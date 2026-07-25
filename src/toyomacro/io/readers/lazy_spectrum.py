"""
Lazy loading spectrum accessor for large HDF5 files.

Compatible with voigtfit/DepthProfiler HDF5 format:
    specdata:  (n_spectra+1, n_energy)  - Row 0 is energy, rows 1+ are spectra
    fitpara:   (n_spectra, maxcomp, 9)  - Fitting parameters
    otherpara: (10, n_spectra)          - Additional parameters
    xytdata:   (3, n_spectra)           - Position data
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from numpy.typing import NDArray

from toyomacro.core import FittingResult, Spectrum
from toyomacro.io.schema import ToyomacroSchema
from toyomacro.io.utils import calculate_batch_size

if TYPE_CHECKING:
    from toyomacro.io.provenance import HDF5Provenance


@dataclass
class DatasetInfo:
    """HDF5 dataset metadata."""

    path: str
    shape: tuple[int, ...]
    dtype: np.dtype
    chunks: tuple[int, ...] | None
    raw_bytes: int
    n_spectra: int
    n_energy: int
    is_contiguous: bool = False


class LazySpectrum:
    """
    Lazy loading spectrum accessor for large HDF5 files.

    Provides memory-efficient access to large HDF5 files without loading
    all data into memory. Supports preview (downsampled), slicing,
    and batch iteration.

    Usage:
        with LazySpectrum('/path/to/data.h5') as spec:
            # Metadata is immediately available
            print(f"Total spectra: {spec.n_spectra}")
            print(f"Energy range: {spec.energy_range}")

            # Preview (downsampled for quick display)
            preview = spec.preview(max_points=1000)

            # Slice specific range
            data = spec.slice(start=0, end=100)

            # Batch iteration
            for batch in spec.iter_batches(batch_size=10000):
                process(batch)

    Attributes:
        filepath: Path to HDF5 file
        n_spectra: Total number of spectra
        n_energy: Number of energy points
        energy: Energy axis (cached on first access)
    """

    # HDF5 cache settings (optimized for sequential read)
    DEFAULT_RDCC_NBYTES = 64 * 1024 * 1024  # 64MB
    DEFAULT_RDCC_NSLOTS = 10007
    DEFAULT_RDCC_W0 = 0.75

    def __init__(
        self,
        filepath: str | Path,
        swmr: bool = False,
        cache_energy: bool = True,
    ):
        """
        Initialize LazySpectrum.

        Args:
            filepath: Path to HDF5 file
            swmr: Enable SWMR (Single Writer Multiple Reader) mode
            cache_energy: Cache energy axis in memory (recommended)
        """
        self.filepath = Path(filepath)
        self._swmr = swmr
        self._cache_energy = cache_energy

        self._file: h5py.File | None = None
        self._info: DatasetInfo | None = None
        self._energy_cache: NDArray[np.float32] | None = None
        self._energy_type: str = "BE"
        self._is_open = False

    # ===== Context Manager =====

    def __enter__(self) -> LazySpectrum:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open the HDF5 file."""
        if self._is_open:
            return

        self._file = h5py.File(
            self.filepath,
            "r",
            swmr=self._swmr,
            rdcc_nbytes=self.DEFAULT_RDCC_NBYTES,
            rdcc_nslots=self.DEFAULT_RDCC_NSLOTS,
            rdcc_w0=self.DEFAULT_RDCC_W0,
        )

        self._info = self._get_dataset_info()

        if self._cache_energy:
            self._energy_cache = self._read_energy()

        # Read energy type from misc/bindingenergysign
        try:
            misc = self._file.get("misc")
            if misc is not None and "bindingenergysign" in misc:
                val = int(misc["bindingenergysign"][()])
                self._energy_type = "BE" if val == 1 else "KE"
        except Exception:
            pass  # keep default "BE"

        self._is_open = True

    def close(self):
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None
        self._is_open = False
        self._energy_cache = None

    # ===== Properties =====

    @property
    def n_spectra(self) -> int:
        """Total number of spectra."""
        self._ensure_open()
        return self._info.n_spectra

    @property
    def n_energy(self) -> int:
        """Number of energy points."""
        self._ensure_open()
        return self._info.n_energy

    @property
    def shape(self) -> tuple[int, int]:
        """Shape as (n_spectra, n_energy)."""
        return (self.n_spectra, self.n_energy)

    @property
    def energy_type(self) -> str:
        """Energy axis type ('BE' or 'KE')."""
        self._ensure_open()
        return self._energy_type

    @property
    def energy(self) -> NDArray[np.float32]:
        """Energy axis (cached)."""
        self._ensure_open()
        if self._energy_cache is not None:
            return self._energy_cache
        return self._read_energy()

    @property
    def energy_range(self) -> tuple[float, float]:
        """Energy range (min, max)."""
        e = self.energy
        return (float(e.min()), float(e.max()))

    @property
    def info(self) -> DatasetInfo:
        """Dataset information."""
        self._ensure_open()
        return self._info

    # ===== Core Read Methods =====

    def preview(
        self,
        max_points: int = 1000,
        spectrum_idx: int = 0,
    ) -> Spectrum:
        """
        Get a downsampled preview for quick display.

        Samples the spectrum at regular intervals to provide a quick
        overview without reading all data.

        Args:
            max_points: Maximum number of points in preview
            spectrum_idx: Index of spectrum to preview

        Returns:
            Downsampled Spectrum object
        """
        self._ensure_open()

        n = self.n_energy
        step = max(1, n // max_points)

        specdata = self._file[ToyomacroSchema.PATH_SPECDATA]

        # Read with step (downsampling)
        # voigtfit format: specdata shape is (n_spectra+1, n_energy)
        # Row 0 is energy axis, rows 1+ are spectra
        energy = specdata[0, ::step].astype(np.float64)
        intensity = specdata[spectrum_idx + 1, ::step].astype(np.float64)

        return Spectrum(
            energy=energy,
            intensity=intensity,
            energy_type=self._energy_type,
            metadata={"preview": True, "step": step, "original_points": n},
        )

    def slice(
        self,
        start: int = 0,
        end: int | None = None,
        as_spectrum: bool = True,
    ) -> Spectrum | NDArray[np.float32]:
        """
        Read a slice of spectra.

        Args:
            start: Start index (0-based, inclusive)
            end: End index (exclusive, None = to end)
            as_spectrum: If True and single spectrum, return Spectrum object

        Returns:
            Single Spectrum or (count, n_energy) array
        """
        self._ensure_open()

        if end is None:
            end = self.n_spectra

        specdata = self._file[ToyomacroSchema.PATH_SPECDATA]

        # voigtfit format: specdata shape is (n_spectra+1, n_energy)
        # Row 0 is energy, rows 1+ are spectra
        # Returns (count, n_energy) - no transpose needed
        intensity = specdata[start + 1 : end + 1, :].astype(np.float32)

        if as_spectrum and intensity.shape[0] == 1:
            return Spectrum(
                energy=self.energy.astype(np.float64),
                intensity=intensity[0].astype(np.float64),
                energy_type=self._energy_type,
            )

        return intensity

    def get_spectrum(self, idx: int) -> Spectrum:
        """
        Get a single spectrum by index.

        Args:
            idx: Spectrum index (0-based)

        Returns:
            Spectrum object
        """
        result = self.slice(start=idx, end=idx + 1, as_spectrum=True)
        if isinstance(result, Spectrum):
            return result
        # Should not happen, but handle array case
        return Spectrum(
            energy=self.energy.astype(np.float64),
            intensity=result[0].astype(np.float64),
            energy_type=self._energy_type,
        )

    # ===== Batch Iteration =====

    def iter_batches(
        self,
        batch_size: int | None = None,
        memory_fraction: float = 0.5,
        yield_indices: bool = False,
        zero_copy: bool = True,
    ) -> Iterator[NDArray[np.float32] | tuple[int, NDArray[np.float32]]]:
        """
        Iterate over spectra in batches.

        Args:
            batch_size: Batch size (None = auto-calculate based on memory)
            memory_fraction: Fraction of available memory to use
            yield_indices: If True, yield (start_idx, batch) tuples
            zero_copy: If True, use read_direct for faster I/O when dtype matches

        Yields:
            Batch of spectra as (batch_size, n_energy) array,
            or (start_idx, batch) tuples if yield_indices=True
        """
        self._ensure_open()

        if batch_size is None:
            batch_size = calculate_batch_size(
                self.n_spectra,
                self.n_energy,
                memory_fraction=memory_fraction,
            )

        specdata = self._file[ToyomacroSchema.PATH_SPECDATA]

        # Pre-allocate buffer for zero-copy optimization
        if zero_copy and specdata.dtype == np.float32:
            buffer = np.empty((batch_size, self.n_energy), dtype=np.float32)

        for start in range(0, self.n_spectra, batch_size):
            end = min(start + batch_size, self.n_spectra)
            actual_count = end - start

            # Try zero-copy path (read_direct) when dtype matches
            if zero_copy and specdata.dtype == np.float32:
                try:
                    specdata.read_direct(
                        buffer[:actual_count],
                        source_sel=np.s_[start + 1 : end + 1, :],
                    )
                    # Copy from buffer since we'll reuse it
                    data = buffer[:actual_count].copy()
                except TypeError:
                    # Fallback if read_direct fails
                    data = specdata[start + 1 : end + 1, :].astype(np.float32)
            else:
                # voigtfit format: specdata[start+1:end+1, :] is (count, n_energy)
                data = specdata[start + 1 : end + 1, :].astype(np.float32)

            if yield_indices:
                yield (start, data)
            else:
                yield data

    # ===== Fitting Result Access =====

    def get_fitpara(
        self,
        start: int = 0,
        end: int | None = None,
    ) -> NDArray[np.float32]:
        """
        Read fitpara array.

        Args:
            start: Start index
            end: End index (None = to end)

        Returns:
            Array of shape (count, maxcomp, 9)
        """
        self._ensure_open()

        if ToyomacroSchema.PATH_FITPARA not in self._file:
            raise KeyError(f"fitpara not found in {self.filepath}")

        fitpara = self._file[ToyomacroSchema.PATH_FITPARA]

        if end is None:
            end = fitpara.shape[0]  # voigtfit format: (n_spectra, maxcomp, 9)

        # voigtfit format: already (n_spectra, maxcomp, 9) - no transpose needed
        return fitpara[start:end, :, :].astype(np.float32)

    def get_otherpara(self, idx: int) -> NDArray[np.float32]:
        """
        Read otherpara for a single spectrum.

        Args:
            idx: Spectrum index

        Returns:
            Array of shape (10,)
        """
        self._ensure_open()

        if ToyomacroSchema.PATH_OTHERPARA not in self._file:
            raise KeyError("otherpara not found")

        otherpara = self._file[ToyomacroSchema.PATH_OTHERPARA]

        # voigtfit format: (10, n_spectra)
        return otherpara[:, idx].astype(np.float32)

    def get_fitting_result(self, idx: int) -> FittingResult:
        """
        Get FittingResult for a single spectrum.

        Args:
            idx: Spectrum index

        Returns:
            FittingResult object
        """
        self._ensure_open()

        fitpara = self.get_fitpara(start=idx, end=idx + 1)[0]  # (maxcomp, 9)
        otherpara = self.get_otherpara(idx)  # (10,)

        return FittingResult.from_matlab(
            fitpara=fitpara.T,  # (9, maxcomp)
            otherpara=otherpara,
            energy=self.energy.astype(np.float64),
        )

    def has_fitting_results(self) -> bool:
        """Check if file contains fitting results."""
        self._ensure_open()
        return ToyomacroSchema.PATH_FITPARA in self._file

    # ===== Metadata Access =====

    def get_provenance(self) -> HDF5Provenance | None:
        """Read the /provenance group (Toyomacro-local schema).

        Returns:
            HDF5Provenance for files written with provenance support,
            None for legacy files without the group.
        """
        self._ensure_open()
        from toyomacro.io.provenance import read_provenance

        return read_provenance(self._file)

    def get_misc(self) -> dict:
        """
        Read misc metadata.

        Returns:
            Dictionary with misc values
        """
        self._ensure_open()

        if ToyomacroSchema.PATH_MISC not in self._file:
            return {}

        misc_group = self._file[ToyomacroSchema.PATH_MISC]
        result = {}

        for key in misc_group:
            value = misc_group[key][()]
            # Decode bytes to string if needed
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.ndarray) and value.size == 1:
                value = value.item()
            result[key] = value

        return result

    # ===== Private Methods =====

    def _ensure_open(self):
        """Ensure file is open."""
        if not self._is_open:
            raise RuntimeError("File not open. Use context manager or call open()")

    def _read_energy(self) -> NDArray[np.float32]:
        """Read energy axis from specdata."""
        specdata = self._file[ToyomacroSchema.PATH_SPECDATA]
        # voigtfit format: row 0 is energy axis
        return specdata[0, :].astype(np.float32)

    def _get_dataset_info(self) -> DatasetInfo:
        """Get dataset information."""
        specdata = self._file[ToyomacroSchema.PATH_SPECDATA]
        # voigtfit format: (n_spectra+1, n_energy)
        # Row 0 is energy, rows 1+ are spectra
        return DatasetInfo(
            path=ToyomacroSchema.PATH_SPECDATA,
            shape=specdata.shape,
            dtype=specdata.dtype,
            chunks=specdata.chunks,
            raw_bytes=int(np.prod(specdata.shape) * specdata.dtype.itemsize),
            n_spectra=specdata.shape[0] - 1,  # Row 0 is energy
            n_energy=specdata.shape[1],
            is_contiguous=specdata.chunks is None,
        )

    def __repr__(self) -> str:
        if self._is_open:
            return (
                f"LazySpectrum({self.filepath.name}, "
                f"n_spectra={self.n_spectra}, n_energy={self.n_energy})"
            )
        return f"LazySpectrum({self.filepath.name}, closed)"
