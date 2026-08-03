"""
HDF5 writer for voigtfit/DepthProfiler compatible files.

Supports two modes:
  - **contiguous** (default for batch fitting): pre-allocated, maximum read speed
  - **chunked** (default for import): lightweight creation, no pre-allocation

HDF5 Layout (voigtfit format):
    specdata:  (n_spectra+1, n_energy)  float32
               Row 0 = energy axis, rows 1+ = spectra
    fitpara:   (n_spectra, maxcomp, 9)  float32
    otherpara: (10, n_spectra)          float32
    xytdata:   (3, n_spectra)           float32
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from toyomacro.core import FittingResult, Spectrum
from toyomacro.io.schema import ToyomacroSchema


@dataclass
class HDF5CreateOptions:
    """HDF5 file creation options.

    Attributes:
        contiguous: If True, use contiguous layout (pre-allocated, fastest
            sequential reads).  If False, use chunked layout (lightweight
            creation, no pre-allocation).
    """

    contiguous: bool = True


class HDF5Writer:
    """
    voigtfit/DepthProfiler compatible HDF5 writer.

    Usage:
        with HDF5Writer('/path/to/output.h5') as writer:
            writer.create(
                n_spectra=10000,
                n_energy=500,
                max_components=6,
                energy=energy_axis,
            )
            writer.write_spectra_batch(0, spectra_array)
            writer.write_fitpara_batch(0, fitpara_array)

    Attributes:
        filepath: Output file path
        options: Creation options
    """

    def __init__(
        self,
        filepath: str | Path,
        options: HDF5CreateOptions | None = None,
    ):
        self.filepath = Path(filepath)
        self.options = options or HDF5CreateOptions()
        self._file: h5py.File | None = None
        self._info: dict[str, Any] = {}

    def __enter__(self) -> HDF5Writer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------
    # Dataset creation helpers
    # ------------------------------------------------------------------

    def _ds_kwargs(self, shape: tuple[int, ...]) -> dict[str, Any]:
        """Return h5py create_dataset kwargs according to options.

        Contiguous: no chunks/compression – pre-allocated, max read speed.
        Chunked:    auto-chunks – lightweight creation, no pre-allocation.
        """
        if self.options.contiguous:
            return {}  # h5py default: contiguous
        # Chunked: let h5py pick chunk sizes automatically
        return {"chunks": True}

    # ------------------------------------------------------------------
    # File creation
    # ------------------------------------------------------------------

    def create(
        self,
        n_spectra: int,
        n_energy: int,
        max_components: int = 6,
        energy: NDArray | None = None,
        misc: dict[str, Any] | None = None,
    ):
        """
        Create a new HDF5 file with voigtfit-compatible structure.

        When *options.contiguous* is False the datasets use chunked layout,
        so the file is created almost instantly regardless of size.
        """
        if self.filepath.exists():
            self.filepath.unlink()

        self._file = h5py.File(self.filepath, "w")

        # File schema version (absent = legacy 1.1.x file). Additive root
        # attribute; every copy path preserves or re-stamps it.
        self._file.attrs[ToyomacroSchema.ATTR_SCHEMA_VERSION] = ToyomacroSchema.VERSION

        # specdata: (n_spectra+1, n_energy) - row 0 is energy, rows 1+ are spectra
        self._file.create_dataset(
            ToyomacroSchema.PATH_SPECDATA,
            shape=(n_spectra + 1, n_energy),
            dtype=ToyomacroSchema.DTYPE_SPECDATA,
            **self._ds_kwargs((n_spectra + 1, n_energy)),
        )

        # fitpara: (n_spectra, maxcomp, 9)
        self._file.create_dataset(
            ToyomacroSchema.PATH_FITPARA,
            shape=(n_spectra, max_components, 9),
            dtype=ToyomacroSchema.DTYPE_FITPARA,
            **self._ds_kwargs((n_spectra, max_components, 9)),
        )

        # otherpara: (10, n_spectra)
        self._file.create_dataset(
            ToyomacroSchema.PATH_OTHERPARA,
            shape=(10, n_spectra),
            dtype=ToyomacroSchema.DTYPE_OTHERPARA,
            **self._ds_kwargs((10, n_spectra)),
        )

        # xytdata: (3, n_spectra)
        self._file.create_dataset(
            ToyomacroSchema.PATH_XYTDATA,
            shape=(3, n_spectra),
            dtype=ToyomacroSchema.DTYPE_XYTDATA,
            **self._ds_kwargs((3, n_spectra)),
        )

        # Write energy axis if provided (row 0 of specdata)
        if energy is not None:
            self._file[ToyomacroSchema.PATH_SPECDATA][0, :] = energy.astype(np.float32)

        # Create misc group
        self._create_misc_group(misc or {}, max_components)

        self._info = {
            "n_spectra": n_spectra,
            "n_energy": n_energy,
            "max_components": max_components,
        }

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def write_energy(self, energy: NDArray):
        """Write energy axis (row 0 of specdata)."""
        self._ensure_open()
        self._file[ToyomacroSchema.PATH_SPECDATA][0, :] = energy.astype(np.float32)

    # ------------------------------------------------------------------
    # Spectra
    # ------------------------------------------------------------------

    def write_spectrum(self, idx: int, spectrum: Spectrum | NDArray):
        """Write a single spectrum at 0-based *idx*."""
        self._ensure_open()
        data = spectrum.intensity if isinstance(spectrum, Spectrum) else spectrum
        self._file[ToyomacroSchema.PATH_SPECDATA][idx + 1, :] = data.astype(np.float32)

    def write_spectra_batch(self, start_idx: int, spectra: NDArray):
        """Write a batch of spectra.  *spectra* shape: (n, n_energy)."""
        self._ensure_open()
        n = spectra.shape[0]
        self._file[ToyomacroSchema.PATH_SPECDATA][
            start_idx + 1 : start_idx + 1 + n, :
        ] = spectra.astype(np.float32)

    # ------------------------------------------------------------------
    # fitpara
    # ------------------------------------------------------------------

    def write_fitpara(self, idx: int, fitpara: NDArray):
        """Write fitpara for a single spectrum."""
        self._ensure_open()
        if fitpara.shape[0] == 9:
            fitpara = fitpara.T
        self._file[ToyomacroSchema.PATH_FITPARA][idx, :, :] = fitpara.astype(np.float32)

    def write_fitpara_batch(self, start_idx: int, fitpara: NDArray):
        """Write fitpara batch.  Shape: (n, maxcomp, 9)."""
        self._ensure_open()
        n = fitpara.shape[0]
        self._file[ToyomacroSchema.PATH_FITPARA][
            start_idx : start_idx + n, :, :
        ] = fitpara.astype(np.float32)

    # ------------------------------------------------------------------
    # otherpara
    # ------------------------------------------------------------------

    def write_otherpara(self, idx: int, otherpara: NDArray):
        """Write otherpara for a single spectrum.  Shape: (10,)."""
        self._ensure_open()
        self._file[ToyomacroSchema.PATH_OTHERPARA][:, idx] = otherpara.astype(np.float32)

    def write_otherpara_batch(self, start_idx: int, otherpara: NDArray):
        """Write otherpara batch.  Input shape: (n, 10)."""
        self._ensure_open()
        n = otherpara.shape[0]
        self._file[ToyomacroSchema.PATH_OTHERPARA][
            :, start_idx : start_idx + n
        ] = otherpara.T.astype(np.float32)

    # ------------------------------------------------------------------
    # xytdata
    # ------------------------------------------------------------------

    def write_xytdata(self, idx: int, xytdata: NDArray):
        """Write xytdata for a single spectrum.  Shape: (3,)."""
        self._ensure_open()
        self._file[ToyomacroSchema.PATH_XYTDATA][:, idx] = xytdata.astype(np.float32)

    def write_xytdata_batch(self, xytdata: NDArray):
        """Write xytdata for all spectra at once.

        Args:
            xytdata: Array of shape (3, n_spectra).
        """
        self._ensure_open()
        self._file[ToyomacroSchema.PATH_XYTDATA][:, :] = xytdata.astype(np.float32)

    # ------------------------------------------------------------------
    # Fitting result convenience
    # ------------------------------------------------------------------

    def write_fitting_result(self, idx: int, result: FittingResult):
        """Write a FittingResult object."""
        fitpara = result.to_fitpara()  # (9, n_components)
        otherpara = result.to_otherpara()  # (10,)

        max_comp = self._info.get("max_components", 6)
        if fitpara.shape[1] < max_comp:
            padded = np.zeros((9, max_comp), dtype=np.float32)
            padded[:, : fitpara.shape[1]] = fitpara
            fitpara = padded

        self.write_fitpara(idx, fitpara)
        self.write_otherpara(idx, otherpara)

    # ------------------------------------------------------------------
    # Provenance (facts from the upstream input + transform history)
    # ------------------------------------------------------------------

    def write_provenance(
        self,
        metadata,
        transforms=(),
        *,
        persist_datetime: bool = False,
        persist_vendor_metadata: bool = False,
        vendor_metadata_allowlist=(),
    ) -> None:
        """Write the /provenance group from Phase A reader output.

        See toyomacro.io.provenance and the provenance design note
        (non-public; CONTRIBUTING.md).
        """
        self._ensure_open()
        from toyomacro.io.provenance import write_provenance

        write_provenance(
            self._file,
            metadata,
            transforms,
            persist_datetime=persist_datetime,
            persist_vendor_metadata=persist_vendor_metadata,
            vendor_metadata_allowlist=vendor_metadata_allowlist,
        )

    # ------------------------------------------------------------------
    # dim_shape (4D navigation metadata)
    # ------------------------------------------------------------------

    def write_dim_shape(self, dim_shape: list[int]) -> None:
        """Write dimension shape metadata to misc group.

        Stores the non-energy dimensions of the original data before
        flattening.  Used by HDF5Cache to auto-configure DimensionConfig.

        Examples:
            4D (1024, 12, 25, 20) -> dim_shape = [12, 25, 20]
            3D (201, 192, 401)    -> dim_shape = [192, 401]
            2D (401, 400)         -> dim_shape = [400]
        """
        self._ensure_open()
        misc = self._file.require_group("misc")
        if "dim_shape" in misc:
            del misc["dim_shape"]
        misc.create_dataset(
            "dim_shape",
            data=np.array(dim_shape, dtype=np.int32),
        )

    def write_dim0_is_angle(self, flag: bool) -> None:
        """Write whether dim0 represents angle (θ) or spatial (X).

        Determined from ``metadata.lens_mode`` during import:
        - ``"Angular"`` and multi-channel → True (θ)
        - ``"Transmission"`` / empty / single-channel → False (X)

        Old files without this flag default to True (backward compat).
        """
        self._ensure_open()
        misc = self._file.require_group("misc")
        if "dim0_is_angle" in misc:
            del misc["dim0_is_angle"]
        misc.create_dataset(
            "dim0_is_angle",
            data=np.array([1 if flag else 0], dtype=np.int32),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def flush(self):
        """Flush data to disk."""
        if self._file is not None:
            self._file.flush()

    def _ensure_open(self):
        if self._file is None:
            raise RuntimeError("File not open. Call create() first.")

    def _create_misc_group(self, misc: dict[str, Any], max_components: int):
        """Create misc metadata group."""
        group = self._file.create_group("misc")

        defaults = ToyomacroSchema.get_misc_defaults(max_components)
        defaults.update(misc)

        for key, value in defaults.items():
            if isinstance(value, str):
                dt = h5py.string_dtype(encoding="utf-8")
                group.create_dataset(key, data=value, dtype=dt)
            elif isinstance(value, (int, np.integer)):
                group.create_dataset(key, data=np.array([value], dtype=np.int32))
            else:
                group.create_dataset(key, data=np.array([value], dtype=np.float32))

    def __repr__(self) -> str:
        if self._file is not None:
            return f"HDF5Writer({self.filepath.name}, open)"
        return f"HDF5Writer({self.filepath.name}, closed)"
