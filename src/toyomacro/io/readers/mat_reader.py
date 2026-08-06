"""MATLAB .mat file reader supporting both v5/v7 and v7.3 (HDF5) formats."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from toyomacro.core import FittingResult, Spectrum


def detect_mat_version(filepath: str | Path) -> str:
    """
    Detect MATLAB file version.

    Args:
        filepath: Path to .mat file

    Returns:
        'v5': scipy.io.loadmat compatible (MATLAB v5-v7)
        'v7.3': HDF5 format (MATLAB v7.3+)
        'unknown': Unknown format
    """
    filepath = Path(filepath)

    with open(filepath, "rb") as f:
        header = f.read(128)

    # HDF5 signature: 0x89 'HDF'
    if header[:4] == b"\x89HDF":
        return "v7.3"

    # MATLAB v5 signature
    if header[:19] == b"MATLAB 5.0 MAT-file":
        return "v5"

    return "unknown"


class _MATBackend(Protocol):
    """Protocol for MAT file backends."""

    @property
    def n_spectra(self) -> int: ...

    @property
    def n_energy(self) -> int: ...

    @property
    def energy(self) -> NDArray: ...

    def get_spectrum(self, idx: int) -> Spectrum: ...

    def get_fitpara(self) -> NDArray: ...

    def get_otherpara(self) -> NDArray: ...

    def close(self) -> None: ...


class _MatV5Backend:
    """Backend for MATLAB v5/v7 files using scipy.io."""

    def __init__(self, filepath: Path):
        import scipy.io

        self._data = scipy.io.loadmat(
            filepath,
            squeeze_me=False,
            struct_as_record=True,
        )
        self._validate_structure()

    def _validate_structure(self):
        """Validate required fields exist."""
        # specdata is required, others are optional
        if "specdata" not in self._data:
            raise ValueError("Required field 'specdata' not found in MAT file")

    @property
    def n_spectra(self) -> int:
        """Number of spectra (columns - 1 for energy)."""
        return self._data["specdata"].shape[1] - 1

    @property
    def n_energy(self) -> int:
        """Number of energy points."""
        return self._data["specdata"].shape[0]

    @property
    def energy(self) -> NDArray:
        """Energy axis (first column of specdata)."""
        return self._data["specdata"][:, 0].astype(np.float64)

    def get_spectrum(self, idx: int) -> Spectrum:
        """Get spectrum by index."""
        return Spectrum(
            energy=self.energy,
            intensity=self._data["specdata"][:, idx + 1].astype(np.float64),
        )

    def get_fitpara(self) -> NDArray:
        """Get fitpara array."""
        if "fitpara" not in self._data:
            raise KeyError("fitpara not found in MAT file")
        return self._data["fitpara"].astype(np.float32)

    def get_otherpara(self) -> NDArray:
        """Get otherpara array."""
        if "otherpara" not in self._data:
            raise KeyError("otherpara not found in MAT file")
        return self._data["otherpara"].astype(np.float32)

    def close(self):
        """No-op for v5 (data already loaded)."""
        pass


class _MatV73Backend:
    """Backend for MATLAB v7.3 (HDF5) files using h5py."""

    def __init__(self, filepath: Path):
        import h5py

        self._file = h5py.File(filepath, "r")
        self._validate_structure()

    def _validate_structure(self):
        """Validate required fields exist."""
        if "specdata" not in self._file:
            raise ValueError("Required field 'specdata' not found in MAT file")

    @property
    def n_spectra(self) -> int:
        """Number of spectra."""
        # MATLAB v7.3 stores arrays transposed
        # specdata shape could be (n_energy, 1+n_spectra) or transposed
        shape = self._file["specdata"].shape
        # Assume larger dimension is n_spectra+1
        if shape[0] > shape[1]:
            return shape[0] - 1
        return shape[1] - 1

    @property
    def n_energy(self) -> int:
        """Number of energy points."""
        shape = self._file["specdata"].shape
        if shape[0] > shape[1]:
            return shape[1]
        return shape[0]

    @property
    def energy(self) -> NDArray:
        """Energy axis."""
        specdata = self._file["specdata"]
        shape = specdata.shape

        # Determine orientation
        if shape[0] > shape[1]:
            # (n_spectra+1, n_energy) - first row is energy
            return specdata[0, :].astype(np.float64)
        else:
            # (n_energy, n_spectra+1) - first column is energy
            return specdata[:, 0].astype(np.float64)

    def get_spectrum(self, idx: int) -> Spectrum:
        """Get spectrum by index."""
        specdata = self._file["specdata"]
        shape = specdata.shape

        if shape[0] > shape[1]:
            # (n_spectra+1, n_energy)
            intensity = specdata[idx + 1, :].astype(np.float64)
        else:
            # (n_energy, n_spectra+1)
            intensity = specdata[:, idx + 1].astype(np.float64)

        return Spectrum(
            energy=self.energy,
            intensity=intensity,
        )

    def get_fitpara(self) -> NDArray:
        """Get fitpara array."""
        if "fitpara" not in self._file:
            raise KeyError("fitpara not found in MAT file")
        return np.array(self._file["fitpara"]).astype(np.float32)

    def get_otherpara(self) -> NDArray:
        """Get otherpara array."""
        if "otherpara" not in self._file:
            raise KeyError("otherpara not found in MAT file")
        return np.array(self._file["otherpara"]).astype(np.float32)

    def close(self):
        """Close HDF5 file."""
        self._file.close()


class MATReader:
    """
    MATLAB .mat file reader.

    Supports both v5/v7 (scipy.io) and v7.3 (HDF5) formats.
    Read-only (writing is done via HDF5Writer).

    Usage:
        with MATReader('/path/to/data.mat') as reader:
            print(f"n_spectra={reader.n_spectra}, n_energy={reader.n_energy}")
            spectrum = reader.get_spectrum(0)
            fitpara = reader.get_fitpara()

    Attributes:
        filepath: Path to MAT file
        version: Detected file version ('v5' or 'v7.3')
    """

    def __init__(self, filepath: str | Path):
        """
        Initialize MATReader.

        Args:
            filepath: Path to .mat file
        """
        self.filepath = Path(filepath)
        self.version = detect_mat_version(filepath)

        if self.version == "v7.3":
            self._backend: _MATBackend = _MatV73Backend(self.filepath)
        elif self.version == "v5":
            self._backend = _MatV5Backend(self.filepath)
        else:
            raise ValueError(f"Unknown MAT file format: {filepath}")

    def __enter__(self) -> MATReader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def n_spectra(self) -> int:
        """Number of spectra."""
        return self._backend.n_spectra

    @property
    def n_energy(self) -> int:
        """Number of energy points."""
        return self._backend.n_energy

    @property
    def energy(self) -> NDArray:
        """Energy axis."""
        return self._backend.energy

    def get_spectrum(self, idx: int) -> Spectrum:
        """
        Get spectrum by index.

        Args:
            idx: Spectrum index (0-based)

        Returns:
            Spectrum object
        """
        return self._backend.get_spectrum(idx)

    def get_fitpara(self) -> NDArray:
        """
        Get fitpara array.

        Returns:
            Array of shape (9, maxcomp, numPeaks) for MATLAB format
        """
        return self._backend.get_fitpara()

    def get_otherpara(self) -> NDArray:
        """
        Get otherpara array.

        Returns:
            Array of shape (numPeaks, 10) or (10, numPeaks)
        """
        return self._backend.get_otherpara()

    def get_fitting_result(self, idx: int) -> FittingResult:
        """
        Get FittingResult for a spectrum.

        Args:
            idx: Spectrum index (0-based)

        Returns:
            FittingResult object
        """
        fitpara = self.get_fitpara()
        otherpara = self.get_otherpara()

        # Extract for single spectrum
        # fitpara shape: (9, maxcomp, numPeaks)
        fp = fitpara[:, :, idx]  # (9, maxcomp)

        # otherpara shape: (numPeaks, 10) or (10, numPeaks)
        if otherpara.shape[0] == 10:
            op = otherpara[:, idx]
        else:
            op = otherpara[idx, :]

        return FittingResult.from_matlab(
            fitpara=fp,
            otherpara=op,
            energy=self.energy,
        )

    def has_fitting_results(self) -> bool:
        """Check if file contains fitting results."""
        try:
            self._backend.get_fitpara()
            return True
        except KeyError:
            return False

    def close(self):
        """Close the file."""
        self._backend.close()

    def __repr__(self) -> str:
        return (
            f"MATReader({self.filepath.name}, version={self.version}, "
            f"n_spectra={self.n_spectra}, n_energy={self.n_energy})"
        )
