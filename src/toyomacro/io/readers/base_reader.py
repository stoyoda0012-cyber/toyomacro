"""
Base reader ABC and common data structures for XPS file format readers.

All format-specific readers (PXT, SES, VAMAS, NPL) inherit from BaseReader
and produce RawSpectrumData as output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SpectrumMetadata:
    """Standardized metadata from raw XPS data files.

    Mirrors MATLAB BaseReader.createEmptyMetadata().
    """

    region: str = ""
    datetime: datetime | None = None
    excitation_energy: float = 0.0
    energy_scale: str = ""  # "Binding" or "Kinetic"
    lens_mode: str = ""  # "Transmission" or "Angular"
    n_slices: int = 1
    n_sweeps: int = 1
    acquisition_mode: str = ""  # "Swept" or "Fixed"
    pass_energy: float = 0.0
    comments: str = ""


@dataclass
class RawSpectrumData:
    """Unified output from all format readers.

    Attributes:
        specdata: Intensity array [nEnergy x nAngle] or [nEnergy x nAngle x nSweep]
        energy: Energy axis [nEnergy]
        angle: Angle axis [nAngle] (scalar 0 if not angle-resolved)
        metadata: Parsed metadata
    """

    specdata: NDArray[np.float64]
    energy: NDArray[np.float64]
    angle: NDArray[np.float64]
    metadata: SpectrumMetadata


class BaseReader(ABC):
    """Abstract base class for XPS file format readers.

    All readers implement:
    - read(region_index) -> RawSpectrumData
    - n_regions -> number of regions in the file
    - region_names -> list of region name strings
    - can_read(filepath) -> bool (static, for format detection)
    """

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

    @abstractmethod
    def read(self, region_index: int = 0) -> RawSpectrumData:
        """Read a region from the file.

        Args:
            region_index: Region index (0-based) for multi-region files.

        Returns:
            RawSpectrumData with specdata, energy, angle, and metadata.
        """
        ...

    @property
    @abstractmethod
    def n_regions(self) -> int:
        """Number of regions in the file."""
        ...

    @property
    @abstractmethod
    def region_names(self) -> list[str]:
        """List of region names."""
        ...

    @staticmethod
    @abstractmethod
    def can_read(filepath: str | Path) -> bool:
        """Check if this reader can handle the given file."""
        ...

    def info(self) -> dict[str, Any]:
        """Summary information about the file."""
        return {
            "filepath": str(self.filepath),
            "format": self.__class__.__name__,
            "n_regions": self.n_regions,
            "region_names": self.region_names,
        }


# --- Format detection and factory ---

_EXTENSION_MAP: dict[str, str] = {
    ".pxt": "pxt",
    ".ibw": "pxt",
    ".vms": "vamas",
    ".npl": "npl",
    ".txt": "ses",  # May be SES or simple; SES reader handles both
}


def detect_format(filepath: str | Path) -> str:
    """Auto-detect file format from extension.

    Args:
        filepath: Path to the data file.

    Returns:
        Format string: 'pxt', 'ses', 'vamas', or 'npl'.

    Raises:
        ValueError: If the extension is not recognized.
    """
    ext = Path(filepath).suffix.lower()
    fmt = _EXTENSION_MAP.get(ext)
    if fmt is None:
        raise ValueError(
            f"Unrecognized file extension '{ext}'. "
            f"Supported: {', '.join(sorted(_EXTENSION_MAP.keys()))}"
        )
    return fmt


def create_reader(filepath: str | Path, format: str = "auto") -> BaseReader:
    """Factory: create the appropriate reader for a file.

    Args:
        filepath: Path to the data file.
        format: 'auto' (detect from extension), 'pxt', 'ses', 'vamas', or 'npl'.

    Returns:
        A BaseReader subclass instance.
    """
    # Lazy imports to avoid circular dependencies
    from toyomacro.io.readers.npl_reader import NPLReader
    from toyomacro.io.readers.pxt_reader import PXTReader
    from toyomacro.io.readers.ses_reader import SESTxtReader
    from toyomacro.io.readers.vamas_reader import VAMASReader

    if format == "auto":
        format = detect_format(filepath)

    reader_map: dict[str, type[BaseReader]] = {
        "pxt": PXTReader,
        "ses": SESTxtReader,
        "vamas": VAMASReader,
        "npl": NPLReader,
    }

    reader_cls = reader_map.get(format)
    if reader_cls is None:
        raise ValueError(
            f"Unknown format '{format}'. Supported: {', '.join(sorted(reader_map.keys()))}"
        )

    return reader_cls(filepath)
