"""
VAMAS file reader (ISO 14976 format).

Reads .vms files commonly used by Omicron instruments.
Based on MATLAB Toyomacro VAMASReader.m.

VAMAS files are line-oriented text, one field per line.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from toyomacro.io.readers.base_reader import (
    BaseReader,
    RawSpectrumData,
    SpectrumMetadata,
)


def _read_lines(fid, n: int) -> list[str]:
    """Read n non-empty lines from file."""
    lines: list[str] = []
    while len(lines) < n:
        line = fid.readline()
        if not line:
            break
        stripped = line.strip().replace("\x00", "")
        if stripped:
            lines.append(stripped)
    return lines


def _read_vamas_file(filepath: Path) -> list[tuple[RawSpectrumData, str]]:
    """Read all regions from a VAMAS file."""
    results: list[tuple[RawSpectrumData, str]] = []

    with open(filepath, encoding="utf-8", errors="replace") as fid:
        # File header: 12 lines
        file_header = _read_lines(fid, 12)
        n_regions = int(file_header[5])

        for region_idx in range(n_regions):
            # Region header: 9 lines
            region_header = _read_lines(fid, 9)

            try:
                year = int(region_header[0])
                month = int(region_header[1])
                day = int(region_header[2])
                hour = int(region_header[3])
                minute = int(region_header[4])
                second = int(region_header[5])
            except (ValueError, IndexError):
                year = month = day = hour = minute = second = 0

            region_name_field = region_header[8] if len(region_header) > 8 else "XPS"

            dt = None
            try:
                dt = datetime(year, month, day, hour, minute, second)
            except (ValueError, OverflowError):
                pass

            # Extended header
            if region_name_field == "XPS":
                ext = _read_lines(fid, 34)
                e_ini = float(ext[18])
                e_step = float(ext[19])
                n_slices = int(ext[20])
                n_sweeps = int(ext[25])
                n_energy = int(ext[31])
                display_name = f"XPS{region_idx + 1}"
            else:
                ext = _read_lines(fid, 35)
                e_ini = float(ext[19])
                e_step = float(ext[20])
                n_slices = int(ext[21])
                n_sweeps = int(ext[26])
                n_energy = int(ext[32])
                display_name = region_name_field

            # Energy axis (kinetic energy)
            energy = np.arange(n_energy, dtype=np.float64) * e_step + e_ini

            # Data: n_energy intensity values
            data_lines = _read_lines(fid, n_energy)
            intensity = np.array(
                [float(line) for line in data_lines], dtype=np.float64
            )

            metadata = SpectrumMetadata(
                region=display_name,
                datetime=dt,
                energy_scale="Kinetic",
                lens_mode="Angular",
                n_slices=n_slices,
                n_sweeps=n_sweeps,
            )

            specdata = intensity.reshape(-1, 1)
            angle = np.array([0.0])

            results.append(
                (
                    RawSpectrumData(
                        specdata=specdata,
                        energy=energy,
                        angle=angle,
                        metadata=metadata,
                    ),
                    display_name,
                )
            )

    return results


class VAMASReader(BaseReader):
    """Reader for VAMAS (ISO 14976) .vms files."""

    def __init__(self, filepath: str | Path):
        super().__init__(filepath)
        self._results: list[tuple[RawSpectrumData, str]] | None = None

    def _ensure_parsed(self):
        if self._results is None:
            self._results = _read_vamas_file(self.filepath)

    @property
    def n_regions(self) -> int:
        self._ensure_parsed()
        return len(self._results)

    @property
    def region_names(self) -> list[str]:
        self._ensure_parsed()
        return [name for _, name in self._results]

    def read(self, region_index: int = 0) -> RawSpectrumData:
        self._ensure_parsed()
        if region_index < 0 or region_index >= len(self._results):
            raise IndexError(
                f"Region index {region_index} out of range [0, {len(self._results)})"
            )
        return self._results[region_index][0]

    @staticmethod
    def can_read(filepath: str | Path) -> bool:
        fp = Path(filepath)
        return fp.suffix.lower() == ".vms"
