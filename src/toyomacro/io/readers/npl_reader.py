"""
NPL file reader (VAMAS variant, Ulvac PHI instruments).

Reads .npl files with work function correction: BE = hv - KE - 4.5 eV.
Based on MATLAB Toyomacro NPLReader.m.

NPL files are line-oriented text, one field per line.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from toyomacro.io.readers.base_reader import (
    BaseReader,
    RawSpectrumData,
    ReaderTransform,
    ReaderWarning,
    SpectrumMetadata,
)

# PHI analyzer work function (eV)
WORK_FUNCTION = 4.5


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


def _read_npl_file(filepath: Path) -> list[tuple[RawSpectrumData, str]]:
    """Read all regions from an NPL file."""
    results: list[tuple[RawSpectrumData, str]] = []

    with open(filepath, encoding="utf-8", errors="replace") as fid:
        # File header: 14 lines
        file_header = _read_lines(fid, 14)
        n_regions = int(file_header[7])

        for region_idx in range(n_regions):
            # Region header: 31 lines
            region_header = _read_lines(fid, 31)

            try:
                year = int(region_header[2])
                month = int(region_header[3])
                day = int(region_header[4])
            except (ValueError, IndexError):
                year = month = day = 0

            try:
                excitation_energy = float(region_header[14])
            except (ValueError, IndexError):
                excitation_energy = None
            if excitation_energy is not None and excitation_energy <= 0:
                # 0 is the legacy "not recorded" sentinel — treat as unknown
                excitation_energy = None

            region_name_flag = region_header[30] if len(region_header) > 30 else "-1"

            dt = None
            try:
                dt = datetime(year, month, day)
            except (ValueError, OverflowError):
                pass

            if region_name_flag == "-1":
                # Unnamed region: 18 extended lines
                ext = _read_lines(fid, 18)
                energy_label = ext[0].lower()  # "binding energy" or "kinetic energy"
                e_ini = float(ext[2])
                e_step = float(ext[3])
                n_slices = int(ext[4])
                n_sweeps = int(ext[9])
                n_energy = int(ext[15])

                # Auto-detect region name
                e_fin = e_ini + e_step * (n_energy - 1)
                if e_fin < 1.0 and abs(e_step * (n_energy - 1)) < 100:
                    region_name = "VB"
                else:
                    region_name = "Wide"
            else:
                # Named region: 19 extended lines
                ext = _read_lines(fid, 19)
                energy_label = ext[1].lower()  # "binding energy" or "kinetic energy"
                e_ini = float(ext[3])
                e_step = float(ext[4])
                n_slices = int(ext[5])
                n_sweeps = int(ext[10])
                n_energy = int(ext[16])
                region_name = region_header[29] + region_name_flag

            # Energy axis
            raw_energy = np.arange(n_energy, dtype=np.float64) * e_step + e_ini
            transforms: list[ReaderTransform] = []

            if "binding" in energy_label:
                # Data is already in binding energy
                energy = raw_energy
                energy_scale_out = "Binding"
            elif excitation_energy is not None:
                # Kinetic energy -> binding energy
                energy = excitation_energy - raw_energy - WORK_FUNCTION
                energy_scale_out = "Binding"
                transforms.append(
                    ReaderTransform(
                        name="energy_scale_conversion",
                        parameters={
                            "from": "Kinetic",
                            "to": "Binding",
                            "excitation_energy_eV": float(excitation_energy),
                            "work_function_eV": WORK_FUNCTION,
                            "formula": "BE = hv - KE - work_function",
                        },
                        source="npl_reader",
                        reason="NPL files are analyzed in binding energy (PHI convention)",
                    )
                )
            else:
                # hv unknown: converting would fabricate an axis. Keep the
                # kinetic scale and say so instead of guessing.
                warnings.warn(
                    f"{filepath.name}[region {region_idx}]: excitation energy "
                    "not recorded; kinetic->binding conversion skipped",
                    ReaderWarning,
                    stacklevel=2,
                )
                energy = raw_energy
                energy_scale_out = "Kinetic"

            # Data: n_energy intensity values
            data_lines = _read_lines(fid, n_energy)
            intensity = np.array(
                [float(line) for line in data_lines], dtype=np.float64
            )

            metadata = SpectrumMetadata(
                region=region_name,
                datetime=dt,
                excitation_energy=excitation_energy,
                energy_scale=energy_scale_out,
                lens_mode="Angular",
                n_slices=n_slices,
                n_sweeps=n_sweeps,
                pass_energy=None,
                source_format="npl",
                source_region_index=region_idx,
                intensity_semantics="unknown",
                intensity_unit="unknown",
                original_shape=(int(n_energy),),
                dimension_roles=("energy",),
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
                        transforms=tuple(transforms),
                    ),
                    region_name,
                )
            )

    return results


class NPLReader(BaseReader):
    """Reader for NPL format .npl files (Ulvac PHI).

    Applies work function correction: BE = hv - KE - 4.5 eV.
    """

    def __init__(self, filepath: str | Path):
        super().__init__(filepath)
        self._results: list[tuple[RawSpectrumData, str]] | None = None

    def _ensure_parsed(self):
        if self._results is None:
            self._results = _read_npl_file(self.filepath)

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
        return fp.suffix.lower() == ".npl"
