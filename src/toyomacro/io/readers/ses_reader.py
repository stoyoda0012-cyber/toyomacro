"""
SES text file reader for Scienta SES text exports.

Supports two sub-formats:
1. SES format: INI-like sections [Info], [Region N], [Info N], [Data N]
2. Simple text format: Column 1 = energy, columns 2+ = intensity

Based on MATLAB Toyomacro SESTxtReader.m.
"""

from __future__ import annotations

import re
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from toyomacro.io.readers.base_reader import (
    BaseReader,
    RawSpectrumData,
    SpectrumMetadata,
)


def _parse_dimension_scale(s: str) -> NDArray:
    """Parse dimension scale values (space or comma separated)."""
    if not s.strip():
        return np.array([])
    # Try comma-separated first, then space-separated
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = s.split()
    return np.array([float(p) for p in parts if p.strip()])


def _is_ses_format(filepath: Path) -> bool:
    """Check if the file has SES header format (contains [Data or [Info marker)."""
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            # Read up to 200 lines looking for SES markers
            for i, line in enumerate(f):
                if i > 200:
                    break
                stripped = line.strip()
                if "[Data" in stripped or "[Info]" in stripped:
                    return True
    except Exception:
        pass
    return False


def _parse_sections(content: str) -> dict[str, str]:
    """Parse INI-like sections from SES text file.

    Returns a dict of {section_name: section_body}.
    Section names: "Info", "Region 1", "Info 1", "Data 1", "Data 1:3", etc.
    """
    sections: dict[str, str] = {}

    # Normalize line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Split into sections using [SectionName] markers
    # Pattern: [SectionName] at the beginning of a line
    parts = re.split(r"^\[([^\]]+)\]\s*$", content, flags=re.MULTILINE)

    # parts[0] = text before first section (might have stray chars)
    # parts[1] = first section name, parts[2] = first section body, etc.
    for i in range(1, len(parts) - 1, 2):
        section_name = parts[i].strip()
        section_body = parts[i + 1] if i + 1 < len(parts) else ""
        sections[section_name] = section_body

    return sections


def _parse_kv_section(body: str) -> dict[str, str]:
    """Parse key=value pairs from a section body."""
    fields: dict[str, str] = {}
    for line in body.strip().split("\n"):
        line = line.strip()
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
    return fields


def _read_ses_format(filepath: Path) -> list[RawSpectrumData]:
    """Read SES INI-like format file with [Info], [Region N], [Data N] sections."""
    with open(filepath, encoding="utf-8", errors="replace") as f:
        content = f.read()

    sections = _parse_sections(content)

    # Global info
    global_info = _parse_kv_section(sections.get("Info", ""))
    n_regions = int(global_info.get("Number of Regions", "1"))

    results: list[RawSpectrumData] = []

    for reg_idx in range(1, n_regions + 1):
        # Parse [Region N] for dimension info
        region_section = sections.get(f"Region {reg_idx}", "")
        region_fields = _parse_kv_section(region_section)

        # Parse [Info N] for metadata
        info_section = sections.get(f"Info {reg_idx}", "")
        info_fields = _parse_kv_section(info_section)

        # Merge: info_fields takes precedence for metadata, region_fields for dimensions
        all_fields = {**region_fields, **info_fields}

        # Build metadata
        region_name = all_fields.get("Region Name", all_fields.get("Spectrum Name", ""))
        excitation_energy = float(all_fields.get("Excitation Energy", "0"))
        energy_scale = all_fields.get("Energy Scale", "Kinetic")
        lens_mode = all_fields.get("Lens Mode", "")
        if lens_mode.startswith("T"):
            lens_mode = "Transmission"
        elif lens_mode.startswith("A"):
            lens_mode = "Angular"
        n_slices = int(all_fields.get("Number of Slices", "1"))
        n_sweeps = int(all_fields.get("Number of Sweeps", "1"))
        acquisition_mode = all_fields.get(
            "Acquisition Mode", all_fields.get("Aquisition Mode", "")
        )
        pass_energy = float(all_fields.get("Pass Energy", "0"))

        # Parse datetime
        dt = None
        date_str = all_fields.get("Date", "")
        time_str = all_fields.get("Time", "")
        if date_str and time_str:
            for fmt in ["%Y-%m-%d %H:%M:%S", "%m/%d/%Y %I:%M:%S %p"]:
                try:
                    dt = datetime.strptime(f"{date_str} {time_str}", fmt)
                    break
                except ValueError:
                    pass

        metadata = SpectrumMetadata(
            region=region_name,
            datetime=dt,
            excitation_energy=excitation_energy,
            energy_scale=energy_scale,
            lens_mode=lens_mode,
            n_slices=n_slices,
            n_sweeps=n_sweeps,
            acquisition_mode=acquisition_mode,
            pass_energy=pass_energy,
        )

        # Parse dimension scales from [Region N]
        d1_str = region_fields.get("Dimension 1 scale", "")
        d2_str = region_fields.get("Dimension 2 scale", "")
        d3_str = region_fields.get("Dimension 3 scale", "")
        d2_name = region_fields.get("Dimension 2 name", "")

        d1 = _parse_dimension_scale(d1_str)
        d2 = _parse_dimension_scale(d2_str)
        d3 = _parse_dimension_scale(d3_str)

        # Collect data sections for this region
        # Patterns: [Data N] (1D/2D) or [Data N:S] (3D, per-sweep)
        data_blocks: list[str] = []
        data_key = f"Data {reg_idx}"
        if data_key in sections:
            data_blocks.append(sections[data_key])

        # Check for sub-blocks [Data N:1], [Data N:2], ...
        sweep_idx = 1
        while True:
            sub_key = f"Data {reg_idx}:{sweep_idx}"
            if sub_key in sections:
                data_blocks.append(sections[sub_key])
                sweep_idx += 1
            else:
                break

        if not data_blocks:
            continue  # No data for this region

        # Parse data
        if len(d1) > 0 and len(d2) > 0:
            # 2D or 3D with dimension scales
            result = _read_region_with_dimensions(
                data_blocks, d1, d2, d3, d2_name, metadata
            )
        elif len(d1) > 0:
            # 1D: dimension 1 scale only, no angle
            result = _read_region_1d(data_blocks, d1, metadata)
        else:
            # Fallback: parse as simple columnar data
            result = _read_region_simple(data_blocks, all_fields, metadata)

        results.append(result)

    if not results:
        raise ValueError("No valid regions found in SES file")

    return results


def _parse_data_block(text: str) -> NDArray:
    """Parse a data block text into a 2D numpy array."""
    lines = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip section headers that might be embedded
        if line.startswith("["):
            continue
        # Must start with a digit, sign, or space+digit (scientific notation)
        if line[0].isdigit() or line[0] in "+-." or (line[0] == " " and len(line) > 1):
            lines.append(line)

    if not lines:
        return np.array([]).reshape(0, 0)

    return np.loadtxt(StringIO("\n".join(lines)), dtype=np.float64)


def _read_region_1d(
    data_blocks: list[str],
    d1: NDArray,
    metadata: SpectrumMetadata,
) -> RawSpectrumData:
    """Read a 1D region (energy vs intensity, no angle)."""
    data = _parse_data_block(data_blocks[0])
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    energy = d1.astype(np.float64)

    if data.shape[1] >= 2:
        # Column 0 = energy (from data), column 1 = intensity
        specdata = data[:, 1:2]
    else:
        specdata = data[:, :1]

    angle = np.array([0.0])

    return RawSpectrumData(
        specdata=specdata,
        energy=energy,
        angle=angle,
        metadata=metadata,
    )


def _read_region_with_dimensions(
    data_blocks: list[str],
    d1: NDArray,
    d2: NDArray,
    d3: NDArray,
    d2_name: str,
    metadata: SpectrumMetadata,
) -> RawSpectrumData:
    """Read a 2D or 3D region with dimension scales."""
    n_sweeps_data = len(data_blocks)

    if len(d3) == 0:
        d3 = np.array([1.0])

    # Check for Seq. Iteration mode — swap d2 and d3
    if "Seq. Iteration" in d2_name:
        d2, d3 = d3, d2

    n_e = len(d1)
    n_a = len(d2)
    n_s = max(len(d3), n_sweeps_data)

    if n_s <= 1 and n_sweeps_data <= 1:
        # Single block: 2D data (energy x angle)
        data = _parse_data_block(data_blocks[0])
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # First column = energy, remaining columns = intensities per angle
        if data.shape[1] > 1:
            specdata = data[:, 1:]  # skip energy column
        else:
            specdata = data

        # Trim to expected size
        specdata = specdata[:n_e, :n_a] if specdata.shape[0] >= n_e and specdata.shape[1] >= n_a else specdata

        if metadata.n_slices > 1:
            specdata = np.fliplr(specdata)

        energy = d1.astype(np.float64)
        angle = np.flipud(d2).astype(np.float64)

        return RawSpectrumData(
            specdata=specdata,
            energy=energy,
            angle=angle,
            metadata=metadata,
        )
    else:
        # Multiple blocks: 3D data (energy x angle x sweep)
        specdata_3d = np.zeros((n_e, n_a, n_s), dtype=np.float64)

        for block_idx, block_text in enumerate(data_blocks):
            if block_idx >= n_s:
                break
            data = _parse_data_block(block_text)
            if data.ndim == 1:
                continue

            # First column = energy, remaining = intensities per angle
            if data.shape[1] > 1:
                block_data = data[:, 1:]
            else:
                block_data = data

            # Fit into 3D array
            rows = min(block_data.shape[0], n_e)
            cols = min(block_data.shape[1], n_a)
            specdata_3d[:rows, :cols, block_idx] = block_data[:rows, :cols]

        if metadata.n_slices > 1:
            specdata_3d = np.flip(specdata_3d, axis=1)

        # Squeeze if single sweep
        if n_s == 1:
            specdata = specdata_3d[:, :, 0]
        else:
            specdata = specdata_3d

        energy = d1.astype(np.float64)
        angle = np.flipud(d2).astype(np.float64)

        return RawSpectrumData(
            specdata=specdata,
            energy=energy,
            angle=angle,
            metadata=metadata,
        )


def _read_region_simple(
    data_blocks: list[str],
    fields: dict[str, str],
    metadata: SpectrumMetadata,
) -> RawSpectrumData:
    """Read a region with simple columnar format (no dimension scales)."""
    data = _parse_data_block(data_blocks[0])
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if data.shape[1] >= 2:
        # Multi-column: col 0 = energy, cols 1+ = intensity
        energy = data[:, 0]
        specdata = data[:, 1:]
    else:
        # Single column — try to build energy from header fields
        low_e = float(fields.get("Low Energy", "0"))
        high_e = float(fields.get("High Energy", "0"))
        n_energy = data.shape[0]
        if low_e > 0 or high_e > 0:
            energy = np.linspace(low_e, high_e, n_energy)
        else:
            energy = np.arange(n_energy, dtype=np.float64)
        specdata = data

    n_angle = specdata.shape[1]
    angle = np.arange(n_angle, dtype=np.float64)

    return RawSpectrumData(
        specdata=specdata,
        energy=energy,
        angle=angle,
        metadata=metadata,
    )


def _read_simple_text(filepath: Path) -> list[RawSpectrumData]:
    """Read simple text format (no SES header)."""
    data = np.loadtxt(filepath, dtype=np.float64)
    if data.ndim == 1:
        raise ValueError("Simple text file must have at least 2 columns")

    energy = data[:, 0]
    specdata = data[:, 1:]
    n_angle = specdata.shape[1]
    angle = np.arange(n_angle, dtype=np.float64)

    metadata = SpectrumMetadata(
        region=filepath.stem,
        energy_scale="Kinetic",  # Default; user can override
    )

    return [
        RawSpectrumData(
            specdata=specdata,
            energy=energy,
            angle=angle,
            metadata=metadata,
        )
    ]


# --- Public reader class ---


class SESTxtReader(BaseReader):
    """Reader for SES Scienta text export files (.txt).

    Handles both SES INI-like format and simple tab/space-delimited text.

    Usage:
        reader = SESTxtReader('data.txt')
        data = reader.read()
    """

    def __init__(self, filepath: str | Path):
        super().__init__(filepath)
        self._is_ses = _is_ses_format(self.filepath)
        self._data: list[RawSpectrumData] | None = None

    def _ensure_parsed(self):
        if self._data is None:
            if self._is_ses:
                self._data = _read_ses_format(self.filepath)
            else:
                self._data = _read_simple_text(self.filepath)

    @property
    def n_regions(self) -> int:
        self._ensure_parsed()
        return len(self._data)

    @property
    def region_names(self) -> list[str]:
        self._ensure_parsed()
        return [d.metadata.region or f"Region {i}" for i, d in enumerate(self._data)]

    def read(self, region_index: int = 0) -> RawSpectrumData:
        self._ensure_parsed()
        if region_index < 0 or region_index >= len(self._data):
            raise IndexError(
                f"Region index {region_index} out of range [0, {len(self._data)})"
            )
        return self._data[region_index]

    @staticmethod
    def can_read(filepath: str | Path) -> bool:
        fp = Path(filepath)
        if fp.suffix.lower() != ".txt":
            return False
        try:
            with open(fp, encoding="utf-8", errors="replace") as f:
                first_line = f.readline().strip()
                # SES format or numeric data
                return bool(first_line)
        except Exception:
            return False
