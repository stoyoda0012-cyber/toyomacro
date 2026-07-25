"""
PXT/IBW file reader for Igor Pro wave files (Scienta SES).

Supports version 3 (*.pxt, multi-region) and version 5 (*.ibw, single-region).
Based on DNNDenoiser/utils/pxt_reader.py and MATLAB Toyomacro PXTReader.
"""

from __future__ import annotations

import struct
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from toyomacro.io.readers.base_reader import (
    BaseReader,
    RawSpectrumData,
    ReaderTransform,
    ReaderWarning,
    SpectrumMetadata,
    coerce_optional_float,
    coerce_optional_int,
)

# --- Low-level binary structures ---


@dataclass
class _BinHeader:
    """Binary header (64 bytes)."""

    version: int
    formula_size: int = 0
    note_size: int = 0


@dataclass
class _WaveHeader:
    """Wave header (v3: 328 bytes, v5: 320 bytes)."""

    type: int
    n_dim: NDArray  # int32[4]
    sf_a: NDArray  # float64[4] (scale factors = X_delta)
    sf_b: NDArray  # float64[4] (offsets = X_ini)
    n_pts: int = 0


@dataclass
class _PXTRegion:
    """Raw region data from PXT file."""

    region_number: int
    n_dim_count: int
    x_delta: NDArray
    x_ini: NDArray
    x_fin: NDArray
    data: NDArray
    wave_notes: str
    wave_header: _WaveHeader
    wave_version: int = 5  # IBW binary version of this wave (3 or 5)


# --- Binary parsing functions ---

_HEADER_SIZE = {3: 64 + 328, 5: 64 + 320}

_DTYPE_MAP = {
    4: np.float64,
    2: np.float32,
    80: np.uint16,
    96: np.uint32,
}


def _read_bin_header(fid, position: int) -> _BinHeader:
    """Read binary header from file."""
    fid.seek(position)
    version = struct.unpack("<h", fid.read(2))[0]

    if version == 3:
        fid.read(14)  # skip
        formula_size = struct.unpack("<i", fid.read(4))[0]
        note_size = struct.unpack("<i", fid.read(4))[0]
    elif version == 5:
        fid.read(6)  # skip
        formula_size = struct.unpack("<i", fid.read(4))[0]
        note_size = struct.unpack("<i", fid.read(4))[0]
    else:
        raise ValueError(
            f"Unsupported PXT version {version}. Only 3 (.pxt) and 5 (.ibw) supported."
        )

    return _BinHeader(version=version, formula_size=formula_size, note_size=note_size)


def _read_wave_header(fid, position: int, version: int) -> _WaveHeader:
    """Read wave header from file."""
    offset = 88 if version == 3 else 80
    fid.seek(position + offset)

    wave_type = struct.unpack("<H", fid.read(2))[0]
    fid.read(50)  # skip

    n_dim = np.array(struct.unpack("<4i", fid.read(16)))
    sf_a = np.array(struct.unpack("<4d", fid.read(32)))
    sf_b = np.array(struct.unpack("<4d", fid.read(32)))

    active_dims = n_dim[n_dim > 0]
    n_pts = int(np.prod(active_dims)) if len(active_dims) > 0 else 0

    return _WaveHeader(type=wave_type, n_dim=n_dim, sf_a=sf_a, sf_b=sf_b, n_pts=n_pts)


def _read_data(fid, wave_header: _WaveHeader) -> NDArray:
    """Read wave data from file."""
    dtype = _DTYPE_MAP.get(wave_header.type)
    if dtype is None:
        raise ValueError(f"Unknown wave data type: {wave_header.type}")

    raw = np.frombuffer(
        fid.read(wave_header.n_pts * np.dtype(dtype).itemsize), dtype=dtype
    )

    # Reshape multi-dimensional data (Fortran order)
    active_dims = wave_header.n_dim[wave_header.n_dim > 0]
    if len(active_dims) > 1:
        raw = raw.reshape(active_dims, order="F")

    return raw


def _read_notes(fid, bin_header: _BinHeader) -> str:
    """Read wave notes from file."""
    fid.read(bin_header.formula_size)
    return fid.read(bin_header.note_size).decode("latin-1", errors="replace")


def _parse_wave_notes(notes: str) -> dict[str, Any]:
    """Parse wave notes key=value pairs."""
    result: dict[str, Any] = {}
    for line in notes.split("\r"):
        line = line.strip()
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value_str = value.strip()
            # Try numeric conversion
            try:
                if "." in value_str:
                    result[key] = float(value_str)
                else:
                    result[key] = int(value_str)
            except ValueError:
                result[key] = value_str
    return result


def _read_all_regions(filepath: Path) -> tuple[str, list[_PXTRegion]]:
    """Read all regions from a PXT/IBW file.

    PXT files use the **Igor Packed Experiment** format: a sequence of
    records, each with an 8-byte header (type:u16, version:u16, size:u32)
    followed by *size* bytes of payload.  Wave records (type=3) contain an
    IBW v5 binary wave inside.  Each spectrum region is typically followed
    by an axis-scaling wave (1D, no key=value notes).

    Records are packed **without** 2-byte alignment padding — the next
    record starts immediately at ``offset + 8 + size``.

    IBW files (v5) contain a single standalone wave (no packed-experiment
    wrapper).

    Returns:
        (container_format, regions) where container_format is
        "scienta_pxt" (packed experiment) or "igor_ibw" (standalone wave).
    """
    file_size = filepath.stat().st_size
    regions: list[_PXTRegion] = []

    with open(filepath, "rb") as fid:
        # Detect whether this is packed-experiment (.pxt) or standalone (.ibw)
        first_header = _read_bin_header(fid, 0)

        if first_header.version == 5:
            # --- Standalone IBW v5 (single wave, no packed-experiment wrapper)
            return "igor_ibw", _read_ibw_regions(fid, first_header, file_size)

        # --- Packed Experiment (.pxt) ---
        # The very first bytes are a packed-experiment record header, NOT a
        # wave bin_header.  Re-read from offset 0 as packed record.
        fid.seek(0)
        offset = 0
        region_num = 0

        while offset + 8 < file_size:
            fid.seek(offset)
            rec_header = fid.read(8)
            if len(rec_header) < 8:
                break
            rec_type, _rec_ver, rec_size = struct.unpack("<HHI", rec_header)

            # Type 0 with size 0 is the END marker
            if rec_type == 0 and rec_size == 0:
                break
            # Only process wave records (type=3)
            if rec_type != 3:
                offset += 8 + rec_size
                continue

            # --- Parse the IBW v5 wave inside this record ---
            wave_start = offset + 8  # skip packed-record header

            try:
                bin_header = _read_bin_header(fid, wave_start)
            except ValueError:
                offset += 8 + rec_size
                continue

            wave_header = _read_wave_header(fid, wave_start, bin_header.version)
            header_size = _HEADER_SIZE[bin_header.version]

            if wave_header.type not in _DTYPE_MAP or wave_header.n_pts <= 0:
                offset += 8 + rec_size
                continue

            n_dim_count = int(np.sum(wave_header.n_dim > 0))
            x_delta = wave_header.sf_a[:n_dim_count].copy()
            x_ini = wave_header.sf_b[:n_dim_count].copy()
            x_fin = x_ini + (wave_header.n_dim[:n_dim_count] - 1) * x_delta

            # Read data + notes from within the wave
            fid.seek(wave_start + header_size)
            data = _read_data(fid, wave_header)
            notes = _read_notes(fid, bin_header)

            # Advance to next packed-experiment record (no padding)
            offset += 8 + rec_size

            # Skip axis scaling waves: 1D data with empty/non-key=value notes
            parsed_notes = _parse_wave_notes(notes)
            if n_dim_count <= 1 and not parsed_notes:
                continue

            region_num += 1
            regions.append(
                _PXTRegion(
                    region_number=region_num,
                    n_dim_count=n_dim_count,
                    x_delta=x_delta,
                    x_ini=x_ini,
                    x_fin=x_fin,
                    data=data,
                    wave_notes=notes,
                    wave_header=wave_header,
                    wave_version=bin_header.version,
                )
            )

    return "scienta_pxt", regions


def _read_ibw_regions(fid, first_header: _BinHeader, file_size: int) -> list[_PXTRegion]:
    """Read a single region from a standalone IBW v5 file."""
    header_size = _HEADER_SIZE[5]
    wave_header = _read_wave_header(fid, 0, 5)

    if wave_header.type not in _DTYPE_MAP or wave_header.n_pts <= 0:
        return []

    n_dim_count = int(np.sum(wave_header.n_dim > 0))
    x_delta = wave_header.sf_a[:n_dim_count].copy()
    x_ini = wave_header.sf_b[:n_dim_count].copy()
    x_fin = x_ini + (wave_header.n_dim[:n_dim_count] - 1) * x_delta

    fid.seek(header_size)
    data = _read_data(fid, wave_header)
    notes = _read_notes(fid, first_header)

    return [
        _PXTRegion(
            region_number=1,
            n_dim_count=n_dim_count,
            x_delta=x_delta,
            x_ini=x_ini,
            x_fin=x_fin,
            data=data,
            wave_notes=notes,
            wave_header=wave_header,
            wave_version=5,
        )
    ]


# --- Metadata helpers ---

# Wave-note keys mapped onto SpectrumMetadata standard fields. Any other
# key survives verbatim in metadata.vendor_metadata.
_NOTE_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %H:%M:%S",
)


def _parse_note_datetime(
    notes: dict[str, Any], source: str
) -> tuple[datetime | None, set[str]]:
    """Type-safe Date/Time extraction from wave notes.

    Returns (datetime or None, set of note keys consumed). On an
    unrecognized format the raw values stay in vendor_metadata and a
    ReaderWarning is emitted.
    """
    date_v = notes.get("Date")
    time_v = notes.get("Time")
    if date_v is None or time_v is None:
        return None, set()
    text = f"{date_v} {time_v}".strip()
    for fmt in _NOTE_DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt), {"Date", "Time"}
        except ValueError:
            pass
    warnings.warn(
        f"{source}: unrecognized Date/Time format: {text!r}",
        ReaderWarning,
        stacklevel=2,
    )
    return None, set()


def _infer_detector_role(lens_mode: str) -> str:
    """Role of the non-energy detector dimension, from the lens mode.

    Only unambiguous lens-mode families are mapped; raw Scienta mode
    strings like "T_HiPPHAXPES" stay "unknown" rather than being guessed.
    """
    lm = lens_mode.strip().lower()
    if lm.startswith("angular"):
        return "emission_angle"
    if lm.startswith("transmission"):
        return "position"
    return "unknown"


# --- Public reader class ---


class PXTReader(BaseReader):
    """Reader for Scienta SES PXT (v3) and Igor IBW (v5) binary files.

    Usage:
        reader = PXTReader('data.pxt')
        print(reader.n_regions, reader.region_names)
        data = reader.read(region_index=0)
    """

    def __init__(self, filepath: str | Path):
        super().__init__(filepath)
        self._regions: list[_PXTRegion] | None = None
        self._container_format: str = "unknown"

    def _ensure_parsed(self):
        if self._regions is None:
            self._container_format, self._regions = _read_all_regions(self.filepath)

    @property
    def n_regions(self) -> int:
        self._ensure_parsed()
        return len(self._regions)

    @property
    def region_names(self) -> list[str]:
        self._ensure_parsed()
        names = []
        for r in self._regions:
            notes = _parse_wave_notes(r.wave_notes)
            name = notes.get("Region Name", notes.get("RegionName", f"Region {r.region_number}"))
            names.append(str(name))
        return names

    def read(self, region_index: int = 0) -> RawSpectrumData:
        self._ensure_parsed()

        if region_index < 0 or region_index >= len(self._regions):
            raise IndexError(
                f"Region index {region_index} out of range [0, {len(self._regions)})"
            )

        region = self._regions[region_index]
        notes = _parse_wave_notes(region.wave_notes)
        src = f"{self.filepath.name}[region {region_index}]"

        # Type-safe extraction of standard fields. Keys whose value was
        # actually adopted are tracked so everything else survives in
        # vendor_metadata (a malformed "Pass Energy=abc" stays there too).
        consumed: set[str] = set()

        def note_value(*keys: str) -> Any:
            for k in keys:
                if k in notes:
                    consumed.add(k)
                    return notes[k]
            return None

        def note_float(*keys: str) -> float | None:
            for k in keys:
                if k in notes:
                    value = coerce_optional_float(notes[k], k, src)
                    if value is not None:
                        consumed.add(k)
                    return value
            return None

        def note_int(*keys: str, default: int) -> int:
            for k in keys:
                if k in notes:
                    value = coerce_optional_int(notes[k], k, src)
                    if value is not None:
                        consumed.add(k)
                        return value
                    return default
            return default

        region_name = str(note_value("Region Name", "RegionName") or "")
        excitation_energy = note_float("Excitation Energy", "ExcitationEnergy")
        energy_scale = str(note_value("Energy Scale", "EnergyScale") or "Kinetic")
        lens_mode = str(note_value("Lens Mode", "LensMode") or "")
        n_slices = note_int("Number of Slices", "NumberOfSlices", default=1)
        n_sweeps = note_int(
            "Number of Sweeps", "NumberOfSweeps", "NumScans", default=1
        )
        acquisition_mode = str(note_value("Acquisition Mode", "AcquisitionMode") or "")
        pass_energy = note_float("Pass Energy", "PassEnergy")
        dt, dt_consumed = _parse_note_datetime(notes, src)
        consumed |= dt_consumed

        # Vendor metadata: parsed key=value pairs not adopted above, plus
        # free-text note lines that are not key=value at all.
        vendor: dict[str, Any] = {
            k: v for k, v in notes.items() if k not in consumed
        }
        unparsed_lines = [
            line.strip()
            for line in region.wave_notes.split("\r")
            if line.strip() and "=" not in line
        ]
        if unparsed_lines:
            vendor["unparsed_notes"] = unparsed_lines

        # Original dimension structure as stored in the wave
        original_shape = tuple(
            int(d) for d in region.wave_header.n_dim if d > 0
        )
        roles = ["energy"]
        if len(original_shape) >= 2:
            roles.append(_infer_detector_role(lens_mode))
        # Further dimensions (sweep? frame? position?) cannot be confirmed
        # from wave notes — never guess.
        roles.extend("unknown" for _ in original_shape[2:])

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
            source_format=self._container_format,
            source_format_version=str(region.wave_version),
            source_region_index=region_index,
            vendor_metadata=vendor,
            intensity_semantics="unknown",
            intensity_unit="unknown",
            original_shape=original_shape,
            dimension_roles=tuple(roles),
        )

        transforms: list[ReaderTransform] = []

        # Energy axis (dimension 0)
        n_energy = region.wave_header.n_dim[0]
        energy = np.linspace(region.x_ini[0], region.x_fin[0], n_energy)

        # Handle dimensions
        data = region.data.copy()

        if region.n_dim_count == 1:
            # 1D: single spectrum, no angle
            specdata = data.reshape(-1, 1)
            angle = np.array([0.0])
        elif region.n_dim_count == 2:
            # 2D: energy x angle
            if data.ndim == 1:
                n_angle = region.wave_header.n_dim[1]
                data = data.reshape(n_energy, n_angle, order="F")
            specdata = data
            if metadata.n_slices > 1:
                specdata = np.fliplr(specdata)
                transforms.append(
                    ReaderTransform(
                        name="angle_axis_reversal",
                        parameters={"target": "specdata", "axis": 1},
                        source="pxt_reader",
                        reason="Scienta slice order convention (Number of Slices > 1)",
                    )
                )
            n_angle = specdata.shape[1]
            angle_ini = region.x_ini[1] if len(region.x_ini) > 1 else 0.0
            angle_fin = region.x_fin[1] if len(region.x_fin) > 1 else float(n_angle - 1)
            angle = np.flipud(np.linspace(angle_ini, angle_fin, n_angle))
            transforms.append(
                ReaderTransform(
                    name="angle_axis_reversal",
                    parameters={
                        "target": "angle_values",
                        "scale_start": float(angle_ini),
                        "scale_stop": float(angle_fin),
                    },
                    source="pxt_reader",
                    reason="legacy Toyomacro convention: angle scale stored descending",
                )
            )
        elif region.n_dim_count >= 3:
            # 3D+: energy x angle x sweep (or energy x sweep if no angle)
            if data.ndim < 3:
                dims = region.wave_header.n_dim[region.wave_header.n_dim > 0]
                data = data.reshape(dims, order="F")
            specdata = data
            if metadata.n_slices > 1 and data.ndim >= 2:
                specdata = np.flip(specdata, axis=1)
                transforms.append(
                    ReaderTransform(
                        name="angle_axis_reversal",
                        parameters={"target": "specdata", "axis": 1},
                        source="pxt_reader",
                        reason="Scienta slice order convention (Number of Slices > 1)",
                    )
                )
            n_angle = specdata.shape[1] if specdata.ndim >= 2 else 1
            angle_ini = region.x_ini[1] if len(region.x_ini) > 1 else 0.0
            angle_fin = region.x_fin[1] if len(region.x_fin) > 1 else float(n_angle - 1)
            angle = np.flipud(np.linspace(angle_ini, angle_fin, n_angle))
            transforms.append(
                ReaderTransform(
                    name="angle_axis_reversal",
                    parameters={
                        "target": "angle_values",
                        "scale_start": float(angle_ini),
                        "scale_stop": float(angle_fin),
                    },
                    source="pxt_reader",
                    reason="legacy Toyomacro convention: angle scale stored descending",
                )
            )
        else:
            raise ValueError(f"Unsupported dimension count: {region.n_dim_count}")

        if specdata.dtype != np.float64:
            transforms.append(
                ReaderTransform(
                    name="dtype_conversion",
                    parameters={"from": str(specdata.dtype), "to": "float64"},
                    source="pxt_reader",
                    reason="unified float64 output",
                )
            )

        return RawSpectrumData(
            specdata=specdata.astype(np.float64),
            energy=energy.astype(np.float64),
            angle=angle.astype(np.float64),
            metadata=metadata,
            transforms=tuple(transforms),
        )

    @staticmethod
    def can_read(filepath: str | Path) -> bool:
        fp = Path(filepath)
        if fp.suffix.lower() not in (".pxt", ".ibw"):
            return False
        try:
            with open(fp, "rb") as f:
                version = struct.unpack("<h", f.read(2))[0]
                return version in (3, 5)
        except Exception:
            return False
