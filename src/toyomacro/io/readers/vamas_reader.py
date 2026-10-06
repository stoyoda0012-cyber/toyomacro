"""
VAMAS file reader (ISO 14976 format).

VAMAS files are line-oriented text, one field per line, in which comment
lines, experimental variables, corresponding variables and additional
parameters are each preceded by their count. The reader follows those
counts; an empty line is a field (an empty transition label is common),
never something to skip.

Supported (v0.5.0): experiment mode ``NORM``, scan mode ``REGULAR``, an
empty parameter-inclusion list and no future-upgrade entries; within such
a file, blocks whose technique is ``XPS``, whose abscissa is kinetic or
binding energy in eV, and which carry at least one corresponding
variable. Verified on the 15 files of Zenodo 10.5281/zenodo.7074887
(Thermo K-Alpha and Kratos Axis Ultra data written by CasaXPS 2.3.24 and
2.3.25, and three without a version line). Any other experiment or scan
mode, or a count that cannot be read, stops the whole file with a
ValueError, because the block boundaries can no longer be known. A block
whose structure is read but whose content is outside that scope, or
whose ordinate values do not parse, is not returned and is listed in
:attr:`VAMASReader.skipped_blocks` with the reason.

``layout="legacy_fixed"`` keeps the fixed-line layout carried over from
the MATLAB-era reader for Omicron files. It does not follow ISO 14976
(it reads the sixth header line, the comment-line count, as the number
of regions) and has not been verified against a real file in this
repository; use it only to reproduce earlier results.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from toyomacro.io.readers.base_reader import (
    BaseReader,
    CorrespondingVariable,
    RawSpectrumData,
    ReaderWarning,
    SpectrumMetadata,
)

#: CasaXPS writes 1e37 where a numeric field is not known. Applied only to
#: the scalar header fields read into metadata, never to ordinate values.
_NOT_KNOWN = 1e37

_ENERGY_AXES = {"kinetic energy": "Kinetic", "binding energy": "Binding"}


@dataclass(frozen=True)
class SkippedBlock:
    """A VAMAS block that was read structurally but not returned.

    Attributes:
        index: 0-based position of the block in the file.
        block_id: The block identifier as written.
        reason: Why it was not returned.
    """

    index: int
    block_id: str
    reason: str


class _Lines:
    """The file's lines, consumed one field at a time."""

    def __init__(self, text: str, source: str):
        self._lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
        self._pos = 0
        self._source = source

    def next(self, what: str) -> str:
        if self._pos >= len(self._lines):
            raise ValueError(f"{self._source}: file ends while reading {what}")
        line = self._lines[self._pos].replace("\x00", "")
        self._pos += 1
        return line

    def count(self, what: str) -> int:
        raw = self.next(what)
        try:
            n = int(raw)
        except ValueError:
            raise ValueError(
                f"{self._source}: {what} is {raw!r}, not an integer; "
                "the block boundaries cannot be determined"
            ) from None
        if n < 0:
            raise ValueError(f"{self._source}: {what} is negative ({n})")
        return n

    def take(self, n: int, what: str) -> list[str]:
        return [self.next(what) for _ in range(n)]

    def rest(self) -> list[str]:
        return [ln for ln in self._lines[self._pos:] if ln]


def _known_float(raw: str) -> float | None:
    """A header number, or None for an unparsable or not-known (1e37) value."""
    try:
        value = float(raw)
    except ValueError:
        return None
    if not np.isfinite(value) or abs(value) >= _NOT_KNOWN:
        return None
    return value


def _acquisition_datetime(fields: list[str]) -> tuple[datetime | None, str | None]:
    """(datetime, note). A two-digit year is left unresolved, not given a century."""
    try:
        year, month, day, hour, minute, second = (int(f) for f in fields)
    except ValueError:
        return None, "unparsable"
    if year < 100:
        return None, "two_digit_year_unresolved"
    try:
        return datetime(year, month, day, hour, minute, second), None
    except ValueError:
        return None, "invalid"


def _read_iso(filepath: Path) -> tuple[list[tuple[RawSpectrumData, str]], list[SkippedBlock]]:
    src = str(filepath.name)
    with open(filepath, encoding="utf-8", errors="replace") as fid:
        lines = _Lines(fid.read(), src)

    format_id = lines.next("format identifier")
    if "VAMAS" not in format_id:
        raise ValueError(f"{src}: first line is not a VAMAS format identifier")
    header_ids = dict(zip(
        ("institution", "instrument_model", "operator", "experiment"),
        lines.take(4, "file header identifiers"),
    ))
    file_comment = lines.take(lines.count("number of file comment lines"), "file comment")
    exp_mode = lines.next("experiment mode")
    scan_mode = lines.next("scan mode")
    if exp_mode != "NORM" or scan_mode != "REGULAR":
        raise ValueError(
            f"{src}: experiment mode {exp_mode!r} / scan mode {scan_mode!r} is not "
            "supported (NORM / REGULAR only)"
        )
    lines.count("number of spectral regions")
    n_exp = lines.count("number of experimental variables")
    exp_vars = [(lines.next("exp. variable label"), lines.next("exp. variable unit"))
                for _ in range(n_exp)]
    if lines.count("number of parameter-inclusion entries") != 0:
        raise ValueError(f"{src}: a parameter inclusion/exclusion list is not supported")
    lines.count("number of manually entered items")
    n_future_exp = lines.count("number of future upgrade experiment entries")
    n_future_block = lines.count("number of future upgrade block entries")
    if n_future_exp or n_future_block:
        raise ValueError(f"{src}: future-upgrade entries are not supported")
    n_blocks = lines.count("number of blocks")

    results: list[tuple[RawSpectrumData, str]] = []
    skipped: list[SkippedBlock] = []
    for index in range(n_blocks):
        where = f"block {index}"
        block_id = lines.next(f"{where} identifier")
        sample_id = lines.next(f"{where} sample identifier")
        date_fields = lines.take(6, f"{where} date")
        gmt_offset = lines.next(f"{where} GMT offset")
        block_comment = lines.take(lines.count(f"{where} comment line count"), f"{where} comment")
        technique = lines.next(f"{where} technique")
        exp_values = lines.take(n_exp, f"{where} experimental variable values")
        source_label = lines.next(f"{where} source label")
        source_energy = _known_float(lines.next(f"{where} source energy"))
        source_strength = _known_float(lines.next(f"{where} source strength"))
        lines.take(2, f"{where} beam width")
        incidence = lines.take(2, f"{where} angle of incidence")
        analyser_mode = lines.next(f"{where} analyser mode")
        pass_or_ratio = _known_float(lines.next(f"{where} pass energy or retard ratio"))
        if technique == "AES diff":
            lines.next(f"{where} differential width")
        magnification = _known_float(lines.next(f"{where} magnification"))
        work_function = _known_float(lines.next(f"{where} work function"))
        lines.next(f"{where} target bias")
        lines.take(2, f"{where} analysis width")
        takeoff = [_known_float(v) for v in lines.take(2, f"{where} take-off angles")]
        species = lines.next(f"{where} species label")
        transition = lines.next(f"{where} transition label")
        lines.next(f"{where} particle charge")
        x_label = lines.next(f"{where} abscissa label")
        x_unit = lines.next(f"{where} abscissa unit")
        x_start_raw = lines.next(f"{where} abscissa start")
        x_step_raw = lines.next(f"{where} abscissa increment")
        n_corr = lines.count(f"{where} number of corresponding variables")
        corr = [(lines.next(f"{where} variable label"), lines.next(f"{where} variable unit"))
                for _ in range(n_corr)]
        signal_mode = lines.next(f"{where} signal mode")
        collection_time = _known_float(lines.next(f"{where} signal collection time"))
        n_scans_raw = lines.next(f"{where} number of scans")
        lines.next(f"{where} signal time correction")
        lines.take(3, f"{where} sample tilt and rotation")
        n_add = lines.count(f"{where} number of additional parameters")
        additional = [
            {"label": lines.next(f"{where} parameter label"),
             "unit": lines.next(f"{where} parameter unit"),
             "value": lines.next(f"{where} parameter value")}
            for _ in range(n_add)
        ]
        n_ord = lines.count(f"{where} number of ordinate values")
        lines.take(2 * n_corr, f"{where} ordinate min/max")
        raw_values = lines.take(n_ord, f"{where} ordinate values")

        def skip(reason: str, block_id: str = block_id, index: int = index) -> None:
            skipped.append(SkippedBlock(index=index, block_id=block_id, reason=reason))

        # From here the block's extent is known: problems reject this
        # block only.
        if technique != "XPS":
            skip(f"technique {technique!r} is not XPS")
            continue
        scale = _ENERGY_AXES.get(x_label.lower())
        if scale is None or x_unit != "eV":
            skip(f"abscissa {x_label!r} [{x_unit}] is not kinetic or binding energy in eV")
            continue
        if n_corr == 0:
            skip("no corresponding variable")
            continue
        if n_ord % n_corr:
            skip(f"{n_ord} ordinate values do not divide into {n_corr} variables")
            continue
        try:
            x_start, x_step = float(x_start_raw), float(x_step_raw)
            values = np.array([float(v) for v in raw_values], dtype=np.float64)
        except ValueError:
            skip("an abscissa or ordinate value does not parse as a number")
            continue
        if not np.all(np.isfinite(values)):
            skip("non-finite ordinate value")
            continue
        try:
            n_scans: int | None = int(n_scans_raw)
        except ValueError:
            n_scans = None

        n_points = n_ord // n_corr
        series = values.reshape(n_points, n_corr)
        variables = tuple(
            CorrespondingVariable(label=label, unit=unit, values=series[:, k].copy())
            for k, (label, unit) in enumerate(corr)
        )
        energy = x_start + x_step * np.arange(n_points, dtype=np.float64)
        dt, date_note = _acquisition_datetime(date_fields)

        transmission = next(
            (v.label for v in variables[1:] if v.label.lower() == "transmission"), None
        )
        vendor: dict[str, Any] = {
            "block_id": block_id,
            "sample_id": sample_id,
            "technique": technique,
            "species_label": species,
            "transition_label": transition,
            "source_label": source_label,
            "source_strength": source_strength,
            "analyser_mode": analyser_mode,
            "magnification": magnification,
            "work_function_eV": work_function,
            "angle_of_incidence": incidence,
            "takeoff_polar_deg": takeoff[0],
            "takeoff_azimuth_deg": takeoff[1],
            "experimental_variables": {
                label: value for (label, _), value in zip(exp_vars, exp_values)
            },
            "additional_parameters": additional,
            "date_raw": " ".join(date_fields) + f" (GMT offset {gmt_offset})",
            "date_note": date_note,
            "header": header_ids,
            # Free text: may carry paths and names; never persisted.
            "unparsed_notes": [*file_comment, *block_comment],
        }
        if analyser_mode != "FAT":
            vendor["retard_ratio"] = pass_or_ratio

        metadata = SpectrumMetadata(
            region=block_id,
            datetime=dt,
            excitation_energy=source_energy,
            energy_scale=scale,
            n_sweeps=n_scans if n_scans is not None else 1,
            pass_energy=pass_or_ratio if analyser_mode == "FAT" else None,
            source_format="vamas",
            source_format_version=format_id,
            source_region_index=index,
            vendor_metadata=vendor,
            intensity_semantics="unknown",
            intensity_unit="unknown",
            original_shape=(int(n_points),),
            dimension_roles=("energy",),
            signal_mode=signal_mode or None,
            signal_collection_time=collection_time,
            transmission_applied="unknown",
            transmission_curve="embedded" if transmission else "none",
            transmission_curve_variable=transmission,
            transmission_curve_normalisation="unknown",
            transmission_basis="unknown",
        )
        data = RawSpectrumData(
            specdata=variables[0].values.reshape(-1, 1),
            energy=energy,
            angle=np.array([0.0]),
            metadata=metadata,
            corresponding_variables=variables,
        )
        results.append((data, block_id))

    trailer = lines.rest()
    if not trailer or trailer[0].lower() != "end of experiment":
        warnings.warn(
            f"{src}: no 'end of experiment' line after {n_blocks} blocks",
            ReaderWarning,
            stacklevel=3,
        )
    return results, skipped


def _read_lines(fid, n: int) -> list[str]:
    """Read n non-empty lines from file (legacy fixed layout only)."""
    lines: list[str] = []
    while len(lines) < n:
        line = fid.readline()
        if not line:
            break
        stripped = line.strip().replace("\x00", "")
        if stripped:
            lines.append(stripped)
    return lines


def _read_legacy_fixed(filepath: Path) -> list[tuple[RawSpectrumData, str]]:
    """The MATLAB-era fixed-line layout. Not ISO 14976; see the module docstring."""
    results: list[tuple[RawSpectrumData, str]] = []

    with open(filepath, encoding="utf-8", errors="replace") as fid:
        # File header: 12 lines
        file_header = _read_lines(fid, 12)
        n_regions = int(file_header[5])

        format_version = file_header[0] if file_header and "VAMAS" in file_header[0] else None

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

            energy = np.arange(n_energy, dtype=np.float64) * e_step + e_ini
            data_lines = _read_lines(fid, n_energy)
            intensity = np.array([float(line) for line in data_lines], dtype=np.float64)

            metadata = SpectrumMetadata(
                region=display_name,
                datetime=dt,
                # Excitation and pass energy are not parsed from this
                # layout — unknown, not 0.
                excitation_energy=None,
                energy_scale="Kinetic",
                lens_mode="Angular",
                n_slices=n_slices,
                n_sweeps=n_sweeps,
                pass_energy=None,
                source_format="vamas",
                source_format_version=format_version,
                source_region_index=region_idx,
                intensity_semantics="unknown",
                intensity_unit="unknown",
                original_shape=(int(n_energy),),
                dimension_roles=("energy",),
            )
            results.append((
                RawSpectrumData(
                    specdata=intensity.reshape(-1, 1),
                    energy=energy,
                    angle=np.array([0.0]),
                    metadata=metadata,
                ),
                display_name,
            ))

    return results


class VAMASReader(BaseReader):
    """Reader for VAMAS (ISO 14976) .vms files.

    Args:
        filepath: Path to the .vms file.
        layout: ``"iso"`` (default) follows ISO 14976 within the scope in
            the module docstring. ``"legacy_fixed"`` is the unverified
            MATLAB-era fixed-line layout, kept to reproduce earlier results.

    Regions are the returned blocks, in file order; ``read(i)`` and
    ``region_names[i]`` index them, and each block's position in the file
    is ``metadata.source_region_index``. Blocks read but not returned are
    in :attr:`skipped_blocks`.
    """

    LAYOUTS = ("iso", "legacy_fixed")

    def __init__(self, filepath: str | Path, *, layout: str = "iso"):
        super().__init__(filepath)
        if layout not in self.LAYOUTS:
            raise ValueError(f"layout must be one of {self.LAYOUTS}, got {layout!r}")
        self.layout = layout
        self._results: list[tuple[RawSpectrumData, str]] | None = None
        self._skipped: list[SkippedBlock] = []

    def _ensure_parsed(self):
        if self._results is not None:
            return
        if self.layout == "legacy_fixed":
            self._results = _read_legacy_fixed(self.filepath)
            return
        self._results, self._skipped = _read_iso(self.filepath)
        if self._skipped:
            warnings.warn(
                f"{self.filepath.name}: {len(self._skipped)} block(s) not returned "
                "(see VAMASReader.skipped_blocks): "
                + "; ".join(f"#{b.index} {b.block_id!r}: {b.reason}" for b in self._skipped[:3])
                + (" ..." if len(self._skipped) > 3 else ""),
                ReaderWarning,
                stacklevel=3,
            )

    @property
    def n_regions(self) -> int:
        self._ensure_parsed()
        return len(self._results)

    @property
    def region_names(self) -> list[str]:
        self._ensure_parsed()
        return [name for _, name in self._results]

    @property
    def skipped_blocks(self) -> tuple[SkippedBlock, ...]:
        """Blocks read structurally but not returned, with the reason."""
        self._ensure_parsed()
        return tuple(self._skipped)

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
