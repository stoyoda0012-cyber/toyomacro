"""
Base reader ABC and common data structures for XPS file format readers.

All format-specific readers (PXT, SES, VAMAS, NPL) inherit from BaseReader
and produce RawSpectrumData as output.
"""

from __future__ import annotations

import copy
import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


class ReaderWarning(UserWarning):
    """Recoverable metadata problem in an instrument file.

    Raised as a warning (never silently swallowed) when a metadata field
    is present but malformed. The array data is still returned; the
    affected field falls back to None / its structural default.
    """


IntensitySemantics = Literal[
    "raw_counts",
    "count_rate",
    "corrected_intensity",
    "arbitrary",
    "unknown",
]

#: Whether the analyzer transmission has been divided out of the stored
#: intensity. "unknown" unless a file states it or the user declares it.
TransmissionApplied = Literal["applied", "not_applied", "unknown"]
#: Where a transmission curve for this spectrum comes from, if anywhere.
TransmissionCurve = Literal["embedded", "user_supplied", "none"]
#: What the applied state rests on.
TransmissionBasis = Literal["file", "user", "vendor_convention", "unknown"]


@dataclass(frozen=True, eq=False)
class CorrespondingVariable:
    """One ordinate series stored alongside the energy axis in a file.

    VAMAS calls these "corresponding variables": a block may carry several
    (intensity, a transmission curve, ...), each with its own label and
    unit, point for point on the same abscissa.

    Attributes:
        label: The label as written in the file (e.g. ``"Counts"``,
            ``"Transmission"``).
        unit: The unit as written in the file (e.g. ``"d"``), not
            interpreted.
        values: One value per energy point, aligned with
            ``RawSpectrumData.energy``.
    """

    label: str
    unit: str
    values: NDArray[np.float64]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CorrespondingVariable):
            return NotImplemented
        return (self.label == other.label and self.unit == other.unit
                and np.array_equal(self.values, other.values))

    __hash__ = None  # type: ignore[assignment]  # holds an array


#: Where a metadata value came from. A field with no recorded origin is
#: not known to be a fact: it may be a structural default (``n_sweeps=1``).
ValueOrigin = Literal["file", "user", "inferred"]

#: Fields whose origin is recorded and which are persisted to HDF5 only
#: together with that origin.
ORIGIN_FIELDS = (
    "excitation_energy",
    "pass_energy",
    "lens_mode",
    "n_sweeps",
    "n_slices",
    "signal_mode",
    "signal_collection_time",
    "transmission_curve",
    "transmission_curve_variable",
    "transmission_curve_normalisation",
)

#: Fields a user may declare with :meth:`RawSpectrumData.declare`.
DECLARABLE_FIELDS = (*ORIGIN_FIELDS, "transmission_applied")


#: Vocabulary for SpectrumMetadata.dimension_roles entries.
DIMENSION_ROLES = (
    "energy",
    "emission_angle",
    "position",
    "frame",
    "sweep",
    "time",
    "unknown",
)


@dataclass(frozen=True)
class ReaderTransform:
    """One transformation applied to the data after reading it from file.

    Readers and the importer record every operation that changes the
    arrays relative to what the instrument file contains (axis reversal,
    KE/BE conversion, flattening, sweep integration, dtype conversion),
    so downstream consumers can reconstruct or re-interpret the raw data.

    Attributes:
        name: Transform identifier, e.g. "energy_scale_conversion",
            "angle_axis_reversal", "dimension_flattening",
            "sweep_integration", "dtype_conversion".
        parameters: JSON-safe key parameters of the operation
            (input/output shapes, hv, axis, ...). Defensively copied.
        source: Which component applied it, e.g. "pxt_reader", "importer".
        reason: Why it was applied (convention, user request, ...).
    """

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    source: str = ""
    reason: str = ""

    def __post_init__(self):
        # Defensive copy: later caller-side mutation must not rewrite history.
        object.__setattr__(self, "parameters", copy.deepcopy(dict(self.parameters)))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict form (parameters copied)."""
        return {
            "name": self.name,
            "parameters": copy.deepcopy(dict(self.parameters)),
            "source": self.source,
            "reason": self.reason,
        }


@dataclass
class SpectrumMetadata:
    """Standardized metadata from raw XPS data files.

    Mirrors MATLAB BaseReader.createEmptyMetadata(), extended with
    provenance fields. Values state facts read from the instrument file;
    ``None`` means "not stated in the file" and is never collapsed to 0.
    """

    region: str = ""
    datetime: datetime | None = None
    excitation_energy: float | None = None  # eV; None = not stated in the file
    energy_scale: str = ""  # "Binding" or "Kinetic"
    lens_mode: str = ""  # "Transmission" or "Angular"
    n_slices: int = 1
    n_sweeps: int = 1
    acquisition_mode: str = ""  # "Swept" or "Fixed"
    pass_energy: float | None = None  # eV; None = not stated in the file
    comments: str = ""
    # --- Provenance ---
    source_format: str = "unknown"  # e.g. "scienta_pxt", "igor_ibw", "vamas"
    source_format_version: str | None = None
    source_region_index: int = 0
    #: Key/value pairs read from the file but not adopted into the standard
    #: fields above. May contain free-text (sample names, user names, paths);
    #: kept in memory only — not persisted to the Toyomacro HDF5 schema.
    vendor_metadata: Mapping[str, Any] = field(default_factory=dict)
    intensity_semantics: IntensitySemantics = "unknown"
    intensity_unit: str = "unknown"
    #: Shape of the data array as stored in the instrument file,
    #: before any reshape/flatten by reader or importer.
    original_shape: tuple[int, ...] = ()
    #: Physical role of each original dimension (see DIMENSION_ROLES).
    #: "unknown" when the file metadata cannot confirm the role.
    dimension_roles: tuple[str, ...] = ()
    # --- Acquisition facts as written, not interpreted ---
    #: Signal mode as written (VAMAS: e.g. "pulse counting"). Says how the
    #: detector worked, not what the stored numbers are.
    signal_mode: str | None = None
    #: Signal collection time in seconds as written (VAMAS). Whether it is
    #: per scan or summed over ``n_sweeps`` is not stated by the format.
    signal_collection_time: float | None = None
    # --- Analyzer transmission: four separate facts ---
    #: Whether the stored intensity has been divided by a transmission curve.
    #: A curve stored with the data does not by itself say this.
    transmission_applied: TransmissionApplied = "unknown"
    #: Whether a curve accompanies the data, and from where.
    transmission_curve: TransmissionCurve = "none"
    #: Label of the corresponding variable holding the curve, when embedded.
    transmission_curve_variable: str | None = None
    #: What the curve is normalised to ("unknown" unless stated).
    transmission_curve_normalisation: str = "unknown"
    #: What ``transmission_applied`` rests on.
    transmission_basis: TransmissionBasis = "unknown"
    #: Origin of each field in :data:`ORIGIN_FIELDS` whose value is known:
    #: ``"file"`` (read from the file), ``"user"`` (declared, see
    #: :meth:`RawSpectrumData.declare`) or ``"inferred"``. A field absent
    #: here is not a known fact, whatever its value.
    field_origins: Mapping[str, ValueOrigin] = field(default_factory=dict)

    def __post_init__(self):
        # Defensive copy: mutating the caller's dict must not change us.
        self.vendor_metadata = copy.deepcopy(dict(self.vendor_metadata))
        self.field_origins = dict(self.field_origins)
        self.original_shape = tuple(int(d) for d in self.original_shape)
        self.dimension_roles = tuple(str(r) for r in self.dimension_roles)


@dataclass
class RawSpectrumData:
    """Unified output from all format readers.

    Attributes:
        specdata: Intensity array [nEnergy x nAngle] or [nEnergy x nAngle x nSweep]
        energy: Energy axis [nEnergy]
        angle: Angle axis [nAngle] (scalar 0 if not angle-resolved)
        metadata: Parsed metadata
        transforms: Operations applied to the arrays since reading the file,
            in application order (reader conventions first, importer last).
        corresponding_variables: All ordinate series stored for the
            spectrum (see :class:`CorrespondingVariable`).
    """

    specdata: NDArray[np.float64]
    energy: NDArray[np.float64]
    angle: NDArray[np.float64]
    metadata: SpectrumMetadata
    transforms: tuple[ReaderTransform, ...] = ()
    #: Every ordinate series the file stores for this spectrum, in file
    #: order; the first is the one in ``specdata``. Empty for formats that
    #: store a single series.
    corresponding_variables: tuple[CorrespondingVariable, ...] = ()

    def declare(self, field_name: str, value: Any, reason: str) -> None:
        """Set a metadata fact on the user's authority, and record it.

        The field's origin becomes ``"user"`` and a ``user_declaration``
        transform keeps the previous value and origin, so the override is
        never mistaken for something the file said. Declaring
        ``transmission_applied`` also sets ``transmission_basis="user"``.

        Args:
            field_name: One of :data:`DECLARABLE_FIELDS`.
            value: The declared value (JSON-safe).
            reason: Why the user knows it (kept in the record).
        """
        if field_name not in DECLARABLE_FIELDS:
            raise ValueError(f"{field_name!r} cannot be declared; one of {DECLARABLE_FIELDS}")
        if not reason:
            raise ValueError("a declaration needs a reason")
        md = self.metadata
        previous = getattr(md, field_name)
        previous_origin = (md.transmission_basis if field_name == "transmission_applied"
                           else md.field_origins.get(field_name))
        setattr(md, field_name, value)
        if field_name == "transmission_applied":
            md.transmission_basis = "user"
        else:
            md.field_origins[field_name] = "user"
        self.record_transform(
            "user_declaration",
            parameters={"field": field_name, "value": value, "previous": previous,
                        "previous_origin": previous_origin},
            source="user",
            reason=reason,
        )

    def record_transform(
        self,
        name: str,
        parameters: Mapping[str, Any] | None = None,
        source: str = "",
        reason: str = "",
    ) -> None:
        """Append a transform record (stored as an immutable tuple)."""
        self.transforms = (
            *self.transforms,
            ReaderTransform(
                name=name,
                parameters=parameters or {},
                source=source,
                reason=reason,
            ),
        )


# --- Type-safe metadata coercion helpers (shared by format readers) ---


def coerce_optional_float(value: Any, key: str, source: str) -> float | None:
    """Convert a metadata value to float; warn and return None if malformed."""
    if value is None:
        return None
    try:
        result = float(value) if isinstance(value, (int, float)) else float(str(value).strip())
        if not math.isfinite(result):
            raise ValueError
        return result
    except (TypeError, ValueError):
        warnings.warn(
            f"{source}: malformed value for '{key}': {value!r} (treated as unknown)",
            ReaderWarning,
            stacklevel=3,
        )
        return None


def coerce_optional_int(value: Any, key: str, source: str) -> int | None:
    """Convert a metadata value to int; warn and return None if malformed."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        numeric = float(str(value).strip())
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError
        return int(numeric)
    except (TypeError, ValueError):
        warnings.warn(
            f"{source}: malformed value for '{key}': {value!r} (treated as unknown)",
            ReaderWarning,
            stacklevel=3,
        )
        return None
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
