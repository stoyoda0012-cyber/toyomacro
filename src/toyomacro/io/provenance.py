"""HDF5 provenance group I/O for the Toyomacro local schema (Phase B-1).

Toyomacro HDF5 is NOT an instrument-vendor format: it is the local
intermediate/storage schema carried over from the MATLAB-era Toyomacro.
The ``/provenance`` group records (a) facts read from the upstream input
file (PXT/IBW/VAMAS/NPL/SES) and (b) the transformations the Toyomacro
reader/importer applied afterwards. ``source_format`` labels which
upstream format the data was read from; it does not imply conformance
to, or endorsement of, any vendor's official schema.

Layout and rules: docs/hdf5_provenance_phase_b1_design.md
  - unknown scalar = attribute absent (never 0 / NaN / ""),
  - scalars live in group attributes (robust against the repack path's
    float64->float32 dataset downcast, see design doc §2.1),
  - transform history is a 1-D vlen-str dataset of JSON records in
    application order; records that cannot be serialized are dropped
    with a warning and flagged via ``transform_history_incomplete`` —
    never replaced by placeholder records,
  - vendor metadata is opt-in + allowlist only; ``unparsed_notes`` is
    never persisted,
  - reading is best-effort: malformed types/shapes/JSON in known fields
    degrade to warnings and safe defaults, never exceptions. Provenance
    problems must never block reading the spectra themselves.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from toyomacro.io.schema import ToyomacroSchema

if TYPE_CHECKING:
    from toyomacro.io.readers.base_reader import ReaderTransform, SpectrumMetadata

_STR = h5py.string_dtype(encoding="utf-8")

# Group name (PATH_PROVENANCE without the leading slash, for h5py keys)
_GROUP = ToyomacroSchema.PATH_PROVENANCE.lstrip("/")

#: Vendor-metadata key that is never persisted, regardless of allowlist:
#: free-text note lines cannot be filtered on a per-key basis.
_NEVER_PERSIST_VENDOR_KEYS = frozenset({"unparsed_notes"})

VENDOR_POLICY_VERSION = "1.0"


class ProvenanceWarning(UserWarning):
    """Recoverable problem while writing or reading /provenance.

    The spectra themselves are unaffected; only provenance completeness
    is degraded, and the degradation is flagged rather than hidden.
    """


@dataclass(frozen=True)
class HDF5Provenance:
    """Typed view of an HDF5 ``/provenance`` group.

    ``None`` (or ``()`` for tuple fields) means the corresponding
    attribute/dataset is absent or unreadable, i.e. the fact is unknown.
    ``lens_mode`` / ``n_sweeps`` / ``n_slices`` are reserved fields that
    B-1 never writes (Phase A cannot guarantee they were explicitly read
    from the file, see design doc §3); when absent they stay ``None`` —
    structural defaults like 1 are never restored.
    ``transforms`` holds the parsed JSON records in application order;
    ``dropped_transform_records`` counts records lost at write time plus
    records that could not be parsed at read time (both also set
    ``transform_history_incomplete``).
    """

    schema_version: str | None
    source_format: str = "unknown"
    source_format_version: str | None = None
    source_region_index: int = 0
    region_name: str = ""
    excitation_energy: float | None = None  # eV
    pass_energy: float | None = None  # eV
    intensity_semantics: str = "unknown"
    intensity_unit: str = "unknown"
    lens_mode: str | None = None  # reserved; not persisted in B-1
    acquisition_mode: str | None = None  # persisted only when explicitly non-empty
    n_sweeps: int | None = None  # reserved; not persisted in B-1
    n_slices: int | None = None  # reserved; not persisted in B-1
    datetime: str | None = None  # ISO 8601; persisted only on opt-in
    original_shape: tuple[int, ...] = ()
    dimension_roles: tuple[str, ...] = ()
    transforms: tuple[dict[str, Any], ...] = ()
    transform_history_incomplete: bool = False
    dropped_transform_records: int = 0
    vendor_metadata: dict[str, Any] | None = None  # None = not persisted
    vendor_policy_version: str | None = None
    vendor_allowlist: tuple[str, ...] = ()


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------


def write_provenance(
    h5file: h5py.File | h5py.Group,
    metadata: SpectrumMetadata,
    transforms: Sequence[ReaderTransform] = (),
    *,
    persist_datetime: bool = False,
    persist_vendor_metadata: bool = False,
    vendor_metadata_allowlist: Sequence[str] = (),
) -> None:
    """Write the ``/provenance`` group from Phase A reader output.

    Unknown values are represented by *absent* attributes (design §5).
    ``datetime`` and vendor metadata are opt-in (design §3, §7).

    ``n_sweeps`` / ``n_slices`` / ``lens_mode`` are NOT persisted: Phase A
    metadata cannot distinguish "the file said 1" from its structural
    default (n_sweeps=1), and NPL/VAMAS ``lens_mode="Angular"`` is a
    legacy fixed value — none of these are guaranteed acquisition facts.
    Persisting them requires an explicit "was read from the file" marker
    in Phase A metadata first (design §3).
    """
    if _GROUP in h5file:
        del h5file[_GROUP]
    grp = h5file.create_group(_GROUP)

    attrs = grp.attrs
    attrs["schema_version"] = ToyomacroSchema.PROVENANCE_SCHEMA_VERSION
    attrs["source_format"] = str(metadata.source_format)
    if metadata.source_format_version is not None:
        attrs["source_format_version"] = str(metadata.source_format_version)
    attrs["source_region_index"] = np.int64(metadata.source_region_index)
    attrs["region_name"] = str(metadata.region)
    if metadata.excitation_energy is not None:
        attrs["excitation_energy_eV"] = np.float64(metadata.excitation_energy)
    if metadata.pass_energy is not None:
        attrs["pass_energy_eV"] = np.float64(metadata.pass_energy)
    attrs["intensity_semantics"] = str(metadata.intensity_semantics)
    attrs["intensity_unit"] = str(metadata.intensity_unit)
    if metadata.acquisition_mode:
        attrs["acquisition_mode"] = str(metadata.acquisition_mode)
    if persist_datetime and metadata.datetime is not None:
        attrs["datetime"] = metadata.datetime.isoformat()
    if metadata.original_shape:
        attrs["original_shape"] = np.asarray(metadata.original_shape, dtype=np.int64)
    if metadata.dimension_roles:
        attrs.create(
            "dimension_roles",
            data=[str(r) for r in metadata.dimension_roles],
            dtype=_STR,
        )

    _write_transform_history(grp, transforms)

    if persist_vendor_metadata and vendor_metadata_allowlist:
        _write_vendor_metadata(grp, metadata, vendor_metadata_allowlist)


def _write_transform_history(
    grp: h5py.Group, transforms: Sequence[ReaderTransform]
) -> None:
    records: list[str] = []
    dropped = 0
    for t in transforms:
        try:
            records.append(
                json.dumps(t.to_dict(), ensure_ascii=False, allow_nan=False)
            )
        except (TypeError, ValueError) as e:
            # Design §6.2: never write a placeholder record — drop, warn,
            # and flag the history as incomplete so the loss is explicit.
            dropped += 1
            warnings.warn(
                f"provenance: transform record '{t.name}' is not JSON-safe "
                f"and was dropped from transform_history ({e})",
                ProvenanceWarning,
                stacklevel=4,
            )

    if records:
        grp.create_dataset("transform_history", data=records, dtype=_STR)
    if dropped:
        grp.attrs["transform_history_incomplete"] = True
        grp.attrs["dropped_transform_records"] = np.int64(dropped)


def _write_vendor_metadata(
    grp: h5py.Group,
    metadata: SpectrumMetadata,
    allowlist: Sequence[str],
) -> None:
    subset: dict[str, Any] = {}
    for key in allowlist:
        if key in _NEVER_PERSIST_VENDOR_KEYS:
            warnings.warn(
                f"provenance: vendor key '{key}' is never persisted "
                "(free-text; cannot be allowlisted)",
                ProvenanceWarning,
                stacklevel=4,
            )
            continue
        if key not in metadata.vendor_metadata:
            continue
        value = metadata.vendor_metadata[key]
        try:
            json.dumps({key: value}, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as e:
            warnings.warn(
                f"provenance: vendor key '{key}' is not JSON-safe and was "
                f"dropped ({e})",
                ProvenanceWarning,
                stacklevel=4,
            )
            continue
        subset[key] = value

    if not subset:
        return

    ds = grp.create_dataset(
        "vendor_metadata",
        data=[json.dumps(subset, ensure_ascii=False, allow_nan=False)],
        dtype=_STR,
    )
    ds.attrs["policy_version"] = VENDOR_POLICY_VERSION
    ds.attrs.create("allowlist", data=[str(k) for k in allowlist], dtype=_STR)


# ----------------------------------------------------------------------
# Reading (best-effort: malformed metadata -> warning + safe default)
# ----------------------------------------------------------------------


def _warn_malformed(src: str, what: str, detail: str) -> None:
    warnings.warn(
        f"{src}: /provenance {what} is malformed ({detail}); using safe default",
        ProvenanceWarning,
        stacklevel=4,
    )


def _attr_str(attrs: h5py.AttributeManager, key: str, src: str) -> str | None:
    if key not in attrs:
        return None
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        _warn_malformed(src, f"attribute '{key}'", "array where scalar string expected")
        return None
    return str(value)


def _attr_float(attrs: h5py.AttributeManager, key: str, src: str) -> float | None:
    if key not in attrs:
        return None
    value = attrs[key]
    try:
        if isinstance(value, np.ndarray):
            raise TypeError("array where scalar expected")
        result = float(value)
    except (TypeError, ValueError) as e:
        _warn_malformed(src, f"attribute '{key}'", str(e))
        return None
    if not np.isfinite(result):
        _warn_malformed(src, f"attribute '{key}'", f"non-finite value {result!r}")
        return None
    return result


def _attr_int(
    attrs: h5py.AttributeManager, key: str, src: str, default: int | None
) -> int | None:
    if key not in attrs:
        return default
    value = attrs[key]
    try:
        if isinstance(value, np.ndarray):
            raise TypeError("array where scalar expected")
        as_float = float(value)
        if not np.isfinite(as_float):
            raise ValueError(f"non-finite value {as_float!r}")
        return int(as_float)
    except (TypeError, ValueError) as e:
        _warn_malformed(src, f"attribute '{key}'", str(e))
        return default


def _attr_bool(attrs: h5py.AttributeManager, key: str, src: str) -> bool:
    if key not in attrs:
        return False
    try:
        return bool(attrs[key])
    except (TypeError, ValueError) as e:
        _warn_malformed(src, f"attribute '{key}'", str(e))
        # Unreadable completeness flag: assume incomplete (conservative).
        return True


def _attr_int_tuple(
    attrs: h5py.AttributeManager, key: str, src: str
) -> tuple[int, ...]:
    if key not in attrs:
        return ()
    try:
        return tuple(int(d) for d in np.atleast_1d(attrs[key]))
    except (TypeError, ValueError) as e:
        _warn_malformed(src, f"attribute '{key}'", str(e))
        return ()


def _attr_str_tuple(
    attrs: h5py.AttributeManager, key: str, src: str
) -> tuple[str, ...]:
    if key not in attrs:
        return ()
    value = attrs[key]
    if isinstance(value, (str, bytes)):
        _warn_malformed(src, f"attribute '{key}'", "scalar where 1-D array expected")
        return ()
    try:
        return tuple(
            v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
            for v in np.atleast_1d(value)
        )
    except (TypeError, ValueError) as e:
        _warn_malformed(src, f"attribute '{key}'", str(e))
        return ()


def read_provenance(h5file: h5py.File | h5py.Group) -> HDF5Provenance | None:
    """Read ``/provenance`` into a typed :class:`HDF5Provenance`.

    Returns ``None`` when the group is absent (all legacy files — normal,
    no warning) or when ``/provenance`` exists but is not a group.
    Provenance problems never raise: unknown attributes, unknown
    transform names, malformed types/shapes and unparseable JSON degrade
    to :class:`ProvenanceWarning` plus safe defaults (design §8).
    """
    if _GROUP not in h5file:
        return None
    grp = h5file[_GROUP]
    src = getattr(h5file, "filename", None) or getattr(
        getattr(h5file, "file", None), "filename", "<h5>"
    )
    if not isinstance(grp, h5py.Group):
        warnings.warn(
            f"{src}: /provenance is not an HDF5 group; provenance ignored",
            ProvenanceWarning,
            stacklevel=2,
        )
        return None
    attrs = grp.attrs

    schema_version = _attr_str(attrs, "schema_version", src)
    if schema_version is None:
        warnings.warn(
            f"{src}: /provenance has no readable schema_version attribute; "
            "reading best-effort",
            ProvenanceWarning,
            stacklevel=2,
        )
    else:
        try:
            major = int(schema_version.split(".")[0])
        except ValueError:
            major = None
        expected_major = int(ToyomacroSchema.PROVENANCE_SCHEMA_VERSION.split(".")[0])
        if major != expected_major:
            warnings.warn(
                f"{src}: /provenance schema_version {schema_version!r} has an "
                f"unsupported major version (supported: {expected_major}.x); "
                "reading known fields best-effort",
                ProvenanceWarning,
                stacklevel=2,
            )

    transforms, parse_dropped, history_unreadable = _read_transform_history(grp, src)
    stored_dropped = _attr_int(attrs, "dropped_transform_records", src, default=0) or 0
    incomplete = (
        _attr_bool(attrs, "transform_history_incomplete", src)
        or parse_dropped > 0
        or history_unreadable
    )

    vendor, policy_version, allowlist = _read_vendor_metadata(grp, src)

    return HDF5Provenance(
        schema_version=schema_version,
        source_format=_attr_str(attrs, "source_format", src) or "unknown",
        source_format_version=_attr_str(attrs, "source_format_version", src),
        source_region_index=_attr_int(attrs, "source_region_index", src, default=0)
        or 0,
        region_name=_attr_str(attrs, "region_name", src) or "",
        excitation_energy=_attr_float(attrs, "excitation_energy_eV", src),
        pass_energy=_attr_float(attrs, "pass_energy_eV", src),
        intensity_semantics=_attr_str(attrs, "intensity_semantics", src) or "unknown",
        intensity_unit=_attr_str(attrs, "intensity_unit", src) or "unknown",
        lens_mode=_attr_str(attrs, "lens_mode", src),
        acquisition_mode=_attr_str(attrs, "acquisition_mode", src),
        # Reserved fields: absent stays None — never restore the Phase A
        # structural default of 1 as if it were an acquisition fact.
        n_sweeps=_attr_int(attrs, "n_sweeps", src, default=None),
        n_slices=_attr_int(attrs, "n_slices", src, default=None),
        datetime=_attr_str(attrs, "datetime", src),
        original_shape=_attr_int_tuple(attrs, "original_shape", src),
        dimension_roles=_attr_str_tuple(attrs, "dimension_roles", src),
        transforms=tuple(transforms),
        transform_history_incomplete=incomplete,
        dropped_transform_records=stored_dropped + parse_dropped,
        vendor_metadata=vendor,
        vendor_policy_version=policy_version,
        vendor_allowlist=allowlist,
    )


def _read_transform_history(
    grp: h5py.Group, src: str
) -> tuple[list[dict[str, Any]], int, bool]:
    """Returns (records, n_unparseable_records, history_unreadable)."""
    if "transform_history" not in grp:
        return [], 0, False
    obj = grp["transform_history"]
    if (
        not isinstance(obj, h5py.Dataset)
        or obj.shape is None
        or len(obj.shape) != 1
    ):
        _warn_malformed(
            src, "transform_history", "not a 1-D dataset; history unreadable"
        )
        return [], 0, True
    if obj.shape[0] == 0:
        _warn_malformed(src, "transform_history", "empty dataset")
        return [], 0, False
    try:
        raw_items = obj[:]
    except (OSError, TypeError, ValueError) as e:
        _warn_malformed(src, "transform_history", f"unreadable dataset ({e})")
        return [], 0, True

    records: list[dict[str, Any]] = []
    dropped = 0
    for i, raw in enumerate(raw_items):
        text = (
            raw.decode("utf-8", errors="replace")
            if isinstance(raw, bytes)
            else str(raw)
        )
        try:
            record = json.loads(text)
        except json.JSONDecodeError as e:
            dropped += 1
            warnings.warn(
                f"{src}: transform_history[{i}] is not valid JSON and was "
                f"skipped ({e})",
                ProvenanceWarning,
                stacklevel=3,
            )
            continue
        if not isinstance(record, dict):
            dropped += 1
            warnings.warn(
                f"{src}: transform_history[{i}] is not a JSON object; skipped",
                ProvenanceWarning,
                stacklevel=3,
            )
            continue
        # Unknown "name" values and unknown keys are returned as-is:
        # interpreting them is the consumer's concern (design §6.2).
        records.append(record)
    return records, dropped, False


def _read_vendor_metadata(
    grp: h5py.Group, src: str
) -> tuple[dict[str, Any] | None, str | None, tuple[str, ...]]:
    if "vendor_metadata" not in grp:
        return None, None, ()
    ds = grp["vendor_metadata"]
    if (
        not isinstance(ds, h5py.Dataset)
        or ds.shape is None
        or len(ds.shape) != 1
        or ds.shape[0] == 0
    ):
        _warn_malformed(src, "vendor_metadata", "not a non-empty 1-D dataset")
        return None, None, ()
    try:
        raw = ds[0]
    except (OSError, TypeError, ValueError) as e:
        _warn_malformed(src, "vendor_metadata", f"unreadable dataset ({e})")
        return None, None, ()
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    try:
        vendor = json.loads(text)
    except json.JSONDecodeError as e:
        warnings.warn(
            f"{src}: vendor_metadata is not valid JSON; ignored ({e})",
            ProvenanceWarning,
            stacklevel=3,
        )
        return None, None, ()
    if not isinstance(vendor, dict):
        warnings.warn(
            f"{src}: vendor_metadata is not a JSON object; ignored",
            ProvenanceWarning,
            stacklevel=3,
        )
        return None, None, ()
    return (
        vendor,
        _attr_str(ds.attrs, "policy_version", src),
        _attr_str_tuple(ds.attrs, "allowlist", src),
    )
