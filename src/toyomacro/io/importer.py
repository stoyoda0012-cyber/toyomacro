"""
Import orchestrator: raw XPS data -> HDF5 -> compression in one pass.

Reads raw data files (.pxt, .ibw, .txt, .vms, .npl), converts to
the Toyomacro HDF5 schema, and optionally compresses with uint16+LZ4.

Usage:
    from toyomacro.io.importer import import_file, ImportConfig

    config = ImportConfig(element="Si2p")
    result = import_file("data.pxt", "outputs/", config)
    print(result.output_path, result.n_spectra)

    # Transparent import (auto-detect format, auto-detect element)
    from toyomacro.io.importer import ensure_h5
    h5_paths = ensure_h5("data.pxt")  # -> [Path("Si2p_240101.h5")]

CLI:
    toyomacro import data.pxt -e Si2p -o output/
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toyomacro.io.readers.base_reader import (
    RawSpectrumData,
    create_reader,
    detect_format,
)
from toyomacro.io.writers.hdf5_writer import HDF5CreateOptions, HDF5Writer


@dataclass
class ImportConfig:
    """Configuration for data import."""

    element: str  # e.g. "Si2p" (used for output filename)
    format: str = "auto"  # auto|pxt|ses|vamas|npl
    max_components: int = 6
    energy_scale: str = "auto"  # auto|BE|KE (override energy scale)
    excitation_energy: float | None = None  # override hv
    region_index: int = 0  # region index for multi-region files
    compress: bool = True  # apply uint16+LZ4 compression
    sweep_mode: str = "individual"  # individual|integrate
    region_name_override: str | None = None  # override region name for filename
    # /provenance opt-ins (docs/hdf5_provenance_phase_b1_design.md §3, §7).
    # The group itself is always written; these gate privacy-sensitive parts.
    persist_datetime: bool = False  # acquisition datetime is an indirect identifier
    persist_vendor_metadata: bool = False
    vendor_metadata_allowlist: tuple[str, ...] = ()  # empty = nothing persisted


@dataclass
class ImportResult:
    """Result of a file import."""

    output_path: Path
    n_spectra: int
    n_energy: int
    n_regions: int
    energy_range: tuple[float, float]
    compressed: bool
    file_size_mb: float
    elapsed_seconds: float
    metadata: dict[str, Any]


_DEDUP_SUFFIX_RE = re.compile(r"^(.+?)(_\d+)$")


def _generate_output_filename(
    element: str, source_path: Path, region_name: str = ""
) -> str:
    """Generate output filename: {prefix}_{source_stem}.h5.

    The prefix is chosen as follows:
    - If region_name yields a recognised element (e.g. "Si2p"),
      the normalised element string is used.
    - Otherwise the raw region_name is used (e.g. "Overview").
    - If no region_name is given, *element* is used as the prefix.
    - De-duplication suffixes (_2, _3, ...) are preserved.
    - If the prefix would be redundant (already starts the stem),
      the stem alone is used to avoid names like "O1s_O1s.h5".

    Examples:
        Si2p_TantalumNanosheet1L.h5      (region "Si2p", stem differs)
        Overview_TantalumNanosheet1L.h5   (region "Overview")
        C1s_2_AlF3_ion_off.h5            (second "C1s" region)
        O1s.h5                           (single-region, stem="O1s")
        O1s_arpes.h5                     (stem already starts with element)
    """
    stem = source_path.stem

    if region_name:
        # Split off de-duplication suffix (_2, _3, ...) before normalising
        m = _DEDUP_SUFFIX_RE.match(region_name)
        if m:
            base_name, dedup = m.group(1), m.group(2)
        else:
            base_name, dedup = region_name, ""

        normalised = detect_element(base_name)
        prefix = (normalised if normalised else base_name) + dedup
    else:
        prefix = element

    # Avoid redundant prefix: "O1s_O1s.h5" → "O1s.h5",
    # "O1s_O1s_arpes.h5" → "O1s_arpes.h5"
    stem_lower = stem.lower()
    prefix_lower = prefix.lower()
    if stem_lower.startswith(prefix_lower):
        # stem already begins with prefix — use stem as-is
        return f"{stem}.h5"

    return f"{prefix}_{stem}.h5"


def _validate_data(data: RawSpectrumData) -> None:
    """Validate spectrum data before writing."""
    if data.energy.size < 3:
        raise ValueError(f"Energy axis too short: {data.energy.size} points")

    if np.any(np.isnan(data.energy)):
        raise ValueError("Energy axis contains NaN values")

    if np.any(np.isnan(data.specdata)):
        raise ValueError("Spectrum data contains NaN values")

    # Check monotonicity (ascending or descending)
    diff = np.diff(data.energy)
    if not (np.all(diff > 0) or np.all(diff < 0)):
        raise ValueError("Energy axis is not monotonic")


def _apply_energy_conversion(
    data: RawSpectrumData, target_scale: str, excitation_energy: float | None
) -> RawSpectrumData:
    """Convert energy scale if needed.

    The conversion (E_out = hv - E_in) is recorded in ``data.transforms``
    together with hv, so the original axis stays reconstructible.

    Args:
        data: Input data
        target_scale: 'BE' or 'KE' or 'auto' (keep as-is)
        excitation_energy: Override photon energy (hv)
    """
    if target_scale == "auto":
        return data

    current_scale = data.metadata.energy_scale
    if not current_scale:
        current_scale = "Kinetic"

    if current_scale.startswith("K") and target_scale == "BE":
        new_scale = "Binding"
    elif current_scale.startswith("B") and target_scale == "KE":
        new_scale = "Kinetic"
    else:
        return data  # already on the requested scale

    # Determine hv (None = unknown; never assume 0)
    hv = (
        excitation_energy
        if excitation_energy is not None
        else data.metadata.excitation_energy
    )
    if hv is None or hv <= 0:
        raise ValueError(
            f"Cannot convert {current_scale} -> {target_scale}: "
            "excitation energy not available. Use --excitation-energy."
        )

    data.energy = hv - data.energy
    data.metadata.energy_scale = new_scale
    data.record_transform(
        "energy_scale_conversion",
        parameters={
            "from": current_scale,
            "to": new_scale,
            "excitation_energy_eV": float(hv),
            "formula": "E_out = hv - E_in",
        },
        source="importer",
        reason=f"requested energy_scale={target_scale}",
    )

    return data


def _apply_sweep_integration(data: RawSpectrumData, sweep_mode: str) -> RawSpectrumData:
    """Handle higher-dimensional data (3D/4D+).

    Input specdata can be:
      1D: (n_energy,) -> keep as-is (reshaped later)
      2D: (n_energy, n_angle) -> keep as-is
      3D: (n_energy, n_angle, n_pos) -> flatten to (n_energy, n_angle*n_pos)
      4D+: (n_energy, n_angle, d3, d4, ...) -> flatten all non-energy dims

    Modes:
      - **individual** (default): Flatten all non-energy dimensions into
        a single spectra axis.  Each slice is preserved as an individual
        spectrum.  Correct for spatial scans and multi-position data.
      - **integrate**: Sum across dimensions beyond the first two
        (energy, angle).  Use only when the extra dimensions are sweep
        repetitions that should be averaged for S/N improvement.

    Note: Scienta SES already integrates sweeps during acquisition, so
    the stored dimensions are typically spatial (position, time, etc.),
    not sweep repetitions.  ``individual`` is the safe default.
    """
    specdata = data.specdata

    if specdata.ndim <= 2:
        return data

    input_shape = tuple(int(s) for s in specdata.shape)

    if sweep_mode == "integrate":
        # Sum across all dimensions beyond the first two (energy, angle)
        summed_axes = list(range(2, specdata.ndim))
        while specdata.ndim > 2:
            specdata = np.sum(specdata, axis=-1)
        data.specdata = specdata
        data.record_transform(
            "sweep_integration",
            parameters={
                "input_shape": list(input_shape),
                "output_shape": [int(s) for s in specdata.shape],
                "summed_axes": summed_axes,
            },
            source="importer",
            reason="sweep_mode='integrate'",
        )
    elif sweep_mode == "individual":
        # Flatten all non-energy dimensions into a single spectra axis
        # Shape: (n_energy, n_angle, d3, d4, ...) -> (n_energy, n_angle*d3*d4*...)
        n_energy = specdata.shape[0]
        n_spectra = int(np.prod(specdata.shape[1:]))
        data.specdata = specdata.reshape(n_energy, n_spectra, order="F")
        data.record_transform(
            "dimension_flattening",
            parameters={
                "input_shape": list(input_shape),
                "output_shape": [n_energy, n_spectra],
                "order": "F",
            },
            source="importer",
            reason="sweep_mode='individual'",
        )

    return data


def import_file(
    input_path: str | Path,
    output_dir: str | Path,
    config: ImportConfig,
) -> ImportResult:
    """Import a single raw data file to HDF5.

    Pipeline: read -> validate -> convert energy -> sweep integration ->
              write HDF5 -> compress

    Args:
        input_path: Path to input file (.pxt, .ibw, .txt, .vms, .npl)
        output_dir: Output directory for HDF5 files
        config: Import configuration

    Returns:
        ImportResult with output path and statistics
    """
    t0 = time.perf_counter()

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create reader and read data
    reader = create_reader(input_path, config.format)
    data = reader.read(config.region_index)

    # 2. Validate
    _validate_data(data)

    # 3. Energy scale conversion
    data = _apply_energy_conversion(data, config.energy_scale, config.excitation_energy)

    # 4. Flatten higher dims to 2D
    # Capture original shape before flattening for xytdata construction
    raw_shape = data.specdata.shape  # e.g. (n_energy, n_angle, n_pos, ...)
    data = _apply_sweep_integration(data, config.sweep_mode)

    # 5. Prepare for HDF5
    specdata = data.specdata
    if specdata.ndim == 1:
        specdata = specdata.reshape(-1, 1)

    # Ensure 2D: (n_energy, n_spectra) — safety net
    if specdata.ndim > 2:
        in_shape = [int(s) for s in specdata.shape]
        n_energy = specdata.shape[0]
        n_flat = int(np.prod(specdata.shape[1:]))
        specdata = specdata.reshape(n_energy, n_flat, order="F")
        data.record_transform(
            "dimension_flattening",
            parameters={
                "input_shape": in_shape,
                "output_shape": [int(n_energy), n_flat],
                "order": "F",
            },
            source="importer",
            reason="HDF5 schema requires 2D (n_energy, n_spectra)",
        )

    n_energy = specdata.shape[0]
    n_spectra = specdata.shape[1]

    energy = data.energy
    angle = data.angle

    # Energy sign for misc
    be_sign = 1 if data.metadata.energy_scale.startswith("B") else 0

    # 6. Generate output filename
    if config.region_name_override is not None:
        region_name = config.region_name_override.replace(" ", "_").replace("\x00", "")
    else:
        region_name = data.metadata.region.replace(" ", "_").replace("\x00", "") if data.metadata.region else ""
    output_filename = _generate_output_filename(config.element, input_path, region_name)
    output_path = output_dir / output_filename

    # 7. Write HDF5
    # dim_shape: non-energy dimensions of the original data before flattening
    # e.g. 4D (1024,12,25,20) -> dim_shape = [12,25,20]
    #      3D (201,192,401)    -> dim_shape = [192,401]
    #      2D (401,400)        -> dim_shape = [400]
    dim_shape = list(raw_shape[1:]) if len(raw_shape) > 1 else []

    # The HDF5 schema (unchanged) stores fermienergy as a plain float;
    # 0.0 remains the legacy sentinel for "unknown". The honest value
    # (None = unknown) stays available on data.metadata / ImportResult.
    hv = data.metadata.excitation_energy
    misc = {
        "fermienergy": float(hv) if hv is not None else 0.0,
        "bindingenergysign": be_sign,
        "maxcomp": config.max_components,
        "numberofslice": data.metadata.n_slices,
        "source_file": input_path.name,
    }

    # Use chunked HDF5 for fast creation (no pre-allocation of zeros)
    opts = HDF5CreateOptions(contiguous=False)
    with HDF5Writer(output_path, options=opts) as writer:
        writer.create(
            n_spectra=n_spectra,
            n_energy=n_energy,
            max_components=config.max_components,
            energy=energy,
            misc=misc,
        )

        # Save dim_shape for 4D navigation
        if dim_shape:
            writer.write_dim_shape(dim_shape)

        # Save dim0_is_angle without re-inferring an unknown detector axis.
        # The reader owns this format-specific classification; extension,
        # channel count, or instrument name alone are not evidence of angle.
        detector_role = (
            data.metadata.dimension_roles[1]
            if len(data.metadata.dimension_roles) > 1
            else "unknown"
        )
        dim0_is_angle = len(angle) > 1 and detector_role == "emission_angle"
        writer.write_dim0_is_angle(dim0_is_angle)

        # Provenance: facts read from the upstream input file plus the
        # reader/importer transform history (Toyomacro-local schema).
        writer.write_provenance(
            data.metadata,
            data.transforms,
            persist_datetime=config.persist_datetime,
            persist_vendor_metadata=config.persist_vendor_metadata,
            vendor_metadata_allowlist=config.vendor_metadata_allowlist,
        )

        # Write spectra
        # specdata is [n_energy, n_spectra], HDF5 wants [n_spectra, n_energy]
        writer.write_spectra_batch(0, specdata.T)

        # Write xytdata in one batch: shape (3, n_spectra)
        # Row 0 = θ (angle), Row 1 = y, Row 2 = t
        # Fortran-order flatten: angle varies fastest, then y, then t
        xyt = np.zeros((3, n_spectra), dtype=np.float32)
        n_angle = len(angle) if len(angle) > 0 else 1

        if n_angle > 0:
            # Row 0: tile angle values (θ cycles every n_angle spectra)
            xyt[0, :] = np.tile(angle, (n_spectra // n_angle) + 1)[:n_spectra]

        if len(dim_shape) >= 3:
            # 4D+: (n_angle, n_y, n_t, ...) — decompose flat index
            n_y = dim_shape[1]
            flat_idx = np.arange(n_spectra, dtype=np.int32)
            remainder = flat_idx // n_angle
            xyt[1, :] = (remainder % n_y).astype(np.float32)   # y
            xyt[2, :] = (remainder // n_y).astype(np.float32)  # t
        elif len(dim_shape) >= 2:
            # 3D: (n_angle, n_pos) — y = position, t = 0
            flat_idx = np.arange(n_spectra, dtype=np.int32)
            xyt[1, :] = (flat_idx // n_angle).astype(np.float32)
            # xyt[2] stays 0
        # else: 2D or less — both y and t stay 0

        writer.write_xytdata_batch(xyt)

    # 8. Compress (optional)
    compressed = False
    final_path = output_path
    if config.compress:
        try:
            from toyomacro.io.compression import compress_h5_file_streaming

            compressed_path = output_path.with_name(
                output_path.stem + "_compressed" + output_path.suffix
            )
            compress_h5_file_streaming(
                input_path=str(output_path),
                output_path=str(compressed_path),
                compression="full",
            )
            compressed = True
            final_path = compressed_path
        except Exception as e:
            # Compression is optional; fall back to uncompressed
            print(f"Warning: compression failed ({e}), keeping uncompressed file")

    elapsed = time.perf_counter() - t0
    file_size_mb = final_path.stat().st_size / (1024 * 1024)

    return ImportResult(
        output_path=final_path,
        n_spectra=n_spectra,
        n_energy=n_energy,
        n_regions=reader.n_regions,
        energy_range=(float(energy.min()), float(energy.max())),
        compressed=compressed,
        file_size_mb=file_size_mb,
        elapsed_seconds=elapsed,
        metadata={
            "element": config.element,
            "region": data.metadata.region,
            "energy_scale": data.metadata.energy_scale,
            "excitation_energy": data.metadata.excitation_energy,  # None = unknown
            "pass_energy": data.metadata.pass_energy,  # None = unknown
            "source_file": str(input_path),
            "source_format": data.metadata.source_format,
            "source_format_version": data.metadata.source_format_version,
            "source_region_index": data.metadata.source_region_index,
            "intensity_semantics": data.metadata.intensity_semantics,
            "intensity_unit": data.metadata.intensity_unit,
            "original_shape": list(data.metadata.original_shape),
            "dimension_roles": list(data.metadata.dimension_roles),
            # Full transform history (reader conventions + importer steps)
            "transforms": [t.to_dict() for t in data.transforms],
        },
    )


def import_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    config: ImportConfig,
    dry_run: bool = False,
) -> list[ImportResult]:
    """Import all compatible files in a folder.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        config: Import configuration
        dry_run: If True, print plan without writing

    Returns:
        List of ImportResult for each file
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all supported files
    supported_exts = {".pxt", ".ibw", ".txt", ".vms", ".npl"}
    files = sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in supported_exts
    )

    if not files:
        print(f"No supported files found in {input_dir}")
        return []

    if dry_run:
        print(f"Found {len(files)} files to import:")
        for f in files:
            fmt = detect_format(f)
            print(f"  {f.name} ({fmt})")
        print(f"\nOutput directory: {output_dir}")
        print(f"Element: {config.element}")
        print(f"Compress: {config.compress}")
        return []

    results: list[ImportResult] = []
    for f in files:
        try:
            result = import_file(f, output_dir, config)
            print(
                f"  {f.name} -> {result.output_path.name} "
                f"({result.n_spectra} spectra, {result.n_energy} pts, "
                f"{result.file_size_mb:.1f} MB, {result.elapsed_seconds:.2f}s)"
            )
            results.append(result)
        except Exception as e:
            print(f"  {f.name} FAILED: {e}")

    return results


# ============================================================
# Transparent import: ensure_h5() gateway
# ============================================================

# Raw formats that can be auto-converted to HDF5
RAW_EXTENSIONS: set[str] = {".pxt", ".ibw", ".txt", ".vms", ".npl"}

# XPS element pattern: matches Si2p, O1s, Au4f, Ta4f, C1s, etc.
_ELEMENT_RE = re.compile(r"([A-Z][a-z]?[\s_]*\d[spdf])", re.IGNORECASE)


def detect_element(filepath: str | Path) -> str | None:
    """Auto-detect XPS element name from a file path or region name.

    Searches for standard XPS orbital patterns (Si2p, O1s, Au4f, etc.)
    in the filename.

    Args:
        filepath: File path or region name string to search

    Returns:
        Normalized element name (e.g. "Si2p") or None if not found
    """
    name = Path(filepath).stem if isinstance(filepath, Path) else str(filepath)
    match = _ELEMENT_RE.search(name)
    if match:
        # Normalize: remove spaces/underscores, capitalize element, lowercase orbital
        raw = match.group(1).replace(" ", "").replace("_", "")
        # Capitalize: first letter upper, rest of element lower, digit+orbital lower
        if len(raw) >= 3 and raw[1].isalpha():
            # Two-letter element: Au4f, Si2p, Ta4f
            return raw[0].upper() + raw[1].lower() + raw[2:]
        else:
            # Single-letter element: O1s, C1s, N1s
            return raw[0].upper() + raw[1:]
    return None


def _find_existing_h5(source_path: Path, output_dir: Path) -> list[Path] | None:
    """Check if converted HDF5 files already exist for a source file.

    Output filenames follow the pattern ``{prefix}_{source_stem}.h5``
    or just ``{source_stem}.h5`` (when prefix is redundant).  We match
    h5 filenames that *end* with the source stem (possibly preceded by
    ``_``) so that "O1s.txt" does not falsely match "O1s_arpes.h5".

    Returns:
        List of existing .h5 paths if found and newer, else None
    """
    source_mtime = source_path.stat().st_mtime
    source_stem = source_path.stem

    candidates = []
    for h5_file in output_dir.glob("*.h5"):
        h5_stem = h5_file.stem
        # Match: h5 stem == source_stem  (e.g. "O1s" == "O1s")
        # or:    h5 stem ends with "_{source_stem}" (e.g. "Si2p_O1s_arpes" ends with "_O1s_arpes")
        if (
            h5_stem == source_stem
            or h5_stem.endswith(f"_{source_stem}")
        ) and h5_file.stat().st_mtime >= source_mtime:
            candidates.append(h5_file)

    return sorted(candidates) if candidates else None


def ensure_h5(
    filepath: str | Path,
    element: str | None = None,
    compress: bool = False,
    output_dir: Path | None = None,
) -> list[Path]:
    """Ensure a file is available as HDF5. Convert from raw format if needed.

    This is the gateway function for transparent file import. All GUI and
    Web API code paths should use this to open files.

    - .h5/.hdf5 files are returned as-is
    - Raw formats (.pxt, .ibw, .txt, .vms, .npl) are auto-converted
    - Previously converted files are reused (mtime-based check)
    - Multi-region files produce one .h5 per region

    Args:
        filepath: Path to any supported XPS data file
        element: Element name override (auto-detected from filename if None)
        compress: Whether to apply uint16+LZ4 compression (default: False)
        output_dir: Output directory for .h5 files (default: same as source)

    Returns:
        List of Path objects to HDF5 files (one per region)

    Raises:
        ValueError: If file format is unsupported
        FileNotFoundError: If file does not exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Already HDF5 -> return as-is
    if filepath.suffix.lower() in (".h5", ".hdf5"):
        return [filepath]

    # Check extension is supported
    if filepath.suffix.lower() not in RAW_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    out_dir = output_dir or filepath.parent

    # Auto-detect element from filename (fallback for single-region files)
    file_element = element or detect_element(filepath) or filepath.stem

    # Check for existing converted .h5 files
    existing = _find_existing_h5(filepath, out_dir)
    if existing is not None:
        return existing

    # Import: check number of regions
    reader = create_reader(filepath)
    n_regions = reader.n_regions

    results: list[Path] = []

    if n_regions <= 1:
        config = ImportConfig(
            element=file_element,
            compress=compress,
            region_index=0,
        )
        result = import_file(filepath, out_dir, config)
        results.append(result.output_path)
    else:
        # Multi-region: import each region as a separate .h5
        # Each region gets its own element detection — no fallback to
        # a file-level element (which can be wrong for Survey/Overview).
        seen_names: dict[str, int] = {}
        for i in range(n_regions):
            rname = reader.region_names[i]

            # Per-region element: detect from region name
            region_element = detect_element(rname)
            if region_element is None:
                # Survey / Overview / Wide / unknown → keep region name
                region_element = rname.replace(" ", "_")

            # De-duplicate: same region name appearing more than once
            name_key = rname
            count = seen_names.get(name_key, 0)
            seen_names[name_key] = count + 1
            region_suffix = rname
            if count > 0:
                region_suffix = f"{rname}_{count + 1}"

            config = ImportConfig(
                element=region_element,
                compress=compress,
                region_index=i,
                region_name_override=region_suffix,
            )
            try:
                result = import_file(filepath, out_dir, config)
                results.append(result.output_path)
            except Exception as e:
                print(f"Warning: region {i} ({rname}) import failed: {e}")

    return results
