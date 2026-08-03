"""
Data path resolution and bundled-table loading.

Handles:
- locating the bundled reference tables under ``_cache/``
- the opt-in path that rebuilds one from a local CSV source
- environment variable overrides
"""

from __future__ import annotations

import csv
import json
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Default Common/data/ path (relative to this file's location)
# paths.py → data/ → toyomacro/ → src/ → toyomacro/ → SourceCode/ → Common/
_DEFAULT_COMMON_PATH = Path(__file__).parent.parent.parent.parent.parent / "Common"

# Bundled reference tables. The directory name is historical: these JSON
# files are shipped, reviewed data, documented in docs/DATA_SOURCES.md and
# pinned by tests -- not build artifacts. The CSV/Excel sources they were
# built from are not part of this repository and have diverged since, so
# rebuilding one (see regenerate_cache()) changes a reference dataset and
# is gated behind REGENERATE_ENV_VAR rather than happening on demand.
_CACHE_DIR = Path(__file__).parent / "_cache"


def get_common_data_path() -> Path:
    """
    Get path to Common/data/ directory.

    Can be overridden with TOYOMACRO_COMMON_PATH environment variable.

    Returns:
        Path to Common/data/ directory

    Raises:
        FileNotFoundError: If the path doesn't exist
    """
    env_path = os.environ.get("TOYOMACRO_COMMON_PATH")
    if env_path:
        common_path = Path(env_path)
    else:
        common_path = _DEFAULT_COMMON_PATH

    data_path = common_path / "data"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Common/data/ not found at {data_path}. "
            f"Set TOYOMACRO_COMMON_PATH environment variable to override."
        )
    return data_path


def get_cache_dir() -> Path:
    """Get cache directory, creating it if needed."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _csv_to_json_binding_energy(csv_path: Path) -> dict[str, Any]:
    """
    Convert BindingEnergyTable.csv to JSON structure.

    Output format:
    {
        "elements": {
            "H": {"Z": 1, "orbitals": {"1s": 13.6}},
            "Si": {"Z": 14, "orbitals": {"1s": 1839.0, "2s": 150.0, ...}},
            ...
        },
        "by_z": {1: "H", 2: "He", ...}
    }
    """
    elements: dict[str, Any] = {}
    by_z: dict[int, str] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            z = int(row["ElementNumber"])
            symbol = row["ElementSymbol"].strip()

            orbitals: dict[str, float] = {}
            for key, value in row.items():
                if key in ("ElementNumber", "ElementSymbol"):
                    continue
                if value and value.strip():
                    try:
                        be = float(value)
                        if be > 0:  # Skip zero/negative values
                            orbitals[key] = be
                    except ValueError:
                        pass

            elements[symbol] = {"Z": z, "orbitals": orbitals}
            by_z[z] = symbol

    return {"elements": elements, "by_z": by_z}


def _csv_to_json_compounds(csv_path: Path) -> dict[str, Any]:
    """
    Convert CompoundTable.csv to JSON structure.

    Output format:
    {
        "SiO2": {"Nv": 16, "density": 2.2, "Mw": 60.08, "Eg": 8.9},
        ...
    }
    """
    compounds: dict[str, Any] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["CompoundName"].strip()
            if not name:
                continue

            props: dict[str, float] = {}

            # ValenceCharge -> Nv
            if row.get("ValenceCharge"):
                try:
                    props["Nv"] = float(row["ValenceCharge"])
                except ValueError:
                    pass

            # Density [g/cm3] -> density
            density_key = next(
                (k for k in row.keys() if "Density" in k), None
            )
            if density_key and row.get(density_key):
                try:
                    props["density"] = float(row[density_key])
                except ValueError:
                    pass

            # MolecularWeight [g/mol] -> Mw
            mw_key = next(
                (k for k in row.keys() if "Molecular" in k or "Weight" in k), None
            )
            if mw_key and row.get(mw_key):
                try:
                    props["Mw"] = float(row[mw_key])
                except ValueError:
                    pass

            # BandGap [eV] -> Eg
            eg_key = next(
                (k for k in row.keys() if "BandGap" in k or "Gap" in k), None
            )
            if eg_key and row.get(eg_key):
                try:
                    props["Eg"] = float(row[eg_key])
                except ValueError:
                    pass

            if props:
                compounds[name] = props

    return compounds


def _csv_to_json_cross_section(csv_path: Path) -> dict[str, Any]:
    """
    Convert CrossSectionTable_Yeh=Lindau.csv to JSON structure.

    Output format:
    {
        "photon_energies": [10.2, 16.7, ..., 1486.6, ...],
        "data": {
            "Si": {
                "2p": {"binding_energy": 99.0, "cross_sections": [...]},
                ...
            },
            ...
        }
    }
    """
    data: dict[str, dict[str, Any]] = {}
    photon_energies: list[float] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Extract photon energies from column names (CrossSec_PE_XXX format)
        for col in fieldnames:
            if col.startswith("CrossSec_PE_"):
                # Extract energy value, handling underscores as decimal points
                energy_str = col.replace("CrossSec_PE_", "").replace("_", ".")
                try:
                    photon_energies.append(float(energy_str))
                except ValueError:
                    pass

        for row in reader:
            symbol = row.get("ElementSymbol", "").strip()
            orbital = row.get("AtomicOrbital", "").strip()

            if not symbol or not orbital:
                continue

            if symbol not in data:
                data[symbol] = {}

            # Get binding energy
            be = None
            if row.get("BindingEnergy"):
                try:
                    be = float(row["BindingEnergy"])
                except ValueError:
                    pass

            # Get cross sections for each photon energy
            cross_sections: list[float | None] = []
            for col in fieldnames:
                if col.startswith("CrossSec_PE_"):
                    val = row.get(col, "")
                    if val and val.strip():
                        try:
                            cross_sections.append(float(val))
                        except ValueError:
                            cross_sections.append(None)
                    else:
                        cross_sections.append(None)

            data[symbol][orbital] = {
                "binding_energy": be,
                "cross_sections": cross_sections,
            }

    return {"photon_energies": photon_energies, "data": data}


#: Env var that permits rebuilding a shipped table from a local CSV.
#: Unset (the normal case), a missing table is an error rather than a
#: silent rebuild — see :func:`_load_shipped_table`.
REGENERATE_ENV_VAR = "TOYOMACRO_REGENERATE_DATA"


def _load_shipped_table(
    cache_name: str,
    csv_name: str,
    converter: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Load one bundled reference table from ``_cache/``.

    The files under ``_cache/`` are named "cache" for historical reasons
    but are **shipped reference data**: reviewed, documented in
    ``docs/DATA_SOURCES.md``, and pinned by tests. The CSV sources they
    were built from are not part of this repository, are not shipped,
    and have since moved on independently.

    So a missing table is not rebuilt on the quiet. Doing that would let
    a table be replaced by whatever a local CSV happens to contain, with
    no review — the change that most needs one. It is not hypothetical:
    on the maintainer's machine, rebuilding ``compounds.json`` from the
    current CSV yields 166 entries rather than 109, drops ``Si3N4``
    (split there into two phases under different names), and moves the
    parameters of Al2O3, GaAs, SiC and SiO2.

    Set :data:`REGENERATE_ENV_VAR` to opt in. The result is unreviewed
    by construction, so it is written where it will be seen in a diff
    rather than returned silently.

    Raises:
        FileNotFoundError: if the table is missing and regeneration was
            not requested.
    """
    cache_file = get_cache_dir() / cache_name

    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    if not os.environ.get(REGENERATE_ENV_VAR):
        raise FileNotFoundError(
            f"Bundled reference table {cache_name} is missing from "
            f"{get_cache_dir()}. It ships with this package and is not a "
            f"disposable cache; restore it (e.g. `git checkout` the file, or "
            f"reinstall). To rebuild it instead from a local "
            f"{csv_name} — which may not hold the same values, and which "
            f"changes a reference dataset without review — set "
            f"{REGENERATE_ENV_VAR}=1."
        )

    warnings.warn(
        f"Rebuilding {cache_name} from {csv_name}. The result is unreviewed "
        f"and may differ from the reference table this package ships; "
        f"check the diff before committing it.",
        UserWarning,
        stacklevel=3,
    )
    data = converter(get_common_data_path() / csv_name)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def load_binding_energy_data() -> dict[str, Any]:
    """Load the bundled elemental binding-energy table."""
    return _load_shipped_table(
        "binding_energy.json", "BindingEnergyTable.csv", _csv_to_json_binding_energy
    )


def load_compound_data() -> dict[str, Any]:
    """Load the bundled compound/element property table."""
    return _load_shipped_table(
        "compounds.json", "CompoundTable.csv", _csv_to_json_compounds
    )


def load_compound_provenance() -> dict[str, Any]:
    """Load the per-field provenance for the compound table.

    Hand-authored, never rebuilt from a CSV: the CSV sources carry no
    provenance, so a rebuild would silently drop it.
    """
    provenance_file = get_cache_dir() / "compounds_provenance.json"
    if not provenance_file.exists():
        raise FileNotFoundError(
            f"Bundled provenance table compounds_provenance.json is missing "
            f"from {get_cache_dir()}. It ships with this package and is "
            f"hand-authored -- there is no source to rebuild it from. Restore "
            f"it (e.g. `git checkout` the file, or reinstall)."
        )
    with open(provenance_file, encoding="utf-8") as f:
        return json.load(f)


def load_cross_section_data() -> dict[str, Any]:
    """Load the bundled Yeh & Lindau cross-section table."""
    return _load_shipped_table(
        "cross_section.json", "CrossSectionTable_Yeh=Lindau.csv", _csv_to_json_cross_section
    )


def _get_trzh2018_xlsx_path() -> Path:
    """Get path to Trzhaskovskaya 2018 HAXPES Excel file.

    Located in SESSAAnalyser database (sibling of Common/).
    """
    env_path = os.environ.get("TOYOMACRO_COMMON_PATH")
    if env_path:
        base = Path(env_path).parent
    else:
        base = _DEFAULT_COMMON_PATH.parent

    xlsx_path = (
        get_common_data_path() / "cross_sections"
        / "excel_trzhaskovskaya_2018_pics.xlsx"
    )
    return xlsx_path


def _xlsx_to_json_trzh2018(xlsx_path: Path) -> dict[str, Any]:
    """Parse Trzhaskovskaya 2018 Excel into JSON structure.

    Each element sheet has orbital blocks of 4 rows:
      Row N:   orbital_name | σ  | σ values (10 energies)
      Row N+1: "Eb="        | β  | β values
      Row N+2: "XX.X eV"    | γ  | γ values
      Row N+3: (empty)      | δ  | δ values

    Returns
    -------
    dict
        {
            "photon_energies": [1500, 2000, ..., 10000],
            "data": {
                "Si": {
                    "2p3/2": {
                        "binding_energy": 98.9,
                        "sigma": [...],
                        "beta": [...],
                        "gamma": [...],
                        "delta": [...]
                    }, ...
                }, ...
            }
        }
    """
    import re

    try:
        import openpyxl
    except ImportError:
        return {"photon_energies": [], "data": {}}

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)

    photon_energies: list[float] = []
    all_data: dict[str, dict[str, Any]] = {}

    for sheet_name in wb.sheetnames:
        if sheet_name.startswith("Explanation"):
            continue

        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if len(rows) < 3:
            continue

        # Extract element symbol from row 0, col B
        elem_symbol = str(rows[0][1]).strip() if rows[0][1] else None
        if not elem_symbol:
            continue

        # Extract photon energies from row 1 (only once)
        if not photon_energies:
            for val in rows[1][2:]:
                if val is not None:
                    try:
                        photon_energies.append(float(val))
                    except (ValueError, TypeError):
                        break

        n_energies = len(photon_energies)
        elem_data: dict[str, Any] = {}

        # Scan for orbital blocks: col B == "σ"
        row_idx = 2  # Start after header rows
        while row_idx < len(rows) - 3:
            row = rows[row_idx]
            if row[1] is not None and str(row[1]).strip() == "σ":
                orbital_name = str(row[0]).strip() if row[0] else None
                if not orbital_name:
                    row_idx += 1
                    continue

                # σ values from this row
                sigma = []
                for v in row[2:2 + n_energies]:
                    sigma.append(float(v) if v is not None else None)

                # β from next row
                beta_row = rows[row_idx + 1]
                beta = []
                for v in beta_row[2:2 + n_energies]:
                    beta.append(float(v) if v is not None else None)

                # Parse binding energy from col A of β row
                be_str = str(beta_row[0]).strip() if beta_row[0] else ""
                be_match = re.search(r"([\d.]+)", be_str)
                binding_energy = float(be_match.group(1)) if be_match else None

                # γ from row+2
                gamma_row = rows[row_idx + 2]
                gamma = []
                for v in gamma_row[2:2 + n_energies]:
                    gamma.append(float(v) if v is not None else None)

                # Also check col A for BE if not found in β row
                if binding_energy is None:
                    be_str2 = str(gamma_row[0]).strip() if gamma_row[0] else ""
                    be_match2 = re.search(r"([\d.]+)", be_str2)
                    if be_match2:
                        binding_energy = float(be_match2.group(1))

                # δ from row+3
                delta_row = rows[row_idx + 3]
                delta = []
                for v in delta_row[2:2 + n_energies]:
                    delta.append(float(v) if v is not None else None)

                elem_data[orbital_name] = {
                    "binding_energy": binding_energy,
                    "sigma": sigma,
                    "beta": beta,
                    "gamma": gamma,
                    "delta": delta,
                }

                row_idx += 4  # Skip to next orbital block
            else:
                row_idx += 1

        if elem_data:
            all_data[elem_symbol] = elem_data

    wb.close()
    return {"photon_energies": photon_energies, "data": all_data}


def _get_trzh2019_xlsx_path() -> Path:
    """Get path to Trzhaskovskaya 2019 HAXPES Excel file (inner shells)."""
    env_path = os.environ.get("TOYOMACRO_COMMON_PATH")
    if env_path:
        base = Path(env_path).parent
    else:
        base = _DEFAULT_COMMON_PATH.parent

    return (
        get_common_data_path() / "cross_sections"
        / "excel_trzhaskovskaya_2019_pics.xlsx"
    )


def _xlsx_to_json_trzh2019(xlsx_path: Path) -> dict[str, Any]:
    """Parse Trzhaskovskaya 2019 Excel (inner shells, variable energy grids).

    Same 4-row orbital block format as 2018, but:
    - Only inner shells (Eb ≥ ~1.5 keV)
    - Energy grid varies per element (2-18 keV range)
    - Z=13 to Z=100

    Returns per-element-orbital structure with individual energy grids:
    {
        "data": {
            "Si": {
                "1s1/2": {
                    "binding_energy": 1838.9,
                    "photon_energies": [2000, 3000, ..., 12000],
                    "sigma": [...], "beta": [...], "gamma": [...], "delta": [...]
                }, ...
            }, ...
        }
    }
    """
    import re

    try:
        import openpyxl
    except ImportError:
        return {"data": {}}

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    all_data: dict[str, dict[str, Any]] = {}

    for sheet_name in wb.sheetnames:
        if sheet_name.startswith("Explanation"):
            continue

        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if len(rows) < 3:
            continue

        elem_symbol = str(rows[0][1]).strip() if rows[0][1] else None
        if not elem_symbol:
            continue

        # Per-sheet photon energies (variable grid)
        sheet_energies: list[float] = []
        for val in rows[1][2:]:
            if val is not None:
                try:
                    sheet_energies.append(float(val))
                except (ValueError, TypeError):
                    break

        n_energies = len(sheet_energies)
        if n_energies == 0:
            continue

        elem_data: dict[str, Any] = {}

        row_idx = 2
        while row_idx < len(rows) - 3:
            row = rows[row_idx]
            if row[1] is not None and str(row[1]).strip() == "σ":
                orbital_name = str(row[0]).strip() if row[0] else None
                if not orbital_name:
                    row_idx += 1
                    continue

                sigma = [float(v) if v is not None else None
                         for v in row[2:2 + n_energies]]

                beta_row = rows[row_idx + 1]
                beta = [float(v) if v is not None else None
                        for v in beta_row[2:2 + n_energies]]

                be_str = str(beta_row[0]).strip() if beta_row[0] else ""
                be_match = re.search(r"([\d.]+)", be_str)
                binding_energy = float(be_match.group(1)) if be_match else None

                gamma_row = rows[row_idx + 2]
                gamma = [float(v) if v is not None else None
                         for v in gamma_row[2:2 + n_energies]]

                if binding_energy is None:
                    be_str2 = str(gamma_row[0]).strip() if gamma_row[0] else ""
                    be_match2 = re.search(r"([\d.]+)", be_str2)
                    if be_match2:
                        binding_energy = float(be_match2.group(1))

                delta_row = rows[row_idx + 3]
                delta = [float(v) if v is not None else None
                         for v in delta_row[2:2 + n_energies]]

                elem_data[orbital_name] = {
                    "binding_energy": binding_energy,
                    "photon_energies": sheet_energies,
                    "sigma": sigma,
                    "beta": beta,
                    "gamma": gamma,
                    "delta": delta,
                }

                row_idx += 4
            else:
                row_idx += 1

        if elem_data:
            all_data[elem_symbol] = elem_data

    wb.close()
    return {"data": all_data}


def load_trzh2018_data() -> dict[str, Any]:
    """Load Trzhaskovskaya 2018 HAXPES data, using cache if available.

    Contains σ, β, γ, δ parameters for Z=1-100 at 1.5-10 keV.
    """
    cache_file = get_cache_dir() / "trzh2018_haxpes.json"

    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    xlsx_path = _get_trzh2018_xlsx_path()
    if not xlsx_path.exists():
        return {"photon_energies": [], "data": {}}

    data = _xlsx_to_json_trzh2018(xlsx_path)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


def load_trzh2019_data() -> dict[str, Any]:
    """Load Trzhaskovskaya 2019 inner-shell data, using cache if available.

    Contains σ, β, γ, δ for inner shells (Eb ≥ ~1.5 keV), Z=13-100.
    Each orbital has its own photon energy grid (variable, 2-18 keV).
    """
    cache_file = get_cache_dir() / "trzh2019_inner.json"

    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    xlsx_path = _get_trzh2019_xlsx_path()
    if not xlsx_path.exists():
        return {"data": {}}

    data = _xlsx_to_json_trzh2019(xlsx_path)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


def clear_cache() -> None:
    """Refuse to delete the bundled tables. Kept so the name still resolves.

    This used to unlink every ``*.json`` under ``_cache/``. Those files
    are the package's shipped reference data — see
    ``docs/DATA_SOURCES.md`` — so the operation had no correct use: it
    destroyed reviewed data, and two of the tables it removed
    (``scofield.json``, ``trzhaskovskaya.json``) were not rebuilt by
    :func:`regenerate_cache` at all, leaving cross-section lookups to
    fall back to an empty table.

    Raises:
        RuntimeError: always.
    """
    raise RuntimeError(
        "clear_cache() would delete this package's bundled reference tables, "
        "not a cache. They ship with the package, are documented in "
        "docs/DATA_SOURCES.md, and are pinned by tests. To rebuild them from "
        f"local CSV/XLSX sources, set {REGENERATE_ENV_VAR}=1 and call "
        "regenerate_cache(), which writes in place and never deletes first."
    )


def regenerate_cache() -> dict[str, str]:
    """Rebuild the bundled tables in place from their local sources.

    Requires :data:`REGENERATE_ENV_VAR` to be set: rebuilding replaces a
    reference dataset with an unreviewed one, which the project treats as
    a change needing an independent audit, not a side effect.

    Nothing is deleted first. A table whose source is unavailable keeps
    the shipped copy and is reported as skipped, so a partial rebuild
    cannot leave the package with missing data.

    Returns:
        Mapping of table filename to ``"rebuilt"`` or a reason it was
        skipped.

    Raises:
        RuntimeError: if the opt-in is not set.
    """
    if not os.environ.get(REGENERATE_ENV_VAR):
        raise RuntimeError(
            "regenerate_cache() rebuilds shipped reference tables from local "
            "CSV/XLSX sources that are not part of this repository, and the "
            "result may differ from the reviewed data this package ships. Set "
            f"{REGENERATE_ENV_VAR}=1 to confirm that is what you want."
        )

    cache_dir = get_cache_dir()
    results: dict[str, str] = {}

    def _write(cache_name: str, data: dict[str, Any]) -> None:
        """Warn immediately before overwriting reviewed data, then write."""
        warnings.warn(
            f"Overwriting the bundled {cache_name} with an unreviewed rebuild "
            f"from local sources. Review the diff before committing it: this "
            f"is a change to a reference dataset.",
            UserWarning,
            stacklevel=3,
        )
        with open(cache_dir / cache_name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    for cache_name, csv_name, converter in (
        ("binding_energy.json", "BindingEnergyTable.csv", _csv_to_json_binding_energy),
        ("compounds.json", "CompoundTable.csv", _csv_to_json_compounds),
        ("cross_section.json", "CrossSectionTable_Yeh=Lindau.csv", _csv_to_json_cross_section),
    ):
        try:
            csv_path = get_common_data_path() / csv_name
        except FileNotFoundError as exc:
            results[cache_name] = f"skipped: {exc}"
            continue
        if not csv_path.exists():
            results[cache_name] = f"skipped: {csv_name} not found"
            continue
        _write(cache_name, converter(csv_path))
        results[cache_name] = "rebuilt"

    for cache_name, path_getter, converter in (
        ("trzh2018_haxpes.json", _get_trzh2018_xlsx_path, _xlsx_to_json_trzh2018),
        ("trzh2019_inner.json", _get_trzh2019_xlsx_path, _xlsx_to_json_trzh2019),
    ):
        xlsx_path = path_getter()
        if not xlsx_path.exists():
            results[cache_name] = f"skipped: {xlsx_path.name} not found"
            continue
        _write(cache_name, converter(xlsx_path))
        results[cache_name] = "rebuilt"

    # scofield.json and trzhaskovskaya.json are rebuilt through
    # CrossSection's own loader, from CSVs in the same Common/data tree.
    # They were silently absent from this function's old list, which is
    # why clear_cache() could remove them with nothing to restore them.
    for cache_name in ("scofield.json", "trzhaskovskaya.json"):
        results[cache_name] = "not rebuilt here: see CrossSection._load_with_cache"

    return results
