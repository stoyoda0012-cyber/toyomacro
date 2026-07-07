"""
Data path resolution and JSON cache management.

Handles:
- Common/data/ CSV source location
- JSON cache generation and loading
- Environment variable overrides
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

# Default Common/data/ path (relative to this file's location)
# paths.py → data/ → toyomacro/ → src/ → toyomacro-python/ → SourceCode/ → Common/
_DEFAULT_COMMON_PATH = Path(__file__).parent.parent.parent.parent.parent / "Common"

# Cache directory inside the package
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


def load_binding_energy_data() -> dict[str, Any]:
    """Load binding energy data, using cache if available."""
    cache_file = get_cache_dir() / "binding_energy.json"

    # Try cache first
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    # Generate from CSV
    csv_path = get_common_data_path() / "BindingEnergyTable.csv"
    data = _csv_to_json_binding_energy(csv_path)

    # Save to cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


def load_compound_data() -> dict[str, Any]:
    """Load compound data, using cache if available."""
    cache_file = get_cache_dir() / "compounds.json"

    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    csv_path = get_common_data_path() / "CompoundTable.csv"
    data = _csv_to_json_compounds(csv_path)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


def load_cross_section_data() -> dict[str, Any]:
    """Load cross section data, using cache if available."""
    cache_file = get_cache_dir() / "cross_section.json"

    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    csv_path = get_common_data_path() / "CrossSectionTable_Yeh=Lindau.csv"
    data = _csv_to_json_cross_section(csv_path)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


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


def clear_cache():
    """Clear all cached JSON files."""
    cache_dir = get_cache_dir()
    for f in cache_dir.glob("*.json"):
        f.unlink()


def regenerate_cache():
    """Force regeneration of all cache files."""
    clear_cache()
    load_binding_energy_data()
    load_compound_data()
    load_cross_section_data()
    load_trzh2018_data()
    load_trzh2019_data()
