"""
Photoionization cross-section lookup for XPS analysis.

Supports multiple databases:
  - yeh_lindau:       Yeh & Lindau (1985), 16 photon energies
  - scofield:         Scofield (1973), long-format CSV, 28 photon energies
  - trzhaskovskaya:   Trzhaskovskaya/Nefedov/Yarzhemsky ADNDT 77 (2001) &
                      82 (2002), relativistic, 10 energies. NOTE: the source
                      tables are gridded in PHOTOELECTRON energy but are
                      interpolated here as photon energy (MATLAB-era
                      convention) — see docs/DATA_SOURCES.md

Usage:
    from toyomacro.data import CrossSection

    sigma = CrossSection.lookup('Si', '2p', 1486.6)  # Al K-α (default: yeh_lindau)
    sigma = CrossSection.lookup('Si', '2p', 1486.6, table='scofield')

    CrossSection.set_default_table('scofield')  # change default
    sigma = CrossSection.lookup('Si', '2p', 1486.6)  # now uses Scofield
"""

from __future__ import annotations

import csv
import json
import math
from typing import Any

import numpy as np

from toyomacro.data.paths import (
    get_cache_dir,
    get_common_data_path,
    load_cross_section_data,
)

# Available table names
AVAILABLE_TABLES = ("yeh_lindau", "scofield", "trzhaskovskaya")


class CrossSection:
    """
    Photoionization cross-section database with multiple table support.

    All methods are class methods - no instantiation needed.
    Cross-section units: Megabarn (Mb)
    """

    _data: dict[str, dict[str, Any]] = {}  # table_name -> data
    _default_table: str = "yeh_lindau"

    @classmethod
    def set_default_table(cls, table: str) -> None:
        """Set the default cross-section table.

        Args:
            table: One of 'yeh_lindau', 'scofield', 'trzhaskovskaya'
        """
        if table not in AVAILABLE_TABLES:
            raise ValueError(f"Unknown table '{table}'. Choose from {AVAILABLE_TABLES}")
        cls._default_table = table

    @classmethod
    def get_default_table(cls) -> str:
        """Return current default table name."""
        return cls._default_table

    @classmethod
    def _get_data(cls, table: str | None = None) -> dict[str, Any]:
        """Get cached data for a specific table, loading if needed."""
        table = table or cls._default_table
        if table not in cls._data:
            cls._data[table] = cls._load_table(table)
        return cls._data[table]

    @classmethod
    def _load_table(cls, table: str) -> dict[str, Any]:
        """Load a specific cross-section table."""
        if table == "yeh_lindau":
            return load_cross_section_data()
        elif table == "scofield":
            return cls._load_scofield()
        elif table == "trzhaskovskaya":
            return cls._load_trzhaskovskaya()
        else:
            raise ValueError(f"Unknown table '{table}'")

    @staticmethod
    def _load_with_cache(cache_name: str, csv_name: str, parser) -> dict[str, Any]:
        """Bundled-JSON-cache-first table load.

        The JSON caches under ``data/_cache`` ship with the package so that
        lookups work on a clean install; the ``Common/data`` CSV sources are
        only needed to regenerate them. Returns an empty table when neither
        the cache nor the CSV source is available.
        """
        cache_file = get_cache_dir() / cache_name
        if cache_file.exists():
            with open(cache_file, encoding="utf-8") as f:
                return json.load(f)
        try:
            csv_path = get_common_data_path() / csv_name
        except FileNotFoundError:
            return {"photon_energies": [], "data": {}}
        if not csv_path.exists():
            return {"photon_energies": [], "data": {}}
        data = parser(csv_path)
        # Compact JSON: these caches are machine-read only, and the Scofield
        # table is large enough that indentation roughly doubles its size.
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        return data

    @classmethod
    def _load_scofield(cls) -> dict[str, Any]:
        """Load Scofield 1973 cross-section data (cache-first)."""
        return cls._load_with_cache(
            "scofield.json",
            "Scofield_1973_cross_sections.csv",
            cls._parse_scofield_csv,
        )

    # The Scofield 1973 report tabulates 1 keV–1.5 MeV; only the soft-X-ray /
    # HAXPES range matters for photoemission, and the full union grid would
    # null-pad the bundled cache to ~30 MB. 30 keV covers every lab and
    # synchrotron HAXPES source with margin.
    _SCOFIELD_MAX_EV = 30_000.0

    @staticmethod
    def _parse_scofield_csv(csv_path) -> dict[str, Any]:
        """Parse the Scofield 1973 long-format CSV (truncated at 30 keV)."""
        # Collect all data grouped by element/orbital
        raw: dict[str, dict[str, list[tuple[float, float]]]] = {}

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                elem = row.get("Element", "").strip()
                orbital = row.get("Orbital", "").strip()
                try:
                    hv = float(row["PhotonEnergy_eV"])
                    sigma = float(row["CrossSection_Mb"])
                except (ValueError, KeyError):
                    continue
                if not elem or not orbital:
                    continue
                if hv > CrossSection._SCOFIELD_MAX_EV:
                    continue
                raw.setdefault(elem, {}).setdefault(orbital, []).append((hv, sigma))

        # Collect all unique photon energies
        all_energies: set[float] = set()
        for elem_data in raw.values():
            for pts in elem_data.values():
                all_energies.update(hv for hv, _ in pts)
        photon_energies = sorted(all_energies)

        # Build same format as Yeh-Lindau
        data: dict[str, dict[str, Any]] = {}
        for elem, orbitals in raw.items():
            data[elem] = {}
            for orbital, pts in orbitals.items():
                pts_dict = {hv: sigma for hv, sigma in pts}
                cross_sections = [pts_dict.get(e) for e in photon_energies]
                data[elem][orbital] = {
                    "binding_energy": None,
                    "cross_sections": cross_sections,
                }

        return {"photon_energies": photon_energies, "data": data}

    @classmethod
    def _load_trzhaskovskaya(cls) -> dict[str, Any]:
        """Load Trzhaskovskaya-Yarzhemsky cross-section data (cache-first)."""
        return cls._load_with_cache(
            "trzhaskovskaya.json",
            "CrossSectionTable_Trzhaskovskaya=Yarzhemsky.csv",
            cls._parse_trzhaskovskaya_csv,
        )

    @staticmethod
    def _parse_trzhaskovskaya_csv(csv_path) -> dict[str, Any]:
        """Parse the Trzhaskovskaya-Yarzhemsky wide CSV (CR line endings)."""
        # This CSV may have \r line endings and be a single line — read raw
        text = csv_path.read_text(encoding="utf-8")
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [l for l in text.split("\n") if l.strip()]
        if len(lines) < 2:
            return {"photon_energies": [], "data": {}}

        reader = csv.DictReader(lines)
        fieldnames = reader.fieldnames or []

        # Extract photon energies from CrossSec_PE_XXX columns
        photon_energies: list[float] = []
        pe_cols: list[str] = []
        for col in fieldnames:
            if col.startswith("CrossSec_PE_"):
                energy_str = col.replace("CrossSec_PE_", "").replace("_", ".")
                try:
                    photon_energies.append(float(energy_str))
                    pe_cols.append(col)
                except ValueError:
                    pass

        data: dict[str, dict[str, Any]] = {}
        for row in reader:
            symbol = row.get("ElementSymbol", "").strip()
            orbital = row.get("AtomicOrbital", "").strip()
            if not symbol or not orbital:
                continue
            # Skip header echo rows
            if symbol == "Photon" or symbol == "ElementSymbol":
                continue

            be = None
            if row.get("BindingEnergy"):
                try:
                    be = float(row["BindingEnergy"])
                except ValueError:
                    pass

            cross_sections: list[float | None] = []
            for col in pe_cols:
                val = row.get(col, "")
                if val and val.strip():
                    try:
                        cross_sections.append(float(val))
                    except ValueError:
                        cross_sections.append(None)
                else:
                    cross_sections.append(None)

            data.setdefault(symbol, {})[orbital] = {
                "binding_energy": be,
                "cross_sections": cross_sections,
            }

        return {"photon_energies": photon_energies, "data": data}

    # Guard to prevent _extrapolate → lookup → _extrapolate recursion
    _extrapolating: bool = False

    @classmethod
    def lookup(
        cls,
        element: str,
        orbital: str,
        photon_energy: float,
        table: str | None = None,
    ) -> float | None:
        """
        Look up cross-section for an element, orbital, and photon energy.

        Interpolates between tabulated values using log-log interpolation.
        Falls back to Z-based extrapolation if no direct data is available.

        Args:
            element: Element symbol (e.g., 'Si', 'Au')
            orbital: Orbital designation (e.g., '2p', '2p3/2')
            photon_energy: Photon energy in eV (e.g., 1486.6 for Al K-α)
            table: Which table to use (default: current default table)

        Returns:
            Cross-section in Megabarn (Mb), or None if not found
        """
        # Check if photon energy exceeds binding energy (physical threshold)
        be = cls._get_binding_energy(element, orbital)
        if be is not None and photon_energy < be:
            return None  # Cannot ionize: hν < BE

        result = cls._lookup_direct(element, orbital, photon_energy, table)
        if result is not None:
            return result

        # Fallback: extrapolate from other elements with the same orbital
        if not cls._extrapolating:
            return cls._extrapolate(element, orbital, photon_energy, table)
        return None

    @classmethod
    def _lookup_direct(
        cls,
        element: str,
        orbital: str,
        photon_energy: float,
        table: str | None = None,
    ) -> float | None:
        """Direct table lookup without extrapolation fallback."""
        data = cls._get_data(table)
        photon_energies = data.get("photon_energies", [])
        elements_data = data.get("data", {})

        elem_data = elements_data.get(element)
        if elem_data is None:
            return None

        # Try exact orbital match first
        orbital_data = elem_data.get(orbital)

        # Try normalized orbital (e.g., '2p' might be stored as '2p3/2')
        if orbital_data is None:
            for suffix in ["3/2", "5/2", "7/2", "1/2"]:
                orbital_data = elem_data.get(f"{orbital}{suffix}")
                if orbital_data is not None:
                    break

        # Also try summing spin-orbit pair for bare orbital (e.g., '2p' = 2p1/2 + 2p3/2)
        if orbital_data is None and len(orbital) == 2 and orbital[1] in "spdf":
            pair = _find_so_pair(elem_data, orbital)
            if pair:
                o1_data, o2_data = pair
                cs1 = o1_data.get("cross_sections", [])
                cs2 = o2_data.get("cross_sections", [])
                summed = []
                for a, b in zip(cs1, cs2):
                    if a is not None and b is not None:
                        summed.append(a + b)
                    elif a is not None:
                        summed.append(a)
                    elif b is not None:
                        summed.append(b)
                    else:
                        summed.append(None)
                orbital_data = {"cross_sections": summed}

        if orbital_data is None:
            return None

        cross_sections = orbital_data.get("cross_sections", [])
        if not cross_sections or not photon_energies:
            return None

        valid_points: list[tuple[float, float]] = []
        for e, sigma in zip(photon_energies, cross_sections):
            if sigma is not None and sigma > 0:
                valid_points.append((e, sigma))

        if len(valid_points) >= 2:
            return cls._interpolate_log_log(valid_points, photon_energy)
        if len(valid_points) == 1:
            return valid_points[0][1]
        return None

    @classmethod
    def get_rsf(
        cls,
        element1: str,
        orbital1: str,
        element2: str,
        orbital2: str,
        photon_energy: float,
        table: str | None = None,
    ) -> float | None:
        """
        Photoionization-cross-section ratio relative to a reference line.

        Returns ``sigma1 / sigma2`` at the given photon energy, both from
        the same table. Each is whatever ``lookup()`` returns, so either
        may be a log-log extrapolation beyond the tabulated energies, or
        a cross-element fit in log(Z) for an orbital the table does not
        carry at all — see ``lookup()``. Neither case is signalled here.

        Despite the historical method name, this is **not** a complete
        relative sensitivity factor and **not** an average-matrix RSF. It
        contains the cross-section term alone. It does not include:

        - the inelastic mean free path or effective attenuation length
          (see ``data.imfp`` and ``data.elastic_scattering``)
        - elastic-scattering corrections
        - the photoelectron angular distribution, x-ray polarization, or
          the source/analyzer geometry
        - analyzer transmission or detector response (see
          ``data.transmission``, which is applied separately and is not
          bundled)
        - matrix-dependent averaging

        Subshell occupancy is *not* among the missing factors: it is
        already carried by the tabulated cross-sections, which is why
        ``2p3/2``/``2p1/2`` comes out near 2:1.

        A published RSF table from an instrument vendor is a different
        quantity and the two are not interchangeable. See §5 of
        ``docs/API.md`` for what a full AMRSF would additionally require.

        Args:
            element1: First element symbol
            orbital1: First orbital
            element2: Reference element symbol (typically 'C')
            orbital2: Reference orbital (typically '1s')
            photon_energy: Photon energy in eV
            table: Which table to use

        Returns:
            The cross-section ratio, or None if either cross-section is
            unavailable.
        """
        sigma1 = cls.lookup(element1, orbital1, photon_energy, table)
        sigma2 = cls.lookup(element2, orbital2, photon_energy, table)

        if sigma1 is None or sigma2 is None or sigma2 == 0:
            return None

        return sigma1 / sigma2

    @classmethod
    def get_available_photon_energies(cls, table: str | None = None) -> list[float]:
        """Get list of tabulated photon energies."""
        data = cls._get_data(table)
        return data.get("photon_energies", [])

    @classmethod
    def get_available_orbitals(cls, element: str, table: str | None = None) -> list[str]:
        """Get list of available orbitals for an element."""
        data = cls._get_data(table)
        elements_data = data.get("data", {})
        elem_data = elements_data.get(element)
        if elem_data is None:
            return []
        return list(elem_data.keys())

    @classmethod
    def to_dict(cls, table: str | None = None) -> dict[str, Any]:
        """Get raw data as dict (for API responses)."""
        return cls._get_data(table)

    # Polynomial order for log-log cross-section interpolation.
    # MATLAB default = 3, DepthProfiler overrides to 6.
    # 3 is safer for general use (less Runge oscillation on extrapolation).
    _poly_order: int = 3

    @classmethod
    def set_poly_order(cls, order: int) -> None:
        """Set the polynomial order for log-log interpolation.

        Args:
            order: Polynomial degree (MATLAB default 3, DepthProfiler default 6).
        """
        cls._poly_order = max(1, order)

    @classmethod
    def _interpolate_log_log(
        cls,
        points: list[tuple[float, float]],
        x: float,
    ) -> float:
        """Interpolate/extrapolate in log-log space using polynomial fit.

        Matches MATLAB ``xps.CrossSection`` behaviour:
        ``polyfit(log10(PE), log10(CS), polyOrder)`` over all valid points,
        then ``10^polyval(p, log10(hv))``.

        For extrapolation beyond the table range, uses at most order-1
        (power-law) from the last few points to avoid high-order
        polynomial divergence (Runge phenomenon).
        """
        points = sorted(points, key=lambda p: p[0])
        n = len(points)

        if n == 1:
            return points[0][1]

        x_min, x_max = points[0][0], points[-1][0]
        extrapolating = x < x_min or x > x_max

        if extrapolating:
            # Use last/first few points with low-order fit (power-law)
            # to avoid high-order polynomial blowup
            if x > x_max:
                # Extrapolate beyond upper range: use last 4 points
                tail = points[-min(4, n):]
            else:
                # Extrapolate below lower range: use first 4 points
                tail = points[:min(4, n)]
            xs = np.array([p[0] for p in tail])
            ys = np.array([p[1] for p in tail])
            log_xs = np.log10(xs)
            log_ys = np.log10(ys)
            order = min(1, len(tail) - 1)  # Linear in log-log = power law
            coeffs = np.polyfit(log_xs, log_ys, order)
        else:
            # Interpolation within range: use all points with full poly_order
            xs = np.array([p[0] for p in points])
            ys = np.array([p[1] for p in points])
            log_xs = np.log10(xs)
            log_ys = np.log10(ys)
            order = min(cls._poly_order, n - 1)
            coeffs = np.polyfit(log_xs, log_ys, order)

        log_y = np.polyval(coeffs, math.log10(x))
        result = 10.0 ** log_y

        return float(result) if result > 0 else points[-1][1]

    @classmethod
    def _get_binding_energy(cls, element: str, orbital: str) -> float | None:
        """Look up binding energy for element/orbital (for threshold check)."""
        try:
            from toyomacro.data.binding_energy import BindingEnergy

            # Try exact match first
            be = BindingEnergy.lookup(element, orbital)
            if be is not None:
                return be
            # Try bare orbital (e.g., '1s' for '1s1/2')
            base = orbital
            for suffix in ["1/2", "3/2", "5/2", "7/2"]:
                if orbital.endswith(suffix):
                    base = orbital[: -len(suffix)]
                    break
            if base != orbital:
                return BindingEnergy.lookup(element, base)
            return None
        except Exception:
            return None

    @classmethod
    def _extrapolate(
        cls,
        element: str,
        orbital: str,
        photon_energy: float,
        table: str | None = None,
    ) -> float | None:
        """Extrapolate cross-section for missing element/orbital.

        Collects cross-sections at ``photon_energy`` for all elements that
        have the same orbital type, then fits a polynomial in log(Z)-log(σ)
        space to estimate the value for the target element.

        σ roughly scales as Z^n, so log-log fit is more physical than
        linear polynomial for extrapolation beyond the data range.
        """
        from toyomacro.data.binding_energy import BindingEnergy

        # Get target atomic number
        be_data = BindingEnergy._get_data()
        elem_entry = be_data.get("elements", {}).get(element)
        if elem_entry is None:
            return None
        target_z = elem_entry["Z"]

        # Normalize orbital to base form (e.g., '1s1/2' → '1s')
        base_orbital = orbital
        for suffix in ["1/2", "3/2", "5/2", "7/2"]:
            if orbital.endswith(suffix):
                base_orbital = orbital[: -len(suffix)]
                break

        data = cls._get_data(table)
        elements_data = data.get("data", {})

        # Collect (Z, sigma) for all elements that have this orbital
        # Set guard to prevent recursive extrapolation
        cls._extrapolating = True
        try:
            z_sigma: list[tuple[int, float]] = []
            for sym, orbitals in elements_data.items():
                sym_entry = be_data.get("elements", {}).get(sym)
                if sym_entry is None:
                    continue
                z = sym_entry["Z"]

                # Use lookup (without extrapolation due to guard)
                sigma = cls.lookup(sym, base_orbital, photon_energy, table)
                if sigma is not None and sigma > 0:
                    z_sigma.append((z, sigma))
        finally:
            cls._extrapolating = False

        if len(z_sigma) < 3:
            return None

        z_sigma.sort()
        zs = np.array([z for z, _ in z_sigma], dtype=np.float64)
        sigmas = np.array([s for _, s in z_sigma], dtype=np.float64)

        # Log-log polynomial fit: σ ∝ Z^n → log(σ) = n·log(Z) + const
        # Z-based extrapolation is inherently approximate — limit to order 2
        # to avoid wild oscillation beyond the data range.
        log_zs = np.log(zs)
        log_sigmas = np.log(sigmas)
        poly_order = min(2, len(z_sigma) - 1)
        coeffs = np.polyfit(log_zs, log_sigmas, poly_order)
        result = float(np.exp(np.polyval(coeffs, math.log(target_z))))

        return result if result > 0 else None


def _find_so_pair(
    elem_data: dict[str, Any], base_orbital: str
) -> tuple[dict, dict] | None:
    """Find spin-orbit pair for a bare orbital like '2p' → ('2p1/2', '2p3/2')."""
    j_pairs = {"s": [("1/2",)], "p": [("1/2", "3/2")], "d": [("3/2", "5/2")], "f": [("5/2", "7/2")]}
    letter = base_orbital[-1]
    pairs = j_pairs.get(letter)
    if not pairs:
        return None
    for js in pairs:
        if len(js) == 2:
            o1 = elem_data.get(f"{base_orbital}{js[0]}")
            o2 = elem_data.get(f"{base_orbital}{js[1]}")
            if o1 is not None and o2 is not None:
                return o1, o2
    return None
