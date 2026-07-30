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

A bare subshell label such as ``'2p'`` means the **whole spin-orbit
doublet**, σ(2p1/2) + σ(2p3/2) — the quantity a measured peak envelope
corresponds to. Pass ``'2p3/2'`` for a single j component. This is the
opposite convention to `BindingEnergy`, where a bare label resolves to
the main line (j = l+1/2), and deliberately so: a binding energy is a
position, of which a doublet has two, whereas a cross-section adds over
the subshells that make up the envelope.

Scofield cross sections already incorporate the report's assumed
fractional subshell occupations (Table A1: B 2p is NE 0.33 + 0.67 = 1
electron, C 0.67 + 1.33 = 2, N 1 + 2 = 3, O 1.33 + 2.67 = 4), so the
components are summed as stored. Do not weight them by occupancy again.

Units are **not** the same across tables — call ``unit_info()`` rather
than assuming. Yeh–Lindau and Scofield are megabarn, each read from its
source; Trzhaskovskaya values run ~10³ larger and their unit is inferred,
not confirmed, so it is reported separately and nothing is rescaled.

Usage:
    from toyomacro.data import CrossSection

    sigma = CrossSection.lookup('Si', '2p', 1486.6)  # Al K-α (default: yeh_lindau)
    CrossSection.unit_info('scofield')               # {'unit': 'Mb', ...}
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

# The j components of each subshell. A bare label denotes their sum.
_J_COMPONENTS: dict[str, tuple[str, ...]] = {
    "s": ("1/2",),
    "p": ("1/2", "3/2"),
    "d": ("3/2", "5/2"),
    "f": ("5/2", "7/2"),
}

_J_SUFFIXES = ("1/2", "3/2", "5/2", "7/2")

# How each table decides which j components to list. This governs what an
# *absent* component means when summing a bare subshell, so it is a
# statement about the data, not a convenience — see the tests.
#
# The evidence differs per table and does not transfer between them.
#
# scofield — evidence: the primary source itself. Scofield, UCRL-51326
#   (1973), Table A1 lists the assumed occupation numbers NE per state,
#   and the tabulated cross-sections already incorporate them: B 2p is
#   NE 0.33 + 0.67 = 1 electron, C 0.67 + 1.33 = 2, N 1 + 2 = 3,
#   O 1.33 + 2.67 = 4. So components are summed exactly as stored, and
#   weighting them by occupancy again would double-correct. Every
#   non-s subshell the bundled cache carries has both components, so a
#   missing one would be a genuine data gap, not an empty orbital.
#   NOTE: Table A1 says nothing about any other table.
# trzhaskovskaya — set False, conservatively, although the structure
#   points the other way. Exactly 38 non-s subshells are half-listed;
#   they are all open valence shells whose upper-j component would be
#   empty, and elements whose ground-state configuration does put an
#   electron there (Cr 3d5, Mo 4d5, Eu 4f7, Re 5d5) carry both. That
#   pattern is consistent with "listed iff occupied" — but it is an
#   inference from the bundled data's own shape, and the inclusion rule
#   has NOT been read from ADNDT 77 (2001) / 82 (2002). Treating an
#   absent component as a hard zero on that basis would put an unverified
#   assumption into returned numbers, so a half-listed bare subshell
#   refuses instead. Flip to True once the source states the rule; the
#   38-count and the exception elements are pinned by test so the
#   evidence is still there to act on. NOT licensed by Scofield's Table
#   A1, which says nothing about this table.
# yeh_lindau — stores bare subshells only; j-resolved requests are refused
#   rather than split by an assumed branching ratio.
_ABSENT_COMPONENT_IS_UNOCCUPIED: dict[str, bool] = {
    "yeh_lindau": False,
    "scofield": False,
    "trzhaskovskaya": False,
}


_UNIT_NOTE_INFERRED = (
    "Unit inferred by comparison with Scofield after correcting the "
    "energy axis, and supported by the same authors' later work; not read "
    "from the primary table heading of ADNDT 77 (2001) / 82 (2002). "
    "A ratio within this table cancels the unknown scale factor, and "
    "only that: this table is gridded in photoelectron energy but "
    "interpolated as photon energy, so a ratio of two lines with "
    "different binding energies still carries that axis error. "
    "Absolute cross-sections, sigma*lambda sensitivities, and any "
    "comparison that mixes tables may be wrong by ~10^3."
)

# (confirmed unit, inferred unit, note). Exactly one of the first two is
# set: a unit read from the source's own heading, or a candidate that a
# caller must not silently convert on.
_TABLE_UNITS: dict[str, tuple[str | None, str | None, str]] = {
    "yeh_lindau": ("Mb", None, ""),
    "scofield": ("Mb", None, ""),
    "trzhaskovskaya": (None, "kb", _UNIT_NOTE_INFERRED),
}


class _Refused:
    """Sentinel: the table covers this subshell and still cannot answer.

    Distinct from ``None``, which means "no data here" and licenses the
    cross-element extrapolation. A refusal must not be extrapolated over:
    doing so would replace a known gap with a fabricated number that is
    indistinguishable from a tabulated one.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<REFUSED>"


_REFUSED = _Refused()


def _absent_component_is_unoccupied(table: str) -> bool:
    """Whether a missing j component means "empty" rather than "no data"."""
    return _ABSENT_COMPONENT_IS_UNOCCUPIED.get(table, False)


def _split_orbital(orbital: str) -> tuple[str, str | None]:
    """Split ``'2p3/2'`` into ``('2p', '3/2')``; ``'2p'`` into ``('2p', None)``."""
    for suffix in _J_SUFFIXES:
        if orbital.endswith(suffix) and len(orbital) > len(suffix):
            return orbital[: -len(suffix)], suffix
    return orbital, None


class CrossSection:
    """
    Photoionization cross-section database with multiple table support.

    All methods are class methods - no instantiation needed.

    Units differ between tables — use :meth:`unit_info` rather than
    assuming a common one. A bare orbital label (``'2p'``) means the
    summed spin-orbit doublet; pass ``'2p3/2'`` for a single component.
    """

    _data: dict[str, dict[str, Any]] = {}  # table_name -> data
    _default_table: str = "yeh_lindau"

    @classmethod
    def unit_info(cls, table: str | None = None) -> dict[str, Any]:
        """What unit ``lookup()`` returns for ``table``, and how well established.

        Returns ``table``, ``unit``, ``inferred_unit``, ``status`` and
        ``values_rescaled``; inferred tables carry a ``note`` as well.

        ``unit`` is populated **only** where it was read from the primary
        source's own table heading. Where the unit is inferred, ``unit`` is
        ``None`` and ``inferred_unit`` carries the candidate, so that no
        caller can convert on an inference by reading a single field. No
        conversion factor is offered for the same reason, and
        ``values_rescaled`` is always False — nothing stored is altered on
        a guess.

        The three tables are **not** on a common scale. A ratio within
        one table cancels the unknown *unit* factor — and only that.
        For `trzhaskovskaya` the separate photoelectron-vs-photon energy
        axis error survives any ratio between lines of different binding
        energy (see `docs/DATA_SOURCES.md`). Absolute cross-sections,
        sigma*lambda sensitivities, and cross-table comparisons carry
        both problems.
        """
        table = table or cls._default_table
        if table not in _TABLE_UNITS:
            raise ValueError(f"Unknown table '{table}'. Choose from {AVAILABLE_TABLES}")
        unit, inferred, note = _TABLE_UNITS[table]
        info: dict[str, Any] = {
            "table": table,
            "unit": unit,
            "inferred_unit": inferred,
            "status": "confirmed" if unit is not None else "inferred",
            "values_rescaled": False,
        }
        if note:
            info["note"] = note
        return info

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
            orbital: Orbital designation. A bare label ('2p') is the summed
                spin-orbit doublet, each component evaluated at
                ``photon_energy`` and then added; pass '2p3/2' for one
                component. A j-resolved label returns None on a table that
                stores only bare subshells (Yeh-Lindau) rather than
                splitting the total by an assumed branching ratio.
            photon_energy: Photon energy in eV (e.g., 1486.6 for Al K-α)
            table: Which table to use (default: current default table)

        Returns:
            Cross-section in the unit reported by :meth:`unit_info` for
            ``table`` — Mb for 'yeh_lindau' and 'scofield', ~10³ larger
            (inferred kb, unconfirmed) for 'trzhaskovskaya' — or None if
            not found. Do not compare across tables without converting.
        """
        # Physical threshold. Gate a *bare* label per component instead,
        # inside `_lookup_direct`: a subshell-level binding energy is the
        # main line's, so gating on it here discards the other component
        # in the window between the two thresholds. Co 3p is the case —
        # bare BE 60 eV, 3p1/2 BE 59 eV — where a subshell gate returns
        # None at 59 eV although 3p1/2 is ionizable and the spectrum
        # contains it.
        base, j = _split_orbital(orbital)
        is_bare_doublet = j is None and len(base) == 2 and base[-1] in "pdf"
        if not is_bare_doublet:
            be = cls._get_binding_energy(element, orbital)
            if be is not None and photon_energy < be:
                return None  # Cannot ionize: hν < BE

        result = cls._lookup_direct(element, orbital, photon_energy, table)
        if result is _REFUSED:
            # The table covers this subshell and still cannot answer —
            # a component missing for an unestablished reason, or nothing
            # ionizable at this energy. Extrapolating from other elements
            # would replace a known gap with a fabricated number that
            # looks tabulated, so refuse instead.
            return None
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
    ) -> float | _Refused | None:
        """Direct table lookup without extrapolation fallback.

        Returns a value, ``None`` when the table has no data for this
        subshell at all (so an extrapolation may legitimately try), or
        ``_REFUSED`` when the table covers the subshell and still cannot
        answer — a component missing for a reason this package has not
        established, or nothing ionizable at this energy.

        A bare subshell label is the sum over its j components, each
        evaluated **independently at** ``photon_energy`` and then added.
        Summing the stored arrays first and interpolating once is not
        equivalent: the interpolator fits one polynomial over an
        element's whole tabulated range, so where the two components are
        listed on different grids — routine just above a split threshold
        — a few dropped points move the result by tens of percent.
        """
        data = cls._get_data(table)
        elements_data = data.get("data", {})

        elem_data = elements_data.get(element)
        if elem_data is None:
            return None

        base, j = _split_orbital(orbital)
        if j is not None:
            # A j-resolved request. Tables that store only bare subshells
            # cannot answer it, and splitting the total by an assumed
            # branching ratio would invent a number indistinguishable
            # from a tabulated one — the statistical ratio is not exact
            # (Scofield's own Si 2p3/2 : 2p1/2 is 1.966, not 2).
            if orbital in elem_data:
                return cls._interp_one(elem_data[orbital], data, photon_energy)
            siblings = _J_COMPONENTS.get(base[-1:], ())
            if any(f"{base}{s}" in elem_data for s in siblings):
                # The table carries this subshell and omits this component.
                # Whatever the reason, it is not a coverage gap that an
                # extrapolation from other elements may fill in.
                return _REFUSED
            return None

        components = _J_COMPONENTS.get(base[-1:]) if len(base) == 2 else None
        if components is None:
            return cls._interp_one(elem_data.get(base), data, photon_energy)

        # A bare label. If the table stores it directly (Yeh-Lindau), that
        # entry already *is* the subshell total. `lookup` skips its
        # threshold gate for bare doublets so that per-component gating
        # can work below, so apply the subshell threshold here.
        if base in elem_data:
            be = cls._get_binding_energy(element, base)
            if be is not None and photon_energy < be:
                return _REFUSED
            return cls._interp_one(elem_data[base], data, photon_energy)

        if not any(f"{base}{s}" in elem_data for s in components):
            return None  # subshell absent entirely: extrapolation may try

        total = 0.0
        contributed = False
        for suffix in components:
            key = f"{base}{suffix}"
            if key in elem_data:
                # Each component carries its own threshold. Between the two
                # thresholds of a split doublet only the lower-BE member is
                # ionizable, and that is exactly what a measured envelope
                # contains — so gate per component, not on the subshell.
                component_be = cls._get_binding_energy(element, key)
                if component_be is not None and photon_energy < component_be:
                    continue
                value = cls._interp_one(elem_data[key], data, photon_energy)
                if value is None:
                    continue
                total += value
                contributed = True
            elif _absent_component_is_unoccupied(table or cls._default_table):
                # The table lists a component iff it is occupied, so an
                # absent one carries no electrons and contributes zero.
                continue
            else:
                # Absent for a reason this package has not established.
                # Summing what is left would be a silent under-count, so
                # refuse the whole subshell rather than guess.
                return _REFUSED

        return total if contributed else _REFUSED

    @classmethod
    def _interp_one(
        cls,
        orbital_data: dict[str, Any] | None,
        data: dict[str, Any],
        photon_energy: float,
    ) -> float | None:
        """Interpolate one stored orbital entry at ``photon_energy``."""
        if orbital_data is None:
            return None
        photon_energies = data.get("photon_energies", [])
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

        Donor elements are sampled with the **same** label that was
        requested. Normalizing '4f7/2' to '4f' here would fit the summed
        doublet and hand it back as a single j component — the same
        miscounting this module fixes in ``_lookup_direct``, laundered
        through an extrapolation so that it looks tabulated.
        """
        from toyomacro.data.binding_energy import BindingEnergy

        # Get target atomic number
        be_data = BindingEnergy._get_data()
        elem_entry = be_data.get("elements", {}).get(element)
        if elem_entry is None:
            return None
        target_z = elem_entry["Z"]
        base_orbital = orbital

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
