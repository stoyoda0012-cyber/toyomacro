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
import os
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.interpolate import PchipInterpolator

from toyomacro.data.paths import (
    REGENERATE_ENV_VAR,
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

    Distinct from ``None``, which means "no data here". Both end as
    ``None`` from :meth:`CrossSection.lookup`; the distinction is kept
    because a refusal is a statement about a table that covers the
    subshell, and must never be filled in from anywhere else.
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


#: Where a cross-section value came from, per j component and summarised
#: per request. The range is the component's own valid cells (non-None,
#: > 0), never the table's whole energy grid, and it is judged on the axis
#: the table is interpolated on. For Trzhaskovskaya that is a photoelectron-
#: energy grid read as photon energy (see the module docstring), so there
#: ``tabulated`` does not mean the source tabulates that kinetic energy.
LookupStatus = Literal[
    "tabulated",            # a valid cell, or interpolation between valid cells
    "extrapolated_above",   # above the last valid cell: power law (value given)
    "outside_range_above",  # above the only valid cell: no basis to extrapolate
    "outside_range_below",  # below the first valid cell: no value by default
    "below_threshold",      # hv < BindingEnergy value: known zero contribution
    "unoccupied",           # absent component the table rule says is empty
    "not_in_table",         # no value, and not established to be empty
]

#: How a value was computed. ``None`` when no value was computed.
LookupMethod = Literal["pchip", "polyfit", "single_cell", "power_law_above",
                       "power_law_below"]

_KNOWN_ZERO: frozenset[str] = frozenset({"below_threshold", "unoccupied"})


@dataclass(frozen=True)
class ComponentLookup:
    """How one stored line of a table answered at one photon energy.

    Attributes:
        orbital: The table key evaluated (``'2p3/2'``; ``'2p'`` where the
            table stores the subshell total, as Yeh-Lindau does).
        status: See :data:`LookupStatus`.
        value: Cross-section in the table's unit; ``0.0`` for a known zero
            (``below_threshold``, ``unoccupied``); ``None`` when there is
            no value. An ``outside_range_below`` value exists only when the
            caller asked for ``extrapolate_below=True``.
        method: See :data:`LookupMethod`; ``None`` when nothing was computed.
        at_tabulated_cell: True when the photon energy is one of the line's
            valid cells (to 1e-12 relative). With ``"pchip"`` the value is
            then the stored cell to round-off; with ``"polyfit"`` it is not.
        valid_range_eV: First and last valid cell of this line, or None,
            on the axis the table is interpolated on (photon energy as read;
            for Trzhaskovskaya a photoelectron-energy grid read as photon
            energy).
        threshold_eV: The binding energy used as the ionization threshold,
            or None when ``BindingEnergy`` has no value for this level — in
            which case no threshold was applied, and a value below the
            range is not known to be above threshold.
    """

    orbital: str
    status: LookupStatus
    value: float | None
    method: LookupMethod | None
    at_tabulated_cell: bool
    valid_range_eV: tuple[float, float] | None
    threshold_eV: float | None


@dataclass(frozen=True)
class CrossSectionLookup:
    """A cross-section with the state of every component behind it.

    ``value`` is the sum over ``components`` when every component either
    has a value or is a known zero; otherwise None. When all components
    are known zeros it is ``0.0`` (where :meth:`CrossSection.lookup`
    returns None). ``status`` summarises the components: a blocking state
    (``not_in_table``, ``outside_range_below`` without a value,
    ``outside_range_above``) wins, then the least supported state among
    the components that contributed a value. ``components`` is empty when
    the table does not carry the element or the subshell at all.
    """

    element: str
    orbital: str
    photon_energy: float
    table: str
    value: float | None
    status: LookupStatus
    components: tuple[ComponentLookup, ...]


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
        """Load one bundled cross-section table from ``data/_cache``.

        Same policy as :func:`toyomacro.data.paths._load_shipped_table`:
        these files are shipped, reviewed reference data, so a missing one
        raises rather than being rebuilt from a local CSV that is not part
        of this repository. It used to return an **empty table** in that
        case, which is worse than either — a lookup then reports "no such
        line" for a table that is simply absent.

        Raises:
            FileNotFoundError: if the table is missing and regeneration
                was not requested via ``TOYOMACRO_REGENERATE_DATA``.
        """
        cache_file = get_cache_dir() / cache_name
        if cache_file.exists():
            with open(cache_file, encoding="utf-8") as f:
                return json.load(f)

        if not os.environ.get(REGENERATE_ENV_VAR):
            raise FileNotFoundError(
                f"Bundled reference table {cache_name} is missing from "
                f"{get_cache_dir()}. It ships with this package and is not a "
                f"disposable cache; restore it (e.g. `git checkout` the file, "
                f"or reinstall). To rebuild it instead from a local "
                f"{csv_name}, set {REGENERATE_ENV_VAR}=1."
            )

        csv_path = get_common_data_path() / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{cache_name} is missing and {csv_name} was not found at "
                f"{csv_path}, so it cannot be rebuilt."
            )
        warnings.warn(
            f"Rebuilding {cache_name} from {csv_name}. The result is "
            f"unreviewed and may differ from the reference table this package "
            f"ships; check the diff before committing it.",
            UserWarning,
            stacklevel=3,
        )
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

    @classmethod
    def lookup(
        cls,
        element: str,
        orbital: str,
        photon_energy: float,
        table: str | None = None,
        *,
        extrapolate_below: bool = False,
    ) -> float | None:
        """
        Look up cross-section for an element, orbital, and photon energy.

        Inside a line's tabulated range the value is a monotone piecewise
        cubic (PCHIP) through the tabulated cells in log-log space, so at a
        tabulated energy it is the tabulated value. Above the range it is a
        power law through the last cells. **Below a line's first valid cell
        it returns None** (since v0.5.0): that is the threshold region,
        where shape resonances and Cooper minima move the cross-section by
        orders of magnitude and a power law has no support — Tl 5d at
        21.2 eV came out at 796 Mb on Yeh-Lindau, whose Table I prints it
        from 40.8 eV (53 Mb). ``extrapolate_below=True`` restores the
        earlier power law for this call only. A line with a single valid
        cell answers only at that cell's energy. The range is each line's
        own valid cells, not the table's energy grid. See
        :meth:`lookup_with_status` for which of these applied, and
        :meth:`set_interpolation` for the earlier whole-line polynomial,
        kept as ``"polyfit"``.

        Only what the table carries is returned. A subshell the table does
        not list for this element returns None; it is not estimated from
        other elements. That covers unoccupied subshells, deep levels the
        table leaves out (Yeh-Lindau lists none with a binding energy above
        about 1.5 keV — for those at HAXPES energies use
        ``table="scofield"``), and a few occupied valence lines Yeh-Lindau
        Table I does not print (K 4s, Ga 4p, Rb 5s, In 5p, Cs 6s, Ce 5d,
        Fr 7s), for which no bundled table covers UPS energies.

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
            extrapolate_below: Return the pre-v0.5.0 power law below a
                line's first valid cell instead of None. For reproducing
                earlier numbers; the value is unsupported by the table.

        Returns:
            Cross-section in the unit reported by :meth:`unit_info` for
            ``table`` — Mb for 'yeh_lindau' and 'scofield', ~10³ larger
            (inferred kb, unconfirmed) for 'trzhaskovskaya' — or None if
            the table does not carry the subshell for this element, the
            photon energy is below its binding energy, or it is below the
            line's first valid cell. Do not compare across tables without
            converting.
        """
        result = cls.lookup_with_status(
            element, orbital, photon_energy, table,
            extrapolate_below=extrapolate_below,
        )
        # A known zero (nothing ionizable here) stays None, as before
        # v0.5.0; only lookup_with_status reports it as 0.0.
        if result.value is None or result.value <= 0.0:
            return None
        return result.value

    @classmethod
    def lookup_with_status(
        cls,
        element: str,
        orbital: str,
        photon_energy: float,
        table: str | None = None,
        *,
        extrapolate_below: bool = False,
    ) -> CrossSectionLookup:
        """Look up a cross-section and say where every part of it came from.

        Same arguments and the same numbers as :meth:`lookup`, returned as
        a :class:`CrossSectionLookup` that carries the state of each j
        component (see :data:`LookupStatus`). A bare doublet label is
        summed only when every component either has a value or is a known
        zero (``below_threshold``, ``unoccupied``); one component outside
        its range or missing for an unestablished reason makes the sum
        None rather than a silent under-count. When every component is a
        known zero the value is ``0.0``.

        The ionization threshold is the :class:`BindingEnergy` value of
        each component, not the table's own binding-energy column. Where
        ``BindingEnergy`` has no value no threshold applies
        (``threshold_eV`` is None) — 147 of 754 Yeh-Lindau lines, 286 of
        1,562 Scofield and 190 of 1,240 Trzhaskovskaya lines, mostly valence
        levels plus Scofield's deep levels of Np to Fm — and
        ``outside_range_below`` then does not mean the energy is above
        threshold. ``below_threshold`` overrides the table where the two
        disagree: it withholds cells the table prints below the
        ``BindingEnergy`` value (47 on Yeh-Lindau, e.g. As 2s at Al K-alpha;
        73 on Scofield, e.g. Ar 1s1/2 at 3199.8 eV; 2,304 on
        Trzhaskovskaya, from its energy axis), as v0.4.0 did. Known limits
        of that source apply too (Co 3p1/2 and 3p3/2 are stored as 59 and
        60 eV, the reverse of the usual order).

        On Trzhaskovskaya the range is judged on the axis it is
        interpolated on, a photoelectron-energy grid read as photon energy:
        Au 4f at 150 eV is ``tabulated`` although its kinetic energy (about
        65 eV) is below the table's first cell (1,170 such components over
        the table's grid and 19 source energies).

        Returns:
            A :class:`CrossSectionLookup`.
        """
        table_name = table or cls._default_table
        data = cls._get_data(table)

        def done(value, status, components=()):
            return CrossSectionLookup(
                element=element, orbital=orbital,
                photon_energy=float(photon_energy), table=table_name,
                value=value, status=status, components=tuple(components),
            )

        elem_data = data.get("data", {}).get(element)
        if elem_data is None:
            return done(None, "not_in_table")

        base, j = _split_orbital(orbital)
        if j is not None:
            # A j-resolved request. Tables that store only bare subshells
            # cannot answer it, and splitting the total by an assumed
            # branching ratio would invent a number indistinguishable
            # from a tabulated one — the statistical ratio is not exact
            # (Scofield's own Si 2p3/2 : 2p1/2 is 1.966, not 2).
            keys: tuple[str, ...] = (orbital,)
        else:
            components = _J_COMPONENTS.get(base[-1:]) if len(base) == 2 else None
            if components is None or base in elem_data:
                # A non-doublet label, or a bare subshell the table stores
                # directly (Yeh-Lindau): that entry already *is* the total.
                keys = (base,)
            else:
                keys = tuple(f"{base}{s}" for s in components)

        if not any(k in elem_data for k in keys):
            # Nothing of this subshell in the table (or a j-resolved request
            # on a table keyed by bare subshells).
            return done(None, "not_in_table")

        parts = []
        for key in keys:
            if key in elem_data:
                # Each component carries its own threshold. Between the two
                # thresholds of a split doublet only the lower-BE member is
                # ionizable, and that is exactly what a measured envelope
                # contains — so gate per component, not on the subshell.
                parts.append(cls._evaluate_component(
                    element, key, elem_data[key], data, photon_energy,
                    extrapolate_below,
                ))
            elif _absent_component_is_unoccupied(table_name):
                # The table lists a component iff it is occupied, so an
                # absent one carries no electrons and contributes zero.
                parts.append(ComponentLookup(
                    orbital=key, status="unoccupied", value=0.0, method=None,
                    at_tabulated_cell=False, valid_range_eV=None,
                    threshold_eV=cls._get_binding_energy(element, key),
                ))
            else:
                # Absent for a reason this package has not established.
                # Summing what is left would be a silent under-count.
                parts.append(ComponentLookup(
                    orbital=key, status="not_in_table", value=None, method=None,
                    at_tabulated_cell=False, valid_range_eV=None,
                    threshold_eV=cls._get_binding_energy(element, key),
                ))

        for blocking in ("not_in_table", "outside_range_above", "outside_range_below"):
            if any(p.status == blocking and p.value is None for p in parts):
                return done(None, blocking, parts)

        contributing = [p for p in parts if p.status not in _KNOWN_ZERO]
        if not contributing:
            zero = "below_threshold" if any(
                p.status == "below_threshold" for p in parts) else "unoccupied"
            return done(0.0, zero, parts)

        total = 0.0
        for p in parts:
            total += p.value
        for weakest in ("outside_range_below", "extrapolated_above"):
            if any(p.status == weakest for p in contributing):
                return done(total, weakest, parts)
        return done(total, "tabulated", parts)

    @classmethod
    def _evaluate_component(
        cls,
        element: str,
        key: str,
        orbital_data: dict[str, Any],
        data: dict[str, Any],
        photon_energy: float,
        extrapolate_below: bool,
    ) -> ComponentLookup:
        """Evaluate one stored line: threshold first, then its own range."""
        threshold = cls._get_binding_energy(element, key)
        if threshold is not None and photon_energy < threshold:
            return ComponentLookup(
                orbital=key, status="below_threshold", value=0.0, method=None,
                at_tabulated_cell=False, valid_range_eV=cls._valid_range(orbital_data, data),
                threshold_eV=threshold,
            )
        status, value, method, at_cell, valid = cls._evaluate_range(
            orbital_data, data, photon_energy, extrapolate_below,
        )
        return ComponentLookup(
            orbital=key, status=status, value=value, method=method,
            at_tabulated_cell=at_cell, valid_range_eV=valid, threshold_eV=threshold,
        )

    @staticmethod
    def _valid_points(
        orbital_data: dict[str, Any] | None, data: dict[str, Any]
    ) -> list[tuple[float, float]]:
        if orbital_data is None:
            return []
        photon_energies = data.get("photon_energies", [])
        cross_sections = orbital_data.get("cross_sections", [])
        return sorted(
            (e, sigma)
            for e, sigma in zip(photon_energies, cross_sections)
            if sigma is not None and sigma > 0
        )

    @classmethod
    def _valid_range(
        cls, orbital_data: dict[str, Any], data: dict[str, Any]
    ) -> tuple[float, float] | None:
        points = cls._valid_points(orbital_data, data)
        return (points[0][0], points[-1][0]) if points else None

    @classmethod
    def _evaluate_range(
        cls,
        orbital_data: dict[str, Any] | None,
        data: dict[str, Any],
        photon_energy: float,
        extrapolate_below: bool,
    ) -> tuple[LookupStatus, float | None, LookupMethod | None, bool, tuple[float, float] | None]:
        """Place ``photon_energy`` against one line's own valid cells.

        Returns (status, value, method, at_tabulated_cell, valid_range).
        """
        points = cls._valid_points(orbital_data, data)
        if not points:
            return "not_in_table", None, None, False, None
        lo, hi = points[0][0], points[-1][0]
        at_cell = any(math.isclose(photon_energy, e, rel_tol=1e-12) for e, _ in points)

        if len(points) == 1:
            # One cell carries no slope: it answers at its own energy only.
            if at_cell:
                return "tabulated", points[0][1], "single_cell", True, (lo, hi)
            if photon_energy > hi:
                return "outside_range_above", None, None, False, (lo, hi)
            if extrapolate_below:
                # Pre-v0.5.0 behaviour: the single value at every energy.
                return "outside_range_below", points[0][1], "single_cell", False, (lo, hi)
            return "outside_range_below", None, None, False, (lo, hi)

        if photon_energy < lo and not at_cell:
            if not extrapolate_below:
                return "outside_range_below", None, None, False, (lo, hi)
            value = cls._interpolate_log_log(points, photon_energy)
            return "outside_range_below", value, "power_law_below", False, (lo, hi)
        value = cls._interpolate_log_log(points, photon_energy)
        if photon_energy > hi and not at_cell:
            return "extrapolated_above", value, "power_law_above", False, (lo, hi)
        return "tabulated", value, cls._interpolation, at_cell, (lo, hi)

    @classmethod
    def _lookup_direct(
        cls,
        element: str,
        orbital: str,
        photon_energy: float,
        table: str | None = None,
    ) -> float | _Refused | None:
        """:meth:`lookup_with_status` reduced to the pre-v0.5.0 return.

        Returns the value, ``None`` when the table has no data for this
        subshell at all, or ``_REFUSED`` when the table covers the
        subshell and still cannot answer here.
        """
        result = cls.lookup_with_status(element, orbital, photon_energy, table)
        if result.value is not None and result.value > 0.0:
            return result.value
        return _REFUSED if result.components else None

    @classmethod
    def _interp_one(
        cls,
        orbital_data: dict[str, Any] | None,
        data: dict[str, Any],
        photon_energy: float,
    ) -> float | None:
        """Value of one stored line at ``photon_energy``, no threshold applied.

        None below the line's first valid cell, or above a single cell.
        """
        _, value, _, _, _ = cls._evaluate_range(orbital_data, data, photon_energy, False)
        return value

    @classmethod
    def get_rsf(
        cls,
        element1: str,
        orbital1: str,
        element2: str,
        orbital2: str,
        photon_energy: float,
        table: str | None = None,
        *,
        extrapolate_below: bool = False,
    ) -> float | None:
        """
        Photoionization-cross-section ratio relative to a reference line.

        Returns ``sigma1 / sigma2`` at the given photon energy, both from
        the same table. Each is whatever ``lookup()`` returns, so either
        may be a power-law extrapolation above a line's tabulated
        energies; this ratio does not say so. Call
        :meth:`lookup_with_status` on each line to see the state. None if
        either line is not in the table, is below its binding energy, or
        is below its first valid cell (unless ``extrapolate_below``).

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
            extrapolate_below: Passed to :meth:`lookup` for both lines.

        Returns:
            The cross-section ratio, or None if either cross-section is
            unavailable.
        """
        sigma1 = cls.lookup(element1, orbital1, photon_energy, table,
                            extrapolate_below=extrapolate_below)
        sigma2 = cls.lookup(element2, orbital2, photon_energy, table,
                            extrapolate_below=extrapolate_below)

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
        """Return the table as loaded: the bundled JSON's own structure.

        ``{"photon_energies": [...], "data": {element: {orbital:
        {"binding_energy": eV or None, "cross_sections": [...]}}}}`` —
        one entry per photon energy, ``None`` where the source table has
        no entry, values in the unit :meth:`unit_info` reports. The same
        shape for every table. This is the live process cache, not a
        copy: read from it, or deep-copy before mutating.
        """
        return cls._get_data(table)

    # Interpolation inside a line's tabulated range. "pchip" passes through
    # every tabulated cell; "polyfit" is the whole-line polynomial used up to
    # v0.3.1 (and by the MATLAB Toyomacro suite), kept for reproducing
    # earlier results.
    INTERPOLATIONS: tuple[str, ...] = ("pchip", "polyfit")
    _interpolation: str = "pchip"

    # Polynomial order for the "polyfit" interpolation only.
    # MATLAB default = 3, DepthProfiler overrides to 6.
    _poly_order: int = 3

    @classmethod
    def set_interpolation(cls, method: str) -> None:
        """Choose how ``lookup()`` interpolates inside a tabulated range.

        Args:
            method: ``"pchip"`` (default since v0.4.0) — a monotone
                piecewise cubic through the tabulated cells in log-log
                space, which returns the tabulated value at a tabulated
                energy; or ``"polyfit"`` — one polynomial of order
                :meth:`set_poly_order` fitted through all the cells of a
                line, the behaviour up to v0.3.1. Left out cells in a
                leave-one-out test across the three bundled tables, the
                median error is 2.0% / 0.02% / 0.53% for "pchip" against
                8.9% / 0.17% / 1.16% for "polyfit" (Yeh-Lindau / Scofield /
                Trzhaskovskaya). Beyond the tabulated range both use the
                same power law through the end cells.
        """
        if method not in cls.INTERPOLATIONS:
            raise ValueError(f"Unknown interpolation '{method}'. Choose from {cls.INTERPOLATIONS}")
        cls._interpolation = method

    @classmethod
    def get_interpolation(cls) -> str:
        """Return the current interpolation method."""
        return cls._interpolation

    @classmethod
    def set_poly_order(cls, order: int) -> None:
        """Set the polynomial order of the ``"polyfit"`` interpolation.

        Has no effect on the default ``"pchip"`` interpolation; see
        :meth:`set_interpolation`.

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
        """Interpolate/extrapolate in log-log space.

        Inside the range: with ``"pchip"`` a monotone piecewise cubic
        through the points; with ``"polyfit"`` the MATLAB ``xps.CrossSection``
        behaviour, ``polyfit(log10(PE), log10(CS), polyOrder)`` over all
        valid points, then ``10^polyval(p, log10(hv))``.

        Beyond the range (either method): a power law through the last or
        first few points, to avoid high-order polynomial divergence.
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
        elif cls._interpolation == "pchip":
            xs = np.log10(np.array([p[0] for p in points]))
            ys = np.log10(np.array([p[1] for p in points]))
            return float(10.0 ** PchipInterpolator(xs, ys)(math.log10(x)))
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
