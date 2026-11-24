"""
Binding energy lookup for XPS analysis.

Provides fast O(1) lookup of core-level binding energies
using cached JSON data.

Usage:
    from toyomacro.data import BindingEnergy

    be = BindingEnergy.lookup('Si', '2p')      # → 99.0 eV
    be = BindingEnergy.from_string('Si2p')     # → 99.0 eV
    so = BindingEnergy.get_spin_orbit_split('Si', '2p')  # → 1.0 eV (integer table)
    ref = BindingEnergy.get_spin_orbit('Si', '2p')       # → split_eV 0.6, provisional
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from toyomacro.data.paths import load_binding_energy_data, load_spin_orbit_data

#: How far a spin-orbit splitting of the reference is supported.
SpinOrbitState = Literal[
    "adopted",      # checked and judged fit for the use; assigned by hand, to no
                    # entry yet
    "provisional",  # both binding energies are the source table's and no other
                    # bundled value contradicts the splitting: approximate
    "to_verify",    # negative as printed, or another value disagrees beyond
                    # tolerance: no position to show
    "not_covered",  # the Trzhaskovskaya tables do not hold both members
]

_J_PAIRS = {"p": ("3/2", "1/2"), "d": ("5/2", "3/2"), "f": ("7/2", "5/2")}


@dataclass(frozen=True)
class SpinOrbitReference:
    """One entry of the spin-orbit splitting reference.

    Attributes:
        element: Element symbol.
        subshell: ``'2p'``, ``'4f'``, ...
        j_plus: Label of the j = l + 1/2 member (``'2p3/2'``); None for s.
        j_minus: Label of the j = l - 1/2 member (``'2p1/2'``); None for s.
        split_eV: BE(j-) - BE(j+) as the source table prints the two
            binding energies, **sign kept** (eight 2018 pairs are
            negative: j- printed less bound than j+). ``0.0`` together with
            ``unresolved_at_precision`` means the two members are printed
            equal, not that the splitting is zero. None when no bundled
            table gives two different members.
        precision_eV: The resolution each binding energy is printed to:
            0.1 for the Trzhaskovskaya tables, None otherwise. It is not
            the uncertainty of ``split_eV``: the difference of two energies
            each rounded to 0.1 eV can be 0.1 eV off from rounding alone.
        source_table: ``'trzh2018'`` (``trzh2018_haxpes.json``),
            ``'trzh2019'`` (``trzh2019_inner.json``),
            ``'trzh2018+trzh2019'`` (the doublet straddles 1.5 keV, where
            the two papers divide: j+ from the first, j- from the second),
            ``'binding_energy'`` (``binding_energy.json``, only where the
            Trzhaskovskaya tables do not hold both members), or None.
        be_j_plus_eV, be_j_minus_eV: The two binding energies the source
            table prints.
        state: See :data:`SpinOrbitState`. Only ``adopted`` and
            ``provisional`` values are meant to be shown as a position,
            and a ``provisional`` one only as approximate.
        state_reason: Why the entry has its state, in words.
        statistical_ratio: (2j- + 1)/(2j+ + 1) = l/(l + 1): 0.5 for p,
            2/3 for d, 0.75 for f; None for s. Not a measured area ratio;
            see :meth:`toyomacro.data.CrossSection.get_spin_orbit_intensity_ratio`.
        unresolved_at_precision: The two members are printed equal, so
            the splitting is below the table's resolution, not zero.
    """

    element: str
    subshell: str
    j_plus: str | None
    j_minus: str | None
    split_eV: float | None
    precision_eV: float | None
    source_table: str | None
    be_j_plus_eV: float | None
    be_j_minus_eV: float | None
    state: SpinOrbitState
    state_reason: str
    statistical_ratio: float | None
    unresolved_at_precision: bool = False


class BindingEnergy:
    """
    Binding energy database for XPS core levels.

    All methods are class methods - no instantiation needed.
    Data is loaded lazily and cached for fast repeated access.
    """

    _data: dict[str, Any] | None = None
    _spin_orbit_data: dict[str, Any] | None = None

    @classmethod
    def _get_data(cls) -> dict[str, Any]:
        """Get cached data, loading if needed."""
        if cls._data is None:
            cls._data = load_binding_energy_data()
        return cls._data

    @classmethod
    def lookup(cls, element: str, orbital: str) -> float | None:
        """
        Look up binding energy for an element and orbital.

        Args:
            element: Element symbol (e.g., 'Si', 'Au')
            orbital: Orbital designation (e.g., '2p', '2p3/2', '4f7/2')

        Returns:
            Binding energy in eV, or None if not found

        Examples:
            >>> BindingEnergy.lookup('Si', '2p')
            99.0
            >>> BindingEnergy.lookup('Au', '4f7/2')
            84.0
        """
        data = cls._get_data()
        elements = data.get("elements", {})

        elem_data = elements.get(element)
        if elem_data is None:
            return None

        orbitals = elem_data.get("orbitals", {})

        # Direct match
        if orbital in orbitals:
            return orbitals[orbital]

        # Normalize orbital name (e.g., '2p' -> '2p3/2' for p,d,f orbitals)
        normalized = cls._normalize_orbital(orbital)
        if normalized in orbitals:
            return orbitals[normalized]

        return None

    @classmethod
    def from_string(cls, core_level: str) -> float | None:
        """
        Parse core-level string and look up binding energy.

        Args:
            core_level: Combined element and orbital string
                        (e.g., 'Si2p', 'Au4f7/2', 'O1s')

        Returns:
            Binding energy in eV, or None if not found

        Examples:
            >>> BindingEnergy.from_string('Si2p')
            99.0
            >>> BindingEnergy.from_string('Au4f7/2')
            84.0
        """
        element, orbital = cls._parse_core_level(core_level)
        if element is None or orbital is None:
            return None
        return cls.lookup(element, orbital)

    @classmethod
    def get_spin_orbit(cls, element: str, subshell: str) -> SpinOrbitReference:
        """Look up the spin-orbit splitting reference for a level.

        Every answer carries its source and state; a level the reference
        cannot settle is returned with its state and the reason, never as
        an exception and never as a splitting of zero.

        **What the value is.** The difference of two binding energies
        printed in Trzhaskovskaya & Yarzhemsky 2018 (lines bound by less
        than 1.5 keV) and 2019 (the others). Both papers call the column
        experimental and take it from Kibedi et al. 2008; that compilation
        was not read for this package, so which entries are measured, and
        whether they are free-atom or solid-state energies, is not
        established here. It is an elemental value: it says nothing about
        a chemical state.

        **How far to trust it.** Each energy is printed to 0.1 eV, so the
        difference can be 0.1 eV off from rounding alone. That is enough
        to show or assign a doublet partner. It is not enough to fix a
        splitting in a fit, and the fitters do not read this reference:
        ``toyomacro.core.xps_database.get_so_info`` keeps its own values,
        which differ from these by up to 0.15 eV for common levels. No
        entry is ``adopted``.

        Where the Trzhaskovskaya tables do not hold both members, a
        ``binding_energy.json`` pair is carried with state ``not_covered``
        when that table stores two different values. See
        ``docs/DATA_SOURCES.md``, *The spin-orbit reference*.

        Args:
            element: Element symbol (``'Si'``).
            subshell: ``'2p'``; a j-resolved label (``'2p3/2'``) is read as
                its subshell.

        Returns:
            A :class:`SpinOrbitReference`. An s level, a level the
            reference does not carry, and an element symbol it does not
            know (symbols are case-sensitive) come back as ``not_covered``
            with no value; the reason says which.

        Raises:
            ValueError: if ``subshell`` is not of the form ``'2p'`` or
                ``'2p3/2'`` with a j that exists for that l.

        Examples:
            >>> ref = BindingEnergy.get_spin_orbit('Si', '2p')
            >>> ref.split_eV, ref.precision_eV, ref.source_table, ref.state
            (0.6, 0.1, 'trzh2018', 'provisional')
        """
        if not isinstance(subshell, str) or not re.fullmatch(
            r"[1-7][spdf]([1357]/2)?", subshell
        ):
            raise ValueError(f"Not a subshell label: {subshell!r} (expected e.g. '2p')")
        base = subshell[:2]
        letter = base[-1]
        if subshell[2:] and subshell[2:] not in (("1/2",) if letter == "s" else _J_PAIRS[letter]):
            raise ValueError(f"No such level: {subshell!r} (no j = {subshell[2:]} for {letter})")
        if letter == "s":
            return SpinOrbitReference(
                element=element, subshell=base, j_plus=None, j_minus=None,
                split_eV=None, precision_eV=None, source_table=None,
                be_j_plus_eV=None, be_j_minus_eV=None, state="not_covered",
                state_reason="s level: j = 1/2 only, no spin-orbit doublet",
                statistical_ratio=None,
            )
        levels = cls._get_spin_orbit_data()["levels"]
        entry = levels.get(element, {}).get(base)
        if entry is None:
            j_hi, j_lo = _J_PAIRS[letter]
            l_qn = "pdf".index(letter) + 1
            return SpinOrbitReference(
                element=element, subshell=base,
                j_plus=f"{base}{j_hi}", j_minus=f"{base}{j_lo}",
                split_eV=None, precision_eV=None, source_table=None,
                be_j_plus_eV=None, be_j_minus_eV=None, state="not_covered",
                state_reason=(
                    "not in the reference: no bundled table holds a member "
                    "of this level"
                    if element in levels
                    else f"not in the reference: no element {element!r}"
                ),
                statistical_ratio=l_qn / (l_qn + 1),
            )
        return SpinOrbitReference(
            element=element, subshell=base,
            j_plus=entry["j_plus"], j_minus=entry["j_minus"],
            split_eV=entry["split_eV"], precision_eV=entry["precision_eV"],
            source_table=entry["source_table"],
            be_j_plus_eV=entry["be_j_plus_eV"], be_j_minus_eV=entry["be_j_minus_eV"],
            state=entry["state"], state_reason=entry["state_reason"],
            statistical_ratio=entry["statistical_ratio"],
            unresolved_at_precision=entry["unresolved_at_precision"],
        )

    @classmethod
    def spin_orbit_levels(cls) -> dict[str, list[str]]:
        """Every (element, subshell) the spin-orbit reference carries."""
        return {el: list(v) for el, v in cls._get_spin_orbit_data()["levels"].items()}

    @classmethod
    def _get_spin_orbit_data(cls) -> dict[str, Any]:
        if cls._spin_orbit_data is None:
            cls._spin_orbit_data = load_spin_orbit_data()
        return cls._spin_orbit_data

    @classmethod
    def get_spin_orbit_split(cls, element: str, orbital: str) -> float | None:
        """
        Get spin-orbit splitting for an orbital from the integer table.

        Calculates the energy difference between j=l-1/2 and j=l+1/2 states
        of ``binding_energy.json``, which stores mostly integer eV and
        stores many pairs with one value for both members: a 0.0 here
        means "one value stored", not "no splitting". Kept for
        compatibility; use :meth:`get_spin_orbit` for the reference value
        with its source and state (Si 2p: 0.6 eV there, 1.0 here).

        Args:
            element: Element symbol
            orbital: Base orbital (e.g., '2p', '3d', '4f')

        Returns:
            Spin-orbit splitting in eV (always positive), or None if unavailable

        Examples:
            >>> BindingEnergy.get_spin_orbit_split('Si', '2p')
            1.0  # 2p1/2 - 2p3/2
        """
        # Extract base orbital (e.g., '2p' from '2p3/2')
        base = cls._get_base_orbital(orbital)
        if base is None:
            return None

        # Get orbital letter (s, p, d, f)
        orbital_letter = base[-1]
        if orbital_letter == "s":
            return 0.0  # s orbitals have no SO splitting

        n = base[:-1]  # quantum number

        # Get both j states
        if orbital_letter == "p":
            j_low, j_high = f"{n}p1/2", f"{n}p3/2"
        elif orbital_letter == "d":
            j_low, j_high = f"{n}d3/2", f"{n}d5/2"
        elif orbital_letter == "f":
            j_low, j_high = f"{n}f5/2", f"{n}f7/2"
        else:
            return None

        be_low = cls.lookup(element, j_low)
        be_high = cls.lookup(element, j_high)

        if be_low is None or be_high is None:
            return None

        return abs(be_low - be_high)

    @classmethod
    def get_orbitals(cls, element: str) -> list[str]:
        """
        Get list of available orbitals for an element.

        Args:
            element: Element symbol

        Returns:
            List of orbital names with non-zero binding energies

        Examples:
            >>> BindingEnergy.get_orbitals('Si')
            ['1s', '2s', '2p1/2', '2p3/2']
        """
        data = cls._get_data()
        elements = data.get("elements", {})

        elem_data = elements.get(element)
        if elem_data is None:
            return []

        return list(elem_data.get("orbitals", {}).keys())

    @classmethod
    def get_atomic_number(cls, element: str) -> int | None:
        """
        Get atomic number for an element symbol.

        Args:
            element: Element symbol

        Returns:
            Atomic number (Z), or None if not found
        """
        data = cls._get_data()
        elements = data.get("elements", {})

        elem_data = elements.get(element)
        if elem_data is None:
            return None

        return elem_data.get("Z")

    @classmethod
    def get_element_symbol(cls, z: int) -> str | None:
        """
        Get element symbol from atomic number.

        Args:
            z: Atomic number

        Returns:
            Element symbol, or None if not found
        """
        data = cls._get_data()
        by_z = data.get("by_z", {})

        # JSON keys are strings
        return by_z.get(str(z))

    @classmethod
    def get_all_elements(cls) -> list[str]:
        """Get list of all element symbols."""
        data = cls._get_data()
        return list(data.get("elements", {}).keys())

    @classmethod
    def to_dict(cls) -> dict[str, Any]:
        """
        Get raw data as dict (for API responses).

        Returns:
            Complete binding energy data structure
        """
        return cls._get_data()

    @staticmethod
    def _normalize_orbital(orbital: str) -> str:
        """
        Normalize orbital name to canonical form.

        For p, d, f orbitals without j value, returns j=l+1/2 (main peak).

        Examples:
            '2p' -> '2p3/2'
            '3d' -> '3d5/2'
            '4f' -> '4f7/2'
            '2p3/2' -> '2p3/2' (unchanged)
        """
        # Already has j value
        if "/" in orbital:
            return orbital

        # Extract quantum number and letter
        match = re.match(r"(\d+)([spdf])", orbital)
        if not match:
            return orbital

        n, letter = match.groups()

        # s orbital - no j suffix needed
        if letter == "s":
            return orbital

        # Return main peak (higher j)
        j_map = {"p": "3/2", "d": "5/2", "f": "7/2"}
        return f"{n}{letter}{j_map[letter]}"

    @staticmethod
    def _get_base_orbital(orbital: str) -> str | None:
        """
        Extract base orbital from full designation.

        Examples:
            '2p3/2' -> '2p'
            '4f7/2' -> '4f'
            '2p' -> '2p'
        """
        match = re.match(r"(\d+[spdf])", orbital)
        if match:
            return match.group(1)
        return None

    @staticmethod
    @lru_cache(maxsize=256)
    def _parse_core_level(core_level: str) -> tuple[str | None, str | None]:
        """
        Parse core-level string into element and orbital.

        Examples:
            'Si2p' -> ('Si', '2p')
            'Au4f7/2' -> ('Au', '4f7/2')
            'O1s' -> ('O', '1s')
        """
        # Pattern: Element (1-2 letters) + orbital (digit + letter + optional j)
        match = re.match(r"([A-Z][a-z]?)(\d+[spdf](?:\d/\d)?)", core_level)
        if match:
            return match.group(1), match.group(2)
        return None, None
