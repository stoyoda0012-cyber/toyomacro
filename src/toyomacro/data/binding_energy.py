"""
Binding energy lookup for XPS analysis.

Provides fast O(1) lookup of core-level binding energies
using cached JSON data.

Usage:
    from toyomacro.data import BindingEnergy

    be = BindingEnergy.lookup('Si', '2p')      # → 99.0 eV
    be = BindingEnergy.from_string('Si2p')     # → 99.0 eV
    so = BindingEnergy.get_spin_orbit_split('Si', '2p')  # → 1.0 eV
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from toyomacro.data.paths import load_binding_energy_data


class BindingEnergy:
    """
    Binding energy database for XPS core levels.

    All methods are class methods - no instantiation needed.
    Data is loaded lazily and cached for fast repeated access.
    """

    _data: dict[str, Any] | None = None

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
    def get_spin_orbit_split(cls, element: str, orbital: str) -> float | None:
        """
        Get spin-orbit splitting for an orbital.

        Calculates the energy difference between j=l-1/2 and j=l+1/2 states.

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
