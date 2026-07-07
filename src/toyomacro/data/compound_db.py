"""
Compound property database for XPS analysis.

Provides physical properties needed for IMFP calculations:
- Valence electrons (Nv)
- Density (g/cm³)
- Molecular weight (g/mol)
- Band gap (eV)

Usage:
    from toyomacro.data import CompoundDB

    props = CompoundDB.get_properties('SiO2')
    # → {'Nv': 16, 'density': 2.2, 'Mw': 60.08, 'Eg': 8.9}
"""

from __future__ import annotations

from typing import Any

from toyomacro.data.paths import load_compound_data


class CompoundDB:
    """
    Compound property database for IMFP and quantitative analysis.

    All methods are class methods - no instantiation needed.
    """

    _data: dict[str, Any] | None = None

    @classmethod
    def _get_data(cls) -> dict[str, Any]:
        """Get cached data, loading if needed."""
        if cls._data is None:
            cls._data = load_compound_data()
        return cls._data

    @classmethod
    def get_properties(cls, compound: str) -> dict[str, float] | None:
        """
        Get physical properties for a compound.

        Args:
            compound: Compound name (e.g., 'SiO2', 'Al2O3', 'Si')

        Returns:
            Dict with keys: Nv, density, Mw, Eg (whichever are available)
            Returns None if compound not found

        Examples:
            >>> CompoundDB.get_properties('SiO2')
            {'Nv': 16.0, 'density': 2.2, 'Mw': 60.08, 'Eg': 8.9}
        """
        data = cls._get_data()
        return data.get(compound)

    @classmethod
    def get_property(cls, compound: str, prop: str) -> float | None:
        """
        Get a single property for a compound.

        Args:
            compound: Compound name
            prop: Property name ('Nv', 'density', 'Mw', or 'Eg')

        Returns:
            Property value, or None if not found
        """
        props = cls.get_properties(compound)
        if props is None:
            return None
        return props.get(prop)

    @classmethod
    def has_compound(cls, compound: str) -> bool:
        """
        Check if a compound exists in the database.

        Args:
            compound: Compound name

        Returns:
            True if compound exists
        """
        data = cls._get_data()
        return compound in data

    @classmethod
    def list_compounds(cls) -> list[str]:
        """
        Get list of all compound names.

        Returns:
            List of compound names
        """
        data = cls._get_data()
        return list(data.keys())

    @classmethod
    def search(cls, query: str) -> list[str]:
        """
        Search for compounds by partial name match.

        Args:
            query: Search string (case-insensitive)

        Returns:
            List of matching compound names
        """
        data = cls._get_data()
        query_lower = query.lower()
        return [name for name in data.keys() if query_lower in name.lower()]

    @classmethod
    def get_average_properties(cls) -> dict[str, float]:
        """
        Get average properties (fallback values).

        Used when a specific compound is not found.

        Returns:
            Dict with average values for Nv, density, Mw, Eg
        """
        data = cls._get_data()
        avg = data.get("AVERAGE")
        if avg:
            return avg

        # Calculate from data if AVERAGE not present
        nv_vals = []
        density_vals = []
        mw_vals = []
        eg_vals = []

        for name, props in data.items():
            if name == "AVERAGE":
                continue
            if "Nv" in props:
                nv_vals.append(props["Nv"])
            if "density" in props:
                density_vals.append(props["density"])
            if "Mw" in props:
                mw_vals.append(props["Mw"])
            if "Eg" in props:
                eg_vals.append(props["Eg"])

        return {
            "Nv": sum(nv_vals) / len(nv_vals) if nv_vals else 4.0,
            "density": sum(density_vals) / len(density_vals) if density_vals else 5.0,
            "Mw": sum(mw_vals) / len(mw_vals) if mw_vals else 50.0,
            "Eg": sum(eg_vals) / len(eg_vals) if eg_vals else 1.0,
        }

    @classmethod
    def to_dict(cls) -> dict[str, Any]:
        """
        Get raw data as dict (for API responses).

        Returns:
            Complete compound data structure
        """
        return cls._get_data()
