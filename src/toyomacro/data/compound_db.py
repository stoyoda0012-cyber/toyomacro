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

from toyomacro.data.paths import load_compound_data, load_compound_provenance


class CompoundDB:
    """
    Compound property database for IMFP and quantitative analysis.

    All methods are class methods - no instantiation needed.
    """

    _data: dict[str, Any] | None = None
    _provenance: dict[str, Any] | None = None

    @classmethod
    def _get_data(cls) -> dict[str, Any]:
        """Get cached data, loading if needed."""
        if cls._data is None:
            cls._data = load_compound_data()
        return cls._data

    @classmethod
    def _get_provenance(cls) -> dict[str, Any]:
        if cls._provenance is None:
            cls._provenance = load_compound_provenance()
        return cls._provenance

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
    def get_provenance(cls, compound: str) -> dict[str, Any] | None:
        """Where this entry's values came from, per field.

        The table's values and their provenance are held apart on
        purpose: :meth:`get_properties` returns numbers and nothing else,
        so adding provenance cannot change what a calculation sees.

        Each field carries an ``availability`` — what is known about the
        basis for the value — and, where there is one, an ``origin``:
        how the value came to be what it is. The two are independent.
        Most entries are ``not_recorded`` with an ``asserted`` origin:
        somebody typed a number in and the basis was never written down.
        That a test can now re-derive the same number does not make its
        origin ``derived``; only having been computed here does. No
        stored value qualifies -- the one field this project touched,
        the Si3N4 molecular weight, is ``corrected``: a transposition
        repaired, which records what the value was and what established
        the new reading rather than claiming it was authored here.

        A ``phase`` record sits alongside the fields, because density
        depends on it and the table itself carries no phase label. It
        can be ``unknown`` — for GeO2 that is the largest uncertainty in
        the entry, its two forms differing by about 47% in density. It
        does not always bear on density: for SiC the ambiguity is in the
        band gap, which is what ``applies_to`` records.

        Args:
            compound: Compound or element name, as in
                :meth:`get_properties`.

        Returns:
            Mapping of field name to its provenance record, or None if
            the compound is not in the table.

        Examples:
            >>> CompoundDB.get_provenance('Si3N4')['Mw']['availability']
            'known'
            >>> CompoundDB.get_provenance('Si3N4')['Mw']['origin']['kind']
            'corrected'
            >>> CompoundDB.get_provenance('SiO2')['density']['availability']
            'not_recorded'
        """
        return cls._get_provenance()["entries"].get(compound)

    @classmethod
    def get_comparisons(cls, compound: str) -> list[dict[str, Any]]:
        """Published values for this entry found *after* it was written.

        These are not sources. Where the bundled value and a published
        one disagree, both are reported; merging the two would fabricate
        a provenance the entry does not have.

        Returns:
            A list, empty when no counterpart has been found.
        """
        return cls._get_provenance()["comparisons"].get(compound, [])

    @classmethod
    def get_investigations(cls) -> list[dict[str, Any]]:
        """What has been searched for sources, and what it found.

        Separate from per-entry provenance because "no published
        counterpart was found" is a fact about a search — including its
        limits — and not a property of the material.
        """
        return cls._get_provenance()["investigation"]

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
