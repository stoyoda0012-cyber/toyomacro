"""Internal consistency of the bundled ``compounds.json`` parameters.

These tests do not check whether an entry describes any particular
sample — most entries carry no recorded source, and that limitation is
documented in ``docs/DATA_SOURCES.md``. They check the one thing that
*is* checkable without a source: that each entry is arithmetically
coherent with its own chemical formula.

That is worth pinning because TPP-2M combines the parameters as
``U = Nv·ρ/M`` (Tanuma, Powell & Penn, *Surf. Interface Anal.* **21**,
165 (1994), Eqn 4e). ``Nv`` and ``M`` must therefore be counted over the
*same* unit — per molecule for a compound. A molecular weight that is
merely mistyped stays plausible-looking and silently rescales every IMFP
computed from that entry.
"""

from __future__ import annotations

import pytest

from toyomacro.data import CompoundDB

# IUPAC standard atomic weights (2021 conventional values), for the
# elements appearing in the bundled compound formulae only.
ATOMIC_WEIGHT = {
    "Al": 26.9815385,
    "As": 74.921595,
    "C": 12.011,
    "Ga": 69.723,
    "Ge": 72.630,
    "Hf": 178.486,
    "N": 14.006703,
    "O": 15.999405,
    "Si": 28.085,
    "Sr": 87.62,
    "Ta": 180.94788,
    "Ti": 47.867,
    "Zr": 91.224,
}

# Group valence electrons per atom, as TPP-2M counts them.
VALENCE = {
    "Al": 3, "As": 5, "C": 4, "Ga": 3, "Ge": 4, "Hf": 4, "N": 5,
    "O": 6, "Si": 4, "Sr": 2, "Ta": 5, "Ti": 4, "Zr": 4,
}

# Every real compound in the bundled table, as (element, count) pairs.
# `AVERAGE` and `Oxide` are excluded: they are not materials and have no
# formula to check against.
FORMULAE = {
    "Al2O3": [("Al", 2), ("O", 3)],
    "GaAs": [("Ga", 1), ("As", 1)],
    "GeO2": [("Ge", 1), ("O", 2)],
    "HfO2": [("Hf", 1), ("O", 2)],
    "Si3N4": [("Si", 3), ("N", 4)],
    "SiC": [("Si", 1), ("C", 1)],
    "SiO2": [("Si", 1), ("O", 2)],
    "SrTiO3": [("Sr", 1), ("Ti", 1), ("O", 3)],
    "Ta2O5": [("Ta", 2), ("O", 5)],
    "TiO2": [("Ti", 1), ("O", 2)],
    "ZrO2": [("Zr", 1), ("O", 2)],
}


def _formula_weight(formula: list[tuple[str, int]]) -> float:
    return sum(ATOMIC_WEIGHT[element] * n for element, n in formula)


def test_formulae_cover_every_bundled_compound():
    """No real compound escapes the checks below by not being listed."""
    entries = set(CompoundDB.list_compounds())
    elements_and_placeholders = {"AVERAGE", "Oxide"}
    compounds = {
        name
        for name in entries
        if name not in elements_and_placeholders and len(name) > 2 and any(c.isdigit() for c in name)
    } | {"GaAs", "SiC"}
    assert compounds == set(FORMULAE), (
        "a compound was added to compounds.json without a formula here; "
        f"unchecked: {sorted(compounds - set(FORMULAE))}"
    )


@pytest.mark.parametrize("name,formula", sorted(FORMULAE.items()))
def test_molecular_weight_matches_formula(name, formula):
    """Stored M is the formula weight, per molecule.

    Si3N4 is the reason this test exists: the bundled value was
    104.28346 g/mol, a digit transposition of the correct 140.28346,
    which inflated U by 34% and the reported IMFP by 11-20% over the
    50-2000 eV range TPP-2M was fitted on.
    """
    stored = CompoundDB.get_property(name, "Mw")
    expected = _formula_weight(formula)
    assert stored == pytest.approx(expected, rel=2e-3), (
        f"{name}: stored M={stored} g/mol, formula weight={expected:.5f} g/mol"
    )


@pytest.mark.parametrize("name,formula", sorted(FORMULAE.items()))
def test_valence_count_is_per_molecule(name, formula):
    """Nv is counted per molecule, which is what makes M per-molecule right.

    If either were per atom the pair would be inconsistent and U wrong,
    without any single value looking out of place.
    """
    stored = CompoundDB.get_property(name, "Nv")
    expected = sum(VALENCE[element] * n for element, n in formula)
    assert stored == pytest.approx(float(expected))


def test_si3n4_is_the_alpha_phase_parameter_set():
    """Pin the entry whose molecular weight was corrected.

    density and Eg match the alpha phase as carried in this project's
    upstream table; the beta phase (3.44 g/cm3, 5.25 eV) is the set
    tabulated by Tanuma, Powell & Penn, *Surf. Interface Anal.* **17**,
    927 (1991), Table 5. The two are not interchangeable, and only the
    molecular weight -- common to both phases -- was changed.
    """
    props = CompoundDB.get_properties("Si3N4")
    assert props == {"Nv": 32.0, "density": 3.2, "Mw": 140.28346, "Eg": 5.3}
