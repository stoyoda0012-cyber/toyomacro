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

# The bundled table carries an isotope mass number for these, since they
# have no standard atomic weight. Excluded from the element check below.
TRANSACTINIDES = {"Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Fl", "Mc", "Lv", "Ts", "Og"}

# IUPAC standard atomic weights (2021 conventional values) for every
# non-transactinide element in the bundled table.
STANDARD_ATOMIC_WEIGHT = {
    "Ac": 227.0, "Ag": 107.8682, "Al": 26.9815385, "Am": 243.0, "As": 74.921595,
    "Au": 196.966569, "B": 10.81, "Ba": 137.327, "Be": 9.0121831, "Bi": 208.98040,
    "Bk": 247.0, "C": 12.011, "Ca": 40.078, "Cd": 112.414, "Ce": 140.116,
    "Cf": 251.0, "Cm": 247.0, "Co": 58.933194, "Cr": 51.9961, "Cs": 132.90545196,
    "Cu": 63.546, "Dy": 162.500, "Er": 167.259, "Es": 252.0, "Eu": 151.964,
    "Fe": 55.845, "Fr": 223.0, "Ga": 69.723, "Gd": 157.25, "Ge": 72.630,
    "Hf": 178.486, "Ho": 164.93033, "In": 114.818, "Ir": 192.217, "K": 39.0983,
    "La": 138.90547, "Li": 6.94, "Lu": 174.9668, "Mg": 24.305, "Mn": 54.938044,
    "Mo": 95.95, "Na": 22.98976928, "Nb": 92.90637, "Nd": 144.242, "Ni": 58.6934,
    "No": 259.0, "Np": 237.0, "Os": 190.23, "P": 30.973761998, "Pa": 231.03588,
    "Pb": 207.2, "Pd": 106.42, "Pm": 145.0, "Po": 209.0, "Pr": 140.90766,
    "Pt": 195.084, "Pu": 244.0, "Ra": 226.0, "Rb": 85.4678, "Re": 186.207,
    "Rh": 102.90550, "Ru": 101.07, "S": 32.06, "Sb": 121.760, "Sc": 44.955908,
    "Se": 78.971, "Si": 28.085, "Sm": 150.36, "Sn": 118.710, "Sr": 87.62,
    "Ta": 180.94788, "Tb": 158.92535, "Tc": 98.0, "Te": 127.60, "Th": 232.0377,
    "Ti": 47.867, "Tl": 204.38, "Tm": 168.93422, "U": 238.02891, "V": 50.9415,
    "W": 183.84, "Y": 88.90584, "Yb": 173.045, "Zn": 65.38, "Zr": 91.224,
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


PLACEHOLDERS = {"AVERAGE", "Oxide"}


def _bundled_compounds() -> set[str]:
    """Every entry that is a compound rather than an element or placeholder.

    Every element symbol is at most two characters, so length alone
    separates them. Nothing subtler works: an earlier version of this
    helper also required a digit in the name, which silently exempted
    MgO, AlN, NaCl, GaN and any other 1:1 formula from the checks below.
    """
    return {
        name for name in CompoundDB.list_compounds() if len(name) > 2 and name not in PLACEHOLDERS
    }


def test_formulae_cover_every_bundled_compound():
    """No real compound escapes the checks below by not being listed."""
    unchecked = _bundled_compounds() - set(FORMULAE)
    assert not unchecked, (
        f"compound(s) added to compounds.json without a formula here: {sorted(unchecked)}"
    )


def test_element_weights_match_standard_atomic_weights():
    """Element entries are their standard atomic weight.

    The transactinides are excluded by construction: they have no
    standard atomic weight, and the table carries the mass number of a
    chosen isotope instead (Rf 261.11 against a commonly cited 267, for
    one). Every remaining element is checked at 0.1%, which is looser
    than the spread between IUPAC revisions of the conventional values
    and far tighter than any plausible transcription slip.
    """
    checked = {
        name
        for name in CompoundDB.list_compounds()
        if len(name) <= 2 and name not in TRANSACTINIDES
    }
    unlisted = checked - set(STANDARD_ATOMIC_WEIGHT)
    assert not unlisted, f"element(s) with no reference weight here: {sorted(unlisted)}"

    off = [
        (name, CompoundDB.get_property(name, "Mw"), STANDARD_ATOMIC_WEIGHT[name])
        for name in sorted(checked)
        if abs(CompoundDB.get_property(name, "Mw") / STANDARD_ATOMIC_WEIGHT[name] - 1) > 1e-3
    ]
    assert not off, f"element weights disagree with IUPAC values: {off}"
    assert len(checked) == 85


@pytest.mark.parametrize("name,formula", sorted(FORMULAE.items()))
def test_molecular_weight_matches_formula(name, formula):
    """Stored M is the formula weight, per molecule.

    Si3N4 is the reason this test exists: the bundled value was
    104.28346 g/mol, a digit transposition of the correct 140.28346,
    which inflated U by 34% and the reported IMFP by 11-26% over the
    50-2000 eV range TPP-2M was fitted on.

    The tolerance is 0.05%. The worst real deviation is GeO2 at 0.010%,
    which comes from the entry rounding Ge to 72.64; 0.05% leaves room
    for that class of rounding while still rejecting a wrong digit in
    any position, the failure mode being guarded against.
    """
    stored = CompoundDB.get_property(name, "Mw")
    expected = _formula_weight(formula)
    assert stored == pytest.approx(expected, rel=5e-4), (
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


def test_si3n4_entry_is_pinned():
    """Pin the entry whose molecular weight was corrected.

    Only Mw changed, from 104.28346. density and Eg are as they were and
    carry no recorded source; in particular this project does not know
    which phase they describe. The alpha and beta phases differ by under
    1% in density (3.18 and 3.20 g/cm3 from their lattice constants), so
    the stored 3.2 does not identify one, and the 3.44 g/cm3 tabulated
    by Tanuma, Powell & Penn, *Surf. Interface Anal.* **17**, 927
    (1991), Table 5 -- which states no phase -- is above both.
    """
    props = CompoundDB.get_properties("Si3N4")
    assert props == {"Nv": 32.0, "density": 3.2, "Mw": 140.28346, "Eg": 5.3}
