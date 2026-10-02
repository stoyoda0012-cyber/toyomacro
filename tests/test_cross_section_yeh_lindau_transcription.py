"""Thirteen cells of the bundled Yeh–Lindau table, pinned to the primary source.

Two independent transcriptions of Yeh & Lindau, *At. Data Nucl. Data
Tables* **32**, 1 (1985), Table I — the CSV this package's
``cross_section.json`` was generated from, and a spreadsheet kept by a
downstream project — were compared cell by cell over 67 elements
(3,942 values). They disagreed in thirteen cells. Each of the thirteen was
then read from Table I itself (journal pages 7–10), and this file pins
what the table says. Seven cells in the shipped JSON were wrong — three
by a factor of 100, one by a factor of 10, three in the last digit — and
were corrected; the other six were right as shipped.

The values are read from the raw table (``to_dict``), not through
``lookup()``: ``lookup`` fits one polynomial through every tabulated
point of a line and does not return the table value at a grid energy.
Three further defects were structural, not numeric: Ta 4s was keyed
``4S``, Ir 6s sat under a mistyped element ``lr``, and the CSV header
row had become an element ``Photon``. The first two made ``lookup()``
miss a tabulated line and extrapolate it from neighbouring elements
instead. The tests at the end pin the keys and those two lines.

Beyond these thirteen cells and two lines, agreement between the two
transcriptions is the only check the table has had.
"""

import pytest

from toyomacro.data.cross_section import CrossSection

# (element, orbital, photon energy / eV) -> cross section / Mb as printed
# in Table I, with the journal page. Entries marked "corrected" were wrong
# in the shipped JSON before 2026-10 and are listed with the old value.
TABLE_I = {
    # corrected
    ("Ar", "3p", 600.0): (0.038, 7, "was 3.8"),
    ("Cu", "4s", 1486.6): (0.00027, 8, "was 0.027"),
    ("As", "4s", 151.4): (0.072, 8, "was 7.2"),
    ("Zn", "3d", 8047.8): (4.7e-06, 8, "was 4.7e-07"),
    ("Fe", "3d", 16.7): (3.576, 8, "was 3.676"),
    ("Se", "4p", 26.8): (3.089, 8, "was 3.069"),
    ("Pd", "3p", 1041.0): (0.2869, 9, "was 0.2859"),
    # confirmed as shipped (the other transcription had these wrong)
    ("Cl", "3p", 26.8): (2.774, 7, "confirmed"),
    ("La", "4d", 1253.6): (0.1343, 10, "confirmed"),
    ("C", "2p", 10.2): (12.72, 7, "confirmed"),
    ("S", "3p", 8047.8): (4.8e-06, 7, "confirmed"),
    ("Nb", "3p", 8047.8): (0.0016, 9, "confirmed"),
    ("Cs", "5p", 1486.6): (0.0068, 10, "confirmed"),
}


@pytest.fixture(scope="module")
def table():
    return CrossSection.to_dict("yeh_lindau")


@pytest.mark.parametrize(
    ("element", "orbital", "hv"), sorted(TABLE_I), ids=lambda v: str(v)
)
def test_cell_matches_table_i(table, element, orbital, hv):
    expected, page, note = TABLE_I[(element, orbital, hv)]
    column = table["photon_energies"].index(hv)
    got = table["data"][element][orbital]["cross_sections"][column]
    # The table prints four significant figures at most; the JSON stores
    # the printed value, so the comparison is exact up to float parsing.
    assert got == pytest.approx(expected, rel=1e-12), (
        f"{element} {orbital} @ {hv} eV: JSON has {got}, Table I (p. {page}) "
        f"prints {expected} ({note})"
    )


def test_corrected_cells_sit_between_their_neighbours(table):
    """The three factor-of-100 corrections restore monotone falloff.

    Along each of these lines the cross section decreases with photon
    energy on both sides of the corrected cell; a cell 100x off broke
    that. Zn 3d's corrected cell is the last column and cannot be
    bracketed. This is a plausibility check, not a second pin.
    """
    E = table["photon_energies"]
    for element, orbital, hv in [
        ("Ar", "3p", 600.0),
        ("Cu", "4s", 1486.6),
        ("As", "4s", 151.4),
    ]:
        row = table["data"][element][orbital]["cross_sections"]
        i = E.index(hv)
        assert row[i - 1] > row[i] > row[i + 1], (element, orbital, hv, row[i - 1 : i + 2])


PERIODIC = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr"
).split()


def test_every_key_is_an_element_and_a_subshell(table):
    """Table I covers Z = 1-103 and nothing else; subshells are `nl`."""
    import re

    assert sorted(table["data"]) == sorted(PERIODIC), (
        set(table["data"]) ^ set(PERIODIC)
    )
    bad = [
        (el, orb)
        for el, rec in table["data"].items()
        for orb in rec
        if not re.fullmatch(r"[1-7][spdf]", orb)
    ]
    assert bad == []


# Two lines that were present but unreachable: Table I, pages 11 and 12.
_NONE = [None] * 10
RELABELLED = {
    ("Ta", "4s"): (470.7, _NONE + [0.106, 0.073, 0.05, 0.038, 0.029, 0.0014]),
    ("Ir", "6s"): (
        7.2,
        [None, 0.015, 0.053, 0.073, 0.082, 0.051, 0.027, 0.022, 0.015, 0.0084,
         0.003, 0.0018, 0.0012, 0.00086, 0.00058, 2.7e-05],
    ),
}


@pytest.mark.parametrize(("element", "orbital"), sorted(RELABELLED), ids=lambda v: str(v))
def test_relabelled_line_matches_table_i(table, element, orbital):
    be, values = RELABELLED[(element, orbital)]
    rec = table["data"][element][orbital]
    assert rec["binding_energy"] == be
    assert rec["cross_sections"] == values


@pytest.mark.parametrize(("element", "orbital"), sorted(RELABELLED), ids=lambda v: str(v))
def test_relabelled_line_is_looked_up_not_extrapolated(element, orbital):
    """Under the old keys `lookup()` fell through to the Z-based
    extrapolation from other elements and returned a number that looked
    tabulated. Now the line is found directly."""
    assert orbital in CrossSection.get_available_orbitals(element, table="yeh_lindau")
    direct = CrossSection._lookup_direct(element, orbital, 1486.6, "yeh_lindau")
    assert direct is not None
    assert CrossSection.lookup(element, orbital, 1486.6, table="yeh_lindau") == direct
