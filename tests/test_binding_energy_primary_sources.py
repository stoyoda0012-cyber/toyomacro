"""Bundled binding energies against the primary sources they rest on.

The table follows the LBNL X-ray Data Booklet (Table 1-1), whose metal
values marked with a dagger come from Fuggle & Martensson, J. Electron
Spectrosc. Relat. Phenom. 21, 275 (1980), doi:10.1016/0368-2048(80)85056-0.
Where the Booklet and its source disagree, the source wins.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from toyomacro.data import BindingEnergy

TABLE = Path(__file__).parents[1] / "src" / "toyomacro" / "data" / "_cache" / "binding_energy.json"


def test_co_3p_is_unresolved_as_fuggle_and_martensson_print_it():
    """Fuggle & Martensson 1980, Table 2: Co 3p is one unresolved level,
    58.8 eV (instrument A) and 59.0 eV (instrument D), printed between the
    3p1/2 and 3p3/2 columns as for Mn (47.2, 47.2) and Fe (52.8, 52.6). The
    Booklet lists Mn and Fe under both columns with one value, but Co as
    58.9 / 59.9: a misprint of 58.9, which put 3p3/2 above 3p1/2. Bearden &
    Burr 1967 (Rev. Mod. Phys. 39, 125, Table I) also give one Co M_II,III
    level, 59.5 +/- 0.3 eV. The table stores integer eV: 59 for both.
    """
    assert BindingEnergy.lookup("Co", "3p1/2") == 59.0
    assert BindingEnergy.lookup("Co", "3p3/2") == 59.0
    assert BindingEnergy.lookup("Co", "3p") == 59.0
    # The resolved neighbours keep their order (same source, same table)
    assert BindingEnergy.lookup("Ni", "3p1/2") == 68.0 and BindingEnergy.lookup("Ni", "3p3/2") == 66.0
    assert BindingEnergy.lookup("Cu", "3p1/2") == 77.0 and BindingEnergy.lookup("Cu", "3p3/2") == 75.0


def test_no_spin_orbit_pair_is_stored_inverted():
    """Over the whole table, the j = l - 1/2 member is never less bound
    than the j = l + 1/2 member. Equal values are stored for unresolved
    pairs, and for some pairs of Np-Lr that the table stores collapsed."""
    elements = json.loads(TABLE.read_text(encoding="utf-8"))["elements"]
    inverted = []
    for el, v in elements.items():
        o = v["orbitals"]
        for n in "1234567":
            for l, (lo, hi) in {"p": ("1/2", "3/2"), "d": ("3/2", "5/2"),
                                "f": ("5/2", "7/2")}.items():
                a, b = o.get(f"{n}{l}{lo}"), o.get(f"{n}{l}{hi}")
                if a is not None and b is not None and a < b:
                    inverted.append(f"{el} {n}{l}: {a} < {b}")
    assert inverted == []
    assert all(re.fullmatch(r"\d[spdf](\d/2)?", k)
               for v in elements.values() for k in v["orbitals"])
