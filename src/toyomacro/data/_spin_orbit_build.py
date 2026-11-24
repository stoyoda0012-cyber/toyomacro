"""Build the bundled spin-orbit reference, ``_cache/spin_orbit.json``.

    python -m toyomacro.data._spin_orbit_build OUT.json

Private. The table is a pure function of four bundled tables and of the
fitters' in-code splittings, so it is rebuilt, never edited by hand, and
``tests/test_spin_orbit_reference.py`` fails when the committed file is
not what :func:`build` returns.

Rules, in the order they are applied to one element and n, l (l = p, d, f):

1. The two binding energies are those of the Trzhaskovskaya tables
   (``trzh2018_haxpes.json`` holds lines bound by less than 1.5 keV,
   ``trzh2019_inner.json`` lines from 1.5 keV to about 11 keV; no line is
   in both). A doublet whose members fall on either side of 1.5 keV takes
   one member from each.
2. With both members: the splitting is BE(j-) - BE(j+), sign kept.
   Negative: ``to_verify``. Printed equal: ``provisional`` and
   ``unresolved_at_precision``. Positive: ``to_verify`` when another value
   disagrees beyond tolerance or when there is no other value to compare
   it with, else ``provisional``.
3. Without both members: ``not_covered``. The ``binding_energy.json`` pair
   is carried when that table stores two different values.

Other values a positive splitting is compared with:

- ``binding_energy.json`` and ``trzhaskovskaya.json`` (2001/2002),
  tolerance 1.1 eV. Both are mostly integer eV: two values each rounded
  to 1 eV can put their difference 1.0 eV off, and two each printed to
  0.1 eV another 0.1 eV. A pair those tables store as one value is not
  compared. It does not say the splitting is zero (for Np 4p the papers
  print members 241.7 eV apart), and it is not counted as agreement.
- the in-code ``_SO_SPLITS`` of ``toyomacro.core.xps_database``, tolerance
  0.15 eV: the 0.1 eV that rounding can put the reference off, plus 0.05
  eV allowed for the in-code value, whose uncertainty is not recorded.
  The 0.05 is a convention, not a derived bound.

No rule adopts a value: ``adopted`` is assigned by hand, to nothing yet.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from toyomacro.data.paths import get_cache_dir

TOL_FINE_EV = 0.15
TOL_INTEGER_EV = 1.1

#: (j+ suffix, j- suffix) per l
_J = {"p": ("3/2", "1/2"), "d": ("5/2", "3/2"), "f": ("7/2", "5/2")}
_L = {"p": 1, "d": 2, "f": 3}

_OTHER_NAME = {
    "binding_energy": "binding_energy.json",
    "trzh2001": "trzhaskovskaya.json (2001/2002)",
    "in_code": "the fitters' in-code value",
}

META = {
    "description": (
        "Spin-orbit splittings, one entry per element and n, l (l = p, d, f). "
        "split_eV = BE(j-) - BE(j+) of the two binding energies as the source "
        "prints them, sign kept. statistical_ratio = (2j- + 1)/(2j+ + 1). "
        "unresolved_at_precision: the source prints the two members equal, so "
        "the splitting is below its resolution, not zero. A reference for "
        "showing and assigning a doublet partner, not a fit constraint."
    ),
    "built_by": "python -m toyomacro.data._spin_orbit_build",
    "sources": {
        "trzh2018": (
            "trzh2018_haxpes.json: Trzhaskovskaya & Yarzhemsky, ADNDT 119, 99 "
            "(2018), Table 1, the column the paper calls 'Experimental "
            "subshell binding energy', taken there from Kibedi et al., NIM A "
            "589, 202 (2008); printed to 0.1 eV"
        ),
        "trzh2019": (
            "trzh2019_inner.json: Trzhaskovskaya & Yarzhemsky, ADNDT 129-130, "
            "101280 (2019), Table 1, same column and compilation; printed to "
            "0.1 eV"
        ),
        "trzh2018+trzh2019": (
            "the j+ member from trzh2018_haxpes.json and the j- member from "
            "trzh2019_inner.json: the doublet straddles 1.5 keV, where the "
            "two papers divide"
        ),
        "binding_energy": (
            "binding_energy.json; carried only where the Trzhaskovskaya "
            "tables do not hold both members and this table stores two "
            "different values. Its resolution is not recorded."
        ),
    },
    "states": {
        "adopted": "checked and judged fit for the use; assigned by hand, to no entry yet",
        "provisional": (
            "both binding energies are those of the source table, and at "
            "least one other bundled value agrees with the splitting within "
            "tolerance while none disagrees; or the two members are printed "
            "equal. Approximate: each energy is printed to 0.1 eV, and "
            "whether the compiled energies are free-atom or solid-state "
            "values is not established"
        ),
        "to_verify": (
            "negative as printed, another value disagrees beyond tolerance, "
            "or there is no other value to compare it with"
        ),
        "not_covered": "the Trzhaskovskaya tables do not hold both members",
    },
    "tolerances_eV": {"in_code": TOL_FINE_EV, "integer_tables": TOL_INTEGER_EV},
}


def _load(name: str) -> dict[str, Any]:
    with open(get_cache_dir() / name, encoding="utf-8") as f:
        return json.load(f)


def _in_code_splits() -> dict[str, float]:
    from toyomacro.core.xps_database import _SO_SPLITS

    return dict(_SO_SPLITS)


def _stored_pair(table: dict[str, Any], el: str, jp: str, jm: str) -> tuple[Any, Any]:
    """(BE(j+), BE(j-)) of a table whose lines are dicts with ``binding_energy``."""
    lines = table.get(el, {})
    a, b = lines.get(jp), lines.get(jm)
    return (
        a.get("binding_energy") if isinstance(a, dict) else None,
        b.get("binding_energy") if isinstance(b, dict) else None,
    )


def _split(be_plus: float | None, be_minus: float | None) -> float | None:
    if be_plus is None or be_minus is None:
        return None
    return round(be_minus - be_plus, 2)


def build() -> dict[str, Any]:
    """Return the whole reference, ``{"_meta": ..., "levels": ...}``."""
    be_raw = _load("binding_energy.json")["elements"]
    be = {el: v["orbitals"] for el, v in be_raw.items()}
    t18 = _load("trzh2018_haxpes.json")["data"]
    t19 = _load("trzh2019_inner.json")["data"]
    t01 = _load("trzhaskovskaya.json")["data"]
    in_code = _in_code_splits()

    z_of = {el: v["Z"] for el, v in be_raw.items()}
    found: set[tuple[str, str]] = set()
    for table in (be, t18, t19, t01):
        for el, lines in table.items():
            for label in lines:
                if len(label) >= 2 and label[1] in _J:
                    found.add((el, label[:2]))
    missing_z = sorted({el for el, _ in found if el not in z_of})
    if missing_z:
        raise ValueError(f"no atomic number for {missing_z}")

    levels: dict[str, dict[str, Any]] = {}
    for el, nl in sorted(found, key=lambda x: (z_of[x[0]], x[1])):
        letter = nl[1]
        jp, jm = nl + _J[letter][0], nl + _J[letter][1]

        p18, m18 = _stored_pair(t18, el, jp, jm)
        p19, m19 = _stored_pair(t19, el, jp, jm)
        be_plus = p18 if p18 is not None else p19
        be_minus = m18 if m18 is not None else m19
        src_plus = "trzh2018" if p18 is not None else ("trzh2019" if p19 is not None else None)
        src_minus = "trzh2018" if m18 is not None else ("trzh2019" if m19 is not None else None)

        be_pair = (be.get(el, {}).get(jp), be.get(el, {}).get(jm))
        others = {
            "binding_energy": _split(*be_pair),
            "trzh2001": _split(*_stored_pair(t01, el, jp, jm)),
        }
        # one value stored for both members says nothing about the splitting
        compared = {k: v for k, v in others.items() if v not in (None, 0)}
        tolerance = dict.fromkeys(compared, TOL_INTEGER_EV)
        if el + nl in in_code:
            compared["in_code"] = in_code[el + nl]
            tolerance["in_code"] = TOL_FINE_EV

        precision: float | None
        unresolved = False
        if src_plus and src_minus:
            source = src_plus if src_plus == src_minus else f"{src_plus}+{src_minus}"
            precision = 0.1
            split = _split(be_plus, be_minus)
            if split < 0:
                state = "to_verify"
                reason = "negative as printed: the j- member is printed less bound than the j+ member"
            elif split == 0:
                state, unresolved = "provisional", True
                reason = (
                    "the two members are printed equal at 0.1 eV: unresolved at "
                    "that resolution, not a splitting of zero"
                )
            else:
                named = {k: f"{_OTHER_NAME[k]} {v:g}" for k, v in compared.items()}
                against = [
                    named[k] for k, v in compared.items()
                    if round(abs(v - split), 2) > tolerance[k]
                ]
                if against:
                    state = "to_verify"
                    reason = "another value disagrees beyond tolerance: " + ", ".join(against)
                elif not compared:
                    state = "to_verify"
                    reason = (
                        "no other value to compare it with: the other bundled "
                        "tables store one value for both members, or none"
                    )
                else:
                    state = "provisional"
                    reason = "agrees within tolerance with: " + ", ".join(named.values())
            if "+" in source:
                reason += "; the doublet straddles 1.5 keV, one member from each table"
        else:
            state = "not_covered"
            tabulated = [lab for lab, s in ((jp, src_plus), (jm, src_minus)) if s]
            reason = (
                f"only {tabulated[0]} is in the 2018/2019 tables"
                if tabulated
                else "neither member is in the 2018/2019 tables"
            )
            if None not in be_pair and be_pair[0] != be_pair[1]:
                be_plus, be_minus = be_pair
                source, precision = "binding_energy", None
                split = _split(be_plus, be_minus)
                reason += "; the binding_energy.json pair is carried, not adopted"
            else:
                be_plus = be_minus = split = source = precision = None
                if None not in be_pair:
                    reason += (
                        "; binding_energy.json stores one value for both "
                        "members, which gives no splitting"
                    )

        levels.setdefault(el, {})[nl] = {
            "Z": z_of[el],
            "j_plus": jp,
            "j_minus": jm,
            "split_eV": split,
            "precision_eV": precision,
            "source_table": source,
            "be_j_plus_eV": be_plus,
            "be_j_minus_eV": be_minus,
            "state": state,
            "state_reason": reason,
            "statistical_ratio": _L[letter] / (_L[letter] + 1),
            "unresolved_at_precision": unresolved,
        }
    return {"_meta": META, "levels": levels}


def dumps(table: dict[str, Any]) -> str:
    """The exact text of the bundled file."""
    return json.dumps(table, indent=2, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m toyomacro.data._spin_orbit_build OUT.json")
    with open(sys.argv[1], "w", encoding="utf-8") as _f:
        _f.write(dumps(build()))
