"""How ``CrossSection.lookup`` interpolates, and what it no longer invents.

Since v0.4.0:

- Inside a line's tabulated range the value is a monotone piecewise cubic
  (PCHIP) through the tabulated cells in log-log space, so a tabulated
  energy returns the tabulated value. Up to v0.3.1 it was one polynomial
  fitted through all the cells of a line, which on Yeh-Lindau missed 39%
  of the cells it was fitted to by more than 10%. That method is kept as
  ``set_interpolation("polyfit")`` and reproduces v0.3.1 to round-off
  (bit for bit on the same platform; the least-squares fit differs in the
  last bits between platforms, about 3e-14 relative).
- A subshell the table does not carry returns None. Up to v0.3.1 it was
  fitted across Z from other elements; against Scofield that put 1s 4.3x
  and 2p 3.9x too high at the median, and it returned numbers (up to
  1e227 Mb, and inf) for subshells with no electrons.

Implementation fidelity (the tabulated cells, the legacy values) and
plausibility (leave-one-out accuracy, finiteness) are tested separately.
"""

import math

import numpy as np
import pytest

from toyomacro.data.cross_section import AVAILABLE_TABLES, CrossSection


@pytest.fixture(autouse=True)
def _restore_interpolation():
    method, order = CrossSection.get_interpolation(), CrossSection._poly_order
    yield
    CrossSection.set_interpolation(method)
    CrossSection.set_poly_order(order)


def _lines(table):
    d = CrossSection.to_dict(table)
    return d, d["photon_energies"]


# --- implementation fidelity -------------------------------------------------

@pytest.mark.parametrize("table", AVAILABLE_TABLES)
def test_pchip_returns_every_tabulated_cell(table):
    """At a tabulated energy the stored line gives back the stored value."""
    d, energies = _lines(table)
    checked = 0
    for element, rec in d["data"].items():
        for orbital, line in rec.items():
            valid = [v for v in line["cross_sections"] if v is not None and v > 0]
            if len(valid) < 2:
                continue
            for e, v in zip(energies, line["cross_sections"]):
                if v is None or v <= 0:
                    continue
                got = CrossSection._interp_one(line, d, e)
                assert got == pytest.approx(v, rel=1e-12), (table, element, orbital, e)
                checked += 1
    assert checked > 7000


# Values lookup() returned in v0.3.1 on macOS/arm64 (whole-line cubic,
# order 3 unless stated). "polyfit" must reproduce them to round-off; the
# least-squares fit's last bits differ between platforms.
V031 = [
    ("yeh_lindau", "Si", "2p", 1486.6, 3, 0.01058780110464214),
    ("yeh_lindau", "Ar", "3p", 8047.8, 3, 1.62378556370801e-05),
    ("yeh_lindau", "Au", "4f", 1253.6, 3, 0.4057849155327707),
    ("yeh_lindau", "C", "1s", 5414.7, 3, 0.0002579794824368428),
    ("yeh_lindau", "O", "1s", 9251.7, 3, 0.00018100209215092514),
    ("scofield", "Si", "2p", 1486.6, 3, 0.0111514214810607),
    ("scofield", "Ir", "3d5/2", 2058.8, 3, 0.31156054742441186),
    ("trzhaskovskaya", "Cu", "2p3/2", 1486.6, 3, 52.596778468462944),
    ("yeh_lindau", "Si", "2p", 1486.6, 6, 0.011033255233477634),
]


@pytest.mark.parametrize(("table", "el", "orb", "hv", "order", "value"), V031)
def test_polyfit_reproduces_v031(table, el, orb, hv, order, value):
    CrossSection.set_interpolation("polyfit")
    CrossSection.set_poly_order(order)
    assert CrossSection.lookup(el, orb, hv, table=table) == pytest.approx(value, rel=1e-12)


def test_poly_order_only_affects_polyfit():
    CrossSection.set_interpolation("pchip")
    a = CrossSection.lookup("Si", "2p", 1486.6, table="yeh_lindau")
    CrossSection.set_poly_order(6)
    assert CrossSection.lookup("Si", "2p", 1486.6, table="yeh_lindau") == a


def test_unknown_interpolation_is_rejected():
    with pytest.raises(ValueError):
        CrossSection.set_interpolation("spline")


@pytest.mark.parametrize(("element", "orbital", "hv", "table", "why"), [
    ("H", "6s", 21.2, "yeh_lindau", "unoccupied; was ~8.8e226 Mb"),
    ("Si", "4s", 21.2, "yeh_lindau", "unoccupied; was 18 Mb"),
    ("Au", "3d", 21.2, "yeh_lindau", "not tabulated, and below its 2206 eV threshold"),
    ("Tm", "3d", 8047.8, "yeh_lindau", "real level Table I does not print"),
    ("Si", "1s", 5414.7, "yeh_lindau", "real level Table I does not print"),
    ("Md", "3d", 8047.8, "scofield", "element not in the bundled Scofield table"),
])
def test_untabulated_subshells_return_none(element, orbital, hv, table, why):
    assert CrossSection.lookup(element, orbital, hv, table=table) is None, why


def test_energy_extrapolation_above_the_grid_is_kept():
    """Beyond the tabulated energies a power law through the end cells
    still answers (Yeh-Lindau stops at 8047.8 eV; Ga K-alpha is 9251.7)."""
    assert CrossSection.lookup("Si", "2p", 9251.7, table="yeh_lindau") > 0


# --- plausibility ------------------------------------------------------------

def test_leave_one_out_beats_the_whole_line_polynomial_on_yeh_lindau():
    """Drop each interior cell, predict it from the rest.

    Measured over 5,557 cells: median 2.0% for PCHIP against 8.9% for the
    order-3 polynomial. The bound 3% leaves room for library versions; the
    comparison is the point.
    """
    d, energies = _lines("yeh_lindau")
    errs = {"pchip": [], "polyfit": []}
    for rec in d["data"].values():
        for line in rec.values():
            pts = [(e, v) for e, v in zip(energies, line["cross_sections"]) if v and v > 0]
            if len(pts) < 5:
                continue
            for i in range(1, len(pts) - 1):
                rest, (x, v) = pts[:i] + pts[i + 1:], pts[i]
                for method in errs:
                    CrossSection.set_interpolation(method)
                    errs[method].append(abs(math.log(CrossSection._interpolate_log_log(rest, x) / v)))
    med = {m: math.expm1(float(np.median(a))) for m, a in errs.items()}
    assert med["pchip"] < 0.03
    assert med["pchip"] < med["polyfit"] / 2


@pytest.mark.parametrize("table", AVAILABLE_TABLES)
def test_everything_returned_is_finite_and_positive(table):
    """Over every element, every label the table uses and its own grid."""
    d, energies = _lines(table)
    labels = sorted({o for rec in d["data"].values() for o in rec})
    for element in d["data"]:
        for orbital in labels:
            for e in energies[:: max(1, len(energies) // 16)]:
                v = CrossSection.lookup(element, orbital, e, table=table)
                assert v is None or (math.isfinite(v) and v > 0), (element, orbital, e, v)
