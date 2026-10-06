"""``toyomacro.composition``: the estimate, its conditions and its refusals.

Synthetic spectra with known areas fix the arithmetic: the result must be
(area / (sigma x lambda)) normalised over the denominator, with the
exposure and transmission handled as stated or assumed. The rest pins
the contract: nothing missing is filled in, and every assumption is
listed beside a ``conditional`` result.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from toyomacro.composition import (
    Conditions,
    Line,
    Matrix,
    composition,
    condition_dependence,
)
from toyomacro.data.cross_section import CrossSection
from toyomacro.data.imfp import IMFP

HV = 1486.6
SIO2 = Matrix(name="SiO2", source="CompoundDB", compound="SiO2")


def _peak(be: float, area: float, base: float = 50.0, sigma: float = 0.6):
    """A Gaussian of the given area on a flat base, on a kinetic-energy axis."""
    ke0 = HV - be
    e = np.linspace(ke0 - 12, ke0 + 12, 481)
    y = base + area / (sigma * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((e - ke0) / sigma) ** 2)
    return e, y, (ke0 - 10, ke0 + 10)


def _line(el, orb, be, area, **kw):
    e, y, win = _peak(be, area)
    kw.setdefault("exposure_s", 1.0)
    return Line(element=el, orbital=orb, energy=e, intensity=y, window=win,
                binding_energy=be, **kw)


def _conditions(**kw):
    base = dict(photon_energy=HV, table="scofield", background="linear", matrix=SIO2,
                intensity_semantics="raw_counts", transmission_applied="applied")
    base.update(kw)
    return Conditions(**base)


def _expected(lines, table="scofield"):
    w = {}
    for ln in lines:
        sigma = CrossSection.lookup(ln.element, ln.orbital, HV, table=table)
        lam = IMFP.tpp2m(kinetic_energy=HV - ln.binding_energy, compound="SiO2")
        w[ln.element] = TRUE_AREA[ln.element] / (sigma * lam)
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


LINES = [("Si", "2p", 99.0, 3000.0), ("O", "1s", 532.0, 9000.0)]
TRUE_AREA = {el: area for el, _, _, area in LINES}


def _lines(**kw):
    return [_line(el, orb, be, area, **kw) for el, orb, be, area in LINES]


# --- the arithmetic -----------------------------------------------------------

def test_fractions_are_area_over_sigma_lambda_normalised():
    lines = _lines()
    r = composition(lines, _conditions())
    assert r.status == "supported"
    expected = _expected(lines)
    for el, f in expected.items():
        assert r.fractions[el] == pytest.approx(f, rel=2e-3)
    assert sum(r.fractions.values()) == pytest.approx(1.0, abs=1e-12)
    si = next(x for x in r.lines if x.element == "Si")
    assert si.area == pytest.approx(3000.0, rel=2e-3)
    assert si.cross_section.status == "tabulated"
    assert "homogeneous-equivalent" in r.quantity
    assert r.statistical_uncertainty is None
    assert any("statistical uncertainty" in s for s in r.not_evaluated)


def test_exposure_divides_integrated_counts_only():
    lines = _lines()
    doubled = [lines[0], replace(lines[1], exposure_s=2.0)]
    r1 = composition(lines, _conditions())
    r2 = composition(doubled, _conditions())
    ratio1 = r1.fractions["O"] / r1.fractions["Si"]
    ratio2 = r2.fractions["O"] / r2.fractions["Si"]
    assert ratio2 == pytest.approx(ratio1 / 2, rel=1e-6)
    # A count rate is not divided again.
    r3 = composition(doubled, _conditions(intensity_semantics="count_rate"))
    assert r3.fractions["O"] / r3.fractions["Si"] == pytest.approx(ratio1, rel=1e-6)
    assert all(x.exposure_s is None for x in r3.lines)


def test_a_curve_is_divided_out_only_when_not_applied():
    lines = _lines()
    halved = [replace(ln, transmission=np.full_like(ln.energy, t))
              for ln, t in zip(lines, (1.0, 0.5))]
    applied = composition(halved, _conditions(transmission_applied="applied"))
    divided = composition(halved, _conditions(transmission_applied="not_applied"))
    a = applied.fractions["O"] / applied.fractions["Si"]
    d = divided.fractions["O"] / divided.fractions["Si"]
    assert d == pytest.approx(2 * a, rel=1e-6)
    assert divided.status == "supported"


# --- refusals: nothing missing is filled in ------------------------------------

@pytest.mark.parametrize(("kw", "why"), [
    (dict(intensity_semantics="unknown"), "assume_intensity"),
    (dict(transmission_applied="unknown"), "assume_transmission"),
    (dict(declared_elements=("Si", "O", "N")), "without a line"),
    (dict(normalise_over=("Si",)), "exclusion_reason"),
    (dict(transmission_applied="not_applied"), "no transmission curve"),
    (dict(background="tougaard"), "background"),
])
def test_missing_facts_refuse_with_a_reason(kw, why):
    r = composition(_lines(), _conditions(**kw))
    assert r.status == "refused" and r.fractions is None
    assert any(why in reason for reason in r.reasons), r.reasons


def test_unknown_exposure_refuses_integrated_counts():
    r = composition(_lines(exposure_s=None), _conditions())
    assert r.status == "refused"
    assert any("exposure unknown" in reason for reason in r.reasons)


def test_a_missing_cross_section_refuses_rather_than_dropping_the_element():
    """Si 2p at 50 eV is below its threshold: no sigma, so no composition —
    never a 100% of the element that is left."""
    r = composition(_lines(), _conditions(photon_energy=50.0, table="yeh_lindau"))
    assert r.status == "refused" and r.fractions is None
    assert any("no cross-section" in reason for reason in r.reasons)


def test_more_than_one_line_per_element_is_refused():
    lines = _lines()
    r = composition([*lines, lines[0]], _conditions())
    assert r.status == "refused"


# --- assumptions make the result conditional -------------------------------------

def test_assumptions_are_listed_beside_a_conditional_result():
    r = composition(_lines(), _conditions(
        intensity_semantics="unknown", assume_intensity="integrated_counts",
        transmission_applied="unknown", assume_transmission="equal_across_lines"))
    assert r.status == "conditional"
    assert any("read as integrated_counts" in a for a in r.assumptions)
    assert any("equal for every line" in a for a in r.assumptions)
    assert any("transmission" in s for s in r.not_evaluated)


def test_a_subset_denominator_is_explicit_and_recorded():
    lines = [*_lines(), _line("N", "1s", 400.0, 500.0)]
    r = composition(lines, _conditions(normalise_over=("Si", "O"),
                                       exclusion_reason="N is adventitious, by the user"))
    assert r.status == "conditional"
    assert r.declared_elements == ("Si", "O", "N")
    assert r.denominator == ("Si", "O")
    assert r.excluded == {"N": "N is adventitious, by the user"}
    assert set(r.fractions) == {"Si", "O"}


def test_an_extrapolated_input_makes_the_result_conditional():
    """Yeh-Lindau stops at 8047.8 eV; at Ga K-alpha sigma is extrapolated."""
    lines = _lines()
    shift = 9251.7 - HV
    ga = [replace(ln, energy=ln.energy + shift,
                  window=(ln.window[0] + shift, ln.window[1] + shift))
          for ln in lines]
    r = composition(ga, _conditions(photon_energy=9251.7, table="yeh_lindau"))
    assert r.status == "conditional"
    assert {x.cross_section.status for x in r.lines} == {"extrapolated_above"}
    assert any("outside its tabulated or fitted range" in a for a in r.assumptions)


# --- condition dependence ---------------------------------------------------------

def test_condition_dependence_is_a_difference_not_an_uncertainty():
    lines = _lines()
    d = condition_dependence(lines, _conditions(), tables=("scofield", "yeh_lindau"),
                             backgrounds=("linear", "shirley"))
    assert len(d.changes) == 4
    base = next(c for c in d.changes if (c.table, c.background) == ("scofield", "linear"))
    assert all(v == pytest.approx(0.0, abs=1e-15) for v in base.change.values())
    cells = {(c.table, c.background): c.fractions for c in d.changes}
    for el in ("Si", "O"):
        expect = (cells[("yeh_lindau", "shirley")][el] - cells[("yeh_lindau", "linear")][el]
                  - cells[("scofield", "shirley")][el] + cells[("scofield", "linear")][el])
        assert d.interaction[el] == pytest.approx(expect, abs=1e-15)
    table_shift = cells[("yeh_lindau", "linear")]["Si"] - cells[("scofield", "linear")]["Si"]
    assert table_shift != 0.0
