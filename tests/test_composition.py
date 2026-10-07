"""``toyomacro.composition``: the estimate, its conditions and its refusals.

Synthetic spectra with known areas fix the arithmetic: the result must be
(area / (sigma x lambda)) normalised over the declared elements, with the
exposure and transmission handled as stated or assumed. The rest pins
the contract: nothing missing is filled in, every assumption and every
extrapolated input is listed beside a ``conditional`` result, and
``inputs_stated`` is only reached when none is.
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
LINES = [("Si", "2p", 99.0, 3000.0), ("O", "1s", 532.0, 9000.0)]
TRUE_AREA = {el: area for el, _, _, area in LINES}


def _peak(be, area, hv=HV, base=50.0, slope=0.0, sigma=0.6, n=481):
    """A Gaussian of the given area on a (sloped) base, on a kinetic-energy axis."""
    ke0 = hv - be
    e = np.linspace(ke0 - 12, ke0 + 12, n)
    y = (base + slope * (e - ke0)
         + area / (sigma * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((e - ke0) / sigma) ** 2))
    return e, y, (ke0 - 10, ke0 + 10)


def _line(el, orb, be, area, hv=HV, slope=0.0, **kw):
    e, y, win = _peak(be, area, hv=hv, slope=slope)
    kw.setdefault("exposure_s", 1.0)
    kw.setdefault("exposure_basis", "synthetic: 1 s per point")
    return Line(element=el, orbital=orb, energy=e, intensity=y, window=win,
                binding_energy=be, **kw)


def _lines(**kw):
    return [_line(el, orb, be, area, **kw) for el, orb, be, area in LINES]


def _conditions(**kw):
    base = dict(photon_energy=HV, table="scofield", background="linear", matrix=SIO2,
                intensity_semantics="raw_counts", transmission_applied="applied",
                declared_elements=("Si", "O"))
    base.update(kw)
    return Conditions(**base)


def _expected():
    w = {el: TRUE_AREA[el] / (CrossSection.lookup(el, orb, HV, table="scofield")
                               * IMFP.tpp2m(kinetic_energy=HV - be, compound="SiO2"))
         for el, orb, be, _ in LINES}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


# --- the arithmetic -----------------------------------------------------------

def test_fractions_are_area_over_sigma_lambda_normalised():
    r = composition(_lines(), _conditions())
    assert r.status == "inputs_stated"
    for el, f in _expected().items():
        assert r.fractions[el] == pytest.approx(f, rel=2e-3)
    assert sum(r.fractions.values()) == pytest.approx(1.0, abs=1e-12)
    si = next(x for x in r.lines if x.element == "Si")
    assert si.area == pytest.approx(3000.0, rel=2e-3)
    assert "homogeneous-equivalent" in r.quantity
    assert r.statistical_uncertainty is None
    assert any("statistical uncertainty" in s for s in r.not_evaluated)


def test_axis_direction_and_scale_do_not_change_the_answer():
    ke = composition(_lines(), _conditions())
    reversed_ke = [replace(ln, energy=ln.energy[::-1], intensity=ln.intensity[::-1])
                   for ln in _lines()]
    on_be = [replace(ln, energy=HV - ln.energy, energy_scale="binding",
                     window=tuple(sorted(HV - w for w in ln.window))) for ln in _lines()]
    for other in (reversed_ke, on_be):
        r = composition(other, _conditions())
        assert r.status == "inputs_stated"
        for el in ("Si", "O"):
            assert r.fractions[el] == pytest.approx(ke.fractions[el], rel=1e-9)


def test_exposure_divides_integrated_counts_only():
    lines = _lines()
    doubled = [lines[0], replace(lines[1], exposure_s=2.0)]
    r1, r2 = composition(lines, _conditions()), composition(doubled, _conditions())
    ratio1 = r1.fractions["O"] / r1.fractions["Si"]
    assert r2.fractions["O"] / r2.fractions["Si"] == pytest.approx(ratio1 / 2, rel=1e-6)
    r3 = composition(doubled, _conditions(intensity_semantics="count_rate"))
    assert r3.fractions["O"] / r3.fractions["Si"] == pytest.approx(ratio1, rel=1e-6)
    assert all(x.exposure_s is None for x in r3.lines)
    assert any("exposure_s ignored" in n for n in r3.notes)


def test_a_curve_is_divided_out_only_when_not_applied():
    halved = [replace(ln, transmission=np.full_like(ln.energy, t))
              for ln, t in zip(_lines(), (1.0, 0.5))]
    applied = composition(halved, _conditions(transmission_applied="applied"))
    divided = composition(halved, _conditions(transmission_applied="not_applied"))
    a = applied.fractions["O"] / applied.fractions["Si"]
    assert divided.fractions["O"] / divided.fractions["Si"] == pytest.approx(2 * a, rel=1e-6)
    # Nothing in a file says two curves share a scale: dividing is conditional.
    assert divided.status == "conditional"
    assert any("share one normalisation" in x for x in divided.assumptions)


# --- refusals: nothing missing is filled in ------------------------------------

@pytest.mark.parametrize(("kw", "why"), [
    (dict(intensity_semantics="unknown"), "assume_intensity"),
    (dict(transmission_applied="unknown"), "assume_transmission"),
    (dict(declared_elements=("Si", "O", "N")), "without a line"),
    (dict(declared_elements=None), "declared_elements is required"),
    (dict(declared_elements=("Si",)), "undeclared elements"),
    (dict(normalise_over=("Si",)), "exclusion_reason"),
    (dict(normalise_over=(), exclusion_reason="x"), "no denominator"),
    (dict(transmission_applied="not_applied"), "no transmission curve"),
    (dict(background="tougaard"), "background"),
    (dict(matrix=Matrix(name="X", source="", compound="SiO2")), "name and a source"),
    (dict(matrix=Matrix(name="X", source="s", Nv=16, density=2.2, Mw=60.08)), "all four"),
])
def test_missing_facts_refuse_with_a_reason(kw, why):
    r = composition(_lines(), _conditions(**kw))
    assert r.status == "refused" and r.fractions is None
    assert any(why in reason for reason in r.reasons), r.reasons


@pytest.mark.parametrize(("line_kw", "why"), [
    (dict(exposure_s=None), "exposure unknown"),
    (dict(exposure_s=-1.0), "must be positive"),
    (dict(exposure_basis=""), "exposure_basis is required"),
])
def test_exposure_must_be_stated_with_its_basis(line_kw, why):
    r = composition(_lines(**line_kw), _conditions())
    assert r.status == "refused"
    assert any(why in reason for reason in r.reasons), r.reasons


def test_a_negative_net_area_refuses():
    """A dip, not a peak: the signed area is negative and must not be
    turned positive."""
    lines = _lines()
    dip = replace(lines[0], intensity=2 * 50.0 - lines[0].intensity)
    r = composition([dip, lines[1]], _conditions())
    assert r.status == "refused"
    assert any("net area is not positive" in reason for reason in r.reasons)


@pytest.mark.parametrize(("change", "why"), [
    (lambda ln: replace(ln, window=(ln.window[0] - 50, ln.window[1])), "beyond the data"),
    (lambda ln: replace(ln, window=(ln.window[0], ln.window[0] + 5)), "outside its window"),
    (lambda ln: replace(ln, transmission=np.ones(3)), "differ in shape"),
    (lambda ln: replace(ln, intensity=np.where(np.arange(ln.energy.size) == 5, np.nan,
                                               ln.intensity)), "non-finite"),
])
def test_ill_formed_lines_refuse(change, why):
    lines = _lines()
    r = composition([change(lines[0]), lines[1]], _conditions())
    assert r.status == "refused"
    assert any(why in reason for reason in r.reasons), r.reasons


def test_shirley_is_refused_on_a_non_uniform_grid():
    lines = _lines()
    e = lines[0].energy
    warped = e[0] + (e - e[0]) ** 1.1 / (e[-1] - e[0]) ** 0.1
    r = composition([replace(lines[0], energy=warped), lines[1]],
                    _conditions(background="shirley"))
    assert r.status == "refused"
    assert any("not uniform" in reason for reason in r.reasons)
    assert composition([replace(lines[0], energy=warped), lines[1]],
                       _conditions(background="linear")).status == "inputs_stated"


def test_a_missing_cross_section_refuses_rather_than_dropping_the_element():
    r = composition(_lines(), _conditions(photon_energy=50.0, table="yeh_lindau"))
    assert r.status == "refused" and r.fractions is None


def test_more_than_one_line_per_element_is_refused():
    lines = _lines()
    assert composition([*lines, lines[0]], _conditions()).status == "refused"


# --- assumptions and extrapolation make the result conditional -------------------

def test_assumptions_are_listed_beside_a_conditional_result():
    r = composition(_lines(), _conditions(
        intensity_semantics="unknown", assume_intensity="integrated_counts",
        transmission_applied="unknown", assume_transmission="equal_across_lines"))
    assert r.status == "conditional"
    assert any("read as integrated_counts" in a for a in r.assumptions)
    assert any("equal for every line" in a for a in r.assumptions)


def test_a_subset_denominator_is_explicit_and_recorded():
    lines = [*_lines(), _line("N", "1s", 400.0, 500.0)]
    r = composition(lines, _conditions(declared_elements=("Si", "O", "N"),
                                       normalise_over=("Si", "O"),
                                       exclusion_reason="N is adventitious, by the user"))
    assert r.status == "conditional"
    assert (r.declared_elements, r.denominator) == (("Si", "O", "N"), ("Si", "O"))
    assert r.excluded == {"N": "N is adventitious, by the user"}
    assert set(r.fractions) == {"Si", "O"}


def test_a_refused_line_inside_the_subset_still_refuses():
    lines = [*_lines(), _line("N", "1s", 400.0, 500.0)]
    lines[0] = replace(lines[0], exposure_s=None)  # Si, inside the subset
    r = composition(lines, _conditions(declared_elements=("Si", "O", "N"),
                                       normalise_over=("Si", "O"), exclusion_reason="r"))
    assert r.status == "refused"


def test_normalising_over_everything_needs_no_reason():
    r = composition(_lines(), _conditions(normalise_over=("Si", "O")))
    assert r.status == "inputs_stated" and r.excluded == {}


def test_an_extrapolated_imfp_alone_makes_the_result_conditional():
    hv = 3000.0  # Scofield is tabulated here; KE above 2000 eV is not TPP-2M's range
    lines = [_line(el, orb, be, area, hv=hv) for el, orb, be, area in LINES]
    r = composition(lines, _conditions(photon_energy=hv))
    assert r.status == "conditional" and not r.assumptions
    assert {x.cross_section.status for x in r.lines} == {"tabulated"}
    assert any("IMFP" in x and "Si 2p" in x for x in r.extrapolated_inputs)


def test_an_extrapolated_cross_section_alone_makes_the_result_conditional():
    """Trzhaskovskaya Pd 2p1/2 ends below 5250 eV; its KE (1920 eV) is in range."""
    pd = _line("Pd", "2p1/2", 3330.0, 1000.0, hv=5250.0)
    r = composition([pd], _conditions(photon_energy=5250.0, table="trzhaskovskaya",
                                      declared_elements=("Pd",)))
    assert r.status == "conditional"
    assert r.extrapolated_inputs == ("Pd 2p1/2: cross-section extrapolated_above",)
    assert r.fractions == {"Pd": 1.0}


# --- condition dependence ---------------------------------------------------------

def test_condition_dependence_is_a_difference_not_an_uncertainty():
    """On a sloped base Shirley and linear differ, so the 2 x 2 is not vacuous."""
    lines = [_line(el, orb, be, area, slope=-3.0) for el, orb, be, area in LINES]
    d = condition_dependence(lines, _conditions(), tables=("scofield", "yeh_lindau"),
                             backgrounds=("linear", "shirley"))
    cells = {(c.table, c.background): c.fractions for c in d.changes}
    assert cells[("scofield", "shirley")]["Si"] != pytest.approx(
        cells[("scofield", "linear")]["Si"], rel=1e-6)
    base = next(c for c in d.changes if (c.table, c.background) == ("scofield", "linear"))
    assert all(v == pytest.approx(0.0, abs=1e-15) for v in base.change.values())
    shirley = next(c for c in d.changes if (c.table, c.background) == ("scofield", "shirley"))
    assert shirley.change["Si"] == pytest.approx(
        cells[("scofield", "shirley")]["Si"] - cells[("scofield", "linear")]["Si"], abs=1e-15)
    for el in ("Si", "O"):
        expect = (cells[("yeh_lindau", "shirley")][el] - cells[("yeh_lindau", "linear")][el]
                  - cells[("scofield", "shirley")][el] + cells[("scofield", "linear")][el])
        assert d.interaction[el] == pytest.approx(expect, abs=1e-15)
        assert d.interaction[el] != 0.0


def test_condition_dependence_rejects_repeated_conditions():
    with pytest.raises(ValueError, match="repeat"):
        condition_dependence(_lines(), _conditions(), tables=("scofield", "scofield"),
                             backgrounds=("linear", "shirley"))
