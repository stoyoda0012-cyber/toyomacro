"""``composition_uncertainty``: admitted by a declaration, never by the data (design record §10).

The validation itself is a recorded run (§8–9); here the public function's
contract is pinned on small cases, the shipped scope is tied to the third
record, and the four tests of §10 "The implementation check in the public
path" stand in for §8 rule 6 on the public path.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import toyomacro.composition as comp
from toyomacro import _composition_validation3 as v3

DESIGN = Path(__file__).parents[1] / "docs" / "design"
RECORD3 = DESIGN / "composition-uncertainty-results-3.json"
RECORD2 = DESIGN / "composition-uncertainty-results-2.json"

#: P/linear/0.02 — an admitted (condition, κ) pair: its κ = (1, 1) r_pred is
#: at or below the linear r* (§9), so the public path reaches the data side.
CID = v3.condition_id("P", "linear", 0.02)
ADMITTED = ("channel_independence", "no_prior_correction", "transfer")


def _si_area(cid=CID) -> float:
    rec = json.loads(RECORD3.read_text(encoding="utf-8"))
    return next(c for c in rec["conditions"] if c["condition"] == cid)["count_setting"]["si_area"]


def _public_conditions(cond):
    """The admitted transmission state (§10 check 3) in place of the harness's label."""
    return replace(cond, transmission_applied="unknown", assume_transmission="equal_across_lines")


def _declaration(mu, cond, assumptions=ADMITTED, **kw):
    lines = tuple(comp.DeclaredLine(element=ln.element, orbital=ln.orbital, energy=ln.energy,
                                    expected=ln.intensity, window=ln.window,
                                    binding_energy=ln.binding_energy, exposure_s=ln.exposure_s,
                                    exposure_basis=ln.exposure_basis) for ln in mu)
    fields = dict(lines=lines, conditions=cond, provenance="planning_model",
                  provenance_basis="synthetic: the registered condition's expected counts",
                  fixed_before_data=True, assumptions=assumptions)
    fields.update(kw)
    return comp.DeclaredSpectrum(**fields)


def _data(mu, trial, cid=CID):
    rng = v3.generator([v3.SEED, 3, 1, cid, trial])
    return [replace(ln, intensity=rng.poisson(ln.intensity).astype(np.float64)) for ln in mu]


@pytest.fixture(scope="module")
def case():
    mu, cond = v3.expected(CID, _si_area())
    pub = _public_conditions(cond)
    return {"mu": mu, "cond": cond, "pub": pub, "declared": _declaration(mu, pub),
            "data": [_data(mu, i) for i in range(3)]}


# --- the declaration ----------------------------------------------------------

def test_declared_arrays_are_copied_and_validated(case):
    ln = case["mu"][0]
    expected = ln.intensity.copy()
    d = comp.DeclaredLine(element="Si", orbital="2p", energy=ln.energy, expected=expected,
                          window=ln.window, binding_energy=ln.binding_energy, exposure_s=1.0,
                          exposure_basis="x")
    expected[0] = -5.0
    assert d.expected[0] >= 0  # the caller's later change does not reach the declaration
    for bad in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            comp.DeclaredLine(element="Si", orbital="2p", energy=ln.energy,
                              expected=np.append(ln.intensity[:-1], bad), window=ln.window,
                              binding_energy=ln.binding_energy)
    with pytest.raises(ValueError, match="real"):
        comp.DeclaredLine(element="Si", orbital="2p", energy=ln.energy,
                          expected=ln.intensity.astype(complex), window=ln.window,
                          binding_energy=ln.binding_energy)
    with pytest.raises(ValueError, match="shape"):
        comp.DeclaredLine(element="Si", orbital="2p", energy=ln.energy,
                          expected=ln.intensity[:-1], window=ln.window,
                          binding_energy=ln.binding_energy)


def test_this_data_unknown_names_and_an_empty_basis_are_refused(case):
    mu, pub = case["mu"], case["pub"]
    with pytest.raises(ValueError, match="this_data"):
        _declaration(mu, pub, provenance="this_data")
    with pytest.raises(ValueError, match="provenance"):
        _declaration(mu, pub, provenance="guess")
    with pytest.raises(ValueError, match="provenance_basis"):
        _declaration(mu, pub, provenance_basis="  ")
    with pytest.raises(ValueError, match="unknown assumption"):
        _declaration(mu, pub, assumptions=("channel_independence", "lucky"))
    assert comp.REQUIRED_ASSUMPTIONS == ADMITTED and comp.ASSUMPTIONS[3] == "more_counts_extrapolation"


def test_a_declaration_that_does_not_describe_the_data_is_a_value_error(case):
    mu, pub, decl, data = case["mu"], case["pub"], case["declared"], case["data"][0]
    with pytest.raises(ValueError, match="energy grid"):
        shifted = [replace(data[0], energy=data[0].energy + 0.01)] + data[1:]
        comp.composition_uncertainty(shifted, pub, declared=decl)
    with pytest.raises(ValueError, match="line order"):
        comp.composition_uncertainty(list(reversed(data)), pub, declared=decl)
    with pytest.raises(ValueError, match="exposure"):
        comp.composition_uncertainty([replace(data[0], exposure_s=2.0)] + data[1:], pub,
                                     declared=decl)
    with pytest.raises(ValueError, match="conditions passed differ"):
        comp.composition_uncertainty(data, replace(pub, background="shirley"), declared=decl)
    with pytest.raises(ValueError, match="assume_intensity"):
        c = replace(pub, assume_intensity="integrated_counts")
        comp.composition_uncertainty(data, c, declared=replace(decl, conditions=c))


# --- statuses, in the order of §10 ----------------------------------------------

def test_not_evaluated_when_the_composition_is_refused_or_the_counts_are_not_poisson(case):
    mu, pub, data = case["mu"], case["pub"], case["data"][0]
    c = replace(pub, declared_elements=None)
    u = comp.composition_uncertainty(data, c, declared=_declaration(mu, c))
    assert u.status == "not_evaluated" and "refused" in u.reasons[0] and u.prediction is None
    c = replace(pub, intensity_semantics="count_rate")
    u = comp.composition_uncertainty(data, c, declared=_declaration(mu, c))
    assert u.status == "not_evaluated" and "Poisson" in u.reasons[0]
    half = [replace(data[0], intensity=data[0].intensity + 0.5)] + data[1:]
    u = comp.composition_uncertainty(half, pub, declared=case["declared"])
    assert u.status == "not_evaluated" and "not integer" in u.reasons[0]
    neg = [replace(data[0], intensity=np.append(data[0].intensity[:-1], -1.0))] + data[1:]
    u = comp.composition_uncertainty(neg, pub, declared=case["declared"])
    assert u.status == "not_evaluated" and "negative" in u.reasons[0]


def test_withheld_by_the_structural_checks_before_any_bootstrap(case):
    mu, pub, data, decl = case["mu"], case["pub"], case["data"][0], case["declared"]
    c = replace(pub, table="yeh_lindau")
    u = comp.composition_uncertainty(data, c, declared=replace(decl, conditions=c))
    assert u.status == "withheld" and "scofield" in u.reasons[0] and u.prediction is None
    with_curves = [replace(ln, transmission=np.ones_like(ln.energy)) for ln in data]
    for state, assume, word in (("applied", None, "already divided"),
                                ("not_applied", None, "divided out"),
                                ("unknown", "divide_by_curve", "divided out")):
        c = replace(pub, transmission_applied=state, assume_transmission=assume)
        u = comp.composition_uncertainty(with_curves, c, declared=replace(decl, conditions=c))
        assert u.status == "withheld" and word in u.reasons[0] and u.data_side is None, state
    c = replace(pub, assume_transmission=None)  # unknown state, no assumption: composition() refuses
    u = comp.composition_uncertainty(data, c, declared=replace(decl, conditions=c))
    assert u.status == "not_evaluated" and "refused" in u.reasons[0]
    u = comp.composition_uncertainty(data, pub, declared=_declaration(mu, pub, assumptions=()))
    assert u.status == "withheld" and u.reasons == tuple(
        f"applicability not declared: {a}" for a in ADMITTED) and u.prediction is None
    u = comp.composition_uncertainty(data, pub, declared=_declaration(mu, pub, fixed_before_data=False))
    assert u.status == "withheld" and u.reasons[0] == "applicability not declared: fixed_before_data"


def test_withheld_when_the_prediction_is_above_the_threshold_and_no_data_bootstrap_runs(case):
    mu, pub, data = case["mu"], case["pub"], case["data"][0]
    # a declaration four times too noisy: half the peak, double the base (kappa = (0.5, 2))
    parts, _ = v3.parts(CID, _si_area())
    noisy = [replace(t, intensity=0.5 * peak + 2.0 * base) for t, peak, base in parts]
    u = comp.composition_uncertainty(data, pub, declared=_declaration(noisy, pub))
    assert u.status == "withheld" and "above the validated 0.02" in u.reasons[0]
    assert u.prediction["r_pred"] > 0.02 and u.prediction["r_star"] == 0.02
    assert u.data_side is None and u.standard_uncertainty is None


def test_evaluated_is_assumed_and_carries_the_record(case):
    pub, data, decl = case["pub"], case["data"][0], case["declared"]
    u = comp.composition_uncertainty(data, pub, declared=decl)
    assert u.status == "evaluated" and u.applicability == "assumed"
    assert u.assumptions == ADMITTED and u.elements == ("Si", "O")
    assert set(u.standard_uncertainty) == {"Si", "O"}
    assert u.standard_uncertainty["Si"] == pytest.approx(u.standard_uncertainty["O"])
    assert u.covariance[0][0] == pytest.approx(u.standard_uncertainty["Si"] ** 2)
    assert u.prediction["key"] == [20261014, 2] and u.prediction["n_boot"] == 1000
    assert u.data_side == {"n_boot": 1000, "n_refused": 0, "key": [20261014, 4]}
    assert u.declared["provenance"] == "planning_model" and u.declared["fixed_before_data"]
    assert len(u.declared["lines"][0]["expected"]) == 241
    assert u.registered["differs"] == [] and u.registered["matches"]["photon_energy_eV"]
    assert u.versions["toyomacro"] == comp.toyomacro.__version__
    assert u.scope == comp._VALIDATED_SCOPE
    again = comp.composition_uncertainty(data, pub, declared=decl)
    assert again.standard_uncertainty == u.standard_uncertainty  # fixed keys: deterministic


def test_the_registered_report_names_what_differs(case):
    pub, data, decl = case["pub"], case["data"][0], case["declared"]
    c = replace(pub, photon_energy=1253.6, matrix=replace(pub.matrix, compound="Si"))
    u = comp.composition_uncertainty(data, c, declared=replace(decl, conditions=c))
    assert u.registered["differs"] == ["matrix_compound", "photon_energy_eV"]
    assert "peak shapes and widths" in u.registered["assumed_only"]


# --- the shipped scope -------------------------------------------------------------

def test_the_shipped_scope_is_the_one_the_third_record_decided():
    decision = json.loads(RECORD3.read_text(encoding="utf-8"))["decision"]
    assert comp._VALIDATED_SCOPE["r_pred_threshold"] == {
        bg: d["r_star"] for bg, d in decision.items()}
    assert comp._VALIDATED_SCOPE["record"].endswith(RECORD3.name)
    assert (comp._VALIDATED_SCOPE["min_elements"], comp._VALIDATED_SCOPE["max_lines"]) == (2, 3)
    # the second record's verdicts still recompute from their intervals (its scope was empty)
    rec2 = json.loads(RECORD2.read_text(encoding="utf-8"))
    assert all(d["r_star"] is None for d in rec2["decision"].values())


# --- the implementation check in the public path (§10; the counterpart of §8 rule 6) ----

def test_no_observed_quantity_enters_the_admission(case):
    pub, decl = case["pub"], case["declared"]
    results = [comp.composition_uncertainty(d, pub, declared=decl) for d in case["data"]]
    preds = [u.prediction for u in results]
    assert all(p == preds[0] for p in preds)
    # admitted for every data set: the data-side bootstrap ran each time (no gate after the prediction)
    assert all(u.data_side is not None for u in results)
    alone = comp._predict_r(decl.as_lines(), pub, n_boot=1000)
    assert preds[0]["r_pred"] == alone.r_pred and preds[0]["n_refused"] == alone.n_refused == 0


def test_the_registered_entry_point_reproduces_the_validation_bit_for_bit(case):
    mu, cond = case["mu"], case["cond"]
    decl = _declaration(mu, cond)  # the harness's own label, accepted only here
    for i in range(3):
        data = _data(mu, i)
        tr = v3.trial(CID, "decision", i, _si_area(), n_boot=1000)
        u = comp._composition_uncertainty_registered(
            data, cond, decl, data_key=[v3.SEED, 4, 1, CID, i])
        assert u.applicability == "registered"
        assert (u.status == "evaluated") == (tr["category"] == 3)
        if tr["category"] == 3:
            assert [u.standard_uncertainty[el] for el in u.elements] == tr["se"]
            assert u.data_side["n_refused"] == 0


def test_the_published_set_is_the_set_with_no_refused_replicate(case):
    pub, decl = case["pub"], case["declared"]
    for data in case["data"]:
        u = comp.composition_uncertainty(data, pub, declared=decl)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261014, 4])))
        rows, refused, _ = comp._bootstrap_detail(data, pub, 1000, rng, u.elements)
        assert (u.status == "evaluated") == (refused == 0)
        if refused == 0:
            assert [u.standard_uncertainty[el] for el in u.elements] == \
                [float(v) for v in np.std(rows, axis=0, ddof=1)]


def test_registered_only_through_the_private_entry_point(case):
    pub, data, decl = case["pub"], case["data"][0], case["declared"]
    assert comp.composition_uncertainty(data, pub, declared=decl).applicability == "assumed"
    assert comp._composition_uncertainty_registered(data, pub, decl).applicability == "registered"
    with pytest.raises(TypeError):
        comp.composition_uncertainty(data, pub, declared=decl, data_key=[1])  # no such parameter


def test_withheld_when_a_data_side_replicate_is_refused(case):
    """An observed line with almost no net area refuses replicates; the declaration still admits."""
    pub, decl, data = case["pub"], case["declared"], case["data"][0]
    parts, _ = v3.parts(CID, _si_area())
    _, peak, base = parts[0]  # the Si line at 2 % of its expected peak: a net area that is
    rng = np.random.default_rng(5)  # positive on the data and not on many replicates
    weak = replace(data[0], intensity=rng.poisson(base + 0.02 * peak).astype(np.float64))
    u = comp.composition_uncertainty([weak, data[1]], pub, declared=decl)
    assert u.status == "withheld" and "replicates were refused" in u.reasons[0]
    assert u.prediction is not None and u.data_side["n_refused"] > 0
    assert u.standard_uncertainty is None


def test_the_returned_scope_is_a_copy_and_the_statement_a_bool(case):
    pub, data, decl = case["pub"], case["data"][0], case["declared"]
    u = comp.composition_uncertainty(data, pub, declared=decl)
    u.scope["r_pred_threshold"]["linear"] = 1.0
    u.registered["scope"]["r_pred_threshold"]["linear"] = 1.0
    assert comp._VALIDATED_SCOPE["r_pred_threshold"]["linear"] == 0.02
    with pytest.raises(ValueError, match="fixed_before_data"):
        _declaration(case["mu"], pub, fixed_before_data="yes")


def test_the_registered_values_are_the_harness_conditions():
    reg = comp._REGISTERED_VALUES
    hvs = {v3.CONFIGS[c][1] for c in v3.CONFIG_NAMES}
    steps = {v3.CONFIGS[c][2] for c in v3.CONFIG_NAMES}
    lines = {tuple((el, orb) for el, orb, *_ in v3.CONFIGS[c][0]) for c in v3.CONFIG_NAMES}
    assert set(reg["photon_energy_eV"]) == hvs and set(reg["channel_step_eV"]) == steps
    assert set(reg["lines"]) == lines and reg["background"] == v3.BACKGROUNDS
    assert reg["exposure_s"] == (v3.REGISTERED["exposure_s"],)
    assert reg["window_half_width_eV"] == (v3.REGISTERED["window_half_eV"],)
    for cid in (0, 10, 20, 30):  # kappa = (1, 1) of each configuration, linear
        mu, cond = v3.expected(cid, 1e4)
        assert comp._registered_report(mu, cond)["differs"] == []


def test_the_second_records_verdicts_still_follow_from_their_intervals():
    """Record 2 (empty scope) stays recomputable from its intervals."""
    rec2 = json.loads(RECORD2.read_text(encoding="utf-8"))
    for s in rec2["scenarios"]:
        for block in (s["R_all"], s["R_admitted"] or {}):
            for el in block.values():
                assert el["verdict"] == v3._verdict(*el["R_mc_95"])


def test_the_first_harness_runs_and_resumes(tmp_path):
    from toyomacro import _composition_validation as val
    out = tmp_path / "S1"
    val.run("S1", n_outer=4, n_boot=5, out=out, chunk=2, workers=1)
    first = sorted(p.read_text() for p in out.glob("chunk_*.json"))
    val.run("S1", n_outer=4, n_boot=5, out=out, chunk=2, workers=1)  # nothing to redo
    assert sorted(p.read_text() for p in out.glob("chunk_*.json")) == first
    assert val.summarize(out)["n_outer_attempted"] == 4


def test_composition_alone_says_the_uncertainty_is_not_computed_here(case):
    r = comp.composition(case["data"][0], case["pub"])
    assert any("not computed in this call" in n and "composition_uncertainty" in n
               for n in r.not_evaluated)
    assert not hasattr(comp, "_composition_uncertainty")
