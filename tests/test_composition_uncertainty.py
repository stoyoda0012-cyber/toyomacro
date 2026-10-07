"""``composition_uncertainty``: reported only where its noise model applies
and inside the scope validated in docs/design/composition-uncertainty.md.

The validation itself (thousands of simulated data sets) is a recorded
run, not a test; here the record is tied to the module's scope constant,
and the function's contract is pinned on small cases.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import toyomacro.composition as comp
from toyomacro import _composition_validation as val

DESIGN = Path(__file__).parents[1] / "docs" / "design"
RECORD = DESIGN / "composition-uncertainty-results.json"
RECORD2 = DESIGN / "composition-uncertainty-results-2.json"


def _case(name="S1"):
    """Integer counts drawn once from the scenario's expected counts."""
    lines, cond = val.expected_lines(name), val.conditions(name)
    rng = np.random.default_rng(7)
    return [replace(ln, intensity=rng.poisson(ln.intensity).astype(float)) for ln in lines], cond


def _integer_case(name="S1"):
    return _case(name)


@pytest.fixture
def scoped(monkeypatch):
    monkeypatch.setattr(comp, "VALIDATED_SCOPE",
                        {"r_threshold": {"linear": 0.02}, "max_lines": 3, "record": "test"})


def test_not_evaluated_unless_the_intensity_is_stated_raw_counts():
    lines, cond = _case()
    for c in (replace(cond, intensity_semantics="count_rate"),
              replace(cond, intensity_semantics="unknown", assume_intensity="integrated_counts")):
        u = comp.composition_uncertainty(lines, c, n_boot=10, seed=1)
        assert u.status == "not_evaluated" and u.standard_uncertainty is None


def test_not_evaluated_when_the_composition_is_refused():
    lines, cond = _case()
    u = comp.composition_uncertainty(lines, replace(cond, declared_elements=None), n_boot=10)
    assert u.status == "not_evaluated"


def test_evaluated_inside_the_scope_with_a_consistent_covariance(scoped):
    lines, cond = _case()
    u = comp.composition_uncertainty(lines, cond, n_boot=200, seed=3)
    assert u.status == "evaluated" and u.reasons == ()
    cov = np.array(u.covariance)
    np.testing.assert_allclose(cov, cov.T)
    for i, el in enumerate(u.elements):
        assert u.standard_uncertainty[el] == pytest.approx(np.sqrt(cov[i, i]))
    # Fractions sum to one, so the covariance has zero row sums.
    np.testing.assert_allclose(cov.sum(axis=1), 0.0, atol=1e-15)
    again = comp.composition_uncertainty(lines, cond, n_boot=200, seed=3)
    assert again.standard_uncertainty == u.standard_uncertainty
    assert "Poisson" in u.noise_model and "before exposure" in u.resampling


def test_withheld_outside_the_validated_scope(scoped):
    lines, cond = _case()
    u = comp.composition_uncertainty(lines, replace(cond, background="shirley"), n_boot=20)
    assert u.status == "withheld" and u.standard_uncertainty is None
    assert any("was not validated" in r for r in u.reasons)
    small, cond3 = _case("S3")  # relative area noise about 0.06
    u = comp.composition_uncertainty(small, cond3, n_boot=50, seed=1)
    assert u.status == "withheld"
    assert any("relative area noise" in r for r in u.reasons)


def test_withheld_for_cases_the_validation_did_not_cover(scoped):
    lines, cond = _case()
    four = [*lines, replace(lines[0], element="N", orbital="1s", binding_energy=400.0,
                            energy=lines[0].energy + 99.0 - 400.0,
                            window=(lines[0].window[0] + 99.0 - 400.0,
                                    lines[0].window[1] + 99.0 - 400.0)),
            replace(lines[0], element="C", orbital="1s", binding_energy=285.0,
                    energy=lines[0].energy + 99.0 - 285.0,
                    window=(lines[0].window[0] + 99.0 - 285.0,
                            lines[0].window[1] + 99.0 - 285.0))]
    u = comp.composition_uncertainty(four, replace(cond, declared_elements=("Si", "O", "N", "C")),
                                     n_boot=30, seed=1)
    assert any("validated up to 3" in r for r in u.reasons)
    curves = [replace(ln, transmission=np.ones_like(ln.energy)) for ln in lines]
    u = comp.composition_uncertainty(curves, replace(cond, transmission_applied="not_applied"),
                                     n_boot=30, seed=1)
    assert any("transmission divided" in r for r in u.reasons)


def test_raw_counts_that_are_not_integers_are_not_evaluated(scoped):
    lines, cond = val.expected_lines("S1"), val.conditions("S1")  # expected, not counts
    u = comp.composition_uncertainty(lines, cond, n_boot=10, seed=1)
    assert u.status == "not_evaluated" and "not integer" in u.reasons[0]


def test_withheld_when_a_replicate_is_refused(scoped):
    """A peak barely above its background: some replicates have no
    positive net area, so a standard deviation over the rest would be
    conditional on success."""
    lines, cond = _case()
    weak = replace(lines[0], intensity=np.round(200.0 + 0.002 * (lines[0].intensity - 200.0)))
    u = comp.composition_uncertainty([weak, lines[1]], cond, n_boot=200, seed=5)
    assert u.status == "withheld" and u.n_refused > 0
    assert any("replicates were refused" in r for r in u.reasons)


def test_withheld_with_no_validated_scope(monkeypatch):
    monkeypatch.setattr(comp, "VALIDATED_SCOPE", None)
    lines, cond = _integer_case()
    u = comp.composition_uncertainty(lines, cond, n_boot=20, seed=1)
    assert u.status == "withheld" and any("no validated scope" in r for r in u.reasons)


def _verdict(ci):
    lo, hi = ci
    return "pass" if 0.9 <= lo and hi <= 1.1 else "fail" if hi < 0.9 or lo > 1.1 else "undecided"


def test_the_records_verdicts_follow_from_their_intervals():
    """Recompute every stored verdict instead of trusting it."""
    for el in (e for s in json.loads(RECORD.read_text())["scenarios"]
               for e in s["elements"].values()):
        assert el["verdict"] == _verdict(el["R_mc_95"])
    record2 = json.loads(RECORD2.read_text())
    for s in record2["scenarios"]:
        assert "definition" in s and s["n_attempted"] == 2000
        for block in (s["R_all"], s["R_admitted"] or {}):
            for el in block.values():
                assert el["verdict"] == _verdict(el["R_mc_95"])


def test_the_shipped_scope_is_the_one_the_second_record_decided():
    """§6 decided no scope for either background, so nothing is published."""
    decision = json.loads(RECORD2.read_text())["decision"]
    if all(d["r_star"] is None for d in decision.values()):
        assert comp.VALIDATED_SCOPE is None
    else:
        assert comp.VALIDATED_SCOPE["r_threshold"] == {
            bg: d["r_threshold"] for bg, d in decision.items() if d["r_star"] is not None}
    for d in decision.values():
        # Every level tried is in the trace, with the admitted R that decided it.
        assert d["trace"] or d["r_star"] is None
    lines, cond = _integer_case()
    u = comp.composition_uncertainty(lines, cond, n_boot=20, seed=1)
    assert u.status == "withheld" and u.standard_uncertainty is None
    assert "no validated scope" in u.reasons[0]


def test_the_harness_runs_and_resumes(tmp_path):
    out = tmp_path / "S1"
    val.run("S1", n_outer=6, n_boot=5, out=out, chunk=3, workers=1)
    first = sorted(p.read_text() for p in out.glob("chunk_*.json"))
    val.run("S1", n_outer=6, n_boot=5, out=out, chunk=3, workers=1)  # nothing to redo
    assert sorted(p.read_text() for p in out.glob("chunk_*.json")) == first
    s = val.summarize(out, n_resample=50)
    assert s["n_outer_attempted"] == 6 and set(s["elements"]) == {"Si", "O"}


def test_the_seed_is_respected_and_changes_the_draws(scoped):
    lines, cond = _case()
    a = comp.composition_uncertainty(lines, cond, n_boot=60, seed=11)
    b = comp.composition_uncertainty(lines, cond, n_boot=60, seed=12)
    assert a.standard_uncertainty != b.standard_uncertainty


def test_negative_counts_are_not_evaluated_rather_than_raising(scoped):
    lines, cond = _case()
    bad = replace(lines[0], intensity=lines[0].intensity - 1e6)
    u = comp.composition_uncertainty([bad, lines[1]], cond, n_boot=10, seed=1)
    assert u.status == "not_evaluated"


def test_the_standard_uncertainty_is_the_sample_sd_of_the_replicates(scoped):
    """ddof = 1: the same seed reproduces the replicates through _bootstrap."""
    import numpy as np

    lines, cond = _case()
    u = comp.composition_uncertainty(lines, cond, n_boot=40, seed=21)
    rows, refused = comp._bootstrap(lines, cond, 40, np.random.default_rng(21), u.elements)
    assert refused == 0
    for k, el in enumerate(u.elements):
        assert u.standard_uncertainty[el] == pytest.approx(np.std(rows[:, k], ddof=1), rel=1e-12)
