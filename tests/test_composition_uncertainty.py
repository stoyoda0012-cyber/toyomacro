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

RECORD = Path(__file__).parents[1] / "docs" / "design" / "composition-uncertainty-results.json"


def _case(name="S1"):
    return val.expected_lines(name), val.conditions(name)


@pytest.fixture
def scoped(monkeypatch):
    monkeypatch.setattr(comp, "VALIDATED_SCOPE",
                        {"backgrounds": ("linear",), "min_raw_area": {"linear": 1000.0}})


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
    small, cond3 = _case("S3")
    u = comp.composition_uncertainty(small, cond3, n_boot=20, seed=1)
    assert u.status == "withheld"
    assert any("below the validated" in r for r in u.reasons)


def test_withheld_when_a_replicate_is_refused(scoped):
    """A peak barely above its background: some replicates have no
    positive net area, so a standard deviation over the rest would be
    conditional on success."""
    lines, cond = _case()
    weak = replace(lines[0], intensity=200.0 + 0.002 * (lines[0].intensity - 200.0))
    u = comp.composition_uncertainty([weak, lines[1]], cond, n_boot=200, seed=5)
    assert u.status == "withheld" and u.n_refused > 0
    assert any("replicates were refused" in r for r in u.reasons)


def test_withheld_with_no_validated_scope(monkeypatch):
    monkeypatch.setattr(comp, "VALIDATED_SCOPE", None)
    lines, cond = _case()
    u = comp.composition_uncertainty(lines, cond, n_boot=20, seed=1)
    assert u.status == "withheld" and any("no validated scope" in r for r in u.reasons)


def test_the_scope_constant_is_the_one_the_record_supports():
    """VALIDATED_SCOPE must follow from the committed validation record:
    the backgrounds of passing acceptance scenarios, and the smallest
    line area among them."""
    record = json.loads(RECORD.read_text())
    passing = [s for s in record["scenarios"]
               if s["role"] == "acceptance" and s["verdict"] == "pass"]
    if not passing:
        assert comp.VALIDATED_SCOPE is None
        return
    assert set(comp.VALIDATED_SCOPE["backgrounds"]) == {s["background"] for s in passing}
    for bg in comp.VALIDATED_SCOPE["backgrounds"]:
        smallest = min(min(val.SCENARIOS[s["scenario"]][:2])
                       for s in passing if s["background"] == bg)
        assert comp.VALIDATED_SCOPE["min_raw_area"][bg] == smallest
    for s in record["scenarios"]:
        for el in s["elements"].values():
            assert el["mc_half_width"] <= 0.05  # the pre-registered precision target
    for s in record["scenarios"]:
        assert s["n_outer_attempted"] == 2000 and s["n_boot"] == 1000


def test_the_published_scope_is_evaluated_at_its_edges():
    """S2 sits at the Shirley minimum; S1 at the linear one."""
    for name in ("S1", "S2"):
        lines, cond = _case(name)
        u = comp.composition_uncertainty(lines, cond, n_boot=50, seed=2)
        assert u.status == "evaluated", (name, u.reasons)
    lines, cond = _case("S2")
    u = comp.composition_uncertainty(lines, replace(cond, background="linear"), n_boot=50, seed=2)
    assert u.status == "withheld"  # linear was validated only at 30,000


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
