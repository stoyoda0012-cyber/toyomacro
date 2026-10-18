"""The §8 harness (``_composition_validation3``) and ``_predict_r``.

The validation itself is a recorded run, not a test. Here each clause of
docs/design/composition-uncertainty.md §8 that the harness implements is
pinned on small cases: the prediction's determinism and inputs, the
count setting's window and failure, the seed series, the per-condition
rule, the implementation check and the execution-identity refusals.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import toyomacro.composition as comp
from toyomacro import _composition_validation3 as v3

CID = v3.condition_id("P", "linear", 0.02)


# --- registered constants ---------------------------------------------------

def test_condition_ids_cover_0_to_39_once():
    ids = [v3.condition_id(c, b, lv) for c in v3.CONFIG_NAMES for b in v3.BACKGROUNDS
           for lv in v3.LEVELS]
    assert sorted(ids) == list(range(40))
    assert all(v3.condition_id(*v3.condition(c)) == c for c in range(40))
    assert v3.condition_id("B", "shirley", 0.08) == 10 + 5 + 4


def test_the_registered_lists_are_the_ones_in_section_8():
    assert v3.CANDIDATES == (0.02, 0.01, 0.005)
    assert len(v3.KAPPA) == 9 and (1.0, 1.0) in v3.KAPPA
    assert v3.RUN_SIZES == {"pilot": (200, 200), "decision": (2000, 1000)}
    assert v3.SHIRLEY == {"max_iter": 50, "tol": 1e-5, "auto_range": False}
    assert (v3.B_EXPLORE, v3.B_PRED, v3.B_DATA, v3.N_RESAMPLE) == (200, 1000, 1000, 5000)
    assert v3.WINDOW == (0.9, 1.0) and v3.AIM == 0.95 and v3.MAX_STEPS == 8


def test_the_shirley_defaults_are_what_section_8_registers():
    import inspect

    from toyomacro.background import Shirley
    sig = inspect.signature(Shirley.calculate)
    for name, value in v3.SHIRLEY.items():
        assert sig.parameters[name].default == value


def test_config_hash_follows_the_constants(monkeypatch):
    before = v3.config_hash()
    assert before == v3.config_hash()
    monkeypatch.setitem(v3.REGISTERED, "B_pred", 999)
    assert v3.config_hash() != before


# --- declared spectra -------------------------------------------------------

def test_kappa_scales_peak_and_base_separately():
    mu, _ = v3.declared(CID, 3e4, (1.0, 1.0))
    d, _ = v3.declared(CID, 3e4, (2.0, 0.5))
    parts, _ = v3.parts(CID, 3e4)
    for ln_mu, ln_d, (_, peak, base) in zip(mu, d, parts):
        assert np.allclose(ln_mu.intensity, peak + base)
        assert np.allclose(ln_d.intensity, 2.0 * peak + 0.5 * base)
    e1, _ = v3.expected(CID, 3e4)
    assert np.array_equal(e1[0].intensity, mu[0].intensity)


# --- the prediction ---------------------------------------------------------

def test_predict_r_is_deterministic_and_takes_non_integer_means():
    lines, cond = v3.declared(CID, 2.5e4, (1.0, 1.0))
    assert not all(np.array_equal(ln.intensity, np.round(ln.intensity)) for ln in lines)
    a = comp._predict_r(lines, cond, n_boot=60)
    np.random.default_rng(0).poisson(5.0, 1000)  # unrelated draws in between
    comp._predict_r(v3.declared(CID, 1e3, (2.0, 2.0))[0], cond, n_boot=60)
    b = comp._predict_r(lines, cond, n_boot=60)
    assert a.r_pred == b.r_pred and a.r_pred > 0 and a.n_refused == b.n_refused == 0
    assert not a.declared_refused and a.elements == ("Si", "O")


def test_predict_r_refuses_bad_arrays_and_reports_a_refused_declaration():
    lines, cond = v3.declared(CID, 2.5e4)
    for bad in (-1.0, math.inf, math.nan):
        broken = [replace(lines[0], intensity=np.append(lines[0].intensity[:-1], bad))] + lines[1:]
        with pytest.raises(ValueError):
            comp._predict_r(broken, cond, n_boot=5)
    p = comp._predict_r(lines, replace(cond, declared_elements=None), n_boot=5)
    assert p.declared_refused and p.r_pred is None and p.elements == ()


# --- count setting ----------------------------------------------------------

def _stub_r(scale):
    """An r that falls as 1/sqrt(A), for the window arithmetic."""
    def explore(lines, cond, cid, k):
        return scale / math.sqrt(lines[0].intensity.max()), None

    def predict(lines, cond, n_boot):
        return comp._Prediction(scale / math.sqrt(lines[0].intensity.max()), False, (), 0, n_boot,
                                ("Si", "O"))
    return explore, predict


def test_count_setting_lands_in_the_one_sided_window():
    explore, predict = _stub_r(scale=40.0)
    out = v3.set_counts(CID, predict=predict, explore=explore)
    level = v3.condition(CID)[2]
    assert out["established"] and 0.9 * level <= out["r_pred"] <= level
    assert out["candidate"] == len(out["trace"]) and out["trace"][-1]["final"]
    assert out["trace"][0]["si_area"] == v3.AREA_1


def test_count_setting_not_established_after_eight_steps_grows_by_four():
    def explore(lines, cond, cid, k):
        return None, "composition(mu) refused: test"
    out = v3.set_counts(CID, predict=None, explore=explore)
    assert not out["established"] and len(out["trace"]) == 8
    areas = [t["si_area"] for t in out["trace"]]
    assert all(b == pytest.approx(4 * a) for a, b in zip(areas, areas[1:]))


# --- outer trials and the per-condition rule --------------------------------

def test_a_trial_returns_category_3_with_a_finite_se_and_r():
    row = v3.trial(CID, "pilot", 3, 3e4, n_boot=8)
    assert row["category"] == 3 and row["cause"] is None
    assert len(row["estimate"]) == 2 and all(math.isfinite(v) for v in row["se"]) and row["r"] > 0
    again = v3.trial(CID, "pilot", 3, 3e4, n_boot=8)
    assert again == row  # series 3 and 4 are keyed by (run kind, condition, trial)
    other = v3.trial(CID, "decision", 3, 3e4, n_boot=8)
    assert other["estimate"] != row["estimate"]  # the run kind is in the key


def test_a_refused_composition_is_category_1(monkeypatch):
    refused = comp.composition(*v3.expected(CID, 3e4))
    refused = replace(refused, status="refused", fractions=None, reasons=("test",))
    monkeypatch.setattr(v3, "composition", lambda lines, cond: refused)
    row = v3.trial(CID, "pilot", 0, 3e4, n_boot=5)
    assert row["category"] == 1 and "test" in row["cause"] and row["estimate"] is None


def test_the_monte_carlo_interval_shares_its_indices_across_elements():
    rng = np.random.default_rng(3)
    si = rng.normal(0.3, 0.01, 50)
    e = np.column_stack([si, 1 - si])
    se = np.column_stack([np.full(50, 0.01), np.full(50, 0.01)])
    out = v3.r_interval(e, se, v3.generator([v3.SEED, 5, 0, CID]))
    assert out["R"][0] == pytest.approx(out["R"][1])
    assert out["R_mc_95"][0] == pytest.approx(out["R_mc_95"][1])


def test_binomial_upper_bound_matches_section_8():
    assert v3.binomial_upper_95(0, 2000) == pytest.approx(0.001497, abs=2e-6)
    assert v3.binomial_upper_95(2000, 2000) == 1.0


def test_rule_4_needs_the_level_above_to_pass_and_fails_at_0_08():
    def cond_summary(passes):
        return {"pass": passes, "admission": []}
    conds = {}
    for cid in range(40):
        cfg, bg, level = v3.condition(cid)
        conds[cid] = cond_summary(True)
        # every configuration admitted at 0.005 and 0.01 (kappa (1,1)), nothing above
        conds[cid]["admission"] = [{"kappa": [1.0, 1.0], "r_pred": level, "declared_refused": False,
                                    "n_refused": 0, "eligible": True}]
    d = v3.decide(conds)
    assert d["linear"]["r_star"] == 0.02 and d["shirley"]["r_star"] == 0.02
    # the level above the highest admitted one fails for B/linear -> 0.02 does not stand
    conds[v3.condition_id("B", "linear", 0.04)]["pass"] = False
    assert v3.decide(conds)["linear"]["r_star"] == 0.01
    # a configuration admitted at 0.08 has no level above
    conds[v3.condition_id("B", "linear", 0.08)]["admission"][0]["r_pred"] = 0.02
    assert v3.decide(conds)["linear"]["r_star"] == 0.01
    conds[v3.condition_id("B", "linear", 0.08)]["admission"][0]["r_pred"] = 0.01
    assert v3.decide(conds)["linear"]["r_star"] == 0.005


# --- execution identity and files -------------------------------------------

def test_identity_refuses_a_dirty_checkout(monkeypatch):
    calls = {"status": " M src/x.py"}
    monkeypatch.setattr(v3, "_git", lambda root, *a: calls["status"] if "status" in a
                        else ("abc" if "HEAD" in a else str(Path(v3.toyomacro.__file__).parents[2])))
    with pytest.raises(RuntimeError, match="modified tracked"):
        v3.identity()
    calls["status"] = ""
    ident = v3.identity()
    assert ident["git_commit"] == "abc" and ident["config_hash"] == v3.config_hash()


def test_output_dir_must_be_outside_or_ignored(tmp_path):
    root = Path(v3.toyomacro.__file__).parents[2]
    v3.check_output_dir(tmp_path, root)
    v3.check_output_dir(root / "outputs" / "x", root)
    with pytest.raises(RuntimeError, match="not ignored"):
        v3.check_output_dir(root / "src" / "x", root)


def test_atomic_write_leaves_no_temporary_file(tmp_path):
    v3._write_atomic(tmp_path / "a.json", {"k": 1})
    assert json.loads((tmp_path / "a.json").read_text()) == {"k": 1}
    assert list(tmp_path.iterdir()) == [tmp_path / "a.json"]


def test_missing_trial_ids_are_refused(tmp_path):
    d = tmp_path / f"c{CID:02d}"
    d.mkdir()
    v3._write_atomic(d / "chunk_000000.json", [{"trial": 0}, {"trial": 1}])
    with pytest.raises(RuntimeError, match="incomplete"):
        v3._load_trials(tmp_path, CID, 4)


@pytest.fixture
def fake_identity(monkeypatch):
    root = Path(v3.toyomacro.__file__).parents[2]
    monkeypatch.setattr(v3, "identity", lambda root=None: {
        "design": v3.DESIGN, "config_hash": v3.config_hash(), "git_commit": "test", "checkout": str(root),
        "python": "3", "numpy": "2", "scipy": "1", "generator": v3.GENERATOR, "machine": "test"})


def test_a_small_run_summarises_and_resumes(tmp_path, fake_identity, monkeypatch):
    monkeypatch.setattr(v3, "B_EXPLORE", 20)
    monkeypatch.setattr(v3, "B_PRED", 40)
    monkeypatch.setitem(v3.REGISTERED, "B_explore", 20)
    monkeypatch.setitem(v3.REGISTERED, "B_pred", 40)
    out = tmp_path / "pilot"
    v3.run("pilot", out, n_outer=6, n_boot=5, chunk=4, workers=1, conditions=(CID,))
    chunks = sorted(out.glob("c*/chunk_*.json"))
    assert [c.name for c in chunks] == ["chunk_000000.json", "chunk_000004.json"]
    first = chunks[0].read_text()
    v3.run("pilot", out, n_outer=6, n_boot=5, chunk=4, workers=1, conditions=(CID,))
    assert chunks[0].read_text() == first  # nothing redone
    rec = v3.summarize(out, workers=1)
    (c,) = rec["conditions"]
    assert c["condition"] == CID and c["n_trials"] == 6 and sum(c["n_category"].values()) == 6
    assert c["implementation_check"]["ok"] and len(c["admission"]) == 9
    assert c["count_setting"]["established"] and 0.9 * 0.02 <= c["count_setting"]["r_pred"] <= 0.02
    assert set(rec["decision"]) == {"linear", "shirley"}
    # a resume under another configuration hash is refused
    monkeypatch.setitem(v3.REGISTERED, "B_pred", 41)
    with pytest.raises(RuntimeError, match="resume refused"):
        v3.run("pilot", out, n_outer=6, n_boot=5, chunk=4, workers=1, conditions=(CID,))


# --- the audit's negative fences ---------------------------------------------

def test_the_seeds_are_the_registered_ones():
    assert v3.SEED == 20261014 and comp._PREDICTION_SEED_KEY == (20261014, 2)
    assert v3.REGISTERED["seed"] == 20261014 and v3.REGISTERED["series"]["2"] == "[seed, 2]"


def test_predict_r_rejects_complex_arrays_and_reports_not_run_when_refused():
    lines, cond = v3.declared(CID, 2.5e4)
    broken = [replace(lines[0], intensity=lines[0].intensity.astype(complex))] + lines[1:]
    with pytest.raises(ValueError, match="real"):
        comp._predict_r(broken, cond, n_boot=5)
    p = comp._predict_r(lines, replace(cond, declared_elements=None), n_boot=5)
    assert p.declared_refused and p.n_refused is None


def test_series_3_and_4_are_distinct_streams():
    a = v3.generator([v3.SEED, 3, 0, CID, 7]).integers(0, 1 << 30, 8)
    b = v3.generator([v3.SEED, 4, 0, CID, 7]).integers(0, 1 << 30, 8)
    assert not np.array_equal(a, b)
    assert np.array_equal(a, v3.generator([v3.SEED, 3, 0, CID, 7]).integers(0, 1 << 30, 8))


def test_a_refused_replicate_is_category_2_withheld(monkeypatch):
    real = v3._bootstrap_detail

    def one_refused(lines, cond, n_boot, rng, elements):
        rows, refused, areas = real(lines, cond, n_boot, rng, elements)
        return rows[:-1], refused + 1, areas[:-1]
    monkeypatch.setattr(v3, "_bootstrap_detail", one_refused)
    row = v3.trial(CID, "pilot", 0, 3e4, n_boot=6)
    assert row["category"] == 2 and row["cause"].startswith("withheld") and row["se"] is None
    s = v3.condition_summary(CID, "pilot", [row], 1)
    assert s["causes"] == {"withheld": 1} and s["rate"] == 1.0 and not s["pass"]


def test_a_condition_with_no_spread_is_not_evaluable():
    rows = [{"trial": i, "category": 3, "cause": None, "estimate": [0.3, 0.7], "se": [0.01, 0.01],
             "r": 0.01} for i in range(5)]
    s = v3.condition_summary(CID, "pilot", rows, 5)
    assert s["evaluable"] is False and s["verdict"] == "not evaluable" and not s["pass"]


def test_the_implementation_check_fails_when_the_published_set_differs():
    rows = [v3.trial(CID, "pilot", i, 3e4, n_boot=6) for i in range(4)]
    s = v3.condition_summary(CID, "pilot", rows, 4)
    assert v3.implementation_check(rows, s)["ok"]
    tampered = [dict(r) for r in rows]
    tampered[0]["se"] = [v * 2 for v in tampered[0]["se"]]
    assert not v3.implementation_check(tampered, s)["ok"]
    dropped = rows[:-1] + [{**rows[-1], "category": 2, "cause": "withheld: test", "se": None}]
    assert not v3.implementation_check(dropped, s)["ok"]


def test_the_decision_run_takes_no_override(tmp_path, fake_identity):
    with pytest.raises(RuntimeError, match="no override"):
        v3.run("decision", tmp_path / "d", n_outer=6, n_boot=5, workers=1, conditions=(CID,))


def test_summary_is_refused_from_another_commit(tmp_path, fake_identity, monkeypatch):
    monkeypatch.setattr(v3, "B_EXPLORE", 20)
    monkeypatch.setattr(v3, "B_PRED", 40)
    monkeypatch.setitem(v3.REGISTERED, "B_explore", 20)
    monkeypatch.setitem(v3.REGISTERED, "B_pred", 40)
    out = tmp_path / "pilot"
    v3.run("pilot", out, n_outer=4, n_boot=5, chunk=4, workers=1, conditions=(CID,))
    rec = v3.summarize(out, workers=1)
    (c,) = rec["conditions"]
    assert all("declared" in a and len(a["declared"]["lines"]) == 2 for a in c["admission"])
    assert all(a["structural"] for a in c["admission"])
    run = json.loads((out / "run.json").read_text())
    run["git_commit"] = "other"
    (out / "run.json").write_text(json.dumps(run))
    with pytest.raises(RuntimeError, match="summary refused"):
        v3.summarize(out, workers=1)
    with pytest.raises(RuntimeError, match="resume refused"):
        v3.run("pilot", out, n_outer=4, n_boot=5, chunk=4, workers=1, conditions=(CID,))
