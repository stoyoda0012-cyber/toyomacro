"""Gate 3 acceptance tests for exact_k orchestration.

Gate 3 (frozen README §2.6): exact-K is guaranteed by test — every
candidate carries exactly K structured components and no ``max_peaks``
semantics exist anywhere in the orchestrator. Plus the Phase 3 review
points: shared ParameterScales, canonical center ordering, multi-start
honesty, fixed background family, negative-amplitude evidence, and a
supported RANGE instead of an argmin-IC K.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import toyomacro.voigtfit.exact_k as exact_k_module
from toyomacro.voigtfit.exact_k import ExactKConfig, run_exact_k
from toyomacro.voigtfit.rank_diagnostics import voigt_fwhm_approx, voigt_profile

SIGMA_TRUE = 0.40
GAMMA = 0.10
FWHM = voigt_fwhm_approx(SIGMA_TRUE, GAMMA)
ENERGY = np.linspace(0.0, 20.0, 401)
NOISE_STD = 0.05


def synth(centers, amplitudes, seed=0, bg=None, noise=NOISE_STD):
    rng = np.random.default_rng(seed)
    y = np.zeros_like(ENERGY)
    for c, a in zip(centers, amplitudes):
        y = y + a * voigt_profile(ENERGY, c, SIGMA_TRUE, GAMMA)
    if bg is not None:
        y = y + bg
    return y + rng.normal(0.0, noise, size=ENERGY.shape)


CFG = ExactKConfig(sigma_init=0.5, gamma=GAMMA, k_max=3, bg_degree=1, n_starts=3)


@pytest.fixture(scope="module")
def report_separated():
    """Two peaks 3 FWHM apart on a linear background, SNR ~ 30."""
    y = synth(
        [6.0, 6.0 + 3 * FWHM], [2.0, 1.6],
        bg=0.3 + 0.02 * (ENERGY - 10.0), seed=0,
    )
    return run_exact_k(ENERGY, y, CFG, sigma_std=NOISE_STD)


@pytest.fixture(scope="module")
def report_close():
    """Two peaks 0.25 FWHM apart — designed to be non-identifiable."""
    y = synth([8.0, 8.0 + 0.25 * FWHM], [2.0, 1.6], seed=1)
    cfg = ExactKConfig(sigma_init=0.5, gamma=GAMMA, k_max=3, bg_degree=None, n_starts=3)
    return run_exact_k(ENERGY, y, cfg, sigma_std=NOISE_STD)


@pytest.fixture(scope="module")
def report_negative():
    """A positive and a NEGATIVE line: unconstrained LS must flag it."""
    y = synth([6.0, 6.0 + 3 * FWHM], [2.0, -0.8], seed=2, noise=0.02)
    cfg = ExactKConfig(sigma_init=0.5, gamma=GAMMA, k_max=2, bg_degree=None, n_starts=3)
    return run_exact_k(ENERGY, y, cfg, sigma_std=0.02)


# ------------------------------------------------- Gate 3 core: exact-K


def test_G3_every_candidate_has_exactly_k_components(report_separated):
    assert [c.k_structured for c in report_separated.candidates] == [1, 2, 3]
    for cand in report_separated.candidates:
        assert len(cand.model.components) == cand.k_structured
        assert cand.model.k_structured == cand.k_structured
        if cand.score.success:
            assert len(cand.centers) == cand.k_structured
            assert len(cand.amplitudes) == cand.k_structured


def test_G3_no_max_peaks_semantics_in_orchestrator_source():
    source = Path(exact_k_module.__file__).read_text()
    assert "max_peaks" not in source.replace("``max_peaks``", "")


# ------------------------------------------------- review point: shared scales


def test_shared_parameter_scales_across_all_candidates(report_separated):
    ref = report_separated.parameter_scales
    for cand in report_separated.candidates:
        assert cand.diagnostics is not None
        assert cand.diagnostics.parameter_scales == ref


# ------------------------------------------------- recovery on a clear case


def test_well_separated_peaks_supported_range_is_2(report_separated):
    r = report_separated
    assert r.supported_k == (2,)
    assert r.verdict == "supported"
    assert r.selection.best_by["aicc"] == 2
    assert r.selection.best_by["bic"] == 2

    k2 = r.candidates[1]
    np.testing.assert_allclose(
        k2.centers, [6.0, 6.0 + 3 * FWHM], atol=0.05
    )
    assert k2.sigma == pytest.approx(SIGMA_TRUE, abs=0.05)
    assert all(a > 0 for a in k2.amplitudes)


def test_close_peaks_yield_ambiguous_on_ic_disagreement(report_close):
    """S2 + Gate 3 review: at 0.25 FWHM AICc and BIC name different best
    K, so the frozen rule demands `ambiguous` — even though the
    both-deltas support range is the single K=1."""
    r = report_close
    assert r.selection.best_by["aicc"] != r.selection.best_by["bic"]
    assert r.verdict == "ambiguous"
    assert any("disagree" in reason for reason in r.verdict_reasons)
    assert not (r.verdict == "supported" and r.supported_k == (2,))


# ------------------------------------------------- review point: label order


def test_centers_are_reported_in_canonical_ascending_order(report_separated):
    for cand in report_separated.candidates:
        if cand.score.success and len(cand.centers) > 1:
            assert list(cand.centers) == sorted(cand.centers)


# ------------------------------------------------- review point: multi-start


def test_multistart_bookkeeping_is_recorded(report_separated):
    for cand in report_separated.candidates:
        assert len(cand.rss_per_start) == CFG.n_starts
        assert len(cand.start_converged) == CFG.n_starts
        conv = [
            r for r, c in zip(cand.rss_per_start, cand.start_converged)
            if c and r is not None
        ]
        assert conv
        if cand.chosen_start is not None:
            assert cand.start_converged[cand.chosen_start]
            assert cand.rss_per_start[cand.chosen_start] == min(conv)


# ------------------------------------------------- review point: fixed bg


def test_background_family_is_fixed_across_candidates(report_separated):
    for cand in report_separated.candidates:
        assert cand.model.bg_degree == CFG.bg_degree
        assert len(cand.bg_coefficients) == CFG.bg_degree + 1


# ------------------------------------------------- review point: neg. amps


def test_negative_amplitude_is_flagged_and_collected_as_nnls_evidence(report_negative):
    r = report_negative
    assert 2 in r.negative_amplitude_k
    k2 = r.candidates[1]
    assert any(f.startswith("negative_amplitude") for f in k2.boundary_flags)
    assert any("NNLS" in w for w in k2.warnings)
    assert any("NNLS" in w for w in r.warnings)
    # the flag also reaches the per-candidate IC warnings via boundary_flags
    assert any("regularity" in w for w in k2.score.warnings)
    # Gate 3 review: a selected candidate with negative amplitudes is
    # never claimed as "supported"
    if r.supported_k == (2,):
        assert r.verdict == "ambiguous"
        assert any("negative" in reason for reason in r.verdict_reasons)


# ------------------------------------------------- review point: range, not argmin


def test_report_returns_supported_range_and_evidence_not_bare_argmin(report_separated):
    r = report_separated
    assert isinstance(r.supported_k, tuple)
    assert len(r.evidence) == len(r.candidates)
    for ev in r.evidence:
        assert set(ev) >= {
            "k", "delta_aicc", "delta_bic", "ic_supported",
            "linear_rank_deficit", "nonlinear_rank_deficit",
            "rank_supported", "negative_amplitudes", "multistart_warning",
        }
    # threshold is echoed, not hidden
    assert r.config["delta_ic_threshold"] == CFG.delta_ic_threshold


# ------------------------------------------------- review: converged starts


class _FakeResult:
    def __init__(self, cost, x, status, message):
        self.cost, self.x, self.status, self.message = cost, x, status, message


def test_nonconverged_start_with_lowest_rss_is_not_chosen(monkeypatch):
    """A non-converged start reporting the smallest RSS must lose to a
    converged start; its RSS stays recorded for the report."""
    calls = {"n": 0}

    def fake_ls(fun, x0, bounds=None):
        i = calls["n"]
        calls["n"] += 1
        if i == 0:
            return _FakeResult(0.001, np.array([9.0, 0.45]), 0, "max iterations")
        return _FakeResult(10.0 + i, np.array([10.0 + 0.01 * i, 0.5]), 1, "converged")

    monkeypatch.setattr(exact_k_module, "least_squares", fake_ls)
    y = synth([10.0], [2.0], seed=3)
    cfg = ExactKConfig(sigma_init=0.5, gamma=GAMMA, k_max=1, bg_degree=None, n_starts=3)
    r = run_exact_k(ENERGY, y, cfg, sigma_std=NOISE_STD)

    cand = r.candidates[0]
    assert cand.score.success
    assert cand.start_converged == (False, True, True)
    assert cand.chosen_start == 1  # min RSS among CONVERGED starts
    assert cand.rss_per_start[0] == pytest.approx(0.002)  # recorded, not chosen
    assert cand.score.rss == pytest.approx(2.0 * 11.0)
    assert any("did not converge" in w for w in cand.warnings)


def test_candidate_fails_only_when_no_start_converges(monkeypatch):
    def fake_ls(fun, x0, bounds=None):
        return _FakeResult(1.0, np.array([9.0, 0.45]), 0, "max iterations")

    monkeypatch.setattr(exact_k_module, "least_squares", fake_ls)
    y = synth([10.0], [2.0], seed=3)
    cfg = ExactKConfig(sigma_init=0.5, gamma=GAMMA, k_max=1, bg_degree=None, n_starts=2)
    r = run_exact_k(ENERGY, y, cfg, sigma_std=NOISE_STD)

    cand = r.candidates[0]
    assert not cand.score.success
    assert cand.start_converged == (False, False)
    assert cand.chosen_start is None
    assert "no start converged" in cand.score.termination_reason
    assert r.verdict == "unsupported"


# ------------------------------------------------- review: descending axis


def test_descending_energy_axis_is_equivalent_to_ascending():
    """XPS binding-energy axes are often descending; the report must be
    identical after internal normalization."""
    y = synth([6.0, 6.0 + 3 * FWHM], [2.0, 1.6], seed=4)
    cfg = ExactKConfig(sigma_init=0.5, gamma=GAMMA, k_max=2, bg_degree=None, n_starts=2)
    r_asc = run_exact_k(ENERGY, y, cfg, sigma_std=NOISE_STD)
    r_desc = run_exact_k(ENERGY[::-1], y[::-1], cfg, sigma_std=NOISE_STD)

    assert r_desc.supported_k == r_asc.supported_k
    assert r_desc.verdict == r_asc.verdict
    for c_asc, c_desc in zip(r_asc.candidates, r_desc.candidates):
        np.testing.assert_allclose(c_desc.centers, c_asc.centers, atol=1e-9)
        np.testing.assert_allclose(c_desc.amplitudes, c_asc.amplitudes, rtol=1e-9)
        assert c_desc.score.rss == pytest.approx(c_asc.score.rss, rel=1e-12)


def test_non_monotonic_energy_axis_is_rejected():
    y = synth([10.0], [2.0], seed=5)
    bad = ENERGY.copy()
    bad[5] = bad[3]
    cfg = ExactKConfig(sigma_init=0.5, gamma=GAMMA, k_max=1)
    with pytest.raises(ValueError, match="monotonic"):
        run_exact_k(bad, y, cfg, sigma_std=NOISE_STD)


# ------------------------------------------------- Phase 3b: plug-in bg


def shirley_like_step(y0: np.ndarray, height: float) -> np.ndarray:
    """Shirley-shaped step: proportional to the integrated peak
    intensity at higher energies (high at the low-energy end)."""
    tail = np.cumsum(y0[::-1])[::-1]
    return height * tail / tail[0]


@pytest.fixture(scope="module")
def report_plugin_shirley():
    y0 = sum(
        a * voigt_profile(ENERGY, c, SIGMA_TRUE, GAMMA)
        for c, a in zip([6.0, 6.0 + 3 * FWHM], [2.0, 1.6])
    )
    rng = np.random.default_rng(7)
    y = y0 + shirley_like_step(y0, 0.3) + rng.normal(0.0, NOISE_STD, ENERGY.shape)
    cfg = ExactKConfig(
        sigma_init=0.5, gamma=GAMMA, k_max=3, bg_degree=None,
        n_starts=3, plugin_background="shirley",
    )
    return run_exact_k(ENERGY, y, cfg, sigma_std=NOISE_STD)


def test_plugin_background_computed_once_and_shared(monkeypatch):
    """Plan 3b item 1-2: the curve is estimated ONCE before the K loop
    and the same curve serves every candidate."""
    from toyomacro.background import Shirley

    calls = {"n": 0}
    orig = Shirley.calculate

    def counting(self, *a, **kw):
        calls["n"] += 1
        return orig(self, *a, **kw)

    monkeypatch.setattr(Shirley, "calculate", counting)
    y = synth([10.0], [2.0], seed=8)
    cfg = ExactKConfig(
        sigma_init=0.5, gamma=GAMMA, k_max=3, n_starts=2,
        plugin_background="shirley",
    )
    run_exact_k(ENERGY, y, cfg, sigma_std=NOISE_STD)
    assert calls["n"] == 1


def test_plugin_background_not_counted_as_free_parameter(report_plugin_shirley):
    """Plan 3b item 3: no linear column, no dof contribution — the
    parameter count equals the no-background model's."""
    for cand in report_plugin_shirley.candidates:
        k = cand.k_structured
        # K amplitudes + K centers + 1 shared sigma + 1 variance
        assert cand.model.n_free_parameters == 2 * k + 2
        assert cand.model.bg_degree is None
        assert cand.bg_coefficients == ()
        assert cand.diagnostics.column_names["linear"] == [
            f"amplitude_{i}" for i in range(k)
        ]


def test_plugin_background_mode_settings_and_heuristic_warning(report_plugin_shirley):
    """Plan 3b items 4-6: mode + settings recorded; IC-heuristic and
    no-cross-family-comparison warnings present."""
    bg = report_plugin_shirley.background
    assert bg["mode"] == "plugin_shirley"
    assert bg["settings"] == {"max_iter": 50, "tol": 1e-5, "auto_range": False}
    assert "curve_summary" in bg
    assert any(
        "heuristic" in w and "families" in w
        for w in report_plugin_shirley.warnings
    )


def test_plugin_shirley_end_to_end_recovers_two_peaks(report_plugin_shirley):
    """K=2 is recovered through the plug-in background. BIC names K=2
    robustly; AICc may chase noise with a (negative-amplitude) third
    component, in which case the criteria disagree and the verdict must
    be ambiguous — never a silent unsupported/overfit claim."""
    r = report_plugin_shirley
    assert r.selection.best_by["bic"] == 2
    assert r.verdict in ("supported", "ambiguous")
    if r.verdict == "ambiguous":
        assert any("disagree" in reason for reason in r.verdict_reasons)
    else:
        assert r.supported_k == (2,)
    k2 = r.candidates[1]
    np.testing.assert_allclose(k2.centers, [6.0, 6.0 + 3 * FWHM], atol=0.06)


def test_plugin_tougaard_records_universal_c():
    y = synth([10.0], [2.0], seed=9, noise=0.02)
    cfg = ExactKConfig(
        sigma_init=0.5, gamma=GAMMA, k_max=1, n_starts=2,
        plugin_background="tougaard",
    )
    r = run_exact_k(ENERGY, y, cfg, sigma_std=0.02)
    assert r.background["mode"] == "plugin_tougaard"
    assert r.background["settings"] == {"C": 1643.0, "universal": True}


def test_no_plugin_records_polynomial_or_none_mode(report_separated, report_close):
    assert report_separated.background == {"mode": "polynomial", "bg_degree": 1}
    assert report_close.background == {"mode": "none", "bg_degree": None}


def test_report_is_json_serializable(report_separated):
    blob = json.dumps(report_separated.to_dict())
    back = json.loads(blob)
    assert set(back) >= {
        "candidates", "selection", "supported_k", "verdict",
        "verdict_reasons", "evidence", "negative_amplitude_k",
        "parameter_scales", "config", "warnings",
    }
    assert back["verdict"] in ("supported", "ambiguous", "unsupported")
