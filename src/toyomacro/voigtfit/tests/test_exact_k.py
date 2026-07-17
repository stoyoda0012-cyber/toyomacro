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


def test_close_peaks_are_not_forced_to_k2(report_close):
    """S2 criterion: at 0.25 FWHM the success condition is reporting
    non-identifiability/ambiguity, NOT recovering K=2."""
    r = report_close
    forced_unique_k2 = r.verdict == "supported" and r.supported_k == (2,)
    assert not forced_unique_k2
    assert r.verdict in ("supported", "ambiguous", "unsupported")
    assert r.verdict_reasons  # the verdict always carries its evidence


# ------------------------------------------------- review point: label order


def test_centers_are_reported_in_canonical_ascending_order(report_separated):
    for cand in report_separated.candidates:
        if cand.score.success and len(cand.centers) > 1:
            assert list(cand.centers) == sorted(cand.centers)


# ------------------------------------------------- review point: multi-start


def test_multistart_bookkeeping_is_recorded(report_separated):
    for cand in report_separated.candidates:
        assert len(cand.rss_per_start) == CFG.n_starts
        finite = [r for r in cand.rss_per_start if r is not None]
        assert finite
        if cand.chosen_start is not None:
            assert cand.rss_per_start[cand.chosen_start] == min(finite)


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


def test_report_is_json_serializable(report_separated):
    blob = json.dumps(report_separated.to_dict())
    back = json.loads(blob)
    assert set(back) >= {
        "candidates", "selection", "supported_k", "verdict",
        "verdict_reasons", "evidence", "negative_amplitude_k",
        "parameter_scales", "config", "warnings",
    }
    assert back["verdict"] in ("supported", "ambiguous", "unsupported")
