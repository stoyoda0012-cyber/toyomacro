"""Gate 2 acceptance tests (E1–E6) for model_selection.

Test IDs follow benchmarks/README_rank_model_selection.md §4.2. All
expected values are hand-computable from the frozen formulas §2.5.
"""

import json
import math

import numpy as np
import pytest

from toyomacro.voigtfit.model_selection import (
    CandidateFit,
    CandidateModel,
    gaussian_ic,
    poisson_deviance,
    poisson_ic,
    poisson_loglik,
    score_candidate,
    select_peak_count,
)
from toyomacro.voigtfit.multipeak_config import ComponentConfig


def comp(center: float, so_split: float = 0.0, branch_ratio: float = 1.0) -> ComponentConfig:
    return ComponentConfig(
        center=center, sigma=0.4, gamma=0.1,
        so_split=so_split, branch_ratio=branch_ratio,
    )


def model_k(k: int, **kw) -> CandidateModel:
    return CandidateModel(components=tuple(comp(5.0 + 2.0 * i) for i in range(k)), **kw)


# ---------------------------------------------------------------- E1


def test_E1_gaussian_ic_matches_hand_computed_values():
    """n=10, RSS=2.5, k=3 — every value written out by hand.

    AIC  = 10 ln(0.25) + 2·3           = -7.8629436112
    AICc = AIC + 2·3·4/(10-3-1)        = -3.8629436112
    BIC  = 10 ln(0.25) + 3 ln(10)      = -6.9551883322
    """
    aic, aicc, bic, warns = gaussian_ic(rss=2.5, n=10, k=3)
    assert aic == pytest.approx(10 * math.log(0.25) + 6, abs=1e-12)
    assert aic == pytest.approx(-7.8629436112, abs=1e-9)
    assert aicc == pytest.approx(aic + 4.0, abs=1e-12)
    assert aicc == pytest.approx(-3.8629436112, abs=1e-9)
    assert bic == pytest.approx(10 * math.log(0.25) + 3 * math.log(10), abs=1e-12)
    assert bic == pytest.approx(-6.9551883322, abs=1e-9)
    assert warns == []


# ---------------------------------------------------------------- E2


def test_E2_poisson_loglik_matches_hand_computed_value():
    """y=[2,1,0], mu=[2,1,0.5]:

    ll = (2 ln2 - 2 - ln2!) + (0 - 1 - ln1!) + (0 - 0.5 - ln0!)
       = (2·0.6931471806 - 2 - 0.6931471806) - 1 - 0.5
       = -2.8068528194
    """
    ll = poisson_loglik(np.array([2.0, 1.0, 0.0]), np.array([2.0, 1.0, 0.5]))
    expected = (2 * math.log(2) - 2 - math.log(2)) + (-1.0) + (-0.5)
    assert ll == pytest.approx(expected, abs=1e-12)
    assert ll == pytest.approx(-2.8068528194, abs=1e-9)

    aic, aicc, bic, warns = poisson_ic(ll, n=3, k=1)
    assert aic == pytest.approx(-2 * ll + 2, abs=1e-12)
    assert bic == pytest.approx(-2 * ll + math.log(3), abs=1e-12)
    assert warns == []


def test_E2_poisson_deviance_zero_at_saturation_and_hand_value():
    y = np.array([2.0, 1.0, 0.0])
    assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-14)
    # 2·[2 ln(2/2) + 1 ln(1/1) + (0 - (0 - 0.5))] = 1.0
    assert poisson_deviance(y, np.array([2.0, 1.0, 0.5])) == pytest.approx(1.0, abs=1e-12)


def test_E2_poisson_rejects_nonpositive_mu_where_y_positive():
    with pytest.raises(ValueError, match="refusing to clip"):
        poisson_loglik(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        poisson_loglik(np.array([1.0]), np.array([-0.5]))
    # mu == 0 where y == 0 is the well-defined limit, not an error
    assert poisson_loglik(np.array([0.0]), np.array([0.0])) == pytest.approx(0.0)


# ---------------------------------------------------------------- E3


def test_E3_aicc_is_inf_not_finite_when_n_le_k_plus_1():
    aic, aicc, bic, warns = gaussian_ic(rss=1.0, n=4, k=3)
    assert math.isinf(aicc) and aicc > 0
    assert math.isfinite(aic) and math.isfinite(bic)
    assert any("AICc undefined" in w for w in warns)

    _, aicc_p, _, warns_p = poisson_ic(-5.0, n=3, k=2)
    assert math.isinf(aicc_p)
    assert any("AICc undefined" in w for w in warns_p)


# ---------------------------------------------------------------- E4


def test_E4_free_parameter_count_under_fixed_shared_so_constraints():
    """2 singlets + 1 fixed SO doublet, shared sigma, gammas fixed,
    linear bg, estimated variance:

    k = 3 (amp) + 3 (centers) + 1 (shared sigma) + 0 (gamma)
      + 2 (bg linear) + 1 (variance) = 10

    The doublet's so_split / branch_ratio are fixed structure — they
    contribute nothing, so replacing the doublet by a singlet keeps k.
    """
    m = CandidateModel(
        components=(comp(3.0), comp(6.0), comp(9.0, so_split=1.8, branch_ratio=1.5)),
        bg_degree=1,
        sigmas_free=True, shared_sigma=True,
        gammas_free=False,
        variance_estimated=True,
    )
    assert m.k_structured == 3
    assert m.n_visible_peaks == 4  # doublet shows two lines
    assert m.n_free_parameters == 10

    m_singlets = CandidateModel(
        components=(comp(3.0), comp(6.0), comp(9.0)),
        bg_degree=1,
        sigmas_free=True, shared_sigma=True,
        gammas_free=False,
        variance_estimated=True,
    )
    assert m_singlets.n_free_parameters == m.n_free_parameters
    assert m_singlets.n_visible_peaks == 3


def test_E4_default_flags_count():
    # defaults: amps + centers + per-component sigma free, no bg, no variance
    assert model_k(2).n_free_parameters == 6
    assert model_k(2, gammas_free=True).n_free_parameters == 8
    assert model_k(2, bg_degree=0).n_free_parameters == 7


# ---------------------------------------------------------------- E5


def test_E5_candidates_carry_exactly_K_structured_components():
    """The scoring layer has no 'up to K' notion: k_structured is
    len(components), preserved verbatim through scoring and selection."""
    scores = []
    for k in (1, 2, 3):
        m = model_k(k)
        assert m.k_structured == k == len(m.components)
        scores.append(
            score_candidate(m, CandidateFit(n_points=200, rss=100.0 / k))
        )
    sel = select_peak_count(scores)
    assert [s.k_structured for s in sel.scores] == [1, 2, 3]
    assert sel.best_by["aicc"] in (1, 2, 3)


# ---------------------------------------------------------------- E6


def test_E6_failed_candidate_with_best_ic_is_never_selected():
    """The failed K=3 fit has by far the lowest RSS (would win every IC);
    selection must ignore it, base deltas on successful fits only, and
    say so in the warnings."""
    s1 = score_candidate(model_k(1), CandidateFit(n_points=200, rss=50.0))
    s2 = score_candidate(model_k(2), CandidateFit(n_points=200, rss=30.0))
    s3 = score_candidate(
        model_k(3),
        CandidateFit(
            n_points=200, rss=1e-6, success=False,
            termination_reason="max_iterations",
        ),
    )
    assert s3.aicc < s1.aicc and s3.aicc < s2.aicc  # it WOULD win on IC alone

    sel = select_peak_count([s1, s2, s3])
    assert sel.best_by == {"aic": 2, "aicc": 2, "bic": 2}
    assert sel.excluded_k == (3,)
    assert sel.delta_aicc[2] is None  # failed candidate has no delta
    assert sel.delta_aicc[1] == pytest.approx(0.0)  # best successful is baseline
    assert sel.delta_aicc[0] > 0
    assert any("excluded failed candidates" in w for w in sel.warnings)
    assert any("excluded from selection" in w for w in s3.warnings)


def test_E6_all_failed_yields_no_selection_with_warning():
    s = score_candidate(
        model_k(1),
        CandidateFit(n_points=100, rss=10.0, success=False, termination_reason="diverged"),
    )
    sel = select_peak_count([s])
    assert sel.best_by == {"aic": None, "aicc": None, "bic": None}
    assert sel.delta_aicc == (None,)
    assert any("no selection possible" in w for w in sel.warnings)


# ------------------------------------------------- misc contracts


def test_boundary_flags_emit_regularity_warning():
    s = score_candidate(
        model_k(2),
        CandidateFit(n_points=200, rss=30.0, boundary_flags=("amplitude_0_at_zero",)),
    )
    assert any("regularity" in w for w in s.warnings)


def test_gaussian_scoring_requires_rss_and_poisson_requires_loglik():
    with pytest.raises(ValueError, match="requires fit.rss"):
        score_candidate(model_k(1), CandidateFit(n_points=10))
    with pytest.raises(ValueError, match="requires fit.loglik"):
        score_candidate(model_k(1), CandidateFit(n_points=10, rss=1.0), "poisson")


def test_selection_is_json_serializable():
    scores = [
        score_candidate(model_k(k), CandidateFit(n_points=200, rss=100.0 / k))
        for k in (1, 2)
    ]
    sel = select_peak_count(scores)
    back = json.loads(json.dumps(sel.to_dict()))
    assert set(back) == {
        "scores", "delta_aicc", "delta_bic", "best_by", "excluded_k", "warnings",
    }
    assert back["scores"][0]["noise_model"] == "gaussian"
    assert back["scores"][0]["n_free_parameters"] == 3
