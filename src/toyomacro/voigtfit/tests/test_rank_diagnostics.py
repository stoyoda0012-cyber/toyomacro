"""Gate 1 acceptance tests (M1–M6) for rank_diagnostics.

Test IDs follow benchmarks/README_rank_model_selection.md §4.1. Whitening
convention: Gaussian whitening input is the noise STANDARD DEVIATION
(sigma_std), matching crlb.compute_multipeak_fisher(noise_std=...); the
sigma-vs-variance distinction is asserted explicitly below.
"""

import json
import math

import numpy as np
import pytest

from toyomacro.voigtfit.crlb import compute_multipeak_fisher
from toyomacro.voigtfit.multipeak_config import ComponentConfig
from toyomacro.voigtfit.rank_diagnostics import (
    ParameterScales,
    RankThresholds,
    compute_rank_diagnostics,
    structured_basis,
    structured_jacobian,
    svd_report,
    voigt_fwhm_approx,
    voigt_profile,
    whitener,
)

ENERGY = np.linspace(0.0, 20.0, 401)
SIGMA = 0.40
GAMMA = 0.10
FWHM = voigt_fwhm_approx(SIGMA, GAMMA)  # ≈ 1.15 eV


def comp(center: float, so_split: float = 0.0, branch_ratio: float = 1.0) -> ComponentConfig:
    return ComponentConfig(
        center=center, sigma=SIGMA, gamma=GAMMA,
        so_split=so_split, branch_ratio=branch_ratio,
    )


def diag(components, amplitudes, **kw):
    kw.setdefault("sigma_std", 0.05)
    return compute_rank_diagnostics(ENERGY, components, np.asarray(amplitudes), **kw)


# ---------------------------------------------------------------- M1


def test_M1_identical_columns_linear_rank_1():
    d = diag([comp(10.0), comp(10.0)], [1.0, 1.0])
    assert d.linear.rank_primary == 1
    assert d.linear.rank_reference == 1
    assert d.linear.n_cols == 2
    # exact duplicates: second singular value is numerically zero
    assert d.linear.singular_values[1] / d.linear.singular_values[0] < 1e-12


# ---------------------------------------------------------------- M2


def test_M2_well_separated_K_peaks_linear_rank_K():
    centers = [5.0, 5.0 + 3 * FWHM, 5.0 + 6 * FWHM]
    d = diag([comp(c) for c in centers], [1.0, 1.0, 1.0])
    assert d.linear.rank_primary == 3
    assert d.linear.rank_reference == 3
    # entropy rank close to K for near-orthogonal equal columns
    assert d.linear.entropy_rank == pytest.approx(3.0, abs=0.15)


# ---------------------------------------------------------------- M3


def test_M3_fixed_so_doublet_is_one_structured_column():
    split, br = 1.8, 4.0 / 3.0
    doublet = comp(8.0, so_split=split, branch_ratio=br)
    phi = structured_basis(ENERGY, [doublet])
    assert phi.shape == (ENERGY.size, 1)

    expected = (
        voigt_profile(ENERGY, 8.0, SIGMA, GAMMA)
        + voigt_profile(ENERGY, 8.0 + split, SIGMA, GAMMA) / br
    )
    np.testing.assert_allclose(phi[:, 0], expected, rtol=1e-14)

    d = diag([doublet], [1.0])
    assert d.linear.rank_primary == 1

    # the same two lines as two FREE peaks are rank 2 — the doublet
    # constraint is what removes the second direction
    d_free = diag([comp(8.0), comp(8.0 + split)], [1.0, 1.0 / br])
    assert d_free.linear.rank_primary == 2


# ---------------------------------------------------------------- M4


def test_M4_background_columns_increment_max_linear_rank():
    comps = [comp(5.0), comp(5.0 + 3 * FWHM)]
    amps = [1.0, 1.0]
    r_none = diag(comps, amps).linear.rank_primary
    r_const = diag(comps, amps, bg_degree=0).linear.rank_primary
    r_lin = diag(comps, amps, bg_degree=1).linear.rank_primary
    assert r_none == 2
    assert r_const == 3
    assert r_lin == 4


# ---------------------------------------------------------------- M5


def _crlb_scale_vector(scales: ParameterScales, n_comp: int, mode: str) -> np.ndarray:
    """D diagonal in crlb's interleaved order [amp_k, dE_k, dsigma_k(, dgamma_k)]."""
    per = [scales.amplitude, scales.center, scales.sigma]
    if mode == "4d":
        per.append(scales.gamma)
    return np.array(per * n_comp)


@pytest.mark.parametrize("mode,include_gamma", [("3d", False), ("4d", True)])
def test_M5_scaled_fisher_matches_scaled_whitened_jacobian_sigma_std_whitening(
    mode, include_gamma
):
    """eig(Dᵀ F D) == s²(whitened scaled Jacobian).

    crlb's Fisher is UNSCALED (natural parameter units), so it is scaled
    here as F_scaled = Dᵀ F D before comparison. Both sides use Gaussian
    whitening parameterized by the noise STANDARD DEVIATION (0.05), not
    the variance.
    """
    noise_std = 0.05
    amplitudes = np.array([2.0, 1.2])
    centers = np.array([7.0, 7.0 + 2.5 * FWHM])
    comps = [comp(c) for c in centers]
    scales = ParameterScales.from_representative(
        representative_amplitude=float(np.max(amplitudes)),
        representative_fwhm=FWHM,
    )

    cr = compute_multipeak_fisher(
        amplitudes=amplitudes,
        centers=centers,
        sigmas=np.full(2, SIGMA),
        gammas=np.full(2, GAMMA),
        energy=ENERGY,
        mode=mode,
        noise_model="gaussian",
        noise_std=noise_std,
    )
    d_vec = _crlb_scale_vector(scales, n_comp=2, mode=mode)
    fisher_scaled = d_vec[:, None] * cr.fisher * d_vec[None, :]  # Dᵀ F D (D diagonal)
    eigs = np.sort(np.linalg.eigvalsh(fisher_scaled))[::-1]

    d = compute_rank_diagnostics(
        ENERGY, comps, amplitudes,
        sigma_std=noise_std, scales=scales, include_gamma=include_gamma,
    )
    s2 = np.array(d.local_full.singular_values) ** 2

    assert eigs.shape == s2.shape
    np.testing.assert_allclose(eigs, s2, rtol=1e-8, atol=eigs.max() * 1e-12)


# ---------------------------------------------------------------- M6


def _mev_problem(comps_ev, amps_ev, scales_ev):
    """Same physical problem expressed in meV.

    Unit-area Voigt columns carry 1/energy units, so amplitudes (areas)
    convert as ×1000 alongside centers/widths. Background columns are
    dimensionless normalized monomials, so their coefficient scale is
    unit-invariant (counts) and stays unchanged — as do the y values and
    the noise standard deviation.
    """
    comps_mev = [
        ComponentConfig(
            center=c.center * 1e3, sigma=c.sigma * 1e3, gamma=c.gamma * 1e3,
            so_split=c.so_split * 1e3, branch_ratio=c.branch_ratio,
        )
        for c in comps_ev
    ]
    amps_mev = np.asarray(amps_ev) * 1e3
    scales_mev = ParameterScales(
        amplitude=scales_ev.amplitude * 1e3,
        center=scales_ev.center * 1e3,
        sigma=scales_ev.sigma * 1e3,
        gamma=scales_ev.gamma * 1e3,
        background=scales_ev.background,
        source="M6 unit-converted (eV -> meV)",
    )
    return comps_mev, amps_mev, scales_mev


def test_M6_unit_change_with_matching_scales_leaves_diagnostics_invariant():
    comps_ev = [comp(6.0), comp(6.0 + 2 * FWHM, so_split=1.8, branch_ratio=1.5)]
    amps_ev = np.array([2.0, 1.0])
    scales_ev = ParameterScales.from_representative(2.0, FWHM)
    comps_mev, amps_mev, scales_mev = _mev_problem(comps_ev, amps_ev, scales_ev)

    d_ev = compute_rank_diagnostics(
        ENERGY, comps_ev, amps_ev,
        sigma_std=0.05, scales=scales_ev, bg_degree=1,
    )
    d_mev = compute_rank_diagnostics(
        ENERGY * 1e3, comps_mev, amps_mev,
        sigma_std=0.05, scales=scales_mev, bg_degree=1,
    )

    # all layers are scaled before SVD (design §2.4) → singular values invariant
    np.testing.assert_allclose(
        d_ev.linear.singular_values, d_mev.linear.singular_values, rtol=1e-9
    )
    np.testing.assert_allclose(
        d_ev.local_full.singular_values, d_mev.local_full.singular_values, rtol=1e-9
    )
    np.testing.assert_allclose(
        d_ev.profiled_nonlinear.singular_values,
        d_mev.profiled_nonlinear.singular_values,
        rtol=1e-9,
    )
    assert d_ev.linear.rank_primary == d_mev.linear.rank_primary
    assert d_ev.local_full.rank_primary == d_mev.local_full.rank_primary
    assert d_ev.profiled_nonlinear.rank_primary == d_mev.profiled_nonlinear.rank_primary
    assert d_ev.linear.entropy_rank == pytest.approx(d_mev.linear.entropy_rank, rel=1e-9)
    assert d_ev.linear.condition_number == pytest.approx(
        d_mev.linear.condition_number, rel=1e-9
    )
    # frozen defaults are themselves unit-consistent: the dimensionless
    # background columns keep a unit-invariant coefficient scale
    auto_ev = ParameterScales.from_representative(2.0, FWHM)
    auto_mev = ParameterScales.from_representative(2.0 * 1e3, FWHM * 1e3)
    assert auto_mev.amplitude == pytest.approx(auto_ev.amplitude * 1e3, rel=1e-12)
    assert auto_mev.background == pytest.approx(auto_ev.background, rel=1e-12)


# ------------------------------------------------- whitening conventions


def test_whitening_sigma_std_and_variance_agree_when_variance_is_sigma_squared():
    sigma = 0.05
    w_std, desc_std = whitener(11, sigma_std=sigma)
    w_var, desc_var = whitener(11, variance=sigma**2)
    np.testing.assert_allclose(w_std, w_var, rtol=1e-15)
    assert desc_std["input"] == "sigma_std"
    assert desc_var["input"] == "variance"


def test_whitening_requires_exactly_one_specification():
    with pytest.raises(ValueError, match="exactly one"):
        whitener(11)
    with pytest.raises(ValueError, match="exactly one"):
        whitener(11, sigma_std=0.05, variance=0.0025)


def test_whitening_choice_is_recorded_in_result():
    d = diag([comp(10.0)], [1.0])
    assert d.whitening == {"model": "gaussian", "input": "sigma_std"}

    counts = 100.0 * structured_basis(ENERGY, [comp(10.0)])[:, 0] + 10.0
    d_p = compute_rank_diagnostics(
        ENERGY, [comp(10.0)], np.array([100.0]), expected_counts=counts
    )
    assert d_p.whitening["model"] == "poisson"


# ------------------------------------------------- data-matrix layer


def test_data_matrix_rank_reflects_independent_directions():
    rng = np.random.default_rng(0)
    comps = [comp(5.0), comp(5.0 + 3 * FWHM)]
    phi = structured_basis(ENERGY, comps)
    y = phi @ rng.uniform(0.5, 2.0, size=(2, 6))  # 6 noise-free spectra, 2 directions
    d = diag(comps, [1.0, 1.0], data_matrix=y)
    assert d.data_matrix is not None
    assert d.data_matrix.rank_primary == 2


# ------------------------------------------------- result type / JSON


def test_result_is_json_serializable_and_complete():
    d = diag(
        [comp(6.0), comp(9.0, so_split=1.8, branch_ratio=1.5)],
        [2.0, 1.0],
        bg_degree=1,
    )
    blob = json.dumps(d.to_dict())
    back = json.loads(blob)
    assert set(back) == {
        "linear", "local_full", "profiled_nonlinear", "data_matrix",
        "whitening", "parameter_scales", "thresholds", "column_names", "warnings",
    }
    assert back["column_names"]["linear"] == ["amplitude_0", "amplitude_1", "bg_0", "bg_1"]
    assert back["column_names"]["nonlinear"] == [
        "center_0", "sigma_0", "gamma_0", "center_1", "sigma_1", "gamma_1",
    ]
    assert back["thresholds"] == {"rel_primary": 1e-2, "rel_reference": 1e-3}
    for layer in ("linear", "local_full", "profiled_nonlinear"):
        assert back[layer]["singular_values"] == sorted(
            back[layer]["singular_values"], reverse=True
        )


def test_structured_jacobian_names_and_shapes():
    s, names = structured_jacobian(
        ENERGY, [comp(6.0)], np.array([1.5]), include_gamma=False
    )
    assert s.shape == (ENERGY.size, 2)
    assert names == ["center_0", "sigma_0"]


def test_svd_report_zero_matrix_is_rank_0_with_inf_condition():
    r = svd_report(np.zeros((10, 3)), RankThresholds())
    assert r.rank_primary == 0
    assert r.entropy_rank == 0.0
    assert math.isinf(r.condition_number)


def test_rank_deficient_linear_basis_emits_projector_warning():
    d = diag([comp(10.0), comp(10.0)], [1.0, 1.0])
    assert any("rank-deficient" in w for w in d.warnings)
