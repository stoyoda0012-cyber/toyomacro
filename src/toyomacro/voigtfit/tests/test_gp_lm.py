"""
Phase-3 tests: GP-LM safeguarded refine (frozen surrogate, CPU).

Semantics under test (hardening brief §6):
    * accepted steps never increase the surrogate objective
    * rejected spectra keep the incoming AP seed bit-for-bit
    * the condition gate keeps GP away from ill-conditioned spectra
    * non-finite input spectra are quarantined, never accepted, and do not
      poison the rest of the batch
    * retries are bounded; diagnostics explain every outcome
    * the batched float32 H/g assembly matches the float64 compact reference
"""

import numpy as np
import pytest
from scipy.special import wofz

from toyomacro.voigtfit.gp_compact import (
    GPLMConfig,
    compact_normal_equations,
)
from toyomacro.voigtfit.multipeak_config import ComponentConfig, MultiPeakConfig
from toyomacro.voigtfit.multipeak_solver import (
    _gp_lm_chunk,
    _post_ap_gp_lm_refine_numpy,
    build_multipeak_dictionaries,
    solve_alternating_projection,
)


def _voigt(energy, center, sigma, gamma):
    z = ((energy - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


ENERGY = np.linspace(280.0, 290.0, 121, dtype=np.float32)


def make_dicts():
    config = MultiPeakConfig(
        peaks=[
            ComponentConfig(center=284.0, sigma=0.5, gamma=0.15,
                            dE_range=0.6, ds_range=0.1, n_dE=7, n_ds=9),
            ComponentConfig(center=286.4, sigma=0.5, gamma=0.15,
                            dE_range=0.6, ds_range=0.1, n_dE=7, n_ds=9),
        ],
        energy_axis=ENERGY,
    )
    return build_multipeak_dictionaries(config)


def make_seed(dicts, n, seed=0, noise=0.01):
    """Spectra + AP-like seed state (best_idx at nominal, offsets nearby)."""
    rng = np.random.default_rng(seed)
    n_comp = len(dicts)
    Y = np.empty((n, len(ENERGY)), dtype=np.float32)
    for i in range(n):
        dE = rng.uniform(-0.2, 0.2, n_comp)
        ds = rng.uniform(-0.04, 0.04, n_comp)
        amps = np.array([1000.0, 700.0]) * rng.uniform(0.8, 1.2, n_comp)
        y = sum(
            amps[k] * _voigt(ENERGY, [284.0, 286.4][k] + dE[k],
                             0.5 + ds[k], 0.15)
            for k in range(n_comp)
        )
        Y[i] = (y + noise * y.max() * rng.standard_normal(len(ENERGY))
                ).astype(np.float32)
    best_idx = np.stack(
        [np.full(n, dc.nominal_index, dtype=np.int32) for dc in dicts],
        axis=1,
    )
    delta_E = rng.uniform(-0.05, 0.05, (n, n_comp)).astype(np.float32)
    delta_sigma = rng.uniform(-0.01, 0.01, (n, n_comp)).astype(np.float32)
    amplitudes = rng.uniform(500, 1500, (n, n_comp)).astype(np.float32)
    chi2_in = rng.uniform(1.0, 2.0, n).astype(np.float32)
    return Y, best_idx, delta_E, delta_sigma, amplitudes, chi2_in


def seed_surrogate_chi2(Y, dicts, best_idx, delta_E, delta_sigma):
    """mean(R²) at the seed on the Taylor surrogate with LLS amplitudes —
    the F0 that GP-LM's accept test is measured against."""
    n, n_E = Y.shape
    n_comp = len(dicts)
    A = np.empty((n, n_E, n_comp), dtype=np.float32)
    for k, dc in enumerate(dicts):
        n_ds = dc.grid_shape[1]
        idx = best_idx[:, k]
        dE_off = delta_E[:, k] - dc.dE_grid[idx // n_ds]
        ds_off = delta_sigma[:, k] - dc.dsigma_grid[idx % n_ds]
        A[:, :, k] = (dc.Phi_per_grid[idx, :, 0]
                      + dE_off[:, None] * dc.Jc_per_grid[idx, :, 0]
                      + ds_off[:, None] * dc.Jsigma_per_grid[idx, :, 0])
    AT = A.transpose(0, 2, 1)
    G = AT @ A
    G[:, np.arange(n_comp), np.arange(n_comp)] += np.float32(1e-10)
    x = np.linalg.solve(G, AT @ Y[:, :, None])[:, :, 0]
    R = Y - (A @ x[:, :, None])[:, :, 0]
    return np.mean(R * R, axis=1)


def test_accepted_steps_never_increase_objective():
    dicts = make_dicts()
    Y, bi, dE, ds, amp, chi2_in = make_seed(dicts, 64, seed=1)
    chi2_seed = seed_surrogate_chi2(Y, dicts, bi, dE, ds)

    amp_o, dE_o, ds_o, chi2_o, _bg, diag = _post_ap_gp_lm_refine_numpy(
        Y, dicts, bi, dE, ds, amp, chi2_in, n_iter=2,
    )
    acc = diag.step_accepted
    assert acc.mean() > 0.5, "expected most spectra to accept a GP-LM step"
    # accepted: surrogate objective strictly reduced vs the seed
    assert np.all(chi2_o[acc] < chi2_seed[acc] * (1 + 1e-6))
    assert np.all(diag.actual_reduction[acc] > 0)
    assert np.all(diag.predicted_reduction[acc] > 0)
    assert np.all(np.isfinite(chi2_o))
    # rejected (if any): seed untouched
    rej = ~acc
    assert np.array_equal(amp_o[rej], amp[rej])
    assert np.array_equal(chi2_o[rej], chi2_in[rej])


def test_all_rejected_keeps_seed_bitwise():
    dicts = make_dicts()
    Y, bi, dE, ds, amp, chi2_in = make_seed(dicts, 16, seed=2)
    cfg = GPLMConfig(eta_accept=1e18, max_retries=1)  # rho can never pass
    amp_o, dE_o, ds_o, chi2_o, _bg, diag = _post_ap_gp_lm_refine_numpy(
        Y, dicts, bi, dE, ds, amp, chi2_in, config=cfg,
    )
    assert not diag.step_accepted.any()
    assert np.array_equal(amp_o, amp.astype(np.float32))
    assert np.array_equal(dE_o, dE)
    assert np.array_equal(ds_o, ds)
    assert np.array_equal(chi2_o, chi2_in)
    assert np.all(diag.fallback_reason == 2)  # lm_exhausted
    assert np.all(diag.retry_count <= cfg.max_retries + 1)


def test_condition_gate_blocks_gp():
    dicts = make_dicts()
    Y, bi, dE, ds, amp, chi2_in = make_seed(dicts, 16, seed=3)
    cfg = GPLMConfig(cond_gate=1.0)  # everything is "too ill-conditioned"
    amp_o, dE_o, ds_o, chi2_o, _bg, diag = _post_ap_gp_lm_refine_numpy(
        Y, dicts, bi, dE, ds, amp, chi2_in, config=cfg,
    )
    assert not diag.step_accepted.any()
    assert np.all(diag.fallback_reason == 1)  # cond_gate
    assert np.array_equal(amp_o, amp.astype(np.float32))
    assert np.array_equal(chi2_o, chi2_in)


def test_nonfinite_spectra_quarantined():
    dicts = make_dicts()
    Y, bi, dE, ds, amp, chi2_in = make_seed(dicts, 32, seed=4)
    bad = np.array([3, 17])
    Y[bad, 5] = np.nan
    amp_o, dE_o, ds_o, chi2_o, _bg, diag = _post_ap_gp_lm_refine_numpy(
        Y, dicts, bi, dE, ds, amp, chi2_in,
    )
    assert not diag.step_accepted[bad].any()
    assert np.all(diag.fallback_reason[bad] == 3)  # nonfinite
    # quarantined outputs are the (finite) seeds — no NaN leaks out
    assert np.all(np.isfinite(amp_o))
    assert np.all(np.isfinite(chi2_o))
    assert np.array_equal(amp_o[bad], amp[bad].astype(np.float32))
    # the rest of the batch is unaffected
    good = np.setdiff1d(np.arange(32), bad)
    assert diag.step_accepted[good].mean() > 0.5


def test_batched_H_g_match_compact_reference():
    """First-iteration float32 batched H/g vs per-spectrum float64
    compact_normal_equations on the same gathered surrogate."""
    dicts = make_dicts()
    Y, bi, dE, ds, amp, chi2_in = make_seed(dicts, 6, seed=5)
    dbg = {}
    _gp_lm_chunk(Y, dicts, bi, dE, ds, amp, chi2_in, GPLMConfig(),
                 debug_out=dbg)
    n_comp = len(dicts)
    for b in range(Y.shape[0]):
        A = np.empty((len(ENERGY), n_comp))
        D_list = []
        for k, dc in enumerate(dicts):
            n_ds = dc.grid_shape[1]
            idx = int(bi[b, k])
            dE_off = float(dE[b, k] - dc.dE_grid[idx // n_ds])
            ds_off = float(ds[b, k] - dc.dsigma_grid[idx % n_ds])
            Jc = dc.Jc_per_grid[idx, :, 0].astype(np.float64)
            Jsig = dc.Jsigma_per_grid[idx, :, 0].astype(np.float64)
            A[:, k] = (dc.Phi_per_grid[idx, :, 0]
                       + dE_off * Jc + ds_off * Jsig)
            Dc = np.zeros_like(A)
            Dc[:, k] = Jc
            Dss = np.zeros_like(A)
            Dss[:, k] = Jsig
            D_list.extend([Dc, Dss])
        y = Y[b].astype(np.float64)
        G = A.T @ A + 1e-10 * np.eye(n_comp)
        x = np.linalg.solve(G, A.T @ y)
        r = y - A @ x
        ref = compact_normal_equations(A, D_list, x, r,
                                       mode="golub_pereyra")
        scale_H = np.linalg.norm(ref.H)
        scale_g = np.linalg.norm(ref.g)
        assert np.linalg.norm(dbg["H"][b] - ref.H) < 2e-3 * scale_H
        assert np.linalg.norm(dbg["g"][b] - ref.g) < 2e-3 * scale_g


def test_public_api_gp_lm_end_to_end():
    dicts = make_dicts()
    Y, *_ = make_seed(dicts, 64, seed=6)
    res_ap = solve_alternating_projection(
        Y, dicts, n_iterations=3, apply_newton=False,
    )
    res = solve_alternating_projection(
        Y, dicts, n_iterations=3, apply_newton=True, newton_iter=2,
        newton_jacobian_mode="gp_lm",
    )
    assert res.gp_diagnostics is not None
    d = res.gp_diagnostics
    assert d.step_accepted.mean() > 0.5
    # accepted spectra improved chi2 vs the AP result; rejected kept it
    acc = d.step_accepted
    assert np.all(res.chi2[acc] <= res_ap.chi2[acc] * (1 + 1e-5))
    assert np.allclose(res.chi2[~acc], res_ap.chi2[~acc], rtol=1e-6)
    assert np.all(np.isfinite(res.delta_E))
    assert np.all(np.isfinite(res.amplitudes))
    # raw default result carries no GP diagnostics
    res_raw = solve_alternating_projection(
        Y, dicts, n_iterations=3, apply_newton=True,
    )
    assert res_raw.gp_diagnostics is None


def test_gp_lm_rejects_invalid_combo():
    dicts = make_dicts()
    Y, *_ = make_seed(dicts, 4, seed=7)
    with pytest.raises(ValueError, match="experimental"):
        solve_alternating_projection(
            Y, dicts, apply_newton=True, quality_flags=True,
            newton_jacobian_mode="gp_lm",
        )


# ---------------------------------------------------------------------------
# Phase 4: MLX-hybrid parity
# ---------------------------------------------------------------------------

try:
    import mlx.core  # noqa: F401

    from toyomacro.voigtfit.multipeak_solver import HAS_MLX
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX unavailable")
def test_mlx_chunk_parity_with_numpy():
    """Same inputs through _gp_lm_chunk (CPU) and _gp_lm_chunk_mlx: accept
    decisions and accepted states must agree up to float32 reduction noise.
    Razor-edge accept flips are tolerated up to 2% of the batch."""
    from toyomacro.voigtfit.multipeak_solver import _gp_lm_chunk_mlx

    dicts = make_dicts()
    Y, bi, dE, ds, amp, chi2_in = make_seed(dicts, 256, seed=11)
    cfg = GPLMConfig()
    out_np = _gp_lm_chunk(Y, dicts, bi, dE.copy(), ds.copy(), amp,
                          chi2_in, cfg, n_iter=2)
    out_mx = _gp_lm_chunk_mlx(Y, dicts, bi, dE.copy(), ds.copy(), amp,
                              chi2_in, cfg, n_iter=2)
    acc_np, acc_mx = out_np[5].step_accepted, out_mx[5].step_accepted
    flip = acc_np != acc_mx
    assert flip.mean() <= 0.02, f"accept-mask disagreement {flip.mean():.1%}"
    both = acc_np & acc_mx
    assert both.mean() > 0.5
    for i_field, tol in ((0, 2e-2), (1, 1e-3), (2, 1e-3)):  # amp, dE, ds
        a, b = out_np[i_field][both], out_mx[i_field][both]
        denom = np.maximum(np.abs(a), 1.0 if i_field == 0 else 1e-3)
        assert np.median(np.abs(a - b) / denom) < tol
    # chi2 close on commonly-accepted spectra
    assert np.median(np.abs(out_np[3][both] - out_mx[3][both])
                     / np.maximum(out_np[3][both], 1e-6)) < 1e-2
    # seeds preserved identically on commonly-rejected spectra
    neither = ~acc_np & ~acc_mx
    if neither.any():
        assert np.array_equal(out_np[0][neither], out_mx[0][neither])


@pytest.mark.skipif(not HAS_MLX, reason="MLX unavailable")
def test_public_api_gp_lm_uses_mlx_and_matches_semantics():
    dicts = make_dicts()
    Y, *_ = make_seed(dicts, 128, seed=12)
    res_ap = solve_alternating_projection(
        Y, dicts, n_iterations=3, apply_newton=False,
    )
    res = solve_alternating_projection(
        Y, dicts, n_iterations=3, apply_newton=True, newton_iter=2,
        newton_jacobian_mode="gp_lm",
    )
    d = res.gp_diagnostics
    acc = d.step_accepted
    assert acc.mean() > 0.5
    assert np.all(res.chi2[acc] <= res_ap.chi2[acc] * (1 + 1e-5))
    assert np.allclose(res.chi2[~acc], res_ap.chi2[~acc], rtol=1e-6)


# ---------------------------------------------------------------------------
# Phase 5: fixed polynomial background in the GP-LM projection
# Scope: FIXED linear designs only (build_bg_design monomials). Data- or
# parameter-dependent backgrounds (Shirley/Tougaard) are explicitly out of
# scope for this projection.
# ---------------------------------------------------------------------------

from toyomacro.voigtfit.multipeak_solver import build_bg_design  # noqa: E402


def make_seed_with_bg(dicts, n, seed=0, noise=0.01, bg=(120.0, 40.0)):
    """Seed spectra + a linear background bg[0] + bg[1]*x, x∈[-1,1]."""
    Y, bi, dE, ds, amp, chi2_in = make_seed(dicts, n, seed=seed, noise=noise)
    x = (ENERGY - ENERGY.mean()) / (np.ptp(ENERGY) / 2)
    Y = Y + (bg[0] + bg[1] * x).astype(np.float32)[None, :]
    return Y, bi, dE, ds, amp, chi2_in


@pytest.mark.parametrize("bg_degree", [0, 1, 2])
def test_gp_lm_background_end_to_end(bg_degree):
    dicts = make_dicts()
    Y, *_ = make_seed_with_bg(make_dicts(), 64, seed=21)
    res_ap = solve_alternating_projection(
        Y, dicts, n_iterations=3, apply_newton=False, bg_degree=bg_degree,
    )
    res = solve_alternating_projection(
        Y, dicts, n_iterations=3, apply_newton=True, newton_iter=2,
        bg_degree=bg_degree, newton_jacobian_mode="gp_lm",
    )
    d = res.gp_diagnostics
    acc = d.step_accepted
    assert acc.mean() > 0.5
    assert np.all(np.isfinite(res.background))
    assert res.background.shape == (64, bg_degree + 1)
    assert np.all(res.chi2[acc] <= res_ap.chi2[acc] * (1 + 1e-5))
    # rejected spectra keep the Phase-2 background untouched
    if (~acc).any():
        assert np.allclose(res.background[~acc], res_ap.background[~acc],
                           rtol=1e-5, atol=1e-3)
    # reconstructed background recovers the injected offset (~120 mean over
    # the window) — basis-independent check: with degree>=2 the constant
    # coefficient legitimately trades against the x**2 column.
    B = build_bg_design(ENERGY, bg_degree).astype(np.float64)
    bg_mean = (res.background[acc].astype(np.float64) @ B.mean(axis=0))
    assert np.median(np.abs(bg_mean - 120.0)) < 25.0


def test_gp_lm_bg_chunk_matches_dense_reference():
    """Batched float32 H/g WITH background columns vs per-spectrum float64
    compact reference on the augmented basis [Phi_hat | B]."""
    dicts = make_dicts()
    Y, bi, dE, ds, amp, chi2_in = make_seed_with_bg(dicts, 5, seed=22)
    B = build_bg_design(ENERGY, 1).astype(np.float64)
    dbg = {}
    _gp_lm_chunk(Y, dicts, bi, dE, ds, amp, chi2_in, GPLMConfig(),
                 bg_design=B.astype(np.float32), debug_out=dbg)
    n_comp = len(dicts)
    n_bg = B.shape[1]
    n_lin = n_comp + n_bg
    for b in range(Y.shape[0]):
        A = np.empty((len(ENERGY), n_lin))
        D_list = []
        for k, dc in enumerate(dicts):
            n_ds = dc.grid_shape[1]
            idx = int(bi[b, k])
            dE_off = float(dE[b, k] - dc.dE_grid[idx // n_ds])
            ds_off = float(ds[b, k] - dc.dsigma_grid[idx % n_ds])
            Jc = dc.Jc_per_grid[idx, :, 0].astype(np.float64)
            Jsig = dc.Jsigma_per_grid[idx, :, 0].astype(np.float64)
            A[:, k] = (dc.Phi_per_grid[idx, :, 0]
                       + dE_off * Jc + ds_off * Jsig)
            Dc = np.zeros_like(A)
            Dc[:, k] = Jc
            Dss = np.zeros_like(A)
            Dss[:, k] = Jsig
            D_list.extend([Dc, Dss])
        A[:, n_comp:] = B
        y = Y[b].astype(np.float64)
        G = A.T @ A + 1e-10 * np.eye(n_lin)
        x = np.linalg.solve(G, A.T @ y)
        r = y - A @ x
        ref = compact_normal_equations(A, D_list, x, r,
                                       mode="golub_pereyra")
        assert np.linalg.norm(dbg["H"][b] - ref.H) < 2e-3 * np.linalg.norm(ref.H)
        assert np.linalg.norm(dbg["g"][b] - ref.g) < 2e-3 * np.linalg.norm(ref.g)


def test_gp_lm_bg_strong_peak_bg_correlation():
    """Broad peaks spanning the window correlate strongly with the linear
    background; GP-LM must stay finite and only accept improving steps."""
    config = MultiPeakConfig(
        peaks=[
            ComponentConfig(center=284.4, sigma=1.6, gamma=0.3,
                            dE_range=0.6, ds_range=0.2, n_dE=7, n_ds=9),
            ComponentConfig(center=286.2, sigma=1.6, gamma=0.3,
                            dE_range=0.6, ds_range=0.2, n_dE=7, n_ds=9),
        ],
        energy_axis=ENERGY,
    )
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # separation warning is expected
        dicts_wide = build_multipeak_dictionaries(config)
    rng = np.random.default_rng(23)
    x = (ENERGY - ENERGY.mean()) / (np.ptp(ENERGY) / 2)
    n = 32
    Y = np.empty((n, len(ENERGY)), dtype=np.float32)
    for i in range(n):
        y = (900.0 * _voigt(ENERGY, 284.4 + rng.uniform(-0.2, 0.2),
                            1.6 + rng.uniform(-0.1, 0.1), 0.3)
             + 700.0 * _voigt(ENERGY, 286.2 + rng.uniform(-0.2, 0.2),
                              1.6 + rng.uniform(-0.1, 0.1), 0.3)
             + 150.0 + 60.0 * x)
        Y[i] = (y + 0.01 * y.max() * rng.standard_normal(len(ENERGY))
                ).astype(np.float32)
    res_ap = solve_alternating_projection(
        Y, dicts_wide, n_iterations=3, apply_newton=False, bg_degree=1,
    )
    res = solve_alternating_projection(
        Y, dicts_wide, n_iterations=3, apply_newton=True, newton_iter=2,
        bg_degree=1, newton_jacobian_mode="gp_lm",
    )
    d = res.gp_diagnostics
    assert np.all(np.isfinite(res.chi2))
    assert np.all(np.isfinite(res.background))
    acc = d.step_accepted
    assert np.all(res.chi2[acc] <= res_ap.chi2[acc] * (1 + 1e-5))
    # high peak-bg correlation shows up in the Gram conditioning
    assert np.median(d.cond_gram) > np.float32(50.0)


# ---------------------------------------------------------------------------
# Phase 6: diagnostics contract
# ---------------------------------------------------------------------------

from toyomacro.voigtfit.gp_compact import FALLBACK_LEGEND  # noqa: E402


def test_diagnostics_contract():
    """Every spectrum's outcome is explained: step_accepted <=>
    fallback_reason == 0, codes stay within the legend, shapes line up,
    and the numbers quoted are finite where they must be."""
    dicts = make_dicts()
    Y, bi, dE, ds, amp, chi2_in = make_seed(dicts, 48, seed=31)
    Y[7, 3] = np.inf  # one quarantined spectrum
    _o = _post_ap_gp_lm_refine_numpy(
        Y, dicts, bi, dE, ds, amp, chi2_in, n_iter=2,
        chunk_size=20,  # force multi-chunk aggregation
    )
    d = _o[5]
    n = 48
    for f in ("cond_gram", "cond_hessian", "lm_lambda", "predicted_reduction",
              "actual_reduction", "reduction_ratio", "step_accepted",
              "retry_count", "step_clipped", "min_amplitude",
              "fallback_reason"):
        assert getattr(d, f).shape == (n,), f
    assert set(np.unique(d.fallback_reason)).issubset(set(FALLBACK_LEGEND))
    np.testing.assert_array_equal(d.step_accepted, d.fallback_reason == 0)
    acc = d.step_accepted
    assert np.all(np.isfinite(d.predicted_reduction[acc]))
    assert np.all(np.isfinite(d.actual_reduction[acc]))
    assert np.all(d.reduction_ratio[acc] > 0)
    assert np.all(np.isfinite(d.cond_gram[acc]))
    assert 0.0 <= d.negative_amplitude_fraction <= 1.0
    assert d.n_basis_evals > 0
