"""
Finite-difference correctness tests for the Golub-Pereyra reference
implementation (gp_reference.py).

The ground truth is the central-difference Jacobian of the FULL reduced
residual r(theta) = y - Phi(theta) a*(theta), with a complete amplitude
re-solve on both sides of every difference — no dictionary argmax, no step
clipping, no frozen state enters the differencing.

Covers:
    * 1 peak: center and sigma columns
    * 2 peaks: all four columns, well-separated and strongly overlapped
    * spin-orbit doublet structured column
    * 5-component small case
    * non-zero residual and near-solution (small residual)
    * near-rank-deficient overlap (documented looser tolerance)
    * fixed background columns in the projection
    * Tikhonov as an explicitly augmented system
    * step-size refinement of the central difference
    * gradient identity J^T r across raw / kaufman / golub_pereyra
    * Kaufman-vs-GP gap linear in the residual scale
    * raw-mode GN step == production normal-equation step
"""

import numpy as np
import pytest

from toyomacro.voigtfit.gp_reference import (
    JACOBIAN_MODES,
    SeparableVoigtModel,
    fd_jacobian,
    gauss_newton_step,
    reduced_jacobian,
    reduced_residual,
)

RNG = np.random.default_rng


def rel_fro(A, B):
    return np.linalg.norm(A - B) / max(np.linalg.norm(B), 1e-300)


def make_case(
    centers,
    sigmas,
    amps,
    gamma=0.25,
    energy=None,
    so_split=None,
    partner_ratio=None,
    fixed_columns=None,
    tikhonov_lambda=0.0,
    noise=0.0,
    seed=0,
):
    """Build (model, theta_true, y) with y = Phi(theta_true) a + noise."""
    centers = np.asarray(centers, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    amps = np.asarray(amps, dtype=np.float64)
    if energy is None:
        lo = centers.min() - 4.0
        hi = centers.max() + 4.0
        energy = np.linspace(lo, hi, 241)
    model = SeparableVoigtModel(
        energy=energy,
        gamma=gamma,
        so_split=so_split,
        partner_ratio=partner_ratio,
        fixed_columns=fixed_columns,
        tikhonov_lambda=tikhonov_lambda,
    )
    theta_true = np.empty(2 * len(centers))
    theta_true[0::2] = centers
    theta_true[1::2] = sigmas
    Phi, _ = model.basis_and_derivs(theta_true)
    y = Phi[:, : len(amps)] @ amps
    if fixed_columns is not None:
        # a mild fixed-column contribution so the bg amplitudes are non-zero
        y = y + fixed_columns @ np.full(fixed_columns.shape[1], 0.1 * amps.max())
    if noise > 0:
        y = y + noise * amps.max() * RNG(seed).standard_normal(len(y))
    return model, theta_true, y


def perturb(theta, dc, ds):
    """Shift every center by +-dc and every sigma by +-ds (alternating)."""
    out = np.asarray(theta, dtype=np.float64).copy()
    sign = 1.0
    for k in range(len(out) // 2):
        out[2 * k] += sign * dc
        out[2 * k + 1] += sign * ds
        sign = -sign
    return out


# ---------------------------------------------------------------------------
# GP vs central difference — the primary correctness criterion
# ---------------------------------------------------------------------------

GP_FD_CASES = [
    # (name, centers, sigmas, amps, extra kwargs, eval offset (dc, ds))
    ("1peak_center_offgrid", [100.0], [0.6], [1000.0], {}, (0.15, 0.0)),
    ("1peak_sigma_offgrid", [100.0], [0.6], [1000.0], {}, (0.0, 0.08)),
    ("2peak_separated", [98.0, 102.0], [0.6, 0.7], [800.0, 500.0], {}, (0.1, -0.05)),
    ("2peak_overlapped", [99.7, 100.3], [0.6, 0.6], [800.0, 500.0], {}, (0.08, 0.04)),
    (
        "so_doublet",
        [100.0, 103.5],
        [0.55, 0.6],
        [900.0, 400.0],
        {"so_split": np.array([0.61, 0.0]), "partner_ratio": np.array([0.75, 0.0])},
        (0.1, 0.05),
    ),
    (
        "5comp",
        [97.0, 98.5, 100.0, 101.5, 103.0],
        [0.5, 0.55, 0.6, 0.55, 0.5],
        [700.0, 300.0, 900.0, 250.0, 500.0],
        {},
        (0.06, 0.03),
    ),
    (
        "near_solution_small_residual",
        [98.0, 102.0],
        [0.6, 0.7],
        [800.0, 500.0],
        {"noise": 1e-4},
        (1e-3, 5e-4),
    ),
]


@pytest.mark.parametrize(
    "name,centers,sigmas,amps,kw,offset",
    GP_FD_CASES,
    ids=[c[0] for c in GP_FD_CASES],
)
def test_gp_matches_central_difference(name, centers, sigmas, amps, kw, offset):
    kw = dict(kw)
    kw.setdefault("noise", 3e-3)  # non-zero residual by default
    model, theta_true, y = make_case(centers, sigmas, amps, **kw)
    theta = perturb(theta_true, *offset)

    J_gp, info = reduced_jacobian(model, theta, y, "golub_pereyra")
    J_fd = fd_jacobian(model, theta, y)

    err = rel_fro(J_gp, J_fd)
    assert err < 1e-5, (
        f"{name}: GP vs FD rel error {err:.2e} (cond(Gram)={info.cond_gram:.1e})"
    )


def test_gp_matches_fd_near_rank_deficient():
    """0.008-sigma separation: Gram nearly singular (cond ~ 1e5). The GP
    formula still has to track the FD derivative, but ill-conditioning
    amplifies both the FD cancellation noise and the sensitivity of
    a*(theta), so the acceptance threshold is looser and the conditioning
    is asserted explicitly."""
    model, theta_true, y = make_case(
        [100.0, 100.005], [0.6, 0.6], [800.0, 600.0], noise=1e-3,
    )
    theta = perturb(theta_true, 0.002, 0.001)
    J_gp, info = reduced_jacobian(model, theta, y, "golub_pereyra")
    J_fd = fd_jacobian(model, theta, y)
    assert info.cond_gram > 1e5, "case is meant to be near-degenerate"
    assert rel_fro(J_gp, J_fd) < 1e-3


def test_gp_matches_fd_with_background_columns():
    energy = np.linspace(94.0, 106.0, 241)
    x = (energy - energy.mean()) / (np.ptp(energy) / 2)
    B = np.stack([np.ones_like(x), x], axis=1)  # constant + linear bg
    model, theta_true, y = make_case(
        [98.0, 102.0], [0.6, 0.7], [800.0, 500.0],
        energy=energy, fixed_columns=B, noise=2e-3,
    )
    theta = perturb(theta_true, 0.1, 0.05)
    J_gp, _ = reduced_jacobian(model, theta, y, "golub_pereyra")
    J_fd = fd_jacobian(model, theta, y)
    assert rel_fro(J_gp, J_fd) < 1e-5


def test_gp_matches_fd_tikhonov_augmented():
    model, theta_true, y = make_case(
        [98.0, 102.0], [0.6, 0.7], [800.0, 500.0],
        tikhonov_lambda=1e-2, noise=2e-3,
    )
    theta = perturb(theta_true, 0.1, 0.05)
    J_gp, _ = reduced_jacobian(model, theta, y, "golub_pereyra")
    J_fd = fd_jacobian(model, theta, y)
    assert rel_fro(J_gp, J_fd) < 1e-5


def test_central_difference_h_refinement():
    """In the truncation-dominated regime the central-difference error is
    O(h^2): shrinking h by 10x must shrink the GP-vs-FD discrepancy by
    roughly 100x (>= 20x asserted to absorb higher-order terms)."""
    model, theta_true, y = make_case(
        [98.0, 102.0], [0.6, 0.7], [800.0, 500.0], noise=3e-3,
    )
    theta = perturb(theta_true, 0.1, 0.05)
    J_gp, _ = reduced_jacobian(model, theta, y, "golub_pereyra")

    sigma_scale = 0.6
    errs = []
    for h_rel in (3e-2, 3e-3):
        J_fd = fd_jacobian(model, theta, y, h=h_rel * sigma_scale)
        errs.append(rel_fro(J_gp, J_fd))
    assert errs[1] < errs[0] / 20.0, f"no O(h^2) decay: {errs}"


# ---------------------------------------------------------------------------
# Structural identities
# ---------------------------------------------------------------------------


def test_gradient_identical_across_modes():
    """Phi^T r = 0 at the LLS optimum, hence J^T r is the same for raw,
    Kaufman and full GP — the modes differ only in curvature."""
    model, theta_true, y = make_case(
        [98.0, 102.0], [0.6, 0.7], [800.0, 500.0], noise=5e-3,
    )
    theta = perturb(theta_true, 0.12, 0.06)
    grads = {}
    for mode in JACOBIAN_MODES:
        J, info = reduced_jacobian(model, theta, y, mode)
        grads[mode] = J.T @ info.r
    g_ref = grads["golub_pereyra"]
    scale = np.linalg.norm(g_ref)
    for mode in ("raw", "kaufman"):
        assert np.linalg.norm(grads[mode] - g_ref) < 1e-8 * scale, mode


def test_kaufman_equals_gp_at_zero_residual():
    model, theta_true, y = make_case(
        [98.0, 102.0], [0.6, 0.7], [800.0, 500.0], noise=0.0,
    )
    J_k, _ = reduced_jacobian(model, theta_true, y, "kaufman")
    J_gp, info = reduced_jacobian(model, theta_true, y, "golub_pereyra")
    assert np.linalg.norm(info.r) < 1e-6  # noiseless at truth
    assert np.linalg.norm(J_gp - J_k) < 1e-9 * np.linalg.norm(J_k)


def test_kaufman_gp_gap_linear_in_residual():
    """The GP second term is linear in r: scaling the noise by 10x must
    scale ||J_gp - J_kaufman|| by exactly 10x (same noise realization)."""
    centers, sigmas, amps = [98.0, 102.0], [0.6, 0.7], [800.0, 500.0]
    model, theta_true, _ = make_case(centers, sigmas, amps, noise=0.0)
    Phi, _ = model.basis_and_derivs(theta_true)
    y_clean = Phi @ np.asarray(amps)
    n = RNG(7).standard_normal(len(y_clean))

    gaps = []
    for eps in (1e-3, 1e-2):
        y = y_clean + eps * n
        J_k, _ = reduced_jacobian(model, theta_true, y, "kaufman")
        J_gp, _ = reduced_jacobian(model, theta_true, y, "golub_pereyra")
        gaps.append(np.linalg.norm(J_gp - J_k))
    assert gaps[1] == pytest.approx(10.0 * gaps[0], rel=1e-9)


def test_raw_step_equals_production_normal_equations():
    """gauss_newton_step('raw') must reproduce the production convention:
    solve (S^T S) delta = S^T r with S the model-side Jacobian, r = y - model,
    then theta + delta (before Tikhonov jitter and clipping)."""
    model, theta_true, y = make_case(
        [98.0, 102.0], [0.6, 0.7], [800.0, 500.0], noise=3e-3,
    )
    theta = perturb(theta_true, 0.1, -0.04)

    delta, _, info = gauss_newton_step(model, theta, y, "raw")

    Phi, dPhi = model.basis_and_derivs(theta)
    S = np.stack([dPhi[j] @ info.a for j in range(dPhi.shape[0])], axis=1)
    r = y - Phi @ info.a
    delta_prod = np.linalg.solve(S.T @ S, S.T @ r)

    np.testing.assert_allclose(delta, delta_prod, rtol=1e-8, atol=1e-12)


def test_reduced_residual_orthogonality():
    """Phi^T r = 0 to machine precision (jitter-free LLS via QR)."""
    model, theta_true, y = make_case(
        [98.0, 102.0], [0.6, 0.7], [800.0, 500.0], noise=1e-2,
    )
    theta = perturb(theta_true, 0.1, 0.05)
    Phi, dPhi = model.basis_and_derivs(theta)
    r, a = reduced_residual(model, theta, y)
    assert np.linalg.norm(Phi.T @ r) < 1e-9 * np.linalg.norm(y)


# ---------------------------------------------------------------------------
# Production hook: newton_jacobian_mode in multipeak_solver
# ---------------------------------------------------------------------------

from scipy.special import wofz  # noqa: E402

from toyomacro.voigtfit.multipeak_config import (  # noqa: E402
    ComponentConfig,
    MultiPeakConfig,
)
from toyomacro.voigtfit.multipeak_solver import (  # noqa: E402
    _post_ap_newton_refine_numpy,
    build_multipeak_dictionaries,
    solve_alternating_projection,
)


def _voigt(energy, center, sigma, gamma):
    z = ((energy - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


ENERGY_2C = np.linspace(280.0, 290.0, 121, dtype=np.float32)


def _dicts_2comp():
    config = MultiPeakConfig(
        peaks=[
            ComponentConfig(center=284.0, sigma=0.5, gamma=0.15,
                            dE_range=0.6, ds_range=0.1, n_dE=7, n_ds=9),
            ComponentConfig(center=286.4, sigma=0.5, gamma=0.15,
                            dE_range=0.6, ds_range=0.1, n_dE=7, n_ds=9),
        ],
        energy_axis=ENERGY_2C,
    )
    return build_multipeak_dictionaries(config)


def _spectra_2comp(n, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    Y = np.empty((n, len(ENERGY_2C)), dtype=np.float32)
    truth = []
    for i in range(n):
        dE = rng.uniform(-0.25, 0.25, size=2)
        ds = rng.uniform(-0.04, 0.04, size=2)
        amps = np.array([1000.0, 700.0]) * rng.uniform(0.8, 1.2, size=2)
        y = (amps[0] * _voigt(ENERGY_2C, 284.0 + dE[0], 0.5 + ds[0], 0.15)
             + amps[1] * _voigt(ENERGY_2C, 286.4 + dE[1], 0.5 + ds[1], 0.15))
        y = y + noise * y.max() * rng.standard_normal(len(ENERGY_2C))
        Y[i] = y.astype(np.float32)
        truth.append((dE, ds, amps))
    return Y, truth


class TestProductionHook:
    def test_raw_mode_is_default_path_identical(self):
        dicts = _dicts_2comp()
        Y, _ = _spectra_2comp(64, seed=1)
        kw = dict(n_iterations=3, apply_newton=True, newton_iter=1)
        res_default = solve_alternating_projection(Y, dicts, **kw)
        res_raw = solve_alternating_projection(
            Y, dicts, newton_jacobian_mode="raw", **kw,
        )
        np.testing.assert_array_equal(res_default.delta_E, res_raw.delta_E)
        np.testing.assert_array_equal(
            res_default.delta_sigma, res_raw.delta_sigma,
        )
        np.testing.assert_array_equal(
            res_default.amplitudes, res_raw.amplitudes,
        )
        np.testing.assert_array_equal(res_default.chi2, res_raw.chi2)

    @pytest.mark.parametrize("mode", ["kaufman", "golub_pereyra"])
    def test_experimental_modes_run_and_do_not_degrade_chi2(self, mode):
        dicts = _dicts_2comp()
        Y, _ = _spectra_2comp(64, seed=2)
        res_ap = solve_alternating_projection(
            Y, dicts, n_iterations=3, apply_newton=False,
        )
        res = solve_alternating_projection(
            Y, dicts, n_iterations=3, apply_newton=True, newton_iter=2,
            newton_jacobian_mode=mode,
        )
        assert np.all(np.isfinite(res.delta_E))
        assert np.all(np.isfinite(res.delta_sigma))
        assert np.all(np.isfinite(res.amplitudes))
        assert np.median(res.chi2) <= 1.05 * np.median(res_ap.chi2)

    def test_invalid_mode_raises(self):
        dicts = _dicts_2comp()
        Y, _ = _spectra_2comp(4, seed=3)
        with pytest.raises(ValueError, match="newton_jacobian_mode"):
            solve_alternating_projection(
                Y, dicts, apply_newton=True, newton_jacobian_mode="nnls",
            )

    def test_incompatible_options_raise(self):
        dicts = _dicts_2comp()
        Y, _ = _spectra_2comp(4, seed=3)
        with pytest.raises(ValueError, match="experimental"):
            solve_alternating_projection(
                Y, dicts, apply_newton=True, newton_exact=True,
                newton_jacobian_mode="golub_pereyra",
            )
        with pytest.raises(ValueError, match="experimental"):
            solve_alternating_projection(
                Y, dicts, apply_newton=True, quality_flags=True,
                newton_jacobian_mode="kaufman",
            )

    def test_kaufman_equals_gp_at_zero_surrogate_residual(self):
        """Y built EXACTLY from the Taylor-surrogate bases: after the
        internal amplitude re-solve the residual is ~0, so the GP second
        term vanishes and kaufman / golub_pereyra must coincide (up to
        float32 solve noise)."""
        dicts = _dicts_2comp()
        rng = np.random.default_rng(4)
        batch = 16
        best_idx = np.stack(
            [np.full(batch, dc.nominal_index, dtype=np.int32) for dc in dicts],
            axis=1,
        )
        n_ds = dicts[0].grid_shape[1]
        amps = np.stack([rng.uniform(800, 1200, batch),
                         rng.uniform(500, 900, batch)], axis=1)
        delta_E = np.empty((batch, 2), dtype=np.float32)
        delta_sigma = np.empty((batch, 2), dtype=np.float32)
        Y = np.zeros((batch, len(ENERGY_2C)), dtype=np.float32)
        for k, dc in enumerate(dicts):
            idx = best_idx[:, k]
            dE_g = dc.dE_grid[idx // n_ds]
            ds_g = dc.dsigma_grid[idx % n_ds]
            dE_off = rng.uniform(-0.03, 0.03, batch).astype(np.float32)
            ds_off = rng.uniform(-0.005, 0.005, batch).astype(np.float32)
            delta_E[:, k] = dE_g + dE_off
            delta_sigma[:, k] = ds_g + ds_off
            Phi_hat = (dc.Phi_per_grid[idx, :, 0]
                       + dE_off[:, None] * dc.Jc_per_grid[idx, :, 0]
                       + ds_off[:, None] * dc.Jsigma_per_grid[idx, :, 0])
            Y += amps[:, k:k + 1].astype(np.float32) * Phi_hat

        out = {}
        for mode in ("kaufman", "golub_pereyra"):
            out[mode] = _post_ap_newton_refine_numpy(
                Y, dicts, best_idx, delta_E.copy(), delta_sigma.copy(),
                amps.astype(np.float32), jacobian_mode=mode,
            )
        for a, b in zip(out["kaufman"], out["golub_pereyra"]):
            np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-5)

    def test_chunk_matches_dense_float64_reference(self):
        """The float32 batched chunk implementation of the projected
        Jacobian step must match a per-spectrum float64 reimplementation
        with an EXPLICIT projector (small batch, golub_pereyra)."""
        dicts = _dicts_2comp()
        Y, _ = _spectra_2comp(6, seed=5)
        batch = Y.shape[0]
        n_comp = 2
        rng = np.random.default_rng(6)
        n_ds = dicts[0].grid_shape[1]
        best_idx = np.stack(
            [rng.integers(0, dicts[k].n_dict, batch).astype(np.int32)
             for k in range(n_comp)], axis=1,
        )
        delta_E = np.empty((batch, n_comp), dtype=np.float32)
        delta_sigma = np.empty((batch, n_comp), dtype=np.float32)
        for k, dc in enumerate(dicts):
            idx = best_idx[:, k]
            delta_E[:, k] = (dc.dE_grid[idx // n_ds]
                             + rng.uniform(-0.04, 0.04, batch))
            delta_sigma[:, k] = (dc.dsigma_grid[idx % n_ds]
                                 + rng.uniform(-0.008, 0.008, batch))
        amps0 = np.full((batch, n_comp), 500.0, dtype=np.float32)  # ignored

        amp_c, dE_c, ds_c, chi2_c = _post_ap_newton_refine_numpy(
            Y, dicts, best_idx, delta_E.copy(), delta_sigma.copy(), amps0,
            jacobian_mode="golub_pereyra",
        )

        # dense float64 reference, explicit projector
        clip_ratio = 0.5
        for b in range(batch):
            Phi = np.empty((len(ENERGY_2C), n_comp))
            Jc = np.empty_like(Phi)
            Js = np.empty_like(Phi)
            steps = np.empty((n_comp, 2))
            for k, dc in enumerate(dicts):
                idx = int(best_idx[b, k])
                dE_off = float(delta_E[b, k] - dc.dE_grid[idx // n_ds])
                ds_off = float(delta_sigma[b, k] - dc.dsigma_grid[idx % n_ds])
                Jc[:, k] = dc.Jc_per_grid[idx, :, 0]
                Js[:, k] = dc.Jsigma_per_grid[idx, :, 0]
                Phi[:, k] = (dc.Phi_per_grid[idx, :, 0]
                             + dE_off * Jc[:, k] + ds_off * Js[:, k])
                steps[k] = (dc.dE_grid[1] - dc.dE_grid[0],
                            dc.dsigma_grid[1] - dc.dsigma_grid[0])
            y = Y[b].astype(np.float64)
            G = Phi.T @ Phi + 1e-10 * np.eye(n_comp)
            a = np.linalg.solve(G, Phi.T @ y)
            r = y - Phi @ a
            P = Phi @ np.linalg.solve(G, Phi.T)
            M = np.empty((len(y), 2 * n_comp))
            for p in range(2 * n_comp):
                k, axis = divmod(p, 2)
                Jcol = Jc[:, k] if axis == 0 else Js[:, k]
                S = a[k] * Jcol
                dPhi = np.zeros_like(Phi)
                dPhi[:, k] = Jcol
                T2 = Phi @ np.linalg.solve(G, dPhi.T @ r)
                M[:, p] = S - P @ S + T2
            H = M.T @ M
            H += 1e-6 * np.trace(H) * np.eye(2 * n_comp)
            delta = np.linalg.solve(H, M.T @ r)
            for k in range(n_comp):
                delta[2 * k] = np.clip(delta[2 * k],
                                       -clip_ratio * steps[k, 0],
                                       clip_ratio * steps[k, 0])
                delta[2 * k + 1] = np.clip(delta[2 * k + 1],
                                           -clip_ratio * steps[k, 1],
                                           clip_ratio * steps[k, 1])
            dE_ref = delta_E[b] + delta[0::2]
            ds_ref = delta_sigma[b] + delta[1::2]
            np.testing.assert_allclose(dE_c[b], dE_ref, atol=2e-3)
            np.testing.assert_allclose(ds_c[b], ds_ref, atol=2e-3)
