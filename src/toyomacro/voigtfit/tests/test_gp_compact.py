"""
Phase-2 correctness tests: compact GP normal equations vs the explicit
projected-Jacobian reference (gp_reference).

Reference: J from gp_reference.reduced_jacobian (J = −M, M the model-side
projected column matrix), so H_explicit = JᵀJ = MᵀM and g_explicit = −Jᵀr.
Compact: H = K − CᵀG⁻¹C (+ QᵀG⁻¹Q), g = Sᵀr from small matrices only.

Battery: 1 comp, 2 comp separated, 2 comp
strongly overlapped, 5 comp, SO doublet, fixed background columns,
non-zero residual, near-zero residual, small amplitude, zero amplitude,
near-rank-deficient, Tikhonov augmented system.

float64 well-conditioned target: relative error ≤ 1e-9 (H, g, step, pred).
Near-rank-deficient: tolerance scales with cond(G) and is reported, not
hard-coded tight.
"""

import numpy as np
import pytest

from toyomacro.voigtfit.gp_compact import (
    compact_normal_equations,
    lm_scale_diag,
    lm_step,
    predicted_reduction,
)
from toyomacro.voigtfit.gp_reference import (
    reduced_jacobian,
    solve_amplitudes_qr,
)
from toyomacro.voigtfit.tests.test_gp_jacobian import make_case, perturb


def build_inputs(model, theta, y):
    """(A, D_list, x, r) at theta with a jitter-free QR amplitude solve."""
    Phi, dPhi = model.basis_and_derivs(theta)
    A, D, y_b = model.augment(Phi, dPhi, y)
    x, _, _ = solve_amplitudes_qr(A, y_b)
    r = y_b - A @ x
    return A, [D[j] for j in range(D.shape[0])], x, r


def rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


CASES = [
    ("1comp", [100.0], [0.6], [1000.0], {}, (0.12, 0.05)),
    ("2comp_sep", [98.0, 102.0], [0.6, 0.7], [800.0, 500.0], {}, (0.1, -0.05)),
    ("2comp_strong", [99.7, 100.3], [0.6, 0.6], [800.0, 500.0], {}, (0.08, 0.04)),
    (
        "5comp",
        [97.0, 98.5, 100.0, 101.5, 103.0],
        [0.5, 0.55, 0.6, 0.55, 0.5],
        [700.0, 300.0, 900.0, 250.0, 500.0],
        {},
        (0.06, 0.03),
    ),
    (
        "so_doublet",
        [100.0, 103.5],
        [0.55, 0.6],
        [900.0, 400.0],
        {"so_split": np.array([0.61, 0.0]),
         "partner_ratio": np.array([0.75, 0.0])},
        (0.1, 0.05),
    ),
    ("near_zero_residual", [98.0, 102.0], [0.6, 0.7], [800.0, 500.0],
     {"noise": 1e-6}, (1e-4, 5e-5)),
    ("small_amplitude", [98.0, 102.0], [0.6, 0.7], [800.0, 4.0], {},
     (0.1, 0.05)),
    ("zero_amplitude", [98.0, 102.0], [0.6, 0.7], [800.0, 0.0], {},
     (0.1, 0.05)),
    ("tikhonov", [98.0, 102.0], [0.6, 0.7], [800.0, 500.0],
     {"tikhonov_lambda": 1e-2}, (0.1, 0.05)),
]


@pytest.mark.parametrize("mode", ["raw", "kaufman", "golub_pereyra"])
@pytest.mark.parametrize(
    "name,centers,sigmas,amps,kw,offset", CASES, ids=[c[0] for c in CASES],
)
def test_compact_matches_explicit(name, centers, sigmas, amps, kw, offset, mode):
    kw = dict(kw)
    kw.setdefault("noise", 3e-3)
    model, theta_true, y = make_case(centers, sigmas, amps, **kw)
    theta = perturb(theta_true, *offset)

    A, D_list, x, r = build_inputs(model, theta, y)
    sys_c = compact_normal_equations(A, D_list, x, r, mode=mode)

    J, info = reduced_jacobian(model, theta, y, mode)
    H_exp = J.T @ J
    g_exp = -J.T @ info.r

    tol = 1e-9 + 1e-15 * sys_c.cond_gram
    assert rel(sys_c.H, H_exp) < tol, (
        f"{name}/{mode}: H rel err {rel(sys_c.H, H_exp):.2e} "
        f"(cond G = {sys_c.cond_gram:.1e})"
    )
    assert rel(sys_c.g, g_exp) < tol

    # Newton step and predicted reduction agree through the same solve
    delta_c = np.linalg.solve(sys_c.H, sys_c.g)
    delta_e = np.linalg.solve(H_exp, g_exp)
    assert rel(delta_c, delta_e) < 1e-6 + 1e-12 * sys_c.cond_gram
    pred_c = predicted_reduction(sys_c.g, sys_c.H, delta_c)
    pred_e = predicted_reduction(g_exp, H_exp, delta_e)
    assert rel(np.atleast_1d(pred_c), np.atleast_1d(pred_e)) < 1e-6

    # LLS orthogonality that the identities rely on (1e-10: when ‖r‖ is at
    # rounding level the normalized metric ‖Aᵀr‖/‖A‖‖r‖ loses ~2 digits)
    assert sys_c.ortho_residual < 1e-10


def test_compact_with_background_columns():
    energy = np.linspace(94.0, 106.0, 241)
    xn = (energy - energy.mean()) / (np.ptp(energy) / 2)
    B = np.stack([np.ones_like(xn), xn], axis=1)
    model, theta_true, y = make_case(
        [98.0, 102.0], [0.6, 0.7], [800.0, 500.0],
        energy=energy, fixed_columns=B, noise=2e-3,
    )
    theta = perturb(theta_true, 0.1, 0.05)
    A, D_list, x, r = build_inputs(model, theta, y)
    for mode in ("kaufman", "golub_pereyra"):
        sys_c = compact_normal_equations(A, D_list, x, r, mode=mode)
        J, info = reduced_jacobian(model, theta, y, mode)
        assert rel(sys_c.H, J.T @ J) < 1e-9
        assert rel(sys_c.g, -J.T @ info.r) < 1e-9
    # background rows of Q are exactly zero (D_j has no bg dependence)
    sys_gp = compact_normal_equations(A, D_list, x, r, mode="golub_pereyra")
    assert np.all(sys_gp.Q[2:, :] == 0.0)


def test_compact_near_rank_deficient_reports_conditioning():
    model, theta_true, y = make_case(
        [100.0, 100.005], [0.6, 0.6], [800.0, 600.0], noise=1e-3,
    )
    theta = perturb(theta_true, 0.002, 0.001)
    A, D_list, x, r = build_inputs(model, theta, y)
    sys_c = compact_normal_equations(A, D_list, x, r, mode="golub_pereyra")
    J, _ = reduced_jacobian(model, theta, y, "golub_pereyra")
    err = rel(sys_c.H, J.T @ J)
    assert sys_c.cond_gram > 1e5, "case is meant to be near-degenerate"
    # tolerance scales with conditioning; report both in the failure message
    assert err < 1e-9 + 1e-14 * sys_c.cond_gram, (
        f"H rel err {err:.2e} at cond(G) {sys_c.cond_gram:.2e}"
    )


def test_gp_minus_kaufman_is_psd():
    """H_GP − H_K = QᵀG⁻¹Q must be PSD (G is SPD)."""
    model, theta_true, y = make_case(
        [99.7, 100.3], [0.6, 0.6], [800.0, 500.0], noise=1e-2,
    )
    theta = perturb(theta_true, 0.08, 0.04)
    A, D_list, x, r = build_inputs(model, theta, y)
    H_k = compact_normal_equations(A, D_list, x, r, mode="kaufman").H
    H_gp = compact_normal_equations(A, D_list, x, r, mode="golub_pereyra").H
    eig = np.linalg.eigvalsh(H_gp - H_k)
    assert eig.min() >= -1e-9 * max(np.abs(eig).max(), 1e-300)


def test_kaufman_equals_gp_at_zero_residual():
    model, theta_true, y = make_case(
        [98.0, 102.0], [0.6, 0.7], [800.0, 500.0], noise=0.0,
    )
    A, D_list, x, r = build_inputs(model, theta_true, y)
    assert np.linalg.norm(r) < 1e-8
    H_k = compact_normal_equations(A, D_list, x, r, mode="kaufman").H
    H_gp = compact_normal_equations(A, D_list, x, r, mode="golub_pereyra").H
    assert rel(H_gp, H_k) < 1e-12


def test_raw_compact_equals_K():
    model, theta_true, y = make_case(
        [98.0, 102.0], [0.6, 0.7], [800.0, 500.0], noise=3e-3,
    )
    theta = perturb(theta_true, 0.1, 0.05)
    A, D_list, x, r = build_inputs(model, theta, y)
    sys_raw = compact_normal_equations(A, D_list, x, r, mode="raw")
    assert np.array_equal(sys_raw.H, 0.5 * (sys_raw.K + sys_raw.K.T))


# ---------------------------------------------------------------------------
# LM primitives
# ---------------------------------------------------------------------------


def test_lm_step_reduces_to_newton_at_zero_lambda():
    model, theta_true, y = make_case(
        [98.0, 102.0], [0.6, 0.7], [800.0, 500.0], noise=3e-3,
    )
    theta = perturb(theta_true, 0.1, 0.05)
    A, D_list, x, r = build_inputs(model, theta, y)
    sys_c = compact_normal_equations(A, D_list, x, r, mode="golub_pereyra")
    d = lm_scale_diag(sys_c.H)
    delta0 = lm_step(sys_c.H, sys_c.g, 0.0, d)
    assert rel(delta0, np.linalg.solve(sys_c.H, sys_c.g)) < 1e-12
    # increasing lambda strictly shrinks the step norm
    norms = [np.linalg.norm(lm_step(sys_c.H, sys_c.g, lam, d))
             for lam in (0.0, 1e-2, 1e0, 1e2)]
    assert all(a > b for a, b in zip(norms, norms[1:]))


def test_predicted_reduction_positive_for_lm_steps():
    """For SPD H and Δ from (H+λD)Δ = g, pred = gᵀΔ − ½ΔᵀHΔ > 0."""
    model, theta_true, y = make_case(
        [99.7, 100.3], [0.6, 0.6], [800.0, 500.0], noise=1e-2,
    )
    theta = perturb(theta_true, 0.08, 0.04)
    A, D_list, x, r = build_inputs(model, theta, y)
    sys_c = compact_normal_equations(A, D_list, x, r, mode="golub_pereyra")
    d = lm_scale_diag(sys_c.H)
    for lam in (0.0, 1e-3, 1e0, 1e3):
        delta = lm_step(sys_c.H, sys_c.g, lam, d)
        assert predicted_reduction(sys_c.g, sys_c.H, delta) > 0.0


def test_predicted_reduction_batched_shapes():
    rng = np.random.default_rng(0)
    p, b = 4, 7
    Jb = rng.standard_normal((b, 30, p))
    H = np.einsum("bec,bef->bcf", Jb, Jb)
    g = rng.standard_normal((b, p))
    d = lm_scale_diag(H)
    assert d.shape == (b, p)
    delta = lm_step(H, g, np.full(b, 1e-2), d)
    assert delta.shape == (b, p)
    pred = predicted_reduction(g, H, delta)
    assert pred.shape == (b,)
    assert np.all(pred > 0)
