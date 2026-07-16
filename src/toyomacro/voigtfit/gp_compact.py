"""
Compact Golub-Pereyra normal equations + LM safeguard (reference layer).
========================================================================

Builds the projected (Kaufman / full Golub-Pereyra) Gauss-Newton system
WITHOUT materialising the projected Jacobian columns, from small matrices
only:

    A ∈ R^{n_rows × n_L}   linear basis (peaks [| background])
    x* = argmin_x ||y - A x||²,     r = y - A x*
    D_j = ∂A/∂θ_j,   S_j = D_j x*,  S = [S_1 … S_p]

    G = AᵀA,  C = AᵀS,  K = SᵀS,  Q[:,j] = D_jᵀ r

    H_raw     = K                        (production model-side curvature)
    H_kaufman = K − Cᵀ G⁻¹ C
    H_gp      = K − Cᵀ G⁻¹ C + Qᵀ G⁻¹ Q
    g         = Sᵀ r                     (identical for all three modes
                                          when Aᵀr = 0 at the LLS optimum)

The GP identity uses the exact orthogonality of the two Jacobian components
(I−P_A)S_j ⊥ A G⁻¹ Q_{:,j}: H_gp = MᵀM with zero cross terms, where
M_j = (I−P_A)S_j + A G⁻¹ Q_{:,j} is the model-side projected column.

Sign system matches production: solve H·Δ = g, θ ← θ + Δ.

Numerical policy
----------------
* No explicit inverse and no explicit projector anywhere: one multi-RHS
  Gram solve `solve(G, [C | Q])` builds both correction terms.
* The DATA Hessian (returned) carries no stabilisation; LM adds its load
  separately at solve time (`lm_step`). Gram jitter, if any, is the
  caller's numerical-stabilisation choice and is reported, never hidden.
* `H ← (H + Hᵀ)/2` at the end guards against asymmetry from float error.
* Tikhonov-regularised objectives must be passed as the explicitly
  augmented system ([A; √λ I], [y; 0]) — same convention as gp_reference.

LM safeguard (Phase 3)
----------------------
Scaled-diagonal Levenberg-Marquardt on the reduced objective
F(θ) = ½‖r(θ)‖²:

    (H + λ·diag(d)) Δ = g,   d_i = max(H_ii, ε·mean(diag H))

predicted reduction (recomputed AFTER any step clip):

    pred = gᵀΔ − ½ ΔᵀHΔ

actual reduction from a FULL re-evaluation (basis → LLS → residual):

    ared = F₀ − F₁,   ρ = ared / pred

accept iff pred > 0, ared > 0, ρ > η_accept and everything is finite;
otherwise the current point is kept and λ is escalated (bounded retries).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CompactSystem",
    "GPLMConfig",
    "GPLMDiagnostics",
    "FALLBACK_LEGEND",
    "compact_normal_equations",
    "lm_scale_diag",
    "lm_step",
    "predicted_reduction",
]

COMPACT_MODES = ("raw", "kaufman", "golub_pereyra")


# ---------------------------------------------------------------------------
# Phase 1: compact normal equations (float64 single-spectrum reference)
# ---------------------------------------------------------------------------


@dataclass
class CompactSystem:
    """Data Hessian + gradient of one reduced-residual GN system.

    H is the UN-stabilised data Hessian in the production sign system
    (solve H·Δ = g). aux small matrices are kept for tests/diagnostics.
    """

    H: np.ndarray            # (p, p) symmetric data Hessian
    g: np.ndarray            # (p,) gradient Sᵀr
    G: np.ndarray            # (n_L, n_L) Gram of the linear basis
    C: np.ndarray            # (n_L, p) AᵀS
    K: np.ndarray            # (p, p) SᵀS (= raw Hessian)
    Q: np.ndarray | None     # (n_L, p) D_jᵀ r columns (None for raw/kaufman-only callers may still get it)
    cond_gram: float         # cond₂(G)
    ortho_residual: float    # ‖Aᵀr‖ / ‖A‖‖r‖ — how well Aᵀr = 0 holds


def compact_normal_equations(
    A: np.ndarray,
    D_list: list[np.ndarray],
    x: np.ndarray,
    r: np.ndarray,
    mode: str = "golub_pereyra",
    gram_jitter: float = 0.0,
) -> CompactSystem:
    """Build the compact GN system for one spectrum (float64 reference).

    Args:
        A: (n_rows, n_L) linear basis at the current θ (row-augmented by the
            caller when a Tikhonov objective is intended).
        D_list: p matrices (n_rows, n_L) — ∂A/∂θ_j (zero rows/cols included;
            sparsity exploitation is the batched layer's job, not this one's).
        x: (n_L,) linear coefficients — MUST be the LLS solution on A for
            the Kaufman/GP identities (and the shared gradient) to hold.
        r: (n_rows,) reduced residual y − A x.
        mode: "raw" | "kaufman" | "golub_pereyra".
        gram_jitter: optional numerical jitter added to diag(G) for the
            G-solves only (reported via the returned G which includes it).

    Returns:
        CompactSystem with the data Hessian for `mode`.
    """
    if mode not in COMPACT_MODES:
        raise ValueError(f"mode must be one of {COMPACT_MODES}, got {mode!r}")
    A = np.asarray(A, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    p = len(D_list)
    n_L = A.shape[1]

    S = np.stack([D @ x for D in D_list], axis=1)      # (n_rows, p)
    G = A.T @ A
    if gram_jitter > 0.0:
        G = G + gram_jitter * np.eye(n_L)
    C = A.T @ S                                        # (n_L, p)
    K = S.T @ S                                        # (p, p)
    g = S.T @ r                                        # (p,)
    Q = np.stack([D.T @ r for D in D_list], axis=1)    # (n_L, p)

    if mode == "raw":
        H = K.copy()
    else:
        rhs = np.concatenate([C, Q], axis=1)           # (n_L, 2p)
        sol = np.linalg.solve(G, rhs)                  # no explicit inverse
        H = K - C.T @ sol[:, :p]
        if mode == "golub_pereyra":
            H = H + Q.T @ sol[:, p:]

    H = 0.5 * (H + H.T)

    denom = max(np.linalg.norm(A) * np.linalg.norm(r), 1e-300)
    return CompactSystem(
        H=H, g=g, G=G, C=C, K=K, Q=Q,
        cond_gram=float(np.linalg.cond(G)),
        ortho_residual=float(np.linalg.norm(A.T @ r) / denom),
    )


# ---------------------------------------------------------------------------
# Phase 3: LM safeguard primitives (shared by reference and batched layers)
# ---------------------------------------------------------------------------


def lm_scale_diag(H: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scaled-diagonal LM metric d_i = max(H_ii, ε·mean(diag H)).

    This is DIAGONAL-scaled LM (Marquardt), not identity-λ Levenberg; the
    floor keeps flat parameters from receiving a zero load. Works on (p, p)
    or batched (batch, p, p) input; returns (p,) or (batch, p).
    """
    diag = np.diagonal(H, axis1=-2, axis2=-1)
    mean_diag = diag.mean(axis=-1, keepdims=True)
    return np.maximum(diag, eps * np.maximum(mean_diag, 1e-300))


def lm_step(
    H: np.ndarray, g: np.ndarray, lam: np.ndarray | float, d: np.ndarray,
) -> np.ndarray:
    """Solve (H + λ·diag(d)) Δ = g. Batched-compatible; no inverse.

    H: (..., p, p), g: (..., p), d: (..., p), lam: scalar or (...,).
    """
    p = H.shape[-1]
    lam_arr = np.asarray(lam, dtype=H.dtype)
    load = lam_arr[..., None] * d if lam_arr.ndim else lam_arr * d
    H_stab = H.copy()
    idx = np.arange(p)
    H_stab[..., idx, idx] += load
    return np.linalg.solve(H_stab, g[..., None])[..., 0]


def predicted_reduction(
    g: np.ndarray, H: np.ndarray, delta: np.ndarray,
) -> np.ndarray:
    """pred = gᵀΔ − ½ΔᵀHΔ for the half-SSR objective F = ½‖r‖².

    Uses the DATA Hessian (no LM load), matching the Gauss-Newton model of
    F. Recompute after any step clipping. Batched-compatible.
    """
    gTd = np.sum(g * delta, axis=-1)
    Hd = (H @ delta[..., None])[..., 0]
    return gTd - 0.5 * np.sum(delta * Hd, axis=-1)


# ---------------------------------------------------------------------------
# Phase 3: GP-LM configuration (used by the batched frozen-surrogate layer
# in multipeak_solver and by the MLX port)
# ---------------------------------------------------------------------------


@dataclass
class GPLMConfig:
    """Tunables of the GP-LM safeguarded step. One place, no magic numbers.

    Thresholds follow the standard trust-region calibration: escalate the
    damping when the quadratic model over-promises (ρ < rho_bad), relax it
    when the model tracks well (ρ > rho_good).
    """

    lambda0: float = 1e-3          # initial LM damping
    lambda_up: float = 4.0         # multiplier on rejection
    lambda_down: float = 1.0 / 3.0  # multiplier on acceptance
    lambda_max: float = 1e8        # escalation cap (reject beyond)
    max_retries: int = 3           # bounded λ escalations per Newton call
    eta_accept: float = 1e-4       # min ρ to accept a step
    rho_bad: float = 0.25          # ρ below → λ increases next time
    rho_good: float = 0.75         # ρ above → λ decreases next time
    diag_eps: float = 1e-8         # floor factor in lm_scale_diag
    gram_jitter: float = 1e-10     # numerical jitter on G (float32 layer)
    cond_gate: float = 1e12        # skip GP when cond(G) estimate exceeds
    clip_step_ratio: float = 0.5   # per-axis |Δ| cap in grid-step units


# fallback_reason codes in GPLMDiagnostics
FALLBACK_LEGEND = {
    0: "accepted",        # at least one LM step accepted
    1: "cond_gate",       # cond(Gram) above config.cond_gate — GP not run
    2: "lm_exhausted",    # every trial rejected within the retry budget
    3: "nonfinite",       # NaN/Inf encountered in step or trial objective
}


@dataclass
class GPLMDiagnostics:
    """Per-spectrum diagnostics of a GP-LM refine call (Phase 6).

    All arrays are (batch,) unless noted. Where several LM iterations ran,
    scalar-per-spectrum entries refer to the LAST attempted step of that
    spectrum. Uncertainty note: any std derived from these Hessians is a
    LOCAL GAUSSIAN approximation; `cond_hessian` above ~1e6 (flat valley)
    means such numbers are unreliable — check `step_accepted` and
    `fallback_reason` before quoting them.
    """

    cond_gram: np.ndarray            # cond₂ of the linear-basis Gram
    cond_hessian: np.ndarray         # cond₂ of the DATA Hessian (no LM load)
    lm_lambda: np.ndarray            # final per-spectrum damping
    predicted_reduction: np.ndarray  # pred of the last attempted step
    actual_reduction: np.ndarray     # ared of the last attempted step
    reduction_ratio: np.ndarray      # rho = ared / pred
    step_accepted: np.ndarray        # bool — >=1 step accepted overall
    retry_count: np.ndarray          # int — LM escalations consumed
    step_clipped: np.ndarray         # bool — accepted step hit the clip box
    min_amplitude: np.ndarray        # min over components (final state)
    negative_amplitude_fraction: float  # fraction of spectra with any a<0
    fallback_reason: np.ndarray      # int8 codes, see FALLBACK_LEGEND
    n_basis_evals: int = 0           # trial basis evaluations (frozen: Taylor)
