"""
Golub-Pereyra reduced-residual Jacobian — NumPy float64 reference.
==================================================================

Verification-grade reference implementation for comparing three Jacobian
approximations of the *separable* (VarPro) Voigt fitting problem used by the
post-AP Newton refinement in :mod:`toyomacro.voigtfit.multipeak_solver`:

    a*(theta) = argmin_a || y - Phi(theta) a ||^2      (unconstrained LLS)
    r(theta)  = y - Phi(theta) a*(theta)               (reduced residual)

with theta the non-linear parameters (per-peak center and sigma, interleaved
``[c0, s0, c1, s1, ...]`` to match the production ``(2k, 2k+1)`` layout).

Jacobian modes (all in the reduced-residual convention ``J = dr/dtheta``):

    raw:            J_j = -S_j                       where S_j = (d_j Phi) a
    kaufman:        J_j = -(I - P) S_j               P = Phi Phi^+
    golub_pereyra:  J_j = -(I - P) S_j - (Phi^+)^T (d_j Phi)^T r

Sign conventions vs production
------------------------------
``_post_ap_newton_refine_chunk`` uses the MODEL-side Jacobian ``+S`` with
``g = S^T R`` and solves ``(S^T S) delta = g``, ``theta += delta``. That is
identical to the Gauss-Newton step in the reduced convention used here:
``(J^T J) delta = -J^T r`` with ``J = -S``. All three modes in this module
share the reduced convention, so their steps are directly comparable and the
raw-mode step reproduces the production step (before Tikhonov/clipping).

Numerical policy
----------------
* float64 throughout; readability over speed (this is NOT a production path).
* No explicit inverse, no explicit projection matrix. Everything goes through
  a thin QR of Phi: ``P v = Q (Q^T v)``, ``(Phi^+)^T q = Q R^{-T} q``.
* The amplitude solve is plain unconstrained LLS. Negative amplitudes are
  allowed and reported, never clipped.
* Optional Tikhonov: handled EXPLICITLY as the augmented system
  ``[Phi; sqrt(lam) I]``, ``[y; 0]`` so the GP identities stay exact for the
  augmented residual. lam=0 (default) is the pure GP setting.
* Optional fixed linear columns (polynomial background): appended to Phi with
  zero derivative; they participate in the projection as required.

Spin-orbit doublets are one structured column:
``Phi_k = V(c_k) + partner_ratio_k * V(c_k + so_split_k)`` (same sigma/gamma),
with the derivative columns composed the same way — matching
``_eval_exact_all`` in the production solver (there ``partner_ratio ==
1/branch_ratio``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_triangular

from .voigt_jacobian import voigt_with_jacobian

JACOBIAN_MODES = ("raw", "kaufman", "golub_pereyra")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class SeparableVoigtModel:
    """Separable Voigt model: y ~ Phi(theta) a  (+ optional fixed columns).

    Attributes:
        energy: (n_E,) energy axis, float64.
        gamma: Lorentzian width — scalar or (n_comp,) per peak.
        so_split: (n_comp,) doublet partner offset per peak (0 = singlet).
        partner_ratio: (n_comp,) partner amplitude multiplier
            (production: 1/branch_ratio). Ignored where so_split == 0.
        fixed_columns: (n_E, n_bg) extra linear columns with zero theta
            derivative (e.g. polynomial background), or None.
        tikhonov_lambda: if > 0, all residual/Jacobian quantities refer to the
            row-augmented system [Phi; sqrt(lam) I], [y; 0].
    """

    energy: np.ndarray
    gamma: float | np.ndarray = 0.2
    so_split: np.ndarray | None = None
    partner_ratio: np.ndarray | None = None
    fixed_columns: np.ndarray | None = None
    tikhonov_lambda: float = 0.0

    def __post_init__(self):
        self.energy = np.asarray(self.energy, dtype=np.float64)
        if self.fixed_columns is not None:
            self.fixed_columns = np.asarray(self.fixed_columns, dtype=np.float64)

    # -- helpers ----------------------------------------------------------

    def n_comp(self, theta: np.ndarray) -> int:
        if len(theta) % 2 != 0:
            raise ValueError("theta must interleave [c0, s0, c1, s1, ...]")
        return len(theta) // 2

    def _gamma_k(self, k: int, n: int) -> float:
        g = np.asarray(self.gamma, dtype=np.float64)
        return float(g) if g.ndim == 0 else float(g[k])

    def n_linear(self, theta: np.ndarray) -> int:
        n_bg = 0 if self.fixed_columns is None else self.fixed_columns.shape[1]
        return self.n_comp(theta) + n_bg

    # -- basis ------------------------------------------------------------

    def basis_and_derivs(
        self, theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (Phi, dPhi) at theta — WITHOUT Tikhonov augmentation.

        Phi:  (n_E, n_lin)   peak columns then fixed columns.
        dPhi: (p, n_E, n_lin) derivative wrt theta_j; fixed columns and
              cross-peak entries are exactly zero.
        """
        theta = np.asarray(theta, dtype=np.float64)
        n = self.n_comp(theta)
        p = 2 * n
        n_E = len(self.energy)
        n_lin = self.n_linear(theta)

        Phi = np.zeros((n_E, n_lin), dtype=np.float64)
        dPhi = np.zeros((p, n_E, n_lin), dtype=np.float64)

        for k in range(n):
            c, s = float(theta[2 * k]), float(theta[2 * k + 1])
            if s <= 0:
                raise ValueError(f"sigma[{k}] = {s} must be positive")
            g = self._gamma_k(k, n)
            V, dVc, dVs, _ = voigt_with_jacobian(self.energy, c, s, g)
            split = 0.0 if self.so_split is None else float(self.so_split[k])
            if split != 0.0:
                ratio = float(self.partner_ratio[k])
                Vp, dVcp, dVsp, _ = voigt_with_jacobian(
                    self.energy, c + split, s, g,
                )
                V = V + ratio * Vp
                dVc = dVc + ratio * dVcp
                dVs = dVs + ratio * dVsp
            Phi[:, k] = V
            dPhi[2 * k, :, k] = dVc
            dPhi[2 * k + 1, :, k] = dVs

        if self.fixed_columns is not None:
            Phi[:, n:] = self.fixed_columns

        return Phi, dPhi

    def augment(
        self, Phi: np.ndarray, dPhi: np.ndarray, y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row-augment (Phi, dPhi, y) with the Tikhonov block if lam > 0."""
        lam = self.tikhonov_lambda
        if lam <= 0.0:
            return Phi, dPhi, np.asarray(y, dtype=np.float64)
        n_lin = Phi.shape[1]
        aug = np.sqrt(lam) * np.eye(n_lin)
        Phi_b = np.vstack([Phi, aug])
        dPhi_b = np.concatenate(
            [dPhi, np.zeros((dPhi.shape[0], n_lin, n_lin))], axis=1,
        )
        y_b = np.concatenate([np.asarray(y, dtype=np.float64), np.zeros(n_lin)])
        return Phi_b, dPhi_b, y_b


# ---------------------------------------------------------------------------
# Linear solve + reduced residual
# ---------------------------------------------------------------------------


def solve_amplitudes_qr(
    Phi: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unconstrained LLS via thin QR. Returns (a, Q, R).

    a = R^{-1} Q^T y. No jitter: the projector identities Phi^T r = 0 must
    hold to machine precision for the GP formulas to be exact.
    """
    Q, R = np.linalg.qr(Phi, mode="reduced")
    a = solve_triangular(R, Q.T @ y, lower=False)
    return a, Q, R


def reduced_residual(
    model: SeparableVoigtModel, theta: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """r(theta) = y_bar - Phi_bar a*(theta) with full amplitude re-solve.

    Returns (r, a). r refers to the augmented system when tikhonov_lambda > 0.
    """
    Phi, dPhi = model.basis_and_derivs(theta)
    Phi_b, _, y_b = model.augment(Phi, dPhi, y)
    a, _, _ = solve_amplitudes_qr(Phi_b, y_b)
    return y_b - Phi_b @ a, a


# ---------------------------------------------------------------------------
# Jacobians
# ---------------------------------------------------------------------------


@dataclass
class JacobianInfo:
    """Byproducts of a Jacobian evaluation at theta."""

    a: np.ndarray            # (n_lin,) LLS amplitudes
    r: np.ndarray            # (n_rows,) reduced residual
    cond_gram: float         # cond(Phi^T Phi) = cond(R)^2


def reduced_jacobian(
    model: SeparableVoigtModel,
    theta: np.ndarray,
    y: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, JacobianInfo]:
    """Reduced-residual Jacobian J = dr/dtheta, (n_rows, p), in `mode`.

    raw:            J_j = -S_j
    kaufman:        J_j = -(I - P) S_j
    golub_pereyra:  J_j = -(I - P) S_j - Q R^{-T} (d_j Phi)^T r

    where S_j = (d_j Phi) a and P = Q Q^T from the thin QR of (augmented) Phi.
    """
    if mode not in JACOBIAN_MODES:
        raise ValueError(f"mode must be one of {JACOBIAN_MODES}, got {mode!r}")

    theta = np.asarray(theta, dtype=np.float64)
    Phi, dPhi = model.basis_and_derivs(theta)
    Phi_b, dPhi_b, y_b = model.augment(Phi, dPhi, y)

    a, Q, R = solve_amplitudes_qr(Phi_b, y_b)
    r = y_b - Phi_b @ a

    p = dPhi_b.shape[0]
    J = np.empty((Phi_b.shape[0], p), dtype=np.float64)

    for j in range(p):
        S_j = dPhi_b[j] @ a                       # (n_rows,)
        if mode == "raw":
            J[:, j] = -S_j
            continue
        T1_j = S_j - Q @ (Q.T @ S_j)              # (I - P) S_j
        if mode == "kaufman":
            J[:, j] = -T1_j
        else:  # golub_pereyra
            q_j = dPhi_b[j].T @ r                 # (n_lin,)
            T2_j = Q @ solve_triangular(R.T, q_j, lower=True)
            J[:, j] = -(T1_j + T2_j)

    cond_R = np.linalg.cond(R)
    return J, JacobianInfo(a=a, r=r, cond_gram=float(cond_R) ** 2)


def fd_jacobian(
    model: SeparableVoigtModel,
    theta: np.ndarray,
    y: np.ndarray,
    h: np.ndarray | float | None = None,
) -> np.ndarray:
    """Central-difference Jacobian of the FULL reduced residual.

    Both sides rebuild Phi(theta +/- h e_j) and completely re-solve the
    amplitude LLS — no dictionary argmax, no clipping, no frozen state.

    h defaults to 3e-6 * scale_j with scale_j = sigma of the peak that
    theta_j belongs to (both the center and the sigma coordinate live on the
    energy-width scale).
    """
    theta = np.asarray(theta, dtype=np.float64)
    p = len(theta)
    if h is None:
        scales = np.empty(p)
        for k in range(p // 2):
            scales[2 * k] = theta[2 * k + 1]
            scales[2 * k + 1] = theta[2 * k + 1]
        h = 3e-6 * scales
    h = np.broadcast_to(np.asarray(h, dtype=np.float64), (p,))

    cols = []
    for j in range(p):
        tp = theta.copy()
        tm = theta.copy()
        tp[j] += h[j]
        tm[j] -= h[j]
        rp, _ = reduced_residual(model, tp, y)
        rm, _ = reduced_residual(model, tm, y)
        cols.append((rp - rm) / (2.0 * h[j]))
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# Gauss-Newton iteration
# ---------------------------------------------------------------------------


@dataclass
class NewtonTrace:
    """Per-iteration record of a run_newton() call."""

    mode: str
    method: str = "lm"
    theta: list = field(default_factory=list)          # theta AFTER each iter
    ssr: list = field(default_factory=list)            # ||r||^2 after iter
    grad_norm: list = field(default_factory=list)      # ||J^T r|| before step
    step_norm: list = field(default_factory=list)      # ||delta|| taken
    cond_gram: list = field(default_factory=list)
    backtrack_count: list = field(default_factory=list)  # halvings / lam bumps
    ssr0: float = np.nan                                # ||r(theta0)||^2
    n_increase: int = 0            # iters where accepted ssr went UP
    converged: bool = False
    stalled: bool = False          # safeguard found no improving step
    n_iter: int = 0
    theta_final: np.ndarray | None = None
    a_final: np.ndarray | None = None


def gauss_newton_step(
    model: SeparableVoigtModel,
    theta: np.ndarray,
    y: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, JacobianInfo]:
    """One undamped GN step: solve min_d ||r + J d||^2 via lstsq.

    Returns (delta, J, info); the caller applies theta + t*delta.
    """
    J, info = reduced_jacobian(model, theta, y, mode)
    delta, *_ = np.linalg.lstsq(J, -info.r, rcond=None)
    return delta, J, info


def _lm_delta(J: np.ndarray, r: np.ndarray, lam: float) -> np.ndarray:
    """Levenberg-Marquardt step: (J^T J + lam*diag(J^T J)) d = -J^T r."""
    A = J.T @ J
    d = np.diag(A).copy()
    d[d <= 0] = max(d.max(), 1e-300)
    return np.linalg.solve(A + lam * np.diag(d), -(J.T @ r))


def run_newton(
    model: SeparableVoigtModel,
    theta0: np.ndarray,
    y: np.ndarray,
    mode: str,
    method: str = "lm",
    max_iter: int = 30,
    tol_step: float = 1e-8,
    tol_rel_ssr: float = 1e-12,
    max_backtracks: int = 12,
    lm_lambda0: float = 1e-3,
    lm_lambda_max: float = 1e10,
    step_clip: np.ndarray | None = None,
    min_sigma: float = 1e-3,
) -> NewtonTrace:
    """Newton-type iteration on the reduced problem with a COMMON safeguard.

    The same globalization serves all Jacobian modes so convergence
    differences come from the Jacobian, not the safeguard. Methods:

      "lm":           Levenberg-Marquardt damping (J^T J + lam*diag), lam
                      adapted x4 up on rejection / x0.5^... down on success.
                      The standard VarPro safeguard; recommended baseline.
      "gn_backtrack": undamped GN direction + halving line search; if no
                      improving step length exists the run STOPS (stalled).
      "gn":           native undamped GN; the full step is accepted even if
                      the objective increases (exposes native stability;
                      n_increase counts those events).

    step_clip: optional (p,) elementwise |delta| cap applied before the
    trial evaluation (mirrors the production clip_step_ratio safeguard).
    min_sigma: hard floor keeping sigma positive during trials.
    Termination: ||step|| < tol_step, or relative SSR improvement below
    tol_rel_ssr, or safeguard stall, or max_iter.
    """
    if method not in ("lm", "gn_backtrack", "gn"):
        raise ValueError(f"unknown method {method!r}")
    theta = np.asarray(theta0, dtype=np.float64).copy()
    trace = NewtonTrace(mode=mode, method=method)

    r, a = reduced_residual(model, theta, y)
    ssr = float(r @ r)
    trace.ssr0 = ssr
    lam = lm_lambda0

    def _try(theta_base, delta, t):
        theta_try = theta_base + t * delta
        theta_try[1::2] = np.maximum(theta_try[1::2], min_sigma)
        r_try, a_try = reduced_residual(model, theta_try, y)
        return theta_try, r_try, a_try, float(r_try @ r_try)

    for _ in range(max_iter):
        J, info = reduced_jacobian(model, theta, y, mode)
        trace.grad_norm.append(float(np.linalg.norm(J.T @ info.r)))
        trace.cond_gram.append(info.cond_gram)

        n_bt = 0
        if method == "lm":
            while True:
                delta = _lm_delta(J, info.r, lam)
                if step_clip is not None:
                    delta = np.clip(delta, -step_clip, step_clip)
                theta_try, r_try, a_try, ssr_try = _try(theta, delta, 1.0)
                if ssr_try < ssr:
                    lam = max(lam / 3.0, 1e-12)
                    break
                lam *= 4.0
                n_bt += 1
                if lam > lm_lambda_max:
                    trace.stalled = True
                    break
            if trace.stalled:
                break
        else:
            delta, *_ = np.linalg.lstsq(J, -info.r, rcond=None)
            if step_clip is not None:
                delta = np.clip(delta, -step_clip, step_clip)
            t = 1.0
            while True:
                theta_try, r_try, a_try, ssr_try = _try(theta, delta, t)
                if method == "gn" or ssr_try < ssr:
                    break
                if n_bt >= max_backtracks:
                    trace.stalled = True
                    break
                t *= 0.5
                n_bt += 1
            if trace.stalled:
                break

        if ssr_try >= ssr:
            trace.n_increase += 1
        step_taken = theta_try - theta  # actual movement (incl. sigma floor)
        ssr_prev = ssr
        theta, ssr, a = theta_try, ssr_try, a_try

        trace.theta.append(theta.copy())
        trace.ssr.append(ssr)
        trace.step_norm.append(float(np.linalg.norm(step_taken)))
        trace.backtrack_count.append(n_bt)
        trace.n_iter += 1

        if trace.step_norm[-1] < tol_step:
            trace.converged = True
            break
        if 0 <= ssr_prev - ssr < tol_rel_ssr * max(ssr_prev, 1e-300):
            trace.converged = True
            break

    trace.theta_final = theta
    trace.a_final = a
    return trace
