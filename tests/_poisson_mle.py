"""Batched Poisson maximum-likelihood fitter for one Voigt peak (test helper).

Exists to check ``toyomacro.voigtfit.identifiability`` against simulated
counts. It is deliberately not part of the package and is not a solver
anyone should fit data with: one peak, a background linear in its
coefficients, NumPy only, no robustness beyond what the tests need.

What it minimises is the exact Poisson deviance

    D(theta) = 2 * sum_E [ mu - y + y * log(y / mu) ],   mu = mu(E; theta)

over theta = (amplitude, center, variance, gamma, background...), subject
to ``variance >= variance_floor``. It is *not* a weighted least-squares
fit. The step is Fisher scoring, ``delta = (J^T diag(1/mu) J)^-1 J^T (y/mu
- 1)``, whose fixed points are the zeros of the exact score; every step
is accepted or rejected on the exact deviance, with Levenberg damping on
the information matrix when it is rejected; and convergence is judged on
the exact score. Weighting by the data (Neyman) or fixing the weights
(Pearson) would move the fixed point away from the maximum-likelihood
estimate, and the replica covariance is being compared with the inverse
Fisher matrix, which is a statement about that estimate.

One concession to double precision: an increase of the deviance below
1e-9 per channel is not treated as a rejection. Two points that close
cannot be told apart by the deviance, and without this the last steps
are refused at random (11 % of replicas then never meet the score
tolerance).

The constraint is handled as an active set. A replica sits on the
boundary when its variance is at the floor and the score there points
outward (``dlogL/dv <= 0``); the variance is then held and the remaining
parameters are solved without it. A step that would cross the floor is
truncated onto it. That is the Karush-Kuhn-Tucker point of the
constrained problem, which ``test_identifiability_mc`` cross-checks
against ``scipy.optimize.minimize(method="L-BFGS-B")`` on the same
deviance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy

from toyomacro.voigtfit.identifiability import voigt_derivatives

I_AMP, I_CENTER, I_VAR, I_GAMMA = 0, 1, 2, 3


@dataclass
class MLEResult:
    """Attributes:
        params: (n_replicas, n_params) estimates, ordered amplitude,
            center, variance, gamma, background coefficients
        deviance: (n_replicas,) exact Poisson deviance at the estimate
        converged: (n_replicas,) projected score below tolerance
        at_floor: (n_replicas,) variance held at the floor by the
            constraint
        n_iter: Iterations used
    """

    params: np.ndarray
    deviance: np.ndarray
    converged: np.ndarray
    at_floor: np.ndarray
    n_iter: int


def model(energy, params, basis):
    """Mean and Jacobian for a batch. ``params`` is (n, p); returns (n, n_E)
    and (n, n_E, p). A row with an infeasible gamma or amplitude gives NaN."""
    amp, center, var, gamma = (params[:, i, None] for i in range(4))
    feasible = (amp[:, 0] > 0) & (gamma[:, 0] > 0) & (var[:, 0] >= 0)
    safe_var = np.where(feasible[:, None], var, 1.0)
    safe_gamma = np.where(feasible[:, None], gamma, 1.0)
    d = voigt_derivatives(energy[None, :], center, safe_var, safe_gamma)
    mean = amp * d.value
    columns = [d.value, amp * d.d_center, amp * d.d_variance, amp * d.d_gamma]
    if basis is not None:
        mean = mean + params[:, 4:] @ basis
        columns += [np.broadcast_to(b, mean.shape) for b in basis]
    mean = np.where(feasible[:, None], mean, np.nan)
    return mean, np.stack(columns, axis=2)


def deviance(counts, mean):
    """Exact Poisson deviance per replica; +inf where the mean is not positive."""
    ok = np.all(np.isfinite(mean) & (mean > 0), axis=1)
    safe = np.where(ok[:, None], mean, 1.0)
    value = 2.0 * np.sum(safe - counts + xlogy(counts, counts) - xlogy(counts, safe), axis=1)
    return np.where(ok, value, np.inf)


def fit_poisson_mle(
    counts: np.ndarray,
    energy: np.ndarray,
    start: np.ndarray,
    basis: np.ndarray | None = None,
    *,
    variance_floor: float = 0.0,
    max_iter: int = 60,
    score_tol: float = 1e-6,
) -> MLEResult:
    """Constrained Poisson MLE for every row of ``counts``.

    Args:
        counts: (n_replicas, n_energy) Poisson counts
        energy: (n_energy,)
        start: (n_params,) starting point, normally the truth
        basis: (n_bg, n_energy) background shape functions, or None
        variance_floor: Lower limit of the Gaussian variance
        max_iter: Iteration cap
        score_tol: Convergence when every free parameter has
            ``|score_i| / sqrt(I_ii) < score_tol``, i.e. the remaining
            step is that fraction of a standard deviation
    """
    counts = np.asarray(counts, dtype=np.float64)
    n, p = counts.shape[0], start.size
    theta = np.tile(np.asarray(start, dtype=np.float64), (n, 1))
    damping = np.zeros(n)
    dev = np.empty(n)
    converged = np.zeros(n, dtype=bool)
    at_floor = np.zeros(n, dtype=bool)
    eye = np.eye(p)
    rounding = 1e-9 * energy.size

    mean, jac = model(energy, theta, basis)
    dev[:] = deviance(counts, mean)
    todo = np.arange(n)  # replicas still iterating; mean and jac are kept for these only

    for iteration in range(1, max_iter + 1):
        y = counts[todo]
        score = np.einsum("nep,ne->np", jac, y / mean - 1.0)
        info = np.einsum("nep,neq->npq", jac / mean[:, :, None], jac)
        scale = np.sqrt(np.einsum("npp->np", info))

        floor_now = (theta[todo, I_VAR] <= variance_floor) & (score[:, I_VAR] <= 0.0)
        free = np.ones((todo.size, p), dtype=bool)
        free[floor_now, I_VAR] = False
        done = np.all(~free | (np.abs(score) / scale < score_tol), axis=1)
        at_floor[todo] = floor_now
        converged[todo] = done
        if done.all():
            break
        keep = ~done
        todo, y, score, info, scale, free = (a[keep] for a in (todo, y, score, info, scale, free))
        mean, jac = mean[keep], jac[keep]

        # held parameters: unit row/column, zero right-hand side -> zero step
        both = free[:, :, None] & free[:, None, :]
        system = np.where(both, info, 0.0) + np.where(free, 0.0, 1.0)[:, :, None] * eye
        system += (damping[todo, None] * scale**2)[:, :, None] * eye * both
        step = np.linalg.solve(system, np.where(free, score, 0.0)[:, :, None])[:, :, 0]

        trial = theta[todo] + step
        trial[:, I_VAR] = np.maximum(trial[:, I_VAR], variance_floor)
        trial_mean, trial_jac = model(energy, trial, basis)
        trial_dev = deviance(y, trial_mean)

        accept = trial_dev <= dev[todo] + rounding
        theta[todo[accept]] = trial[accept]
        dev[todo[accept]] = trial_dev[accept]
        mean = np.where(accept[:, None], trial_mean, mean)
        jac = np.where(accept[:, None, None], trial_jac, jac)
        damping[todo] = np.where(accept, damping[todo] / 10.0, np.maximum(damping[todo] * 10.0, 1e-4))

    return MLEResult(theta, dev, converged, at_floor, iteration)


def fit_in_chunks(counts, energy, start, basis=None, *, chunk=2500, **kwargs) -> MLEResult:
    """``fit_poisson_mle`` over slices of the replicas; the Jacobian of ten
    thousand replicas at once is several hundred megabytes."""
    parts = [
        fit_poisson_mle(counts[lo:lo + chunk], energy, start, basis, **kwargs)
        for lo in range(0, counts.shape[0], chunk)
    ]
    return MLEResult(
        params=np.concatenate([q.params for q in parts]),
        deviance=np.concatenate([q.deviance for q in parts]),
        converged=np.concatenate([q.converged for q in parts]),
        at_floor=np.concatenate([q.at_floor for q in parts]),
        n_iter=max(q.n_iter for q in parts),
    )


def fit_reference(counts_row, energy, start, basis=None, *, variance_floor=0.0):
    """One replica by L-BFGS-B on the same exact deviance: an independent solver."""
    scale = np.abs(start) + (start == 0)
    scale[I_VAR] = max(start[I_VAR], 1e-3 * start[I_GAMMA] ** 2)

    def objective(u):
        theta = (u * scale)[None, :]
        mean, jac = model(energy, theta, basis)
        value = deviance(counts_row[None, :], mean)[0]
        if not np.isfinite(value):
            return 1e300, np.zeros_like(u)
        gradient = -2.0 * np.einsum("ep,e->p", jac[0], counts_row / mean[0] - 1.0)
        return value, gradient * scale

    bounds = [(1e-12, None), (None, None), (variance_floor / scale[I_VAR], None), (1e-9, None)]
    bounds += [(None, None)] * (start.size - 4)
    best = minimize(
        objective, start / scale, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10, "maxcor": 30},
    )
    return best.x * scale, best.fun
