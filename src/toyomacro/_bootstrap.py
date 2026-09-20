"""Poisson resampling and a constrained Poisson maximum-likelihood fit (private).

Two things that are the same machinery seen from two sides:

- **Monte Carlo against a bound.** Draw counts from the mean a model
  gives at stated parameters, fit them back, and compare the spread of
  the estimates with the inverse Fisher matrix.
- **The parametric bootstrap.** Draw counts from the mean a *fit* gives,
  fit them back, and read the spread as the estimator's own.

Only the mean drawn from differs. The nonparametric variant draws from
the observed counts themselves (``kind='nonparametric'``), which needs
the counts to be integers and says so when they are not.

What is reported is the **distribution of an estimator**: quantiles, a
covariance, the share of draws that end on a boundary. It is not a
posterior distribution, and nothing here places a prior on anything.

The fitter here minimises the exact Poisson deviance

    D(theta) = 2 sum [ mu - y + y log(y / mu) ]

subject to lower bounds on any parameters that have them. The step is
Fisher scoring, ``delta = (J^T diag(1/mu) J)^-1 J^T (y/mu - 1)``, whose
fixed points are the zeros of the exact score; every step is accepted or
rejected on the exact deviance, with Levenberg damping when it is
rejected; convergence is judged on the exact score. An increase of the
deviance below 1e-9 per channel is not treated as a rejection: two
points that close cannot be told apart by it in double precision.

A bound is handled as an active set. A replica sits on one when the
parameter is at the bound and the score points outward; that parameter
is then held and the rest are solved without it, which is the
Karush-Kuhn-Tucker point of the constrained problem. Tests cross-check
it against a different optimiser on the same deviance.

Any estimator can be bootstrapped, not only this one: ``bootstrap``
takes the fitting step as a callable, and the Fermi-edge tests pass it
the package's own least-squares fitter to see whether that estimator's
sandwich covariance describes its actual spread.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import xlogy

__all__ = [
    "BootstrapResult",
    "MLEResult",
    "bootstrap",
    "draw_poisson",
    "fit_poisson_mle",
    "poisson_deviance",
    "poisson_mle_fitter",
]

#: (mean, jacobian) for a batch of parameter vectors: (n, p) -> (n, n_channels),
#: (n, n_channels, p). Rows with infeasible parameters may return NaN.
ModelBatch = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def poisson_deviance(counts: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Exact Poisson deviance per row; ``+inf`` where the mean is not positive."""
    ok = np.all(np.isfinite(mean) & (mean > 0.0), axis=1)
    safe = np.where(ok[:, np.newaxis], mean, 1.0)
    value = 2.0 * np.sum(safe - counts + xlogy(counts, counts) - xlogy(counts, safe), axis=1)
    return np.where(ok, value, np.inf)


@dataclass
class MLEResult:
    """Attributes:
        params: (n, p) estimates
        deviance: (n,) exact Poisson deviance at the estimate
        converged: (n,) projected score below the tolerance
        at_bound: (n, p) parameters held by their lower bound
        n_iter: Iterations used
    """

    params: np.ndarray
    deviance: np.ndarray
    converged: np.ndarray
    at_bound: np.ndarray
    n_iter: int


def fit_poisson_mle(
    counts: np.ndarray,
    model: ModelBatch,
    start: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    max_iter: int = 60,
    score_tol: float = 1e-6,
    chunk: int = 2500,
) -> MLEResult:
    """Constrained Poisson MLE for every row of ``counts``.

    Args:
        counts: (n_replicas, n_channels)
        model: Mean and Jacobian for a batch of parameter vectors
        start: (p,) starting point, or (n_replicas, p)
        lower: (p,) lower bounds; ``-inf`` where there is none
        max_iter: Iteration cap
        score_tol: Converged when every free parameter has
            ``|score_i| / sqrt(I_ii) < score_tol``, i.e. the step left is
            that fraction of a standard deviation
        chunk: Replicas per batch; the Jacobian of ten thousand at once
            is several hundred megabytes
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"counts must be (n_replicas, n_channels), got {counts.shape}")
    if counts.shape[0] > chunk:
        parts = [fit_poisson_mle(counts[lo:lo + chunk], model, start, lower=lower,
                                 max_iter=max_iter, score_tol=score_tol, chunk=chunk)
                 for lo in range(0, counts.shape[0], chunk)]
        return MLEResult(
            params=np.concatenate([q.params for q in parts]),
            deviance=np.concatenate([q.deviance for q in parts]),
            converged=np.concatenate([q.converged for q in parts]),
            at_bound=np.concatenate([q.at_bound for q in parts]),
            n_iter=max(q.n_iter for q in parts))

    start = np.asarray(start, dtype=np.float64)
    n = counts.shape[0]
    theta = np.tile(start, (n, 1)) if start.ndim == 1 else start.copy()
    p = theta.shape[1]
    bounds = (np.full(p, -np.inf) if lower is None else np.asarray(lower, dtype=np.float64))
    damping = np.zeros(n)
    dev = np.empty(n)
    converged = np.zeros(n, dtype=bool)
    at_bound = np.zeros((n, p), dtype=bool)
    eye = np.eye(p)
    rounding = 1e-9 * counts.shape[1]

    mean, jac = model(theta)
    dev[:] = poisson_deviance(counts, mean)
    todo = np.arange(n)  # replicas still iterating; mean and jac are kept for these only
    iteration = 0

    for iteration in range(1, max_iter + 1):
        y = counts[todo]
        score = np.einsum("nep,ne->np", jac, y / mean - 1.0)
        info = np.einsum("nep,neq->npq", jac / mean[:, :, np.newaxis], jac)
        scale = np.sqrt(np.einsum("npp->np", info))

        held = (theta[todo] <= bounds) & (score <= 0.0)
        free = ~held
        done = np.all(held | (np.abs(score) / scale < score_tol), axis=1)
        at_bound[todo] = held
        converged[todo] = done
        if done.all():
            break
        keep = ~done
        todo, y, score, info, scale, free = (a[keep] for a in (todo, y, score, info, scale, free))
        mean, jac = mean[keep], jac[keep]

        # held parameters: unit row/column, zero right-hand side -> zero step
        both = free[:, :, np.newaxis] & free[:, np.newaxis, :]
        system = np.where(both, info, 0.0) + np.where(free, 0.0, 1.0)[:, :, np.newaxis] * eye
        system += (damping[todo, np.newaxis] * scale**2)[:, :, np.newaxis] * eye * both
        step = np.linalg.solve(system, np.where(free, score, 0.0)[:, :, np.newaxis])[:, :, 0]

        trial = np.maximum(theta[todo] + step, bounds)
        trial_mean, trial_jac = model(trial)
        trial_dev = poisson_deviance(y, trial_mean)

        accept = trial_dev <= dev[todo] + rounding
        theta[todo[accept]] = trial[accept]
        dev[todo[accept]] = trial_dev[accept]
        mean = np.where(accept[:, np.newaxis], trial_mean, mean)
        jac = np.where(accept[:, np.newaxis, np.newaxis], trial_jac, jac)
        damping[todo] = np.where(accept, damping[todo] / 10.0,
                                 np.maximum(damping[todo] * 10.0, 1e-4))

    return MLEResult(theta, dev, converged, at_bound, iteration)


def poisson_mle_fitter(model: ModelBatch, start: np.ndarray, lower: np.ndarray | None = None,
                       **kwargs) -> Callable[[np.ndarray], MLEResult]:
    """``fit_poisson_mle`` with the model, start and bounds bound in."""
    def fit(counts: np.ndarray) -> MLEResult:
        return fit_poisson_mle(counts, model, start, lower=lower, **kwargs)
    return fit


def draw_poisson(mean: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """(n_draws, n_channels) counts from a Poisson with this mean per channel."""
    mean = np.asarray(mean, dtype=np.float64)
    if mean.ndim != 1 or np.any(mean < 0.0) or not np.all(np.isfinite(mean)):
        raise ValueError("mean must be a finite, non-negative vector")
    return rng.poisson(np.broadcast_to(mean, (int(n_draws), mean.size))).astype(np.float64)


@dataclass
class BootstrapResult:
    """The distribution of an estimator over resampled counts.

    Not a posterior: no prior enters, and the spread is the estimator's
    own under repeated sampling from the stated mean.

    Attributes:
        estimates: (n_ok, p) estimates from the draws that converged
        names: Parameter names
        kind: 'parametric' (drawn from a model's mean) or
            'nonparametric' (drawn from observed counts)
        n_draws: Draws attempted
        n_failed: Draws whose fit did not converge, and are excluded
        at_bound: Fraction of converged draws with each parameter held at
            its lower bound
        mean_counts: The mean each draw was taken from
    """

    estimates: np.ndarray
    names: tuple[str, ...]
    kind: str
    n_draws: int
    n_failed: int
    at_bound: np.ndarray
    mean_counts: np.ndarray

    def sd(self) -> np.ndarray:
        """Standard deviation of each parameter over the draws."""
        return self.estimates.std(axis=0, ddof=1)

    def covariance(self) -> np.ndarray:
        return np.cov(self.estimates, rowvar=False)

    def quantiles(self, q) -> np.ndarray:
        """Quantiles of each parameter, shape (len(q), p)."""
        return np.quantile(self.estimates, np.atleast_1d(q), axis=0)


def bootstrap(
    source: np.ndarray,
    fit: Callable[[np.ndarray], MLEResult],
    n_draws: int,
    *,
    rng: np.random.Generator,
    kind: str = "parametric",
    names: tuple[str, ...] | None = None,
) -> BootstrapResult:
    """Resample counts, fit each draw, and collect the estimates.

    Args:
        source: Expected counts per channel ('parametric'), or the
            observed counts ('nonparametric'), which must be integers
        fit: Takes (n, n_channels) counts and returns an ``MLEResult``
        n_draws: Number of draws
        rng: Generator
        kind: 'parametric' or 'nonparametric'
        names: Parameter names for the result
    """
    source = np.asarray(source, dtype=np.float64)
    if kind == "nonparametric":
        if np.any(source < 0.0) or np.any(source != np.rint(source)):
            raise ValueError(
                "a nonparametric bootstrap draws from observed counts: they must be "
                "non-negative integers. Pass the expected counts with kind='parametric' "
                "to bootstrap a model instead.")
    elif kind != "parametric":
        raise ValueError(f"kind must be 'parametric' or 'nonparametric', got {kind!r}")

    counts = draw_poisson(source, n_draws, rng)
    result = fit(counts)
    ok = result.converged
    estimates = result.params[ok]
    at_bound = (result.at_bound[ok].mean(axis=0) if estimates.size
                else np.zeros(result.params.shape[1]))
    return BootstrapResult(
        estimates=estimates,
        names=tuple(names) if names is not None else tuple(
            f"p{i}" for i in range(result.params.shape[1])),
        kind=kind, n_draws=int(n_draws), n_failed=int((~ok).sum()), at_bound=at_bound,
        mean_counts=source)
