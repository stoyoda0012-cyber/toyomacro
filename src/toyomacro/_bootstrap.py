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
the unconstrained case against a different optimiser on the same
deviance, and check the constrained one directly -- raising a held
parameter off its bound must not lower the deviance. The two are
separate because the cross-check is at an interior point, where nothing
is held and the active set has nothing to do.

``bootstrap`` takes the fitting step as a callable, so another
estimator could be bootstrapped here. Nothing in this repository does:
every call passes ``poisson_mle_fitter``, and the callable has to
return an ``MLEResult`` -- ``params``, ``converged`` and ``at_bound``
-- which the package's least-squares fitter does not, and for which no
adapter exists. The extension point has one implementation. Where the
least-squares fitter's spread is checked against its sandwich
covariance, the draws are taken directly with ``draw_poisson`` and the
fitter looped over them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy
from scipy.stats import chi2

__all__ = [
    "BootstrapResult",
    "MLEResult",
    "bootstrap",
    "draw_poisson",
    "estimate_gain",
    "fit_poisson_mle",
    "poisson_deviance",
    "poisson_mle_fitter",
    "profile_interval",
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
    start = np.asarray(start, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"counts must be (n_replicas, n_channels), got {counts.shape}")
    if start.ndim == 2 and start.shape[0] != counts.shape[0]:
        raise ValueError(f"one start per replica or one for all: start {start.shape} "
                         f"against counts {counts.shape}")
    if counts.shape[0] > chunk:
        # a per-replica start has to be cut with the counts it belongs to
        parts = [fit_poisson_mle(counts[lo:lo + chunk], model,
                                 start[lo:lo + chunk] if start.ndim == 2 else start,
                                 lower=lower, max_iter=max_iter, score_tol=score_tol,
                                 chunk=chunk)
                 for lo in range(0, counts.shape[0], chunk)]
        return MLEResult(
            params=np.concatenate([q.params for q in parts]),
            deviance=np.concatenate([q.deviance for q in parts]),
            converged=np.concatenate([q.converged for q in parts]),
            at_bound=np.concatenate([q.at_bound for q in parts]),
            n_iter=max(q.n_iter for q in parts))

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
        rhs = np.where(free, score, 0.0)[:, :, np.newaxis]
        try:
            step = np.linalg.solve(system, rhs)[:, :, 0]
        except np.linalg.LinAlgError:
            # One singular replica (e.g. a resampled spectrum whose channels
            # cannot separate two parameters) used to abort the whole batch.
            # Its least-norm step lets the rest proceed; it then converges
            # or is reported as not converged like any other replica. Batches
            # without a singular system take the branch above, unchanged.
            step = (np.linalg.pinv(system) @ rhs)[:, :, 0]

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

    def quantiles(self, q, *, method: str = "weibull") -> np.ndarray:
        """Quantiles of each parameter, shape (len(q), p).

        The default is **not** numpy's. ``method='linear'`` interpolates
        at index ``q(B-1)``, which at q = 0.025 and B = 200 draws reads
        the 5.975-th of 200 order statistics -- whose expected position
        in the distribution is 5.975/201 = 2.97%, not 2.5%. A two-sided
        "95%" interval built that way is nominally 94.05% at B = 200 and
        94.81% at B = 1000, so it under-covers by an amount that shrinks
        with B and is easily mistaken for a property of the bootstrap.
        Measured on a nested check, the convention alone was worth about
        a point of coverage at B = 200.

        ``'weibull'`` places order statistic k at k/(B+1), the plotting
        position the percentile interval is defined with (Efron &
        Tibshirani 1993, §13.3), and is nominally 95.00% at any B. Pass
        ``method='linear'`` for numpy's default if you want it.
        """
        return np.quantile(self.estimates, np.atleast_1d(q), axis=0, method=method)


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

    Three ways to resample:

    - ``'parametric'``: Poisson draws from expected counts (a fit's mean).
    - ``'nonparametric'``: Poisson draws from the observed counts.
    - ``'sweep'``: the spectrum is the sum of repeated sweeps; each draw
      resamples the sweeps with replacement and sums them. It needs no
      noise model, so it also carries whatever the sweeps share beyond
      Poisson noise (source intensity, drift), and it does not need the
      intensities to be counts.

    Intensities from an analog (ADC) detector are not counts. Dividing
    them by the slope from :func:`estimate_gain` matches each channel's
    variance to a Poisson draw's, and nothing more: the result is in
    general not integer (``'nonparametric'`` rejects it; round only
    knowingly), and a correlation between channels is not modelled by
    either Poisson kind.

    Args:
        source: Expected counts per channel ('parametric'), the observed
            counts ('nonparametric', non-negative integers), or the sweeps,
            shape (n_sweeps, n_channels) ('sweep')
        fit: Takes (n, n_channels) spectra and returns an ``MLEResult``
        n_draws: Number of draws
        rng: Generator
        kind: 'parametric', 'nonparametric' or 'sweep'
        names: Parameter names for the result
    """
    source = np.asarray(source, dtype=np.float64)
    if kind == "nonparametric":
        if np.any(source < 0.0) or np.any(source != np.rint(source)):
            raise ValueError(
                "a nonparametric bootstrap draws from observed counts: they must be "
                "non-negative integers. Pass the expected counts with kind='parametric' "
                "to bootstrap a model instead.")
    elif kind == "sweep":
        if source.ndim != 2 or source.shape[0] < 2:
            raise ValueError("a sweep bootstrap needs the sweeps as (n_sweeps >= 2, n_channels)")
    elif kind != "parametric":
        raise ValueError(f"kind must be 'parametric', 'nonparametric' or 'sweep', got {kind!r}")

    if kind == "sweep":
        n_sweeps = source.shape[0]
        picks = rng.integers(0, n_sweeps, size=(int(n_draws), n_sweeps))
        counts = source[picks].sum(axis=1)
    else:
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
        mean_counts=source.sum(axis=0) if kind == "sweep" else source)


@dataclass
class ProfileInterval:
    """A likelihood-ratio interval for one parameter.

    Attributes:
        low, high: Ends. ``low`` equals the lower bound when the profile
            stays under the threshold all the way down to it.
        bound_included: Whether the lower bound itself belongs to the set.
            Judged with the Self & Liang threshold (see ``profile_interval``),
            so it can be False while ``low`` equals the bound: the set is
            then open at the bound. This is what decides coverage when the
            true value sits exactly on the bound.
        n_unconverged: How many of the constrained refits behind the
            profile did not report convergence. Not zero means the
            profile was read from fits that stopped at ``max_iter``.
    """

    low: float
    high: float
    bound_included: bool
    n_unconverged: int = 0

    def covers(self, value: float, lower_bound: float = -np.inf) -> bool:
        if value == lower_bound:
            return self.bound_included
        return self.low <= value <= self.high


def profile_interval(counts: np.ndarray, model: ModelBatch, estimate: np.ndarray, index: int,
                     lower: np.ndarray | None = None, *, level: float = 0.95,
                     max_iter: int = 200) -> ProfileInterval:
    """Profile-likelihood interval for parameter ``index`` from one spectrum.

    The parameter is fixed at t, every other parameter is refitted
    (constrained Poisson MLE), and the set is ``{t : D_p(t) - D_min <= c}``.
    Away from a bound ``c`` is the ``level`` quantile of chi2 with one
    degree of freedom (3.84 at 95%). At the lower bound itself the null
    distribution of the statistic is the 50:50 mixture of chi2_0 and chi2_1
    (Self & Liang 1987), so the bound is in the set when the excess is under
    the ``2*level - 1`` quantile of chi2_1 (2.71 at 95%). That threshold is
    always used at the bound, and it is exact only for one parameter on its
    bound with the others interior; when a nuisance parameter sits on a
    bound too the true mixture differs and ``bound_included`` is
    approximate.

    ``estimate`` must be the constrained MLE for ``counts``: ``D_min`` is
    taken there and nothing checks it. The result is the connected set
    around the estimate; a separate region elsewhere under the threshold
    is not reported.

    Measured on the Fermi edge for E_F, v and tau (design record section
    7): nominal at an interior point, and for tau with v on its bound; for
    v there 98.1% at the true value and 94.6% judged at the bound, a
    reading chosen after the measurement. Over-covering (98%) where the width
    split is ``not_separable``, where an interval for v or tau should be
    withheld as ``sd_tau`` is.

    Args:
        counts: (n_channels,) one spectrum
        model: Mean and Jacobian for a batch of parameter vectors
        estimate: (p,) the constrained MLE for ``counts``
        index: Which parameter
        lower: (p,) lower bounds
        level: Coverage level
    """
    counts = np.asarray(counts, dtype=np.float64)[None, :]
    theta = np.asarray(estimate, dtype=np.float64)
    p = theta.size
    lo_b = np.full(p, -np.inf) if lower is None else np.asarray(lower, dtype=np.float64)
    keep = [i for i in range(p) if i != index]
    d_min = float(poisson_deviance(counts, model(theta[None])[0])[0])
    c_in = float(chi2.ppf(level, 1))
    c_bound = float(chi2.ppf(2.0 * level - 1.0, 1))

    def reduced(t):
        def m(theta_r):
            full = np.insert(np.asarray(theta_r, dtype=np.float64), index, t, axis=1)
            mean, jac = model(full)
            return mean, np.delete(jac, index, axis=2)
        return m

    unconverged = [0]

    def excess(t):
        fit = fit_poisson_mle(counts, reduced(t), theta[keep], lower=lo_b[keep], max_iter=max_iter)
        unconverged[0] += int(not fit.converged[0])
        return float(fit.deviance[0]) - d_min

    mean, jac = model(theta[None])
    info = jac[0].T @ (jac[0] / np.maximum(mean[0], 1e-300)[:, None])
    step = float(np.sqrt(max(np.linalg.pinv(info)[index, index], 0.0)))
    if not np.isfinite(step) or step <= 0.0:
        step = max(abs(theta[index]), 1.0) * 1e-3

    def bracket(direction):
        a, h = theta[index], step
        for _ in range(60):
            b = a + direction * h
            if direction < 0 and b <= lo_b[index]:
                return None
            if excess(b) > c_in:
                return b
            h *= 2.0
        return np.inf * direction

    hi_edge = bracket(+1.0)
    high = (hi_edge if not np.isfinite(hi_edge)
            else brentq(lambda t: excess(t) - c_in, theta[index], hi_edge, xtol=step * 1e-4))

    bound_included = False
    at_bound = excess(lo_b[index]) if np.isfinite(lo_b[index]) else np.inf
    if at_bound <= c_in:
        low = float(lo_b[index])
        bound_included = at_bound <= c_bound
    else:
        lo_edge = bracket(-1.0)
        if lo_edge is None:
            lo_edge = lo_b[index]
        low = (lo_edge if not np.isfinite(lo_edge)
               else brentq(lambda t: excess(t) - c_in, lo_edge, theta[index], xtol=step * 1e-4))
    return ProfileInterval(low=float(low), high=float(high), bound_included=bool(bound_included),
                           n_unconverged=unconverged[0])


@dataclass
class GainEstimate:
    """Detector gain from repeated sweeps.

    Attributes:
        gain: Intensity per count, ``b`` in ``intensity = b * counts``
        offset: Intercept of the variance-mean line, intensity^2 per
            channel; read noise or a pedestal shows here, zero for a
            pure scaled Poisson
        sweep_factor_sd: Standard deviation of the per-sweep total
            relative to its mean (source intensity and Poisson together)
        n_sweeps, n_channels: Data used
    """

    gain: float
    offset: float
    sweep_factor_sd: float
    n_sweeps: int
    n_channels: int


def estimate_gain(sweeps: np.ndarray) -> GainEstimate:
    """Estimate ``b`` in ``intensity = b * counts`` from repeated sweeps.

    Each sweep is first divided by its total relative to the mean, which
    removes changes of source intensity from sweep to sweep. Differences
    of adjacent sweeps then remove what drifts slowly, and per channel
    ``E[(x_s - x_{s+1})^2] / 2 = b * mean + offset`` for counts that are
    Poisson up to the factor ``b``. ``b`` and ``offset`` are the
    least-squares line through all channels.

    Valid for: a detector whose output is proportional to Poisson counts,
    sweeps taken under the same conditions, and drift slow on the scale
    of one sweep. Two small biases of opposite sign: normalising by the
    total takes out a part of the Poisson noise of the total itself,
    which biases ``b`` low by about one part in the number of channels,
    and dividing by the noisy per-sweep factor adds its variance,
    biasing ``b`` high by about ``sweep_factor_sd**2``.

    Not valid as a per-event gain when channels are correlated (for
    instance by charge spreading or rebinning): ``b`` is then the
    per-channel variance-to-mean slope on the stored scale, which is what
    a per-channel noise model needs, but not intensity per detected
    electron.

    Args:
        sweeps: (n_sweeps, n_channels), raw intensities of each sweep
    """
    x = np.asarray(sweeps, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError("estimate_gain needs (n_sweeps >= 3, n_channels)")
    totals = x.sum(axis=1)
    factor = totals / totals.mean()
    xn = x / factor[:, None]
    mean = xn.mean(axis=0)
    half_var = 0.5 * ((xn[:-1] - xn[1:]) ** 2).mean(axis=0)
    design = np.stack([mean, np.ones_like(mean)], axis=1)
    (b, offset), *_ = np.linalg.lstsq(design, half_var, rcond=None)
    return GainEstimate(gain=float(b), offset=float(offset),
                        sweep_factor_sd=float(factor.std(ddof=1)),
                        n_sweeps=x.shape[0], n_channels=x.shape[1])
