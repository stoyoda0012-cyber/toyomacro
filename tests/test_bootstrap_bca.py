"""BCa intervals in the private bootstrap (v0.4.0, not public yet).

Fidelity: the acceleration reproduces the textbook value for a Poisson
mean, and BCa reduces to the percentile interval when there is nothing to
correct. Plausibility: on a Poisson mean, where the exact (Garwood)
interval is known, BCa lands closer to it than the percentile interval.
Whether BCa helps on the Fermi edge is a separate, recorded measurement
(design record section 7).
"""

import math

import numpy as np
import pytest
from scipy.stats import chi2

from toyomacro._bootstrap import (
    BootstrapResult,
    bca_acceleration,
    draw_poisson,
    fit_poisson_mle,
)


def _one_channel(theta):
    theta = np.asarray(theta, dtype=float)
    return theta[:, :1].copy(), np.ones((theta.shape[0], 1, 1))


def _result(estimates):
    est = np.asarray(estimates, dtype=float).reshape(-1, 1)
    return BootstrapResult(estimates=est, names=("x",), kind="parametric", n_draws=est.shape[0],
                           n_failed=0, at_bound=np.zeros(1), mean_counts=np.zeros(1))


@pytest.mark.parametrize("theta", [4.0, 20.0, 500.0])
def test_acceleration_is_the_textbook_value_for_a_poisson_mean(theta):
    a = bca_acceleration(_one_channel, np.array([theta]), np.array([theta]))
    assert a[0] == pytest.approx(1.0 / (6.0 * math.sqrt(theta)), rel=1e-12)


def test_bca_is_the_percentile_interval_when_there_is_nothing_to_correct():
    """Draws symmetric about the estimate (z0 = 0) and no acceleration."""
    r = _result(np.arange(-500, 501))
    bca = r.interval(method="bca", estimate=np.array([0.0]), acceleration=np.array([0.0]))
    assert np.array_equal(bca, r.interval(method="percentile"))


def test_positive_acceleration_moves_the_interval_up():
    r = _result(np.random.default_rng(1).normal(size=4000))
    est = np.array([0.0])
    base = r.interval(method="bca", estimate=est, acceleration=np.array([0.0]))
    up = r.interval(method="bca", estimate=est, acceleration=np.array([0.1]))
    assert (up > base).all()


def test_the_estimate_and_acceleration_are_required():
    with pytest.raises(ValueError):
        _result([1.0, 2.0, 3.0]).interval(method="bca")


def test_bca_is_closer_than_percentile_to_the_exact_poisson_interval():
    """One channel, 20 observed counts, parametric bootstrap with 40,000
    draws. The exact central 95% interval for a Poisson mean is
    (chi2(0.025, 2k)/2, chi2(0.975, 2k+2)/2) = (12.22, 30.89). The
    percentile interval reads (12, 29) and BCa (12, 30): the upper end
    moves toward the exact one. The draws are integers, so an end can only
    move by whole counts; the lower end is held to within one count."""
    k = 20.0
    exact = np.array([chi2.ppf(0.025, 2 * k) / 2, chi2.ppf(0.975, 2 * k + 2) / 2])
    draws = draw_poisson(np.array([k]), 40_000, np.random.default_rng(5))
    fit = fit_poisson_mle(draws, _one_channel, np.array([k]))
    r = _result(fit.params[:, 0])
    a = bca_acceleration(_one_channel, np.array([k]), np.array([k]))
    pct = r.interval(method="percentile")[:, 0]
    bca = r.interval(method="bca", estimate=np.array([k]), acceleration=a)[:, 0]
    assert abs(bca[1] - exact[1]) < abs(pct[1] - exact[1])
    assert abs(bca[0] - exact[0]) <= abs(pct[0] - exact[0]) + 1.0


# --- profile-likelihood interval --------------------------------------------

from scipy.optimize import brentq as _brentq  # noqa: E402
from scipy.stats import chi2 as _chi2  # noqa: E402

from toyomacro._bootstrap import profile_interval  # noqa: E402


def _signal_on_background(bg):
    def model(theta):
        theta = np.asarray(theta, dtype=float)
        return theta[:, :1] + bg, np.ones((theta.shape[0], 1, 1))
    return model


def _lr(y, mu):
    """Poisson deviance of one channel, written out independently."""
    return 2.0 * (mu - y + (y * math.log(y / mu) if y > 0 else 0.0))


def test_profile_interval_is_the_likelihood_ratio_interval_for_a_poisson_mean():
    y = 20.0
    iv = profile_interval(np.array([y]), _one_channel, np.array([y]), 0)
    c = _chi2.ppf(0.95, 1)
    lo = _brentq(lambda t: _lr(y, t) - c, 1e-6, y)
    hi = _brentq(lambda t: _lr(y, t) - c, y, 10 * y)
    assert iv.low == pytest.approx(lo, rel=1e-5)
    assert iv.high == pytest.approx(hi, rel=1e-5)
    assert not iv.bound_included


@pytest.mark.parametrize(("y", "included"), [(8.0, True), (20.0, False)])
def test_self_liang_threshold_decides_whether_the_bound_is_in_the_set(y, included):
    """Signal theta >= 0 on a known background of 10.

    y = 8: the MLE sits on the bound, the excess at 0 is 0 -> included.
    y = 20: theta_hat = 10; the excess at 0 is about 7.7, above both
    thresholds -> the bound is excluded and the lower end is interior."""
    model = _signal_on_background(10.0)
    est = np.array([max(y - 10.0, 0.0)])
    iv = profile_interval(np.array([y]), model, est, 0, lower=np.array([0.0]))
    assert iv.bound_included is included
    assert iv.covers(0.0, lower_bound=0.0) is included
    c = _chi2.ppf(0.95, 1)
    d_min = _lr(y, est[0] + 10.0)
    hi = _brentq(lambda t: _lr(y, t + 10.0) - d_min - c, est[0], 100.0)
    assert iv.high == pytest.approx(hi, rel=1e-5)


def test_the_bound_uses_the_mixture_threshold_not_chi2_1():
    """An excess at the bound between 2.71 and 3.84 keeps the bound out of
    the set (Self & Liang) although the naive chi2_1 threshold would let it
    in. Background 10, y chosen so that the excess at 0 is about 3.2."""
    model = _signal_on_background(10.0)
    y = _brentq(lambda yy: _lr(yy, 10.0) - _lr(yy, yy) - 3.2, 10.5, 40.0)
    iv = profile_interval(np.array([y]), model, np.array([y - 10.0]), 0, lower=np.array([0.0]))
    assert iv.low == 0.0
    assert not iv.bound_included
