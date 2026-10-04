"""Profile-likelihood intervals in the private bootstrap (v0.4.0, not public yet).

Fidelity: the interval equals the likelihood-ratio interval for a Poisson
mean, and at a lower bound the Self & Liang mixture threshold (2.71 at
95%), not chi2_1 (3.84), decides whether the bound is in the set.
How it covers on the Fermi edge is a recorded measurement (design record
section 7). BCa was measured on the same trials and not adopted.
"""

import math

import numpy as np
import pytest
from scipy.optimize import brentq as _brentq
from scipy.stats import chi2 as _chi2

from toyomacro._bootstrap import profile_interval


def _one_channel(theta):
    theta = np.asarray(theta, dtype=float)
    return theta[:, :1].copy(), np.ones((theta.shape[0], 1, 1))


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
    assert iv.n_unconverged == 0


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
