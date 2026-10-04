"""Sweep resampling and the detector gain, in the private bootstrap (v0.4.0).

Synthetic sweeps only; the real-data check (repeated Au Fermi-edge sweeps)
is recorded in design record section 7 and uses data that is not bundled.
"""

import numpy as np
import pytest

from toyomacro._bootstrap import bootstrap, estimate_gain, fit_poisson_mle


def _edge(n=160, scale=1.0, shift=0.0):
    """Counts per sweep per channel; ``shift`` moves the edge (scalar or per sweep)."""
    x = np.linspace(-1.0, 1.0, n)
    shift = np.atleast_1d(shift)[:, None]
    return scale * (0.04 + 1.2 / (1.0 + np.exp((x - shift) / 0.08)))


def _sweeps(rng, n_sweeps, gain, *, drift=0.10, jitter=0.03, read_sd=0.0, scale=1.0,
            edge_drift=0.0):
    t = np.linspace(0.0, 1.0, n_sweeps)
    mu = _edge(scale=scale, shift=edge_drift * np.sin(2 * np.pi * t))
    factor = (1.0 + drift * np.sin(2 * np.pi * t)) * (1.0 + jitter * rng.standard_normal(n_sweeps))
    counts = rng.poisson(factor[:, None] * mu)
    return gain * counts + read_sd * rng.standard_normal(counts.shape)


@pytest.mark.parametrize("seed", [3, 4, 5])
@pytest.mark.parametrize("gain", [0.58, 2.0])
def test_estimate_gain_recovers_the_gain_through_source_drift(gain, seed):
    """10% slow drift and 3% sweep-to-sweep jitter of the source, and a slow
    drift of the edge position by 0.6 of its width; about 60 counts per
    sweep on the plateau, 1000 sweeps of 160 channels. Both steps are
    needed at this tolerance: without the per-sweep scaling the jitter
    reaches the variance and b comes out 4.4-6.8% high on these seeds,
    without the sweep differences the edge drift does, 4.1-5.1% high.
    The full estimate is within 1.0%."""
    g = estimate_gain(_sweeps(np.random.default_rng(seed), 1000, gain, scale=50.0,
                              edge_drift=0.05))
    assert g.gain == pytest.approx(gain, rel=0.025)


def test_pure_scaled_poisson_has_no_offset():
    """Without read noise the intercept is zero up to sampling (low counts,
    source drift and jitter only)."""
    gain = 0.58
    g = estimate_gain(_sweeps(np.random.default_rng(3), 1000, gain))
    assert abs(g.offset) < 0.02 * gain**2


def test_read_noise_shows_as_the_offset_not_the_gain():
    read_sd = 0.3
    g = estimate_gain(_sweeps(np.random.default_rng(4), 1000, 0.58, read_sd=read_sd))
    assert g.gain == pytest.approx(0.58, rel=0.05)
    assert g.offset == pytest.approx(read_sd**2, rel=0.15)


def _mean_model(theta):
    theta = np.asarray(theta, dtype=float)
    return theta[:, :1].copy(), np.ones((theta.shape[0], 1, 1))


def test_sweep_bootstrap_matches_poisson_for_independent_sweeps():
    """One channel, 200 sweeps of mean 3 counts: the summed-count MLE has
    sd sqrt(600) = 24.5 under Poisson; resampling sweeps gives the same."""
    rng = np.random.default_rng(5)
    sweeps = rng.poisson(3.0, size=(200, 1)).astype(float)
    fit = lambda c: fit_poisson_mle(c, _mean_model, np.array([600.0]))  # noqa: E731
    out = bootstrap(sweeps, fit, 2000, rng=rng, kind="sweep")
    assert out.n_failed == 0
    assert out.sd()[0] == pytest.approx(np.sqrt(sweeps.sum()), rel=0.07)
    assert out.mean_counts.shape == (1,)


def test_sweep_bootstrap_sees_what_poisson_does_not():
    """Each sweep's source intensity varies by 60% (gamma, mean 1), so the
    sum is over-dispersed: variance 200 * (3 + (0.6 * 3)^2) instead of
    200 * 3, a factor sqrt(6.24 / 3) = 1.44 in sd. Only the sweep
    bootstrap widens by it."""
    rng = np.random.default_rng(6)
    factor = rng.gamma(1 / 0.36, 0.36, size=200)
    sweeps = rng.poisson(3.0 * factor[:, None]).astype(float)
    fit = lambda c: fit_poisson_mle(c, _mean_model, np.array([600.0]))  # noqa: E731
    sweep_sd = bootstrap(sweeps, fit, 4000, rng=rng, kind="sweep").sd()[0]
    pois_sd = bootstrap(sweeps.sum(axis=0), fit, 4000, rng=rng, kind="parametric").sd()[0]
    assert pois_sd == pytest.approx(np.sqrt(sweeps.sum()), rel=0.05)
    assert sweep_sd / pois_sd == pytest.approx(np.sqrt(6.24 / 3.0), rel=0.10)


def test_sweep_kind_needs_a_sweep_axis():
    with pytest.raises(ValueError):
        bootstrap(np.ones(5), lambda c: None, 10, rng=np.random.default_rng(0), kind="sweep")
    with pytest.raises(ValueError):
        estimate_gain(np.ones((2, 5)))
