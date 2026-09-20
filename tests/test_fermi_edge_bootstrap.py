"""Resampling: the Fermi-edge bounds against simulated counts, and the
spread of the package's own least-squares fitter against its sandwich.

The same machinery does both. Drawing from a model's mean at stated
parameters and fitting back is the Monte Carlo check of a bound; drawing
from a fit's mean is the parametric bootstrap. What is compared differs,
not the code.

Pass/fail only at an interior point, where the bound is supposed to be
attained. At a boundary, and where the width cannot be split, the
estimator's distribution is recorded instead: the inverse Fisher matrix
is not its variance there, and a Monte Carlo need not agree with it
(Self & Liang 1987).

To keep the check from being circular -- the fitter used here shares the
model with what it tests -- two things are pinned independently: the
constrained fit against a different optimiser on the same deviance, and
the drawn counts against the Poisson law.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import minimize

from toyomacro._bootstrap import (
    bootstrap,
    draw_poisson,
    fit_poisson_mle,
    poisson_deviance,
    poisson_mle_fitter,
)
from toyomacro.fitting import fermi_edge_identifiability as fi
from toyomacro.fitting.fermi_edge import fit_fermi_edge

ENERGY = np.arange(-0.6, 0.6 + 1e-9, 0.01)
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def _background():
    return fi.constant_background(ENERGY, 50.0)


def _interior_edge():
    """kT/sigma = 1 at 300 K: v is 11 bounds from zero, tau 34, and the
    instrument's share of the width is bounded at 0.021 -- separable, and
    nowhere near either boundary."""
    return fi.FermiEdge(ef=0.0, amplitude=2000.0, sigma=fi.KB_EV * 300.0, temperature=300.0,
                        dos_c1=0.3)


def _run(edge, *, exposure=1.0, n_draws=400, seed=11, mode="free", max_iter=200):
    fisher = fi.edge_fisher(ENERGY, edge, _background(), exposure=exposure,
                            temperature_mode=mode)
    model, names, lower, start = fi.edge_mle_model(ENERGY, edge, _background(),
                                                   exposure=exposure)
    fit = poisson_mle_fitter(model, start, lower, max_iter=max_iter)
    out = bootstrap(fisher.expected_counts, fit, n_draws, rng=np.random.default_rng(seed),
                    names=names)
    return out, np.sqrt(np.diag(np.linalg.inv(fisher.fisher))), fisher


def test_monte_carlo_matches_the_bound_at_an_interior_point():
    """400 draws at exposure 50 (100,000 counts per channel on the plateau).
    Measured ratios of the Monte Carlo spread to the bound: E_F 1.027,
    amplitude 0.980, DOS slope 0.970, v 0.976, tau 0.963, background
    1.007. The sampling error of a standard deviation from 400 draws is
    3.5%, so the band here is +-10%."""
    out, crb, _ = _run(_interior_edge(), exposure=50.0)
    assert out.n_failed == 0
    assert not out.at_bound.any()
    for name, sd, bound in zip(out.names, out.sd(), crb):
        assert 0.90 < sd / bound < 1.10, (name, sd / bound)


def test_the_bound_is_not_the_spread_at_the_boundary(record_property):
    """A pure thermal edge (v = 0) at 300 K. Recorded, not asserted beyond
    the half-mass that Self & Liang predict for one parameter at its
    boundary: about half the draws end with v held at zero (measured
    0.536 over 300 draws) and the spread of the rest is several times the
    bound (3.7)."""
    edge = fi.FermiEdge(ef=0.0, amplitude=1000.0, sigma=1e-4, temperature=300.0, dos_c1=0.3)
    out, crb, _ = _run(edge, n_draws=300, seed=3)
    i = out.names.index("variance")
    at_bound = float(out.at_bound[i])
    ratio = float(out.sd()[i] / crb[i])
    record_property("variance_at_boundary_fraction", at_bound)
    record_property("variance_sd_over_bound", ratio)
    record_property("variance_quantiles_5_50_95", out.quantiles([0.05, 0.5, 0.95])[:, i].tolist())
    assert 0.35 < at_bound < 0.65
    assert ratio > 2.0


def test_where_the_width_cannot_be_split_the_distribution_is_recorded(record_property):
    """kT/sigma = 0.5 with the temperature free: the assessment calls this
    'not_separable', and the Monte Carlo spread of v and tau sits below
    the bound (0.82 and 0.81 measured) because the likelihood is not
    quadratic there. Recorded, not judged."""
    edge = fi.FermiEdge(ef=0.0, amplitude=1000.0, sigma=2.0 * fi.KB_EV * 300.0,
                        temperature=300.0, dos_c1=0.3)
    report = fi.assess_edge_identifiability(ENERGY, edge, _background())
    assert report.separation == "not_separable"
    out, crb, _ = _run(edge, n_draws=300, seed=7)
    for name in ("variance", "tau"):
        i = out.names.index(name)
        record_property(f"{name}_sd_over_bound", float(out.sd()[i] / crb[i]))
        record_property(f"{name}_quantiles_5_50_95",
                        out.quantiles([0.05, 0.5, 0.95])[:, i].tolist())
    assert out.n_failed < 0.1 * out.n_draws


def test_the_constrained_fit_agrees_with_a_different_optimiser():
    """Not circular: the same exact deviance, minimised by Nelder-Mead,
    on two replicas (each takes a few thousand evaluations of the model,
    which is why there are only two)."""
    edge = _interior_edge()
    model, names, lower, start = fi.edge_mle_model(ENERGY, edge, _background(), exposure=10.0)
    fisher = fi.edge_fisher(ENERGY, edge, _background(), exposure=10.0)
    counts = draw_poisson(fisher.expected_counts, 2, np.random.default_rng(23))
    ours = fit_poisson_mle(counts, model, start, lower=lower, max_iter=200)
    assert ours.converged.all()

    def objective(theta, y):
        mean, _ = model(np.atleast_2d(theta))
        if np.any(theta[[1, 3, 4]] < 0.0):
            return 1e300
        return float(poisson_deviance(y[None, :], mean)[0])

    for row in range(counts.shape[0]):
        scale = np.where(np.abs(start) > 0, np.abs(start), 1.0)
        best = minimize(lambda u, r=row: objective(u * scale, counts[r]), start / scale,
                        method="Nelder-Mead",
                        options=dict(xatol=1e-9, fatol=1e-9, maxiter=6000, maxfev=6000))
        theirs = best.x * scale
        assert best.fun >= ours.deviance[row] - 1e-6
        for i, name in enumerate(names):
            assert abs(theirs[i] - ours.params[row, i]) < 0.05 * max(
                abs(ours.params[row, i]), scale[i] * 1e-3), name


def test_the_draws_are_poisson():
    """Independent of any fit: 20,000 draws from a known mean."""
    mean = np.array([1.0, 7.5, 60.0, 500.0])
    counts = draw_poisson(mean, 20000, np.random.default_rng(5))
    assert counts.shape == (20000, mean.size)
    assert np.all(counts == np.rint(counts)) and np.all(counts >= 0)
    # standard errors at 20,000 draws: 0.7% on the mean of the largest, 1% on its variance
    np.testing.assert_allclose(counts.mean(axis=0), mean, rtol=0.03)
    np.testing.assert_allclose(counts.var(axis=0, ddof=1), mean, rtol=0.06)
    with pytest.raises(ValueError, match="non-negative"):
        draw_poisson(np.array([1.0, -1.0]), 3, np.random.default_rng(0))


def test_the_nonparametric_draw_needs_integer_counts():
    edge = _interior_edge()
    fisher = fi.edge_fisher(ENERGY, edge, _background(), exposure=10.0)
    model, names, lower, start = fi.edge_mle_model(ENERGY, edge, _background(), exposure=10.0)
    fit = poisson_mle_fitter(model, start, lower, max_iter=200)
    with pytest.raises(ValueError, match="observed counts"):
        bootstrap(fisher.expected_counts, fit, 2, rng=np.random.default_rng(0),
                  kind="nonparametric")
    with pytest.raises(ValueError, match="kind"):
        bootstrap(fisher.expected_counts, fit, 2, rng=np.random.default_rng(0), kind="wild")

    observed = draw_poisson(fisher.expected_counts, 1, np.random.default_rng(2))[0]
    parametric = bootstrap(fisher.expected_counts, fit, 120, rng=np.random.default_rng(4),
                           names=names)
    nonparametric = bootstrap(observed, fit, 120, rng=np.random.default_rng(4), names=names,
                              kind="nonparametric")
    assert nonparametric.kind == "nonparametric"
    # at ~20,000 counts per channel the two draw from means that differ by 0.7%
    for name in ("variance", "tau", "ef"):
        i = names.index(name)
        assert 0.7 < nonparametric.sd()[i] / parametric.sd()[i] < 1.4, name


def test_the_least_squares_fitter_scatters_as_its_sandwich_says():
    """What V2 rests on. 200 draws at 2000 counts on a background of 50,
    fitted by ``fit_fermi_edge`` with its default unit weights and the
    temperature fixed: the spread of sigma is 1.896e-3 eV, the sandwich
    covariance of that estimator says 1.876e-3 (ratio 1.011), and the
    Poisson bound says 1.424e-3 (ratio 1.331). Pairing a fit like this
    with the bound would understate its error by a third."""
    edge = fi.FermiEdge(ef=0.0, amplitude=2000.0, sigma=0.05, temperature=300.0, dos_c1=0.3)
    background = _background()
    fisher = fi.edge_fisher(ENERGY, edge, background, temperature_mode="fixed_temperature")
    cov, names, label = fi.edge_wls_covariance(ENERGY, edge, background, weights="unit",
                                               temperature_mode="fixed_temperature")
    i = names.index("variance")
    sandwich = math.sqrt(cov[i, i]) / (2.0 * edge.sigma)
    bound = math.sqrt(np.linalg.inv(fisher.fisher)[i, i]) / (2.0 * edge.sigma)

    rng = np.random.default_rng(5)
    counts = draw_poisson(fisher.expected_counts, 200, rng)
    sigmas = []
    for row in counts:
        fit = fit_fermi_edge(ENERGY, row, convention="BE", dos="linear", background="constant",
                             temperature=300.0, resolution=FWHM_PER_SIGMA * edge.sigma)
        if fit.success:
            sigmas.append(fit.resolution / FWHM_PER_SIGMA)
    sigmas = np.asarray(sigmas)
    assert sigmas.size > 190
    spread = sigmas.std(ddof=1)
    assert 0.9 < spread / sandwich < 1.1
    assert spread / bound > 1.2
    assert abs(sigmas.mean() - edge.sigma) < 0.1 * spread * math.sqrt(sigmas.size)
    assert "unit" in label
