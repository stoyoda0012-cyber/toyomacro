"""Resampling: the Fermi-edge bounds against simulated counts, and the
spread of the package's own least-squares fitter against its sandwich.

The same machinery does both. Drawing from a model's mean at stated
parameters and fitting back is the Monte Carlo check of a bound; drawing
from a fit's mean is the parametric bootstrap. What is compared differs,
not the code.

A third question is nested inside those: whether an interval a user
would actually quote holds the truth. A Monte Carlo compares a spread
with a bound; it says nothing about coverage. So each Monte Carlo
replica is bootstrapped in turn and its 95% percentile interval checked
against the true value.

Pass/fail only at an interior point, where the bound is supposed to be
attained. At a boundary, and where the width cannot be split, the
estimator's distribution is recorded instead: the inverse Fisher matrix
is not its variance there, and a Monte Carlo need not agree with it
(Self & Liang 1987).

To keep the checks from being circular -- the fitter used here shares
the model with what it tests -- four things are pinned independently:
the constrained fit against a different optimiser on the same deviance,
a replica held at a bound against the Karush-Kuhn-Tucker condition, the
deviance against its own value rather than a difference of it, and the
drawn counts against the Poisson law.
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
    bound (3.7).

    A nested bootstrap at the same point (200 trials x 200 draws,
    binomial se 1.56%) shows what that does to an interval a user would
    quote. 98.5% of the 95% percentile intervals for v begin at the
    bound, so they hold the true zero 98.5% of the time (97.4%
    nonparametric) while being 4.5 times wider than the Cramer-Rao
    interval -- coverage bought with width, not with information. tau,
    which takes up what v cannot, covers 83.7% and 88.3%, well under
    nominal. The four parameters away from the boundary cover 91.3 to
    96.4% either way.

    (Those three sets were measured before the quantile convention was
    corrected, with an interval nominally 94.05% rather than 95% at 200
    draws. They are recorded and not judged, so they were not
    re-measured; read each as about a point low.)"""
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
    quadratic there. Recorded, not judged.

    A nested bootstrap at the same point (200 x 200, se 1.56%): E_F, the
    amplitude, the DOS slope and the background cover 92.9 to 95.4%
    either way, but v and tau cover 95.4% parametric against 86.2%
    nonparametric -- the one place measured where the two versions part
    company. Their bootstrap sd is 0.70 to 0.78 of the Monte Carlo sd,
    and a third to a half of the intervals begin at the bound.

    (Those three sets were measured before the quantile convention was
    corrected, with an interval nominally 94.05% rather than 95% at 200
    draws. They are recorded and not judged, so they were not
    re-measured; read each as about a point low.)"""
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


def _nested(edge, n_mc, n_boot, *, exposure=1.0, kind="parametric", seed=41, max_iter=200):
    """MC draw -> fit -> bootstrap that fit -> 95% percentile interval.

    Returns (coverage per parameter, bootstrap sd / Monte Carlo sd per
    parameter, parameter names, fraction of intervals whose lower end sits
    on a bound).
    """
    bg = _background()
    fisher = fi.edge_fisher(ENERGY, edge, bg, exposure=exposure)
    model, names, lower, start = fi.edge_mle_model(ENERGY, edge, bg, exposure=exposure)
    truth = np.asarray(start, dtype=float)
    rng = np.random.default_rng(seed)

    counts = draw_poisson(fisher.expected_counts, n_mc, rng)
    first = fit_poisson_mle(counts, model, truth, lower=lower, max_iter=max_iter)
    source = np.maximum(model(first.params)[0], 1e-12) if kind == "parametric" else counts
    draws = rng.poisson(np.repeat(source, n_boot, axis=0)).astype(np.float64)
    out = fit_poisson_mle(draws, model, np.repeat(first.params, n_boot, axis=0),
                          lower=lower, max_iter=max_iter)
    est = np.where(out.converged.reshape(n_mc, n_boot)[:, :, None],
                   out.params.reshape(n_mc, n_boot, -1), np.nan)
    # 'weibull' puts order statistic k at k/(n_boot+1), so the interval is
    # nominally 95% at any n_boot; numpy's default 'linear' is 94.05% at 200
    # draws and 94.81% at 1000, which reads as a bootstrap defect and is not.
    lo = np.nanquantile(est, 0.025, axis=1, method="weibull")
    hi = np.nanquantile(est, 0.975, axis=1, method="weibull")
    use = first.converged
    ratio = (np.nanstd(est[use], axis=1, ddof=1).mean(axis=0)
             / first.params[use].std(axis=0, ddof=1))
    return (((lo <= truth) & (truth <= hi))[use].mean(axis=0), ratio, names,
            (lo[use] <= 1e-12).mean(axis=0))


@pytest.mark.parametrize("kind", ["parametric", "nonparametric"])
def test_the_bootstrap_interval_covers_the_truth_at_an_interior_point(kind):
    """The nested question A9 did not answer: A9 compared the spread of the
    estimator with a bound, which says nothing about whether an interval a
    user would actually quote holds the truth.

    Measured at exposure 50 with 200 bootstrap draws, pooled over three
    independent sets of 200 Monte Carlo trials (600 trials, binomial
    standard error 0.89%):

        param       parametric   nonparametric
        E_F         94.33        94.67
        amplitude   95.00        94.83
        DOS slope   95.17        95.67
        v           96.50        96.00
        tau         95.83        96.00
        background  94.50        94.17

    All twelve within 1.7 standard errors of nominal, six above and six
    below; the bootstrap sd is 0.97 to 1.06 of the Monte Carlo sd.

    An earlier version of this file reported 92.5 to 95.1% here and
    blamed a percentile interval's first-order accuracy. An audit found
    the real cause: ``np.quantile``'s default method interpolates at
    index ``q(B-1)``, which at q = 0.025 and B = 200 reads the 5.975-th
    of 200 order statistics, whose expected position is 5.975/201 =
    2.97%. That interval is nominally 94.05%, not 95%. The experiment
    meant to rule the draw count out -- holding the trials fixed and
    raising B to 1000 -- measured the nominal level climbing toward 95%
    and recorded it as "moves nothing downward"; it could only have
    failed in one direction. ``_nested`` now uses the (B+1) plotting
    position, nominally 95% at any B.

    What runs here is 20 x 40, where the standard error on a coverage is
    4.9% and a Monte Carlo sd from 20 trials carries 16% of its own, on a
    fixed seed: it pins the machinery, coverage above 0.75 and a sd ratio
    within a third of 1, not the numbers above."""
    covered, ratio, names, _ = _nested(_interior_edge(), 20, 40, exposure=50.0, kind=kind)
    for name, c, r in zip(names, covered, ratio):
        assert c > 0.75, (name, c)
        assert 0.67 < r < 1.5, (name, r)


def test_the_deviance_is_the_exact_poisson_one_and_not_a_shifted_copy():
    """Deliberate-defect testing found this hole. Every consumer of
    ``poisson_deviance`` reads only differences of it on the same counts --
    the fitter's accept test, and the cross-check below against a different
    optimiser, which minimises this same function and so carries the same
    offset on both sides. A version that drops the saturated term is
    therefore off by a constant that depends only on the data, returns
    negative deviances, and passed the entire suite. The value itself has
    to be pinned. ``voigtfit.model_selection`` has its own copy of this
    function and its own value test; these are the same checks for this
    one."""
    y = np.array([[4.0, 9.0, 16.0]])
    assert poisson_deviance(y, y)[0] == pytest.approx(0.0, abs=1e-12)   # saturated
    # 2 sum[mu - y + y log(y/mu)] at mu = 1.3 y is 2 (0.3 - log 1.3) sum(y)
    assert poisson_deviance(y, 1.3 * y)[0] == pytest.approx(
        2.0 * (0.3 - math.log(1.3)) * y.sum(), rel=1e-12)
    assert poisson_deviance(y, 0.7 * y)[0] > 0.0                        # and never negative
    # an empty channel contributes its mean alone
    assert poisson_deviance(np.zeros((1, 2)), np.array([[2.0, 3.0]]))[0] == pytest.approx(10.0)
    assert np.isinf(poisson_deviance(y, np.array([[1.0, 0.0, 1.0]]))[0])


def test_a_replica_held_at_a_bound_satisfies_the_kkt_condition():
    """Also from deliberate-defect testing: removing the active set and
    leaving the clip to enforce the bound left the cross-check against a
    different optimiser passing, because at an interior point nothing is
    held and there is nothing for an active set to do. This checks the
    boundary on its own terms and without a second optimiser -- where v
    comes back at zero, raising it must not lower the deviance, and the
    replica must be reported as converged rather than merely stopped."""
    edge = fi.FermiEdge(ef=0.0, amplitude=1000.0, sigma=1e-4, temperature=300.0, dos_c1=0.3)
    fisher = fi.edge_fisher(ENERGY, edge, _background())
    model, names, lower, start = fi.edge_mle_model(ENERGY, edge, _background())
    counts = draw_poisson(fisher.expected_counts, 12, np.random.default_rng(19))
    out = fit_poisson_mle(counts, model, start, lower=lower, max_iter=200)
    i = names.index("variance")
    held = out.at_bound[:, i] & out.converged
    assert held.sum() >= 3, out.at_bound[:, i].sum()

    rows = np.flatnonzero(held)
    base = poisson_deviance(counts[rows], model(out.params[rows])[0])
    for step in (1e-8, 1e-6, 1e-4):
        moved = out.params[rows].copy()
        moved[:, i] += step
        assert np.all(poisson_deviance(counts[rows], model(moved)[0]) >= base - 1e-9)
    assert np.all(out.params[rows, i] == 0.0)


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


def test_a_start_per_replica_survives_chunking():
    """A nested bootstrap starts every draw at the fit it came from, so the
    start is one row per replica. Found by writing one: the chunked path
    handed each chunk the whole start array, which raised a broadcasting
    error as soon as the batch was larger than a chunk. Seven replicas in
    chunks of three must give the same fits as seven in one chunk."""
    edge = _interior_edge()
    fisher = fi.edge_fisher(ENERGY, edge, _background(), exposure=10.0)
    model, _, lower, start = fi.edge_mle_model(ENERGY, edge, _background(), exposure=10.0)
    counts = draw_poisson(fisher.expected_counts, 7, np.random.default_rng(13))
    rng = np.random.default_rng(14)
    starts = start * rng.uniform(0.97, 1.03, size=(7, start.size))
    whole = fit_poisson_mle(counts, model, starts, lower=lower, chunk=99)
    cut = fit_poisson_mle(counts, model, starts, lower=lower, chunk=3)
    np.testing.assert_array_equal(cut.params, whole.params)
    np.testing.assert_array_equal(cut.converged, whole.converged)
    assert whole.converged.all()
    with pytest.raises(ValueError, match="one start per replica"):
        fit_poisson_mle(counts, model, starts[:3], lower=lower)


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


def test_low_counts_do_not_split_the_two_kinds_where_the_parameter_is_determined():
    """A nonparametric draw resamples the observed counts, so a channel that
    came back empty stays empty in every draw, and the spread can come out
    too small. Measured at 1/5000 of the interior exposure -- 0.5 counts
    per channel on the background, 20 on the plateau, 25% of channels
    zero, 200 trials x 200 draws -- the two versions agree to within 3%
    on E_F, the amplitude, the DOS slope and the background, and cover
    93.2 to 95.5% (with the pre-correction quantile convention, about a
    point low; recorded, not judged). They part on v and tau, where the
    nonparametric sd is
    12% and 10% below the parametric one; but 98% and 81% of those
    intervals begin at the bound, so there the effect cannot be told
    apart from the truncation. The anticipated under-statement is not
    visible in a parameter that is actually determined.

    Those numbers are conditional on the fit converging: at this exposure
    161 and 154 of 200 trials did. What runs here is the cheap half of
    the comparison -- that the two versions agree on a single low-count
    spectrum -- not the nested coverage."""
    edge = fi.FermiEdge(ef=0.0, amplitude=2000.0, sigma=fi.KB_EV * 300.0, temperature=300.0,
                        dos_c1=0.3)
    fisher = fi.edge_fisher(ENERGY, edge, _background(), exposure=0.01)
    model, names, lower, start = fi.edge_mle_model(ENERGY, edge, _background(), exposure=0.01)
    observed = draw_poisson(fisher.expected_counts, 1, np.random.default_rng(6))[0]
    assert np.mean(observed == 0.0) > 0.15
    fit = poisson_mle_fitter(model, start, lower, max_iter=200)
    parametric = bootstrap(fisher.expected_counts, fit, 80, rng=np.random.default_rng(8),
                           names=names)
    nonparametric = bootstrap(observed, fit, 80, rng=np.random.default_rng(8), names=names,
                              kind="nonparametric")
    for name in ("ef", "amplitude", "dos_c1", "bg_level"):
        i = names.index(name)
        assert 0.6 < nonparametric.sd()[i] / parametric.sd()[i] < 1.6, name


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
