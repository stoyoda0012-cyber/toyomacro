"""What `fit_fermi_edge` returns when the true density of states is not
the one it assumes.

The fitted model puts the DOS on the occupied side only, so its slope
changes at E_F: ``D(y) = 1 + c1 y`` for y > 0 and ``D = 1`` below. That
kink is an assumption about the sample, and the identifiability scan
shows it is worth a factor of 1.6 to 3 in the standard deviation of the
width split. This file asks the other question: if the truth has no
kink, how wrong are sigma, T and E_F, and by how many error bars.

The truth is generated here and not by either module under test: the
product ``D(y) f(y;T)`` on a dense grid, convolved with a Gaussian by
direct summation, read back at the channels. The 'occupied' form of it
is the control, and must reproduce ``fermi_edge`` -- if it did not, the
bias measured for the other forms would be the generator's.

Every bias here is exact, not a Monte Carlo estimate. ``fit_fermi_edge``
minimises an unweighted sum of squares, and

    E[ sum (y - m(theta))^2 ] = sum (mu - m(theta))^2 + sum Var(y),

whose second term does not depend on theta. The minimiser of the
expected sum of squares is therefore exactly the fit to the noiseless
mean, so a single noiseless fit gives the asymptotic ("pseudo-true")
value the estimator converges to. One Monte Carlo test is kept, to pin
the scale of the error bars that the biases are quoted against.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from toyomacro.fitting import fermi_edge_identifiability as fi
from toyomacro.fitting.fermi_edge import compare_dos_forms, fermi_edge, fit_fermi_edge

KB = fi.KB_EV
FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))

ROOT_K2 = 0.1                                  # sqrt(kappa_2), eV: one edge width
HALF = 6.0 * ROOT_K2                           # window half-width, the scan's middle one
ENERGY = np.arange(-HALF, HALF + 1e-9, 0.01)   # 121 channels
AMPLITUDE = 2000.0                             # counts per channel on the occupied plateau
BG = 50.0

# The two misspecified forms are matched on DELTA, the relative change of the DOS
# at the window edge: c1 * HALF = c2 * HALF**2 = DELTA. DELTA < 1 is forced -- a
# linear DOS through E_F reaches zero at the window edge at DELTA = 1 -- which caps
# the slope at 0.15 per edge width, half of the 0.3 the scan uses.


def truth(energy, ef, amplitude, sigma, temperature, form, delta, *, bg=BG):
    """``D(y) f(y;T)`` convolved with a Gaussian, on a dense grid. BE convention."""
    c1, c2 = delta / HALF, delta / (HALF * HALF)
    kt = KB * temperature
    step = min(sigma, kt) / 50.0
    pad = 10.0 * sigma + 25.0 * kt
    lo, hi = float(np.min(energy)) - pad, float(np.max(energy)) + pad
    fine = lo + step * np.arange(int(math.ceil((hi - lo) / step)) + 1)
    y = fine - ef
    if form == "occupied":                     # what fit_fermi_edge models: a kink at E_F
        dos = 1.0 + c1 * np.maximum(y, 0.0)
    elif form == "both_sides":                 # the same slope through E_F, no kink
        dos = 1.0 + c1 * y
    elif form == "quadratic":                  # smooth curvature through E_F, no kink
        dos = 1.0 + c2 * y * y
    else:
        raise ValueError(form)
    profile = amplitude * np.maximum(dos, 0.0) / (1.0 + np.exp(np.clip(-y / kt, -700, 700)))
    half = int(math.ceil(8.0 * sigma / step))
    kernel = np.exp(-0.5 * (step * np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(profile, kernel, mode="same")
    return np.interp(np.asarray(energy, dtype=float), fine, smooth) + bg


def conditions(ratio):
    """(sigma, T) at this kT/sigma, holding kappa_2 = ROOT_K2**2."""
    var = ROOT_K2**2 / (1.0 + (math.pi**2 / 3.0) * ratio * ratio)
    sigma = math.sqrt(var)
    return sigma, ratio * sigma / KB


def pseudo_true(mean, sigma, temperature, *, dos="linear", free_t=False,
                dos_form="occupied"):
    """The value the estimator converges to: the fit to the noiseless mean."""
    return fit_fermi_edge(ENERGY, mean, convention="BE", dos=dos, background="constant",
                          temperature=temperature, fit_temperature=free_t,
                          resolution=FWHM * sigma, ef_init=0.0, dos_form=dos_form)


def sandwich(sigma, temperature, delta, mode, *, dos="linear"):
    """The error bars a user would quote here if the model were right: the
    sandwich covariance of the unit-weight least-squares estimator."""
    edge = fi.FermiEdge(ef=0.0, amplitude=AMPLITUDE, sigma=sigma, temperature=temperature,
                        dos_c1=delta / HALF, dos=dos)
    cov, names, _ = fi.edge_wls_covariance(
        ENERGY, edge, fi.constant_background(ENERGY, BG), weights="unit",
        temperature_mode=mode)
    out = {"ef": math.sqrt(cov[names.index("ef"), names.index("ef")]),
           "sigma": math.sqrt(cov[names.index("variance"), names.index("variance")])
           / (2.0 * sigma)}
    if "tau" in names:
        i = names.index("tau")
        out["temperature"] = math.sqrt(cov[i, i]) / (2.0 * KB * KB * temperature)
    return out


# ---------------------------------------------------------------------------
# the generator, and the control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.1, 1.0, 3.0])
def test_the_generator_reproduces_the_model_it_is_compared_against(ratio):
    """The 'occupied' truth is exactly what ``fermi_edge`` models. They are
    built differently -- a dense grid and a direct sum here, a refined grid
    and an FFT there -- so their agreement is a check of both. Without it,
    a bias below could be this file's own.

    Measured worst difference on these 10 meV channels: 0.072, 0.121 and
    0.283 counts on an amplitude of 2000, at kT/sigma = 0.1, 1 and 3. The
    largest is ``fermi_edge``'s own grid discretisation, not this
    generator's: on channels 5 times finer it falls to 0.016 counts, and
    20 times finer to 0.010. It is 0.6% of the Poisson noise of a channel
    on the plateau, and 2000 times smaller than the smallest bias this
    file measures."""
    sigma, temperature = conditions(ratio)
    mine = truth(ENERGY, 0.0, AMPLITUDE, sigma, temperature, "occupied", 0.9)
    theirs = fermi_edge(ENERGY, ef=0.0, amplitude=AMPLITUDE, fwhm_g=FWHM * sigma,
                        temperature=temperature, dos_c1=0.9 / HALF, bg_const=BG,
                        convention="BE")
    assert np.max(np.abs(mine - theirs)) < 3e-4 * AMPLITUDE


@pytest.mark.parametrize("ratio", [0.1, 1.0, 3.0])
@pytest.mark.parametrize("delta", [0.1, 0.9])
def test_a_correctly_specified_dos_leaves_no_bias(ratio, delta):
    """The control. Over nine cells (three kT/sigma x three DOS slopes) the
    worst |bias| / sandwich sd is 0.0032 for sigma and 0.0043 for E_F --
    three orders of magnitude below what the misspecified forms give, and
    consistent with the generator's own agreement with ``fermi_edge``
    above. Anything the other forms show below is theirs, not the
    fitter's and not the generator's."""
    sigma, temperature = conditions(ratio)
    mean = truth(ENERGY, 0.0, AMPLITUDE, sigma, temperature, "occupied", delta)
    star = pseudo_true(mean, sigma, temperature)
    sd = sandwich(sigma, temperature, delta, "fixed_temperature")
    assert star.success
    assert abs(star.resolution / FWHM - sigma) < 0.01 * sd["sigma"]
    assert abs(star.ef) < 0.01 * sd["ef"]


# ---------------------------------------------------------------------------
# the kink alone
# ---------------------------------------------------------------------------


def test_the_kink_costs_nothing_when_the_thermal_tail_is_short():
    """'both_sides' differs from the fitted model only where y < 0, which the
    occupation suppresses. At kT/sigma = 0.1 (sigma 98.4 meV, T 114 K) the
    strongest DOS tested moves sigma by -0.02 meV, which is -0.01 of its
    error bar: the assumption is free here."""
    sigma, temperature = conditions(0.1)
    mean = truth(ENERGY, 0.0, AMPLITUDE, sigma, temperature, "both_sides", 0.9)
    star = pseudo_true(mean, sigma, temperature)
    sd = sandwich(sigma, temperature, 0.9, "fixed_temperature")
    assert star.success
    assert abs(star.resolution / FWHM - sigma) < 0.1 * sd["sigma"]


@pytest.mark.parametrize(
    "ratio, delta, expected",           # expected = (sigma* - sigma) / sandwich sd
    [(1.0, 0.1, -0.19), (1.0, 0.3, -0.55), (1.0, 0.9, -1.55),
     (3.0, 0.1, -0.30), (3.0, 0.3, -1.09)])
def test_the_kink_biases_the_resolution_once_the_tail_is_long(ratio, delta, expected):
    """With the temperature fixed at its true value and no kink in the truth,
    the fitted resolution comes out low, and the more so the longer the
    thermal tail and the steeper the DOS. In meV, sigma* - sigma is
    -0.67, -2.07, -6.58 at kT/sigma = 1 (sigma 48.3 meV) and -2.69, -10.3 at
    kT/sigma = 3 (sigma 18.1 meV). In units of the error bar a user would
    quote, the cost passes 1 between a 1.7% and a 5% DOS rise per edge
    width at kT/sigma = 3 (-0.30, -1.09), and between 5% and 15% at
    kT/sigma = 1 (-0.55, -1.55). The three DOS changes tested across the
    window, 0.1, 0.3 and 0.9, are 1.7%, 5% and 15% per edge width."""
    sigma, temperature = conditions(ratio)
    mean = truth(ENERGY, 0.0, AMPLITUDE, sigma, temperature, "both_sides", delta)
    star = pseudo_true(mean, sigma, temperature)
    sd = sandwich(sigma, temperature, delta, "fixed_temperature")
    assert star.success
    assert (star.resolution / FWHM - sigma) / sd["sigma"] == pytest.approx(expected, abs=0.05)


def test_the_strongest_kink_misspecification_is_caught_by_the_success_gate():
    """At kT/sigma = 3 with a 0.9 DOS change across the window, the fitted
    resolution collapses onto its lower bound (0.42 meV against a true
    18.1 meV) rather than coming back merely biased -- and the fit is
    reported as a failure, naming the parameter at a bound. The gate, not
    the number, is what protects a user here."""
    sigma, temperature = conditions(3.0)
    mean = truth(ENERGY, 0.0, AMPLITUDE, sigma, temperature, "both_sides", 0.9)
    star = pseudo_true(mean, sigma, temperature)
    assert not star.success
    assert "bound" in star.message
    assert star.resolution / FWHM < 0.05 * sigma


def test_with_the_temperature_free_the_kink_is_paid_for_in_temperature():
    """The same truth fitted with T free does not collapse: the fit buys the
    missing occupied-side slope with a wider Gaussian and a colder sample.
    At kT/sigma = 3, DOS change 0.9: sigma 18.1 -> 40.2 meV (+0.68 of its
    error bar) and T 629 -> 538 K (-91 K, -2.11 of its error bar). The
    temperature is the parameter that absorbs the DOS error."""
    sigma, temperature = conditions(3.0)
    mean = truth(ENERGY, 0.0, AMPLITUDE, sigma, temperature, "both_sides", 0.9)
    star = pseudo_true(mean, sigma, temperature, free_t=True)
    sd = sandwich(sigma, temperature, 0.9, "free")
    assert star.success
    assert (star.resolution / FWHM - sigma) / sd["sigma"] == pytest.approx(0.68, abs=0.05)
    assert (star.temperature - temperature) / sd["temperature"] == pytest.approx(-2.11, abs=0.05)


# ---------------------------------------------------------------------------
# curvature is a different fault from the kink
# ---------------------------------------------------------------------------


def test_the_quadratic_truth_fails_the_linear_fit_through_its_curvature():
    """A DOS with smooth curvature through E_F is misspecified twice over for
    ``dos='linear'``: no kink, and a shape the linear DOS cannot hold.
    Fitting it with ``dos='quadratic'`` -- which keeps the kink but gains
    the curvature -- removes essentially all of the bias, so what the
    linear fit measures there is the missing curvature, not the kink.
    At kT/sigma = 1 with a 0.9 DOS change: E_F -49.3 meV and the resolution
    on its bound with 'linear', E_F +0.14 meV and sigma +1.3 meV (+0.3 of
    its error bar) with 'quadratic'."""
    sigma, temperature = conditions(1.0)
    mean = truth(ENERGY, 0.0, AMPLITUDE, sigma, temperature, "quadratic", 0.9)
    sd = sandwich(sigma, temperature, 0.9, "fixed_temperature")

    linear = pseudo_true(mean, sigma, temperature, dos="linear")
    assert not linear.success
    assert abs(linear.ef) > 10.0 * sd["ef"]

    quad = pseudo_true(mean, sigma, temperature, dos="quadratic")
    assert quad.success
    assert abs(quad.ef) < 0.2 * sd["ef"]
    assert abs(quad.resolution / FWHM - sigma) < 0.5 * sd["sigma"]


def test_a_quadratic_dos_does_not_repair_the_missing_kink():
    """The complement of the test above, and the reason the two faults are
    separate axes: against the 'both_sides' truth, letting the fit have a
    quadratic term does not recover the resolution -- it makes it slightly
    worse (-6.58 meV with 'linear', -8.09 meV with 'quadratic', at
    kT/sigma = 1 and a 0.9 DOS change). A smoother DOS is not a substitute
    for the right behaviour at E_F."""
    sigma, temperature = conditions(1.0)
    mean = truth(ENERGY, 0.0, AMPLITUDE, sigma, temperature, "both_sides", 0.9)
    linear = pseudo_true(mean, sigma, temperature, dos="linear")
    quad = pseudo_true(mean, sigma, temperature, dos="quadratic")
    assert linear.success and quad.success
    assert quad.resolution / FWHM - sigma < linear.resolution / FWHM - sigma < -0.1 * sigma


# ---------------------------------------------------------------------------
# the bias is measurable without knowing the truth
# ---------------------------------------------------------------------------


def _fit_smooth(counts, sigma, temperature):
    """The shipped smooth-DOS fit: the same estimator as the default, with
    the DOS continued through E_F instead of kinked there. Returns sigma."""
    return pseudo_true(counts, sigma, temperature, dos_form="both_sides").resolution / FWHM


@pytest.mark.parametrize(
    "ratio, delta, kinked_on_smooth",                       # meV
    [(0.1, 0.9, -0.025), (1.0, 0.1, -0.667), (1.0, 0.3, -2.068),
     (1.0, 0.9, -6.584), (3.0, 0.3, -10.259), (3.0, 0.9, -17.650)])
def test_refitting_with_both_dos_forms_measures_the_bias(ratio, delta, kinked_on_smooth):
    """Why ``compare_dos_forms`` exists, and how far it can be trusted.

    The bias is nearly antisymmetric in which form is wrong. Fitting the
    kinked model to a smooth truth gives the values parametrised above;
    fitting the smooth model to a kinked truth gives +0.030, +0.674,
    +2.018, +6.006, +6.487 and +16.280 meV -- the same size with the
    opposite sign in five of the six, and 0.63 of it in the sixth.

    So the spread between the two fits of *one* spectrum stands in for a
    bias nobody can measure without knowing the true DOS: 0.027, 0.663,
    2.007, 5.993, 6.482 and 16.251 meV, which is 63 to 110% of what it
    stands for. The 63% is at kT/sigma = 3 with a DOS changing 30% across
    the window, where the antisymmetry is weakest.

    Both fits succeed in all six cells, so the spread is usable in all
    six; where one of them fails, ``DosFormComparison.success`` is False
    and the spread mixes a systematic with a failure."""
    sigma, temperature = conditions(ratio)
    smooth = truth(ENERGY, 0.0, AMPLITUDE, sigma, temperature, "both_sides", delta)
    kinked = truth(ENERGY, 0.0, AMPLITUDE, sigma, temperature, "occupied", delta)

    on_smooth = pseudo_true(smooth, sigma, temperature).resolution / FWHM - sigma
    on_kinked = _fit_smooth(kinked, sigma, temperature) - sigma
    assert on_smooth * 1e3 == pytest.approx(kinked_on_smooth, abs=0.01)
    assert on_smooth < 0.0 < on_kinked

    comparison = compare_dos_forms(
        ENERGY, kinked, convention="BE", dos="linear", background="constant",
        temperature=temperature, resolution=FWHM * sigma, ef_init=0.0)
    assert comparison.success
    spread = comparison.spread["resolution"] / FWHM
    assert 0.6 < spread / abs(on_smooth) < 1.2
    # the spread is the difference of the two fits, not something else
    values = comparison.values["resolution"]
    assert comparison.spread["resolution"] == pytest.approx(
        abs(values["occupied"] - values["both_sides"]), rel=1e-12)
    assert comparison.fits["occupied"].resolution == values["occupied"]


# ---------------------------------------------------------------------------
# what the biases are quoted against
# ---------------------------------------------------------------------------


def test_the_sandwich_is_the_right_scale_and_the_reported_ef_error_is_not():
    """100 Poisson realisations of the correctly specified truth at
    kT/sigma = 1, fitted with the temperature fixed. Normalised by the
    sandwich covariance the pulls are 0.02 +- 0.99 (E_F) and -0.09 +- 0.92
    (sigma): the sandwich is the right size, which is why the biases above
    are quoted against it. Normalised by what ``fit_fermi_edge`` itself
    reports, the E_F pull is 0.02 +- 1.18: the reported E_F error is too
    small, because it is the unweighted least-squares covariance scaled
    by the reduced chi-squared, not the sandwich. Over nine correctly
    specified cells of 200 realisations (three kT/sigma x three DOS
    slopes, temperature fixed) the E_F pull sd runs 1.15 to 1.29 -- the
    reported error is 13 to 22% smaller than the scatter it describes --
    while the sandwich gives 0.93 to 1.08. For sigma the reported error is
    within 20% either way (pull sd 0.76 to 1.06). The mechanism is pinned
    separately, below."""
    sigma, temperature = conditions(1.0)
    mean = truth(ENERGY, 0.0, AMPLITUDE, sigma, temperature, "occupied", 0.3)
    sd = sandwich(sigma, temperature, 0.3, "fixed_temperature")
    rng = np.random.default_rng(31)
    ef, res, ef_err = [], [], []
    for _ in range(100):
        f = pseudo_true(rng.poisson(mean).astype(float), sigma, temperature)
        if f.success:
            ef.append(f.ef)
            res.append(f.resolution / FWHM)
            ef_err.append(f.ef_err)
    ef, res, ef_err = np.array(ef), np.array(res), np.array(ef_err)
    assert ef.size > 90
    # sampling error of a standard deviation from 100 draws is 7%; the band is +-25%
    assert 0.75 < np.std(ef / sd["ef"], ddof=1) < 1.25
    assert 0.75 < np.std((res - sigma) / sd["sigma"], ddof=1) < 1.25
    assert abs(np.mean(ef / sd["ef"])) < 0.3
    assert np.std(ef / ef_err, ddof=1) > 1.05     # the reported error is too small


@pytest.mark.parametrize("ratio", [0.1, 1.0, 3.0])
def test_why_the_reported_ef_error_is_small(ratio):
    """No Monte Carlo: the two covariances side by side. What a
    least-squares solver reports is the unweighted covariance scaled by one
    number, the reduced chi-squared -- which is the right answer only if
    every channel had the same variance. Here the mean runs from 50 counts
    on the background to 2650 on the plateau, a factor of 53, and E_F's
    Jacobian sits where the mean is above its average. Replacing
    ``diag(mu)`` in the sandwich by its mean shrinks sd(E_F) by a factor
    1.19 to 1.21 and leaves sd(sigma) within 1.2%, which is what the pulls
    above measure (1.15 to 1.29 for E_F, 0.76 to 1.06 for sigma)."""
    sigma, temperature = conditions(ratio)
    edge = fi.FermiEdge(ef=0.0, amplitude=AMPLITUDE, sigma=sigma, temperature=temperature,
                        dos_c1=0.3 / HALF)
    fisher = fi.edge_fisher(ENERGY, edge, fi.constant_background(ENERGY, BG),
                            temperature_mode="fixed_temperature")
    jac, mean = fisher.jacobian, fisher.expected_counts
    inner = np.linalg.inv(jac.T @ jac)
    sandwich_cov = inner @ (jac.T @ (jac * mean[:, np.newaxis])) @ inner
    equal_variance = inner * mean.mean()
    i = fisher.param_names.index("ef")
    v = fisher.param_names.index("variance")
    assert mean.max() / mean.min() == pytest.approx(53.0, abs=1.0)
    assert 1.15 < math.sqrt(sandwich_cov[i, i] / equal_variance[i, i]) < 1.25
    assert 0.98 < math.sqrt(sandwich_cov[v, v] / equal_variance[v, v]) < 1.02


@pytest.mark.parametrize("ratio", [0.1, 1.0, 3.0])
def test_the_poisson_error_field_describes_the_scatter(ratio):
    """``FermiEdgeResult.poisson_err`` against 150 realisations, temperature
    fixed. Over 300 realisations at each of the three kT/sigma the scatter
    is 1.27, 1.22 and 1.09 times ``ef_err`` -- the reported error is 8 to
    21% small -- and 1.04, 1.00 and 0.88 times ``poisson_err['ef']``. For
    the resolution the two are within 20% of the scatter and of each
    other. Fewer realisations here, so the band is wider."""
    sigma, temperature = conditions(ratio)
    mean = truth(ENERGY, 0.0, AMPLITUDE, sigma, temperature, "occupied", 0.3)
    rng = np.random.default_rng(23)
    ef, reported, poisson = [], [], []
    for _ in range(150):
        f = pseudo_true(rng.poisson(mean).astype(float), sigma, temperature)
        if f.success:
            ef.append(f.ef)
            reported.append(f.ef_err)
            poisson.append(f.poisson_err["ef"])
            assert set(f.poisson_err) == {"ef", "amplitude", "resolution", "dos_c1", "bg_const"}
    spread = np.std(ef, ddof=1)
    assert len(ef) > 100
    # sampling error of a standard deviation from ~150 draws is 6%
    assert spread / np.median(poisson) == pytest.approx(1.0, abs=0.25)
    assert spread / np.median(reported) > spread / np.median(poisson)
