"""toyomacro.fitting.fermi_edge: model, recovery, error calibration, invariances.

Implementation fidelity is tested against the existing
``toyomacro.lineshape.FermiDirac`` (same model on a binding-energy axis)
and against exact symmetries of the problem (a shifted axis shifts E_F
by the same amount; a kinetic axis is the mirror of a binding one).
Statistical claims -- that ``ef_err`` is a 1-sigma error -- are tested
on a population of Poisson realisations, not on one seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from toyomacro.fitting.fermi_edge import (
    DOS_FORMS,
    KB_EV,
    compare_dos_forms,
    detect_convention,
    differential_ef,
    fermi_edge,
    fit_fermi_edge,
    to_binding_energy,
)

BE = np.arange(-2.0, 2.0 + 1e-9, 0.02)
TRUTH = dict(ef=0.0, amplitude=2000.0, fwhm_g=0.30, temperature=300.0,
             dos_c1=0.35, dos_c2=0.0, bg_const=100.0, bg_slope=15.0)


def _noisy(seed, energy=BE, truth=TRUTH, convention="BE"):
    clean = fermi_edge(energy, convention=convention, bg_ref=float(np.mean(energy)), **truth)
    return np.random.default_rng(seed).poisson(np.maximum(clean, 0.0)).astype(float)


def _fit(y, energy=BE, **kw):
    kw.setdefault("convention", "BE")
    return fit_fermi_edge(energy, y, weights=1.0 / np.maximum(y, 1.0), **kw)


# --- model ---------------------------------------------------------------


def test_model_matches_lineshape_fermidirac():
    """Same model as the lineshape, away from the axis ends.

    ``FermiDirac`` convolves on the given axis only (clamped ends), which
    :func:`fermi_edge` deliberately does not; they must agree where the
    ends are more than 4 sigma away.
    """
    from toyomacro.lineshape import FermiDirac

    params = dict(amplitude=3.0, fwhm_g=0.25, temperature=450.0, dos_c1=0.4, dos_c2=-0.1)
    ref = FermiDirac().evaluate(BE, center=0.1, **params)
    ours = fermi_edge(BE, ef=0.1, convention="BE", **params)
    inner = np.abs(BE) < 2.0 - 4 * 0.25 / 2.3548
    np.testing.assert_allclose(ours[inner], ref[inner], rtol=1e-6, atol=1e-9)


def test_window_of_a_spectrum_equals_the_spectrum_windowed():
    """Evaluating on a cut axis gives the full spectrum's values there."""
    params = dict(ef=0.0, amplitude=1.0, fwhm_g=0.4, dos_c1=0.6, bg_ref=0.0)
    full = fermi_edge(BE, **params)
    cut = np.abs(BE) <= 1.0
    np.testing.assert_allclose(fermi_edge(BE[cut], **params), full[cut], rtol=1e-9)


def test_unbroadened_edge_is_half_at_ef_and_has_thermal_width():
    e = np.linspace(-0.5, 0.5, 20001)
    y = fermi_edge(e, ef=0.0, amplitude=1.0, fwhm_g=0.0, temperature=300.0)
    assert np.interp(0.0, e, y) == pytest.approx(0.5, abs=1e-6)
    # 10-90% width of a Fermi-Dirac step is 2 ln 9 kB T
    w = np.interp(0.9, y, e) - np.interp(0.1, y, e)
    assert w == pytest.approx(2 * np.log(9) * KB_EV * 300.0, rel=1e-3)


def test_kinetic_axis_is_the_mirror_of_binding():
    hv = 100.0
    y_be = fermi_edge(BE, ef=0.2, fwhm_g=0.3, dos_c1=0.5, convention="BE")
    y_ke = fermi_edge(hv - BE, ef=hv - 0.2, fwhm_g=0.3, dos_c1=0.5, convention="KE")
    np.testing.assert_allclose(y_ke, y_be, rtol=1e-9, atol=1e-12)


# --- recovery and invariances -------------------------------------------------


@pytest.mark.parametrize("dos", ["linear", "quadratic"])
def test_recovers_ef_and_resolution(dos):
    res = _fit(_noisy(1), dos=dos)
    assert res.success
    assert abs(res.ef - TRUTH["ef"]) < 4 * res.ef_err
    assert abs(res.resolution - TRUTH["fwhm_g"]) < 4 * res.resolution_err
    assert np.isnan(res.temperature_err)  # fixed by default


def test_shifting_the_axis_shifts_ef_by_the_same_amount():
    y = _noisy(2)
    a = _fit(y)
    b = _fit(y, energy=BE + 83.7)
    assert b.ef - a.ef == pytest.approx(83.7, abs=1e-5)
    assert b.resolution == pytest.approx(a.resolution, rel=1e-4)


def test_kinetic_fit_equals_binding_fit():
    hv = 1253.6
    y = _noisy(3)
    be_fit = _fit(y, convention="BE")
    ke_fit = _fit(y, energy=hv - BE, convention="KE")
    assert hv - ke_fit.ef == pytest.approx(be_fit.ef, abs=1e-5)
    assert ke_fit.resolution == pytest.approx(be_fit.resolution, rel=1e-4)
    assert ke_fit.ef_err == pytest.approx(be_fit.ef_err, rel=1e-3)


def test_convention_detection():
    y = _noisy(4)
    assert detect_convention(BE, y) == "BE"
    assert detect_convention(1253.6 - BE, y) == "KE"
    assert _fit(y, convention="auto").convention == "BE"


def test_errors_are_one_sigma_over_a_population():
    """Pulls of E_F and the resolution are unit normal on a windowed fit.

    The window matters: before the model padded its convolution grid, a
    cut window biased E_F by -0.4 sigma (mean pull) through the clamped
    ends.
    """
    ef_pulls, res_pulls = [], []
    for seed in range(200):
        res = _fit(_noisy(100 + seed), window=(-1.2, 1.2))
        ef_pulls.append((res.ef - TRUTH["ef"]) / res.ef_err)
        res_pulls.append((res.resolution - TRUTH["fwhm_g"]) / res.resolution_err)
    for pulls in (np.asarray(ef_pulls), np.asarray(res_pulls)):
        # 200 draws of a unit normal: sample std within ~0.1 and mean within
        # ~0.14 of the ideal at 2 sigma
        assert 0.85 < pulls.std() < 1.15
        assert abs(pulls.mean()) < 0.2


def test_flat_dos_on_a_rising_edge_biases_ef_beyond_its_error():
    """``ef_err`` is statistical: it does not cover a wrong DOS model."""
    res = _fit(_noisy(5), dos="flat")
    assert abs(res.ef - TRUTH["ef"]) > 3 * res.ef_err


def test_differential_ef_recovers_a_rigid_shift():
    shifted = dict(TRUTH, ef=0.15)
    out = differential_ef(BE, _noisy(6), BE, _noisy(7, truth=shifted),
                          convention="BE", window=(-1.5, 1.5))
    assert abs(out["delta_ef"] - 0.15) < 4 * out["delta_ef_err"]


def test_fixed_resolution_and_fitted_temperature():
    res = _fit(_noisy(8), fit_resolution=False, resolution=0.30, fit_temperature=True)
    assert res.resolution == 0.30 and np.isnan(res.resolution_err)
    assert np.isfinite(res.temperature_err)


def test_a_parameter_on_its_bound_marks_the_fit_unsuccessful():
    # E_F bounded just past the edge: the solver pins it to the lower bound
    res = _fit(_noisy(10), ef_init=0.55, ef_bounds=(0.5, 0.6))
    assert not res.success
    assert "at a bound: ef" in res.message
    assert _fit(_noisy(10)).success  # and a normal fit is not flagged


def test_a_fit_forced_off_the_edge_is_not_successful():
    """E_F bounded far from the edge ends wherever the platform's solver
    stops -- on a bound, or inside with a 1-sigma error wider than the
    window; either way it must not be reported as a success."""
    res = _fit(_noisy(10), ef_init=1.5, ef_bounds=(1.4, 1.6))
    assert not res.success


def test_fine_steps_still_find_the_edge():
    """The starting E_F smooths over a width in eV, not a point count."""
    # 5 meV steps at ~200 counts: a 5-point smoothing started >0.3 eV off
    # the edge in most seeds and 40% of the fits failed.
    e = np.arange(-1.5, 1.5 + 1e-9, 0.005)
    truth = dict(TRUTH, amplitude=200.0, bg_const=10.0, bg_slope=1.6)
    ok = 0
    for seed in range(20):
        y = _noisy(300 + seed, energy=e, truth=truth)
        res = _fit(y, energy=e)
        ok += res.success and abs(res.ef) < 4 * res.ef_err
    assert ok >= 19


def test_low_temperature_fit_is_not_flagged_as_on_a_bound():
    """T = 10 K sits near the lower end of its [1, 1e4] K range.

    The resolution (1 meV) is well below the thermal 10-90% width
    (3.4 meV) so T is identifiable; the error bound on T keeps the test
    from passing on a degenerate fit with a huge error.
    """
    truth = dict(TRUTH, temperature=10.0, fwhm_g=0.001)
    e = np.arange(-0.04, 0.04 + 1e-9, 0.0005)
    res = _fit(_noisy(11, energy=e, truth=truth), energy=e, fit_resolution=False,
               resolution=0.001, fit_temperature=True, temperature=30.0)
    assert res.success, res.message
    assert res.temperature_err < 2.0
    assert abs(res.temperature - 10.0) < 4 * res.temperature_err


def test_unidentifiable_temperature_is_not_a_success():
    """Resolution 20 meV against a 3.4 meV thermal width: T cannot be fitted."""
    truth = dict(TRUTH, temperature=10.0, fwhm_g=0.02)
    e = np.arange(-0.3, 0.3 + 1e-9, 0.002)
    res = _fit(_noisy(11, energy=e, truth=truth), energy=e, fit_resolution=False,
               resolution=0.02, fit_temperature=True, temperature=50.0)
    assert not res.success


_W = np.linspace(-12.0, 12.0, 24001)
_PHI = np.exp(-0.5 * _W**2) / np.sqrt(2.0 * np.pi) * (_W[1] - _W[0])


def _edge_by_quadrature(energy, ef, fwhm_g, temperature):
    """Flat-DOS edge, amplitude 1, by quadrature over the Gaussian variable.

    The occupation is evaluated at every quadrature node, so nothing is
    tied to the channel grid. The integrand is analytic in a strip of
    half-width pi kT around the real axis, so the trapezoid rule converges
    exponentially; at 3 K and FWHM 0.3 eV the node spacing is half of kT
    and the rule error is below 1e-15 of the step.
    """
    sigma = fwhm_g / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    x = energy[:, None] - ef - sigma * _W[None, :]
    return (0.5 * (1.0 + np.tanh(x / (2.0 * KB_EV * temperature)))) @ _PHI


@pytest.mark.parametrize("temperature", [100.0, 30.0, 10.0, 3.0])
@pytest.mark.parametrize("offset", [0.0, 0.37])
def test_model_follows_ef_and_t_when_kt_is_below_a_channel(temperature, offset):
    """20 meV channels, kT from 0.43 down to 0.013 of a channel, E_F on a
    grid point or 0.37 of a channel off it.

    The occupation used to be sampled on the channel grid before the
    convolution: below about half a channel the model stopped following
    E_F inside a channel (d/dE_F off by 55% at 30 K and 100% at 10 K) and
    T (off by 88% at 30 K). Refined, the errors measured here are at most
    3.2e-5 in the model, 1.7e-4 in d/dE_F and 1.5e-3 in d/dT, over these
    cases and 5 meV channels; the bounds are about three times that.
    """
    e = np.arange(-1.0, 1.0 + 1e-9, 0.02)
    ef, fwhm, h = offset * 0.02, 0.30, 1e-5

    def model(ef_, t_):
        return fermi_edge(e, ef=ef_, fwhm_g=fwhm, temperature=t_)

    def ref(ef_, t_):
        return _edge_by_quadrature(e, ef_, fwhm, t_)

    assert np.max(np.abs(model(ef, temperature) - ref(ef, temperature))) < 1e-4
    d_model = (model(ef + h, temperature) - model(ef - h, temperature)) / (2 * h)
    d_ref = (ref(ef + h, temperature) - ref(ef - h, temperature)) / (2 * h)
    assert np.linalg.norm(d_model - d_ref) / np.linalg.norm(d_ref) < 1e-3
    t_up, t_dn = 1.001 * temperature, 0.999 * temperature
    dt_model = (model(ef, t_up) - model(ef, t_dn)) / (t_up - t_dn)
    dt_ref = (ref(ef, t_up) - ref(ef, t_dn)) / (t_up - t_dn)
    assert np.linalg.norm(dt_model - dt_ref) / np.linalg.norm(dt_ref) < 5e-3


@pytest.mark.parametrize("offset", [0.0, 0.37])
def test_fermi_level_below_a_channel_is_unbiased_with_its_poisson_scatter(offset):
    """10 K on 20 meV channels (kT = 0.04 channel), data drawn from the
    quadrature model, 80 Poisson realisations.

    Before the refinement E_F was pulled toward the nearest grid point:
    0.37 channel off it, bias -4.6 meV and pull width 6.8 over 150
    realisations; on it, a scatter of 0.31 meV, a sixth of the Poisson
    bound, because the model could not move E_F inside a channel. The
    bound here is 1.74 meV (Poisson Fisher of this model with E_F, the
    amplitude, the resolution and the background estimated); after the
    fix the scatter over 150 realisations was 1.00 and 1.08 times it and
    the pulls had unit width.
    """
    e = np.arange(-1.2, 1.2 + 1e-9, 0.02)
    ef, fwhm, temperature = offset * 0.02, 0.30, 10.0
    mean = 2000.0 * _edge_by_quadrature(e, ef, fwhm, temperature) + 100.0
    rng = np.random.default_rng(700 + int(100 * offset))
    est, err = [], []
    for _ in range(80):
        y = rng.poisson(mean).astype(float)
        res = fit_fermi_edge(e, y, convention="BE", dos="flat", background="constant",
                             weights=1.0 / np.maximum(y, 1.0), temperature=temperature,
                             resolution=fwhm)
        est.append(res.ef)
        err.append(res.ef_err)
    est, err = np.asarray(est), np.asarray(err)
    pulls = (est - ef) / err
    # 80 draws: the pull mean has a standard error of 0.11, the sample sd one of about 0.08
    assert abs(pulls.mean()) < 0.35
    assert 0.8 < pulls.std(ddof=1) < 1.25
    assert 0.8 < est.std(ddof=1) / 1.74e-3 < 1.25


def test_width_1090_does_not_depend_on_the_window():
    y = _noisy(12)
    wide = _fit(y, window=(-1.5, 1.5))
    tight = _fit(y, window=(-1.5, 0.1))  # ends just past the edge
    # both are the 10-90% width of their own fitted edge, not of the window
    for res in (wide, tight):
        edge = fermi_edge(np.linspace(-3, 3, 30001), ef=res.ef, fwhm_g=res.resolution)
        u = np.linspace(-3, 3, 30001)
        expected = np.interp(0.9, edge, u) - np.interp(0.1, edge, u)
        assert res.width_1090 == pytest.approx(expected, rel=2e-3)


def test_float32_axis_matches_float64_and_repeats_raise():
    y = _noisy(13)
    a = _fit(y)
    b = _fit(y, energy=BE.astype(np.float32))
    assert b.ef == pytest.approx(a.ef, abs=1e-5)
    with pytest.raises(ValueError, match="repeated"):
        fermi_edge(np.r_[BE, BE[-1]], ef=0.0)


def test_degenerate_low_count_fits_are_not_reported_as_successful():
    """At ~100 counts on an equal background a converged fit can collapse.

    The edge then sits between two channels (FWHM of a few meV, singular
    covariance) and E_F can leave the window; none of that may come back
    as ``success=True``.
    """
    truth = dict(TRUTH, amplitude=100.0, bg_const=100.0, bg_slope=15.0)
    for seed in range(9000, 9400):
        res = _fit(_noisy(seed, truth=truth), window=(-1.2, 1.2))
        if res.success:
            assert np.isfinite(res.ef_err) and np.isfinite(res.resolution_err)
            assert abs(res.ef) < 1.0, (seed, res.ef, res.message)


def test_an_edge_located_less_well_than_its_own_width_is_not_a_success():
    """Seed 9074 of the population above. Once the model could follow a
    resolution below a channel, the fit settled on a noise feature: E_F
    1.14 eV, FWHM 16 meV, and a 1-sigma of 0.26 eV on E_F, twice the
    fitted edge's own 10-90% width. The earlier gates (window, half a
    channel) let it through. Over the 400 seeds the ratio of the E_F error
    to that width is below 0.08 for 99% of the successful fits and above 1
    for two, this one and seed 9315 (E_F 0.89 eV); over the 200-seed pull
    population at 2000 counts it never exceeds 0.009."""
    truth = dict(TRUTH, amplitude=100.0, bg_const=100.0, bg_slope=15.0)
    res = _fit(_noisy(9074, truth=truth), window=(-1.2, 1.2))
    assert not res.success
    assert "10-90% width" in res.message
    assert res.ef_err > res.width_1090


def test_zero_weights_do_not_count_as_degrees_of_freedom():
    y = _noisy(14)
    w = 1.0 / np.maximum(y, 1.0)
    w_masked = w.copy()
    w_masked[:40] = 0.0  # masked channels
    masked = fit_fermi_edge(BE, y, convention="BE", weights=w_masked)
    trimmed = fit_fermi_edge(BE[40:], y[40:], convention="BE", weights=w[40:])
    assert masked.reduced_chi2 == pytest.approx(trimmed.reduced_chi2, rel=1e-3)


# --- helpers and input checks ---------------------------------------------


def test_to_binding_energy():
    np.testing.assert_allclose(to_binding_energy([1.0, 2.0], 0.5, "BE"), [0.5, 1.5])
    np.testing.assert_allclose(to_binding_energy([1253.0, 1250.0], 1253.5, "KE"),
                               [0.5, 3.5])


@pytest.mark.parametrize("kw", [dict(dos="cubic"), dict(background="shirley"),
                                dict(convention="XE"), dict(window=(5.0, 5.1)),
                                dict(dos_form="smooth")])
def test_bad_arguments_raise(kw):
    with pytest.raises(ValueError):
        fit_fermi_edge(BE, _noisy(9), **kw)


# --- the DOS form --------------------------------------------------------


def test_the_default_dos_form_changes_nothing():
    """`dos_form` was added after v0.2.0; its default must leave every
    number exactly where it was. These are values computed before the
    argument existed, compared bit for bit -- not to a tolerance. They
    were also checked against the previous commit's code directly, over
    three parameter sets and all ten scalar fit results.

    **If this test fails, do not update the numbers to make it pass.**
    They pin the behaviour of ``dos_form='occupied'`` as v0.3.0 shipped
    it, and they are meant to fail whenever the model moves -- including
    for changes that are entirely legitimate, such as a finer
    discretisation, a different convolution, or a new default. Replacing
    them is only correct once the change has been confirmed as
    deliberate and recorded in CHANGELOG.md with what moved and by how
    much; the entry for the low-temperature E_F fix is the shape that
    takes. A failure here without such an entry is a regression, not a
    stale constant."""
    model = fermi_edge(BE, ef=0.02, amplitude=2000.0, fwhm_g=0.1137, temperature=560.3,
                       dos_c1=1.5, bg_const=50.0)
    assert np.array_equal(model, fermi_edge(BE, ef=0.02, amplitude=2000.0, fwhm_g=0.1137,
                                            temperature=560.3, dos_c1=1.5, bg_const=50.0,
                                            dos_form="occupied"))
    grid = np.arange(-0.6, 0.6 + 1e-9, 0.01)
    before = np.array([50.00872830450744, 54.333831213484196, 904.1136269069085,
                       2876.712487344952, 3789.964068966334])
    now = fermi_edge(grid, ef=0.02, amplitude=2000.0, fwhm_g=0.1137, temperature=560.3,
                     dos_c1=1.5, bg_const=50.0)[::30]
    assert np.array_equal(now, before)

    counts = np.random.default_rng(3).poisson(
        fermi_edge(grid, ef=0.02, amplitude=2000.0, fwhm_g=0.1137, temperature=560.3,
                   dos_c1=1.5, bg_const=50.0)).astype(float)
    fit = fit_fermi_edge(grid, counts, convention="BE", dos="linear", background="constant",
                         temperature=560.3, resolution=0.1137)
    assert fit.ef == 0.02220324184703952
    assert fit.resolution == 0.12235167495258374
    assert fit.reduced_chi2 == 1119.9023959702502
    assert fit.dos_form == "occupied"


@pytest.mark.parametrize(
    "fwhm_g, temperature, dos_c1, peak_fraction",
    [(0.30, 300.0, 0.35, 5.6e-4),        # kT/sigma = 0.2: the forms are nearly the same edge
     (0.1137, 560.0, 1.5, 1.5e-2)])      # kT/sigma = 1 on a steep DOS: 27 times more
def test_the_two_dos_forms_differ_only_near_ef(fwhm_g, temperature, dos_c1, peak_fraction):
    """They put the same polynomial on different domains, so they can only
    differ where the occupation is neither 0 nor 1. The difference peaks
    two or three kT below E_F -- 0.06 and 0.08 eV in these two cases --
    and is 5.6e-4 and 1.5e-2 of the step height there. Ten kT out it is
    1.8e-4 and 3.5e-5 on the empty side and gone on the occupied one.

    That is also why the choice between them matters only when kT is not
    far below the resolution: at kT/sigma = 0.2 the whole difference is
    below a tenth of a percent of the step."""
    kw = dict(ef=0.0, amplitude=1000.0, fwhm_g=fwhm_g, temperature=temperature,
              dos_c1=dos_c1)
    difference = np.abs(fermi_edge(BE, dos_form="both_sides", **kw)
                        - fermi_edge(BE, dos_form="occupied", **kw))
    kt = KB_EV * temperature
    assert -4.0 * kt < BE[np.argmax(difference)] < 0.0
    assert difference.max() / 1000.0 == pytest.approx(peak_fraction, rel=0.15)
    far = np.abs(BE) > 10.0 * kt
    assert np.max(difference[far]) / 1000.0 < 2e-4


def test_compare_dos_forms_reports_each_fit_and_their_spread():
    """The helper fits every form and reports the spread. It does not
    average them, choose between them, or rank them by fit quality --
    that is the point of it, since fit quality is what cannot decide
    this."""
    counts = _noisy(11)
    out = compare_dos_forms(BE, counts, convention="BE", dos="linear",
                            background="linear", temperature=300.0)
    assert out.forms == DOS_FORMS
    assert set(out.fits) == set(DOS_FORMS)
    for form in DOS_FORMS:
        assert out.fits[form].dos_form == form
        assert out.values["ef"][form] == out.fits[form].ef
    assert out.spread["ef"] == abs(out.values["ef"]["occupied"]
                                   - out.values["ef"]["both_sides"])
    assert out.spread["resolution"] > 0.0
    assert out.success is all(f.success for f in out.fits.values())
    assert "spread" in out.summary()

    # the spread is a systematic, kept apart from the statistical error
    assert not hasattr(out, "combined")
    assert not hasattr(out, "best")


def test_compare_dos_forms_rejects_a_form_it_cannot_set():
    with pytest.raises(TypeError, match="sets dos_form itself"):
        compare_dos_forms(BE, _noisy(12), dos_form="occupied")
    for bad in [("occupied",), ("occupied", "occupied"), ("occupied", "smooth")]:
        with pytest.raises(ValueError, match="forms must be"):
            compare_dos_forms(BE, _noisy(12), forms=bad)


def test_a_failed_fit_marks_the_comparison_unusable():
    """When a fit fails, the spread mixes a systematic with a failure. The
    flag says so; the fits are still returned so the caller can see which
    one went wrong and why. Flat noise with no edge in it fails both
    forms, on E_F and the DOS slope running to their bounds."""
    flat = np.random.default_rng(13).poisson(np.full(BE.size, 300.0)).astype(float)
    out = compare_dos_forms(BE, flat, convention="BE", dos="linear",
                            background="linear", temperature=300.0)
    assert not out.success
    assert all(not f.success for f in out.fits.values())
    assert all("bound" in f.message for f in out.fits.values())
    assert set(out.spread) == set(out.values)


def test_the_poisson_error_is_only_offered_for_counts():
    """The sandwich it carries assumes the intensities are raw Poisson
    counts. Nothing in a float array distinguishes counts from counts per
    second or from a background-subtracted spectrum, and on the wrong
    scale the field is wrong by sqrt(dwell) -- a larger error than the
    9 to 20% it exists to remove. So it is offered only when every
    intensity is a non-negative integer, and is empty otherwise. The
    `*_err` fields, which make no such assumption, are unaffected.

    The check is necessary, not sufficient: an audit noted that counts
    scaled by an integer factor still pass it. Nothing available here
    can tell those apart, which is why the docstring states the
    condition rather than promising to enforce it."""
    counts = _noisy(21)
    assert _fit(counts).poisson_err                       # integers: offered
    assert _fit(counts / 3.0).poisson_err == {}           # counts per second: withheld
    assert _fit(counts - counts.min()).poisson_err        # still integers
    shifted = counts - 0.5                                # background-subtracted
    assert _fit(shifted).poisson_err == {}
    assert not np.isnan(_fit(shifted).ef_err)             # the usual errors still come back
