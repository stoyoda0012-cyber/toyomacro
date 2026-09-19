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
    KB_EV,
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
                                dict(convention="XE"), dict(window=(5.0, 5.1))])
def test_bad_arguments_raise(kw):
    with pytest.raises(ValueError):
        fit_fermi_edge(BE, _noisy(9), **kw)
