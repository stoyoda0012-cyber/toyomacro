"""
Tests for the Voigt width identifiability diagnostics
=====================================================

Two kinds of assertion, kept apart:

- *Identities* that hold exactly or to rounding: the heat equation, the
  v = 0 limit, parity on a symmetric window, linearity in exposure, the
  Loewner order of effective and conditional information, the Hessian of
  the expected Poisson deviance.
- *Invariances* of the assessment: it must not move with the energy unit,
  and its labels must move with exposure the way the docstring says.
- *Measured numbers* with a written tolerance: the agreement of the two
  derivative routes across the switch, the Olivero-Longbothum error, the
  conditioning against window width.
"""

import json
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from toyomacro.lineshape import Voigt
from toyomacro.voigtfit import identifiability as idf
from toyomacro.voigtfit.identifiability import (
    PARAMETERIZATIONS,
    Background,
    IdentifiabilityThresholds,
    ScanGrid,
    VoigtPeak,
    assess_identifiability,
    constant_background,
    effective_information,
    linear_background,
    poisson_fisher,
    scan_identifiability,
    shirley_background,
    voigt_derivatives,
    voigt_fwhm,
)
from toyomacro.voigtfit.voigt_jacobian import voigt_with_hessian

GAMMA = 0.3
AMP = 1.0e5


def _close(a, b, rtol):
    """Relative agreement, judged against the largest magnitude in ``b``.

    Derivatives cross zero inside any window, so a pointwise relative
    tolerance is meaningless at those points.
    """
    return np.allclose(a, b, rtol=rtol, atol=rtol * np.max(np.abs(b)))


def _quadrature_reference(x, variance, gamma):
    """Independent route: trapezoid rule on the convolution integral.

    V = int L(x - sigma t) phi(t) dt and its derivatives, from derivatives
    of the Lorentzian under the integral. The integrand is analytic in a
    strip of half-width gamma/sigma, so the trapezoid rule converges like
    exp(-2 pi (gamma/sigma) / h); with h <= gamma / (8 sigma) that is
    exp(-50). No Faddeeva function and no expansion in the variance.
    """
    sigma = math.sqrt(variance)
    h = min(0.25, gamma / (8.0 * sigma))
    t = np.arange(-13.0, 13.0 + h / 2, h)
    phi = np.exp(-0.5 * t * t) / math.sqrt(2 * math.pi) * h
    u = (x[:, None] - sigma * t[None, :]) - 1j * gamma
    value = ((1 / u).imag @ phi) / math.pi
    q = (1 / u**2) @ phi
    d_variance = ((1 / u**3).imag @ phi) / math.pi
    return value, q.imag / math.pi, d_variance, q.real / math.pi


def _canonical_from_widths(name, widths, variance):
    """Inverse of ``idf._width_coordinates``, written independently of it.

    ``variance`` is only used by 'fixed_instrument': the fixed Gaussian
    variance, or the floor when the excess over it is a coordinate.
    """
    if name == "sigma_gamma":
        return float(widths[0]) ** 2, float(widths[1])
    if name == "var_gamma":
        return float(widths[0]), float(widths[1])
    if name == "fwhm_shape":
        fwhm, r = float(widths[0]), float(widths[1])
        f_g_squared = (fwhm - 0.5346 * r * fwhm) ** 2 - 0.2166 * (r * fwhm) ** 2
        return f_g_squared / (8 * math.log(2)), 0.5 * r * fwhm
    if name == "fixed_instrument":
        if len(widths) == 2:  # with a variance floor: (var_extra, gamma)
            return variance + float(widths[0]), float(widths[1])
        return variance, float(widths[0])
    raise ValueError(name)


def _corr_cond(fisher):
    d = np.sqrt(np.diag(fisher))
    ev = np.linalg.eigvalsh(fisher / np.outer(d, d))
    return ev[-1] / ev[0]


def _is_psd(m, scale):
    return np.linalg.eigvalsh(0.5 * (m + m.T)).min() >= -1e-10 * scale


# ===================================================================
# Derivatives
# ===================================================================


class TestVoigtDerivatives:
    @pytest.mark.parametrize("ratio", [0.3, 1.0, 3.0])
    def test_matches_the_engine_jacobian_where_that_is_valid(self, ratio):
        """sigma/gamma >= 0.3, |x| <= 10 gamma: the legacy formulas are fine."""
        sigma = ratio * GAMMA
        energy = np.linspace(-3.0, 3.0, 601)
        d = voigt_derivatives(energy, 0.1, sigma**2, GAMMA)
        V, dV_dc, dV_dsigma, dV_dgamma, d2V_dc2 = voigt_with_hessian(energy, 0.1, sigma, GAMMA)
        assert _close(d.value, V, 1e-12)
        assert _close(d.d_center, dV_dc, 1e-11)
        assert _close(d.d_gamma, dV_dgamma, 1e-10)
        # chain rule and heat equation, two routes to the same number
        assert _close(d.d_variance, dV_dsigma / (2 * sigma), 1e-9)
        assert _close(d.d_variance, 0.5 * d2V_dc2, 1e-9)

    def test_zero_variance_is_the_lorentzian_exactly(self):
        x = np.linspace(-10.0, 10.0, 2001)
        d = voigt_derivatives(x, 0.0, 0.0, GAMMA)
        den = x * x + GAMMA**2
        lorentz = GAMMA / (math.pi * den)
        half_curvature = GAMMA * (3 * x * x - GAMMA**2) / (math.pi * den**3)
        assert np.allclose(d.value, lorentz, rtol=1e-14, atol=0)
        assert _close(d.d_variance, half_curvature, 1e-14)
        assert _close(d.d_center, 2 * x * GAMMA / (math.pi * den**2), 1e-14)
        assert _close(d.d_gamma, (x * x - GAMMA**2) / (math.pi * den**2), 1e-14)

    @pytest.mark.parametrize("ratio", [1e-4, 1e-2, 0.1, 0.3, 1.0])
    def test_against_quadrature_of_the_convolution(self, ratio):
        """All four outputs, window +-33 gamma, against an independent route.

        Tolerance 1e-9: the measured worst case is the variance derivative
        on the Faddeeva route just below the switch, 3e-11 (module
        docstring); 1e-9 leaves a factor 30 and is still more than two
        orders below the 4.5e-7 that a second-order expansion leaves in
        that derivative at sigma/gamma = 0.01.
        """
        x = np.linspace(-10.0, 10.0, 801)
        variance = (ratio * GAMMA) ** 2
        d = voigt_derivatives(x, 0.0, variance, GAMMA)
        value, d_x, d_variance, d_gamma = _quadrature_reference(x, variance, GAMMA)
        assert _close(d.value, value, 1e-9)
        assert _close(d.d_center, d_x, 1e-9)
        assert _close(d.d_variance, d_variance, 1e-9)
        assert _close(d.d_gamma, d_gamma, 1e-9)

    @pytest.mark.parametrize("ratio", [0.11, 0.3, 1.0, 3.0])
    def test_routes_agree_on_both_sides_of_the_switch(self, ratio):
        """|z| from 6.5 to 12 straddles the switch at 7.

        Both routes are usable there: the expansion is at 5e-11 just below
        7, the Faddeeva form at 9e-11 just above. min |z| = gamma / (sigma
        sqrt 2), so sigma/gamma must exceed 0.109 to reach |z| = 6.5 at all.
        """
        sigma = ratio * GAMMA
        z_abs = np.linspace(6.5, 12.0, 45)
        x = np.sqrt((z_abs * sigma * math.sqrt(2)) ** 2 - GAMMA**2)
        series = voigt_derivatives(x, 0.0, sigma**2, GAMMA, route="series")
        faddeeva = voigt_derivatives(x, 0.0, sigma**2, GAMMA, route="faddeeva")
        auto = voigt_derivatives(x, 0.0, sigma**2, GAMMA)
        for s, f, a in zip(series, faddeeva, auto):
            assert _close(s, f, 1e-9)
            assert _close(a, f, 1e-9)

    def test_term_count_stays_below_the_turning_point_of_the_series(self):
        """N < |z_c|**2: every term summed is on the shrinking side.

        Asserted on the constants because the accuracy tests cannot see a
        mild violation: at the shipped switch the terms near n = |z|**2
        are of order exp(-49), and the measured error is at rounding
        level for every N from 24 to 60. What the accuracy tests do catch
        is the other way of breaking it, a switch lowered to |z| = 4
        (16 < 24), and a gross excess of terms (N >= 100).
        """
        assert idf._SERIES_TERMS < idf._SERIES_SWITCH_Z**2

    def test_expansion_is_asymptotic_more_terms_eventually_hurt(self):
        """At |z| = 4 terms grow once n > 16; the error has a floor, then diverges.

        d_variance at sigma/gamma = 1 against the quadrature reference.
        Measured: 1.9e-3 (N=4), 2.9e-5 (N=12, the best), 2.5e-3 (N=24),
        3e2 (N=40). The floor is why the switch cannot be lowered, and the
        divergence is why the term count is bounded by |z_c|**2.
        """
        sigma = GAMMA
        z_abs = np.linspace(4.0, 4.08, 9)
        x = np.sqrt((z_abs * sigma * math.sqrt(2)) ** 2 - GAMMA**2)
        reference = _quadrature_reference(x, sigma**2, GAMMA)[2]

        def error(n_terms):
            got = idf._derivatives_series(x, sigma**2, GAMMA, n_terms=n_terms)[2]
            return np.max(np.abs(got - reference)) / np.max(np.abs(reference))

        assert error(12) < error(4) < 1e-2
        assert 1e-6 < error(12) < 1e-4  # a floor of order exp(-16), not rounding
        assert error(24) > 10 * error(12)
        assert error(40) > 1.0

    def test_overridden_term_count_matches_the_shipped_one(self):
        x = np.linspace(3.0, 10.0, 50)
        shipped = idf._derivatives_series(x, 0.01, GAMMA)
        explicit = idf._derivatives_series(x, 0.01, GAMMA, n_terms=idf._SERIES_TERMS)
        for a, b in zip(shipped, explicit):
            assert np.array_equal(a, b)

    def test_broadcasts_over_profiles(self):
        """(n, 1) parameters against an (n_energy,) axis: n profiles at
        once, each point on its own route, identical to the scalar calls."""
        energy = np.linspace(-6.0, 6.0, 241)
        center = np.array([0.0, 0.3, -1.0, 0.0])
        variance = np.array([0.0, 1e-6, 0.04, 1.0])
        gamma = np.array([GAMMA, GAMMA, 0.1, 0.0])
        batch = voigt_derivatives(energy, center[:, None], variance[:, None], gamma[:, None])
        for k in range(4):
            single = voigt_derivatives(energy, center[k], variance[k], gamma[k])
            for a, b in zip(batch, single):
                assert a.shape == (4, 241)
                assert np.array_equal(a[k], b)
        with pytest.raises(ValueError):
            voigt_derivatives(energy, 0.0, np.array([[0.1], [-0.1]]), GAMMA)

    def test_rejects_points_outside_the_domain(self):
        x = np.linspace(-1, 1, 5)
        with pytest.raises(ValueError):
            voigt_derivatives(x, 0.0, -1e-6, GAMMA)
        with pytest.raises(ValueError):
            voigt_derivatives(x, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            voigt_derivatives(x, 0.0, 0.0, GAMMA, route="faddeeva")
        with pytest.raises(ValueError):
            voigt_derivatives(x, 0.0, 0.01, GAMMA, route="taylor")


class TestVoigtFwhm:
    def test_fwhm_is_the_public_lineshape_expression(self):
        for sigma, gamma in ((0.4247, 0.125), (0.0, 0.3), (0.2, 0.0)):
            fwhm_g = math.sqrt(8 * math.log(2)) * sigma
            assert voigt_fwhm(sigma**2, gamma) == pytest.approx(
                Voigt().fwhm(fwhm_g=fwhm_g, fwhm_l=2 * gamma), rel=1e-14
            )

    def test_fwhm_error_against_the_exact_half_maximum(self):
        """The number the docstrings quote: 2.37e-4 at f_L/f_G = 0.29.

        Slightly above the nominal 0.02 % attached to this expression, so
        "within 0.02 %" is not a bound. lineshape/voigt.py and
        rank_diagnostics.py state the same figure and rely on this test.
        """

        def exact(variance, gamma):
            peak = voigt_derivatives(np.array([0.0]), 0.0, variance, gamma).value[0]

            def f(x):
                return voigt_derivatives(np.array([x]), 0.0, variance, gamma).value[0] - peak / 2

            return 2 * brentq(f, 0.0, 10 * (math.sqrt(variance) + gamma), xtol=1e-15, rtol=1e-14)

        ratios = np.concatenate([np.logspace(-4, 4, 81), np.linspace(0.25, 0.33, 9)])
        errors = []
        for ratio in ratios:  # f_L / f_G
            variance, gamma = 1.0 / (8 * math.log(2)), ratio / 2
            errors.append(abs(voigt_fwhm(variance, gamma) / exact(variance, gamma) - 1))
        worst = int(np.argmax(errors))
        assert errors[worst] == pytest.approx(2.37e-4, rel=5e-3)
        assert ratios[worst] == pytest.approx(0.29, abs=0.011)


# ===================================================================
# Width coordinates
# ===================================================================


WIDTH_MAPS = ["sigma_gamma", "var_gamma", "fwhm_shape", "fixed_instrument"]


class TestWidthCoordinates:
    @pytest.mark.parametrize("name", WIDTH_MAPS)
    @pytest.mark.parametrize("variance", [0.0, 1e-6, 0.04, 1.0])
    def test_round_trip(self, name, variance):
        widths = idf._width_coordinates(name, variance, GAMMA)
        v, g = _canonical_from_widths(name, widths, variance)
        # In (F, r) the Gaussian variance is a difference of two numbers
        # of order F**2 that cancels as r -> 1, so it comes back only to
        # about eps * F**2 in absolute terms: the coordinate is poorly
        # conditioned at the Lorentzian end. The others are exact.
        absolute = 1e-15 * voigt_fwhm(variance, GAMMA) ** 2 if name == "fwhm_shape" else 1e-18
        assert v == pytest.approx(variance, rel=1e-12, abs=absolute)
        assert g == pytest.approx(GAMMA, rel=1e-12)

    @pytest.mark.parametrize("name", WIDTH_MAPS)
    def test_transform_is_the_jacobian_of_the_map(self, name):
        variance = 0.04
        w0 = idf._width_coordinates(name, variance, GAMMA)
        numeric = np.zeros((2, w0.size))
        for j in range(w0.size):
            h = 1e-6 * max(abs(w0[j]), 1.0)
            up, dn = w0.copy(), w0.copy()
            up[j] += h
            dn[j] -= h
            numeric[:, j] = (
                np.array(_canonical_from_widths(name, up, variance))
                - np.array(_canonical_from_widths(name, dn, variance))
            ) / (2 * h)
        assert np.allclose(idf._width_transform(name, variance, GAMMA), numeric, rtol=1e-7, atol=1e-9)


# ===================================================================
# Fisher matrix
# ===================================================================


def _mean_from_params(result, energy, phi, background):
    """Poisson mean at parameter vector ``phi`` in ``result``'s coordinates."""
    name = result.parameterization
    floor = result.config["variance_floor"]
    n_width = len(idf._WIDTH_NAMES[name if floor is None else "instrument_floor"])
    n_peaks = len(result.config["peaks"])
    mean = np.zeros_like(energy)
    pos = 0
    for k in range(n_peaks):
        amp, center = phi[pos], phi[pos + 1]
        widths = phi[pos + 2: pos + 2 + n_width]
        pos += 2 + n_width
        if name == "pvoigt":
            mean += amp * idf._pvoigt_derivatives(energy - center, widths[0], widths[1])[0]
        else:
            fixed_variance = result.config["peaks"][k][2] ** 2 if floor is None else floor
            v, g = _canonical_from_widths(name, widths, fixed_variance)
            mean += amp * voigt_derivatives(energy, center, v, g).value
    if background is not None:
        coefficients = background.coefficients.copy()
        for j, is_estimated in enumerate(background.estimated):
            if is_estimated:
                coefficients[j] = phi[pos]
                pos += 1
        mean += coefficients @ background.basis
    assert pos == phi.size
    return result.exposure * mean


class TestPoissonFisher:
    def test_fisher_matrix_is_the_same_from_either_route(self, monkeypatch):
        """The brief's criterion: agreement of the matrix, not of the profile."""
        energy = np.linspace(-3.0, 3.0, 1201)
        peak = [VoigtPeak(AMP, 0.0, 0.03 * GAMMA, GAMMA)]  # min |z| = 23.6: all series
        auto = poisson_fisher(energy, peak).fisher

        def forced(energy, center, variance, gamma, *, route="auto"):
            return original(energy, center, variance, gamma, route="faddeeva")

        original = idf.voigt_derivatives
        monkeypatch.setattr(idf, "voigt_derivatives", forced)
        legacy = poisson_fisher(energy, peak).fisher
        d = np.sqrt(np.diag(auto))
        assert np.allclose(legacy / np.outer(d, d), auto / np.outer(d, d), rtol=0, atol=1e-7)

    def test_sigma_information_slope_is_two(self):
        """I_sigma_sigma ~ sigma**2 when the Lorentzian dominates.

        Asserted over 3e-3 <= sigma/gamma <= 3e-2, the window in which
        the legacy Jacobian also shows it; the slope departs from 2 by
        O((sigma/gamma)**2) at the top, hence 0.01.
        """
        energy = np.linspace(-10.0, 10.0, 2001)
        ratios = np.logspace(math.log10(3e-3), math.log10(3e-2), 9)
        info = np.array([
            poisson_fisher(
                energy, [VoigtPeak(AMP, 0.0, r * GAMMA, GAMMA)], parameterization="sigma_gamma"
            ).fisher[2, 2]
            for r in ratios
        ])
        slopes = np.diff(np.log(info)) / np.diff(np.log(ratios))
        assert np.all(np.abs(slopes - 2.0) < 0.01)

    def test_slope_holds_where_the_legacy_jacobian_has_broken_down(self):
        """Down to sigma/gamma = 1e-7; voigt_with_jacobian fails below 1e-3."""
        energy = np.linspace(-10.0, 10.0, 2001)
        ratios = np.logspace(-7, -4, 7)
        info = np.array([
            poisson_fisher(
                energy, [VoigtPeak(AMP, 0.0, r * GAMMA, GAMMA)], parameterization="sigma_gamma"
            ).fisher[2, 2]
            for r in ratios
        ])
        slopes = np.diff(np.log(info)) / np.diff(np.log(ratios))
        assert np.all(np.abs(slopes - 2.0) < 1e-6)

    def test_variance_information_is_finite_and_nonzero_at_zero(self):
        energy = np.linspace(-10.0, 10.0, 2001)
        at_zero = poisson_fisher(energy, [VoigtPeak(AMP, 0.0, 0.0, GAMMA)])
        assert at_zero.param_names[2] == "var_0"
        i_vv = at_zero.fisher[2, 2]
        assert np.isfinite(i_vv) and i_vv > 0
        assert np.linalg.eigvalsh(at_zero.fisher).min() > 0

        sigma = 1e-4 * GAMMA
        nearby = poisson_fisher(
            energy, [VoigtPeak(AMP, 0.0, sigma, GAMMA)], parameterization="sigma_gamma"
        )
        assert nearby.fisher[2, 2] / (4 * sigma**2) == pytest.approx(i_vv, rel=1e-6)

    def test_sigma_coordinate_is_singular_at_zero(self):
        energy = np.linspace(-10.0, 10.0, 2001)
        r = poisson_fisher(
            energy, [VoigtPeak(AMP, 0.0, 0.0, GAMMA)], parameterization="sigma_gamma"
        )
        assert np.all(r.fisher[2, :] == 0.0) and np.all(r.fisher[:, 2] == 0.0)

    @staticmethod
    def _correlation(energy, peak, background):
        r = poisson_fisher(energy, peak, background)
        d = np.sqrt(np.diag(r.fisher))
        return r.param_names, r.fisher / np.outer(d, d)

    def test_center_decouples_when_the_mean_is_even(self):
        """Symmetric window *and* an even mean: odd and even parameters split.

        The center is odd, amplitude, widths and a flat level are even. A
        background slope is odd too, so it couples to the center and to
        nothing else -- provided its true value is zero, because the
        weights 1/mu must be even as well.
        """
        energy = np.linspace(-5.0, 5.0, 1001)
        peak = [VoigtPeak(AMP, 0.0, 0.2, GAMMA)]

        for background in (None, constant_background(energy, 40.0)):
            names, c = self._correlation(energy, peak, background)
            row = np.delete(c[names.index("center_0")], names.index("center_0"))
            assert np.max(np.abs(row)) < 1e-12

        names, c = self._correlation(energy, peak, linear_background(energy, 40.0, 0.0))
        center, slope = names.index("center_0"), names.index("bg_slope")
        assert abs(c[center, slope]) > 1e-3
        even = [i for i, n in enumerate(names) if n not in ("center_0", "bg_slope")]
        assert np.max(np.abs(c[np.ix_([center, slope], even)])) < 1e-12

    def test_a_sloping_background_recouples_the_center(self):
        """A non-zero slope makes 1/mu uneven; the symmetric window no
        longer protects the position. Measured at 4e-4 (center) to 1e-2
        (slope) against the even parameters for this configuration."""
        energy = np.linspace(-5.0, 5.0, 1001)
        peak = [VoigtPeak(AMP, 0.0, 0.2, GAMMA)]
        names, c = self._correlation(energy, peak, linear_background(energy, 40.0, 1.5))
        center = names.index("center_0")
        even = [i for i, n in enumerate(names) if n not in ("center_0", "bg_slope")]
        assert np.max(np.abs(c[center, even])) > 1e-4

    def test_center_couples_on_an_asymmetric_window(self):
        energy = np.linspace(-6.0, 10.0, 1601)
        r = poisson_fisher(energy, [VoigtPeak(AMP, 0.0, 0.3, GAMMA)])
        d = np.sqrt(np.diag(r.fisher))
        row = (r.fisher / np.outer(d, d))[1]
        assert np.max(np.abs(np.delete(row, 1))) > 1e-4

    def test_linear_in_exposure(self):
        energy = np.linspace(-3.0, 3.0, 601)
        peak = [VoigtPeak(AMP, 0.0, 0.2, GAMMA)]
        background = linear_background(energy, 40.0, 1.5)
        one = poisson_fisher(energy, peak, background)
        many = poisson_fisher(energy, peak, background, exposure=37.0)
        assert np.allclose(many.fisher, 37.0 * one.fisher, rtol=1e-13, atol=0)
        assert np.allclose(many.expected_counts, 37.0 * one.expected_counts, rtol=1e-15)
        assert np.allclose(many.jacobian, 37.0 * one.jacobian, rtol=1e-15)
        assert np.array_equal(many.param_values, one.param_values)

    @pytest.mark.parametrize("name", [*PARAMETERIZATIONS, "fixed_instrument+floor"])
    def test_is_the_hessian_of_the_expected_deviance(self, name):
        """E[-log L] has Hessian exactly I at the truth; checked by differences.

        Independent of every derivative in the module: only model values
        enter. Done in each coordinate system, so it checks the transform
        T as well, and with two peaks and an estimated linear background.
        """
        energy = np.linspace(-3.0, 4.0, 701)
        peaks = [VoigtPeak(AMP, 0.0, 0.15, GAMMA), VoigtPeak(0.4 * AMP, 1.1, 0.25, 0.2)]
        background = linear_background(energy, 60.0, 3.0)
        floor = None
        if name.endswith("+floor"):
            name, floor = "fixed_instrument", 0.1**2
        r = poisson_fisher(
            energy, peaks, background, parameterization=name, exposure=2.0, variance_floor=floor
        )
        mu0, phi0 = r.expected_counts, r.param_values

        def half_deviance(phi):
            delta = (_mean_from_params(r, energy, phi, background) - mu0) / mu0
            return np.sum(mu0 * (delta - np.log1p(delta)))

        n = phi0.size
        steps = 1e-2 / np.sqrt(np.diag(r.fisher))
        hessian = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                total = 0.0
                for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    phi = phi0.copy()
                    phi[i] += si * steps[i]
                    phi[j] += sj * steps[j]
                    total += si * sj * half_deviance(phi)
                hessian[i, j] = hessian[j, i] = total / (4 * steps[i] * steps[j])

        d = np.sqrt(np.diag(r.fisher))
        assert np.allclose(hessian / np.outer(d, d), r.fisher / np.outer(d, d), rtol=0, atol=2e-4)

    def test_amplitude_bound_does_not_depend_on_width_coordinates(self):
        """Re-parameterising nuisance directions cannot move another bound."""
        energy = np.linspace(-3.0, 3.0, 601)
        peak = [VoigtPeak(AMP, 0.0, 0.15, GAMMA)]
        background = constant_background(energy, 60.0)
        bounds = []
        for name in ("sigma_gamma", "var_gamma", "fwhm_shape"):
            r = poisson_fisher(energy, peak, background, parameterization=name)
            bounds.append(np.diag(np.linalg.inv(r.fisher))[:2])
        assert np.allclose(bounds[0], bounds[1], rtol=1e-9)
        assert np.allclose(bounds[2], bounds[1], rtol=1e-9)

    def test_conditioning_against_window_width(self):
        """No background, sigma/gamma = 1/30, condition number of the
        unit-diagonal matrix.

        +-33.3 gamma reproduces the 6.18 of the inventory. The narrow
        windows are given as ranges because the number there depends on
        the sampling at the several-percent level (measured: 936 on 2001
        points against 912 on a 0.003 eV step at +-gamma, 5.5e6 against
        4.9e6 at +-0.3 gamma); an independent recomputation got 917 and
        5.2e6.
        """
        conds = []
        for half in (100.0 / 3.0, 1.0, 0.3):
            energy = np.linspace(-half * GAMMA, half * GAMMA, 2001)
            r = poisson_fisher(energy, [VoigtPeak(AMP, 0.0, GAMMA / 30, GAMMA)])
            conds.append(_corr_cond(r.fisher))
        assert conds[0] == pytest.approx(6.18, abs=0.02)
        assert 8.0e2 < conds[1] < 1.1e3
        assert 4.0e6 < conds[2] < 7.0e6

    def test_pvoigt_is_regular_at_a_pure_lorentzian(self):
        energy = np.linspace(-10.0, 10.0, 2001)
        r = poisson_fisher(energy, [VoigtPeak(AMP, 0.0, 0.0, GAMMA)], parameterization="pvoigt")
        assert r.param_names == ("amp_0", "center_0", "w_0", "eta_0")
        assert r.param_values[3] == pytest.approx(1.0, abs=1e-5)
        assert r.param_values[2] == pytest.approx(2 * GAMMA, rel=1e-12)
        assert np.linalg.eigvalsh(r.fisher).min() > 0

    def test_layout_for_two_peaks_with_background(self):
        energy = np.linspace(-3.0, 4.0, 701)
        peaks = [VoigtPeak(AMP, 0.0, 0.15, GAMMA), VoigtPeak(AMP, 1.1, 0.25, 0.2)]
        background = constant_background(energy, 60.0) + shirley_background(energy, peaks, 30.0)
        r = poisson_fisher(energy, peaks, background)
        assert r.param_names == (
            "amp_0", "center_0", "var_0", "gamma_0",
            "amp_1", "center_1", "var_1", "gamma_1",
            "bg_level", "bg_shirley_step",
        )
        assert r.width_index == (2, 3, 6, 7)
        assert r.fisher.shape == (10, 10) and r.jacobian.shape == (701, 10)
        assert np.allclose(r.fisher, r.fisher.T, rtol=1e-14)

        fixed = poisson_fisher(energy, peaks, background, parameterization="fixed_instrument")
        assert fixed.width_index == (2, 5)
        keep = [0, 1, 3, 4, 5, 7, 8, 9]
        assert np.allclose(fixed.fisher, r.fisher[np.ix_(keep, keep)], rtol=1e-13)

    def test_variance_floor_moves_the_origin_and_nothing_else(self):
        """Excess variance over a calibrated instrument width.

        Same matrix as 'var_gamma'; the coordinate value is the excess,
        and it may be exactly zero -- that is the boundary of interest.
        """
        energy = np.linspace(-3.0, 3.0, 601)
        peaks = [VoigtPeak(AMP, 0.0, 0.15, GAMMA), VoigtPeak(AMP, 1.2, 0.25, 0.2)]
        reference = poisson_fisher(energy, peaks)
        floor = 0.15**2
        r = poisson_fisher(
            energy, peaks, parameterization="fixed_instrument", variance_floor=floor
        )
        assert r.param_names == (
            "amp_0", "center_0", "var_extra_0", "gamma_0",
            "amp_1", "center_1", "var_extra_1", "gamma_1",
        )
        assert np.array_equal(r.fisher, reference.fisher)
        assert r.param_values[2] == 0.0
        assert r.param_values[6] == pytest.approx(0.25**2 - floor, rel=1e-12)
        assert r.config["variance_floor"] == floor

        with pytest.raises(ValueError, match="variance_floor"):
            poisson_fisher(
                energy, peaks, parameterization="fixed_instrument", variance_floor=0.2**2
            )
        with pytest.raises(ValueError, match="only applies"):
            poisson_fisher(energy, peaks, variance_floor=floor)

    def test_rejects_a_channel_with_no_expected_counts(self):
        energy = np.linspace(-20.0, 20.0, 401)
        gaussian = [VoigtPeak(AMP, 0.0, 0.2, 0.0)]
        with pytest.raises(ValueError, match="positive in every channel"):
            poisson_fisher(energy, gaussian)
        poisson_fisher(energy, gaussian, constant_background(energy, 5.0, estimated=False))

    def test_rejects_bad_arguments(self):
        energy = np.linspace(-1.0, 1.0, 11)
        peak = [VoigtPeak(AMP, 0.0, 0.2, GAMMA)]
        with pytest.raises(ValueError):
            poisson_fisher(energy, peak, parameterization="eta_fwhm")
        with pytest.raises(ValueError):
            poisson_fisher(energy, [])
        with pytest.raises(ValueError):
            poisson_fisher(energy, peak, exposure=0.0)
        with pytest.raises(ValueError):
            poisson_fisher(energy, peak, constant_background(energy[:-1], 5.0))
        with pytest.raises(ValueError):
            VoigtPeak(0.0, 0.0, 0.2, GAMMA)
        with pytest.raises(ValueError):
            VoigtPeak(AMP, 0.0, 0.0, 0.0)


class TestBackground:
    def test_shirley_shape_runs_from_zero_to_one(self):
        energy = np.linspace(-4.0, 4.0, 801)
        peaks = [VoigtPeak(AMP, 0.0, 0.2, GAMMA)]
        high = shirley_background(energy, peaks, 30.0)
        low = shirley_background(energy, peaks, 30.0, rises_toward="low")
        assert high.basis[0, 0] == 0.0 and high.basis[0, -1] == pytest.approx(1.0)
        assert np.all(np.diff(high.basis[0]) > 0)
        assert np.allclose(low.basis[0], 1.0 - high.basis[0])
        assert high.counts().max() == pytest.approx(30.0)
        with pytest.raises(ValueError):
            shirley_background(energy[::-1], peaks, 30.0)

    def test_known_terms_enter_the_mean_but_not_the_parameters(self):
        energy = np.linspace(-3.0, 3.0, 601)
        peak = [VoigtPeak(AMP, 0.0, 0.2, GAMMA)]
        known = poisson_fisher(energy, peak, constant_background(energy, 50.0, estimated=False))
        bare = poisson_fisher(energy, peak)
        assert known.param_names == bare.param_names
        assert np.allclose(known.expected_counts - bare.expected_counts, 50.0)

    def test_sum_concatenates_terms(self):
        energy = np.linspace(-3.0, 3.0, 601)
        both = constant_background(energy, 50.0, estimated=False) + linear_background(
            energy, 0.0, 2.0
        )
        assert both.estimated == (False, True, True)
        assert both.basis.shape == (3, 601)
        with pytest.raises(ValueError):
            Background(np.ones((2, 5)), np.ones(1), ("a",), (True,))


# ===================================================================
# Effective information
# ===================================================================


class TestEffectiveInformation:
    @pytest.fixture
    def cases(self):
        energy = np.linspace(-3.0, 3.0, 601)
        peak = [VoigtPeak(AMP, 0.0, 0.1, GAMMA)]
        return energy, peak, {
            "none": None,
            "known": constant_background(energy, 50.0, estimated=False),
            "estimated": constant_background(energy, 50.0),
            "estimated_shirley": constant_background(energy, 50.0)
            + shirley_background(energy, peak, 30.0),
        }

    def test_effective_never_exceeds_conditional(self, cases):
        energy, peak, backgrounds = cases
        for background in backgrounds.values():
            for name in ("var_gamma", "sigma_gamma", "fwhm_shape", "pvoigt"):
                r = poisson_fisher(energy, peak, background, parameterization=name)
                info = effective_information(r.fisher, r.width_index)
                scale = np.linalg.eigvalsh(info.conditional).max()
                assert _is_psd(info.conditional - info.effective, scale)
                assert _is_psd(info.effective, scale)
                assert info.nuisance_null_dim == 0

    def test_inverse_is_the_width_block_of_the_full_inverse(self, cases):
        energy, peak, backgrounds = cases
        for background in backgrounds.values():
            r = poisson_fisher(energy, peak, background)
            info = effective_information(r.fisher, r.width_index)
            block = np.linalg.inv(r.fisher)[np.ix_(r.width_index, r.width_index)]
            assert np.allclose(np.linalg.inv(info.effective), block, rtol=1e-9)

    def test_background_only_removes_information(self, cases):
        """none >= known >= estimated, in the Loewner order."""
        energy, peak, backgrounds = cases
        eff = {}
        for key, background in backgrounds.items():
            r = poisson_fisher(energy, peak, background)
            eff[key] = effective_information(r.fisher, r.width_index).effective
        scale = np.linalg.eigvalsh(eff["none"]).max()
        assert _is_psd(eff["none"] - eff["known"], scale)
        assert _is_psd(eff["known"] - eff["estimated"], scale)
        # and an estimated flat background costs the widths a lot more
        # than its shot noise does: the soft direction loses over half
        soft = {k: np.linalg.eigvalsh(v)[0] for k, v in eff.items()}
        assert soft["known"] > 0.95 * soft["none"]
        assert soft["estimated"] < 0.5 * soft["known"]

    def test_does_not_move_with_the_amplitude_unit(self, cases):
        """A -> cA is g -> D g D; the width block must not notice."""
        energy, peak, backgrounds = cases
        r = poisson_fisher(energy, peak, backgrounds["estimated"])
        reference = effective_information(r.fisher, r.width_index).effective
        for c in (1e-6, 1e6):
            scale = np.ones(len(r.param_names))
            scale[0] = c
            rescaled = r.fisher * np.outer(scale, scale)
            moved = effective_information(rescaled, r.width_index).effective
            assert np.allclose(moved, reference, rtol=1e-9)

    def test_redundant_nuisance_direction_is_dropped_not_inverted(self, cases):
        energy, peak, backgrounds = cases
        single = poisson_fisher(energy, peak, backgrounds["estimated"])
        reference = effective_information(single.fisher, single.width_index)

        duplicate = Background(
            basis=np.ones((2, energy.size)),
            coefficients=np.array([25.0, 25.0]),
            names=("bg_a", "bg_b"),
            estimated=(True, True),
        )
        doubled = poisson_fisher(energy, peak, duplicate)
        info = effective_information(doubled.fisher, doubled.width_index)
        assert info.nuisance_null_dim == 1
        assert np.allclose(info.effective, reference.effective, rtol=1e-9)

    def test_no_nuisance_means_no_loss(self):
        m = np.array([[4.0, 1.0], [1.0, 3.0]])
        info = effective_information(m, (0, 1))
        assert np.array_equal(info.effective, m) and info.nuisance_null_dim == 0

    def test_rejects_bad_index(self):
        m = np.eye(3)
        for bad in ((), (0, 0), (3,), (-1,)):
            with pytest.raises(ValueError):
                effective_information(m, bad)


# ===================================================================
# Assessment
# ===================================================================


def _flat(energy, peak, fraction=0.1):
    """Flat background at a fraction of the peak height."""
    height = peak.amplitude * voigt_derivatives(
        np.array([peak.center]), peak.center, peak.variance, peak.gamma
    ).value[0]
    return constant_background(energy, fraction * height)


class TestAssessment:
    PEAK = VoigtPeak(AMP, 0.0, GAMMA / 3, GAMMA)

    def _report(self, half_width, **kwargs):
        energy = np.arange(-half_width * GAMMA, half_width * GAMMA + 1e-9, 0.01)
        return assess_identifiability(energy, [self.PEAK], _flat(energy, self.PEAK), **kwargs)

    def test_wide_window_is_identified(self):
        report = self._report(10.0)
        status = {a.name: a.status for a in report.parameters}
        assert status == {
            "amp_0": "identified", "center_0": "identified", "var_0": "identified",
            "gamma_0": "identified", "bg_level": "not_assessed",
        }
        width = report.widths[0]
        assert width.status == "identified" and not width.near_boundary
        assert report.null_space_dim == 0 and np.isfinite(report.condition_number)

    def test_narrow_window_with_an_estimated_background_is_weak(self):
        """+-gamma: sd(gamma)/FWHM is 0.24 with everything estimated and
        7e-4 with the rest known -- the loss a sub-block diagnostic hides."""
        report = self._report(1.0)
        by_name = {a.name: a for a in report.parameters}
        assert by_name["gamma_0"].status == "weakly_identified"
        assert by_name["var_0"].status == "weakly_identified"
        assert by_name["center_0"].status == "identified"
        assert report.widths[0].status == "weakly_identified"
        assert by_name["gamma_0"].relative_sd == pytest.approx(0.24, abs=0.02)
        assert by_name["gamma_0"].sd / by_name["gamma_0"].sd_conditional > 100

    def test_weak_is_not_structural_more_exposure_lifts_it(self):
        weak = self._report(1.0)
        strong = self._report(1.0, exposure=1.0e4)
        assert weak.widths[0].status == "weakly_identified"
        assert strong.widths[0].status == "identified"
        for a, b in zip(weak.parameters, strong.parameters):
            assert b.sd == pytest.approx(a.sd / 100.0, rel=1e-9)
        # 8e6, so its smallest eigenvalue carries cond * eps ~ 1e-9 of rounding
        assert strong.condition_number == pytest.approx(weak.condition_number, rel=1e-6)

    def test_identified_is_not_detection_of_a_small_gaussian(self):
        """sigma/gamma = 0.03: fine on the FWHM scale, hopeless against itself."""
        energy = np.arange(-3.0, 3.0 + 1e-9, 0.01)
        peak = VoigtPeak(AMP, 0.0, 0.03 * GAMMA, GAMMA)
        width = assess_identifiability(energy, [peak], _flat(energy, peak)).widths[0]
        assert width.status == "identified" and width.worst_relative_sd < 0.01
        assert width.variance_relative_sd > 1.0
        assert width.near_boundary

    def test_pure_lorentzian_sits_on_the_boundary(self):
        energy = np.arange(-3.0, 3.0 + 1e-9, 0.01)
        peak = VoigtPeak(AMP, 0.0, 0.0, GAMMA)
        width = assess_identifiability(energy, [peak], _flat(energy, peak)).widths[0]
        assert width.near_boundary and width.variance_relative_sd == np.inf
        assert width.status == "identified" and np.isfinite(width.sd_variance)

    def test_boundary_follows_the_variance_floor(self):
        energy = np.arange(-3.0, 3.0 + 1e-9, 0.01)
        peak = VoigtPeak(AMP, 0.0, 0.15, GAMMA)
        background = _flat(energy, peak)
        free = assess_identifiability(energy, [peak], background).widths[0]
        assert not free.near_boundary and free.variance_excess == pytest.approx(0.15**2)

        at_floor = assess_identifiability(
            energy, [peak], background, variance_floor=0.15**2
        ).widths[0]
        assert at_floor.near_boundary and at_floor.variance_excess == 0.0
        assert at_floor.variance_relative_sd == np.inf
        # the floor moves the boundary and nothing else
        assert at_floor.sd_variance == free.sd_variance
        assert at_floor.worst_relative_sd == free.worst_relative_sd

        just_above = peak.variance - 2.0 * free.sd_variance
        assert assess_identifiability(
            energy, [peak], background, variance_floor=just_above
        ).widths[0].near_boundary
        well_above = peak.variance - 4.0 * free.sd_variance
        assert not assess_identifiability(
            energy, [peak], background, variance_floor=well_above
        ).widths[0].near_boundary
        with pytest.raises(ValueError):
            assess_identifiability(energy, [peak], background, variance_floor=0.16**2)

    def test_sd_is_the_full_inverse_and_the_conditional_one_is_smaller(self):
        report = self._report(3.0)
        full = np.sqrt(np.diag(np.linalg.inv(report.fisher.fisher)))
        for a, expected in zip(report.parameters, full):
            assert a.sd == pytest.approx(expected, rel=1e-8)
            if a.role == "width":
                assert a.sd_conditional < a.sd
            else:
                assert a.sd_conditional is None

    def test_worst_direction_is_no_better_than_either_width(self):
        for half_width in (1.0, 3.0, 10.0):
            report = self._report(half_width)
            by_name = {a.name: a for a in report.parameters}
            width = report.widths[0]
            assert width.worst_relative_sd >= by_name["var_0"].relative_sd
            assert width.worst_relative_sd >= by_name["gamma_0"].relative_sd
            assert np.hypot(*width.worst_direction) == pytest.approx(1.0)

    def test_does_not_depend_on_the_energy_unit(self):
        """The same spectrum described in meV instead of eV.

        Energies scale by c, the unit-area profile by 1/c, so the area
        scales by c for the same counts per channel; a slope by 1/c; a
        variance floor by c**2. Every relative number must stay put.
        """
        c = 1000.0
        energy = np.arange(-1.0, 1.5 + 1e-9, 0.01)
        peaks = [VoigtPeak(AMP, 0.0, 0.12, GAMMA), VoigtPeak(0.5 * AMP, 0.6, 0.2, 0.15)]
        background = linear_background(energy, 4000.0, 300.0)
        ev = assess_identifiability(energy, peaks, background, variance_floor=0.1**2)

        scaled_peaks = [
            VoigtPeak(c * p.amplitude, c * p.center, c * p.sigma, c * p.gamma) for p in peaks
        ]
        mev = assess_identifiability(
            c * energy, scaled_peaks, linear_background(c * energy, 4000.0, 300.0 / c),
            variance_floor=(c * 0.1) ** 2,
        )
        assert mev.condition_number == pytest.approx(ev.condition_number, rel=1e-8)
        for a, b in zip(ev.parameters, mev.parameters):
            assert a.status == b.status
            if a.relative_sd is not None:
                assert b.relative_sd == pytest.approx(a.relative_sd, rel=1e-8)
        for a, b in zip(ev.widths, mev.widths):
            assert (a.status, a.near_boundary) == (b.status, b.near_boundary)
            assert b.worst_relative_sd == pytest.approx(a.worst_relative_sd, rel=1e-8)
            assert b.variance_relative_sd == pytest.approx(a.variance_relative_sd, rel=1e-8)

    def test_coincident_peaks_are_rank_deficient(self):
        energy = np.arange(-3.0, 3.0 + 1e-9, 0.01)
        twin = VoigtPeak(AMP, 0.0, 0.1, GAMMA)
        report = assess_identifiability(energy, [twin, twin])
        assert report.null_space_dim == 4 and report.condition_number == np.inf
        assert all(a.status == "rank_deficient" and a.sd == np.inf for a in report.parameters)
        assert all(w.status == "rank_deficient" and w.near_boundary for w in report.widths)

    def test_redundant_background_does_not_condemn_the_peak(self):
        """The null direction bg_a - bg_b touches no peak axis."""
        energy = np.arange(-3.0, 3.0 + 1e-9, 0.01)
        level = _flat(energy, self.PEAK).coefficients[0]
        duplicate = Background(
            basis=np.ones((2, energy.size)),
            coefficients=np.array([level / 2, level / 2]),
            names=("bg_a", "bg_b"),
            estimated=(True, True),
        )
        doubled = assess_identifiability(energy, [self.PEAK], duplicate)
        single = assess_identifiability(energy, [self.PEAK], _flat(energy, self.PEAK))
        assert doubled.null_space_dim == 1
        status = {a.name: a.status for a in doubled.parameters}
        assert status["bg_a"] == status["bg_b"] == "rank_deficient"
        for a, b in zip(doubled.parameters[:4], single.parameters[:4]):
            assert a.status == b.status == "identified"
            assert a.sd == pytest.approx(b.sd, rel=1e-6)

    def test_thresholds_are_the_callers(self):
        strict = self._report(10.0, thresholds=IdentifiabilityThresholds(weak_relative_sd=1e-4))
        assert strict.widths[0].status == "weakly_identified"
        lax = self._report(1.0, thresholds=IdentifiabilityThresholds(weak_relative_sd=10.0))
        assert lax.widths[0].status == "identified"
        eager = self._report(10.0, thresholds=IdentifiabilityThresholds(boundary_sd=1e3))
        assert eager.widths[0].near_boundary
        with pytest.raises(ValueError):
            IdentifiabilityThresholds(weak_relative_sd=0.0)

    def test_report_serialises(self):
        text = json.dumps(self._report(1.0).to_dict())
        assert "weakly_identified" in text and "worst_relative_sd" in text


# ===================================================================
# Scan
# ===================================================================


class TestScan:
    GRID = ScanGrid(
        half_widths=(0.3, 1.0, 5.0),
        levels=(1.0, 100.0),
        ratios=(0.0, 1e-3, 1e-2, 1.0),
        separations=(None, 0.5),
    )

    @pytest.fixture(scope="class")
    def scan(self):
        return scan_identifiability(self.GRID)

    @staticmethod
    def _b(scan, name):
        return list(scan["backgrounds"]).index(name)

    def test_shapes_and_round_trip_through_npz(self, scan, tmp_path):
        shape = (3, 2, 4, 3, 2)
        for key in ("sd_variance", "sd_sigma", "status", "near_boundary", "condition_number"):
            assert scan[key].shape == shape
        assert np.isnan(scan["separations"][0]) and scan["separations"][1] == 0.5
        path = tmp_path / "scan.npz"
        np.savez(path, **scan)
        loaded = np.load(path)
        assert set(loaded.files) == set(scan)
        assert np.array_equal(loaded["status"], scan["status"])
        assert str(loaded["normalization"]) == "exposure"

    def test_cell_is_the_direct_assessment(self, scan):
        """Background 'estimated', single peak, sigma/gamma = 1, +-1 FWHM, level 100."""
        sigma, gamma = idf._widths_at_fwhm(1.0, 1.0)
        assert voigt_fwhm(sigma**2, gamma) == pytest.approx(1.0, rel=1e-14)
        energy = np.linspace(-1.0, 1.0, 101)
        peak = VoigtPeak(1.0e4, 0.0, sigma, gamma)
        report = assess_identifiability(energy, [peak], _flat(energy, peak), exposure=100.0)
        cell = (self._b(scan, "estimated"), 0, 3, 1, 1)
        assert scan["sd_gamma"][cell] == pytest.approx(report.parameters[3].relative_sd, rel=1e-12)
        assert scan["worst_relative_sd"][cell] == report.widths[0].worst_relative_sd
        assert scan["n_energy"][cell] == 101

    def test_coordinate_degeneracy_and_information_loss_come_apart(self, scan):
        """Towards sigma = 0 only the sigma bound moves; towards a narrow
        window both do."""
        est = self._b(scan, "estimated")
        along_ratio = (est, 0, slice(0, 3), 2, 0)  # ratios 0, 1e-3, 1e-2 at +-5 FWHM
        variance = scan["sd_variance"][along_ratio]
        assert np.ptp(variance) / variance.mean() < 1e-3
        assert scan["sd_sigma"][along_ratio][0] == np.inf
        assert scan["sd_sigma"][est, 0, 1, 2, 0] / scan["sd_sigma"][est, 0, 2, 2, 0] == pytest.approx(
            10.0, rel=1e-3
        )
        # information on sigma vanishes like sigma**2, on the variance it does not
        assert scan["info_sigma"][est, 0, 0, 2, 0] == 0.0
        assert scan["info_sigma"][est, 0, 2, 2, 0] / scan["info_sigma"][est, 0, 1, 2, 0] == pytest.approx(
            100.0, rel=1e-3
        )
        # and the label, taken in var_gamma, does not react to the coordinate
        assert np.all(scan["status"][est, 0, :3, 2, 0] == 0)

        along_window = (est, 0, 3, slice(None), 0)
        assert np.all(np.diff(scan["sd_variance"][along_window]) < 0)
        assert np.all(np.diff(scan["sd_sigma"][along_window]) < 0)
        assert scan["status"][est, 0, 3, 0, 0] == 1

    def test_window_dependence_without_any_background(self, scan):
        none = self._b(scan, "none")
        cond = scan["condition_number"][none, 0, 3, :, 0]
        assert np.all(np.diff(cond) < 0) and cond[0] > 100 * cond[-1]

    def test_background_ordering(self, scan):
        """none <= known <= estimated in every cell, for both widths."""
        none, known, est = (self._b(scan, k) for k in ("none", "known", "estimated"))
        for key in ("sd_variance", "sd_gamma"):
            assert np.all(scan[key][none] <= scan[key][known] * (1 + 1e-9))
            assert np.all(scan[key][known] <= scan[key][est] * (1 + 1e-9))
            assert np.all(scan[key + "_conditional"] <= scan[key] * (1 + 1e-9))

    def test_levels_scale_the_bounds_and_nothing_else(self, scan):
        ratio = scan["sd_gamma"][..., 0] / scan["sd_gamma"][..., 1]
        assert np.allclose(ratio, 10.0, rtol=1e-6)
        assert np.allclose(
            scan["condition_number"][..., 0], scan["condition_number"][..., 1], rtol=1e-5
        )
        assert np.allclose(scan["total_counts"][..., 1], 100.0 * scan["total_counts"][..., 0])

    def test_a_close_second_peak_costs_information(self, scan):
        """Compared at +-5 FWHM only. The window is measured outward from
        the outermost center, so a pair gets a window wider by its
        separation; at +-0.3 FWHM that extra half FWHM of data outweighs
        the overlap and the pair comes out *better* (1.9 against 2.5 for
        sd_gamma). Cells are not comparable across separations there."""
        est = self._b(scan, "estimated")
        assert np.all(scan["sd_gamma"][est, 1, :, 2] > 1.5 * scan["sd_gamma"][est, 0, :, 2])
        assert scan["sd_gamma"][est, 1, 3, 0, 0] < scan["sd_gamma"][est, 0, 3, 0, 0]

    def test_total_counts_normalisation(self):
        """Same counts in every window: what is left is the window's shape."""
        grid = ScanGrid(
            half_widths=(1.0, 5.0), levels=(1.0e6,), backgrounds=("none", "estimated"),
            normalization="total_counts",
        )
        scan = scan_identifiability(grid)
        assert np.allclose(scan["total_counts"], 1.0e6, rtol=1e-12)
        assert np.all(scan["sd_gamma"][:, 0, 0, 0, 0] > scan["sd_gamma"][:, 0, 0, 1, 0])

    @pytest.mark.parametrize("kind", ["linear", "shirley"])
    def test_other_background_shapes(self, kind):
        grid = ScanGrid(half_widths=(2.0,), backgrounds=("estimated",), background_kind=kind)
        scan = scan_identifiability(grid)
        assert np.isfinite(scan["sd_gamma"]).all() and scan["status"].min() >= 0

    def test_rejects_unknown_options(self):
        with pytest.raises(ValueError):
            scan_identifiability(ScanGrid(half_widths=(1.0,), backgrounds=("fitted",)))
        with pytest.raises(ValueError):
            scan_identifiability(ScanGrid(half_widths=(1.0,), normalization="counts"))
        with pytest.raises(ValueError):
            scan_identifiability(ScanGrid(half_widths=(1.0,), background_kind="tougaard"))

