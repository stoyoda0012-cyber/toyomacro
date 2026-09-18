"""
Tests for the Voigt width identifiability diagnostics
=====================================================

Two kinds of assertion, kept apart:

- *Identities* that hold exactly or to rounding: the heat equation and
  the v = 0 limit.
- *Measured numbers* with a written tolerance: the agreement of the two
  derivative routes across the switch and the Olivero-Longbothum error.
"""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from toyomacro.lineshape import Voigt
from toyomacro.voigtfit.identifiability import voigt_derivatives, voigt_fwhm
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
        """Implementation fidelity: the approximation is good to 2.4e-4."""

        def exact(variance, gamma):
            peak = voigt_derivatives(np.array([0.0]), 0.0, variance, gamma).value[0]

            def f(x):
                return voigt_derivatives(np.array([x]), 0.0, variance, gamma).value[0] - peak / 2

            return 2 * brentq(f, 0.0, 10 * (math.sqrt(variance) + gamma), xtol=1e-15, rtol=1e-14)

        worst = 0.0
        for ratio in np.logspace(-4, 4, 81):  # f_L / f_G
            variance, gamma = 1.0 / (8 * math.log(2)), ratio / 2
            worst = max(worst, abs(voigt_fwhm(variance, gamma) / exact(variance, gamma) - 1))
        assert 1e-4 < worst < 2.5e-4
