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
    identifiability_at_estimate,
    identifiability_from_exact_k,
    identifiability_modes,
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


def _unit_free(m, reference):
    """``m`` in the coordinates that give ``reference`` a unit diagonal.

    A width block mixes eV**2 and eV, so its eigenvalues are statements
    about the energy unit. After this congruence they are not.
    """
    d = np.sqrt(np.diag(reference))
    return m / np.outer(d, d)


def _is_psd(m, reference):
    """Positive semi-definite to rounding, judged unit-free."""
    scaled = _unit_free(0.5 * (m + m.T), reference)
    return np.linalg.eigvalsh(scaled).min() >= -1e-10


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

    def test_accuracy_over_a_population_of_shapes(self):
        """The table in the module docstring, as a bound: 31 shapes from
        sigma/gamma = 0.01 to 10 on +-33.3 gamma, every output, error in a
        band of |z| over the largest magnitude in that band.

        Measured worst case 3e-11 (d_variance on 6 <= |z| < 7, the
        Faddeeva side of the switch; 2e-11 to 4e-11 as the grid is
        shifted, 5e-11 on ten times as many shapes); asserted at 1e-10.
        """
        bands = [0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 12.0, 50.0, np.inf]
        x = np.linspace(-33.3 * GAMMA, 33.3 * GAMMA, 1601) + 1.234e-4
        worst = 0.0
        for ratio in np.logspace(-2, 1, 31):
            sigma = ratio * GAMMA
            z_abs = np.hypot(x, GAMMA) / (sigma * math.sqrt(2))
            got = voigt_derivatives(x, 0.0, sigma**2, GAMMA)
            reference = _quadrature_reference(x, sigma**2, GAMMA)
            for lo, hi in zip(bands[:-1], bands[1:]):
                band = (z_abs >= lo) & (z_abs < hi)
                if band.sum() < 5:
                    continue
                for a, b in zip(got, reference):
                    worst = max(worst, np.max(np.abs(a[band] - b[band])) / np.max(np.abs(b[band])))
        assert worst < 1e-10

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
        """Agreement of the matrix, not only of the profile."""
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
        """Down to sigma/gamma = 1e-7. The engine's Jacobian gives this
        element wrong by a factor of about 6 at 3e-4 and by four orders at
        1e-4."""
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

        # Achieved: 1.5e-10 .. 6e-10 in every coordinate system, for steps
        # of 1e-3 to 1e-2 sd alike. 2e-8 leaves a factor 30 and still fails
        # a column that is wrong by one part in 10**7.
        d = np.sqrt(np.diag(r.fisher))
        assert np.allclose(hessian / np.outer(d, d), r.fisher / np.outer(d, d), rtol=0, atol=2e-8)

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

        6.18 at +-33.3 gamma. The narrow windows are given as ranges
        because the number there depends on how finely the window is
        sampled: at +-gamma 719, 854, 912, 936, 939 on 21, 61, 201, 2001
        and 20001 points, at +-0.3 gamma 4.0e6, 4.9e6, 5.3e6, 5.5e6, 5.5e6.
        Coarse sampling understates it, by 23 % and 27 % on 21 points;
        2001 points are within 0.4 % of the limit.
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
        for bad in ((np.inf, 0.0, 0.2, GAMMA), (AMP, np.nan, 0.2, GAMMA), (AMP, 0.0, np.nan, GAMMA),
                    (AMP, 0.0, 0.2, np.inf)):
            with pytest.raises(ValueError, match="finite"):
                VoigtPeak(*bad)
        with pytest.raises(ValueError, match="pvoigt"):
            poisson_fisher(energy, [VoigtPeak(AMP, 0.0, 1e-12, 1e-12)], parameterization="pvoigt")


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

    def test_shirley_shape_of_two_lorentzians_is_the_arctangent_sum(self):
        """Closed form: the running integral of A/pi * g/(x^2+g^2) is
        A/pi * arctan(x/g). Two peaks of unequal area, so the weights
        matter; trapezoid rule on a 0.002 eV step, so 1e-6 -- a rectangle
        rule would be out by 1e-3."""
        energy = np.linspace(-4.0, 5.0, 4501)
        peaks = [VoigtPeak(AMP, 0.0, 0.0, 0.3), VoigtPeak(0.25 * AMP, 1.5, 0.0, 0.2)]
        running = sum(
            p.amplitude / math.pi * np.arctan((energy - p.center) / p.gamma) for p in peaks
        )
        expected = (running - running[0]) / (running[-1] - running[0])
        shape = shirley_background(energy, peaks, 30.0).basis[0]
        assert np.max(np.abs(shape - expected)) < 1e-6

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
                assert _is_psd(info.conditional - info.effective, info.conditional)
                assert _is_psd(info.effective, info.conditional)
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
        assert _is_psd(eff["none"] - eff["known"], eff["none"])
        assert _is_psd(eff["known"] - eff["estimated"], eff["none"])

    def test_a_small_estimated_background_costs_more_than_its_shot_noise(self, cases):
        """Flat background at 0.05 % of the peak height, +-10 gamma.

        Information left in the least determined direction of the width
        block: 99 % with the background known, 61 % with it estimated. The
        direction is taken in the scaled coordinates of the assessment
        (variance over FWHM**2/(8 ln2), gamma over FWHM); the smallest
        eigenvalue of the raw (eV**2, eV) block is not a physical quantity,
        and gives 43 %, 63 % or 41 % for the same spectrum described in eV,
        meV or keV.
        """
        energy, peak, backgrounds = cases
        fwhm = voigt_fwhm(peak[0].variance, peak[0].gamma)
        scales = np.array([fwhm**2 / (8 * math.log(2)), fwhm])
        soft = {}
        for key in ("none", "known", "estimated"):
            r = poisson_fisher(energy, peak, backgrounds[key])
            effective = effective_information(r.fisher, r.width_index).effective
            soft[key] = np.linalg.eigvalsh(effective * np.outer(scales, scales))[0]
        assert soft["known"] / soft["none"] == pytest.approx(0.993, abs=0.002)
        assert soft["estimated"] / soft["none"] == pytest.approx(0.605, abs=0.005)

    def test_that_share_does_not_depend_on_the_energy_unit(self):
        shares = []
        for c in (1.0, 1.0e3, 1.0e-3):  # eV, meV, keV
            energy = c * np.linspace(-3.0, 3.0, 601)
            peak = [VoigtPeak(c * AMP, 0.0, c * 0.1, c * GAMMA)]
            fwhm = voigt_fwhm(peak[0].variance, peak[0].gamma)
            scales = np.array([fwhm**2 / (8 * math.log(2)), fwhm])
            soft = []
            for background in (None, constant_background(energy, 50.0)):
                r = poisson_fisher(energy, peak, background)
                effective = effective_information(r.fisher, r.width_index).effective
                soft.append(np.linalg.eigvalsh(effective * np.outer(scales, scales))[0])
            shares.append(soft[1] / soft[0])
        assert np.allclose(shares, shares[0], rtol=1e-8)

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

    def test_rank_rule_is_n_times_eps(self):
        """As np.linalg.matrix_rank and crlb: an eigenvalue of 1e-15 next
        to 1 is null among ten, not among two."""
        ten = np.array([1e-15] + [0.5] * 8 + [1.0])
        assert idf._null_directions(ten).tolist() == [True] + [False] * 9
        assert idf._null_directions(np.array([1e-15, 1.0])).tolist() == [False, False]

    def test_shared_core_does_not_import_the_voigt_engine(self):
        """The model-independent parts live outside voigtfit so that a
        model other than a Voigt peak can use them without importing the
        engine; that only holds while the module imports nothing from the
        package. Checked in a fresh interpreter, where nothing else has
        loaded voigtfit already."""
        import subprocess
        import sys

        code = ("import sys, toyomacro._identifiability; "
                "print(sorted(m for m in sys.modules if m.startswith('toyomacro')))")
        loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                check=True).stdout.strip()
        assert loaded == "['toyomacro', 'toyomacro._identifiability']"
        for name in ("Background", "EffectiveInformation", "IdentifiabilityThresholds",
                     "constant_background", "linear_background", "effective_information",
                     "Mode", "mode_decomposition"):
            assert getattr(idf, name) is getattr(sys.modules["toyomacro._identifiability"], name)

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

    def test_conditional_sd_is_the_inverse_of_the_width_block(self):
        """Both widths of both peaks jointly, the rest known -- not
        1/sqrt(I_ii), which would also take the other width as known."""
        energy = np.arange(-1.0, 1.5 + 1e-9, 0.01)
        peaks = [VoigtPeak(AMP, 0.0, 0.12, GAMMA), VoigtPeak(0.5 * AMP, 0.6, 0.2, 0.15)]
        report = assess_identifiability(energy, peaks, linear_background(energy, 4000.0, 0.0))
        index = list(report.fisher.width_index)
        block = report.fisher.fisher[np.ix_(index, index)]
        expected = np.sqrt(np.diag(np.linalg.inv(block)))
        got = [report.parameters[i].sd_conditional for i in index]
        assert np.allclose(got, expected, rtol=1e-9)
        single = 1.0 / np.sqrt(np.diag(block))
        assert np.all(expected > 1.2 * single)  # the two are not the same number

    def test_thresholds_are_applied_as_stated(self):
        """Not only what the defaults are, but where the code puts them:
        the boundary flag flips between 2.9 and 3.1 bounds from the floor,
        the label between 0.99 and 1.01 of the reported worst direction."""
        energy = np.arange(-3.0, 3.0 + 1e-9, 0.01)
        peak = VoigtPeak(AMP, 0.0, 0.15, GAMMA)
        background = _flat(energy, peak)
        free = assess_identifiability(energy, [peak], background).widths[0]
        for distance, expected in ((2.9, True), (3.1, False)):
            floor = peak.variance - distance * free.sd_variance
            width = assess_identifiability(
                energy, [peak], background, variance_floor=floor
            ).widths[0]
            assert width.near_boundary is expected
        for factor, expected in ((0.99, "weakly_identified"), (1.01, "identified")):
            thresholds = IdentifiabilityThresholds(
                weak_relative_sd=factor * free.worst_relative_sd
            )
            width = assess_identifiability(
                energy, [peak], background, thresholds=thresholds
            ).widths[0]
            assert width.status == expected

    def test_default_thresholds_are_the_documented_ones(self):
        """A tenth of the scale; three bounds from the boundary. Stated in
        the design record and the changelog, so a change must be loud."""
        defaults = IdentifiabilityThresholds()
        assert (defaults.weak_relative_sd, defaults.boundary_sd) == (0.1, 3.0)

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
            assert (a.status, a.near_boundary, a.near_boundary_gamma) == (
                b.status, b.near_boundary, b.near_boundary_gamma)
            assert b.worst_relative_sd == pytest.approx(a.worst_relative_sd, rel=1e-8)
            assert b.variance_relative_sd == pytest.approx(a.variance_relative_sd, rel=1e-8)
            assert b.gamma_relative_sd == pytest.approx(a.gamma_relative_sd, rel=1e-8)

    def test_coincident_peaks_are_rank_deficient(self):
        energy = np.arange(-3.0, 3.0 + 1e-9, 0.01)
        twin = VoigtPeak(AMP, 0.0, 0.1, GAMMA)
        report = assess_identifiability(energy, [twin, twin])
        assert report.null_space_dim == 4 and report.condition_number == np.inf
        assert all(a.status == "rank_deficient" and a.sd == np.inf for a in report.parameters)
        assert all(w.status == "rank_deficient" and w.near_boundary for w in report.widths)
        assert all(w.near_boundary_gamma and w.gamma_relative_sd == np.inf for w in report.widths)

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
        for key in ("sd_variance", "sd_sigma", "status", "near_boundary", "near_boundary_gamma",
                    "condition_number"):
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
        assert scan["near_boundary_gamma"][cell] == report.widths[0].near_boundary_gamma
        assert scan["n_energy"][cell] == 101

    def test_gamma_flag_reaches_the_cells(self, scan):
        """A flagged cell, rebuilt directly: single peak, sigma/gamma = 1,
        known background, +-0.3 FWHM, level 1."""
        cell = (self._b(scan, "known"), 0, 3, 0, 0)
        sigma, gamma = idf._widths_at_fwhm(1.0, 1.0)
        energy = np.linspace(-0.3, 0.3, 31)
        peak = VoigtPeak(1.0e4, 0.0, sigma, gamma)
        known = constant_background(energy, _flat(energy, peak).coefficients[0], estimated=False)
        width = assess_identifiability(energy, [peak], known).widths[0]
        assert width.near_boundary_gamma
        assert scan["near_boundary_gamma"][cell]
        assert scan["near_boundary_gamma"].any() and not scan["near_boundary_gamma"].all()

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
        the overlap and the pair comes out *better* (sd_gamma 15.7 against
        39.7 at sigma/gamma = 1). Cells are not comparable across
        separations there."""
        est = self._b(scan, "estimated")
        assert np.all(scan["sd_gamma"][est, 1, :, 2] > 1.5 * scan["sd_gamma"][est, 0, :, 2])
        assert scan["sd_gamma"][est, 1, 3, 0, 0] < scan["sd_gamma"][est, 0, 3, 0, 0]

    def test_total_counts_normalisation(self):
        """Same counts in every window: what is left is the window's shape.

        The total is checked against a mean rebuilt here, background
        included, not against the number the scan was given.
        """
        grid = ScanGrid(
            half_widths=(1.0, 5.0), levels=(1.0e6,), backgrounds=("none", "estimated"),
            normalization="total_counts",
        )
        scan = scan_identifiability(grid)
        assert np.all(scan["sd_gamma"][:, 0, 0, 0, 0] > scan["sd_gamma"][:, 0, 0, 1, 0])

        sigma, gamma = idf._widths_at_fwhm(1.0, 1.0)
        peak = VoigtPeak(grid.area, 0.0, sigma, gamma)
        for h, half in enumerate(grid.half_widths):
            energy = np.linspace(-half, half, int(round(2 * half / grid.step)) + 1)
            background = _flat(energy, peak)
            unit = poisson_fisher(energy, [peak], background).expected_counts.sum()
            report = assess_identifiability(energy, [peak], background, exposure=1.0e6 / unit)
            assert report.fisher.expected_counts.sum() == pytest.approx(1.0e6, rel=1e-12)
            cell = (1, 0, 0, h, 0)
            assert scan["sd_gamma"][cell] == pytest.approx(
                report.parameters[3].relative_sd, rel=1e-10
            )
            assert scan["total_counts"][cell] == pytest.approx(1.0e6, rel=1e-12)

    @pytest.mark.parametrize("normalization, level", [("exposure", 1.0), ("total_counts", 1e6)])
    def test_fwhm_only_sets_the_unit(self, normalization, level):
        """Step, window, area and floor all scale with it, so no result
        moves: every array the scan returns, both normalisations, with a
        floor in force. rtol 1e-6 because the matrices here reach a
        condition number of 1e9, i.e. 3e-7 of rounding."""
        sigma, _ = idf._widths_at_fwhm(1.0, 1.0)
        kwargs = dict(
            half_widths=(0.5, 3.0), ratios=(1.0, 3.0), separations=(None, 0.5),
            levels=(level,), normalization=normalization, variance_floor=0.5 * sigma**2,
        )
        reference = scan_identifiability(ScanGrid(**kwargs))
        axes = {"backgrounds", "separations", "ratios", "half_widths", "levels",
                "normalization", "background_kind", "fwhm"}
        for fwhm in (0.5, 0.7, 3.0):
            other = scan_identifiability(ScanGrid(fwhm=fwhm, **kwargs))
            for key in set(reference) - axes:
                if reference[key].dtype.kind == "f":
                    assert np.allclose(other[key], reference[key], rtol=1e-6), key
                else:
                    assert np.array_equal(other[key], reference[key]), key

    def test_variance_floor_reaches_the_cells(self):
        """In units of fwhm**2; at the floor the peak is on the boundary."""
        sigma, _ = idf._widths_at_fwhm(1.0, 1.0)
        free = scan_identifiability(ScanGrid(half_widths=(5.0,), backgrounds=("estimated",)))
        held = scan_identifiability(
            ScanGrid(half_widths=(5.0,), backgrounds=("estimated",), variance_floor=sigma**2)
        )
        assert not free["near_boundary"].any() and held["near_boundary"].all()
        assert np.isfinite(free["variance_relative_sd"]).all()
        assert np.isinf(held["variance_relative_sd"]).all()
        assert np.array_equal(held["sd_variance"], free["sd_variance"])

    def test_information_is_the_inverse_square_of_the_scaled_bound(self, scan):
        finite = np.isfinite(scan["sd_sigma"])
        assert np.all(scan["info_sigma"][~finite] == 0.0)
        assert np.allclose(scan["info_sigma"][finite] * scan["sd_sigma"][finite] ** 2, 1.0, rtol=1e-10)
        assert np.allclose(scan["info_variance"] * scan["sd_variance"] ** 2, 1.0, rtol=1e-10)

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



class TestVarianceFloorUncertainty:
    """``variance_floor_sd``: the floor is itself an estimate."""

    def _report(self, **kw):
        energy = np.linspace(-6.0, 6.0, 241)
        peak = [VoigtPeak(amplitude=2.0e4, center=0.0, sigma=0.35, gamma=0.3)]
        return assess_identifiability(energy, peak, constant_background(energy, 50.0),
                                      variance_floor=0.05, **kw)

    def test_zero_reproduces_the_bounds_exactly(self):
        plain = self._report()
        zero = self._report(variance_floor_sd=0.0)
        for a, b in zip(plain.widths, zero.widths):
            assert (a.sd_variance, a.variance_relative_sd, a.near_boundary, a.worst_relative_sd,
                    a.worst_direction, a.status) == (
                b.sd_variance, b.variance_relative_sd, b.near_boundary, b.worst_relative_sd,
                b.worst_direction, b.status)

    def test_the_floors_own_sd_adds_in_quadrature(self):
        """The excess is a difference of two independent estimates, so its
        bound is the quadrature sum; the width block's worst direction
        widens with it."""
        base = self._report().widths[0]
        for floor_sd in (0.5 * base.sd_variance, 2.0 * base.sd_variance):
            got = self._report(variance_floor_sd=floor_sd).widths[0]
            assert got.sd_variance == pytest.approx(math.hypot(base.sd_variance, floor_sd))
            assert got.variance_relative_sd == pytest.approx(
                got.sd_variance / got.variance_excess)
            assert got.worst_relative_sd > base.worst_relative_sd
            assert got.variance_floor_sd == floor_sd

    def test_it_can_turn_a_determined_excess_into_a_boundary_case(self):
        base = self._report().widths[0]
        assert not base.near_boundary
        loose = self._report(variance_floor_sd=base.variance_excess).widths[0]
        assert loose.near_boundary

    def test_a_negative_floor_sd_raises(self):
        with pytest.raises(ValueError, match="variance_floor_sd"):
            self._report(variance_floor_sd=-1e-6)


# ===================================================================
# The gamma = 0 boundary
# ===================================================================


class TestGammaBoundary:
    """gamma = 0 is a boundary as the variance floor is, and gets the same
    two numbers: a precision relative to itself and a flag."""

    def test_a_nearly_gaussian_peak_is_flagged(self):
        """The example of the docstring: sigma/gamma = 100, default scan
        peak, +-5 FWHM, estimated flat background. gamma is 'identified'
        against the FWHM, yet known only to about 58 % of itself."""
        sigma, gamma = idf._widths_at_fwhm(100.0, 1.0)
        energy = np.linspace(-5.0, 5.0, 501)
        peak = VoigtPeak(1.0e4, 0.0, sigma, gamma)
        report = assess_identifiability(energy, [peak], _flat(energy, peak))
        assert report.parameters[3].status == "identified"
        width = report.widths[0]
        assert width.gamma_relative_sd == pytest.approx(0.58, abs=0.01)
        assert width.gamma_relative_sd == pytest.approx(report.parameters[3].sd / gamma)
        assert width.near_boundary_gamma and not width.near_boundary

    def test_a_lorentzian_peak_is_not(self):
        report = TestAssessment()._report(10.0)
        assert not report.widths[0].near_boundary_gamma
        assert report.widths[0].gamma_relative_sd < 0.05

    def test_gamma_at_zero(self):
        energy = np.linspace(-1.0, 1.0, 201)
        peak = VoigtPeak(1.0e4, 0.0, 0.2, 0.0)
        width = assess_identifiability(energy, [peak], _flat(energy, peak)).widths[0]
        assert width.gamma_relative_sd == np.inf and width.near_boundary_gamma

    def test_follows_the_boundary_threshold(self):
        energy = np.linspace(-5.0, 5.0, 501)
        sigma, gamma = idf._widths_at_fwhm(100.0, 1.0)
        peak = VoigtPeak(1.0e4, 0.0, sigma, gamma)
        width = assess_identifiability(energy, [peak], _flat(energy, peak)).widths[0]
        ratio = 1.0 / width.gamma_relative_sd  # gamma in units of its own sd
        below = IdentifiabilityThresholds(boundary_sd=0.9 * ratio)
        above = IdentifiabilityThresholds(boundary_sd=1.1 * ratio)
        assert not assess_identifiability(energy, [peak], _flat(energy, peak),
                                          thresholds=below).widths[0].near_boundary_gamma
        assert assess_identifiability(energy, [peak], _flat(energy, peak),
                                      thresholds=above).widths[0].near_boundary_gamma


# ===================================================================
# Modes
# ===================================================================


MODE_ENERGY = np.arange(-3.0, 3.0 + 1e-9, 0.02)
MODE_PEAK = VoigtPeak(2.0e4, 0.0, 0.3, 0.3)


def _mode_cases():
    """(name, energy, peaks, background, exposure): one peak, a pair, a
    Shirley step, a narrow window, a higher exposure."""
    e = MODE_ENERGY
    pair = [VoigtPeak(2.0e4, -0.4, 0.2, 0.3), VoigtPeak(1.0e4, 0.5, 0.3, 0.2)]
    narrow = np.arange(-0.3, 0.3 + 1e-9, 0.02)
    soft = VoigtPeak(2.0e4, 0.0, 0.1, 0.3)
    return [
        ("single", e, [MODE_PEAK], linear_background(e, 300.0, 10.0), 1.0),
        ("pair", e, pair, linear_background(e, 300.0, 10.0), 1.0),
        ("shirley", e, [MODE_PEAK],
         constant_background(e, 300.0) + shirley_background(e, [MODE_PEAK], 200.0), 1.0),
        ("narrow", narrow, [soft], linear_background(narrow, 300.0, 10.0), 1.0),
        ("exposure", e, pair, linear_background(e, 300.0, 10.0), 7.3),
    ]


def _scaled_effective(report):
    """I_s = S I_eff S from scratch, for checking the eigen-relation."""
    f = report.fisher
    target = [i for i, r in enumerate(f.param_roles) if r != "background"]
    scales = np.array([report.scales[f.param_names[i]] for i in target])
    effective = effective_information(f.fisher, target).effective
    return effective * np.outer(scales, scales), [f.param_names[i] for i in target]


class TestModes:
    """``identifiability_modes``: the identities and invariances of §9.1 of
    the design record. The Monte Carlo is in tests/test_identifiability_mc.py."""

    @pytest.mark.parametrize("case", _mode_cases(), ids=lambda c: c[0])
    def test_every_mode_is_one_unit_of_counting_noise(self, case):
        _, energy, peaks, background, exposure = case
        report = identifiability_modes(energy, peaks, background, exposure=exposure, k=20)
        assert report.modes and not report.null_modes
        for mode in report.modes:
            # 1 in exact arithmetic; 1e-7 leaves room for the BLAS on the
            # narrow window, while any real break of the identity is O(1)
            assert mode.whitened_norm == pytest.approx(1.0, abs=1e-7)
            mu = report.fisher.expected_counts
            assert np.sqrt(np.sum(mode.signature**2 / mu)) == pytest.approx(mode.whitened_norm)

    @pytest.mark.parametrize("case", _mode_cases(), ids=lambda c: c[0])
    def test_eigen_relation(self, case):
        """I_s w = w / lambda, lambda = relative_sd**2, and the modes are
        orthonormal; every non-null mode is listed with k large enough."""
        _, energy, peaks, background, exposure = case
        report = identifiability_modes(energy, peaks, background, exposure=exposure, k=20)
        scaled, names = _scaled_effective(report)
        vectors = np.array([[m.weights[n] for n in names] for m in report.modes]).T
        assert vectors.shape == scaled.shape
        assert np.allclose(vectors.T @ vectors, np.eye(len(names)), atol=1e-10)
        for mode, w in zip(report.modes, vectors.T):
            assert np.allclose(scaled @ w, w / mode.relative_sd**2,
                               rtol=1e-7, atol=1e-7 * np.abs(scaled).max())
        sds = [m.relative_sd for m in report.modes]
        assert sds == sorted(sds, reverse=True)
        ratios = [m.eigenvalue_ratio for m in report.modes]
        assert ratios[-1] is None
        assert all(r == pytest.approx((a / b) ** 2) for r, a, b in zip(ratios, sds, sds[1:]))

    @pytest.mark.parametrize("case", _mode_cases(), ids=lambda c: c[0])
    def test_softest_is_no_better_than_any_parameter(self, case):
        """The largest eigenvalue bounds every diagonal element and every
        principal sub-block (no floor uncertainty in the comparison)."""
        _, energy, peaks, background, exposure = case
        softest = identifiability_modes(energy, peaks, background, exposure=exposure).modes[0]
        report = assess_identifiability(energy, peaks, background, exposure=exposure)
        bounded = [a.relative_sd for a in report.parameters
                   if a.relative_sd is not None and np.isfinite(a.relative_sd)]
        assert softest.relative_sd >= max(bounded) * (1.0 - 1e-9)
        assert softest.relative_sd >= max(w.worst_relative_sd for w in report.widths) * (1.0 - 1e-9)

    def test_a_mode_can_be_weak_when_no_parameter_is(self):
        """The converse does not hold; the docstring says so. Not every
        configuration shows it -- here the single-peak case does not."""
        energy, peaks, background = MODE_ENERGY, [MODE_PEAK], linear_background(
            MODE_ENERGY, 300.0, 10.0)
        softest = identifiability_modes(energy, peaks, background).modes[0]
        largest = max(a.relative_sd for a in assess_identifiability(
            energy, peaks, background).parameters if a.relative_sd is not None)
        between = IdentifiabilityThresholds(weak_relative_sd=0.5 * (softest.relative_sd + largest))
        assert identifiability_modes(energy, peaks, background,
                                     thresholds=between).modes[0].status == "weakly_identified"
        assert all(a.status != "weakly_identified" for a in assess_identifiability(
            energy, peaks, background, thresholds=between).parameters)

    def test_does_not_depend_on_the_energy_unit(self):
        c = 1000.0
        e = MODE_ENERGY
        pair = [VoigtPeak(2.0e4, -0.4, 0.2, 0.3), VoigtPeak(1.0e4, 0.5, 0.3, 0.2)]
        ev = identifiability_modes(e, pair, linear_background(e, 300.0, 10.0), k=8)
        mev = identifiability_modes(
            c * e, [VoigtPeak(c * p.amplitude, c * p.center, c * p.sigma, c * p.gamma)
                    for p in pair],
            linear_background(c * e, 300.0, 10.0 / c), k=8)
        for a, b in zip(ev.modes, mev.modes):
            assert b.relative_sd == pytest.approx(a.relative_sd, rel=1e-8)
            for name in a.weights:
                assert b.weights[name] == pytest.approx(a.weights[name], abs=1e-8)
            assert b.background_shift["bg_level"] == pytest.approx(
                a.background_shift["bg_level"], rel=1e-6, abs=1e-9)
            assert b.background_shift["bg_slope"] == pytest.approx(
                a.background_shift["bg_slope"] / c, rel=1e-6, abs=1e-12)

    def test_does_not_depend_on_how_the_background_is_written(self):
        """[1, E - E_mid] and [1, E + 100] span the same background. The
        modes, and the background curve that follows each, are the same."""
        e = MODE_ENERGY
        mid = linear_background(e, 300.0, 10.0)
        shifted = Background(
            basis=np.vstack([np.ones(e.size), e + 100.0]),
            coefficients=np.array([300.0 - 10.0 * 100.0, 10.0]),
            names=("bg_a", "bg_b"), estimated=(True, True),
        )
        assert np.allclose(mid.counts(), shifted.counts())
        a = identifiability_modes(e, [MODE_PEAK], mid, k=4)
        b = identifiability_modes(e, [MODE_PEAK], shifted, k=4)
        for x, y in zip(a.modes, b.modes):
            assert y.relative_sd == pytest.approx(x.relative_sd, rel=1e-8)
            assert all(y.weights[n] == pytest.approx(x.weights[n], abs=1e-8) for n in x.weights)
            curve_x = mid.basis.T @ np.array(list(x.background_shift.values()))
            curve_y = shifted.basis.T @ np.array(list(y.background_shift.values()))
            assert np.allclose(curve_x, curve_y, rtol=1e-6, atol=1e-6 * np.abs(curve_x).max())

    def test_scales_with_exposure_and_the_directions_do_not_move(self):
        e = MODE_ENERGY
        bg = linear_background(e, 300.0, 10.0)
        one = identifiability_modes(e, [MODE_PEAK], bg, k=4)
        more = identifiability_modes(e, [MODE_PEAK], bg, k=4, exposure=7.3)
        for a, b in zip(one.modes, more.modes):
            assert b.relative_sd * math.sqrt(7.3) == pytest.approx(a.relative_sd, rel=1e-10)
            assert all(b.weights[n] == pytest.approx(a.weights[n], abs=1e-10) for n in a.weights)

    def test_null_space_counts_on_constructed_cases(self):
        """Coincident peaks: 4 null modes. A background term duplicated: a
        null direction, but wholly in the background, so no null mode. A
        background term with the peak's own shape: 1 null mode."""
        e = MODE_ENERGY
        twin = VoigtPeak(2.0e4, 0.0, 0.1, 0.3)
        coincident = identifiability_modes(e, [twin, twin], linear_background(e, 300.0, 10.0))
        assert (coincident.null_space_dim, coincident.nuisance_null_dim) == (4, 0)
        assert len(coincident.null_modes) == 4

        twice = constant_background(e, 150.0) + Background(
            basis=np.ones((1, e.size)), coefficients=np.array([150.0]),
            names=("bg_level_2",), estimated=(True,))
        duplicated = identifiability_modes(e, [MODE_PEAK], twice)
        assert (duplicated.null_space_dim, duplicated.nuisance_null_dim) == (1, 1)
        assert not duplicated.null_modes

        shape = voigt_derivatives(e, 0.0, MODE_PEAK.variance, MODE_PEAK.gamma).value
        peak_shaped = constant_background(e, 300.0) + Background(
            basis=shape[np.newaxis, :], coefficients=np.array([0.0]),
            names=("bg_peak_shape",), estimated=(True,))
        aliased = identifiability_modes(e, [MODE_PEAK], peak_shaped)
        assert (aliased.null_space_dim, aliased.nuisance_null_dim) == (1, 0)
        assert len(aliased.null_modes) == 1
        assert aliased.null_modes[0].weights["amp_0"] > 0.99

        cases = (
            (coincident, [twin, twin], linear_background(e, 300.0, 10.0)),
            (duplicated, [MODE_PEAK], twice),
            (aliased, [MODE_PEAK], peak_shaped),
        )
        for report, peaks, background in cases:
            assert report.null_space_dim == assess_identifiability(
                e, peaks, background).null_space_dim
            for mode in report.null_modes:
                assert mode.relative_sd == np.inf and mode.status == "rank_deficient"
                assert mode.whitened_norm < 1e-6
                assert all(v is None for v in mode.shares.values())

    def test_coincident_peaks_leave_the_sum_determined(self):
        """Every single peak parameter is rank deficient, yet A0 + A1 lies
        in the span of the soft modes."""
        e = MODE_ENERGY
        twin = VoigtPeak(2.0e4, 0.0, 0.1, 0.3)
        report = identifiability_modes(e, [twin, twin], linear_background(e, 300.0, 10.0), k=8)
        assert len(report.modes) == 4
        names = list(report.modes[0].weights)
        span = np.array([[m.weights[n] for n in names] for m in report.modes]).T
        total = np.zeros(len(names))
        total[names.index("amp_0")] = 1.0 / report.scales["amp_0"]
        total[names.index("amp_1")] = 1.0 / report.scales["amp_1"]
        total /= np.linalg.norm(total)
        assert np.linalg.norm(span.T @ total) == pytest.approx(1.0, abs=1e-8)
        assert all(np.isfinite(m.relative_sd) for m in report.modes)
        # every peak parameter's own bound is infinite, so it has no share
        assert all(report.modes[0].shares[n] is None for n in names)

    def test_shares(self):
        """Over all modes the shares of a peak parameter add to 1; those of a
        background coefficient to 1 - (I_bb^-1)_bb / Var(b)."""
        e = MODE_ENERGY
        bg = linear_background(e, 300.0, 10.0)
        report = identifiability_modes(e, [MODE_PEAK], bg, k=20)
        f = report.fisher
        for name in ("amp_0", "center_0", "var_0", "gamma_0"):
            assert sum(m.shares[name] for m in report.modes) == pytest.approx(1.0, abs=1e-9)
        b = [i for i, r in enumerate(f.param_roles) if r == "background"]
        known_peaks = np.linalg.inv(f.fisher[np.ix_(b, b)])
        full = np.linalg.inv(f.fisher)
        for j, i in enumerate(b):
            expected = 1.0 - known_peaks[j, j] / full[i, i]
            got = sum(m.shares[f.param_names[i]] for m in report.modes)
            assert got == pytest.approx(expected, abs=1e-9)
        assert all(0.0 <= v <= 1.0 + 1e-12 for m in report.modes for v in m.shares.values())

    def test_sign_convention(self):
        from toyomacro._identifiability import _oriented

        report = identifiability_modes(MODE_ENERGY, [MODE_PEAK],
                                       linear_background(MODE_ENERGY, 300.0, 10.0), k=4)
        for mode in report.modes:
            values = list(mode.weights.values())
            assert max(values, key=abs) > 0.0
        assert _oriented(np.array([0.6, -0.8])).tolist() == [-0.6, 0.8]
        assert _oriented(np.array([-0.6, 0.8])).tolist() == [-0.6, 0.8]
        tie = np.array([-0.5, 0.5 * (1.0 + 1e-9), 0.1])
        assert _oriented(tie)[0] > 0.0  # within the tie tolerance the first decides
        gap = np.array([-0.5, 0.5 * (1.0 + 1e-4), 0.1])
        assert _oriented(gap)[1] > 0.0  # outside it the largest decides

    def test_k_and_the_status(self):
        e = MODE_ENERGY
        bg = linear_background(e, 300.0, 10.0)
        assert len(identifiability_modes(e, [MODE_PEAK], bg).modes) == 3
        assert len(identifiability_modes(e, [MODE_PEAK], bg, k=1).modes) == 1
        assert len(identifiability_modes(e, [MODE_PEAK], bg, k=50).modes) == 4
        assert len(identifiability_modes(e, [MODE_PEAK], bg, k=np.int64(2)).modes) == 2
        for bad in (0, -1, 2.0):
            with pytest.raises(ValueError, match="k must be"):
                identifiability_modes(e, [MODE_PEAK], bg, k=bad)
        modes = identifiability_modes(e, [MODE_PEAK], bg, k=4).modes
        cut = IdentifiabilityThresholds(
            weak_relative_sd=0.5 * (modes[1].relative_sd + modes[2].relative_sd))
        status = [m.status for m in identifiability_modes(e, [MODE_PEAK], bg, k=4,
                                                          thresholds=cut).modes]
        assert status == ["weakly_identified", "weakly_identified", "identified", "identified"]

    def test_variance_floor_moves_no_mode(self):
        e = MODE_ENERGY
        bg = linear_background(e, 300.0, 10.0)
        plain = identifiability_modes(e, [MODE_PEAK], bg)
        floored = identifiability_modes(e, [MODE_PEAK], bg, variance_floor=0.05)
        assert floored.variance_floor == 0.05
        for a, b in zip(plain.modes, floored.modes):
            assert (a.relative_sd, a.weights) == (b.relative_sd, b.weights)
        with pytest.raises(ValueError, match="variance_floor"):
            identifiability_modes(e, [MODE_PEAK], bg, variance_floor=0.2)

    def test_report_serialises_without_arrays(self):
        report = identifiability_modes(MODE_ENERGY, [MODE_PEAK],
                                       linear_background(MODE_ENERGY, 300.0, 10.0))
        data = report.to_dict()
        text = json.dumps(data)
        assert "signature" not in text and "whitened_norm" in text
        assert len(data["modes"]) == 3 and data["null_modes"] == []
        assert set(data["scales"]) == {"amp_0", "center_0", "var_0", "gamma_0"}

    def test_null_count_follows_the_labels(self):
        """Two background terms that are nearly, but not numerically,
        dependent: [1, 1 + delta E]. Counting null modes as the difference
        of two n*eps decisions on different matrices put a null mode here
        for some delta while every peak parameter was labelled; the rank of
        the null space's peak rows, with the labels' tolerance, does not."""
        e = MODE_ENERGY
        flat = identifiability_modes(e, [MODE_PEAK], constant_background(e, 300.0), k=4)
        linear = identifiability_modes(e, [MODE_PEAK], linear_background(e, 300.0, 0.0), k=4)
        for delta in np.logspace(-10.0, -6.0, 81):
            bg = Background(basis=np.vstack([np.ones(e.size), np.ones(e.size) + delta * e]),
                            coefficients=np.array([150.0, 150.0]), names=("bg_a", "bg_b"),
                            estimated=(True, True))
            report = identifiability_modes(e, [MODE_PEAK], bg, k=4)
            labels = assess_identifiability(e, [MODE_PEAK], bg).parameters
            deficient = any(a.status == "rank_deficient" for a in labels if a.role != "background")
            assert bool(report.null_modes) == deficient, delta
            assert report.null_space_dim - report.nuisance_null_dim == len(report.null_modes)
            # The two terms span [1, E], numerically [1] as delta shrinks. The
            # three widest modes do not see the slope (a centred peak). The
            # fourth, the center, couples to a slope that sits at rounding
            # level in this band, so its digits depend on the BLAS; only that
            # it is a finite bound is asserted.
            for mode, reference in zip(report.modes[:3], flat.modes):
                assert mode.relative_sd == pytest.approx(reference.relative_sd, rel=1e-6)
            assert np.isfinite(report.modes[3].relative_sd)
        assert linear.modes[3].relative_sd > flat.modes[3].relative_sd

    def test_an_oblique_null_direction(self):
        """A background term built from a mix of the center, variance and
        gamma columns: the null direction is not along any axis, so the
        mapping into scaled coordinates is what is being tested."""
        e = MODE_ENERGY
        peak = VoigtPeak(2.0e4, 0.1, 0.3, 0.3)
        jac = poisson_fisher(e, [peak], constant_background(e, 300.0)).jacobian
        fwhm = voigt_fwhm(peak.variance, peak.gamma)
        shape = 3.0 * jac[:, 1] / fwhm + jac[:, 3] / fwhm + 0.5 * jac[:, 2] / (
            fwhm**2 / (8.0 * math.log(2.0)))
        bg = constant_background(e, 300.0) + Background(
            basis=shape[np.newaxis, :], coefficients=np.array([0.0]),
            names=("bg_oblique",), estimated=(True,))
        report = identifiability_modes(e, [peak], bg, k=10)
        assert len(report.null_modes) == 1 and len(report.modes) == 3
        scaled, names = _scaled_effective(report)
        q = np.array([report.null_modes[0].weights[n] for n in names])
        assert np.linalg.norm(scaled @ q) < 1e-8 * np.abs(scaled).max()
        assert report.null_modes[0].whitened_norm < 1e-6
        vectors = np.array([[m.weights[n] for n in names] for m in report.modes]).T
        assert np.allclose(vectors.T @ q, 0.0, atol=1e-8)
        for mode, w in zip(report.modes, vectors.T):
            assert np.allclose(scaled @ w, w / mode.relative_sd**2,
                               rtol=1e-7, atol=1e-7 * np.abs(scaled).max())
            assert mode.whitened_norm == pytest.approx(1.0, abs=1e-9)

    def test_the_stiff_end_of_a_near_singular_design(self):
        """Two close peaks, every mode requested. The covariance side keeps
        the soft end; at the stiff end digits can go, and a bound rounding
        leaves non-positive is reported as 'not_resolved', never as 0."""
        e = MODE_ENERGY
        peaks = [VoigtPeak(2.0e4, -0.028, 0.3, 0.3), VoigtPeak(2.0e3, 0.028, 0.3, 0.3)]
        report = identifiability_modes(e, peaks, constant_background(e, 300.0), k=8)
        # The softest bound here is ~1e5 of the scales: the design is singular
        # to within rounding, and how many digits survive depends on the BLAS.
        for mode in report.modes:
            if mode.status == "not_resolved":
                assert math.isnan(mode.relative_sd) and mode.eigenvalue_ratio is None
                assert all(v is None for v in mode.shares.values())
            else:
                assert np.isfinite(mode.relative_sd) and mode.relative_sd > 0.0
                assert mode.eigenvalue_ratio is None or mode.eigenvalue_ratio >= 1.0 - 1e-9

    def test_a_bound_rounding_leaves_non_positive_is_not_resolved(self, monkeypatch):
        """Deterministically, whatever the BLAS: the pseudo-inverse is made
        to carry one negative eigenvalue in the peak block, so the stiffest
        mode has no positive bound."""
        import toyomacro._identifiability as core

        real = core.covariance_bound

        def tampered(matrix, tol):
            covariance, unbounded, dim, cond = real(matrix, tol)
            peak = np.arange(4)
            eigenvalues, vectors = np.linalg.eigh(covariance[np.ix_(peak, peak)])
            eigenvalues[0] = -abs(eigenvalues[0])
            covariance = covariance.copy()
            covariance[np.ix_(peak, peak)] = (vectors * eigenvalues) @ vectors.T
            return covariance, unbounded, dim, cond

        monkeypatch.setattr(core, "covariance_bound", tampered)
        report = identifiability_modes(MODE_ENERGY, [MODE_PEAK],
                                       linear_background(MODE_ENERGY, 300.0, 10.0), k=4)
        last = report.modes[-1]
        assert last.status == "not_resolved" and math.isnan(last.relative_sd)
        assert last.eigenvalue_ratio is None and all(v is None for v in last.shares.values())
        assert report.modes[-2].eigenvalue_ratio is None
        assert all(m.status != "not_resolved" and m.relative_sd > 0.0 for m in report.modes[:-1])

# ===================================================================
# At a fitted estimate
# ===================================================================


EK_ENERGY = np.arange(280.0, 292.0 + 1e-9, 0.05)


def _exact_k_report(*, doublet=False, plugin=None):
    """A noise-free two-component spectrum on a linear background, fitted
    by exact_k (which then recovers the truth to rounding)."""
    from toyomacro.voigtfit.exact_k import ExactKConfig, run_exact_k
    from toyomacro.voigtfit.multipeak_solver import build_bg_design

    e = EK_ENERGY
    split, ratio = (1.2, 2.0) if doublet else (0.0, 1.0)
    y = build_bg_design(e, 1).astype(np.float64) @ np.array([200.0, 30.0])
    for amp, center in ((3.0e4, 284.8), (1.2e4, 286.4)):
        y = y + amp * voigt_derivatives(e, center, 0.16, 0.15).value
        if doublet:
            y = y + amp / ratio * voigt_derivatives(e, center + split, 0.16, 0.15).value
    cfg = ExactKConfig(sigma_init=0.4, gamma=0.15, k_max=2, so_split=split,
                       branch_ratio=ratio,
                       bg_degree=None if plugin else 1, plugin_background=plugin)
    return run_exact_k(e, y, cfg, sigma_std=np.sqrt(y))


@pytest.fixture(scope="module")
def ek_singlet():
    return _exact_k_report()


@pytest.fixture(scope="module")
def ek_doublet():
    return _exact_k_report(doublet=True)


def _tied_mean(theta, e, n_comp, split, ratio, gamma):
    """The exact_k model written out directly, for finite differences."""
    from toyomacro.voigtfit.multipeak_solver import build_bg_design

    amps, centers, var, bg = theta[:n_comp], theta[n_comp:2 * n_comp], theta[2 * n_comp], \
        theta[2 * n_comp + 1:]
    mu = build_bg_design(e, bg.size - 1).astype(np.float64) @ bg
    for a, c in zip(amps, centers):
        mu = mu + a * voigt_derivatives(e, c, var, gamma).value
        if split:
            mu = mu + a / ratio * voigt_derivatives(e, c + split, var, gamma).value
    return mu


class TestEstimate:
    """``identifiability_at_estimate`` and the ``exact_k`` adapter."""

    def test_vocabulary_is_the_readers(self):
        import typing

        from toyomacro.io.readers.base_reader import IntensitySemantics

        assert idf.INTENSITY_SEMANTICS == typing.get_args(IntensitySemantics)

    def test_anything_but_raw_counts_is_withheld(self):
        e = MODE_ENERGY
        bg = linear_background(e, 300.0, 10.0)
        for intensity in idf.INTENSITY_SEMANTICS[1:]:
            report = identifiability_at_estimate(e, [MODE_PEAK], bg, intensity=intensity)
            assert report.withheld and intensity in report.withheld
            assert report.parameters == () and report.modes is None and not report.near_boundary
            json.dumps(report.to_dict())
        with pytest.raises(ValueError, match="intensity must be one of"):
            identifiability_at_estimate(e, [MODE_PEAK], bg, intensity="counts")
        with pytest.raises(TypeError):
            identifiability_at_estimate(e, [MODE_PEAK], bg)  # no default

    def test_raw_counts_is_the_labels_and_the_modes(self):
        e = MODE_ENERGY
        bg = linear_background(e, 300.0, 10.0)
        report = identifiability_at_estimate(e, [MODE_PEAK], bg, intensity="raw_counts")
        assert report.withheld is None and report.model == "every peak parameter free"
        direct = assess_identifiability(e, [MODE_PEAK], bg)
        assert [a.sd for a in report.parameters] == [a.sd for a in direct.parameters]
        modes = identifiability_modes(e, [MODE_PEAK], bg)
        assert [m.relative_sd for m in report.modes.modes] == [m.relative_sd for m in modes.modes]
        assert report.near_boundary == {"var_0": False, "gamma_0": False}
        assert "not a confidence interval" in report.caveat
        text = json.dumps(report.to_dict())
        assert "signature" not in text

    def test_the_adapter_evaluates_the_tied_model(self, ek_singlet, ek_doublet):
        """The Jacobian of the sub-model against finite differences of the
        exact_k model written out directly: that pins the ties."""
        for rep, split, ratio in ((ek_singlet, 0.0, 1.0), (ek_doublet, 1.2, 2.0)):
            cand = rep.candidates[1]
            assert cand.score.success and not cand.boundary_flags
            report = identifiability_from_exact_k(rep, 2, EK_ENERGY, intensity="raw_counts")
            assert report.withheld is None and report.assessment is None
            assert ("doublet" in report.model) == bool(split)
            fisher = report.modes.fisher
            assert fisher.param_names == ("amp_0", "amp_1", "center_0", "center_1",
                                          "var_shared", "bg_0", "bg_1")
            theta = fisher.param_values.copy()
            jac = np.empty_like(fisher.jacobian)
            for i in range(theta.size):
                h = 1e-6 * max(abs(theta[i]), 1e-3)
                up, down = theta.copy(), theta.copy()
                up[i] += h
                down[i] -= h
                jac[:, i] = (_tied_mean(up, EK_ENERGY, 2, split, ratio, 0.15)
                             - _tied_mean(down, EK_ENERGY, 2, split, ratio, 0.15)) / (2 * h)
            assert np.allclose(fisher.jacobian, jac, rtol=1e-5,
                               atol=1e-6 * np.abs(jac).max(axis=0))
            assert np.allclose(fisher.expected_counts,
                               _tied_mean(theta, EK_ENERGY, 2, split, ratio, 0.15), rtol=1e-12)
            for mode in report.modes.modes:
                assert mode.whitened_norm == pytest.approx(1.0, abs=1e-9)

    def test_tying_never_loses_information(self, ek_singlet):
        """The same estimate with every parameter free: for the parameters
        both models have, the tied bound is never the larger (Loewner)."""
        from toyomacro.voigtfit.multipeak_solver import build_bg_design

        cand = ek_singlet.candidates[1]
        tied = identifiability_from_exact_k(ek_singlet, 2, EK_ENERGY, intensity="raw_counts")
        peaks = [VoigtPeak(a, c, cand.sigma, 0.15) for a, c in zip(cand.amplitudes, cand.centers)]
        bg = Background(basis=build_bg_design(EK_ENERGY, 1).astype(np.float64).T,
                        coefficients=np.array(cand.bg_coefficients), names=("bg_0", "bg_1"),
                        estimated=(True, True))
        free = identifiability_at_estimate(EK_ENERGY, peaks, bg, intensity="raw_counts")
        free_sd = {a.name: a.sd for a in free.parameters}
        tied_sd = {a.name: a.sd for a in tied.parameters}
        for name in ("amp_0", "amp_1", "center_0", "center_1", "bg_0", "bg_1"):
            assert tied_sd[name] <= free_sd[name] * (1.0 + 1e-9)

    def test_scales_and_labels_of_the_sub_model(self, ek_doublet):
        report = identifiability_from_exact_k(ek_doublet, 2, EK_ENERGY, intensity="raw_counts")
        cand = ek_doublet.candidates[1]
        fwhm = voigt_fwhm(cand.sigma ** 2, 0.15)
        assert report.modes.scales["var_shared"] == pytest.approx(fwhm**2 / (8 * math.log(2)))
        assert report.modes.scales["amp_1"] == pytest.approx(cand.amplitudes[1])
        covariance = np.linalg.inv(report.modes.fisher.fisher)
        for i, row in enumerate(report.parameters):
            assert row.sd == pytest.approx(math.sqrt(covariance[i, i]), rel=1e-8)
            if row.role == "background":
                assert row.status == "not_assessed" and row.reference_scale is None
            else:
                assert row.relative_sd == pytest.approx(row.sd / row.reference_scale)
        assert set(report.near_boundary) == {"var_shared", "var_shared_fit_bounds"}
        json.dumps(report.to_dict())

    def test_the_axis_order_does_not_matter(self, ek_singlet):
        forward = identifiability_from_exact_k(ek_singlet, 2, EK_ENERGY, intensity="raw_counts")
        backward = identifiability_from_exact_k(ek_singlet, 2, EK_ENERGY[::-1],
                                                intensity="raw_counts")
        for a, b in zip(forward.parameters, backward.parameters):
            assert b.sd == pytest.approx(a.sd, rel=1e-9)

    def test_withheld_cases(self, ek_singlet):
        import dataclasses

        def with_candidate(**changes):
            cands = list(ek_singlet.candidates)
            cands[1] = dataclasses.replace(cands[1], **changes)
            return dataclasses.replace(ek_singlet, candidates=tuple(cands))

        def withheld(rep, **kw):
            return identifiability_from_exact_k(rep, 2, EK_ENERGY, intensity="raw_counts",
                                                **kw).withheld

        cand = ek_singlet.candidates[1]
        assert "count_rate" in identifiability_from_exact_k(
            ek_singlet, 2, EK_ENERGY, intensity="count_rate").withheld
        assert "not positive" in withheld(with_candidate(amplitudes=(cand.amplitudes[0], -1.0)))
        assert "sigma_at_bound" in withheld(with_candidate(boundary_flags=("sigma_at_bound",)))
        assert "center_0_at_energy_bound" in withheld(
            with_candidate(boundary_flags=("center_0_at_energy_bound",)))
        assert "mean" in withheld(with_candidate(bg_coefficients=(-1.0e5, 0.0)))
        assert "converge" in withheld(with_candidate(
            score=dataclasses.replace(cand.score, success=False)))
        plugin = _exact_k_report(plugin="shirley")
        assert "plug-in" in withheld(plugin)
        with pytest.raises(ValueError, match="no candidate"):
            identifiability_from_exact_k(ek_singlet, 7, EK_ENERGY, intensity="raw_counts")
        with pytest.raises(ValueError, match="points"):
            identifiability_from_exact_k(ek_singlet, 2, EK_ENERGY[:-1], intensity="raw_counts")
        with pytest.raises(ValueError, match="intensity must be one of"):
            identifiability_from_exact_k(ek_singlet, 2, EK_ENERGY, intensity="counts")

    @staticmethod
    def _variant(rep, **changes):
        """The report with candidate K = 2 changed; sigma and centers are
        changed in the components too, where the adapter reads them."""
        import dataclasses

        cand = rep.candidates[1]
        comps = list(cand.model.components)
        if "sigma" in changes:
            comps = [dataclasses.replace(c, sigma=changes["sigma"]) for c in comps]
        if "centers" in changes:
            comps = [dataclasses.replace(c, center=x) for c, x in zip(comps, changes["centers"])]
        cand = dataclasses.replace(cand, model=dataclasses.replace(
            cand.model, components=tuple(comps)), **changes)
        return dataclasses.replace(rep, candidates=(rep.candidates[0], cand))

    def test_labels_and_flags_of_the_sub_model(self, ek_singlet):
        """Coincident components are rank deficient with two null modes; a
        threshold below every bound makes every row weak; a small,
        faint shared width is flagged against 0; the reference scales and
        the conditional bound are what the docstring says."""
        twin = self._variant(ek_singlet, centers=(285.5, 285.5))
        report = identifiability_from_exact_k(twin, 2, EK_ENERGY, intensity="raw_counts")
        status = {a.name: a.status for a in report.parameters}
        assert all(status[n] == "rank_deficient"
                   for n in ("amp_0", "amp_1", "center_0", "center_1"))
        assert len(report.modes.null_modes) == 2

        strict = identifiability_from_exact_k(
            ek_singlet, 2, EK_ENERGY, intensity="raw_counts",
            thresholds=IdentifiabilityThresholds(weak_relative_sd=1e-9))
        assert all(a.status == "weakly_identified" for a in strict.parameters
                   if a.role != "background")

        faint = self._variant(ek_singlet, sigma=0.05, amplitudes=(300.0, 100.0))
        report = identifiability_from_exact_k(faint, 2, EK_ENERGY, intensity="raw_counts")
        row = {a.name: a for a in report.parameters}["var_shared"]
        assert row.value < 3.0 * row.sd and report.near_boundary["var_shared"]
        assert row.peak is None and row.sd_conditional == pytest.approx(
            1.0 / math.sqrt(report.modes.fisher.fisher[4, 4]))
        wide = identifiability_from_exact_k(ek_singlet, 2, EK_ENERGY, intensity="raw_counts")
        assert not wide.near_boundary["var_shared"]
        cand = ek_singlet.candidates[1]
        by_name = {a.name: a for a in wide.parameters}
        assert by_name["center_1"].reference_scale == pytest.approx(
            voigt_fwhm(cand.sigma ** 2, 0.15))
        assert (by_name["amp_1"].peak, by_name["center_1"].peak) == (1, 1)

    def test_flag_at_the_fits_own_sigma_bound(self, ek_singlet):
        """exact_k constrains sigma to sigma_bounds (0.02 eV at the low end):
        a variance a few bounds from 0.02**2 is flagged there, though far
        from 0 in its own units."""
        near = self._variant(ek_singlet, sigma=0.0205, amplitudes=(1.0e3, 1.0e3))
        report = identifiability_from_exact_k(near, 2, EK_ENERGY, intensity="raw_counts")
        assert report.near_boundary["var_shared_fit_bounds"]
        far = identifiability_from_exact_k(ek_singlet, 2, EK_ENERGY, intensity="raw_counts")
        assert not far.near_boundary["var_shared_fit_bounds"]

    def test_generic_flags_are_per_width(self):
        """A nearly Gaussian peak: gamma is flagged, the variance is not."""
        sigma, gamma = idf._widths_at_fwhm(100.0, 1.0)
        energy = np.linspace(-5.0, 5.0, 501)
        peak = VoigtPeak(1.0e4, 0.0, sigma, gamma)
        report = identifiability_at_estimate(energy, [peak], _flat(energy, peak),
                                             intensity="raw_counts")
        assert report.near_boundary == {"var_0": False, "gamma_0": True}

    def test_arguments_reach_the_evaluation(self):
        e = MODE_ENERGY
        bg = linear_background(e, 300.0, 10.0)
        cut = IdentifiabilityThresholds(weak_relative_sd=1e-9)
        report = identifiability_at_estimate(e, [MODE_PEAK], bg, intensity="raw_counts", k=2,
                                             variance_floor=0.05, thresholds=cut)
        assert len(report.modes.modes) == 2 and report.modes.variance_floor == 0.05
        assert report.assessment.variance_floor == 0.05
        assert all(m.status == "weakly_identified" for m in report.modes.modes)
        for word in ("not a confidence interval", "unbiased", "correct"):
            assert word in report.caveat

    def test_malformed_inputs(self, ek_singlet):
        import dataclasses

        assert "not finite" in identifiability_from_exact_k(
            self._variant(ek_singlet, bg_coefficients=(200.0,)), 2, EK_ENERGY,
            intensity="raw_counts").withheld
        assert "not finite" in identifiability_from_exact_k(
            self._variant(ek_singlet, bg_coefficients=(200.0, float("nan"))), 2, EK_ENERGY,
            intensity="raw_counts").withheld
        unrecorded = dataclasses.replace(ek_singlet, background={})
        assert "does not record" in identifiability_from_exact_k(
            unrecorded, 2, EK_ENERGY, intensity="raw_counts").withheld
        with pytest.raises(ValueError, match="does not contain the fitted centers"):
            identifiability_from_exact_k(ek_singlet, 2, EK_ENERGY + 1000.0,
                                         intensity="raw_counts")
        doublet = _exact_k_report(doublet=True)
        cand = doublet.candidates[1]
        broken = dataclasses.replace(cand, model=dataclasses.replace(
            cand.model, components=tuple(dataclasses.replace(c, branch_ratio=0.0)
                                         for c in cand.model.components)))
        with pytest.raises(ValueError, match="branch_ratio"):
            identifiability_from_exact_k(
                dataclasses.replace(doublet, candidates=(doublet.candidates[0], broken)),
                2, EK_ENERGY, intensity="raw_counts")
