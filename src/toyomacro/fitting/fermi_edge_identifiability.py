"""Identifiability of the resolution and the temperature on a Fermi edge (experimental).

How much a counting spectrum of a metal Fermi edge says about the
instrumental Gaussian variance ``v = sigma**2`` and the thermal scale
``tau = (kT)**2``, once E_F, the amplitude, the density of states and
the background are estimated from the same spectrum.

The model is the one ``fitting.fermi_edge`` fits, written in these
coordinates::

    mu(E) = A * [D(y) f(y; tau)] (x) G_v  +  background(E)

    y       distance from E_F into the occupied side, y = s*(E - E_F),
            s = +1 on a binding-energy axis and -1 on a kinetic one
    D(y)    1 + c1*y+ + c2*y+**2 with y+ = max(y, 0): the density of
            states rises past the edge and is flat on the empty side, as in
            ``fermi_edge``. The slope therefore changes at E_F -- inside
            the thermal tail, where the temperature information lives.
    f       Fermi-Dirac occupation, f = 1 / (1 + exp(-y / sqrt(tau)))
    G_v     Gaussian of variance v

Why these coordinates. Convolution adds cumulants: the occupation's
derivative -df/dy is a logistic density of scale kT, with second and
fourth cumulants (pi**2/3) tau and (2 pi**4/15) tau**2; the Gaussian
adds v to the second and nothing to the fourth. So the data weigh
``v + (pi**2/3) tau`` strongly, and only the fourth cumulant separates v
from tau. In ``sigma`` and ``T`` both first-order sensitivities vanish at
zero (``d/dsigma = 2 sigma d/dv``); in ``v`` and ``tau`` they do not.

Derivatives. Every column is an integral against the Gaussian,
``int G_v(u - y) q(y) dy``, of a function q that is smooth except at
y = 0:

- d/dE_F from ``h' = (D f)'`` (h is continuous, so no delta term);
- d/dv from the heat equation, ``d mu/dv = (1/2) d2 mu/du2 = (1/2) [G (x) h''
  + (c1/2) G_v(u)]`` -- the second term is the kink of D at E_F, where h'
  jumps by ``c1 f(0) = c1/2``. Written this way the integrand is bounded
  and nothing cancels as v goes to 0;
- d/dtau from ``df/dtau = -(y / (2 tau)) df/dy``;
- d/dc1, d/dc2 from ``y+ f`` and ``y+**2 f``.

Two routes evaluate them:

- 'quadrature': Gauss-Legendre panels (16 points) with breakpoints at
  y = 0 and graded in kT around it, and graded in sigma around each
  energy point, cut at 10 sigma. The Fermi function's poles sit on the
  imaginary axis at odd multiples of i pi kT, so panels graded from
  y = 0 keep every pole well outside their Bernstein ellipses.
- 'series': for kT << sigma, the Gaussian expanded about E_F against the
  moments of ``D (f - step)``, which are Dirichlet eta values times powers
  of kT; the step part is closed form in the normal CDF and density.
  The series is asymptotic in kT/sigma, not convergent (see
  ``_SERIES_TERMS``).

Measured against an adaptive quadrature (``scipy.integrate.quad``) of the
same integrals, for kT/sigma from 0.02 to 3, both DOS forms and five sets
of coefficients: the quadrature route is within 2.6e-13 of it in every
column, relative to the column's largest magnitude, and the series within
6e-13 where it is used. The quadrature's d/dtau integrand is odd about
E_F with a magnitude of 1/kT, so it loses about eps * sigma/kT: 9e-12 at
kT/sigma = 1e-3, 9e-10 at 1e-5, against the series. 'auto' therefore
takes the series below kT/sigma = 0.07, where the two agree to 2e-13;
above 0.15 the series fails (1e-7 at 0.15, order one at 0.3). These are
numbers for these settings, not bounds.

At ``tau = 0`` the occupation is a step and everything is closed form,
including the limit ``d mu/dtau = A [(pi**2/6) G_v'(u) - (pi**2/12) c1
G_v(u)]``. At ``v = 0`` the convolution is the identity and the columns
are the integrands at the energy point. ``v = 0`` together with
``tau = 0`` is a bare step, where none of this is defined.

A Fisher matrix computed from this model is a statement about the
model, not a measurement; what ``voigtfit.crlb``'s module docstring says
about unbiasedness and correct specification applies.

Not exported from ``toyomacro.fitting``; not wired to any CLI.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np
from scipy.special import expit, ndtr, zeta

from .fermi_edge import KB_EV, _sign_of

__all__ = [
    "EdgeDerivatives",
    "edge_derivatives",
    "tau_from_temperature",
    "temperature_from_tau",
]

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_PI2 = math.pi * math.pi

# Gauss-Legendre panels: 16 points each.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
# Breakpoints around each energy point, in units of sigma; +-10 bounds the
# integral (the Gaussian is 2e-22 of its peak there).
_GAUSS_BREAKS = np.array([-10.0, -6.0, -3.5, -2.0, -1.0, -0.35, 0.35, 1.0, 2.0, 3.5, 6.0, 10.0])
# Breakpoints around E_F, in units of kT, on both sides of y = 0. Past 60 kT
# f differs from the step by e**-60.
_THERMAL_BREAKS = np.array([0.5, 1.5, 3.5, 7.0, 13.0, 22.0, 36.0, 60.0])


def tau_from_temperature(temperature: float) -> float:
    """tau = (k_B T)**2 in eV**2 for T in K."""
    return (KB_EV * float(temperature)) ** 2


def temperature_from_tau(tau: float) -> float:
    """T in K for tau = (k_B T)**2 in eV**2."""
    return math.sqrt(max(float(tau), 0.0)) / KB_EV


class EdgeDerivatives(NamedTuple):
    """Mean and first derivatives of the edge, without background.

    All arrays have the shape of ``energy``. ``value`` is the expected
    count per channel at unit exposure; the derivatives are of that.
    """

    value: np.ndarray
    d_ef: np.ndarray
    d_amplitude: np.ndarray
    d_dos_c1: np.ndarray
    d_dos_c2: np.ndarray
    d_variance: np.ndarray
    d_tau: np.ndarray


# ---------------------------------------------------------------------------
# Pieces of the integrand, as functions of y (occupied side positive)
# ---------------------------------------------------------------------------


def _dos(y, c1, c2, form):
    """dD/dc1 (= y or y+), D, D' and D''.

    'occupied': D = 1 + c1 y+ + c2 y+**2, flat for y < 0 with a kink at
    y = 0, as ``fermi_edge`` has it. 'both_sides': the same polynomial on
    both sides of E_F, smooth.
    """
    if form == "both_sides":
        return y, 1.0 + c1 * y + c2 * y * y, c1 + 2.0 * c2 * y, np.full_like(y, 2.0 * c2)
    occupied = y > 0.0
    yp = np.where(occupied, y, 0.0)
    d = 1.0 + c1 * yp + c2 * yp * yp
    d1 = np.where(occupied, c1 + 2.0 * c2 * y, 0.0)
    d2 = np.where(occupied, 2.0 * c2, 0.0)
    return yp, d, d1, d2


def _integrands(y, tau, c1, c2, form):
    """q(y) for every column: h, h', h'', dD/dc1 f, dD/dc2 f, D df/dtau."""
    kt = math.sqrt(tau)
    f = expit(y / kt)
    f1 = f * (1.0 - f) / kt
    f2 = f1 * (1.0 - 2.0 * f) / kt
    yc, d, d1, d2 = _dos(y, c1, c2, form)
    h = d * f
    h1 = d1 * f + d * f1
    h2 = d2 * f + 2.0 * d1 * f1 + d * f2
    f_tau = -(y / (2.0 * tau)) * f1
    return h, h1, h2, yc * f, yc * yc * f, d * f_tau


def _gauss(x, variance):
    return np.exp(-0.5 * x * x / variance) / math.sqrt(2.0 * math.pi * variance)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _nodes(u: np.ndarray, sigma: float, kt: float):
    """Quadrature nodes and weights per energy point, shape (n, n_nodes)."""
    n = u.size
    lo, hi = u - 10.0 * sigma, u + 10.0 * sigma
    thermal = np.concatenate([-kt * _THERMAL_BREAKS[::-1], [0.0], kt * _THERMAL_BREAKS])
    cand = np.concatenate([u[:, None] + sigma * _GAUSS_BREAKS[None, :],
                           np.broadcast_to(thermal, (n, thermal.size))], axis=1)
    cand = np.sort(np.clip(cand, lo[:, None], hi[:, None]), axis=1)
    a, b = cand[:, :-1], cand[:, 1:]
    half = 0.5 * (b - a)
    y = (a + half)[..., None] + half[..., None] * _GL_X
    w = half[..., None] * _GL_W
    return y.reshape(n, -1), w.reshape(n, -1)


def _quadrature(u, variance, tau, c1, c2, form):
    sigma = math.sqrt(variance)
    y, w = _nodes(u, sigma, math.sqrt(tau))
    g = w * _gauss(u[:, None] - y, variance)
    h, h1, h2, qc1, qc2, qtau = _integrands(y, tau, c1, c2, form)
    # 'occupied': h' jumps by c1 f(0) = c1/2 at y = 0, a delta in h''
    kink = 0.5 * c1 * _gauss(u, variance) if form == "occupied" else 0.0
    return ((g * h).sum(1), (g * h1).sum(1), 0.5 * ((g * h2).sum(1) + kink),
            (g * qc1).sum(1), (g * qc2).sum(1), (g * qtau).sum(1))


def _step_part(u, variance, c1, c2):
    """[D step] (x) G_v and its derivatives, closed form (sigma > 0)."""
    sigma = math.sqrt(variance)
    t = u / sigma
    cdf, pdf = ndtr(t), np.exp(-0.5 * t * t) / _SQRT_2PI
    gau = pdf / sigma                      # G_v(u)
    gau1 = -t * pdf / variance             # G_v'(u)
    value = cdf * (1.0 + c1 * u + c2 * (u * u + variance)) + sigma * pdf * (c1 + c2 * u)
    d_u = gau + c1 * cdf + 2.0 * c2 * (u * cdf + sigma * pdf)
    d_v = 0.5 * (gau1 + c1 * gau + 2.0 * c2 * cdf)
    d_c1 = u * cdf + sigma * pdf
    d_c2 = (u * u + variance) * cdf + u * sigma * pdf
    return value, d_u, d_v, d_c1, d_c2, gau, gau1


def _eta(s: np.ndarray) -> np.ndarray:
    """Dirichlet eta; eta(1) = ln 2."""
    s = np.asarray(s, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (1.0 - 2.0 ** (1.0 - s)) * zeta(s)
    return np.where(s == 1.0, math.log(2.0), out)


def _hermite_functions(t: np.ndarray, n_max: int) -> np.ndarray:
    """He_n(t) phi(t) for n = 0..n_max, shape (n_max + 1, len(t))."""
    phi = np.exp(-0.5 * t * t) / _SQRT_2PI
    out = np.empty((n_max + 1, t.size))
    out[0] = phi
    if n_max >= 1:
        out[1] = t * phi
    for n in range(1, n_max):
        out[n + 1] = t * out[n] - n * out[n - 1]
    return out


# Terms of the asymptotic expansion in kT / sigma. Term n carries He_n(t) phi(t)
# (kT/sigma)**(n+1) eta(n+1); He_n phi grows like sqrt(n!), so successive terms
# shrink only while n < (sigma/kT)**2 -- the same structure as the large-|z|
# expansion in voigtfit.identifiability. _SERIES_TERMS must stay below
# (sigma/kT)**2 at the switch; asserted in the tests.
_SERIES_TERMS = 30


def _series(u, variance, tau, c1, c2, form, n_terms=None):
    """Step part in closed form plus the moment expansion of D (f - step)."""
    n_terms = _SERIES_TERMS if n_terms is None else n_terms
    sigma, kt = math.sqrt(variance), math.sqrt(tau)
    value, d_u, d_v, d_c1, d_c2, _, _ = _step_part(u, variance, c1, c2)
    t = u / sigma
    he = _hermite_functions(t, n_terms + 2)  # He_n phi, up to n_terms + 2 for d/du and d/dv
    n = np.arange(n_terms + 1, dtype=np.float64)
    # M_n / n! = kT**(n+1) (a_n + c1 kT b_n + c2 kT**2 c_n), from
    # int_0^inf y**m / (1 + exp(y/kT)) dy = kT**(m+1) m! eta(m+1): f - step is
    # +1/(1 + exp(|y|/kT)) below E_F and -1/(1 + exp(y/kT)) above it.
    eta1, eta2, eta3 = _eta(n + 1.0), _eta(n + 2.0), _eta(n + 3.0)
    a = ((-1.0) ** n - 1.0) * eta1
    if form == "both_sides":  # the DOS terms reach below E_F too
        b = ((-1.0) ** (n + 1.0) - 1.0) * (n + 1.0) * eta2
        c = ((-1.0) ** n - 1.0) * (n + 1.0) * (n + 2.0) * eta3
    else:                      # 'occupied': above E_F only, hence odd powers of kT
        b = -(n + 1.0) * eta2
        c = -(n + 1.0) * (n + 2.0) * eta3
    # r**(n+1) with r = kT/sigma; the c1 and c2 terms carry one and two more kT
    r = kt / sigma
    rp = r ** (n + 1.0)
    coef = rp * (a + c1 * kt * b + c2 * tau * c)
    # d coef / d tau: kT**(n+1+k) -> ((n+1+k)/2) kT**(n+1+k) / tau
    dcoef_tau = rp * ((n + 1.0) * a + c1 * kt * (n + 2.0) * b + c2 * tau * (n + 3.0) * c) / (
        2.0 * tau)
    # term n = He_n(t) phi(t) (kT/sigma)**(n+1) (a_n + ...): the Gaussian's n-th
    # derivative, G^(n)(u) = (-1)**n He_n(t) phi(t) / sigma**(n+1), against M_n / n!
    base = he[: n_terms + 1]
    value = value + coef @ base
    d_u = d_u + coef @ (-he[1 : n_terms + 2] / sigma)
    # heat equation term by term: dG^(n)/dv = G^(n+2) / 2
    d_v = d_v + 0.5 * coef @ (he[2 : n_terms + 3] / variance)
    d_c1 = d_c1 + (rp * kt * b) @ base
    d_c2 = d_c2 + (rp * tau * c) @ base
    d_tau = dcoef_tau @ base
    return value, d_u, d_v, d_c1, d_c2, d_tau


def _pointwise(u, tau, c1, c2, form):
    """v = 0: the convolution is the identity."""
    if form == "occupied" and c1 != 0.0 and np.any(u == 0.0):
        raise ValueError("variance = 0 with a channel exactly at E_F and c1 != 0: d/dv is a delta")
    h, h1, h2, qc1, qc2, qtau = _integrands(u, tau, c1, c2, form)
    return h, h1, 0.5 * h2, qc1, qc2, qtau


def edge_derivatives(
    energy: np.ndarray,
    ef: float,
    amplitude: float,
    variance: float,
    tau: float,
    *,
    dos_c1: float = 0.0,
    dos_c2: float = 0.0,
    convention: str = "BE",
    dos_form: Literal["occupied", "both_sides"] = "occupied",
    route: Literal["auto", "quadrature", "series"] = "auto",
) -> EdgeDerivatives:
    """Mean of the Fermi-edge model and its derivatives, without background.

    Args:
        energy: Energy axis (eV)
        ef: Fermi level (eV), where f = 1/2 before broadening
        amplitude: Height of the occupied-side step at E_F, counts per
            channel at unit exposure
        variance: Gaussian variance v = sigma**2 (eV**2), >= 0
        tau: (kT)**2 (eV**2), >= 0
        dos_c1, dos_c2: Density-of-states coefficients (1/eV, 1/eV**2) on
            the occupied side; D must stay positive over the energies used
        convention: 'BE' or 'KE', as in ``fermi_edge``
        dos_form: 'occupied' (default) is ``fermi_edge``'s density of
            states, flat below E_F with a kink at E_F; 'both_sides' is the
            same polynomial continued smoothly below E_F, for comparison
        route: 'auto' uses the series where kT/sigma is below the switch
            and quadrature elsewhere

    Returns:
        EdgeDerivatives
    """
    if not (math.isfinite(variance) and math.isfinite(tau)) or variance < 0.0 or tau < 0.0:
        raise ValueError(f"need finite variance >= 0 and tau >= 0, got {variance}, {tau}")
    if variance == 0.0 and tau == 0.0:
        raise ValueError("variance = 0 and tau = 0 is a bare step: nothing here is defined")
    if dos_form not in ("occupied", "both_sides"):
        raise ValueError(f"dos_form must be 'occupied' or 'both_sides', got {dos_form!r}")
    if route not in ("auto", "quadrature", "series"):
        raise ValueError(f"route must be 'auto', 'quadrature' or 'series', got {route!r}")
    energy = np.asarray(energy, dtype=np.float64)
    s = _sign_of(convention)
    u = s * (energy - ef)
    # D must be positive wherever the integrand is not negligible: the occupied
    # side up to the convolution's reach, and below E_F (for 'both_sides') down
    # to where f has fallen to e**-37
    y_hi = max(float(np.max(u)), 0.0) + 10.0 * math.sqrt(variance) + 37.0 * math.sqrt(tau)
    y_lo = -37.0 * math.sqrt(tau) if dos_form == "both_sides" else 0.0
    _, d_check, _, _ = _dos(np.linspace(y_lo, y_hi, 513), dos_c1, dos_c2, dos_form)
    if np.any(d_check <= 0.0):
        raise ValueError("the density of states is not positive over the energies the "
                         "convolution reaches")

    if variance == 0.0:
        value, d_u, d_v, d_c1, d_c2, d_tau = _pointwise(u, tau, dos_c1, dos_c2, dos_form)
    elif tau == 0.0:
        value, d_u, d_v, d_c1, d_c2, gau, gau1 = _step_part(u, variance, dos_c1, dos_c2)
        # the tau**1 term of the series: n = 1 (flat part) and n = 0 (c1 part)
        c1_limit = 12.0 if dos_form == "occupied" else 6.0
        d_tau = (_PI2 / 6.0) * gau1 - (_PI2 / c1_limit) * dos_c1 * gau
    else:
        if route == "auto":
            route = "series" if math.sqrt(tau / variance) < _SERIES_SWITCH else "quadrature"
        if route == "series":
            value, d_u, d_v, d_c1, d_c2, d_tau = _series(u, variance, tau, dos_c1, dos_c2,
                                                         dos_form)
        else:
            value, d_u, d_v, d_c1, d_c2, d_tau = _quadrature(u, variance, tau, dos_c1, dos_c2,
                                                             dos_form)

    a = float(amplitude)
    return EdgeDerivatives(
        value=a * value, d_ef=-s * a * d_u, d_amplitude=value, d_dos_c1=a * d_c1,
        d_dos_c2=a * d_c2, d_variance=a * d_v, d_tau=a * d_tau,
    )


# kT / sigma below which 'auto' takes the series. Set from measurement; see the tests.
_SERIES_SWITCH = 0.07
