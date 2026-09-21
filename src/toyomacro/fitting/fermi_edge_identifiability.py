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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import numpy as np
from scipy.special import expit, ndtr, zeta

from .._identifiability import (
    Background,
    EffectiveInformation,
    constant_background,
    covariance_bound,
    effective_information,
    linear_background,
    poisson_fisher_matrix,
)
from ..voigtfit.crlb import _NULL_PROJECTION_TOL
from .fermi_edge import KB_EV, _sign_of

__all__ = [
    "EDGE_PARAMETERIZATIONS",
    "TEMPERATURE_MODES",
    "Background",
    "EdgeDerivatives",
    "EdgeFisherResult",
    "EdgeIdentifiabilityReport",
    "EdgeParameterAssessment",
    "EdgeScanGrid",
    "EdgeThresholds",
    "EdgeWidthInformation",
    "FermiEdge",
    "InstrumentalResolution",
    "constant_background",
    "edge_derivatives",
    "edge_fisher",
    "edge_mle_model",
    "edge_width_information",
    "edge_wls_covariance",
    "assess_edge_identifiability",
    "scan_edge_identifiability",
    "linear_background",
    "tau_from_temperature",
    "temperature_from_tau",
]

EdgeParameterization = Literal["var_tau", "sigma_T"]
EDGE_PARAMETERIZATIONS: tuple[str, ...] = ("var_tau", "sigma_T")
TemperatureMode = Literal["free", "fixed_temperature", "temperature_prior"]
TEMPERATURE_MODES: tuple[str, ...] = ("free", "fixed_temperature", "temperature_prior")

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
    # D must be positive where it carries weight: the occupied side up to the
    # convolution's reach, and, for 'both_sides', the first few kT below E_F,
    # where f is still of order 1. Deeper than that f suppresses D
    # exponentially (e**-3 at 3 kT), and the check that every expected count is
    # positive covers what is left.
    y_hi = max(float(np.max(u)), 0.0) + 10.0 * math.sqrt(variance) + 37.0 * math.sqrt(tau)
    y_lo = -3.0 * math.sqrt(tau) if dos_form == "both_sides" else 0.0
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


# ---------------------------------------------------------------------------
# Model specification and Poisson Fisher matrix
# ---------------------------------------------------------------------------

_DOS_TERMS = {"flat": (), "linear": ("dos_c1",), "quadratic": ("dos_c1", "dos_c2")}


@dataclass(frozen=True)
class FermiEdge:
    """A Fermi edge, in the units a measurement is quoted in.

    Attributes:
        ef: Fermi level (eV)
        amplitude: Height of the occupied-side step at E_F, counts per
            channel at unit exposure; must be positive
        sigma: Instrumental Gaussian standard deviation (eV), >= 0.
            FWHM = 2 sqrt(2 ln 2) sigma
        temperature: Electron temperature (K), >= 0
        dos_c1, dos_c2: Density-of-states coefficients (1/eV, 1/eV**2)
        dos: Which DOS coefficients are estimated, as in ``fit_fermi_edge``:
            'flat' (neither; both must be 0), 'linear' (c1; c2 must be 0)
            or 'quadratic'
    """

    ef: float
    amplitude: float
    sigma: float
    temperature: float
    dos_c1: float = 0.0
    dos_c2: float = 0.0
    dos: Literal["flat", "linear", "quadratic"] = "linear"

    def __post_init__(self) -> None:
        values = (self.ef, self.amplitude, self.sigma, self.temperature, self.dos_c1, self.dos_c2)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"edge parameters must be finite, got {values}")
        if not self.amplitude > 0.0:
            raise ValueError(f"amplitude must be positive, got {self.amplitude}")
        if self.sigma < 0.0 or self.temperature < 0.0:
            raise ValueError("need sigma >= 0 and temperature >= 0")
        if self.sigma == 0.0 and self.temperature == 0.0:
            raise ValueError("sigma = 0 and temperature = 0 is a bare step")
        if self.dos not in _DOS_TERMS:
            raise ValueError(f"dos must be 'flat', 'linear' or 'quadratic', got {self.dos!r}")
        if (self.dos == "flat" and (self.dos_c1 or self.dos_c2)) or (
                self.dos == "linear" and self.dos_c2):
            raise ValueError(f"dos={self.dos!r} holds the omitted coefficients at 0")

    @property
    def variance(self) -> float:
        return self.sigma * self.sigma

    @property
    def tau(self) -> float:
        return tau_from_temperature(self.temperature)


@dataclass
class EdgeFisherResult:
    """Poisson Fisher matrix of an edge-plus-background model.

    Attributes:
        fisher: Fisher matrix in the chosen coordinates, (n_params, n_params)
        param_names: e.g. ('ef', 'amplitude', 'dos_c1', 'variance', 'tau', 'bg_level')
        param_roles: 'position', 'amplitude', 'dos', 'width' or 'background'
        param_values: Parameter values in the chosen coordinates
        width_index: Positions of the two width parameters
        jacobian: d(expected counts)/d(parameters) at the stated exposure,
            (n_energy, n_params)
        expected_counts: Poisson mean per channel at the stated exposure
        parameterization: 'var_tau' or 'sigma_T'
        exposure: Multiplier on the whole mean. The data part of ``fisher``
            is linear in it; a prior is not
        temperature_mode: 'free', 'fixed_temperature' or 'temperature_prior'
        prior: The prior information included in ``fisher``, same shape
            (zero unless ``temperature_mode='temperature_prior'``); the
            information from the data alone is ``fisher - prior``
        config: Model echo
    """

    fisher: np.ndarray
    param_names: tuple[str, ...]
    param_roles: tuple[str, ...]
    param_values: np.ndarray
    width_index: tuple[int, ...]
    jacobian: np.ndarray
    expected_counts: np.ndarray
    parameterization: str
    exposure: float
    temperature_mode: str = "free"
    prior: np.ndarray | None = None
    config: dict[str, Any] = field(default_factory=dict)


def edge_fisher(
    energy: np.ndarray,
    edge: FermiEdge,
    background: Background | None = None,
    *,
    parameterization: EdgeParameterization = "var_tau",
    temperature_mode: TemperatureMode = "free",
    temperature_sd: float | None = None,
    exposure: float = 1.0,
    convention: str = "BE",
    dos_form: Literal["occupied", "both_sides"] = "occupied",
    route: Literal["auto", "quadrature", "series"] = "auto",
) -> EdgeFisherResult:
    """Fisher matrix ``J^T diag(1/mu) J`` for Poisson counts on a Fermi edge.

    ``mu(E) = exposure * (edge(E) + background(E))``. Parameters, in
    order: E_F, the amplitude, the DOS coefficients ``edge.dos`` estimates,
    the two width coordinates, and every background term flagged
    ``estimated``. A known background raises ``mu`` without adding a
    parameter.

    Width coordinates:

    - 'var_tau': (v, tau) = (sigma**2, (kT)**2), eV**2 both. Regular at
      sigma = 0 and at T = 0 in the sense that each column stays finite.
      At tau = 0 the tau column is, to first order, a combination of the
      v, E_F and DOS columns (the Sommerfeld expansion; see the module
      docstring), so what the data say about tau apart from v comes from
      higher orders and vanishes there. That is a property of the model,
      not of the coordinates, and no coordinate removes it.
    - 'sigma_T': (sigma, T) in eV and K, related by ``I_phi = T^T I T``
      with ``dv/dsigma = 2 sigma`` and ``dtau/dT = 2 k_B**2 T``: singular
      at sigma = 0 and at T = 0 by construction of the coordinates.

    The temperature, three ways:

    - 'free': estimated with everything else.
    - 'fixed_temperature': held at ``edge.temperature``; the temperature
      coordinate is dropped. **Fixing the temperature is an assumption
      that the sample's electron temperature equals the given value** --
      a thermocouple reading is not that. If it is wrong, what the fit
      calls resolution absorbs the difference through ``v + (pi**2/3)
      tau``.
    - 'temperature_prior': estimated, with a normal prior of standard
      deviation ``temperature_sd`` (K) centred on ``edge.temperature``.
      Its information is added to the temperature coordinate's diagonal:
      ``1/sd_T**2`` in 'sigma_T', ``1/sd_tau**2`` in 'var_tau' with
      ``sd_tau = 2 k_B**2 T sd_T``, the same prior carried through the
      linearised map, so the two coordinates stay one model. This is the
      prior's information at the stated temperature added to the data's;
      the van Trees bound, which averages over the prior, is not computed.
      As ``temperature_sd`` goes to 0 the result tends to
      'fixed_temperature', and as it grows, to 'free'.

    Args:
        energy: Energy axis (eV)
        edge: The edge
        background: None, or a ``Background`` on the same axis
        parameterization: 'var_tau' (default) or 'sigma_T'
        temperature_mode: 'free' (default), 'fixed_temperature' or
            'temperature_prior'
        temperature_sd: Standard deviation of the temperature prior (K),
            for 'temperature_prior' only
        exposure: Multiplier on the whole mean (acquisition time, flux)
        convention: 'BE' or 'KE'
        dos_form: As in ``edge_derivatives``
        route: As in ``edge_derivatives``

    Returns:
        EdgeFisherResult. The matrix is returned as computed and can be
        singular; nothing here inverts it.

    Raises:
        ValueError: if the expected count is not positive in every
            channel (without a background the unoccupied side underflows
            a few sigma past the edge: add one or narrow the window).
    """
    if parameterization not in EDGE_PARAMETERIZATIONS:
        raise ValueError(
            f"parameterization must be one of {EDGE_PARAMETERIZATIONS}, got {parameterization!r}")
    if not exposure > 0.0:
        raise ValueError(f"exposure must be positive, got {exposure}")
    if temperature_mode not in TEMPERATURE_MODES:
        raise ValueError(
            f"temperature_mode must be one of {TEMPERATURE_MODES}, got {temperature_mode!r}")
    if temperature_mode == "temperature_prior":
        if temperature_sd is None or not (math.isfinite(temperature_sd) and temperature_sd > 0.0):
            raise ValueError("temperature_prior needs a positive, finite temperature_sd (K)")
    elif temperature_sd is not None:
        raise ValueError("temperature_sd applies to temperature_mode='temperature_prior' only")
    energy = np.asarray(energy, dtype=np.float64)
    d = edge_derivatives(energy, edge.ef, edge.amplitude, edge.variance, edge.tau,
                         dos_c1=edge.dos_c1, dos_c2=edge.dos_c2, convention=convention,
                         dos_form=dos_form, route=route)

    if parameterization == "var_tau":
        width_names = ("variance", "tau")
        width_values = [edge.variance, edge.tau]
        width_columns = [d.d_variance, d.d_tau]
    else:
        width_names = ("sigma", "temperature")
        width_values = [edge.sigma, edge.temperature]
        width_columns = [2.0 * edge.sigma * d.d_variance,
                         2.0 * KB_EV * KB_EV * edge.temperature * d.d_tau]

    if temperature_mode == "fixed_temperature":
        width_names, width_values, width_columns = width_names[:1], width_values[:1], width_columns[:1]
    dos_names = _DOS_TERMS[edge.dos]
    columns = [d.d_ef, d.d_amplitude, *(getattr(d, "d_" + n) for n in dos_names), *width_columns]
    names = ["ef", "amplitude", *dos_names, *width_names]
    roles = ["position", "amplitude", *(["dos"] * len(dos_names)), *(["width"] * len(width_names))]
    values = [edge.ef, edge.amplitude, *(getattr(edge, n) for n in dos_names), *width_values]

    mean = d.value.copy()
    if background is not None:
        if background.basis.shape[1] != energy.size:
            raise ValueError("background is on a different energy axis")
        mean += background.counts()
        for term, name, coef, estimated in zip(background.basis, background.names,
                                               background.coefficients, background.estimated):
            if estimated:
                columns.append(term)
                names.append(name)
                roles.append("background")
                values.append(float(coef))
    if not np.all(mean > 0.0):
        raise ValueError(
            "expected counts must be positive in every channel; the minimum is "
            f"{mean.min():.3e}. Add a background or narrow the window."
        )

    jac_unit = np.stack(columns, axis=1)
    data = poisson_fisher_matrix(jac_unit, mean, exposure)
    prior = np.zeros_like(data)
    if temperature_mode == "temperature_prior":
        i = names.index(width_names[1])
        if parameterization == "sigma_T":
            prior[i, i] = 1.0 / temperature_sd**2
        else:
            sd_tau = 2.0 * KB_EV * KB_EV * edge.temperature * temperature_sd
            if not sd_tau > 0.0:
                raise ValueError("a temperature prior at T = 0 has no width in tau")
            prior[i, i] = 1.0 / sd_tau**2
    return EdgeFisherResult(
        fisher=data + prior,
        param_names=tuple(names),
        param_roles=tuple(roles),
        param_values=np.array(values, dtype=np.float64),
        width_index=tuple(i for i, r in enumerate(roles) if r == "width"),
        jacobian=exposure * jac_unit,
        expected_counts=exposure * mean,
        parameterization=parameterization,
        exposure=float(exposure),
        temperature_mode=temperature_mode,
        prior=prior,
        config={
            "temperature_sd": temperature_sd,
            "edge": (edge.ef, edge.amplitude, edge.sigma, edge.temperature, edge.dos_c1,
                     edge.dos_c2, edge.dos),
            "background_terms": () if background is None else background.names,
            "background_estimated": () if background is None else background.estimated,
            "convention": convention,
            "dos_form": dos_form,
            "n_energy": int(energy.size),
            "energy_range": (float(energy[0]), float(energy[-1])),
        },
    )


# ---------------------------------------------------------------------------
# What the data say about (v, tau), in the scale of the edge itself
# ---------------------------------------------------------------------------

_C_TAU = _PI2 / 3.0  # kappa_2 = v + (pi^2/3) tau


@dataclass(frozen=True)
class EdgeWidthInformation:
    """The (v, tau) block with everything else profiled out, made unit-free.

    Coordinates are the two contributions to the edge's second cumulant
    as fractions of it: ``x_v = v / kappa_2`` and ``x_tau = (pi^2/3) tau /
    kappa_2`` with ``kappa_2 = v + (pi^2/3) tau``, so ``x_v + x_tau = 1`` at
    the stated parameters. Both are dimensionless; nothing reported here
    depends on the energy unit.

    The prediction from the cumulants is that the data fix ``x_v + x_tau``
    (the total width) well and ``x_v - x_tau`` (how the width divides
    between instrument and temperature) poorly. ``stiff_direction`` and
    ``soft_direction`` are the eigenvectors of the effective information
    in these coordinates, and ``alignment`` measures how close the stiff
    one is to (1, 1)/sqrt(2).

    Attributes:
        kappa2: v + (pi^2/3) tau (eV**2)
        shares: (x_v, x_tau) at the stated parameters
        effective: 2x2 effective information, everything else estimated
        conditional: 2x2 sub-block, everything else known exactly;
            ``conditional - effective`` is positive semi-definite
        eigenvalues: (soft, stiff) eigenvalues of ``effective``
        stiff_direction, soft_direction: unit eigenvectors (x_v, x_tau),
            signs fixed so the first non-zero component is positive
        alignment: |cos| between the stiff direction and (1, 1)/sqrt(2)
        sd_kappa2: bound on sd(kappa_2)/kappa_2 from the inverse of
            ``effective``; inf if it is singular
        sd_share: bound on sd(x_v), the instrument's share of the width;
            inf if singular
        nuisance_null_dim: As in ``EffectiveInformation``
        temperature_mode: Echo of the Fisher matrix's mode -- which is
            'free' or 'temperature_prior', never 'fixed_temperature',
            since dividing a width needs both of its parts estimated.
            When an ``EdgeIdentifiabilityReport`` was asked for a fixed
            temperature this field says 'free' while the report's own
            ``fisher.temperature_mode`` says 'fixed_temperature', and
            the report's ``separation`` is 'assumed'.
    """

    kappa2: float
    shares: tuple[float, float]
    effective: np.ndarray
    conditional: np.ndarray
    eigenvalues: tuple[float, float]
    stiff_direction: tuple[float, float]
    soft_direction: tuple[float, float]
    alignment: float
    sd_kappa2: float
    sd_share: float
    nuisance_null_dim: int
    temperature_mode: str


def _oriented(vector: np.ndarray) -> tuple[float, float]:
    first = vector[np.flatnonzero(np.abs(vector) > 1e-300)[0]]
    vector = vector if first > 0 else -vector
    return float(vector[0]), float(vector[1])


def edge_width_information(fisher: EdgeFisherResult) -> EdgeWidthInformation:
    """Effective information on (v, tau) in the edge's own scale.

    Needs a 'var_tau' matrix with both width coordinates, i.e. the
    temperature 'free' or with a prior: with it fixed there is nothing to
    divide the width between.

    The bounds are inverse-Fisher numbers at the stated parameters. Near
    ``tau = 0`` the separating direction's information vanishes (see
    ``edge_fisher``) and they grow without limit; near either boundary the
    normal approximation behind them fails (Self & Liang 1987). They are
    not confidence intervals.
    """
    if fisher.parameterization != "var_tau" or len(fisher.width_index) != 2:
        raise ValueError("needs a 'var_tau' matrix with v and tau both estimated "
                         "(temperature_mode 'free' or 'temperature_prior')")
    iv, it = fisher.width_index
    v, tau = float(fisher.param_values[iv]), float(fisher.param_values[it])
    kappa2 = v + _C_TAU * tau
    scale = np.diag([kappa2, kappa2 / _C_TAU])  # d(v, tau)/d(x_v, x_tau)

    info: EffectiveInformation = effective_information(fisher.fisher, [iv, it])
    effective = scale @ info.effective @ scale
    conditional = scale @ info.conditional @ scale
    eigenvalues, eigenvectors = np.linalg.eigh(effective)
    stiff, soft = eigenvectors[:, 1], eigenvectors[:, 0]
    alignment = float(abs(stiff @ np.array([1.0, 1.0])) / math.sqrt(2.0))

    shares = (v / kappa2, _C_TAU * tau / kappa2)
    try:
        cov = np.linalg.inv(effective)
        ones = np.ones(2)
        gradient = np.array([1.0 - shares[0], -shares[0]])  # of x_v / (x_v + x_tau) at x_v + x_tau = 1
        sd_kappa2 = math.sqrt(max(float(ones @ cov @ ones), 0.0))
        sd_share = math.sqrt(max(float(gradient @ cov @ gradient), 0.0))
        if not (math.isfinite(sd_kappa2) and math.isfinite(sd_share)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        sd_kappa2 = sd_share = math.inf

    return EdgeWidthInformation(
        kappa2=kappa2, shares=shares, effective=effective, conditional=conditional,
        eigenvalues=(float(eigenvalues[0]), float(eigenvalues[1])),
        stiff_direction=_oriented(stiff), soft_direction=_oriented(soft), alignment=alignment,
        sd_kappa2=sd_kappa2, sd_share=sd_share, nuisance_null_dim=info.nuisance_null_dim,
        temperature_mode=fisher.temperature_mode,
    )


# ---------------------------------------------------------------------------
# Judgement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeThresholds:
    """Where the labels change. Conventions, not physics.

    Attributes:
        weak_relative_sd: A parameter is 'weakly_identified' when its
            bound exceeds this fraction of its reference scale. 0.1, as
            in ``voigtfit.identifiability``: a tenth of the edge's own
            width is a working rule of thumb for when a number has
            stopped being usable, with no deeper justification.
        boundary_sd: v (or tau) is 'near_boundary' when it lies within
            this many bounds of zero. The probability behind 3 comes from
            the normal approximation for an unconstrained estimator,
            which is exactly what fails at a boundary (Self & Liang
            1987): read it as a margin of caution.
        separation_sd: v and tau count as separable when the bound on
            ``x_v = v / kappa_2``, the instrument's share of the edge's
            second cumulant, is at most this. 0.1 means "the split is
            known to a tenth" -- a working threshold, not physics.
    """

    weak_relative_sd: float = 0.1
    boundary_sd: float = 3.0
    separation_sd: float = 0.1

    def __post_init__(self) -> None:
        if not (self.weak_relative_sd > 0.0 and self.boundary_sd > 0.0
                and self.separation_sd > 0.0):
            raise ValueError("thresholds must be positive")


@dataclass(frozen=True)
class EdgeParameterAssessment:
    """One parameter of the model.

    Attributes:
        name, role: As in ``EdgeFisherResult``
        value: Its value
        sd: Bound on its standard deviation; ``inf`` when rank deficient
        reference_scale: What ``sd`` is judged against -- sqrt(kappa_2)
            for E_F, |A| for the amplitude, kappa_2 for v and tau,
            1/sqrt(kappa_2) and 1/kappa_2 for the DOS coefficients, None
            for a background term
        relative_sd: ``sd / reference_scale``, or None
        status: 'rank_deficient', 'weakly_identified', 'identified',
            'undersampled' (widths only) or 'not_assessed'
    """

    name: str
    role: str
    value: float
    sd: float
    reference_scale: float | None
    relative_sd: float | None
    status: str


@dataclass(frozen=True)
class InstrumentalResolution:
    """The Gaussian width, with what it is worth and what it rests on.

    ``variance`` and one of the two standard deviations are what
    ``voigtfit.identifiability.assess_identifiability`` takes as
    ``variance_floor`` and ``variance_floor_sd``, when the Voigt peaks
    come from the same instrument and settings.

    Attributes:
        variance, sigma, fwhm: The Gaussian width, three ways (eV**2, eV, eV)
        sd_bound: Bound on sd(v) from the Poisson Fisher matrix in this
            temperature mode -- what an efficient estimator could reach.
            **Do not pair it with an estimate from a fit whose weights
            are not the Poisson ones**: an unweighted least-squares fit
            of this model scatters 1.25 to 2.55 times wider over the
            conditions measured, so its own covariance -- ``sd_estimator``
            below -- is the honest partner.
        sd_estimator: sd(v) of the weighted least-squares estimator whose
            weights were given (sandwich covariance), or None
        estimator: What ``sd_estimator`` belongs to
        sd_sigma_bound: ``sd_bound / (2 sigma)``, the same bound on sigma
        temperature_sensitivity: **d sigma / dT (eV per K)**: how far the
            fitted resolution moves when the temperature held fixed is
            wrong by one kelvin, to first order and with everything else
            estimated. The most practical number here when the
            temperature is fixed, because a thermocouple reading is not
            the electron temperature.
        temperature_mode, temperature: The mode and the temperature used
    """

    variance: float
    sigma: float
    fwhm: float
    sd_bound: float
    sd_estimator: float | None
    estimator: str | None
    sd_sigma_bound: float
    temperature_sensitivity: float
    temperature_mode: str
    temperature: float


@dataclass(frozen=True)
class EdgeIdentifiabilityReport:
    """What a counting spectrum of this edge determines.

    Attributes:
        parameters: One assessment per parameter
        width: The (v, tau) block, unit-free. **With
            ``temperature_mode='fixed_temperature'`` this comes from
            the matrix that estimates both widths, not from the one
            that was asked for** -- a matrix with tau held has no
            (v, tau) block to report. So it answers 'could this
            spectrum have divided the width?', not 'how well did this
            fit divide it?', which is why ``separation`` is then
            'assumed'. ``width.temperature_mode`` reads 'free' there
            while ``fisher.temperature_mode`` reads
            'fixed_temperature'; the two differing is the signal, not
            a mistake. Every other field of this report comes from the
            mode that was asked for.
        resolution: The Gaussian width and what it rests on
        separation: 'separable' when the bound on the instrument's share
            of the width is within ``thresholds.separation_sd``;
            'not_separable' when it is not, and ``sd_tau`` is then None;
            'assumed' when the temperature was held fixed -- the split is
            then an assumption and not a measurement, whatever ``width``
            reports (see ``width`` above); 'undersampled' when the edge
            is thinner than two channels
        sd_variance: Bound on sd(v) (eV**2)
        sd_tau: Bound on sd(tau) (eV**2), or None when not separable
        near_boundary_v: v within ``boundary_sd`` bounds of 0
        near_boundary_tau: The same for tau, or None when not separable
            or the temperature is fixed
        undersampled: sqrt(kappa_2) below twice the channel spacing,
            where a model sampled at channel centres says little about a
            width; every width label is then 'undersampled'
        null_space_dim, condition_number: Of the unit-diagonal Fisher matrix
        thresholds: The thresholds used
        fisher: The Fisher matrix it all came from
    """

    parameters: tuple[EdgeParameterAssessment, ...]
    width: EdgeWidthInformation
    resolution: InstrumentalResolution
    separation: str
    sd_variance: float
    sd_tau: float | None
    near_boundary_v: bool
    near_boundary_tau: bool | None
    undersampled: bool
    null_space_dim: int
    condition_number: float
    thresholds: EdgeThresholds
    fisher: EdgeFisherResult = field(repr=False)


def edge_wls_covariance(
    energy: np.ndarray,
    edge: FermiEdge,
    background: Background | None = None,
    *,
    weights: np.ndarray | Literal["poisson", "unit"] = "poisson",
    exposure: float = 1.0,
    **kwargs: Any,
) -> tuple[np.ndarray, tuple[str, ...], str]:
    """Sandwich covariance of a weighted least-squares fit of this model.

    ``fit_fermi_edge`` minimises a weighted sum of squares, not the
    Poisson likelihood, so what it returns is the weighted least-squares
    estimator, whose covariance is
    ``(J^T W J)^-1 J^T W diag(mu) W J (J^T W J)^-1``. With
    ``weights = 1/mu`` that is the inverse Fisher matrix; with unit
    weights -- the fitter's default -- it is not, and the difference is
    not small.

    Args:
        energy, edge, background, exposure: As in ``edge_fisher``
        weights: 'poisson' (1/mu, what the fitter's docstring
            recommends), 'unit' (its default), or an array
        kwargs: Passed to ``edge_fisher`` (convention, dos_form, route,
            temperature_mode, temperature_sd)

    Returns:
        (covariance, parameter names, a label for the estimator)
    """
    fisher = edge_fisher(energy, edge, background, exposure=exposure, **kwargs)
    jac, mean = fisher.jacobian, fisher.expected_counts
    if isinstance(weights, str):
        if weights == "poisson":
            w, label = 1.0 / mean, "weighted least squares (poisson weights)"
        elif weights == "unit":
            w, label = np.ones_like(mean), "least squares (unit weights)"
        else:
            raise ValueError(f"weights must be 'poisson', 'unit' or an array, got {weights!r}")
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != mean.shape or np.any(w < 0.0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite, non-negative and one per channel")
        label = "weighted least squares (given weights)"
    a = jac.T @ (jac * w[:, np.newaxis])
    b = jac.T @ (jac * (w * w * mean)[:, np.newaxis])
    a_inv = np.linalg.inv(a)
    return a_inv @ b @ a_inv, fisher.param_names, label


def assess_edge_identifiability(
    energy: np.ndarray,
    edge: FermiEdge,
    background: Background | None = None,
    *,
    exposure: float = 1.0,
    temperature_mode: TemperatureMode = "free",
    temperature_sd: float | None = None,
    estimator_weights: np.ndarray | Literal["poisson", "unit"] | None = None,
    thresholds: EdgeThresholds | None = None,
    convention: str = "BE",
    dos_form: Literal["occupied", "both_sides"] = "occupied",
    route: Literal["auto", "quadrature", "series"] = "auto",
) -> EdgeIdentifiabilityReport:
    """Sort the parameters of an edge-plus-background model by identifiability.

    Always in 'var_tau' coordinates, and always from the full inverse
    Fisher matrix: every other parameter is estimated from the same
    spectrum.

    What the labels do not mean:

    - 'identified' on the width scale is not a measurement of the
      temperature or of the resolution separately. Read ``separation``
      and the bound on the instrument's share of the width for that.
    - 'not_separable' says the data in *this* window, at *this* exposure,
      cannot divide the width between instrument and temperature. The
      information that separates them is real, but it vanishes as
      tau**2 as the temperature falls (see ``edge_fisher``), so at low
      temperature no exposure within reach recovers it.
    - Near a boundary, or where the split is not separable, the inverse
      Fisher matrix is not the variance of a constrained estimator and a
      Monte Carlo variance need not agree with it (Self & Liang 1987).
    - Everything is computed from a model at stated parameters. It is
      not a property of any data set, and the conditions in the module
      docstring of ``voigtfit.crlb`` apply.

    Args:
        energy, edge, background, exposure, convention, dos_form, route:
            As in ``edge_fisher``
        temperature_mode, temperature_sd: As in ``edge_fisher``
        estimator_weights: When given, ``resolution.sd_estimator`` is the
            sandwich sd of a weighted least-squares fit with these
            weights ('poisson', 'unit', or an array)
        thresholds: Defaults to ``EdgeThresholds()``

    Returns:
        EdgeIdentifiabilityReport.

        One field does not come from ``temperature_mode``: with
        ``'fixed_temperature'`` there is no (v, tau) block to report, so
        ``report.width`` is computed from the matrix that estimates both
        widths and ``report.separation`` is ``'assumed'``. It answers
        what this spectrum could have said about the split, not what a
        fit holding T fixed measured -- that fit measured nothing about
        it. ``report.width.temperature_mode`` reads ``'free'`` there,
        beside a ``report.fisher.temperature_mode`` of
        ``'fixed_temperature'``.
    """
    thresholds = thresholds or EdgeThresholds()
    fisher = edge_fisher(energy, edge, background, parameterization="var_tau",
                         temperature_mode=temperature_mode, temperature_sd=temperature_sd,
                         exposure=exposure, convention=convention, dos_form=dos_form, route=route)
    covariance, unbounded, null_dim, condition = covariance_bound(fisher.fisher,
                                                                  _NULL_PROJECTION_TOL)
    kappa2 = edge.variance + _C_TAU * edge.tau
    root = math.sqrt(kappa2)
    scales = {"ef": root, "amplitude": abs(edge.amplitude), "dos_c1": 1.0 / root,
              "dos_c2": 1.0 / kappa2, "variance": kappa2, "tau": kappa2}
    step = float(np.median(np.diff(np.sort(np.asarray(energy, dtype=np.float64)))))
    undersampled = root < 2.0 * step

    parameters = []
    for i, (name, role) in enumerate(zip(fisher.param_names, fisher.param_roles)):
        sd = math.inf if unbounded[i] else math.sqrt(max(covariance[i, i], 0.0))
        scale = scales.get(name)
        relative = None if scale is None else sd / scale
        if undersampled and role == "width":
            # the sampling, not the matrix, is why there is nothing to read
            status = "undersampled"
        elif unbounded[i]:
            status = "rank_deficient"
        elif relative is None:
            status = "not_assessed"
        else:
            status = ("weakly_identified" if relative > thresholds.weak_relative_sd
                      else "identified")
        parameters.append(EdgeParameterAssessment(
            name=name, role=role, value=float(fisher.param_values[i]), sd=float(sd),
            reference_scale=scale, relative_sd=relative, status=status))

    iv = fisher.width_index[0]
    sd_variance = math.inf if unbounded[iv] else math.sqrt(max(covariance[iv, iv], 0.0))
    if temperature_mode == "fixed_temperature":
        # the split is an assumption here; the trade between v and tau still
        # comes from the matrix that estimates both
        free = edge_fisher(energy, edge, background, exposure=exposure, convention=convention,
                           dos_form=dos_form, route=route)
        width = edge_width_information(free)
        separation, sd_tau, near_boundary_tau = "assumed", None, None
        block = effective_information(free.fisher, free.width_index).effective
    else:
        width = edge_width_information(fisher)
        separation = ("not_separable" if width.sd_share > thresholds.separation_sd
                      else "separable")
        it = fisher.width_index[1]
        sd_tau_value = math.inf if unbounded[it] else math.sqrt(max(covariance[it, it], 0.0))
        sd_tau = None if separation == "not_separable" else sd_tau_value
        near_boundary_tau = (None if sd_tau is None
                             else bool(edge.tau < thresholds.boundary_sd * sd_tau))
        block = effective_information(fisher.fisher, fisher.width_index).effective
    if undersampled:
        separation = "undersampled"

    # How far a fit's resolution moves when the temperature it holds fixed is
    # wrong by 1 K: -(I_vv)^-1 I_v_tau dtau/dT on the effective block, i.e.
    # the first-order trade along the edge's width with everything else estimated.
    d_tau_d_t = 2.0 * KB_EV * KB_EV * edge.temperature
    d_variance_d_t = (-block[0, 1] / block[0, 0] * d_tau_d_t) if block[0, 0] > 0.0 else math.nan
    sigma = edge.sigma
    sd_estimator = estimator = None
    if estimator_weights is not None:
        cov_wls, names_wls, estimator = edge_wls_covariance(
            energy, edge, background, weights=estimator_weights, exposure=exposure,
            convention=convention, dos_form=dos_form, route=route,
            temperature_mode=temperature_mode, temperature_sd=temperature_sd)
        i = names_wls.index("variance")
        sd_estimator = math.sqrt(max(cov_wls[i, i], 0.0))
    resolution = InstrumentalResolution(
        variance=edge.variance, sigma=sigma, fwhm=2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma,
        sd_bound=sd_variance, sd_estimator=sd_estimator, estimator=estimator,
        sd_sigma_bound=sd_variance / (2.0 * sigma) if sigma > 0.0 else math.inf,
        temperature_sensitivity=d_variance_d_t / (2.0 * sigma) if sigma > 0.0 else math.nan,
        temperature_mode=temperature_mode, temperature=edge.temperature)

    return EdgeIdentifiabilityReport(
        parameters=tuple(parameters), width=width, resolution=resolution, separation=separation,
        sd_variance=sd_variance, sd_tau=sd_tau,
        near_boundary_v=bool(edge.variance < thresholds.boundary_sd * sd_variance),
        near_boundary_tau=near_boundary_tau, undersampled=undersampled,
        null_space_dim=null_dim, condition_number=condition, thresholds=thresholds, fisher=fisher,
    )


# ---------------------------------------------------------------------------
# Scan over measurement conditions
# ---------------------------------------------------------------------------

_SEPARATION_CODE = {"separable": 0, "not_separable": 1, "assumed": 2, "undersampled": 3}
_STATUS_CODE = {"identified": 0, "weakly_identified": 1, "rank_deficient": 2, "undersampled": 3,
                "not_assessed": 4}


@dataclass(frozen=True)
class EdgeScanGrid:
    """Conditions for ``scan_edge_identifiability``.

    Lengths are in units of the edge's own width sqrt(kappa_2), which is
    held fixed, so the grid carries no energy unit: ``width`` only sets
    what that unit is, and no result depends on it.

    Attributes:
        ratios: kT/sigma. Covers the three regimes: instrument-dominated
            (<< 1), comparable (~1) and temperature-dominated (>> 1)
        half_widths: Window half-widths in units of sqrt(kappa_2),
            measured from E_F
        levels: Exposure multipliers, or total counts in the window when
            ``normalization='total_counts'``
        slopes: DOS coefficient c1 in units of 1/sqrt(kappa_2), i.e. the
            fractional rise of the DOS over one edge width
        backgrounds: Any of 'none', 'known', 'estimated'
        temperature_modes: Any of 'free', 'fixed_temperature',
            'temperature_prior'
        dos_forms: Any of 'occupied', 'both_sides' -- ``fermi_edge``'s
            density of states, flat below E_F, and the same polynomial
            continued smoothly, to see what the change of slope at E_F
            is worth
        normalization: 'exposure' keeps the energy step and the exposure
            per point fixed, so a wider window has more points and more
            counts -- how a measurement actually trades. 'total_counts'
            rescales every window to the same total.
        background_kind: 'constant' or 'linear' (zero true slope)
        background_fraction: Background level as a fraction of the
            occupied-side step height
        prior_fraction: sd(T)/T for 'temperature_prior'
        step: Energy step in units of sqrt(kappa_2)
        width: sqrt(kappa_2) in eV. Sets the unit of the energy axis and
            nothing else
        amplitude: Step height in counts per channel at unit exposure
        thresholds: As in ``assess_edge_identifiability``
    """

    ratios: tuple[float, ...] = (0.1, 0.5, 1.0, 3.0)
    half_widths: tuple[float, ...] = (3.0, 6.0, 12.0)
    levels: tuple[float, ...] = (1.0,)
    slopes: tuple[float, ...] = (0.0, 0.3)
    backgrounds: tuple[str, ...] = ("none", "known", "estimated")
    temperature_modes: tuple[str, ...] = ("free", "fixed_temperature", "temperature_prior")
    dos_forms: tuple[str, ...] = ("occupied", "both_sides")
    normalization: Literal["exposure", "total_counts"] = "exposure"
    background_kind: Literal["constant", "linear"] = "constant"
    background_fraction: float = 0.05
    prior_fraction: float = 0.1
    step: float = 0.1
    width: float = 0.1
    amplitude: float = 1000.0
    thresholds: EdgeThresholds = field(default_factory=EdgeThresholds)

    def __post_init__(self) -> None:
        if not all(r > 0.0 for r in self.ratios) or not self.ratios:
            raise ValueError("ratios must be positive")
        if not all(h > 0.0 for h in self.half_widths) or not all(v > 0.0 for v in self.levels):
            raise ValueError("half_widths and levels must be positive")
        if not (self.step > 0.0 and self.width > 0.0 and self.amplitude > 0.0):
            raise ValueError("step, width and amplitude must be positive")
        for name, allowed in (("backgrounds", ("none", "known", "estimated")),
                              ("temperature_modes", TEMPERATURE_MODES),
                              ("dos_forms", ("occupied", "both_sides"))):
            bad = set(getattr(self, name)) - set(allowed)
            if bad or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty subset of {allowed}, got {bad}")
        if self.normalization not in ("exposure", "total_counts"):
            raise ValueError("normalization must be 'exposure' or 'total_counts'")
        if self.background_kind not in ("constant", "linear"):
            raise ValueError("background_kind must be 'constant' or 'linear'")
        if self.background_fraction < 0.0 or self.prior_fraction <= 0.0:
            raise ValueError("background_fraction must be >= 0 and prior_fraction > 0")


def _scan_edge(grid: EdgeScanGrid, ratio: float) -> tuple[float, float]:
    """(sigma, temperature) at this kT/sigma with kappa_2 = width**2 fixed."""
    kappa2 = grid.width * grid.width
    sigma = math.sqrt(kappa2 / (1.0 + _C_TAU * ratio * ratio))
    return sigma, ratio * sigma / KB_EV


def scan_edge_identifiability(grid: EdgeScanGrid) -> dict[str, np.ndarray]:
    """Assess an edge over a grid of measurement conditions.

    Five of the arrays are the same for ``'free'`` and
    ``'fixed_temperature'``: ``sd_kappa2``, ``sd_share``, ``alignment``,
    ``soft_over_stiff`` and ``temperature_sensitivity_relative`` all come
    from ``report.width``, which under a fixed temperature is computed
    from the matrix that estimates both widths (see
    ``assess_edge_identifiability``). Plot them against the temperature
    mode and two of the curves will lie on top of each other, by
    construction and not by accident. Only ``sd_sigma``, ``separation``,
    ``condition_number`` and ``near_boundary_v`` differ between those two
    modes.

    Returns a dict of arrays of shape ``(len(ratios), len(half_widths),
    len(levels), len(slopes), len(backgrounds), len(temperature_modes),
    len(dos_forms))``, plus the axes themselves, ready for ``np.savez``:

    - 'sd_kappa2': bound on sd(kappa_2)/kappa_2
    - 'sd_share': bound on the instrument's share of the width
    - 'sd_sigma': bound on sd(sigma)/sigma
    - 'alignment': |cos| between the stiff direction and v + (pi^2/3) tau
    - 'soft_over_stiff': the eigenvalue ratio of the effective block
    - 'temperature_sensitivity_relative': (d sigma/dT) T / sigma, the fractional
      change in the fitted resolution per fractional error in a fixed
      temperature
    - 'separation', 'variance_status': codes, see 'separation_codes' and
      'status_codes' in the result
    - 'near_boundary_v': 1.0 where v is within boundary_sd bounds of 0
    - 'condition_number', 'null_space_dim', 'n_energy', 'total_counts'

    Nothing here depends on ``grid.width``: every reported number is
    dimensionless.

    Cells the model refuses stay NaN. Two ways that happens: a rising DOS
    continued below E_F ('both_sides') turns negative inside the thermal
    tail, which it does whenever the slope times the tail's reach exceeds
    1; and, with no background, the mean underflows in a window many
    sigma below the edge.
    """
    shape = (len(grid.ratios), len(grid.half_widths), len(grid.levels), len(grid.slopes),
             len(grid.backgrounds), len(grid.temperature_modes), len(grid.dos_forms))
    names = ("sd_kappa2", "sd_share", "sd_sigma", "alignment", "soft_over_stiff",
             "temperature_sensitivity_relative", "separation", "variance_status",
             "near_boundary_v",
             "condition_number", "null_space_dim", "n_energy", "total_counts")
    out = {name: np.full(shape, np.nan) for name in names}

    for i, ratio in enumerate(grid.ratios):
        sigma, temperature = _scan_edge(grid, ratio)
        for j, half in enumerate(grid.half_widths):
            energy = np.arange(-half, half + 1e-12, grid.step) * grid.width
            for m, slope in enumerate(grid.slopes):
                dos = "flat" if slope == 0.0 else "linear"
                edge = FermiEdge(ef=0.0, amplitude=grid.amplitude, sigma=sigma,
                                 temperature=temperature, dos_c1=slope / grid.width, dos=dos)
                for k, level in enumerate(grid.levels):
                    for n, background in enumerate(grid.backgrounds):
                        level_counts = grid.background_fraction * grid.amplitude
                        if background == "none":
                            bg = None
                        elif grid.background_kind == "constant":
                            bg = constant_background(energy, level_counts,
                                                     estimated=background == "estimated")
                        else:
                            bg = linear_background(energy, level_counts, 0.0,
                                                   estimated=background == "estimated")
                        exposure = level
                        if grid.normalization == "total_counts":
                            unit = edge_fisher(energy, edge, bg, dos_form=grid.dos_forms[0])
                            exposure = level / float(unit.expected_counts.sum())
                        for p, mode in enumerate(grid.temperature_modes):
                            sd = (grid.prior_fraction * temperature
                                  if mode == "temperature_prior" else None)
                            for q, form in enumerate(grid.dos_forms):
                                try:
                                    report = assess_edge_identifiability(
                                        energy, edge, bg, exposure=exposure,
                                        temperature_mode=mode, temperature_sd=sd,
                                        thresholds=grid.thresholds, dos_form=form)
                                except ValueError:
                                    continue  # refused cell, left as NaN (see the docstring)
                                index = (i, j, k, m, n, p, q)
                                width = report.width
                                out["sd_kappa2"][index] = width.sd_kappa2
                                out["sd_share"][index] = width.sd_share
                                out["sd_sigma"][index] = report.resolution.sd_sigma_bound / sigma
                                out["alignment"][index] = width.alignment
                                out["soft_over_stiff"][index] = (width.eigenvalues[0]
                                                                 / width.eigenvalues[1])
                                out["temperature_sensitivity_relative"][index] = (
                                    report.resolution.temperature_sensitivity * temperature / sigma)
                                out["separation"][index] = _SEPARATION_CODE[report.separation]
                                out["variance_status"][index] = _STATUS_CODE[
                                    next(a.status for a in report.parameters
                                         if a.name == "variance")]
                                out["near_boundary_v"][index] = float(report.near_boundary_v)
                                out["condition_number"][index] = report.condition_number
                                out["null_space_dim"][index] = report.null_space_dim
                                out["n_energy"][index] = energy.size
                                out["total_counts"][index] = float(
                                    report.fisher.expected_counts.sum())
    out.update({"ratios": np.asarray(grid.ratios, dtype=np.float64),
                "half_widths": np.asarray(grid.half_widths, dtype=np.float64),
                "levels": np.asarray(grid.levels, dtype=np.float64),
                "slopes": np.asarray(grid.slopes, dtype=np.float64),
                "backgrounds": np.asarray(grid.backgrounds),
                "temperature_modes": np.asarray(grid.temperature_modes),
                "dos_forms": np.asarray(grid.dos_forms),
                "separation_codes": np.asarray(list(_SEPARATION_CODE)),
                "status_codes": np.asarray(list(_STATUS_CODE))})
    return out


# ---------------------------------------------------------------------------
# The model as a batch, for resampling
# ---------------------------------------------------------------------------


def edge_mle_model(
    energy: np.ndarray,
    edge: FermiEdge,
    background: Background | None = None,
    *,
    exposure: float = 1.0,
    convention: str = "BE",
    dos_form: Literal["occupied", "both_sides"] = "occupied",
    route: Literal["auto", "quadrature", "series"] = "auto",
) -> tuple[Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], tuple[str, ...],
           np.ndarray, np.ndarray]:
    """The edge as ``toyomacro._bootstrap`` wants it: a batched model.

    Returns ``(model, names, lower, start)``:

    - ``model(theta)`` takes (n, p) parameter vectors in 'var_tau'
      coordinates and returns the expected counts (n, n_channels) and
      their Jacobian (n, n_channels, p), at the stated exposure. Rows
      with an infeasible parameter (a negative width, an amplitude at or
      below zero, a DOS that is not positive) come back as NaN, which the
      fitter reads as an infinite deviance.
    - ``names`` are the parameters, in the order ``edge_fisher`` uses.
    - ``lower`` holds 0 for the amplitude, v and tau, ``-inf`` elsewhere.
    - ``start`` is ``edge`` itself, the usual starting point for a Monte
      Carlo check; for a bootstrap, pass the fitted values instead.

    One parameter set at a time is evaluated inside; the edge model is
    vectorised over channels, not over replicas.
    """
    energy = np.asarray(energy, dtype=np.float64)
    dos_names = _DOS_TERMS[edge.dos]
    names = ("ef", "amplitude", *dos_names, "variance", "tau")
    basis = None
    if background is not None:
        estimated = [i for i, flag in enumerate(background.estimated) if flag]
        names = names + tuple(background.names[i] for i in estimated)
        basis = background.basis[estimated] if estimated else None
        known = background.counts() - (background.coefficients[estimated] @ basis
                                       if basis is not None else 0.0)
    else:
        known = np.zeros_like(energy)
    n_edge = 2 + len(dos_names)

    def model(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        n = theta.shape[0]
        mean = np.empty((n, energy.size))
        jac = np.empty((n, energy.size, theta.shape[1]))
        for i, row in enumerate(theta):
            dos_values = dict(zip(dos_names, row[2:n_edge]))
            try:
                d = edge_derivatives(energy, row[0], row[1], row[n_edge], row[n_edge + 1],
                                     dos_c1=dos_values.get("dos_c1", 0.0),
                                     dos_c2=dos_values.get("dos_c2", 0.0),
                                     convention=convention, dos_form=dos_form, route=route)
            except ValueError:
                mean[i] = np.nan
                jac[i] = np.nan
                continue
            columns = [d.d_ef, d.d_amplitude, *(getattr(d, "d_" + n_) for n_ in dos_names),
                       d.d_variance, d.d_tau]
            total = d.value + known
            if basis is not None:
                total = total + row[n_edge + 2:] @ basis
                columns += [b for b in basis]
            mean[i] = exposure * total
            jac[i] = exposure * np.stack(columns, axis=1)
        return mean, jac

    lower = np.full(len(names), -np.inf)
    lower[1] = 0.0                      # amplitude
    lower[n_edge] = 0.0                 # v
    lower[n_edge + 1] = 0.0             # tau
    start = np.array([edge.ef, edge.amplitude, *(getattr(edge, n_) for n_ in dos_names),
                      edge.variance, edge.tau,
                      *(background.coefficients[i] for i in
                        ([j for j, f in enumerate(background.estimated) if f]
                         if background is not None else []))], dtype=np.float64)
    return model, names, lower, start
