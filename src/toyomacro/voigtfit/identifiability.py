"""Identifiability diagnostics for Voigt widths (SciPy/NumPy reference).

Research/diagnostic layer answering "how much does a counting spectrum
say about the Gaussian and Lorentzian widths, once everything else that
has to be estimated from the same spectrum is accounted for?".

Three things live here that the older Fisher modules do not have:

- A Poisson Fisher matrix whose mean is *peaks plus background*, with the
  background absent, known, or estimated. ``fisher_information`` and
  ``crlb`` carry no background term at all, so a bound computed there
  conditions on a background-free spectrum.
- The effective information for a subset of parameters with the rest
  profiled out (a Schur complement), next to the plain sub-block that
  treats the rest as known. ``fisher_information.analyze_sigma_gamma_axes``
  reports the latter only.
- Voigt derivatives in the Gaussian *variance* ``v = sigma**2`` that stay
  accurate down to and including ``v = 0``.

On top of those, ``assess_identifiability`` sorts each parameter into
rank deficient, weakly identified or identified, and separately says
whether the Gaussian variance sits near its boundary. What those labels
do and do not mean is in its docstring; none of them is a statement
about a fit.

Why the variance. A Voigt profile is a Lorentzian convolved with a
Gaussian, so it obeys the heat equation in ``v``::

    dV/dv = (1/2) d2V/dx2

and depends on ``sigma`` only through ``v``. In the ``sigma`` coordinate
the first-order sensitivity ``dV/dsigma = 2 sigma dV/dv`` vanishes at
``sigma = 0`` and ``I_sigma_sigma ~ sigma**2``: a degeneracy of the
coordinate, removed by working in ``v``. What re-parameterisation does
*not* remove is the precision problem -- ``sd(sigma)/sigma =
sd(v)/(2 v)``, so a finite bound on ``v`` still means an unbounded
relative error on a vanishing Gaussian component -- nor the fact that
``v = 0`` is a boundary of the parameter space, where the usual
asymptotics of constrained estimators do not apply (Self & Liang 1987).

Numerical validity. The Faddeeva-based derivative formulas cancel
catastrophically at large ``|z| = |x + i gamma| / (sigma sqrt 2)``: in
``w''(z)`` a term of order ``4z`` cancels down to ``2/z**3``. On a window
of +-33 gamma that is why ``voigt_jacobian.voigt_with_jacobian`` and
``voigt_with_hessian`` break down for ``sigma/gamma`` below about 1e-3.
Here every energy point with ``|z| >= 7`` is evaluated from the
large-``|z|`` expansion instead,

    V = L + (v/2) L'' + (v**2/8) L'''' + ...

summed to 24 terms in closed form from derivatives of the Lorentzian.

Measured against an exponentially convergent quadrature of the
convolution integral, as the largest error in a band of ``|z|`` relative
to the largest magnitude in that band, worst over 31 values of
``sigma/gamma`` from 0.01 to 10 on a window of +-33.3 gamma:

    ==========  =========  ============  ===========  =========
    route       value      d_variance    d_gamma      d_center
    ==========  =========  ============  ===========  =========
    as shipped  6e-14      3e-11         5e-12        7e-13
    ==========  =========  ============  ===========  =========

The worst case sits just below the switch, on the Faddeeva side. The
expansion is at rounding level (3e-14) from ``|z| = 7`` up, while the
Faddeeva route keeps degrading with ``|z|`` (d_variance: 9e-11 on
7 <= |z| < 8, 4e-9 on 20 <= |z| < 50, 2e-7 beyond), so the switch is put
at the lowest ``|z|`` the expansion can take; below it the 24-term sum
fails fast -- 5e-11 on 6 <= |z| < 7, 2e-2 on 4 <= |z| < 6. Those are
numbers for this window and these shapes, not a bound.

The expansion is asymptotic, not convergent. Successive terms are in the
ratio ``(2n + 1) / (2 |z|**2)``, so they shrink only while ``n < |z|**2``
and grow without limit after it; the smallest term, and with it the best
accuracy available at that ``|z|``, is of order ``exp(-|z|**2)``. Two
constraints on the constants follow:

- The number of terms N must stay below ``|z_c|**2`` at the switch
  ``|z_c|`` -- 24 < 49 here -- so that every term summed is on the
  shrinking side at every point that uses the expansion.
- The switch cannot be lowered to buy back the Faddeeva error. Measured
  on d_variance at sigma/gamma = 1, the best any N can do is 4e-8 at
  ``|z| = 5`` and 3e-5 at ``|z| = 4``; N = 24 gives 2.5e-3 at ``|z| = 4``
  and N = 40 gives 3e2. More terms make it worse, not better.

At ``|z| = 7`` the same measurement is at rounding level for every N from
24 to 60, 6e-8 at N = 100 and 0.5 at N = 120. So exceeding ``|z_c|**2``
mildly has no numerical effect at this switch and the accuracy tests
would not notice it; ``N < |z_c|**2`` is therefore asserted on the
constants directly.

Conventions inherited from the existing engine:

- Peaks are unit-area Voigt profiles; ``sigma`` is the Gaussian standard
  deviation and ``gamma`` the Lorentzian half width at half maximum.
- ``amplitude * V(E)`` is the expected count *per channel* at unit
  exposure. As in ``crlb``, nothing can check that the caller's
  amplitudes are on that scale; normalised or arbitrary-unit amplitudes
  rescale every bound silently.

A Fisher matrix is computed from a model; it is not a measurement. What
``crlb``'s module docstring says about unbiasedness, correct
specification and non-singularity applies here unchanged.

Not exported from ``toyomacro.voigtfit.__init__``; not wired to any CLI.

References:
    J. J. Olivero and R. L. Longbothum, J. Quant. Spectrosc. Radiat.
        Transfer 17, 233-236 (1977), doi:10.1016/0022-4073(77)90161-3
    P. Thompson, D. E. Cox and J. B. Hastings, J. Appl. Cryst. 20, 79-83
        (1987), doi:10.1107/S0021889887087090
    S. G. Self and K.-Y. Liang, J. Am. Stat. Assoc. 82, 605-610 (1987),
        doi:10.1080/01621459.1987.10478472
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import numpy as np
from scipy import special as sps

from ..lineshape.pseudovoigt import PseudoVoigt
from .crlb import _NULL_PROJECTION_TOL

__all__ = [
    "PARAMETERIZATIONS",
    "VoigtDerivatives",
    "VoigtPeak",
    "Background",
    "PoissonFisherResult",
    "EffectiveInformation",
    "IdentifiabilityThresholds",
    "ParameterAssessment",
    "WidthAssessment",
    "IdentifiabilityReport",
    "ScanGrid",
    "voigt_derivatives",
    "voigt_fwhm",
    "constant_background",
    "linear_background",
    "shirley_background",
    "poisson_fisher",
    "effective_information",
    "assess_identifiability",
    "scan_identifiability",
]

Parameterization = Literal[
    "sigma_gamma", "var_gamma", "fwhm_shape", "pvoigt", "fixed_instrument"
]
PARAMETERIZATIONS: tuple[str, ...] = (
    "sigma_gamma", "var_gamma", "fwhm_shape", "pvoigt", "fixed_instrument"
)

_SQRT_PI = math.sqrt(math.pi)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_EIGHT_LN2 = 8.0 * math.log(2.0)  # f_G**2 = 8 ln2 * sigma**2

# Olivero & Longbothum (1977): f_V = a f_L + sqrt(b f_L**2 + f_G**2)
_OL_A = 0.5346
_OL_B = 0.2166

# Energy points with |z| at or above this use the large-|z| expansion.
# The lowest value at which the expansion is at rounding level: the
# Faddeeva route only gets worse with |z|, so nothing is gained by
# waiting. Measurements in the module docstring, pinned by the tests.
_SERIES_SWITCH_Z = 7.0
# The series is asymptotic: terms grow again once n exceeds |z|**2, so
# this must stay below _SERIES_SWITCH_Z**2 (the tests assert it).
_SERIES_TERMS = 24


def _series_coefficients(n_terms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coefficients (2n-1)!!, (2n+1)!! and (n+1)(2n+1)!! for n = 0..n_terms.

    Those of the value, of d/dx and d/dgamma, and of d/dvariance, as
    power series in t = variance / (x - i gamma)**2.
    """
    a = np.ones(n_terms + 1)
    b = np.ones(n_terms + 1)
    for n in range(1, n_terms + 1):
        a[n] = a[n - 1] * (2 * n - 1)
        b[n] = b[n - 1] * (2 * n + 1)
    c = b * (np.arange(n_terms + 1) + 1.0)
    return a, b, c


_SERIES_A, _SERIES_B, _SERIES_C = _series_coefficients(_SERIES_TERMS)


# ---------------------------------------------------------------------------
# Voigt derivatives in (center, variance, gamma)
# ---------------------------------------------------------------------------


class VoigtDerivatives(NamedTuple):
    """Unit-area Voigt profile and its first derivatives.

    Attributes:
        value: V(E), shape (n_energy,)
        d_center: dV/d(center)
        d_variance: dV/d(sigma**2), equal to (1/2) d2V/dE2
        d_gamma: dV/d(gamma)
    """

    value: np.ndarray
    d_center: np.ndarray
    d_variance: np.ndarray
    d_gamma: np.ndarray


def _horner(coefficients: np.ndarray, t: np.ndarray) -> np.ndarray:
    acc = np.full(t.shape, coefficients[-1], dtype=np.complex128)
    for c in coefficients[-2::-1]:
        acc = acc * t + c
    return acc


def _derivatives_series(
    x: np.ndarray, variance: float, gamma: float, n_terms: int | None = None
):
    """Large-|z| expansion; exact at variance = 0.

    ``n_terms`` overrides the shipped number of terms. It exists so the
    tests can show the series diverging; nothing else should pass it.
    """
    coeff_a, coeff_b, coeff_c = (
        (_SERIES_A, _SERIES_B, _SERIES_C) if n_terms is None else _series_coefficients(n_terms)
    )
    # L(x) = Im[1/u]/pi with u = x - i gamma, and every x-derivative of L
    # is a power of 1/u, so the heat-equation series in the variance
    # becomes a power series in t = variance / u**2.
    u = x - 1j * gamma
    t = variance / (u * u)
    value = (_horner(coeff_a, t) / u).imag / math.pi
    q = _horner(coeff_b, t) / (u * u)
    d_x = -q.imag / math.pi
    d_gamma = q.real / math.pi
    d_variance = (_horner(coeff_c, t) / u**3).imag / math.pi
    return value, -d_x, d_variance, d_gamma


def _derivatives_faddeeva(x: np.ndarray, variance: float, gamma: float):
    """Closed form from w(z), w'(z), w''(z); needs variance > 0."""
    sigma = math.sqrt(variance)
    z = (x + 1j * gamma) / (sigma * math.sqrt(2.0))
    w = sps.wofz(z)
    w1 = -2.0 * z * w + 2j / _SQRT_PI
    w2 = (4.0 * z * z - 2.0) * w - 4j * z / _SQRT_PI
    value = w.real / (sigma * _SQRT_2PI)
    d_x = w1.real / (2.0 * variance * _SQRT_PI)
    d_gamma = -w1.imag / (2.0 * variance * _SQRT_PI)
    d_variance = w2.real / (4.0 * variance * sigma * _SQRT_2PI)
    return value, -d_x, d_variance, d_gamma


def voigt_derivatives(
    energy: np.ndarray,
    center: float,
    variance: float,
    gamma: float,
    *,
    route: Literal["auto", "faddeeva", "series"] = "auto",
) -> VoigtDerivatives:
    """Voigt profile and derivatives w.r.t. center, Gaussian variance, gamma.

    Valid for ``variance >= 0`` and ``gamma >= 0``, not both zero,
    including ``variance = 0`` exactly, where the profile is the
    Lorentzian ``L`` and ``d_variance`` is ``L''/2``.

    Args:
        energy: Energy axis (eV), shape (n_energy,)
        center: Peak center (eV)
        variance: Gaussian variance sigma**2 (eV**2)
        gamma: Lorentzian half width at half maximum (eV)
        route: 'auto' picks per energy point -- the large-|z| expansion
            where ``|z| >= 7``, the Faddeeva closed form elsewhere. The
            other two force one route everywhere and exist so the switch
            can be verified; 'series' is wrong at small |z| and
            'faddeeva' loses accuracy at large |z| (module docstring).

    Returns:
        VoigtDerivatives
    """
    if variance < 0.0 or gamma < 0.0 or (variance == 0.0 and gamma == 0.0):
        raise ValueError(
            "need variance >= 0 and gamma >= 0, not both zero; "
            f"got variance={variance}, gamma={gamma}"
        )
    x = np.asarray(energy, dtype=np.float64) - float(center)

    if route == "series":
        return VoigtDerivatives(*_derivatives_series(x, variance, gamma))
    if route == "faddeeva":
        if variance == 0.0:
            raise ValueError("route='faddeeva' needs variance > 0")
        return VoigtDerivatives(*_derivatives_faddeeva(x, variance, gamma))
    if route != "auto":
        raise ValueError(f"route must be 'auto', 'faddeeva' or 'series', got '{route}'")

    # |z|**2 >= Z**2 written without dividing by the variance
    far = x * x + gamma * gamma >= 2.0 * variance * _SERIES_SWITCH_Z**2
    out = [np.empty_like(x) for _ in range(4)]
    if far.any():
        for dst, src in zip(out, _derivatives_series(x[far], variance, gamma)):
            dst[far] = src
    if not far.all():
        near = ~far
        for dst, src in zip(out, _derivatives_faddeeva(x[near], variance, gamma)):
            dst[near] = src
    return VoigtDerivatives(*out)


def voigt_fwhm(variance: float, gamma: float) -> float:
    """Voigt FWHM, Olivero & Longbothum (1977) approximation.

    ``f_V = 0.5346 f_L + sqrt(0.2166 f_L**2 + f_G**2)`` with
    ``f_G**2 = 8 ln2 * variance`` and ``f_L = 2 gamma``. The same
    expression as ``toyomacro.lineshape.Voigt.fwhm``. Against the exact
    half-maximum width of the profile its error is at most 2.4e-4
    (measured over f_L/f_G from 1e-4 to 1e4, pinned by the tests), exact
    for a Gaussian and 3e-6 for a Lorentzian. It is used here as the
    *definition* of the width coordinate in 'fwhm_shape', so that map is
    exactly invertible whatever the approximation error.
    """
    f_l = 2.0 * gamma
    return _OL_A * f_l + math.sqrt(_OL_B * f_l * f_l + _EIGHT_LN2 * variance)


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoigtPeak:
    """One Voigt component.

    Attributes:
        amplitude: Peak area; ``amplitude * V(E)`` is the expected count
            per channel at unit exposure. Must be positive.
        center: Peak center (eV)
        sigma: Gaussian standard deviation (eV), ``>= 0``. Zero is
            allowed and means a pure Lorentzian.
        gamma: Lorentzian half width at half maximum (eV), ``>= 0``
    """

    amplitude: float
    center: float
    sigma: float
    gamma: float

    def __post_init__(self) -> None:
        if not self.amplitude > 0.0:
            raise ValueError(f"amplitude must be positive, got {self.amplitude}")
        if self.sigma < 0.0 or self.gamma < 0.0 or (self.sigma == 0.0 and self.gamma == 0.0):
            raise ValueError(
                "need sigma >= 0 and gamma >= 0, not both zero; "
                f"got sigma={self.sigma}, gamma={self.gamma}"
            )

    @property
    def variance(self) -> float:
        return self.sigma * self.sigma


@dataclass(frozen=True)
class Background:
    """Background that is linear in its coefficients.

    ``counts = coefficients @ basis`` per channel at unit exposure. Terms
    flagged in ``estimated`` become parameters of the Fisher matrix; the
    others enter the Poisson mean only, i.e. they are treated as known.
    Backgrounds add: ``constant_background(...) + shirley_background(...)``.

    Attributes:
        basis: Shape functions, shape (n_terms, n_energy)
        coefficients: Shape (n_terms,)
        names: Parameter name per term
        estimated: Boolean per term
    """

    basis: np.ndarray
    coefficients: np.ndarray
    names: tuple[str, ...]
    estimated: tuple[bool, ...]

    def __post_init__(self) -> None:
        n = len(self.names)
        if self.basis.shape[0] != n or self.coefficients.shape != (n,) or len(self.estimated) != n:
            raise ValueError("basis, coefficients, names and estimated disagree on n_terms")

    def counts(self) -> np.ndarray:
        return self.coefficients @ self.basis

    def __add__(self, other: Background) -> Background:
        if self.basis.shape[1] != other.basis.shape[1]:
            raise ValueError("backgrounds are on different energy axes")
        return Background(
            basis=np.vstack([self.basis, other.basis]),
            coefficients=np.concatenate([self.coefficients, other.coefficients]),
            names=self.names + other.names,
            estimated=self.estimated + other.estimated,
        )


def constant_background(
    energy: np.ndarray, level: float, *, estimated: bool = True
) -> Background:
    """Flat background of ``level`` counts per channel at unit exposure."""
    energy = np.asarray(energy, dtype=np.float64)
    return Background(
        basis=np.ones((1, energy.size)),
        coefficients=np.array([float(level)]),
        names=("bg_level",),
        estimated=(estimated,),
    )


def linear_background(
    energy: np.ndarray, level: float, slope: float, *, estimated: bool = True
) -> Background:
    """``level + slope * (E - E_mid)`` with E_mid the window midpoint.

    Referring the slope to the midpoint only changes the coordinates of
    the background block; the information left for the peak parameters
    depends on the span of the basis, not on this choice.
    """
    energy = np.asarray(energy, dtype=np.float64)
    e_mid = 0.5 * (energy[0] + energy[-1])
    return Background(
        basis=np.vstack([np.ones(energy.size), energy - e_mid]),
        coefficients=np.array([float(level), float(slope)]),
        names=("bg_level", "bg_slope"),
        estimated=(estimated, estimated),
    )


def shirley_background(
    energy: np.ndarray,
    peaks: list[VoigtPeak],
    step: float,
    *,
    estimated: bool = True,
    rises_toward: Literal["high", "low"] = "high",
) -> Background:
    """Shirley-type step of fixed shape; only its height is a parameter.

    The shape is the running integral of the peak sum across the window,
    normalised to go from 0 to 1, so ``step`` is the background rise
    across the window in counts per channel (Shirley 1972; the endpoint
    convention of ``toyomacro.background.Shirley``). It is evaluated once
    at the stated peak parameters and then held fixed: the dependence of
    a real Shirley background on the peak parameters is *not*
    differentiated, so this understates the coupling between an
    iteratively determined Shirley background and the widths.

    Args:
        energy: Energy axis (eV), ascending
        peaks: Peaks whose integral shapes the step
        step: Rise across the window, counts per channel at unit exposure
        estimated: Whether ``step`` is a parameter or known
        rises_toward: 'high' puts the step-up at the high-energy end of
            the axis (a binding-energy axis), 'low' at the low end (a
            kinetic-energy axis)
    """
    energy = np.asarray(energy, dtype=np.float64)
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError("energy must be strictly ascending")
    total = np.zeros_like(energy)
    for p in peaks:
        total += p.amplitude * voigt_derivatives(energy, p.center, p.variance, p.gamma).value
    running = np.concatenate(
        [[0.0], np.cumsum(0.5 * (total[1:] + total[:-1]) * np.diff(energy))]
    )
    shape = running / running[-1]
    if rises_toward == "low":
        shape = 1.0 - shape
    elif rises_toward != "high":
        raise ValueError(f"rises_toward must be 'high' or 'low', got '{rises_toward}'")
    return Background(
        basis=shape[np.newaxis, :],
        coefficients=np.array([float(step)]),
        names=("bg_shirley_step",),
        estimated=(estimated,),
    )


# ---------------------------------------------------------------------------
# Width coordinates
# ---------------------------------------------------------------------------

_WIDTH_NAMES = {
    "sigma_gamma": ("sigma", "gamma"),
    "var_gamma": ("var", "gamma"),
    "fwhm_shape": ("fwhm", "shape"),
    "pvoigt": ("w", "eta"),
    "fixed_instrument": ("gamma",),
    # 'fixed_instrument' with a variance floor: the excess over the floor
    "instrument_floor": ("var_extra", "gamma"),
}


def _shape_slope(r: float) -> float:
    """g'(r) for g(r) = (1 - a r)**2 - b r**2, where f_G**2 = F**2 g(r)."""
    return -2.0 * _OL_A * (1.0 - _OL_A * r) - 2.0 * _OL_B * r


def _width_coordinates(
    name: str, variance: float, gamma: float, variance_floor: float = 0.0
) -> np.ndarray:
    """Width coordinates of the point (variance, gamma)."""
    if name == "sigma_gamma":
        return np.array([math.sqrt(variance), gamma])
    if name == "var_gamma":
        return np.array([variance, gamma])
    if name == "fwhm_shape":
        fwhm = voigt_fwhm(variance, gamma)
        return np.array([fwhm, 2.0 * gamma / fwhm])
    if name == "fixed_instrument":
        return np.array([gamma])
    if name == "instrument_floor":
        return np.array([variance - variance_floor, gamma])
    raise ValueError(f"no (variance, gamma) coordinates for '{name}'")


def _width_transform(name: str, variance: float, gamma: float) -> np.ndarray:
    """T = d(variance, gamma) / d(width coordinates), shape (2, n_width)."""
    if name == "sigma_gamma":
        return np.array([[2.0 * math.sqrt(variance), 0.0], [0.0, 1.0]])
    if name in ("var_gamma", "instrument_floor"):
        return np.eye(2)
    if name == "fwhm_shape":
        fwhm = voigt_fwhm(variance, gamma)
        r = 2.0 * gamma / fwhm
        # variance = F**2 g(r) / (8 ln2); its F-derivative is written as
        # 2 variance / F because g(r) itself cancels to nothing as r -> 1.
        return np.array([
            [2.0 * variance / fwhm, fwhm * fwhm * _shape_slope(r) / _EIGHT_LN2],
            [0.5 * r, 0.5 * fwhm],
        ])
    if name == "fixed_instrument":
        return np.array([[0.0], [1.0]])
    raise ValueError(f"no (variance, gamma) coordinates for '{name}'")


def _pvoigt_derivatives(x: np.ndarray, w: float, eta: float):
    """Pseudo-Voigt eta*L + (1-eta)*G at common FWHM w, and derivatives."""
    h = 0.5 * w
    den = x * x + h * h
    lor = h / (math.pi * den)
    lor_x = -2.0 * x * h / (math.pi * den * den)
    lor_w = 0.5 * (x * x - h * h) / (math.pi * den * den)

    s = w / math.sqrt(_EIGHT_LN2)
    gau = np.exp(-0.5 * (x / s) ** 2) / (s * _SQRT_2PI)
    gau_x = -x / (s * s) * gau
    gau_w = gau * (x * x / s**3 - 1.0 / s) / math.sqrt(_EIGHT_LN2)

    value = eta * lor + (1.0 - eta) * gau
    d_center = -(eta * lor_x + (1.0 - eta) * gau_x)
    d_w = eta * lor_w + (1.0 - eta) * gau_w
    d_eta = lor - gau
    return value, d_center, d_w, d_eta


# ---------------------------------------------------------------------------
# Poisson Fisher matrix with background
# ---------------------------------------------------------------------------


@dataclass
class PoissonFisherResult:
    """Poisson Fisher matrix of a peaks-plus-background model.

    Attributes:
        fisher: Fisher matrix in the chosen coordinates, (n_params, n_params)
        param_names: e.g. ('amp_0', 'center_0', 'var_0', 'gamma_0', 'bg_level')
        param_roles: 'amplitude', 'center', 'width' or 'background' per
            parameter
        param_values: Parameter values in the chosen coordinates
        width_index: Positions of the width parameters
        jacobian: d(expected counts)/d(parameters) at the stated exposure,
            shape (n_energy, n_params)
        expected_counts: Poisson mean per channel at the stated exposure,
            peaks plus every background term, known or estimated
        parameterization: Width coordinates used
        exposure: Multiplier on the whole mean; ``fisher`` is linear in it
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
    config: dict[str, Any] = field(default_factory=dict)


def poisson_fisher(
    energy: np.ndarray,
    peaks: list[VoigtPeak],
    background: Background | None = None,
    *,
    parameterization: Parameterization = "var_gamma",
    exposure: float = 1.0,
    variance_floor: float | None = None,
) -> PoissonFisherResult:
    """Fisher matrix ``J^T diag(1/mu) J`` for Poisson counts.

    ``mu(E) = exposure * (sum_k A_k V_k(E) + background(E))``. Each peak
    contributes amplitude, center and its width coordinates; each
    background term flagged ``estimated`` contributes its coefficient. A
    known background raises ``mu`` -- and with it the counting noise
    under the peak -- without adding parameters; an estimated one
    additionally competes with the peak parameters for the same counts.

    Width coordinates:

    - 'sigma_gamma': (sigma, gamma). Singular at sigma = 0 by
      construction of the coordinate, see the module docstring.
    - 'var_gamma': (sigma**2, gamma). Regular at sigma = 0.
    - 'fwhm_shape': (F, r) with F the Olivero-Longbothum total FWHM and
      r = f_L / F the Lorentzian share of it, 0 for a Gaussian and
      0.999997 (not 1: the published coefficients are rounded) for a
      Lorentzian.
    - 'fixed_instrument': the instrument's Gaussian width is calibrated
      and not estimated. Without ``variance_floor`` that is all of the
      Gaussian width: (gamma,) only, each peak's ``sigma`` taken as the
      calibrated value, no sample-side Gaussian broadening allowed for.
      With ``variance_floor = sigma_inst**2`` the sample may add to it:
      (var_extra, gamma) with ``var_extra = sigma**2 - variance_floor >=
      0``, the usual XPS question of whether there is broadening beyond
      the known resolution. That matrix equals the 'var_gamma' one --
      shifting the origin of a coordinate changes no derivative -- and
      what moves is the boundary, from ``sigma**2 = 0`` to ``var_extra =
      0``.
    - 'pvoigt': (w, eta) of the pseudo-Voigt ``eta L + (1 - eta) G`` at
      common FWHM ``w``. This is a **different lineshape model**, not a
      re-parameterisation of the Voigt: each peak's (sigma, gamma) is
      mapped to (w, eta) by Thompson, Cox & Hastings (1987) and mean and
      derivatives are those of the pseudo-Voigt at that point.

    The first four are related by ``I_phi = T^T I_theta T`` and describe
    the same statistical model; only 'sigma_gamma' has a singular ``T``.

    Args:
        energy: Energy axis (eV), shape (n_energy,)
        peaks: One or more peaks
        background: None for no background, else a ``Background``
        parameterization: Width coordinates, see above
        exposure: Multiplier on the whole mean (acquisition time, flux).
            Parameters are defined at unit exposure, so the matrix is
            exactly linear in it.
        variance_floor: Calibrated instrument variance (eV**2), for
            'fixed_instrument' only; see above. Every peak must have
            ``sigma**2 >= variance_floor``.

    Returns:
        PoissonFisherResult. The matrix is returned as computed and can
        be singular; nothing here inverts it.

    Raises:
        ValueError: if the expected count is not positive in every
            channel. Without a background that happens where a profile
            underflows (a pure Gaussian far from its center); the
            information is then undefined, not large. Also if
            ``variance_floor`` is given with another parameterization
            (it would be silently ignored) or exceeds a peak's variance.
    """
    if parameterization not in PARAMETERIZATIONS:
        raise ValueError(
            f"parameterization must be one of {PARAMETERIZATIONS}, got '{parameterization}'"
        )
    if not peaks:
        raise ValueError("need at least one peak")
    if not exposure > 0.0:
        raise ValueError(f"exposure must be positive, got {exposure}")
    coordinates: str = parameterization
    if variance_floor is not None:
        if parameterization != "fixed_instrument":
            raise ValueError("variance_floor only applies to parameterization='fixed_instrument'")
        if variance_floor < 0.0 or any(p.variance < variance_floor for p in peaks):
            raise ValueError(
                f"need 0 <= variance_floor <= sigma**2 for every peak, got {variance_floor}"
            )
        coordinates = "instrument_floor"
    energy = np.asarray(energy, dtype=np.float64)

    mean = np.zeros_like(energy)
    columns: list[np.ndarray] = []
    names: list[str] = []
    roles: list[str] = []
    values: list[float] = []

    for k, p in enumerate(peaks):
        if parameterization == "pvoigt":
            w, eta = PseudoVoigt.eta_from_voigt_params(
                math.sqrt(_EIGHT_LN2) * p.sigma, 2.0 * p.gamma
            )
            value, d_center, d_w, d_eta = _pvoigt_derivatives(energy - p.center, w, float(eta))
            width_columns = p.amplitude * np.stack([d_w, d_eta], axis=1)
            width_values = np.array([w, float(eta)])
        else:
            d = voigt_derivatives(energy, p.center, p.variance, p.gamma)
            value, d_center = d.value, d.d_center
            canonical = p.amplitude * np.stack([d.d_variance, d.d_gamma], axis=1)
            width_columns = canonical @ _width_transform(coordinates, p.variance, p.gamma)
            width_values = _width_coordinates(
                coordinates, p.variance, p.gamma, variance_floor or 0.0
            )

        mean += p.amplitude * value
        columns += [value, p.amplitude * d_center, *width_columns.T]
        width_names = _WIDTH_NAMES[coordinates]
        names += [f"amp_{k}", f"center_{k}", *(f"{n}_{k}" for n in width_names)]
        roles += ["amplitude", "center", *(["width"] * len(width_names))]
        values += [p.amplitude, p.center, *width_values]

    if background is not None:
        if background.basis.shape[1] != energy.size:
            raise ValueError("background is on a different energy axis")
        mean += background.counts()
        for j, is_estimated in enumerate(background.estimated):
            if is_estimated:
                columns.append(background.basis[j])
                names.append(background.names[j])
                roles.append("background")
                values.append(float(background.coefficients[j]))
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate parameter names: {names}")

    if not np.all(mean > 0.0):
        raise ValueError(
            "expected counts must be positive in every channel; "
            f"minimum is {mean.min():.3e}. Add a background or narrow the window."
        )

    jac_unit = np.stack(columns, axis=1)  # (n_energy, n_params), unit exposure
    weighted = jac_unit / np.sqrt(mean)[:, np.newaxis]
    fisher = exposure * (weighted.T @ weighted)

    return PoissonFisherResult(
        fisher=fisher,
        param_names=tuple(names),
        param_roles=tuple(roles),
        param_values=np.array(values, dtype=np.float64),
        width_index=tuple(i for i, r in enumerate(roles) if r == "width"),
        jacobian=exposure * jac_unit,
        expected_counts=exposure * mean,
        parameterization=parameterization,
        exposure=float(exposure),
        config={
            "peaks": [(p.amplitude, p.center, p.sigma, p.gamma) for p in peaks],
            "background_terms": () if background is None else background.names,
            "background_estimated": () if background is None else background.estimated,
            "variance_floor": variance_floor,
            "n_energy": int(energy.size),
            "energy_range": (float(energy[0]), float(energy[-1])),
        },
    )


# ---------------------------------------------------------------------------
# Effective information
# ---------------------------------------------------------------------------


@dataclass
class EffectiveInformation:
    """Information about a target block, with and without the rest known.

    Attributes:
        target_index: Positions of the target parameters in the full matrix
        effective: ``I_uu - I_uq pinv(I_qq) I_qu``: what is left for the
            targets ``u`` once the other parameters ``q`` are estimated
            from the same data. Where it is invertible its inverse is the
            target block of the full inverse Fisher matrix.
        conditional: ``I_uu``: the information if ``q`` were known
            exactly. ``conditional - effective`` is positive
            semi-definite, so this is never the smaller of the two; it is
            what ``fisher_information.analyze_sigma_gamma_axes`` reports.
        nuisance_null_dim: Number of numerically null directions in the
            nuisance block, by the ``n * eps`` rule on its correlation
            matrix. Non-zero means some combination of nuisance
            parameters is itself not estimable; ``effective`` is then
            taken with those directions dropped.
    """

    target_index: tuple[int, ...]
    effective: np.ndarray
    conditional: np.ndarray
    nuisance_null_dim: int


def effective_information(
    fisher: np.ndarray, target_index: tuple[int, ...] | list[int]
) -> EffectiveInformation:
    """Schur complement of the nuisance block, in a scale-free metric.

    The Fisher matrix mixes units (counts, eV, eV**2), so the nuisance
    block is inverted after rescaling the whole matrix to unit diagonal
    and the result is scaled back. The rank decision inside the
    pseudo-inverse therefore does not move with the units of any
    parameter -- the same gauge ``crlb.compute_multipeak_fisher`` fixes,
    with the same ``n * eps`` threshold.

    Args:
        fisher: Symmetric positive semi-definite matrix, (n, n)
        target_index: Positions of the target parameters ``u``; every
            other parameter is a nuisance parameter ``q``

    Returns:
        EffectiveInformation
    """
    fisher = np.asarray(fisher, dtype=np.float64)
    n = fisher.shape[0]
    u = np.array(sorted(target_index), dtype=int)
    if u.size == 0 or u.size != len(set(target_index)) or u.min() < 0 or u.max() >= n:
        raise ValueError(f"invalid target_index {tuple(target_index)} for a {n}x{n} matrix")
    q = np.array([i for i in range(n) if i not in set(u.tolist())], dtype=int)

    conditional = fisher[np.ix_(u, u)].copy()
    if q.size == 0:
        return EffectiveInformation(tuple(u.tolist()), conditional.copy(), conditional, 0)

    d = np.sqrt(np.maximum(np.diag(fisher), 0.0))
    d_safe = np.where(d > 0.0, d, 1.0)
    scaled = fisher / np.outer(d_safe, d_safe)

    eigenvalues, eigenvectors = np.linalg.eigh(scaled[np.ix_(q, q)])
    threshold = max(eigenvalues[-1] * q.size * np.finfo(np.float64).eps, 1e-300)
    null = eigenvalues <= threshold
    inverse = np.where(null, 0.0, 1.0 / np.where(null, 1.0, eigenvalues))
    c_uq = scaled[np.ix_(u, q)] @ eigenvectors
    schur = scaled[np.ix_(u, u)] - (c_uq * inverse[np.newaxis, :]) @ c_uq.T
    schur = 0.5 * (schur + schur.T)

    return EffectiveInformation(
        target_index=tuple(u.tolist()),
        effective=schur * np.outer(d_safe[u], d_safe[u]),
        conditional=conditional,
        nuisance_null_dim=int(null.sum()),
    )


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentifiabilityThresholds:
    """Where 'identified' ends. Conventions, not physics.

    Attributes:
        weak_relative_sd: A parameter is 'weakly_identified' when its
            standard-deviation bound exceeds this fraction of its
            reference scale. The default 0.1 means "a tenth of the total
            FWHM" for positions and widths: a working rule of thumb for
            when a width has stopped being a usable number, with no
            deeper justification. Change it to suit the question.
        boundary_sd: The Gaussian variance is 'near_boundary' when it
            lies within this many standard-deviation bounds of its lower
            limit. For an unconstrained, normally distributed estimator
            the chance of falling below the limit would be 0.13 % at 3
            and 2.3 % at 2 -- but that normal approximation is exactly
            what fails near a boundary (Self & Liang 1987), so read the
            default 3 as a margin of caution, not as a probability.
    """

    weak_relative_sd: float = 0.1
    boundary_sd: float = 3.0

    def __post_init__(self) -> None:
        if not (self.weak_relative_sd > 0.0 and self.boundary_sd > 0.0):
            raise ValueError("thresholds must be positive")


@dataclass(frozen=True)
class ParameterAssessment:
    """One parameter of the model, in 'var_gamma' coordinates.

    Attributes:
        name: As in ``PoissonFisherResult.param_names``
        role: 'amplitude', 'center', 'width' or 'background'
        peak: Peak index, None for background coefficients
        value: Parameter value
        sd: ``sqrt([I^-1]_ii)``, the bound with every other parameter
            estimated from the same spectrum; ``inf`` when rank deficient
        sd_conditional: Widths only. The same from the width sub-block
            alone, i.e. with amplitude, position and background taken as
            known. Never larger than ``sd``; the gap is what estimating
            the rest costs.
        reference_scale: |amplitude| for an amplitude, the peak's total
            FWHM for its center and gamma, FWHM**2 / (8 ln2) for its
            variance; None for background coefficients, for which no
            natural scale exists
        relative_sd: ``sd / reference_scale``
        status: 'rank_deficient', 'weakly_identified' or 'identified';
            'not_assessed' for a background coefficient that is not rank
            deficient, because without a scale its precision is not
            judged
    """

    name: str
    role: str
    peak: int | None
    value: float
    sd: float
    sd_conditional: float | None
    reference_scale: float | None
    relative_sd: float | None
    status: str


@dataclass(frozen=True)
class WidthAssessment:
    """The (variance, gamma) pair of one peak, judged as a block.

    Attributes:
        peak: Peak index
        fwhm: Olivero-Longbothum total FWHM, the reference scale
        variance_excess: ``sigma**2 - variance_floor``
        sd_variance: Bound on the standard deviation of the variance
        variance_relative_sd: ``sd_variance / variance_excess`` -- the
            precision of the Gaussian component *relative to itself*.
            Grows without limit as the component vanishes however good
            ``worst_relative_sd`` looks; ``inf`` on the boundary.
        near_boundary: ``variance_excess < boundary_sd * sd_variance``.
            Independent of ``status``.
        worst_relative_sd: Square root of the largest eigenvalue of the
            2x2 covariance bound after dividing variance by
            FWHM**2/(8 ln2) and gamma by FWHM: the relative bound along
            the least determined direction of the pair. Never smaller
            than either parameter's own ``relative_sd``.
        worst_direction: That direction, in the same scaled coordinates
        status: 'rank_deficient' if either width is; else
            'weakly_identified' if ``worst_relative_sd`` exceeds the
            threshold; else 'identified'
    """

    peak: int
    fwhm: float
    variance_excess: float
    sd_variance: float
    variance_relative_sd: float
    near_boundary: bool
    worst_relative_sd: float
    worst_direction: tuple[float, float]
    status: str


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Result of ``assess_identifiability``.

    Attributes:
        parameters: One entry per Fisher-matrix parameter
        widths: One entry per peak
        null_space_dim: Numerically null directions of the full matrix
        condition_number: Of the unit-diagonal (correlation) matrix, so
            it does not move with the units of any parameter
        thresholds: The thresholds used
        variance_floor: The lower limit of the Gaussian variance used
        fisher: The underlying 'var_gamma' ``PoissonFisherResult``
    """

    parameters: tuple[ParameterAssessment, ...]
    widths: tuple[WidthAssessment, ...]
    null_space_dim: int
    condition_number: float
    thresholds: IdentifiabilityThresholds
    variance_floor: float
    fisher: PoissonFisherResult = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python summary (no arrays), for JSON or a table row."""
        return {
            "null_space_dim": self.null_space_dim,
            "condition_number": self.condition_number,
            "variance_floor": self.variance_floor,
            "weak_relative_sd": self.thresholds.weak_relative_sd,
            "boundary_sd": self.thresholds.boundary_sd,
            "parameters": [vars(a).copy() for a in self.parameters],
            "widths": [vars(w).copy() for w in self.widths],
        }


def _covariance_bound(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Inverse of a Fisher matrix with the rank decision of ``crlb``.

    Returns the pseudo-inverse taken on the unit-diagonal matrix (``n *
    eps`` rule), a mask of the axes that project onto its null space and
    whose bound is therefore infinite, the null-space dimension and the
    condition number of the unit-diagonal matrix. Entries of the
    pseudo-inverse on masked axes are not bounds and must not be read.
    """
    n = matrix.shape[0]
    d = np.sqrt(np.maximum(np.diag(matrix), 0.0))
    d_safe = np.where(d > 0.0, d, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix / np.outer(d_safe, d_safe))
    threshold = max(eigenvalues[-1] * n * np.finfo(np.float64).eps, 1e-300)
    null = eigenvalues <= threshold
    inverse = np.where(null, 0.0, 1.0 / np.where(null, 1.0, eigenvalues))
    covariance = (eigenvectors * inverse[np.newaxis, :]) @ eigenvectors.T
    covariance /= np.outer(d_safe, d_safe)
    unbounded = (eigenvectors[:, null] ** 2).sum(axis=1) > _NULL_PROJECTION_TOL
    positive = eigenvalues[eigenvalues > 0.0]
    condition = float(positive[-1] / positive[0]) if positive.size >= 2 and not null.any() else np.inf
    return covariance, unbounded, int(null.sum()), condition


def assess_identifiability(
    energy: np.ndarray,
    peaks: list[VoigtPeak],
    background: Background | None = None,
    *,
    exposure: float = 1.0,
    variance_floor: float = 0.0,
    thresholds: IdentifiabilityThresholds | None = None,
) -> IdentifiabilityReport:
    """Sort the parameters of a peaks-plus-background model by identifiability.

    Always evaluated in 'var_gamma' coordinates, so that the degeneracy
    of the ``sigma`` coordinate at ``sigma = 0`` cannot raise a flag by
    itself, and from the full inverse Fisher matrix, i.e. from the
    effective information with everything else estimated.

    Three exclusive labels, in this order of precedence:

    - 'rank_deficient': the unit-diagonal Fisher matrix is numerically
      singular and this parameter's axis projects onto its null space.
      The rank rule and both tolerances are those of
      ``crlb.compute_multipeak_fisher``. The bound is infinite.
    - 'weakly_identified': the bound on the standard deviation exceeds
      ``thresholds.weak_relative_sd`` times the reference scale.
    - 'identified': neither.

    and one independent flag per peak, 'near_boundary', for a Gaussian
    variance within ``thresholds.boundary_sd`` bounds of its lower limit
    ``variance_floor`` (0, or a calibrated instrument variance when the
    quantity of interest is the broadening in excess of it).

    What the labels do not mean:

    - 'identified' on the FWHM scale is **not** detection of a small
      Gaussian component. The variance is judged against
      FWHM**2/(8 ln2), a scale that stays finite as the component
      vanishes, so a peak with ``sigma/gamma = 0.03`` can be
      'identified' while ``variance_relative_sd`` is above 1. Read that
      field, and ``near_boundary``, for the component itself.
    - 'weakly_identified' is **not** structural non-identifiability. It
      says the information in *this* window, at *this* exposure and with
      *this* background is small against a chosen scale; more counts or
      a wider window change it. Only 'rank_deficient' is a statement
      about the model, and only a numerical one.
    - Near the boundary the inverse Fisher matrix is not the variance of
      a constrained estimator, and a Monte Carlo variance need not agree
      with it (Self & Liang 1987). The bounds reported for a
      'near_boundary' peak are still the Fisher numbers; they are not
      corrected.
    - Everything is computed at the stated parameters from a model. It
      is not a property of any data set, and the conditions in the
      module docstring of ``crlb`` apply.

    Args:
        energy: Energy axis (eV)
        peaks: One or more peaks
        background: None, or a ``Background``
        exposure: Multiplier on the whole mean; bounds scale as
            ``1/sqrt(exposure)``
        variance_floor: Lower limit of every peak's Gaussian variance
            (eV**2). 0 for the plain Voigt; ``sigma_inst**2`` when the
            instrument width is calibrated and only the excess is in
            question.
        thresholds: Defaults to ``IdentifiabilityThresholds()``

    Returns:
        IdentifiabilityReport
    """
    thresholds = thresholds or IdentifiabilityThresholds()
    if variance_floor < 0.0 or any(p.variance < variance_floor for p in peaks):
        raise ValueError(
            f"need 0 <= variance_floor <= sigma**2 for every peak, got {variance_floor}"
        )
    fisher = poisson_fisher(energy, peaks, background, exposure=exposure)
    covariance, unbounded, null_dim, condition = _covariance_bound(fisher.fisher)

    width_index = list(fisher.width_index)
    information = effective_information(fisher.fisher, width_index)
    cond_cov, cond_unbounded, _, _ = _covariance_bound(information.conditional)
    sd_conditional = {
        i: (np.inf if cond_unbounded[j] else math.sqrt(max(cond_cov[j, j], 0.0)))
        for j, i in enumerate(information.target_index)
    }

    fwhm = [voigt_fwhm(p.variance, p.gamma) for p in peaks]
    peak_of = []
    for k in range(len(peaks)):
        peak_of += [k] * 4
    peak_of += [None] * (len(fisher.param_names) - len(peak_of))

    parameters = []
    for i, (name, role) in enumerate(zip(fisher.param_names, fisher.param_roles)):
        k = peak_of[i]
        sd = np.inf if unbounded[i] else math.sqrt(max(covariance[i, i], 0.0))
        if role == "amplitude":
            scale = abs(peaks[k].amplitude)
        elif role == "background":
            scale = None
        elif name.startswith("var_"):
            scale = fwhm[k] ** 2 / _EIGHT_LN2
        else:
            scale = fwhm[k]
        relative = None if scale is None else sd / scale
        if unbounded[i]:
            status = "rank_deficient"
        elif relative is None:
            status = "not_assessed"
        else:
            status = "weakly_identified" if relative > thresholds.weak_relative_sd else "identified"
        parameters.append(ParameterAssessment(
            name=name, role=role, peak=k, value=float(fisher.param_values[i]), sd=float(sd),
            sd_conditional=sd_conditional.get(i), reference_scale=scale,
            relative_sd=relative, status=status,
        ))

    widths = []
    for k, p in enumerate(peaks):
        iv, ig = 4 * k + 2, 4 * k + 3
        excess = p.variance - variance_floor
        if unbounded[iv] or unbounded[ig]:
            sd_v = np.inf if unbounded[iv] else math.sqrt(max(covariance[iv, iv], 0.0))
            worst, direction, status = np.inf, (math.nan, math.nan), "rank_deficient"
        else:
            sd_v = math.sqrt(max(covariance[iv, iv], 0.0))
            scales = np.array([fwhm[k] ** 2 / _EIGHT_LN2, fwhm[k]])
            block = covariance[np.ix_([iv, ig], [iv, ig])] / np.outer(scales, scales)
            eigenvalues, eigenvectors = np.linalg.eigh(block)
            worst = math.sqrt(max(eigenvalues[-1], 0.0))
            direction = (float(eigenvectors[0, -1]), float(eigenvectors[1, -1]))
            status = "weakly_identified" if worst > thresholds.weak_relative_sd else "identified"
        widths.append(WidthAssessment(
            peak=k, fwhm=fwhm[k], variance_excess=excess, sd_variance=float(sd_v),
            variance_relative_sd=float(sd_v / excess) if excess > 0.0 else np.inf,
            near_boundary=bool(excess < thresholds.boundary_sd * sd_v),
            worst_relative_sd=float(worst), worst_direction=direction, status=status,
        ))

    return IdentifiabilityReport(
        parameters=tuple(parameters), widths=tuple(widths), null_space_dim=null_dim,
        condition_number=condition, thresholds=thresholds,
        variance_floor=float(variance_floor), fisher=fisher,
    )


# ---------------------------------------------------------------------------
# Scan over measurement conditions
# ---------------------------------------------------------------------------

_STATUS_CODE = {"identified": 0, "weakly_identified": 1, "rank_deficient": 2}


@dataclass(frozen=True)
class ScanGrid:
    """Conditions for ``scan_identifiability``. Lengths are in units of the
    total FWHM, which is held fixed, so the grid is free of energy units.

    Attributes:
        half_widths: Window half-widths, measured outward from the
            outermost peak center
        levels: Exposure multipliers, or total counts in the window when
            ``normalization='total_counts'``
        ratios: ``sigma/gamma`` at fixed total FWHM. 0 is a pure
            Lorentzian; use a large finite number, not ``inf``, for a
            Gaussian (a pure Gaussian without background has channels
            with no expected counts).
        backgrounds: Any of 'none', 'known', 'estimated'
        separations: Peak spacings for a second, identical peak; None is
            a single peak. Because the window is measured outward from
            the outermost center, a pair is given a window wider by its
            separation, so at narrow half-widths a pair can come out
            better determined than a single peak in the "same" cell.
            Compare across separations at wide windows only.
        normalization: 'exposure' keeps the energy step and the exposure
            per point fixed, so a wider window has more points *and* more
            counts -- how a measurement actually trades. 'total_counts'
            rescales every window to the same total, which isolates what
            the window's shape contributes from what its counts do.
        background_kind: 'constant', 'linear' (zero true slope) or
            'shirley' (a constant plus a Shirley-type step of the same
            height); applies to 'known' and 'estimated'
        background_fraction: Background level as a fraction of the height
            of a single peak
        step: Energy step
        fwhm: Total FWHM in eV; only sets the unit of the energy axis
        area: Peak area at unit exposure, in counts x eV per step, i.e.
            ``area * V(E)`` counts per channel
        variance_floor: As in ``assess_identifiability``, in units of
            ``fwhm**2``
        thresholds: As in ``assess_identifiability``
    """

    half_widths: tuple[float, ...]
    levels: tuple[float, ...] = (1.0,)
    ratios: tuple[float, ...] = (1.0,)
    backgrounds: tuple[str, ...] = ("none", "known", "estimated")
    separations: tuple[float | None, ...] = (None,)
    normalization: Literal["exposure", "total_counts"] = "exposure"
    background_kind: Literal["constant", "linear", "shirley"] = "constant"
    background_fraction: float = 0.1
    step: float = 0.02
    fwhm: float = 1.0
    area: float = 1.0e4
    variance_floor: float = 0.0
    thresholds: IdentifiabilityThresholds = field(default_factory=IdentifiabilityThresholds)


def _widths_at_fwhm(ratio: float, fwhm: float) -> tuple[float, float]:
    """(sigma, gamma) with sigma/gamma = ratio and Olivero-Longbothum FWHM = fwhm."""
    gamma = fwhm / (2.0 * _OL_A + math.sqrt(4.0 * _OL_B + _EIGHT_LN2 * ratio * ratio))
    return ratio * gamma, gamma


def _scan_background(grid: ScanGrid, mode: str, energy: np.ndarray, peaks: list[VoigtPeak]):
    if mode == "none":
        return None
    if mode not in ("known", "estimated"):
        raise ValueError(f"background must be 'none', 'known' or 'estimated', got '{mode}'")
    estimated = mode == "estimated"
    first = peaks[0]
    height = first.amplitude * voigt_derivatives(
        np.array([first.center]), first.center, first.variance, first.gamma
    ).value[0]
    level = grid.background_fraction * height
    if grid.background_kind == "constant":
        return constant_background(energy, level, estimated=estimated)
    if grid.background_kind == "linear":
        return linear_background(energy, level, 0.0, estimated=estimated)
    if grid.background_kind == "shirley":
        return constant_background(energy, level, estimated=estimated) + shirley_background(
            energy, peaks, level, estimated=estimated
        )
    raise ValueError(f"unknown background_kind '{grid.background_kind}'")


def scan_identifiability(grid: ScanGrid) -> dict[str, np.ndarray]:
    """Evaluate the assessment over a grid of measurement conditions.

    Every cell is evaluated twice, in 'var_gamma' and in 'sigma_gamma',
    so that the two ways a Gaussian width can look badly determined come
    apart: along ``ratios`` towards 0 the ``sigma`` bound diverges while
    the variance bound stays put (the coordinate), whereas along
    ``half_widths`` towards 0, or with an estimated background, both
    grow (the information). The no-background rows show the window
    dependence on its own.

    Returns:
        A flat dict that ``np.savez(path, **result)`` stores as is. Axis
        arrays ``backgrounds``, ``separations`` (NaN for a single peak),
        ``ratios``, ``half_widths``, ``levels``, and result arrays of
        shape ``(n_backgrounds, n_separations, n_ratios, n_half_widths,
        n_levels)``, all for the first peak:

        - ``total_counts``, ``n_energy``
        - ``condition_number``: of the unit-diagonal matrix ('var_gamma')
        - ``sd_variance``, ``sd_gamma``: full bounds over their reference
          scales FWHM**2/(8 ln2) and FWHM; ``*_conditional`` the same
          from the width sub-block with everything else known
        - ``worst_relative_sd``, ``variance_relative_sd``,
          ``near_boundary``, ``status`` (0 identified, 1 weakly
          identified, 2 rank deficient): as in ``WidthAssessment``
        - ``sd_sigma``: bound on sigma in 'sigma_gamma' coordinates over
          FWHM/sqrt(8 ln2); ``inf`` at sigma = 0
        - ``info_variance``, ``info_sigma``: effective information on the
          single parameter, times its reference scale squared; the second
          vanishes like sigma**2, the first does not

        and ``normalization``, ``background_kind``, ``fwhm`` as 0-d
        arrays.
    """
    if grid.normalization not in ("exposure", "total_counts"):
        raise ValueError(f"unknown normalization '{grid.normalization}'")
    shape = (
        len(grid.backgrounds), len(grid.separations), len(grid.ratios),
        len(grid.half_widths), len(grid.levels),
    )
    keys = (
        "total_counts", "n_energy", "condition_number", "sd_variance", "sd_gamma",
        "sd_variance_conditional", "sd_gamma_conditional", "worst_relative_sd",
        "variance_relative_sd", "sd_sigma", "info_variance", "info_sigma",
    )
    out = {key: np.full(shape, np.nan) for key in keys}
    out["near_boundary"] = np.zeros(shape, dtype=bool)
    out["status"] = np.full(shape, -1, dtype=np.int8)

    fwhm = grid.fwhm
    sigma_scale = fwhm / math.sqrt(_EIGHT_LN2)
    floor = grid.variance_floor * fwhm * fwhm
    for index in np.ndindex(shape[:4]):
        b, s_, r, h = index
        sigma, gamma = _widths_at_fwhm(grid.ratios[r], fwhm)
        separation = grid.separations[s_]
        centers = [0.0] if separation is None else [-0.5 * separation * fwhm, 0.5 * separation * fwhm]
        peaks = [VoigtPeak(grid.area, c, sigma, gamma) for c in centers]
        lo = centers[0] - grid.half_widths[h] * fwhm
        hi = centers[-1] + grid.half_widths[h] * fwhm
        energy = np.linspace(lo, hi, int(round((hi - lo) / (grid.step * fwhm))) + 1)
        background = _scan_background(grid, grid.backgrounds[b], energy, peaks)
        unit_counts = poisson_fisher(energy, peaks, background).expected_counts.sum()

        for j, level in enumerate(grid.levels):
            exposure = level if grid.normalization == "exposure" else level / unit_counts
            report = assess_identifiability(
                energy, peaks, background, exposure=exposure,
                variance_floor=floor, thresholds=grid.thresholds,
            )
            cell = index + (j,)
            var_, gam_ = report.parameters[2], report.parameters[3]
            width = report.widths[0]
            out["total_counts"][cell] = exposure * unit_counts
            out["n_energy"][cell] = energy.size
            out["condition_number"][cell] = report.condition_number
            out["sd_variance"][cell] = var_.relative_sd
            out["sd_gamma"][cell] = gam_.relative_sd
            out["sd_variance_conditional"][cell] = var_.sd_conditional / var_.reference_scale
            out["sd_gamma_conditional"][cell] = gam_.sd_conditional / gam_.reference_scale
            out["worst_relative_sd"][cell] = width.worst_relative_sd
            out["variance_relative_sd"][cell] = width.variance_relative_sd
            out["near_boundary"][cell] = width.near_boundary
            out["status"][cell] = _STATUS_CODE[width.status]
            out["info_variance"][cell] = (var_.reference_scale / var_.sd) ** 2

            in_sigma = poisson_fisher(
                energy, peaks, background, parameterization="sigma_gamma", exposure=exposure
            )
            covariance, unbounded, _, _ = _covariance_bound(in_sigma.fisher)
            sd_sigma = np.inf if unbounded[2] else math.sqrt(max(covariance[2, 2], 0.0))
            out["sd_sigma"][cell] = sd_sigma / sigma_scale
            out["info_sigma"][cell] = (sigma_scale / sd_sigma) ** 2

    out["backgrounds"] = np.array(grid.backgrounds)
    out["separations"] = np.array(
        [np.nan if s_ is None else s_ for s_ in grid.separations], dtype=np.float64
    )
    out["ratios"] = np.array(grid.ratios, dtype=np.float64)
    out["half_widths"] = np.array(grid.half_widths, dtype=np.float64)
    out["levels"] = np.array(grid.levels, dtype=np.float64)
    out["normalization"] = np.array(grid.normalization)
    out["background_kind"] = np.array(grid.background_kind)
    out["fwhm"] = np.array(fwhm)
    return out

