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
whether the Gaussian variance, or gamma, sits near its boundary;
``identifiability_modes`` reports which combinations of peak parameters
are determined worst and what each looks like in the spectrum.
``identifiability_at_estimate`` and ``identifiability_from_exact_k``
evaluate both at a fit's result -- still bounds from a model, now
plugged in at a noisy estimate, not a statement about that fit's error.
What each does and does not mean is in its docstring.

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
``w''(z)`` a term of order ``4z`` cancels down to ``2/z**3``. The wings go
first, and the Fisher matrix follows: on a window of +-33 gamma the
variance element built from ``voigt_jacobian.voigt_with_hessian`` (or
from ``voigt_with_jacobian`` by the chain rule) is off by 1.5e-3 at
``sigma/gamma`` = 1e-3, by a factor of about 6 at 3e-4 (5.2 to 6.5
depending on the grid) and by four orders of magnitude at 1e-4. Here every energy point with ``|z| >= 7`` is
evaluated from the large-``|z|`` expansion instead,

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
  on d_variance at sigma/gamma = 1 for every N from 2 to 70, the best
  that any of them reaches is 1e-8 at ``|z| = 5`` (N = 25) and 3e-5 at
  ``|z| = 4`` (N = 12); N = 24 gives 2.5e-3 at ``|z| = 4`` and N = 40
  gives 3e2. Past the turning point more terms make it worse.

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

The parts that do not depend on the lineshape -- ``Background`` and its
constant and linear builders, the Fisher product, the rank decision,
``effective_information``, ``IdentifiabilityThresholds``, ``Mode`` and
the mode decomposition itself -- are defined in the private module
``toyomacro._identifiability`` and re-exported here unchanged; this
module is where they are documented and supported.

References:
    J. J. Olivero and R. L. Longbothum, J. Quant. Spectrosc. Radiat.
        Transfer 17, 233-236 (1977), doi:10.1016/0022-4073(77)90161-3
    P. Thompson, D. E. Cox and J. B. Hastings, J. Appl. Cryst. 20, 79-83
        (1987), doi:10.1107/S0021889887087090
    Y. Wang, B. Zhou, R. Zhao, B. Wang, Q. Liu and M. Dai, Mathematics
        10, 210 (2022), doi:10.3390/math10020210
    S. G. Self and K.-Y. Liang, J. Am. Stat. Assoc. 82, 605-610 (1987),
        doi:10.1080/01621459.1987.10478472
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import numpy as np
from scipy import special as sps

from .._identifiability import (
    Background,
    EffectiveInformation,
    IdentifiabilityThresholds,
    Mode,
    constant_background,
    covariance_bound,
    effective_information,
    linear_background,
    mode_decomposition,
    poisson_fisher_matrix,
)
from .._identifiability import null_directions as _null_directions
from ..lineshape.pseudovoigt import PseudoVoigt
from .crlb import _NULL_PROJECTION_TOL

__all__ = [
    "INTENSITY_SEMANTICS",
    "PARAMETERIZATIONS",
    "PROFILE_GRID",
    "VoigtDerivatives",
    "VoigtPeak",
    "Background",
    "PoissonFisherResult",
    "EffectiveInformation",
    "IdentifiabilityThresholds",
    "ParameterAssessment",
    "WidthAssessment",
    "IdentifiabilityReport",
    "EstimateReport",
    "Mode",
    "ModeProfile",
    "ModeReport",
    "ProfilePoint",
    "ProfileSide",
    "ScanGrid",
    "voigt_derivatives",
    "voigt_fwhm",
    "constant_background",
    "linear_background",
    "shirley_background",
    "poisson_fisher",
    "effective_information",
    "assess_identifiability",
    "identifiability_at_estimate",
    "identifiability_from_exact_k",
    "identifiability_modes",
    "mode_profile",
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


def _derivatives_series(x: np.ndarray, variance, gamma, n_terms: int | None = None):
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


def _derivatives_faddeeva(x: np.ndarray, variance, gamma):
    """Closed form from w(z), w'(z), w''(z); needs variance > 0."""
    sigma = np.sqrt(variance)
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
    center: float | np.ndarray,
    variance: float | np.ndarray,
    gamma: float | np.ndarray,
    *,
    route: Literal["auto", "faddeeva", "series"] = "auto",
) -> VoigtDerivatives:
    """Voigt profile and derivatives w.r.t. center, Gaussian variance, gamma.

    Valid for ``variance >= 0`` and ``gamma >= 0``, not both zero,
    including ``variance = 0`` exactly, where the profile is the
    Lorentzian ``L`` and ``d_variance`` is ``L''/2``.

    ``center``, ``variance`` and ``gamma`` may be arrays that broadcast
    against ``energy`` -- shape (n, 1) against an axis of shape
    (n_energy,) evaluates n profiles at once, each point taking its own
    route -- and the outputs then have the broadcast shape.

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
    variance = np.asarray(variance, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    if np.any(variance < 0.0) or np.any(gamma < 0.0) or np.any((variance == 0.0) & (gamma == 0.0)):
        raise ValueError(
            "need variance >= 0 and gamma >= 0, not both zero; "
            f"got variance={variance}, gamma={gamma}"
        )
    x = np.asarray(energy, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    x, variance, gamma = np.broadcast_arrays(x, variance, gamma)

    if route == "series":
        return VoigtDerivatives(*_derivatives_series(x, variance, gamma))
    if route == "faddeeva":
        if np.any(variance == 0.0):
            raise ValueError("route='faddeeva' needs variance > 0")
        return VoigtDerivatives(*_derivatives_faddeeva(x, variance, gamma))
    if route != "auto":
        raise ValueError(f"route must be 'auto', 'faddeeva' or 'series', got '{route}'")

    # |z|**2 >= Z**2 written without dividing by the variance
    far = x * x + gamma * gamma >= 2.0 * variance * _SERIES_SWITCH_Z**2
    out = [np.empty(x.shape) for _ in range(4)]
    if far.any():
        for dst, src in zip(out, _derivatives_series(x[far], variance[far], gamma[far])):
            dst[far] = src
    if not far.all():
        near = ~far
        for dst, src in zip(out, _derivatives_faddeeva(x[near], variance[near], gamma[near])):
            dst[near] = src
    return VoigtDerivatives(*out)


def voigt_fwhm(variance: float, gamma: float) -> float:
    """Voigt FWHM, Olivero & Longbothum (1977) approximation.

    ``f_V = 0.5346 f_L + sqrt(0.2166 f_L**2 + f_G**2)`` with
    ``f_G**2 = 8 ln2 * variance`` and ``f_L = 2 gamma``. The same
    expression as ``toyomacro.lineshape.Voigt.fwhm``: the simpler of the
    two fits in that paper, stated there as accurate to about 0.02 % (the
    0.01 % belongs to the other one). Against the exact half-maximum width
    of the profile its error is at most 2.37e-4 (measured over f_L/f_G
    from 1e-4 to 1e4 and pinned by the tests; Wang et al. 2022 report the
    same maximum), exact for a Gaussian and 3e-6 for a Lorentzian. It is
    used here as the
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
        values = (self.amplitude, self.center, self.sigma, self.gamma)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"peak parameters must be finite, got {values}")
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
      derivatives are those of the pseudo-Voigt at that point. Needs a
      total FWHM above 1e-10 in whatever unit the axis is in: the mapping
      in ``lineshape.PseudoVoigt`` has an absolute threshold there.

    'sigma_gamma', 'var_gamma', 'fwhm_shape' and 'fixed_instrument' with
    a floor are coordinates on one statistical model, related by ``I_phi
    = T^T I_theta T``; ``T`` is singular only for 'sigma_gamma', and only
    at sigma = 0. 'fixed_instrument' without a floor is a sub-model of it,
    one parameter per peak fewer.

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
            if not w > 0.0:
                raise ValueError(
                    "pvoigt needs a total FWHM above 1e-10 (absolute threshold in "
                    "PseudoVoigt.eta_from_voigt_params)"
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
    fisher = poisson_fisher_matrix(jac_unit, mean, exposure)

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
# Assessment
# ---------------------------------------------------------------------------


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
        sd_variance: Bound on the standard deviation of the excess. With
            a ``variance_floor_sd``, the floor's own uncertainty is added
            in quadrature: the floor is measured elsewhere (a Fermi edge,
            say) and the two estimates are taken as independent
        variance_relative_sd: ``sd_variance / variance_excess`` -- the
            precision of the Gaussian component *relative to itself*.
            Grows without limit as the component vanishes however good
            ``worst_relative_sd`` looks; ``inf`` on the boundary.
        near_boundary: The Gaussian variance's flag: ``variance_excess <
            boundary_sd * sd_variance``. Independent of ``status``.
        worst_relative_sd: Square root of the largest eigenvalue of the
            2x2 covariance bound after dividing variance by
            FWHM**2/(8 ln2) and gamma by FWHM: the relative bound along
            the least determined direction of the pair. Never smaller
            than either parameter's own ``relative_sd``.
        worst_direction: That direction, in the same scaled coordinates
        variance_floor_sd: The floor's own standard deviation, as given
        status: 'rank_deficient' if either width is; else
            'weakly_identified' if ``worst_relative_sd`` exceeds the
            threshold; else 'identified'
        gamma_relative_sd: ``sd(gamma) / gamma``, the precision of the
            Lorentzian width relative to itself; ``inf`` at gamma = 0.
            gamma is judged against the FWHM elsewhere, a scale that
            stays finite as the Lorentzian component vanishes.
        near_boundary_gamma: gamma's flag, ``gamma < boundary_sd *
            sd(gamma)``; gamma = 0 is a boundary just as the variance
            floor is. True when gamma is rank deficient. Independent of
            ``status``.
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
    variance_floor_sd: float = 0.0
    gamma_relative_sd: float = math.nan
    near_boundary_gamma: bool = False


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
        variance_floor_sd: The floor's own standard deviation, as given
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
    variance_floor_sd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python summary (no arrays), for a table row or ``json.dumps``.

        Bounds are ``inf`` at a boundary or when rank deficient, and
        ``json.dumps`` writes those as ``Infinity``, which strict JSON
        parsers reject.
        """
        return {
            "null_space_dim": self.null_space_dim,
            "condition_number": self.condition_number,
            "variance_floor": self.variance_floor,
            "variance_floor_sd": self.variance_floor_sd,
            "weak_relative_sd": self.thresholds.weak_relative_sd,
            "boundary_sd": self.thresholds.boundary_sd,
            "parameters": [vars(a).copy() for a in self.parameters],
            "widths": [vars(w).copy() for w in self.widths],
        }


def _covariance_bound(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    """``covariance_bound`` with the projection tolerance calibrated for Voigt designs in ``crlb``."""
    return covariance_bound(matrix, _NULL_PROJECTION_TOL)


def assess_identifiability(
    energy: np.ndarray,
    peaks: list[VoigtPeak],
    background: Background | None = None,
    *,
    exposure: float = 1.0,
    variance_floor: float = 0.0,
    variance_floor_sd: float = 0.0,
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
      Same decision as ``crlb.compute_multipeak_fisher``: its ``n * eps``
      rank rule, re-implemented here, and its projection tolerance,
      imported. The bound is infinite.
    - 'weakly_identified': the bound on the standard deviation exceeds
      ``thresholds.weak_relative_sd`` times the reference scale.
    - 'identified': neither.

    and two independent flags per peak: 'near_boundary', for a Gaussian
    variance within ``thresholds.boundary_sd`` bounds of its lower limit
    ``variance_floor`` (0, or a calibrated instrument variance when the
    quantity of interest is the broadening in excess of it), and
    'near_boundary_gamma', for gamma within as many bounds of 0.

    What the labels do not mean:

    - 'identified' on the FWHM scale is **not** detection of a small
      Gaussian component. The variance is judged against
      FWHM**2/(8 ln2), a scale that stays finite as the component
      vanishes, so a peak with ``sigma/gamma = 0.03`` can be
      'identified' while ``variance_relative_sd`` is above 1. Read that
      field, and ``near_boundary``, for the component itself.
    - The same holds at the other end. ``gamma = 0`` is a boundary too,
      and gamma is judged against the FWHM: a nearly Gaussian peak can
      have gamma 'identified' while sd(gamma)/gamma is of order one (0.58
      at sigma/gamma = 100 for the default ``ScanGrid`` peak on +-5 FWHM
      with an estimated flat background, where sd(gamma)/FWHM is 2e-3).
      Read ``gamma_relative_sd`` and ``near_boundary_gamma`` for it.
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
        variance_floor_sd: Standard deviation of that floor (eV**2), when
            it is itself an estimate -- from a Fermi-edge fit, say. It is
            added in quadrature to the bound on the excess, which assumes
            the two are independent (they are, when they come from
            different spectra). The Fisher matrix does not change; what
            changes is ``sd_variance``, ``variance_relative_sd``,
            ``near_boundary`` and the width block's worst direction. The
            default 0 reproduces the bounds exactly.
        thresholds: Defaults to ``IdentifiabilityThresholds()``

    Returns:
        IdentifiabilityReport
    """
    thresholds = thresholds or IdentifiabilityThresholds()
    if variance_floor < 0.0 or any(p.variance < variance_floor for p in peaks):
        raise ValueError(
            f"need 0 <= variance_floor <= sigma**2 for every peak, got {variance_floor}"
        )
    if not (variance_floor_sd >= 0.0 and math.isfinite(variance_floor_sd)):
        raise ValueError(f"variance_floor_sd must be finite and >= 0, got {variance_floor_sd}")
    floor_variance = variance_floor_sd * variance_floor_sd
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
            sd_v = np.inf if unbounded[iv] else math.sqrt(
                max(covariance[iv, iv], 0.0) + floor_variance)
            worst, direction, status = np.inf, (math.nan, math.nan), "rank_deficient"
        else:
            sd_v = math.sqrt(max(covariance[iv, iv], 0.0) + floor_variance)
            scales = np.array([fwhm[k] ** 2 / _EIGHT_LN2, fwhm[k]])
            block = covariance[np.ix_([iv, ig], [iv, ig])].copy()
            block[0, 0] += floor_variance  # the floor's own uncertainty, independent
            block = block / np.outer(scales, scales)
            eigenvalues, eigenvectors = np.linalg.eigh(block)
            worst = math.sqrt(max(eigenvalues[-1], 0.0))
            direction = (float(eigenvectors[0, -1]), float(eigenvectors[1, -1]))
            status = "weakly_identified" if worst > thresholds.weak_relative_sd else "identified"
        sd_g = np.inf if unbounded[ig] else math.sqrt(max(covariance[ig, ig], 0.0))
        widths.append(WidthAssessment(
            peak=k, fwhm=fwhm[k], variance_excess=excess, sd_variance=float(sd_v),
            variance_relative_sd=float(sd_v / excess) if excess > 0.0 else np.inf,
            near_boundary=bool(excess < thresholds.boundary_sd * sd_v),
            worst_relative_sd=float(worst), worst_direction=direction, status=status,
            variance_floor_sd=float(variance_floor_sd),
            gamma_relative_sd=float(sd_g / p.gamma) if p.gamma > 0.0 else np.inf,
            near_boundary_gamma=bool(p.gamma < thresholds.boundary_sd * sd_g),
        ))

    return IdentifiabilityReport(
        parameters=tuple(parameters), widths=tuple(widths), null_space_dim=null_dim,
        condition_number=condition, thresholds=thresholds,
        variance_floor=float(variance_floor), variance_floor_sd=float(variance_floor_sd),
        fisher=fisher,
    )


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeReport:
    """Result of ``identifiability_modes``.

    Attributes:
        modes: Up to ``k`` non-null modes of the peak parameters, softest
            first (``Mode``)
        null_modes: A basis of the directions with an infinite bound, in
            the same scaled coordinates. Only their span means anything.
        null_space_dim: As ``IdentifiabilityReport.null_space_dim``
        nuisance_null_dim: Null directions with no component on the peak
            parameters (by the tolerance behind the labels); the other
            ``null_space_dim - nuisance_null_dim`` are the null modes
        scales: Peak parameter name -> its reference scale
        thresholds: The thresholds used
        exposure: As given
        variance_floor: As given; it moves a boundary, not the matrix
        fisher: The underlying 'var_gamma' ``PoissonFisherResult``
    """

    modes: tuple[Mode, ...]
    null_modes: tuple[Mode, ...]
    null_space_dim: int
    nuisance_null_dim: int
    scales: dict[str, float]
    thresholds: IdentifiabilityThresholds
    exposure: float
    variance_floor: float
    fisher: PoissonFisherResult = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python summary for ``json.dumps``: no arrays and no matrix.

        The signatures are left out (they are arrays over the energy
        axis); read them from ``modes``. Infinite bounds are written as
        ``Infinity`` and the bound of a 'not_resolved' mode as ``NaN``,
        which strict JSON parsers reject.
        """
        def entry(mode: Mode) -> dict[str, Any]:
            out = vars(mode).copy()
            del out["signature"]
            return out

        return {
            "null_space_dim": self.null_space_dim,
            "nuisance_null_dim": self.nuisance_null_dim,
            "weak_relative_sd": self.thresholds.weak_relative_sd,
            "exposure": self.exposure,
            "variance_floor": self.variance_floor,
            "scales": dict(self.scales),
            "modes": [entry(m) for m in self.modes],
            "null_modes": [entry(m) for m in self.null_modes],
        }


def identifiability_modes(
    energy: np.ndarray,
    peaks: list[VoigtPeak],
    background: Background | None = None,
    *,
    exposure: float = 1.0,
    k: int = 3,
    variance_floor: float = 0.0,
    thresholds: IdentifiabilityThresholds | None = None,
) -> ModeReport:
    """Which combinations of peak parameters the spectrum determines worst.

    ``assess_identifiability`` bounds each parameter on its own. This
    returns the least determined *directions*: the peak block of the
    inverse Fisher matrix in 'var_gamma' coordinates -- amplitude,
    center, variance and gamma of every peak, with every estimated
    background term profiled out, so the cost of estimating the
    background is included -- divided by the reference scales of
    ``assess_identifiability`` (|A|, FWHM, FWHM**2/(8 ln2), FWHM) and
    diagonalised. Each ``Mode`` gives the relative bound along its
    direction, the direction as named weights, how the background
    follows when the peaks move one SD along it, what that move does to
    the expected spectrum, and how much of each parameter's bound lies
    along it.

    What depends on a convention and what does not:

    - The directions and their order depend on the reference scales;
      other scales would list other directions. The background is not in
      the directions because no scale for it survived checking: one tied
      to the peak height moved the modes when the same background was
      written on another basis.
    - The null space does not depend on them, and neither does the size
      of a signature: a one-SD move along any mode changes the expected
      spectrum by one unit of counting noise, ``whitened_norm = 1`` in
      exact arithmetic. The convention decides which directions are
      listed, not how visible they are. A departure of ``whitened_norm``
      from 1 measures that mode's numerical error: it appears at the stiff
      end of a near-singular design, in every mode of a design singular
      to within rounding, or when the background is itself nearly
      singular. Where rounding leaves no positive bound the mode is
      'not_resolved'.
    - The softest mode is never better determined than any single peak
      parameter with a finite bound, nor than a width block's
      ``worst_relative_sd`` without a floor uncertainty. The converse does
      not hold: a mode can be 'weakly_identified' while no parameter is,
      because the threshold then applies to a combination of fractions of
      different scales.
    - Null directions are reported apart, in ``null_modes``: as many as
      the rank of the null space's peak-parameter rows, by the tolerance
      that marks a single parameter 'rank_deficient', so there is a null
      mode exactly when some peak parameter carries that label. The soft modes are taken on their
      complement, so they are combinations the data determine: with two
      coincident peaks, A0 + A1 lies in their span while every single
      peak parameter is 'rank_deficient'.

    This is a diagnostic, not a set of weights: nothing here is meant to
    be fed to a downstream fit as a covariance, and nothing here is a
    confidence interval. Like the labels it is computed from a model at
    stated parameters, under the conditions of ``crlb``'s module
    docstring, and near a boundary the Fisher numbers are not the spread
    of a constrained estimator (Self & Liang 1987). ``variance_floor_sd``
    is not taken: the floor's own uncertainty is not information carried
    by the spectrum, and adding it would break ``whitened_norm = 1``.

    Args:
        energy: Energy axis (eV)
        peaks: One or more peaks
        background: None, or a ``Background``
        exposure: Multiplier on the whole mean; relative bounds scale as
            ``1/sqrt(exposure)`` and the directions do not move
        k: Number of non-null modes to report, at most
        variance_floor: Lower limit of every peak's Gaussian variance,
            as in ``assess_identifiability``. It moves a boundary, not
            the matrix, so it changes no mode.
        thresholds: Defaults to ``IdentifiabilityThresholds()``; only
            ``weak_relative_sd`` is used

    Returns:
        ModeReport
    """
    thresholds = thresholds or IdentifiabilityThresholds()
    if variance_floor < 0.0 or any(p.variance < variance_floor for p in peaks):
        raise ValueError(
            f"need 0 <= variance_floor <= sigma**2 for every peak, got {variance_floor}"
        )
    fisher = poisson_fisher(energy, peaks, background, exposure=exposure)
    target = [i for i, role in enumerate(fisher.param_roles) if role != "background"]
    scales = []
    for p in peaks:
        fwhm = voigt_fwhm(p.variance, p.gamma)
        scales += [abs(p.amplitude), fwhm, fwhm * fwhm / _EIGHT_LN2, fwhm]
    decomposition = mode_decomposition(
        fisher.fisher, fisher.jacobian, fisher.expected_counts, fisher.param_names,
        target, scales, k=k, weak_relative_sd=thresholds.weak_relative_sd,
        projection_tol=_NULL_PROJECTION_TOL,
    )
    return ModeReport(
        modes=decomposition.modes, null_modes=decomposition.null_modes,
        null_space_dim=decomposition.null_space_dim,
        nuisance_null_dim=decomposition.nuisance_null_dim,
        scales=dict(zip(decomposition.target_names, decomposition.scales)),
        thresholds=thresholds, exposure=float(exposure),
        variance_floor=float(variance_floor), fisher=fisher,
    )


# ---------------------------------------------------------------------------
# At a fitted estimate
# ---------------------------------------------------------------------------

# The readers' IntensitySemantics vocabulary (toyomacro.io.readers.base_reader),
# repeated here so that this module does not import the I/O layer; a test
# keeps the two equal. Only raw counts carry the scale the bound needs.
INTENSITY_SEMANTICS: tuple[str, ...] = (
    "raw_counts", "count_rate", "corrected_intensity", "arbitrary", "unknown",
)

_ESTIMATE_CAVEAT = (
    "a plug-in at a noisy estimate, not a confidence interval; if the fit's "
    "estimator is unbiased and the model -- including the fit's ties and "
    "independent Poisson channels -- is correct, its variance is at least the "
    "bound at the true parameters, of which this is a plug-in estimate"
)


@dataclass(frozen=True)
class EstimateReport:
    """Result of ``identifiability_at_estimate`` and ``identifiability_from_exact_k``.

    Attributes:
        withheld: None, or why no numbers are reported (the intensity is
            not raw counts, the background cannot be rebuilt, ...). When
            set, every other result field is empty.
        model: Which model the bounds are for: every peak parameter free,
            or the sub-model a fit actually used
        intensity: As declared
        parameters: One ``ParameterAssessment`` per parameter of that
            model; background coefficients are 'not_assessed' unless rank
            deficient. For the ``exact_k`` sub-model the same fields mean
            slightly different things: ``peak`` is the component index (a
            doublet's two lines share it) and is None for ``var_shared``
            as well as for the background; ``amp_j`` is the main line's
            area, the partner's being ``amp_j / branch_ratio``.
        near_boundary: Width parameter name -> whether it lies within
            ``boundary_sd`` bounds of its lower limit (variance: the
            floor; gamma: 0). For the ``exact_k`` sub-model also
            'var_shared_fit_bounds': whether the shared variance lies
            within as many bounds of either of the fit's own
            ``sigma_bounds`` (squared), where the constrained fit stops
        modes: The ``ModeReport`` of that model
        assessment: The full ``IdentifiabilityReport`` when every peak
            parameter is free; None for a sub-model, whose layout it does
            not fit
        caveat: What the numbers are, in one sentence
    """

    withheld: str | None
    model: str
    intensity: str
    parameters: tuple[ParameterAssessment, ...] = ()
    near_boundary: dict[str, bool] = field(default_factory=dict)
    modes: ModeReport | None = None
    assessment: IdentifiabilityReport | None = field(default=None, repr=False)
    caveat: str = _ESTIMATE_CAVEAT

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python summary for ``json.dumps`` (no arrays, no matrix)."""
        return {
            "withheld": self.withheld,
            "model": self.model,
            "intensity": self.intensity,
            "caveat": self.caveat,
            "parameters": [vars(a).copy() for a in self.parameters],
            "near_boundary": dict(self.near_boundary),
            "modes": None if self.modes is None else self.modes.to_dict(),
        }


def _check_intensity(intensity: str) -> str | None:
    if intensity not in INTENSITY_SEMANTICS:
        raise ValueError(
            f"intensity must be one of {INTENSITY_SEMANTICS}, got {intensity!r}"
        )
    if intensity == "raw_counts":
        return None
    return (
        f"intensity is '{intensity}': the bound needs amplitude * V(E) in counts "
        "per channel as recorded, and nothing here can recover that scale"
    )


def identifiability_at_estimate(
    energy: np.ndarray,
    peaks: list[VoigtPeak],
    background: Background | None = None,
    *,
    intensity: str,
    k: int = 3,
    variance_floor: float = 0.0,
    thresholds: IdentifiabilityThresholds | None = None,
) -> EstimateReport:
    """The labels and the modes evaluated at a fitted estimate.

    ``peaks`` and ``background`` are a fit's result. The report is
    ``assess_identifiability`` and ``identifiability_modes`` at those
    values: a plug-in at a noisy estimate, not a confidence interval.
    The fitters in this package that produce such estimates are least
    squares, not Poisson maximum likelihood. If the fit's estimator is
    unbiased and the model --
    including the fit's ties and independent Poisson channels -- is
    correct, its variance is at least the bound at the true parameters;
    this is a plug-in estimate of that bound.

    Every peak parameter is evaluated free. For a fit that tied or fixed
    parameters (a shared sigma, a fixed gamma, a doublet) that answers
    another question -- it can name a direction the fit did not have;
    ``identifiability_from_exact_k`` evaluates the sub-model ``exact_k``
    used.

    The intensity is declared, never guessed. The bound needs ``amplitude
    * V(E)`` to be counts per channel as recorded, and no fit result in
    the package records what its intensities are. ``intensity`` takes the
    readers' vocabulary (``INTENSITY_SEMANTICS``); only 'raw_counts'
    proceeds, anything else returns a report with ``withheld`` set and no
    numbers. With raw counts the exposure is 1, so there is no
    ``exposure`` argument: it would count the acquisition twice.

    Args:
        energy: Energy axis (eV)
        peaks: The fitted peaks
        background: The fitted background, its terms flagged estimated
            as the fit estimated them; or None
        intensity: One of ``INTENSITY_SEMANTICS``; required
        k: Number of modes, as in ``identifiability_modes``
        variance_floor: As in ``assess_identifiability``
        thresholds: Defaults to ``IdentifiabilityThresholds()``

    Raises:
        ValueError: for an ``intensity`` outside the vocabulary, an
            invalid ``variance_floor``, or a mean that is not positive in
            every channel (the inputs are the caller's, as for
            ``assess_identifiability``)
    """
    model = "every peak parameter free"
    withheld = _check_intensity(intensity)
    if withheld is not None:
        return EstimateReport(withheld=withheld, model=model, intensity=intensity)
    assessment = assess_identifiability(
        energy, peaks, background, variance_floor=variance_floor, thresholds=thresholds,
    )
    modes = identifiability_modes(
        energy, peaks, background, k=k, variance_floor=variance_floor, thresholds=thresholds,
    )
    near = {}
    for width in assessment.widths:
        near[f"var_{width.peak}"] = width.near_boundary
        near[f"gamma_{width.peak}"] = width.near_boundary_gamma
    return EstimateReport(
        withheld=None, model=model, intensity=intensity, parameters=assessment.parameters,
        near_boundary=near, modes=modes, assessment=assessment,
    )


def identifiability_from_exact_k(
    report: Any,
    k_structured: int,
    energy: np.ndarray,
    *,
    intensity: str,
    k: int = 3,
    thresholds: IdentifiabilityThresholds | None = None,
) -> EstimateReport:
    """The labels and the modes at an ``exact_k`` candidate, for its own model.

    ``exact_k`` fits K structured components that share one Gaussian
    width, with gamma fixed and, for a doublet, the splitting and
    branching ratio fixed. All of these are affine constraints in
    'var_gamma' coordinates, so the sub-model is evaluated directly: the
    Voigt Jacobian of every line (doublet partners included) times the
    matrix L of the ties, giving parameters ``amp_j`` and ``center_j``
    per component, one ``var_shared`` and the background coefficients.
    The shared variance is scaled by FWHM**2/(8 ln2) of the narrowest
    line that shares it, the smallest candidate scale and so the largest
    relative bound (in ``exact_k`` every line has the same width, so the
    choice does not arise).

    The same caveats as ``identifiability_at_estimate``: a plug-in at a
    noisy estimate, not a confidence interval. ``exact_k`` is a weighted
    least-squares fit, not Poisson maximum likelihood; if it is unbiased
    and its model -- these ties and independent Poisson channels -- is
    correct, its variance is at least the bound at the true parameters,
    of which this is a plug-in estimate. The rows read as described under
    ``EstimateReport.parameters``: ``amp_j`` is the main line's area and
    ``peak`` the component index.

    Withheld, with the reason and no numbers, when:

    - ``intensity`` is not 'raw_counts';
    - the candidate did not converge;
    - the background was a plug-in Shirley or Tougaard (only a summary of
      it is stored, so it cannot be rebuilt);
    - an amplitude is not positive (``exact_k`` allows negative ones);
    - the candidate carries ``sigma_at_bound`` or a
      ``center_*_at_energy_bound`` flag (the estimate is not a
      stationary point);
    - its background coefficients are missing or not finite;
    - the fitted mean is not positive in every channel.

    Args:
        report: An ``exact_k.ExactKReport``
        k_structured: Which candidate K
        energy: The energy axis the fit ran on, in either order (the
            report does not store it). Its length is checked and it must
            contain the fitted centers; another axis of the same length
            that still contains them cannot be told apart.
        intensity: One of ``INTENSITY_SEMANTICS``; required
        k: Number of modes
        thresholds: Defaults to ``IdentifiabilityThresholds()``

    Raises:
        ValueError: for an ``intensity`` outside the vocabulary, a K the
            report has no candidate for, an energy axis of another length
            or one that does not contain the fitted centers, or a
            non-positive branching ratio
    """
    from .multipeak_solver import build_bg_design

    thresholds = thresholds or IdentifiabilityThresholds()
    energy = np.asarray(energy, dtype=np.float64)
    matches = [c for c in report.candidates if c.k_structured == k_structured]
    if not matches:
        raise ValueError(f"the report has no candidate with K = {k_structured}")
    cand = matches[0]
    if cand.score.n_points != energy.size:
        raise ValueError(
            f"energy has {energy.size} points, the fit ran on {cand.score.n_points}"
        )
    components = list(cand.model.components)
    doublets = any(c.is_doublet for c in components)
    model = ("exact_k sub-model: shared sigma, fixed gamma"
             + (", fixed doublet splitting and ratio" if doublets else ""))

    def withheld(reason: str) -> EstimateReport:
        return EstimateReport(withheld=reason, model=model, intensity=intensity)

    reason = _check_intensity(intensity)
    if reason is not None:
        return withheld(reason)
    if not cand.score.success:
        return withheld(f"candidate K = {k_structured} did not converge")
    mode = report.background.get("mode")
    if mode is None:
        return withheld("the report does not record how the background was treated")
    if mode not in ("polynomial", "none"):
        return withheld(f"background '{mode}' was a plug-in; only its summary is stored")
    amplitudes = np.asarray(cand.amplitudes, dtype=np.float64)
    if not np.all(amplitudes > 0.0):
        return withheld("an amplitude is not positive")
    flagged = [f for f in cand.boundary_flags
               if f == "sigma_at_bound" or f.endswith("_at_energy_bound")]
    if flagged:
        return withheld(f"the estimate is at a bound: {', '.join(flagged)}")

    if any(c.is_doublet and not c.branch_ratio > 0.0 for c in components):
        raise ValueError("a doublet's branch_ratio must be positive")
    if any(not energy.min() <= c.center <= energy.max() for c in components):
        raise ValueError("the energy axis does not contain the fitted centers: "
                         "pass the axis the fit ran on")

    # Every line as a VoigtPeak, and L: (amp_j, center_j per component,
    # var_shared, background) -> (amp, center, var, gamma per line, background)
    lines: list[VoigtPeak] = []
    owner: list[tuple[int, float]] = []  # (component, amplitude factor) per line
    for j, (comp, amp) in enumerate(zip(components, amplitudes)):
        lines.append(VoigtPeak(float(amp), comp.center, comp.sigma, comp.gamma))
        owner.append((j, 1.0))
        if comp.is_doublet:
            lines.append(VoigtPeak(float(amp) / comp.branch_ratio, comp.center + comp.so_split,
                                   comp.sigma, comp.gamma))
            owner.append((j, 1.0 / comp.branch_ratio))
    degree = cand.model.bg_degree
    background = None
    if degree is not None:
        coefficients = np.asarray(cand.bg_coefficients, dtype=np.float64)
        if coefficients.size != degree + 1 or not np.all(np.isfinite(coefficients)):
            return withheld("the background coefficients are missing or not finite")
        background = Background(
            basis=build_bg_design(energy, degree).astype(np.float64).T,
            coefficients=coefficients,
            names=tuple(f"bg_{i}" for i in range(degree + 1)),
            estimated=(True,) * (degree + 1),
        )
    mean = sum(p.amplitude * voigt_derivatives(energy, p.center, p.variance, p.gamma).value
               for p in lines)
    if background is not None:
        mean = mean + background.counts()
    if not np.all(mean > 0.0):
        return withheld("the fitted mean is not positive in every channel")

    full = poisson_fisher(energy, lines, background)
    n_comp, n_bg = len(components), 0 if background is None else degree + 1
    names = ([f"amp_{j}" for j in range(n_comp)] + [f"center_{j}" for j in range(n_comp)]
             + ["var_shared"] + ([] if background is None else list(background.names)))
    roles = (["amplitude"] * n_comp + ["center"] * n_comp + ["width"]
             + ["background"] * n_bg)
    ties = np.zeros((len(full.param_names), len(names)))
    for line, (j, factor) in enumerate(owner):
        ties[4 * line, j] = factor                # amplitude
        ties[4 * line + 1, n_comp + j] = 1.0      # center: a doublet shifts rigidly
        ties[4 * line + 2, 2 * n_comp] = 1.0      # one shared variance; gamma fixed
    for b in range(n_bg):
        ties[4 * len(lines) + b, 2 * n_comp + 1 + b] = 1.0
    jacobian = full.jacobian @ ties
    fisher = ties.T @ full.fisher @ ties
    values = np.concatenate([amplitudes, [c.center for c in components],
                             [components[0].sigma ** 2],
                             [] if background is None else background.coefficients])

    fwhm = [voigt_fwhm(p.variance, p.gamma) for p in lines]
    narrowest = min(fwhm)
    scales = ([abs(a) for a in amplitudes]
              + [voigt_fwhm(c.sigma ** 2, c.gamma) for c in components]
              + [narrowest * narrowest / _EIGHT_LN2])
    result = PoissonFisherResult(
        fisher=fisher, param_names=tuple(names), param_roles=tuple(roles),
        param_values=values, width_index=(2 * n_comp,), jacobian=jacobian,
        expected_counts=full.expected_counts, parameterization="var_gamma",
        exposure=1.0,
        config={
            "model": model,
            "peaks": [(p.amplitude, p.center, p.sigma, p.gamma) for p in lines],
            "background_terms": () if background is None else background.names,
            "n_energy": int(energy.size),
            "energy_range": (float(energy[0]), float(energy[-1])),
        },
    )
    target = list(range(2 * n_comp + 1))
    decomposition = mode_decomposition(
        fisher, jacobian, full.expected_counts, names, target, scales, k=k,
        weak_relative_sd=thresholds.weak_relative_sd, projection_tol=_NULL_PROJECTION_TOL,
    )
    modes = ModeReport(
        modes=decomposition.modes, null_modes=decomposition.null_modes,
        null_space_dim=decomposition.null_space_dim,
        nuisance_null_dim=decomposition.nuisance_null_dim,
        scales=dict(zip(decomposition.target_names, decomposition.scales)),
        thresholds=thresholds, exposure=1.0, variance_floor=0.0, fisher=result,
    )

    covariance, unbounded, _, _ = _covariance_bound(fisher)
    width_info = effective_information(fisher, [2 * n_comp])
    cond_cov, cond_unbounded, _, _ = _covariance_bound(width_info.conditional)
    parameters = []
    for i, (name, role) in enumerate(zip(names, roles)):
        sd = np.inf if unbounded[i] else math.sqrt(max(covariance[i, i], 0.0))
        scale = scales[i] if i < len(scales) else None
        relative = None if scale is None else sd / scale
        if unbounded[i]:
            status = "rank_deficient"
        elif relative is None:
            status = "not_assessed"
        else:
            status = "weakly_identified" if relative > thresholds.weak_relative_sd else "identified"
        conditional = None
        if role == "width":
            conditional = np.inf if cond_unbounded[0] else math.sqrt(max(cond_cov[0, 0], 0.0))
        peak = i if i < n_comp else (i - n_comp if i < 2 * n_comp else None)
        parameters.append(ParameterAssessment(
            name=name, role=role, peak=peak, value=float(values[i]), sd=float(sd),
            sd_conditional=conditional, reference_scale=scale, relative_sd=relative,
            status=status,
        ))
    var_row = parameters[2 * n_comp]
    reach = thresholds.boundary_sd * var_row.sd
    near = {"var_shared": bool(var_row.value < reach)}
    bounds = report.config.get("sigma_bounds")
    if bounds is not None:
        near["var_shared_fit_bounds"] = bool(
            any(abs(var_row.value - float(b) ** 2) < reach for b in bounds))
    return EstimateReport(
        withheld=None, model=model, intensity=intensity, parameters=tuple(parameters),
        near_boundary=near, modes=modes, assessment=None,
    )


# ---------------------------------------------------------------------------
# The profile along a mode (P3)
# ---------------------------------------------------------------------------

# Registered in design record §9.4; not to be retuned after the registered
# profiles have been seen.
_PROFILE_TOL = 1e-6          # KKT residual, bound proximity, amplitude at zero, agreement
_PROFILE_BISECT = 1e-6       # crossing bracket in t
_PROFILE_MAX_ITER = 60       # route A
_PROFILE_SLSQP = {"ftol": 1e-12, "maxiter": 1000}
PROFILE_GRID: tuple[float, ...] = tuple(k / 10.0 for k in range(-30, 31))


@dataclass(frozen=True)
class ProfilePoint:
    """One grid point of ``mode_profile``.

    Attributes:
        t: Displacement along the mode, in units of its SD bound
        outcome: 'ok', 'not_obtained', or the stop's event
            ('amplitude_at_zero', 'corner') at and beyond a stop
        delta: The lower successful deviance of the two routes; None
            unless ``outcome`` is 'ok'
        route: 'A' or 'B', the route ``delta`` came from
        delta_a, delta_b: Each route's deviance: its best successful
            result, or for a route that failed the deviance of its
            lowest-deviance candidate; None if it found no candidate or
            the point was not solved (beyond a stop found outward). A
            point between a stop found in the inward pass and the
            outward one keeps what was solved there.
        success_a, success_b: Whether each route succeeded
        kkt_a, kkt_b: Each route's recorded KKT residual, belonging to the
            result whose deviance is recorded
        flagged: A solved point where the routes disagree by more than
            1e-6 (relative above 1), or either failed -- a stop included
        active: Bounds active at the summary solution ('variance', 'gamma')
        background_negative: The re-estimated background curve is
            negative in some channel
        parameters: The summary solution, name -> value
    """

    t: float
    outcome: str
    delta: float | None
    route: str | None
    delta_a: float | None
    delta_b: float | None
    success_a: bool
    success_b: bool
    kkt_a: float | None
    kkt_b: float | None
    flagged: bool
    active: tuple[str, ...]
    background_negative: bool
    parameters: dict[str, float] | None


@dataclass(frozen=True)
class ProfileSide:
    """What one side (t > 0 or t < 0) of ``mode_profile`` comes to.

    Attributes:
        crossing: |t| where the profile first reaches 1, a float; else
            'beyond the grid', the stop's event, or 'not obtained'
        delta_1, delta_2: The profile at |t| = 1 and 2: a float, the
            stop's event, 'not obtained', or 'not on the grid'
        activations: Bound -> (t, delta) at the first grid point with that
            bound active
        stop: None, or {'event', 't', 'delta_before' (a float, or 'not
            obtained'), 'background_only_deviance'}. 'amplitude_at_zero'
            means the amplitude reached 1e-6 of its stated value or went
            below it, a sign change included; where a solution shows both
            events, amplitude at zero is the one reported.
        background_negative: Grid points with a negative background curve
    """

    crossing: float | str
    delta_1: float | str
    delta_2: float | str
    activations: dict[str, tuple[float, float | None]]
    stop: dict[str, Any] | None
    background_negative: tuple[float, ...]


@dataclass(frozen=True)
class ModeProfile:
    """Result of ``mode_profile``.

    Attributes:
        mode: The ``Mode`` profiled
        points: One ``ProfilePoint`` per grid value
        plus, minus: The two sides
        exposure, variance_floor: As given
        caveat: What the numbers are
    """

    mode: Mode
    points: tuple[ProfilePoint, ...]
    plus: ProfileSide
    minus: ProfileSide
    exposure: float
    variance_floor: float
    caveat: str = (
        "the minimum Poisson deviance along a mode on expected counts (no noise), "
        "against the Fisher prediction t**2; a property of the model, not the "
        "coverage of any interval; no boundary correction"
    )

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python summary for ``json.dumps`` (the mode without its signature)."""
        mode = vars(self.mode).copy()
        del mode["signature"]
        return {
            "mode": mode,
            "points": [vars(p).copy() for p in self.points],
            "plus": vars(self.plus).copy(),
            "minus": vars(self.minus).copy(),
            "exposure": self.exposure,
            "variance_floor": self.variance_floor,
            "caveat": self.caveat,
        }


class _Profile:
    """The constrained problem at one mode: coordinates, deviance, the KKT
    test and the two routes of design record §9.4."""

    _BOUNDS = ("variance", "gamma")

    def __init__(self, energy, peak, background, exposure, variance_floor, mode_index):
        report = identifiability_modes(energy, [peak], background, exposure=exposure, k=4,
                                       variance_floor=variance_floor)
        if not 0 <= mode_index < len(report.modes):
            raise ValueError(
                f"mode must index one of the {len(report.modes)} non-null modes; a null "
                "mode has no finite SD in which to measure t"
            )
        self.mode = report.modes[mode_index]
        if self.mode.status == "not_resolved":
            raise ValueError("the mode has no resolved bound to measure t in")
        names = ("amp_0", "center_0", "var_0", "gamma_0")
        vec = lambda m: np.array([m.weights[n] for n in names])  # noqa: E731
        others = [vec(m) for m in report.modes if m is not self.mode] + [
            vec(m) for m in report.null_modes]
        self.w = vec(self.mode)
        self.others = np.array(others).T                      # (4, 3)
        self.scales = np.array([report.scales[n] for n in names])
        self.sd_w = self.mode.relative_sd
        self.energy = np.asarray(energy, dtype=np.float64)
        self.exposure = float(exposure)
        self.floor = float(variance_floor)
        self.peak0 = np.array([peak.amplitude, peak.center, peak.variance, peak.gamma])
        if background is None:
            self.basis = np.zeros((0, self.energy.size))
            self.b0 = np.zeros(0)
            self.known = np.zeros(self.energy.size)
            self.bg_names: tuple[str, ...] = ()
        else:
            est = np.array(background.estimated, dtype=bool)
            self.basis = background.basis[est]
            self.b0 = background.coefficients[est]
            self.known = background.coefficients[~est] @ background.basis[~est]
            self.bg_names = tuple(n for n, e in zip(background.names, est) if e)
        self.y = report.fisher.expected_counts
        self.n_z = 3 + self.b0.size
        # theta_peak = peak0 + S (t sd w + others z_p) ; b = z_b
        self.dpeak_dz = self.scales[:, None] * self.others   # (4, 3)
        self.bound_index = {"variance": 2, "gamma": 3}
        self.bound_value = {"variance": self.floor, "gamma": 0.0}
        self.amp_floor = _PROFILE_TOL * abs(peak.amplitude)
        self.names = names + self.bg_names
        # rounding below a bound, after a projection onto it, counts as on it
        self.slack = {b: 1e-12 * self.scales[i] for b, i in self.bound_index.items()}

    # -- coordinates ------------------------------------------------------
    def theta(self, t, z):
        peak = self.peak0 + self.scales * (t * self.sd_w * self.w) + self.dpeak_dz @ z[:3]
        return peak, z[3:]

    def z_of(self, t, peak, b):
        """The coordinates of (peak, b) once moved onto the constraint plane."""
        x = (np.asarray(peak) - self.peak0) / self.scales - t * self.sd_w * self.w
        return np.concatenate([self.others.T @ x, np.asarray(b, dtype=np.float64)])

    def g_row(self, bound):
        """g(z) = a . z + c for one width bound, as (a, c) at t = 0 offsets."""
        i = self.bound_index[bound]
        a = np.zeros(self.n_z)
        a[:3] = self.dpeak_dz[i]
        return a, i

    def g(self, t, z, bound):
        peak, _ = self.theta(t, z)
        i = self.bound_index[bound]
        return peak[i] - self.bound_value[bound]

    # -- model --------------------------------------------------------------
    def evaluate(self, t, z, *, derivatives=True):
        """(mean, J_z, deviance) or None outside the domain."""
        peak, b = self.theta(t, z)
        amp, center, var, gamma = self.on_bounds(peak)
        if var < 0.0 or gamma < 0.0 or (var == 0.0 and gamma == 0.0):
            return None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            d = voigt_derivatives(self.energy, center, var, gamma)  # extremes give non-finite
        mean = self.exposure * (amp * d.value + b @ self.basis + self.known)
        if not np.all(np.isfinite(mean)) or not np.all(mean > 0.0):
            return None
        dev = _deviance(self.y, mean)
        if not derivatives:
            return mean, None, dev
        j_peak = self.exposure * np.stack(
            [d.value, amp * d.d_center, amp * d.d_variance, amp * d.d_gamma], axis=1)
        jac = np.hstack([j_peak @ self.dpeak_dz, self.exposure * self.basis.T])
        return mean, jac, dev

    def on_bounds(self, peak):
        """Peak values with rounding just below a width bound put on it."""
        peak = np.array(peak, dtype=np.float64)
        for bound, i in self.bound_index.items():
            low = self.bound_value[bound]
            if low - self.slack[bound] <= peak[i] < low:
                peak[i] = low
        return peak

    def feasible(self, t, z, bound):
        return self.g(t, z, bound) >= -self.slack[bound]

    def start(self, t, z, held=()):
        """z on the held bounds, and on any other width bound it violates;
        None if that is the corner or outside the domain."""
        z = self.project(t, z, tuple(held))
        violated = tuple(b for b in self._BOUNDS if b not in held and not self.feasible(t, z, b))
        if violated:
            z = self.project(t, z, tuple(held) + violated)
        if self.evaluate(t, z, derivatives=False) is None:
            return None
        return z

    def deviance(self, t, z):
        out = self.evaluate(t, z, derivatives=False)
        return np.inf if out is None else out[2]

    # -- the KKT test ---------------------------------------------------------
    def active(self, t, z):
        """Bounds the solution lies within 1e-6 of, in units of the reference
        scale, on either side: a larger violation is not active but infeasible."""
        out = []
        for bound in self._BOUNDS:
            i = self.bound_index[bound]
            if abs(self.g(t, z, bound)) < _PROFILE_TOL * self.scales[i]:
                out.append(bound)
        return tuple(out)

    def project(self, t, z, bounds):
        """Least-squares move of z onto g = 0 for the given bounds."""
        if not bounds:
            return z
        rows = np.array([self.g_row(b)[0] for b in bounds])
        resid = np.array([self.g(t, z, b) for b in bounds])
        return z - np.linalg.lstsq(rows, resid, rcond=None)[0]

    def kkt(self, t, z):
        """(passed, residual, active, projected z, deviance, solution)."""
        act = self.active(t, z)
        zp = self.project(t, z, act)
        out = self.evaluate(t, zp)
        if out is None:
            return False, np.inf, act, zp, np.inf
        mean, jac, dev = out
        info = (jac / mean[:, None]).T @ jac
        half_grad = jac.T @ (1.0 - self.y / mean)
        rows = np.array([self.g_row(b)[0] for b in act]).reshape(len(act), self.n_z)
        if act:
            nu = np.linalg.lstsq(rows.T, half_grad, rcond=None)[0]
        else:
            nu = np.zeros(0)
        resid = half_grad - rows.T @ nu
        diag = np.diag(info)
        if not (np.all(np.isfinite(info)) and np.all(np.isfinite(resid)) and np.all(diag > 0.0)):
            return False, np.inf, act, zp, dev
        res = float(np.max(np.abs(resid) / np.sqrt(diag)))
        # singular: the n*eps rule on the unit-diagonal I_z, as for the labels
        root = np.sqrt(diag)
        unit = info / root[:, None] / root[None, :]
        try:
            singular = _null_directions(np.linalg.eigvalsh(0.5 * (unit + unit.T))).any()
            info_inv = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            singular = True
        if singular:
            return False, res, act, zp, dev
        normalised = [float(n * math.sqrt(max(r @ info_inv @ r, 0.0))) for n, r in zip(nu, rows)]
        inactive_ok = all(self.feasible(t, zp, b) for b in self._BOUNDS if b not in act)
        passed = inactive_ok and res < _PROFILE_TOL and all(v > -_PROFILE_TOL for v in normalised)
        return passed, res, act, zp, dev

    # -- route A: active sets, Fisher scoring ---------------------------------
    def admissible_sets(self):
        sets = [(), ("variance",), ("gamma",)]
        if self.floor > 0.0:
            sets.append(("variance", "gamma"))
        return sets

    def scoring(self, t, z0, held):
        """Fisher scoring with the bounds in ``held`` as equalities; None if
        the start cannot be put on them inside the domain."""
        z = self.start(t, z0, held)
        if z is None:
            return None
        free = [b for b in self._BOUNDS if b not in held]
        if held:
            rows = np.array([self.g_row(b)[0] for b in held])
            null = np.linalg.svd(rows)[2][len(held):].T
        else:
            null = np.eye(self.n_z)
        out = self.evaluate(t, z)
        if out is None:
            return None
        damping = 1e-3
        for _ in range(_PROFILE_MAX_ITER):
            mean, jac, dev = out
            if self.kkt(t, z)[0]:
                break
            info = null.T @ ((jac / mean[:, None]).T @ jac) @ null
            grad = null.T @ (jac.T @ (1.0 - self.y / mean))
            accepted = False
            while damping < 1e12:
                lhs = info + damping * np.diag(np.diag(info))
                try:
                    step = -np.linalg.solve(lhs, grad)
                except np.linalg.LinAlgError:
                    damping *= 10.0
                    continue
                trial = z + null @ step
                if all(self.feasible(t, trial, b) for b in free):
                    trial_out = self.evaluate(t, trial)
                    if trial_out is not None and trial_out[2] <= dev:
                        z, out = trial, trial_out
                        damping = max(damping / 10.0, 1e-12)
                        accepted = True
                        break
                damping *= 10.0
            if not accepted:
                break
        return z

    # -- route B: SLSQP -------------------------------------------------------
    def slsqp(self, t, z0):
        """SLSQP from z0, in coordinates preconditioned by the Fisher diagonal
        at the start; returns (z, scipy success) or None."""
        from scipy.optimize import minimize

        z0 = self.start(t, z0)
        if z0 is None:
            return None
        mean0, jac0, _ = self.evaluate(t, z0)
        precond = 1.0 / np.sqrt(np.maximum(np.diag((jac0 / mean0[:, None]).T @ jac0), 1e-300))
        floor_mean = 1e-12 * float(self.y.max())

        def unpack(u):
            return u * precond

        def fun(u):
            out = self.evaluate(t, unpack(u))
            if out is None:
                return 1e30, np.zeros_like(u)
            mean, jac, dev = out
            return dev, precond * (2.0 * jac.T @ (1.0 - self.y / mean))

        constraints = []
        for bound in self._BOUNDS:
            a, i = self.g_row(bound)
            constraints.append({
                "type": "ineq",
                "fun": lambda u, b=bound: np.array([self.g(t, unpack(u), b)]),
                "jac": lambda u, a=a: (a * precond)[np.newaxis, :],
            })

        def mean_con(u):
            peak, b = self.theta(t, unpack(u))
            amp, center, var, gamma = self.on_bounds(peak)
            if var < 0.0 or gamma < 0.0 or (var == 0.0 and gamma == 0.0):
                return -np.ones(self.energy.size)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                d = voigt_derivatives(self.energy, center, var, gamma)
            out = self.exposure * (amp * d.value + b @ self.basis + self.known) - floor_mean
            return np.where(np.isfinite(out), out, -1.0)

        def mean_jac(u):
            out = self.evaluate(t, unpack(u))
            if out is None:
                return np.zeros((self.energy.size, u.size))
            return out[1] * precond

        constraints.append({"type": "ineq", "fun": mean_con, "jac": mean_jac})
        res = minimize(fun, z0 / precond, jac=True, method="SLSQP", constraints=constraints,
                       options=dict(_PROFILE_SLSQP))
        return unpack(res.x), bool(res.success)

    # -- one point -------------------------------------------------------------
    def solve(self, t, starts):
        """Both routes from every start. Returns per route the best successful
        (deviance, z, residual) or None and the lowest-deviance candidate, and
        every final iterate as (start index, z), for the stop rule when
        neither route succeeds."""
        result = {}
        iterates = []
        for route in ("A", "B"):
            best = None
            best_any = None
            for index, z0 in enumerate(starts):
                runs = []
                if route == "A":
                    for held in self.admissible_sets():
                        z = self.scoring(t, z0, held)
                        if z is not None:
                            runs.append((z, True))
                else:
                    out = self.slsqp(t, z0)
                    if out is not None:
                        runs.append(out)
                for z, ok in runs:
                    iterates.append((index, z))
                    passed, res, _, zp, dev = self.kkt(t, z)
                    if best_any is None or dev < best_any[0]:
                        best_any = (dev, zp, res)
                    if ok and passed and (best is None or dev < best[0]):
                        best = (dev, zp, res)
            result[route] = (best, best_any)
        return result, iterates

    def event(self, t, z):
        """'amplitude_at_zero', 'corner' or None for a solution."""
        peak, _ = self.theta(t, z)
        if peak[0] <= self.amp_floor:
            return "amplitude_at_zero"
        if self.floor == 0.0 and set(self.active(t, z)) == set(self._BOUNDS):
            return "corner"
        return None

    def background_only_deviance(self):
        """D minimised over the estimated background coefficients with A = 0."""
        b = self.b0.copy()
        basis = self.exposure * self.basis
        known = self.exposure * self.known

        def dev_of(b):
            mean = b @ basis + known
            if not np.all(mean > 0.0):
                return np.inf, mean
            return _deviance(self.y, mean), mean

        dev, mean = dev_of(b)
        if b.size == 0:
            return dev
        for _ in range(200):
            grad = basis @ (1.0 - self.y / mean)
            info = (basis / mean) @ basis.T
            step = -np.linalg.solve(info, grad)
            scale = 1.0
            while scale > 1e-12:
                trial = b + scale * step
                trial_dev, trial_mean = dev_of(trial)
                if trial_dev <= dev:
                    break
                scale *= 0.5
            if scale <= 1e-12 or dev - trial_dev < 1e-12 * max(1.0, dev):
                if trial_dev <= dev:
                    b, dev, mean = trial, trial_dev, trial_mean
                break
            b, dev, mean = trial, trial_dev, trial_mean
        return dev


def _deviance(y: np.ndarray, mean: np.ndarray) -> float:
    """2 sum(mu - y + y ln(y/mu)), as 2 sum y (x - log1p(x)) with x = mu/y - 1,
    which does not cancel near the minimum and so stays at or above 0."""
    x = mean / y - 1.0
    return 2.0 * float(np.sum(y * (x - np.log1p(x))))


def _better(a, b):
    """The lower-deviance of two (deviance, z, residual) results, either None."""
    if a is None:
        return b
    if b is None:
        return a
    return a if a[0] <= b[0] else b


def mode_profile(
    energy: np.ndarray,
    peak: VoigtPeak,
    background: Background | None = None,
    *,
    mode: int = 0,
    t: tuple[float, ...] | list[float] | np.ndarray = PROFILE_GRID,
    exposure: float = 1.0,
    variance_floor: float = 0.0,
) -> ModeProfile:
    """The Poisson deviance along one mode, against the Fisher prediction.

    Design record §9.4, registered before this code. The data are the
    expected counts y = mu(theta_0) at the stated parameters: no noise and
    no seed (the "Asimov" data set of Cowan et al. 2011). Along the mode
    ``w`` of ``identifiability_modes`` with bound ``sqrt(lambda)``, the
    constraint ``w^T S^-1 (theta_peak - theta_0) = t sqrt(lambda)`` is
    imposed and the deviance minimised over everything else -- the other
    peak directions and the background -- with the Gaussian variance held
    at or above ``variance_floor`` and gamma at or above 0. At theta_0 the
    Hessian of the deviance is exactly twice the Fisher matrix, so the
    Fisher prediction is Delta(t) = t**2 and any departure is curvature or
    a boundary. It is a property of the model, not the coverage of any
    interval, and no boundary correction is applied.

    Two routes solve every grid point: A enumerates the admissible sets of
    active width bounds and solves each by Fisher scoring; B is SLSQP.
    Both are judged by the same KKT test; the summaries use the lower
    successful deviance, and a disagreement beyond 1e-6 is flagged, never
    resolved. Where the amplitude reaches zero, or both widths reach zero,
    the side stops and every later point is reported as that event.

    Args:
        energy: Energy axis (eV)
        peak: One ``VoigtPeak`` (a list of one is accepted)
        background: None, or a ``Background``; its estimated terms are
            profiled out with the peak directions
        mode: Index into the non-null modes of ``identifiability_modes``
            (0 the softest)
        t: Strictly increasing grid in units of the mode's SD bound,
            containing 0
        exposure: As in ``identifiability_modes``
        variance_floor: Lower bound of the Gaussian variance

    Returns:
        ModeProfile

    Raises:
        ValueError: for more than one peak, a null or unresolved mode, or
            a grid that is not strictly increasing or lacks 0
    """
    if isinstance(peak, (list, tuple)):
        if len(peak) != 1:
            raise ValueError("mode_profile is for a single peak only")
        peak = peak[0]
    grid = np.asarray(t, dtype=np.float64)
    if (grid.ndim != 1 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0)
            or not np.any(grid == 0.0)):
        raise ValueError("t must be finite, strictly increasing and contain 0")
    problem = _Profile(energy, peak, background, exposure, variance_floor, mode)
    zero = int(np.flatnonzero(grid == 0.0)[0])
    stated = {float(x): problem.z_of(float(x), problem.peak0, problem.b0) for x in grid}

    best: dict[float, dict[str, Any]] = {}      # t -> route -> (dev, z, res) best successful
    best_any: dict[float, dict[str, Any]] = {}  # t -> route -> lowest-deviance candidate
    events: dict[float, str] = {}               # stop events

    def run(x, starts):
        res, iterates = problem.solve(x, starts)
        for route in ("A", "B"):
            best.setdefault(x, {})[route] = _better(best.get(x, {}).get(route), res[route][0])
            best_any.setdefault(x, {})[route] = _better(
                best_any.get(x, {}).get(route), res[route][1])
        return iterates

    def summary(x):
        """(deviance, z, route) of the lower successful route, or None."""
        options = [(v[0], v[1], r) for r, v in best.get(x, {}).items() if v is not None]
        return min(options, key=lambda o: o[0]) if options else None

    def classify(x, iterates, path_starts):
        """'ok', 'not_obtained' or the event that makes x a stop. When
        neither route succeeds, only iterates from the starts that follow the
        path count (§9.4, amendment of 2026-10-23): the stated point, moved
        onto a distant constraint plane, can sit at any amplitude."""
        s = summary(x)
        if s is not None:
            return problem.event(x, s[1]) or "ok"
        found = [problem.event(x, z) for index, z in iterates if index in path_starts]
        for name in ("amplitude_at_zero", "corner"):
            if name in found:
                return name
        return "not_obtained"

    run(0.0, [stated[0.0]])
    sides = {"plus": [float(x) for x in grid[zero + 1:]],
             "minus": [float(x) for x in grid[:zero][::-1]]}
    stops: dict[str, int | None] = {}
    outcome: dict[float, str] = {0.0: classify(0.0, [], ())}
    for side, xs in sides.items():
        stop = None
        last = stated[0.0] if summary(0.0) is None else summary(0.0)[1]
        for i, x in enumerate(xs):
            iterates = run(x, [last, stated[x]])  # the neighbour follows the path
            outcome[x] = classify(x, iterates, (0,))
            if outcome[x] in ("amplitude_at_zero", "corner"):
                stop = i
                events[x] = outcome[x]
                break
            if outcome[x] == "ok":
                last = summary(x)[1]
        # inward from the point before the stop (or the end)
        end = len(xs) if stop is None else stop
        for i in range(end - 1, -1, -1):
            x = xs[i]
            starts = []
            if summary(x) is not None:
                starts.append(summary(x)[1])
            if i + 1 < end and summary(xs[i + 1]) is not None:
                starts.append(summary(xs[i + 1])[1])
            if not starts:
                continue
            run(x, starts)
            s = summary(x)
            if s is not None:
                ev = problem.event(x, s[1])
                if ev is not None:
                    stop, end = i, i
                    events[x] = ev
                    outcome[x] = ev
                else:
                    outcome[x] = "ok"
        stops[side] = stop
        if stop is not None:
            for x in xs[stop + 1:]:
                outcome[x] = events[xs[stop]]

    def point(x):
        out = outcome.get(x, "not_obtained")
        routes = best.get(x, {})
        anyr = best_any.get(x, {})
        solved = x in best

        def record(route):
            """(success, deviance, residual): the best successful result, or
            for a route that failed its lowest-deviance candidate."""
            if routes.get(route) is not None:
                return True, routes[route][0], routes[route][2]
            if anyr.get(route) is not None:
                return False, anyr[route][0], anyr[route][2]
            return False, None, None

        ok_a, da, ka = record("A")
        ok_b, db, kb = record("B")
        s = summary(x) if out == "ok" else None
        if s is not None:
            peak_values, b = problem.theta(x, s[1])
            peak_values = problem.on_bounds(peak_values)
            active = problem.active(x, s[1])
            for bound in active:  # projected onto it: report it there, not at rounding
                peak_values[problem.bound_index[bound]] = problem.bound_value[bound]
            curve = b @ problem.basis + problem.known
            params = dict(zip(problem.names, [float(v) for v in np.concatenate([peak_values, b])]))
            negative = bool(np.any(curve < 0.0))
        else:
            params, active, negative = None, (), False
        flagged = solved and (
            not ok_a or not ok_b or abs(da - db) > _PROFILE_TOL * max(1.0, min(da, db)))
        return ProfilePoint(
            t=x, outcome=out, delta=None if s is None else float(s[0]),
            route=None if s is None else s[2], delta_a=None if da is None else float(da),
            delta_b=None if db is None else float(db), success_a=ok_a, success_b=ok_b,
            kkt_a=None if ka is None else float(ka), kkt_b=None if kb is None else float(kb),
            flagged=bool(flagged), active=active, background_negative=negative,
            parameters=params,
        )

    points = {float(x): point(float(x)) for x in grid}

    def side_summary(side):
        xs = sides[side]
        stop = stops[side]
        before = xs if stop is None else xs[:stop]
        stop_event = None if stop is None else events[xs[stop]]

        def at(target):
            for x in xs:
                if abs(abs(x) - target) < 1e-12:
                    p = points[x]
                    if p.outcome == "ok":
                        return float(p.delta)
                    return p.outcome.replace("_", " ")
            return "not on the grid"

        crossing: float | str
        chain = [0.0] + before
        crossing = "beyond the grid" if stop is None else stop_event.replace("_", " ")
        for lo_x, hi_x in zip(chain, chain[1:]):
            lo_p, hi_p = points[lo_x], points[hi_x]
            if lo_p.outcome != "ok" or hi_p.outcome != "ok":
                crossing = "not obtained"
                break
            if lo_p.delta < 1.0 <= hi_p.delta:
                lo, hi = (lo_x, summary(lo_x)[1]), (hi_x, summary(hi_x)[1])
                crossing = None
                while abs(hi[0] - lo[0]) > _PROFILE_BISECT:
                    mid = 0.5 * (lo[0] + hi[0])
                    res, _ = problem.solve(mid, [lo[1], hi[1]])
                    options = [v for v in (res["A"][0], res["B"][0]) if v is not None]
                    if not options:
                        crossing = "not obtained"
                        break
                    dev, z, _ = min(options, key=lambda o: o[0])
                    if dev < 1.0:
                        lo = (mid, z)
                    else:
                        hi = (mid, z)
                if crossing is None:
                    crossing = float(abs(0.5 * (lo[0] + hi[0])))
                break
        activations = {}
        for x in before:
            p = points[x]
            for bound in p.active:
                activations.setdefault(bound, (x, p.delta))
        stop_info = None
        if stop is not None:
            prev = xs[stop - 1] if stop > 0 else 0.0
            prev_p = points[prev]
            stop_info = {
                "event": stop_event, "t": xs[stop],
                "delta_before": prev_p.delta if prev_p.outcome == "ok" else "not obtained",
                "background_only_deviance": float(problem.background_only_deviance()),
            }
        return ProfileSide(
            crossing=crossing, delta_1=at(1.0), delta_2=at(2.0), activations=activations,
            stop=stop_info,
            background_negative=tuple(x for x in xs if points[x].background_negative),
        )

    return ModeProfile(
        mode=problem.mode, points=tuple(points[float(x)] for x in grid),
        plus=side_summary("plus"), minus=side_summary("minus"),
        exposure=float(exposure), variance_floor=float(variance_floor),
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
        fwhm: Total FWHM in eV. Only sets the unit of the energy axis:
            step and window are in units of it and the peak area scales
            with it, so no result depends on it.
        area: Peak area at unit exposure in counts x FWHM: the peak is
            given ``amplitude = area * fwhm``, i.e. ``area * fwhm * V(E)``
            counts per channel, which does not change with ``fwhm``
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
          ``near_boundary``, ``near_boundary_gamma``, ``status`` (0
          identified, 1 weakly identified, 2 rank deficient): as in
          ``WidthAssessment``
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
    out["near_boundary_gamma"] = np.zeros(shape, dtype=bool)
    out["status"] = np.full(shape, -1, dtype=np.int8)

    fwhm = grid.fwhm
    sigma_scale = fwhm / math.sqrt(_EIGHT_LN2)
    floor = grid.variance_floor * fwhm * fwhm
    for index in np.ndindex(shape[:4]):
        b, s_, r, h = index
        sigma, gamma = _widths_at_fwhm(grid.ratios[r], fwhm)
        separation = grid.separations[s_]
        centers = [0.0] if separation is None else [-0.5 * separation * fwhm, 0.5 * separation * fwhm]
        peaks = [VoigtPeak(grid.area * fwhm, c, sigma, gamma) for c in centers]
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
            out["near_boundary_gamma"][cell] = width.near_boundary_gamma
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

