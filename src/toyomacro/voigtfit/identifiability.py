"""Identifiability diagnostics for Voigt widths (SciPy/NumPy reference).

Numerical foundation of the diagnostic: Voigt derivatives in the
Gaussian *variance* ``v = sigma**2`` that stay accurate down to and
including ``v = 0``, where the engine's Faddeeva-based Jacobian does not.

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
at the lowest ``|z|`` the expansion can take. It cannot go lower: the
expansion is asymptotic, not convergent, and below the switch it fails
fast -- 5e-11 on 6 <= |z| < 7, 2e-2 on 4 <= |z| < 6. Those are numbers
for this window and these shapes, not a bound.

Conventions inherited from the existing engine:

- Peaks are unit-area Voigt profiles; ``sigma`` is the Gaussian standard
  deviation and ``gamma`` the Lorentzian half width at half maximum.

Not exported from ``toyomacro.voigtfit.__init__``; not wired to any CLI.

References:
    J. J. Olivero and R. L. Longbothum, J. Quant. Spectrosc. Radiat.
        Transfer 17, 233-236 (1977), doi:10.1016/0022-4073(77)90161-3
    S. G. Self and K.-Y. Liang, J. Am. Stat. Assoc. 82, 605-610 (1987),
        doi:10.1080/01621459.1987.10478472
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np
from scipy import special as sps

__all__ = [
    "VoigtDerivatives",
    "voigt_derivatives",
    "voigt_fwhm",
]

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
_SERIES_TERMS = 24


def _series_coefficients(n_terms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coefficients (2n-1)!!, (2n+1)!! and (n+1)(2n+1)!! for n = 0..n_terms."""
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


def _derivatives_series(x: np.ndarray, variance: float, gamma: float):
    """Large-|z| expansion; exact at variance = 0."""
    # L(x) = Im[1/u]/pi with u = x - i gamma, and every x-derivative of L
    # is a power of 1/u, so the heat-equation series in the variance
    # becomes a power series in t = variance / u**2.
    u = x - 1j * gamma
    t = variance / (u * u)
    value = (_horner(_SERIES_A, t) / u).imag / math.pi
    q = _horner(_SERIES_B, t) / (u * u)
    d_x = -q.imag / math.pi
    d_gamma = q.real / math.pi
    d_variance = (_horner(_SERIES_C, t) / u**3).imag / math.pi
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
    *definition* of a total-width coordinate, so that a map through it
    is exactly invertible whatever the approximation error.
    """
    f_l = 2.0 * gamma
    return _OL_A * f_l + math.sqrt(_OL_B * f_l * f_l + _EIGHT_LN2 * variance)
