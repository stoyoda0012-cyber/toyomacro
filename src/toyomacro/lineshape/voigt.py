"""Voigt lineshape profile using Faddeeva function."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import wofz

from toyomacro.lineshape.base import BaseLineshape


class Voigt(BaseLineshape):
    """
    Voigt lineshape profile.

    The Voigt profile is the convolution of Gaussian and Lorentzian:
    V(x) = Re[w(z)] / (σ√(2π))

    where:
        - w(z) is the Faddeeva function
        - z = (x - x₀ + iγ) / (σ√2)
        - σ is Gaussian sigma
        - γ is Lorentzian half-width

    This implementation uses scipy.special.wofz for the Faddeeva function.
    """

    name = "voigt"

    # FWHM to sigma conversion for Gaussian component
    FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def evaluate(
        self,
        x: NDArray[np.float64],
        center: float,
        amplitude: float,
        fwhm_g: float = 0.5,
        fwhm_l: float = 0.5,
        **params,
    ) -> NDArray[np.float64]:
        """
        Evaluate Voigt profile using Faddeeva function.

        Args:
            x: Energy axis
            center: Peak center
            amplitude: Peak area
            fwhm_g: Gaussian FWHM
            fwhm_l: Lorentzian FWHM

        Returns:
            Voigt profile values
        """
        # Handle edge case where Gaussian width is very small
        if fwhm_g < 1e-10:
            # Pure Lorentzian
            gamma = fwhm_l / 2.0
            return (amplitude / np.pi) * (gamma / ((x - center) ** 2 + gamma**2))

        sigma = fwhm_g * self.FWHM_TO_SIGMA
        gamma = fwhm_l / 2.0

        # Complex argument for Faddeeva function
        z = (x - center + 1j * gamma) / (sigma * np.sqrt(2))

        # Voigt profile: real part of Faddeeva function, normalized
        return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

    def fwhm(self, fwhm_g: float = 0.5, fwhm_l: float = 0.5, **params) -> float:
        """
        Approximate FWHM of Voigt profile.

        Uses the Olivero-Longbothum approximation (1977):
        FWHM_V ≈ 0.5346 * fwhm_l + √(0.2166 * fwhm_l² + fwhm_g²)

        Olivero & Longbothum give two fits. This is the simpler one, a
        modification of Whiting's expression, for which their abstract
        states an accuracy of about 0.02%; the 0.01% stated there belongs
        to their other, more elaborate expression, which is not the one
        used here. The largest error measured against the exact
        half-maximum width of the profile is 2.37e-4, at
        fwhm_l/fwhm_g = 0.29, over ratios from 1e-4 to 1e4 (exact for a
        Gaussian, 3e-6 for a Lorentzian); Wang et al. (2022) report the
        same maximum, 2.37e-4. Pinned in
        voigtfit/tests/test_identifiability.py.

        References:
            J. J. Olivero and R. L. Longbothum, J. Quant. Spectrosc.
                Radiat. Transfer 17, 233-236 (1977),
                doi:10.1016/0022-4073(77)90161-3
            Y. Wang, B. Zhou, R. Zhao, B. Wang, Q. Liu and M. Dai,
                Mathematics 10, 210 (2022), doi:10.3390/math10020210
        """
        return 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)

    def area_to_height(
        self, area: float, fwhm_g: float = 0.5, fwhm_l: float = 0.5, **params
    ) -> float:
        """
        Convert area to peak height.

        Peak height is V(center), which depends on both widths.
        """
        if fwhm_g < 1e-10:
            # Pure Lorentzian
            gamma = fwhm_l / 2.0
            return area / (np.pi * gamma)

        sigma = fwhm_g * self.FWHM_TO_SIGMA
        gamma = fwhm_l / 2.0

        # At center, z is pure imaginary
        z = 1j * gamma / (sigma * np.sqrt(2))
        return area * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

    def height_to_area(
        self, height: float, fwhm_g: float = 0.5, fwhm_l: float = 0.5, **params
    ) -> float:
        """Convert peak height to area."""
        if fwhm_g < 1e-10:
            gamma = fwhm_l / 2.0
            return height * np.pi * gamma

        sigma = fwhm_g * self.FWHM_TO_SIGMA
        gamma = fwhm_l / 2.0

        z = 1j * gamma / (sigma * np.sqrt(2))
        factor = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
        return height / factor
