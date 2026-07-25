"""Gaussian lineshape profile."""

import numpy as np
from numpy.typing import NDArray

from toyomacro.lineshape.base import BaseLineshape


class Gaussian(BaseLineshape):
    """
    Gaussian lineshape profile.

    G(x) = (A / (σ√(2π))) * exp(-(x-x₀)² / (2σ²))

    where:
        - A is the area
        - σ = FWHM / (2√(2ln2)) ≈ FWHM / 2.3548
        - x₀ is the center
    """

    name = "gaussian"

    # FWHM to sigma conversion factor
    FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ≈ 0.4247

    def evaluate(
        self,
        x: NDArray[np.float64],
        center: float,
        amplitude: float,
        fwhm: float = 1.0,
        **params,
    ) -> NDArray[np.float64]:
        """
        Evaluate Gaussian profile.

        Args:
            x: Energy axis
            center: Peak center
            amplitude: Peak area
            fwhm: Full Width at Half Maximum

        Returns:
            Gaussian profile values
        """
        sigma = fwhm * self.FWHM_TO_SIGMA
        norm = amplitude / (sigma * np.sqrt(2 * np.pi))
        return norm * np.exp(-0.5 * ((x - center) / sigma) ** 2)

    def fwhm(self, fwhm: float = 1.0, **params) -> float:
        """Return FWHM (trivial for Gaussian as it's a direct parameter)."""
        return fwhm

    def area_to_height(self, area: float, fwhm: float = 1.0, **params) -> float:
        """Convert area to peak height."""
        sigma = fwhm * self.FWHM_TO_SIGMA
        return area / (sigma * np.sqrt(2 * np.pi))

    def height_to_area(self, height: float, fwhm: float = 1.0, **params) -> float:
        """Convert peak height to area."""
        sigma = fwhm * self.FWHM_TO_SIGMA
        return height * sigma * np.sqrt(2 * np.pi)
