"""Lorentzian lineshape profile."""

import numpy as np
from numpy.typing import NDArray

from toyomacro.lineshape.base import BaseLineshape


class Lorentzian(BaseLineshape):
    """
    Lorentzian (Cauchy) lineshape profile.

    L(x) = (A / π) * (γ / ((x - x₀)² + γ²))

    where:
        - A is the area
        - γ = FWHM / 2
        - x₀ is the center
    """

    name = "lorentzian"

    def evaluate(
        self,
        x: NDArray[np.float64],
        center: float,
        amplitude: float,
        fwhm: float = 1.0,
        **params,
    ) -> NDArray[np.float64]:
        """
        Evaluate Lorentzian profile.

        Args:
            x: Energy axis
            center: Peak center
            amplitude: Peak area
            fwhm: Full Width at Half Maximum

        Returns:
            Lorentzian profile values
        """
        gamma = fwhm / 2.0
        return (amplitude / np.pi) * (gamma / ((x - center) ** 2 + gamma**2))

    def fwhm(self, fwhm: float = 1.0, **params) -> float:
        """Return FWHM."""
        return fwhm

    def area_to_height(self, area: float, fwhm: float = 1.0, **params) -> float:
        """Convert area to peak height."""
        gamma = fwhm / 2.0
        return area / (np.pi * gamma)

    def height_to_area(self, height: float, fwhm: float = 1.0, **params) -> float:
        """Convert peak height to area."""
        gamma = fwhm / 2.0
        return height * np.pi * gamma
