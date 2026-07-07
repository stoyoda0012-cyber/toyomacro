"""Pseudo-Voigt lineshape profile."""

import numpy as np
from numpy.typing import NDArray

from toyomacro.lineshape.base import BaseLineshape
from toyomacro.lineshape.gaussian import Gaussian
from toyomacro.lineshape.lorentzian import Lorentzian


class PseudoVoigt(BaseLineshape):
    """
    Pseudo-Voigt lineshape profile.

    A linear combination of Gaussian and Lorentzian:
    PV(x) = η * L(x) + (1 - η) * G(x)

    where η (eta) is the mixing parameter [0, 1]:
        - η = 0: Pure Gaussian
        - η = 1: Pure Lorentzian

    Faster than true Voigt but less physically accurate.
    """

    name = "pseudovoigt"

    def __init__(self):
        self._gaussian = Gaussian()
        self._lorentzian = Lorentzian()

    def evaluate(
        self,
        x: NDArray[np.float64],
        center: float,
        amplitude: float,
        fwhm: float = 1.0,
        eta: float = 0.5,
        fwhm_g: float | None = None,
        fwhm_l: float | None = None,
        **params,
    ) -> NDArray[np.float64]:
        """
        Evaluate Pseudo-Voigt profile.

        Args:
            x: Energy axis
            center: Peak center
            amplitude: Peak area
            fwhm: Full Width at Half Maximum (used for both G and L if
                fwhm_g/fwhm_l not given)
            eta: Lorentzian fraction [0, 1]
            fwhm_g: Gaussian FWHM (overrides fwhm for G component)
            fwhm_l: Lorentzian FWHM (overrides fwhm for L component)

        Returns:
            Pseudo-Voigt profile values
        """
        eta = np.clip(eta, 0.0, 1.0)
        fg = fwhm_g if fwhm_g is not None else fwhm
        fl = fwhm_l if fwhm_l is not None else fwhm

        g = self._gaussian.evaluate(x, center, amplitude, fwhm=fg)
        l = self._lorentzian.evaluate(x, center, amplitude, fwhm=fl)

        return eta * l + (1 - eta) * g

    def fwhm(self, fwhm: float = 1.0, **params) -> float:
        """Return FWHM (same for both Gaussian and Lorentzian components)."""
        return fwhm

    def area_to_height(
        self, area: float, fwhm: float = 1.0, eta: float = 0.5,
        fwhm_g: float | None = None, fwhm_l: float | None = None, **params,
    ) -> float:
        """Convert area to peak height."""
        eta = np.clip(eta, 0.0, 1.0)
        fg = fwhm_g if fwhm_g is not None else fwhm
        fl = fwhm_l if fwhm_l is not None else fwhm
        h_g = self._gaussian.area_to_height(area, fwhm=fg)
        h_l = self._lorentzian.area_to_height(area, fwhm=fl)
        return eta * h_l + (1 - eta) * h_g

    def height_to_area(
        self, height: float, fwhm: float = 1.0, eta: float = 0.5,
        fwhm_g: float | None = None, fwhm_l: float | None = None, **params,
    ) -> float:
        """Convert peak height to area."""
        eta = np.clip(eta, 0.0, 1.0)
        fg = fwhm_g if fwhm_g is not None else fwhm
        fl = fwhm_l if fwhm_l is not None else fwhm
        # Inverse of the linear combination
        factor = eta / (np.pi * fl / 2) + (1 - eta) / (
            fg * Gaussian.FWHM_TO_SIGMA * np.sqrt(2 * np.pi)
        )
        return height / factor

    @staticmethod
    def eta_from_voigt_params(fwhm_g: float, fwhm_l: float) -> tuple[float, float]:
        """
        Estimate Pseudo-Voigt parameters from Voigt parameters.

        Uses Thompson-Cox-Hastings approximation (1987).

        Args:
            fwhm_g: Gaussian FWHM
            fwhm_l: Lorentzian FWHM

        Returns:
            Tuple of (total_fwhm, eta)
        """
        fg, fl = fwhm_g, fwhm_l

        # Total FWHM approximation
        fwhm_total = (
            fg**5
            + 2.69269 * fg**4 * fl
            + 2.42843 * fg**3 * fl**2
            + 4.47163 * fg**2 * fl**3
            + 0.07842 * fg * fl**4
            + fl**5
        ) ** 0.2

        if fwhm_total < 1e-10:
            return 0.0, 0.5

        # Lorentzian fraction
        ratio = fl / fwhm_total
        eta = (
            1.36603 * ratio
            - 0.47719 * ratio**2
            + 0.11116 * ratio**3
        )

        return fwhm_total, np.clip(eta, 0.0, 1.0)
