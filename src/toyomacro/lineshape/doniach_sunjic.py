"""Doniach-Sunjic lineshape for metallic XPS systems.

Ported from MATLAB Toyomacro +toyomacro/+lineshape/DoniachSunjic.m and
COMPONENT_GPUAcceleration.m.

The Doniach-Sunjic lineshape describes the asymmetric photoemission line
shape observed in metals due to electron-hole pair excitations near the
Fermi level.

Formula:
    DS(x) = Gamma(1-alpha) * cos(pi*alpha/2 + (1-alpha)*arctan((x-E0)/gamma))
            / ((x-E0)^2 + gamma^2)^((1-alpha)/2)

where:
    - alpha: asymmetry (singularity) index, 0 <= alpha < 1
    - gamma: Lorentzian half-width (fwhm_l / 2)
    - E0: peak center position
    - Gamma(): gamma function (scipy.special.gamma)

Special cases:
    - alpha = 0: reduces to a symmetric Lorentzian
    - Gaussian broadening (fwhm_g > 0): convolve DS with Gaussian via FFT
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma as gamma_func

from toyomacro.lineshape.base import BaseLineshape


class DoniachSunjic(BaseLineshape):
    """Doniach-Sunjic lineshape for metallic XPS peaks.

    Typical usage for metals (Au 4f, Ag 3d, Cu 2p, etc.)
    where the photoemission peak shows an asymmetric tail
    toward higher binding energy.

    Parameters:
        fwhm_g: Gaussian FWHM for instrumental/phonon broadening
        fwhm_l: Lorentzian FWHM (gamma = fwhm_l / 2)
        asymmetry: Singularity index alpha, typically 0.02 - 0.3 for metals
    """

    name = "doniach_sunjic"

    # Gaussian FWHM to sigma
    FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def evaluate(
        self,
        x: NDArray[np.float64],
        center: float,
        amplitude: float,
        fwhm_g: float = 0.0,
        fwhm_l: float = 0.5,
        asymmetry: float = 0.1,
        **params,
    ) -> NDArray[np.float64]:
        """Evaluate Doniach-Sunjic profile with optional Gaussian broadening.

        Args:
            x: Energy axis (binding energy, eV)
            center: Peak center position (eV)
            amplitude: Integrated area (arbitrary units)
            fwhm_g: Gaussian FWHM for broadening (0 = no broadening)
            fwhm_l: Lorentzian FWHM (eV)
            asymmetry: Singularity index alpha in [0, 1)

        Returns:
            DS profile values at each x position, area-normalized
        """
        alpha = np.clip(asymmetry, 0.0, 0.9999)
        gamma = max(fwhm_l / 2.0, 1e-10)
        energy_type = params.get("energy_type", "BE")

        # --- Core DS formula ---
        profile = self._ds_core(x, center, alpha, gamma, energy_type=energy_type)

        # --- Optional Gaussian convolution ---
        if fwhm_g > 1e-10:
            profile = self._convolve_gaussian(x, profile, fwhm_g)

        # --- Area normalization ---
        # Normalize so that trapz(profile, x) == amplitude
        raw_area = np.trapezoid(profile, x)
        if abs(raw_area) > 1e-30:
            profile = amplitude * profile / abs(raw_area)

        return profile

    @staticmethod
    def _ds_core(
        x: NDArray[np.float64],
        center: float,
        alpha: float,
        gamma: float,
        energy_type: str = "BE",
    ) -> NDArray[np.float64]:
        """Compute raw (unnormalized) DS profile.

        The asymmetric tail extends toward the energy-loss side:
        - BE axis: tail toward higher BE (delta = center - x)
        - KE axis: tail toward lower KE (delta = x - center)
        """
        # Sign convention: tail toward energy loss
        delta = (x - center) if energy_type == "KE" else (center - x)

        # Denominator: ((x - E0)^2 + gamma^2)^((1-alpha)/2)
        denom = (delta**2 + gamma**2) ** ((1.0 - alpha) / 2.0)

        # Phase: pi*alpha/2 + (1-alpha)*arctan(delta/gamma)
        phase = np.pi * alpha / 2.0 + (1.0 - alpha) * np.arctan(delta / gamma)

        # Gamma function factor
        gf = gamma_func(1.0 - alpha)

        # Full profile
        profile = gf * np.cos(phase) / denom

        return profile

    def _convolve_gaussian(
        self,
        x: NDArray[np.float64],
        profile: NDArray[np.float64],
        fwhm_g: float,
    ) -> NDArray[np.float64]:
        """Convolve profile with Gaussian kernel via FFT.

        Uses zero-padded FFT convolution for efficiency and
        to avoid circular convolution artifacts.
        """
        sigma = fwhm_g * self.FWHM_TO_SIGMA
        dx = abs(x[1] - x[0]) if len(x) > 1 else 1.0

        # Build Gaussian kernel (same x spacing as data)
        # Extend to ±4 sigma for 99.99% coverage
        half_width = max(int(4.0 * sigma / dx), 1)
        k = np.arange(-half_width, half_width + 1) * dx
        kernel = np.exp(-0.5 * (k / sigma) ** 2)
        kernel /= kernel.sum()  # Normalize

        # FFT convolution (zero-padded to avoid wrap-around)
        n = len(profile)
        nk = len(kernel)
        nfft = n + nk - 1

        conv = np.real(np.fft.ifft(
            np.fft.fft(profile, nfft) * np.fft.fft(kernel, nfft)
        ))

        # Extract same-length centered result
        start = nk // 2
        return conv[start : start + n]

    def fwhm(
        self,
        fwhm_g: float = 0.0,
        fwhm_l: float = 0.5,
        asymmetry: float = 0.1,
        **params,
    ) -> float:
        """Estimate FWHM numerically.

        For alpha = 0 (Lorentzian), FWHM = fwhm_l exactly.
        For alpha > 0, the profile becomes asymmetric so we find the
        half-maximum points numerically.
        """
        alpha = np.clip(asymmetry, 0.0, 0.9999)
        gamma = max(fwhm_l / 2.0, 1e-10)

        if alpha < 1e-10 and fwhm_g < 1e-10:
            return fwhm_l  # Pure Lorentzian

        # Generate fine grid and find half-max
        extent = max(fwhm_l * 10, fwhm_g * 10, 5.0)
        x = np.linspace(-extent, extent, 10001)
        profile = self._ds_core(x, 0.0, alpha, gamma)

        if fwhm_g > 1e-10:
            profile = self._convolve_gaussian(x, profile, fwhm_g)

        peak_val = profile.max()
        if peak_val <= 0:
            return fwhm_l

        half_max = peak_val / 2.0
        above = profile >= half_max
        indices = np.where(above)[0]
        if len(indices) < 2:
            return fwhm_l

        return float(x[indices[-1]] - x[indices[0]])

    def area_to_height(
        self,
        area: float,
        fwhm_g: float = 0.0,
        fwhm_l: float = 0.5,
        asymmetry: float = 0.1,
        **params,
    ) -> float:
        """Convert integrated area to peak height.

        Uses the analytical DS value at center (delta=0):
            DS(0) = Gamma(1-alpha) * cos(pi*alpha/2) / gamma^(1-alpha)
        """
        alpha = np.clip(asymmetry, 0.0, 0.9999)
        gamma = max(fwhm_l / 2.0, 1e-10)

        ds_center = gamma_func(1.0 - alpha) * np.cos(np.pi * alpha / 2.0) / (
            gamma ** (1.0 - alpha)
        )

        if ds_center <= 0:
            return 0.0

        # For Gaussian-broadened case, the peak height is lower
        if fwhm_g > 1e-10:
            # Quick numerical estimate
            extent = max(fwhm_l * 10, fwhm_g * 10, 5.0)
            x = np.linspace(-extent, extent, 2001)
            profile = self.evaluate(x, 0.0, area, fwhm_g, fwhm_l, asymmetry)
            return float(profile.max())

        # No broadening: height = area * ds_center / raw_area
        # raw_area is the integral of the unnormalized DS
        x_int = np.linspace(-50 * gamma, 50 * gamma, 50001)
        raw = self._ds_core(x_int, 0.0, alpha, gamma)
        raw_area = np.trapezoid(raw, x_int)
        if abs(raw_area) < 1e-30:
            return 0.0

        return area * ds_center / abs(raw_area)

    def height_to_area(
        self,
        height: float,
        fwhm_g: float = 0.0,
        fwhm_l: float = 0.5,
        asymmetry: float = 0.1,
        **params,
    ) -> float:
        """Convert peak height to integrated area."""
        # Use inverse of area_to_height
        test_area = 1.0
        test_height = self.area_to_height(
            test_area, fwhm_g=fwhm_g, fwhm_l=fwhm_l, asymmetry=asymmetry
        )
        if test_height <= 0:
            return 0.0
        return height / test_height
