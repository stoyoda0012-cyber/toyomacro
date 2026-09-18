"""Fermi-Dirac edge lineshape for valence band XPS analysis.

Ported from MATLAB Toyomacro +toyomacro/+lineshape/FermiDirac.m and a
valence-band analysis script.

For *fitting* a Fermi edge use :mod:`toyomacro.fitting.fermi_edge`,
which evaluates the same model with two differences that matter near
the ends of a cut spectrum: this lineshape convolves on the given axis
only, so channels within ~2 FWHM of either end see a clamped copy of the
end value (a fit in a narrow window is biased by it), and it skips the
broadening entirely when the Gaussian sigma is at most half a channel.

The Fermi-Dirac lineshape models the valence band Fermi edge as:
    I(E) = A × DOS(E) × f(E, EF, T) ⊗ G(σ_instr) + bg

where:
    - f(E, EF, T) = 1 / (1 + exp(-(BE - EF) / (kB * T)))
      in the binding energy (BE) convention (EnergySign = -1)
    - DOS(E) = 1 + c1*x + c2*x^2   (x = max(BE - EF, 0))
      captures the d-band onset curvature near EF
    - G(σ_instr): Gaussian instrument broadening via FFT convolution

Parameters mapping to BaseLineshape interface:
    center    → EF (Fermi energy position in eV)
    amplitude → A  (overall scale factor)
    fwhm_g    → Gaussian FWHM for instrument broadening
    temperature → Temperature in Kelvin
    dos_c1    → linear DOS coefficient
    dos_c2    → quadratic DOS coefficient (0 for linear DOS)
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from toyomacro.lineshape.base import BaseLineshape

# Boltzmann constant in eV/K
KB_EV = 8.617333e-5

# FWHM to sigma conversion: σ = FWHM / (2√(2ln2))
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


class FermiDirac(BaseLineshape):
    """Fermi-Dirac edge lineshape for valence band XPS.

    Models the Fermi edge cutoff in photoemission spectra, including:
    - Fermi-Dirac distribution (thermal broadening)
    - Polynomial density of states (flat, linear, or quadratic)
    - Gaussian instrument broadening (FFT convolution)

    Parameters:
        fwhm_g: Gaussian FWHM for instrument broadening (eV)
        temperature: Sample temperature in Kelvin (default 300 K)
        dos_c1: Linear DOS coefficient (default 0)
        dos_c2: Quadratic DOS coefficient (default 0)
    """

    name = "fermi_dirac"

    def evaluate(
        self,
        x: NDArray[np.float64],
        center: float,
        amplitude: float,
        fwhm_g: float = 0.0,
        temperature: float = 300.0,
        dos_c1: float = 0.0,
        dos_c2: float = 0.0,
        **params,
    ) -> NDArray[np.float64]:
        """Evaluate FD × DOS profile with optional Gaussian broadening.

        Args:
            x: Energy axis (binding energy, eV)
            center: Fermi energy position EF (eV)
            amplitude: Absolute scale factor (DOS height at EF)
            fwhm_g: Gaussian FWHM for instrument broadening (eV)
            temperature: Temperature in Kelvin (must be > 0)
            dos_c1: Linear DOS coefficient
            dos_c2: Quadratic DOS coefficient

        Returns:
            amplitude × DOS(E) × f(E, EF, T) ⊗ G(σ_instr)
        """
        ef = center
        t_k = max(temperature, 1.0)

        # Ensure sorted for convolution
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]

        # DOS: polynomial above EF (occupied side in BE convention)
        dx_be = np.maximum(x_sorted - ef, 0.0)
        dos = 1.0 + dos_c1 * dx_be + dos_c2 * dx_be**2
        dos = np.maximum(dos, 0.0)

        # Fermi-Dirac distribution (BE convention)
        fermi = self._fermi_function(x_sorted, ef, t_k)

        profile = amplitude * dos * fermi

        # Gaussian convolution (scipy gaussian_filter1d for correct normalization)
        if fwhm_g > 1e-10 and len(x_sorted) > 1:
            dx_grid = abs(x_sorted[1] - x_sorted[0])
            sigma_pts = (fwhm_g * FWHM_TO_SIGMA) / dx_grid
            if sigma_pts > 0.5:
                profile = gaussian_filter1d(profile, sigma_pts, mode='nearest')

        # Unsort back to original order
        unsort_idx = np.argsort(sort_idx)
        return profile[unsort_idx]

    @staticmethod
    def _fermi_function(
        be: NDArray[np.float64],
        ef: float,
        t_k: float,
    ) -> NDArray[np.float64]:
        """Fermi-Dirac distribution in binding energy convention.

        f(BE) = 1 / (1 + exp(-(BE - EF) / (kB * T)))

        In BE convention (EnergySign = -1):
        - BE < EF (unoccupied): f → 0
        - BE > EF (occupied):   f → 1
        """
        arg = -(be - ef) / (KB_EV * t_k)
        arg = np.clip(arg, -700, 700)
        return 1.0 / (1.0 + np.exp(arg))

    # NOTE: _convolve_gaussian_fft removed — replaced by scipy.ndimage.gaussian_filter1d
    # in evaluate() for correct normalization of absolute amplitude.

    def fwhm(
        self,
        fwhm_g: float = 0.0,
        temperature: float = 300.0,
        **params,
    ) -> float:
        """Estimate the effective edge width.

        For a Fermi edge, the "FWHM" concept is the 10-90% width,
        dominated by max(thermal width, instrument width).
        Thermal width ≈ 4 * kB * T.
        """
        thermal_width = 4.0 * KB_EV * max(temperature, 1.0)
        if fwhm_g > 1e-10:
            # Quadrature sum as rough estimate
            return float(np.sqrt(thermal_width**2 + fwhm_g**2))
        return thermal_width

    def area_to_height(self, area: float, **params) -> float:
        """Not meaningful for Fermi edge (step function, not peak)."""
        raise NotImplementedError(
            "area_to_height is not applicable to the Fermi-Dirac edge lineshape"
        )

    def height_to_area(self, height: float, **params) -> float:
        """Not meaningful for Fermi edge (step function, not peak)."""
        raise NotImplementedError(
            "height_to_area is not applicable to the Fermi-Dirac edge lineshape"
        )
