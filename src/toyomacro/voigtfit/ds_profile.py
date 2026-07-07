"""
Doniach-Sunjic Profile for VoigtFit Basis Generation
=====================================================

Core DS lineshape + optional Gaussian broadening via FFT convolution.

Formula:
    DS(x) = Γ(1-α) × cos(πα/2 + (1-α)arctan((x-c)/γ))
            / ((x-c)² + γ²)^((1-α)/2)

where:
    α: asymmetry (singularity) index, 0 ≤ α < 1
    γ: Lorentzian half-width
    c: peak center
    Γ(): gamma function

Special cases:
    α = 0: symmetric Lorentzian
    σ > 0: convolve with Gaussian via FFT (instrumental broadening)

This module is self-contained (no toyomacro dependency).
Same formula as toyomacro.lineshape.doniach_sunjic._ds_core.
"""

import numpy as np
from scipy.special import gamma as gamma_func

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def ds_profile(
    energy: np.ndarray,
    center: float,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    """Compute raw (unnormalized) Doniach-Sunjic profile.

    Args:
        energy: Energy axis (n_energy,)
        center: Peak center position
        alpha: Asymmetry index in [0, 1)
        gamma: Lorentzian half-width (> 0)

    Returns:
        profile: DS values (n_energy,), unnormalized
    """
    alpha = np.clip(alpha, 0.0, 0.9999)
    delta = energy - center

    denom = (delta**2 + gamma**2) ** ((1.0 - alpha) / 2.0)
    phase = np.pi * alpha / 2.0 + (1.0 - alpha) * np.arctan(delta / gamma)
    gf = gamma_func(1.0 - alpha)

    return gf * np.cos(phase) / denom


def ds_with_gaussian(
    energy: np.ndarray,
    center: float,
    alpha: float,
    gamma: float,
    sigma: float,
) -> np.ndarray:
    """DS profile with Gaussian broadening, area-normalized to 1.0.

    Args:
        energy: Energy axis (n_energy,)
        center: Peak center position
        alpha: Asymmetry index in [0, 1)
        gamma: Lorentzian half-width (> 0)
        sigma: Gaussian σ for broadening (0 = no broadening)

    Returns:
        profile: Area-normalized DS profile (n_energy,), float32
    """
    profile = ds_profile(energy, center, alpha, gamma)

    if sigma > 1e-10:
        profile = _convolve_gaussian(energy, profile, sigma)

    # Area normalize
    area = np.trapezoid(profile, energy)
    if abs(area) > 1e-30:
        profile = profile / abs(area)

    return profile.astype(np.float32)


def _convolve_gaussian(
    energy: np.ndarray,
    profile: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Convolve profile with Gaussian kernel via zero-padded FFT.

    Args:
        energy: Energy axis (uniform spacing assumed)
        profile: Input profile
        sigma: Gaussian standard deviation

    Returns:
        Convolved profile (same length as input)
    """
    dx = abs(energy[1] - energy[0]) if len(energy) > 1 else 1.0

    # Gaussian kernel: ±4σ coverage
    half_width = max(int(4.0 * sigma / dx), 1)
    k = np.arange(-half_width, half_width + 1) * dx
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()

    # FFT convolution (zero-padded)
    n = len(profile)
    nk = len(kernel)
    nfft = n + nk - 1

    conv = np.real(np.fft.ifft(
        np.fft.fft(profile, nfft) * np.fft.fft(kernel, nfft)
    ))

    # Extract centered result
    start = nk // 2
    return conv[start: start + n]
