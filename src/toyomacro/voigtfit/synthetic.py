"""Synthetic XPS spectrum generation for benchmarks and tests.

Provides element-specific profiles and helper functions to create
ComponentConfig lists and spectra with known ground truth.

Usage::

    from toyomacro.voigtfit.synthetic import (
        ELEMENT_PROFILES, make_synthetic_configs, make_synthetic_spectra,
    )
    profile = ELEMENT_PROFILES["Si2p"]
    configs, nominal_amps = make_synthetic_configs(profile, n_comp=5)
    spectra, true_amps = make_synthetic_spectra(configs, profile, n_spectra=1000)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import wofz

from .multipeak_config import ComponentConfig

_SQRT2 = float(np.sqrt(2.0))
_SQRT2PI = float(np.sqrt(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Element profiles
# ---------------------------------------------------------------------------


@dataclass
class ElementProfile:
    """Physical parameters for an XPS element line.

    Attributes:
        name: Element/line identifier (e.g. "Si2p").
        sigma: Gaussian width (eV).
        gamma: Lorentzian width (eV).
        so_split: Spin-orbit splitting (eV); ``None`` for singlets.
            Partner peak at ``center + so_split``.
        branch_ratio: ``I_main / I_partner`` — same convention as
            :class:`ComponentConfig`.  ``None`` for singlets.
        typical_dE: Typical spacing between chemical states (eV).
        n_comp_range: Recommended range of component counts.
    """

    name: str
    sigma: float
    gamma: float
    so_split: float | None = None
    branch_ratio: float | None = None
    typical_dE: float = 1.0
    n_comp_range: tuple[int, int] = (1, 5)


ELEMENT_PROFILES: dict[str, ElementProfile] = {
    "Si2p": ElementProfile(
        name="Si2p",
        sigma=0.55,
        gamma=0.10,
        so_split=0.608,
        branch_ratio=2.0,  # I(2p3/2) / I(2p1/2) = 2:1
        typical_dE=1.0,
    ),
    "O1s": ElementProfile(
        name="O1s",
        sigma=0.70,
        gamma=0.05,
        typical_dE=1.5,
    ),
    "Au4f": ElementProfile(
        name="Au4f",
        sigma=0.55,
        gamma=0.30,
        so_split=3.70,
        branch_ratio=4 / 3,  # I(4f7/2) / I(4f5/2) = 4:3
        typical_dE=1.0,
    ),
    "C1s": ElementProfile(
        name="C1s",
        sigma=0.60,
        gamma=0.05,
        typical_dE=1.5,
    ),
}


# ---------------------------------------------------------------------------
# Voigt helpers (standalone, no voigtfit engine dependency)
# ---------------------------------------------------------------------------


def _voigt(
    energy: np.ndarray, center: float, sigma: float, gamma: float
) -> np.ndarray:
    """Single Voigt profile via Faddeeva function."""
    z = ((energy - center) + 1j * gamma) / (sigma * _SQRT2)
    return np.real(wofz(z)) / (sigma * _SQRT2PI)


def _doublet(
    energy: np.ndarray,
    center: float,
    sigma: float,
    gamma: float,
    so_split: float,
    branch_ratio: float,
) -> np.ndarray:
    """SO doublet: main + partner / branch_ratio."""
    main = _voigt(energy, center, sigma, gamma)
    partner = _voigt(energy, center + so_split, sigma, gamma)
    return main + partner / branch_ratio


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def make_synthetic_configs(
    profile: ElementProfile,
    n_comp: int,
    amp_distribution: str = "uniform",
    E_start: float = 100.0,
    seed: int = 42,
) -> tuple[list[ComponentConfig], np.ndarray]:
    """Generate ComponentConfig list and nominal amplitudes.

    Parameters
    ----------
    profile : ElementProfile
        Element physical parameters.
    n_comp : int
        Number of chemical states (components).
    amp_distribution : str
        ``"uniform"`` : amp in [0.8, 1.2].
        ``"moderate"`` : amp in [0.3, 1.0].
        ``"random"`` : amp in [0.05, 1.0].
    E_start : float
        Center of the first component (eV).
    seed : int
        Random seed for amplitude sampling.

    Returns
    -------
    configs : list[ComponentConfig]
        One per component, centres spaced by >= 3.5 sigma.
    true_amps : ndarray (n_comp,) float32
        Nominal amplitudes.
    """
    rng = np.random.default_rng(seed)

    # Amplitude sampling
    ranges = {
        "uniform": (0.8, 1.2),
        "moderate": (0.3, 1.0),
        "random": (0.05, 1.0),
    }
    if amp_distribution not in ranges:
        raise ValueError(f"Unknown amp_distribution: {amp_distribution!r}")
    lo, hi = ranges[amp_distribution]
    amps = rng.uniform(lo, hi, n_comp).astype(np.float32)

    # Centre positions: spaced by >= 3.5 sigma
    min_spacing = max(3.5 * profile.sigma, profile.typical_dE)
    centers = [E_start + i * min_spacing for i in range(n_comp)]

    # dE_range: constrained to prevent cross-assignment
    if n_comp > 1:
        dE_range = min(2.0, max(0.1, min_spacing / 2 - profile.sigma / 2))
    else:
        dE_range = 2.0

    is_dbl = profile.so_split is not None and profile.so_split > 0
    configs = [
        ComponentConfig(
            center=c,
            sigma=profile.sigma,
            gamma=profile.gamma,
            dE_range=dE_range,
            ds_range=0.3,
            n_dE=10,
            n_ds=31,
            so_split=profile.so_split if is_dbl else 0.0,
            branch_ratio=profile.branch_ratio if is_dbl else 1.0,
        )
        for c in centers
    ]
    return configs, amps


# ---------------------------------------------------------------------------
# Spectra generation
# ---------------------------------------------------------------------------


def make_synthetic_spectra(
    configs: list[ComponentConfig],
    profile: ElementProfile,
    n_spectra: int,
    snr_db: float = 40.0,
    E_range: tuple[float, float] | None = None,
    n_channels: int = 128,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic spectra with known ground truth.

    Each spectrum is a weighted sum of peak-normalised basis profiles
    with per-spectrum amplitudes independently sampled.

    Parameters
    ----------
    configs : list[ComponentConfig]
        Component configurations (from :func:`make_synthetic_configs`).
    profile : ElementProfile
        Element profile for sigma/gamma reference.
    n_spectra : int
        Number of spectra.
    snr_db : float
        Signal-to-noise ratio in dB (SNR = peak_signal / noise_std).
        Values >= 100 dB are treated as noiseless.
    E_range : tuple[float, float] | None
        ``(E_min, E_max)``.  ``None`` auto-calculates to cover all
        components including SO partners with 3 sigma margin.
    n_channels : int
        Number of energy channels.
    seed : int
        Random seed.

    Returns
    -------
    spectra : ndarray (n_spectra, n_channels) float32
    true_amps : ndarray (n_spectra, n_comp) float32
    """
    rng = np.random.default_rng(seed)
    n_comp = len(configs)

    # --- Energy axis ---
    if E_range is None:
        positions: list[float] = [c.center for c in configs]
        for c in configs:
            if c.is_doublet:
                positions.append(c.center + c.so_split)
        margin = 3.0 * profile.sigma
        E_min = min(positions) - margin
        E_max = max(positions) + margin
    else:
        E_min, E_max = E_range

    energy = np.linspace(E_min, E_max, n_channels, dtype=np.float64)

    # --- Peak-normalised basis (n_comp, n_channels) ---
    basis = np.zeros((n_comp, n_channels), dtype=np.float64)
    for k, cc in enumerate(configs):
        if cc.is_doublet:
            prof = _doublet(
                energy, cc.center, cc.sigma, cc.gamma,
                cc.so_split, cc.branch_ratio,
            )
        else:
            prof = _voigt(energy, cc.center, cc.sigma, cc.gamma)
        prof /= prof.max() + 1e-30
        basis[k] = prof

    # --- Per-spectrum amplitudes ---
    true_amps = rng.uniform(0.1, 1.0, (n_spectra, n_comp)).astype(np.float32)

    # --- Spectra = amps @ basis ---
    spectra = true_amps.astype(np.float64) @ basis

    # --- Gaussian noise (SNR = peak / noise_std) ---
    if snr_db < 100.0:
        noise_std = 10.0 ** (-snr_db / 20.0)
        peak_vals = spectra.max(axis=1, keepdims=True)
        noise = rng.normal(0.0, 1.0, spectra.shape) * (peak_vals * noise_std)
        spectra += noise

    return spectra.astype(np.float32), true_amps
