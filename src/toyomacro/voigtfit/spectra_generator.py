"""
Spectra Generator for PSNR Benchmark Testing.

Generates synthetic XPS spectra from images for algorithm evaluation.
Supports noise injection (Poisson, Gaussian, mixed) to simulate
real imaging XPS conditions.

Workflow:
    Original Image → Color Decomposition → Voigt Spectra + Noise → H5 Files

Based on MATLAB DepthProfiler.m SaveAngleProfileSpectra_Callback (line 3744)
and NoiseGenerator.m
Date: 2026-01-22
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import h5py
import numpy as np

from .frame_io import open_frames

# MLX for GPU-accelerated noise generation
try:
    import mlx.core as mx

    from ._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND a Metal device works
except ImportError:
    HAS_MLX = False

# Numba for parallel noise generation (~11x faster than NumPy)
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(parallel=True)
    def _poisson_noise_numba(scaled, out, use_gaussian):
        """Poisson noise generation, parallelized with Numba.

        For large λ (use_gaussian=True): Gaussian approximation
            Poisson(λ) ≈ N(λ, √λ), error < 1% for λ > 5.
        For small λ (use_gaussian=False): exact Poisson via inversion.
            Uses Knuth's algorithm: count uniform draws until product < e^(-λ).
        """
        n, m = scaled.shape
        for i in prange(n):
            for j in range(m):
                lam = scaled[i, j]
                if use_gaussian or lam > 20.0:
                    # Gaussian approximation
                    v = lam + np.sqrt(lam) * np.random.randn()
                    out[i, j] = max(v, 0.0)
                else:
                    # Exact Poisson (Knuth's algorithm)
                    exp_neg_lam = np.exp(-lam)
                    k = 0
                    p = 1.0
                    while True:
                        p *= np.random.random()
                        if p <= exp_neg_lam:
                            break
                        k += 1
                    out[i, j] = float(k)


@dataclass
class ElementSpec:
    """Specification for a single element's spectral parameters.

    Note: Gaussian and Lorentzian widths are specified as FWHM (Full Width at Half Maximum).
    Internally converted to σ and γ for Voigt profile calculation.
    """
    symbol: str             # e.g., 'C', 'O', 'Si'
    orbital: str            # e.g., '1s', '2p'
    binding_energy: float   # Center position (eV)
    gaussian_fwhm: float = 1.0   # Gaussian FWHM (eV)
    lorentzian_fwhm: float = 0.25  # Lorentzian FWHM (eV)
    asymmetry: float = 0.0       # Asymmetry parameter (Doniach-Sunjic)
    branching_ratio: float = 0.0  # Branching ratio for spin-orbit split
    spinorbit_split: float = 0.0  # Spin-orbit splitting (eV)
    cross_section: float = 1.0    # Relative cross section

    @property
    def file_key(self) -> str:
        return f"{self.symbol}{self.orbital}"

    @property
    def sigma(self) -> float:
        """Convert Gaussian FWHM to standard deviation σ."""
        # FWHM = 2 * sqrt(2 * ln(2)) * σ ≈ 2.3548 * σ
        return self.gaussian_fwhm / (2 * np.sqrt(2 * np.log(2)))

    @property
    def gamma(self) -> float:
        """Get Lorentzian half-width at half-maximum (γ = FWHM/2)."""
        return self.lorentzian_fwhm / 2


@dataclass
class ElementPreset:
    """Preset definition for element configuration + color mapping.

    Bundles element specs, RGB color mapping, and component order
    so that any image can be decomposed/reconstructed with a consistent
    element-to-color assignment.

    Example:
        preset = ElementPreset(
            name='simple_3elem',
            elements=[
                ElementSpec('C', '1s', 284.4),
                ElementSpec('O', '1s', 531.0),
                ElementSpec('Si', '1s', 1839.1),
            ],
            color_mapping=np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32).T,
            component_order=[('C1s',0), ('O1s',0), ('Si1s',0)],
        )
        generator.configure_elements(preset)
    """
    name: str
    elements: list[ElementSpec]
    color_mapping: np.ndarray              # (3, n_components) RGB weights
    component_order: list[tuple[str, int]] # (file_key, comp_idx_in_file)


# ============================================================================
# Built-in Presets
# ============================================================================

FUJI_PRESET = ElementPreset(
    name='fuji',
    elements=[
        ElementSpec('Si', '1s', 1839.1, gaussian_fwhm=1.0, lorentzian_fwhm=0.25),   # Si(0)
        ElementSpec('Si', '1s', 1843.8, gaussian_fwhm=1.0, lorentzian_fwhm=0.25),   # Si(+4)
        ElementSpec('Ti', '1s', 4968.7, gaussian_fwhm=1.0, lorentzian_fwhm=0.25),   # Ti(+4)
        ElementSpec('Al', '1s', 1561.8, gaussian_fwhm=1.0, lorentzian_fwhm=0.25),   # Al(+3)
        ElementSpec('C', '1s', 284.4, gaussian_fwhm=1.0, lorentzian_fwhm=0.25),     # C(0)
        ElementSpec('O', '1s', 531.0, gaussian_fwhm=1.0, lorentzian_fwhm=0.25),     # O(-2)
    ],
    color_mapping=np.array([
        [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # R: Si(0), Si(+4), O(-2)
        [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],  # G: Si(+4), Ti(+4), Al(+3)
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],  # B: Al(+3), C(0), O(-2)
    ], dtype=np.float32),
    component_order=[
        ('Si1s', 0),   # Si(0)
        ('Si1s', 1),   # Si(+4)
        ('Ti1s', 0),   # Ti(+4)
        ('Al1s', 0),   # Al(+3)
        ('C1s', 0),    # C(0)
        ('O1s', 0),    # O(-2)
    ],
)

# Preset registry
_PRESETS: dict[str, ElementPreset] = {'fuji': FUJI_PRESET}


def register_element_preset(preset: ElementPreset) -> None:
    """Register a custom element preset for use with configure_elements()."""
    _PRESETS[preset.name] = preset


def get_element_preset(name: str) -> ElementPreset:
    """Get a registered element preset by name.

    Args:
        name: Preset name (e.g., 'fuji')

    Returns:
        ElementPreset

    Raises:
        KeyError: If preset name is not registered
    """
    if name not in _PRESETS:
        available = ', '.join(sorted(_PRESETS.keys()))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return _PRESETS[name]


def list_element_presets() -> list[str]:
    """List all registered preset names."""
    return sorted(_PRESETS.keys())


@dataclass
class NoiseConfig:
    """Configuration for noise injection.

    Attributes:
        poisson_level: Poisson noise level (0 = none)
        gaussian_std: Gaussian noise std (fraction of signal)
        noise_type: Type of noise ('poisson', 'gaussian', 'mixed', 'none')
        heteroscedastic: If True (default), use global_max normalization for
            Poisson noise. This preserves intensity-dependent noise: bright
            pixels get higher λ → better S/N, dark pixels get lower λ → worse
            S/N. Physically correct for XPS where count rate ∝ signal intensity.
            Set to False for legacy homoscedastic behavior (all pixels get same
            λ_peak regardless of amplitude).
    """
    poisson_level: float = 0.0      # Poisson noise level (0 = none)
    gaussian_std: float = 0.0       # Gaussian noise std (fraction of signal)
    noise_type: Literal['poisson', 'gaussian', 'mixed', 'none'] = 'none'
    heteroscedastic: bool = True     # Default: physically correct Poisson noise

    def __str__(self) -> str:
        homo_tag = '_homo' if not self.heteroscedastic else ''
        if self.noise_type == 'none':
            return 'noise_none'
        elif self.noise_type == 'poisson':
            return f'poisson_{self.poisson_level:.0e}{homo_tag}'
        elif self.noise_type == 'gaussian':
            return f'gaussian_{self.gaussian_std:.0e}'
        else:
            return f'mixed_p{self.poisson_level:.0e}_g{self.gaussian_std:.0e}{homo_tag}'


@dataclass
class SpectralConfig:
    """
    Spectral generation parameters with strict defaults.

    Based on MATLAB DepthProfiler.m SaveAngleProfileSpectra_Callback (line 3744-3894)
    and ToyomacroSchema definitions.

    Default values selected for:
    - Core-level spectra (spectypecore=True)
    - Optimal balance between resolution and file size
    - Realistic background levels
    """
    # Spectrum type
    spectypecore: bool = True       # True=CoreLevel, False=Survey

    # Energy axis (Core-level defaults)
    engedge_core: float = 2.5       # Edge width for core-level (eV)
    engstep_core: float = 0.05      # Energy step for core-level (eV)
    engedge_survey: float = 50.0    # Edge width for survey (eV)
    engstep_survey: float = 1.0     # Energy step for survey (eV)

    # Function type: 1=Voigt, 2=PseudoVoigt, 3=Gauss, 4=Lorentz, 5=DoniachSunjic, 6=FermiDirac
    FCtype: int = 1

    # Background type: 1=Shirley, 2=Tougaard, 3=Level, 4=Linear, 5=None
    BGtype: int = 1

    # Background parameters
    # BGbase: VeryHigh=1, High=0.1, Medium=0.01, Low=0.001, VeryLow=0.0001
    BGbase: float = 0.001           # Low (default)
    # BGlevel = factor * BGbase: High=50x, Medium=10x, Low=2x
    BGlevel_factor: float = 10.0    # Medium (default)

    # Noise settings
    # NoiseDegree: None=0, Minimal=1e-2, Negligible=1e-1, Subtle=1e0, Weak=1e1,
    #              Small=1e2, Moderate=1e3, Strong=1e4, Intense=1e5, Extreme=1e6, Maximal=1e7
    NoiseDegree: float = 0.0        # None (default)
    # NoiseType: 1=Poisson, 2=Gaussian, 3=Uniform
    NoiseType: int = 1              # Poisson (default)

    # Data format
    DataTypeMATHDF5: bool = False   # False=HDF5, True=MAT
    SpectralModelDepthProfile: bool = True  # True=DepthProfile, False=ElementImage

    # Amplitude scaling
    amplitude_scale: float = 1e6

    @property
    def engedge(self) -> float:
        """Get energy edge based on spectrum type."""
        return self.engedge_core if self.spectypecore else self.engedge_survey

    @property
    def engstep(self) -> float:
        """Get energy step based on spectrum type."""
        return self.engstep_core if self.spectypecore else self.engstep_survey

    @property
    def BGlevel(self) -> float:
        """Calculate background level."""
        return self.BGlevel_factor * self.BGbase

    def energy_points(self, binding_energies: list[float]) -> int:
        """Calculate number of energy points for given binding energies."""
        xmin = min(binding_energies) - self.engedge
        xmax = max(binding_energies) + self.engedge
        return int((xmax - xmin) / self.engstep) + 1

    def energy_range(self, binding_energies: list[float]) -> tuple[float, float]:
        """Get energy range for given binding energies."""
        return (min(binding_energies) - self.engedge,
                max(binding_energies) + self.engedge)

    @classmethod
    def noise_degree_name(cls, value: float) -> str:
        """Convert NoiseDegree value to descriptive name."""
        mapping = {
            0: 'None', 1e-2: 'Minimal', 1e-1: 'Negligible', 1e0: 'Subtle',
            1e1: 'Weak', 1e2: 'Small', 1e3: 'Moderate', 1e4: 'Strong',
            1e5: 'Intense', 1e6: 'Extreme', 1e7: 'Maximal'
        }
        return mapping.get(value, f'{value:.0e}')

    @classmethod
    def noise_degree_value(cls, name: str) -> float:
        """Convert descriptive name to NoiseDegree value."""
        import warnings
        mapping = {
            'None': 0, 'Minimal': 1e-2, 'Negligible': 1e-1, 'Subtle': 1e0,
            'Weak': 1e1, 'Small': 1e2, 'Moderate': 1e3, 'Strong': 1e4,
            'Intense': 1e5, 'Extreme': 1e6, 'Maximal': 1e7,
            'Light': 1e2, 'Medium': 1e3, 'Heavy': 1e4, 'VeryHeavy': 1e5,
        }
        if name not in mapping:
            warnings.warn(
                f"Unknown noise level '{name}', defaulting to 0 (noise-free). "
                f"Valid names: {sorted(mapping.keys())}",
                stacklevel=2,
            )
        return mapping.get(name, 0)


# Noise level presets for convenience
# Named noise-severity levels.  The value is the dimensionless `level`
# parameter of ``add_poisson_noise`` — it is NOT a Poisson mean.  The
# physical interpretation is fixed by the conversions
#
#     SNR_peak    = 10^4 / level          (counting SNR at the signal max)
#     lambda_peak = (10^4 / level)^2      (Poisson mean at the signal max)
#
# provided by :func:`level_to_peak_snr` / :func:`level_to_peak_lambda`.
NOISE_LEVELS = {
    'None': 0,
    'Minimal': 1e-2,
    'Negligible': 1e-1,
    'Subtle': 1e0,
    'Weak': 1e1,
    'Small': 1e2,
    'Moderate': 1e3,
    'Strong': 1e4,
    'Intense': 1e5,
    'Extreme': 1e6,
    'Maximal': 1e7,
    # Intuitive aliases
    'Light': 1e2,       # = Small
    'Medium': 1e3,      # = Moderate
    'Heavy': 1e4,       # = Strong
    'VeryHeavy': 1e5,   # = Intense
}

#: Reference count scale of the noise model: level == NOISE_SCALE gives
#: SNR_peak = 1 (peak counts equal to shot noise).
NOISE_SCALE = 1e4


def level_to_peak_snr(level: float) -> float:
    """Peak-count SNR implied by a noise-severity ``level``.

    ``SNR_peak = 10^4 / level`` — the counting signal-to-noise ratio at
    the normalized signal maximum.  ``level=0`` (no noise) maps to
    ``inf``.
    """
    if level <= 0:
        return float('inf')
    return NOISE_SCALE / level


def level_to_peak_lambda(level: float) -> float:
    """Peak Poisson mean implied by a noise-severity ``level``.

    ``lambda_peak = (10^4 / level)^2`` — the Poisson mean assigned to
    the normalized signal maximum, so that ``SNR = sqrt(lambda)``
    reproduces :func:`level_to_peak_snr`.  ``level=0`` maps to ``inf``.
    """
    snr = level_to_peak_snr(level)
    return snr * snr


@dataclass
class GeneratorConfig:
    """Configuration for spectra generation (legacy compatibility)."""
    # Energy axis (now using SpectralConfig defaults)
    energy_edge: float = 2.5        # Edge width around center (eV) - Core-level default
    energy_step: float = 0.05       # Energy step (eV) - Core-level default

    # Spectral parameters
    func_type: int = 1              # 1=Voigt, 2=PseudoVoigt, 3=Gauss, 4=Lorentz
    amplitude_scale: float = 1e6    # Base amplitude scaling

    # Background
    bg_type: int = 1                # 1=Shirley (default)
    bg_base: float = 0.001          # Low (default)
    bg_level: float = 0.01          # 10 * bg_base (Medium factor)
    bg_level_factor: float = 10.0   # Factor for bgParamA

    # Output
    max_components: int = 9         # Max components per file (default for < 100k pixels)


def voigt_profile(x: np.ndarray, center: float, sigma: float, gamma: float) -> np.ndarray:
    """
    Compute true Voigt profile using Faddeeva function.

    The Voigt profile is the convolution of Gaussian and Lorentzian:
        V(x) = Re[w(z)] / (σ√(2π))
    where w(z) is the Faddeeva function and z = (x - x0 + iγ) / (σ√2)
    """
    from scipy.special import wofz  # Faddeeva function

    # Compute complex argument for Faddeeva function
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))

    # Voigt profile = Real part of Faddeeva function, normalized
    profile = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

    return profile


def add_poisson_noise(
    data: np.ndarray,
    level: float,
    legacy_scale: bool = False,
    use_gaussian_approx: bool = True,
    gaussian_threshold: float = 20.0,
    global_max: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add Poisson (shot) noise.

    Poisson noise: var(X) = λ, so SNR = √λ

    New scale (normalized, SNR = 10000 / level):
        level=1     → SNR ≈ 10000 (negligible noise, ~1s measurement)
        level=10    → SNR ≈ 1000  (minimal noise)
        level=100   → SNR ≈ 100   (slight noise)
        level=1000  → SNR ≈ 10    (visible noise)
        level=1e4   → SNR ≈ 1     (very noisy, ~100μs measurement)
        level=1e5   → SNR ≈ 0.1   (mostly noise, ~10μs)
        level=1e6   → SNR ≈ 0.01  (destroyed, ~1μs measurement)

    Legacy scale (MATLAB compatible, legacy_scale=True):
        Iout = poissrnd(1e4 * Iin / noiseLevel)

    Heteroscedastic mode (global_max is not None):
        When global_max is provided, normalization uses this fixed value
        instead of per-batch data_max. This preserves intensity-dependent
        noise: bright pixels (high amplitude) get higher λ → better S/N,
        dark pixels get lower λ → worse S/N. This is physically correct
        for XPS where count rate ∝ signal intensity.

        Homoscedastic (default): λ_peak = (10000/level)² for ALL pixels
        Heteroscedastic:         λ_peak = (10000/level)² × (pixel_max/global_max)

    Args:
        data: Input intensity data
        level: Noise level (1 to 1e6, higher = more noise)
        legacy_scale: Use MATLAB-compatible scaling
        use_gaussian_approx: Use Gaussian approximation for large λ (much faster)
        gaussian_threshold: Use Gaussian when λ > threshold (default 20)
        global_max: If provided, use this as normalization maximum instead of
            per-call data_max. Enables heteroscedastic Poisson noise.
        rng: Optional seeded numpy Generator for reproducible noise.
            When provided, sampling uses this generator (bypassing the
            Numba parallel path, which has its own uncontrollable RNG
            state), so identical inputs + identical seed reproduce the
            identical noisy output.

    Returns:
        Noisy data
    """
    if level <= 0:
        return data

    if legacy_scale:
        # Original MATLAB-compatible scaling
        scaled = 1e4 * np.maximum(data, 0) / level
        _poisson = rng.poisson if rng is not None else np.random.poisson
        noisy = _poisson(scaled).astype(np.float32)
        noisy[noisy == 0] = 1
        return noisy * level / 1e4
    else:
        # New signal-normalized scaling: SNR = 10000 / level
        # λ = (10000 / level)² * normalized_signal
        # SNR = √λ = 10000 / level
        data_pos = np.maximum(data, 0)
        data_max = data_pos.max()
        if data_max == 0:
            return data

        # Use global_max for heteroscedastic mode
        norm_max = global_max if global_max is not None else data_max

        # Normalize to [0, 1] range (relative to norm_max)
        normalized = data_pos / norm_max

        # λ = (10000 / level)² * normalized
        snr_target = 10000.0 / level
        lambda_scale = snr_target * snr_target  # SNR² = λ for normalized signal
        scaled = lambda_scale * normalized

        # Clip to avoid overflow (1e15 allows for very low noise levels)
        scaled = np.clip(scaled, 0, 1e15)

        if rng is None and HAS_NUMBA and data.ndim == 2:
            # Numba parallel: ~11x faster than NumPy for both
            # Gaussian approx (large λ) and exact Poisson (small λ).
            # Skipped when a seeded rng is requested — the Numba path's
            # per-thread RNG state cannot be seeded reproducibly.
            noisy = np.empty_like(scaled, dtype=np.float32)
            use_gauss = (use_gaussian_approx
                         and lambda_scale > gaussian_threshold)
            _poisson_noise_numba(
                scaled.astype(np.float32), noisy, use_gauss)
        elif use_gaussian_approx and lambda_scale > gaussian_threshold:
            # Gaussian approximation: Poisson(λ) ≈ N(λ, √λ) for large λ
            # For λ > 20, the approximation error is < 1%
            std = np.sqrt(scaled)
            _rng = rng if rng is not None else np.random.default_rng()
            noisy = scaled + _rng.standard_normal(
                scaled.shape, dtype=np.float32) * std
            noisy = np.maximum(noisy, 0)
        else:
            # Exact Poisson sampling (slow for large arrays)
            _poisson = rng.poisson if rng is not None else np.random.poisson
            noisy = _poisson(scaled).astype(np.float64)

        # Scale back to original range
        return (noisy / lambda_scale * norm_max).astype(np.float32)


def add_poisson_noise_mlx(
    data: np.ndarray,
    level: float,
    force_gaussian: bool = False,
    global_max: float | None = None,
) -> np.ndarray:
    """
    Add Poisson noise using MLX (GPU acceleration).

    Uses Cornish-Fisher approximation: Gaussian + skewness correction.
    z_cf = z + (1/√λ)·(z²-1)/6 restores Poisson skewness at ~10% cost.
    For λ > 20 (most practical cases), error < 1%.
    For λ < 20 (very high noise), set force_gaussian=True to stay on GPU
    (slightly less accurate but much faster), or False for NumPy fallback.

    Args:
        data: Input intensity data (np.ndarray)
        level: Noise level (1 to 1e6, higher = more noise)
        force_gaussian: If True, always use Gaussian approx even for small λ.
            Useful for batch pipelines where speed matters more than exact
            Poisson statistics at extreme noise levels.
        global_max: If provided, use this as normalization maximum instead of
            per-call data_max. Enables heteroscedastic Poisson noise.

    Returns:
        Noisy data (np.ndarray)
    """
    if not HAS_MLX:
        return add_poisson_noise(data, level, use_gaussian_approx=True,
                                 global_max=global_max)

    if level <= 0:
        return data

    data_max = float(np.maximum(data, 0).max())
    if data_max == 0:
        return data

    # Use global_max for heteroscedastic mode
    norm_max = global_max if global_max is not None else data_max

    # λ = (10000 / level)² * normalized
    snr_target = 10000.0 / level
    lambda_scale = snr_target * snr_target

    if not force_gaussian and lambda_scale < 20.0:
        # Small λ: Gaussian approx is inaccurate, fall back to NumPy/Numba
        return add_poisson_noise(data, level, legacy_scale=False,
                                 use_gaussian_approx=True,
                                 global_max=global_max)

    # Full GPU path: single np→mlx transfer, all computation on GPU
    data_mx = mx.array(data, dtype=mx.float32)
    data_pos = mx.maximum(data_mx, 0)

    # Normalize and scale (all on GPU, lazy evaluation)
    scaled = (lambda_scale / norm_max) * data_pos
    del data_mx, data_pos
    scaled = mx.clip(scaled, 0, 1e9)

    # Cornish-Fisher approximation: Poisson(λ) ≈ N(λ, √λ) + skewness correction
    std = mx.sqrt(mx.maximum(scaled, 1.0))
    z = mx.random.normal(shape=scaled.shape, dtype=mx.float32)
    gamma1 = 1.0 / std  # skewness = 1/sqrt(lambda)
    z = z + gamma1 * (z * z - 1.0) / 6.0
    noisy = mx.maximum(scaled + std * z, 0)
    del scaled, std, z, gamma1

    # Scale back to original range
    result = noisy * (norm_max / lambda_scale)
    del noisy

    # Single eval + single mlx→np transfer
    mx.eval(result)
    out = np.array(result, dtype=np.float32)
    del result
    return out


def _poisson_noise_cf_mlx(data_mx: 'mx.array', level_mx: 'mx.array') -> 'mx.array':
    """Cornish-Fisher Poisson noise kernel (compilable, homoscedastic).

    All arguments and return values are mx.array so that mx.compile()
    can fuse intermediate operations into a single GPU kernel, avoiding
    per-operation memory round-trips.

    Without compilation, 8 element-wise ops each write ~200MB to DRAM.
    With mx.compile(), these fuse into 1-2 kernel launches → 4-5x speedup.
    """
    data_pos = mx.maximum(data_mx, 0)
    data_max = mx.max(data_pos)

    snr = 10000.0 / level_mx
    ls = snr * snr

    scaled = (ls / data_max) * data_pos
    scaled = mx.clip(scaled, 0, 1e9)

    # Cornish-Fisher approximation: Poisson(λ) ≈ N(λ,√λ) + skewness correction
    std = mx.sqrt(mx.maximum(scaled, 1.0))
    z = mx.random.normal(shape=data_mx.shape, dtype=mx.float32)
    gamma1 = 1.0 / std
    z = z + gamma1 * (z * z - 1.0) / 6.0
    noisy = mx.maximum(scaled + std * z, 0)

    return noisy * (data_max / ls)


def _poisson_noise_cf_hetero_mlx(
    data_mx: 'mx.array', level_mx: 'mx.array', gmax_mx: 'mx.array',
) -> 'mx.array':
    """Cornish-Fisher Poisson noise kernel (compilable, heteroscedastic).

    Uses a fixed global_max for normalization instead of per-batch max.
    This preserves intensity-dependent noise: bright pixels get higher λ
    → better S/N, dark pixels get lower λ → worse S/N.
    """
    data_pos = mx.maximum(data_mx, 0)

    snr = 10000.0 / level_mx
    ls = snr * snr

    scaled = (ls / gmax_mx) * data_pos
    scaled = mx.clip(scaled, 0, 1e9)

    # Cornish-Fisher approximation
    std = mx.sqrt(mx.maximum(scaled, 1.0))
    z = mx.random.normal(shape=data_mx.shape, dtype=mx.float32)
    gamma1 = 1.0 / std
    z = z + gamma1 * (z * z - 1.0) / 6.0
    noisy = mx.maximum(scaled + std * z, 0)

    return noisy * (gmax_mx / ls)


# Module-level compiled versions — compilation is deferred to the first
# call (LazyCompiled) so importing this module never touches the Metal
# compiler.  Fuses 8 element-wise ops into minimal GPU kernel launches
# (4-5x faster).
if HAS_MLX:
    from ._mlx_support import LazyCompiled
    _poisson_noise_cf_compiled = LazyCompiled(_poisson_noise_cf_mlx)
    _poisson_noise_cf_hetero_compiled = LazyCompiled(
        _poisson_noise_cf_hetero_mlx)
else:
    _poisson_noise_cf_compiled = None
    _poisson_noise_cf_hetero_compiled = None


def add_poisson_noise_mlx_fused(
    data_mx: 'mx.array',
    level: float,
    global_max: float | None = None,
) -> 'mx.array':
    """
    Add Poisson noise using MLX, taking and returning mx.array directly.

    Cornish-Fisher approximation for Poisson noise.  Unlike
    add_poisson_noise_mlx(), this avoids NumPy↔MLX round-trips:
    the caller keeps data on GPU throughout matmul→noise→fit.

    Uses mx.compile() to fuse element-wise operations into a single
    GPU kernel, eliminating intermediate DRAM writes (4-5x faster).

    Args:
        data_mx: Clean spectra on MLX (n_spectra, n_energy), float32
        level: Noise level (1 to 1e6, higher = more noise)
        global_max: If provided, use fixed normalization for heteroscedastic
            noise. Bright pixels get higher λ (better S/N).

    Returns:
        Noisy spectra as mx.array (same shape, lazy — not yet evaluated)
    """
    if level <= 0:
        return data_mx

    if _poisson_noise_cf_compiled is None:
        raise RuntimeError(
            'add_poisson_noise_mlx_fused requires a usable MLX backend '
            '(none detected — headless macOS or TOYOMACRO_DISABLE_MLX). '
            'Use add_poisson_noise (NumPy) instead.')

    level_mx = mx.array(level, dtype=mx.float32)
    if global_max is not None:
        gmax_mx = mx.array(global_max, dtype=mx.float32)
        return _poisson_noise_cf_hetero_compiled(data_mx, level_mx, gmax_mx)
    else:
        return _poisson_noise_cf_compiled(data_mx, level_mx)


def add_gaussian_noise(data: np.ndarray, std_fraction: float) -> np.ndarray:
    """
    Add Gaussian noise.

    Args:
        data: Input intensity data
        std_fraction: Noise std as fraction of max signal

    Returns:
        Noisy data
    """
    if std_fraction <= 0:
        return data

    std = std_fraction * np.max(data)
    noisy = data + np.random.normal(0, std, data.shape).astype(np.float32)
    return np.maximum(noisy, 0)  # Clip negative values


def add_mixed_noise(
    data: np.ndarray,
    poisson_level: float,
    gaussian_std: float,
    global_max: float | None = None,
) -> np.ndarray:
    """
    Add mixed Poisson + Gaussian noise.

    Args:
        data: Input intensity data
        poisson_level: Poisson noise level
        gaussian_std: Gaussian noise std fraction
        global_max: If provided, use for heteroscedastic Poisson noise.

    Returns:
        Noisy data
    """
    result = data
    if poisson_level > 0:
        result = add_poisson_noise(result, poisson_level, global_max=global_max)
    if gaussian_std > 0:
        result = add_gaussian_noise(result, gaussian_std)
    return result


def decompose_image_to_amplitudes(
    image: np.ndarray,
    color_mapping: np.ndarray,
    method: Literal['pinv', 'lsq', 'nnls'] = 'pinv',
) -> np.ndarray:
    """
    Decompose RGB image to element amplitudes using color mapping.

    Based on MATLAB DepthProfiler.m line 3911-3950

    Args:
        image: (H, W, 3) RGB image, values 0-255 or 0-1
        color_mapping: (3, n_elements) RGB color coefficients
        method: Decomposition method
            'pinv': Moore-Penrose pseudo-inverse (fast, can have negatives)
            'lsq': Least squares (balanced)
            'nnls': Non-negative least squares (physically meaningful)

    Returns:
        amplitudes: (n_elements, H*W) element amplitudes
    """
    H, W = image.shape[:2]
    n_pixels = H * W
    n_elements = color_mapping.shape[1]

    # Normalize image to [0, 1]
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0

    # Reshape to (3, n_pixels) in column-major order for MATLAB compatibility
    rgb = image.reshape(n_pixels, 3, order='F').T  # (3, n_pixels)

    if method == 'pinv':
        # Moore-Penrose pseudo-inverse
        pinv_col = np.linalg.pinv(color_mapping)  # (n_elements, 3)
        amplitudes = pinv_col @ rgb  # (n_elements, n_pixels)

    elif method == 'lsq':
        # Least squares per pixel
        amplitudes = np.zeros((n_elements, n_pixels), dtype=np.float32)
        C = color_mapping.T  # (3, n_elements) -> (n_elements, 3).T for lstsq
        for i in range(n_pixels):
            amplitudes[:, i], _, _, _ = np.linalg.lstsq(
                color_mapping.T, rgb[:, i], rcond=None
            )

    elif method == 'nnls':
        # Non-negative least squares (slower but physically meaningful)
        from scipy.optimize import nnls
        amplitudes = np.zeros((n_elements, n_pixels), dtype=np.float32)
        C = color_mapping.T
        for i in range(n_pixels):
            amplitudes[:, i], _ = nnls(C, rgb[:, i])

    # Scale to [0, 1] range
    amplitudes = amplitudes / (amplitudes.max() + 1e-10)

    return amplitudes.astype(np.float32)


class SpectraGenerator:
    """
    Generate synthetic XPS spectra from images.

    Example usage:
        generator = SpectraGenerator(
            output_dir='/path/to/output',
            image_path='/path/to/image.jpg',
        )
        generator.configure_fuji_elements()
        generator.generate(noise_config=NoiseConfig(poisson_level=1e3))
    """

    def __init__(
        self,
        output_dir: str | Path,
        image_path: str | Path,
        project_name: str | None = None,
        config: GeneratorConfig | None = None,
    ):
        """
        Initialize generator.

        Args:
            output_dir: Output directory for H5 files
            image_path: Path to source image
            project_name: Project name for file naming (auto-generated if None)
            config: Generator configuration
        """
        self.output_dir = Path(output_dir)
        self.image_path = Path(image_path)
        self.config = config or GeneratorConfig()

        # Load image
        self.image = open_frames(image_path)[0]
        self.height, self.width = self.image.shape[:2]
        self.n_pixels = self.height * self.width

        # Auto-generate project name
        if project_name is None:
            timestamp = time.strftime('%y%m%d')
            self.project_name = f"{timestamp}_{self.image_path.stem}"
        else:
            self.project_name = project_name

        # Element configurations
        self.elements: list[ElementSpec] = []
        self.color_mapping: np.ndarray | None = None
        self.component_order: list[tuple[str, int]] = []

        print(f"Loaded image: {self.width} x {self.height} ({self.n_pixels:,} pixels)")

    def configure_elements(self, preset: Union[str, 'ElementPreset'] = 'fuji') -> None:
        """
        Configure element specs, color mapping, and component order.

        Args:
            preset: Preset name (e.g., 'fuji') or an ElementPreset object.
                    Default is 'fuji' (6-component Si/Ti/Al/C/O mapping).

        Example:
            generator.configure_elements('fuji')           # built-in preset
            generator.configure_elements(my_custom_preset) # custom ElementPreset
        """
        if isinstance(preset, str):
            preset = get_element_preset(preset)

        self.elements = list(preset.elements)
        self.color_mapping = preset.color_mapping.copy()
        self.component_order = list(preset.component_order)

        print(f"Configured {len(self.elements)} element components (preset: {preset.name})")

    def configure_fuji_elements(self) -> None:
        """Configure for Fuji 8K test data with standard 6 components.

        Backward-compatible wrapper around configure_elements('fuji').
        """
        self.configure_elements('fuji')

    def _create_output_dir(self, noise_config: NoiseConfig) -> Path:
        """Create output directory with noise info in name."""
        dir_name = f"{self.project_name}_{noise_config}"
        output_path = self.output_dir / dir_name
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def _generate_energy_axis(self, center: float) -> np.ndarray:
        """Generate energy axis around center."""
        edge = self.config.energy_edge
        step = self.config.energy_step
        return np.arange(center - edge, center + edge + step, step, dtype=np.float32)

    def _generate_voigt_spectrum(
        self,
        energy: np.ndarray,
        amplitude: float,
        center: float,
        sigma: float,
        gamma: float,
    ) -> np.ndarray:
        """Generate single Voigt spectrum."""
        profile = voigt_profile(energy, center, sigma, gamma)
        # Normalize and scale
        profile = profile / (profile.max() + 1e-10)
        return (amplitude * self.config.amplitude_scale * profile).astype(np.float32)

    def _generate_voigt_spectra_batch(
        self,
        energy: np.ndarray,
        amplitudes: np.ndarray,
        center: float,
        sigma: float,
        gamma: float,
    ) -> np.ndarray:
        """
        Generate Voigt spectra for a batch of amplitudes (vectorized).

        Args:
            energy: (n_energy,) energy axis
            amplitudes: (n_spectra,) amplitude values
            center, sigma, gamma: Voigt parameters

        Returns:
            spectra: (n_spectra, n_energy) spectra
        """
        # Compute base profile once
        profile = voigt_profile(energy, center, sigma, gamma)
        profile = profile / (profile.max() + 1e-10)

        # Broadcast: (n_spectra, 1) * (n_energy,) -> (n_spectra, n_energy)
        spectra = amplitudes[:, np.newaxis] * self.config.amplitude_scale * profile[np.newaxis, :]
        return spectra.astype(np.float32)

    def _add_background(
        self,
        spectrum: np.ndarray,
        amplitude: float,
    ) -> np.ndarray:
        """Add flat background to spectrum."""
        bg = amplitude * self.config.amplitude_scale * self.config.bg_base
        return spectrum + bg

    def generate(
        self,
        noise_config: NoiseConfig | None = None,
        decomposition_method: str = 'pinv',
        batch_size: int = 100_000,
        verbose: bool = True,
        create_subdir: bool = True,
    ) -> Path:
        """
        Generate spectra and save to H5 files.

        Args:
            noise_config: Noise configuration (None = no noise)
            decomposition_method: 'pinv', 'lsq', or 'nnls'
            batch_size: Spectra per batch for memory efficiency
            verbose: Print progress
            create_subdir: If True, create subdirectory for output (default).
                          If False, write directly to output_dir.

        Returns:
            Path to output directory
        """
        if not self.elements:
            raise ValueError("No elements configured. Call configure_* first.")

        noise_config = noise_config or NoiseConfig()
        if create_subdir:
            output_path = self._create_output_dir(noise_config)
        else:
            output_path = self.output_dir
            output_path.mkdir(parents=True, exist_ok=True)

        if verbose:
            print("=" * 70)
            print(f"Generating spectra: {self.project_name}")
            print(f"Noise: {noise_config}")
            print(f"Output: {output_path}")
            print("=" * 70)

        # Decompose image to element amplitudes
        if verbose:
            print("\nDecomposing image to element amplitudes...")

        amplitudes = decompose_image_to_amplitudes(
            self.image, self.color_mapping, method=decomposition_method
        )

        if verbose:
            print(f"  Amplitudes shape: {amplitudes.shape}")
            for i, elem in enumerate(self.elements):
                amp = amplitudes[i, :]
                print(f"  {elem.symbol}{elem.orbital}: min={amp.min():.4f}, max={amp.max():.4f}")

        # Group elements by file (Si has 2 components)
        file_groups: dict[str, list[int]] = {}
        for i, elem in enumerate(self.elements):
            key = elem.file_key
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(i)

        # Generate spectra for each file
        start_time = time.perf_counter()

        for file_key, elem_indices in file_groups.items():
            if verbose:
                print(f"\n--- {file_key} ({len(elem_indices)} components) ---")

            # Get element specs for this file
            elem_specs = [self.elements[i] for i in elem_indices]
            n_comp = len(elem_specs)

            # Energy axis - cover all peaks with edge margin
            # For multiple components, ensure all peaks are within range
            # Use float64 for arange to avoid cumulative rounding errors,
            # then convert to float32 for H5 storage
            binding_energies = [e.binding_energy for e in elem_specs]
            e_min = min(binding_energies) - self.config.energy_edge
            e_max = max(binding_energies) + self.config.energy_edge
            energy = np.arange(e_min, e_max + self.config.energy_step / 2,
                              self.config.energy_step, dtype=np.float64)
            energy = energy.astype(np.float32)
            n_energy = len(energy)

            # Create H5 file
            h5_filename = f"{file_key}_{self.project_name}.h5"
            h5_path = output_path / h5_filename

            # Determine energy range and binding energy sign
            xRangeMin = energy.min()
            xRangeMax = energy.max()

            # Use max_components from config (default 9 for MATLAB compatibility)
            max_comp = self.config.max_components

            with h5py.File(h5_path, 'w') as f:
                # Create datasets - MATLAB compatible format
                # specdata: (N+1, energy) - first row is energy axis
                f.create_dataset('specdata', shape=(self.n_pixels + 1, n_energy),
                               dtype='float32')
                # fitpara: (N, maxcomp, 9) - comp dimension uses max_components (default 9)
                f.create_dataset('fitpara', shape=(self.n_pixels, max_comp, 9),
                               dtype='float32')
                # xytdata: (3, N) - coordinates
                f.create_dataset('xytdata', shape=(3, self.n_pixels),
                               dtype='float32')
                # otherpara: (10, N) - MATLAB format is transposed!
                f.create_dataset('otherpara', shape=(10, self.n_pixels),
                               dtype='float32')

                # Create misc group for metadata
                # MATLAB compatibility: use variable-length UTF-8 string type
                # MATLAB h5create(fp, '/misc/xdata_str', 1, 'Datatype', 'string')
                # creates H5T_VARIABLE length string datasets
                misc = f.create_group('misc')
                dt_vlen_str = h5py.string_dtype(encoding='utf-8')
                misc.create_dataset('xdata_str', data='Horizontal', dtype=dt_vlen_str)
                misc.create_dataset('ydata_str', data='Vertical', dtype=dt_vlen_str)
                misc.create_dataset('tdata_str', data='Time', dtype=dt_vlen_str)
                misc.create_dataset('numberofslice', data=np.array(1, dtype='uint16'))
                misc.create_dataset('fermienergy', data=np.array(0.0, dtype='float32'))
                misc.create_dataset('bindingenergysign', data=np.array(1, dtype='uint8'))
                misc.create_dataset('sodeconv', data=np.array(0, dtype='uint8'))
                misc.create_dataset('maxcomp', data=np.array(n_comp, dtype='uint8'))

                # Write energy axis
                f['specdata'][0, :] = energy

                # Write xytdata (x, y coordinates)
                x_coords = np.tile(np.arange(1, self.width + 1), self.height)
                y_coords = np.repeat(np.arange(self.height, 0, -1), self.width)
                f['xytdata'][0, :] = x_coords
                f['xytdata'][1, :] = y_coords
                f['xytdata'][2, :] = 1  # t = 1

                # Pre-compute global_max for heteroscedastic Poisson noise.
                # This ensures all batches use the same normalization, so
                # bright pixels get higher λ (better S/N) and dark pixels
                # get lower λ (worse S/N), matching real XPS physics.
                global_max_val = None
                if (noise_config.noise_type in ('poisson', 'mixed')
                        and noise_config.heteroscedastic):
                    # Estimate max spectrum intensity from max amplitude sum
                    # across all pixels: max_spectrum ≈ max_sI0 × (1 + bg_base)
                    # since each component profile is normalized to peak=1
                    max_sI0 = 0.0
                    for elem_idx in elem_indices:
                        max_sI0 += np.maximum(amplitudes[elem_idx], 0).max()
                    max_sI0 *= self.config.amplitude_scale
                    # Spectrum max ≈ max single component peak + background
                    # More precisely, iterate to find the pixel with max total
                    # For simplicity, use sum of all component maxima + bg
                    global_max_val = float(max_sI0 * (1.0 + self.config.bg_base))
                    if verbose:
                        print(f"  Heteroscedastic mode: global_max = {global_max_val:.2f}")

                # Generate spectra in batches
                for batch_start in range(0, self.n_pixels, batch_size):
                    batch_end = min(batch_start + batch_size, self.n_pixels)
                    batch_size_actual = batch_end - batch_start

                    # Initialize batch arrays
                    spectra_batch = np.zeros((batch_size_actual, n_energy), dtype=np.float32)
                    # fitpara: (batch, maxcomp, 9) - comp uses max_components
                    # Unused components should be NaN (MATLAB convention)
                    fitpara_batch = np.full((batch_size_actual, max_comp, 9), np.nan, dtype=np.float32)
                    # comp_spectra_batch: store individual component spectra for Area calculation
                    comp_spectra_batch = np.zeros((n_comp, batch_size_actual, n_energy), dtype=np.float32)
                    otherpara_batch = np.zeros((10, batch_size_actual), dtype=np.float32)

                    # Generate spectra for each component (vectorized)
                    for j, (elem_idx, elem) in enumerate(zip(elem_indices, elem_specs)):
                        amp_values = amplitudes[elem_idx, batch_start:batch_end]

                        # Clip negative amplitudes to zero (from image decomposition artifacts)
                        amp_values_clipped = np.maximum(amp_values, 0)

                        # Generate Voigt spectra (batch)
                        comp_spectra = self._generate_voigt_spectra_batch(
                            energy, amp_values_clipped,
                            elem.binding_energy, elem.sigma, elem.gamma
                        )
                        spectra_batch += comp_spectra
                        comp_spectra_batch[j, :, :] = comp_spectra  # Store for Area calculation

                        # Store fitpara (vectorized)
                        # Format: (N, maxcomp, 9) = [Int0, E0, GW, LW, alpha, BR, SO, ft, Area]
                        fitpara_batch[:, j, 0] = amp_values_clipped * self.config.amplitude_scale  # Int0
                        fitpara_batch[:, j, 1] = elem.binding_energy                        # E0
                        fitpara_batch[:, j, 2] = elem.gaussian_fwhm                         # GW (FWHM)
                        fitpara_batch[:, j, 3] = elem.lorentzian_fwhm                       # LW (FWHM)
                        fitpara_batch[:, j, 4] = elem.asymmetry                             # alpha
                        fitpara_batch[:, j, 5] = elem.branching_ratio                       # BR
                        fitpara_batch[:, j, 6] = elem.spinorbit_split                       # SO
                        fitpara_batch[:, j, 7] = self.config.func_type                      # ft
                        # Area = abs(trapz(energy, comp_spectrum)) - calculated below

                    # Calculate Area for each component: abs(trapz(energy, comp_spectrum))
                    for j in range(n_comp):
                        fitpara_batch[:, j, 8] = np.abs(np.trapezoid(
                            comp_spectra_batch[j, :, :], energy, axis=1
                        ))

                    # Calculate sI0 = sum of all peak intensities for each pixel
                    # MATLAB: sI0 = sum(par.Int0); a = sI0*BGlevel; b = sI0*BGbase;
                    # fitpara shape is (batch, maxcomp, 9), so Int0 is fitpara_batch[:, :, 0]
                    sI0 = fitpara_batch[:, :n_comp, 0].sum(axis=1)  # sum over components -> (batch,)

                    # Add background based on sI0
                    # bg_actual = sI0 * BGbase for each pixel
                    bg_actual = sI0 * self.config.bg_base  # (batch_size,)
                    spectra_batch += bg_actual[:, np.newaxis]  # broadcast to (batch_size, n_energy)

                    # Add noise
                    if noise_config.noise_type == 'poisson':
                        spectra_batch = add_poisson_noise(
                            spectra_batch, noise_config.poisson_level,
                            global_max=global_max_val)
                    elif noise_config.noise_type == 'gaussian':
                        spectra_batch = add_gaussian_noise(spectra_batch, noise_config.gaussian_std)
                    elif noise_config.noise_type == 'mixed':
                        spectra_batch = add_mixed_noise(
                            spectra_batch, noise_config.poisson_level,
                            noise_config.gaussian_std,
                            global_max=global_max_val)

                    # Calculate Chi2: sum(background^2) / length(energy)
                    # Background = sI0 * BGbase (flat background we added)
                    chi2 = (bg_actual ** 2) / n_energy

                    # Store otherpara - (10, N)
                    # [numComponents, xRangeMin, xRangeMax, bgType, bgParamA, bgParamB, bgParamC, maxHeight, totalArea, chi2]
                    # MATLAB: a = sI0*BGlevel; b = sI0*BGbase; c = 1;
                    otherpara_batch[0, :] = n_comp                           # numComponents
                    otherpara_batch[1, :] = xRangeMin                        # xRangeMin
                    otherpara_batch[2, :] = xRangeMax                        # xRangeMax
                    otherpara_batch[3, :] = self.config.bg_type              # bgType
                    otherpara_batch[4, :] = sI0 * self.config.bg_level       # bgParamA = sI0 * BGlevel
                    otherpara_batch[5, :] = sI0 * self.config.bg_base        # bgParamB = sI0 * BGbase
                    otherpara_batch[6, :] = 1                                # bgParamC
                    otherpara_batch[7, :] = spectra_batch.max(axis=1)        # maxHeight
                    otherpara_batch[8, :] = np.trapezoid(spectra_batch, energy, axis=1)  # totalArea
                    otherpara_batch[9, :] = chi2                             # chi2

                    # Write to file
                    f['specdata'][batch_start + 1:batch_end + 1, :] = spectra_batch
                    f['fitpara'][batch_start:batch_end, :, :] = fitpara_batch
                    f['otherpara'][:, batch_start:batch_end] = otherpara_batch

                    if verbose and (batch_start // batch_size) % 10 == 0:
                        progress = (batch_end / self.n_pixels) * 100
                        print(f"  Progress: {progress:.1f}%")

            if verbose:
                print(f"  Saved: {h5_path}")

        elapsed = time.perf_counter() - start_time

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Generation complete: {elapsed:.1f}s")
            print(f"Output: {output_path}")
            print("=" * 70)

        # Save project info
        self._save_project_info(output_path, noise_config)

        return output_path

    def generate_inmemory(
        self,
        noise_config: NoiseConfig | None = None,
        decomposition_method: str = 'pinv',
        verbose: bool = True,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Generate spectra in memory (no H5 I/O).

        Returns dict keyed by file_key, each containing:
            'energy':  (n_energy,) float32
            'spectra': (n_spectra, n_energy) float32
            'amplitudes': (n_comp, n_spectra) float32  (ground truth)

        This is typically 2-5x faster than generate() because it avoids
        H5 file creation, writes, and subsequent reads.

        Args:
            noise_config: Noise configuration (None = no noise)
            decomposition_method: 'pinv', 'lsq', or 'nnls'
            verbose: Print progress

        Returns:
            Dict[file_key, {'energy', 'spectra', 'amplitudes'}]
        """
        if not self.elements:
            raise ValueError("No elements configured. Call configure_* first.")

        noise_config = noise_config or NoiseConfig()

        if verbose:
            print("=" * 70)
            print(f"Generating spectra (in-memory): {self.project_name}")
            print(f"Noise: {noise_config}")
            print("=" * 70)

        # Decompose image to element amplitudes
        amplitudes = decompose_image_to_amplitudes(
            self.image, self.color_mapping, method=decomposition_method
        )

        if verbose:
            print(f"  Amplitudes shape: {amplitudes.shape}")

        # Group elements by file
        file_groups: dict[str, list[int]] = {}
        for i, elem in enumerate(self.elements):
            key = elem.file_key
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(i)

        result = {}
        start_time = time.perf_counter()

        for file_key, elem_indices in file_groups.items():
            elem_specs = [self.elements[i] for i in elem_indices]
            n_comp = len(elem_specs)

            # Energy axis
            binding_energies = [e.binding_energy for e in elem_specs]
            e_min = min(binding_energies) - self.config.energy_edge
            e_max = max(binding_energies) + self.config.energy_edge
            energy = np.arange(e_min, e_max + self.config.energy_step / 2,
                              self.config.energy_step, dtype=np.float64)
            energy = energy.astype(np.float32)
            n_energy = len(energy)

            if verbose:
                print(f"\n--- {file_key} ({n_comp} components, {n_energy} channels) ---")

            # Generate all spectra at once (no batching needed for memory-only)
            spectra_all = np.zeros((self.n_pixels, n_energy), dtype=np.float32)
            amp_gt = np.zeros((n_comp, self.n_pixels), dtype=np.float32)

            for j, (elem_idx, elem) in enumerate(zip(elem_indices, elem_specs)):
                amp_values = amplitudes[elem_idx, :]
                amp_values_clipped = np.maximum(amp_values, 0).astype(np.float32)
                amp_gt[j, :] = amp_values_clipped

                comp_spectra = self._generate_voigt_spectra_batch(
                    energy, amp_values_clipped,
                    elem.binding_energy, elem.sigma, elem.gamma
                )
                spectra_all += comp_spectra

            # Add background
            sI0 = amp_gt.sum(axis=0) * self.config.amplitude_scale
            bg_actual = sI0 * self.config.bg_base
            spectra_all += bg_actual[:, np.newaxis]

            # Compute global_max for heteroscedastic mode
            global_max_val = None
            if (noise_config.noise_type in ('poisson', 'mixed')
                    and noise_config.heteroscedastic):
                global_max_val = float(spectra_all.max())
                if verbose:
                    print(f"  Heteroscedastic mode: global_max = {global_max_val:.2f}")

            # Add noise
            if noise_config.noise_type == 'poisson':
                spectra_all = add_poisson_noise(
                    spectra_all, noise_config.poisson_level,
                    global_max=global_max_val)
            elif noise_config.noise_type == 'gaussian':
                spectra_all = add_gaussian_noise(spectra_all, noise_config.gaussian_std)
            elif noise_config.noise_type == 'mixed':
                spectra_all = add_mixed_noise(
                    spectra_all, noise_config.poisson_level,
                    noise_config.gaussian_std,
                    global_max=global_max_val)

            result[file_key] = {
                'energy': energy,
                'spectra': spectra_all,
                'amplitudes': amp_gt,
            }

            if verbose:
                print(f"  Spectra: {spectra_all.shape}, Memory: {spectra_all.nbytes / 1e6:.1f} MB")

        elapsed = time.perf_counter() - start_time
        if verbose:
            total_mem = sum(v['spectra'].nbytes + v['amplitudes'].nbytes
                          for v in result.values()) / 1e6
            print(f"\nGeneration complete: {elapsed:.1f}s, Total memory: {total_mem:.1f} MB")

        return result

    def _save_project_info(self, output_path: Path, noise_config: NoiseConfig) -> None:
        """Save project information to H5 file."""
        info_path = output_path / f"{self.project_name}.h5"

        with h5py.File(info_path, 'w') as f:
            # ezdeprof group for compatibility
            ezdeprof = f.create_group('ezdeprof')

            # Color mapping
            ezdeprof.create_dataset('Col', data=self.color_mapping)

            # Element info
            n_elem = len(self.elements)
            dt = h5py.special_dtype(vlen=str)

            element_names = np.array([[f"{e.symbol}({i})"] for i, e in enumerate(self.elements)], dtype=object)
            symbols = np.array([[e.symbol] for e in self.elements], dtype=object)
            orbitals = np.array([[e.orbital] for e in self.elements], dtype=object)

            ezdeprof.create_dataset('element', data=element_names.astype('S20'))
            ezdeprof.create_dataset('elementsymbol', data=symbols.astype('S10'))
            ezdeprof.create_dataset('orbital', data=orbitals.astype('S10'))

            # Noise config
            noise_grp = f.create_group('noise_config')
            noise_grp.attrs['type'] = noise_config.noise_type
            noise_grp.attrs['poisson_level'] = noise_config.poisson_level
            noise_grp.attrs['gaussian_std'] = noise_config.gaussian_std


def generate_noise_sweep(
    image_path: str | Path,
    output_base_dir: str | Path,
    poisson_levels: list[float] = [0, 1e2, 1e3, 1e4, 1e5],
    gaussian_std: float = 0.01,
    project_name: str | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Generate spectra with multiple noise levels for PSNR testing.

    Args:
        image_path: Path to source image
        output_base_dir: Base output directory
        poisson_levels: List of Poisson noise levels to test
        gaussian_std: Fixed Gaussian noise std (added to all)
        project_name: Project name
        verbose: Print progress

    Returns:
        Dict mapping noise description to output path
    """
    results = {}

    for poisson_level in poisson_levels:
        if poisson_level == 0 and gaussian_std == 0:
            noise_config = NoiseConfig(noise_type='none')
        elif poisson_level == 0:
            noise_config = NoiseConfig(
                gaussian_std=gaussian_std,
                noise_type='gaussian',
            )
        elif gaussian_std == 0:
            noise_config = NoiseConfig(
                poisson_level=poisson_level,
                noise_type='poisson',
            )
        else:
            noise_config = NoiseConfig(
                poisson_level=poisson_level,
                gaussian_std=gaussian_std,
                noise_type='mixed',
            )

        generator = SpectraGenerator(
            output_dir=output_base_dir,
            image_path=image_path,
            project_name=project_name,
        )
        generator.configure_fuji_elements()

        output_path = generator.generate(
            noise_config=noise_config,
            verbose=verbose,
        )

        results[str(noise_config)] = output_path

    return results


def generate_with_noise_levels(
    image_path: str | Path,
    output_base_dir: str | Path,
    noise_levels: list[str] | None = None,
    project_name: str | None = None,
    spectral_config: SpectralConfig | None = None,
    verbose: bool = True,
) -> Path:
    """
    Generate spectra with hierarchical folder structure by noise level.

    Creates MATLAB-compatible folder structure:
        日付時刻/
        ├── NoiseDegree_None/
        │   ├── Si1s_*.h5
        │   ├── O1s_*.h5
        │   └── ...
        ├── NoiseDegree_Minimal/
        │   └── ...
        └── ...

    Args:
        image_path: Path to source image
        output_base_dir: Base output directory
        noise_levels: List of noise level names (default: all levels)
                     e.g., ['None', 'Minimal', 'Moderate', 'Strong']
        project_name: Project name (auto-generated if None)
        spectral_config: Spectral configuration (uses defaults if None)
        verbose: Print progress

    Returns:
        Path to root output directory (日付時刻/)
    """
    # Default: use all noise levels for comprehensive testing
    if noise_levels is None:
        noise_levels = list(NOISE_LEVELS.keys())

    # Auto-generate timestamp-based project name
    timestamp = time.strftime('%y%m%d_%H%M%S')
    if project_name is None:
        image_stem = Path(image_path).stem
        project_name = f"{image_stem}"

    # Create root directory: 日付時刻/
    root_dir = Path(output_base_dir) / timestamp
    root_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("Generating spectra with hierarchical folder structure")
        print(f"Root: {root_dir}")
        print(f"Noise levels: {', '.join(noise_levels)}")
        print("=" * 70)

    # Use default spectral config if not provided
    spectral_config = spectral_config or SpectralConfig()

    # Generate for each noise level
    for noise_name in noise_levels:
        noise_value = NOISE_LEVELS.get(noise_name, 0)

        # Create subdirectory for this noise level
        noise_dir = root_dir / f"NoiseDegree_{noise_name}"
        noise_dir.mkdir(exist_ok=True)

        if verbose:
            print(f"\n--- NoiseDegree: {noise_name} ({noise_value:.0e}) ---")

        # Configure noise
        if noise_value == 0:
            noise_config = NoiseConfig(noise_type='none')
        else:
            # Use spectral_config.NoiseType to determine noise type
            if spectral_config.NoiseType == 1:
                noise_config = NoiseConfig(
                    poisson_level=noise_value,
                    noise_type='poisson'
                )
            elif spectral_config.NoiseType == 2:
                noise_config = NoiseConfig(
                    gaussian_std=noise_value / 1e6,  # Scale for Gaussian
                    noise_type='gaussian'
                )
            else:
                noise_config = NoiseConfig(
                    poisson_level=noise_value,
                    noise_type='poisson'
                )

        # Create generator with updated config
        gen_config = GeneratorConfig(
            energy_edge=spectral_config.engedge,
            energy_step=spectral_config.engstep,
            func_type=spectral_config.FCtype,
            amplitude_scale=spectral_config.amplitude_scale,
            bg_type=spectral_config.BGtype,
            bg_base=spectral_config.BGbase,
            bg_level=spectral_config.BGlevel,
        )

        generator = SpectraGenerator(
            output_dir=noise_dir,
            image_path=image_path,
            project_name=project_name,
            config=gen_config,
        )
        generator.configure_fuji_elements()

        # Generate (files go directly into noise_dir, not a subdirectory)
        generator.generate(
            noise_config=noise_config,
            verbose=verbose,
            create_subdir=False,  # Don't create another subdirectory
        )

    if verbose:
        print(f"\n{'=' * 70}")
        print("All generations complete!")
        print(f"Root directory: {root_dir}")
        print(f"Subdirectories: {len(noise_levels)}")
        print("=" * 70)

    return root_dir


def create_prj_file(
    output_dir: Path,
    file_keys: list[str],
    project_name: str,
) -> Path:
    """
    Create MATLAB .prj file for compatibility.

    Based on DepthProfiler.m line 4396-4401

    Args:
        output_dir: Output directory
        file_keys: List of file keys (e.g., ['Si1s', 'O1s', 'C1s'])
        project_name: Project name

    Returns:
        Path to .prj file
    """
    import platform

    import scipy.io as sio

    prj_path = output_dir / f"{project_name}.prj"

    # Prepare data for .mat format
    data_txt = [f"{key}_{project_name}" for key in file_keys]
    pathname = [str(output_dir) + '/' for _ in file_keys]

    prj_data = {
        'data_txt': np.array(data_txt, dtype=object),
        'pathname': np.array(pathname, dtype=object),
        'isComputer': platform.system(),
        'fileSep': '/',
        'fittingspeed': np.array(['0' for _ in file_keys], dtype=object),
        'calculationtime': np.array(['0' for _ in file_keys], dtype=object),
        'remainingtime': np.array(['0' for _ in file_keys], dtype=object),
    }

    sio.savemat(prj_path, prj_data)
    return prj_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate XPS spectra from images')
    parser.add_argument('--image', type=str, required=True, help='Path to source image')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--noise-levels', type=str, nargs='+',
                        default=['None', 'Moderate', 'Strong'],
                        choices=list(NOISE_LEVELS.keys()),
                        help='Noise levels to generate')
    parser.add_argument('--single-noise', type=str, default=None,
                        choices=list(NOISE_LEVELS.keys()),
                        help='Generate single noise level (legacy mode)')

    args = parser.parse_args()

    if args.single_noise:
        # Legacy mode: single noise level
        noise_value = NOISE_LEVELS[args.single_noise]
        if noise_value == 0:
            noise_config = NoiseConfig(noise_type='none')
        else:
            noise_config = NoiseConfig(poisson_level=noise_value, noise_type='poisson')

        generator = SpectraGenerator(
            output_dir=args.output,
            image_path=args.image,
        )
        generator.configure_fuji_elements()

        output_path = generator.generate(noise_config=noise_config)
        print(f"\nOutput: {output_path}")
    else:
        # New mode: hierarchical folders with multiple noise levels
        root_dir = generate_with_noise_levels(
            image_path=args.image,
            output_base_dir=args.output,
            noise_levels=args.noise_levels,
        )
        print(f"\nRoot output: {root_dir}")
