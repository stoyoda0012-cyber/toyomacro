"""
3-Parameter Roundtrip Benchmark: Image → 3-Param Spectra → 4-Step Fit → Image

End-to-end evaluation of the 4-step residual projection pipeline for
recovering amplitude, center shift (δE), and Gaussian width (FWHM) from
single-peak Voigt spectra.

Encoding:
    R = amplitude [0, 1]
    G = center shift δE [-1, +1] eV
    B = Gaussian FWHM [0.7, 1.5] eV

Pipeline:
    1. RGB Image → SinglePeakEncoder → (amp, δE, FWHM)
    2. Taylor Voigt spectra: amp × (Φ + δE·J_c + δσ·J_σ) + background
    3. Optional Poisson noise injection
    4. 4-step recovery: process_rowmajor_extended_3param()
    5. Decode → RGB image
    6. Per-channel PSNR evaluation (R/G/B independently)

Usage:
    python -m toyomacro.voigtfit.benchmarks.param_roundtrip_benchmark \\
        --image /path/to/image.jpg \\
        --noise-levels None Moderate Strong \\
        --output /path/to/outputs/
"""

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toyomacro.voigtfit.image_utils import load_image
from toyomacro.voigtfit.param_encoder import (
    C1S_SINGLE_PRESET,
    SinglePeakEncoder,
    SinglePeakPreset,
)
from toyomacro.voigtfit.pipeline import HybridPipeline
from toyomacro.voigtfit.spectra_generator import NOISE_LEVELS
from toyomacro.voigtfit.weight_cache import WeightMatrixCache

try:
    import mlx.core as mx

    from toyomacro.voigtfit._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND the default device can execute work
except ImportError:
    HAS_MLX = False


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ChannelPSNR:
    """Per-channel PSNR results."""
    amplitude: float   # R channel PSNR (dB)
    shift: float       # G channel PSNR (dB)
    fwhm: float        # B channel PSNR (dB)
    total: float       # Mean of three channels

    def __str__(self):
        return (f"R(amp)={self.amplitude:.1f}dB  "
                f"G(δE)={self.shift:.1f}dB  "
                f"B(FWHM)={self.fwhm:.1f}dB  "
                f"avg={self.total:.1f}dB")


@dataclass
class ParamRoundtripResult:
    """Result from a single 3-parameter roundtrip run."""
    noise_level: str
    noise_value: float
    image_shape: tuple[int, int]
    n_spectra: int

    # Timing
    gen_time: float
    fit_time: float
    total_time: float
    throughput: float  # spectra/sec for fit

    # Per-channel PSNR vs original image (quantization-limited for noise-free)
    psnr_vs_original: ChannelPSNR

    # Per-channel PSNR vs noise-free reconstruction (None for noise-free run)
    psnr_vs_noisefree: ChannelPSNR | None = None

    # Parameter-space metrics (before RGB quantization)
    amp_corr: float = 1.0
    shift_corr: float = 1.0
    fwhm_corr: float = 1.0
    amp_rmse: float = 0.0
    shift_rmse: float = 0.0
    fwhm_rmse: float = 0.0

    # Images for visualization
    original_image: np.ndarray | None = None
    reconstructed_image: np.ndarray | None = None


def parse_noise_level(s: str) -> str:
    """Parse noise level string, supporting 'lam' notation.

    'lamX' sets the dimensionless noise-severity *level* to 10^X.
    Despite the historical name, this is NOT a Poisson mean: peak-count
    SNR = 1e4/level and peak Poisson mean = (1e4/level)^2 (see
    spectra_generator.level_to_peak_snr / level_to_peak_lambda).

    'lam3' -> 'Moderate' (level 10^3 = 1000, peak SNR 10)
    'lam-1' -> level 0.1
    'lam3.5' -> 'Custom_3162' (level 10^3.5 ≈ 3162)
    'None', 'Moderate', 'Strong' -> passthrough
    """
    if s.startswith('lam'):
        exp = float(s[3:])
        val = 10 ** exp
        for name, nval in NOISE_LEVELS.items():
            if abs(nval - val) / max(val, 1e-10) < 1e-4:
                return name
        # No preset match — return custom name with value
        return f'Custom_{val:.0f}'
    return s


@dataclass
class ParamNoiseSweepResult:
    """Results from a multi-noise sweep."""
    results: dict[str, ParamRoundtripResult]
    preset_name: str
    image_path: str

    def save_npz(self, path: str | Path) -> None:
        """Save results to NPZ for fast re-plotting."""
        data = {}
        names = []
        for name, r in self.results.items():
            names.append(name)
            p = r.psnr_vs_original
            data[f'{name}_psnr_orig'] = np.array([p.amplitude, p.shift, p.fwhm, p.total])
            data[f'{name}_noise_value'] = np.array(r.noise_value)
            data[f'{name}_corr'] = np.array([r.amp_corr, r.shift_corr, r.fwhm_corr])
            data[f'{name}_rmse'] = np.array([r.amp_rmse, r.shift_rmse, r.fwhm_rmse])
            data[f'{name}_timing'] = np.array([r.gen_time, r.fit_time, r.throughput])
            if r.psnr_vs_noisefree is not None:
                pnf = r.psnr_vs_noisefree
                data[f'{name}_psnr_nf'] = np.array([pnf.amplitude, pnf.shift, pnf.fwhm, pnf.total])

        data['_names'] = np.array(names)
        data['_preset'] = np.array(self.preset_name)
        data['_image'] = np.array(self.image_path)
        np.savez_compressed(str(path), **data)
        print(f"Saved: {path}")

    @classmethod
    def load_npz(cls, path: str | Path) -> 'ParamNoiseSweepResult':
        """Load results from NPZ."""
        d = np.load(str(path), allow_pickle=True)
        names = list(d['_names'])
        results = {}
        for name in names:
            psnr_arr = d[f'{name}_psnr_orig']
            nf_key = f'{name}_psnr_nf'
            psnr_nf = None
            if nf_key in d:
                nf = d[nf_key]
                psnr_nf = ChannelPSNR(float(nf[0]), float(nf[1]), float(nf[2]), float(nf[3]))
            corr = d[f'{name}_corr']
            rmse = d[f'{name}_rmse']
            timing = d[f'{name}_timing']
            results[name] = ParamRoundtripResult(
                noise_level=name,
                noise_value=float(d[f'{name}_noise_value']),
                image_shape=(0, 0),
                n_spectra=0,
                gen_time=float(timing[0]),
                fit_time=float(timing[1]),
                total_time=float(timing[0] + timing[1]),
                throughput=float(timing[2]),
                psnr_vs_original=ChannelPSNR(
                    float(psnr_arr[0]), float(psnr_arr[1]),
                    float(psnr_arr[2]), float(psnr_arr[3]),
                ),
                psnr_vs_noisefree=psnr_nf,
                amp_corr=float(corr[0]),
                shift_corr=float(corr[1]),
                fwhm_corr=float(corr[2]),
                amp_rmse=float(rmse[0]),
                shift_rmse=float(rmse[1]),
                fwhm_rmse=float(rmse[2]),
            )
        return cls(
            results=results,
            preset_name=str(d['_preset']),
            image_path=str(d['_image']),
        )

    solver_name: str = 'taylor'

    @property
    def summary_table(self) -> str:
        lines = [
            "=" * 90,
            "3-PARAMETER ROUNDTRIP NOISE SWEEP",
            "=" * 90,
            f"Image: {self.image_path}",
            f"Preset: {self.preset_name}, Solver: {self.solver_name}",
            "-" * 90,
            f"{'Noise':<12} {'R(amp)':<10} {'G(δE)':<10} {'B(FWHM)':<10} "
            f"{'avg':<10} {'fit M/s':<10} {'time':<8}",
            "-" * 90,
        ]
        for name, r in self.results.items():
            p = r.psnr_vs_original
            lines.append(
                f"{name:<12} {p.amplitude:<10.1f} {p.shift:<10.1f} {p.fwhm:<10.1f} "
                f"{p.total:<10.1f} {r.throughput/1e6:<10.1f} {r.total_time:<8.2f}"
            )
        lines.append("-" * 90)

        # vs noise-free section
        has_nf = any(r.psnr_vs_noisefree is not None for r in self.results.values())
        if has_nf:
            lines.append("")
            lines.append(f"{'Noise':<12} {'R vs NF':<10} {'G vs NF':<10} {'B vs NF':<10} {'avg vs NF':<10}")
            lines.append("-" * 90)
            for name, r in self.results.items():
                if r.psnr_vs_noisefree is not None:
                    p = r.psnr_vs_noisefree
                    lines.append(
                        f"{name:<12} {p.amplitude:<10.1f} {p.shift:<10.1f} "
                        f"{p.fwhm:<10.1f} {p.total:<10.1f}"
                    )
            lines.append("-" * 90)

        return "\n".join(lines)


# ============================================================================
# PSNR Utilities
# ============================================================================

def _channel_psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    """PSNR between two uint8 channel images. MAX_I = 255."""
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse < 1e-20:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_per_channel_psnr(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> ChannelPSNR:
    """Compute PSNR for each RGB channel independently.

    Args:
        original: (H, W, 3) uint8
        reconstructed: (H, W, 3) uint8

    Returns:
        ChannelPSNR with R (amplitude), G (shift), B (FWHM) PSNR values
    """
    psnr_r = _channel_psnr(original[:, :, 0], reconstructed[:, :, 0])
    psnr_g = _channel_psnr(original[:, :, 1], reconstructed[:, :, 1])
    psnr_b = _channel_psnr(original[:, :, 2], reconstructed[:, :, 2])
    return ChannelPSNR(
        amplitude=psnr_r,
        shift=psnr_g,
        fwhm=psnr_b,
        total=(psnr_r + psnr_g + psnr_b) / 3,
    )


# ============================================================================
# Benchmark Class
# ============================================================================

class ParamRoundtripBenchmark:
    """3-parameter roundtrip benchmark for single-peak Voigt fitting.

    Encodes an RGB image into (amplitude, δE, FWHM) via SinglePeakEncoder,
    generates Voigt spectra (Taylor or exact), optionally adds noise, then
    recovers parameters via the 4-step pipeline or dictionary hybrid solver.

    Solver modes:
        'taylor': Standard 4-step residual projection (fast, ~19M spec/s)
        'dictionary': Dictionary hybrid with 4-step refinement (~7M spec/s)
                      Uses exact Voigt spectra for generation to test beyond
                      Taylor validity range.
        'dict2d': Dense 2D dictionary (δE × δσ) + 4-step refinement.
                  Uses denser δσ grid than 'dictionary'.
        'dict2d_amp_only': Dense 2D dictionary + amplitude-only Phase 2.
                           No sub-grid δE/δσ correction — relies on grid density.
    """

    def __init__(
        self,
        image_path: str | Path | None = None,
        preset: SinglePeakPreset = C1S_SINGLE_PRESET,
        solver: str = 'taylor',
        batch_size: int = 500_000,
        bg_fraction: float = 0.001,
        amplitude_scale: float = 1000.0,
        verbose: bool = True,
        exact_voigt: bool = False,
        amp_correction: str = 'full',
        image: np.ndarray | None = None,
    ):
        if image_path is None and image is None:
            raise ValueError("Either image_path or image must be provided")
        self.image_path = Path(image_path) if image_path is not None else Path('<array>')
        self.preset = preset
        self.solver = solver
        self.batch_size = batch_size
        self.bg_fraction = bg_fraction
        self.amplitude_scale = amplitude_scale
        self.verbose = verbose
        self.exact_voigt = exact_voigt
        self.amp_correction = amp_correction

        # Load image (or use the provided array directly)
        self.image = np.asarray(image) if image is not None else load_image(self.image_path)
        if self.image.ndim == 2:
            self.image = np.stack([self.image] * 3, axis=-1)
        self.image_shape = self.image.shape[:2]
        self.n_pixels = self.image_shape[0] * self.image_shape[1]

        # Setup encoder
        self.encoder = SinglePeakEncoder(preset)

        # Encode ground truth parameters
        self.gt_amp, self.gt_dE, self.gt_fwhm = self.encoder.encode(self.image)
        self.gt_dsigma = self.encoder.delta_sigma(self.gt_fwhm)

        # Pre-build basis matrices
        self.energy = preset.energy
        center = preset.element.binding_energy
        sigma = preset.element.sigma
        gamma = preset.element.gamma

        self.peak_config = {
            'centers': np.array([center], dtype=np.float32),
            'sigmas': np.array([sigma], dtype=np.float32),
            'gamma': gamma,
        }

        # Setup pipeline (used by taylor mode and for basis generation)
        self.cache = WeightMatrixCache()
        self.pipeline = HybridPipeline(
            cache=self.cache,
            enable_shift_correction=True,
            enable_stage2=False,
            use_mlx=True,
        )

        # Build Phi, J_c, J_sigma for spectra generation
        # Use raw (un-normalized) basis so that recovered amplitudes
        # match the scale used in spectra generation (amp * amplitude_scale).
        self.Phi, self.J_c, self.J_sigma = self.cache.build_basis_voigt_with_full_jacobian(
            self.energy,
            self.peak_config['centers'],
            self.peak_config['sigmas'],
            gamma,
        )

        # Dictionary cache (built on first use for dictionary mode)
        self._dict_cache = None

        if self.verbose:
            H, W = self.image_shape
            print(f"Image: {self.image_path.name} ({W}x{H}, {self.n_pixels:,} px)")
            print(f"Preset: {preset.name}, Solver: {solver}")
            print(f"Energy: {self.energy[0]:.1f}-{self.energy[-1]:.1f} eV "
                  f"({len(self.energy)} pts)")
            print(f"Params: amp[{preset.amplitude_range.min_val},{preset.amplitude_range.max_val}] "
                  f"δE[{preset.shift_range.min_val},{preset.shift_range.max_val}]eV "
                  f"FWHM[{preset.fwhm_range.min_val},{preset.fwhm_range.max_val}]eV")

    def _get_dict_cache(self):
        """Get or build the dictionary cache (lazy initialization)."""
        if self._dict_cache is not None:
            return self._dict_cache

        from toyomacro.voigtfit.dictionary_solver import build_dictionary

        sigma = self.preset.element.sigma
        # δE range: cover full encoder range plus margin
        dE_margin = 0.2
        dE_lo = self.preset.shift_range.min_val - dE_margin
        dE_hi = self.preset.shift_range.max_val + dE_margin
        # δσ range: cover full encoder range
        ds_min = float(self.gt_dsigma.min())
        ds_max = float(self.gt_dsigma.max())
        ds_margin = 0.02
        ds_lo = ds_min - ds_margin
        ds_hi = ds_max + ds_margin

        # Dict2D modes: denser δσ grid (step = 0.02σ instead of 0.1σ)
        is_dict2d = self.solver in ('dict2d', 'dict2d_amp_only', 'dict2d_parabola')
        dE_step = 0.1 * sigma
        dsigma_step = 0.02 * sigma if is_dict2d else 0.1 * sigma

        if self.verbose:
            print(f"Building dictionary: δE=[{dE_lo:.2f},{dE_hi:.2f}] step={dE_step:.4f}, "
                  f"δσ=[{ds_lo:.3f},{ds_hi:.3f}] step={dsigma_step:.4f}")

        t0 = time.perf_counter()
        self._dict_cache = build_dictionary(
            energy=self.energy,
            centers=self.peak_config['centers'],
            sigmas=self.peak_config['sigmas'],
            gamma=self.peak_config['gamma'],
            dE_range=(dE_lo, dE_hi),
            dE_step=dE_step,
            dsigma_range=(ds_lo, ds_hi),
            dsigma_step=dsigma_step,
            include_hessian=(self.solver == 'hybrid_6step'),
        )
        build_time = time.perf_counter() - t0

        if self.verbose:
            dc = self._dict_cache
            mem_mb = dc.D.nbytes / 1e6
            print(f"  Dictionary: {dc.n_dE}×{dc.n_dsigma} = {dc.n_dict} entries "
                  f"(D={mem_mb:.1f}MB, {build_time:.1f}s)")

        return self._dict_cache

    def _generate_spectra_batch(
        self,
        amp: np.ndarray,
        dE: np.ndarray,
        dsigma: np.ndarray,
    ) -> np.ndarray:
        """Generate Taylor-expanded Voigt spectra for a batch.

        Args:
            amp: (n,) amplitude values
            dE: (n,) center shift values (eV)
            dsigma: (n,) sigma shift values

        Returns:
            spectra: (n, n_energy) float32
        """
        Phi = self.Phi[:, 0]       # (n_energy,)
        J_c = self.J_c[:, 0]       # (n_energy,)
        J_s = self.J_sigma[:, 0]   # (n_energy,)

        # Taylor expansion: amp * (Phi + dE * J_c + dsigma * J_sigma) + background
        profiles = (Phi[np.newaxis, :]
                    + dE[:, np.newaxis] * J_c[np.newaxis, :]
                    + dsigma[:, np.newaxis] * J_s[np.newaxis, :])

        scaled_amp = amp * self.amplitude_scale
        spectra = scaled_amp[:, np.newaxis] * profiles

        # Add flat background proportional to amplitude
        bg = scaled_amp * self.bg_fraction
        spectra += bg[:, np.newaxis]

        return spectra.astype(np.float32)

    def _generate_spectra_exact(
        self,
        amp: np.ndarray,
        dE: np.ndarray,
        dsigma: np.ndarray,
    ) -> np.ndarray:
        """Generate exact Voigt spectra for a batch (no Taylor approximation).

        Used in dictionary mode to test recovery beyond Taylor validity.
        Vectorized using scipy.special.wofz broadcasting.
        """
        from scipy import special as sps

        energy = self.energy.astype(np.float64)  # (n_E,)
        center = float(self.preset.element.binding_energy)
        sigma_nom = float(self.preset.element.sigma)
        gamma = float(self.preset.element.gamma)

        SQRT2 = np.sqrt(2.0)
        SQRT2PI = np.sqrt(2.0 * np.pi)

        # Vectorized: (n, n_E)
        centers = (center + dE.astype(np.float64))[:, np.newaxis]  # (n, 1)
        sigmas = (sigma_nom + dsigma.astype(np.float64))[:, np.newaxis]  # (n, 1)

        z = ((energy[np.newaxis, :] - centers) + 1j * gamma) / (sigmas * SQRT2)
        profiles = np.real(sps.wofz(z)) / (sigmas * SQRT2PI)  # (n, n_E)

        scaled_amp = (amp * self.amplitude_scale).astype(np.float64)
        spectra = scaled_amp[:, np.newaxis] * profiles

        # Background
        bg = scaled_amp * self.bg_fraction
        spectra += bg[:, np.newaxis]

        return spectra.astype(np.float32)

    def _add_noise(
        self,
        spectra: np.ndarray,
        noise_value: float,
        global_max: float | None = None,
    ) -> np.ndarray:
        """Add Poisson noise to spectra.

        Args:
            spectra: (n, n_energy) float32
            noise_value: Poisson noise level (0 = no noise)
            global_max: For heteroscedastic noise normalization

        Returns:
            noisy spectra (n, n_energy) float32
        """
        if noise_value <= 0:
            return spectra

        if HAS_MLX:
            from toyomacro.voigtfit.spectra_generator import add_poisson_noise_mlx_fused
            spectra_mx = mx.array(spectra)
            noisy_mx = add_poisson_noise_mlx_fused(
                spectra_mx, noise_value, global_max=global_max,
            )
            mx.eval(noisy_mx)
            return np.array(noisy_mx, dtype=np.float32)
        else:
            from toyomacro.voigtfit.spectra_generator import add_poisson_noise
            return add_poisson_noise(spectra, noise_value, global_max=global_max)

    def run(
        self,
        noise_level: str | float = 'None',
        noisefree_result: ParamRoundtripResult | None = None,
        keep_images: bool = True,
    ) -> ParamRoundtripResult:
        """Run single 3-parameter roundtrip benchmark.

        Args:
            noise_level: Noise level name or numeric value
            noisefree_result: Reference for PSNR vs noise-free comparison
            keep_images: Include images in result

        Returns:
            ParamRoundtripResult
        """
        # Parse noise level
        if isinstance(noise_level, str):
            noise_name = noise_level
            noise_value = NOISE_LEVELS.get(noise_level, 0)
            if noise_value == 0 and noise_name.startswith('Custom_'):
                noise_value = float(noise_name.split('_', 1)[1])
        else:
            noise_value = float(noise_level)
            # Find closest name
            noise_name = 'Custom'
            for name, val in NOISE_LEVELS.items():
                if abs(val - noise_value) < 1e-6:
                    noise_name = name
                    break

        if self.verbose:
            print(f"\n--- {noise_name} (level={noise_value}) ---")

        # Compute global_max for heteroscedastic noise
        global_max = None
        if noise_value > 0:
            max_amp = self.gt_amp.max() * self.amplitude_scale
            global_max = max_amp * (1.0 + self.bg_fraction)

        # Build dictionary cache if needed
        use_dict = self.solver in ('dictionary', 'hybrid_6step', 'dict2d', 'dict2d_amp_only', 'dict2d_parabola')
        use_taylor6 = self.solver == 'taylor6'
        use_hybrid_6step = self.solver == 'hybrid_6step'
        use_dict2d_amp_only = self.solver == 'dict2d_amp_only'
        use_dict2d_parabola = self.solver == 'dict2d_parabola'
        if use_dict:
            if use_hybrid_6step:
                from toyomacro.voigtfit.dictionary_solver import solve_hybrid_6step_sorted
            elif use_dict2d_amp_only:
                from toyomacro.voigtfit.dictionary_solver import solve_dict2d_amp_only
            elif use_dict2d_parabola:
                from toyomacro.voigtfit.dictionary_solver import solve_dict2d_parabola
            else:
                from toyomacro.voigtfit.dictionary_solver import solve_hybrid_sorted
            dict_cache = self._get_dict_cache()

        # Select spectra generator:
        # - dictionary/dict2d/dict2d_amp_only/taylor6/exact_voigt: exact Voigt
        # - taylor (default): Taylor-expanded spectra (linear, self-inverse)
        use_exact = use_dict or use_taylor6 or self.exact_voigt
        gen_func = self._generate_spectra_exact if use_exact else self._generate_spectra_batch

        # Generate spectra and fit in batches
        all_amps = []
        all_dE = []
        all_dsigma = []
        gen_time = 0.0
        fit_time = 0.0

        for batch_start in range(0, self.n_pixels, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.n_pixels)

            amp_b = self.gt_amp[batch_start:batch_end]
            dE_b = self.gt_dE[batch_start:batch_end]
            ds_b = self.gt_dsigma[batch_start:batch_end]

            # Generate spectra
            t0 = time.perf_counter()
            spectra = gen_func(amp_b, dE_b, ds_b)
            spectra = self._add_noise(spectra, noise_value, global_max)
            gen_time += time.perf_counter() - t0

            # Fit
            t0 = time.perf_counter()
            if use_dict:
                if use_hybrid_6step:
                    amps_b, _, dE_b_out, ds_b_out, _ = solve_hybrid_6step_sorted(
                        spectra, dict_cache
                    )
                elif use_dict2d_parabola:
                    amps_b, _, dE_b_out, ds_b_out, _ = solve_dict2d_parabola(
                        spectra, dict_cache
                    )
                elif use_dict2d_amp_only:
                    amps_b, _, dE_b_out, ds_b_out, _ = solve_dict2d_amp_only(
                        spectra, dict_cache
                    )
                else:
                    amps_b, _, dE_b_out, ds_b_out, _ = solve_hybrid_sorted(
                        spectra, dict_cache
                    )
                all_amps.append(amps_b[0, :])
                all_dE.append(dE_b_out)
                all_dsigma.append(ds_b_out)
            else:
                result = self.pipeline.process_rowmajor_extended_3param(
                    spectra,
                    self.preset.element.symbol,
                    self.preset.element.orbital,
                    self.energy,
                    self.peak_config,
                    amplitudes_only=True,
                    n_steps=6 if use_taylor6 else 4,
                    amp_correction=self.amp_correction,
                )
                all_amps.append(result.amplitudes[0, :])
                all_dE.append(result.energy_shifts)
                all_dsigma.append(result.sigma_shifts)
                del result
            fit_time += time.perf_counter() - t0

            del spectra
            gc.collect()

        # Concatenate all batches
        rec_amp = np.concatenate(all_amps)
        rec_dE = np.concatenate(all_dE)
        rec_dsigma = np.concatenate(all_dsigma)

        # Convert dsigma back to FWHM
        nominal_sigma = self.preset.element.sigma
        rec_fwhm = self.encoder.fwhm_from_sigma(nominal_sigma + rec_dsigma)

        # Decode to RGB image
        rec_image = self.encoder.decode(
            rec_amp / self.amplitude_scale,  # back to [0, 1] range
            rec_dE,
            rec_fwhm,
            self.image_shape,
        )

        # Per-channel PSNR vs original
        psnr_orig = compute_per_channel_psnr(self.image, rec_image)

        # Parameter-space correlation and RMSE
        gt_amp_scaled = self.gt_amp * self.amplitude_scale
        amp_corr = float(np.corrcoef(gt_amp_scaled.ravel(), rec_amp.ravel())[0, 1])
        shift_corr = float(np.corrcoef(self.gt_dE.ravel(), rec_dE.ravel())[0, 1])
        fwhm_corr = float(np.corrcoef(self.gt_dsigma.ravel(), rec_dsigma.ravel())[0, 1])

        amp_rmse = float(np.sqrt(np.mean((gt_amp_scaled - rec_amp) ** 2)))
        shift_rmse = float(np.sqrt(np.mean((self.gt_dE - rec_dE) ** 2)))
        fwhm_rmse = float(np.sqrt(np.mean((self.gt_dsigma - rec_dsigma) ** 2)))

        # PSNR vs noise-free
        psnr_nf = None
        if noisefree_result is not None and noisefree_result.reconstructed_image is not None:
            psnr_nf = compute_per_channel_psnr(
                noisefree_result.reconstructed_image, rec_image
            )

        total_time = gen_time + fit_time
        throughput = self.n_pixels / fit_time if fit_time > 0 else float('inf')

        if self.verbose:
            print(f"  Gen: {gen_time:.2f}s  Fit: {fit_time:.2f}s  "
                  f"({throughput/1e6:.1f}M spec/s)")
            print(f"  vs Original: {psnr_orig}")
            if psnr_nf is not None:
                print(f"  vs NoiseFree: {psnr_nf}")
            print(f"  Corr: amp={amp_corr:.4f} δE={shift_corr:.4f} δσ={fwhm_corr:.4f}")
            print(f"  RMSE: amp={amp_rmse:.2f} δE={shift_rmse:.4f}eV δσ={fwhm_rmse:.4f}")

        return ParamRoundtripResult(
            noise_level=noise_name,
            noise_value=noise_value,
            image_shape=self.image_shape,
            n_spectra=self.n_pixels,
            gen_time=gen_time,
            fit_time=fit_time,
            total_time=total_time,
            throughput=throughput,
            psnr_vs_original=psnr_orig,
            psnr_vs_noisefree=psnr_nf,
            amp_corr=amp_corr,
            shift_corr=shift_corr,
            fwhm_corr=fwhm_corr,
            amp_rmse=amp_rmse,
            shift_rmse=shift_rmse,
            fwhm_rmse=fwhm_rmse,
            original_image=self.image if keep_images else None,
            reconstructed_image=rec_image if keep_images else None,
        )

    def run_noise_sweep(
        self,
        noise_levels: list[str] | None = None,
        output_dir: str | Path | None = None,
        save_images: bool = True,
    ) -> ParamNoiseSweepResult:
        """Run benchmark for multiple noise levels.

        Args:
            noise_levels: List of noise level names (default: representative set)
            output_dir: Directory for output images
            save_images: Save comparison images per noise level

        Returns:
            ParamNoiseSweepResult with all results and summary table
        """
        if noise_levels is None:
            noise_levels = ['None', 'Moderate', 'Strong']

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print("3-PARAMETER ROUNDTRIP NOISE SWEEP")
        print(f"{'='*70}")

        results = {}
        noisefree_result = None

        # Run noise-free first as reference
        if 'None' in noise_levels:
            noisefree_result = self.run('None', keep_images=True)
            results['None'] = noisefree_result

            if save_images and output_dir is not None:
                self._save_comparison(
                    output_dir / 'param_roundtrip_noisefree.png',
                    noisefree_result,
                )

        # Run noisy levels
        for level in noise_levels:
            if level == 'None':
                continue

            r = self.run(level, noisefree_result=noisefree_result, keep_images=True)
            results[level] = r

            if save_images and output_dir is not None:
                self._save_comparison(
                    output_dir / f'param_roundtrip_{level.lower()}.png',
                    r,
                )

        solver_label = self.solver
        if self.exact_voigt and self.solver == 'taylor':
            solver_label = 'taylor(exact)'
        if self.amp_correction != 'full':
            solver_label = f'{solver_label}[amp={self.amp_correction}]'
        sweep_result = ParamNoiseSweepResult(
            results=results,
            preset_name=self.preset.name,
            image_path=str(self.image_path),
            solver_name=solver_label,
        )

        print(f"\n{sweep_result.summary_table}")

        # Save NPZ cache
        if output_dir is not None:
            npz_path = output_dir / 'param_roundtrip_cache.npz'
            sweep_result.save_npz(npz_path)

        return sweep_result

    def _save_comparison(
        self,
        path: Path,
        result: ParamRoundtripResult,
    ) -> None:
        """Save side-by-side original vs reconstructed comparison."""
        try:
            from PIL import Image
        except ImportError:
            return

        orig = Image.fromarray(self.image)
        recon = Image.fromarray(result.reconstructed_image)

        H, W = self.image_shape
        # Side-by-side with labels
        canvas_w = W * 2 + 20
        canvas_h = H + 40
        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        canvas.paste(orig, (0, 30))
        canvas.paste(recon, (W + 20, 30))

        # Add simple text via pixel manipulation (no font dependency)
        canvas.save(str(path), quality=95)
        if self.verbose:
            print(f"  Saved: {path.name}")


# ============================================================================
# CLI
# ============================================================================

def plot_noise_trajectory(
    sweep: ParamNoiseSweepResult,
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot per-channel PSNR vs noise level trajectory.

    Args:
        sweep: Results from noise sweep
        output_path: Save path for PNG
        show: Show interactive plot
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Collect data
    names = []
    noise_vals = []
    psnr_r, psnr_g, psnr_b, psnr_avg = [], [], [], []

    for name, r in sweep.results.items():
        names.append(name)
        noise_vals.append(r.noise_value if r.noise_value > 0 else 0.01)
        p = r.psnr_vs_original
        psnr_r.append(p.amplitude)
        psnr_g.append(p.shift)
        psnr_b.append(p.fwhm)
        psnr_avg.append(p.total)

    noise_vals = np.array(noise_vals)
    sort_idx = np.argsort(noise_vals)
    noise_vals = noise_vals[sort_idx]
    psnr_r = np.array(psnr_r)[sort_idx]
    psnr_g = np.array(psnr_g)[sort_idx]
    psnr_b = np.array(psnr_b)[sort_idx]
    psnr_avg = np.array(psnr_avg)[sort_idx]
    names = [names[i] for i in sort_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PSNR vs noise level
    ax1.semilogx(noise_vals, psnr_r, 'ro-', label='R (amplitude)', markersize=7)
    ax1.semilogx(noise_vals, psnr_g, 'gs-', label='G (δE shift)', markersize=7)
    ax1.semilogx(noise_vals, psnr_b, 'b^-', label='B (FWHM)', markersize=7)
    ax1.semilogx(noise_vals, psnr_avg, 'k--', label='Average', linewidth=2, alpha=0.7)

    for i, name in enumerate(names):
        if name != 'None':
            ax1.annotate(name, (noise_vals[i], psnr_avg[i]),
                        textcoords='offset points', xytext=(5, 5),
                        fontsize=8, alpha=0.7)

    ax1.set_xlabel('Noise Level (Poisson λ)')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Per-Channel PSNR vs Noise Level')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Right: Correlation vs noise level
    corr_a, corr_s, corr_f = [], [], []
    for name in names:
        r = sweep.results[name]
        corr_a.append(r.amp_corr)
        corr_s.append(r.shift_corr)
        corr_f.append(r.fwhm_corr)

    ax2.semilogx(noise_vals, corr_a, 'ro-', label='Amplitude', markersize=7)
    ax2.semilogx(noise_vals, corr_s, 'gs-', label='δE shift', markersize=7)
    ax2.semilogx(noise_vals, corr_f, 'b^-', label='FWHM', markersize=7)

    ax2.set_xlabel('Noise Level (Poisson λ)')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Parameter Correlation vs Noise Level')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.2, 1.05)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='3-Parameter Roundtrip Benchmark (amp + δE + FWHM)',
    )
    parser.add_argument(
        '--image', type=str, default=None,
        help='Input image path (JPG, PNG, GIF)',
    )
    parser.add_argument(
        '--noise-levels', nargs='+', default=['None', 'Moderate', 'Strong'],
        help='Noise levels (names or lam notation: lam3=Moderate)',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory for images and results',
    )
    parser.add_argument(
        '--npz', type=str, default=None,
        help='Load cached results from NPZ (skip computation)',
    )
    parser.add_argument(
        '--solver', type=str, default='taylor',
        choices=['taylor', 'taylor6', 'dictionary', 'hybrid_6step',
                 'dict2d', 'dict2d_amp_only'],
        help='Solver: taylor (4-step), taylor6 (6-step), dictionary (dict+4-step), '
             'hybrid_6step (dict+6-step), dict2d (dense 2D dict+4-step), '
             'dict2d_amp_only (dense 2D dict, amp only)',
    )
    parser.add_argument(
        '--batch-size', type=int, default=500_000,
        help='Batch size for spectra generation/fitting',
    )
    parser.add_argument(
        '--bg-fraction', type=float, default=0.001,
        help='Background fraction of amplitude',
    )
    parser.add_argument(
        '--amp-scale', type=float, default=1000.0,
        help='Amplitude scale factor',
    )
    parser.add_argument(
        '--exact-voigt', action='store_true',
        help='Use exact Voigt spectra (not Taylor expansion) for all solvers',
    )
    parser.add_argument(
        '--amp-correction', type=str, default='full',
        choices=['full', 'sigma_only', 'skip', 'refit'],
        help='Amplitude correction mode: full (M matrix), sigma_only (C_σ only), skip (none), refit (corrected basis)',
    )
    parser.add_argument(
        '--no-images', action='store_true',
        help='Skip saving comparison images',
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate noise trajectory plot',
    )

    args = parser.parse_args()

    output_dir = args.output
    if output_dir is None:
        output_dir = str(
            Path(__file__).resolve().parent.parent.parent / 'outputs' / 'param_roundtrip'
        )

    if args.npz:
        # Load from cache
        sweep = ParamNoiseSweepResult.load_npz(args.npz)
        print(sweep.summary_table)
    else:
        if args.image is None:
            parser.error('--image is required when not using --npz')

        # Parse noise levels with lam notation
        noise_levels = [parse_noise_level(s) for s in args.noise_levels]

        bench = ParamRoundtripBenchmark(
            image_path=args.image,
            solver=args.solver,
            batch_size=args.batch_size,
            bg_fraction=args.bg_fraction,
            amplitude_scale=args.amp_scale,
            exact_voigt=args.exact_voigt,
            amp_correction=args.amp_correction,
        )

        sweep = bench.run_noise_sweep(
            noise_levels=noise_levels,
            output_dir=output_dir,
            save_images=not args.no_images,
        )

    if args.plot:
        plot_path = Path(output_dir) / 'param_roundtrip_trajectory.png'
        plot_noise_trajectory(sweep, output_path=plot_path)


if __name__ == '__main__':
    main()
