"""
Roundtrip Benchmark: Image → Spectra → VoigtFit → Image

Complete evaluation pipeline for VoigtFit reconstruction quality.
Supports noise injection at various levels for robustness testing.

Workflow:
    1. Input Image (RGB)
    2. Color Decomposition → Element Amplitudes
    3. Voigt Spectra Generation (with optional noise)
    4. VoigtFit Reconstruction
    5. Image Reconstruction from fitted amplitudes
    6. PSNR Evaluation vs Original and vs Noise-Free

Usage:
    from benchmark import run_noise_sweep, run_roundtrip_benchmark

    # Quick benchmark
    result = run_roundtrip_benchmark(image_path='my_image.jpg')

    # Full noise sweep with timestamp folder
    results = run_noise_sweep(
        image_path='my_image.jpg',
        output_dir='/path/to/output',
        save_spectra=True,  # Keep H5 files
    )
    print(results.summary_table)
Date: 2026-01-23
"""

import gc
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from toyomacro.voigtfit.image_utils import (
    amplitudes_to_rgb,
    load_image,
    save_image,
)
from toyomacro.voigtfit.pipeline import HybridPipeline
from toyomacro.voigtfit.spectra_generator import (
    NOISE_LEVELS,
    ElementPreset,
    GeneratorConfig,
    NoiseConfig,
    SpectraGenerator,
    SpectralConfig,
    decompose_image_to_amplitudes,
    get_element_preset,
)
from toyomacro.voigtfit.weight_cache import WeightMatrixCache

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AmplitudeMetrics:
    """Metrics for amplitude quality evaluation."""
    gain: float  # Total area ratio (noisy/noisefree)
    correlation: float  # Pixel-wise correlation
    rmse: float  # Root mean square error
    psnr: float  # PSNR of normalized amplitudes
    nf_total: float  # Noise-free total amplitude
    noisy_total: float  # Noisy total amplitude


def calculate_amplitude_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate PSNR between original and reconstructed amplitude maps.

    Both are normalized to [0, 1] based on their max values before comparison.
    This matches the approach in comprehensive_benchmark.py / BENCHMARK_RESULTS.md.

    Args:
        original: Original amplitude map
        reconstructed: Reconstructed amplitude map

    Returns:
        PSNR in dB
    """
    # Normalize both to [0, 1]
    orig_norm = original / (original.max() + 1e-10)
    recon_norm = reconstructed / (reconstructed.max() + 1e-10)

    # Calculate MSE and PSNR
    mse = np.mean((orig_norm - recon_norm) ** 2)
    if mse < 1e-20:
        return float('inf')

    # Max value is 1.0 after normalization
    return 10 * np.log10(1.0 / mse)


@dataclass
class RoundtripResult:
    """Result from a single roundtrip benchmark run."""
    # Configuration
    noise_level: str
    noise_value: float
    image_shape: tuple
    n_elements: int

    # Timing
    generation_time: float
    reconstruction_time: float
    total_time: float

    # Performance
    n_spectra: int
    throughput: float  # spectra/sec

    # Quality - Amplitude level (primary metrics, matching BENCHMARK_RESULTS.md)
    psnr_vs_original: float = 0.0  # PSNR of normalized amplitudes vs original
    psnr_vs_noisefree: float | None = None  # PSNR vs noise-free reconstruction

    # Quality - Additional amplitude metrics
    amp_gain_avg: float = 1.0
    amp_correlation: float = 1.0
    amp_details: dict[str, Any] = field(default_factory=dict)

    # Images (optional, for visualization only)
    original_image: np.ndarray | None = None
    reconstructed_image: np.ndarray | None = None

    # Output path
    output_dir: str | None = None

    def __repr__(self):
        psnr_nf = f", NF={self.psnr_vs_noisefree:.1f}dB" if self.psnr_vs_noisefree else ""
        return (
            f"RoundtripResult(noise='{self.noise_level}', "
            f"PSNR={self.psnr_vs_original:.1f}dB{psnr_nf}, "
            f"Corr={self.amp_correlation:.4f}, time={self.total_time:.1f}s)"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'noise_level': self.noise_level,
            'noise_value': float(self.noise_value),
            'image_shape': list(self.image_shape),
            'n_elements': self.n_elements,
            'generation_time': float(self.generation_time),
            'reconstruction_time': float(self.reconstruction_time),
            'total_time': float(self.total_time),
            'n_spectra': self.n_spectra,
            'throughput': float(self.throughput),
            'psnr_vs_original': float(self.psnr_vs_original),
            'psnr_vs_noisefree': float(self.psnr_vs_noisefree) if self.psnr_vs_noisefree else None,
            'amp_gain_avg': float(self.amp_gain_avg),
            'amp_correlation': float(self.amp_correlation),
            'output_dir': self.output_dir,
        }


@dataclass
class NoiseSweepResult:
    """Results from running all noise levels."""
    results: dict[str, RoundtripResult]
    summary_table: str
    total_time: float
    output_dir: str | None = None
    timestamp: str | None = None

    def get_psnr_table(self) -> dict[str, float]:
        """Get PSNR values by noise level."""
        return {
            name: r.psnr_vs_original.psnr_total
            for name, r in self.results.items()
            if r.psnr_vs_original is not None
        }

    def get_correlation_table(self) -> dict[str, float]:
        """Get amplitude correlation values by noise level."""
        return {
            name: r.amp_correlation
            for name, r in self.results.items()
        }

    def __repr__(self):
        n_levels = len(self.results)
        return f"NoiseSweepResult({n_levels} levels, {self.total_time:.1f}s, dir={self.output_dir})"


# ============================================================================
# Roundtrip Benchmark Class
# ============================================================================

class RoundtripBenchmark:
    """
    Complete Image → Spectra → VoigtFit → Image roundtrip benchmark.

    Example:
        # Basic usage
        benchmark = RoundtripBenchmark('my_image.jpg')
        result = benchmark.run(noise_level='Moderate')
        print(f"PSNR: {result.psnr_vs_original.psnr_total:.1f} dB")
        print(f"Correlation: {result.amp_correlation:.4f}")

        # Run all noise levels with timestamp folder
        sweep = benchmark.run_noise_sweep(save_spectra=True)
        print(sweep.summary_table)
    """

    def __init__(
        self,
        image_path: str | Path,
        output_dir: str | Path | None = None,
        generator_config: GeneratorConfig | None = None,
        elements: str | ElementPreset = 'demo',
        batch_size: int = 2_000_000,
        use_mlx: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize roundtrip benchmark.

        Args:
            image_path: Path to input image (RGB)
            output_dir: Base directory for output (timestamp subfolder created)
            generator_config: Custom generator configuration
            elements: Preset name (e.g., 'demo') or ElementPreset object.
                      Default is 'demo' (6-component Si/Ti/Al/C/O mapping).
            batch_size: Spectra per batch for processing
            use_mlx: Use MLX GPU acceleration
            verbose: Print progress
        """
        self.image_path = Path(image_path)
        self.base_output_dir = Path(output_dir) if output_dir else Path('/tmp/roundtrip_benchmark')
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        self.generator_config = generator_config or GeneratorConfig()
        self.batch_size = batch_size
        self.use_mlx = use_mlx
        self.verbose = verbose

        # Load image
        self.original_image = load_image(self.image_path)
        self.image_shape = self.original_image.shape[:2]
        self.n_pixels = self.image_shape[0] * self.image_shape[1]

        # Initialize generator (output_dir set per-run)
        self._generator = None
        self._current_output_dir = None

        # Element configuration: resolve to ElementPreset
        if isinstance(elements, str):
            self._elements_preset = get_element_preset(elements)
        else:
            self._elements_preset = elements

        # Initialize VoigtFit pipeline
        self.cache = WeightMatrixCache()
        self.pipeline = HybridPipeline(
            self.cache,
            enable_stage2=False,
            use_mlx=use_mlx,
        )

        # Lazy initialization
        self._elem_by_file = None
        self._color_mapping = None
        self._component_order = None
        self._original_amplitudes = None

        if self.verbose:
            print("RoundtripBenchmark initialized:")
            print(f"  Image: {self.image_path.name} ({self.image_shape[1]}x{self.image_shape[0]})")
            print(f"  Pixels: {self.n_pixels:,}")
            print(f"  Elements: {self._elements_preset.name} ({len(self._elements_preset.elements)} components)")

    def _init_generator(self, output_dir: Path, project_name: str = 'benchmark'):
        """Initialize or reinitialize generator with given output directory."""
        self._current_output_dir = output_dir
        self._generator = SpectraGenerator(
            output_dir=output_dir,
            image_path=str(self.image_path),
            project_name=project_name,
            config=self.generator_config,
        )

        self._generator.configure_elements(self._elements_preset)

        # Update cached properties
        self._color_mapping = self._generator.color_mapping
        self._component_order = self._generator.component_order

        self._elem_by_file = {}
        for idx, elem in enumerate(self._generator.elements):
            file_key = elem.file_key
            if file_key not in self._elem_by_file:
                self._elem_by_file[file_key] = []
            self._elem_by_file[file_key].append((idx, elem))

        # Compute original amplitudes
        self._compute_original_amplitudes()

    def _compute_original_amplitudes(self):
        """Decompose original image to element amplitudes."""
        image_array = np.array(self.original_image).astype(np.float64) / 255.0
        amp_raw = decompose_image_to_amplitudes(
            image_array, self._color_mapping, method='pinv'
        )

        self._original_amplitudes = {}
        for idx, (file_key, comp_idx) in enumerate(self._component_order):
            key = f"{file_key}_{comp_idx}"
            self._original_amplitudes[key] = amp_raw[idx, :].reshape(self.image_shape)

    @property
    def n_elements(self) -> int:
        if self._generator is None:
            return len(self._elements_preset.elements)
        return len(self._generator.elements)

    def run(
        self,
        noise_level: str | float = 'None',
        output_dir: str | Path | None = None,
        project_name: str = 'benchmark',
        save_spectra: bool = True,
        keep_images: bool = True,
        noisefree_amplitudes: dict[str, np.ndarray] | None = None,
        noisefree_reconstructed: np.ndarray | None = None,
        use_inmemory: bool | str = False,
    ) -> RoundtripResult:
        """
        Run single roundtrip benchmark.

        Args:
            noise_level: Noise level name ('None', 'Moderate', etc.) or numeric value
            output_dir: Output directory (default: base_output_dir)
            project_name: Project name for H5 files
            save_spectra: Keep generated H5 files
            keep_images: Include images in result
            noisefree_amplitudes: Reference amplitudes for comparison
            noisefree_reconstructed: Reference image for PSNR vs NF
            use_inmemory: Processing mode:
                False - H5 path (write to disk, read back)
                True or 'inmemory' - Generate all spectra in memory, then fit
                'streaming' - Fused generate+fit per batch (lowest memory, fastest)

        Returns:
            RoundtripResult with metrics and optionally images
        """
        # Parse noise level
        if isinstance(noise_level, str):
            noise_name = noise_level
            if noise_level not in NOISE_LEVELS:
                import warnings
                warnings.warn(
                    f"Unknown noise level '{noise_level}', defaulting to 0 (noise-free). "
                    f"Valid names: {sorted(NOISE_LEVELS.keys())}",
                    stacklevel=2,
                )
            noise_value = NOISE_LEVELS.get(noise_level, 0)
        else:
            noise_value = float(noise_level)
            noise_name = SpectralConfig.noise_degree_name(noise_value)

        # Setup output directory
        if output_dir is None:
            output_dir = self.base_output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize generator
        self._init_generator(output_dir, project_name)

        if self.verbose:
            print(f"\n--- {noise_name} (level={noise_value}) ---")

        # Configure noise
        if noise_value == 0:
            noise_config = NoiseConfig(noise_type='none')
        else:
            noise_config = NoiseConfig(noise_type='poisson', poisson_level=noise_value)

        # Normalize mode string
        mode_str = str(use_inmemory).lower() if isinstance(use_inmemory, str) else ''
        is_streaming = mode_str == 'streaming'
        is_inmemory = use_inmemory is True or mode_str == 'inmemory'

        if is_streaming:
            # === Streaming path: fused generate+fit per batch ===
            fitted_amplitudes, gen_time, recon_time = self._run_fused_streaming(
                noise_config
            )
            spectra_dir = None
            if self.verbose:
                print(f"  Streaming gen+fit: gen={gen_time:.2f}s, fit={recon_time:.2f}s")
        elif is_inmemory:
            # === In-memory path: generate all, then fit ===
            t0 = time.perf_counter()
            spectra_data = self._generator.generate_inmemory(
                noise_config, verbose=False
            )
            gen_time = time.perf_counter() - t0
            spectra_dir = None

            if self.verbose:
                print(f"  Generation (in-memory): {gen_time:.2f}s")

            t0 = time.perf_counter()
            fitted_amplitudes = self._run_voigtfit_inmemory(spectra_data)
            recon_time = time.perf_counter() - t0
        else:
            # === H5 path: write to disk, read back ===
            t0 = time.perf_counter()
            spectra_dir = self._generator.generate(noise_config, verbose=False)
            gen_time = time.perf_counter() - t0

            if self.verbose:
                print(f"  Generation: {gen_time:.2f}s")
                print(f"  Output: {spectra_dir}")

            t0 = time.perf_counter()
            fitted_amplitudes = self._run_voigtfit(spectra_dir, project_name)
            recon_time = time.perf_counter() - t0

        n_spectra = self.n_pixels * len(self._elem_by_file)
        if self.verbose:
            mode_label = "streaming" if is_streaming else ("in-memory" if is_inmemory else "H5")
            print(f"  VoigtFit ({mode_label}): {recon_time:.2f}s ({n_spectra/recon_time/1e6:.1f}M spec/s)")

        # Step 3: Calculate amplitude-level PSNR vs Original
        # This matches the approach in BENCHMARK_RESULTS.md
        psnr_orig_list = []
        for file_key, elem_list in self._elem_by_file.items():
            if file_key not in fitted_amplitudes:
                continue
            fitted = fitted_amplitudes[file_key]  # shape: (n_comp, n_spectra)

            for j, (elem_idx, elem) in enumerate(elem_list):
                comp_idx = self._component_order[elem_idx][1]
                key = f"{file_key}_{comp_idx}"

                if key in self._original_amplitudes:
                    original = self._original_amplitudes[key]
                    reconstructed_amp = fitted[j, :].reshape(self.image_shape)

                    psnr = calculate_amplitude_psnr(original, reconstructed_amp)
                    psnr_orig_list.append(psnr)

        psnr_vs_original = np.mean(psnr_orig_list) if psnr_orig_list else 0.0

        # Step 4: Calculate amplitude-level PSNR vs Noise-Free
        psnr_vs_noisefree = None
        psnr_nf_list = []
        amp_metrics = {}

        if noisefree_amplitudes is not None and noise_name != 'None':
            for file_key, elem_list in self._elem_by_file.items():
                if file_key not in fitted_amplitudes or file_key not in noisefree_amplitudes:
                    continue

                fitted = fitted_amplitudes[file_key]
                nf_fitted = noisefree_amplitudes[file_key]

                for j, (elem_idx, elem) in enumerate(elem_list):
                    comp_idx = self._component_order[elem_idx][1]
                    key = f"{file_key}_{comp_idx}"

                    noisy_amp = fitted[j, :].reshape(self.image_shape)
                    nf_amp = nf_fitted[j, :].reshape(self.image_shape)

                    # PSNR vs noise-free
                    psnr_nf = calculate_amplitude_psnr(nf_amp, noisy_amp)
                    psnr_nf_list.append(psnr_nf)

                    # Additional metrics
                    nf_total = float(np.sum(nf_amp))
                    noisy_total = float(np.sum(noisy_amp))
                    gain = noisy_total / nf_total if nf_total > 0 else 1.0
                    corr = float(np.corrcoef(nf_amp.flatten(), noisy_amp.flatten())[0, 1])
                    rmse = float(np.sqrt(np.mean((nf_amp - noisy_amp)**2)))

                    amp_metrics[key] = AmplitudeMetrics(
                        gain=gain,
                        correlation=corr,
                        rmse=rmse,
                        psnr=psnr_nf,
                        nf_total=nf_total,
                        noisy_total=noisy_total,
                    )

            psnr_vs_noisefree = np.mean(psnr_nf_list) if psnr_nf_list else None

        avg_gain = np.mean([m.gain for m in amp_metrics.values()]) if amp_metrics else 1.0
        avg_corr = np.mean([m.correlation for m in amp_metrics.values()]) if amp_metrics else 1.0

        # Step 5: Generate reconstructed image (for visualization only)
        reconstructed = amplitudes_to_rgb(
            amplitudes=fitted_amplitudes,
            color_mapping=self._color_mapping,
            component_order=self._component_order,
            image_shape=self.image_shape,
        )

        if self.verbose:
            psnr_nf_str = f'{psnr_vs_noisefree:.2f}' if psnr_vs_noisefree else 'N/A'
            print(f"  PSNR Orig: {psnr_vs_original:.2f} dB, NF: {psnr_nf_str} dB")
            print(f"  Amp Gain: {avg_gain:.4f}, Corr: {avg_corr:.4f}")

        total_time = gen_time + recon_time

        # Cleanup if not saving spectra
        if not save_spectra and spectra_dir is not None:
            import shutil
            if spectra_dir.exists():
                shutil.rmtree(spectra_dir)

        return RoundtripResult(
            noise_level=noise_name,
            noise_value=noise_value,
            image_shape=self.image_shape,
            n_elements=self.n_elements,
            generation_time=gen_time,
            reconstruction_time=recon_time,
            total_time=total_time,
            n_spectra=n_spectra,
            throughput=n_spectra / recon_time,
            psnr_vs_original=psnr_vs_original,
            psnr_vs_noisefree=psnr_vs_noisefree,
            amp_gain_avg=avg_gain,
            amp_correlation=avg_corr,
            amp_details={k: vars(v) for k, v in amp_metrics.items()},
            original_image=self.original_image if keep_images else None,
            reconstructed_image=reconstructed if keep_images else None,
            output_dir=str(spectra_dir) if save_spectra else None,
        )

    def _run_voigtfit(self, spectra_dir: Path, project_name: str) -> dict[str, np.ndarray]:
        """Run VoigtFit on generated spectra."""
        amplitudes = {}

        for file_key, elem_list in self._elem_by_file.items():
            h5_path = spectra_dir / f"{file_key}_{project_name}.h5"
            if not h5_path.exists():
                continue

            with h5py.File(h5_path, 'r') as f:
                specdata = f['specdata'][:]
                energy = specdata[0, :].astype(np.float32)
                spectra = specdata[1:, :].astype(np.float32)

            elem = elem_list[0][1]
            peak_config = {
                'centers': np.array([e.binding_energy for _, e in elem_list], dtype=np.float32),
                'sigmas': np.array([e.sigma for _, e in elem_list], dtype=np.float32),
                'gamma': elem.gamma,
            }

            all_amps = []
            for batch_start in range(0, spectra.shape[0], self.batch_size):
                batch_end = min(batch_start + self.batch_size, spectra.shape[0])
                Y_batch = spectra[batch_start:batch_end, :]

                result = self.pipeline.process_rowmajor(
                    Y_batch,
                    elem.symbol,
                    elem.orbital,
                    energy,
                    peak_config,
                )
                all_amps.append(result.amplitudes)

            amplitudes[file_key] = np.concatenate(all_amps, axis=1)

        return amplitudes

    def _run_voigtfit_inmemory(
        self, spectra_data: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """Run VoigtFit on in-memory spectra (no H5 I/O).

        Bypasses disk I/O entirely. Uses process_rowmajor() which leverages
        sampled chi2 for faster anomaly detection than process_rowmajor_fast().

        Args:
            spectra_data: Dict from SpectraGenerator.generate_inmemory()
                Each value has 'energy', 'spectra', 'amplitudes' keys.

        Returns:
            Dict mapping file_key to fitted amplitudes (n_comp, n_spectra)
        """
        amplitudes = {}

        for file_key, elem_list in self._elem_by_file.items():
            if file_key not in spectra_data:
                continue

            data = spectra_data[file_key]
            energy = data['energy']
            spectra = data['spectra']  # (n_spectra, n_energy) float32

            elem = elem_list[0][1]
            peak_config = {
                'centers': np.array([e.binding_energy for _, e in elem_list], dtype=np.float32),
                'sigmas': np.array([e.sigma for _, e in elem_list], dtype=np.float32),
                'gamma': elem.gamma,
            }

            all_amps = []
            for batch_start in range(0, spectra.shape[0], self.batch_size):
                batch_end = min(batch_start + self.batch_size, spectra.shape[0])
                Y_batch = spectra[batch_start:batch_end, :]

                result = self.pipeline.process_rowmajor(
                    Y_batch,
                    elem.symbol,
                    elem.orbital,
                    energy,
                    peak_config,
                )
                all_amps.append(result.amplitudes)

            amplitudes[file_key] = np.concatenate(all_amps, axis=1)

        return amplitudes

    def _run_fused_streaming(
        self,
        noise_config: 'NoiseConfig',
        decomposition_method: str = 'pinv',
    ) -> tuple[dict[str, np.ndarray], float, float]:
        """Fused generate-and-fit pipeline: batch-level streaming.

        MLX-fused architecture:
        - Matmul on GPU (A_mx @ P_mx + bg_mx)
        - Poisson noise via mx.compile()-fused kernel (4-5x vs unfused)
        - amplitudes_only=True (skip anomaly_detect + tile)
        - Gaussian/mixed noise falls back to NumPy path

        Only one batch (~0.8 GB) is alive at a time instead of entire
        file (~13 GB), avoiding page fault/swapping on large images.

        Returns:
            (fitted_amplitudes, gen_time, recon_time)
        """
        from toyomacro.voigtfit.spectra_generator import (
            add_gaussian_noise,
            add_mixed_noise,
            add_poisson_noise,
            add_poisson_noise_mlx_fused,
            decompose_image_to_amplitudes,
            voigt_profile,
        )

        try:
            import mlx.core as mx
            _has_mlx = True
        except ImportError:
            _has_mlx = False

        gen_time_total = 0.0
        recon_time_total = 0.0
        amplitudes = {}

        # Decompose image (shared across all files)
        t0 = time.perf_counter()
        image_amplitudes = decompose_image_to_amplitudes(
            self._generator.image, self._color_mapping, method=decomposition_method
        )
        gen_time_total += time.perf_counter() - t0

        config = self._generator.config

        for file_key, elem_list in self._elem_by_file.items():
            elem_specs = [e for _, e in elem_list]
            elem_indices = [idx for idx, _ in elem_list]

            # Energy axis (once per file)
            binding_energies = [e.binding_energy for e in elem_specs]
            e_min = min(binding_energies) - config.energy_edge
            e_max = max(binding_energies) + config.energy_edge
            energy = np.arange(e_min, e_max + config.energy_step / 2,
                              config.energy_step, dtype=np.float64).astype(np.float32)

            # Pre-compute normalized profiles (once per element, tiny)
            profiles = []
            for elem in elem_specs:
                prof = voigt_profile(energy, elem.binding_energy, elem.sigma, elem.gamma)
                prof = (prof / (prof.max() + 1e-10)).astype(np.float32)
                profiles.append(prof)

            peak_config = {
                'centers': np.array([e.binding_energy for e in elem_specs], dtype=np.float32),
                'sigmas': np.array([e.sigma for e in elem_specs], dtype=np.float32),
                'gamma': elem_specs[0].gamma,
            }

            # Pre-convert profiles to MLX (once per file_key)
            if _has_mlx:
                P_mx = mx.array(np.stack(profiles, axis=0))  # (n_comp, n_energy)

            # Pre-compute global_max for heteroscedastic Poisson noise
            global_max_val = None
            if (noise_config.noise_type in ('poisson', 'mixed')
                    and noise_config.heteroscedastic):
                # max spectrum ≈ max(sum_amps) × amp_scale × (1 + bg_base)
                sum_amps = np.zeros(self.n_pixels, dtype=np.float32)
                for elem_idx in elem_indices:
                    sum_amps += np.maximum(image_amplitudes[elem_idx], 0)
                max_sI0 = float(sum_amps.max()) * config.amplitude_scale
                global_max_val = max_sI0 * (1.0 + config.bg_base)
                del sum_amps
                if self.verbose:
                    print(f"  Heteroscedastic: global_max = {global_max_val:.2f}")

            all_amps = []
            n_spectra = self.n_pixels

            for batch_start in range(0, n_spectra, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_spectra)
                bs = batch_end - batch_start

                # --- Generate this batch's spectra ---
                t0 = time.perf_counter()

                # Build amplitude matrix A on NumPy (indexing is fast)
                n_profiles = len(profiles)
                A_np = np.zeros((bs, n_profiles), dtype=np.float32)
                for j, elem_idx in enumerate(elem_indices):
                    A_np[:, j] = np.maximum(
                        image_amplitudes[elem_idx, batch_start:batch_end], 0
                    )

                if _has_mlx:
                    # === MLX-fused path: matmul + noise on GPU ===
                    A_mx = mx.array(A_np * config.amplitude_scale)
                    bg_mx = mx.array(
                        (A_np.sum(axis=1, keepdims=True)
                         * (config.amplitude_scale * config.bg_base)).astype(np.float32)
                    )
                    del A_np

                    # Matmul on MLX (lazy)
                    spectra_mx = A_mx @ P_mx + bg_mx
                    del A_mx, bg_mx

                    # Noise on MLX (compiled kernel) or fallback to NumPy
                    if noise_config.noise_type == 'poisson':
                        spectra_mx = add_poisson_noise_mlx_fused(
                            spectra_mx, noise_config.poisson_level,
                            global_max=global_max_val,
                        )
                    elif noise_config.noise_type in ('gaussian', 'mixed'):
                        # Fallback: materialize, apply NumPy noise, convert back
                        mx.eval(spectra_mx)
                        spectra_np = np.array(spectra_mx, dtype=np.float32)
                        del spectra_mx
                        if noise_config.noise_type == 'gaussian':
                            spectra_np = add_gaussian_noise(
                                spectra_np, noise_config.gaussian_std)
                        else:
                            spectra_np = add_mixed_noise(
                                spectra_np,
                                noise_config.poisson_level,
                                noise_config.gaussian_std,
                                global_max=global_max_val)
                        spectra_mx = mx.array(spectra_np)
                        del spectra_np

                    # Materialize lazy graph before fit
                    mx.eval(spectra_mx)
                    gen_time_total += time.perf_counter() - t0

                    # --- Fit this batch immediately ---
                    t0 = time.perf_counter()
                    result = self.pipeline.process_rowmajor(
                        spectra_mx,
                        elem_specs[0].symbol,
                        elem_specs[0].orbital,
                        energy,
                        peak_config,
                        amplitudes_only=True,
                    )
                    recon_time_total += time.perf_counter() - t0
                    del spectra_mx
                else:
                    # === NumPy fallback path ===
                    spectra_batch = (config.amplitude_scale * A_np) @ np.stack(profiles, axis=0)
                    bg = (A_np.sum(axis=1) * (config.amplitude_scale * config.bg_base))
                    spectra_batch += bg[:, np.newaxis]
                    del A_np

                    if noise_config.noise_type == 'poisson':
                        spectra_batch = add_poisson_noise(
                            spectra_batch, noise_config.poisson_level)
                    elif noise_config.noise_type == 'gaussian':
                        spectra_batch = add_gaussian_noise(
                            spectra_batch, noise_config.gaussian_std)
                    elif noise_config.noise_type == 'mixed':
                        spectra_batch = add_mixed_noise(
                            spectra_batch,
                            noise_config.poisson_level,
                            noise_config.gaussian_std)

                    gen_time_total += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    result = self.pipeline.process_rowmajor(
                        spectra_batch,
                        elem_specs[0].symbol,
                        elem_specs[0].orbital,
                        energy,
                        peak_config,
                        amplitudes_only=True,
                    )
                    recon_time_total += time.perf_counter() - t0
                    del spectra_batch

                all_amps.append(result.amplitudes)
                gc.collect()

            amplitudes[file_key] = np.concatenate(all_amps, axis=1)

        return amplitudes, gen_time_total, recon_time_total

    def run_noise_sweep(
        self,
        noise_levels: list[str] | None = None,
        save_spectra: bool = True,
        use_timestamp: bool = True,
    ) -> NoiseSweepResult:
        """
        Run benchmark for multiple noise levels.

        Args:
            noise_levels: List of noise level names (default: all)
            save_spectra: Keep generated H5 files
            use_timestamp: Create timestamped subfolder

        Returns:
            NoiseSweepResult with all results and summary
        """
        if noise_levels is None:
            noise_levels = list(NOISE_LEVELS.keys())

        # Create timestamp folder
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S') if use_timestamp else ''
        if use_timestamp:
            output_dir = self.base_output_dir / timestamp
        else:
            output_dir = self.base_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        total_start = time.perf_counter()

        print(f"\n{'='*70}")
        print("NOISE SWEEP BENCHMARK")
        print(f"{'='*70}")
        print(f"Image: {self.image_path.name}")
        print(f"Output: {output_dir}")
        print(f"Noise levels: {len(noise_levels)}")

        # Run noise-free first as reference
        noisefree_result = None
        noisefree_amplitudes = None
        noisefree_reconstructed = None

        if 'None' in noise_levels:
            noisefree_result = self.run(
                'None',
                output_dir=output_dir,
                save_spectra=save_spectra,
                keep_images=True,
            )
            results['None'] = noisefree_result

            # Store reference data
            noisefree_reconstructed = noisefree_result.reconstructed_image

            # Get amplitudes for correlation calculation
            self._init_generator(output_dir, 'benchmark')
            spectra_dir = output_dir / 'benchmark_noise_none'
            if spectra_dir.exists():
                noisefree_amplitudes = self._run_voigtfit(spectra_dir, 'benchmark')

            noise_levels = [n for n in noise_levels if n != 'None']

        # Run other noise levels
        for noise_name in noise_levels:
            result = self.run(
                noise_name,
                output_dir=output_dir,
                save_spectra=save_spectra,
                keep_images=False,
                noisefree_amplitudes=noisefree_amplitudes,
                noisefree_reconstructed=noisefree_reconstructed,
            )
            results[noise_name] = result

        total_time = time.perf_counter() - total_start

        # Generate summary table
        summary = self._generate_summary_table(results)

        # Save results JSON
        results_dict = {k: v.to_dict() for k, v in results.items()}
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(summary)
        print(f"\nResults saved to: {output_dir}")

        return NoiseSweepResult(
            results=results,
            summary_table=summary,
            total_time=total_time,
            output_dir=str(output_dir),
            timestamp=timestamp,
        )

    def _generate_summary_table(self, results: dict[str, RoundtripResult]) -> str:
        """Generate summary table string."""
        lines = []
        lines.append("")
        lines.append("=" * 90)
        lines.append("ROUNDTRIP BENCHMARK SUMMARY")
        lines.append("=" * 90)
        lines.append("")
        lines.append(f"{'Level':>12} | {'PSNR Orig':>10} | {'PSNR NF':>10} | "
                     f"{'Gain':>8} | {'Corr':>8} | {'Time':>8} | {'Efficiency'}")
        lines.append("-" * 90)

        for noise_name in NOISE_LEVELS.keys():
            if noise_name not in results:
                continue
            r = results[noise_name]
            psnr_orig = r.psnr_vs_original  # Now a float
            psnr_nf = r.psnr_vs_noisefree   # Optional float
            psnr_nf_str = f'{psnr_nf:.2f}dB' if psnr_nf else 'N/A'

            # Efficiency: PSNR per second
            if psnr_nf:
                efficiency = psnr_nf / r.total_time
            else:
                efficiency = psnr_orig / r.total_time

            lines.append(
                f"{noise_name:>12} | {psnr_orig:>8.2f}dB | {psnr_nf_str:>10} | "
                f"{r.amp_gain_avg:>8.4f} | {r.amp_correlation:>8.4f} | "
                f"{r.total_time:>6.1f}s | {efficiency:.2f} dB/s"
            )

        lines.append("")
        lines.append("=" * 90)
        return "\n".join(lines)

    def save_result(
        self,
        result: RoundtripResult,
        output_dir: str | Path,
        prefix: str = '',
    ):
        """Save result images and metrics."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if prefix:
            prefix = f"{prefix}_"

        # Save images
        if result.reconstructed_image is not None:
            save_image(
                result.reconstructed_image,
                output_dir / f"{prefix}reconstructed_{result.noise_level}.png"
            )

        # Save metrics as JSON
        metrics_path = output_dir / f"{prefix}metrics_{result.noise_level}.json"
        with open(metrics_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)


# ============================================================================
# Convenience Functions
# ============================================================================

from ._data_paths import find_default_image, roundtrip_image_dir

# First image found under the (unbundled) roundtrip data tree; None when
# the user has not pointed VOIGTFIT_DATA_ROOT at a dataset yet.
_DEFAULT_IMAGE_PATH = find_default_image()


def run_roundtrip_benchmark(
    image_path: str | Path | None = None,
    noise_level: str | float = 'None',
    output_dir: str | Path | None = None,
    elements: str | ElementPreset = 'demo',
    save_spectra: bool = True,
    use_inmemory: bool | str = False,
    verbose: bool = True,
) -> RoundtripResult:
    """
    Run a single roundtrip benchmark.

    Args:
        image_path: Input image path (default: first image under the roundtrip data tree)
        noise_level: 'None', 'Moderate', etc. or numeric value
        output_dir: Output directory for results
        elements: Preset name or ElementPreset object (default: 'demo')
        save_spectra: Keep H5 files
        use_inmemory: False, True/'inmemory', or 'streaming' (fused gen+fit)
        verbose: Print progress

    Returns:
        RoundtripResult
    """
    if image_path is None:
        image_path = _DEFAULT_IMAGE_PATH

    benchmark = RoundtripBenchmark(
        image_path=image_path,
        output_dir=output_dir,
        elements=elements,
        verbose=verbose,
    )

    return benchmark.run(
        noise_level=noise_level,
        save_spectra=save_spectra,
        use_inmemory=use_inmemory,
    )


def run_noise_sweep(
    image_path: str | Path | None = None,
    noise_levels: list[str] | None = None,
    output_dir: str | Path | None = None,
    elements: str | ElementPreset = 'demo',
    save_spectra: bool = True,
    verbose: bool = True,
) -> NoiseSweepResult:
    """
    Run roundtrip benchmark across multiple noise levels.

    Args:
        image_path: Input image path (default: first image under the roundtrip data tree)
        noise_levels: List of noise levels (default: all 11 levels)
        output_dir: Output directory for results
        elements: Preset name or ElementPreset object (default: 'demo')
        save_spectra: Keep H5 files
        verbose: Print progress

    Returns:
        NoiseSweepResult with all results
    """
    if image_path is None:
        image_path = _DEFAULT_IMAGE_PATH

    benchmark = RoundtripBenchmark(
        image_path=image_path,
        output_dir=output_dir,
        elements=elements,
        verbose=verbose,
    )

    return benchmark.run_noise_sweep(
        noise_levels=noise_levels,
        save_spectra=save_spectra,
    )


def run_media_roundtrip(
    input_path: str | Path,
    output_path: str | Path | None = None,
    noise_levels: list[str] | None = None,
    noise_labels: list[str] | None = None,
    elements: str | ElementPreset = 'demo',
    duration: int = 50,
    verbose: bool = True,
    denoise_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    denoise_label: str = 'Denoised',
    image_denoise_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    image_denoise_label: str = 'Denoised',
) -> dict[str, Any]:
    """
    Media roundtrip demo: load any media (GIF, image, video, etc.),
    process each frame through Image -> Spectra -> VoigtFit -> Image,
    save result as side-by-side animation.

    Output format is auto-detected from output_path extension:
      .gif  -> GIF with global palette quantization
      .apng -> APNG with full 24-bit RGB
      .mp4  -> MP4 video (requires imageio[pyav])
      dir   -> Numbered PNGs

    Args:
        input_path: Input media path (GIF, PNG, JPG, MP4, directory, etc.)
        output_path: Output path (default: {input}_roundtrip.{ext})
        noise_levels: Noise levels to compare (preset names or numeric strings).
            Default: ['None'] -> 2 columns: Ground Truth | Reconstructed
        noise_labels: Optional display labels for each noise level column.
            If None, auto-generated from noise_levels.
        elements: Element preset name or object
        duration: Frame duration in ms (default: 50ms = 20fps)
        verbose: Print progress
        denoise_fn: Optional spectral denoiser function. Signature:
            denoise_fn(spectra: np.ndarray) -> np.ndarray
            Input/output shape: (n_spectra, n_energy) float32.
            Applied to noisy spectra BEFORE fitting.
        denoise_label: Label for spectral-denoised columns (default: 'Denoised')
        image_denoise_fn: Optional image denoiser function. Signature:
            image_denoise_fn(image: np.ndarray) -> np.ndarray
            Input/output shape: (H, W, 3) uint8.
            Applied to reconstructed RGB image AFTER fitting.
        image_denoise_label: Label for image-denoised columns (default: 'Denoised')

    Returns:
        Dict with 'psnr_per_frame', 'mean_psnr', 'total_time', 'output_path',
        'n_frames', 'throughput_spec_per_s'
    """
    from toyomacro.voigtfit.frame_io import open_frames
    from toyomacro.voigtfit.image_utils import (
        amplitudes_to_rgb,
        save_side_by_side,
    )
    from toyomacro.voigtfit.spectra_generator import (
        add_gaussian_noise,
        add_mixed_noise,
        add_poisson_noise_mlx,
        add_poisson_noise_mlx_fused,
        decompose_image_to_amplitudes,
        voigt_profile,
    )
    from toyomacro.voigtfit.spectra_generator import (
        get_element_preset as _get_preset,
    )
    try:
        import mlx.core as mx
        _HAS_MLX = True
    except ImportError:
        _HAS_MLX = False

    input_path = Path(input_path)
    if output_path is None:
        # Default: same directory, same extension
        out_ext = input_path.suffix if input_path.suffix else '.gif'
        output_path = input_path.parent / f"{input_path.stem}_roundtrip{out_ext}"
    output_path = Path(output_path)

    # Default noise levels
    if noise_levels is None:
        noise_levels = ['None']

    # Resolve preset
    if isinstance(elements, str):
        preset = _get_preset(elements)
    else:
        preset = elements

    # Load all frames (format-agnostic via frame_io)
    source = open_frames(input_path)
    frames = source.load_all()  # (n_frames, H, W, 3)
    n_frames, h, w, _ = frames.shape
    n_pixels = h * w
    image_shape = (h, w)

    if verbose:
        print("Media Roundtrip Demo")
        print(f"  Input: {input_path.name} ({w}x{h}, {n_frames} frames)")
        print(f"  Noise levels: {noise_levels}")
        print(f"  Elements: {preset.name} ({len(preset.elements)} components)")

    # Initialize pipeline (once, shared across all frames)
    cache = WeightMatrixCache()
    pipeline = HybridPipeline(cache, enable_stage2=False, use_mlx=True)

    color_mapping = preset.color_mapping
    component_order = preset.component_order
    elements_list = preset.elements

    # Group elements by file
    elem_by_file: dict[str, list[tuple[int, Any]]] = {}
    for idx, elem in enumerate(elements_list):
        fk = elem.file_key
        if fk not in elem_by_file:
            elem_by_file[fk] = []
        elem_by_file[fk].append((idx, elem))

    # Pre-compute Voigt profiles and peak configs (shared across frames)
    config = GeneratorConfig()
    file_configs = {}
    for file_key, elem_list in elem_by_file.items():
        elem_specs = [e for _, e in elem_list]
        elem_indices = [idx for idx, _ in elem_list]
        binding_energies = [e.binding_energy for e in elem_specs]
        e_min = min(binding_energies) - config.energy_edge
        e_max = max(binding_energies) + config.energy_edge
        energy = np.arange(e_min, e_max + config.energy_step / 2,
                          config.energy_step, dtype=np.float64).astype(np.float32)
        profiles = []
        for elem in elem_specs:
            prof = voigt_profile(energy, elem.binding_energy, elem.sigma, elem.gamma)
            prof = (prof / (prof.max() + 1e-10)).astype(np.float32)
            profiles.append(prof)
        peak_config = {
            'centers': np.array([e.binding_energy for e in elem_specs], dtype=np.float32),
            'sigmas': np.array([e.sigma for e in elem_specs], dtype=np.float32),
            'gamma': elem_specs[0].gamma,
        }
        file_configs[file_key] = {
            'elem_specs': elem_specs,
            'elem_indices': elem_indices,
            'energy': energy,
            'profiles': profiles,
            'peak_config': peak_config,
        }

    # Process each noise level (supports preset names or numeric values)
    noise_configs = {}
    for nl in noise_levels:
        if nl in NOISE_LEVELS:
            nv = NOISE_LEVELS[nl]
        else:
            try:
                nv = float(nl)
            except ValueError:
                import warnings
                warnings.warn(
                    f"Unknown noise level '{nl}', defaulting to 0 (noise-free). "
                    f"Valid names: {sorted(NOISE_LEVELS.keys())}",
                    stacklevel=2,
                )
                nv = 0
        if nv == 0:
            noise_configs[nl] = NoiseConfig(noise_type='none')
        else:
            noise_configs[nl] = NoiseConfig(noise_type='poisson', poisson_level=nv)

    # Ensure 'None' (noise-free) is always processed for vs-NoiseFree PSNR reference
    _nf_key = '_noisefree_ref'  # internal key, not shown in GIF columns
    _has_noisefree = any(noise_configs[nl].noise_type == 'none' for nl in noise_levels)
    if not _has_noisefree:
        noise_configs[_nf_key] = NoiseConfig(noise_type='none')

    # Build processing keys: each noise level, plus denoised variants
    # proc_keys tracks all columns to process
    # Variants: 'raw', 'spec_denoised' (spectral denoise), 'img_denoised' (image denoise)
    proc_keys = []
    for nl in noise_levels:
        proc_keys.append((nl, 'raw'))
        nc = noise_configs[nl]
        if denoise_fn is not None and nc.noise_type != 'none':
            proc_keys.append((nl, 'spec_denoised'))
        if image_denoise_fn is not None and nc.noise_type != 'none':
            proc_keys.append((nl, 'img_denoised'))

    # Results storage: {proc_key: (n_frames, H, W, 3) uint8}
    recon_frames = {pk: np.zeros_like(frames) for pk in proc_keys}
    psnr_per_frame = {pk: [] for pk in proc_keys}         # RGB-space PSNR
    amp_psnr_per_frame = {pk: [] for pk in proc_keys}     # Amplitude-space PSNR

    total_spectra = 0
    t_start = time.perf_counter()
    import gc

    from toyomacro.voigtfit.image_utils import psnr as _psnr_fn

    # ================================================================
    # Phase 1: Decompose all frames → element amplitudes
    # ================================================================
    all_image_amplitudes = []  # list of (n_comp, n_pixels) per frame
    for fi in range(n_frames):
        amp = decompose_image_to_amplitudes(
            frames[fi], color_mapping, method='pinv'
        )
        all_image_amplitudes.append(amp)

    if verbose:
        t_decompose = time.perf_counter() - t_start
        print(f"  Decomposition: {t_decompose:.2f}s")

    # ================================================================
    # Phase 2+3+4 Fused: chunk frames → gen → fit → PSNR → GC
    # ================================================================
    # Streaming architecture (like 8K mode): process frame chunks to bound
    # peak memory.  Only ONE chunk's spectra live in memory at a time.
    #   chunk_size = n_frames for small images, smaller for large ones.
    # Benchmark: With MLX-fused pipeline, 2M optimal.
    #   Old NumPy pipeline: 0.5M was optimal (CPU L2 cache pressure).
    #   MLX-fused: GPU prefers larger batches; Python per-chunk overhead dominates.
    #   2M → 17.7M spec/s; 0.5M → 11.5M spec/s; 8M → 9.0M spec/s.
    _MAX_CHUNK_SPECTRA = 2_000_000  # 2M pixels per chunk
    chunk_size = max(1, _MAX_CHUNK_SPECTRA // n_pixels)
    chunk_size = min(chunk_size, n_frames)  # don't exceed total frames

    # Determine variant list per noise level
    variant_map: dict[str, list[str]] = {}
    for nl in noise_levels:
        nc = noise_configs[nl]
        variants = ['raw']
        if denoise_fn is not None and nc.noise_type != 'none':
            variants.append('spec_denoised')
        variant_map[nl] = variants
    if not _has_noisefree:
        variant_map[_nf_key] = ['raw']

    # Which noise-free key to use for vs-NF PSNR reference
    _nf_ref_key = _nf_key
    if _has_noisefree:
        for nl in noise_levels:
            if noise_configs[nl].noise_type == 'none':
                _nf_ref_key = nl
                break

    all_nl_for_fit = list(noise_levels)
    if not _has_noisefree:
        all_nl_for_fit.append(_nf_key)

    n_comp_total = len(component_order)
    amp_psnr_vs_nf_per_frame = {pk: [] for pk in proc_keys}

    n_chunks = (n_frames + chunk_size - 1) // chunk_size
    if verbose:
        print(f"  Chunk: {chunk_size} frames × {n_chunks} chunks "
              f"({chunk_size * n_pixels:,} px/chunk)")

    for ci in range(n_chunks):
        f_start = ci * chunk_size
        f_end = min(f_start + chunk_size, n_frames)
        chunk_frames = f_end - f_start
        chunk_batch = chunk_frames * n_pixels

        # Concatenate this chunk's amplitudes: (n_comp, chunk_batch)
        chunk_amp = np.concatenate(
            all_image_amplitudes[f_start:f_end], axis=1
        )

        # --- Gen + Fit for each (file_key, nl) ---
        # Store fitted amplitudes for this chunk only
        chunk_fitted: dict[tuple[str, str, str], np.ndarray] = {}

        for file_key, fc in file_configs.items():
            elem_indices = fc['elem_indices']
            profiles = fc['profiles']
            energy = fc['energy']
            peak_config = fc['peak_config']
            elem_specs = fc['elem_specs']

            # Build amplitude matrix A on NumPy (indexing), then transfer once
            n_profiles = len(profiles)
            A_np = np.zeros((chunk_batch, n_profiles), dtype=np.float32)
            for j, elem_idx in enumerate(elem_indices):
                A_np[:, j] = np.maximum(chunk_amp[elem_idx, :], 0)

            # Pre-compute global_max for heteroscedastic mode
            _any_hetero = any(
                nc.heteroscedastic for nc in noise_configs.values()
                if nc.noise_type in ('poisson', 'mixed'))
            _gmax_val = None
            if _any_hetero:
                max_sI0 = float(A_np.sum(axis=1).max()) * config.amplitude_scale
                _gmax_val = max_sI0 * (1.0 + config.bg_base)

            if _HAS_MLX:
                # === MLX-fused path: matmul + noise + fit all on GPU ===
                P_mx = mx.array(np.stack(profiles, axis=0))
                A_mx = mx.array(A_np * config.amplitude_scale)
                bg_mx = mx.array(
                    (A_np.sum(axis=1, keepdims=True)
                     * (config.amplitude_scale * config.bg_base)).astype(np.float32)
                )
                del A_np

                # Matmul on MLX (lazy evaluation)
                all_clean_mx = A_mx @ P_mx + bg_mx
                del A_mx, bg_mx, P_mx

                for nl in all_nl_for_fit:
                    nc = noise_configs[nl]
                    _nc_gmax = _gmax_val if nc.heteroscedastic else None
                    if nc.noise_type == 'none':
                        spectra_mx = all_clean_mx
                    elif nc.noise_type == 'poisson':
                        # Poisson noise entirely on GPU — no copy, no transfer
                        spectra_mx = add_poisson_noise_mlx_fused(
                            all_clean_mx, nc.poisson_level,
                            global_max=_nc_gmax,
                        )
                    else:
                        # Gaussian/mixed: fall back to NumPy path
                        mx.eval(all_clean_mx)
                        all_clean_np = np.array(all_clean_mx, dtype=np.float32)
                        if nc.noise_type == 'gaussian':
                            spectra_np = add_gaussian_noise(
                                all_clean_np.copy(), nc.gaussian_std)
                        elif nc.noise_type == 'mixed':
                            spectra_np = add_mixed_noise(
                                all_clean_np.copy(),
                                nc.poisson_level, nc.gaussian_std,
                                global_max=_nc_gmax)
                        else:
                            spectra_np = all_clean_np
                        spectra_mx = mx.array(spectra_np)
                        del spectra_np, all_clean_np

                    for variant in variant_map[nl]:
                        if variant == 'spec_denoised':
                            # Denoise needs NumPy
                            mx.eval(spectra_mx)
                            fit_np = denoise_fn(
                                np.array(spectra_mx, dtype=np.float32).copy()
                            )
                            fit_input = mx.array(fit_np)
                            del fit_np
                        else:
                            fit_input = spectra_mx

                        # Materialize lazy graph, pass mx.array to VoigtFit
                        mx.eval(fit_input)
                        result = pipeline.process_rowmajor(
                            fit_input,
                            elem_specs[0].symbol,
                            elem_specs[0].orbital,
                            energy,
                            peak_config,
                            amplitudes_only=True,
                        )
                        chunk_fitted[(file_key, nl, variant)] = result.amplitudes
                        total_spectra += fit_input.shape[0]

                    if spectra_mx is not all_clean_mx:
                        del spectra_mx

                del all_clean_mx
            else:
                # === NumPy fallback path (no MLX) ===
                P = np.stack(profiles, axis=0)
                all_clean = (config.amplitude_scale * A_np) @ P
                bg_per_pixel = A_np.sum(axis=1) * (config.amplitude_scale * config.bg_base)
                all_clean += bg_per_pixel[:, np.newaxis]
                del A_np, P, bg_per_pixel

                for nl in all_nl_for_fit:
                    nc = noise_configs[nl]
                    if nc.noise_type == 'none':
                        spectra = all_clean
                    elif nc.noise_type == 'poisson':
                        spectra = add_poisson_noise_mlx(
                            all_clean.copy(), nc.poisson_level, force_gaussian=True
                        )
                    elif nc.noise_type == 'gaussian':
                        spectra = add_gaussian_noise(all_clean.copy(), nc.gaussian_std)
                    elif nc.noise_type == 'mixed':
                        spectra = add_mixed_noise(
                            all_clean.copy(), nc.poisson_level, nc.gaussian_std
                        )
                    else:
                        spectra = all_clean

                    for variant in variant_map[nl]:
                        fit_input = spectra
                        if variant == 'spec_denoised':
                            fit_input = denoise_fn(spectra.copy())

                        result = pipeline.process_rowmajor(
                            fit_input,
                            elem_specs[0].symbol,
                            elem_specs[0].orbital,
                            energy,
                            peak_config,
                            amplitudes_only=True,
                        )
                        chunk_fitted[(file_key, nl, variant)] = result.amplitudes
                        total_spectra += fit_input.shape[0]

                    if spectra is not all_clean:
                        del spectra

                del all_clean

            gc.collect()

        del chunk_amp

        # --- PSNR + RGB reconstruction for this chunk ---
        for fi_local in range(chunk_frames):
            fi = f_start + fi_local
            sl = slice(fi_local * n_pixels, (fi_local + 1) * n_pixels)
            gt_amp = all_image_amplitudes[fi]

            # Noise-free reference amplitudes
            nf_recon_amp = np.zeros((n_comp_total, n_pixels), dtype=np.float32)
            for k, (fk, comp_idx) in enumerate(component_order):
                nf_recon_amp[k] = chunk_fitted[(fk, _nf_ref_key, 'raw')][comp_idx, sl]

            for nl in noise_levels:
                nc = noise_configs[nl]
                for variant in variant_map[nl]:
                    pk = (nl, variant)

                    fitted_amplitudes: dict[str, np.ndarray] = {}
                    for file_key in file_configs:
                        fitted_amplitudes[file_key] = chunk_fitted[
                            (file_key, nl, variant)
                        ][:, sl]

                    # Amplitude-space PSNR vs GT
                    recon_amp = np.zeros((n_comp_total, n_pixels), dtype=np.float32)
                    for k, (fk, comp_idx) in enumerate(component_order):
                        recon_amp[k] = fitted_amplitudes[fk][comp_idx, :]
                    frame_amp_psnr = calculate_amplitude_psnr(gt_amp, recon_amp)
                    amp_psnr_per_frame[pk].append(frame_amp_psnr)

                    # Amplitude-space PSNR vs NoiseFree
                    frame_amp_psnr_nf = calculate_amplitude_psnr(
                        nf_recon_amp, recon_amp
                    )
                    amp_psnr_vs_nf_per_frame[pk].append(frame_amp_psnr_nf)

                    # Reconstruct RGB
                    recon_rgb = amplitudes_to_rgb(
                        amplitudes=fitted_amplitudes,
                        color_mapping=color_mapping,
                        component_order=component_order,
                        image_shape=image_shape,
                    )
                    recon_frames[pk][fi] = recon_rgb

                    # RGB-space PSNR
                    frame_psnr = _psnr_fn(frames[fi], recon_rgb)
                    psnr_per_frame[pk].append(frame_psnr)

                    # Image-space denoising
                    if (variant == 'raw'
                            and image_denoise_fn is not None
                            and nc.noise_type != 'none'):
                        pk_img = (nl, 'img_denoised')
                        denoised_rgb = image_denoise_fn(recon_rgb)
                        recon_frames[pk_img][fi] = denoised_rgb
                        img_psnr = _psnr_fn(frames[fi], denoised_rgb)
                        psnr_per_frame[pk_img].append(img_psnr)
                        amp_psnr_per_frame[pk_img].append(frame_amp_psnr)
                        amp_psnr_vs_nf_per_frame[pk_img].append(frame_amp_psnr_nf)

        del chunk_fitted
        gc.collect()

    if verbose:
        t_fit = time.perf_counter() - t_start
        print(f"  Spectra gen + VoigtFit: {t_fit:.2f}s "
              f"({total_spectra/1e6:.1f}M spectra, "
              f"{total_spectra/t_fit/1e6:.1f}M spec/s overall)")

    total_time = time.perf_counter() - t_start

    # Build side-by-side GIF columns from proc_keys
    frame_groups = [frames]  # Ground truth first
    labels = ['Ground Truth']
    psnr_columns = [None]  # GT has no PSNR

    for i, nl in enumerate(noise_levels):
        # Raw (noisy) column
        pk_raw = (nl, 'raw')
        frame_groups.append(recon_frames[pk_raw])
        if noise_labels is not None and i < len(noise_labels):
            raw_label = noise_labels[i]
        else:
            nc = noise_configs[nl]
            if nc.noise_type == 'none':
                raw_label = 'Reconstructed'
            else:
                raw_label = f'{nl} (noise={nc.poisson_level:.0f})'
        labels.append(raw_label)
        psnr_columns.append(amp_psnr_per_frame[pk_raw])

        # Spectral-denoised column (if exists)
        pk_spec = (nl, 'spec_denoised')
        if pk_spec in recon_frames:
            frame_groups.append(recon_frames[pk_spec])
            labels.append(denoise_label)
            psnr_columns.append(amp_psnr_per_frame[pk_spec])

        # Image-denoised column (if exists)
        pk_img = (nl, 'img_denoised')
        if pk_img in recon_frames:
            frame_groups.append(recon_frames[pk_img])
            labels.append(image_denoise_label)
            psnr_columns.append(amp_psnr_per_frame[pk_img])

    save_side_by_side(
        frame_groups=frame_groups,
        labels=labels,
        path=output_path,
        duration=duration,
        psnr_per_frame=psnr_columns,
    )

    # Summary
    mean_psnr = {}
    mean_amp_psnr = {}
    mean_amp_psnr_vs_nf = {}
    for pk in proc_keys:
        mean_psnr[pk] = float(np.mean(psnr_per_frame[pk]))
        mean_amp_psnr[pk] = float(np.mean(amp_psnr_per_frame[pk]))
        mean_amp_psnr_vs_nf[pk] = float(np.mean(amp_psnr_vs_nf_per_frame[pk]))

    if verbose:
        print(f"\n{'='*70}")
        print("Media Roundtrip Complete")
        print(f"  Frames: {n_frames}, Time: {total_time:.2f}s")
        print(f"  Total spectra: {total_spectra:,}")
        print(f"  Throughput: {total_spectra/total_time/1e6:.1f}M spec/s")
        print()
        hdr = f"  {'Noise':<20s} {'RGB PSNR':>10s} {'Amp vs GT':>10s} {'Amp vs NF':>10s}"
        print(hdr)
        print(f"  {'-'*50}")
        for pk in proc_keys:
            nl, variant = pk
            if variant == 'raw':
                tag = nl
            elif variant == 'spec_denoised':
                tag = f"{nl}+{denoise_label}"
            else:
                tag = f"{nl}+{image_denoise_label}"
            rgb_s = f"{mean_psnr[pk]:.2f} dB"
            gt_s = f"{mean_amp_psnr[pk]:.2f} dB"
            nf_s = f"{mean_amp_psnr_vs_nf[pk]:.2f} dB"
            print(f"  {tag:<20s} {rgb_s:>10s} {gt_s:>10s} {nf_s:>10s}")
        print(f"\n  Output: {output_path}")
        print(f"{'='*70}")

    return {
        'psnr_per_frame': psnr_per_frame,
        'amp_psnr_per_frame': amp_psnr_per_frame,
        'amp_psnr_vs_nf_per_frame': amp_psnr_vs_nf_per_frame,
        'mean_psnr': mean_psnr,
        'mean_amp_psnr': mean_amp_psnr,
        'mean_amp_psnr_vs_nf': mean_amp_psnr_vs_nf,
        'total_time': total_time,
        'output_path': str(output_path),
        'n_frames': n_frames,
        'total_spectra': total_spectra,
        'throughput_spec_per_s': total_spectra / total_time,
    }


def run_gif_roundtrip(
    gif_path: str | Path,
    output_path: str | Path | None = None,
    noise_levels: list[str] | None = None,
    noise_labels: list[str] | None = None,
    elements: str | ElementPreset = 'demo',
    duration: int = 50,
    verbose: bool = True,
    denoise_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    denoise_label: str = 'Denoised',
    image_denoise_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    image_denoise_label: str = 'Denoised',
) -> dict[str, Any]:
    """Backward-compatible alias for run_media_roundtrip().

    See run_media_roundtrip() for full documentation.
    """
    return run_media_roundtrip(
        input_path=gif_path,
        output_path=output_path,
        noise_levels=noise_levels,
        noise_labels=noise_labels,
        elements=elements,
        duration=duration,
        verbose=verbose,
        denoise_fn=denoise_fn,
        denoise_label=denoise_label,
        image_denoise_fn=image_denoise_fn,
        image_denoise_label=image_denoise_label,
    )


def quick_psnr_check(
    image_path: str | Path,
    noise_level: str = 'Moderate',
) -> float:
    """
    Quick PSNR check for a given image and noise level.

    Args:
        image_path: Input image
        noise_level: Noise level name

    Returns:
        PSNR value in dB
    """
    benchmark = RoundtripBenchmark(
        image_path=image_path,
        verbose=False,
    )
    result = benchmark.run(noise_level=noise_level, save_spectra=False, keep_images=False)
    return result.psnr_vs_original.psnr_total


# ============================================================================
# CLI Support
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Roundtrip Benchmark: Image -> Spectra -> VoigtFit -> Image'
    )
    parser.add_argument('--image', '-i', type=str, help='Input image path')
    parser.add_argument('--noise', '-n', type=str, default='None',
                       help='Noise level (None, Moderate, Strong, etc.)')
    parser.add_argument('--sweep', '-s', action='store_true',
                       help='Run all noise levels')
    parser.add_argument('--output', '-o', type=str, help='Output directory')
    parser.add_argument('--elements', '-e', type=str, default='demo',
                       help='Element preset name (default: demo)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save H5 files')
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--inmemory', action='store_true',
                           help='In-memory path (no H5 I/O)')
    mode_group.add_argument('--streaming', action='store_true',
                           help='Fused generate+fit per batch (fastest, lowest memory)')
    # Media roundtrip mode (--input replaces --gif, both supported)
    parser.add_argument('--input', type=str, default=None,
                       help='Media roundtrip: input path (GIF, image, MP4, etc.)')
    parser.add_argument('--gif', type=str, default=None,
                       help='(deprecated, use --input) Input animated GIF path')
    parser.add_argument('--noise-levels', type=str, nargs='*', default=None,
                       help='Noise levels for roundtrip (e.g., None Moderate Strong)')
    parser.add_argument('--gif-noise', type=str, nargs='*', default=None,
                       help='(deprecated, use --noise-levels) Noise levels for roundtrip')

    args = parser.parse_args()

    # Resolve --input / --gif (--input takes precedence)
    media_input = args.input or args.gif
    media_noise = args.noise_levels or args.gif_noise

    # Media roundtrip mode
    if media_input:
        noise = media_noise if media_noise else ['None']
        result = run_media_roundtrip(
            input_path=media_input,
            output_path=args.output,
            noise_levels=noise,
            elements=args.elements,
        )
        print(f"\nMedia roundtrip: {result['n_frames']} frames, "
              f"{result['throughput_spec_per_s']/1e6:.1f}M spec/s")
    elif args.sweep:
        # Determine mode
        if args.streaming:
            use_inmemory = 'streaming'
        elif args.inmemory:
            use_inmemory = True
        else:
            use_inmemory = False

        result = run_noise_sweep(
            image_path=args.image,
            output_dir=args.output,
            elements=args.elements,
            save_spectra=not args.no_save,
        )
        print(result.summary_table)
    else:
        # Determine mode
        if args.streaming:
            use_inmemory = 'streaming'
        elif args.inmemory:
            use_inmemory = True
        else:
            use_inmemory = False

        result = run_roundtrip_benchmark(
            image_path=args.image,
            noise_level=args.noise,
            output_dir=args.output,
            elements=args.elements,
            save_spectra=not args.no_save,
            use_inmemory=use_inmemory,
        )
        print(f"\nResult: {result}")
