"""
PSNR vs Noise Level Benchmark.

Runs complete pipeline: Image → Spectra (with noise) → VoigtFit → Reconstruct → PSNR
Date: 2026-01-22
"""

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..image_utils import PSNRResult, load_image, save_image
from ..spectra_generator import NoiseConfig, SpectraGenerator
from .reconstruction_benchmark import ReconstructionBenchmark


@dataclass
class NoiseBenchmarkResult:
    """Result from a single noise level test."""
    noise_config: NoiseConfig
    generation_time: float
    fitting_time: float
    psnr_result: PSNRResult
    output_dir: Path


def run_psnr_noise_benchmark(
    image_path: str,
    output_base_dir: str,
    poisson_levels: list[float] = [0, 1e2, 1e3, 1e4, 1e5],
    gaussian_std: float = 0.0,
    project_name: str = 'psnr_benchmark',
    verbose: bool = True,
) -> dict[str, NoiseBenchmarkResult]:
    """
    Run complete PSNR benchmark across multiple noise levels.

    Args:
        image_path: Path to source image
        output_base_dir: Base directory for output files
        poisson_levels: List of Poisson noise levels to test
        gaussian_std: Fixed Gaussian noise std (0 = pure Poisson test)
        project_name: Project name for file naming
        verbose: Print progress

    Returns:
        Dict mapping noise description to NoiseBenchmarkResult
    """
    image_path = Path(image_path)
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Load original image for PSNR comparison
    original_image = load_image(image_path)

    results = {}

    for poisson_level in poisson_levels:
        # Create noise config
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

        noise_key = str(noise_config)

        if verbose:
            print("\n" + "=" * 70)
            print(f"Testing: {noise_key}")
            print("=" * 70)

        # ===== Step 1: Generate spectra with noise =====
        if verbose:
            print("\n[Step 1] Generating spectra...")

        gen_start = time.perf_counter()

        generator = SpectraGenerator(
            output_dir=output_base_dir,
            image_path=image_path,
            project_name=project_name,
        )
        generator.configure_fuji_elements()

        output_dir = generator.generate(
            noise_config=noise_config,
            verbose=verbose,
        )

        gen_time = time.perf_counter() - gen_start

        # ===== Step 2: Run VoigtFit =====
        if verbose:
            print("\n[Step 2] Running VoigtFit...")

        fit_start = time.perf_counter()

        benchmark = ReconstructionBenchmark(
            project_dir=output_dir,
            project_name=project_name,
            original_image_path=image_path,
        )
        benchmark.configure_fuji_8k()
        # Override image shape based on actual image
        benchmark.image_shape = original_image.shape[:2]

        fit_result = benchmark.run(verbose=verbose)

        fit_time = time.perf_counter() - fit_start

        # ===== Step 3: Calculate PSNR =====
        psnr_result = fit_result.psnr_vs_original

        if verbose:
            print(f"\n[Result] PSNR vs Original: {psnr_result.psnr_total:.2f} dB")
            print(f"  Generation: {gen_time:.1f}s, Fitting: {fit_time:.1f}s")

        # Save reconstructed image
        recon_path = output_dir / f"reconstructed_{noise_key}.png"
        save_image(fit_result.reconstructed_image, recon_path)

        results[noise_key] = NoiseBenchmarkResult(
            noise_config=noise_config,
            generation_time=gen_time,
            fitting_time=fit_time,
            psnr_result=psnr_result,
            output_dir=output_dir,
        )

    return results


def plot_psnr_vs_noise(
    results: dict[str, NoiseBenchmarkResult],
    output_path: str | None = None,
    title: str = "PSNR vs Noise Level",
) -> None:
    """
    Plot PSNR vs noise level.

    Args:
        results: Dict from run_psnr_noise_benchmark
        output_path: Optional path to save figure
        title: Plot title
    """
    # Extract data
    noise_levels = []
    psnr_values = []
    psnr_r = []
    psnr_g = []
    psnr_b = []
    labels = []

    for key, result in results.items():
        noise_config = result.noise_config

        if noise_config.noise_type == 'none':
            noise_level = 0
            label = 'No noise'
        elif noise_config.noise_type == 'poisson':
            noise_level = noise_config.poisson_level
            label = f'Poisson {noise_level:.0e}'
        elif noise_config.noise_type == 'gaussian':
            noise_level = noise_config.gaussian_std
            label = f'Gaussian {noise_level:.0e}'
        else:
            noise_level = noise_config.poisson_level
            label = f'Mixed P{noise_level:.0e}'

        noise_levels.append(noise_level if noise_level > 0 else 0.1)  # Avoid log(0)
        psnr_values.append(result.psnr_result.psnr_total)
        psnr_r.append(result.psnr_result.psnr_r)
        psnr_g.append(result.psnr_result.psnr_g)
        psnr_b.append(result.psnr_result.psnr_b)
        labels.append(label)

    # Sort by noise level
    sorted_idx = np.argsort(noise_levels)
    noise_levels = [noise_levels[i] for i in sorted_idx]
    psnr_values = [psnr_values[i] for i in sorted_idx]
    psnr_r = [psnr_r[i] for i in sorted_idx]
    psnr_g = [psnr_g[i] for i in sorted_idx]
    psnr_b = [psnr_b[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Total PSNR vs noise level
    ax1.semilogx(noise_levels, psnr_values, 'ko-', markersize=10, linewidth=2)
    for i, (x, y, label) in enumerate(zip(noise_levels, psnr_values, labels)):
        ax1.annotate(f'{y:.1f} dB', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    ax1.set_xlabel('Noise Level (Poisson λ)', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title(f'{title} - Total', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Add reference lines
    ax1.axhline(y=40, color='g', linestyle='--', alpha=0.5, label='Excellent (>40 dB)')
    ax1.axhline(y=30, color='y', linestyle='--', alpha=0.5, label='Good (30-40 dB)')
    ax1.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Fair (20-30 dB)')
    ax1.legend(loc='lower left')

    # Plot 2: Per-channel PSNR
    x = np.arange(len(labels))
    width = 0.25

    ax2.bar(x - width, psnr_r, width, label='R', color='red', alpha=0.7)
    ax2.bar(x, psnr_g, width, label='G', color='green', alpha=0.7)
    ax2.bar(x + width, psnr_b, width, label='B', color='blue', alpha=0.7)

    ax2.set_xlabel('Noise Configuration', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title(f'{title} - Per Channel', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_path}")

    plt.show()


def print_benchmark_summary(results: dict[str, NoiseBenchmarkResult]) -> None:
    """Print formatted benchmark summary."""
    print("\n" + "=" * 80)
    print("PSNR NOISE BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"\n{'Noise Config':<25} {'PSNR (dB)':<12} {'R':<8} {'G':<8} {'B':<8} {'MAE':<8}")
    print("-" * 80)

    for key, result in results.items():
        r = result.psnr_result
        print(f"{key:<25} {r.psnr_total:<12.2f} {r.psnr_r:<8.2f} {r.psnr_g:<8.2f} {r.psnr_b:<8.2f} {r.mean_abs_error:<8.2f}")

    print("-" * 80)
    print("\nInterpretation:")
    print("  > 40 dB: Excellent (nearly identical)")
    print("  30-40 dB: Good (minor differences)")
    print("  20-30 dB: Fair (visible differences)")
    print("  < 20 dB: Poor (significant differences)")
    print("=" * 80)


if __name__ == '__main__':
    from ._data_paths import fuji_dir

    _fuji = fuji_dir()
    image_path = str(_fuji / 'churei-tower-mount-fuji-in-japan-8k-68-7680x4320.jpg')
    output_dir = str(_fuji)

    results = run_psnr_noise_benchmark(
        image_path=image_path,
        output_base_dir=output_dir,
        poisson_levels=[0, 1e2, 1e3, 1e4],  # Test subset first
        project_name='fuji_psnr_test',
    )

    print_benchmark_summary(results)

    # Plot results
    plot_psnr_vs_noise(
        results,
        output_path=f'{output_dir}/psnr_vs_noise.png',
    )
