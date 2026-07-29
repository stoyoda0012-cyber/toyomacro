"""
MLX-Fused Spectra Generation Pipeline Benchmark

Compares the current NumPy→MLX→NumPy pipeline with an MLX-fused pipeline
that keeps data on GPU throughout matmul→poisson→fit stages.

Before: NumPy(matmul) → np→mx(poisson) → mx→np(copy) → np→mx(fit)
After:  mx(matmul → poisson) → mx(fit) — only 1 transfer in, 1 out

Usage:
    python -m voigtfit.benchmarks.benchmark_mlx_pipeline
    python -m voigtfit.benchmarks.benchmark_mlx_pipeline --frames 20
Date: 2026-02-17
"""

import gc
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx

    from toyomacro.voigtfit._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND the default device can execute work
except ImportError:
    HAS_MLX = False


@dataclass
class PipelineTimings:
    """Detailed timings for a single pipeline run."""
    decompose: float = 0
    concat: float = 0
    matmul: float = 0
    copy: float = 0
    noise: float = 0
    transfer_to_mlx: float = 0  # explicit np→mx conversions
    transfer_to_np: float = 0   # explicit mx→np conversions
    fit: float = 0
    gc_time: float = 0
    total: float = 0
    total_spectra: int = 0
    n_fit_calls: int = 0
    n_noise_calls: int = 0

    @property
    def throughput(self) -> float:
        return self.total_spectra / self.total if self.total > 0 else 0

    @property
    def genfit_throughput(self) -> float:
        t = self.total - self.decompose
        return self.total_spectra / t if t > 0 else 0


def _get_peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / (1024 * 1024)


def run_before_pipeline(
    frames: np.ndarray,
    file_configs: dict,
    all_amps: list,
    noise_configs: dict,
    all_nl: list,
    config: Any,
    pipeline: Any,
    n_pixels: int,
    n_test: int,
) -> PipelineTimings:
    """Run the BEFORE (current) pipeline with detailed timing."""
    from toyomacro.voigtfit.spectra_generator import add_poisson_noise_mlx

    t = PipelineTimings()
    t0 = time.perf_counter()

    for ci in range(n_test):
        chunk_amp = all_amps[ci]

        for file_key, fc in file_configs.items():
            elem_indices = fc['elem_indices']
            profiles = fc['profiles']
            energy = fc['energy']
            peak_config = fc['peak_config']
            elem_specs = fc['elem_specs']

            # Matmul (NumPy)
            tm0 = time.perf_counter()
            n_profiles = len(profiles)
            A = np.zeros((n_pixels, n_profiles), dtype=np.float32)
            for j, elem_idx in enumerate(elem_indices):
                A[:, j] = np.maximum(chunk_amp[elem_idx, :], 0)
            P = np.stack(profiles, axis=0)
            all_clean = (config.amplitude_scale * A) @ P
            bg = A.sum(axis=1) * (config.amplitude_scale * config.bg_base)
            all_clean += bg[:, np.newaxis]
            del A, P, bg
            t.matmul += time.perf_counter() - tm0

            for nl in all_nl:
                nc = noise_configs[nl]
                if nc.noise_type == 'none':
                    spectra = all_clean
                elif nc.noise_type == 'poisson':
                    tc = time.perf_counter()
                    spectra_copy = all_clean.copy()
                    t.copy += time.perf_counter() - tc

                    tn = time.perf_counter()
                    spectra = add_poisson_noise_mlx(
                        spectra_copy, nc.poisson_level, force_gaussian=True
                    )
                    t.noise += time.perf_counter() - tn
                    t.n_noise_calls += 1
                else:
                    spectra = all_clean

                tf = time.perf_counter()
                result = pipeline.process_rowmajor(
                    spectra, elem_specs[0].symbol, elem_specs[0].orbital,
                    energy, peak_config,
                    amplitudes_only=True,
                )
                t.fit += time.perf_counter() - tf
                t.total_spectra += spectra.shape[0]
                t.n_fit_calls += 1

                if spectra is not all_clean:
                    del spectra

            del all_clean
            tg = time.perf_counter()
            gc.collect()
            t.gc_time += time.perf_counter() - tg

    t.total = time.perf_counter() - t0
    return t


def run_after_pipeline(
    frames: np.ndarray,
    file_configs: dict,
    all_amps: list,
    noise_configs: dict,
    all_nl: list,
    config: Any,
    pipeline: Any,
    n_pixels: int,
    n_test: int,
) -> PipelineTimings:
    """Run the AFTER (MLX-fused) pipeline with detailed timing.

    Key changes:
    1. Matmul on MLX (A@P+bg all on GPU)
    2. Poisson noise takes mx.array directly (no np→mx transfer)
    3. Pass mx.array to process_rowmajor (no mx→np→mx roundtrip)
    4. GC only per chunk, not per file_key
    """
    t = PipelineTimings()
    t0 = time.perf_counter()

    # Pre-convert profiles to MLX (done once per file_key)
    mlx_profiles = {}
    for file_key, fc in file_configs.items():
        P_np = np.stack(fc['profiles'], axis=0)  # (n_profiles, n_energy)
        mlx_profiles[file_key] = mx.array(P_np)

    for ci in range(n_test):
        chunk_amp = all_amps[ci]

        for file_key, fc in file_configs.items():
            elem_indices = fc['elem_indices']
            energy = fc['energy']
            peak_config = fc['peak_config']
            elem_specs = fc['elem_specs']
            n_profiles = len(fc['profiles'])

            # Build A on NumPy (indexing is fast), then transfer once
            tm0 = time.perf_counter()
            A_np = np.zeros((n_pixels, n_profiles), dtype=np.float32)
            for j, elem_idx in enumerate(elem_indices):
                A_np[:, j] = np.maximum(chunk_amp[elem_idx, :], 0)

            # Transfer to MLX once
            A_mx = mx.array(A_np * config.amplitude_scale)
            bg_mx = mx.array(
                A_np.sum(axis=1, keepdims=True) * (config.amplitude_scale * config.bg_base)
            )
            del A_np
            P_mx = mlx_profiles[file_key]

            # Matmul on MLX (lazy — not computed yet)
            all_clean_mx = A_mx @ P_mx + bg_mx
            del A_mx, bg_mx
            t.matmul += time.perf_counter() - tm0

            for nl in all_nl:
                nc = noise_configs[nl]
                if nc.noise_type == 'none':
                    # No noise: evaluate matmul result
                    spectra_mx = all_clean_mx
                elif nc.noise_type == 'poisson':
                    # Poisson noise entirely on MLX — NO copy, NO np→mx transfer
                    tn = time.perf_counter()
                    spectra_mx = _add_poisson_noise_mlx_fused(
                        all_clean_mx, nc.poisson_level
                    )
                    t.noise += time.perf_counter() - tn
                    t.n_noise_calls += 1
                else:
                    spectra_mx = all_clean_mx

                # Materialize the lazy graph (matmul+noise) before timing fit
                te = time.perf_counter()
                mx.eval(spectra_mx)
                t_eval = time.perf_counter() - te
                # Attribute eval time to matmul+noise proportionally
                if nc.noise_type == 'poisson':
                    t.noise += t_eval * 0.5
                    t.matmul += t_eval * 0.5
                else:
                    t.matmul += t_eval

                # Pass mx.array directly to VoigtFit — skips np→mx in process_rowmajor
                tf = time.perf_counter()
                result = pipeline.process_rowmajor(
                    spectra_mx, elem_specs[0].symbol, elem_specs[0].orbital,
                    energy, peak_config,
                    amplitudes_only=True,
                )
                t.fit += time.perf_counter() - tf
                t.total_spectra += spectra_mx.shape[0]
                t.n_fit_calls += 1

                if spectra_mx is not all_clean_mx:
                    del spectra_mx

            del all_clean_mx

        # GC once per chunk (not per file_key)
        tg = time.perf_counter()
        gc.collect()
        t.gc_time += time.perf_counter() - tg

    t.total = time.perf_counter() - t0
    return t


def _add_poisson_noise_mlx_fused(
    data_mx: 'mx.array',
    level: float,
) -> 'mx.array':
    """Cornish-Fisher Poisson noise, MLX-native input/output.

    Unlike add_poisson_noise_mlx(), this takes and returns mx.array directly,
    avoiding NumPy↔MLX round-trips.

    Args:
        data_mx: Clean spectra on MLX (n_spectra, n_energy), float32
        level: Noise level (1 to 1e6)

    Returns:
        Noisy spectra as mx.array (same shape)
    """
    if level <= 0:
        return data_mx

    data_pos = mx.maximum(data_mx, 0)
    data_max = mx.max(data_pos)

    snr_target = 10000.0 / level
    lambda_scale = snr_target * snr_target

    # Normalize and scale (all lazy on GPU)
    scaled = (lambda_scale / data_max) * data_pos
    scaled = mx.clip(scaled, 0, 1e9)

    # Cornish-Fisher approximation
    std = mx.sqrt(mx.maximum(scaled, 1.0))
    z = mx.random.normal(shape=scaled.shape, dtype=mx.float32)
    gamma1 = 1.0 / std
    z = z + gamma1 * (z * z - 1.0) / 6.0
    noisy = mx.maximum(scaled + std * z, 0)

    # Scale back
    result = noisy * (data_max / lambda_scale)
    return result


def run_benchmark(
    n_test: int = 10,
    gif_path: str | None = None,
    input_path: str | None = None,
) -> tuple[PipelineTimings, PipelineTimings]:
    """Run before/after benchmark comparison."""
    from toyomacro.voigtfit.image_utils import load_image
    from toyomacro.voigtfit.pipeline import HybridPipeline
    from toyomacro.voigtfit.spectra_generator import (
        GeneratorConfig,
        NoiseConfig,
        decompose_image_to_amplitudes,
        get_element_preset,
        voigt_profile,
    )
    from toyomacro.voigtfit.weight_cache import WeightMatrixCache

    from ._data_paths import roundtrip_image_dir

    # Resolve input path (--input takes precedence over --gif)
    media_path = input_path or gif_path
    if media_path is None:
        from ._data_paths import find_default_image

        found = find_default_image()
        if found is None:
            raise SystemExit(
                'No input media found — pass --input/--gif or set '
                'VOIGTFIT_DATA_ROOT to a tree containing images.'
            )
        media_path = str(found)

    frames = load_image(media_path, all_frames=True)
    n_frames, h, w, _ = frames.shape
    n_pixels = h * w
    print(f"Loaded: {w}x{h}, {n_frames} frames, {n_pixels:,} px/frame")

    preset = get_element_preset('demo')
    config = GeneratorConfig()

    color_mapping = preset.color_mapping
    elements_list = preset.elements

    # Group elements by file
    elem_by_file = {}
    for idx, elem in enumerate(elements_list):
        fk = elem.file_key
        if fk not in elem_by_file:
            elem_by_file[fk] = []
        elem_by_file[fk].append((idx, elem))

    # Pre-compute profiles
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
            'elem_specs': elem_specs, 'elem_indices': elem_indices,
            'energy': energy, 'profiles': profiles, 'peak_config': peak_config,
        }

    print(f"File keys: {list(file_configs.keys())} "
          f"(channels: {[len(fc['energy']) for fc in file_configs.values()]})")

    noise_levels = ['None', 'Moderate']
    noise_configs = {
        'None': NoiseConfig(noise_type='none'),
        'Moderate': NoiseConfig(noise_type='poisson', poisson_level=1e3),
    }
    all_nl = list(noise_levels)

    # Decompose frames
    print(f"\nDecomposing {n_test} frames...", flush=True)
    all_amps = []
    for fi in range(min(n_test, n_frames)):
        amp = decompose_image_to_amplitudes(
            frames[fi], color_mapping, method='pinv'
        )
        all_amps.append(amp)
    print(f"  Done ({len(all_amps)} frames)")

    # Warmup both pipelines
    print("Warmup (before)...", end='', flush=True)
    cache_b = WeightMatrixCache()
    pipeline_b = HybridPipeline(cache_b, enable_stage2=False, use_mlx=True)
    _ = run_before_pipeline(
        frames, file_configs, all_amps[:2], noise_configs, all_nl,
        config, pipeline_b, n_pixels, 2,
    )
    gc.collect()
    print(" done")

    print("Warmup (after)...", end='', flush=True)
    cache_a = WeightMatrixCache()
    pipeline_a = HybridPipeline(cache_a, enable_stage2=False, use_mlx=True)
    _ = run_after_pipeline(
        frames, file_configs, all_amps[:2], noise_configs, all_nl,
        config, pipeline_a, n_pixels, 2,
    )
    gc.collect()
    print(" done")

    # Run BEFORE
    print(f"\nRunning BEFORE ({n_test} frames)...", flush=True)
    gc.collect()
    rss_before_start = _get_peak_rss_mb()
    t_before = run_before_pipeline(
        frames, file_configs, all_amps, noise_configs, all_nl,
        config, pipeline_b, n_pixels, n_test,
    )
    rss_before = _get_peak_rss_mb()
    print(f"  Done: {t_before.throughput/1e6:.2f} M spec/s, {t_before.total:.2f}s")

    gc.collect()

    # Run AFTER
    print(f"Running AFTER ({n_test} frames)...", flush=True)
    rss_after_start = _get_peak_rss_mb()
    t_after = run_after_pipeline(
        frames, file_configs, all_amps, noise_configs, all_nl,
        config, pipeline_a, n_pixels, n_test,
    )
    rss_after = _get_peak_rss_mb()
    print(f"  Done: {t_after.throughput/1e6:.2f} M spec/s, {t_after.total:.2f}s")

    # Print report
    print(f"\n{'='*70}")
    print(f"MLX-Fused Pipeline Benchmark: {w}x{h}, {n_test} frames")
    print(f"  Total spectra: {t_before.total_spectra:,} (each)")
    print(f"{'='*70}")

    phases = [
        ('Matmul',        t_before.matmul,  t_after.matmul),
        ('Copy (.copy())', t_before.copy,   t_after.copy),
        ('Poisson noise', t_before.noise,   t_after.noise),
        ('VoigtFit',      t_before.fit,     t_after.fit),
        ('GC',            t_before.gc_time, t_after.gc_time),
    ]

    print(f"\n{'Phase':<18s} {'Before(s)':>10s} {'After(s)':>10s} {'Speedup':>8s}")
    print('-' * 50)
    for name, tb, ta in phases:
        if tb > 0 and ta > 0:
            speedup = f"{tb/ta:.1f}x"
        elif ta == 0:
            speedup = "inf"
        else:
            speedup = "-"
        print(f"  {name:<16s} {tb:>9.3f}s {ta:>9.3f}s {speedup:>7s}")

    t_before_sum = sum(p[1] for p in phases)
    t_after_sum = sum(p[2] for p in phases)
    print(f"  {'Sum':<16s} {t_before_sum:>9.3f}s {t_after_sum:>9.3f}s "
          f"{t_before_sum/t_after_sum:.1f}x" if t_after_sum > 0 else "")
    print(f"  {'TOTAL':<16s} {t_before.total:>9.3f}s {t_after.total:>9.3f}s "
          f"{t_before.total/t_after.total:.1f}x" if t_after.total > 0 else "")

    print("\nThroughput:")
    print(f"  Before: {t_before.throughput/1e6:.2f} M spec/s")
    print(f"  After:  {t_after.throughput/1e6:.2f} M spec/s")
    if t_after.throughput > 0:
        print(f"  Speedup: {t_after.throughput/t_before.throughput:.2f}x")

    print("\nPoisson noise detail:")
    if t_before.n_noise_calls > 0:
        print(f"  Before: {t_before.noise/t_before.n_noise_calls*1000:.2f} ms/call "
              f"({t_before.n_noise_calls * n_pixels / t_before.noise / 1e6:.1f} M spec/s)")
    if t_after.n_noise_calls > 0:
        print(f"  After:  {t_after.noise/t_after.n_noise_calls*1000:.2f} ms/call "
              f"({t_after.n_noise_calls * n_pixels / t_after.noise / 1e6:.1f} M spec/s)")

    print(f"\nMemory (peak RSS): Before={rss_before:.0f}MB, After={rss_after:.0f}MB")

    return t_before, t_after


def plot_results(t_before: PipelineTimings, t_after: PipelineTimings,
                 output_path: str | None = None) -> None:
    """Generate 4-panel comparison plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Phase breakdown stacked bar
    ax = axes[0, 0]
    phases = {
        'Matmul': (t_before.matmul, t_after.matmul),
        'Copy': (t_before.copy, t_after.copy),
        'Poisson': (t_before.noise, t_after.noise),
        'VoigtFit': (t_before.fit, t_after.fit),
        'GC': (t_before.gc_time, t_after.gc_time),
    }
    labels = list(phases.keys())
    before_vals = [phases[k][0] for k in labels]
    after_vals = [phases[k][1] for k in labels]
    other_before = t_before.total - sum(before_vals)
    other_after = t_after.total - sum(after_vals)
    labels.append('Other')
    before_vals.append(max(other_before, 0))
    after_vals.append(max(other_after, 0))

    x = np.arange(2)
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#95a5a6', '#bdc3c7']
    bottom_b = np.zeros(1)
    bottom_a = np.zeros(1)
    for i, label in enumerate(labels):
        ax.bar(0, before_vals[i], bottom=bottom_b[0], color=colors[i], label=label, width=0.6)
        ax.bar(1, after_vals[i], bottom=bottom_a[0], color=colors[i], width=0.6)
        bottom_b[0] += before_vals[i]
        bottom_a[0] += after_vals[i]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Before\n(NumPy+MLX)', 'After\n(MLX-fused)'])
    ax.set_ylabel('Time (s)')
    ax.set_title('Phase Breakdown')
    ax.legend(fontsize=8, loc='upper right')

    # Panel 2: Throughput comparison
    ax = axes[0, 1]
    tp_before = t_before.throughput / 1e6
    tp_after = t_after.throughput / 1e6
    bars = ax.bar(['Before', 'After'], [tp_before, tp_after],
                  color=['#3498db', '#e74c3c'], width=0.5)
    ax.set_ylabel('Throughput (M spec/s)')
    ax.set_title(f'Overall Throughput ({tp_after/tp_before:.1f}x speedup)')
    for bar, val in zip(bars, [tp_before, tp_after]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Panel 3: Per-phase speedup
    ax = axes[1, 0]
    phase_names = ['Matmul', 'Copy', 'Poisson', 'VoigtFit', 'GC', 'TOTAL']
    speedups = []
    for name in phase_names:
        if name == 'TOTAL':
            speedups.append(t_before.total / t_after.total if t_after.total > 0 else 0)
        elif name == 'Matmul':
            speedups.append(t_before.matmul / t_after.matmul if t_after.matmul > 0 else 0)
        elif name == 'Copy':
            speedups.append(t_before.copy / max(t_after.copy, 0.001))
        elif name == 'Poisson':
            speedups.append(t_before.noise / t_after.noise if t_after.noise > 0 else 0)
        elif name == 'VoigtFit':
            speedups.append(t_before.fit / t_after.fit if t_after.fit > 0 else 0)
        elif name == 'GC':
            speedups.append(t_before.gc_time / t_after.gc_time if t_after.gc_time > 0 else 0)
    bar_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#95a5a6', '#8e44ad']
    bars = ax.bar(phase_names, speedups, color=bar_colors, width=0.6)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Per-Phase Speedup (Before/After)')
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.1f}x', ha='center', va='bottom', fontsize=9)

    # Panel 4: Time breakdown pie charts
    ax = axes[1, 1]
    # Show after pipeline breakdown
    after_phases = {
        'Matmul': t_after.matmul,
        'Poisson': t_after.noise,
        'VoigtFit': t_after.fit,
        'GC': t_after.gc_time,
        'Other': max(t_after.total - t_after.matmul - t_after.noise
                     - t_after.fit - t_after.gc_time - t_after.copy, 0),
    }
    if t_after.copy > 0.01:
        after_phases['Copy'] = t_after.copy
    vals = [v for v in after_phases.values() if v > 0.01]
    lbls = [k for k, v in after_phases.items() if v > 0.01]
    pie_colors = ['#e74c3c', '#3498db', '#2ecc71', '#95a5a6', '#bdc3c7', '#f39c12']
    ax.pie(vals, labels=lbls, colors=pie_colors[:len(vals)],
           autopct='%1.1f%%', startangle=90)
    ax.set_title('After Pipeline Breakdown')

    fig.suptitle('MLX-Fused Pipeline: Before vs After (960x540, M3 Max)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path is None:
        output_path = (Path(__file__).parent.parent.parent
                       / 'outputs' / 'mlx_pipeline_benchmark_results.png')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='MLX-Fused Spectra Pipeline Benchmark')
    parser.add_argument('--frames', type=int, default=10,
                       help='Number of frames to test (default: 10)')
    parser.add_argument('--input', type=str, default=None,
                       help='Input media path (image, GIF, directory, etc.)')
    parser.add_argument('--gif', type=str, default=None,
                       help='(deprecated alias for --input) Input GIF path')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output plot path')
    args = parser.parse_args()

    t_before, t_after = run_benchmark(
        n_test=args.frames,
        input_path=args.input,
        gif_path=args.gif,
    )
    plot_results(t_before, t_after, output_path=args.output)
