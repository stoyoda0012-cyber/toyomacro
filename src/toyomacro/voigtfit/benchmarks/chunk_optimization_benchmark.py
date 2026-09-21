"""
Chunk Size Optimization Benchmark for GIF Roundtrip Pipeline

Measures throughput vs chunk size across multiple resolutions to find
the optimal _MAX_CHUNK_SPECTRA parameter. Analyzes cache hierarchy
effects (L1d=64KB, L2=16MB, L3=none on M3 Max) on spectral processing.

Key insight: As resolution increases, per-chunk overhead (GC, MLX sync,
Python loop) is amortized over more spectra, revealing the asymptotic
throughput ceiling.

Usage:
    python -m toyomacro.voigtfit.benchmarks.chunk_optimization_benchmark
    python -m toyomacro.voigtfit.benchmarks.chunk_optimization_benchmark --quick
    python -m toyomacro.voigtfit.benchmarks.chunk_optimization_benchmark --resolutions 320x180 640x360 1280x720
"""

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toyomacro.voigtfit.memory import get_peak_rss_bytes


@dataclass
class ChunkBenchmarkResult:
    """Result of a single chunk-size × resolution measurement."""
    resolution: str  # e.g. "640x360"
    width: int
    height: int
    n_pixels: int
    n_frames: int
    chunk_spectra: int  # _MAX_CHUNK_SPECTRA value used
    chunk_size_frames: int  # frames per chunk
    n_chunks: int
    total_spectra: int
    wall_time: float  # seconds
    throughput: float  # spec/s
    peak_rss_mb: float
    # Phase breakdown
    decompose_time: float
    gen_fit_time: float
    gen_fit_throughput: float  # spec/s for gen+fit only (excl. decompose+PSNR)


def _get_peak_rss_mb() -> float:
    """Get peak RSS in MiB."""
    return get_peak_rss_bytes() / (1024 * 1024)


def _make_synthetic_gif(width: int, height: int, n_frames: int = 100
                        ) -> np.ndarray:
    """Create a synthetic animated GIF-like array for benchmarking.

    Uses sine+cosine color patterns that vary across frames to simulate
    a real GIF without needing file I/O.
    """
    frames = np.zeros((n_frames, height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    yy = yy.astype(np.float32) / max(height, 1)
    xx = xx.astype(np.float32) / max(width, 1)

    for fi in range(n_frames):
        phase = fi / max(n_frames, 1) * 2 * np.pi
        r = np.clip(128 + 120 * np.sin(2*np.pi*xx + phase), 0, 255)
        g = np.clip(128 + 120 * np.sin(2*np.pi*yy + phase * 0.7), 0, 255)
        b = np.clip(128 + 120 * np.cos(2*np.pi*(xx+yy) + phase * 1.3), 0, 255)
        frames[fi, :, :, 0] = r.astype(np.uint8)
        frames[fi, :, :, 1] = g.astype(np.uint8)
        frames[fi, :, :, 2] = b.astype(np.uint8)

    return frames


def _load_or_synthesize(resolution: str, gif_dir: Path | None) -> np.ndarray:
    """Load a real GIF if available, otherwise synthesize."""
    w, h = map(int, resolution.split('x'))

    if gif_dir is not None:
        # Try to find a pre-existing GIF at this resolution
        candidates = list(gif_dir.glob(f'*_{resolution}.gif'))
        if candidates:
            from toyomacro.voigtfit.image_utils import load_image
            frames = load_image(candidates[0], all_frames=True)
            print(f"    Loaded {candidates[0].name} ({frames.shape[0]} frames)")
            return frames

    # Synthesize
    n_frames = max(10, min(100, 200_000_000 // (w * h)))  # ~200M total px cap
    frames = _make_synthetic_gif(w, h, n_frames)
    print(f"    Synthesized {w}x{h}x{n_frames} ({w*h*n_frames/1e6:.1f}M px)")
    return frames


def run_chunk_benchmark(
    frames: np.ndarray,
    chunk_spectra: int,
    elements: str = 'demo',
    noise_levels: list[str] | None = None,
) -> ChunkBenchmarkResult:
    """Run a single chunk-size benchmark measurement.

    This is a stripped-down version of run_gif_roundtrip() that measures
    only throughput without saving GIF output.
    """
    from toyomacro.voigtfit.pipeline import HybridPipeline
    from toyomacro.voigtfit.spectra_generator import (
        NOISE_LEVELS,
        GeneratorConfig,
        NoiseConfig,
        add_poisson_noise_mlx,
        add_poisson_noise_mlx_fused,
        decompose_image_to_amplitudes,
        get_element_preset,
        voigt_profile,
    )
    from toyomacro.voigtfit.weight_cache import WeightMatrixCache
    try:
        import mlx.core as mx
        _HAS_MLX = True
    except ImportError:
        _HAS_MLX = False

    if noise_levels is None:
        noise_levels = ['None', 'Moderate']

    n_frames, h, w, _ = frames.shape
    n_pixels = h * w
    resolution = f"{w}x{h}"

    preset = get_element_preset(elements)
    cache = WeightMatrixCache()
    pipeline = HybridPipeline(cache, enable_stage2=False, use_mlx=True)

    color_mapping = preset.color_mapping
    component_order = preset.component_order
    elements_list = preset.elements
    config = GeneratorConfig()

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
            'elem_specs': elem_specs,
            'elem_indices': elem_indices,
            'energy': energy,
            'profiles': profiles,
            'peak_config': peak_config,
        }

    # Noise configs
    noise_configs = {}
    for nl in noise_levels:
        if nl in NOISE_LEVELS:
            nv = NOISE_LEVELS[nl]
        else:
            try:
                nv = float(nl)
            except ValueError:
                nv = 0
        if nv == 0:
            noise_configs[nl] = NoiseConfig(noise_type='none')
        else:
            noise_configs[nl] = NoiseConfig(noise_type='poisson', poisson_level=nv)

    _nf_key = '_nf'
    _has_nf = any(nc.noise_type == 'none' for nc in noise_configs.values())
    if not _has_nf:
        noise_configs[_nf_key] = NoiseConfig(noise_type='none')
    all_nl = list(noise_levels) + ([] if _has_nf else [_nf_key])

    # ================================================================
    # Phase 1: Decompose
    # ================================================================
    t0 = time.perf_counter()
    all_amps = []
    for fi in range(n_frames):
        amp = decompose_image_to_amplitudes(
            frames[fi], color_mapping, method='pinv'
        )
        all_amps.append(amp)
    t_decompose = time.perf_counter() - t0

    # ================================================================
    # Phase 2+3: Chunk gen+fit (no PSNR/GIF save for speed measurement)
    # ================================================================
    chunk_size = max(1, chunk_spectra // n_pixels)
    chunk_size = min(chunk_size, n_frames)
    n_chunks = (n_frames + chunk_size - 1) // chunk_size

    total_spectra = 0
    t_genfit_start = time.perf_counter()

    for ci in range(n_chunks):
        f_start = ci * chunk_size
        f_end = min(f_start + chunk_size, n_frames)
        chunk_frames = f_end - f_start
        chunk_batch = chunk_frames * n_pixels

        chunk_amp = np.concatenate(all_amps[f_start:f_end], axis=1)

        for file_key, fc in file_configs.items():
            elem_indices = fc['elem_indices']
            profiles = fc['profiles']
            energy = fc['energy']
            peak_config = fc['peak_config']
            elem_specs = fc['elem_specs']

            n_profiles = len(profiles)
            A_np = np.zeros((chunk_batch, n_profiles), dtype=np.float32)
            for j, elem_idx in enumerate(elem_indices):
                A_np[:, j] = np.maximum(chunk_amp[elem_idx, :], 0)

            if _HAS_MLX:
                P_mx = mx.array(np.stack(profiles, axis=0))
                A_mx = mx.array(A_np * config.amplitude_scale)
                bg_mx = mx.array(
                    (A_np.sum(axis=1, keepdims=True)
                     * (config.amplitude_scale * config.bg_base)).astype(np.float32)
                )
                del A_np
                all_clean_mx = A_mx @ P_mx + bg_mx
                del A_mx, bg_mx, P_mx

                for nl in all_nl:
                    nc = noise_configs[nl]
                    if nc.noise_type == 'none':
                        spectra_mx = all_clean_mx
                    elif nc.noise_type == 'poisson':
                        spectra_mx = add_poisson_noise_mlx_fused(
                            all_clean_mx, nc.poisson_level
                        )
                    else:
                        spectra_mx = all_clean_mx

                    mx.eval(spectra_mx)
                    result = pipeline.process_rowmajor(
                        spectra_mx,
                        elem_specs[0].symbol,
                        elem_specs[0].orbital,
                        energy,
                        peak_config,
                        amplitudes_only=True,
                    )
                    total_spectra += spectra_mx.shape[0]

                    if spectra_mx is not all_clean_mx:
                        del spectra_mx

                del all_clean_mx
            else:
                P = np.stack(profiles, axis=0)
                all_clean = (config.amplitude_scale * A_np) @ P
                bg = A_np.sum(axis=1) * (config.amplitude_scale * config.bg_base)
                all_clean += bg[:, np.newaxis]
                del A_np, P, bg

                for nl in all_nl:
                    nc = noise_configs[nl]
                    if nc.noise_type == 'none':
                        spectra = all_clean
                    elif nc.noise_type == 'poisson':
                        spectra = add_poisson_noise_mlx(
                            all_clean.copy(), nc.poisson_level, force_gaussian=True
                        )
                    else:
                        spectra = all_clean

                    result = pipeline.process_rowmajor(
                        spectra,
                        elem_specs[0].symbol,
                        elem_specs[0].orbital,
                        energy,
                        peak_config,
                        amplitudes_only=True,
                    )
                    total_spectra += spectra.shape[0]

                    if spectra is not all_clean:
                        del spectra

                del all_clean

            gc.collect()

        del chunk_amp

    t_genfit = time.perf_counter() - t_genfit_start
    t_total = time.perf_counter() - t0
    peak_rss = _get_peak_rss_mb()

    return ChunkBenchmarkResult(
        resolution=resolution,
        width=w,
        height=h,
        n_pixels=n_pixels,
        n_frames=n_frames,
        chunk_spectra=chunk_spectra,
        chunk_size_frames=chunk_size,
        n_chunks=n_chunks,
        total_spectra=total_spectra,
        wall_time=t_total,
        throughput=total_spectra / t_total,
        peak_rss_mb=peak_rss,
        decompose_time=t_decompose,
        gen_fit_time=t_genfit,
        gen_fit_throughput=total_spectra / t_genfit if t_genfit > 0 else 0,
    )


def run_sweep(
    resolutions: list[str] | None = None,
    chunk_sizes: list[int] | None = None,
    gif_dir: str | None = None,
    noise_levels: list[str] | None = None,
    quick: bool = False,
) -> list[ChunkBenchmarkResult]:
    """Run a full resolution × chunk-size sweep.

    Args:
        resolutions: List of "WxH" strings. Default covers 320x180 to 1920x1080.
        chunk_sizes: List of _MAX_CHUNK_SPECTRA values to test.
            Default: geometric series from 100K to 20M.
        gif_dir: Directory containing pre-existing GIF files.
        noise_levels: Noise levels for roundtrip.
        quick: If True, use fewer resolutions and chunk sizes.
    """
    if resolutions is None:
        if quick:
            resolutions = ['320x180', '640x360', '1280x720']
        else:
            resolutions = [
                '320x180', '480x270', '640x360',
                '960x540', '1280x720', '1920x1080',
            ]

    if chunk_sizes is None:
        if quick:
            chunk_sizes = [500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000]
        else:
            chunk_sizes = [
                100_000, 250_000, 500_000,
                1_000_000, 2_000_000, 4_000_000,
                8_000_000, 16_000_000,
            ]

    if noise_levels is None:
        noise_levels = ['None', 'Moderate']

    gif_path = Path(gif_dir) if gif_dir else None

    results = []
    print("=" * 80)
    print("Chunk Size Optimization Benchmark")
    print(f"  Resolutions: {resolutions}")
    print(f"  Chunk sizes: {[f'{cs/1e6:.1f}M' for cs in chunk_sizes]}")
    print(f"  Noise levels: {noise_levels}")
    print()

    # Cache hierarchy reference (M3 Max)
    print("  M3 Max Cache Hierarchy:")
    print("    L1d = 64 KB/core (128B line)")
    print("    L2  = 16 MB (P-core cluster, 6 cores shared)")
    print("    No dedicated L3 (system-level cache in SoC)")
    print()

    # Precompute cache-relevant sizes
    for cs in chunk_sizes:
        # float32 spectra: cs × 240ch × 4B = cs × 960B ≈ cs KB
        mem_mb = cs * 960 / (1024 * 1024)
        fits_l2 = "L2" if mem_mb < 16 else "RAM"
        print(f"    chunk={cs/1e6:.1f}M → spectra ~{mem_mb:.0f} MB → {fits_l2}")
    print()

    for res in resolutions:
        print(f"\n{'─'*60}")
        print(f"Resolution: {res}")
        frames = _load_or_synthesize(res, gif_path)

        # JIT warmup: run smallest chunk once to warm VoigtFit/MLX
        warmup_cs = min(chunk_sizes)
        print(f"  Warmup (chunk={warmup_cs/1e6:.1f}M)...", end='', flush=True)
        # Use just first 2 frames for warmup
        warmup_frames = frames[:min(2, len(frames))]
        _ = run_chunk_benchmark(
            warmup_frames, warmup_cs,
            noise_levels=['None'],
        )
        gc.collect()
        print(" done")

        for cs in chunk_sizes:
            w, h = map(int, res.split('x'))
            chunk_frames_est = max(1, cs // (w * h))
            n_chunks_est = (frames.shape[0] + chunk_frames_est - 1) // chunk_frames_est
            total_spec_est = frames.shape[0] * w * h * len(noise_levels) * 5  # ~5 file_keys

            print(f"  chunk={cs/1e6:.1f}M "
                  f"({chunk_frames_est}f × {n_chunks_est}c)... ",
                  end='', flush=True)

            gc.collect()
            r = run_chunk_benchmark(
                frames, cs,
                noise_levels=noise_levels,
            )
            results.append(r)

            print(f"{r.gen_fit_throughput/1e6:.2f}M spec/s "
                  f"(total {r.throughput/1e6:.2f}M/s, "
                  f"{r.wall_time:.1f}s, "
                  f"RSS {r.peak_rss_mb:.0f}MB)")

            gc.collect()

        del frames
        gc.collect()

    return results


def print_summary_table(results: list[ChunkBenchmarkResult]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: Gen+Fit Throughput (M spec/s)")
    print("=" * 100)

    # Group by resolution
    resolutions = sorted(set(r.resolution for r in results),
                         key=lambda s: int(s.split('x')[0]))
    chunk_sizes = sorted(set(r.chunk_spectra for r in results))

    # Header
    hdr = f"{'Chunk':>10s}"
    for res in resolutions:
        hdr += f" {res:>12s}"
    print(hdr)
    print("-" * (10 + 13 * len(resolutions)))

    # Find best per resolution
    best = {}
    for res in resolutions:
        res_results = [r for r in results if r.resolution == res]
        if res_results:
            best_r = max(res_results, key=lambda r: r.gen_fit_throughput)
            best[res] = best_r.chunk_spectra

    for cs in chunk_sizes:
        row = f"{cs/1e6:>8.1f}M"
        for res in resolutions:
            match = [r for r in results
                     if r.resolution == res and r.chunk_spectra == cs]
            if match:
                tp = match[0].gen_fit_throughput / 1e6
                marker = " *" if best.get(res) == cs else "  "
                row += f" {tp:>10.2f}{marker}"
            else:
                row += f" {'—':>12s}"
        print(row)

    print("\n* = best chunk size for that resolution")

    # Memory table
    print(f"\n{'SUMMARY TABLE: Peak RSS (MB)':>60s}")
    print("-" * (10 + 13 * len(resolutions)))
    hdr = f"{'Chunk':>10s}"
    for res in resolutions:
        hdr += f" {res:>12s}"
    print(hdr)
    print("-" * (10 + 13 * len(resolutions)))
    for cs in chunk_sizes:
        row = f"{cs/1e6:>8.1f}M"
        for res in resolutions:
            match = [r for r in results
                     if r.resolution == res and r.chunk_spectra == cs]
            if match:
                row += f" {match[0].peak_rss_mb:>10.0f}  "
            else:
                row += f" {'—':>12s}"
        print(row)


def plot_results(results: list[ChunkBenchmarkResult],
                 output_path: str | None = None) -> None:
    """Generate a 4-panel analysis plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    resolutions = sorted(set(r.resolution for r in results),
                         key=lambda s: int(s.split('x')[0]))
    chunk_sizes = sorted(set(r.chunk_spectra for r in results))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Color map for resolutions
    cmap = plt.cm.viridis
    colors = {res: cmap(i / max(len(resolutions)-1, 1))
              for i, res in enumerate(resolutions)}

    # Panel 1: Throughput vs chunk size (per resolution)
    ax = axes[0, 0]
    for res in resolutions:
        res_r = sorted([r for r in results if r.resolution == res],
                       key=lambda r: r.chunk_spectra)
        if res_r:
            xs = [r.chunk_spectra / 1e6 for r in res_r]
            ys = [r.gen_fit_throughput / 1e6 for r in res_r]
            ax.plot(xs, ys, 'o-', color=colors[res], label=res, markersize=5)
    ax.set_xscale('log')
    ax.set_xlabel('Chunk size (M spectra)')
    ax.set_ylabel('Gen+Fit throughput (M spec/s)')
    ax.set_title('Throughput vs Chunk Size')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    # Cache size reference lines
    ax.axvline(x=16.0, color='red', alpha=0.3, linestyle='--', label='L2=16MB boundary')

    # Panel 2: Throughput vs resolution (per chunk size)
    ax = axes[0, 1]
    for cs in chunk_sizes:
        cs_r = sorted([r for r in results if r.chunk_spectra == cs],
                      key=lambda r: r.n_pixels)
        if cs_r:
            xs = [r.n_pixels / 1e3 for r in cs_r]
            ys = [r.gen_fit_throughput / 1e6 for r in cs_r]
            ax.plot(xs, ys, 'o-', label=f'{cs/1e6:.1f}M', markersize=4)
    ax.set_xscale('log')
    ax.set_xlabel('Pixels per frame (K)')
    ax.set_ylabel('Gen+Fit throughput (M spec/s)')
    ax.set_title('Throughput vs Resolution')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: Memory (RSS) vs chunk size
    ax = axes[1, 0]
    for res in resolutions:
        res_r = sorted([r for r in results if r.resolution == res],
                       key=lambda r: r.chunk_spectra)
        if res_r:
            xs = [r.chunk_spectra / 1e6 for r in res_r]
            ys = [r.peak_rss_mb / 1024 for r in res_r]  # GB
            ax.plot(xs, ys, 'o-', color=colors[res], label=res, markersize=5)
    ax.set_xscale('log')
    ax.set_xlabel('Chunk size (M spectra)')
    ax.set_ylabel('Peak RSS (GB)')
    ax.set_title('Memory vs Chunk Size')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 4: Overhead analysis — (total time - gen_fit time) / total time
    ax = axes[1, 1]
    for res in resolutions:
        res_r = sorted([r for r in results if r.resolution == res],
                       key=lambda r: r.chunk_spectra)
        if res_r:
            xs = [r.chunk_spectra / 1e6 for r in res_r]
            ys = [(r.wall_time - r.gen_fit_time) / r.wall_time * 100
                  for r in res_r]
            ax.plot(xs, ys, 'o-', color=colors[res], label=res, markersize=5)
    ax.set_xscale('log')
    ax.set_xlabel('Chunk size (M spectra)')
    ax.set_ylabel('Overhead fraction (%)')
    ax.set_title('Overhead (decompose + PSNR + GC) fraction')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle('GIF Roundtrip: Chunk Size Optimization (M3 Max)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / 'outputs' / 'chunk_optimization.png'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Chunk Size Optimization for GIF Roundtrip Pipeline'
    )
    parser.add_argument('--resolutions', '-r', nargs='*', type=str,
                       help='Resolution list (e.g. 320x180 640x360)')
    parser.add_argument('--chunk-sizes', '-c', nargs='*', type=float,
                       help='Chunk sizes in millions (e.g. 0.5 1 2 4 8)')
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Directory with pre-existing media files')
    parser.add_argument('--gif-dir', type=str, default=None,
                       help='(deprecated alias for --input-dir) Directory with GIF files')
    parser.add_argument('--noise', nargs='*', type=str, default=None,
                       help='Noise levels (default: None Moderate)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer resolutions and chunk sizes')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output plot path')

    args = parser.parse_args()

    chunk_sizes = None
    if args.chunk_sizes:
        chunk_sizes = [int(cs * 1_000_000) for cs in args.chunk_sizes]

    media_dir = args.input_dir or args.gif_dir
    results = run_sweep(
        resolutions=args.resolutions,
        chunk_sizes=chunk_sizes,
        gif_dir=media_dir,
        noise_levels=args.noise,
        quick=args.quick,
    )

    print_summary_table(results)
    plot_results(results, output_path=args.output)
