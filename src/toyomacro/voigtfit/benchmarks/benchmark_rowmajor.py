"""
Benchmark: process_rowmajor vs process_rowmajor_fast
====================================================

Compares the original and optimized pipeline implementations.
"""

import sys
import time

import numpy as np

try:
    import mlx.core as mx

    from toyomacro.voigtfit._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND the default device can execute work
except ImportError:
    HAS_MLX = False
    print("MLX not available")
    sys.exit(1)

from ..pipeline import HybridPipeline
from ..weight_cache import WeightMatrixCache


def generate_test_data(n_spectra: int, n_energy: int = 151, seed: int = 42):
    """Generate synthetic row-major test data."""
    np.random.seed(seed)

    # Row-major data (C-contiguous) like HDF5
    Y = np.random.randn(n_spectra, n_energy).astype(np.float32)
    energy = np.linspace(95, 110, n_energy).astype(np.float32)

    peak_config = {
        "centers": np.array([99.5, 103.0, 107.0], dtype=np.float32),
        "sigmas": np.array([0.6, 0.8, 0.7], dtype=np.float32),
        "gamma": 0.25,
    }

    return Y, energy, peak_config


def benchmark_comparison(n_spectra_list=[100_000, 1_000_000, 5_000_000], n_iterations=3):
    """Compare original vs optimized pipeline."""
    print("=" * 70)
    print("Benchmark: process_rowmajor vs process_rowmajor_fast")
    print("=" * 70)

    cache = WeightMatrixCache()
    pipeline = HybridPipeline(cache, enable_stage2=False)

    for n_spectra in n_spectra_list:
        print(f"\n{'='*70}")
        print(f"Testing with {n_spectra:,} spectra")
        print(f"{'='*70}")

        Y, energy, peak_config = generate_test_data(n_spectra)

        # Warmup
        _ = pipeline.process_rowmajor(
            Y[:1000], "Fe", "2p", energy, peak_config
        )
        _ = pipeline.process_rowmajor_fast(
            Y[:1000], "Fe", "2p", energy, peak_config
        )

        # Benchmark original
        times_original = []
        for i in range(n_iterations):
            t0 = time.perf_counter()
            result_orig = pipeline.process_rowmajor(
                Y, "Fe", "2p", energy, peak_config
            )
            times_original.append(time.perf_counter() - t0)

        # Benchmark optimized
        times_fast = []
        for i in range(n_iterations):
            t0 = time.perf_counter()
            result_fast = pipeline.process_rowmajor_fast(
                Y, "Fe", "2p", energy, peak_config
            )
            times_fast.append(time.perf_counter() - t0)

        # Results
        mean_orig = np.mean(times_original)
        mean_fast = np.mean(times_fast)
        speedup = mean_orig / mean_fast

        print("\nOriginal (process_rowmajor):")
        print(f"  Time: {mean_orig*1000:.1f}ms ± {np.std(times_original)*1000:.1f}ms")
        print(f"  Rate: {n_spectra/mean_orig:,.0f} spec/s")
        if result_orig.timing:
            print(f"  Breakdown: {result_orig.timing}")

        print("\nOptimized (process_rowmajor_fast):")
        print(f"  Time: {mean_fast*1000:.1f}ms ± {np.std(times_fast)*1000:.1f}ms")
        print(f"  Rate: {n_spectra/mean_fast:,.0f} spec/s")
        if result_fast.timing:
            print(f"  Breakdown: {result_fast.timing}")

        print(f"\nSpeedup: {speedup:.2f}x")

        # Verify results match
        amp_diff = np.max(np.abs(result_orig.amplitudes - result_fast.amplitudes))
        chi2_diff = np.max(np.abs(result_orig.chi2 - result_fast.chi2))
        print(f"Max amplitude difference: {amp_diff:.2e}")
        print(f"Max chi2 difference: {chi2_diff:.2e}")


def benchmark_detailed_timing(n_spectra: int = 1_000_000):
    """Detailed timing breakdown for optimized pipeline."""
    print("=" * 70)
    print(f"Detailed Timing: {n_spectra:,} spectra")
    print("=" * 70)

    cache = WeightMatrixCache()
    pipeline = HybridPipeline(cache, enable_stage2=False)

    Y, energy, peak_config = generate_test_data(n_spectra)

    # Warmup
    _ = pipeline.process_rowmajor_fast(Y[:1000], "Fe", "2p", energy, peak_config)

    # Run with detailed timing
    result = pipeline.process_rowmajor_fast(Y, "Fe", "2p", energy, peak_config)

    print("\nTiming breakdown:")
    total = result.timing.get("total", 0)
    for key, value in sorted(result.timing.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0):
        if isinstance(value, float) and key != "total":
            pct = 100 * value / total if total > 0 else 0
            print(f"  {key:<20}: {value*1000:>8.2f}ms ({pct:>5.1f}%)")
        elif key == "stage1_rate":
            print(f"  {key:<20}: {value:>12,.0f} spec/s")
        elif key not in ["n_anomaly", "anomaly_ratio"]:
            print(f"  {key:<20}: {value}")

    print(f"\n  {'total':<20}: {total*1000:>8.2f}ms")
    print(f"  {'throughput':<20}: {n_spectra/total:>12,.0f} spec/s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-spectra", type=int, default=1_000_000)
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    if args.detailed:
        benchmark_detailed_timing(args.n_spectra)
    else:
        benchmark_comparison([100_000, args.n_spectra])
