"""
Benchmark for Stage 1 Pipeline
===============================

Tests Stage 1 (weight matrix amp-only) throughput at various scales:
    - 1K spectra: quick validation
    - 1M spectra: standard workload
    - 33M spectra: 8K image simulation

Expected performance on M3 Max:
    Stage 1: 400-500M spectra/s (memory bandwidth limited)
"""

import time

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from ..pipeline import HybridPipeline
from ..weight_cache import WeightMatrixCache


def generate_synthetic_spectra(
    n_spectra: int,
    n_energy: int = 100,
    n_components: int = 3,
    noise_level: float = 0.01,
    anomaly_ratio: float = 0.01,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic Voigt spectra for benchmarking.

    Args:
        n_spectra: Number of spectra to generate
        n_energy: Number of energy points
        n_components: Number of Voigt peaks
        noise_level: Gaussian noise level (fraction of max)
        anomaly_ratio: Fraction of anomalous spectra
        seed: Random seed

    Returns:
        (Y, energy, peak_config): Spectra, energy axis, and peak configuration
    """
    np.random.seed(seed)

    # Energy axis
    energy = np.linspace(700, 730, n_energy).astype(np.float32)

    # Peak configuration
    centers = np.array([707.0, 714.0, 721.0])[:n_components].astype(np.float32)
    sigmas = np.array([0.8, 0.9, 0.85])[:n_components].astype(np.float32)
    gamma = 0.2

    peak_config = {
        "centers": centers,
        "sigmas": sigmas,
        "gamma": gamma,
    }

    # Generate basis
    from scipy import special as sps
    SQRT2 = np.sqrt(2.0)
    SQRT2PI = np.sqrt(2.0 * np.pi)

    Phi = np.zeros((n_energy, n_components), dtype=np.float32)
    for k in range(n_components):
        z = ((energy - centers[k]) + 1j * gamma) / (sigmas[k] * SQRT2)
        Phi[:, k] = np.real(sps.wofz(z)).astype(np.float32) / (sigmas[k] * SQRT2PI)

    # Random amplitudes
    A = np.random.uniform(0.5, 2.0, (n_components, n_spectra)).astype(np.float32)

    # Generate spectra
    Y = Phi @ A

    # Add noise
    Y += np.random.normal(0, noise_level * Y.max(), Y.shape).astype(np.float32)

    # Create anomalies (shifted peaks)
    n_anomaly = int(n_spectra * anomaly_ratio)
    if n_anomaly > 0:
        anomaly_indices = np.random.choice(n_spectra, n_anomaly, replace=False)
        for idx in anomaly_indices:
            # Shift centers by random amount
            shift = np.random.uniform(-1.5, 1.5, n_components)
            centers_shifted = centers + shift

            Phi_shifted = np.zeros((n_energy, n_components), dtype=np.float32)
            for k in range(n_components):
                z = ((energy - centers_shifted[k]) + 1j * gamma) / (sigmas[k] * SQRT2)
                Phi_shifted[:, k] = np.real(sps.wofz(z)).astype(np.float32) / (sigmas[k] * SQRT2PI)

            Y[:, idx] = Phi_shifted @ A[:, idx]

    return Y, energy, peak_config


def run_benchmark(
    n_spectra_list: list = [1000, 10000, 100000, 1000000],
    n_energy: int = 100,
    anomaly_ratio: float = 0.01,
    enable_stage2: bool = False,
    warmup: bool = True,
) -> dict:
    """
    Run benchmark at various scales.

    Args:
        n_spectra_list: List of spectrum counts to test
        n_energy: Energy points per spectrum
        anomaly_ratio: Fraction of anomalous spectra
        enable_stage2: Enable Stage 2 refinement
        warmup: Run warmup iteration first

    Returns:
        Dict with benchmark results
    """
    results = {
        "mlx_available": HAS_MLX,
        "n_energy": n_energy,
        "anomaly_ratio": anomaly_ratio,
        "runs": [],
    }

    cache = WeightMatrixCache()
    pipeline = HybridPipeline(cache, enable_stage2=enable_stage2)

    for n_spectra in n_spectra_list:
        print(f"\nBenchmarking {n_spectra:,} spectra...")

        # Generate data
        Y, energy, peak_config = generate_synthetic_spectra(
            n_spectra=n_spectra,
            n_energy=n_energy,
            anomaly_ratio=anomaly_ratio,
        )

        # Warmup
        if warmup:
            _ = pipeline.process(
                Y=Y[:, :min(1000, n_spectra)],
                element="Fe",
                orbital="2p3/2",
                energy=energy,
                peak_config=peak_config,
            )

        # Benchmark
        t0 = time.perf_counter()
        result = pipeline.process(
            Y=Y,
            element="Fe",
            orbital="2p3/2",
            energy=energy,
            peak_config=peak_config,
        )
        total_time = time.perf_counter() - t0

        run_result = {
            "n_spectra": n_spectra,
            "total_time": total_time,
            "throughput": n_spectra / total_time,
            "timing": result.timing,
            "n_anomaly": np.sum(result.anomaly_mask),
        }
        results["runs"].append(run_result)

        print(f"  Total: {total_time:.4f}s")
        print(f"  Throughput: {run_result['throughput']:,.0f} spec/s")
        print(f"  Stage 1: {result.timing.get('stage1', 0):.4f}s ({result.timing.get('stage1_rate', 0):,.0f} spec/s)")
        if result.timing.get("stage2"):
            print(f"  Stage 2: {result.timing['stage2']:.4f}s ({result.timing.get('stage2_rate', 0):,.0f} spec/s)")
        print(f"  Anomalies: {run_result['n_anomaly']:,} ({100*run_result['n_anomaly']/n_spectra:.2f}%)")

    return results


def benchmark_8k_simulation():
    """
    Simulate 8K image processing (33M spectra, Stage 1 only).

    Expected:
        Stage 1: 33M @ 400M/s = ~0.08s
    """
    print("=" * 60)
    print("8K Image Simulation (33M spectra)")
    print("=" * 60)

    # Use subset for actual benchmark (full 33M requires ~13GB memory)
    n_spectra = 10_000_000  # 10M spectra (feasible on 64GB Mac)

    print(f"\nRunning with {n_spectra:,} spectra (memory-limited subset)")
    print("Extrapolated 8K (33M) time shown in parentheses")
    print("-" * 60)

    results = run_benchmark(
        n_spectra_list=[n_spectra],
        anomaly_ratio=0.001,  # 0.1% anomaly
        enable_stage2=True,
    )

    # Extrapolate to 33M
    run = results["runs"][0]
    scale = 33_000_000 / n_spectra

    print("\n" + "=" * 60)
    print("Extrapolated 8K (33M) Performance:")
    print(f"  Stage 1: {run['timing']['stage1'] * scale:.3f}s")
    if run['timing'].get('stage2'):
        print(f"  Stage 2: {run['timing']['stage2'] * scale:.3f}s")
    print(f"  Total: {run['total_time'] * scale:.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    print("VoigtFit Stage 1 Pipeline Benchmark")
    print("=" * 60)
    print(f"MLX available: {HAS_MLX}")
    print()

    # Standard benchmark
    results = run_benchmark(
        n_spectra_list=[1000, 10000, 100000, 1000000],
        anomaly_ratio=0.01,
    )

    # 8K simulation (if enough memory)
    print("\n")
    benchmark_8k_simulation()
