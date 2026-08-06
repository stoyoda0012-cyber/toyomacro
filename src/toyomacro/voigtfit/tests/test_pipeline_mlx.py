"""
MLX integration tests for the legacy compatibility fallback (Stage 2).

These cover the Gauss-Newton path that `HybridPipeline` runs on
anomalous spectra. It is retained for backward compatibility and may
change in a future release; these tests would go with it.

Verifies:
1. MLX Stage 2 is automatically used when available
2. CPU fallback works when MLX disabled
3. Performance comparison between backends
4. Accuracy validation
"""

import time

import numpy as np


def generate_test_data(n_spectra: int, n_energy: int = 151):
    """Generate test spectra with known parameters."""
    from toyomacro.voigtfit.faddeeva_mlx import voigt_profile_numpy

    energy = np.linspace(95, 110, n_energy).astype(np.float32)

    # True parameters
    true_centers = np.array([99.5, 103.0], dtype=np.float32)
    true_sigmas = np.array([0.6, 0.8], dtype=np.float32)
    true_gamma = 0.25
    true_amps = np.array([1000.0, 600.0], dtype=np.float32)

    # Generate spectra with random center shifts (simulating charging)
    center_shifts = np.random.uniform(-0.3, 0.3, n_spectra).astype(np.float32)

    Y = np.zeros((n_energy, n_spectra), dtype=np.float32)
    for i in range(n_spectra):
        for c, s, a in zip(true_centers + center_shifts[i], true_sigmas, true_amps):
            Y[:, i] += a * voigt_profile_numpy(energy, c, s, true_gamma)

    # Add noise
    noise_level = Y.max() / 100
    Y += noise_level * np.random.randn(n_energy, n_spectra).astype(np.float32)

    peak_config = {
        "centers": true_centers,
        "sigmas": true_sigmas,
        "gamma": true_gamma,
    }

    return Y, energy, peak_config, center_shifts


def test_pipeline_mlx_integration():
    """Test HybridPipeline with MLX Stage 2."""
    from toyomacro.voigtfit import HAS_MLX
    from toyomacro.voigtfit.pipeline import HybridPipeline
    from toyomacro.voigtfit.weight_cache import WeightMatrixCache

    print("=" * 70)
    print("HybridPipeline + MLX Stage 2 Integration Test")
    print("=" * 70)
    print(f"MLX available: {HAS_MLX}")

    np.random.seed(42)

    # Generate test data
    n_spectra = 10000
    Y, energy, peak_config, true_shifts = generate_test_data(n_spectra)

    # Artificially inflate anomaly rate by making all spectra anomalous
    # (for testing Stage 2 performance)

    cache = WeightMatrixCache()

    # Test 1: MLX Stage 2 (if available)
    if HAS_MLX:
        print("\n--- Test 1: MLX Stage 2 Backend ---")
        pipeline_mlx = HybridPipeline(
            cache=cache,
            chi2_threshold=0.1,  # Low threshold to force all to Stage 2
            use_mlx=True,
            enable_stage2=True,
            stage2_mode="light",
            use_mlx_stage2=True,
        )

        t0 = time.perf_counter()
        result_mlx = pipeline_mlx.process(
            Y=Y,
            element="Si",
            orbital="2p",
            energy=energy,
            peak_config=peak_config,
        )
        elapsed_mlx = time.perf_counter() - t0

        n_anomaly = result_mlx.timing.get("n_anomaly", 0)
        stage2_rate = result_mlx.timing.get("stage2_rate", 0)

        print(f"Total spectra: {n_spectra}")
        print(f"Anomalies (Stage 2): {n_anomaly} ({n_anomaly/n_spectra*100:.1f}%)")
        print(f"Stage 2 backend: {result_mlx.timing.get('stage2_backend', 'N/A')}")
        print(f"Stage 2 rate: {stage2_rate:,.0f} spec/s")
        print(f"Total time: {elapsed_mlx*1000:.1f}ms")

        if n_anomaly > 0:
            # Validate accuracy
            anomaly_idx = np.where(result_mlx.anomaly_mask)[0]
            detected_shifts = result_mlx.centers[0, anomaly_idx] - peak_config["centers"][0]
            true_shifts_anomaly = true_shifts[anomaly_idx]
            shift_error = detected_shifts - true_shifts_anomaly
            rmse = np.sqrt(np.mean(shift_error**2))
            print(f"Center shift RMSE: {rmse:.4f} eV")

    # Test 2: CPU Stage 2 (forced)
    print("\n--- Test 2: CPU Stage 2 Backend ---")
    pipeline_cpu = HybridPipeline(
        cache=cache,
        chi2_threshold=0.1,
        use_mlx=True,  # Stage 1 still uses MLX
        enable_stage2=True,
        stage2_mode="light",
        use_mlx_stage2=False,  # Force CPU backend
    )

    # Use smaller dataset for CPU test
    n_cpu_test = 1000
    Y_cpu = Y[:, :n_cpu_test]
    true_shifts_cpu = true_shifts[:n_cpu_test]

    t0 = time.perf_counter()
    result_cpu = pipeline_cpu.process(
        Y=Y_cpu,
        element="Si",
        orbital="2p",
        energy=energy,
        peak_config=peak_config,
    )
    elapsed_cpu = time.perf_counter() - t0

    n_anomaly_cpu = result_cpu.timing.get("n_anomaly", 0)
    stage2_rate_cpu = result_cpu.timing.get("stage2_rate", 0)

    print(f"Total spectra: {n_cpu_test}")
    print(f"Anomalies (Stage 2): {n_anomaly_cpu} ({n_anomaly_cpu/n_cpu_test*100:.1f}%)")
    print(f"Stage 2 backend: {result_cpu.timing.get('stage2_backend', 'N/A')}")
    print(f"Stage 2 rate: {stage2_rate_cpu:,.0f} spec/s")
    print(f"Total time: {elapsed_cpu*1000:.1f}ms")

    if n_anomaly_cpu > 0:
        anomaly_idx_cpu = np.where(result_cpu.anomaly_mask)[0]
        detected_shifts_cpu = result_cpu.centers[0, anomaly_idx_cpu] - peak_config["centers"][0]
        true_shifts_anomaly_cpu = true_shifts_cpu[anomaly_idx_cpu]
        shift_error_cpu = detected_shifts_cpu - true_shifts_anomaly_cpu
        rmse_cpu = np.sqrt(np.mean(shift_error_cpu**2))
        print(f"Center shift RMSE: {rmse_cpu:.4f} eV")

    # Performance comparison
    if HAS_MLX and n_anomaly > 0 and n_anomaly_cpu > 0:
        print("\n--- Performance Comparison ---")
        speedup = stage2_rate / stage2_rate_cpu if stage2_rate_cpu > 0 else 0
        print(f"MLX Stage 2: {stage2_rate:,.0f} spec/s")
        print(f"CPU Stage 2: {stage2_rate_cpu:,.0f} spec/s")
        print(f"Speedup: {speedup:.1f}x")

    print("\n" + "=" * 70)
    print("Integration test completed!")
    print("=" * 70)


def benchmark_8k_image():
    """Benchmark full 8K image processing (simulated)."""
    from toyomacro.voigtfit import HAS_MLX
    from toyomacro.voigtfit.pipeline import HybridPipeline
    from toyomacro.voigtfit.weight_cache import WeightMatrixCache

    if not HAS_MLX:
        print("MLX not available, skipping 8K benchmark")
        return

    print("\n" + "=" * 70)
    print("8K Image Benchmark (Simulated)")
    print("=" * 70)

    np.random.seed(42)

    # 8K image: 7680 x 4320 = 33M pixels
    # For testing, we'll use a representative subset
    n_spectra_test = 100000  # 100K spectra
    anomaly_rate = 0.01  # 1%

    Y, energy, peak_config, true_shifts = generate_test_data(n_spectra_test)

    cache = WeightMatrixCache()
    pipeline = HybridPipeline(
        cache=cache,
        chi2_threshold=3.0,  # Realistic threshold
        use_mlx=True,
        enable_stage2=True,
        stage2_mode="light",
        use_mlx_stage2=True,
    )

    # Warm up
    _ = pipeline.process(Y[:, :100], "Si", "2p", energy, peak_config)

    # Benchmark
    t0 = time.perf_counter()
    result = pipeline.process(Y, "Si", "2p", energy, peak_config)
    elapsed = time.perf_counter() - t0

    timing = result.timing

    print(f"\nResults for {n_spectra_test:,} spectra:")
    print(f"  Stage 1: {timing['stage1']*1000:.1f}ms ({timing['stage1_rate']:,.0f} spec/s)")
    print(f"  Anomaly detection: {timing['anomaly_detection']*1000:.2f}ms")
    print(f"  Anomalies: {timing['n_anomaly']:,} ({timing['anomaly_ratio']*100:.2f}%)")

    if 'stage2' in timing:
        print(f"  Stage 2 ({timing.get('stage2_backend', 'N/A')}): {timing['stage2']*1000:.1f}ms ({timing['stage2_rate']:,.0f} spec/s)")
        print(f"  Stage 2 converged: {timing['stage2_converged']}/{timing['n_anomaly']}")

    print(f"  Total: {elapsed*1000:.1f}ms")

    # Extrapolate to full 8K
    scale = 33_000_000 / n_spectra_test
    print(f"\nExtrapolated to 8K image (33M spectra, {anomaly_rate*100:.0f}% anomaly):")
    print(f"  Stage 1: ~{timing['stage1']*scale*1000:.0f}ms")

    if 'stage2' in timing and timing['n_anomaly'] > 0:
        # Scale Stage 2 based on anomaly count
        expected_anomalies = 33_000_000 * anomaly_rate
        stage2_time = expected_anomalies / timing['stage2_rate']
        print(f"  Stage 2 (MLX): ~{stage2_time*1000:.0f}ms for {expected_anomalies:,.0f} anomalies")
        print(f"  Estimated total: ~{(timing['stage1']*scale + stage2_time)*1000:.0f}ms")


if __name__ == "__main__":
    test_pipeline_mlx_integration()
    benchmark_8k_image()
