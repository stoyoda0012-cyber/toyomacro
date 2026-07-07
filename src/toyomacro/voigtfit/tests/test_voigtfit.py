#!/usr/bin/env python3
"""
Test script for VoigtFit Stage 1 Pipeline
==========================================

Tests WeightMatrixCache, HybridPipeline (Stage 1), and pipeline throughput.

Run with:
    pytest src/toyomacro/voigtfit/tests/test_voigtfit.py -v
"""

import sys

import numpy as np

from toyomacro.voigtfit import FitResult, HybridPipeline, WeightMatrixCache


def test_weight_cache():
    """Test weight matrix cache."""
    print("Testing WeightMatrixCache...")

    cache = WeightMatrixCache(cache_dir="/tmp/voigtfit_test_cache")
    cache.clear_all()

    # Parameters
    energy = np.linspace(700, 730, 100).astype(np.float32)
    centers = np.array([707.0, 714.0, 721.0], dtype=np.float32)
    sigmas = np.array([0.8, 0.9, 0.85], dtype=np.float32)
    gamma = 0.2

    # Get or create
    Wt, Phi = cache.get_or_create(
        element="Fe",
        orbital="2p3/2",
        energy=energy,
        centers=centers,
        sigmas=sigmas,
        gamma=gamma,
        use_mlx=False,  # Test with numpy first
    )

    assert Wt.shape == (3, 100), f"Wt shape mismatch: {Wt.shape}"
    assert Phi.shape == (100, 3), f"Phi shape mismatch: {Phi.shape}"

    # Test cache hit
    Wt2, Phi2 = cache.get_or_create(
        element="Fe",
        orbital="2p3/2",
        energy=energy,
        centers=centers,
        sigmas=sigmas,
        gamma=gamma,
        use_mlx=False,
    )

    assert np.allclose(Wt, Wt2), "Cache hit should return same values"

    info = cache.info()
    assert info["memory_entries"] >= 1
    assert info["disk_entries"] >= 1

    print("  WeightMatrixCache tests passed")


def test_hybrid_pipeline():
    """Test hybrid pipeline (Stage 1 only)."""
    print("Testing HybridPipeline...")

    from scipy import special as sps

    # Generate synthetic data
    n_spectra = 1000
    n_energy = 100

    energy = np.linspace(700, 730, n_energy).astype(np.float32)
    centers = np.array([707.0, 714.0, 721.0], dtype=np.float32)
    sigmas = np.array([0.8, 0.9, 0.85], dtype=np.float32)
    gamma = 0.2

    SQRT2 = np.sqrt(2.0)
    SQRT2PI = np.sqrt(2.0 * np.pi)

    # Build basis
    Phi = np.zeros((n_energy, 3), dtype=np.float32)
    for k in range(3):
        z = ((energy - centers[k]) + 1j * gamma) / (sigmas[k] * SQRT2)
        Phi[:, k] = np.real(sps.wofz(z)).astype(np.float32) / (sigmas[k] * SQRT2PI)

    # Random amplitudes
    np.random.seed(42)
    A_true = np.random.uniform(0.5, 2.0, (3, n_spectra)).astype(np.float32)
    Y = Phi @ A_true

    # Add noise
    Y += np.random.normal(0, 0.01 * Y.max(), Y.shape).astype(np.float32)

    # Create anomalies (10 spectra with shifted peaks)
    for i in range(10):
        centers_shifted = centers + np.random.uniform(-1, 1, 3).astype(np.float32)
        Phi_shifted = np.zeros((n_energy, 3), dtype=np.float32)
        for k in range(3):
            z = ((energy - centers_shifted[k]) + 1j * gamma) / (sigmas[k] * SQRT2)
            Phi_shifted[:, k] = np.real(sps.wofz(z)).astype(np.float32) / (sigmas[k] * SQRT2PI)
        Y[:, i] = Phi_shifted @ A_true[:, i]

    # Run pipeline
    cache = WeightMatrixCache(cache_dir="/tmp/voigtfit_test_cache")
    pipeline = HybridPipeline(cache, enable_stage2=False)

    result = pipeline.process(
        Y=Y,
        element="Fe",
        orbital="2p3/2",
        energy=energy,
        peak_config={
            "centers": centers,
            "sigmas": sigmas,
            "gamma": gamma,
        },
    )

    assert isinstance(result, FitResult), "Should return FitResult"
    assert result.amplitudes.shape == (3, n_spectra), f"Amplitudes shape mismatch: {result.amplitudes.shape}"
    assert result.chi2.shape == (n_spectra,), f"chi2 shape mismatch: {result.chi2.shape}"
    assert result.anomaly_mask.shape == (n_spectra,), "Anomaly mask shape mismatch"

    # Check anomaly detection found some anomalies
    n_detected = np.sum(result.anomaly_mask)
    assert n_detected >= 5, f"Should detect some anomalies, got {n_detected}"

    # Check timing
    assert "stage1" in result.timing
    assert result.timing["stage1"] > 0

    print(f"  Detected {n_detected} anomalies")
    print(f"  Stage 1 time: {result.timing['stage1']*1000:.2f} ms")
    print(f"  Stage 1 rate: {result.timing['stage1_rate']:,.0f} spec/s")
    print("  HybridPipeline tests passed")


def test_performance():
    """Run Stage 1 pipeline throughput benchmark."""
    print("\nPerformance Benchmark...")

    from toyomacro.voigtfit.benchmarks.benchmark_stage1_pipeline import run_benchmark

    results = run_benchmark(
        n_spectra_list=[1000, 10000, 100000],
        anomaly_ratio=0.01,
        enable_stage2=False,
        warmup=True,
    )

    print("\n  Performance benchmark completed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("VoigtFit Test Suite")
    print("=" * 60)

    tests = [
        ("WeightMatrixCache", test_weight_cache),
        ("HybridPipeline", test_hybrid_pipeline),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  {name} failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    # Run performance benchmark if all tests pass
    if failed == 0:
        try:
            test_performance()
        except Exception as e:
            print(f"Performance benchmark failed: {e}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
