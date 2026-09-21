"""
Integration Tests for VoigtFit + DepthProfiler
==============================================

Tests:
1. Synthetic data accuracy
2. MATLAB bridge compatibility
3. Performance benchmarks
"""

import os
import tempfile
import time

import numpy as np
import pytest


class TestSyntheticAccuracy:
    """Test accuracy on synthetic data with known ground truth."""

    def test_single_peak_noiseless(self):
        """Test single peak fitting without noise."""
        from toyomacro.voigtfit.integration import DepthProfilerBridge, PeakConfig
        from toyomacro.voigtfit.voigt_jacobian import voigt_profile

        bridge = DepthProfilerBridge(enable_stage2=False)

        # Ground truth
        energy = np.linspace(280, 292, 100).astype(np.float32)
        true_center = 285.0
        true_sigma = 0.8
        true_gamma = 0.3
        true_amp = 1000.0

        # Generate clean spectrum
        Y = true_amp * voigt_profile(energy, true_center, true_sigma, true_gamma)
        Y = Y[:, np.newaxis].astype(np.float32)

        peak_config = PeakConfig(
            element="C", orbital="1s",
            centers=np.array([true_center]),
            sigmas=np.array([true_sigma]),
            gamma=true_gamma,
            energy=energy,
        )

        result = bridge.process_spectra(Y, peak_config)

        # Check amplitude (should be very close)
        amp_error = abs(result.amplitudes[0, 0] - true_amp) / true_amp
        assert amp_error < 0.01, f"Amplitude error {amp_error:.4f} > 1%"

    def test_three_peaks_noisy(self):
        """Test three peaks with Gaussian noise."""
        from toyomacro.voigtfit.integration import DepthProfilerBridge, PeakConfig
        from toyomacro.voigtfit.voigt_jacobian import voigt_profile

        np.random.seed(42)
        bridge = DepthProfilerBridge(enable_stage2=False)

        energy = np.linspace(280, 292, 100).astype(np.float32)
        centers = np.array([284.5, 286.0, 288.5])
        sigmas = np.array([0.8, 0.9, 0.7])
        gamma = 0.3
        true_amps = np.array([1000.0, 800.0, 600.0])

        n_spectra = 1000

        # Generate spectra
        Y = np.zeros((len(energy), n_spectra), dtype=np.float32)
        for c, s, a in zip(centers, sigmas, true_amps):
            profile = voigt_profile(energy, c, s, gamma)
            Y += a * profile[:, np.newaxis]

        # Add noise (SNR ~ 100)
        noise_level = Y.max() / 100
        Y += noise_level * np.random.randn(*Y.shape).astype(np.float32)

        peak_config = PeakConfig(
            element="C", orbital="1s",
            centers=centers, sigmas=sigmas, gamma=gamma,
            energy=energy,
        )

        result = bridge.process_spectra(Y, peak_config)

        # Check mean amplitudes (average over spectra)
        mean_amps = result.amplitudes.mean(axis=1)
        for i, (est, true) in enumerate(zip(mean_amps, true_amps)):
            rel_error = abs(est - true) / true
            assert rel_error < 0.05, f"Peak {i} mean amp error {rel_error:.4f} > 5%"

    def test_depth_profile_workflow(self):
        """Test full depth profile → spectra → fit workflow."""
        from toyomacro.voigtfit.integration import DepthProfilerBridge, PeakConfig

        bridge = DepthProfilerBridge(enable_stage2=False)

        # Depth profile parameters
        n_depth = 50
        n_angles = 10
        n_spatial = 100

        depth = np.linspace(0, 10, n_depth)
        depth_profile = np.exp(-depth / 3)[:, np.newaxis] * np.ones((1, n_spatial))

        # Transformation matrix
        angles = np.linspace(0, 60, n_angles)
        imfp = 2.0
        cos_theta = np.cos(np.radians(angles))
        T = np.exp(-depth[np.newaxis, :] / (imfp * cos_theta[:, np.newaxis]))

        peak_config = PeakConfig(
            element="C", orbital="1s",
            centers=np.array([284.5, 286.0]),
            sigmas=np.array([0.8, 0.9]),
            gamma=0.3,
            energy=np.linspace(280, 292, 80),
        )

        result, clean = bridge.process_synthetic(
            depth_profile=depth_profile,
            transformation_matrix=T,
            peak_config=peak_config,
            snr=200.0,
        )

        # Verify output shape
        n_spectra = n_angles * n_spatial
        assert result.amplitudes.shape == (2, n_spectra)
        assert result.chi2.shape == (n_spectra,)

        # Low anomaly rate for high SNR
        anomaly_rate = result.anomaly_mask.mean()
        assert anomaly_rate < 0.05, f"Anomaly rate {anomaly_rate:.2%} > 5% at SNR=200"


class TestMatlabBridge:
    """Test MATLAB file exchange interface."""

    def test_file_roundtrip(self):
        """Test HDF5 file read/write."""
        import h5py

        from toyomacro.voigtfit.matlab_bridge import process_file

        input_file = tempfile.mktemp(suffix='.h5')
        output_file = tempfile.mktemp(suffix='.h5')

        try:
            n_energy, n_spectra = 100, 1000

            # Create input
            with h5py.File(input_file, 'w') as f:
                f.create_dataset('Y', data=np.random.randn(n_energy, n_spectra).astype(np.float32))
                f.create_dataset('energy', data=np.linspace(280, 292, n_energy).astype(np.float32))
                f.create_dataset('centers', data=np.array([284.5, 286.0], dtype=np.float32))
                f.create_dataset('sigmas', data=np.array([0.8, 0.9], dtype=np.float32))
                f.attrs['gamma'] = 0.3
                f.attrs['element'] = 'C'
                f.attrs['orbital'] = '1s'

            # Process
            result = process_file(input_file, output_file)

            assert result['n_spectra'] == n_spectra
            assert result['total_time'] > 0

            # Verify output
            with h5py.File(output_file, 'r') as f:
                assert f['amplitudes'].shape == (2, n_spectra)
                assert f['chi2'].shape == (n_spectra,)
                assert f['anomaly_mask'].shape == (n_spectra,)

        finally:
            for f in [input_file, output_file]:
                if os.path.exists(f):
                    os.remove(f)


@pytest.mark.perf
class TestPerformance:
    """Performance benchmarks."""

    def test_throughput_100k(self):
        """Test throughput with 100K spectra."""
        from toyomacro.voigtfit.integration import DepthProfilerBridge, PeakConfig

        bridge = DepthProfilerBridge(enable_stage2=False)

        n_spectra = 100_000
        n_energy = 100

        Y = np.random.randn(n_energy, n_spectra).astype(np.float32)
        energy = np.linspace(280, 292, n_energy).astype(np.float32)

        peak_config = PeakConfig(
            element="C", orbital="1s",
            centers=np.array([284.5, 286.0, 288.5]),
            sigmas=np.array([0.8, 0.9, 0.7]),
            gamma=0.3,
            energy=energy,
        )

        # Warmup
        _ = bridge.process_spectra(Y[:, :1000], peak_config)

        # Benchmark
        t0 = time.perf_counter()
        result = bridge.process_spectra(Y, peak_config)
        elapsed = time.perf_counter() - t0

        rate = n_spectra / elapsed
        print(f"\n100K benchmark: {elapsed:.3f}s ({rate:,.0f} spec/s)")

        # Minimum expected rate
        min_rate = 100_000  # 100K/s minimum
        assert rate > min_rate, f"Rate {rate:,.0f} < {min_rate:,} spec/s"


def run_quick_tests():
    """Run quick validation tests."""
    print("=" * 60)
    print("VoigtFit Integration Quick Tests")
    print("=" * 60)

    tests = TestSyntheticAccuracy()

    print("\n[1] Single peak (noiseless)...")
    tests.test_single_peak_noiseless()
    print("    PASS")

    print("\n[2] Three peaks (noisy)...")
    tests.test_three_peaks_noisy()
    print("    PASS")

    print("\n[3] Depth profile workflow...")
    tests.test_depth_profile_workflow()
    print("    PASS")

    print("\n[4] MATLAB bridge file roundtrip...")
    bridge_tests = TestMatlabBridge()
    bridge_tests.test_file_roundtrip()
    print("    PASS")

    print("\n[5] Performance (100K spectra)...")
    perf_tests = TestPerformance()
    perf_tests.test_throughput_100k()
    print("    PASS")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_quick_tests()
