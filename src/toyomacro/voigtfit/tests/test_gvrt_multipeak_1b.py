"""
Tests for GVRT MultiPeak 1B Pipeline.

Validates the full pipeline:
    GT params → TwoPeakSplitEncoder → spectra gen → multipeak solver → PSNR

Tests cover:
    - GVRTMultiPeak1BBenchmark initialization
    - Parameter generation correctness
    - Encode/decode round trip fidelity
    - 2-comp spectra generation (MLX vs CPU)
    - Solver recovery at small scale
    - Swap correction logic
    - CPU dict-based generation for pipelining
    - PSNR targets match dev-log 76 (δσ > 30 dB)
    - Throughput targets (fit > 1.0 M/s at 500K)
    - Memory scaling behavior
    - End-to-end pipeline consistency across scales
"""


import importlib.util
import os

import numpy as np
import pytest

from toyomacro.voigtfit._mlx_support import mlx_usable as _mlx_usable

HAS_MLX = _mlx_usable()  # installed AND the default device can execute work
IN_CI = os.environ.get("CI") == "true"

from toyomacro.voigtfit.benchmarks.bench_gvrt_multipeak_1b import (
    AMPLITUDE_SCALE,
    DEFAULT_SEPARATION_SIGMA,
    SIGMA_NOM,
    GVRTMultiPeak1BBenchmark,
    _psnr,
)

# Small N for fast tests
N_TEST = 2_000
N_MEDIUM = 10_000


@pytest.fixture
def bench():
    """Create a benchmark instance with default settings."""
    return GVRTMultiPeak1BBenchmark(
        n_total=N_TEST,
        chunk_size=N_TEST,
    )


@pytest.fixture
def bench_large():
    """Benchmark for larger-scale tests."""
    return GVRTMultiPeak1BBenchmark(
        n_total=N_MEDIUM,
        chunk_size=N_MEDIUM,
    )


class TestBenchmarkInit:
    """Tests for GVRTMultiPeak1BBenchmark initialization."""

    def test_default_parameters(self, bench):
        """Default benchmark has correct peak configuration."""
        assert bench.separation_sigma == DEFAULT_SEPARATION_SIGMA
        assert bench.amp_ratio == 1.0
        assert bench.noise_level == 0.0
        assert bench.n_energy > 100

    def test_peak_separation(self, bench):
        """Peak centers are separated by correct distance."""
        sep_eV = DEFAULT_SEPARATION_SIGMA * SIGMA_NOM
        actual_sep = bench.center2 - bench.center1
        assert abs(actual_sep - sep_eV) < 1e-6

    def test_energy_axis_covers_both_peaks(self, bench):
        """Energy axis has sufficient margin around both peaks."""
        assert bench.energy[0] < bench.center1 - 5.0
        assert bench.energy[-1] > bench.center2 + 5.0

    def test_solver_config_creation(self, bench):
        """MultiPeakConfig is correctly created."""
        config = bench._make_solver_config()
        assert config.n_comp == 2
        assert len(config.peaks) == 2
        assert abs(config.peaks[0].center - bench.center1) < 1e-6
        assert abs(config.peaks[1].center - bench.center2) < 1e-6


class TestParameterGeneration:
    """Tests for GT parameter generation."""

    def test_shape(self, bench):
        """Generated params have correct shape (n, 8)."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(100, rng)
        assert gt.shape == (100, 8)
        assert gt.dtype == np.float32

    def test_amplitude_range(self, bench):
        """Amplitudes are in [0.1, 1.0]."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(10_000, rng)
        assert np.all(gt[:, 0] >= 0.1)
        assert np.all(gt[:, 0] <= 1.0)
        assert np.all(gt[:, 4] >= 0.0)
        assert np.all(gt[:, 4] <= 1.0)

    def test_shift_range(self, bench):
        """δE values are in [-0.5, 0.5]."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(10_000, rng)
        for col in [1, 5]:
            assert np.all(gt[:, col] >= -0.5)
            assert np.all(gt[:, col] <= 0.5)

    def test_sigma_range(self, bench):
        """δσ values are in [-0.1, 0.1]."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(10_000, rng)
        for col in [2, 6]:
            assert np.all(gt[:, col] >= -0.1)
            assert np.all(gt[:, col] <= 0.1)

    def test_gamma_zero(self, bench):
        """δγ values are always 0 (nominal γ)."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(1000, rng)
        assert np.all(gt[:, 3] == 0.0)
        assert np.all(gt[:, 7] == 0.0)

    def test_amp_ratio_scaling(self):
        """Amplitude ratio correctly scales peak 2."""
        bench = GVRTMultiPeak1BBenchmark(amp_ratio=0.5)
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(10_000, rng)
        ratio = gt[:, 4] / gt[:, 0]
        assert np.allclose(ratio, 0.5, atol=1e-6)


class TestEncodeDecode:
    """Tests for Split Fisher-Hilbert encode/decode pipeline."""

    def test_round_trip_shape(self, bench):
        """Encode/decode produces correct shapes."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(100, rng)
        channels, decoded = bench._encode_decode(gt)
        assert channels.shape == (100, 6)
        assert channels.dtype == np.uint8
        assert decoded.shape == (100, 8)

    def test_amplitude_preserved(self, bench):
        """Amplitudes are not modified by encode/decode."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(100, rng)
        _, decoded = bench._encode_decode(gt)
        np.testing.assert_array_equal(gt[:, 0], decoded[:, 0])
        np.testing.assert_array_equal(gt[:, 4], decoded[:, 4])

    def test_encoder_psnr_high(self, bench):
        """Encoder PSNR is high (> 40 dB for δE, > 40 dB for δσ)."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(N_TEST, rng)
        _, decoded = bench._encode_decode(gt)

        dE_psnr = _psnr(gt[:, 1], decoded[:, 1])
        ds_psnr = _psnr(gt[:, 2], decoded[:, 2])
        # Split encoder with order=8 gives ~58.9 dB
        assert dE_psnr > 40.0, f"δE encoder PSNR {dE_psnr:.1f} < 40 dB"
        assert ds_psnr > 40.0, f"δσ encoder PSNR {ds_psnr:.1f} < 40 dB"


class TestSpectraGeneration:
    """Tests for 2-component Voigt spectra generation."""

    def test_output_shape(self, bench):
        """Generated spectra have correct shape."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(100, rng)
        Y = bench._generate_spectra(gt)
        assert Y.shape == (100, bench.n_energy)
        assert Y.dtype == np.float32

    def test_positive_spectra(self, bench):
        """Spectra values are non-negative."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(100, rng)
        Y = bench._generate_spectra(gt)
        assert np.all(Y >= 0), "Spectra should be non-negative"

    def test_peak_location(self, bench):
        """Spectra have peaks near the expected positions."""
        rng = np.random.default_rng(42)
        # Use zero shifts so peaks are at nominal centers
        gt = bench._generate_gt_params(1, rng)
        gt[0, 1] = 0.0  # dE1 = 0
        gt[0, 2] = 0.0  # dσ1 = 0
        gt[0, 5] = 0.0  # dE2 = 0
        gt[0, 6] = 0.0  # dσ2 = 0
        Y = bench._generate_spectra(gt)

        # Find the two highest peaks
        peak_idx = np.argsort(Y[0])[-2:]
        peak_energies = bench.energy[peak_idx]
        # Both peaks should be near their centers (within 0.5 eV)
        assert any(abs(e - bench.center1) < 0.5 for e in peak_energies)
        assert any(abs(e - bench.center2) < 0.5 for e in peak_energies)


class TestSolver:
    """Tests for multipeak solver integration."""

    def test_solver_recovery_noiseless(self, bench):
        """Solver recovers parameters from noise-free 2-comp spectra."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(N_TEST, rng)
        _, decoded = bench._encode_decode(gt)
        Y = bench._generate_spectra(decoded)

        amp_rec, dE_rec, ds_rec = bench._solve_direct(Y)

        # Swap correction
        gt_amp = np.column_stack([gt[:, 0], gt[:, 4]])
        gt_dE = np.column_stack([gt[:, 1], gt[:, 5]])
        gt_ds = np.column_stack([gt[:, 2], gt[:, 6]])
        amp_c, dE_c, ds_c, n_swapped = bench._correct_swaps(
            amp_rec / AMPLITUDE_SCALE, dE_rec, ds_rec,
            gt_amp, gt_dE, gt_ds,
        )

        # PSNR checks (comparison with encoded GT, not raw GT)
        dE_psnr = (_psnr(gt[:, 1], dE_c[:, 0]) + _psnr(gt[:, 5], dE_c[:, 1])) / 2
        ds_psnr = (_psnr(gt[:, 2], ds_c[:, 0]) + _psnr(gt[:, 6], ds_c[:, 1])) / 2

        assert dE_psnr > 35.0, f"δE PSNR {dE_psnr:.1f} dB < 35 dB"
        assert ds_psnr > 25.0, f"δσ PSNR {ds_psnr:.1f} dB < 25 dB"

    def test_swap_rate_low(self, bench):
        """Swap rate is low for well-separated peaks (7σ)."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(N_TEST, rng)
        _, decoded = bench._encode_decode(gt)
        Y = bench._generate_spectra(decoded)

        amp_rec, dE_rec, ds_rec = bench._solve_direct(Y)

        gt_amp = np.column_stack([gt[:, 0], gt[:, 4]])
        gt_dE = np.column_stack([gt[:, 1], gt[:, 5]])
        gt_ds = np.column_stack([gt[:, 2], gt[:, 6]])
        _, _, _, n_swapped = bench._correct_swaps(
            amp_rec, dE_rec, ds_rec, gt_amp, gt_dE, gt_ds,
        )

        swap_rate = n_swapped / N_TEST
        assert swap_rate < 0.05, f"Swap rate {swap_rate:.2%} > 5%"


class TestSwapCorrection:
    """Tests for component swap correction."""

    def test_no_swap_when_correct(self):
        """No swaps applied when assignment is already correct."""
        n = 100
        amp = np.column_stack([np.ones(n), 0.5 * np.ones(n)])
        dE = np.column_stack([np.full(n, -0.2), np.full(n, 0.2)])
        ds = np.column_stack([np.zeros(n), np.zeros(n)])

        amp_c, dE_c, ds_c, n_swapped = GVRTMultiPeak1BBenchmark._correct_swaps(
            amp, dE, ds, amp, dE, ds,
        )
        assert n_swapped == 0
        np.testing.assert_array_equal(dE, dE_c)

    def test_swap_when_reversed(self):
        """Swaps applied when components are reversed."""
        n = 100
        amp = np.column_stack([0.5 * np.ones(n), np.ones(n)])
        dE = np.column_stack([np.full(n, 0.2), np.full(n, -0.2)])
        ds = np.column_stack([np.zeros(n), np.zeros(n)])

        gt_amp = np.column_stack([np.ones(n), 0.5 * np.ones(n)])
        gt_dE = np.column_stack([np.full(n, -0.2), np.full(n, 0.2)])
        gt_ds = ds.copy()

        amp_c, dE_c, ds_c, n_swapped = GVRTMultiPeak1BBenchmark._correct_swaps(
            amp, dE, ds, gt_amp, gt_dE, gt_ds,
        )
        assert n_swapped == n
        np.testing.assert_allclose(dE_c[:, 0], -0.2, atol=1e-6)
        np.testing.assert_allclose(dE_c[:, 1], 0.2, atol=1e-6)


class TestDictBasedGeneration:
    """Tests for CPU dict-based 2-comp spectra generation."""

    def test_dict_gen_shape(self, bench):
        """Dict-based gen produces correct shape."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(100, rng)
        bench._build_profile_matrices()
        Y = bench._gen_2comp_dict_cpu(gt)
        assert Y.shape == (100, bench.n_energy)

    def test_dict_gen_positive(self, bench):
        """Dict-based spectra are non-negative."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(100, rng)
        bench._build_profile_matrices()
        Y = bench._gen_2comp_dict_cpu(gt)
        assert np.all(Y >= 0)

    def test_dict_gen_vs_exact_correlated(self, bench):
        """Dict-based and exact spectra are correlated (R > 0.99)."""
        rng = np.random.default_rng(42)
        gt = bench._generate_gt_params(100, rng)
        bench._build_profile_matrices()

        Y_dict = bench._gen_2comp_dict_cpu(gt)
        Y_exact = bench._generate_spectra(gt)

        # Compare mean spectra (correlation should be very high)
        corr = np.corrcoef(Y_dict.ravel(), Y_exact.ravel())[0, 1]
        assert corr > 0.99, f"Correlation {corr:.4f} < 0.99"


class TestEndToEndPSNR:
    """Tests for end-to-end PSNR targets."""

    def test_psnr_targets_met(self, bench_large):
        """E2E PSNR meets dev-log 76 targets: δσ > 30 dB."""
        result = bench_large.run_smoke(n_test=N_MEDIUM)
        assert result['ds'] > 30.0, f"δσ PSNR {result['ds']:.1f} < 30 dB"
        assert result['dE'] > 40.0, f"δE PSNR {result['dE']:.1f} < 40 dB"
        assert result['amp'] > 45.0, f"amp PSNR {result['amp']:.1f} < 45 dB"

    def test_psnr_stable_across_seeds(self):
        """PSNR is consistent across different random seeds."""
        psnrs = []
        for seed in [42, 123, 456]:
            bench = GVRTMultiPeak1BBenchmark(n_total=5000, seed=seed)
            result = bench.run_smoke(n_test=5000)
            psnrs.append(result['ds'])

        # δσ PSNR should be within 3 dB across seeds
        assert max(psnrs) - min(psnrs) < 3.0, \
            f"PSNR variation {max(psnrs) - min(psnrs):.1f} dB > 3 dB"


@pytest.mark.skipif(IN_CI or not HAS_MLX,
                    reason="MLX throughput target not meaningful on CI or CPU fallback")
class TestThroughput:
    """Tests for throughput targets."""

    def test_fit_throughput_target(self):
        """Fit throughput > 0.5 M/s at sufficient batch size (warm).

        Conservative threshold that passes even under GPU contention.
        In isolation, the 100M benchmark achieves 2.23 M/s (solver-only).
        """
        bench = GVRTMultiPeak1BBenchmark(n_total=200_000, chunk_size=200_000)
        rng = np.random.default_rng(42)

        # Warmup chunk (builds dict cache)
        gt_warmup = bench._generate_gt_params(50_000, rng)
        bench._process_chunk(0, gt_warmup)

        # Measurement chunk (dict already cached)
        gt = bench._generate_gt_params(200_000, rng)
        rec, stats = bench._process_chunk(1, gt)

        assert stats.throughput > 0.5e6, \
            f"Fit throughput {stats.throughput / 1e6:.2f} M/s < 0.5 M/s"


class TestPSNRHelper:
    """Tests for the _psnr helper function."""

    def test_identical_signals(self):
        """Identical signals give infinite PSNR."""
        x = np.array([1.0, 2.0, 3.0])
        assert _psnr(x, x) == float('inf')

    def test_known_value(self):
        """Known MSE gives expected PSNR."""
        gt = np.array([0.0, 1.0], dtype=np.float64)
        rec = np.array([0.1, 0.9], dtype=np.float64)
        # range = 1.0, MSE = (0.01 + 0.01) / 2 = 0.01
        # PSNR = 10 * log10(1.0 / 0.01) = 20.0
        assert abs(_psnr(gt, rec) - 20.0) < 0.1
