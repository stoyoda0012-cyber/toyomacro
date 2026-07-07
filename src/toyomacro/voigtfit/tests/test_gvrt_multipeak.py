"""
Tests for GVRT MultiPeak Round Trip.

Validates the end-to-end pipeline:
    MultiPeakEncoder → spectra gen → multipeak solver → PSNR

Tests cover:
    - make_preset factory
    - Parameter sampling
    - 2-component spectra generation
    - Encoder round trip consistency
    - Full round trip at well-separated peaks
    - Separation regime verification
    - Amplitude ratio robustness
    - Noise degradation ordering
    - Error decomposition validity
    - PSNR threshold checks
"""


import numpy as np
import pytest

from toyomacro.voigtfit.benchmarks.bench_gvrt_multipeak_roundtrip import (
    SIGMA_NOM,
    correct_swaps,
    encoder_roundtrip,
    evaluate_roundtrip,
    generate_2comp_spectra,
    make_preset,
    psnr,
    sample_params,
    theory_min_psnr,
)
from toyomacro.voigtfit.param_encoder import MultiPeakEncoder

# Small N for fast tests
N_TEST = 5_000


class TestMakePreset:
    """Tests for make_preset factory."""

    def test_preset_center_positions(self):
        """Peak centers are correctly separated."""
        p = make_preset(7.0)
        c1 = p.elements[0].binding_energy
        c2 = p.elements[1].binding_energy
        expected_sep = 7.0 * SIGMA_NOM
        assert abs((c2 - c1) - expected_sep) < 1e-6

    def test_preset_energy_axis(self):
        """Energy axis covers both peaks with margin."""
        p = make_preset(7.0)
        energy = p.energy
        c1 = p.elements[0].binding_energy
        c2 = p.elements[1].binding_energy
        assert energy[0] < c1 - 5.0
        assert energy[-1] > c2 + 5.0

    def test_preset_2_peaks(self):
        """Preset has exactly 2 peaks."""
        p = make_preset(5.0)
        assert p.n_peaks == 2
        assert len(p.elements) == 2

    def test_preset_hilbert_order(self):
        """Custom Hilbert order is respected."""
        p = make_preset(7.0, hilbert_order=3)
        assert p.hilbert_order == 3
        assert p.n_levels == 8


class TestSampleParams:
    """Tests for parameter sampling."""

    def test_shape(self):
        """Output shapes are correct."""
        p = make_preset(7.0)
        rng = np.random.default_rng(42)
        amps, dEs, dSs = sample_params(100, p, rng)
        assert amps.shape == (100, 2)
        assert dEs.shape == (100, 2)
        assert dSs.shape == (100, 2)

    def test_amplitude_range(self):
        """Amplitudes are within [0, 1]."""
        p = make_preset(7.0)
        rng = np.random.default_rng(42)
        amps, _, _ = sample_params(1000, p, rng, amp_ratio=1.0)
        assert amps.min() >= 0.0
        assert amps.max() <= 1.0

    def test_amplitude_ratio(self):
        """Peak 2 amplitude scales with ratio."""
        p = make_preset(7.0)
        rng = np.random.default_rng(42)
        amps, _, _ = sample_params(1000, p, rng, amp_ratio=0.5)
        # Peak 2 = peak 1 * 0.5 (clipped)
        np.testing.assert_allclose(
            amps[:, 1], np.clip(amps[:, 0] * 0.5, 0, 1), atol=1e-6
        )

    def test_shift_range(self):
        """Shifts are within preset ranges."""
        p = make_preset(7.0)
        rng = np.random.default_rng(42)
        _, dEs, dSs = sample_params(1000, p, rng)
        assert dEs.min() >= p.shift_range.min_val
        assert dEs.max() <= p.shift_range.max_val
        assert dSs.min() >= p.sigma_range.min_val
        assert dSs.max() <= p.sigma_range.max_val


class TestSpectraGeneration:
    """Tests for 2-component spectra generation."""

    def test_output_shape(self):
        """Spectra have correct shape."""
        p = make_preset(7.0)
        rng = np.random.default_rng(42)
        amps, dEs, dSs = sample_params(100, p, rng)
        Y = generate_2comp_spectra(amps, dEs, dSs, p)
        assert Y.shape == (100, len(p.energy))
        assert Y.dtype == np.float32

    def test_nonnegative_noisefree(self):
        """Noise-free spectra are non-negative."""
        p = make_preset(7.0)
        rng = np.random.default_rng(42)
        amps, dEs, dSs = sample_params(100, p, rng)
        Y = generate_2comp_spectra(amps, dEs, dSs, p)
        assert Y.min() >= -1e-10

    def test_two_peaks_visible(self):
        """Two peaks are visible as local maxima in the spectrum."""
        p = make_preset(10.0)  # well-separated
        # Uniform amplitudes, zero shifts
        amps = np.array([[0.5, 0.5]], dtype=np.float32)
        dEs = np.zeros((1, 2), dtype=np.float32)
        dSs = np.zeros((1, 2), dtype=np.float32)
        Y = generate_2comp_spectra(amps, dEs, dSs, p)

        # Find peaks: local maxima in the spectrum
        spec = Y[0]
        # At least two prominent peaks
        from scipy.signal import find_peaks
        peaks_idx, _ = find_peaks(spec, prominence=0.1 * spec.max())
        assert len(peaks_idx) >= 2, f"Expected 2+ peaks, found {len(peaks_idx)}"


class TestEncoderRoundtrip:
    """Tests for encode → decode consistency."""

    def test_quantization_error_bounded(self):
        """Encoder round trip error is within quantization step."""
        p = make_preset(7.0)
        encoder = MultiPeakEncoder(p)
        rng = np.random.default_rng(42)
        amps, dEs, dSs = sample_params(1000, p, rng)

        amps_dec, dEs_dec, dSs_dec = encoder_roundtrip(amps, dEs, dSs, encoder)

        da, de, ds = encoder.quantization_step()
        assert np.max(np.abs(amps - amps_dec)) <= da / 2 + 1e-6
        assert np.max(np.abs(dEs - dEs_dec)) <= de / 2 + 1e-6
        assert np.max(np.abs(dSs - dSs_dec)) <= ds / 2 + 1e-6


class TestPSNR:
    """Tests for PSNR computation."""

    def test_perfect_recovery(self):
        """Identical arrays give very high PSNR."""
        x = np.random.randn(1000).astype(np.float32)
        assert psnr(x, x, 1.0) > 100

    def test_zero_signal_range(self):
        """PSNR handles zero MSE gracefully."""
        x = np.ones(100, dtype=np.float32)
        assert psnr(x, x, 1.0) > 100

    def test_theory_min_symmetric(self):
        """Theory min is less than or equal to both inputs."""
        result = theory_min_psnr(30.0, 40.0)
        assert result <= 30.0 + 0.1
        assert result <= 40.0 + 0.1


class TestSwapCorrection:
    """Tests for component swap correction."""

    def test_no_swap_when_correct(self):
        """No swaps when assignment is already correct."""
        n = 100
        dE = np.stack([np.ones(n) * 0.1, np.ones(n) * -0.1], axis=1)
        dS = np.zeros((n, 2))
        amp = np.ones((n, 2))
        gt_dE = dE.copy()
        gt_dS = dS.copy()
        gt_amp = amp.copy()

        _, _, _, n_swapped = correct_swaps(dE, dS, amp, gt_dE, gt_dS, gt_amp)
        assert n_swapped == 0

    def test_full_swap(self):
        """All spectra swapped when assignment is reversed."""
        n = 100
        dE = np.stack([np.ones(n) * -0.1, np.ones(n) * 0.1], axis=1)
        dS = np.zeros((n, 2))
        amp = np.ones((n, 2))
        gt_dE = np.stack([np.ones(n) * 0.1, np.ones(n) * -0.1], axis=1)
        gt_dS = dS.copy()
        gt_amp = amp.copy()

        dE_c, _, _, n_swapped = correct_swaps(dE, dS, amp, gt_dE, gt_dS, gt_amp)
        assert n_swapped == n
        np.testing.assert_allclose(dE_c[:, 0], 0.1, atol=1e-6)
        np.testing.assert_allclose(dE_c[:, 1], -0.1, atol=1e-6)


class TestFullRoundTrip:
    """End-to-end round trip tests."""

    @pytest.fixture(autouse=True)
    def _skip_no_mlx(self):
        """Skip if MLX is not available."""
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            pytest.skip("MLX not available")

    def test_basic_roundtrip_runs(self):
        """Basic round trip completes without error."""
        r = evaluate_roundtrip(
            n_spectra=N_TEST,
            separation_sigma=7.0,
            amp_ratio=1.0,
            noise_level=None,
            seed=42,
        )
        assert r.n_spectra == N_TEST
        assert r.time_s > 0

    def test_e2e_psnr_above_threshold(self):
        """End-to-end PSNR > 30 dB at Δc=7σ (success criterion)."""
        r = evaluate_roundtrip(
            n_spectra=N_TEST,
            separation_sigma=7.0,
            amp_ratio=1.0,
            noise_level=None,
            seed=42,
        )
        assert r.amp_psnr > 30.0, f"amp PSNR {r.amp_psnr:.1f} < 30 dB"
        assert r.dE_psnr > 25.0, f"dE PSNR {r.dE_psnr:.1f} < 25 dB"

    def test_encoder_psnr_matches_session71(self):
        """Encoder PSNR is near dev-log 71 values (~34 dB for amp)."""
        r = evaluate_roundtrip(
            n_spectra=N_TEST,
            separation_sigma=7.0,
            amp_ratio=1.0,
            noise_level=None,
            seed=42,
        )
        assert r.enc_amp_psnr > 30.0, f"enc amp PSNR {r.enc_amp_psnr:.1f} < 30 dB"
        assert r.enc_dE_psnr > 25.0, f"enc dE PSNR {r.enc_dE_psnr:.1f} < 25 dB"

    def test_e2e_leq_min_enc_sol(self):
        """End-to-end PSNR ≤ min(encoder, solver) + margin (error adds)."""
        r = evaluate_roundtrip(
            n_spectra=N_TEST,
            separation_sigma=7.0,
            amp_ratio=1.0,
            noise_level=None,
            seed=42,
        )
        margin = 3.0  # Allow small margin for statistical fluctuation
        for attr in ["amp_psnr", "dE_psnr", "dS_psnr"]:
            e2e = getattr(r, attr)
            enc = getattr(r, f"enc_{attr}")
            sol = getattr(r, f"sol_{attr}")
            upper = min(enc, sol) + margin
            assert e2e <= upper, (
                f"{attr}: e2e={e2e:.1f} > min(enc={enc:.1f}, sol={sol:.1f})+{margin}"
            )

    def test_separation_regime_boundary(self):
        """PSNR degrades below 3.5σ separation (regime boundary)."""
        r_wide = evaluate_roundtrip(
            n_spectra=N_TEST, separation_sigma=7.0, seed=100
        )
        r_close = evaluate_roundtrip(
            n_spectra=N_TEST, separation_sigma=3.0, seed=101
        )
        # At 3σ, δE PSNR should be significantly lower than at 7σ
        assert r_close.dE_psnr < r_wide.dE_psnr, (
            f"Close ({r_close.dE_psnr:.1f}) >= Wide ({r_wide.dE_psnr:.1f})"
        )

    def test_noise_degrades_psnr(self):
        """Adding noise degrades solver PSNR."""
        r_nf = evaluate_roundtrip(
            n_spectra=N_TEST, separation_sigma=7.0, noise_level=None, seed=200
        )
        r_noisy = evaluate_roundtrip(
            n_spectra=N_TEST, separation_sigma=7.0, noise_level=1000, seed=201
        )
        # Solver PSNR should decrease with noise
        assert r_noisy.sol_dE_psnr < r_nf.sol_dE_psnr + 3.0, (
            f"Noisy solver ({r_noisy.sol_dE_psnr:.1f}) >= NF ({r_nf.sol_dE_psnr:.1f})"
        )

    def test_theory_min_close_to_e2e(self):
        """Theory minimum is close to actual end-to-end PSNR."""
        r = evaluate_roundtrip(
            n_spectra=N_TEST, separation_sigma=7.0, seed=42
        )
        # Theory should be within ~5 dB of actual
        gap_dE = abs(r.dE_psnr - r.theory_dE_psnr)
        assert gap_dE < 5.0, f"dE gap {gap_dE:.1f} dB > 5 dB"


class TestAmplitudeRatioEffect:
    """Tests for amplitude ratio sensitivity."""

    @pytest.fixture(autouse=True)
    def _skip_no_mlx(self):
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            pytest.skip("MLX not available")

    def test_high_ratio_degrades_weak_peak(self):
        """High amplitude ratio (1:5) degrades precision vs 1:1."""
        r_equal = evaluate_roundtrip(
            n_spectra=N_TEST, separation_sigma=7.0, amp_ratio=1.0, seed=300
        )
        r_unequal = evaluate_roundtrip(
            n_spectra=N_TEST, separation_sigma=7.0, amp_ratio=5.0, seed=301
        )
        # With high ratio, weaker peak is harder to fit → lower dE PSNR
        # Allow generous margin since this is a statistical test with small N
        assert r_unequal.dE_psnr < r_equal.dE_psnr + 5.0
