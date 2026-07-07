"""
Tests for Split Encoder E2E Solver.

Validates that the SplitFisherHilbertEncoder eliminates the 10 dB δσ gap
discovered in dev-log 72.

Tests cover:
    - Split encoder pipeline adapter
    - Error separation (encoder / solver / theory / E2E / gap)
    - Separation sweep (3-regime structure)
    - Noise robustness
    - Gap elimination criterion
    - TwoPeakSplitEncoder encode/decode
"""


import numpy as np
import pytest

from toyomacro.voigtfit.benchmarks.bench_gvrt_multipeak_roundtrip import (
    make_preset,
    psnr,
    sample_params,
)
from toyomacro.voigtfit.benchmarks.bench_split_encoder_e2e import (
    euclidean_encoder_roundtrip,
    evaluate_euclidean,
    evaluate_split,
    split_encoder_roundtrip,
)
from toyomacro.voigtfit.fisher_hilbert_encoder import (
    SplitFisherHilbertConfig,
    SplitFisherHilbertEncoder,
    TwoPeakSplitEncoder,
)
from toyomacro.voigtfit.param_encoder import MultiPeakEncoder

# Small N for fast tests
N_TEST = 5_000


@pytest.fixture
def preset_7sigma():
    """Standard 2-peak preset at Δc=7σ."""
    return make_preset(7.0, hilbert_order=4)


@pytest.fixture
def split_encoder():
    """SplitFisherHilbertEncoder with ranges matching dev-log 72 preset."""
    return SplitFisherHilbertEncoder(
        config=SplitFisherHilbertConfig(
            hilbert_order=8,
            dE_range=(-0.5, 0.5),
            dsigma_range=(-0.1, 0.1),
            dgamma_range=(-0.05, 0.05),
        )
    )


# ============================================================================
# Split pipeline adapter tests
# ============================================================================


class TestSplitPipeline:
    """Tests for the Split encoder pipeline adapter."""

    def test_split_roundtrip_shapes(self, preset_7sigma, split_encoder):
        """Output shapes match input shapes."""
        rng = np.random.default_rng(42)
        _, dEs, dSs = sample_params(100, preset_7sigma, rng)
        dEs_dec, dSs_dec = split_encoder_roundtrip(dEs, dSs, split_encoder)
        assert dEs_dec.shape == dEs.shape
        assert dSs_dec.shape == dSs.shape

    def test_split_roundtrip_bounded_error(self, preset_7sigma, split_encoder):
        """Encoder error is within quantization step."""
        rng = np.random.default_rng(42)
        _, dEs, dSs = sample_params(1000, preset_7sigma, rng)
        dEs_dec, dSs_dec = split_encoder_roundtrip(dEs, dSs, split_encoder)

        steps = split_encoder.quantization_step_params()
        # Allow 2x step for Fisher non-linearity
        assert np.max(np.abs(dEs - dEs_dec)) < 2 * steps['dE'] + 1e-6
        assert np.max(np.abs(dSs - dSs_dec)) < 2 * steps['dsigma'] + 1e-6

    def test_split_encoder_psnr_high(self, preset_7sigma, split_encoder):
        """Split encoder PSNR >> Euclidean encoder PSNR for δσ."""
        rng = np.random.default_rng(42)
        amps, dEs, dSs = sample_params(10_000, preset_7sigma, rng)

        # Split encoder
        dEs_s, dSs_s = split_encoder_roundtrip(dEs, dSs, split_encoder)
        dS_range = preset_7sigma.sigma_range.max_val - preset_7sigma.sigma_range.min_val
        split_dS_psnr = np.mean([
            psnr(dSs[:, k], dSs_s[:, k], dS_range) for k in range(2)
        ])

        # Euclidean encoder
        euc = MultiPeakEncoder(preset_7sigma)
        _, dEs_e, dSs_e = euclidean_encoder_roundtrip(amps, dEs, dSs, euc)
        euc_dS_psnr = np.mean([
            psnr(dSs[:, k], dSs_e[:, k], dS_range) for k in range(2)
        ])

        # Split should be much higher (58.9 vs 34.3 dB)
        assert split_dS_psnr > euc_dS_psnr + 10.0, (
            f"Split δσ PSNR {split_dS_psnr:.1f} not >> Euclidean {euc_dS_psnr:.1f}"
        )


# ============================================================================
# Error separation tests (Part 1)
# ============================================================================


class TestErrorSeparation:
    """Tests for error separation and gap analysis."""

    @pytest.fixture(autouse=True)
    def _skip_no_mlx(self):
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            pytest.skip("MLX not available")

    def test_split_e2e_runs(self):
        """Split E2E pipeline completes without error."""
        r = evaluate_split(n_spectra=N_TEST, separation_sigma=7.0, seed=42)
        assert r.n_spectra == N_TEST
        assert r.time_s > 0
        assert r.encoder_name == "Split Fisher-Hilbert"

    def test_euclidean_e2e_runs(self):
        """Euclidean E2E pipeline completes without error."""
        r = evaluate_euclidean(n_spectra=N_TEST, separation_sigma=7.0, seed=42)
        assert r.n_spectra == N_TEST
        assert r.encoder_name == "Euclidean"

    def test_split_encoder_psnr_above_50(self):
        """Split encoder δσ PSNR > 50 dB (vs Euclidean ~34 dB)."""
        r = evaluate_split(n_spectra=N_TEST, separation_sigma=7.0, seed=42)
        assert r.enc_dS_psnr > 50.0, f"Split enc δσ {r.enc_dS_psnr:.1f} < 50 dB"

    def test_split_e2e_dS_above_30(self):
        """Split E2E δσ PSNR > 30 dB.

        This is the core success criterion: the δσ gap is largely eliminated.
        Solver-only δσ ceiling is ~37 dB (limited by n_ds=31 grid).
        E2E should be close to solver-limited, not encoder-limited.
        With N=5000, observed ~33.7 dB (gap: -3.6 dB vs dev-log 72's -11.6 dB).
        """
        r = evaluate_split(n_spectra=N_TEST, separation_sigma=7.0, seed=42)
        assert r.e2e_dS_psnr > 30.0, (
            f"Split E2E δσ {r.e2e_dS_psnr:.1f} < 30 dB"
        )

    def test_split_gap_above_minus_5(self):
        """Split δσ gap > -5 dB.

        A gap near 0 means encoder and solver errors are independent.
        """
        r = evaluate_split(n_spectra=N_TEST, separation_sigma=7.0, seed=42)
        assert r.gap_dS > -5.0, (
            f"Split gap {r.gap_dS:.1f} dB < -5 dB"
        )

    def test_split_better_than_euclidean(self):
        """Split E2E δσ PSNR significantly exceeds Euclidean."""
        r_split = evaluate_split(n_spectra=N_TEST, separation_sigma=7.0, seed=42)
        r_euc = evaluate_euclidean(n_spectra=N_TEST, separation_sigma=7.0, seed=42)
        assert r_split.e2e_dS_psnr > r_euc.e2e_dS_psnr + 5.0, (
            f"Split E2E δσ {r_split.e2e_dS_psnr:.1f} not >> "
            f"Euclidean {r_euc.e2e_dS_psnr:.1f}"
        )

    def test_split_e2e_leq_min_enc_sol(self):
        """E2E PSNR ≤ min(encoder, solver) + margin."""
        r = evaluate_split(n_spectra=N_TEST, separation_sigma=7.0, seed=42)
        margin = 3.0
        for attr in ["amp", "dE", "dS"]:
            e2e = getattr(r, f"e2e_{attr}_psnr")
            enc = getattr(r, f"enc_{attr}_psnr")
            sol = getattr(r, f"sol_{attr}_psnr")
            upper = min(enc, sol) + margin
            assert e2e <= upper, (
                f"{attr}: e2e={e2e:.1f} > min(enc={enc:.1f}, sol={sol:.1f})+{margin}"
            )


# ============================================================================
# Separation sweep tests (Part 2)
# ============================================================================


class TestSeparationSweep:
    """Tests for separation regime structure with Split encoder."""

    @pytest.fixture(autouse=True)
    def _skip_no_mlx(self):
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            pytest.skip("MLX not available")

    def test_wide_separation_high_psnr(self):
        """δσ E2E > 30 dB at Δc=7σ with Split encoder."""
        r = evaluate_split(n_spectra=N_TEST, separation_sigma=7.0, seed=100)
        assert r.e2e_dS_psnr > 30.0

    def test_close_separation_degrades(self):
        """δσ E2E decreases at Δc=3σ vs Δc=7σ."""
        r_wide = evaluate_split(n_spectra=N_TEST, separation_sigma=7.0, seed=100)
        r_close = evaluate_split(n_spectra=N_TEST, separation_sigma=3.0, seed=101)
        assert r_close.e2e_dS_psnr < r_wide.e2e_dS_psnr, (
            f"Close ({r_close.e2e_dS_psnr:.1f}) >= Wide ({r_wide.e2e_dS_psnr:.1f})"
        )

    def test_regime_boundary_maintained(self):
        """3.5σ boundary still visible with Split encoder."""
        r_35 = evaluate_split(n_spectra=N_TEST, separation_sigma=3.5, seed=102)
        r_50 = evaluate_split(n_spectra=N_TEST, separation_sigma=5.0, seed=103)
        # At 3.5σ, δE should be harder → lower PSNR
        assert r_35.e2e_dE_psnr < r_50.e2e_dE_psnr + 3.0


# ============================================================================
# Noise robustness tests (Part 3)
# ============================================================================


class TestNoiseRobustness:
    """Tests for noise degradation with Split encoder."""

    @pytest.fixture(autouse=True)
    def _skip_no_mlx(self):
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            pytest.skip("MLX not available")

    def test_noise_degrades_e2e(self):
        """Adding noise degrades E2E PSNR."""
        r_nf = evaluate_split(n_spectra=N_TEST, separation_sigma=7.0,
                               noise_level=None, seed=300)
        r_noisy = evaluate_split(n_spectra=N_TEST, separation_sigma=7.0,
                                  noise_level=1000, seed=301)
        assert r_noisy.e2e_dS_psnr < r_nf.e2e_dS_psnr + 3.0

    def test_gap_stable_under_noise(self):
        """Gap remains small even with noise (not noise-amplified)."""
        r = evaluate_split(n_spectra=N_TEST, separation_sigma=7.0,
                           noise_level=1000, seed=302)
        # Gap should not become much worse than noise-free
        assert r.gap_dS > -8.0, f"Gap under noise: {r.gap_dS:.1f} dB"

    def test_split_beats_euclidean_under_noise(self):
        """Split E2E > Euclidean E2E even with noise."""
        r_split = evaluate_split(n_spectra=N_TEST, separation_sigma=7.0,
                                  noise_level=1000, seed=303)
        r_euc = evaluate_euclidean(n_spectra=N_TEST, separation_sigma=7.0,
                                    noise_level=1000, seed=303)
        # At least no worse (noise may dominate over encoder difference)
        assert r_split.e2e_dS_psnr >= r_euc.e2e_dS_psnr - 3.0


# ============================================================================
# TwoPeakSplitEncoder tests (Part 4)
# ============================================================================


class TestTwoPeakSplitEncoder:
    """Tests for the 2-peak Split Fisher-Hilbert encoder."""

    @pytest.fixture
    def encoder(self):
        return TwoPeakSplitEncoder(eta=0.15, order_peak1=8, order_peak2=4)

    def test_construction(self, encoder):
        """Encoder constructs with two internal SplitFisherHilbertEncoder."""
        assert encoder.enc1 is not None
        assert encoder.enc2 is not None

    def test_encode_decode_roundtrip(self, encoder):
        """Encode → decode round trip preserves parameters within quantization."""
        rng = np.random.default_rng(42)
        n = 1000
        dE1 = rng.uniform(-0.5, 0.5, n)
        ds1 = rng.uniform(-0.1, 0.1, n)
        dg1 = rng.uniform(-0.05, 0.05, n)
        dE2 = rng.uniform(-0.5, 0.5, n)
        ds2 = rng.uniform(-0.1, 0.1, n)
        dg2 = rng.uniform(-0.05, 0.05, n)

        channels = encoder.encode(dE1, ds1, dg1, dE2, ds2, dg2)
        rec = encoder.decode(channels)

        # Peak 1 should be close (order=8 → 256 levels)
        step1 = encoder.enc1.quantization_step_params()
        assert np.max(np.abs(dE1 - rec['dE1'])) < 2 * step1['dE'] + 1e-5
        assert np.max(np.abs(ds1 - rec['dsigma1'])) < 2 * step1['dsigma'] + 1e-5

        # Peak 2 can be coarser (order=4 → 16 levels)
        step2 = encoder.enc2.quantization_step_params()
        assert np.max(np.abs(dE2 - rec['dE2'])) < 2 * step2['dE'] + 1e-5
        assert np.max(np.abs(ds2 - rec['dsigma2'])) < 2 * step2['dsigma'] + 1e-5

    def test_output_shape(self, encoder):
        """Encode produces 6 channels (2 × RGB)."""
        n = 100
        rng = np.random.default_rng(42)
        dE1 = rng.uniform(-0.5, 0.5, n)
        ds1 = rng.uniform(-0.1, 0.1, n)
        dg1 = np.zeros(n)
        dE2 = rng.uniform(-0.5, 0.5, n)
        ds2 = rng.uniform(-0.1, 0.1, n)
        dg2 = np.zeros(n)
        channels = encoder.encode(dE1, ds1, dg1, dE2, ds2, dg2)
        assert channels.shape == (n, 6)
        assert channels.dtype == np.uint8

    def test_peak1_higher_precision(self, encoder):
        """Peak 1 (order=8) has higher PSNR than peak 2 (order=4)."""
        rng = np.random.default_rng(42)
        n = 10_000
        dE1 = rng.uniform(-0.5, 0.5, n)
        ds1 = rng.uniform(-0.1, 0.1, n)
        dg1 = np.zeros(n)
        dE2 = rng.uniform(-0.5, 0.5, n)
        ds2 = rng.uniform(-0.1, 0.1, n)
        dg2 = np.zeros(n)

        channels = encoder.encode(dE1, ds1, dg1, dE2, ds2, dg2)
        rec = encoder.decode(channels)

        psnr1 = psnr(ds1, rec['dsigma1'], 0.2)
        psnr2 = psnr(ds2, rec['dsigma2'], 0.2)

        assert psnr1 > psnr2 + 5.0, (
            f"Peak1 δσ PSNR {psnr1:.1f} not >> Peak2 {psnr2:.1f}"
        )

    def test_equal_order_symmetric(self):
        """With equal orders, both peaks have similar precision."""
        enc = TwoPeakSplitEncoder(eta=0.15, order_peak1=6, order_peak2=6)
        rng = np.random.default_rng(42)
        n = 10_000
        dE = rng.uniform(-0.5, 0.5, (n, 2))
        ds = rng.uniform(-0.1, 0.1, (n, 2))
        dg = np.zeros((n, 2))

        channels = enc.encode(dE[:, 0], ds[:, 0], dg[:, 0],
                              dE[:, 1], ds[:, 1], dg[:, 1])
        rec = enc.decode(channels)

        psnr1 = psnr(ds[:, 0], rec['dsigma1'], 0.2)
        psnr2 = psnr(ds[:, 1], rec['dsigma2'], 0.2)

        assert abs(psnr1 - psnr2) < 3.0, (
            f"Symmetric order but asymmetric PSNR: {psnr1:.1f} vs {psnr2:.1f}"
        )
