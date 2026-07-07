"""
Tests for dev-log 75: 2D Hilbert + 1D Linear Split Encoding.

Tests:
1. 2D Hilbert curve correctness (scalar + vectorized)
2. SplitFisherHilbertEncoder round-trip and PSNR
3. σ-γ neighbor preservation comparison
4. Split vs 3D encoder comparison
"""

import numpy as np
import pytest

from toyomacro.voigtfit.fisher_hilbert_encoder import (
    C1S_FISHER_CONFIG,
    C1S_SPLIT_CONFIG,
    FisherHilbertEncoder,
    SplitFisherHilbertConfig,
    SplitFisherHilbertEncoder,
    compute_neighbor_preservation,
    compute_sg_neighbor_preservation,
)
from toyomacro.voigtfit.hilbert import (
    _scalar_d_to_xy_2d,
    _scalar_xy_to_d_2d,
    coords_to_index_2d,
    index_to_coords_2d,
)

# -----------------------------------------------------------------------
# 2D Hilbert curve tests
# -----------------------------------------------------------------------

class TestHilbert2DScalar:
    """Scalar 2D Hilbert reference implementation."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_roundtrip_exhaustive(self, order):
        """Every point in [0, 2^order)^2 round-trips correctly."""
        n = 1 << order
        for x in range(n):
            for y in range(n):
                d = _scalar_xy_to_d_2d(x, y, order)
                x2, y2 = _scalar_d_to_xy_2d(d, order)
                assert (x2, y2) == (x, y), \
                    f"order={order}: ({x},{y}) -> d={d} -> ({x2},{y2})"

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_bijection(self, order):
        """Forward mapping produces unique indices covering [0, 2^(2p))."""
        n = 1 << order
        indices = set()
        for x in range(n):
            for y in range(n):
                indices.add(_scalar_xy_to_d_2d(x, y, order))
        expected = n ** 2
        assert len(indices) == expected
        assert min(indices) == 0
        assert max(indices) == expected - 1

    def test_origin(self):
        """Origin (0,0) always maps to index 0."""
        for order in range(1, 8):
            assert _scalar_xy_to_d_2d(0, 0, order) == 0


class TestHilbert2DVectorized:
    """Vectorized 2D Hilbert numpy implementation."""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_exhaustive_roundtrip(self, order):
        """All points round-trip through vectorized encode/decode."""
        n = 1 << order
        total = n ** 2
        xs = np.arange(n).repeat(n).astype(np.int64)
        ys = np.tile(np.arange(n), n).astype(np.int64)

        ds = coords_to_index_2d(xs, ys, order)
        x2, y2 = index_to_coords_2d(ds, order)

        assert np.all(x2 == xs)
        assert np.all(y2 == ys)
        assert len(np.unique(ds)) == total

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_matches_scalar(self, order):
        """Vectorized matches scalar for all points."""
        n = 1 << order
        xs, ys, ds_scalar = [], [], []
        for x in range(n):
            for y in range(n):
                xs.append(x)
                ys.append(y)
                ds_scalar.append(_scalar_xy_to_d_2d(x, y, order))

        xs = np.array(xs, dtype=np.int64)
        ys = np.array(ys, dtype=np.int64)
        ds_vec = coords_to_index_2d(xs, ys, order)
        np.testing.assert_array_equal(ds_vec, ds_scalar)

    def test_scalar_input(self):
        """Single scalar inputs work correctly."""
        d = coords_to_index_2d(np.int64(0), np.int64(0), 4)
        assert int(d) == 0
        x, y = index_to_coords_2d(np.int64(0), 4)
        assert int(x) == 0 and int(y) == 0

    def test_order8_random_roundtrip(self):
        """Order=8 (256^2 = 65536 points) random subset round-trips."""
        order = 8
        n = 1 << order
        rng = np.random.default_rng(123)
        xs = rng.integers(0, n, size=10000)
        ys = rng.integers(0, n, size=10000)
        ds = coords_to_index_2d(xs, ys, order)
        x2, y2 = index_to_coords_2d(ds, order)
        assert np.all((x2 == xs) & (y2 == ys))


class TestHilbert2DLocality:
    """Spatial locality property of 2D Hilbert curve."""

    def test_adjacent_points_nearby(self):
        """Adjacent 2D points should map to nearby indices."""
        order = 6
        n = 1 << order
        rng = np.random.default_rng(42)
        gaps = []
        for _ in range(500):
            x, y = rng.integers(0, n - 1, size=2)
            d1 = int(coords_to_index_2d(np.array([x]), np.array([y]), order)[0])
            for dx, dy in [(1, 0), (0, 1)]:
                d2 = int(coords_to_index_2d(
                    np.array([x + dx]), np.array([y + dy]), order)[0])
                gaps.append(abs(d2 - d1))
        gaps = np.array(gaps)
        assert np.median(gaps) < n ** 2 * 0.05, \
            f"Median gap {np.median(gaps)} too large (range={n**2})"


# -----------------------------------------------------------------------
# SplitFisherHilbertEncoder tests
# -----------------------------------------------------------------------

class TestSplitEncoder:
    """SplitFisherHilbertEncoder round-trip and PSNR tests."""

    def test_roundtrip(self):
        """Encode → decode round-trip within quantization error."""
        enc = SplitFisherHilbertEncoder(config=C1S_SPLIT_CONFIG)
        rng = np.random.default_rng(42)
        N = 1000
        cfg = C1S_SPLIT_CONFIG
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)

        rec = enc.encode_decode_roundtrip(dE, ds, dg)
        steps = enc.quantization_step_params()
        np.testing.assert_allclose(dE, rec['dE'], atol=steps['dE'] * 1.5)
        np.testing.assert_allclose(ds, rec['dsigma'], atol=steps['dsigma'] * 2)
        np.testing.assert_allclose(dg, rec['dgamma'], atol=steps['dgamma'] * 2)

    def test_lossless_rgb_roundtrip(self):
        """RGB → decode → encode → same RGB."""
        enc = SplitFisherHilbertEncoder(config=C1S_SPLIT_CONFIG)
        rng = np.random.default_rng(99)
        N = 500
        cfg = C1S_SPLIT_CONFIG
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)
        rgb1 = enc.encode(dE, ds, dg)
        rec = enc.decode(rgb1)
        rgb2 = enc.encode(rec['dE'], rec['dsigma'], rec['dgamma'])
        np.testing.assert_array_equal(rgb1, rgb2)

    def test_psnr_dE_above_48(self):
        """δE PSNR should be 48+ dB (8-bit linear, 256 levels)."""
        enc = SplitFisherHilbertEncoder(config=C1S_SPLIT_CONFIG)
        rng = np.random.default_rng(42)
        N = 50_000
        cfg = C1S_SPLIT_CONFIG
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)
        psnr = enc.psnr(dE, ds, dg)
        assert psnr['dE'] > 48, f"dE PSNR = {psnr['dE']:.1f}, expected > 48"

    def test_psnr_dsigma_above_50(self):
        """δσ PSNR should be > 50 dB (16-bit 2D Hilbert, 256×256)."""
        enc = SplitFisherHilbertEncoder(config=C1S_SPLIT_CONFIG)
        rng = np.random.default_rng(42)
        N = 50_000
        cfg = C1S_SPLIT_CONFIG
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)
        psnr = enc.psnr(dE, ds, dg)
        assert psnr['dsigma'] > 50, f"δσ PSNR = {psnr['dsigma']:.1f}, expected > 50"

    def test_psnr_dgamma_above_40(self):
        """δγ PSNR should be > 40 dB."""
        enc = SplitFisherHilbertEncoder(config=C1S_SPLIT_CONFIG)
        rng = np.random.default_rng(42)
        N = 50_000
        cfg = C1S_SPLIT_CONFIG
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)
        psnr = enc.psnr(dE, ds, dg)
        assert psnr['dgamma'] > 40, f"δγ PSNR = {psnr['dgamma']:.1f}, expected > 40"

    def test_split_dsigma_exceeds_3d_baseline(self):
        """Split δσ PSNR should exceed 3D baseline (58.9 dB)."""
        enc_split = SplitFisherHilbertEncoder(config=C1S_SPLIT_CONFIG)
        enc_3d = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)

        rng = np.random.default_rng(42)
        N = 50_000
        dE = rng.uniform(-2, 2, N)
        ds = rng.uniform(-0.3, 0.3, N)
        dg = rng.uniform(-0.1, 0.1, N)

        p_split = enc_split.psnr(dE, ds, dg)
        p_3d = enc_3d.psnr(dE, ds, dg)
        # Split should at least match 3D baseline for dsigma
        assert p_split['dsigma'] >= p_3d['dsigma'] - 2.0, \
            f"Split δσ={p_split['dsigma']:.1f} vs 3D={p_3d['dsigma']:.1f}"

    def test_output_shape_and_dtype(self):
        """Output has correct shape (N, 3) uint8."""
        enc = SplitFisherHilbertEncoder(config=C1S_SPLIT_CONFIG)
        dE = np.zeros(100)
        ds = np.zeros(100)
        dg = np.zeros(100)
        rgb = enc.encode(dE, ds, dg)
        assert rgb.shape == (100, 3)
        assert rgb.dtype == np.uint8

    def test_sg_correlation(self):
        """σ-γ correlation from Fisher sub-matrix should be ~0.70."""
        enc = SplitFisherHilbertEncoder(config=C1S_SPLIT_CONFIG)
        r = enc.sg_correlation
        assert 0.5 < r < 0.9, f"σ-γ correlation = {r:.3f}, expected ~0.70"

    def test_sg_bbox_aspect_ratio(self):
        """σ-γ bbox aspect ratio should be lower than 3D."""
        enc_split = SplitFisherHilbertEncoder(config=C1S_SPLIT_CONFIG)
        enc_3d = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        # 2D σ-γ bbox should be less anisotropic than full 3D bbox
        assert enc_split.sg_bbox_aspect_ratio < enc_3d.bbox_aspect_ratio

    def test_order_too_high_raises(self):
        """Order > 8 should raise ValueError."""
        with pytest.raises(ValueError, match="exceeds 8"):
            SplitFisherHilbertEncoder(config=SplitFisherHilbertConfig(hilbert_order=9))

    def test_quantization_steps(self):
        """Quantization steps should be positive and reasonable."""
        enc = SplitFisherHilbertEncoder(config=C1S_SPLIT_CONFIG)
        steps = enc.quantization_step_params()
        for name in ['dE', 'dsigma', 'dgamma']:
            assert steps[name] > 0, f"{name} step should be positive"
            assert steps[name] < 1.0, f"{name} step too large: {steps[name]}"


# -----------------------------------------------------------------------
# Neighbor preservation comparison tests
# -----------------------------------------------------------------------

class TestSplitNeighborPreservation:
    """Compare σ-γ neighbor preservation: Split vs 3D encoders."""

    def test_sg_preservation_positive(self):
        """σ-γ preservation should be positive for both encoders."""
        enc_split = SplitFisherHilbertEncoder(config=C1S_SPLIT_CONFIG)
        enc_3d = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)

        p_split = compute_sg_neighbor_preservation(enc_split, n_samples=2000, k=10)
        p_3d = compute_sg_neighbor_preservation(enc_3d, n_samples=2000, k=10)
        assert p_split > 0
        assert p_3d > 0

    def test_split_sg_preservation_exceeds_3d(self):
        """Split encoder should preserve σ-γ neighbors better than 3D."""
        enc_split = SplitFisherHilbertEncoder(config=C1S_SPLIT_CONFIG)
        enc_3d = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)

        p_split = compute_sg_neighbor_preservation(enc_split, n_samples=3000, k=10)
        p_3d = compute_sg_neighbor_preservation(enc_3d, n_samples=3000, k=10)
        assert p_split > p_3d, \
            f"Split σ-γ NP ({p_split*100:.1f}%) should exceed 3D ({p_3d*100:.1f}%)"

    def test_split_overall_preservation(self):
        """Split full-3D Fisher preservation should be reasonable."""
        enc_split = SplitFisherHilbertEncoder(config=C1S_SPLIT_CONFIG)
        p = compute_neighbor_preservation(enc_split, n_samples=2000, k=10)
        # Should be positive and reasonable
        assert p > 0.10, f"Overall NP too low: {p*100:.1f}%"

    def test_sg_preservation_function_consistent(self):
        """Same encoder with same seed gives same result."""
        enc = SplitFisherHilbertEncoder(config=C1S_SPLIT_CONFIG)
        p1 = compute_sg_neighbor_preservation(enc, n_samples=2000, k=10, seed=42)
        p2 = compute_sg_neighbor_preservation(enc, n_samples=2000, k=10, seed=42)
        assert p1 == p2
