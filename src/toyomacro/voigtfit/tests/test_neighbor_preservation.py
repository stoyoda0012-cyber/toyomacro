"""
Tests for dev-log 74: Hilbert Neighbor Preservation Improvement.

Tests:
1. FisherWhitenedTransform properties
2. Cubic bounding box behavior
3. Neighbor breakdown analysis
4. Grid-normalized preservation metric
5. Systematic comparison: whitened vs cubic vs baseline
6. PSNR-preservation tradeoff characterization
"""

import numpy as np
import pytest

from toyomacro.voigtfit.fisher_hilbert_encoder import (
    C1S_FISHER_CONFIG,
    FisherHilbertConfig,
    FisherHilbertEncoder,
    NeighborBreakdownResult,
    analyze_neighbor_breakdown,
    compute_grid_neighbor_preservation,
    compute_neighbor_preservation,
)
from toyomacro.voigtfit.fisher_transform import (
    FisherCoordinateTransform,
    FisherWhitenedTransform,
)

# -----------------------------------------------------------------------
# FisherWhitenedTransform tests
# -----------------------------------------------------------------------

class TestFisherWhitenedTransform:
    """Whitened (g^{1/2}) transform properties."""

    def test_roundtrip(self):
        """θ → ξ_w → θ round trip."""
        ft = FisherWhitenedTransform(eta=0.15)
        theta = np.array([0.1, -0.02, 0.005])
        xi = ft.to_fisher(theta)
        theta_rec = ft.from_fisher(xi)
        np.testing.assert_allclose(theta_rec, theta, atol=1e-12)

    def test_roundtrip_batch(self):
        """Batch (N, 3) round trip."""
        ft = FisherWhitenedTransform(eta=0.15)
        rng = np.random.default_rng(42)
        theta = rng.uniform(-0.5, 0.5, (200, 3))
        xi = ft.to_fisher(theta)
        theta_rec = ft.from_fisher(xi)
        np.testing.assert_allclose(theta_rec, theta, atol=1e-10)

    def test_preserves_fisher_distance(self):
        """Whitened transform preserves Fisher distances (W^T W = g)."""
        ft_chol = FisherCoordinateTransform(eta=0.15)
        ft_white = FisherWhitenedTransform(eta=0.15)

        theta1 = np.array([0.1, 0.02, -0.01])
        theta2 = np.array([-0.05, 0.01, 0.005])

        d_chol = ft_chol.fisher_distance(theta1, theta2)
        d_white = ft_white.fisher_distance(theta1, theta2)
        # Both satisfy T^T T = g; small numerical difference from
        # Cholesky vs eigenvalue square root computation
        np.testing.assert_allclose(d_white, d_chol, rtol=0.01)

    def test_L_is_symmetric(self):
        """Whitened W = g^{1/2} should be symmetric."""
        ft = FisherWhitenedTransform(eta=0.15)
        L = ft.info.L
        np.testing.assert_allclose(L, L.T, atol=1e-10)

    def test_L_squared_equals_g(self):
        """W^2 = g (since W = g^{1/2}, W^T W = W W = g)."""
        ft = FisherWhitenedTransform(eta=0.15)
        info = ft.info
        g_rec = info.L @ info.L
        np.testing.assert_allclose(g_rec, info.g, atol=1e-6)

    def test_eigenvalues_stored(self):
        """Eigenvalues should be accessible and positive."""
        ft = FisherWhitenedTransform(eta=0.15)
        evals = ft.eigenvalues
        assert len(evals) == 3
        assert np.all(evals > 0)

    def test_eigenvectors_orthogonal(self):
        """Eigenvectors should form an orthogonal basis."""
        ft = FisherWhitenedTransform(eta=0.15)
        V = ft.eigenvectors
        np.testing.assert_allclose(V @ V.T, np.eye(3), atol=1e-10)

    @pytest.mark.parametrize("eta", [0.05, 0.15, 0.5, 0.9])
    def test_various_eta(self, eta):
        """Whitened transform works across eta values."""
        ft = FisherWhitenedTransform(eta=eta)
        theta = np.array([0.05, 0.01, -0.005])
        xi = ft.to_fisher(theta)
        theta_rec = ft.from_fisher(xi)
        np.testing.assert_allclose(theta_rec, theta, atol=1e-10)


# -----------------------------------------------------------------------
# BBox aspect ratio tests
# -----------------------------------------------------------------------

class TestBBoxAspectRatio:
    """Bounding box aspect ratio diagnostics."""

    def test_cholesky_aspect_ratio(self):
        """Cholesky baseline has aspect ratio ~5.8 for C 1s."""
        ft = FisherCoordinateTransform(eta=0.15)
        ar = ft.bbox_aspect_ratio((-2, 2), (-0.3, 0.3), (-0.1, 0.1))
        assert 4.0 < ar < 8.0, f"Expected ~5.8, got {ar:.2f}"

    def test_whitened_aspect_ratio(self):
        """Whitened transform has different (possibly worse) aspect ratio."""
        ft = FisherWhitenedTransform(eta=0.15)
        ar = ft.bbox_aspect_ratio((-2, 2), (-0.3, 0.3), (-0.1, 0.1))
        assert ar > 1.0

    def test_equal_ranges_lower_aspect(self):
        """Equal parameter ranges should give lower aspect ratio."""
        ft = FisherCoordinateTransform(eta=0.15)
        ar_unequal = ft.bbox_aspect_ratio((-2, 2), (-0.3, 0.3), (-0.1, 0.1))
        ar_equal = ft.bbox_aspect_ratio((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))
        assert ar_equal < ar_unequal


# -----------------------------------------------------------------------
# Cubic bounding box tests
# -----------------------------------------------------------------------

class TestCubicBBox:
    """Cubic bounding box option for isotropic Hilbert grid."""

    def test_cubic_bbox_aspect_ratio_is_one(self):
        """Cubic bbox should have aspect ratio exactly 1."""
        cfg = FisherHilbertConfig(cubic_bbox=True)
        enc = FisherHilbertEncoder(config=cfg)
        assert enc.bbox_aspect_ratio == pytest.approx(1.0)

    def test_default_bbox_anisotropic(self):
        """Default (non-cubic) bbox should be anisotropic."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        assert enc.bbox_aspect_ratio > 3.0

    def test_cubic_roundtrip(self):
        """Cubic bbox encoder round-trips correctly."""
        cfg = FisherHilbertConfig(cubic_bbox=True)
        enc = FisherHilbertEncoder(config=cfg)
        rng = np.random.default_rng(42)
        N = 500
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)

        rec = enc.encode_decode_roundtrip(dE, ds, dg)
        steps = enc.quantization_step_params()
        np.testing.assert_allclose(dE, rec['dE'], atol=steps['dE'] * 3)

    def test_cubic_lossless_rgb(self):
        """Cubic: RGB → decode → encode → same RGB."""
        cfg = FisherHilbertConfig(cubic_bbox=True)
        enc = FisherHilbertEncoder(config=cfg)
        rng = np.random.default_rng(99)
        N = 300
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)
        rgb1 = enc.encode(dE, ds, dg)
        rec = enc.decode(rgb1)
        rgb2 = enc.encode(rec['dE'], rec['dsigma'], rec['dgamma'])
        np.testing.assert_array_equal(rgb1, rgb2)

    def test_cubic_improves_preservation(self):
        """Cubic bbox should improve Fisher preservation over baseline."""
        enc_base = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        enc_cubic = FisherHilbertEncoder(
            config=FisherHilbertConfig(cubic_bbox=True))

        p_base = compute_neighbor_preservation(enc_base, n_samples=2000, k=10)
        p_cubic = compute_neighbor_preservation(enc_cubic, n_samples=2000, k=10)
        assert p_cubic > p_base, \
            f"Cubic ({p_cubic:.3f}) should exceed baseline ({p_base:.3f})"

    def test_cubic_preservation_above_30_pct(self):
        """Cubic bbox should achieve > 30% Fisher preservation at k=10."""
        cfg = FisherHilbertConfig(cubic_bbox=True)
        enc = FisherHilbertEncoder(config=cfg)
        p = compute_neighbor_preservation(enc, n_samples=3000, k=10)
        assert p > 0.30, f"Expected > 30%, got {p*100:.1f}%"

    def test_cubic_dE_psnr_unchanged(self):
        """dE PSNR should be unaffected by cubic bbox (dE has largest range)."""
        enc_base = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        enc_cubic = FisherHilbertEncoder(
            config=FisherHilbertConfig(cubic_bbox=True))

        rng = np.random.default_rng(42)
        N = 50_000
        dE = rng.uniform(-2, 2, N)
        ds = rng.uniform(-0.3, 0.3, N)
        dg = rng.uniform(-0.1, 0.1, N)

        p_base = enc_base.psnr(dE, ds, dg)
        p_cubic = enc_cubic.psnr(dE, ds, dg)
        # dE has the largest range → same step → same PSNR
        assert abs(p_base['dE'] - p_cubic['dE']) < 1.0


# -----------------------------------------------------------------------
# Grid-normalized preservation tests
# -----------------------------------------------------------------------

class TestGridPreservation:
    """Grid-normalized neighbor preservation metric."""

    def test_grid_preservation_positive(self):
        """Grid preservation should be positive."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        p = compute_grid_neighbor_preservation(enc, n_samples=2000, k=10)
        assert p > 0.0

    def test_grid_exceeds_fisher_for_anisotropic(self):
        """Grid preservation > Fisher preservation for anisotropic grid.

        The gap between grid and Fisher metrics measures the anisotropy
        penalty on the baseline encoder.
        """
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        p_fisher = compute_neighbor_preservation(enc, n_samples=2000, k=10)
        p_grid = compute_grid_neighbor_preservation(enc, n_samples=2000, k=10)
        assert p_grid > p_fisher, \
            f"Grid ({p_grid:.3f}) should exceed Fisher ({p_fisher:.3f}) " \
            f"for anisotropic grid (aspect={enc.bbox_aspect_ratio:.1f})"

    def test_grid_equals_fisher_for_cubic(self):
        """Grid ≈ Fisher for cubic bbox (isotropic grid)."""
        cfg = FisherHilbertConfig(cubic_bbox=True)
        enc = FisherHilbertEncoder(config=cfg)
        p_fisher = compute_neighbor_preservation(enc, n_samples=2000, k=10)
        p_grid = compute_grid_neighbor_preservation(enc, n_samples=2000, k=10)
        # Should be close (not exactly equal due to quantization effects)
        assert abs(p_grid - p_fisher) < 0.05, \
            f"Grid ({p_grid:.3f}) and Fisher ({p_fisher:.3f}) should be " \
            f"close for cubic bbox"


# -----------------------------------------------------------------------
# Neighbor breakdown analysis tests
# -----------------------------------------------------------------------

class TestNeighborBreakdown:
    """Neighbor breakdown analysis function."""

    def test_returns_correct_type(self):
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        result = analyze_neighbor_breakdown(enc, n_samples=1000, k=5)
        assert isinstance(result, NeighborBreakdownResult)

    def test_direction_histogram_sums_to_one(self):
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        result = analyze_neighbor_breakdown(enc, n_samples=1000, k=5)
        assert result.direction_histogram.sum() == pytest.approx(1.0, abs=0.01)

    def test_fisher_step_ratios(self):
        """Fisher step ratios should reflect bbox aspect ratio."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        result = analyze_neighbor_breakdown(enc, n_samples=1000, k=5)
        assert result.fisher_step_ratios.max() == pytest.approx(
            result.bbox_aspect_ratio, rel=0.01)
        assert result.fisher_step_ratios.min() == pytest.approx(1.0, rel=0.01)

    def test_cubic_has_uniform_steps(self):
        """Cubic bbox should have all step ratios ≈ 1."""
        cfg = FisherHilbertConfig(cubic_bbox=True)
        enc = FisherHilbertEncoder(config=cfg)
        result = analyze_neighbor_breakdown(enc, n_samples=1000, k=5)
        np.testing.assert_allclose(result.fisher_step_ratios, 1.0, atol=0.01)

    def test_broken_distances_larger(self):
        """Broken pairs should have larger median distance than preserved."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        result = analyze_neighbor_breakdown(enc, n_samples=2000, k=10)
        if len(result.broken_distances) > 0 and len(result.preserved_distances) > 0:
            assert np.median(result.broken_distances) >= np.median(
                result.preserved_distances) * 0.8


# -----------------------------------------------------------------------
# Whitened + cubic combined tests
# -----------------------------------------------------------------------

class TestWhitenedEncoder:
    """Whitened transform in encoder context."""

    def test_whitened_roundtrip(self):
        """Whitened encoder round-trips correctly."""
        cfg = FisherHilbertConfig(whitened=True)
        enc = FisherHilbertEncoder(config=cfg)
        rng = np.random.default_rng(42)
        N = 500
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)
        rec = enc.encode_decode_roundtrip(dE, ds, dg)
        steps = enc.quantization_step_params()
        np.testing.assert_allclose(dE, rec['dE'], atol=steps['dE'] * 2)

    def test_whitened_psnr_reasonable(self):
        """Whitened PSNR should be > 40 dB for all params."""
        cfg = FisherHilbertConfig(whitened=True)
        enc = FisherHilbertEncoder(config=cfg)
        rng = np.random.default_rng(42)
        N = 50_000
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)
        psnr = enc.psnr(dE, ds, dg)
        for name, val in psnr.items():
            assert val > 40, f"Whitened {name} PSNR = {val:.1f}, expected > 40"

    def test_whitened_cubic_roundtrip(self):
        """Whitened + cubic combined round-trip."""
        cfg = FisherHilbertConfig(whitened=True, cubic_bbox=True)
        enc = FisherHilbertEncoder(config=cfg)
        rng = np.random.default_rng(42)
        N = 300
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)
        rec = enc.encode_decode_roundtrip(dE, ds, dg)
        steps = enc.quantization_step_params()
        np.testing.assert_allclose(dE, rec['dE'], atol=steps['dE'] * 3)


# -----------------------------------------------------------------------
# Effective levels diagnostic test
# -----------------------------------------------------------------------

class TestEffectiveLevels:
    """Diagnostic: effective quantization levels per axis."""

    def test_baseline_all_256(self):
        """Baseline encoder uses all 256 levels per axis."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        levels = enc.effective_levels
        np.testing.assert_allclose(levels, 256.0, rtol=0.01)

    def test_cubic_short_axes_fewer(self):
        """Cubic bbox: short axes have fewer effective levels."""
        cfg = FisherHilbertConfig(cubic_bbox=True)
        enc = FisherHilbertEncoder(config=cfg)
        levels = enc.effective_levels
        # Longest axis still ~256
        assert levels.max() > 200
        # Shortest axis should have fewer effective levels
        assert levels.min() < levels.max()
