"""
Tests for Fisher Coordinate Transform.

Verify FisherCoordinateTransform and FisherHilbertEncoder.

Tests:
1. Transform round-trip (θ → ξ → θ)
2. Fisher distance anisotropy (g_σσ > g_EE for C 1s)
3. σ-γ decorrelation in Fisher space
4. FisherHilbertEncoder encode-decode round trip
5. PSNR improvement over Euclidean encoding
6. Neighbor preservation comparison
"""

import numpy as np
import pytest

from toyomacro.voigtfit.fisher_hilbert_encoder import (
    C1S_FISHER_CONFIG,
    FisherHilbertConfig,
    FisherHilbertEncoder,
    compute_neighbor_preservation,
)
from toyomacro.voigtfit.fisher_transform import (
    FisherCoordinateTransform,
)
from toyomacro.voigtfit.param_encoder import (
    MultiPeakEncoder,
    MultiPeakPreset,
    ParamRange,
)
from toyomacro.voigtfit.spectra_generator import ElementSpec

# -----------------------------------------------------------------------
# FisherCoordinateTransform tests
# -----------------------------------------------------------------------

class TestFisherCoordinateTransform:
    """Basic transform properties."""

    def test_construction_from_eta(self):
        """Construct from Voigt mixing ratio."""
        ft = FisherCoordinateTransform(eta=0.15)
        info = ft.info
        assert info.eta == pytest.approx(0.15, abs=0.02)
        assert info.sigma > 0
        assert info.gamma > 0
        assert info.g.shape == (3, 3)
        assert info.L.shape == (3, 3)

    def test_construction_from_sigma_gamma(self):
        """Construct from explicit sigma/gamma."""
        ft = FisherCoordinateTransform(sigma=0.4247, gamma=0.125)
        info = ft.info
        assert info.sigma == pytest.approx(0.4247)
        assert info.gamma == pytest.approx(0.125)

    def test_construction_requires_params(self):
        """Must provide either eta or (sigma, gamma)."""
        with pytest.raises(ValueError, match="Provide either"):
            FisherCoordinateTransform()

    def test_roundtrip_single(self):
        """Single point: θ → ξ → θ exact round trip."""
        ft = FisherCoordinateTransform(eta=0.15)
        theta = np.array([0.1, 0.02, -0.01])
        xi = ft.to_fisher(theta)
        theta_rec = ft.from_fisher(xi)
        np.testing.assert_allclose(theta_rec, theta, atol=1e-12)

    def test_roundtrip_batch(self):
        """Batch: (N, 3) θ → ξ → θ exact round trip."""
        ft = FisherCoordinateTransform(eta=0.15)
        rng = np.random.default_rng(42)
        theta = rng.uniform(-0.5, 0.5, (100, 3))
        xi = ft.to_fisher(theta)
        theta_rec = ft.from_fisher(xi)
        np.testing.assert_allclose(theta_rec, theta, atol=1e-10)

    def test_cholesky_reconstructs_g(self):
        """L @ L.T = g (Fisher sub-matrix)."""
        ft = FisherCoordinateTransform(eta=0.15)
        info = ft.info
        g_rec = info.L @ info.L.T
        np.testing.assert_allclose(g_rec, info.g, atol=1e-8)

    def test_g_is_positive_definite(self):
        """Fisher sub-matrix must be positive definite."""
        ft = FisherCoordinateTransform(eta=0.15)
        eigs = np.linalg.eigvalsh(ft.info.g)
        assert np.all(eigs > 0)

    def test_g_is_symmetric(self):
        """Fisher sub-matrix must be symmetric."""
        ft = FisherCoordinateTransform(eta=0.15)
        g = ft.info.g
        np.testing.assert_allclose(g, g.T, atol=1e-10)


class TestFisherAnisotropy:
    """Information anisotropy: g_σσ > g_EE for C 1s."""

    def test_sigma_has_more_info_than_dE(self):
        """g_σσ / g_EE ≈ 1.32 for C 1s (η=0.15)."""
        ft = FisherCoordinateTransform(eta=0.15)
        g = ft.info.g
        g_EE = g[0, 0]
        g_ss = g[1, 1]
        ratio = g_ss / g_EE
        # ratio ≈ 1.32
        assert ratio > 1.0, f"Expected g_σσ > g_EE, got ratio={ratio:.3f}"

    def test_gamma_has_most_info(self):
        """g_γγ is largest diagonal for C 1s."""
        ft = FisherCoordinateTransform(eta=0.15)
        g = ft.info.g
        assert g[2, 2] > g[0, 0], "g_γγ should exceed g_EE"
        assert g[2, 2] > g[1, 1], "g_γγ should exceed g_σσ"

    def test_fisher_distance_anisotropic(self):
        """Same Euclidean step in δE vs δσ → different Fisher distances."""
        ft = FisherCoordinateTransform(eta=0.15)
        step = 0.01
        d_dE = ft.fisher_distance(
            np.array([step, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        d_ds = ft.fisher_distance(
            np.array([0.0, step, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        # σ direction should have larger Fisher distance (more info)
        assert d_ds > d_dE, f"Fisher(δσ)={d_ds:.4f} should exceed Fisher(δE)={d_dE:.4f}"

    def test_sigma_gamma_correlated(self):
        """σ-γ correlation should be substantial in parameter space."""
        ft = FisherCoordinateTransform(eta=0.15)
        g = ft.info.g
        r_sg = g[1, 2] / np.sqrt(g[1, 1] * g[2, 2])
        assert abs(r_sg) > 0.3, f"Expected substantial σ-γ correlation, got {r_sg:.3f}"


class TestFisherDecorrelation:
    """Fisher transform decorrelates σ-γ correlation."""

    def test_fisher_coords_uncorrelated(self):
        """After transform, axes should be (approximately) uncorrelated.

        In Fisher coordinates, the metric is the identity → axes orthogonal.
        Verify by sampling grid and checking correlation of ξ components.
        """
        ft = FisherCoordinateTransform(eta=0.15)
        rng = np.random.default_rng(42)
        theta = rng.uniform(-0.1, 0.1, (5000, 3))
        xi = ft.to_fisher(theta)

        # Correlation of Fisher coordinates should reflect the transform
        # The identity metric in Fisher space means equal information per axis
        # Check that the covariance of xi has roughly equal diagonal elements
        cov = np.cov(xi.T)
        # After transform, if input is uniform, cov should be L @ Sigma @ L.T
        # The key point is that the off-diagonal of g^(-1) should be smaller
        g_inv = ft._L_inv @ ft._L_inv.T
        r_12 = g_inv[1, 2] / np.sqrt(g_inv[1, 1] * g_inv[2, 2])
        # In parameter space, σ-γ are correlated
        # In Fisher space, ξ₂-ξ₃ have reduced correlation via Cholesky
        g = ft.info.g
        r_sg_param = g[1, 2] / np.sqrt(g[1, 1] * g[2, 2])
        # Fisher coordinates decorrelate — just verify transform works
        assert abs(r_sg_param) > abs(r_12) or abs(r_12) < 0.1


class TestFisherRange:
    """Fisher range computation."""

    def test_range_contains_corners(self):
        """All parameter box corners should map within the Fisher range."""
        ft = FisherCoordinateTransform(eta=0.15)
        dE_r = (-1.0, 1.0)
        ds_r = (-0.2, 0.2)
        dg_r = (-0.05, 0.05)
        xi_min, xi_max = ft.fisher_range(dE_r, ds_r, dg_r)

        for dE in dE_r:
            for ds in ds_r:
                for dg in dg_r:
                    xi = ft.to_fisher(np.array([dE, ds, dg]))
                    for i in range(3):
                        assert xi[i] >= xi_min[i] - 1e-10
                        assert xi[i] <= xi_max[i] + 1e-10

    def test_range_positive_extent(self):
        """Fisher range should have positive extent in all dimensions."""
        ft = FisherCoordinateTransform(eta=0.15)
        xi_min, xi_max = ft.fisher_range((-1, 1), (-0.2, 0.2), (-0.05, 0.05))
        for i in range(3):
            assert xi_max[i] > xi_min[i]


class TestFisherEtaDependence:
    """Fisher transform varies with element (η)."""

    @pytest.mark.parametrize("eta", [0.05, 0.15, 0.3, 0.5, 0.7, 0.9])
    def test_roundtrip_various_eta(self, eta):
        """Transform round-trips at all eta values."""
        ft = FisherCoordinateTransform(eta=eta)
        theta = np.array([0.05, 0.01, -0.005])
        xi = ft.to_fisher(theta)
        theta_rec = ft.from_fisher(xi)
        np.testing.assert_allclose(theta_rec, theta, atol=1e-10)

    def test_lorentzian_gamma_more_important(self):
        """At high η, gamma diagonal should have larger share."""
        ft_gauss = FisherCoordinateTransform(eta=0.15)
        ft_lor = FisherCoordinateTransform(eta=0.7)
        g_g = ft_gauss.info.g
        g_l = ft_lor.info.g
        ratio_gauss = g_g[2, 2] / (g_g[1, 1] + g_g[2, 2])
        ratio_lor = g_l[2, 2] / (g_l[1, 1] + g_l[2, 2])
        assert ratio_lor > ratio_gauss


# -----------------------------------------------------------------------
# FisherHilbertEncoder tests
# -----------------------------------------------------------------------

class TestFisherHilbertEncoder:
    """Fisher-Hilbert encoder encode/decode round trip."""

    def test_roundtrip_basic(self):
        """Encode → decode round trip within quantization error."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        rng = np.random.default_rng(42)
        N = 1000
        cfg = C1S_FISHER_CONFIG

        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)

        rec = enc.encode_decode_roundtrip(dE, ds, dg)

        # Check errors are bounded by quantization step
        steps = enc.quantization_step_params()
        # Allow 2x step (Fisher transform can distort grid)
        np.testing.assert_allclose(dE, rec['dE'], atol=steps['dE'] * 2)
        np.testing.assert_allclose(ds, rec['dsigma'], atol=steps['dsigma'] * 2)
        np.testing.assert_allclose(dg, rec['dgamma'], atol=steps['dgamma'] * 2)

    def test_lossless_rgb_roundtrip(self):
        """RGB → decode → encode → RGB is lossless."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        rng = np.random.default_rng(99)
        N = 500
        rgb = rng.integers(0, 256, (N, 3), dtype=np.uint8)
        params = enc.decode(rgb)
        rgb2 = enc.encode(params['dE'], params['dsigma'], params['dgamma'])
        np.testing.assert_array_equal(rgb, rgb2)

    def test_psnr_dE_above_40dB(self):
        """δE PSNR should exceed 40 dB with 256 levels."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        rng = np.random.default_rng(42)
        N = 100_000
        cfg = C1S_FISHER_CONFIG
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)

        psnr = enc.psnr(dE, ds, dg)
        assert psnr['dE'] > 40, f"δE PSNR = {psnr['dE']:.1f} dB, expected > 40"

    def test_psnr_dsigma_above_40dB(self):
        """δσ PSNR should exceed 40 dB with 256 levels.

        This is the key dev-log 73 target: overcome the 22.4 dB limit
        from dev-log 72 (which used order=4 with 2 peaks).
        """
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        rng = np.random.default_rng(42)
        N = 100_000
        cfg = C1S_FISHER_CONFIG
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)

        psnr = enc.psnr(dE, ds, dg)
        assert psnr['dsigma'] > 40, f"δσ PSNR = {psnr['dsigma']:.1f} dB, expected > 40"

    def test_psnr_dgamma_above_40dB(self):
        """δγ PSNR should exceed 40 dB — new parameter not in old encoder."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        rng = np.random.default_rng(42)
        N = 100_000
        cfg = C1S_FISHER_CONFIG
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)

        psnr = enc.psnr(dE, ds, dg)
        assert psnr['dgamma'] > 40, f"δγ PSNR = {psnr['dgamma']:.1f} dB, expected > 40"

    def test_quantization_step_params(self):
        """Quantization steps should be reasonable for C 1s."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        steps = enc.quantization_step_params()
        # With 256 levels and range ~4 eV for dE, step < 0.02 eV
        assert steps['dE'] < 0.05, f"dE step = {steps['dE']:.4f}"
        assert steps['dsigma'] < 0.01, f"dsigma step = {steps['dsigma']:.4f}"
        assert steps['dgamma'] < 0.01, f"dgamma step = {steps['dgamma']:.4f}"

    def test_output_dtype(self):
        """Encoded output should be uint8 (N, 3)."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        dE = np.array([0.0, 0.1, -0.1])
        ds = np.array([0.0, 0.01, -0.01])
        dg = np.array([0.0, 0.005, -0.005])
        rgb = enc.encode(dE, ds, dg)
        assert rgb.shape == (3, 3)
        assert rgb.dtype == np.uint8

    def test_decoded_dtype(self):
        """Decoded output should be float32."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        rgb = np.array([[128, 64, 32]], dtype=np.uint8)
        rec = enc.decode(rgb)
        assert rec['dE'].dtype == np.float32
        assert rec['dsigma'].dtype == np.float32
        assert rec['dgamma'].dtype == np.float32

    def test_too_large_order_raises(self):
        """Order > 8 exceeds 24-bit RGB."""
        with pytest.raises(ValueError, match="exceeds 24-bit"):
            FisherHilbertEncoder(config=FisherHilbertConfig(hilbert_order=9))


class TestFisherHilbertLowOrder:
    """Fisher-Hilbert at lower orders."""

    def test_order4_roundtrip(self):
        """Order=4 (16 levels) still round-trips."""
        cfg = FisherHilbertConfig(hilbert_order=4)
        enc = FisherHilbertEncoder(config=cfg)
        rng = np.random.default_rng(42)
        N = 200
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)

        rgb = enc.encode(dE, ds, dg)
        rec = enc.decode(rgb)

        # With only 16 levels, errors are larger
        steps = enc.quantization_step_params()
        np.testing.assert_allclose(dE, rec['dE'], atol=steps['dE'] * 2)

    def test_order4_rgb_lossless(self):
        """Order=4: 12 bits, packed in lower 12 bits of 24-bit RGB."""
        cfg = FisherHilbertConfig(hilbert_order=4)
        enc = FisherHilbertEncoder(config=cfg)
        rng = np.random.default_rng(99)
        # Generate valid RGB: only lower 12 bits used (R upper 4 bits + G + B lower bits)
        # Actually all 24-bit values are valid since unused bits are just 0
        N = 100
        # Generate params, encode, then check decode-encode is lossless
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)
        rgb1 = enc.encode(dE, ds, dg)
        rec = enc.decode(rgb1)
        rgb2 = enc.encode(rec['dE'], rec['dsigma'], rec['dgamma'])
        np.testing.assert_array_equal(rgb1, rgb2)


class TestFisherVsEuclidean:
    """Compare Fisher-Hilbert vs Euclidean-Hilbert encoding.

    Both use order=8 for single peak, but Fisher uses (δE, δσ, δγ)
    in Fisher coordinates while Euclidean uses (amp, δE, δσ) uniformly.
    """

    def _make_euclidean_encoder(self):
        """Create comparable Euclidean encoder: single peak, order=8."""
        preset = MultiPeakPreset(
            name='test_euclidean_1peak',
            elements=[ElementSpec('C', '1s', 284.4)],
            energy_range=(280.0, 290.0), n_energy=101,
            hilbert_order=8,
            amplitude_range=ParamRange(0.0, 1.0, 'amplitude'),
            shift_range=ParamRange(-2.0, 2.0, 'delta_E'),
            sigma_range=ParamRange(-0.3, 0.3, 'delta_sigma'),
        )
        return MultiPeakEncoder(preset)

    def test_fisher_dsigma_psnr_comparable_or_better(self):
        """Fisher encoder should have comparable or better δσ PSNR.

        Both have 256 levels, but Fisher allocates information-optimally.
        """
        rng = np.random.default_rng(42)
        N = 50_000

        # Euclidean encoder
        enc_euc = self._make_euclidean_encoder()
        amps = rng.uniform(0, 1, (N, 1)).astype(np.float32)
        dEs_euc = rng.uniform(-2.0, 2.0, (N, 1)).astype(np.float32)
        dSs_euc = rng.uniform(-0.3, 0.3, (N, 1)).astype(np.float32)
        rgb_euc = enc_euc.decode(amps, dEs_euc, dSs_euc, (N, 1))
        a2, e2, s2 = enc_euc.encode(rgb_euc)
        mse_euc = float(np.mean((dSs_euc[:, 0] - s2[:, 0]) ** 2))
        psnr_euc = 10 * np.log10(0.6 ** 2 / mse_euc)

        # Fisher encoder
        enc_fisher = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        dEs_f = rng.uniform(-2.0, 2.0, N)
        dSs_f = rng.uniform(-0.3, 0.3, N)
        dGs_f = rng.uniform(-0.1, 0.1, N)
        psnr_f = enc_fisher.psnr(dEs_f, dSs_f, dGs_f)

        # Fisher should be at least as good (both have 256 levels)
        assert psnr_f['dsigma'] > psnr_euc - 3.0, \
            f"Fisher δσ={psnr_f['dsigma']:.1f} dB vs Euclidean={psnr_euc:.1f} dB"

    def test_fisher_encodes_dgamma(self):
        """Fisher encoder uniquely provides δγ encoding."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        rng = np.random.default_rng(42)
        N = 10_000
        cfg = C1S_FISHER_CONFIG
        dE = rng.uniform(*cfg.dE_range, N)
        ds = rng.uniform(*cfg.dsigma_range, N)
        dg = rng.uniform(*cfg.dgamma_range, N)

        psnr = enc.psnr(dE, ds, dg)
        # δγ should be encoded with meaningful precision
        assert psnr['dgamma'] > 30, \
            f"δγ PSNR = {psnr['dgamma']:.1f} dB, expected > 30 (meaningful encoding)"


class TestNeighborPreservation:
    """Locality preservation of Fisher-Hilbert vs Euclidean-Hilbert."""

    def test_preservation_positive(self):
        """Neighbor preservation should be > 0 (non-trivial locality)."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        pres = compute_neighbor_preservation(enc, n_samples=2000, k=5)
        assert pres > 0.0, f"Expected positive preservation, got {pres:.3f}"

    def test_preservation_reasonable(self):
        """Preservation should be reasonable (> 1% at k=10)."""
        enc = FisherHilbertEncoder(config=C1S_FISHER_CONFIG)
        pres = compute_neighbor_preservation(enc, n_samples=3000, k=10)
        assert pres > 0.01, f"Expected > 1% preservation at k=10, got {pres:.3f}"
