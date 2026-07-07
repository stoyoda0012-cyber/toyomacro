"""
Tests for Fisher Information Matrix computation.

Verifies:
1. Symmetry and positive semi-definiteness of g_ij
2. Known scaling: g_ij ~ amplitude (Poisson)
3. Off-diagonal structure: g_13 (amp-dsigma) correlation
4. 3D vs 4D consistency (3D = upper-left block of 4D)
5. Eta conversion roundtrip
6. Cholesky transform roundtrip
7. Eigenvalue ordering and positivity
8. Numerical verification of Fisher via finite differences of KL divergence
"""

import numpy as np
import pytest

from toyomacro.voigtfit.fisher_information import (
    PARAM_NAMES_3D,
    PARAM_NAMES_4D,
    amplitude_sweep,
    compute_fisher_matrix,
    eta_from_sigma_gamma,
    eta_sweep,
    fisher_cholesky,
    fisher_inverse_transform,
    fisher_transform,
    sigma_gamma_from_eta,
)
from toyomacro.voigtfit.voigt_jacobian import voigt_profile

# Test parameters (C 1s typical)
CENTER = 284.4
SIGMA = 0.4247  # C 1s nominal Gaussian sigma
GAMMA = 0.125   # C 1s nominal Lorentzian half-width
AMP = 1000.0
ENERGY = np.linspace(279.4, 289.4, 201)


class TestFisherBasic:
    """Basic properties of Fisher Information Matrix."""

    def test_symmetry_3d(self):
        """g_ij must be symmetric."""
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        np.testing.assert_allclose(result.g, result.g.T, atol=1e-10)

    def test_symmetry_4d(self):
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='4d')
        np.testing.assert_allclose(result.g, result.g.T, atol=1e-10)

    def test_positive_semidefinite_3d(self):
        """All eigenvalues must be >= 0."""
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        assert np.all(result.eigenvalues >= -1e-10)

    def test_positive_semidefinite_4d(self):
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='4d')
        assert np.all(result.eigenvalues >= -1e-10)

    def test_shape_3d(self):
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        assert result.g.shape == (3, 3)
        assert result.correlation.shape == (3, 3)
        assert result.eigenvalues.shape == (3,)
        assert result.eigenvectors.shape == (3, 3)
        assert result.param_names == PARAM_NAMES_3D

    def test_shape_4d(self):
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='4d')
        assert result.g.shape == (4, 4)
        assert result.eigenvalues.shape == (4,)
        assert result.param_names == PARAM_NAMES_4D

    def test_eigenvalue_ascending(self):
        """Eigenvalues should be in ascending order."""
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='4d')
        for i in range(len(result.eigenvalues) - 1):
            assert result.eigenvalues[i] <= result.eigenvalues[i+1] + 1e-10

    def test_correlation_diagonal_one(self):
        """Diagonal of correlation matrix must be 1."""
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        np.testing.assert_allclose(np.diag(result.correlation), 1.0, atol=1e-10)

    def test_correlation_bounded(self):
        """Correlation coefficients must be in [-1, 1]."""
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='4d')
        assert np.all(result.correlation >= -1.0 - 1e-10)
        assert np.all(result.correlation <= 1.0 + 1e-10)

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='5d')


class TestFisherScaling:
    """Verify known scaling properties."""

    def test_amplitude_linear_scaling(self):
        """g_ij should scale linearly with amplitude for Poisson noise.

        This is because g_ij = sum J_i*J_j / f, and for f = A*V:
            J_amp = V, J_dE = A*dV/dc, J_dsigma = A*dV/dsigma
            g_amp_amp = sum V^2 / (A*V) = sum V/A ~ 1/A  ... wait
            Actually g_amp_amp = sum V*V / (A*V) = (1/A) * sum V ~ 1/A
            g_dE_dE = sum (A*dV/dc)^2 / (A*V) = A * sum (dV/dc)^2 / V ~ A

        So: g_amp_amp ~ 1/A, g_dE_dE ~ A, g_dsigma_dsigma ~ A
        The shape parameters scale linearly with amplitude.
        """
        result_lo = compute_fisher_matrix(100.0, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        result_hi = compute_fisher_matrix(1000.0, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')

        # g_amp_amp should scale as 1/A
        ratio_amp = result_hi.g[0, 0] / result_lo.g[0, 0]
        np.testing.assert_allclose(ratio_amp, 0.1, rtol=0.01)

        # g_dE_dE should scale as A
        ratio_dE = result_hi.g[1, 1] / result_lo.g[1, 1]
        np.testing.assert_allclose(ratio_dE, 10.0, rtol=0.01)

        # g_dsigma_dsigma should scale as A
        ratio_dsigma = result_hi.g[2, 2] / result_lo.g[2, 2]
        np.testing.assert_allclose(ratio_dsigma, 10.0, rtol=0.01)

    def test_condition_number_independent_of_amplitude(self):
        """Condition number of shape sub-matrix should not depend on amplitude.

        The (dE, dsigma) block scales uniformly with A, so the condition
        number (ratio of eigenvalues) is amplitude-independent.
        """
        r1 = compute_fisher_matrix(100.0, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        r2 = compute_fisher_matrix(10000.0, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')

        # Extract 2x2 shape sub-matrix (dE, dsigma)
        cond1 = np.linalg.cond(r1.g[1:, 1:])
        cond2 = np.linalg.cond(r2.g[1:, 1:])
        np.testing.assert_allclose(cond1, cond2, rtol=0.01)


class TestFisher3Dvs4D:
    """3D Fisher should be the upper-left block of 4D Fisher."""

    def test_3d_is_submatrix_of_4d(self):
        r3 = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        r4 = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='4d')

        np.testing.assert_allclose(r3.g, r4.g[:3, :3], atol=1e-10)


class TestFisherPhysics:
    """Physical expectations from XPS spectroscopy.

    KEY FINDING (Phase 1): g(delta_sigma) > g(delta_E) for C 1s parameters.
    This means the FUNDAMENTAL estimation limit is better for sigma than for dE.
    The empirical observation that delta_sigma PSNR degrades first
    is therefore a SOLVER bias issue, not a Fisher information limit.

    Also: amp and dsigma are nearly Fisher-orthogonal (r ~ 0.0003).
    The Voigt normalization ensures integral(dV/dsigma) ~ 0.
    Empirical crosstalk is solver-induced, not information-geometric.
    """

    def test_dsigma_has_substantial_info(self):
        """delta_sigma has comparable or larger Fisher info than delta_E.

        For C 1s (Gaussian-dominated), the sigma Jacobian dV/dsigma has
        large amplitude in the wings where 1/f weighting amplifies it.
        """
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        g_dE = result.g[1, 1]
        g_dsigma = result.g[2, 2]
        # Both should be substantial and of similar magnitude
        assert g_dsigma > 0
        assert g_dE > 0
        ratio = g_dsigma / g_dE
        assert 0.1 < ratio < 10, f"Expected similar magnitude, got ratio={ratio:.2f}"

    def test_amp_dsigma_nearly_orthogonal(self):
        """amp and dsigma are nearly Fisher-orthogonal.

        Voigt profile is normalized: integral(V) = 1, so integral(dV/dsigma) ~ 0.
        This means the Fisher cross-term g(amp, dsigma) ~ sum(V * dV/dsigma) / (A*V)
        = (1/A) * sum(dV/dsigma) ~ 0.

        The amp-width crosstalk observed in dev-log 40-43 is therefore
        a SOLVER artifact (linear projection bias), not a fundamental correlation.
        """
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        r_amp_dsigma = result.correlation[0, 2]
        assert abs(r_amp_dsigma) < 0.05, \
            f"Expected near-zero amp-dsigma correlation, got {r_amp_dsigma:.4f}"

    def test_sigma_gamma_correlation_4d(self):
        """sigma and gamma should be correlated in 4D.

        Both affect peak width — the Voigt identifiability problem.
        """
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='4d')
        r_sigma_gamma = result.correlation[2, 3]
        assert abs(r_sigma_gamma) > 0.01, \
            f"Expected nonzero sigma-gamma correlation, got {r_sigma_gamma:.4f}"

    def test_lorentzian_dominated_gamma_relative_importance(self):
        """In Lorentzian regime, gamma's RELATIVE share of total info should increase.

        Absolute g_gamma_gamma may decrease (broader peak = lower S/N in wings),
        but the fraction g_gamma / (g_sigma + g_gamma) should increase with eta.
        """
        # C 1s: eta ~ 0.15 (Gaussian dominated)
        r_c1s = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='4d')
        ratio_c1s = r_c1s.g[3, 3] / (r_c1s.g[2, 2] + r_c1s.g[3, 3])

        # Lorentzian-dominated (eta ~ 0.7)
        sigma_lor, gamma_lor = sigma_gamma_from_eta(0.7, fwhm_total=1.0)
        r_lor = compute_fisher_matrix(AMP, CENTER, sigma_lor, gamma_lor, ENERGY, mode='4d')
        ratio_lor = r_lor.g[3, 3] / (r_lor.g[2, 2] + r_lor.g[3, 3])

        assert ratio_lor > ratio_c1s, \
            f"Gamma share should increase: C1s={ratio_c1s:.3f}, Lor={ratio_lor:.3f}"

    def test_shape_params_scale_with_amplitude(self):
        """g_dE and g_dsigma should scale linearly with amplitude.

        Fisher info for shape parameters: g ~ A * sum((dV/dtheta)^2 / V)
        """
        r1 = compute_fisher_matrix(100.0, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        r2 = compute_fisher_matrix(1000.0, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        np.testing.assert_allclose(r2.g[1, 1] / r1.g[1, 1], 10.0, rtol=0.01)
        np.testing.assert_allclose(r2.g[2, 2] / r1.g[2, 2], 10.0, rtol=0.01)


class TestEtaConversion:
    """Eta <-> sigma/gamma conversion roundtrip."""

    @pytest.mark.parametrize("eta_target", [0.05, 0.2, 0.5, 0.8, 0.90])
    def test_roundtrip(self, eta_target):
        """sigma_gamma_from_eta -> eta_from_sigma_gamma should roundtrip."""
        sigma, gamma = sigma_gamma_from_eta(eta_target, fwhm_total=1.0)
        eta_recovered = eta_from_sigma_gamma(sigma, gamma)
        np.testing.assert_allclose(eta_recovered, eta_target, atol=0.02)

    def test_pure_gaussian(self):
        """eta=0 should give gamma=0."""
        sigma, gamma = sigma_gamma_from_eta(0.0, fwhm_total=1.0)
        assert gamma == 0.0
        assert sigma > 0

    def test_near_pure_lorentzian(self):
        """eta close to 1 should give small sigma relative to gamma."""
        sigma, gamma = sigma_gamma_from_eta(0.95, fwhm_total=1.0)
        assert sigma < gamma, "sigma should be much smaller than gamma"
        assert gamma > 0.3  # Lorentzian dominates

    def test_fwhm_scales(self):
        """Doubling fwhm_total should double sigma and gamma."""
        s1, g1 = sigma_gamma_from_eta(0.5, fwhm_total=1.0)
        s2, g2 = sigma_gamma_from_eta(0.5, fwhm_total=2.0)
        np.testing.assert_allclose(s2 / s1, 2.0, rtol=0.01)
        np.testing.assert_allclose(g2 / g1, 2.0, rtol=0.01)


class TestEtaSweep:
    """Eta sweep functionality."""

    def test_sweep_returns_expected_keys(self):
        result = eta_sweep(etas=np.array([0.1, 0.5, 0.9]), mode='4d')
        assert 'etas' in result
        assert 'diagonal' in result
        assert 'eigenvalues' in result
        assert 'info_retention_3d' in result
        assert result['diagonal'].shape == (3, 4)
        assert result['eigenvalues'].shape == (3, 4)

    def test_sweep_3d_no_retention(self):
        result = eta_sweep(etas=np.array([0.1, 0.5]), mode='3d')
        assert result['info_retention_3d'] is None

    def test_info_retention_near_unity(self):
        """For C 1s (eta=0.15), 3D retention should be very high.

        The 4th eigenvalue (gamma direction) should be negligible.
        """
        result = eta_sweep(etas=np.array([0.15]), mode='4d')
        assert result['info_retention_3d'][0] > 0.95


class TestAmplitudeSweep:
    """Amplitude sweep functionality."""

    def test_sweep_shape(self):
        amps = np.array([100, 1000, 10000])
        result = amplitude_sweep(amplitudes=amps, mode='3d')
        assert result['diagonal'].shape == (3, 3)


class TestFisherCholesky:
    """Cholesky decomposition and Fisher coordinate transform."""

    def test_cholesky_reconstructs_g(self):
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        L = fisher_cholesky(result.g)
        g_reconstructed = L @ L.T
        np.testing.assert_allclose(g_reconstructed, result.g, atol=1e-8)

    def test_transform_roundtrip(self):
        """theta -> xi -> theta should roundtrip exactly."""
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        L = fisher_cholesky(result.g)

        theta_ref = np.array([AMP, 0.0, SIGMA])
        theta = np.array([AMP + 50, 0.1, SIGMA + 0.02])

        xi = fisher_transform(theta, theta_ref, L)
        theta_recovered = fisher_inverse_transform(xi, theta_ref, L)
        np.testing.assert_allclose(theta_recovered, theta, atol=1e-10)

    def test_transform_batch(self):
        """Batch transform should work for (n_points, n_params) input."""
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        L = fisher_cholesky(result.g)

        theta_ref = np.array([AMP, 0.0, SIGMA])
        thetas = np.array([
            [AMP + 50, 0.1, SIGMA + 0.02],
            [AMP - 30, -0.05, SIGMA - 0.01],
        ])

        xi = fisher_transform(thetas, theta_ref, L)
        assert xi.shape == (2, 3)

        recovered = fisher_inverse_transform(xi, theta_ref, L)
        np.testing.assert_allclose(recovered, thetas, atol=1e-10)

    def test_fisher_distance_vs_euclidean(self):
        """Fisher distance should differ from Euclidean distance.

        Two points equidistant in parameter space should have different
        Fisher distances if the metric is non-trivial.
        """
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')
        L = fisher_cholesky(result.g)
        theta_ref = np.array([AMP, 0.0, SIGMA])

        # Same Euclidean distance, different directions
        d = 0.01
        theta_dE = theta_ref + np.array([0, d, 0])
        theta_dsigma = theta_ref + np.array([0, 0, d])

        xi_dE = fisher_transform(theta_dE, theta_ref, L)
        xi_dsigma = fisher_transform(theta_dsigma, theta_ref, L)

        fisher_dist_dE = np.linalg.norm(xi_dE)
        fisher_dist_dsigma = np.linalg.norm(xi_dsigma)

        # Fisher distance should be different (anisotropic metric)
        assert abs(fisher_dist_dE - fisher_dist_dsigma) / max(fisher_dist_dE, fisher_dist_dsigma) > 0.01


class TestNumericalVerification:
    """Verify Fisher matrix against numerical KL divergence."""

    def test_fisher_approx_kl(self):
        """g_ij * dtheta_i * dtheta_j should approximate 2 * KL divergence.

        For small perturbations: KL(p_theta || p_{theta+d}) ~ 0.5 * d^T g d
        This is the fundamental definition of Fisher metric.
        """
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')

        eps = 1e-4  # Small perturbation in delta_E direction
        dtheta = np.array([0, eps, 0])

        # Fisher approximation: KL ~ 0.5 * dtheta^T @ g @ dtheta
        kl_fisher = 0.5 * dtheta @ result.g @ dtheta

        # Numerical KL divergence between two Poisson distributions
        V0 = voigt_profile(ENERGY, CENTER, SIGMA, GAMMA)
        V1 = voigt_profile(ENERGY, CENTER + eps, SIGMA, GAMMA)
        f0 = AMP * V0
        f1 = AMP * V1

        # KL(Poisson(f0) || Poisson(f1)) = sum(f0 * log(f0/f1) - f0 + f1)
        mask = f0 > 1e-20
        kl_numerical = np.sum(
            f0[mask] * np.log(f0[mask] / f1[mask]) - f0[mask] + f1[mask]
        )

        # Should agree to O(eps^3) accuracy
        np.testing.assert_allclose(kl_fisher, kl_numerical, rtol=0.01)

    def test_fisher_approx_kl_dsigma(self):
        """Same KL verification for delta_sigma direction."""
        result = compute_fisher_matrix(AMP, CENTER, SIGMA, GAMMA, ENERGY, mode='3d')

        eps = 1e-4
        dtheta = np.array([0, 0, eps])

        kl_fisher = 0.5 * dtheta @ result.g @ dtheta

        V0 = voigt_profile(ENERGY, CENTER, SIGMA, GAMMA)
        V1 = voigt_profile(ENERGY, CENTER, SIGMA + eps, GAMMA)
        f0 = AMP * V0
        f1 = AMP * V1

        mask = f0 > 1e-20
        kl_numerical = np.sum(
            f0[mask] * np.log(f0[mask] / f1[mask]) - f0[mask] + f1[mask]
        )

        np.testing.assert_allclose(kl_fisher, kl_numerical, rtol=0.01)
