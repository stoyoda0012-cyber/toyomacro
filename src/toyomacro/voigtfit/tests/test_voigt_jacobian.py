"""Tests for Voigt Jacobian and Hessian analytical derivatives.

Verifies d²V/dσ² and d²V/dcdσ implementations alongside
existing d²V/dc² for completeness.
"""

import numpy as np
import pytest

from toyomacro.voigtfit.voigt_jacobian import (
    verify_sigma_hessian,
    voigt_hessian_batch,
    voigt_sigma_hessian_batch,
    voigt_with_jacobian,
    voigt_with_sigma_hessian,
)

# Reference parameters (C 1s nominal)
ENERGY = np.linspace(-10, 10, 201)
CENTER = 0.0
SIGMA = 0.4247
GAMMA = 0.125


class TestSigmaHessianNumerical:
    """Verify d²V/dσ² and d²V/dcdσ against numerical central differences."""

    EPS = 1e-5
    TOL = 1e-4

    @pytest.mark.parametrize("sigma,gamma", [
        (0.3, 0.1),   # small sigma, small gamma
        (0.5, 0.3),   # typical XPS
        (1.0, 0.5),   # balanced
        (2.0, 0.8),   # large sigma, large gamma
        (0.4247, 0.125),  # C 1s nominal
    ])
    def test_d2V_dsigma2_numerical(self, sigma, gamma):
        """d²V/dσ² matches numerical central difference of dV/dσ."""
        _, _, _, _, _, d2V_dsigma2, _ = voigt_with_sigma_hessian(
            ENERGY, CENTER, sigma, gamma
        )

        _, _, dV_ds_p, _ = voigt_with_jacobian(ENERGY, CENTER, sigma + self.EPS, gamma)
        _, _, dV_ds_m, _ = voigt_with_jacobian(ENERGY, CENTER, sigma - self.EPS, gamma)
        d2V_dsigma2_num = (dV_ds_p - dV_ds_m) / (2 * self.EPS)

        mask = np.abs(d2V_dsigma2) > 1e-10
        assert mask.sum() > 0, "d²V/dσ² should have non-zero values"
        rel_err = np.max(np.abs(d2V_dsigma2[mask] - d2V_dsigma2_num[mask]) / np.abs(d2V_dsigma2[mask]))
        assert rel_err < self.TOL, f"d²V/dσ² rel error {rel_err:.2e} > {self.TOL}"

    @pytest.mark.parametrize("sigma,gamma", [
        (0.3, 0.1),
        (0.5, 0.3),
        (1.0, 0.5),
        (2.0, 0.8),
        (0.4247, 0.125),
    ])
    def test_d2V_dcdsigma_numerical(self, sigma, gamma):
        """d²V/dcdσ matches numerical d/dσ(dV/dc)."""
        _, _, _, _, _, _, d2V_dcdsigma = voigt_with_sigma_hessian(
            ENERGY, CENTER, sigma, gamma
        )

        _, dV_dc_p, _, _ = voigt_with_jacobian(ENERGY, CENTER, sigma + self.EPS, gamma)
        _, dV_dc_m, _, _ = voigt_with_jacobian(ENERGY, CENTER, sigma - self.EPS, gamma)
        d2V_dcdsigma_num = (dV_dc_p - dV_dc_m) / (2 * self.EPS)

        mask = np.abs(d2V_dcdsigma) > 1e-10
        assert mask.sum() > 0, "d²V/dcdσ should have non-zero values"
        rel_err = np.max(np.abs(d2V_dcdsigma[mask] - d2V_dcdsigma_num[mask]) / np.abs(d2V_dcdsigma[mask]))
        assert rel_err < self.TOL, f"d²V/dcdσ rel error {rel_err:.2e} > {self.TOL}"

    def test_d2V_dcdsigma_symmetry(self):
        """d²V/dcdσ = d²V/dσdc (mixed partials commute)."""
        eps = self.EPS
        _, _, _, _, _, _, d2V_dcdsigma = voigt_with_sigma_hessian(
            ENERGY, CENTER, SIGMA, GAMMA
        )

        # d/dc(dV/dσ): numerically via center perturbation
        _, _, dV_ds_p, _ = voigt_with_jacobian(ENERGY, CENTER + eps, SIGMA, GAMMA)
        _, _, dV_ds_m, _ = voigt_with_jacobian(ENERGY, CENTER - eps, SIGMA, GAMMA)
        d2V_dsdc_num = (dV_ds_p - dV_ds_m) / (2 * eps)

        mask = np.abs(d2V_dcdsigma) > 1e-10
        rel_err = np.max(np.abs(d2V_dcdsigma[mask] - d2V_dsdc_num[mask]) / np.abs(d2V_dcdsigma[mask]))
        assert rel_err < self.TOL, f"Mixed partial symmetry error {rel_err:.2e}"


class TestSigmaHessianProperties:
    """Test mathematical properties of the sigma Hessian."""

    def test_d2V_dsigma2_symmetric_about_center(self):
        """d²V/dσ² is symmetric about center for a centered profile."""
        energy_sym = np.linspace(-5, 5, 201)
        _, _, _, _, _, d2V_dsigma2, _ = voigt_with_sigma_hessian(
            energy_sym, 0.0, SIGMA, GAMMA
        )
        np.testing.assert_allclose(
            d2V_dsigma2, d2V_dsigma2[::-1], atol=1e-14,
            err_msg="d²V/dσ² should be symmetric about center"
        )

    def test_d2V_dcdsigma_antisymmetric_about_center(self):
        """d²V/dcdσ is antisymmetric about center for a centered profile."""
        energy_sym = np.linspace(-5, 5, 201)
        _, _, _, _, _, _, d2V_dcdsigma = voigt_with_sigma_hessian(
            energy_sym, 0.0, SIGMA, GAMMA
        )
        np.testing.assert_allclose(
            d2V_dcdsigma, -d2V_dcdsigma[::-1], atol=1e-14,
            err_msg="d²V/dcdσ should be antisymmetric about center"
        )

    def test_d2V_dsigma2_positive_at_peak(self):
        """d²V/dσ² should be positive at peak center (V(0)∝1/σ is convex: d²/dσ²(1/σ)=2/σ³>0)."""
        energy_fine = np.linspace(-0.01, 0.01, 3)
        _, _, _, _, _, d2V_dsigma2, _ = voigt_with_sigma_hessian(
            energy_fine, 0.0, SIGMA, GAMMA
        )
        assert d2V_dsigma2[1] > 0, "d²V/dσ² should be positive at peak"

    def test_gaussian_limit(self):
        """When γ → 0, d²V/dσ² should approach d²G/dσ² for Gaussian."""
        gamma_tiny = 1e-6
        sigma = 1.0
        energy = np.linspace(-5, 5, 201)

        _, _, _, _, _, d2V_dsigma2, _ = voigt_with_sigma_hessian(
            energy, 0.0, sigma, gamma_tiny
        )

        # Analytical d²G/dσ²: G = 1/(σ√(2π)) * exp(-x²/(2σ²))
        # dG/dσ = G * (x²/σ³ - 1/σ)
        # d²G/dσ² = G * [(x²/σ³ - 1/σ)² + (-3x²/σ⁴ + 1/σ²)]
        x = energy
        G = np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        term1 = (x**2 / sigma**3 - 1 / sigma)
        term2 = -3 * x**2 / sigma**4 + 1 / sigma**2
        d2G_dsigma2 = G * (term1**2 + term2)

        # Should match closely (not exactly due to tiny γ)
        mask = np.abs(d2G_dsigma2) > 1e-12
        rel_err = np.max(np.abs(d2V_dsigma2[mask] - d2G_dsigma2[mask]) / np.abs(d2G_dsigma2[mask]))
        assert rel_err < 1e-3, f"Gaussian limit error {rel_err:.2e}"


class TestSigmaHessianBatch:
    """Test batch computation against single-component calls."""

    def test_batch_matches_single(self):
        """voigt_sigma_hessian_batch matches per-component calls."""
        centers = np.array([284.0, 285.0, 286.0])
        sigmas = np.array([0.4, 0.5, 0.6])
        gammas = np.array([0.1, 0.15, 0.2])
        energy = np.linspace(282, 288, 101)

        Phi, J_c, J_s, J_g, H_cc, H_ss, H_cs = voigt_sigma_hessian_batch(
            energy, centers, sigmas, gammas
        )

        for k in range(3):
            V, dVc, dVs, dVg, d2cc, d2ss, d2cs = voigt_with_sigma_hessian(
                energy, centers[k], sigmas[k], gammas[k]
            )
            np.testing.assert_allclose(Phi[:, k], V, atol=1e-14)
            np.testing.assert_allclose(J_c[:, k], dVc, atol=1e-14)
            np.testing.assert_allclose(J_s[:, k], dVs, atol=1e-14)
            np.testing.assert_allclose(J_g[:, k], dVg, atol=1e-14)
            np.testing.assert_allclose(H_cc[:, k], d2cc, atol=1e-14)
            np.testing.assert_allclose(H_ss[:, k], d2ss, atol=1e-14)
            np.testing.assert_allclose(H_cs[:, k], d2cs, atol=1e-14)

    def test_batch_scalar_gamma(self):
        """Scalar gamma works in batch version."""
        centers = np.array([284.0, 285.0])
        sigmas = np.array([0.4, 0.5])
        energy = np.linspace(282, 288, 101)

        Phi, J_c, J_s, J_g, H_cc, H_ss, H_cs = voigt_sigma_hessian_batch(
            energy, centers, sigmas, 0.125
        )
        assert Phi.shape == (101, 2)
        assert H_ss.shape == (101, 2)
        assert H_cs.shape == (101, 2)

    def test_batch_consistent_with_hessian_batch(self):
        """First 5 outputs match voigt_hessian_batch."""
        centers = np.array([284.0, 285.0])
        sigmas = np.array([0.4, 0.5])
        gammas = 0.125
        energy = np.linspace(282, 288, 101)

        Phi1, Jc1, Js1, Jg1, Hcc1 = voigt_hessian_batch(
            energy, centers, sigmas, gammas
        )
        Phi2, Jc2, Js2, Jg2, Hcc2, _, _ = voigt_sigma_hessian_batch(
            energy, centers, sigmas, gammas
        )

        np.testing.assert_allclose(Phi1, Phi2, atol=1e-14)
        np.testing.assert_allclose(Jc1, Jc2, atol=1e-14)
        np.testing.assert_allclose(Js1, Js2, atol=1e-14)
        np.testing.assert_allclose(Jg1, Jg2, atol=1e-14)
        np.testing.assert_allclose(Hcc1, Hcc2, atol=1e-14)


class TestVerifySigmaHessian:
    """Test the self-verification function."""

    def test_verify_passes(self):
        """verify_sigma_hessian reports all errors < 1e-4."""
        results = verify_sigma_hessian(n_tests=10, verbose=False)
        assert results["all_passed"]
        assert results["max_error_d2V_dsigma2"] < 1e-4
        assert results["max_error_d2V_dcdsigma"] < 1e-4
