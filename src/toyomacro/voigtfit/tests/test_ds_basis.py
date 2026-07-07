"""Tests for Doniach-Sunjic basis generation in VoigtFit.

Covers:
- DS profile accuracy (alpha=0 → Lorentzian, alpha>0)
- Gaussian convolution
- Area normalization
- Mixed Voigt+DS basis and weight matrix
- PeakConfig backward compatibility
- DS guard on Jacobian/Hessian pipelines
"""

import numpy as np
import pytest


class TestDSProfile:
    """Test ds_profile.py core functions."""

    def test_alpha_zero_is_lorentzian(self):
        """DS with alpha=0 should reduce to Lorentzian."""
        from toyomacro.voigtfit.ds_profile import ds_profile

        energy = np.linspace(-10, 10, 501, dtype=np.float64)
        center = 0.0
        gamma = 0.5  # half-width

        ds = ds_profile(energy, center, alpha=0.0, gamma=gamma)

        # Analytical Lorentzian: L(x) = 1/(pi*gamma) * gamma^2/(x^2 + gamma^2)
        # But ds_profile is unnormalized: DS(alpha=0) = Gamma(1) * cos(arctan(x/gamma)) / (x^2+gamma^2)^0.5
        # = 1 * gamma / (x^2 + gamma^2) = gamma / (x^2 + gamma^2) which is Lorentzian * pi * gamma
        lorentz_unnorm = gamma / (energy**2 + gamma**2)

        # Check shape match (relative)
        ratio = ds / lorentz_unnorm
        np.testing.assert_allclose(ratio, ratio[len(ratio) // 2], rtol=1e-10)

    def test_alpha_positive_asymmetric(self):
        """DS with alpha>0 should be asymmetric."""
        from toyomacro.voigtfit.ds_profile import ds_profile

        energy = np.linspace(-10, 10, 1001, dtype=np.float64)
        ds_asym = ds_profile(energy, center=0.0, alpha=0.1, gamma=0.5)
        ds_sym = ds_profile(energy, center=0.0, alpha=0.0, gamma=0.5)

        # With alpha>0, the profile should differ from the symmetric (alpha=0) case
        mid = len(energy) // 2
        # Check that the asymmetry breaks left-right symmetry
        left = ds_asym[:mid]
        right = ds_asym[mid + 1:][::-1]
        assert not np.allclose(left, right, rtol=0.01), \
            "DS with alpha>0 should not be symmetric"
        # Symmetric case should be symmetric
        left_s = ds_sym[:mid]
        right_s = ds_sym[mid + 1:][::-1]
        np.testing.assert_allclose(left_s, right_s, rtol=1e-10)

    def test_ds_with_gaussian_area_normalized(self):
        """ds_with_gaussian should return area=1 profile."""
        from toyomacro.voigtfit.ds_profile import ds_with_gaussian

        energy = np.linspace(-15, 15, 1001, dtype=np.float64)
        profile = ds_with_gaussian(energy, center=0.0, alpha=0.1, gamma=0.5, sigma=0.3)

        area = np.trapezoid(profile, energy)
        np.testing.assert_allclose(area, 1.0, atol=0.01)

    def test_ds_with_gaussian_zero_sigma(self):
        """sigma=0 should skip convolution."""
        from toyomacro.voigtfit.ds_profile import ds_with_gaussian

        energy = np.linspace(-15, 15, 1001, dtype=np.float64)
        profile = ds_with_gaussian(energy, center=0.0, alpha=0.1, gamma=0.5, sigma=0.0)

        area = np.trapezoid(profile, energy)
        np.testing.assert_allclose(area, 1.0, atol=0.01)

    def test_ds_with_gaussian_broadens(self):
        """Gaussian convolution should broaden the peak."""
        from toyomacro.voigtfit.ds_profile import ds_with_gaussian

        energy = np.linspace(-15, 15, 1001, dtype=np.float64)
        narrow = ds_with_gaussian(energy, center=0.0, alpha=0.05, gamma=0.3, sigma=0.0)
        broad = ds_with_gaussian(energy, center=0.0, alpha=0.05, gamma=0.3, sigma=0.8)

        assert narrow.max() > broad.max(), "Gaussian broadening should reduce peak height"

    def test_ds_with_gaussian_float32(self):
        """Output should be float32."""
        from toyomacro.voigtfit.ds_profile import ds_with_gaussian

        energy = np.linspace(-10, 10, 201, dtype=np.float64)
        profile = ds_with_gaussian(energy, center=0.0, alpha=0.1, gamma=0.5, sigma=0.3)
        assert profile.dtype == np.float32


class TestWeightCacheDSBasis:
    """Test DS basis generation in WeightMatrixCache."""

    def setup_method(self):
        import tempfile

        from toyomacro.voigtfit.weight_cache import WeightMatrixCache
        self.tmpdir = tempfile.mkdtemp()
        self.cache = WeightMatrixCache(cache_dir=self.tmpdir)
        self.energy = np.linspace(700, 720, 100, dtype=np.float32)

    def test_build_basis_ds(self):
        """build_basis_ds should return (n_energy, n_comp) float32 matrix."""
        centers = np.array([707.0, 711.0], dtype=np.float32)
        sigmas = np.array([0.4, 0.5], dtype=np.float32)
        alphas = np.array([0.1, 0.05], dtype=np.float32)

        Phi = self.cache.build_basis_ds(self.energy, centers, sigmas, gamma=0.3, alphas=alphas)

        assert Phi.shape == (100, 2)
        assert Phi.dtype == np.float32
        # Each column should be area-normalized
        for k in range(2):
            area = np.trapezoid(Phi[:, k], self.energy)
            np.testing.assert_allclose(area, 1.0, atol=0.05)

    def test_build_basis_mixed_voigt_only(self):
        """Mixed basis with all-Voigt should match build_basis_voigt."""
        centers = np.array([707.0, 711.0], dtype=np.float32)
        sigmas = np.array([0.8, 0.9], dtype=np.float32)

        Phi_voigt = self.cache.build_basis_voigt(self.energy, centers, sigmas, gamma=0.3)
        Phi_mixed = self.cache.build_basis_mixed(
            self.energy, centers, sigmas, gamma=0.3,
            lineshape_types=["voigt", "voigt"],
        )

        np.testing.assert_allclose(Phi_mixed, Phi_voigt, atol=1e-6)

    def test_build_basis_mixed_ds_and_voigt(self):
        """Mixed basis with DS+Voigt should have correct shapes."""
        centers = np.array([707.0, 709.0, 711.0], dtype=np.float32)
        sigmas = np.array([0.4, 0.8, 0.9], dtype=np.float32)
        alphas = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        lineshape_types = ["ds", "voigt", "voigt"]

        Phi = self.cache.build_basis_mixed(
            self.energy, centers, sigmas, gamma=0.3,
            lineshape_types=lineshape_types, alphas=alphas,
        )

        assert Phi.shape == (100, 3)
        assert Phi.dtype == np.float32
        # DS column (0) should be asymmetric
        peak_idx = np.argmax(Phi[:, 0])
        left_area = np.sum(Phi[:peak_idx, 0])
        right_area = np.sum(Phi[peak_idx:, 0])
        # DS tail is on high-energy side
        assert right_area != left_area  # Should not be symmetric

    def test_build_basis_mixed_ds_requires_alphas(self):
        """DS components without alphas should raise ValueError."""
        centers = np.array([707.0], dtype=np.float32)
        sigmas = np.array([0.4], dtype=np.float32)

        with pytest.raises(ValueError, match="alphas required"):
            self.cache.build_basis_mixed(
                self.energy, centers, sigmas, gamma=0.3,
                lineshape_types=["ds"], alphas=None,
            )


class TestGetOrCreateMixed:
    """Test get_or_create with DS parameters."""

    def setup_method(self):
        import tempfile

        from toyomacro.voigtfit.weight_cache import WeightMatrixCache
        self.tmpdir = tempfile.mkdtemp()
        self.cache = WeightMatrixCache(cache_dir=self.tmpdir)
        self.energy = np.linspace(700, 720, 100, dtype=np.float32)

    def test_backward_compatible_voigt_only(self):
        """Calling without alphas/lineshape_types should work as before."""
        centers = np.array([707.0, 711.0], dtype=np.float32)
        sigmas = np.array([0.8, 0.9], dtype=np.float32)

        Wt, Phi = self.cache.get_or_create(
            element="Fe", orbital="2p3/2",
            energy=self.energy, centers=centers, sigmas=sigmas, gamma=0.3,
            use_mlx=False,
        )

        assert Wt.shape == (2, 100)
        assert Phi.shape == (100, 2)

    def test_mixed_ds_voigt(self):
        """get_or_create with DS+Voigt should return valid Wt, Phi."""
        centers = np.array([707.0, 709.0, 711.0], dtype=np.float32)
        sigmas = np.array([0.4, 0.8, 0.9], dtype=np.float32)
        alphas = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        lineshape_types = ["ds", "voigt", "voigt"]

        Wt, Phi = self.cache.get_or_create(
            element="Ta", orbital="4f7/2",
            energy=self.energy, centers=centers, sigmas=sigmas, gamma=0.3,
            use_mlx=False,
            alphas=alphas, lineshape_types=lineshape_types,
        )

        assert Wt.shape == (3, 100)
        assert Phi.shape == (100, 3)

    def test_cache_key_differs_with_ds(self):
        """Cache key should change when DS params are added."""
        centers = np.array([707.0], dtype=np.float32)
        sigmas = np.array([0.8], dtype=np.float32)

        key_voigt = self.cache.get_key(
            "Fe", "2p", self.energy, centers, sigmas, 0.3,
        )
        key_ds = self.cache.get_key(
            "Fe", "2p", self.energy, centers, sigmas, 0.3,
            alphas=np.array([0.1]), lineshape_types=["ds"],
        )

        assert key_voigt != key_ds


class TestMixedBasisFitting:
    """End-to-end test: generate synthetic spectra with DS+Voigt, fit with mixed basis."""

    def test_amplitude_recovery(self):
        """Mixed basis should correctly recover amplitudes from synthetic data."""
        import tempfile

        from toyomacro.voigtfit.weight_cache import WeightMatrixCache

        energy = np.linspace(700, 720, 200, dtype=np.float32)
        tmpdir = tempfile.mkdtemp()
        cache = WeightMatrixCache(cache_dir=tmpdir)

        # Ground truth: 1 DS + 2 Voigt
        centers = np.array([707.0, 710.0, 714.0], dtype=np.float32)
        sigmas = np.array([0.4, 0.8, 0.7], dtype=np.float32)
        alphas = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        lineshape_types = ["ds", "voigt", "voigt"]
        gamma = 0.3
        true_amps = np.array([1000.0, 500.0, 800.0], dtype=np.float32)

        # Build basis and generate synthetic spectrum
        Wt, Phi = cache.get_or_create(
            element="Ta", orbital="4f",
            energy=energy, centers=centers, sigmas=sigmas, gamma=gamma,
            use_mlx=False, alphas=alphas, lineshape_types=lineshape_types,
        )

        # Y = Phi @ A (single spectrum, column vector)
        Y = (Phi @ true_amps[:, np.newaxis]).astype(np.float32)  # (200, 1)

        # Recover: A = Wt @ Y
        recovered = Wt @ Y  # (3, 1)
        recovered = recovered.squeeze()

        np.testing.assert_allclose(recovered, true_amps, rtol=0.01)

    def test_amplitude_recovery_noisy(self):
        """Recovery should be robust to moderate noise."""
        import tempfile

        from toyomacro.voigtfit.weight_cache import WeightMatrixCache

        energy = np.linspace(700, 720, 200, dtype=np.float32)
        tmpdir = tempfile.mkdtemp()
        cache = WeightMatrixCache(cache_dir=tmpdir)

        centers = np.array([707.0, 710.0, 714.0], dtype=np.float32)
        sigmas = np.array([0.4, 0.8, 0.7], dtype=np.float32)
        alphas = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        lineshape_types = ["ds", "voigt", "voigt"]
        gamma = 0.3
        true_amps = np.array([1000.0, 500.0, 800.0], dtype=np.float32)

        Wt, Phi = cache.get_or_create(
            element="Ta", orbital="4f",
            energy=energy, centers=centers, sigmas=sigmas, gamma=gamma,
            use_mlx=False, alphas=alphas, lineshape_types=lineshape_types,
        )

        # Generate 1000 noisy spectra
        rng = np.random.default_rng(42)
        Y_clean = Phi @ true_amps[:, np.newaxis]  # (200, 1)
        noise = rng.normal(0, Y_clean.max() * 0.02, (200, 1000)).astype(np.float32)
        Y = Y_clean + noise  # (200, 1000)

        recovered = Wt @ Y  # (3, 1000)
        mean_amps = recovered.mean(axis=1)

        np.testing.assert_allclose(mean_amps, true_amps, rtol=0.05)


class TestDSGuardOnJacobianPipelines:
    """DS components should raise ValueError on extended SVD pipelines."""

    def setup_method(self):
        import tempfile

        from toyomacro.voigtfit.pipeline import HybridPipeline
        from toyomacro.voigtfit.weight_cache import WeightMatrixCache
        tmpdir = tempfile.mkdtemp()
        cache = WeightMatrixCache(cache_dir=tmpdir)
        self.pipeline = HybridPipeline(cache, use_mlx=False)

    def test_extended_3param_rejects_ds(self):
        """process_rowmajor_extended_3param should reject DS components."""
        energy = np.linspace(700, 720, 100, dtype=np.float32)
        Y = np.random.randn(10, 100).astype(np.float32)

        peak_config = {
            "centers": np.array([707.0, 711.0]),
            "sigmas": np.array([0.8, 0.9]),
            "gamma": 0.3,
            "lineshape_types": ["ds", "voigt"],
            "alphas": np.array([0.1, 0.0]),
        }

        with pytest.raises(ValueError, match="DS components"):
            self.pipeline.process_rowmajor_extended_3param(
                Y=Y, energy=energy, element="Ta", orbital="4f",
                peak_config=peak_config,
            )

    def test_extended_rejects_ds(self):
        """process_rowmajor_extended should reject DS components."""
        energy = np.linspace(700, 720, 100, dtype=np.float32)
        Y = np.random.randn(10, 100).astype(np.float32)

        peak_config = {
            "centers": np.array([707.0, 711.0]),
            "sigmas": np.array([0.8, 0.9]),
            "gamma": 0.3,
            "lineshape_types": ["ds", "voigt"],
            "alphas": np.array([0.1, 0.0]),
        }

        with pytest.raises(ValueError, match="DS components"):
            self.pipeline.process_rowmajor_extended(
                Y=Y, energy=energy, element="Ta", orbital="4f",
                peak_config=peak_config,
            )
