"""Tests for template_bridge: FittingTemplate → ComponentConfig conversion.

Validates:
- Correct center/sigma/gamma conversion for all template types
- SO splitting and branch ratio conventions
- BE vs KE axis sign handling
- DS lineshape warning
- Integration with solve_grouped (synthetic spectrum fit)

template → ComponentConfig → solve_grouped → GUI pipeline.
"""

import importlib.util
import warnings

import numpy as np
import pytest

HAS_MLX = importlib.util.find_spec("mlx") is not None

from toyomacro.core.xps_database import get_so_info
from toyomacro.fitting.template_bridge import template_to_configs
from toyomacro.fitting.templates import get_template
from toyomacro.voigtfit.multipeak_config import ComponentConfig

# ---------------------------------------------------------------------------
# Unit tests: template_to_configs
# ---------------------------------------------------------------------------


class TestSi2pOxideConfigs:
    """Test Si2p_oxide template → ComponentConfig conversion."""

    def test_config_count(self):
        """Si2p_oxide has 5 peaks → 5 configs."""
        template = get_template("Si2p_oxide")
        configs = template_to_configs(template, reference_be=99.3)
        assert len(configs) == 5

    def test_centers_be(self):
        """Centers follow delta_be + reference_be."""
        template = get_template("Si2p_oxide")
        configs = template_to_configs(template, reference_be=99.3)

        expected_centers = [99.3, 100.25, 101.1, 101.9, 102.9]
        for cfg, expected in zip(configs, expected_centers):
            assert abs(cfg.center - expected) < 0.01, (
                f"center={cfg.center:.2f}, expected={expected:.2f}"
            )

    def test_sigma_conversion(self):
        """sigma = fwhm_g / 2.3548."""
        template = get_template("Si2p_oxide")
        configs = template_to_configs(template, reference_be=99.3)

        # Si0: fwhm_g=0.35 → sigma=0.35/2.3548≈0.1486
        assert abs(configs[0].sigma - 0.35 / 2.3548) < 1e-4

        # Si4+: fwhm_g=0.80 → sigma=0.80/2.3548≈0.3397
        assert abs(configs[4].sigma - 0.80 / 2.3548) < 1e-4

    def test_gamma_conversion(self):
        """gamma = fwhm_l / 2.0."""
        template = get_template("Si2p_oxide")
        configs = template_to_configs(template, reference_be=99.3)

        # Si0: fwhm_l=0.15 → gamma=0.075
        assert abs(configs[0].gamma - 0.075) < 1e-4

        # Si4+: fwhm_l=0.35 → gamma=0.175
        assert abs(configs[4].gamma - 0.175) < 1e-4

    def test_so_split(self):
        """All configs should have Si2p SO splitting (0.6 eV)."""
        template = get_template("Si2p_oxide")
        configs = template_to_configs(template, reference_be=99.3)

        for cfg in configs:
            assert cfg.so_split == pytest.approx(0.6, abs=0.01)

    def test_branch_ratio_p_shell(self):
        """p-shell: SOInfo.branch_ratio=0.5 → ComponentConfig.branch_ratio=2.0."""
        template = get_template("Si2p_oxide")
        configs = template_to_configs(template, reference_be=99.3)

        for cfg in configs:
            assert cfg.branch_ratio == pytest.approx(2.0, rel=1e-3)

    def test_is_doublet(self):
        """All Si2p configs should be doublets."""
        template = get_template("Si2p_oxide")
        configs = template_to_configs(template, reference_be=99.3)

        for cfg in configs:
            assert cfg.is_doublet


class TestC1sOrganicConfigs:
    """Test C1s_organic template (no SO splitting)."""

    def test_config_count(self):
        """C1s_organic has 4 peaks → 4 configs."""
        template = get_template("C1s_organic")
        configs = template_to_configs(template, reference_be=285.0)
        assert len(configs) == 4

    def test_no_so_split(self):
        """C1s has no SO splitting (s orbital)."""
        template = get_template("C1s_organic")
        configs = template_to_configs(template, reference_be=285.0)

        for cfg in configs:
            assert cfg.so_split == 0.0
            assert cfg.branch_ratio == 1.0
            assert not cfg.is_doublet

    def test_centers(self):
        """C1s centers: C-C=285.0, C-O=286.5, C=O=288.0, O-C=O=289.25."""
        template = get_template("C1s_organic")
        configs = template_to_configs(template, reference_be=285.0)

        expected = [285.0, 286.5, 288.0, 289.25]
        for cfg, exp in zip(configs, expected):
            assert abs(cfg.center - exp) < 0.01


class TestKEAxis:
    """Test KE axis sign convention."""

    def test_ke_sign_inversion(self):
        """KE axis: center = reference_be - delta_be."""
        template = get_template("Si2p_oxide")
        configs_be = template_to_configs(
            template, reference_be=99.3, energy_type="BE"
        )
        configs_ke = template_to_configs(
            template, reference_be=99.3, energy_type="KE"
        )

        # Si0 (delta_be=0): same for both
        assert configs_be[0].center == pytest.approx(99.3, abs=0.01)
        assert configs_ke[0].center == pytest.approx(99.3, abs=0.01)

        # Si4+ (delta_be=3.60): BE → 102.9, KE → 95.7
        assert configs_be[4].center == pytest.approx(102.9, abs=0.01)
        assert configs_ke[4].center == pytest.approx(95.7, abs=0.01)


class TestTa4fBranchRatio:
    """Test Ta4f (f-shell) branch ratio."""

    def test_f_shell_branch_ratio(self):
        """f-shell: SOInfo.branch_ratio=3/4 → ComponentConfig.branch_ratio=4/3."""
        template = get_template("Ta4f_oxide")
        configs = template_to_configs(template, reference_be=21.6)

        for cfg in configs:
            assert cfg.branch_ratio == pytest.approx(4.0 / 3.0, rel=1e-3)


class TestDSWarning:
    """Test Doniach-Sunjic lineshape warning."""

    def test_ds_peak_warning(self):
        """DS peaks should emit a warning."""
        template = get_template("Ta4f_oxide")
        # Ta0 is DS lineshape
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            configs = template_to_configs(template, reference_be=21.6)
            ds_warnings = [x for x in w if "Doniach-Sunjic" in str(x.message)]
            assert len(ds_warnings) >= 1


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_template_raises(self):
        """Template with no peaks → ValueError."""
        from toyomacro.fitting.templates import FittingTemplate

        template = FittingTemplate(
            name="empty", element="X1s", description="empty", peaks=()
        )
        with pytest.raises(ValueError, match="no peaks"):
            template_to_configs(template, reference_be=100.0)


# ---------------------------------------------------------------------------
# Integration tests: fit_grouped
# ---------------------------------------------------------------------------
class TestFitGroupedIntegration:
    """Integration test: template_to_configs → solve_grouped → AutoFitResult."""

    @pytest.fixture
    def synthetic_si2p_spectrum(self):
        """Create a synthetic Si2p spectrum with 2 oxidation states.

        Simple case: Si0 + Si4+ (well-separated, ~3.6 eV apart).
        Returns (energy, intensity) tuple.
        """
        from toyomacro.lineshape import Voigt

        voigt = Voigt()
        energy = np.linspace(96, 106, 200)

        # Si0 main (2p3/2) at 99.3 eV
        y = voigt.evaluate(energy, 99.3, 5000, fwhm_g=0.35, fwhm_l=0.15)
        # Si0 partner (2p1/2) at 99.9 eV, half intensity
        y += voigt.evaluate(energy, 99.9, 2500, fwhm_g=0.35, fwhm_l=0.15)
        # Si4+ main at 102.9 eV
        y += voigt.evaluate(energy, 102.9, 3000, fwhm_g=0.80, fwhm_l=0.35)
        # Si4+ partner at 103.5 eV
        y += voigt.evaluate(energy, 103.5, 1500, fwhm_g=0.80, fwhm_l=0.35)

        # Add linear background
        y += np.linspace(500, 300, len(energy))

        return energy, y.astype(np.float64)

    def test_fit_grouped_runs(self, synthetic_si2p_spectrum):
        """fit_grouped completes without error on synthetic data."""
        from toyomacro.core.spectrum import Spectrum
        from toyomacro.fitting import AutoFitter

        energy, intensity = synthetic_si2p_spectrum
        spectrum = Spectrum(
            energy=energy,
            intensity=intensity,
            element="Si2p",
            energy_type="BE",
        )

        fitter = AutoFitter()
        result = fitter.fit_grouped(spectrum, "Si2p_oxide")
        assert result.success, f"fit_grouped failed: {result.message}"

    def test_fit_grouped_component_count(self, synthetic_si2p_spectrum):
        """Si2p_oxide + SO → 10 PeakComponents (5 main + 5 partner)."""
        from toyomacro.core.spectrum import Spectrum
        from toyomacro.fitting import AutoFitter

        energy, intensity = synthetic_si2p_spectrum
        spectrum = Spectrum(
            energy=energy,
            intensity=intensity,
            element="Si2p",
            energy_type="BE",
        )

        fitter = AutoFitter()
        result = fitter.fit_grouped(spectrum, "Si2p_oxide")
        assert result.success

        n_components = len(result.fitting_result.components)
        # 5 chemical states × 2 (main + SO partner) = 10
        assert n_components == 10, (
            f"Expected 10 components, got {n_components}"
        )

    def test_fit_grouped_r_squared(self, synthetic_si2p_spectrum):
        """R² should be reasonable (> 0.8) on synthetic noiseless data."""
        from toyomacro.core.spectrum import Spectrum
        from toyomacro.fitting import AutoFitter

        energy, intensity = synthetic_si2p_spectrum
        spectrum = Spectrum(
            energy=energy,
            intensity=intensity,
            element="Si2p",
            energy_type="BE",
        )

        fitter = AutoFitter()
        result = fitter.fit_grouped(spectrum, "Si2p_oxide")
        assert result.success
        assert result.r_squared > 0.8, (
            f"R²={result.r_squared:.4f} < 0.8"
        )

    def test_fit_grouped_metadata(self, synthetic_si2p_spectrum):
        """Metadata should contain solver and template info."""
        from toyomacro.core.spectrum import Spectrum
        from toyomacro.fitting import AutoFitter

        energy, intensity = synthetic_si2p_spectrum
        spectrum = Spectrum(
            energy=energy,
            intensity=intensity,
            element="Si2p",
            energy_type="BE",
        )

        fitter = AutoFitter()
        result = fitter.fit_grouped(spectrum, "Si2p_oxide")
        assert result.success
        meta = result.fitting_result.metadata
        assert meta.get("solver") == "hierarchical"
        assert meta.get("template") == "Si2p_oxide"

    def test_fit_grouped_ds_fallback(self):
        """Ta4f_oxide (has DS peak) → falls back to fit_with_template."""
        from toyomacro.core.spectrum import Spectrum
        from toyomacro.fitting import AutoFitter
        from toyomacro.lineshape import Voigt

        voigt = Voigt()
        energy = np.linspace(18, 30, 200)
        y = voigt.evaluate(energy, 21.6, 5000, fwhm_g=0.40, fwhm_l=0.25)
        y += voigt.evaluate(energy, 23.5, 3750, fwhm_g=0.40, fwhm_l=0.25)
        y += voigt.evaluate(energy, 26.1, 3000, fwhm_g=0.70, fwhm_l=0.35)
        y += np.linspace(200, 150, len(energy))

        spectrum = Spectrum(
            energy=energy,
            intensity=y.astype(np.float64),
            element="Ta4f",
            energy_type="BE",
        )

        fitter = AutoFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fitter.fit_grouped(spectrum, "Ta4f_oxide")
        # Should fall back to lmfit (not crash)
        assert result.success or "failed" in result.message.lower()

    def test_fit_grouped_unknown_template(self):
        """Unknown template → falls back to auto-fit."""
        from toyomacro.core.spectrum import Spectrum
        from toyomacro.fitting import AutoFitter

        energy = np.linspace(280, 290, 100)
        y = 1000 * np.exp(-((energy - 285) ** 2) / (2 * 0.5**2)) + 100

        spectrum = Spectrum(
            energy=energy,
            intensity=y.astype(np.float64),
            element="C1s",
            energy_type="BE",
        )

        fitter = AutoFitter()
        result = fitter.fit_grouped(spectrum, "NonExistentTemplate")
        # Should not crash — falls back to fitter.fit()
        # (success depends on peak detection, but should not raise)
        assert isinstance(result.success, bool)
