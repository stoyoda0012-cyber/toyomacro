"""Tests for the fitting template system."""

from __future__ import annotations

import re

import numpy as np
import pytest

from tests._testdata import fitting_dir
from toyomacro.core.fitting_result import LineshapeType
from toyomacro.core.spectrum import Spectrum
from toyomacro.fitting.autofitter import AutoFitConfig, AutoFitter
from toyomacro.fitting.templates import (
    FittingTemplate,
    TemplatePeak,
    get_template,
    get_templates_for_element,
    list_templates,
    register_template,
)
from toyomacro.lineshape import Voigt

# ------------------------------------------------------------------
# Template registry tests
# ------------------------------------------------------------------

class TestTemplateRegistry:
    """Test the template lookup and registration API."""

    def test_list_templates(self):
        names = list_templates()
        assert "Si2p_oxide" in names
        assert "Ta4f_oxide" in names
        assert "C1s_organic" in names

    def test_get_template_existing(self):
        t = get_template("Si2p_oxide")
        assert t is not None
        assert t.element == "Si2p"
        assert t.n_chemical_states == 5

    def test_get_template_unknown(self):
        t = get_template("nonexistent_template")
        assert t is None

    def test_get_templates_for_element(self):
        si_templates = get_templates_for_element("Si2p")
        assert len(si_templates) >= 1
        assert all(t.element == "Si2p" for t in si_templates)

        c_templates = get_templates_for_element("C1s")
        assert len(c_templates) >= 1

    def test_get_templates_for_unknown_element(self):
        templates = get_templates_for_element("Xx9z")
        assert templates == []

    def test_register_custom_template(self):
        custom = FittingTemplate(
            name="_test_custom",
            element="Au4f",
            description="Test template",
            peaks=(
                TemplatePeak(
                    name="Au0", delta_be=0.0, is_reference=True,
                ),
            ),
            apply_so=True,
            width_ordering=False,
        )
        register_template(custom)
        assert get_template("_test_custom") is not None

        # Duplicate registration should raise
        with pytest.raises(ValueError, match="already registered"):
            register_template(custom)

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError):
            register_template(get_template("Si2p_oxide"))


# ------------------------------------------------------------------
# Template data validation
# ------------------------------------------------------------------

class TestTemplateValidation:
    """Validate built-in template data integrity."""

    @pytest.mark.parametrize("name", list_templates())
    def test_has_reference_peak(self, name: str):
        t = get_template(name)
        ref = t.reference_peak
        assert ref.delta_be == 0.0
        assert ref.is_reference is True

    @pytest.mark.parametrize("name", list_templates())
    def test_peak_names_unique(self, name: str):
        t = get_template(name)
        names = [p.name for p in t.peaks]
        assert len(names) == len(set(names)), f"Duplicate peak names in {name}"

    def test_si2p_oxide_structure(self):
        t = get_template("Si2p_oxide")
        assert t.apply_so is True
        assert t.width_ordering is True
        assert t.n_chemical_states == 5
        # Width fractions should be monotonically increasing
        fracs = [p.width_fraction for p in t.peaks]
        assert fracs == [0.0, 0.25, 0.50, 0.75, 1.0]

    def test_ta4f_oxide_doniach_sunjic(self):
        t = get_template("Ta4f_oxide")
        ta0 = t.reference_peak
        assert ta0.lineshape == LineshapeType.DONIACH_SUNJIC
        assert ta0.asymmetry > 0

    def test_c1s_organic_no_so(self):
        t = get_template("C1s_organic")
        assert t.apply_so is False
        assert t.width_ordering is False
        assert t.n_chemical_states == 4


# ------------------------------------------------------------------
# Synthetic spectrum helper
# ------------------------------------------------------------------

def _make_synthetic_si2p_ke(
    photon_energy: float = 1486.7,
    noise_level: float = 100.0,
) -> Spectrum:
    """Create a synthetic Si 2p spectrum on KE axis.

    5 oxidation states × SO doublet = 10 Voigt components + Shirley BG.
    """
    voigt = Voigt()
    si0_be = 99.3
    so_split = 0.6
    br = 0.5
    states = [
        # (name, delta_be, area, fwhm_g, fwhm_l)
        ("Si0", 0.0, 12000, 0.35, 0.15),
        ("Si1+", 0.95, 2000, 0.45, 0.20),
        ("Si2+", 1.80, 1500, 0.55, 0.25),
        ("Si3+", 2.60, 3000, 0.65, 0.30),
        ("Si4+", 3.60, 8000, 0.80, 0.35),
    ]

    energy = np.linspace(
        photon_energy - si0_be - 5.0,
        photon_energy - si0_be + 2.0,
        201,
    )

    signal = np.zeros_like(energy)
    for _, dbe, area, fwhm_g, fwhm_l in states:
        center_ke = photon_energy - (si0_be + dbe)
        # Main (j+)
        signal += voigt.evaluate(energy, center_ke, area, fwhm_g=fwhm_g, fwhm_l=fwhm_l)
        # Partner (j-)
        partner_ke = center_ke - so_split
        signal += voigt.evaluate(
            energy, partner_ke, area * br, fwhm_g=fwhm_g, fwhm_l=fwhm_l,
        )

    # Add simple linear background
    bg = np.linspace(5000, 8000, len(energy))
    intensity = signal + bg

    # Add noise
    rng = np.random.default_rng(42)
    intensity += rng.normal(0, noise_level, len(intensity))
    intensity = np.maximum(intensity, 0)

    return Spectrum(
        energy=energy,
        intensity=intensity,
        energy_type="KE",
        element="Si2p",
        metadata={"photon_energy": photon_energy},
    )


def _make_synthetic_c1s_be(noise_level: float = 50.0) -> Spectrum:
    """Create a synthetic C 1s spectrum on BE axis (no SO)."""
    voigt = Voigt()
    cc_be = 285.0
    states = [
        ("C-C", 0.0, 5000, 0.60, 0.20),
        ("C-O", 1.5, 1500, 0.65, 0.20),
        ("C=O", 3.0, 800, 0.65, 0.20),
        ("O-C=O", 4.25, 500, 0.70, 0.25),
    ]

    energy = np.linspace(283.0, 292.0, 181)
    signal = np.zeros_like(energy)
    for _, dbe, area, fwhm_g, fwhm_l in states:
        center = cc_be + dbe
        signal += voigt.evaluate(energy, center, area, fwhm_g=fwhm_g, fwhm_l=fwhm_l)

    bg = np.linspace(2000, 3000, len(energy))
    intensity = signal + bg

    rng = np.random.default_rng(123)
    intensity += rng.normal(0, noise_level, len(intensity))
    intensity = np.maximum(intensity, 0)

    return Spectrum(
        energy=energy,
        intensity=intensity,
        energy_type="BE",
        element="C1s",
    )


# ------------------------------------------------------------------
# Fitting tests
# ------------------------------------------------------------------

class TestTemplateFitting:
    """Test fit_with_template() on synthetic spectra."""

    def test_si2p_oxide_ke(self):
        """Si2p_oxide template on KE-axis synthetic spectrum."""
        spectrum = _make_synthetic_si2p_ke()
        fitter = AutoFitter()
        result = fitter.fit_with_template(spectrum, "Si2p_oxide")

        assert result.success, f"Fit failed: {result.message}"
        assert result.r_squared > 0.99, f"R²={result.r_squared:.4f} too low"

        # 5 states × 2 SO components = 10 components
        n_comp = len(result.fitting_result.components)
        assert n_comp == 10, f"Expected 10 components, got {n_comp}"

        # Verify template name in metadata
        assert result.fitting_result.metadata.get("template") == "Si2p_oxide"

    def test_si2p_oxide_width_ordering(self):
        """Verify fitted widths are monotonically ordered (Si0 < Si4+)."""
        spectrum = _make_synthetic_si2p_ke(noise_level=50.0)
        fitter = AutoFitter()
        result = fitter.fit_with_template(spectrum, "Si2p_oxide")

        assert result.success

        # Extract sigma (FWHM_G / 2.3548) for main peaks (every other component)
        components = result.fitting_result.components
        # Main peaks are at indices 0, 2, 4, 6, 8 (j+ components)
        main_fwhm_g = [components[i].fwhm_g for i in range(0, 10, 2)]

        # Should be monotonically non-decreasing
        for i in range(len(main_fwhm_g) - 1):
            assert main_fwhm_g[i] <= main_fwhm_g[i + 1] + 0.01, (
                f"Width ordering violated: {main_fwhm_g}"
            )

    def test_c1s_organic_no_so(self):
        """C1s_organic template produces 4 components (no SO)."""
        spectrum = _make_synthetic_c1s_be()
        fitter = AutoFitter(AutoFitConfig(background_type="linear"))
        result = fitter.fit_with_template(spectrum, "C1s_organic")

        assert result.success, f"Fit failed: {result.message}"
        assert result.r_squared > 0.99, f"R²={result.r_squared:.4f} too low"

        n_comp = len(result.fitting_result.components)
        assert n_comp == 4, f"Expected 4 components (no SO), got {n_comp}"

    def test_unknown_template_fallback(self):
        """Unknown template falls back to normal auto-fit."""
        spectrum = _make_synthetic_si2p_ke()
        fitter = AutoFitter()
        result = fitter.fit_with_template(spectrum, "nonexistent_xyz")

        # Should still succeed (fallback to auto-fit)
        assert result.success

    def test_reference_center_override(self):
        """Explicit reference_center skips auto-detection."""
        spectrum = _make_synthetic_si2p_ke()
        fitter = AutoFitter()

        # Provide exact Si0 KE position
        si0_ke = 1486.7 - 99.3
        result = fitter.fit_with_template(
            spectrum, "Si2p_oxide", reference_center=si0_ke,
        )

        assert result.success
        assert result.r_squared > 0.99


# ------------------------------------------------------------------
# Real data test (Si2p ARPES)
# ------------------------------------------------------------------

SI2P_ARPES_PATH = fitting_dir() / "arpes" / "Si2p_arpes.txt"


@pytest.mark.skipif(
    not SI2P_ARPES_PATH.exists(),
    reason="Si2p ARPES test data not available",
)
class TestSi2pArpesRealData:
    """Test template fitting on the real Si2p ARPES data."""

    @staticmethod
    def _load_si2p_arpes() -> Spectrum:
        raw = SI2P_ARPES_PATH.read_text()
        values = re.split(r"[\t\n]+", raw.strip())
        data = np.array([float(v) for v in values]).reshape(-1, 8)
        energy = data[:, 0]
        avg_intensity = np.mean(data[:, 1:], axis=1)
        return Spectrum(
            energy=energy,
            intensity=avg_intensity,
            energy_type="KE",
            element="Si2p",
            metadata={"photon_energy": 126.0},
        )

    def test_si2p_oxide_template(self):
        spectrum = self._load_si2p_arpes()
        fitter = AutoFitter()

        # Si0 peak is at ~26.8 KE in this data
        result = fitter.fit_with_template(
            spectrum, "Si2p_oxide", reference_center=26.80,
        )

        assert result.success, f"Fit failed: {result.message}"
        assert result.r_squared > 0.99, f"R²={result.r_squared:.4f}"
        assert len(result.fitting_result.components) == 10


# ------------------------------------------------------------------
# Auto-template routing tests
# ------------------------------------------------------------------


class TestAutoTemplateRouting:
    """Test that fit() automatically routes to template when applicable."""

    def test_fit_auto_routes_to_template_si2p(self):
        """fit() with element='Si2p' auto-selects Si2p_oxide template."""
        spectrum = _make_synthetic_si2p_ke()
        fitter = AutoFitter()
        result = fitter.fit(spectrum)

        assert result.success
        # Template gives 10 components (5 states × 2 SO)
        assert len(result.fitting_result.components) == 10
        assert result.r_squared > 0.99
        assert result.fitting_result.metadata.get("template") == "Si2p_oxide"

    def test_fit_auto_routes_to_template_c1s(self):
        """fit() with element='C1s' auto-selects C1s_organic template."""
        spectrum = _make_synthetic_c1s_be()
        fitter = AutoFitter(AutoFitConfig(background_type="linear"))
        result = fitter.fit(spectrum)

        assert result.success
        assert len(result.fitting_result.components) == 4
        assert result.fitting_result.metadata.get("template") == "C1s_organic"

    def test_fit_no_element_skips_template(self):
        """fit() with element=None uses peak-detection (no template)."""
        spectrum = _make_synthetic_si2p_ke()
        spectrum = Spectrum(
            energy=spectrum.energy,
            intensity=spectrum.intensity,
            energy_type="KE",
            element=None,
        )
        fitter = AutoFitter()
        result = fitter.fit(spectrum)

        assert result.success
        # Without template, should detect fewer than 10 components
        assert "template" not in (result.fitting_result.metadata or {})

    def test_fit_unknown_element_skips_template(self):
        """fit() with unknown element falls back to peak-detection."""
        spectrum = _make_synthetic_si2p_ke()
        spectrum = Spectrum(
            energy=spectrum.energy,
            intensity=spectrum.intensity,
            energy_type="KE",
            element="Zr3d",
        )
        fitter = AutoFitter()
        result = fitter.fit(spectrum)

        assert result.success
        assert "template" not in (result.fitting_result.metadata or {})

    def test_fit_use_template_false(self):
        """fit(use_template=False) skips template even for known element."""
        spectrum = _make_synthetic_si2p_ke()
        fitter = AutoFitter()
        result = fitter.fit(spectrum, use_template=False)

        assert result.success
        # Without template, fewer than 10 components
        assert len(result.fitting_result.components) < 10
        assert "template" not in (result.fitting_result.metadata or {})


# ------------------------------------------------------------------
# extract_peak_config helper tests
# ------------------------------------------------------------------

class TestExtractPeakConfig:
    """Tests for the extract_peak_config shared helper."""

    def test_extract_from_successful_result(self):
        """extract_peak_config returns dict with centers, sigmas, gamma."""
        from toyomacro.fitting import extract_peak_config

        spectrum = _make_synthetic_si2p_ke()
        result = AutoFitter().fit(spectrum)
        assert result.success

        config = extract_peak_config(result)
        assert config is not None
        assert 'centers' in config
        assert 'sigmas' in config
        assert 'gamma' in config
        assert len(config['centers']) == len(result.fitting_result.components)
        assert len(config['sigmas']) == len(result.fitting_result.components)
        assert config['gamma'] > 0
        # sigmas = fwhm_g / 2.355
        for i, comp in enumerate(result.fitting_result.components):
            np.testing.assert_allclose(
                config['sigmas'][i], comp.fwhm_g / 2.355, rtol=1e-10
            )

    def test_extract_from_failed_result(self):
        """extract_peak_config returns None for failed result."""
        from toyomacro.core.fitting_result import FittingResult
        from toyomacro.fitting import AutoFitResult, extract_peak_config

        z = np.zeros(10)
        failed = AutoFitResult(
            success=False,
            fitting_result=FittingResult(
                components=[], background=z,
                energy=np.linspace(0, 10, 10),
                fitted_intensity=z,
                background_curve=z,
                residuals=z,
            ),
            detected_peaks=[],
            r_squared=0.0,
        )
        assert extract_peak_config(failed) is None
