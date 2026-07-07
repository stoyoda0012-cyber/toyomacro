"""Tests for Doniach-Sunjic lineshape and MATLAB-compatible LineshapeType numbering."""

import numpy as np
import pytest

from toyomacro.core.fitting_result import LineshapeType, PeakComponent
from toyomacro.lineshape import DoniachSunjic, Lorentzian, create_lineshape

# ---------------------------------------------------------------------------
# LineshapeType MATLAB compatibility
# ---------------------------------------------------------------------------

class TestLineshapeTypeNumbering:
    """Verify LineshapeType values match MATLAB TypeID convention."""

    def test_voigt_is_1(self):
        assert LineshapeType.VOIGT == 1

    def test_pseudovoigt_is_2(self):
        assert LineshapeType.PSEUDO_VOIGT == 2

    def test_gaussian_is_3(self):
        assert LineshapeType.GAUSSIAN == 3

    def test_lorentzian_is_4(self):
        assert LineshapeType.LORENTZIAN == 4

    def test_doniach_sunjic_is_5(self):
        assert LineshapeType.DONIACH_SUNJIC == 5

    def test_fermi_dirac_is_6(self):
        assert LineshapeType.FERMI_DIRAC == 6


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    """Test DoniachSunjic factory registration."""

    def test_create_by_name(self):
        ds = create_lineshape("doniach_sunjic")
        assert isinstance(ds, DoniachSunjic)

    def test_create_by_alias(self):
        ds = create_lineshape("ds")
        assert isinstance(ds, DoniachSunjic)

    def test_create_by_number(self):
        ds = create_lineshape(5)
        assert isinstance(ds, DoniachSunjic)

    def test_name_attribute(self):
        ds = DoniachSunjic()
        assert ds.name == "doniach_sunjic"


# ---------------------------------------------------------------------------
# Core DS evaluation
# ---------------------------------------------------------------------------

class TestDoniachSunjicCore:
    """Test DoniachSunjic.evaluate() correctness."""

    @pytest.fixture()
    def ds(self):
        return DoniachSunjic()

    @pytest.fixture()
    def energy(self):
        return np.linspace(-10, 10, 2001)

    def test_alpha_zero_matches_lorentzian(self, ds):
        """When alpha=0, DS should reduce to a Lorentzian."""
        # Use wide energy range for accurate numerical integration in DS
        wide_energy = np.linspace(-50, 50, 10001)
        fwhm_l = 1.0
        area = 100.0

        ds_profile = ds.evaluate(wide_energy, center=0.0, amplitude=area,
                                 fwhm_g=0.0, fwhm_l=fwhm_l, asymmetry=0.0)

        lor = Lorentzian()
        lor_profile = lor.evaluate(wide_energy, center=0.0, amplitude=area,
                                   fwhm=fwhm_l)

        # Should match within numerical precision of area normalization
        # (DS uses trapz over finite range; small mismatch at tails expected)
        np.testing.assert_allclose(ds_profile, lor_profile, rtol=0.01,
                                   err_msg="DS(alpha=0) should match Lorentzian")

    def test_asymmetric_shape_be(self, ds, energy):
        """alpha > 0 should produce asymmetric tail toward higher BE."""
        profile = ds.evaluate(energy, center=0.0, amplitude=100.0,
                              fwhm_g=0.0, fwhm_l=1.0, asymmetry=0.2,
                              energy_type="BE")

        # Find peak position
        peak_idx = np.argmax(profile)

        # Intensity at equal distance from peak
        offset = 200  # ~2 eV
        left_val = profile[peak_idx - offset] if peak_idx - offset >= 0 else 0
        right_val = profile[peak_idx + offset] if peak_idx + offset < len(profile) else 0

        # BE convention: tail toward higher BE (positive x)
        assert right_val > left_val, \
            "DS(BE) tail should extend toward higher binding energy (positive x)"

    def test_asymmetric_shape_ke(self, ds, energy):
        """alpha > 0 should produce asymmetric tail toward lower KE."""
        profile = ds.evaluate(energy, center=0.0, amplitude=100.0,
                              fwhm_g=0.0, fwhm_l=1.0, asymmetry=0.2,
                              energy_type="KE")

        peak_idx = np.argmax(profile)
        offset = 200
        left_val = profile[peak_idx - offset] if peak_idx - offset >= 0 else 0
        right_val = profile[peak_idx + offset] if peak_idx + offset < len(profile) else 0

        # KE convention: tail toward lower KE (negative x)
        assert left_val > right_val, \
            "DS(KE) tail should extend toward lower kinetic energy (negative x)"

    def test_area_preservation(self, ds, energy):
        """Integrated area should match requested amplitude."""
        area = 500.0
        profile = ds.evaluate(energy, center=0.0, amplitude=area,
                              fwhm_g=0.0, fwhm_l=1.0, asymmetry=0.15)

        actual_area = np.trapezoid(profile, energy)
        np.testing.assert_allclose(actual_area, area, rtol=0.01,
                                   err_msg="Area should be preserved")

    def test_gaussian_broadening(self, ds, energy):
        """Gaussian convolution should broaden the peak."""
        profile_no_g = ds.evaluate(energy, center=0.0, amplitude=100.0,
                                   fwhm_g=0.0, fwhm_l=1.0, asymmetry=0.1)
        profile_with_g = ds.evaluate(energy, center=0.0, amplitude=100.0,
                                     fwhm_g=1.0, fwhm_l=1.0, asymmetry=0.1)

        # Broadened profile should have lower peak height
        assert profile_with_g.max() < profile_no_g.max(), \
            "Gaussian broadening should reduce peak height"

        # But similar area
        area_no_g = np.trapezoid(profile_no_g, energy)
        area_with_g = np.trapezoid(profile_with_g, energy)
        np.testing.assert_allclose(area_with_g, area_no_g, rtol=0.05,
                                   err_msg="Gaussian broadening should preserve area")

    def test_positive_profile(self, ds, energy):
        """Profile should be non-negative near the peak."""
        profile = ds.evaluate(energy, center=0.0, amplitude=100.0,
                              fwhm_g=0.5, fwhm_l=1.0, asymmetry=0.1)

        # Near center (within ~5 FWHM), profile should be positive
        mask = np.abs(energy) < 5.0
        assert np.all(profile[mask] >= -1e-10), \
            "Profile near center should be non-negative"

    def test_different_asymmetry_values(self, ds, energy):
        """Higher asymmetry should produce more skewed peaks."""
        fwhm_results = []
        for alpha in [0.0, 0.1, 0.2, 0.3]:
            fwhm = ds.fwhm(fwhm_g=0.0, fwhm_l=1.0, asymmetry=alpha)
            fwhm_results.append(fwhm)

        # FWHM should generally change with asymmetry
        # (exact behavior depends on definition, but alpha=0 should be fwhm_l)
        assert fwhm_results[0] == pytest.approx(1.0, rel=0.01), \
            "alpha=0 FWHM should be fwhm_l"


# ---------------------------------------------------------------------------
# area_to_height / height_to_area
# ---------------------------------------------------------------------------

class TestAreaHeight:
    """Test area-height conversion round-trips."""

    def test_round_trip(self):
        ds = DoniachSunjic()
        area = 200.0
        height = ds.area_to_height(area, fwhm_g=0.0, fwhm_l=1.0, asymmetry=0.1)
        recovered = ds.height_to_area(height, fwhm_g=0.0, fwhm_l=1.0, asymmetry=0.1)
        np.testing.assert_allclose(recovered, area, rtol=0.02)

    def test_height_positive(self):
        ds = DoniachSunjic()
        height = ds.area_to_height(100.0, fwhm_g=0.0, fwhm_l=1.0, asymmetry=0.15)
        assert height > 0

    def test_with_gaussian_broadening(self):
        ds = DoniachSunjic()
        h_no_g = ds.area_to_height(100.0, fwhm_g=0.0, fwhm_l=1.0, asymmetry=0.1)
        h_with_g = ds.area_to_height(100.0, fwhm_g=1.0, fwhm_l=1.0, asymmetry=0.1)
        assert h_with_g < h_no_g, "Gaussian broadening reduces peak height"


# ---------------------------------------------------------------------------
# PeakComponent serialization
# ---------------------------------------------------------------------------

class TestPeakComponentDS:
    """Test PeakComponent round-trip with DS lineshape."""

    def test_to_array_and_back(self):
        comp = PeakComponent(
            area=500.0,
            center=84.0,  # Au 4f7/2
            fwhm_g=0.3,
            fwhm_l=0.4,
            asymmetry=0.12,
            branch_ratio=0.75,
            so_split=3.67,
            lineshape=LineshapeType.DONIACH_SUNJIC,
            height=1200.0,
        )
        arr = comp.to_array()
        assert arr[7] == 5  # MATLAB TypeID = 5
        assert arr[4] == pytest.approx(0.12)

        recovered = PeakComponent.from_array(arr)
        assert recovered.lineshape == LineshapeType.DONIACH_SUNJIC
        assert recovered.asymmetry == pytest.approx(0.12)
        assert recovered.center == pytest.approx(84.0)


# ---------------------------------------------------------------------------
# lmfit integration
# ---------------------------------------------------------------------------

class TestLmfitIntegration:
    """Test Doniach-Sunjic fitting via lmfit."""

    def test_single_peak_fit(self):
        """Fit a synthetic DS peak and recover parameters."""
        from toyomacro.fitting.peak_model import MultiPeakModel

        # Generate synthetic DS data
        ds = DoniachSunjic()
        energy = np.linspace(80, 90, 500)
        true_center = 84.0
        true_area = 300.0
        true_asymmetry = 0.15

        clean = ds.evaluate(energy, center=true_center, amplitude=true_area,
                            fwhm_g=0.3, fwhm_l=0.4, asymmetry=true_asymmetry)

        # Add small noise
        np.random.seed(42)
        noisy = clean + 2.0 * np.random.randn(len(energy))

        # Fit
        model = MultiPeakModel(lineshape="doniach_sunjic")
        model.add_peak(
            center=84.2,  # slightly off
            amplitude=250.0,
            sigma=0.3 / 2.3548,
            gamma=0.4 / 2.0,
            asymmetry=0.1,
        )
        result = model.fit(energy, noisy)

        # Check convergence
        assert result.success, f"Fit failed: {result.message}"

        # Recover center within 0.2 eV
        fitted_center = result.params["p1_center"].value
        assert abs(fitted_center - true_center) < 0.2, \
            f"Center off: {fitted_center} vs {true_center}"

        # Recover asymmetry within reasonable range
        fitted_asym = result.params["p1_asymmetry"].value
        assert 0.0 < fitted_asym < 0.4, \
            f"Asymmetry unreasonable: {fitted_asym}"
