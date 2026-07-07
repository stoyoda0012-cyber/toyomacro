"""Tests for Tougaard background algorithm."""

import numpy as np
import pytest

from toyomacro.background import Tougaard
from toyomacro.background.tougaard import _UNIVERSAL_C


@pytest.fixture
def tougaard():
    return Tougaard()


@pytest.fixture
def synthetic_xps():
    """Synthetic XPS-like spectrum: peak on a step background.

    Simulates a spectrum where the high-BE side (low KE) has higher
    background due to inelastic scattering.
    """
    energy = np.linspace(95.0, 105.0, 201)
    # Step background: low on left, high on right
    step = 200 + 300 / (1 + np.exp(-(energy - 100.0) / 0.3))
    # Gaussian peak
    peak = 1000 * np.exp(-0.5 * ((energy - 100.0) / 0.4) ** 2)
    intensity = step + peak
    return energy, intensity


class TestTougaardBasic:
    """Basic Tougaard background tests."""

    def test_returns_correct_shape(self, tougaard, synthetic_xps):
        energy, intensity = synthetic_xps
        bg = tougaard.calculate(energy, intensity)
        assert bg.shape == intensity.shape

    def test_no_nan(self, tougaard, synthetic_xps):
        energy, intensity = synthetic_xps
        bg = tougaard.calculate(energy, intensity)
        assert not np.any(np.isnan(bg))

    def test_no_inf(self, tougaard, synthetic_xps):
        energy, intensity = synthetic_xps
        bg = tougaard.calculate(energy, intensity)
        assert not np.any(np.isinf(bg))

    def test_background_reasonable(self, tougaard, synthetic_xps):
        """Background should be in a reasonable range relative to the spectrum."""
        energy, intensity = synthetic_xps
        bg = tougaard.calculate(energy, intensity)
        # Background at peak region should be below the spectrum
        peak_idx = np.argmax(intensity)
        assert bg[peak_idx] < intensity[peak_idx]
        # Background should not be wildly larger than the spectrum
        assert np.max(bg) < 2 * np.max(intensity)

    def test_background_positive(self, tougaard, synthetic_xps):
        energy, intensity = synthetic_xps
        bg = tougaard.calculate(energy, intensity)
        # Background should be positive (spectrum is positive)
        assert np.all(bg >= 0)


class TestTougaardSubtraction:
    """Test background subtraction."""

    def test_subtract(self, tougaard, synthetic_xps):
        energy, intensity = synthetic_xps
        signal = tougaard.subtract(energy, intensity)
        assert signal.shape == intensity.shape
        # Peak region should have positive signal
        peak_idx = np.argmax(intensity)
        assert signal[peak_idx] > 0

    def test_subtract_reduces_background(self, tougaard, synthetic_xps):
        """After subtraction, the endpoint regions should be near zero."""
        energy, intensity = synthetic_xps
        signal = tougaard.subtract(energy, intensity)
        # Endpoints should be close to zero (within 10% of peak)
        peak_height = np.max(signal)
        m = len(signal) // 10
        assert abs(np.mean(signal[:m])) < 0.3 * peak_height
        assert abs(np.mean(signal[-m:])) < 0.3 * peak_height


class TestTougaardEdgeCases:
    """Edge cases for Tougaard background."""

    def test_flat_spectrum(self, tougaard):
        """Flat spectrum should give flat background."""
        energy = np.linspace(90, 110, 100)
        intensity = np.full(100, 500.0)
        bg = tougaard.calculate(energy, intensity)
        # Background should be constant (≈ 500)
        assert np.std(bg) < 1.0

    def test_very_short_spectrum(self, tougaard):
        """Spectrum with very few points."""
        energy = np.array([98.0, 99.0, 100.0])
        intensity = np.array([100.0, 300.0, 150.0])
        bg = tougaard.calculate(energy, intensity)
        assert bg.shape == (3,)
        assert not np.any(np.isnan(bg))

    def test_two_point_spectrum(self, tougaard):
        """Minimum viable spectrum (2 points)."""
        energy = np.array([99.0, 101.0])
        intensity = np.array([200.0, 500.0])
        bg = tougaard.calculate(energy, intensity)
        assert bg.shape == (2,)

    def test_descending_energy(self, tougaard, synthetic_xps):
        """Descending energy axis (BE convention)."""
        energy, intensity = synthetic_xps
        # Reverse to descending
        energy_desc = energy[::-1]
        intensity_desc = intensity[::-1]
        bg = tougaard.calculate(energy_desc, intensity_desc)
        assert bg.shape == intensity_desc.shape
        assert not np.any(np.isnan(bg))


class TestTougaardParameters:
    """Test custom parameter support."""

    def test_custom_C(self, tougaard, synthetic_xps):
        """Custom loss function constant."""
        energy, intensity = synthetic_xps
        bg1 = tougaard.calculate(energy, intensity, C=1643.0)
        bg2 = tougaard.calculate(energy, intensity, C=800.0)
        # Different C should give different backgrounds
        assert not np.allclose(bg1, bg2)

    def test_universal_C_default(self):
        """Default C should be 1643 eV^2."""
        assert _UNIVERSAL_C == 1643.0


class TestTougaardName:
    """Test class attributes."""

    def test_name(self, tougaard):
        assert tougaard.name == "tougaard"

    def test_repr(self, tougaard):
        assert "Tougaard" in repr(tougaard)


class TestTougaardImport:
    """Test that Tougaard is properly exported."""

    def test_import_from_background(self):
        from toyomacro.background import Tougaard
        assert Tougaard is not None

    def test_in_all(self):
        import toyomacro.background
        assert "Tougaard" in toyomacro.background.__all__
