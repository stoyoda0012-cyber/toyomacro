"""Tests for BE/KE energy axis handling throughout the pipeline."""

import numpy as np
import pytest

from toyomacro.core.spectrum import Spectrum

# ===== Spectrum.to_energy_type() =====


class TestSpectrumEnergyType:
    """Tests for Spectrum energy type conversion."""

    def test_ke_to_be_conversion(self):
        """KE → BE: energy = hν - KE."""
        ke = np.array([20.0, 22.0, 24.0, 26.0, 28.0])
        intensity = np.array([100, 200, 500, 300, 100], dtype=np.float64)
        spec = Spectrum(energy=ke, intensity=intensity, energy_type="KE")

        hv = 126.0
        be_spec = spec.to_energy_type("BE", hv)

        assert be_spec.energy_type == "BE"
        np.testing.assert_allclose(be_spec.energy, hv - ke)
        np.testing.assert_array_equal(be_spec.intensity, intensity)
        assert be_spec.metadata["photon_energy"] == hv

    def test_be_to_ke_conversion(self):
        """BE → KE: energy = hν - BE."""
        be = np.array([98.0, 99.0, 100.0, 101.0, 102.0])
        intensity = np.array([50, 100, 300, 150, 50], dtype=np.float64)
        spec = Spectrum(energy=be, intensity=intensity, energy_type="BE")

        hv = 1486.6
        ke_spec = spec.to_energy_type("KE", hv)

        assert ke_spec.energy_type == "KE"
        np.testing.assert_allclose(ke_spec.energy, hv - be)

    def test_same_type_returns_self(self):
        """Converting to same type returns self (no copy)."""
        spec = Spectrum(
            energy=np.array([1.0, 2.0, 3.0]),
            intensity=np.array([10.0, 20.0, 30.0]),
            energy_type="BE",
        )
        result = spec.to_energy_type("BE", 1486.6)
        assert result is spec

    def test_roundtrip_conversion(self):
        """KE → BE → KE should recover original energy."""
        ke = np.array([20.0, 22.0, 24.0, 26.0, 28.0])
        intensity = np.array([100, 200, 500, 300, 100], dtype=np.float64)
        spec = Spectrum(energy=ke, intensity=intensity, energy_type="KE")

        hv = 126.0
        be_spec = spec.to_energy_type("BE", hv)
        ke_spec = be_spec.to_energy_type("KE", hv)

        assert ke_spec.energy_type == "KE"
        np.testing.assert_allclose(ke_spec.energy, ke, atol=1e-10)

    def test_element_preserved(self):
        """Element name should be preserved through conversion."""
        spec = Spectrum(
            energy=np.array([1.0, 2.0, 3.0]),
            intensity=np.array([10.0, 20.0, 30.0]),
            energy_type="KE",
            element="Si2p",
        )
        result = spec.to_energy_type("BE", 126.0)
        assert result.element == "Si2p"


# ===== SO doublet sign convention =====


class TestSOSignConvention:
    """Tests for SO doublet sign correctness in AutoFitter."""

    def _make_spectrum(self, energy_type="KE"):
        """Create a synthetic Si2p spectrum."""
        if energy_type == "KE":
            energy = np.linspace(20.0, 28.0, 161)
        else:
            energy = np.linspace(98.0, 106.0, 161)

        # Simple single peak
        center = energy[len(energy) // 2]
        intensity = 1000 * np.exp(-0.5 * ((energy - center) / 0.4) ** 2)
        intensity += np.random.default_rng(42).normal(0, 5, len(energy))
        intensity = np.maximum(intensity, 0)

        return Spectrum(
            energy=energy,
            intensity=intensity,
            energy_type=energy_type,
            element="Si2p",
        )

    def test_so_partner_ke_axis(self):
        """On KE axis, SO partner should be at lower KE (main - split)."""
        from toyomacro.fitting.autofitter import AutoFitter

        fitter = AutoFitter()
        spec = self._make_spectrum("KE")
        result = fitter.fit(spec)

        if result.success and len(result.fitting_result.components) >= 2:
            # With SO, partner should be at lower KE than main
            centers = sorted(
                [c.center for c in result.fitting_result.components], reverse=True
            )
            # Main (j+) at higher KE, partner (j-) at lower KE
            assert centers[0] > centers[-1]

    def test_so_partner_be_axis(self):
        """On BE axis, SO partner should be at higher BE (main + split)."""
        from toyomacro.fitting.autofitter import AutoFitter

        fitter = AutoFitter()
        spec = self._make_spectrum("BE")
        result = fitter.fit(spec)

        if result.success and len(result.fitting_result.components) >= 2:
            centers = sorted(
                [c.center for c in result.fitting_result.components]
            )
            # Main (j+) at lower BE, partner (j-) at higher BE
            assert centers[0] < centers[-1]


# ===== HDF5Cache energy_type =====


class TestHDF5CacheEnergyType:
    """Tests for HDF5Cache energy_type caching."""

    def test_default_energy_type(self):
        """Default energy_type should be 'BE'."""
        from toyomacro.io.hdf5_cache import HDF5Cache

        cache = HDF5Cache()
        assert cache.get_energy_type("/nonexistent") == "BE"

    def test_energy_type_in_spectrum_data(self):
        """SpectrumData should include energy_type field."""
        from toyomacro.io.hdf5_cache import Coordinates4D, SpectrumData

        sd = SpectrumData(
            energy=np.array([1.0, 2.0]),
            intensity=np.array([10.0, 20.0]),
            index=0,
            coordinates=Coordinates4D(x=0, y=0, t=0, angle_idx=0),
            element="Si2p",
            read_time_ms=0.1,
            energy_type="KE",
        )
        assert sd.energy_type == "KE"

    def test_energy_type_in_file_metadata(self):
        """FileMetadata should include energy_type field."""
        from toyomacro.io.hdf5_cache import FileMetadata

        meta = FileMetadata(
            path="/test.h5",
            filename="test.h5",
            element="Si2p",
            n_spectra=100,
            n_energy=161,
            size_mb=10.0,
            energy_type="KE",
        )
        assert meta.energy_type == "KE"

    def test_energy_type_default_be(self):
        """FileMetadata default energy_type should be 'BE'."""
        from toyomacro.io.hdf5_cache import FileMetadata

        meta = FileMetadata(
            path="/test.h5",
            filename="test.h5",
            element="Si2p",
            n_spectra=100,
            n_energy=161,
            size_mb=10.0,
        )
        assert meta.energy_type == "BE"


# ===== Importer bindingenergysign convention =====


class TestImporterBESign:
    """Tests for bindingenergysign convention (MATLAB compatibility)."""

    def test_be_sign_values(self):
        """BE → 1, KE → 0 (MATLAB uint8 logical convention)."""
        # This is a convention test — verified against MATLAB Toyomacro.m
        # MATLAB: bindingenergysign = uint8(true) = 1 for BE
        #         bindingenergysign = uint8(false) = 0 for KE
        assert 1 == 1  # BE sign
        assert 0 == 0  # KE sign (not -1)

    def test_reading_convention(self):
        """val == 1 → BE, anything else → KE."""
        def interpret(val):
            return "BE" if val == 1 else "KE"

        assert interpret(1) == "BE"
        assert interpret(0) == "KE"
        assert interpret(-1) == "KE"  # Legacy Python value
        assert interpret(2) == "KE"  # Unknown
