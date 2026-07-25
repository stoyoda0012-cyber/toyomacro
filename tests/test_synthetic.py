"""Tests for voigtfit synthetic data generation."""

import numpy as np
import pytest

from toyomacro.voigtfit.synthetic import (
    ELEMENT_PROFILES,
    make_synthetic_configs,
    make_synthetic_spectra,
)

# ── TestSyntheticConfigs ──────────────────────────────────────────────────


class TestSyntheticConfigs:
    """Test make_synthetic_configs."""

    def test_si2p_5comp_separation(self):
        """Si2p n_comp=5: ΔE ≥ 3.5σ between all neighbours."""
        profile = ELEMENT_PROFILES["Si2p"]
        configs, amps = make_synthetic_configs(profile, n_comp=5)
        assert len(configs) == 5
        assert amps.shape == (5,)
        assert amps.dtype == np.float32
        for i in range(len(configs) - 1):
            dE = configs[i + 1].center - configs[i].center
            assert dE >= 3.5 * profile.sigma - 1e-9, (
                f"Components {i}-{i + 1}: ΔE={dE:.4f} < 3.5σ={3.5 * profile.sigma:.4f}"
            )

    def test_au4f_2comp_doublet(self):
        """Au4f n_comp=2: is_doublet=True, so_split=3.70, branch_ratio=4/3."""
        profile = ELEMENT_PROFILES["Au4f"]
        configs, amps = make_synthetic_configs(profile, n_comp=2)
        assert len(configs) == 2
        for cc in configs:
            assert cc.is_doublet is True
            assert cc.so_split == pytest.approx(3.70)
            assert cc.branch_ratio == pytest.approx(4 / 3)

    def test_singlet_no_doublet(self):
        """O1s / C1s are singlets (no SO splitting)."""
        for name in ("O1s", "C1s"):
            profile = ELEMENT_PROFILES[name]
            configs, _ = make_synthetic_configs(profile, n_comp=2)
            for cc in configs:
                assert cc.is_doublet is False
                assert cc.so_split == 0.0

    def test_amp_distributions(self):
        """All amp_distribution values produce valid ranges."""
        profile = ELEMENT_PROFILES["Si2p"]
        for dist, (lo, hi) in [
            ("uniform", (0.8, 1.2)),
            ("moderate", (0.3, 1.0)),
            ("random", (0.05, 1.0)),
        ]:
            _, amps = make_synthetic_configs(
                profile, n_comp=10, amp_distribution=dist
            )
            assert amps.min() >= lo - 1e-6
            assert amps.max() <= hi + 1e-6

    def test_invalid_distribution(self):
        """Invalid amp_distribution raises ValueError."""
        profile = ELEMENT_PROFILES["Si2p"]
        with pytest.raises(ValueError, match="Unknown"):
            make_synthetic_configs(profile, n_comp=2, amp_distribution="bad")

    def test_single_component(self):
        """n_comp=1: dE_range stays at default 2.0."""
        profile = ELEMENT_PROFILES["Si2p"]
        configs, amps = make_synthetic_configs(profile, n_comp=1)
        assert len(configs) == 1
        assert configs[0].dE_range == 2.0

    def test_dE_range_constrained(self):
        """n_comp>1: dE_range is constrained to prevent cross-assignment."""
        profile = ELEMENT_PROFILES["Si2p"]
        configs, _ = make_synthetic_configs(profile, n_comp=3)
        spacing = max(3.5 * profile.sigma, profile.typical_dE)
        expected_dE = min(2.0, max(0.1, spacing / 2 - profile.sigma / 2))
        for cc in configs:
            assert cc.dE_range == pytest.approx(expected_dE)


# ── TestSyntheticSpectra ──────────────────────────────────────────────────


class TestSyntheticSpectra:
    """Test make_synthetic_spectra."""

    def test_shape_and_dtype(self):
        """Output shapes and dtypes are correct."""
        profile = ELEMENT_PROFILES["Si2p"]
        configs, _ = make_synthetic_configs(profile, n_comp=3)
        spectra, true_amps = make_synthetic_spectra(
            configs, profile, n_spectra=100
        )
        assert spectra.shape == (100, 128)
        assert true_amps.shape == (100, 3)
        assert spectra.dtype == np.float32
        assert true_amps.dtype == np.float32

    def test_snr(self):
        """Measured SNR matches target within ±3 dB."""
        profile = ELEMENT_PROFILES["O1s"]
        configs, _ = make_synthetic_configs(profile, n_comp=2)
        target_snr = 20.0
        # Clean reference (snr_db >= 100 → noiseless)
        spectra_clean, _ = make_synthetic_spectra(
            configs, profile, n_spectra=500, snr_db=200.0
        )
        spectra_noisy, _ = make_synthetic_spectra(
            configs, profile, n_spectra=500, snr_db=target_snr
        )
        # Same seed → same amps; difference = noise
        noise = spectra_noisy.astype(np.float64) - spectra_clean.astype(np.float64)
        peaks = spectra_clean.astype(np.float64).max(axis=1)
        noise_stds = noise.std(axis=1)
        measured_snr_db = 20.0 * np.log10(peaks / (noise_stds + 1e-30))
        median_snr = float(np.median(measured_snr_db))
        assert abs(median_snr - target_snr) < 3.0, (
            f"Median SNR {median_snr:.1f} dB != {target_snr} dB"
        )

    def test_custom_energy_range(self):
        """Custom E_range and n_channels are respected."""
        profile = ELEMENT_PROFILES["C1s"]
        configs, _ = make_synthetic_configs(profile, n_comp=1)
        spectra, _ = make_synthetic_spectra(
            configs, profile, n_spectra=10,
            E_range=(90.0, 110.0), n_channels=200,
        )
        assert spectra.shape == (10, 200)

    def test_doublet_spectra(self):
        """Au4f doublet produces valid spectra."""
        profile = ELEMENT_PROFILES["Au4f"]
        configs, _ = make_synthetic_configs(profile, n_comp=2)
        spectra, amps = make_synthetic_spectra(
            configs, profile, n_spectra=50
        )
        assert spectra.shape == (50, 128)
        assert amps.shape == (50, 2)
        assert np.all(spectra.max(axis=1) > 0)

    def test_no_nan_inf(self):
        """No NaN or Inf in generated spectra."""
        profile = ELEMENT_PROFILES["Si2p"]
        configs, _ = make_synthetic_configs(profile, n_comp=5)
        spectra, amps = make_synthetic_spectra(
            configs, profile, n_spectra=100, snr_db=20.0
        )
        assert np.all(np.isfinite(spectra))
        assert np.all(np.isfinite(amps))

    def test_noiseless(self):
        """snr_db >= 100 produces noiseless spectra."""
        profile = ELEMENT_PROFILES["Si2p"]
        configs, _ = make_synthetic_configs(profile, n_comp=2)
        s1, a1 = make_synthetic_spectra(
            configs, profile, n_spectra=10, snr_db=100.0
        )
        s2, a2 = make_synthetic_spectra(
            configs, profile, n_spectra=10, snr_db=200.0
        )
        # Both noiseless with same seed → identical
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(a1, a2)

    def test_positive_spectra(self):
        """Noiseless spectra are strictly non-negative."""
        profile = ELEMENT_PROFILES["Si2p"]
        configs, _ = make_synthetic_configs(profile, n_comp=3)
        spectra, _ = make_synthetic_spectra(
            configs, profile, n_spectra=50, snr_db=200.0
        )
        assert np.all(spectra >= -1e-7)


# ── Smoke tests ───────────────────────────────────────────────────────────


class TestSmokeAllProfiles:
    """Smoke test: all ELEMENT_PROFILES × n_comp=1,2,3."""

    @pytest.mark.parametrize("name", list(ELEMENT_PROFILES.keys()))
    @pytest.mark.parametrize("n_comp", [1, 2, 3])
    def test_smoke(self, name: str, n_comp: int):
        profile = ELEMENT_PROFILES[name]
        configs, amps = make_synthetic_configs(profile, n_comp=n_comp)
        assert len(configs) == n_comp
        assert amps.shape == (n_comp,)
        spectra, true_amps = make_synthetic_spectra(
            configs, profile, n_spectra=10
        )
        assert spectra.shape == (10, 128)
        assert true_amps.shape == (10, n_comp)
        assert np.all(np.isfinite(spectra))
