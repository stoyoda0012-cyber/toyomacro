"""Tests pinning the statistical meaning of the noise-severity `level`.

The generator's ``level`` is a dimensionless severity parameter, not a
Poisson mean.  These tests verify the documented conversions

    SNR_peak    = 1e4 / level
    lambda_peak = (1e4 / level)^2

both as pure functions and empirically against the actual sampler, so
the paper's Figure 2 axis (peak-count SNR) is guaranteed to describe
what the implementation does.
"""

import numpy as np
import pytest

from toyomacro.voigtfit.spectra_generator import (
    NOISE_LEVELS,
    NOISE_SCALE,
    add_poisson_noise,
    level_to_peak_lambda,
    level_to_peak_snr,
)


class TestConversionFunctions:
    def test_scale_constant(self):
        assert NOISE_SCALE == 1e4

    @pytest.mark.parametrize('level,snr', [
        (1e0, 1e4), (1e1, 1e3), (1e2, 1e2), (1e3, 1e1),
        (1e4, 1.0), (1e5, 1e-1),
    ])
    def test_level_to_peak_snr(self, level, snr):
        assert level_to_peak_snr(level) == pytest.approx(snr)

    @pytest.mark.parametrize('level', [1e0, 1e2, 1e3, 1e4, 1e5])
    def test_lambda_is_snr_squared(self, level):
        snr = level_to_peak_snr(level)
        assert level_to_peak_lambda(level) == pytest.approx(snr * snr)

    def test_zero_level_is_noise_free(self):
        assert level_to_peak_snr(0) == float('inf')
        assert level_to_peak_lambda(0) == float('inf')

    def test_named_levels_cover_collapse_point(self):
        # 'Strong' is the documented collapse point: peak SNR = 1.
        assert level_to_peak_snr(NOISE_LEVELS['Strong']) == pytest.approx(1.0)
        assert level_to_peak_lambda(NOISE_LEVELS['Strong']) == pytest.approx(1.0)


class TestSamplerMatchesConversion:
    """Empirical moments of add_poisson_noise must match the formulas."""

    N = 200_000

    def _empirical_snr(self, level: float, **kwargs) -> float:
        data = np.full((self.N, 1), 1000.0, dtype=np.float32)
        noisy = add_poisson_noise(data, level, **kwargs)
        return float(noisy.mean() / noisy.std())

    def test_exact_poisson_snr_moderate(self):
        # level=1e3 -> lambda_peak=100, SNR_peak=10 (exact-Poisson path)
        snr = self._empirical_snr(1e3, use_gaussian_approx=False)
        assert snr == pytest.approx(level_to_peak_snr(1e3), rel=0.05)

    def test_exact_poisson_snr_strong(self):
        # level=1e4 -> lambda_peak=1, SNR_peak=1: the collapse point
        snr = self._empirical_snr(1e4, use_gaussian_approx=False)
        assert snr == pytest.approx(1.0, rel=0.05)

    def test_gaussian_approx_agrees_with_exact(self):
        # In the approximation regime (lambda > 20) both paths must give
        # the same first two moments within sampling error.
        snr_exact = self._empirical_snr(1e2, use_gaussian_approx=False)
        snr_approx = self._empirical_snr(1e2, use_gaussian_approx=True)
        assert snr_exact == pytest.approx(level_to_peak_snr(1e2), rel=0.05)
        assert snr_approx == pytest.approx(snr_exact, rel=0.05)

    def test_mean_is_preserved(self):
        data = np.full((self.N, 1), 1000.0, dtype=np.float32)
        noisy = add_poisson_noise(data, 1e3, use_gaussian_approx=False)
        assert float(noisy.mean()) == pytest.approx(1000.0, rel=0.02)

    def test_heteroscedastic_shot_noise_scaling(self):
        # A pixel at half the normalization max receives lambda_peak/2,
        # so its SNR is sqrt(lambda_peak/2) — shot-noise scaling.
        level = 1e3
        lam_peak = level_to_peak_lambda(level)
        data = np.full((self.N, 1), 500.0, dtype=np.float32)
        noisy = add_poisson_noise(
            data, level, use_gaussian_approx=False, global_max=1000.0)
        snr = float(noisy.mean() / noisy.std())
        assert snr == pytest.approx(np.sqrt(lam_peak / 2), rel=0.05)
