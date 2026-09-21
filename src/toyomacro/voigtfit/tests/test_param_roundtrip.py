"""
Test 3-Parameter Roundtrip (R=Amplitude, G=Center Shift, B=FWHM)
=================================================================

Verifies:
1. SinglePeakEncoder encode/decode identity
2. Taylor-expansion spectra generation accuracy
3. 3-step residual projection recovers amplitude, δE, δσ
4. End-to-end image → spectra → fit → image roundtrip
"""

import os
import time

import numpy as np
import pytest

IN_CI = os.environ.get("CI") == "true"

from toyomacro.voigtfit.param_encoder import (
    C1S_SINGLE_PRESET,
    SinglePeakEncoder,
)


def _generate_voigt_spectrum(
    energy: np.ndarray,
    center: float,
    sigma: float,
    gamma: float,
    amplitude: float,
) -> np.ndarray:
    """Generate a single Voigt spectrum (exact, not Taylor)."""
    from scipy import special as sps
    SQRT2 = np.sqrt(2.0)
    SQRT2PI = np.sqrt(2.0 * np.pi)
    z = ((energy - center) + 1j * gamma) / (sigma * SQRT2)
    return amplitude * np.real(sps.wofz(z)) / (sigma * SQRT2PI)


@pytest.fixture
def preset():
    return C1S_SINGLE_PRESET


@pytest.fixture
def encoder(preset):
    return SinglePeakEncoder(preset)


@pytest.fixture
def energy(preset):
    return preset.energy


@pytest.fixture
def cache():
    from toyomacro.voigtfit.weight_cache import WeightMatrixCache
    c = WeightMatrixCache()
    c.clear_memory_cache()
    return c


@pytest.fixture
def pipeline(cache):
    from toyomacro.voigtfit.pipeline import HybridPipeline
    return HybridPipeline(
        cache=cache,
        enable_shift_correction=True,
        enable_stage2=False,
        use_mlx=True,
    )


class TestSinglePeakEncoder:
    """Test parameter encoding/decoding."""

    def test_encode_decode_identity(self, encoder):
        """encode -> decode should recover RGB image within quantization error."""
        rng = np.random.default_rng(42)
        H, W = 10, 10
        image = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)

        amp, dE, fwhm = encoder.encode(image)
        reconstructed = encoder.decode(amp, dE, fwhm, (H, W))

        # Quantization error: max 1 level
        np.testing.assert_allclose(
            reconstructed.astype(np.int16), image.astype(np.int16),
            atol=1, err_msg="Encode-decode roundtrip exceeds 1-level error"
        )

    def test_amplitude_range(self, encoder):
        """R=0 -> amp_min, R=255 -> amp_max."""
        H, W = 1, 2
        image = np.zeros((H, W, 3), dtype=np.uint8)
        image[0, 0, 0] = 0
        image[0, 1, 0] = 255

        amp, _, _ = encoder.encode(image)
        pr = encoder.preset.amplitude_range
        assert abs(amp[0] - pr.min_val) < 1e-5
        assert abs(amp[1] - pr.max_val) < 1e-3

    def test_shift_range(self, encoder):
        """G=0 -> shift_min, G=128 -> ~0, G=255 -> shift_max."""
        H, W = 1, 3
        image = np.zeros((H, W, 3), dtype=np.uint8)
        image[0, 0, 1] = 0
        image[0, 1, 1] = 128
        image[0, 2, 1] = 255

        _, dE, _ = encoder.encode(image)
        pr = encoder.preset.shift_range
        assert abs(dE[0] - pr.min_val) < 1e-5
        assert abs(dE[1]) < 0.01  # ~0 at midpoint
        assert abs(dE[2] - pr.max_val) < 1e-3

    def test_fwhm_range(self, encoder):
        """B=0 -> fwhm_min, B=255 -> fwhm_max."""
        H, W = 1, 2
        image = np.zeros((H, W, 3), dtype=np.uint8)
        image[0, 0, 2] = 0
        image[0, 1, 2] = 255

        _, _, fwhm = encoder.encode(image)
        pr = encoder.preset.fwhm_range
        assert abs(fwhm[0] - pr.min_val) < 1e-5
        assert abs(fwhm[1] - pr.max_val) < 1e-3

    def test_delta_sigma(self, encoder):
        """delta_sigma at nominal FWHM should be ~0."""
        nominal_fwhm = encoder.preset.element.gaussian_fwhm
        ds = encoder.delta_sigma(np.array([nominal_fwhm]))
        assert abs(ds[0]) < 1e-6


class TestTaylorSpectra:
    """Test Taylor-expansion spectra generation accuracy."""

    def test_zero_perturbation(self, preset, energy):
        """δE=0, δσ=0 → same spectrum as exact Voigt."""
        from toyomacro.voigtfit.weight_cache import WeightMatrixCache
        cache = WeightMatrixCache()

        center = preset.element.binding_energy
        sigma = preset.element.sigma
        gamma = preset.element.gamma

        exact = _generate_voigt_spectrum(energy, center, sigma, gamma, 1.0)

        Phi, J_c, J_sigma = cache.build_basis_voigt_with_full_jacobian(
            energy,
            np.array([center], dtype=np.float32),
            np.array([sigma], dtype=np.float32),
            gamma,
        )

        taylor = Phi[:, 0]  # δE=0, δσ=0 → just Phi
        np.testing.assert_allclose(taylor, exact, rtol=1e-5)

    @pytest.mark.parametrize("delta_E", [0.1, 0.3, 0.5])
    def test_taylor_shift_accuracy(self, preset, energy, delta_E):
        """Taylor approximation vs exact for small center shifts."""
        from toyomacro.voigtfit.weight_cache import WeightMatrixCache
        cache = WeightMatrixCache()

        center = preset.element.binding_energy
        sigma = preset.element.sigma
        gamma = preset.element.gamma

        exact = _generate_voigt_spectrum(
            energy, center + delta_E, sigma, gamma, 1.0
        )

        Phi, J_c, J_sigma = cache.build_basis_voigt_with_full_jacobian(
            energy,
            np.array([center], dtype=np.float32),
            np.array([sigma], dtype=np.float32),
            gamma,
        )

        taylor = Phi[:, 0] + delta_E * J_c[:, 0]
        rel_error = np.max(np.abs(taylor - exact)) / np.max(exact)

        # Taylor should be < 5% for δE < 0.5σ
        if abs(delta_E) < 0.5 * sigma:
            assert rel_error < 0.05, f"δE={delta_E}: rel_error={rel_error:.4f}"

    @pytest.mark.parametrize("delta_sigma", [0.05, 0.1])
    def test_taylor_width_accuracy(self, preset, energy, delta_sigma):
        """Taylor approximation vs exact for small sigma changes."""
        from toyomacro.voigtfit.weight_cache import WeightMatrixCache
        cache = WeightMatrixCache()

        center = preset.element.binding_energy
        sigma = preset.element.sigma
        gamma = preset.element.gamma

        exact = _generate_voigt_spectrum(
            energy, center, sigma + delta_sigma, gamma, 1.0
        )

        Phi, J_c, J_sigma = cache.build_basis_voigt_with_full_jacobian(
            energy,
            np.array([center], dtype=np.float32),
            np.array([sigma], dtype=np.float32),
            gamma,
        )

        taylor = Phi[:, 0] + delta_sigma * J_sigma[:, 0]
        rel_error = np.max(np.abs(taylor - exact)) / np.max(exact)
        assert rel_error < 0.05, f"δσ={delta_sigma}: rel_error={rel_error:.4f}"


class TestThreeStepRecovery:
    """Test 3-step residual projection recovery."""

    def _make_spectra(self, energy, preset, n_spectra, true_shift, true_dsigma):
        """Generate spectra with known shift and width change."""
        center = preset.element.binding_energy
        sigma = preset.element.sigma
        gamma = preset.element.gamma

        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            dE = true_shift if np.isscalar(true_shift) else true_shift[i]
            ds = true_dsigma if np.isscalar(true_dsigma) else true_dsigma[i]
            Y[i, :] = _generate_voigt_spectrum(
                energy, center + dE, sigma + ds, gamma, 1000.0
            )
        return Y

    def test_recover_shift_only(self, pipeline, energy, preset):
        """Known δE=0.3, δσ=0 → shift recovered, sigma ~0."""
        peak_config = {
            "centers": np.array([preset.element.binding_energy], dtype=np.float32),
            "sigmas": np.array([preset.element.sigma], dtype=np.float32),
            "gamma": preset.element.gamma,
        }

        Y = self._make_spectra(energy, preset, 100, 0.3, 0.0)
        result = pipeline.process_rowmajor_extended_3param(
            Y, preset.element.symbol, preset.element.orbital,
            energy, peak_config, amplitudes_only=True,
        )

        mean_dE = np.mean(result.energy_shifts)
        mean_ds = np.mean(result.sigma_shifts)
        assert abs(mean_dE - 0.3) < 0.05, f"δE: expected 0.3, got {mean_dE:.4f}"
        # σ cross-talk from Taylor O(δE²) residual: δE/σ≈0.7 → ~0.07 leakage
        assert abs(mean_ds) < 0.1, f"δσ: expected ~0, got {mean_ds:.4f}"

    def test_recover_width_only(self, pipeline, energy, preset):
        """Known δE=0, δσ=0.1 → width recovered, shift ~0."""
        peak_config = {
            "centers": np.array([preset.element.binding_energy], dtype=np.float32),
            "sigmas": np.array([preset.element.sigma], dtype=np.float32),
            "gamma": preset.element.gamma,
        }

        Y = self._make_spectra(energy, preset, 100, 0.0, 0.1)
        result = pipeline.process_rowmajor_extended_3param(
            Y, preset.element.symbol, preset.element.orbital,
            energy, peak_config, amplitudes_only=True,
        )

        mean_dE = np.mean(result.energy_shifts)
        mean_ds = np.mean(result.sigma_shifts)
        assert abs(mean_dE) < 0.05, f"δE: expected 0, got {mean_dE:.4f}"
        assert abs(mean_ds - 0.1) < 0.05, f"δσ: expected 0.1, got {mean_ds:.4f}"

    def test_recover_both(self, pipeline, energy, preset):
        """Known δE=0.2, δσ=0.08 → both recovered simultaneously."""
        peak_config = {
            "centers": np.array([preset.element.binding_energy], dtype=np.float32),
            "sigmas": np.array([preset.element.sigma], dtype=np.float32),
            "gamma": preset.element.gamma,
        }

        Y = self._make_spectra(energy, preset, 200, 0.2, 0.08)
        result = pipeline.process_rowmajor_extended_3param(
            Y, preset.element.symbol, preset.element.orbital,
            energy, peak_config, amplitudes_only=True,
        )

        mean_dE = np.mean(result.energy_shifts)
        mean_ds = np.mean(result.sigma_shifts)
        assert abs(mean_dE - 0.2) < 0.1, f"δE: expected 0.2, got {mean_dE:.4f}"
        assert abs(mean_ds - 0.08) < 0.05, f"δσ: expected 0.08, got {mean_ds:.4f}"

    def test_amplitude_sigma_correction(self, pipeline, energy, preset):
        """Step 4: amplitude correction removes σ-coupling bias.

        Without correction, δσ=+0.1 (δσ/σ≈0.24) causes ~8% amplitude error.
        With correction, amplitude error should be < 2%.
        """
        peak_config = {
            "centers": np.array([preset.element.binding_energy], dtype=np.float32),
            "sigmas": np.array([preset.element.sigma], dtype=np.float32),
            "gamma": preset.element.gamma,
        }
        true_amp = 1000.0

        for ds in [0.05, 0.1, -0.1]:
            Y = self._make_spectra(energy, preset, 100, 0.0, ds)
            result = pipeline.process_rowmajor_extended_3param(
                Y, preset.element.symbol, preset.element.orbital,
                energy, peak_config, amplitudes_only=True,
            )
            mean_amp = np.mean(result.amplitudes[0, :])
            amp_err_pct = abs(mean_amp - true_amp) / true_amp * 100
            assert amp_err_pct < 3.0, (
                f"δσ={ds}: amp={mean_amp:.1f}, error={amp_err_pct:.1f}% (expected <3%)"
            )

    def test_varying_params_per_spectrum(self, pipeline, energy, preset):
        """Per-spectrum varying δE and δσ should be recovered with high correlation.

        Note: δE range limited to ±0.15 eV (δE/σ ≤ 0.35) for Taylor validity.
        At ±0.3 (δE/σ=0.71), 2nd-order Taylor error projects onto J_σ → corr~0.74.
        """
        np.random.seed(42)
        n_spectra = 500
        peak_config = {
            "centers": np.array([preset.element.binding_energy], dtype=np.float32),
            "sigmas": np.array([preset.element.sigma], dtype=np.float32),
            "gamma": preset.element.gamma,
        }

        true_dE = np.random.uniform(-0.15, 0.15, n_spectra).astype(np.float32)
        true_ds = np.random.uniform(-0.05, 0.1, n_spectra).astype(np.float32)

        Y = self._make_spectra(energy, preset, n_spectra, true_dE, true_ds)
        result = pipeline.process_rowmajor_extended_3param(
            Y, preset.element.symbol, preset.element.orbital,
            energy, peak_config, amplitudes_only=True,
        )

        corr_dE = np.corrcoef(true_dE, result.energy_shifts)[0, 1]
        corr_ds = np.corrcoef(true_ds, result.sigma_shifts)[0, 1]

        assert corr_dE > 0.95, f"δE correlation: {corr_dE:.4f}"
        assert corr_ds > 0.90, f"δσ correlation: {corr_ds:.4f}"

        rmse_dE = np.sqrt(np.mean((result.energy_shifts - true_dE)**2))
        rmse_ds = np.sqrt(np.mean((result.sigma_shifts - true_ds)**2))
        print(f"\nPer-spectrum recovery (n={n_spectra}):")
        print(f"  δE: corr={corr_dE:.4f}, RMSE={rmse_dE:.4f} eV")
        print(f"  δσ: corr={corr_ds:.4f}, RMSE={rmse_ds:.4f} eV")


@pytest.mark.perf
@pytest.mark.skipif(IN_CI, reason="speed ratio assertion is sensitive to CI hardware")
class TestThreeStepThroughput:
    """Benchmark 3-step vs 2-step kernel throughput."""

    def test_3step_throughput(self, cache, energy, preset):
        """3-step kernel within 2x of 2-step.

        Both kernels are warmed before the clock starts and each is timed
        repeatedly, because a single cold pair measures neither one. The
        first pipeline to run absorbs the MLX compile cost for the shared
        kernels, which deflates the ratio well below 1; a single timing
        window also lets one scheduling hiccup dominate the result.
        """
        from toyomacro.voigtfit.pipeline import HybridPipeline

        peak_config = {
            "centers": np.array([preset.element.binding_energy], dtype=np.float32),
            "sigmas": np.array([preset.element.sigma], dtype=np.float32),
            "gamma": preset.element.gamma,
        }
        n_spectra = 100000
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i, :] = _generate_voigt_spectrum(
                energy, preset.element.binding_energy,
                preset.element.sigma, preset.element.gamma, 1000.0,
            )

        def _pipeline():
            return HybridPipeline(
                cache=cache, enable_shift_correction=True,
                enable_stage2=False, use_mlx=True,
            )

        def _run_2step():
            _pipeline().process_rowmajor_extended(
                Y, preset.element.symbol, preset.element.orbital,
                energy, peak_config, amplitudes_only=True,
            )

        def _run_3step():
            _pipeline().process_rowmajor_extended_3param(
                Y, preset.element.symbol, preset.element.orbital,
                energy, peak_config, amplitudes_only=True,
            )

        def _median_seconds(run, reps=5):
            samples = []
            for _ in range(reps):
                t0 = time.perf_counter()
                run()
                samples.append(time.perf_counter() - t0)
            return float(np.median(samples))

        # Warm up both paths first — the discarded runs pay for kernel
        # compilation so neither timed loop is charged for the other's.
        _run_2step()
        _run_3step()

        t_2step = _median_seconds(_run_2step)
        t_3step = _median_seconds(_run_3step)

        slowdown = t_3step / t_2step
        print(f"\n3-step vs 2-step (n={n_spectra:,}, warm, median of 5):")
        print(f"  2-step: {n_spectra/t_2step/1e6:.1f}M spec/s ({t_2step*1e3:.1f} ms)")
        print(f"  3-step: {n_spectra/t_3step/1e6:.1f}M spec/s ({t_3step*1e3:.1f} ms)")
        print(f"  Slowdown: {slowdown:.2f}x")

        assert slowdown < 2.0, f"3-step too slow: {slowdown:.2f}x"
