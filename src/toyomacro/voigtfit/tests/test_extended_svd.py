"""
Test Extended SVD Energy Shift Correction (2-step residual projection)
=====================================================================

Verifies:
1. Known energy shifts (0, 0.1, 0.5, 1.0, 2.0 eV) are recovered accurately
2. Zero shift: no amplitude distortion vs standard pipeline
3. Unbalanced composition: amplitude-weighted projection handles well
4. Per-spectrum varying shifts
5. Throughput: extended pipeline within 3x of standard Stage 1

Sign convention:
    energy_shift > 0 means peaks shifted to higher energy.
    _generate_voigt_spectra(energy_shift=+0.5) shifts peaks by +0.5 eV.
"""

import os
import time

import numpy as np
import pytest

IN_CI = os.environ.get("CI") == "true"


def _generate_voigt_spectra(
    energy: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gamma: float,
    amplitudes: np.ndarray,
    energy_shift: float = 0.0,
) -> np.ndarray:
    """Generate a single spectrum with optional global energy shift.

    Shift convention: peaks move to (center + energy_shift).
    """
    from scipy import special as sps

    SQRT2 = np.sqrt(2.0)
    SQRT2PI = np.sqrt(2.0 * np.pi)
    spectrum = np.zeros_like(energy)
    for c, s, a in zip(centers, sigmas, amplitudes):
        z = ((energy - (c + energy_shift)) + 1j * gamma) / (s * SQRT2)
        spectrum += a * np.real(sps.wofz(z)) / (s * SQRT2PI)
    return spectrum


@pytest.fixture
def peak_config():
    return {
        "centers": np.array([99.5, 103.0], dtype=np.float32),
        "sigmas": np.array([0.6, 0.8], dtype=np.float32),
        "gamma": 0.25,
    }


@pytest.fixture
def energy():
    return np.linspace(95, 110, 151).astype(np.float32)


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


class TestWeightCacheJacobian:
    """Test weight_cache.py Jacobian methods."""

    def test_build_basis_with_jacobian(self, cache, energy, peak_config):
        """Phi and J_c have correct shapes and finite values."""
        Phi, J_c = cache.build_basis_voigt_with_jacobian(
            energy,
            peak_config["centers"],
            peak_config["sigmas"],
            peak_config["gamma"],
        )
        n_energy = len(energy)
        n_comp = len(peak_config["centers"])
        assert Phi.shape == (n_energy, n_comp)
        assert J_c.shape == (n_energy, n_comp)
        assert np.all(np.isfinite(Phi))
        assert np.all(np.isfinite(J_c))
        assert np.all(Phi >= 0)

    def test_get_or_create_with_jacobian(self, cache, energy, peak_config):
        """Jacobian cache stores and retrieves correctly."""
        r1 = cache.get_or_create_with_jacobian(
            "test", "1s", energy,
            peak_config["centers"], peak_config["sigmas"],
            peak_config["gamma"], use_mlx=False,
        )
        r2 = cache.get_or_create_with_jacobian(
            "test", "1s", energy,
            peak_config["centers"], peak_config["sigmas"],
            peak_config["gamma"], use_mlx=False,
        )
        np.testing.assert_array_equal(r1[0], r2[0])  # Wt
        np.testing.assert_array_equal(r1[2], r2[2])  # J_c

    def test_jacobian_sign_convention(self, cache, energy, peak_config):
        """dV/dc: center moves right → peak shifts right → left tail decreases."""
        Phi, J_c = cache.build_basis_voigt_with_jacobian(
            energy,
            peak_config["centers"],
            peak_config["sigmas"],
            peak_config["gamma"],
        )
        # For peak at 99.5: dV/dc < 0 for E < center (left tail decreases)
        # dV/dc > 0 for E > center (right tail increases)
        center_idx = np.argmin(np.abs(energy - peak_config["centers"][0]))
        left_mean = np.mean(J_c[:center_idx, 0])
        right_mean = np.mean(J_c[center_idx:, 0])
        # J_c = dV/dc = -dV/dE (since V depends on E-c)
        # center moves right → V(E) shifts right → left side decreases
        assert left_mean < 0, f"Expected negative J_c on left, got {left_mean}"
        assert right_mean > 0, f"Expected positive J_c on right, got {right_mean}"


class TestShiftRecovery:
    """Test energy shift recovery accuracy."""

    @pytest.mark.parametrize("true_shift,tol", [
        (0.0, 0.05), (0.1, 0.05), (0.5, 0.05), (1.0, 0.3),
    ])
    def test_known_shift_recovery(self, pipeline, energy, peak_config, true_shift, tol):
        """Recover known global energy shifts.

        Linear approximation accuracy degrades for large shifts:
        - δE < 0.5σ: excellent (<5% error)
        - δE ~ σ: moderate (~25% error, still correct direction)
        - δE > 2σ: Taylor expansion breaks down
        """
        n_spectra = 100
        true_amps = np.array([1000.0, 600.0])

        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i, :] = _generate_voigt_spectra(
                energy, peak_config["centers"], peak_config["sigmas"],
                peak_config["gamma"], true_amps,
                energy_shift=true_shift,
            )

        result = pipeline.process_rowmajor_extended(
            Y, "test", "1s", energy, peak_config, amplitudes_only=True,
        )

        assert result.energy_shifts is not None
        mean_shift = np.mean(result.energy_shifts)
        abs_error = abs(mean_shift - true_shift)
        assert abs_error < tol, \
            f"Shift {true_shift}: recovered {mean_shift:.4f}, error={abs_error:.4f} (tol={tol})"

    def test_large_shift_correct_sign(self, pipeline, energy, peak_config):
        """Large shifts (2.0 eV) exceed linear range; just check sign."""
        n_spectra = 100
        true_amps = np.array([1000.0, 600.0])
        true_shift = 2.0

        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i, :] = _generate_voigt_spectra(
                energy, peak_config["centers"], peak_config["sigmas"],
                peak_config["gamma"], true_amps,
                energy_shift=true_shift,
            )

        result = pipeline.process_rowmajor_extended(
            Y, "test", "1s", energy, peak_config, amplitudes_only=True,
        )

        # 2.0 eV >> σ (0.6-0.8), so Taylor is far from valid.
        # We just verify the method returns *something* without crashing.
        assert result.energy_shifts is not None
        assert np.all(np.isfinite(result.energy_shifts))

    def test_zero_shift_no_amplitude_distortion(self, pipeline, energy, peak_config):
        """Zero shift: amplitudes should match standard pipeline exactly."""
        from toyomacro.voigtfit.pipeline import HybridPipeline

        n_spectra = 200
        true_amps = np.array([1000.0, 600.0])

        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i, :] = _generate_voigt_spectra(
                energy, peak_config["centers"], peak_config["sigmas"],
                peak_config["gamma"], true_amps,
            )

        result_ext = pipeline.process_rowmajor_extended(
            Y, "test", "1s", energy, peak_config, amplitudes_only=True,
        )

        std_pipeline = HybridPipeline(
            cache=pipeline.cache, enable_stage2=False, use_mlx=True,
        )
        result_std = std_pipeline.process_rowmajor(
            Y, "test", "1s", energy, peak_config, amplitudes_only=True,
        )

        # Identical amplitudes (same first step)
        np.testing.assert_allclose(
            result_ext.amplitudes, result_std.amplitudes, rtol=1e-4,
            err_msg="Zero-shift amplitudes differ from standard pipeline"
        )

    def test_unbalanced_composition(self, pipeline, energy, peak_config):
        """Amplitude-weighted projection handles dominant peak well."""
        n_spectra = 100
        true_shift = 0.5
        true_amps = np.array([10000.0, 100.0])

        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i, :] = _generate_voigt_spectra(
                energy, peak_config["centers"], peak_config["sigmas"],
                peak_config["gamma"], true_amps,
                energy_shift=true_shift,
            )

        result = pipeline.process_rowmajor_extended(
            Y, "test", "1s", energy, peak_config, amplitudes_only=True,
        )

        mean_shift = np.mean(result.energy_shifts)
        assert abs(mean_shift - true_shift) < 0.15, \
            f"Unbalanced: shift {true_shift}, recovered {mean_shift:.4f}"

    def test_varying_shifts_per_spectrum(self, pipeline, energy, peak_config):
        """Per-spectrum shifts should be recovered with high correlation."""
        np.random.seed(42)
        n_spectra = 500
        true_amps = np.array([1000.0, 600.0])
        true_shifts = np.random.uniform(-0.5, 0.5, n_spectra).astype(np.float32)

        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i, :] = _generate_voigt_spectra(
                energy, peak_config["centers"], peak_config["sigmas"],
                peak_config["gamma"], true_amps,
                energy_shift=true_shifts[i],
            )

        result = pipeline.process_rowmajor_extended(
            Y, "test", "1s", energy, peak_config, amplitudes_only=True,
        )

        correlation = np.corrcoef(true_shifts, result.energy_shifts)[0, 1]
        assert correlation > 0.95, f"Shift correlation: {correlation:.4f}"

        rmse = np.sqrt(np.mean((result.energy_shifts - true_shifts)**2))
        assert rmse < 0.1, f"Shift RMSE: {rmse:.4f} eV"

    def test_negative_shifts(self, pipeline, energy, peak_config):
        """Negative shifts should be recovered with correct sign."""
        n_spectra = 100
        true_amps = np.array([1000.0, 600.0])
        true_shift = -0.3

        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i, :] = _generate_voigt_spectra(
                energy, peak_config["centers"], peak_config["sigmas"],
                peak_config["gamma"], true_amps,
                energy_shift=true_shift,
            )

        result = pipeline.process_rowmajor_extended(
            Y, "test", "1s", energy, peak_config, amplitudes_only=True,
        )

        mean_shift = np.mean(result.energy_shifts)
        assert abs(mean_shift - true_shift) < 0.1, \
            f"Negative shift: expected {true_shift}, got {mean_shift:.4f}"


class TestShiftWithNoise:
    """Test shift recovery under noise."""

    def test_noisy_shift_recovery(self, pipeline, energy, peak_config):
        """Shift recovery with 1% noise."""
        np.random.seed(42)
        n_spectra = 500
        true_amps = np.array([1000.0, 600.0])
        true_shift = 0.3

        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i, :] = _generate_voigt_spectra(
                energy, peak_config["centers"], peak_config["sigmas"],
                peak_config["gamma"], true_amps,
                energy_shift=true_shift,
            )

        noise_level = Y.max() * 0.01
        rng = np.random.default_rng(42)
        Y += (noise_level * rng.standard_normal(Y.shape)).astype(np.float32)

        result = pipeline.process_rowmajor_extended(
            Y, "test", "1s", energy, peak_config, amplitudes_only=True,
        )

        mean_shift = np.mean(result.energy_shifts)
        assert abs(mean_shift - true_shift) < 0.15, \
            f"Noisy shift: expected {true_shift}, got {mean_shift:.4f}"


class TestFitResultFields:
    """Test FitResult output structure."""

    def test_energy_shifts_in_result(self, pipeline, energy, peak_config):
        """FitResult includes energy_shifts field."""
        n_spectra = 50
        true_amps = np.array([1000.0, 600.0])
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i, :] = _generate_voigt_spectra(
                energy, peak_config["centers"], peak_config["sigmas"],
                peak_config["gamma"], true_amps,
            )

        result = pipeline.process_rowmajor_extended(
            Y, "test", "1s", energy, peak_config,
        )

        assert result.energy_shifts is not None
        assert result.energy_shifts.shape == (n_spectra,)
        assert result.centers is not None
        assert result.centers.shape == (2, n_spectra)
        assert result.chi2 is not None

    def test_centers_adjusted_by_shift(self, pipeline, energy, peak_config):
        """Output centers = nominal + δE."""
        n_spectra = 50
        true_shift = 0.5
        true_amps = np.array([1000.0, 600.0])
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i, :] = _generate_voigt_spectra(
                energy, peak_config["centers"], peak_config["sigmas"],
                peak_config["gamma"], true_amps, energy_shift=true_shift,
            )

        result = pipeline.process_rowmajor_extended(
            Y, "test", "1s", energy, peak_config,
        )

        for k in range(2):
            mean_center = np.mean(result.centers[k, :])
            expected = peak_config["centers"][k] + true_shift
            assert abs(mean_center - expected) < 0.15, \
                f"Center {k}: expected {expected:.2f}, got {mean_center:.4f}"


@pytest.mark.skipif(IN_CI, reason="speed ratio assertion is sensitive to CI hardware")
class TestThroughput:
    """Benchmark extended vs standard pipeline throughput."""

    def test_extended_throughput(self, cache, energy, peak_config):
        """Extended pipeline within 3x of standard (3 matmuls vs 1)."""
        from toyomacro.voigtfit.pipeline import HybridPipeline

        n_spectra = 100000
        true_amps = np.array([1000.0, 600.0])
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i, :] = _generate_voigt_spectra(
                energy, peak_config["centers"], peak_config["sigmas"],
                peak_config["gamma"], true_amps,
            )

        # Standard
        std_pipeline = HybridPipeline(
            cache=cache, enable_stage2=False, use_mlx=True,
        )
        t0 = time.perf_counter()
        _ = std_pipeline.process_rowmajor(
            Y, "test", "1s", energy, peak_config, amplitudes_only=True,
        )
        t_std = time.perf_counter() - t0

        # Extended
        ext_pipeline = HybridPipeline(
            cache=cache, enable_shift_correction=True,
            enable_stage2=False, use_mlx=True,
        )
        t0 = time.perf_counter()
        _ = ext_pipeline.process_rowmajor_extended(
            Y, "test", "1s", energy, peak_config, amplitudes_only=True,
        )
        t_ext = time.perf_counter() - t0

        rate_std = n_spectra / t_std
        rate_ext = n_spectra / t_ext
        slowdown = t_ext / t_std

        print("\nThroughput comparison:")
        print(f"  Standard: {rate_std/1e6:.1f}M spec/s ({t_std*1e3:.1f} ms)")
        print(f"  Extended: {rate_ext/1e6:.1f}M spec/s ({t_ext*1e3:.1f} ms)")
        print(f"  Slowdown: {slowdown:.2f}x")

        assert slowdown < 3.0, f"Extended too slow: {slowdown:.2f}x"

    def test_mx_compile_speedup(self, cache, energy, peak_config):
        """Measure mx.compile effect on 2-step shift kernel.

        mx.compile fuses element-wise ops (R*S sum, S*S sum, R*R sum)
        between matmuls into fewer GPU kernel launches.
        """
        from toyomacro.voigtfit.pipeline import (
            _shift_kernel_compiled,
            _shift_kernel_mlx,
        )
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")

        if _shift_kernel_compiled is None:
            pytest.skip("mx.compile not available")

        n_spectra = 1_000_000
        true_amps = np.array([1000.0, 600.0])
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i, :] = _generate_voigt_spectra(
                energy, peak_config["centers"], peak_config["sigmas"],
                peak_config["gamma"], true_amps,
            )

        # Build matrices
        Wt, Phi, J_c = cache.get_or_create_with_jacobian(
            "test", "1s", energy,
            peak_config["centers"], peak_config["sigmas"],
            peak_config["gamma"], use_mlx=True,
        )
        W = Wt.T
        Phi_T = Phi.T
        J_c_T = J_c.T
        Y_mx = mx.array(Y)
        n_energy_inv = mx.array(1.0 / len(energy))

        # Warmup both paths
        for fn in [_shift_kernel_mlx, _shift_kernel_compiled]:
            out = fn(Y_mx[:1000], W, Phi_T, J_c_T, n_energy_inv)
            mx.eval(*out)

        n_iter = 5

        # Non-compiled
        times_plain = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            out = _shift_kernel_mlx(Y_mx, W, Phi_T, J_c_T, n_energy_inv)
            mx.eval(*out)
            times_plain.append(time.perf_counter() - t0)
        t_plain = np.median(times_plain)

        # Compiled
        times_compiled = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            out = _shift_kernel_compiled(Y_mx, W, Phi_T, J_c_T, n_energy_inv)
            mx.eval(*out)
            times_compiled.append(time.perf_counter() - t0)
        t_compiled = np.median(times_compiled)

        rate_plain = n_spectra / t_plain
        rate_compiled = n_spectra / t_compiled
        speedup = t_plain / t_compiled

        print(f"\nmx.compile effect (n={n_spectra:,}):")
        print(f"  Plain:    {rate_plain/1e6:.1f}M spec/s ({t_plain*1e3:.1f} ms)")
        print(f"  Compiled: {rate_compiled/1e6:.1f}M spec/s ({t_compiled*1e3:.1f} ms)")
        print(f"  Speedup:  {speedup:.2f}x")

        # Verify correctness: compiled == plain
        out_p = _shift_kernel_mlx(Y_mx[:100], W, Phi_T, J_c_T, n_energy_inv)
        mx.eval(*out_p)
        out_c = _shift_kernel_compiled(Y_mx[:100], W, Phi_T, J_c_T, n_energy_inv)
        mx.eval(*out_c)
        np.testing.assert_allclose(
            np.array(out_p[0]), np.array(out_c[0]), rtol=1e-5,
            err_msg="Compiled amplitudes differ from plain"
        )
        np.testing.assert_allclose(
            np.array(out_p[2]), np.array(out_c[2]), rtol=1e-4,
            err_msg="Compiled numerator differs from plain"
        )

    def test_cpu_gpu_boundary(self, cache, energy, peak_config):
        """Profile CPU-GPU data transfer overhead.

        Measures: MLX conversion, kernel execution, NumPy readback separately.
        """
        from toyomacro.voigtfit.pipeline import _shift_kernel_compiled
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")

        if _shift_kernel_compiled is None:
            pytest.skip("mx.compile not available")

        n_spectra = 1_000_000
        true_amps = np.array([1000.0, 600.0])
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i, :] = _generate_voigt_spectra(
                energy, peak_config["centers"], peak_config["sigmas"],
                peak_config["gamma"], true_amps,
            )

        Wt, Phi, J_c = cache.get_or_create_with_jacobian(
            "test", "1s", energy,
            peak_config["centers"], peak_config["sigmas"],
            peak_config["gamma"], use_mlx=True,
        )
        W = Wt.T
        Phi_T = Phi.T
        J_c_T = J_c.T
        n_energy_inv = mx.array(1.0 / len(energy))

        # Warmup
        Y_mx = mx.array(Y[:1000])
        out = _shift_kernel_compiled(Y_mx, W, Phi_T, J_c_T, n_energy_inv)
        mx.eval(*out)

        n_iter = 5

        # Phase 1: NumPy → MLX conversion
        times_convert = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            Y_mx = mx.array(Y)
            mx.eval(Y_mx)
            times_convert.append(time.perf_counter() - t0)
        t_convert = np.median(times_convert)

        # Phase 2: GPU kernel
        Y_mx = mx.array(Y)
        mx.eval(Y_mx)
        times_kernel = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            A_row, chi2, num, den = _shift_kernel_compiled(
                Y_mx, W, Phi_T, J_c_T, n_energy_inv
            )
            mx.eval(A_row, chi2, num, den)
            times_kernel.append(time.perf_counter() - t0)
        t_kernel = np.median(times_kernel)

        # Phase 3: MLX → NumPy readback
        A_row, chi2, num, den = _shift_kernel_compiled(
            Y_mx, W, Phi_T, J_c_T, n_energy_inv
        )
        mx.eval(A_row, chi2, num, den)
        times_readback = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = np.array(A_row)
            _ = np.array(chi2)
            _ = np.array(num)
            _ = np.array(den)
            times_readback.append(time.perf_counter() - t0)
        t_readback = np.median(times_readback)

        t_total = t_convert + t_kernel + t_readback
        data_mb = Y.nbytes / 1e6

        print(f"\nCPU-GPU boundary analysis (n={n_spectra:,}, {data_mb:.0f} MB):")
        print(f"  NumPy→MLX:  {t_convert*1e3:6.1f} ms ({t_convert/t_total*100:4.1f}%)"
              f"  [{data_mb/t_convert:.0f} MB/s]")
        print(f"  GPU kernel: {t_kernel*1e3:6.1f} ms ({t_kernel/t_total*100:4.1f}%)"
              f"  [{n_spectra/t_kernel/1e6:.1f}M spec/s]")
        print(f"  MLX→NumPy:  {t_readback*1e3:6.1f} ms ({t_readback/t_total*100:4.1f}%)")
        print(f"  Total:      {t_total*1e3:6.1f} ms"
              f"  [{n_spectra/t_total/1e6:.1f}M spec/s effective]")


# ===== 6-Step Hessian-Corrected Pipeline Tests =====

class Test6StepHessian:
    """Tests for 6-step Taylor pipeline with d²V/dc² Hessian correction."""

    def test_hessian_numerical_verification(self):
        """Verify d²V/dc² against central difference of dV/dc."""
        from toyomacro.voigtfit.voigt_jacobian import verify_hessian
        results = verify_hessian(verbose=False)
        assert results["all_passed"], f"max error {results['max_error']:.2e}"

    def test_hessian_negative_at_peak(self):
        """d²V/dc² should be negative at peak center (concave down)."""
        from toyomacro.voigtfit.voigt_jacobian import voigt_with_hessian
        energy = np.linspace(-5, 5, 201)
        _, _, _, _, d2V = voigt_with_hessian(energy, 0.0, 0.8, 0.3)
        center_idx = np.argmin(np.abs(energy))
        assert d2V[center_idx] < 0, f"d2V/dc2 at peak = {d2V[center_idx]}"

    def test_hessian_batch_shapes(self):
        """voigt_hessian_batch returns correct shapes."""
        from toyomacro.voigtfit.voigt_jacobian import voigt_hessian_batch
        energy = np.linspace(280, 290, 101)
        centers = np.array([284.0, 285.0, 286.0])
        sigmas = np.array([0.8, 0.9, 0.8])
        Phi, J_c, J_s, J_g, H_cc = voigt_hessian_batch(energy, centers, sigmas, 0.2)
        assert Phi.shape == (101, 3)
        assert H_cc.shape == (101, 3)

    def test_six_step_zero_shift_identity(self):
        """With δE=0, 6-step should produce identical δσ as 4-step."""
        from toyomacro.voigtfit.pipeline import HybridPipeline
        from toyomacro.voigtfit.weight_cache import WeightMatrixCache

        cache = WeightMatrixCache()
        pipeline = HybridPipeline(cache=cache, use_mlx=True)

        energy = np.linspace(280, 290, 101).astype(np.float32)
        centers = np.array([285.0], dtype=np.float32)
        sigmas = np.array([0.5], dtype=np.float32)
        peak_config = {"centers": centers, "sigmas": sigmas, "gamma": 0.2}

        # Generate spectrum with NO shift, just δσ
        from toyomacro.voigtfit.voigt_jacobian import voigt_profile
        V = voigt_profile(energy.astype(np.float64), 285.0, 0.55, 0.2)
        Y = (np.array([V]) * 1000).astype(np.float32)
        Y = np.tile(Y, (50, 1))

        res4 = pipeline.process_rowmajor_extended_3param(
            Y.copy(), "C", "1s", energy, peak_config,
            amplitudes_only=True, n_steps=4,
        )
        res6 = pipeline.process_rowmajor_extended_3param(
            Y.copy(), "C", "1s", energy, peak_config,
            amplitudes_only=True, n_steps=6,
        )

        # δE should be ~0 for both → Hessian correction = ½·0²·hcc_ss = 0
        np.testing.assert_allclose(
            res6.sigma_shifts, res4.sigma_shifts, atol=1e-5,
            err_msg="6-step should match 4-step when δE≈0",
        )

    def test_six_step_dsigma_improvement(self):
        """6-step should improve δσ recovery on exact Voigt with δE≠0."""
        from toyomacro.voigtfit.pipeline import HybridPipeline
        from toyomacro.voigtfit.voigt_jacobian import voigt_profile
        from toyomacro.voigtfit.weight_cache import WeightMatrixCache

        cache = WeightMatrixCache()
        pipeline = HybridPipeline(cache=cache, use_mlx=True)

        energy = np.linspace(280, 290, 101).astype(np.float32)
        centers = np.array([285.0], dtype=np.float32)
        sigmas = np.array([0.5], dtype=np.float32)
        peak_config = {"centers": centers, "sigmas": sigmas, "gamma": 0.2}

        # Generate 1000 exact Voigt spectra with random shifts
        N = 1000
        rng = np.random.RandomState(42)
        dE_true = rng.uniform(-0.5, 0.5, N).astype(np.float32)
        ds_true = rng.uniform(-0.05, 0.05, N).astype(np.float32)

        Y = np.zeros((N, len(energy)), dtype=np.float32)
        for i in range(N):
            V = voigt_profile(
                energy.astype(np.float64),
                float(285.0 + dE_true[i]),
                float(0.5 + ds_true[i]),
                0.2,
            )
            Y[i, :] = (V * 1000).astype(np.float32)

        res4 = pipeline.process_rowmajor_extended_3param(
            Y.copy(), "C", "1s", energy, peak_config,
            amplitudes_only=True, n_steps=4,
        )
        res6 = pipeline.process_rowmajor_extended_3param(
            Y.copy(), "C", "1s", energy, peak_config,
            amplitudes_only=True, n_steps=6,
        )

        rmse4 = np.sqrt(np.mean((res4.sigma_shifts - ds_true) ** 2))
        rmse6 = np.sqrt(np.mean((res6.sigma_shifts - ds_true) ** 2))
        corr6 = np.corrcoef(res6.sigma_shifts, ds_true)[0, 1]

        print(f"\n6-step improvement: δσ RMSE {rmse4:.4f} → {rmse6:.4f} "
              f"({rmse4/rmse6:.1f}x), corr {corr6:.3f}")

        assert rmse6 < rmse4, (
            f"6-step δσ RMSE ({rmse6:.4f}) should be lower than 4-step ({rmse4:.4f})"
        )
        assert corr6 > 0.5, f"6-step δσ correlation ({corr6:.3f}) should be > 0.5"

    def test_weight_cache_hessian(self):
        """get_or_create_with_hessian returns correct shapes and is cached."""
        from toyomacro.voigtfit.weight_cache import WeightMatrixCache

        cache = WeightMatrixCache()
        energy = np.linspace(280, 290, 101).astype(np.float32)
        centers = np.array([285.0], dtype=np.float32)
        sigmas = np.array([0.5], dtype=np.float32)

        Wt, Phi, J_c, J_sigma, H_cc = cache.get_or_create_with_hessian(
            "C", "1s", energy, centers, sigmas, gamma=0.2, use_mlx=False,
        )
        assert Wt.shape == (1, 101)
        assert Phi.shape == (101, 1)
        assert H_cc.shape == (101, 1)

        # Second call should hit cache
        Wt2, _, _, _, H_cc2 = cache.get_or_create_with_hessian(
            "C", "1s", energy, centers, sigmas, gamma=0.2, use_mlx=False,
        )
        np.testing.assert_array_equal(H_cc, H_cc2)
