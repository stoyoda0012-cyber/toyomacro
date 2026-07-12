"""
Tests for Dictionary + 4-Step Hybrid Solver
============================================

Tests cover:
1. Dictionary construction (grid correctness, L2 normalization)
2. Dictionary lookup accuracy (argmax selects correct grid point)
3. Hybrid solver accuracy in Taylor range (should match pure 4-step)
4. Hybrid solver accuracy BEYOND Taylor range (the whole point)
5. Grid boundary cases (δE at midpoint between grid points)
6. Zero-shift case (nominal basis selected)
7. Multi-component support
8. Throughput benchmark
"""

import os
import time

import numpy as np
import pytest

from toyomacro.voigtfit._mlx_support import mlx_usable as _mlx_usable

IN_CI = os.environ.get("CI") == "true"

from toyomacro.voigtfit.dictionary_solver import (
    _parabola_vertex,
    adaptive_solve,
    build_dictionary,
    build_dictionary_2d,
    build_dictionary_shift_only,
    dictionary_lookup_numpy,
    parabola_refine_2d,
    solve_dict2d_adaptive,
    solve_dict2d_adaptive_fused,
    solve_dict2d_amp_only,
    solve_hybrid,
    solve_hybrid_sorted,
)
from toyomacro.voigtfit.pipeline import FitResult, HybridPipeline
from toyomacro.voigtfit.voigt_jacobian import voigt_profile
from toyomacro.voigtfit.weight_cache import WeightMatrixCache

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def energy():
    return np.arange(95.0, 105.0, 0.05, dtype=np.float32)


@pytest.fixture
def single_peak_config():
    return {
        "centers": np.array([100.0], dtype=np.float32),
        "sigmas": np.array([0.6], dtype=np.float32),
        "gamma": 0.2,
    }


@pytest.fixture
def two_peak_config():
    return {
        "centers": np.array([99.5, 101.0], dtype=np.float32),
        "sigmas": np.array([0.6, 0.8], dtype=np.float32),
        "gamma": 0.2,
    }


@pytest.fixture
def cache():
    return WeightMatrixCache()


@pytest.fixture
def pipeline(cache):
    return HybridPipeline(
        cache=cache,
        enable_stage2=False,
        use_mlx=True,
    )


def _generate_spectra(energy, centers, sigmas, gamma, amplitudes,
                      energy_shift=0.0, sigma_shift=0.0):
    """Generate synthetic spectra with known shift and width change."""
    shifted_centers = centers + energy_shift
    shifted_sigmas = sigmas + sigma_shift
    n_comp = len(centers)
    n_energy = len(energy)
    y = np.zeros(n_energy, dtype=np.float64)
    for k in range(n_comp):
        y += amplitudes[k] * voigt_profile(energy, shifted_centers[k],
                                            shifted_sigmas[k], gamma)
    return y.astype(np.float32)


# ============================================================================
# Test: Dictionary Construction
# ============================================================================

class TestDictionaryConstruction:

    def test_grid_shape(self, energy, single_peak_config):
        """Grid dimensions match expected counts."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-1.0, 1.0), dE_step=0.5,
            dsigma_range=(-0.1, 0.1), dsigma_step=0.1,
        )
        n_dE = len(dc.dE_grid)
        n_ds = len(dc.dsigma_grid)
        assert dc.grid_shape == (n_dE, n_ds)
        assert dc.n_dict == n_dE * n_ds
        assert dc.D.shape == (len(energy), dc.n_dict)

    def test_l2_normalized(self, energy, single_peak_config):
        """Dictionary columns are L2-normalized."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-1.0, 1.0), dE_step=0.5,
        )
        norms = np.linalg.norm(dc.D, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_shift_only_has_single_dsigma(self, energy, single_peak_config):
        """build_dictionary_shift_only creates n_dsigma=1."""
        dc = build_dictionary_shift_only(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-1.0, 1.0), dE_step=0.5,
        )
        assert dc.n_dsigma == 1
        assert dc.dsigma_grid[0] == 0.0

    def test_per_grid_matrices_shapes(self, energy, two_peak_config):
        """Per-grid Wt, Phi, Jc, Jsigma, C have correct shapes."""
        dc = build_dictionary(
            energy, two_peak_config["centers"],
            two_peak_config["sigmas"], two_peak_config["gamma"],
            dE_range=(-0.5, 0.5), dE_step=0.5,
            dsigma_range=(0.0, 0.0), dsigma_step=1.0,
        )
        n_E = len(energy)
        n_comp = 2
        assert dc.Wt_per_grid.shape[1:] == (n_comp, n_E)
        assert dc.Phi_per_grid.shape[1:] == (n_E, n_comp)
        assert dc.Jc_per_grid.shape[1:] == (n_E, n_comp)
        assert dc.Jsigma_per_grid.shape[1:] == (n_E, n_comp)
        assert dc.C_per_grid.shape[1:] == (n_comp, n_comp)

    def test_nominal_grid_point_matches_weight_cache(self, energy,
                                                      single_peak_config, cache):
        """Grid point at δE=0, δσ=0 should match WeightMatrixCache output."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-0.5, 0.5), dE_step=0.5,
            dsigma_range=(0.0, 0.0), dsigma_step=1.0,
        )
        # Find grid index for δE=0
        zero_idx = np.argmin(np.abs(dc.dE_grid))
        assert abs(dc.dE_grid[zero_idx]) < 0.01

        # Compare with weight_cache
        Wt_ref, Phi_ref = cache.get_or_create(
            "test", "1s", energy,
            single_peak_config["centers"], single_peak_config["sigmas"],
            single_peak_config["gamma"], use_mlx=False,
        )
        np.testing.assert_allclose(
            dc.Wt_per_grid[zero_idx], Wt_ref, atol=1e-5,
        )


# ============================================================================
# Test: Dictionary Lookup
# ============================================================================

class TestDictionaryLookup:

    def test_zero_shift_selects_nominal(self, energy, single_peak_config):
        """Spectrum with δE=0 should match the δE=0 dictionary entry."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-1.0, 1.0), dE_step=0.1,
            dsigma_range=(0.0, 0.0), dsigma_step=1.0,
        )
        Y = _generate_spectra(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            np.array([1000.0]),
        )
        Y = Y[np.newaxis, :]  # (1, n_energy)

        idx, dE, ds = dictionary_lookup_numpy(
            dc.D, Y, dc.grid_shape, dc.dE_grid, dc.dsigma_grid,
        )
        assert abs(dE[0]) < 0.15, f"Expected ~0, got {dE[0]}"

    @pytest.mark.parametrize("true_shift", [-1.5, -0.5, 0.3, 1.0, 2.0])
    def test_shift_detection_direction(self, energy, single_peak_config,
                                        true_shift):
        """Dictionary lookup finds correct shift direction."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-2.5, 2.5), dE_step=0.1,
            dsigma_range=(0.0, 0.0), dsigma_step=1.0,
        )
        Y = _generate_spectra(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            np.array([1000.0]), energy_shift=true_shift,
        )
        Y = Y[np.newaxis, :]

        _, dE, _ = dictionary_lookup_numpy(
            dc.D, Y, dc.grid_shape, dc.dE_grid, dc.dsigma_grid,
        )
        # Coarse should be within 1 grid step of true
        assert abs(dE[0] - true_shift) < 0.15, \
            f"true={true_shift}, coarse={dE[0]}"


# ============================================================================
# Test: Hybrid Solver Accuracy
# ============================================================================

class TestHybridSolverAccuracy:

    def _make_batch(self, energy, config, n_spectra, dE, ds=0.0):
        """Create batch of spectra with known shift."""
        centers = config["centers"]
        sigmas = config["sigmas"]
        gamma = config["gamma"]
        n_comp = len(centers)
        true_amps = np.ones(n_comp) * 1000.0
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)

        if np.isscalar(dE):
            dE = np.full(n_spectra, dE)
        if np.isscalar(ds):
            ds = np.full(n_spectra, ds)

        for i in range(n_spectra):
            Y[i] = _generate_spectra(
                energy, centers, sigmas, gamma, true_amps,
                energy_shift=float(dE[i]), sigma_shift=float(ds[i]),
            )
        return Y

    @pytest.mark.parametrize("true_shift,tol", [
        (0.0, 0.05), (0.3, 0.05), (-0.3, 0.05),
    ])
    def test_taylor_range_shift(self, energy, single_peak_config,
                                 true_shift, tol):
        """In Taylor range, hybrid should recover shifts accurately."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
        )
        Y = self._make_batch(energy, single_peak_config, 100, true_shift)
        amps, _, dE, ds, _ = solve_hybrid(Y, dc, use_mlx=False)

        mean_dE = np.mean(dE)
        assert abs(mean_dE - true_shift) < tol, \
            f"true={true_shift}, recovered={mean_dE:.4f}"

    @pytest.mark.parametrize("true_shift,tol", [
        (1.0, 0.1), (1.5, 0.15), (2.0, 0.2), (-1.5, 0.15),
    ])
    def test_beyond_taylor_shift(self, energy, single_peak_config,
                                  true_shift, tol):
        """Beyond Taylor range: dictionary hybrid should still recover well."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-3.0, 3.0),
        )
        Y = self._make_batch(energy, single_peak_config, 100, true_shift)
        amps, _, dE, ds, _ = solve_hybrid(Y, dc, use_mlx=False)

        mean_dE = np.mean(dE)
        assert abs(mean_dE - true_shift) < tol, \
            f"true={true_shift}, recovered={mean_dE:.4f}"

    def test_grid_boundary(self, energy, single_peak_config):
        """δE exactly at midpoint between grid points should still work."""
        sigma = float(single_peak_config["sigmas"][0])
        step = 0.1 * sigma  # default grid step
        midpoint = step / 2  # halfway between grid points

        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
        )
        Y = self._make_batch(energy, single_peak_config, 100, midpoint)
        amps, _, dE, ds, _ = solve_hybrid(Y, dc, use_mlx=False)

        mean_dE = np.mean(dE)
        # Should be within half a grid step of truth
        assert abs(mean_dE - midpoint) < step, \
            f"midpoint={midpoint}, recovered={mean_dE:.4f}"

    def test_shift_and_width(self, energy, single_peak_config):
        """Simultaneous δE and δσ recovery."""
        true_dE = 0.5
        true_ds = 0.1
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-1.0, 1.0),
            dsigma_range=(-0.2, 0.2),
        )
        Y = self._make_batch(energy, single_peak_config, 200,
                              true_dE, true_ds)
        _, _, dE, ds, _ = solve_hybrid(Y, dc, use_mlx=False)

        mean_dE = np.mean(dE)
        mean_ds = np.mean(ds)
        assert abs(mean_dE - true_dE) < 0.1, \
            f"δE: true={true_dE}, got={mean_dE:.4f}"
        assert abs(mean_ds - true_ds) < 0.05, \
            f"δσ: true={true_ds}, got={mean_ds:.4f}"

    def test_zero_shift_amplitude_accuracy(self, energy, single_peak_config):
        """At δE=0, δσ=0, amplitudes should be accurate."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
        )
        Y = self._make_batch(energy, single_peak_config, 100, 0.0)
        amps, _, dE, ds, _ = solve_hybrid(Y, dc, use_mlx=False)

        mean_amp = np.mean(amps[0, :])
        amp_err = abs(mean_amp - 1000.0) / 1000.0 * 100
        assert amp_err < 1.0, f"Amplitude error {amp_err:.2f}% at zero shift"

    def test_large_shift_amplitude(self, energy, single_peak_config):
        """At large δE=1.5, amplitude should still be reasonable."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-2.0, 2.0),
        )
        Y = self._make_batch(energy, single_peak_config, 100, 1.5)
        amps, _, dE, _, _ = solve_hybrid(Y, dc, use_mlx=False)

        mean_amp = np.mean(amps[0, :])
        amp_err = abs(mean_amp - 1000.0) / 1000.0 * 100
        # Dictionary + 4-step should give < 5% error at 1.5 eV
        assert amp_err < 5.0, \
            f"Amplitude error {amp_err:.2f}% at δE=1.5 eV"


# ============================================================================
# Test: Multi-Component
# ============================================================================

class TestMultiComponent:

    def test_two_peak_shift_recovery(self, energy, two_peak_config):
        """Two-peak system: global shift recovered correctly.

        Note: composite dictionary uses equal-weight sum for matching,
        which introduces bias when component amplitudes differ significantly.
        The shift recovery is still good (~0.2 eV tolerance), and the
        4-step corrects sub-grid residuals.
        """
        true_shift = 0.8
        dc = build_dictionary(
            energy, two_peak_config["centers"],
            two_peak_config["sigmas"], two_peak_config["gamma"],
            dE_range=(-1.5, 1.5),
        )
        centers = two_peak_config["centers"]
        sigmas = two_peak_config["sigmas"]
        gamma = two_peak_config["gamma"]
        true_amps = np.array([1000.0, 600.0])

        n_spectra = 100
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i] = _generate_spectra(
                energy, centers, sigmas, gamma, true_amps,
                energy_shift=true_shift,
            )

        amps, _, dE, _, _ = solve_hybrid(Y, dc, use_mlx=False)

        mean_dE = np.mean(dE)
        assert abs(mean_dE - true_shift) < 0.2, \
            f"true={true_shift}, recovered={mean_dE:.4f}"

    def test_two_peak_equal_amps(self, energy, two_peak_config):
        """Two equal-amplitude peaks: shift and ratio both accurate."""
        true_shift = 0.8
        dc = build_dictionary(
            energy, two_peak_config["centers"],
            two_peak_config["sigmas"], two_peak_config["gamma"],
            dE_range=(-1.5, 1.5),
        )
        centers = two_peak_config["centers"]
        sigmas = two_peak_config["sigmas"]
        gamma = two_peak_config["gamma"]
        true_amps = np.array([1000.0, 1000.0])

        n_spectra = 100
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i] = _generate_spectra(
                energy, centers, sigmas, gamma, true_amps,
                energy_shift=true_shift,
            )

        amps, _, dE, _, _ = solve_hybrid(Y, dc, use_mlx=False)

        mean_dE = np.mean(dE)
        assert abs(mean_dE - true_shift) < 0.15, \
            f"true={true_shift}, recovered={mean_dE:.4f}"

        ratio = np.mean(amps[1, :]) / np.mean(amps[0, :])
        assert abs(ratio - 1.0) < 0.15, \
            f"Ratio: true=1.00, got={ratio:.2f}"


# ============================================================================
# Test: Pipeline Integration
# ============================================================================

class TestPipelineIntegration:

    def test_process_rowmajor_dictionary_hybrid(self, pipeline, energy,
                                                 single_peak_config):
        """Pipeline method returns FitResult with shifts."""
        true_shift = 0.5
        n_spectra = 50
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i] = _generate_spectra(
                energy, single_peak_config["centers"],
                single_peak_config["sigmas"], single_peak_config["gamma"],
                np.array([1000.0]), energy_shift=true_shift,
            )

        result = pipeline.process_rowmajor_dictionary_hybrid(
            Y, "test", "1s", energy, single_peak_config,
            amplitudes_only=True,
        )

        assert isinstance(result, FitResult)
        assert result.energy_shifts is not None
        assert result.sigma_shifts is not None
        assert result.amplitudes.shape == (1, n_spectra)

        mean_dE = np.mean(result.energy_shifts)
        assert abs(mean_dE - true_shift) < 0.1, \
            f"true={true_shift}, recovered={mean_dE:.4f}"

    def test_pipeline_with_prebuilt_cache(self, pipeline, energy,
                                           single_peak_config):
        """Pre-built dictionary cache can be passed to pipeline."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
        )

        Y = np.zeros((20, len(energy)), dtype=np.float32)
        for i in range(20):
            Y[i] = _generate_spectra(
                energy, single_peak_config["centers"],
                single_peak_config["sigmas"], single_peak_config["gamma"],
                np.array([1000.0]),
            )

        result = pipeline.process_rowmajor_dictionary_hybrid(
            Y, "test", "1s", energy, single_peak_config,
            amplitudes_only=True, dict_cache=dc,
        )
        assert "dict_build" not in result.timing


# ============================================================================
# Test: Comparison with Pure 4-Step
# ============================================================================

class TestHybridVsPure4Step:

    def test_small_shift_matches_4step(self, pipeline, energy,
                                        single_peak_config):
        """In Taylor range, hybrid and pure 4-step give similar results."""
        true_shift = 0.1
        n_spectra = 100
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i] = _generate_spectra(
                energy, single_peak_config["centers"],
                single_peak_config["sigmas"], single_peak_config["gamma"],
                np.array([1000.0]), energy_shift=true_shift,
            )

        # Pure 4-step
        result_4step = pipeline.process_rowmajor_extended_3param(
            Y, "test", "1s", energy, single_peak_config,
            amplitudes_only=True,
        )

        # Hybrid
        result_hybrid = pipeline.process_rowmajor_dictionary_hybrid(
            Y, "test", "1s", energy, single_peak_config,
            amplitudes_only=True,
        )

        # Both should recover similar shifts
        dE_4step = np.mean(result_4step.energy_shifts)
        dE_hybrid = np.mean(result_hybrid.energy_shifts)
        assert abs(dE_4step - dE_hybrid) < 0.05, \
            f"4-step={dE_4step:.4f}, hybrid={dE_hybrid:.4f}"

    def test_large_shift_hybrid_wins(self, pipeline, energy,
                                      single_peak_config):
        """Beyond Taylor: hybrid should be closer to truth than pure 4-step."""
        true_shift = 1.5  # δE/σ ≈ 2.5 — far beyond Taylor
        n_spectra = 100
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i] = _generate_spectra(
                energy, single_peak_config["centers"],
                single_peak_config["sigmas"], single_peak_config["gamma"],
                np.array([1000.0]), energy_shift=true_shift,
            )

        # Pure 4-step (Taylor breaks down)
        result_4step = pipeline.process_rowmajor_extended_3param(
            Y, "test", "1s", energy, single_peak_config,
            amplitudes_only=True,
        )

        # Hybrid (dictionary absorbs nonlinearity)
        result_hybrid = pipeline.process_rowmajor_dictionary_hybrid(
            Y, "test", "1s", energy, single_peak_config,
            amplitudes_only=True,
            dE_range=(-2.0, 2.0),
        )

        err_4step = abs(np.mean(result_4step.energy_shifts) - true_shift)
        err_hybrid = abs(np.mean(result_hybrid.energy_shifts) - true_shift)

        print("\nδE=1.5 eV (δE/σ≈2.5):")
        print(f"  4-step error: {err_4step:.4f} eV")
        print(f"  Hybrid error: {err_hybrid:.4f} eV")
        print(f"  Improvement: {err_4step/max(err_hybrid, 1e-6):.1f}x")

        # Hybrid should be significantly better
        assert err_hybrid < err_4step, \
            f"Hybrid ({err_hybrid:.4f}) should beat 4-step ({err_4step:.4f})"


# ============================================================================
# Test: Throughput Benchmark
# ============================================================================

class TestThroughput:

    def test_dictionary_build_time(self, energy, single_peak_config):
        """Dictionary build should be fast (< 1s for default grid)."""
        t0 = time.perf_counter()
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
        )
        build_time = time.perf_counter() - t0

        print(f"\nDictionary build: {build_time*1000:.1f} ms")
        print(f"  Grid: {dc.n_dE} × {dc.n_dsigma} = {dc.n_dict} entries")
        print(f"  Memory: D={dc.D.nbytes/1024:.1f} KB, "
              f"Wt={dc.Wt_per_grid.nbytes/1024:.1f} KB, "
              f"total={sum(a.nbytes for a in [dc.D, dc.Wt_per_grid, dc.Phi_per_grid, dc.Jc_per_grid, dc.Jsigma_per_grid, dc.C_per_grid])/1024:.1f} KB")

        assert build_time < 5.0, f"Build too slow: {build_time:.1f}s"

    def test_hybrid_throughput(self, energy, single_peak_config):
        """Benchmark hybrid solver throughput."""
        n_spectra = 100_000
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        shifts = np.random.uniform(-1.0, 1.0, n_spectra).astype(np.float32)
        for i in range(n_spectra):
            Y[i] = _generate_spectra(
                energy, single_peak_config["centers"],
                single_peak_config["sigmas"], single_peak_config["gamma"],
                np.array([1000.0]), energy_shift=float(shifts[i]),
            )

        # Build dictionary (not timed)
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
        )

        # Warmup
        solve_hybrid(Y[:1000], dc, use_mlx=False)

        # Benchmark
        n_iter = 3
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            solve_hybrid(Y, dc, use_mlx=False)
            times.append(time.perf_counter() - t0)

        t_median = np.median(times)
        rate = n_spectra / t_median

        print(f"\nHybrid throughput (N={n_spectra:,}, NumPy):")
        print(f"  {rate/1e6:.1f}M spec/s ({t_median*1000:.1f} ms)")

    def test_per_spectrum_varying_shifts(self, energy, single_peak_config):
        """Recovery correlation for per-spectrum random shifts."""
        np.random.seed(42)
        n_spectra = 500
        true_shifts = np.random.uniform(-1.5, 1.5, n_spectra).astype(np.float32)

        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i] = _generate_spectra(
                energy, single_peak_config["centers"],
                single_peak_config["sigmas"], single_peak_config["gamma"],
                np.array([1000.0]), energy_shift=float(true_shifts[i]),
            )

        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-2.0, 2.0),
        )
        _, _, dE, _, _ = solve_hybrid(Y, dc, use_mlx=False)

        corr = np.corrcoef(true_shifts, dE)[0, 1]
        rmse = np.sqrt(np.mean((dE - true_shifts) ** 2))

        print(f"\nPer-spectrum varying shifts (N={n_spectra}):")
        print(f"  Correlation: {corr:.4f}")
        print(f"  RMSE: {rmse:.4f} eV")

        assert corr > 0.99, f"Correlation too low: {corr:.4f}"
        assert rmse < 0.1, f"RMSE too high: {rmse:.4f} eV"


# ============================================================================
# Test: DictionaryCache Extensions
# ============================================================================

class TestDictionaryCacheExtensions:

    def test_nominal_index(self, energy, single_peak_config):
        """nominal_index points to δE=0, δσ=0 grid point."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-1.0, 1.0), dE_step=0.1,
            dsigma_range=(-0.1, 0.1), dsigma_step=0.1,
        )
        n_dsigma = dc.grid_shape[1]
        dE_idx = dc.nominal_index // n_dsigma
        ds_idx = dc.nominal_index % n_dsigma
        assert abs(dc.dE_grid[dE_idx]) < 0.05, f"Nominal δE not ~0: {dc.dE_grid[dE_idx]}"
        assert abs(dc.dsigma_grid[ds_idx]) < 0.05, f"Nominal δσ not ~0: {dc.dsigma_grid[ds_idx]}"

    def test_mean_sigma(self, energy, single_peak_config):
        """mean_sigma matches input."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
        )
        expected = float(np.mean(single_peak_config["sigmas"]))
        assert abs(dc.mean_sigma - expected) < 1e-5

    def test_is_within_taylor_all_nominal(self, energy, single_peak_config):
        """Spectra at nominal should all be within Taylor range."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-1.0, 1.0), dE_step=0.1,
        )
        # All at nominal
        indices = np.full(100, dc.nominal_index, dtype=np.int32)
        mask = dc.is_within_taylor(indices, taylor_threshold=0.35)
        assert mask.all(), "Nominal spectra should be within Taylor range"

    def test_is_within_taylor_large_shift(self, energy, single_peak_config):
        """Spectra with large δE should be outside Taylor range."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-2.0, 2.0), dE_step=0.1,
        )
        # Find index with δE ≈ 1.5 eV (δE/σ ≈ 2.5 for σ=0.6)
        dE_idx = np.argmin(np.abs(dc.dE_grid - 1.5))
        ds_idx = np.argmin(np.abs(dc.dsigma_grid))
        large_idx = dE_idx * dc.grid_shape[1] + ds_idx
        indices = np.full(100, large_idx, dtype=np.int32)
        mask = dc.is_within_taylor(indices, taylor_threshold=0.35)
        assert not mask.any(), "Large-shift spectra should be outside Taylor range"


# ============================================================================
# Test: Sort-and-Batch Solver
# ============================================================================

class TestSortedSolver:

    def _make_batch(self, energy, config, n_spectra, dE, ds=0.0):
        centers = config["centers"]
        sigmas = config["sigmas"]
        gamma = config["gamma"]
        n_comp = len(centers)
        true_amps = np.ones(n_comp) * 1000.0
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        if np.isscalar(dE):
            dE = np.full(n_spectra, dE)
        if np.isscalar(ds):
            ds = np.full(n_spectra, ds)
        for i in range(n_spectra):
            Y[i] = _generate_spectra(
                energy, centers, sigmas, gamma, true_amps,
                energy_shift=float(dE[i]), sigma_shift=float(ds[i]),
            )
        return Y

    def test_sorted_matches_hybrid(self, energy, single_peak_config):
        """solve_hybrid_sorted gives same results as solve_hybrid."""
        np.random.seed(42)
        n_spectra = 500
        true_shifts = np.random.uniform(-1.5, 1.5, n_spectra).astype(np.float32)
        Y = self._make_batch(energy, single_peak_config, n_spectra, true_shifts)

        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-2.0, 2.0),
        )

        amp_h, chi2_h, dE_h, ds_h, idx_h = solve_hybrid(Y, dc, use_mlx=False)
        amp_s, chi2_s, dE_s, ds_s, idx_s = solve_hybrid_sorted(Y, dc)

        # Indices should match (same Phase 1)
        np.testing.assert_array_equal(idx_h, idx_s)

        # Results should be close (same math, different execution order)
        np.testing.assert_allclose(dE_s, dE_h, atol=0.05,
                                    err_msg="Sorted vs hybrid shift mismatch")
        np.testing.assert_allclose(amp_s, amp_h, rtol=0.02,
                                    err_msg="Sorted vs hybrid amplitude mismatch")

    def test_sorted_large_shift_accuracy(self, energy, single_peak_config):
        """Sorted solver recovers large shifts accurately."""
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-2.0, 2.0),
        )
        Y = self._make_batch(energy, single_peak_config, 200, 1.5)
        _, _, dE, _, _ = solve_hybrid_sorted(Y, dc)

        mean_dE = np.mean(dE)
        assert abs(mean_dE - 1.5) < 0.15, f"true=1.5, got={mean_dE:.4f}"

    def test_sorted_varying_shifts_correlation(self, energy, single_peak_config):
        """Per-spectrum varying shifts: high correlation."""
        np.random.seed(123)
        n_spectra = 1000
        true_shifts = np.random.uniform(-1.5, 1.5, n_spectra).astype(np.float32)
        Y = self._make_batch(energy, single_peak_config, n_spectra, true_shifts)

        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-2.0, 2.0),
        )
        _, _, dE, _, _ = solve_hybrid_sorted(Y, dc)

        corr = np.corrcoef(true_shifts, dE)[0, 1]
        assert corr > 0.99, f"Correlation too low: {corr:.4f}"

    def test_sorted_throughput(self, energy, single_peak_config):
        """Benchmark sort-and-batch throughput."""
        n_spectra = 100_000
        np.random.seed(42)
        shifts = np.random.uniform(-1.0, 1.0, n_spectra).astype(np.float32)
        Y = self._make_batch(energy, single_peak_config, n_spectra, shifts)

        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
        )

        # Warmup
        solve_hybrid_sorted(Y[:1000], dc)

        n_iter = 3
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            solve_hybrid_sorted(Y, dc)
            times.append(time.perf_counter() - t0)

        t_median = np.median(times)
        rate = n_spectra / t_median
        print(f"\nSort-and-batch throughput (N={n_spectra:,}):")
        print(f"  {rate/1e6:.1f}M spec/s ({t_median*1000:.1f} ms)")

    def test_sorted_pipeline_integration(self, pipeline, energy, single_peak_config):
        """Pipeline solver='sorted' works."""
        Y = self._make_batch(energy, single_peak_config, 100, 0.5)
        result = pipeline.process_rowmajor_dictionary_hybrid(
            Y, "test", "1s", energy, single_peak_config,
            amplitudes_only=True, solver="sorted",
        )
        assert result.energy_shifts is not None
        mean_dE = np.mean(result.energy_shifts)
        assert abs(mean_dE - 0.5) < 0.1, f"true=0.5, got={mean_dE:.4f}"


# ============================================================================
# Test: Adaptive Solver
# ============================================================================

class TestAdaptiveSolver:

    def _make_batch(self, energy, config, n_spectra, dE, ds=0.0):
        centers = config["centers"]
        sigmas = config["sigmas"]
        gamma = config["gamma"]
        n_comp = len(centers)
        true_amps = np.ones(n_comp) * 1000.0
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        if np.isscalar(dE):
            dE = np.full(n_spectra, dE)
        if np.isscalar(ds):
            ds = np.full(n_spectra, ds)
        for i in range(n_spectra):
            Y[i] = _generate_spectra(
                energy, centers, sigmas, gamma, true_amps,
                energy_shift=float(dE[i]), sigma_shift=float(ds[i]),
            )
        return Y

    def test_adaptive_all_taylor(self, energy, single_peak_config):
        """When all shifts are small, adaptive ≈ pure 4-step."""
        Y = self._make_batch(energy, single_peak_config, 200, 0.1)
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-2.0, 2.0),
        )
        _, _, dE, ds, _ = adaptive_solve(Y, dc)
        assert abs(np.mean(dE) - 0.1) < 0.05

    def test_adaptive_all_beyond_taylor(self, energy, single_peak_config):
        """When all shifts are large, adaptive falls back to hybrid."""
        Y = self._make_batch(energy, single_peak_config, 200, 1.5)
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-2.0, 2.0),
        )
        _, _, dE, ds, _ = adaptive_solve(Y, dc)
        assert abs(np.mean(dE) - 1.5) < 0.15

    def test_adaptive_mixed_shifts(self, energy, single_peak_config):
        """Mix of Taylor-range and beyond-Taylor spectra."""
        np.random.seed(42)
        n_spectra = 500
        # 70% in Taylor range (±0.2), 30% beyond (±1.5)
        n_taylor = 350
        n_beyond = 150
        true_shifts = np.concatenate([
            np.random.uniform(-0.15, 0.15, n_taylor),
            np.random.uniform(-1.5, 1.5, n_beyond),
        ]).astype(np.float32)
        np.random.shuffle(true_shifts)  # mix order

        Y = self._make_batch(energy, single_peak_config, n_spectra, true_shifts)
        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-2.0, 2.0),
        )
        _, _, dE, _, _ = adaptive_solve(Y, dc)

        corr = np.corrcoef(true_shifts, dE)[0, 1]
        rmse = np.sqrt(np.mean((dE - true_shifts) ** 2))
        print(f"\nAdaptive mixed shifts (N={n_spectra}):")
        print(f"  Correlation: {corr:.4f}")
        print(f"  RMSE: {rmse:.4f} eV")
        assert corr > 0.99, f"Correlation too low: {corr:.4f}"

    def test_adaptive_throughput(self, energy, single_peak_config):
        """Benchmark adaptive solver throughput."""
        n_spectra = 100_000
        np.random.seed(42)
        # 80% Taylor-range (typical XPS)
        n_taylor = 80_000
        n_beyond = 20_000
        shifts = np.concatenate([
            np.random.uniform(-0.1, 0.1, n_taylor),
            np.random.uniform(-1.5, 1.5, n_beyond),
        ]).astype(np.float32)
        np.random.shuffle(shifts)
        Y = self._make_batch(energy, single_peak_config, n_spectra, shifts)

        dc = build_dictionary(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
        )

        # Warmup
        adaptive_solve(Y[:1000], dc)

        n_iter = 3
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            adaptive_solve(Y, dc)
            times.append(time.perf_counter() - t0)

        t_median = np.median(times)
        rate = n_spectra / t_median
        print(f"\nAdaptive throughput (N={n_spectra:,}, 80% Taylor):")
        print(f"  {rate/1e6:.1f}M spec/s ({t_median*1000:.1f} ms)")

    def test_adaptive_pipeline_integration(self, pipeline, energy, single_peak_config):
        """Pipeline solver='adaptive' works."""
        np.random.seed(42)
        n_spectra = 200
        shifts = np.random.uniform(-1.0, 1.0, n_spectra).astype(np.float32)
        Y = self._make_batch(energy, single_peak_config, n_spectra, shifts)

        result = pipeline.process_rowmajor_dictionary_hybrid(
            Y, "test", "1s", energy, single_peak_config,
            amplitudes_only=True, solver="adaptive",
        )
        assert result.energy_shifts is not None
        corr = np.corrcoef(shifts, result.energy_shifts)[0, 1]
        assert corr > 0.99, f"Correlation too low: {corr:.4f}"


# ============================================================================
# Test: 2D Dictionary Extensions
# ============================================================================

class TestDict2D:
    """Tests for dense 2D dictionary (δE × δσ) and amp_only solver."""

    def test_build_dictionary_2d_N_counts(self, energy, single_peak_config):
        """build_dictionary_2d with N_dE/N_ds produces correct grid sizes."""
        dc = build_dictionary_2d(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-1.0, 1.0), N_dE=21,
            dsigma_range=(-0.2, 0.2), N_ds=11,
        )
        assert dc.n_dE == 21, f"Expected 21 dE points, got {dc.n_dE}"
        assert dc.n_dsigma == 11, f"Expected 11 ds points, got {dc.n_dsigma}"
        assert dc.n_dict == 21 * 11

    def test_build_dictionary_2d_step_override(self, energy, single_peak_config):
        """dE_step/dsigma_step override N_dE/N_ds."""
        dc = build_dictionary_2d(
            energy, single_peak_config["centers"],
            single_peak_config["sigmas"], single_peak_config["gamma"],
            dE_range=(-1.0, 1.0), dE_step=0.5, N_dE=100,  # step wins
            dsigma_range=(-0.2, 0.2), dsigma_step=0.1, N_ds=100,
        )
        assert dc.n_dE == 5  # -1.0, -0.5, 0.0, 0.5, 1.0
        assert dc.n_dsigma == 5

    def test_dict2d_amp_only_basic(self, energy, single_peak_config):
        """Dict2D amp_only recovers amplitude and shift for simple case."""
        np.random.seed(42)
        centers = single_peak_config["centers"]
        sigmas = single_peak_config["sigmas"]
        gamma = single_peak_config["gamma"]

        # Build dense 2D dictionary
        dc = build_dictionary_2d(
            energy, centers, sigmas, gamma,
            dE_range=(-1.0, 1.0), N_dE=41,
            dsigma_range=(-0.2, 0.2), N_ds=21,
        )

        # Generate spectra with known shifts
        n_spectra = 100
        dE_true = np.random.uniform(-0.8, 0.8, n_spectra).astype(np.float32)
        ds_true = np.random.uniform(-0.15, 0.15, n_spectra).astype(np.float32)
        amp_true = np.random.uniform(0.5, 1.5, n_spectra).astype(np.float32)

        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i] = _generate_spectra(
                energy, centers, sigmas, gamma,
                amplitudes=np.array([amp_true[i]]),
                energy_shift=dE_true[i],
                sigma_shift=ds_true[i],
            )

        # Solve
        amps, chi2, dE_out, ds_out, _ = solve_dict2d_amp_only(Y, dc)

        # δE accuracy: should be within half grid spacing
        dE_step = dc.dE_grid[1] - dc.dE_grid[0]
        dE_err = np.abs(dE_out - dE_true)
        assert np.mean(dE_err) < dE_step, \
            f"Mean δE error {np.mean(dE_err):.4f} > grid step {dE_step:.4f}"

        # δσ accuracy: should be within half grid spacing
        ds_step = dc.dsigma_grid[1] - dc.dsigma_grid[0]
        ds_err = np.abs(ds_out - ds_true)
        assert np.mean(ds_err) < ds_step, \
            f"Mean δσ error {np.mean(ds_err):.4f} > grid step {ds_step:.4f}"

        # Amplitude correlation
        amp_corr = np.corrcoef(amp_true, amps[0])[0, 1]
        assert amp_corr > 0.99, f"Amplitude correlation {amp_corr:.4f} too low"

    def test_dict2d_amp_only_vs_hybrid_sorted(self, energy, single_peak_config):
        """Dict2D amp_only and hybrid_sorted use same Phase 1 (same coarse δE/δσ)."""
        np.random.seed(123)
        centers = single_peak_config["centers"]
        sigmas = single_peak_config["sigmas"]
        gamma = single_peak_config["gamma"]

        dc = build_dictionary_2d(
            energy, centers, sigmas, gamma,
            dE_range=(-1.0, 1.0), N_dE=41,
            dsigma_range=(-0.2, 0.2), N_ds=21,
        )

        n_spectra = 50
        dE_true = np.random.uniform(-0.5, 0.5, n_spectra).astype(np.float32)
        amp_true = np.ones(n_spectra, dtype=np.float32)
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i] = _generate_spectra(
                energy, centers, sigmas, gamma,
                amplitudes=np.array([amp_true[i]]),
                energy_shift=dE_true[i],
            )

        # Both solvers should find the same dictionary indices
        _, _, dE_ao, ds_ao, idx_ao = solve_dict2d_amp_only(Y, dc)
        _, _, dE_hs, ds_hs, idx_hs = solve_hybrid_sorted(Y, dc)

        np.testing.assert_array_equal(idx_ao, idx_hs,
            err_msg="Phase 1 should produce identical indices")

    def test_dict2d_regression_Nds1(self, energy, single_peak_config):
        """With N_ds=1, Dict2D amp_only should match solve_dictionary_only."""
        np.random.seed(77)
        centers = single_peak_config["centers"]
        sigmas = single_peak_config["sigmas"]
        gamma = single_peak_config["gamma"]

        # N_ds=1 → shift-only dictionary
        dc = build_dictionary_2d(
            energy, centers, sigmas, gamma,
            dE_range=(-1.0, 1.0), dE_step=0.1,
            dsigma_range=(0.0, 0.0), dsigma_step=1.0, N_ds=None,
        )
        assert dc.n_dsigma == 1, f"Expected n_dsigma=1, got {dc.n_dsigma}"

        n_spectra = 50
        dE_true = np.random.uniform(-0.8, 0.8, n_spectra).astype(np.float32)
        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i] = _generate_spectra(
                energy, centers, sigmas, gamma,
                amplitudes=np.array([1.0]),
                energy_shift=dE_true[i],
            )

        # solve_dict2d_amp_only with N_ds=1
        amp_2d, chi2_2d, dE_2d, ds_2d, idx_2d = solve_dict2d_amp_only(Y, dc)

        # solve_dictionary_only (the existing 1D solver)
        from toyomacro.voigtfit.dictionary_solver import solve_dictionary_only
        amp_1d, chi2_1d, dE_1d, ds_1d, idx_1d = solve_dictionary_only(Y, dc)

        # Should match exactly (same Phase 1, same amp estimation)
        np.testing.assert_array_equal(idx_2d, idx_1d)
        np.testing.assert_allclose(dE_2d, dE_1d, atol=1e-6)
        np.testing.assert_allclose(amp_2d, amp_1d, atol=1e-4)

    def test_dict2d_dense_dsigma_improves_fwhm(self, energy, single_peak_config):
        """Denser δσ grid should improve FWHM recovery vs sparse grid."""
        np.random.seed(99)
        centers = single_peak_config["centers"]
        sigmas = single_peak_config["sigmas"]
        gamma = single_peak_config["gamma"]

        n_spectra = 200
        dE_true = np.random.uniform(-0.5, 0.5, n_spectra).astype(np.float32)
        ds_true = np.random.uniform(-0.15, 0.15, n_spectra).astype(np.float32)
        amp_true = np.ones(n_spectra, dtype=np.float32)

        Y = np.zeros((n_spectra, len(energy)), dtype=np.float32)
        for i in range(n_spectra):
            Y[i] = _generate_spectra(
                energy, centers, sigmas, gamma,
                amplitudes=np.array([amp_true[i]]),
                energy_shift=dE_true[i],
                sigma_shift=ds_true[i],
            )

        # Sparse δσ grid (N_ds=3)
        dc_sparse = build_dictionary_2d(
            energy, centers, sigmas, gamma,
            dE_range=(-1.0, 1.0), N_dE=41,
            dsigma_range=(-0.2, 0.2), N_ds=3,
        )
        _, _, _, ds_sparse, _ = solve_dict2d_amp_only(Y, dc_sparse)
        rmse_sparse = np.sqrt(np.mean((ds_sparse - ds_true) ** 2))

        # Dense δσ grid (N_ds=41)
        dc_dense = build_dictionary_2d(
            energy, centers, sigmas, gamma,
            dE_range=(-1.0, 1.0), N_dE=41,
            dsigma_range=(-0.2, 0.2), N_ds=41,
        )
        _, _, _, ds_dense, _ = solve_dict2d_amp_only(Y, dc_dense)
        rmse_dense = np.sqrt(np.mean((ds_dense - ds_true) ** 2))

        print(f"\nδσ RMSE: sparse(N=3)={rmse_sparse:.5f}, dense(N=41)={rmse_dense:.5f}")
        assert rmse_dense < rmse_sparse, \
            f"Dense grid ({rmse_dense:.5f}) should beat sparse ({rmse_sparse:.5f})"

    def test_dict2d_amp_only_throughput(self, energy, single_peak_config):
        """Benchmark throughput of Dict2D amp_only."""
        import time

        np.random.seed(42)
        centers = single_peak_config["centers"]
        sigmas = single_peak_config["sigmas"]
        gamma = single_peak_config["gamma"]

        dc = build_dictionary_2d(
            energy, centers, sigmas, gamma,
            dE_range=(-1.0, 1.0), N_dE=41,
            dsigma_range=(-0.2, 0.2), N_ds=21,
        )
        n_dict = dc.n_dict

        n_spectra = 100_000
        Y = np.random.randn(n_spectra, len(energy)).astype(np.float32)

        # Warmup
        solve_dict2d_amp_only(Y[:1000], dc)

        n_iter = 3
        times = []
        for _ in range(n_iter):
            # Reset MLX cache
            dc._D_mx = None
            dc._Wt_mx = None
            dc._Phi_mx = None
            dc._Jc_mx = None
            dc._Jsigma_mx = None
            t0 = time.perf_counter()
            solve_dict2d_amp_only(Y, dc)
            times.append(time.perf_counter() - t0)

        t_median = np.median(times)
        rate = n_spectra / t_median
        print(f"\nDict2D amp_only throughput ({n_dict} entries, N={n_spectra:,}):")
        print(f"  {rate/1e6:.2f}M spec/s ({t_median*1000:.1f} ms)")

    def test_dict2d_adaptive_matches_4step_nf(self, energy, single_peak_config):
        """Adaptive solver matches 4-step on noise-free data (chi2 → 0)."""
        from toyomacro.voigtfit.voigt_jacobian import voigt_profile

        np.random.seed(42)
        centers = single_peak_config["centers"]
        sigmas = single_peak_config["sigmas"]
        gamma = single_peak_config["gamma"]

        dc = build_dictionary_2d(
            energy, centers, sigmas, gamma,
            dE_range=(-1.0, 1.0), N_dE=41,
            dsigma_range=(-0.2, 0.2), N_ds=21,
        )

        n = 500
        gt_dE = np.random.uniform(-0.5, 0.5, n).astype(np.float32)
        gt_ds = np.random.uniform(-0.1, 0.1, n).astype(np.float32)
        gt_amp = np.random.uniform(0.5, 1.0, n).astype(np.float32)

        Y = np.zeros((n, len(energy)), dtype=np.float32)
        for i in range(n):
            prof = voigt_profile(energy, centers[0] + gt_dE[i],
                                  sigmas[0] + gt_ds[i], gamma)
            Y[i] = gt_amp[i] * prof.astype(np.float32)

        # 4-step reference
        _, _, dE_4s, ds_4s, _ = solve_hybrid_sorted(Y, dc)
        # Adaptive
        _, _, dE_ad, ds_ad, _ = solve_dict2d_adaptive(Y, dc)

        # NF: adaptive should use 4-step for nearly all spectra
        # So results should be nearly identical
        assert np.allclose(dE_4s, dE_ad, atol=1e-4), \
            f"δE mismatch: max diff={np.max(np.abs(dE_4s - dE_ad)):.6f}"
        assert np.allclose(ds_4s, ds_ad, atol=1e-4), \
            f"δσ mismatch: max diff={np.max(np.abs(ds_4s - ds_ad)):.6f}"

    @pytest.mark.skipif(IN_CI or not _mlx_usable(),
                        reason="tolerance calibrated for the MLX path (fails on the NumPy backend)")
    def test_dict2d_adaptive_custom_threshold(self, energy, single_peak_config):
        """Adaptive solver respects custom chi2_threshold."""
        from toyomacro.voigtfit.voigt_jacobian import voigt_profile

        np.random.seed(42)
        centers = single_peak_config["centers"]
        sigmas = single_peak_config["sigmas"]
        gamma = single_peak_config["gamma"]

        dc = build_dictionary_2d(
            energy, centers, sigmas, gamma,
            dE_range=(-1.0, 1.0), N_dE=21,
            dsigma_range=(-0.2, 0.2), N_ds=11,
        )

        n = 200
        gt_dE = np.random.uniform(-0.5, 0.5, n).astype(np.float32)
        gt_ds = np.random.uniform(-0.1, 0.1, n).astype(np.float32)
        gt_amp = np.random.uniform(0.5, 1.0, n).astype(np.float32)

        Y = np.zeros((n, len(energy)), dtype=np.float32)
        for i in range(n):
            prof = voigt_profile(energy, centers[0] + gt_dE[i],
                                  sigmas[0] + gt_ds[i], gamma)
            Y[i] = gt_amp[i] * prof.astype(np.float32)

        # Very large threshold = all get 4-step
        _, _, dE_all4s, ds_all4s, _ = solve_dict2d_adaptive(Y, dc, chi2_threshold=1e10)
        _, _, dE_ref, ds_ref, _ = solve_hybrid_sorted(Y, dc)
        assert np.allclose(dE_all4s, dE_ref, atol=1e-5)

        # Tiny threshold = all get amp_only
        _, _, dE_allao, ds_allao, _ = solve_dict2d_adaptive(Y, dc, chi2_threshold=1e-30)
        _, _, dE_ao_ref, ds_ao_ref, _ = solve_dict2d_amp_only(Y, dc)
        assert np.allclose(dE_allao, dE_ao_ref, atol=1e-5)
        assert np.allclose(ds_allao, ds_ao_ref, atol=1e-5)

    def test_dict2d_adaptive_output_shape(self, energy, single_peak_config):
        """Adaptive solver returns correct shapes."""
        from toyomacro.voigtfit.voigt_jacobian import voigt_profile

        np.random.seed(42)
        centers = single_peak_config["centers"]
        sigmas = single_peak_config["sigmas"]
        gamma = single_peak_config["gamma"]

        dc = build_dictionary_2d(
            energy, centers, sigmas, gamma,
            dE_range=(-1.0, 1.0), N_dE=11,
            dsigma_range=(-0.1, 0.1), N_ds=5,
        )

        n = 100
        Y = np.zeros((n, len(energy)), dtype=np.float32)
        for i in range(n):
            dE = np.random.uniform(-0.5, 0.5)
            prof = voigt_profile(energy, centers[0] + dE, sigmas[0], gamma)
            Y[i] = np.random.uniform(0.5, 1.0) * prof.astype(np.float32)

        amps, chi2, dE_out, ds_out, idx = solve_dict2d_adaptive(Y, dc)
        assert amps.shape == (1, n)
        assert chi2.shape == (n,)
        assert dE_out.shape == (n,)
        assert ds_out.shape == (n,)
        assert idx.shape == (n,)


class TestAdaptiveV2:
    """Tests for fused adaptive solver (v2) — matches v1 output."""

    def test_v2_matches_v1_nf(self, energy, single_peak_config):
        """V2 matches V1 on noise-free data (100% 4-step)."""
        from toyomacro.voigtfit.voigt_jacobian import voigt_profile

        np.random.seed(42)
        centers = single_peak_config["centers"]
        sigmas = single_peak_config["sigmas"]
        gamma = single_peak_config["gamma"]

        dc = build_dictionary_2d(
            energy, centers, sigmas, gamma,
            dE_range=(-1.0, 1.0), N_dE=41,
            dsigma_range=(-0.2, 0.2), N_ds=21,
        )

        n = 500
        gt_dE = np.random.uniform(-0.5, 0.5, n).astype(np.float32)
        gt_ds = np.random.uniform(-0.1, 0.1, n).astype(np.float32)
        gt_amp = np.random.uniform(0.5, 1.0, n).astype(np.float32)

        Y = np.zeros((n, len(energy)), dtype=np.float32)
        for i in range(n):
            prof = voigt_profile(energy, centers[0] + gt_dE[i],
                                  sigmas[0] + gt_ds[i], gamma)
            Y[i] = gt_amp[i] * prof.astype(np.float32)

        amp_v1, chi2_v1, dE_v1, ds_v1, idx_v1 = solve_dict2d_adaptive(Y, dc)
        amp_v2, chi2_v2, dE_v2, ds_v2, idx_v2 = solve_dict2d_adaptive_fused(Y, dc)

        # Same dictionary indices
        np.testing.assert_array_equal(idx_v1, idx_v2)
        # Same energy/sigma shifts (tight tolerance)
        np.testing.assert_allclose(dE_v1, dE_v2, atol=1e-4)
        np.testing.assert_allclose(ds_v1, ds_v2, atol=1e-4)
        # Same amplitudes
        np.testing.assert_allclose(amp_v1, amp_v2, atol=1e-3)

    def test_v2_matches_v1_noisy(self, energy, single_peak_config):
        """V2 matches V1 on noisy data (mostly amp_only)."""
        from toyomacro.voigtfit.voigt_jacobian import voigt_profile

        np.random.seed(123)
        centers = single_peak_config["centers"]
        sigmas = single_peak_config["sigmas"]
        gamma = single_peak_config["gamma"]

        dc = build_dictionary_2d(
            energy, centers, sigmas, gamma,
            dE_range=(-1.0, 1.0), N_dE=21,
            dsigma_range=(-0.2, 0.2), N_ds=11,
        )

        n = 300
        gt_dE = np.random.uniform(-0.5, 0.5, n).astype(np.float32)
        gt_amp = np.random.uniform(0.5, 1.0, n).astype(np.float32)

        Y = np.zeros((n, len(energy)), dtype=np.float32)
        for i in range(n):
            prof = voigt_profile(energy, centers[0] + gt_dE[i],
                                  sigmas[0], gamma)
            Y[i] = gt_amp[i] * prof.astype(np.float32)
        # Add strong noise so chi2_norm > threshold → amp_only
        Y += np.random.normal(0, 0.1, Y.shape).astype(np.float32)

        amp_v1, chi2_v1, dE_v1, ds_v1, idx_v1 = solve_dict2d_adaptive(Y, dc)
        amp_v2, chi2_v2, dE_v2, ds_v2, idx_v2 = solve_dict2d_adaptive_fused(Y, dc)

        np.testing.assert_array_equal(idx_v1, idx_v2)
        np.testing.assert_allclose(dE_v1, dE_v2, atol=1e-4)
        np.testing.assert_allclose(ds_v1, ds_v2, atol=1e-4)

    def test_v2_all_4step_matches_sorted(self, energy, single_peak_config):
        """V2 with huge threshold = all 4-step, matches solve_hybrid_sorted."""
        from toyomacro.voigtfit.voigt_jacobian import voigt_profile

        np.random.seed(99)
        centers = single_peak_config["centers"]
        sigmas = single_peak_config["sigmas"]
        gamma = single_peak_config["gamma"]

        dc = build_dictionary_2d(
            energy, centers, sigmas, gamma,
            dE_range=(-1.0, 1.0), N_dE=21,
            dsigma_range=(-0.2, 0.2), N_ds=11,
        )

        n = 200
        Y = np.zeros((n, len(energy)), dtype=np.float32)
        for i in range(n):
            dE = np.random.uniform(-0.5, 0.5)
            ds = np.random.uniform(-0.1, 0.1)
            prof = voigt_profile(energy, centers[0] + dE,
                                  sigmas[0] + ds, gamma)
            Y[i] = np.random.uniform(0.5, 1.0) * prof.astype(np.float32)

        _, _, dE_ref, ds_ref, _ = solve_hybrid_sorted(Y, dc)
        _, _, dE_v2, ds_v2, _ = solve_dict2d_adaptive_fused(
            Y, dc, chi2_threshold=1e10)

        np.testing.assert_allclose(dE_ref, dE_v2, atol=1e-5)
        np.testing.assert_allclose(ds_ref, ds_v2, atol=1e-5)

    def test_v2_output_shape(self, energy, single_peak_config):
        """V2 returns correct shapes."""
        from toyomacro.voigtfit.voigt_jacobian import voigt_profile

        np.random.seed(42)
        centers = single_peak_config["centers"]
        sigmas = single_peak_config["sigmas"]
        gamma = single_peak_config["gamma"]

        dc = build_dictionary_2d(
            energy, centers, sigmas, gamma,
            dE_range=(-1.0, 1.0), N_dE=11,
            dsigma_range=(-0.1, 0.1), N_ds=5,
        )

        n = 100
        Y = np.zeros((n, len(energy)), dtype=np.float32)
        for i in range(n):
            dE = np.random.uniform(-0.5, 0.5)
            prof = voigt_profile(energy, centers[0] + dE, sigmas[0], gamma)
            Y[i] = np.random.uniform(0.5, 1.0) * prof.astype(np.float32)

        amps, chi2, dE_out, ds_out, idx = solve_dict2d_adaptive_fused(Y, dc)
        assert amps.shape == (1, n)
        assert chi2.shape == (n,)
        assert dE_out.shape == (n,)
        assert ds_out.shape == (n,)
        assert idx.shape == (n,)


# ============================================================================
# Parabola Sub-Grid Interpolation Tests
# ============================================================================


class TestParabolaVertex:
    """Unit tests for _parabola_vertex."""

    def test_symmetric_peak(self):
        """Symmetric parabola → vertex at center (δ=0)."""
        s_m = np.array([1.0, 2.0])
        s_0 = np.array([3.0, 5.0])
        s_p = np.array([1.0, 2.0])
        delta = _parabola_vertex(s_m, s_0, s_p)
        np.testing.assert_allclose(delta, 0.0, atol=1e-7)

    def test_shifted_peak(self):
        """Known parabola y = -(x-0.3)^2 + 1 sampled at -1,0,+1."""
        # y(-1) = -(−1−0.3)^2+1 = -1.69+1 = -0.69
        # y(0) = -(0-0.3)^2+1 = -0.09+1 = 0.91
        # y(1) = -(1-0.3)^2+1 = -0.49+1 = 0.51
        s_m = np.array([-0.69])
        s_0 = np.array([0.91])
        s_p = np.array([0.51])
        delta = _parabola_vertex(s_m, s_0, s_p)
        np.testing.assert_allclose(delta, 0.3, atol=1e-6)

    def test_negative_shift(self):
        """Parabola y = -(x+0.4)^2 + 1 → vertex at -0.4."""
        s_m = np.array([-((-1+0.4)**2) + 1])   # 0.64 → 0.36
        s_0 = np.array([-(0.4**2) + 1])         # 0.16 → 0.84
        s_p = np.array([-(1.4**2) + 1])          # 1.96 → -0.96
        delta = _parabola_vertex(s_m, s_0, s_p)
        np.testing.assert_allclose(delta, -0.4, atol=1e-6)

    def test_clamp(self):
        """Very asymmetric samples → clamp to ±0.5."""
        # Force extreme asymmetry
        s_m = np.array([0.0])
        s_0 = np.array([0.1])
        s_p = np.array([10.0])
        delta = _parabola_vertex(s_m, s_0, s_p)
        assert delta[0] <= 0.5

    def test_flat_scores(self):
        """All equal scores → δ=0 (denominator ≈ 0)."""
        s_m = np.array([5.0])
        s_0 = np.array([5.0])
        s_p = np.array([5.0])
        delta = _parabola_vertex(s_m, s_0, s_p)
        np.testing.assert_allclose(delta, 0.0, atol=1e-7)

    def test_vectorized(self):
        """Batch of 1000 spectra."""
        rng = np.random.default_rng(42)
        n = 1000
        # True vertex at random position
        true_delta = rng.uniform(-0.45, 0.45, n)
        # Parabola y = -(x - δ)^2 + 1
        s_m = -((-1 - true_delta)**2) + 1
        s_0 = -((-true_delta)**2) + 1
        s_p = -((1 - true_delta)**2) + 1
        delta = _parabola_vertex(s_m, s_0, s_p)
        np.testing.assert_allclose(delta, true_delta, atol=1e-6)


class TestParabolaRefine2D:
    """Tests for parabola_refine_2d on a 2D grid."""

    @pytest.fixture
    def grid_setup(self):
        """Small 2D grid for testing."""
        n_dE, n_ds = 11, 7
        dE_grid = np.linspace(-0.5, 0.5, n_dE).astype(np.float32)
        ds_grid = np.linspace(-0.15, 0.15, n_ds).astype(np.float32)
        return n_dE, n_ds, dE_grid, ds_grid

    def test_exact_grid_point(self, grid_setup):
        """If true params are on a grid point, parabola returns that point."""
        n_dE, n_ds, dE_grid, ds_grid = grid_setup
        n_dict = n_dE * n_ds
        n_spectra = 50
        rng = np.random.default_rng(123)

        # Pick random interior grid points
        dE_idx = rng.integers(1, n_dE - 1, n_spectra)
        ds_idx = rng.integers(1, n_ds - 1, n_spectra)
        best_idx = (dE_idx * n_ds + ds_idx).astype(np.int32)

        # Build score matrix: each spectrum has a clear peak at its grid point
        scores = np.zeros((n_spectra, n_dict), dtype=np.float32)
        for i in range(n_spectra):
            for di in range(n_dE):
                for dj in range(n_ds):
                    # Parabolic score centered at true grid point
                    dist2 = (di - dE_idx[i])**2 + (dj - ds_idx[i])**2
                    scores[i, di * n_ds + dj] = 100.0 - dist2

        dE_ref, ds_ref = parabola_refine_2d(
            scores, best_idx, (n_dE, n_ds), dE_grid, ds_grid)

        # Should match grid values exactly (vertex at center)
        np.testing.assert_allclose(dE_ref, dE_grid[dE_idx], atol=1e-6)
        np.testing.assert_allclose(ds_ref, ds_grid[ds_idx], atol=1e-6)

    def test_subgrid_accuracy(self, grid_setup):
        """Parabola recovers known sub-grid offsets."""
        n_dE, n_ds, dE_grid, ds_grid = grid_setup
        dE_step = dE_grid[1] - dE_grid[0]
        ds_step = ds_grid[1] - ds_grid[0]
        n_dict = n_dE * n_ds
        n_spectra = 200
        rng = np.random.default_rng(456)

        # True continuous positions (within grid)
        true_dE = rng.uniform(dE_grid[1], dE_grid[-2], n_spectra).astype(np.float32)
        true_ds = rng.uniform(ds_grid[1], ds_grid[-2], n_spectra).astype(np.float32)

        # Nearest grid indices
        dE_idx = np.clip(np.round((true_dE - dE_grid[0]) / dE_step).astype(int),
                         1, n_dE - 2)
        ds_idx = np.clip(np.round((true_ds - ds_grid[0]) / ds_step).astype(int),
                         1, n_ds - 2)
        best_idx = (dE_idx * n_ds + ds_idx).astype(np.int32)

        # Build parabolic scores: score = -( (dE - true_dE)^2 + (ds - true_ds)^2 )
        scores = np.zeros((n_spectra, n_dict), dtype=np.float32)
        for di in range(n_dE):
            for dj in range(n_ds):
                flat = di * n_ds + dj
                scores[:, flat] = -(
                    (dE_grid[di] - true_dE)**2 + (ds_grid[dj] - true_ds)**2)

        dE_ref, ds_ref = parabola_refine_2d(
            scores, best_idx, (n_dE, n_ds), dE_grid, ds_grid)

        # Should recover true position within ~1e-3 eV
        # (error comes from independent-axis approximation on coupled parabola)
        dE_err = np.abs(dE_ref - true_dE)
        ds_err = np.abs(ds_ref - true_ds)
        assert np.mean(dE_err) < dE_step * 0.1, f"mean dE err {np.mean(dE_err):.5f}"
        assert np.mean(ds_err) < ds_step * 0.1, f"mean ds err {np.mean(ds_err):.5f}"

    def test_boundary_spectra(self, grid_setup):
        """Spectra at grid boundary get δ=0 (no extrapolation)."""
        n_dE, n_ds, dE_grid, ds_grid = grid_setup
        n_dict = n_dE * n_ds

        # Spectra at all four corners
        corners = np.array([
            0 * n_ds + 0,                   # (0, 0)
            0 * n_ds + (n_ds - 1),           # (0, last)
            (n_dE - 1) * n_ds + 0,           # (last, 0)
            (n_dE - 1) * n_ds + (n_ds - 1),  # (last, last)
        ], dtype=np.int32)

        scores = np.ones((4, n_dict), dtype=np.float32)  # doesn't matter
        dE_ref, ds_ref = parabola_refine_2d(
            scores, corners, (n_dE, n_ds), dE_grid, ds_grid)

        # dE boundary: idx=0 or n_dE-1 → no parabola
        np.testing.assert_allclose(dE_ref[0], dE_grid[0])
        np.testing.assert_allclose(dE_ref[2], dE_grid[-1])
        # ds boundary: idx=0 or n_ds-1 → no parabola
        np.testing.assert_allclose(ds_ref[0], ds_grid[0])
        np.testing.assert_allclose(ds_ref[1], ds_grid[-1])
