"""
Tests for Multi-Peak Alternating Projection Solver.

Covers:
    1. Configuration (ComponentConfig, MultiPeakConfig, separation checks)
    2. Dictionary building (per-component, shapes, grid types)
    3. Batch LLS amplitude estimation (2×2 Cramer, 1-comp, general)
    4. Alternating projection accuracy (1-comp equivalence, 2-comp NF/noisy)
    5. Parabola refinement integration
    6. Chunked processing consistency
    7. Pipeline integration
"""

import warnings

import numpy as np
import pytest
from scipy.special import wofz

from toyomacro.voigtfit.dictionary_solver import build_dictionary
from toyomacro.voigtfit.multipeak_config import ComponentConfig, MultiPeakConfig
from toyomacro.voigtfit.multipeak_solver import (
    MultiPeak2StageResult,
    MultiPeakResult,
    _batch_lls_amplitude,
    _project_orthogonal,
    build_multipeak_dictionaries,
    process_multipeak,
    solve_alternating_projection,
    solve_multipeak_2stage,
    solve_multipeak_chunked,
)

# ============================================================================
# Helpers
# ============================================================================


def voigt_profile(energy, center, sigma, gamma):
    """Single Voigt profile via Faddeeva function."""
    z = ((energy - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


FWHM_TO_SIGMA = 1.0 / (2 * np.sqrt(2 * np.log(2)))
SIGMA_NOM = 1.0 * FWHM_TO_SIGMA  # 0.4247 eV
GAMMA = 0.125
ENERGY = np.linspace(280.0, 290.0, 101, dtype=np.float32)


def generate_2comp_spectra(
    n_spectra,
    center1=284.8,
    center2=286.5,
    sigma=SIGMA_NOM,
    gamma=GAMMA,
    dE1_range=(-0.3, 0.3),
    dE2_range=(-0.3, 0.3),
    ds1_range=(-0.05, 0.05),
    ds2_range=(-0.05, 0.05),
    amp_ratio=1.0,
    noise_sigma=0.0,
    seed=42,
):
    """Generate 2-component Voigt spectra with known ground truth."""
    rng = np.random.default_rng(seed)
    dE1 = rng.uniform(*dE1_range, n_spectra).astype(np.float32)
    dE2 = rng.uniform(*dE2_range, n_spectra).astype(np.float32)
    ds1 = rng.uniform(*ds1_range, n_spectra).astype(np.float32)
    ds2 = rng.uniform(*ds2_range, n_spectra).astype(np.float32)
    a1 = np.ones(n_spectra, dtype=np.float32)
    a2 = np.full(n_spectra, amp_ratio, dtype=np.float32)

    Y = np.zeros((n_spectra, len(ENERGY)), dtype=np.float32)
    for i in range(n_spectra):
        p1 = voigt_profile(ENERGY, center1 + dE1[i], max(sigma + ds1[i], 0.01), gamma)
        p2 = voigt_profile(ENERGY, center2 + dE2[i], max(sigma + ds2[i], 0.01), gamma)
        p1 /= p1.max() + 1e-30
        p2 /= p2.max() + 1e-30
        Y[i] = a1[i] * p1 + a2[i] * p2

    if noise_sigma > 0:
        Y += rng.normal(0, noise_sigma, Y.shape).astype(np.float32)

    gt = dict(dE1=dE1, dE2=dE2, ds1=ds1, ds2=ds2, a1=a1, a2=a2)
    return Y, gt


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def two_comp_config():
    """Well-separated 2-component config (4.0σ separation)."""
    return MultiPeakConfig(
        peaks=[
            ComponentConfig(center=284.8, sigma=SIGMA_NOM, gamma=GAMMA),
            ComponentConfig(center=286.5, sigma=SIGMA_NOM, gamma=GAMMA),
        ],
        energy_axis=ENERGY,
    )


@pytest.fixture
def single_comp_config():
    """Single component config."""
    return MultiPeakConfig(
        peaks=[ComponentConfig(center=284.8, sigma=SIGMA_NOM, gamma=GAMMA)],
        energy_axis=ENERGY,
    )


@pytest.fixture
def close_comp_config():
    """Close 2-component config (1.2σ separation — below threshold)."""
    return MultiPeakConfig(
        peaks=[
            ComponentConfig(center=284.8, sigma=SIGMA_NOM, gamma=GAMMA),
            ComponentConfig(center=284.3, sigma=SIGMA_NOM, gamma=GAMMA),
        ],
        energy_axis=ENERGY,
    )


# ============================================================================
# Test: Configuration
# ============================================================================


class TestMultiPeakConfig:
    def test_separation_sigma(self, two_comp_config):
        sep = two_comp_config.separation_sigma()
        expected = abs(286.5 - 284.8) / SIGMA_NOM
        assert abs(sep - expected) < 0.01

    def test_separation_single_comp(self, single_comp_config):
        assert single_comp_config.separation_sigma() == float("inf")

    def test_separation_warning(self, close_comp_config):
        with pytest.warns(UserWarning, match="separation"):
            close_comp_config.check_separation()

    def test_no_warning_well_separated(self, two_comp_config):
        # Should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            two_comp_config.check_separation()

    def test_n_comp(self, two_comp_config, single_comp_config):
        assert two_comp_config.n_comp == 2
        assert single_comp_config.n_comp == 1


# ============================================================================
# Test: Dictionary Building
# ============================================================================


class TestComponentDictionary:
    def test_builds_correct_count(self, two_comp_config):
        dicts = build_multipeak_dictionaries(two_comp_config)
        assert len(dicts) == 2

    def test_dict_shapes(self, two_comp_config):
        dicts = build_multipeak_dictionaries(two_comp_config)
        for dc in dicts:
            n_dE, n_ds = dc.grid_shape
            assert dc.D.shape[0] == len(ENERGY)
            assert dc.D.shape[1] == n_dE * n_ds
            assert dc.Phi_per_grid.shape == (n_dE * n_ds, len(ENERGY), 1)
            assert dc.Wt_per_grid.shape == (n_dE * n_ds, 1, len(ENERGY))

    def test_grid_sizes(self, two_comp_config):
        dicts = build_multipeak_dictionaries(two_comp_config)
        for dc in dicts:
            n_dE, n_ds = dc.grid_shape
            assert n_dE == 10  # default
            assert n_ds == 31  # default

    def test_custom_grid_sizes(self):
        config = MultiPeakConfig(
            peaks=[
                ComponentConfig(
                    center=284.8, sigma=SIGMA_NOM, gamma=GAMMA, n_dE=5, n_ds=11
                ),
                ComponentConfig(
                    center=286.5, sigma=SIGMA_NOM, gamma=GAMMA, n_dE=15, n_ds=21
                ),
            ],
            energy_axis=ENERGY,
        )
        dicts = build_multipeak_dictionaries(config)
        assert dicts[0].grid_shape[0] == 5
        assert dicts[0].grid_shape[1] == 11
        assert dicts[1].grid_shape[0] == 15
        assert dicts[1].grid_shape[1] == 21

    def test_d_normalized(self, two_comp_config):
        dicts = build_multipeak_dictionaries(two_comp_config)
        for dc in dicts:
            norms = np.linalg.norm(dc.D, axis=0)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_grid_types(self, two_comp_config):
        for grid_type in ["uniform", "chebyshev", "sinh"]:
            dicts = build_multipeak_dictionaries(two_comp_config, grid_type=grid_type)
            assert len(dicts) == 2
            for dc in dicts:
                assert dc.D.shape[1] > 0


# ============================================================================
# Test: Batch LLS Amplitude
# ============================================================================


class TestBatchLLS:
    def test_2x2_analytic(self, two_comp_config):
        """2×2 Cramer's rule gives correct amplitudes."""
        dicts = build_multipeak_dictionaries(two_comp_config)
        batch = 50
        best_indices = np.zeros((batch, 2), dtype=np.int32)
        for k in range(2):
            best_indices[:, k] = dicts[k].nominal_index

        # Build Y from known amplitudes
        a1_true = np.full(batch, 2.0, dtype=np.float32)
        a2_true = np.full(batch, 0.7, dtype=np.float32)
        phi1 = dicts[0].Phi_per_grid[best_indices[:, 0], :, 0]
        phi2 = dicts[1].Phi_per_grid[best_indices[:, 1], :, 0]
        Y = a1_true[:, None] * phi1 + a2_true[:, None] * phi2

        amp = _batch_lls_amplitude(Y, dicts, best_indices)
        np.testing.assert_allclose(amp[:, 0], a1_true, atol=1e-3)
        np.testing.assert_allclose(amp[:, 1], a2_true, atol=1e-3)

    def test_single_comp(self, single_comp_config):
        """Single component LLS."""
        dicts = build_multipeak_dictionaries(single_comp_config)
        batch = 30
        best_indices = np.full((batch, 1), dicts[0].nominal_index, dtype=np.int32)
        a_true = np.linspace(0.5, 3.0, batch, dtype=np.float32)
        phi = dicts[0].Phi_per_grid[best_indices[:, 0], :, 0]
        Y = a_true[:, None] * phi

        amp = _batch_lls_amplitude(Y, dicts, best_indices)
        np.testing.assert_allclose(amp[:, 0], a_true, atol=1e-3)

    def test_batch_consistency(self, two_comp_config):
        """Batch LLS matches per-spectrum lstsq."""
        dicts = build_multipeak_dictionaries(two_comp_config)
        batch = 20
        rng = np.random.default_rng(42)
        best_indices = np.zeros((batch, 2), dtype=np.int32)
        for k in range(2):
            best_indices[:, k] = rng.integers(0, dicts[k].n_dict, batch)

        a1 = rng.uniform(0.5, 2.0, batch).astype(np.float32)
        a2 = rng.uniform(0.3, 1.5, batch).astype(np.float32)
        phi1 = dicts[0].Phi_per_grid[best_indices[:, 0], :, 0]
        phi2 = dicts[1].Phi_per_grid[best_indices[:, 1], :, 0]
        Y = a1[:, None] * phi1 + a2[:, None] * phi2

        # Batch result
        amp_batch = _batch_lls_amplitude(Y, dicts, best_indices)

        # Per-spectrum lstsq reference
        for i in range(batch):
            Phi = np.column_stack([phi1[i], phi2[i]])
            amps_ref, _, _, _ = np.linalg.lstsq(Phi, Y[i], rcond=None)
            np.testing.assert_allclose(amp_batch[i], amps_ref, atol=1e-3)


# ============================================================================
# Test: Orthogonal Projection
# ============================================================================


class TestOrthogonalProjection:
    def test_single_comp_identity(self, single_comp_config):
        """n_comp=1: projection is identity."""
        dicts = build_multipeak_dictionaries(single_comp_config)
        batch = 10
        Y = np.random.default_rng(42).standard_normal((batch, len(ENERGY))).astype(
            np.float32
        )
        best_np = np.full((batch, 1), dicts[0].nominal_index, dtype=np.int32)
        Y_proj = _project_orthogonal(Y, dicts, best_np, 0, 1)
        np.testing.assert_array_equal(Y_proj, Y)

    def test_two_comp_removes_other(self, two_comp_config):
        """Projecting ⊥ φ₂ removes φ₂ component from Y."""
        dicts = build_multipeak_dictionaries(two_comp_config)
        batch = 10
        best_np = np.zeros((batch, 2), dtype=np.int32)
        for k in range(2):
            best_np[:, k] = dicts[k].nominal_index

        phi1 = dicts[0].Phi_per_grid[best_np[:, 0], :, 0]
        phi2 = dicts[1].Phi_per_grid[best_np[:, 1], :, 0]
        Y = 2.0 * phi1 + 3.0 * phi2

        # Project ⊥ comp 1 (remove phi2)
        Y_proj = _project_orthogonal(Y, dicts, best_np, 0, 2)

        # Projected Y should be orthogonal to phi2
        dots = np.sum(Y_proj * phi2, axis=1)
        np.testing.assert_allclose(dots, 0.0, atol=1e-3)


# ============================================================================
# Test: Alternating Projection Solver
# ============================================================================

# Skip if MLX not usable (installed alone is not enough: a headless
# Mac imports mlx but cannot execute Metal work)
from toyomacro.voigtfit._mlx_support import mlx_usable

mlx_available = mlx_usable()
if mlx_available:
    import mlx.core as mx

requires_mlx = pytest.mark.skipif(not mlx_available, reason="MLX not available")


@requires_mlx
class TestAlternatingProjection:
    def test_single_component_equivalence(self, single_comp_config):
        """n_comp=1 should work and give reasonable results."""
        n_spectra = 100
        rng = np.random.default_rng(42)
        dE_true = rng.uniform(-0.3, 0.3, n_spectra).astype(np.float32)

        Y = np.zeros((n_spectra, len(ENERGY)), dtype=np.float32)
        for i in range(n_spectra):
            p = voigt_profile(ENERGY, 284.8 + dE_true[i], SIGMA_NOM, GAMMA)
            p /= p.max() + 1e-30
            Y[i] = p

        dicts = build_multipeak_dictionaries(single_comp_config)
        result = solve_alternating_projection(Y, dicts, n_iterations=1)

        assert result.amplitudes.shape == (n_spectra, 1)
        assert result.delta_E.shape == (n_spectra, 1)
        # δE should be close to ground truth
        dE_rmse = np.sqrt(np.mean((result.delta_E[:, 0] - dE_true) ** 2))
        assert dE_rmse < 0.1  # within 0.1 eV (grid + parabola)

    def test_two_component_noisefree(self, two_comp_config):
        """2-component NF with 4σ separation: δE < 0.05 eV RMSE."""
        n_spectra = 200
        Y, gt = generate_2comp_spectra(
            n_spectra, center1=284.8, center2=286.5, amp_ratio=1.0
        )

        dicts = build_multipeak_dictionaries(two_comp_config)
        result = solve_alternating_projection(
            Y, dicts, n_iterations=3, parabola_dE=True
        )

        assert result.amplitudes.shape == (n_spectra, 2)

        # δE RMSE for both components
        dE1_rmse = np.sqrt(np.mean((result.delta_E[:, 0] - gt["dE1"]) ** 2))
        dE2_rmse = np.sqrt(np.mean((result.delta_E[:, 1] - gt["dE2"]) ** 2))
        # 10-point grid (step ≈ 0.44 eV) + parabola → 0.06-0.08 eV RMSE typical
        assert dE1_rmse < 0.08, f"dE1 RMSE={dE1_rmse:.4f}"
        assert dE2_rmse < 0.08, f"dE2 RMSE={dE2_rmse:.4f}"

        # Amplitude ratio should be close to 1.0.
        # Coarse grid (10pt, step ≈ 0.44 eV) introduces ~6% systematic bias.
        ratio = result.amplitudes[:, 1] / (result.amplitudes[:, 0] + 1e-10)
        ratio_err = np.abs(np.mean(ratio) - 1.0)
        assert ratio_err < 0.10, f"Amp ratio error={ratio_err:.4f}"

    def test_two_component_noisy(self, two_comp_config):
        """2-component with moderate noise still recovers parameters."""
        n_spectra = 200
        Y, gt = generate_2comp_spectra(
            n_spectra,
            center1=284.8,
            center2=286.5,
            amp_ratio=1.0,
            noise_sigma=0.01,
        )

        dicts = build_multipeak_dictionaries(two_comp_config)
        result = solve_alternating_projection(
            Y, dicts, n_iterations=3, parabola_dE=True
        )

        dE1_rmse = np.sqrt(np.mean((result.delta_E[:, 0] - gt["dE1"]) ** 2))
        dE2_rmse = np.sqrt(np.mean((result.delta_E[:, 1] - gt["dE2"]) ** 2))
        # Noise adds variance but should still be reasonable
        assert dE1_rmse < 0.15, f"dE1 RMSE={dE1_rmse:.4f}"
        assert dE2_rmse < 0.15, f"dE2 RMSE={dE2_rmse:.4f}"

    def test_symmetry(self, two_comp_config):
        """Swapping component order should not change results significantly."""
        n_spectra = 50
        Y, gt = generate_2comp_spectra(
            n_spectra, center1=284.8, center2=286.5, amp_ratio=1.0
        )

        dicts_fwd = build_multipeak_dictionaries(two_comp_config)
        result_fwd = solve_alternating_projection(Y, dicts_fwd, n_iterations=3)

        # Reversed config
        config_rev = MultiPeakConfig(
            peaks=list(reversed(two_comp_config.peaks)),
            energy_axis=ENERGY,
        )
        dicts_rev = build_multipeak_dictionaries(config_rev)
        result_rev = solve_alternating_projection(Y, dicts_rev, n_iterations=3)

        # Component 0 forward ≈ component 1 reversed.
        # Alternating projection is sensitive to update order, so
        # some spectra may converge to different solutions with finite
        # iterations. Check that most spectra agree.
        diff_01 = np.abs(result_fwd.delta_E[:, 0] - result_rev.delta_E[:, 1])
        diff_10 = np.abs(result_fwd.delta_E[:, 1] - result_rev.delta_E[:, 0])
        # At least 80% of spectra should agree within 0.1 eV
        agree_01 = np.mean(diff_01 < 0.1)
        agree_10 = np.mean(diff_10 < 0.1)
        assert agree_01 > 0.8, f"Only {agree_01:.0%} agree for comp 0↔1"
        assert agree_10 > 0.8, f"Only {agree_10:.0%} agree for comp 1↔0"

    def test_convergence(self):
        """More iterations should not degrade accuracy."""
        config = MultiPeakConfig(
            peaks=[
                ComponentConfig(center=284.8, sigma=SIGMA_NOM, gamma=GAMMA),
                ComponentConfig(center=286.5, sigma=SIGMA_NOM, gamma=GAMMA),
            ],
            energy_axis=ENERGY,
        )
        Y, gt = generate_2comp_spectra(100, center1=284.8, center2=286.5)

        dicts = build_multipeak_dictionaries(config)

        rmse_by_iter = []
        for n_iter in [1, 3, 5]:
            result = solve_alternating_projection(Y, dicts, n_iterations=n_iter)
            dE1_rmse = np.sqrt(np.mean((result.delta_E[:, 0] - gt["dE1"]) ** 2))
            rmse_by_iter.append(dE1_rmse)

        # 3 iterations should be at least as good as 1
        assert rmse_by_iter[1] <= rmse_by_iter[0] + 0.01
        # 5 iterations should be at least as good as 3
        assert rmse_by_iter[2] <= rmse_by_iter[1] + 0.01

    def test_chi2_output(self, two_comp_config):
        """chi2 is computed and positive."""
        Y, _ = generate_2comp_spectra(50, center1=284.8, center2=286.5)
        dicts = build_multipeak_dictionaries(two_comp_config)
        result = solve_alternating_projection(Y, dicts)
        assert result.chi2.shape == (50,)
        assert np.all(result.chi2 >= 0)

    def test_timing_info(self, two_comp_config):
        """Timing dict is populated."""
        Y, _ = generate_2comp_spectra(30, center1=284.8, center2=286.5)
        dicts = build_multipeak_dictionaries(two_comp_config)
        result = solve_alternating_projection(Y, dicts)
        assert result.timing is not None
        assert "alternating" in result.timing
        assert "lls" in result.timing
        assert "total" in result.timing


# ============================================================================
# Test: Parabola Integration
# ============================================================================


@requires_mlx
class TestParabolaMultiPeak:
    def test_parabola_improves_dE(self, two_comp_config):
        """Parabola refinement should improve δE accuracy over grid-only."""
        n_spectra = 200
        Y, gt = generate_2comp_spectra(
            n_spectra, center1=284.8, center2=286.5, amp_ratio=1.0
        )
        dicts = build_multipeak_dictionaries(two_comp_config)

        result_no_para = solve_alternating_projection(
            Y, dicts, parabola_dE=False, parabola_ds=False
        )
        result_para = solve_alternating_projection(
            Y, dicts, parabola_dE=True, parabola_ds=False
        )

        rmse_no = np.sqrt(np.mean((result_no_para.delta_E[:, 0] - gt["dE1"]) ** 2))
        rmse_yes = np.sqrt(np.mean((result_para.delta_E[:, 0] - gt["dE1"]) ** 2))

        # Parabola should give equal or better δE
        assert rmse_yes <= rmse_no + 0.001, (
            f"Parabola worse: {rmse_yes:.4f} > {rmse_no:.4f}"
        )


# ============================================================================
# Test: Chunked Processing
# ============================================================================


@requires_mlx
class TestChunkedProcessing:
    def test_chunk_vs_full(self, two_comp_config):
        """Chunked and full processing give identical results."""
        n_spectra = 200
        Y, _ = generate_2comp_spectra(n_spectra, center1=284.8, center2=286.5)
        dicts = build_multipeak_dictionaries(two_comp_config)

        result_full = solve_multipeak_chunked(Y, dicts, chunk_size=None)
        result_chunk = solve_multipeak_chunked(Y, dicts, chunk_size=50)

        np.testing.assert_allclose(
            result_full.amplitudes, result_chunk.amplitudes, atol=1e-5
        )
        np.testing.assert_allclose(
            result_full.delta_E, result_chunk.delta_E, atol=1e-5
        )
        np.testing.assert_allclose(result_full.chi2, result_chunk.chi2, atol=1e-5)

    def test_auto_chunk_size(self, two_comp_config):
        """Auto chunk size is reasonable."""
        Y, _ = generate_2comp_spectra(100, center1=284.8, center2=286.5)
        dicts = build_multipeak_dictionaries(two_comp_config)
        result = solve_multipeak_chunked(Y, dicts)
        assert result.amplitudes.shape == (100, 2)


# ============================================================================
# Test: Pipeline Integration
# ============================================================================


@requires_mlx
class TestProcessMultipeak:
    def test_end_to_end(self, two_comp_config):
        """Full pipeline: config → dicts → solve → result."""
        n_spectra = 100
        Y, gt = generate_2comp_spectra(
            n_spectra, center1=284.8, center2=286.5, amp_ratio=1.0
        )
        result = process_multipeak(Y, two_comp_config)

        assert isinstance(result, MultiPeakResult)
        assert result.amplitudes.shape == (n_spectra, 2)
        assert result.delta_E.shape == (n_spectra, 2)
        assert result.delta_sigma.shape == (n_spectra, 2)
        assert result.chi2.shape == (n_spectra,)
        assert result.timing is not None
        assert "dict_build" in result.timing

    def test_separation_warning_flag(self, close_comp_config):
        """Separation warning is set in result."""
        Y, _ = generate_2comp_spectra(
            30, center1=284.8, center2=284.3, amp_ratio=1.0
        )
        with pytest.warns(UserWarning, match="separation"):
            result = process_multipeak(Y, close_comp_config)
        assert result.separation_warning == True

    def test_pipeline_wrapper(self, two_comp_config):
        """HybridPipeline.process_multipeak works."""
        from toyomacro.voigtfit.pipeline import HybridPipeline
        from toyomacro.voigtfit.weight_cache import WeightMatrixCache

        cache = WeightMatrixCache()
        pipeline = HybridPipeline(cache=cache, enable_stage2=False)

        Y, _ = generate_2comp_spectra(50, center1=284.8, center2=286.5)
        result = pipeline.process_multipeak(Y, two_comp_config)

        assert isinstance(result, MultiPeakResult)
        assert result.amplitudes.shape == (50, 2)


# ============================================================================
# 2-Stage γ-calibrated multi-peak solver
# ============================================================================


def generate_2comp_spectra_with_gamma(
    n_spectra,
    center1=284.8,
    center2=286.5,
    sigma=SIGMA_NOM,
    gamma=GAMMA,
    dgamma1=0.0,
    dgamma2=0.0,
    dE1_range=(-0.3, 0.3),
    dE2_range=(-0.3, 0.3),
    ds1_range=(-0.05, 0.05),
    ds2_range=(-0.05, 0.05),
    amp_ratio=1.0,
    noise_sigma=0.0,
    seed=42,
):
    """Generate 2-component Voigt spectra with per-component γ offsets.

    Critical: ds_range must be non-zero for Fisher σ-γ coupling
    to enable γ estimation.
    """
    rng = np.random.default_rng(seed)
    dE1 = rng.uniform(*dE1_range, n_spectra).astype(np.float32)
    dE2 = rng.uniform(*dE2_range, n_spectra).astype(np.float32)
    ds1 = rng.uniform(*ds1_range, n_spectra).astype(np.float32)
    ds2 = rng.uniform(*ds2_range, n_spectra).astype(np.float32)
    a1 = np.ones(n_spectra, dtype=np.float32)
    a2 = np.full(n_spectra, amp_ratio, dtype=np.float32)

    Y = np.zeros((n_spectra, len(ENERGY)), dtype=np.float32)
    for i in range(n_spectra):
        p1 = voigt_profile(
            ENERGY, center1 + dE1[i], max(sigma + ds1[i], 0.01), gamma + dgamma1,
        )
        p2 = voigt_profile(
            ENERGY, center2 + dE2[i], max(sigma + ds2[i], 0.01), gamma + dgamma2,
        )
        p1 /= p1.max() + 1e-30
        p2 /= p2.max() + 1e-30
        Y[i] = a1[i] * p1 + a2[i] * p2

    if noise_sigma > 0:
        Y += rng.normal(0, noise_sigma, Y.shape).astype(np.float32)

    gt = dict(
        dE1=dE1, dE2=dE2, ds1=ds1, ds2=ds2, a1=a1, a2=a2,
        dgamma1=dgamma1, dgamma2=dgamma2,
    )
    return Y, gt


@requires_mlx
class TestMultiPeak2Stage:
    """Tests for 2-stage γ-calibrated multi-peak solver."""

    def test_basic_shapes(self, two_comp_config):
        """2-stage result should have correct shapes and fields."""
        n = 200
        Y, _ = generate_2comp_spectra_with_gamma(n)
        result = solve_multipeak_2stage(Y, two_comp_config)

        assert isinstance(result, MultiPeak2StageResult)
        assert result.amplitudes.shape == (n, 2)
        assert result.delta_E.shape == (n, 2)
        assert result.delta_sigma.shape == (n, 2)
        assert result.chi2.shape == (n,)
        assert result.corrected_gammas.shape == (2,)
        assert result.global_dgammas.shape == (2,)
        assert len(result.dgamma_per_spectrum) == 2
        assert result.dgamma_stds.shape == (2,)
        assert result.n_calibration > 0

    def test_no_gamma_offset(self, two_comp_config):
        """At dgamma=0, corrected gammas should stay near nominal.

        Known: deflation artifact introduces systematic negative bias of
        ~0.01 eV in per-component γ estimation (σ-γ coupling in overlap
        region, dev-log 67). Tolerance accounts for this.
        """
        n = 1000
        Y, _ = generate_2comp_spectra_with_gamma(n, dgamma1=0.0, dgamma2=0.0)
        result = solve_multipeak_2stage(Y, two_comp_config)

        for k in range(2):
            # Deflation bias ~0.01: allow ±0.02 tolerance
            assert abs(result.global_dgammas[k]) < 0.02, (
                f"comp {k}: global_dg={result.global_dgammas[k]:.4f}, expected ~0"
            )

    def test_gamma_correction_direction(self, two_comp_config):
        """Per-component γ correction goes in the correct direction.

        With large dgamma offsets (0.04, -0.03), the estimated correction
        should track the direction and relative magnitude, even though
        absolute accuracy is limited by deflation bias (~0.01).
        """
        dgamma1, dgamma2 = 0.04, -0.03
        n = 1000
        Y, _ = generate_2comp_spectra_with_gamma(
            n, dgamma1=dgamma1, dgamma2=dgamma2, seed=777,
        )
        result = solve_multipeak_2stage(Y, two_comp_config)

        # Comp 0 (true dg=+0.04) should estimate positive (or less negative)
        # Comp 1 (true dg=-0.03) should estimate more negative
        assert result.global_dgammas[0] > result.global_dgammas[1], (
            f"comp0 dg={result.global_dgammas[0]:.4f} should be > "
            f"comp1 dg={result.global_dgammas[1]:.4f}"
        )
        # The DIFFERENCE between components should track the true difference.
        # Attenuated by σ-γ coupling (~24% recovery at σ/γ≈3.4).
        true_diff = dgamma1 - dgamma2  # 0.07
        est_diff = float(result.global_dgammas[0] - result.global_dgammas[1])
        assert est_diff > true_diff * 0.15, (
            f"est_diff={est_diff:.4f}, true_diff={true_diff:.4f}"
        )

    def test_large_gamma_offset_improves_sigma(self, two_comp_config):
        """With large dgamma (0.04), 2-stage should improve delta-sigma.

        At σ/γ≈3.4, a 0.04 eV gamma offset creates measurable sigma bias.
        The 2-stage correction should reduce this bias even with the ~0.01
        deflation artifact, since the net correction (0.04 - 0.01 = 0.03)
        is still in the right direction.
        """
        dgamma1 = 0.04  # Large offset to overcome deflation bias
        n = 1000
        Y, gt = generate_2comp_spectra_with_gamma(
            n, dgamma1=dgamma1, dgamma2=0.0,
            ds1_range=(-0.05, 0.05), ds2_range=(-0.05, 0.05), seed=888,
        )

        # Standard (no γ correction) — uses wrong gamma for comp 0
        result_std = process_multipeak(Y, two_comp_config)
        ds1_rmse_std = float(np.sqrt(np.mean(
            (result_std.delta_sigma[:, 0] - gt["ds1"]) ** 2
        )))

        # 2-stage (with γ correction)
        result_2s = solve_multipeak_2stage(Y, two_comp_config)
        ds1_rmse_2s = float(np.sqrt(np.mean(
            (result_2s.delta_sigma[:, 0] - gt["ds1"]) ** 2
        )))

        # 2-stage RMSE should be lower (correction helps despite bias)
        assert ds1_rmse_2s < ds1_rmse_std, (
            f"2-stage RMSE={ds1_rmse_2s:.5f} should be < std RMSE={ds1_rmse_std:.5f}"
        )

    def test_calibration_subset(self, two_comp_config):
        """n_calibration should limit Stage 1 to a subset."""
        n = 500
        Y, _ = generate_2comp_spectra_with_gamma(n, dgamma1=0.01)
        result = solve_multipeak_2stage(Y, two_comp_config, n_calibration=100)

        assert result.n_calibration == 100
        # Stage 2 should process all spectra
        assert result.amplitudes.shape[0] == n
        assert result.delta_E.shape[0] == n

    def test_timing_populated(self, two_comp_config):
        """Timing dict should have stage1a, stage1b, stage2 entries."""
        n = 100
        Y, _ = generate_2comp_spectra_with_gamma(n)
        result = solve_multipeak_2stage(Y, two_comp_config)

        assert "stage1a_alternating" in result.timing
        assert "stage1b_gamma_estimation" in result.timing
        assert "stage2_dict_build" in result.timing
        assert "stage2_solve" in result.timing
        assert "total" in result.timing

    def test_pipeline_wrapper(self, two_comp_config):
        """HybridPipeline.process_multipeak_2stage works."""
        from toyomacro.voigtfit.pipeline import HybridPipeline
        from toyomacro.voigtfit.weight_cache import WeightMatrixCache

        cache = WeightMatrixCache()
        pipeline = HybridPipeline(cache=cache, enable_stage2=False)

        n = 100
        Y, _ = generate_2comp_spectra_with_gamma(n, dgamma1=0.01)
        result = pipeline.process_multipeak_2stage(Y, two_comp_config)

        assert isinstance(result, MultiPeak2StageResult)
        assert result.corrected_gammas is not None
        assert result.amplitudes.shape == (n, 2)

    def test_aggregation_median(self, two_comp_config):
        """Median aggregation path should execute without error."""
        n = 500
        Y, _ = generate_2comp_spectra_with_gamma(n, dgamma1=0.03, seed=999)
        result_mean = solve_multipeak_2stage(
            Y, two_comp_config, aggregation="mean",
        )
        result_median = solve_multipeak_2stage(
            Y, two_comp_config, aggregation="median",
        )
        # Both should complete and give reasonable results
        assert result_mean.corrected_gammas is not None
        assert result_median.corrected_gammas is not None
        # Median and mean should be in the same ballpark
        assert abs(result_mean.global_dgammas[0] - result_median.global_dgammas[0]) < 0.015


# ============================================================================
# Exact-Jacobian Newton (Task A1) + GPU closed-form solve (Task B1)
# ============================================================================


def _two_peak_problem(n, sep_sigma=7.0, sigma=SIGMA_NOM, gamma=GAMMA,
                      dE_amp=0.15, ds_amp=0.12, seed=0):
    """Per-spectrum noise-free 2-peak problem (dev-log 77 ceiling-style setup)."""
    c0 = 285.0
    c1 = c0 + sep_sigma * sigma
    amp_peak = 1000.0
    energy = np.linspace(c0 - 6, c1 + 6, 160, dtype=np.float32)

    rng = np.random.default_rng(seed)
    dE_true = rng.uniform(-dE_amp, dE_amp, (n, 2)).astype(np.float32)
    ds_true = rng.uniform(-ds_amp, ds_amp, (n, 2)).astype(np.float32)
    amp_true = rng.uniform(0.6 * amp_peak, 1.4 * amp_peak, (n, 2)).astype(np.float32)

    centers = np.stack([c0 + dE_true[:, 0], c1 + dE_true[:, 1]], axis=1)
    sigmas = sigma + ds_true
    Y = np.zeros((n, len(energy)), dtype=np.float32)
    for k in range(2):
        for i in range(n):
            Y[i] += amp_true[i, k] * voigt_profile(
                energy, centers[i, k], sigmas[i, k], gamma)

    dicts = [
        build_dictionary(energy, np.array([c0]), np.array([sigma]), gamma,
                         dE_range=(-0.6, 0.6), dE_step=0.04,
                         dsigma_range=(-0.3, 0.3), dsigma_step=0.03),
        build_dictionary(energy, np.array([c1]), np.array([sigma]), gamma,
                         dE_range=(-0.6, 0.6), dE_step=0.04,
                         dsigma_range=(-0.3, 0.3), dsigma_step=0.03),
    ]
    return Y, dicts, dE_true, ds_true, amp_true, amp_peak


def _psnr(true, est, peak):
    mse = np.mean((true - est) ** 2)
    return 10 * np.log10(peak ** 2 / mse) if mse > 0 else float("inf")


@requires_mlx
class TestExactJacobianNewton:
    """Task A1 — exact-Jacobian Gauss-Newton breaks the surrogate δσ ceiling."""

    def test_breaks_dsigma_ceiling(self):
        Y, dicts, dE_true, ds_true, _, _ = _two_peak_problem(n=1500)
        common = dict(n_iterations=3, parabola_dE=True, parabola_ds=True,
                      apply_newton=True, newton_iter=5)
        res_surr = solve_alternating_projection(Y, dicts, newton_exact=False, **common)
        res_exact = solve_alternating_projection(Y, dicts, newton_exact=True, **common)

        dS_surr = _psnr(ds_true, res_surr.delta_sigma, 1.0)
        dS_exact = _psnr(ds_true, res_exact.delta_sigma, 1.0)
        # Exact must beat the surrogate by a clear margin and clear ~50 dB.
        assert dS_exact > dS_surr + 5.0, (dS_surr, dS_exact)
        assert dS_exact > 50.0, dS_exact

    def test_amplitude_recovery(self):
        Y, dicts, _, ds_true, amp_true, _ = _two_peak_problem(n=200)
        res = solve_alternating_projection(
            Y, dicts, n_iterations=3, parabola_dE=True, parabola_ds=True,
            apply_newton=True, newton_iter=3, newton_exact=True)
        rel = np.abs(res.amplitudes - amp_true) / amp_true
        assert np.median(rel) < 0.05
        assert np.all(np.isfinite(res.delta_sigma))

    def test_single_peak_exact(self):
        rng = np.random.default_rng(7)
        sigma, gamma = 0.5, 0.15
        energy = np.linspace(96, 104, 120, dtype=np.float32)
        n = 100
        dE_true = rng.uniform(-0.15, 0.15, (n, 1)).astype(np.float32)
        ds_true = rng.uniform(-0.10, 0.10, (n, 1)).astype(np.float32)
        amp_true = rng.uniform(700, 1300, (n, 1)).astype(np.float32)
        Y = np.zeros((n, len(energy)), dtype=np.float32)
        for i in range(n):
            Y[i] = amp_true[i, 0] * voigt_profile(
                energy, 100.0 + dE_true[i, 0], sigma + ds_true[i, 0], gamma)
        d = build_dictionary(energy, np.array([100.0]), np.array([sigma]), gamma,
                             dE_range=(-0.6, 0.6), dE_step=0.04,
                             dsigma_range=(-0.3, 0.3), dsigma_step=0.03)
        res = solve_alternating_projection(
            Y, [d], n_iterations=3, parabola_dE=True, parabola_ds=True,
            apply_newton=True, newton_iter=3, newton_exact=True)
        assert res.amplitudes.shape == (n, 1)
        assert np.all(np.isfinite(res.delta_sigma))

    def test_default_is_surrogate(self):
        """newton_exact defaults to False — identical to explicit surrogate."""
        Y, dicts, _, _, _, _ = _two_peak_problem(n=80)
        common = dict(n_iterations=3, parabola_dE=True, parabola_ds=True,
                      apply_newton=True, newton_iter=2)
        res_default = solve_alternating_projection(Y, dicts, **common)
        res_surr = solve_alternating_projection(Y, dicts, newton_exact=False, **common)
        np.testing.assert_allclose(res_default.delta_sigma, res_surr.delta_sigma)


@requires_mlx
class TestGpuSmallSystemSolve:
    """Task B1 — GPU Schur-complement solve matches numpy's LAPACK solve."""

    @staticmethod
    def _ref_solve(H, g):
        n_params = H.shape[1]
        Hc = H.copy()
        tr = np.trace(Hc, axis1=1, axis2=2)
        Hc[:, np.arange(n_params), np.arange(n_params)] += (1e-6 * tr)[:, None]
        return np.linalg.solve(Hc, g[:, :, None])[:, :, 0]

    @staticmethod
    def _entries(H):
        n_params = H.shape[1]
        return {(i, j): mx.array(H[:, i, j].copy())
                for i in range(n_params) for j in range(i, n_params)}

    @pytest.mark.parametrize("n_comp", [1, 2])
    def test_matches_numpy_solve(self, n_comp):
        from toyomacro.voigtfit.multipeak_solver import _solve_small_system_gpu

        rng = np.random.default_rng(11)
        n_params = 2 * n_comp
        batch = 3000
        H = np.zeros((batch, n_params, n_params), dtype=np.float32)
        for b in range(batch):
            root = rng.standard_normal((n_params, n_params)).astype(np.float32)
            H[b] = root @ root.T + 0.5 * np.eye(n_params, dtype=np.float32)
        g = rng.standard_normal((batch, n_params)).astype(np.float32)
        g_entries = [mx.array(g[:, i].copy()) for i in range(n_params)]
        delta_gpu = _solve_small_system_gpu(self._entries(H), g_entries, n_comp, batch)
        delta_ref = self._ref_solve(H, g)
        np.testing.assert_allclose(delta_gpu, delta_ref, atol=1e-3, rtol=1e-3)


@requires_mlx
class TestSpdSolveGpu:
    """Unrolled GPU Cholesky solve matches LAPACK on small SPD systems."""

    @staticmethod
    def _spd_problem(m, batch, seed, ridge=0.5):
        rng = np.random.default_rng(seed)
        root = rng.standard_normal((batch, m, m))
        G = (root @ np.transpose(root, (0, 2, 1))
             + ridge * np.eye(m)).astype(np.float32)
        rhs = rng.standard_normal((batch, m)).astype(np.float32)
        ref = np.linalg.solve(
            G.astype(np.float64), rhs.astype(np.float64)[:, :, None],
        )[:, :, 0]
        return G, rhs, ref

    @pytest.mark.parametrize("m", [2, 3, 4, 6, 13, 20])
    def test_matches_numpy_solve(self, m):
        from toyomacro.voigtfit.multipeak_solver import _spd_solve_gpu

        G, rhs, ref = self._spd_problem(m, batch=3000, seed=5)
        x_gpu = _spd_solve_gpu(mx.array(G), mx.array(rhs), m)
        mx.eval(x_gpu)
        np.testing.assert_allclose(np.asarray(x_gpu), ref, atol=2e-3, rtol=2e-3)

    def test_gram_solve_dispatch_large_m(self):
        """m > _SPD_GPU_MAX_M must still solve via the CPU-stream fallback."""
        from toyomacro.voigtfit.multipeak_solver import _SPD_GPU_MAX_M, _gram_solve_mx

        m = _SPD_GPU_MAX_M + 4
        G, rhs, ref = self._spd_problem(m, batch=64, seed=6, ridge=1.0)
        x = _gram_solve_mx(mx.array(G), mx.array(rhs), m)
        mx.eval(x)
        np.testing.assert_allclose(np.asarray(x), ref, atol=2e-3, rtol=2e-3)


@requires_mlx
class TestBlockDeflationBias:
    """n_comp >= 3 AP must use block deflation (sequential is biased).

    Sequential 1-D deflation is not a true orthogonal projection for
    n_comp >= 3: at 3σ spacing it leaves a deterministic, noise-independent
    δE bias of ≈ -0.07 eV on the interior/lower peaks. Block deflation
    (joint LLS residual + per-component add-back) removes it.
    """

    @staticmethod
    def _three_peak_problem(n=3000, seed=3):
        spacing = 3.0 * SIGMA_NOM
        centers = [284.0, 284.0 + spacing, 284.0 + 2 * spacing]
        rng = np.random.default_rng(seed)
        dE = rng.uniform(-0.10, 0.10, (n, 3)).astype(np.float32)
        ds = rng.uniform(-0.08, 0.08, (n, 3)).astype(np.float32)
        amp = rng.uniform(600.0, 1400.0, (n, 3)).astype(np.float32)
        Y = np.zeros((n, len(ENERGY)), dtype=np.float64)
        for k in range(3):
            for i in range(n):
                p = voigt_profile(ENERGY.astype(np.float64),
                                  centers[k] + dE[i, k],
                                  SIGMA_NOM + ds[i, k], GAMMA)
                Y[i] += amp[i, k] * p / (p.max() + 1e-30)
        dicts = [build_dictionary(ENERGY, np.array([c]), np.array([SIGMA_NOM]),
                                  GAMMA,
                                  dE_range=(-0.6, 0.6), dE_step=0.04,
                                  dsigma_range=(-0.3, 0.3), dsigma_step=0.03)
                 for c in centers]
        return Y.astype(np.float32), dicts, dE

    def test_three_comp_deflation_unbiased(self):
        Y, dicts, dE_true = self._three_peak_problem()
        result = solve_alternating_projection(Y, dicts)
        bias = np.mean(result.delta_E - dE_true, axis=0)
        # Sequential deflation gives |bias| ~ 0.06-0.07 eV on comps 0/1;
        # block deflation stays below a few mdE.
        assert np.max(np.abs(bias)) < 0.01, f"per-comp δE bias too large: {bias}"


@requires_mlx
class TestQualityFlags:
    """B4: per-spectrum quality diagnostics from the Newton Hessian."""

    def _doublet(self, sep_sigma, n=1500, snr=1000.0, seed=0):
        sigma, gamma = 0.4247, 0.125
        sep = sep_sigma * sigma
        c0, c1 = 100.0, 100.0 + sep
        energy = np.linspace(c0 - 6, c1 + 6, 160).astype(np.float32)
        rng = np.random.default_rng(seed)
        dE = rng.uniform(-0.1, 0.1, (n, 2)).astype(np.float32)
        ds = rng.uniform(-0.08, 0.08, (n, 2)).astype(np.float32)
        amp = rng.uniform(700, 1300, (n, 2)).astype(np.float32)
        sqrt2, sqrt2pi = np.sqrt(2.0), np.sqrt(2.0 * np.pi)
        e64 = energy.astype(np.float64)
        Y = np.zeros((n, len(energy)), dtype=np.float64)
        for k, ck in enumerate((c0, c1)):
            c = (ck + dE[:, k]).astype(np.float64)[:, None]
            s = np.maximum(ds[:, k] + sigma, 0.01).astype(np.float64)[:, None]
            z = ((e64[None, :] - c) + 1j * gamma) / (s * sqrt2)
            P = np.real(wofz(z)) / (s * sqrt2pi)
            P /= P.max(axis=1, keepdims=True) + 1e-30
            Y += amp[:, k][:, None] * P
        if snr is not None:
            rng2 = np.random.default_rng(seed + 7)
            Y = Y + rng2.normal(0, float(Y.max()) / snr, Y.shape)
        d0 = build_dictionary(energy, np.array([c0]), np.array([sigma]), gamma,
                              dE_range=(-0.6, 0.6), dE_step=0.04,
                              dsigma_range=(-0.3, 0.3), dsigma_step=0.03)
        d1 = build_dictionary(energy, np.array([c1]), np.array([sigma]), gamma,
                              dE_range=(-0.6, 0.6), dE_step=0.04,
                              dsigma_range=(-0.3, 0.3), dsigma_step=0.03)
        return Y.astype(np.float32), [d0, d1]

    def test_flags_off_by_default(self):
        Y, dicts = self._doublet(7)
        res = solve_alternating_projection(
            Y, dicts, n_iterations=3, parabola_dE=True, parabola_ds=True,
            apply_newton=True, newton_iter=3, newton_exact=True)
        assert res.cond_number is None
        assert res.param_std is None
        assert res.soft_std is None

    def test_flags_shapes_and_finiteness(self):
        Y, dicts = self._doublet(7)
        n = Y.shape[0]
        res = solve_alternating_projection(
            Y, dicts, n_iterations=3, parabola_dE=True, parabola_ds=True,
            apply_newton=True, newton_iter=3, newton_exact=True,
            quality_flags=True)
        assert res.cond_number.shape == (n,)
        assert res.param_std.shape == (n, 4)   # [δE0, δσ0, δE1, δσ1]
        assert res.soft_std.shape == (n,)
        assert np.all(np.isfinite(res.cond_number))
        assert np.all(np.isfinite(res.param_std))
        assert np.all(np.isfinite(res.soft_std))
        assert np.all(res.cond_number >= 1.0)
        assert np.all(res.param_std >= 0.0)
        # soft_std is the least-constrained direction => >= every marginal std
        assert np.all(res.soft_std >= res.param_std.max(axis=1) - 1e-6)

    def test_overlap_raises_condition_number(self):
        # Heavier overlap (3σ) => stiffer δσ1-δσ2 degeneracy => larger median cond
        Y_far, d_far = self._doublet(10, seed=1)
        Y_near, d_near = self._doublet(3, seed=1)
        common = dict(n_iterations=3, parabola_dE=True, parabola_ds=True,
                      apply_newton=True, newton_iter=3, newton_exact=True,
                      quality_flags=True)
        r_far = solve_alternating_projection(Y_far, d_far, **common)
        r_near = solve_alternating_projection(Y_near, d_near, **common)
        assert np.median(r_near.cond_number) > np.median(r_far.cond_number)


# ============================================================================
# Test: B3 polynomial background coupling
# ============================================================================


def _two_peak_with_bg(n, bg_const=0.0, bg_slope=0.0, seed=0):
    """Two-peak batch (dev-log 77 setup) plus a const+linear background.

    Returns (Y, dicts, ds_true). The background is bg_const·A + bg_slope·A·x on
    the normalized energy axis x = (E - E.mean()) / (ptp(E)/2), with A = 1000
    (the per-peak amplitude scale of ``_two_peak_problem``).
    """
    Y, dicts, _dE, ds, _amp, A = _two_peak_problem(n, seed=seed)
    c0 = 285.0
    energy = np.linspace(c0 - 6, c0 + 7.0 * SIGMA_NOM + 6, 160, dtype=np.float32)
    x = (energy - energy.mean()) / (np.ptp(energy) / 2)
    bg = bg_const * A + bg_slope * A * x[None, :]
    return (Y + bg).astype(np.float32), dicts, ds


def _ds_rms(result, ds_true):
    """Pooled δσ RMS error over both peaks."""
    return float(np.mean([
        np.std(result.delta_sigma[:, k] - ds_true[:, k]) for k in range(2)
    ]))


class TestBackgroundCoupling:
    """B3: opt-in polynomial background in the joint amplitude LLS."""

    def test_bg_off_identical_to_default(self):
        """bg_degree=-1 must reproduce the default behavior bit-for-bit."""
        Y, dicts, _ds = _two_peak_with_bg(200, seed=1)
        r_default = solve_alternating_projection(
            Y, dicts, parabola_dE=True, parabola_ds=True,
        )
        r_off = solve_alternating_projection(
            Y, dicts, parabola_dE=True, parabola_ds=True, bg_degree=-1,
        )
        assert r_default.background is None
        assert r_off.background is None
        np.testing.assert_array_equal(r_default.amplitudes, r_off.amplitudes)
        np.testing.assert_array_equal(r_default.delta_E, r_off.delta_E)
        np.testing.assert_array_equal(r_default.delta_sigma, r_off.delta_sigma)
        np.testing.assert_array_equal(r_default.chi2, r_off.chi2)

    @requires_mlx
    def test_bg_on_improves_dsigma(self):
        """With an injected background, bg_degree=1 gives lower δσ RMS than OFF.

        Uses the exact-Jacobian Newton polish, which is MLX-only (the numpy
        fallback supports the frozen-surrogate step only).
        """
        Y, dicts, ds_true = _two_peak_with_bg(
            2000, bg_const=0.05, bg_slope=0.02, seed=2,
        )
        common = dict(parabola_dE=True, parabola_ds=True,
                      apply_newton=True, newton_iter=5, newton_exact=True)
        r_off = solve_alternating_projection(Y, dicts, bg_degree=-1, **common)
        r_on = solve_alternating_projection(Y, dicts, bg_degree=1, **common)
        rms_off = _ds_rms(r_off, ds_true)
        rms_on = _ds_rms(r_on, ds_true)
        # Modeling the background removes the leak into the width estimate.
        assert rms_on < rms_off

    def test_background_field_shape(self):
        """result.background is (batch, bg_degree+1) when on, None when off."""
        Y, dicts, _ds = _two_peak_with_bg(
            128, bg_const=0.05, bg_slope=0.02, seed=3,
        )
        r_off = solve_alternating_projection(Y, dicts, bg_degree=-1)
        assert r_off.background is None
        for deg in (0, 1, 2, 3):
            r = solve_alternating_projection(Y, dicts, bg_degree=deg)
            assert r.background is not None
            assert r.background.shape == (128, deg + 1)
            assert r.amplitudes.shape == (128, 2)  # peak-amp semantics unchanged


def _two_peak_with_gamma(n, gamma_jitter=0.0, seed=0):
    """Two-peak batch (dev-log 77-style) with per-spectrum/peak TRUE γ jitter.

    Identical geometry to ``_two_peak_problem`` (7σ separation, A ≈ 1000,
    noise-free) but the Lorentzian width of each peak in each spectrum is
    γ_nom + U(−gamma_jitter, +gamma_jitter). Returns
    (Y, dicts, ds_true, dg_true).
    """
    sigma = SIGMA_NOM
    gamma = GAMMA
    c0 = 285.0
    c1 = c0 + 7.0 * sigma
    amp_peak = 1000.0
    energy = np.linspace(c0 - 6, c1 + 6, 160, dtype=np.float32)

    rng = np.random.default_rng(seed)
    dE_true = rng.uniform(-0.15, 0.15, (n, 2)).astype(np.float32)
    ds_true = rng.uniform(-0.12, 0.12, (n, 2)).astype(np.float32)
    amp_true = rng.uniform(0.6 * amp_peak, 1.4 * amp_peak, (n, 2)).astype(np.float32)
    if gamma_jitter > 0:
        dg_true = rng.uniform(-gamma_jitter, gamma_jitter, (n, 2)).astype(np.float32)
    else:
        dg_true = np.zeros((n, 2), dtype=np.float32)

    centers = np.stack([c0 + dE_true[:, 0], c1 + dE_true[:, 1]], axis=1)
    sigmas = sigma + ds_true
    Y = np.zeros((n, len(energy)), dtype=np.float32)
    for k in range(2):
        for i in range(n):
            Y[i] += amp_true[i, k] * voigt_profile(
                energy, centers[i, k], sigmas[i, k], gamma + dg_true[i, k])

    dicts = [
        build_dictionary(energy, np.array([c0]), np.array([sigma]), gamma,
                         dE_range=(-0.6, 0.6), dE_step=0.04,
                         dsigma_range=(-0.3, 0.3), dsigma_step=0.03),
        build_dictionary(energy, np.array([c1]), np.array([sigma]), gamma,
                         dE_range=(-0.6, 0.6), dE_step=0.04,
                         dsigma_range=(-0.3, 0.3), dsigma_step=0.03),
    ]
    return Y, dicts, ds_true, dg_true


@requires_mlx
class TestGammaRefinement:
    """B3b: opt-in δγ refinement in the post-AP exact Newton step."""

    def test_fit_gamma_off_unchanged_vs_default(self):
        """fit_gamma=False must reproduce the default behavior bit-for-bit."""
        Y, dicts, _ds, _dg = _two_peak_with_gamma(200, gamma_jitter=0.0, seed=1)
        common = dict(n_iterations=3, parabola_dE=True, parabola_ds=True,
                      apply_newton=True, newton_iter=3, newton_exact=True)
        r_default = solve_alternating_projection(Y, dicts, **common)
        r_off = solve_alternating_projection(Y, dicts, fit_gamma=False, **common)
        assert r_default.delta_gamma is None
        assert r_off.delta_gamma is None
        np.testing.assert_array_equal(r_default.amplitudes, r_off.amplitudes)
        np.testing.assert_array_equal(r_default.delta_E, r_off.delta_E)
        np.testing.assert_array_equal(r_default.delta_sigma, r_off.delta_sigma)
        np.testing.assert_array_equal(r_default.chi2, r_off.chi2)

    def test_fit_gamma_requires_newton_exact(self):
        """fit_gamma=True without newton_exact must raise ValueError."""
        Y, dicts, _ds, _dg = _two_peak_with_gamma(64, gamma_jitter=0.03, seed=2)
        # exact surrogate (newton_exact=False) → not allowed
        with pytest.raises(ValueError, match="newton_exact"):
            solve_alternating_projection(
                Y, dicts, n_iterations=3, apply_newton=True, newton_iter=3,
                newton_exact=False, fit_gamma=True)
        # apply_newton=False → also not allowed
        with pytest.raises(ValueError, match="apply_newton"):
            solve_alternating_projection(
                Y, dicts, n_iterations=3, apply_newton=False,
                newton_exact=True, fit_gamma=True)

    def test_fit_gamma_recovers_and_improves_dsigma(self):
        """γ-jittered: fit_gamma gives δγ (batch, n_comp) and lower δσ RMS."""
        Y, dicts, ds_true, dg_true = _two_peak_with_gamma(
            2000, gamma_jitter=0.04, seed=3)
        common = dict(n_iterations=3, parabola_dE=True, parabola_ds=True,
                      apply_newton=True, newton_iter=5, newton_exact=True)
        r_off = solve_alternating_projection(Y, dicts, fit_gamma=False, **common)
        r_on = solve_alternating_projection(Y, dicts, fit_gamma=True, **common)

        assert r_on.delta_gamma is not None
        assert r_on.delta_gamma.shape == (2000, 2)
        assert np.all(np.isfinite(r_on.delta_gamma))

        # δγ is recovered well below the injected ±0.04 jitter range.
        dg_rms = _ds_rms_named(r_on.delta_gamma, dg_true)
        assert dg_rms < 0.04, dg_rms

        # Fitting γ lowers the δσ RMS (the frozen-γ mismatch otherwise leaks
        # into the width estimate).
        rms_off = _ds_rms(r_off, ds_true)
        rms_on = _ds_rms(r_on, ds_true)
        assert rms_on < rms_off, (rms_off, rms_on)

    def test_delta_gamma_none_when_off(self):
        """result.delta_gamma is None when fit_gamma=False."""
        Y, dicts, _ds, _dg = _two_peak_with_gamma(128, gamma_jitter=0.03, seed=4)
        r = solve_alternating_projection(
            Y, dicts, n_iterations=3, apply_newton=True, newton_iter=3,
            newton_exact=True, fit_gamma=False)
        assert r.delta_gamma is None


@requires_mlx
class TestGammaChi2Gate:
    """Per-spectrum χ² gate on the δγ refinement (default-on when fit_gamma)."""

    _COMMON = dict(n_iterations=3, parabola_dE=True, parabola_ds=True,
                   apply_newton=True, newton_iter=5, newton_exact=True)

    def test_gate_default_on(self):
        """gamma_chi2_gate defaults to True (the safe behaviour)."""
        import inspect
        sig = inspect.signature(solve_alternating_projection)
        assert sig.parameters["gamma_chi2_gate"].default is True
        assert sig.parameters["gamma_chi2_margin"].default == 0.01

    def test_gate_no_damage_when_gamma_correct(self):
        """γ-correct: gated fit_gamma δσ RMS ≈ the fit_gamma=False RMS.

        The ungated δγ fit degrades δσ badly on γ-correct spectra (the extra
        DoF only fits noise via the σ↔γ valley). The gate must keep the damage
        small: gated δσ RMS within a modest factor of the no-γ RMS, and FAR
        below the ungated δγ RMS.
        """
        Y, dicts, ds_true, _dg = _two_peak_with_gamma(
            2000, gamma_jitter=0.0, seed=11)
        r_off = solve_alternating_projection(Y, dicts, fit_gamma=False,
                                             **self._COMMON)
        r_ungated = solve_alternating_projection(
            Y, dicts, fit_gamma=True, gamma_chi2_gate=False, **self._COMMON)
        r_gated = solve_alternating_projection(
            Y, dicts, fit_gamma=True, gamma_chi2_gate=True, **self._COMMON)

        rms_off = _ds_rms(r_off, ds_true)
        rms_ungated = _ds_rms(r_ungated, ds_true)
        rms_gated = _ds_rms(r_gated, ds_true)

        # The gate keeps δσ close to the no-γ baseline (within 25%), unlike the
        # ungated fit which inflates it substantially.
        assert rms_gated < rms_off * 1.25, (rms_off, rms_gated)
        assert rms_gated < rms_ungated, (rms_ungated, rms_gated)
        # δγ is 0 on the spectra the gate rejected.
        assert r_gated.delta_gamma is not None
        assert np.any(r_gated.delta_gamma == 0.0)

    def test_gate_keeps_gain_when_gamma_off(self):
        """γ-jittered: gated fit_gamma still improves δσ vs fit_gamma=False."""
        Y, dicts, ds_true, _dg = _two_peak_with_gamma(
            2000, gamma_jitter=0.04, seed=12)
        r_off = solve_alternating_projection(Y, dicts, fit_gamma=False,
                                             **self._COMMON)
        r_gated = solve_alternating_projection(
            Y, dicts, fit_gamma=True, gamma_chi2_gate=True, **self._COMMON)
        rms_off = _ds_rms(r_off, ds_true)
        rms_gated = _ds_rms(r_gated, ds_true)
        # Where γ is genuinely off, the gate keeps the δγ fit and δσ improves.
        assert rms_gated < rms_off, (rms_off, rms_gated)
        # The gate accepts δγ on a substantial fraction of γ-off spectra.
        take_on = float(np.mean(np.any(r_gated.delta_gamma != 0.0, axis=1)))
        assert take_on > 0.5, take_on

    def test_gate_only_affects_fit_gamma(self):
        """gamma_chi2_gate is a no-op unless fit_gamma=True."""
        Y, dicts, _ds, _dg = _two_peak_with_gamma(
            256, gamma_jitter=0.03, seed=13)
        common = dict(n_iterations=3, apply_newton=True, newton_iter=3,
                      newton_exact=True)
        r_gate_on = solve_alternating_projection(
            Y, dicts, fit_gamma=False, gamma_chi2_gate=True, **common)
        r_gate_off = solve_alternating_projection(
            Y, dicts, fit_gamma=False, gamma_chi2_gate=False, **common)
        assert r_gate_on.delta_gamma is None
        assert r_gate_off.delta_gamma is None
        np.testing.assert_array_equal(r_gate_on.delta_sigma,
                                      r_gate_off.delta_sigma)
        np.testing.assert_array_equal(r_gate_on.chi2, r_gate_off.chi2)


def _ds_rms_named(est, true):
    """Pooled RMS error over both peaks for an arbitrary (batch, 2) array."""
    return float(np.mean([
        np.std(est[:, k] - true[:, k]) for k in range(2)
    ]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
