"""
Tests for Cramer-Rao Lower Bound (CRLB) multi-peak module.

Validates:
1. Single-peak consistency with existing fisher_information.py
2. Symmetry properties for equal-spacing configs
3. Overlap scaling (ill-conditioning)
4. SNR scaling
5. Well-separated limit (block-diagonal)
6. Positive semi-definiteness
7. Grid sweep smoke test
"""

import importlib.util

import numpy as np
import pytest

HAS_MLX = importlib.util.find_spec("mlx") is not None
needs_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX required for multipeak solver")

from toyomacro.voigtfit.crlb import (
    FWHM_TO_SIGMA,
    SolvabilityInfo,
    SolvabilityLevel,
    _build_equal_spacing_centers,
    _voigt_fwhm,
    classify_solvability,
    compute_efficiency_point,
    compute_multipeak_fisher,
    crlb_grid_sweep,
    generate_ncomp_spectra,
)
from toyomacro.voigtfit.fisher_information import compute_fisher_matrix
from toyomacro.voigtfit.multipeak_config import ComponentConfig, MultiPeakConfig

# Standard test parameters
SIGMA = 0.5
GAMMA = 0.3
ENERGY = np.linspace(-5, 5, 256)
AMP = 1000.0


class TestSinglePeakConsistency:
    """n_comp=1 must reproduce existing fisher_information.py results."""

    def test_fisher_matrix_matches(self):
        """compute_multipeak_fisher(n_comp=1) == compute_fisher_matrix()."""
        # Existing single-peak
        fr = compute_fisher_matrix(AMP, 0.0, SIGMA, GAMMA, ENERGY, mode='3d')

        # New multi-peak with n_comp=1
        cr = compute_multipeak_fisher(
            np.array([AMP]), np.array([0.0]),
            np.array([SIGMA]), np.array([GAMMA]),
            ENERGY, mode='3d',
        )

        np.testing.assert_allclose(cr.fisher, fr.g, rtol=1e-10)

    def test_crlb_matches_inverse(self):
        """CRLB = diag(inv(Fisher)) for well-conditioned single peak."""
        cr = compute_multipeak_fisher(
            np.array([AMP]), np.array([0.0]),
            np.array([SIGMA]), np.array([GAMMA]),
            ENERGY, mode='3d',
        )
        # Direct inverse
        fisher_inv_direct = np.linalg.inv(cr.fisher)
        expected_crlb = np.diag(fisher_inv_direct)

        np.testing.assert_allclose(cr.crlb, expected_crlb, rtol=1e-8)

    def test_4d_mode_matches(self):
        """4D mode also matches existing fisher_information."""
        fr = compute_fisher_matrix(AMP, 0.0, SIGMA, GAMMA, ENERGY, mode='4d')
        cr = compute_multipeak_fisher(
            np.array([AMP]), np.array([0.0]),
            np.array([SIGMA]), np.array([GAMMA]),
            ENERGY, mode='4d',
        )
        np.testing.assert_allclose(cr.fisher, fr.g, rtol=1e-10)
        assert cr.n_params_per_comp == 4


class TestSymmetry:
    """Equal-spacing, equal-amplitude configs must have symmetric CRLB."""

    def test_2comp_symmetric(self):
        """Two equal peaks → same CRLB for both."""
        centers = np.array([-1.0, 1.0])
        amps = np.array([AMP, AMP])
        sigs = np.array([SIGMA, SIGMA])
        gams = np.array([GAMMA, GAMMA])

        cr = compute_multipeak_fisher(amps, centers, sigs, gams, ENERGY)

        # CRLB for comp 0 and comp 1 should be identical
        for name in ('amp', 'dE', 'dsigma'):
            val0 = cr.crlb_per_component[0][name]
            val1 = cr.crlb_per_component[1][name]
            np.testing.assert_allclose(val0, val1, rtol=1e-8,
                                       err_msg=f"CRLB[{name}] not symmetric")

    def test_3comp_symmetric(self):
        """Three equal peaks at equal spacing."""
        centers = np.array([-2.0, 0.0, 2.0])
        amps = np.full(3, AMP)
        sigs = np.full(3, SIGMA)
        gams = np.full(3, GAMMA)

        cr = compute_multipeak_fisher(amps, centers, sigs, gams, ENERGY)

        # Outer peaks (0 and 2) should have same CRLB
        for name in ('amp', 'dE', 'dsigma'):
            np.testing.assert_allclose(
                cr.crlb_per_component[0][name],
                cr.crlb_per_component[2][name],
                rtol=1e-8,
            )

        # Center peak may differ from outer peaks (that's fine)
        # But all should be finite and positive
        for k in range(3):
            for name in ('amp', 'dE', 'dsigma'):
                assert cr.crlb_per_component[k][name] > 0


class TestOverlapScaling:
    """Closer peaks → higher condition number → larger CRLB."""

    def test_condition_number_increases_with_overlap(self):
        """Reducing separation increases condition number."""
        cond_numbers = []
        for sep in [3.0, 2.0, 1.0, 0.5]:
            centers = np.array([-sep / 2, sep / 2])
            cr = compute_multipeak_fisher(
                np.array([AMP, AMP]), centers,
                np.array([SIGMA, SIGMA]), np.array([GAMMA, GAMMA]),
                ENERGY,
            )
            cond_numbers.append(cr.condition_number)

        # Condition number should be monotonically increasing
        for i in range(len(cond_numbers) - 1):
            assert cond_numbers[i + 1] > cond_numbers[i], (
                f"cond[sep={[3,2,1,0.5][i+1]}]={cond_numbers[i+1]:.1e} "
                f"<= cond[sep={[3,2,1,0.5][i]}]={cond_numbers[i]:.1e}"
            )

    def test_crlb_dE_increases_with_overlap(self):
        """CRLB for center shift grows as peaks overlap more."""
        crlb_dE_values = []
        for sep in [3.0, 1.5, 0.5]:
            centers = np.array([-sep / 2, sep / 2])
            cr = compute_multipeak_fisher(
                np.array([AMP, AMP]), centers,
                np.array([SIGMA, SIGMA]), np.array([GAMMA, GAMMA]),
                ENERGY,
            )
            crlb_dE_values.append(cr.crlb_per_component[0]['dE'])

        # Monotonically increasing
        for i in range(len(crlb_dE_values) - 1):
            assert crlb_dE_values[i + 1] > crlb_dE_values[i]


class TestSNRScaling:
    """CRLB scales inversely with signal strength."""

    def test_crlb_scales_inversely_with_amplitude(self):
        """Doubling amplitude halves CRLB (for well-separated peak)."""
        cr1 = compute_multipeak_fisher(
            np.array([AMP]), np.array([0.0]),
            np.array([SIGMA]), np.array([GAMMA]),
            ENERGY,
        )
        cr2 = compute_multipeak_fisher(
            np.array([2 * AMP]), np.array([0.0]),
            np.array([SIGMA]), np.array([GAMMA]),
            ENERGY,
        )
        # For Poisson noise, CRLB(amp) * amp ∝ 1/amp
        # So CRLB(dE) at 2*AMP should be ~ 0.5 * CRLB(dE) at AMP
        ratio = cr2.crlb_per_component[0]['dE'] / cr1.crlb_per_component[0]['dE']
        np.testing.assert_allclose(ratio, 0.5, rtol=0.05)


class TestWellSeparatedLimit:
    """Well-separated peaks → block-diagonal Fisher (independent estimation)."""

    def test_block_diagonal_at_large_separation(self):
        """Fisher off-diagonal blocks vanish for widely separated peaks."""
        # Large separation: 50 * FWHM
        fwhm = _voigt_fwhm(SIGMA, GAMMA)
        sep = 50 * fwhm
        wide_energy = np.linspace(-sep - 5, sep + 5, 2048)

        centers = np.array([-sep / 2, sep / 2])
        cr = compute_multipeak_fisher(
            np.array([AMP, AMP]), centers,
            np.array([SIGMA, SIGMA]), np.array([GAMMA, GAMMA]),
            wide_energy,
        )

        # Off-diagonal block (params 0-2 vs params 3-5) should be ~0
        n_per = 3
        off_diag_block = cr.fisher[:n_per, n_per:]
        on_diag_norm = np.linalg.norm(cr.fisher[:n_per, :n_per])

        relative_coupling = np.linalg.norm(off_diag_block) / on_diag_norm
        assert relative_coupling < 1e-4, (
            f"Off-diagonal coupling {relative_coupling:.2e} too large for separated peaks"
        )

    def test_crlb_matches_independent_at_large_separation(self):
        """CRLB of multi-peak ≈ CRLB of independent single peaks when well-separated."""
        fwhm = _voigt_fwhm(SIGMA, GAMMA)
        sep = 50 * fwhm
        # Use same energy density (points per FWHM) for both
        n_pts = 2048
        wide_energy = np.linspace(-sep - 5, sep + 5, n_pts)

        # Multi-peak
        cr_multi = compute_multipeak_fisher(
            np.array([AMP, AMP]),
            np.array([-sep / 2, sep / 2]),
            np.array([SIGMA, SIGMA]),
            np.array([GAMMA, GAMMA]),
            wide_energy,
        )

        # Single peak with matched energy density
        # Energy points per unit ≈ n_pts / (2*sep + 10)
        pts_per_unit = n_pts / (2 * sep + 10)
        single_range = 10.0  # ±5 around peak
        n_single = max(128, int(pts_per_unit * single_range))
        single_energy = np.linspace(-5, 5, n_single)
        cr_single = compute_multipeak_fisher(
            np.array([AMP]), np.array([0.0]),
            np.array([SIGMA]), np.array([GAMMA]),
            single_energy,
        )

        # CRLB for dE and dsigma should roughly match
        # Fisher ∝ n_energy_points covering the peak, so ratio depends on sampling
        # Use generous tolerance since energy grids differ
        for name in ('dE', 'dsigma'):
            ratio = (cr_multi.crlb_per_component[0][name]
                     / cr_single.crlb_per_component[0][name])
            assert 0.2 < ratio < 5.0, (
                f"CRLB[{name}] ratio {ratio:.2f} outside [0.2, 5.0]"
            )


class TestPositiveDefiniteness:
    """Fisher matrix must be PSD; CRLB must be non-negative."""

    def test_fisher_psd(self):
        """All eigenvalues of Fisher matrix must be >= 0."""
        centers = np.array([-0.5, 0.5, 1.5])
        cr = compute_multipeak_fisher(
            np.full(3, AMP), centers,
            np.full(3, SIGMA), np.full(3, GAMMA),
            ENERGY,
        )
        assert np.all(cr.eigenvalues >= -1e-10), (
            f"Negative eigenvalue: {cr.eigenvalues.min()}"
        )

    def test_crlb_non_negative(self):
        """CRLB values must be non-negative."""
        centers = np.array([-0.3, 0.3])  # Close overlap
        cr = compute_multipeak_fisher(
            np.array([AMP, AMP]), centers,
            np.array([SIGMA, SIGMA]), np.array([GAMMA, GAMMA]),
            ENERGY,
        )
        assert np.all(cr.crlb >= 0), f"Negative CRLB: {cr.crlb.min()}"


class TestGridSweep:
    """Smoke tests for crlb_grid_sweep."""

    def test_basic_sweep(self):
        """Grid sweep runs and returns expected structure."""
        result = crlb_grid_sweep(
            n_comp_list=[2, 3],
            overlap_ratios=np.array([0.5, 1.0, 2.0]),
            snr_levels=np.array([10, 100]),
            sigma=SIGMA, gamma=GAMMA,
            n_energy=128,
        )

        assert 'crlb_amp' in result
        assert 'crlb_dE' in result
        assert 'condition_number' in result

        # Check shapes
        for nc in [2, 3]:
            assert result['crlb_amp'][nc].shape == (3, 2)
            assert result['crlb_dE'][nc].shape == (3, 2)

    def test_sweep_crlb_ordering(self):
        """Higher SNR → lower CRLB in sweep results."""
        result = crlb_grid_sweep(
            n_comp_list=[2],
            overlap_ratios=np.array([1.5]),
            snr_levels=np.array([10, 50, 200]),
            sigma=SIGMA, gamma=GAMMA,
            n_energy=128,
        )

        crlb_dE = result['crlb_dE'][2]  # shape (1, 3)
        # Higher SNR → lower CRLB
        assert crlb_dE[0, 0] > crlb_dE[0, 1] > crlb_dE[0, 2]


class TestHelpers:
    """Test helper functions."""

    def test_voigt_fwhm(self):
        """Thompson approximation gives reasonable FWHM."""
        fwhm = _voigt_fwhm(SIGMA, GAMMA)
        f_G = SIGMA / FWHM_TO_SIGMA
        f_L = 2 * GAMMA
        # FWHM should be between Gaussian and sum
        assert fwhm >= max(f_G, f_L)
        assert fwhm <= f_G + f_L

    def test_equal_spacing_centers(self):
        """Centers are symmetric and equally spaced."""
        centers = _build_equal_spacing_centers(3, 1.5, SIGMA, GAMMA)
        assert len(centers) == 3
        np.testing.assert_allclose(centers[1], 0.0, atol=1e-10)
        np.testing.assert_allclose(centers[2] - centers[1],
                                   centers[1] - centers[0], rtol=1e-10)

    def test_equal_spacing_overlap_ratio(self):
        """Spacing matches requested overlap ratio * FWHM."""
        fwhm = _voigt_fwhm(SIGMA, GAMMA)
        ratio = 2.0
        centers = _build_equal_spacing_centers(2, ratio, SIGMA, GAMMA)
        actual_spacing = centers[1] - centers[0]
        np.testing.assert_allclose(actual_spacing, ratio * fwhm, rtol=1e-10)


class TestGaussianNoiseModel:
    """Tests for Gaussian noise CRLB."""

    def test_gaussian_crlb_scales_with_noise_std(self):
        """Doubling noise_std quadruples CRLB (variance ∝ sigma^2)."""
        cr1 = compute_multipeak_fisher(
            np.array([AMP]), np.array([0.0]),
            np.array([SIGMA]), np.array([GAMMA]),
            ENERGY, noise_model='gaussian', noise_std=1.0,
        )
        cr2 = compute_multipeak_fisher(
            np.array([AMP]), np.array([0.0]),
            np.array([SIGMA]), np.array([GAMMA]),
            ENERGY, noise_model='gaussian', noise_std=2.0,
        )
        # CRLB ∝ sigma_noise^2
        ratio = cr2.crlb_per_component[0]['dE'] / cr1.crlb_per_component[0]['dE']
        np.testing.assert_allclose(ratio, 4.0, rtol=1e-10)

    def test_gaussian_requires_noise_std(self):
        """Gaussian model without noise_std raises ValueError."""
        with pytest.raises(ValueError, match="noise_std"):
            compute_multipeak_fisher(
                np.array([AMP]), np.array([0.0]),
                np.array([SIGMA]), np.array([GAMMA]),
                ENERGY, noise_model='gaussian',
            )

    def test_gaussian_fisher_psd(self):
        """Gaussian Fisher matrix is PSD."""
        cr = compute_multipeak_fisher(
            np.array([AMP, AMP]), np.array([-0.5, 0.5]),
            np.array([SIGMA, SIGMA]), np.array([GAMMA, GAMMA]),
            ENERGY, noise_model='gaussian', noise_std=0.1,
        )
        assert np.all(cr.eigenvalues >= -1e-10)


class TestNcompSpectraGeneration:
    """Tests for generate_ncomp_spectra."""

    def test_output_shape(self):
        """Output shapes match expected dimensions."""
        energy = np.linspace(-5, 5, 128, dtype=np.float32)
        Y, gt = generate_ncomp_spectra(
            n_spectra=100, centers=np.array([0.0, 2.0]),
            sigmas=np.array([SIGMA, SIGMA]),
            gammas=np.array([GAMMA, GAMMA]),
            amplitudes=np.array([1.0, 1.0]),
            energy=energy, snr=100.0,
        )
        assert Y.shape == (100, 128)
        assert gt['dE'].shape == (100, 2)
        assert gt['ds'].shape == (100, 2)
        assert gt['amp'].shape == (100, 2)

    def test_noiseless_positive(self):
        """Noiseless spectra are non-negative."""
        energy = np.linspace(-5, 5, 128, dtype=np.float32)
        Y, _ = generate_ncomp_spectra(
            n_spectra=10, centers=np.array([0.0]),
            sigmas=np.array([SIGMA]),
            gammas=np.array([GAMMA]),
            amplitudes=np.array([1.0]),
            energy=energy, snr=float('inf'),
        )
        assert np.all(Y >= 0)

    def test_reproducible_with_seed(self):
        """Same seed gives same output."""
        energy = np.linspace(-5, 5, 64, dtype=np.float32)
        kwargs = dict(
            n_spectra=50, centers=np.array([0.0, 1.0]),
            sigmas=np.array([SIGMA, SIGMA]),
            gammas=np.array([GAMMA, GAMMA]),
            amplitudes=np.array([1.0, 1.0]),
            energy=energy, snr=100.0, seed=123,
        )
        Y1, gt1 = generate_ncomp_spectra(**kwargs)
        Y2, gt2 = generate_ncomp_spectra(**kwargs)
        np.testing.assert_array_equal(Y1, Y2)
        np.testing.assert_array_equal(gt1['dE'], gt2['dE'])
class TestEfficiencyPoint:
    """Smoke test for compute_efficiency_point (requires MLX for solver)."""

    def test_well_separated_high_snr(self):
        """Well-separated peaks at high SNR: efficiency of order 1, amplitudes unbiased.

        The amplitude assertion is the load-bearing one. The generated
        spectra, the Fisher matrix and the solver basis must all use the
        same amplitude convention -- the coefficient of a unit-area
        Voigt. When they did not (spectra peak-normalised, Fisher and
        solver area-normalised), the recovered amplitudes were offset by
        1/V_max - 1 and `rmse_amp` measured that offset rather than any
        estimator error: 0.917 instead of 0.0025 at this configuration,
        a factor of 363. `efficiency_dE` moved too, but *upward*, to
        2.04 -- past the bound it is measured against -- so the previous
        one-sided `> 0.01` could not see it. Hence the two-sided bracket.

        Both thresholds are set from measurement with ~20x margin; the
        bracket is deliberately wide because the remaining known
        artefacts in this ratio (see EfficiencyResult) are unfixed.
        """
        try:
            er = compute_efficiency_point(
                n_comp=2, overlap_ratio=3.0, snr=500,
                sigma=SIGMA, gamma=GAMMA,
                n_energy=128, n_spectra=2_000, seed=42,
            )
            assert er.rmse_amp < 0.05, (
                f"Amplitude RMSE {er.rmse_amp:.5f} is too large to be estimator "
                "error at this SNR; the generator, Fisher and solver amplitude "
                "conventions have diverged"
            )
            assert 0.1 < er.efficiency_dE < 1.5, (
                f"Efficiency out of range: {er.efficiency_dE:.4f}"
            )
            assert er.rmse_dE > 0, "RMSE should be positive"
            assert er.n_comp == 2
        except ImportError:
            pytest.skip("MLX not available")


class TestSolvabilityClassification:
    """Tests for classify_solvability and SolvabilityLevel."""

    def test_well_separated_is_easy(self):
        """Well-separated peaks at high SNR → EASY."""
        info = classify_solvability(
            centers=np.array([-3.0, 3.0]),
            sigmas=np.array([SIGMA, SIGMA]),
            gammas=np.array([GAMMA, GAMMA]),
            snr=500,
        )
        assert info.level == SolvabilityLevel.EASY
        assert info.crlb_dE_meV < 10

    def test_overlapping_is_impossible(self):
        """Heavily overlapping peaks → SHOULDER or IMPOSSIBLE."""
        fwhm = _voigt_fwhm(SIGMA, GAMMA)
        sep = 0.3 * fwhm  # 0.3 FWHM
        info = classify_solvability(
            centers=np.array([-sep / 2, sep / 2]),
            sigmas=np.array([SIGMA, SIGMA]),
            gammas=np.array([GAMMA, GAMMA]),
            snr=100,
        )
        assert info.level in (SolvabilityLevel.SHOULDER, SolvabilityLevel.IMPOSSIBLE)

    def test_str_representation(self):
        """SolvabilityInfo has readable str."""
        info = classify_solvability(
            centers=np.array([0.0, 2.0]),
            sigmas=np.array([SIGMA, SIGMA]),
            gammas=np.array([GAMMA, GAMMA]),
        )
        s = str(info)
        assert 'CRLB(dE)' in s
        assert 'n_comp=2' in s

    def test_per_component_info(self):
        """Per-component solvability is populated."""
        info = classify_solvability(
            centers=np.array([-1.0, 0.0, 1.0]),
            sigmas=np.array([SIGMA, SIGMA, SIGMA]),
            gammas=np.array([GAMMA, GAMMA, GAMMA]),
        )
        assert len(info.per_component) == 3
        for level, crlb_meV in info.per_component:
            assert isinstance(level, SolvabilityLevel)
            assert crlb_meV >= 0

    def test_overall_is_worst_component(self):
        """Overall level should be worst (hardest) component."""
        info = classify_solvability(
            centers=np.array([-0.3, 0.0, 5.0]),  # 2 close + 1 far
            sigmas=np.array([SIGMA, SIGMA, SIGMA]),
            gammas=np.array([GAMMA, GAMMA, GAMMA]),
            snr=100,
        )
        comp_levels = [lev for lev, _ in info.per_component]
        level_order = [SolvabilityLevel.EASY, SolvabilityLevel.HARD,
                       SolvabilityLevel.SHOULDER, SolvabilityLevel.IMPOSSIBLE]
        worst = max(comp_levels, key=lambda x: level_order.index(x))
        assert info.level == worst


class TestMultiPeakConfigSolvability:
    """Tests for MultiPeakConfig.solvability() integration."""

    def test_config_solvability(self):
        """MultiPeakConfig.solvability() returns SolvabilityInfo."""
        config = MultiPeakConfig(
            peaks=[
                ComponentConfig(center=284.0, sigma=SIGMA, gamma=GAMMA),
                ComponentConfig(center=286.0, sigma=SIGMA, gamma=GAMMA),
            ],
            energy_axis=np.linspace(280, 290, 128),
        )
        info = config.solvability(snr=100)
        assert isinstance(info, SolvabilityInfo)
        assert info.n_comp == 2

    def test_config_close_peaks_harder(self):
        """Closer peaks → worse solvability level."""
        energy = np.linspace(280, 290, 128)

        config_far = MultiPeakConfig(
            peaks=[
                ComponentConfig(center=282.0, sigma=SIGMA, gamma=GAMMA),
                ComponentConfig(center=288.0, sigma=SIGMA, gamma=GAMMA),
            ],
            energy_axis=energy,
        )
        config_close = MultiPeakConfig(
            peaks=[
                ComponentConfig(center=284.5, sigma=SIGMA, gamma=GAMMA),
                ComponentConfig(center=285.5, sigma=SIGMA, gamma=GAMMA),
            ],
            energy_axis=energy,
        )
        info_far = config_far.solvability(snr=100)
        info_close = config_close.solvability(snr=100)

        assert info_close.crlb_dE_meV >= info_far.crlb_dE_meV
    def test_solver_annotates_result(self):
        """process_multipeak populates solvability field."""
        try:
            from toyomacro.voigtfit.multipeak_solver import process_multipeak

            config = MultiPeakConfig(
                peaks=[
                    ComponentConfig(center=284.0, sigma=0.425, gamma=0.125),
                    ComponentConfig(center=287.0, sigma=0.425, gamma=0.125),
                ],
                energy_axis=np.linspace(280, 290, 101, dtype=np.float32),
            )
            Y = np.random.default_rng(42).normal(0.5, 0.01, (100, 101)).astype(np.float32)
            result = process_multipeak(Y, config)

            assert result.solvability is not None
            assert isinstance(result.solvability.level, SolvabilityLevel)
        except ImportError:
            pytest.skip("MLX not available")
