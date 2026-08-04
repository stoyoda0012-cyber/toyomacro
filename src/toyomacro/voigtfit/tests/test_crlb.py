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
    _classify_crlb_dE,
    _voigt_fwhm,
    classify_solvability,
    compute_efficiency_point,
    compute_multipeak_fisher,
    crlb_grid_sweep,
    generate_ncomp_spectra,
    mean_efficiency,
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
        """All eigenvalues of the Fisher matrix must be >= 0.

        Asserted on `fisher` directly. `cr.eigenvalues` is the
        correlation matrix's spectrum, which has unit trace per
        parameter and so would not exercise the same tolerance.
        """
        centers = np.array([-0.5, 0.5, 1.5])
        cr = compute_multipeak_fisher(
            np.full(3, AMP), centers,
            np.full(3, SIGMA), np.full(3, GAMMA),
            ENERGY,
        )
        fisher_eigenvalues = np.linalg.eigvalsh(cr.fisher)
        tol = -1e-10 * max(float(fisher_eigenvalues[-1]), 1.0)
        assert np.all(fisher_eigenvalues >= tol), (
            f"Negative eigenvalue: {fisher_eigenvalues.min()}"
        )
        assert np.all(cr.eigenvalues >= -1e-10), (
            f"Negative correlation eigenvalue: {cr.eigenvalues.min()}"
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

    def test_sweep_crlb_scales_as_inverse_snr_squared(self):
        """The swept bound must follow 1/SNR**2 exactly, at every overlap.

        Ordering alone is too weak to see the defect this guards. Under
        the old rank test the SNR axis stayed monotone at all four
        overlaps here -- the values still descended, just too fast (31x
        and 186x too small at overlap 0.8 and 0.5). The exact law is
        what breaks, so the exact law is what is asserted.

        `crlb_grid_sweep` passes no `noise_model`, so this is the Poisson
        branch; its 1/f weighting with the sweep's `amplitude ~ snr**2`
        makes the law exact rather than asymptotic.
        """
        overlaps = np.array([1.5, 1.0, 0.8, 0.5])
        snr = np.array([10.0, 50.0, 200.0])
        result = crlb_grid_sweep(
            n_comp_list=[2],
            overlap_ratios=overlaps,
            snr_levels=snr,
            sigma=SIGMA, gamma=GAMMA,
            n_energy=128,
        )

        crlb_dE = np.asarray(result['crlb_dE'][2])  # (n_overlap, n_snr)

        for i, ov in enumerate(overlaps):
            row = crlb_dE[i]
            assert np.all(np.isfinite(row)), f"overlap {ov}: {row}"
            invariant = row * snr ** 2
            np.testing.assert_allclose(
                invariant, invariant[0], rtol=1e-8,
                err_msg=f"overlap {ov} breaks the inverse-square law: {row}",
            )


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
class TestSingularFisher:
    """A singular Fisher matrix must produce an infinite bound, not a small one."""

    @staticmethod
    def _fisher(n_comp, overlap, noise_std=1e-2):
        centers = _build_equal_spacing_centers(n_comp, overlap, SIGMA, GAMMA)
        fwhm = _voigt_fwhm(SIGMA, GAMMA)
        energy = np.linspace(centers[0] - 5 * fwhm, centers[-1] + 5 * fwhm, 256)
        return compute_multipeak_fisher(
            np.ones(n_comp), centers, np.full(n_comp, SIGMA), np.full(n_comp, GAMMA),
            energy, mode='3d', noise_model='gaussian', noise_std=noise_std,
        )

    def test_full_rank_case_is_untouched(self):
        """Where nothing is truncated, the bound and the pseudo-inverse agree."""
        cr = self._fisher(n_comp=3, overlap=0.3)
        assert cr.null_space_dim == 0
        assert not cr.unbounded.any()
        assert np.all(np.isfinite(cr.crlb))
        np.testing.assert_array_equal(cr.crlb, cr.crlb_pseudo)

    def test_singular_case_is_infinite_and_pseudo_stays_finite(self):
        """Four peaks at overlap 0.2 are not individually identifiable."""
        cr = self._fisher(n_comp=4, overlap=0.2)
        assert cr.null_space_dim == 1, f"expected a 1-D null space, got {cr.null_space_dim}"
        assert cr.unbounded.all(), "the null direction touches every parameter axis here"
        assert np.all(np.isinf(cr.crlb))
        assert np.all(np.isfinite(cr.crlb_pseudo))

    def test_bound_does_not_fall_as_the_problem_gets_harder(self):
        """Monotonicity, asserted on `crlb` only.

        Four peaks at overlap 0.3 are identifiable; at 0.2 they are not.
        The bound must move toward less precision across that step --
        here from finite to infinite.

        The claim is restricted to `crlb` on purpose. The same
        monotonicity does *not* hold for `crlb_pseudo`, and that is not
        an oversight: see
        test_pseudo_inverse_can_still_fall_as_information_is_lost.
        """
        easier = self._fisher(n_comp=4, overlap=0.3)
        harder = self._fisher(n_comp=4, overlap=0.2)
        assert harder.null_space_dim > easier.null_space_dim

        dE = slice(1, None, 3)
        assert np.all(np.isfinite(easier.crlb[dE])), "overlap 0.3 is still computable"
        assert np.all(np.isinf(harder.crlb[dE]))

    def test_decoupled_parameters_keep_a_finite_bound(self):
        """A singular Fisher matrix does not make *every* bound infinite.

        Two nearly coincident peaks plus one placed 10 FWHM away: the
        pair is not individually identifiable, the distant peak is. Its
        parameter axes project onto the null space at the 1e-20 level
        against 5e-01 for the degenerate pair, and they must keep their
        finite bound. This is the band `_NULL_PROJECTION_TOL` has to
        stay below.
        """
        fwhm = _voigt_fwhm(SIGMA, GAMMA)
        centers = np.array([-0.01, 0.01, 10 * fwhm])
        energy = np.linspace(centers.min() - 5 * fwhm, centers.max() + 5 * fwhm, 1024)
        cr = compute_multipeak_fisher(
            np.ones(3), centers, np.full(3, SIGMA), np.full(3, GAMMA),
            energy, mode='3d', noise_model='gaussian', noise_std=1e-2,
        )
        assert cr.null_space_dim > 0, "the coincident pair should be degenerate"

        assert np.all(np.isfinite(cr.crlb[6:])), "the distant peak is estimable"
        assert np.all(np.isinf(cr.crlb[:6])), "the coincident pair is not"

    @pytest.mark.parametrize("noise_std", [1e-2, 6.6e-3, 1e-4])
    def test_round_off_projection_does_not_buy_a_finite_bound(self, noise_std):
        """The other band: a coupled axis must not escape on rounding noise.

        Three peaks at overlap 0.15 are rank deficient and every axis is
        coupled into the degeneracy, but one projects at only 5.5e-12 --
        rounding noise in an eigenvector of a numerically singular
        matrix, not decoupling. With the tolerance inside that band it
        kept a finite `crlb` taken from the truncated pseudo-inverse,
        and which axis it was moved with `noise_std`, a scalar that
        multiplies the Fisher matrix and cannot change identifiability.

        Parameterised over noise_std for exactly that reason: the answer
        must not depend on it.
        """
        centers = _build_equal_spacing_centers(3, 0.15, SIGMA, GAMMA)
        fwhm = _voigt_fwhm(SIGMA, GAMMA)
        energy = np.linspace(centers[0] - 5 * fwhm, centers[-1] + 5 * fwhm, 256)
        cr = compute_multipeak_fisher(
            np.ones(3), centers, np.full(3, SIGMA), np.full(3, GAMMA),
            energy, mode='3d', noise_model='gaussian', noise_std=noise_std,
        )
        assert cr.null_space_dim > 0
        assert np.all(np.isinf(cr.crlb)), (
            f"{int(np.isfinite(cr.crlb).sum())} axes escaped as finite"
        )

    @pytest.mark.parametrize("n_comp,easier,harder", [
        (6, 0.3, 0.2),
        (5, 0.20, 0.15),
        (4, 0.20, 0.10),
    ])
    def test_pseudo_inverse_can_still_fall_as_information_is_lost(
        self, n_comp, easier, harder
    ):
        """The artefact that makes `crlb_pseudo` unusable as a bound is live.

        Dropping the null directions instead of letting them diverge
        makes the remaining diagonal *smaller*, so between two
        rank-deficient configurations the strictly less identifiable one
        can report the smaller number. This is a property of the
        pseudo-inverse itself, not of the units, and it survived the move
        to the scaled metric unchanged.

        It is pinned rather than fixed because `crlb_pseudo` is a
        deliberately finite diagnostic; `crlb` is the bound, and it is
        inf in both configurations here.
        """
        lo = self._fisher(n_comp=n_comp, overlap=easier)
        hi = self._fisher(n_comp=n_comp, overlap=harder)
        assert hi.null_space_dim > lo.null_space_dim > 0

        dE = slice(1, None, 3)
        assert hi.crlb_pseudo[dE].min() < lo.crlb_pseudo[dE].min(), (
            "the artefact is expected here; if it has genuinely gone away, "
            "the crlb_pseudo docstring and CHANGELOG need updating too"
        )
        assert np.all(np.isinf(hi.crlb[dE])) and np.all(np.isinf(lo.crlb[dE]))

    def test_pseudo_inverse_flatters_a_configuration_the_bound_rejects(self):
        """Why `crlb_pseudo` must never be presented as a bound.

        `process_multipeak` attaches a SolvabilityInfo to every result,
        and the level comes from sqrt(CRLB[dE]) relative to sigma. For
        four peaks at overlap 0.2 at SNR 1e8 -- a configuration whose
        Fisher matrix is rank deficient, so the individual centres are
        not identifiable at all -- the pseudo-inverse diagonal is small
        enough to be classified EASY, "meV precision, well-resolved".
        The bound says IMPOSSIBLE. The gap widens with SNR, because the
        bound scales as noise_std**2 and only the surviving directions
        shrink.

        The noise level is chosen for robustness, not for drama: at
        1e-6 the pseudo classification flips between SHOULDER and
        IMPOSSIBLE with the energy grid, so pinning a level there would
        be a platform-dependent test. At 1e-8 it is EASY across
        n_energy 128-1024 and padding 5-7 FWHM, and for three other
        rank-deficient configurations besides this one.
        """
        cr = self._fisher(n_comp=4, overlap=0.2, noise_std=1e-8)
        assert cr.null_space_dim == 1

        n_comp = 4
        pseudo_dE = float(np.mean([cr.crlb_pseudo[k * 3 + 1] for k in range(n_comp)]))
        true_dE = float(np.mean([cr.crlb[k * 3 + 1] for k in range(n_comp)]))

        assert _classify_crlb_dE(true_dE, SIGMA) is SolvabilityLevel.IMPOSSIBLE
        assert _classify_crlb_dE(pseudo_dE, SIGMA) is SolvabilityLevel.EASY


class TestScaleInvariance:
    """The rank decision must depend on the physics, not on the units.

    The Fisher matrix mixes units -- amplitude in area, dE and dsigma in
    eV -- so a threshold on its eigenvalues moves with the amplitude
    parameterisation. `A -> cA` sends `g -> D g D`, which is not a
    similarity transform. The bound itself transforms correctly; only
    the numerical rank test did not, and `crlb_grid_sweep` made the
    amplitude proportional to snr**2, putting that defect on its own
    SNR axis.
    """

    @pytest.mark.parametrize("amplitude", [1e-3, 1e0, 1e3])
    def test_identity_holds_for_several_peaks(self, amplitude):
        """diag(inv(C))/d**2 == diag(inv(g)), which is what makes this safe.

        The whole change rests on that identity. The single-peak
        `test_crlb_matches_inverse` exercises a 3x3 at one amplitude,
        which is where a scaling defect would be least visible.
        """
        centers = _build_equal_spacing_centers(3, 0.8, SIGMA, GAMMA)
        fwhm = _voigt_fwhm(SIGMA, GAMMA)
        energy = np.linspace(centers[0] - 5 * fwhm, centers[-1] + 5 * fwhm, 256)

        cr = compute_multipeak_fisher(
            np.full(3, amplitude), centers, np.full(3, SIGMA), np.full(3, GAMMA),
            energy, mode='3d', noise_model='gaussian', noise_std=1e-2,
        )
        assert cr.null_space_dim == 0, "this configuration should be full rank"

        np.testing.assert_allclose(
            cr.crlb, np.diag(np.linalg.inv(cr.fisher)), rtol=1e-6,
        )

    def test_correlation_inverse_diagonal_is_at_least_one(self):
        """[C^-1]_ii >= 1 identically for a correlation matrix.

        A structural sanity check on the mapping back to the caller's
        units: `crlb_pseudo * diag(g)` is `[C^-1]_ii` by construction, so
        a sign error or a misplaced `d**2` shows up here immediately.

        It cannot be a near-singularity guard, and that is structural
        rather than a matter of picking better configurations:
        `[C^-1]_ii` diverges as C approaches singularity, so the margin
        necessarily widens exactly where the inversion is least
        accurate. Measured: 1.0005 at two well-separated peaks, 2.3e7 at
        four peaks / overlap 0.3 where `cond(C)` is already 3.5e13. The
        check therefore catches gross errors -- a sign flip, a misplaced
        `d**2`, the wrong matrix inverted -- and says nothing about
        accuracy. A negative bound reaching `_classify_crlb_dE` (which
        floors at zero and would return EASY) has not been observed in
        any configuration.
        """
        for n_comp, overlap in [(2, 3.0), (3, 0.8), (5, 0.5), (3, 0.3), (4, 0.3)]:
            centers = _build_equal_spacing_centers(n_comp, overlap, SIGMA, GAMMA)
            fwhm = _voigt_fwhm(SIGMA, GAMMA)
            energy = np.linspace(centers[0] - 5 * fwhm, centers[-1] + 5 * fwhm, 256)
            cr = compute_multipeak_fisher(
                np.ones(n_comp), centers, np.full(n_comp, SIGMA),
                np.full(n_comp, GAMMA), energy,
                mode='3d', noise_model='gaussian', noise_std=1e-2,
            )
            scaled = cr.crlb_pseudo * np.diag(cr.fisher)
            assert np.all(scaled >= 1.0 - 1e-6), (
                f"n_comp={n_comp} overlap={overlap}: [C^-1]_ii dipped to "
                f"{scaled.min():.6f}; the inversion is failing"
            )

    def test_rank_decision_is_invariant_under_amplitude_rescaling(self):
        """Same physics, five amplitude units, one answer."""
        centers = _build_equal_spacing_centers(3, 0.3, SIGMA, GAMMA)
        fwhm = _voigt_fwhm(SIGMA, GAMMA)
        energy = np.linspace(centers[0] - 5 * fwhm, centers[-1] + 5 * fwhm, 256)

        results = [
            compute_multipeak_fisher(
                np.full(3, amp), centers, np.full(3, SIGMA), np.full(3, GAMMA),
                energy, mode='3d', noise_model='gaussian', noise_std=1e-2,
            )
            for amp in (1e-2, 1e-1, 1e0, 1e1, 1e2)
        ]

        dims = {r.null_space_dim for r in results}
        assert len(dims) == 1, f"null_space_dim moved with the amplitude unit: {dims}"

        # The tolerance is set from the conditioning, not from what
        # passes: cond(C) is 2.7e10 here, so the rounding floor on any
        # quantity derived from it is eps * cond = 6.0e-6, and the
        # measured spread is 7.0e-6. 1e-4 leaves ~14x over that floor
        # while remaining far tighter than the defect it guards, where
        # cond(g) moved from 2.6e12 to 2.1e14 across the same range.
        conds = np.array([r.condition_number for r in results])
        np.testing.assert_allclose(conds, conds[0], rtol=1e-4)

    def test_bound_follows_the_inverse_square_snr_law(self):
        """The swept CRLB scales exactly as 1/SNR**2 at fixed geometry.

        `crlb_grid_sweep` passes no `noise_model`, so this is the
        Poisson branch, whose 1/f weighting combined with the sweep's
        `amplitude ~ snr**2` gives the exact inverse-square law. Under
        Gaussian weighting at fixed `noise_std` the same amplitude law
        would give 1/SNR**4.

        `crlb_grid_sweep` sets the amplitude from snr**2, so before the
        fix this ladder read [4.82e-1, 5.36e-2, inf, 2.87e-6, 2.59e-7]:
        infinite at SNR 100, finite and 186x too small at 300 and above,
        for a configuration whose correlation-matrix condition number is
        2.48e4 at every point.
        """
        snr = np.array([10.0, 30.0, 100.0, 300.0, 1000.0])
        sweep = crlb_grid_sweep(
            n_comp_list=[2], overlap_ratios=np.array([0.5]), snr_levels=snr,
            sigma=SIGMA, gamma=GAMMA,
        )
        crlb_dE = np.asarray(sweep['crlb_dE'][2])[0]

        assert np.all(np.isfinite(crlb_dE)), f"not identifiable anywhere? {crlb_dE}"
        assert np.all(np.diff(crlb_dE) < 0), f"not monotone in SNR: {crlb_dE}"
        np.testing.assert_allclose(crlb_dE * snr ** 2, crlb_dE[0] * snr[0] ** 2, rtol=1e-6)


class TestMeanEfficiency:
    """The efficiency aggregation must give 1.0 for an ideal estimator."""

    @pytest.mark.parametrize("crlb_per", [
        np.array([1e-4, 1e-4, 1e-4]),          # equal bounds
        np.array([8.8e1, 3.0e2, 7.7e2]),       # 3 peaks, overlap 0.3
        np.array([2.2e1, 1.5e2, 4.0e2, 6.0e2, 8.4e2]),  # 5 peaks, overlap 0.5
    ])
    def test_ideal_estimator_returns_one(self, crlb_per):
        """RMSE_k = sqrt(CRLB_k) in every component must give exactly 1.0.

        The previous aggregation, mean(CRLB) / mean(RMSE)**2, divides a
        mean of variances by the square of a mean of standard
        deviations, so Jensen puts it above 1 whenever the
        per-component bounds differ: 1.17 and 1.23 for the second and
        third vectors here, and 1.16 (three peaks at overlap 0.3) and
        1.51 (five at overlap 0.5) on the Fisher matrices those vectors
        are modelled on. Only the equal-bounds case hid the defect,
        which is why the well-separated two-peak smoke test never saw
        it.
        """
        rmse_per = np.sqrt(crlb_per)
        assert mean_efficiency(crlb_per, rmse_per) == pytest.approx(1.0, rel=1e-12)

        old = float(np.mean(crlb_per) / np.mean(rmse_per) ** 2)
        if len(set(crlb_per.tolist())) > 1:
            assert old > 1.0, "the old formula should be the one that inflates"

    def test_efficiency_point_actually_uses_this_aggregation(self):
        """`mean_efficiency` being right does not prove the caller uses it.

        Reverting only the call site leaves the rest of the suite green,
        because the solver-level smoke test sits at a symmetric two-peak
        configuration where the per-component bounds are equal and both
        formulas agree. This one picks a configuration where they do not.
        """
        try:
            er = compute_efficiency_point(
                n_comp=3, overlap_ratio=0.5, snr=100.0,
                sigma=SIGMA, gamma=GAMMA,
                n_energy=128, n_spectra=500, seed=42,
            )
        except ImportError:
            pytest.skip("MLX not available")

        crlb_per = np.array([er.crlb.crlb_per_component[k]['dE'] for k in range(3)])
        rmse_per = np.asarray(er.rmse_dE_per)

        per_component = float(np.mean(crlb_per / rmse_per ** 2))
        pooled = float(np.mean(crlb_per) / np.mean(rmse_per) ** 2)

        assert per_component != pytest.approx(pooled, rel=1e-3), (
            "this configuration does not discriminate between the two formulas"
        )
        assert er.efficiency_dE == pytest.approx(per_component, rel=1e-9)

    def test_zero_rmse_is_floored_not_infinite(self):
        """A noiseless component must not turn the mean into inf."""
        eff = mean_efficiency(np.array([1e-4, 1e-4]), np.array([1e-2, 0.0]))
        assert np.isfinite(eff)


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
