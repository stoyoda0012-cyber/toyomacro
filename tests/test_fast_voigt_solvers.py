"""Tests for FastVoigtFitter solver routing and template BindingEnergy integration."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from toyomacro.fitting.fast_voigt import VOIGTFIT_AVAILABLE, FastFitConfig

HAS_MLX = importlib.util.find_spec("mlx") is not None

needs_voigtfit = pytest.mark.skipif(
    not VOIGTFIT_AVAILABLE, reason="voigtfit not available"
)

# FastVoigtFitter routes through the multipeak solver which requires
# ------------------------------------------------------------------
# FastFitConfig solver_name mapping
# ------------------------------------------------------------------


class TestFastFitConfigSolverName:
    """Verify mode → solver_name mapping."""

    def test_preview_maps_to_amp_only(self):
        assert FastFitConfig(mode="preview").solver_name == "amp_only"

    def test_fast_maps_to_4step(self):
        assert FastFitConfig(mode="fast").solver_name == "4step"

    def test_precise_maps_to_adaptive(self):
        assert FastFitConfig(mode="precise").solver_name == "adaptive"

    def test_default_mode_is_fast(self):
        assert FastFitConfig().mode == "fast"
        assert FastFitConfig().solver_name == "4step"


# ------------------------------------------------------------------
# FastVoigtFitter batch fitting
# ------------------------------------------------------------------


def _make_synthetic_batch(
    n_spectra: int = 100, n_energy: int = 200, n_comp: int = 3, seed: int = 42,
) -> tuple:
    """Create synthetic Voigt spectra for testing.

    Returns (spectra_rowmajor, energy, centers, sigmas, gamma).
    """
    from toyomacro.lineshape import Voigt

    rng = np.random.default_rng(seed)
    voigt = Voigt()

    energy = np.linspace(95.0, 105.0, n_energy)
    centers = np.array([98.0, 100.0, 102.0])[:n_comp]
    sigmas = np.array([0.4, 0.5, 0.4])[:n_comp]
    gamma = 0.15

    spectra = np.zeros((n_spectra, n_energy), dtype=np.float32)
    for i in range(n_spectra):
        for j in range(n_comp):
            amp = rng.uniform(500, 2000)
            spectra[i] += voigt.evaluate(
                energy, center=centers[j], amplitude=amp,
                fwhm_g=sigmas[j] * 2.3548, fwhm_l=gamma * 2,
            ).astype(np.float32)
        spectra[i] += rng.normal(0, 50, n_energy).astype(np.float32)

    return spectra, energy, centers, sigmas, gamma


@needs_voigtfit
class TestFastVoigtFitterPreview:
    """Test preview mode (amp-only)."""

    def test_fit_batch_rowmajor_shape(self):
        from toyomacro.fitting import FastVoigtFitter

        spectra, energy, centers, sigmas, gamma = _make_synthetic_batch()
        fitter = FastVoigtFitter(FastFitConfig(mode="preview"))
        result = fitter.fit_batch_rowmajor(
            spectra, energy, centers, sigmas, gamma,
        )
        assert result.amplitudes.shape == (3, 100)
        assert result.chi2.shape == (100,)
        assert result.spectra_per_second > 0

    def test_fit_batch_colmajor_delegates(self):
        """fit_batch() with col-major input produces same result."""
        from toyomacro.fitting import FastVoigtFitter

        spectra, energy, centers, sigmas, gamma = _make_synthetic_batch(n_spectra=50)
        fitter = FastVoigtFitter(FastFitConfig(mode="preview"))

        # Row-major path
        result_row = fitter.fit_batch_rowmajor(
            spectra, energy, centers, sigmas, gamma,
        )
        # Col-major path (transpose input)
        result_col = fitter.fit_batch(
            spectra.T, energy, centers, sigmas, gamma,
        )
        np.testing.assert_allclose(
            result_row.amplitudes, result_col.amplitudes, rtol=1e-5,
        )

    def test_fit_single(self):
        from toyomacro.fitting import FastVoigtFitter

        spectra, energy, centers, sigmas, gamma = _make_synthetic_batch(n_spectra=1)
        fitter = FastVoigtFitter(FastFitConfig(mode="preview"))
        result = fitter.fit_single(spectra[0], energy, centers, sigmas, gamma)
        assert result.n_spectra == 1
        assert result.n_components == 3


@needs_voigtfit
class TestFastVoigtFitterFast:
    """Test fast mode (4-step solver)."""

    def test_4step_returns_per_spectrum_params(self):
        from toyomacro.fitting import FastVoigtFitter

        spectra, energy, centers, sigmas, gamma = _make_synthetic_batch()
        fitter = FastVoigtFitter(FastFitConfig(mode="fast"))
        result = fitter.fit_batch_rowmajor(
            spectra, energy, centers, sigmas, gamma,
        )
        assert result.amplitudes.shape == (3, 100)
        assert result.chi2.shape == (100,)
        # 4-step returns per-spectrum centers/sigmas
        assert result.centers.ndim == 2
        assert result.centers.shape == (3, 100)
        assert result.sigmas.ndim == 2
        assert result.sigmas.shape == (3, 100)


@needs_voigtfit
class TestFastVoigtFitterPrecise:
    """Test precise mode (adaptive solver)."""

    def test_adaptive_runs_without_error(self):
        from toyomacro.fitting import FastVoigtFitter

        spectra, energy, centers, sigmas, gamma = _make_synthetic_batch(n_spectra=50)
        fitter = FastVoigtFitter(FastFitConfig(mode="precise"))
        result = fitter.fit_batch_rowmajor(
            spectra, energy, centers, sigmas, gamma,
        )
        assert result.amplitudes.shape[0] == 3
        assert result.chi2.shape[0] == 50


# ------------------------------------------------------------------
# to_fitpara with per-spectrum params
# ------------------------------------------------------------------


@needs_voigtfit
class TestToFitpara:
    """Test FastFitResult.to_fitpara() with 1D and 2D params."""

    def test_to_fitpara_1d_params(self):
        from toyomacro.fitting import FastVoigtFitter

        spectra, energy, centers, sigmas, gamma = _make_synthetic_batch(n_spectra=10)
        fitter = FastVoigtFitter(FastFitConfig(mode="preview"))
        result = fitter.fit_batch_rowmajor(spectra, energy, centers, sigmas, gamma)

        fitpara = result.to_fitpara(energy=energy)
        assert fitpara.shape == (10, 6, 9)  # max_components default=6
        # Col 0 = amplitude, should be non-negative for NNLS
        assert np.all(fitpara[:, :3, 0] >= 0)
        # Col 7 = function type = 1 (Voigt)
        assert np.all(fitpara[:, :3, 7] == 1)

    def test_to_fitpara_2d_params(self):
        from toyomacro.fitting import FastVoigtFitter

        spectra, energy, centers, sigmas, gamma = _make_synthetic_batch(n_spectra=10)
        fitter = FastVoigtFitter(FastFitConfig(mode="fast"))
        result = fitter.fit_batch_rowmajor(spectra, energy, centers, sigmas, gamma)

        fitpara = result.to_fitpara(energy=energy)
        assert fitpara.shape == (10, 6, 9)
        # Areas should be computed
        assert np.any(fitpara[:, :3, 8] > 0)


# ------------------------------------------------------------------
# FittingTemplate.get_reference_be
# ------------------------------------------------------------------


class TestTemplateGetReferenceBe:
    """Test FittingTemplate.get_reference_be() with explicit and fallback."""

    def test_explicit_reference_be(self):
        from toyomacro.fitting.templates import get_template

        t = get_template("Si2p_oxide")
        assert t.get_reference_be() == 99.3  # Explicit value

    def test_explicit_overrides_database(self):
        """Explicit reference_be takes priority over database."""
        from toyomacro.fitting.templates import get_template

        t = get_template("Si2p_oxide")
        # Database has 99.0, template has 99.3
        assert t.get_reference_be() == 99.3

    def test_database_fallback(self):
        """When reference_be is None, falls back to BindingEnergy database."""
        from toyomacro.fitting.templates import FittingTemplate, TemplatePeak

        t = FittingTemplate(
            name="_test_no_ref_be",
            element="Si2p",
            description="Test without explicit reference_be",
            peaks=(TemplatePeak(name="Si0", delta_be=0.0, is_reference=True),),
            reference_be=None,
        )
        be = t.get_reference_be()
        # Should look up from BindingEnergy database
        assert be is not None
        assert abs(be - 99.0) < 1.0  # ~99 eV for Si 2p

    def test_unknown_element_fallback(self):
        """Unknown element returns None from database."""
        from toyomacro.fitting.templates import FittingTemplate, TemplatePeak

        t = FittingTemplate(
            name="_test_unknown_elem",
            element="Xx9z",
            description="Unknown element",
            peaks=(TemplatePeak(name="X0", delta_be=0.0, is_reference=True),),
            reference_be=None,
        )
        assert t.get_reference_be() is None
