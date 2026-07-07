"""Tests for GVRT Noise Robustness Visualization."""

import numpy as np
import pytest


class TestPSNRCellColor:
    """Test PSNR color threshold utility."""

    def test_excellent(self):
        from toyomacro.voigtfit.visualization.common import psnr_cell_color
        assert psnr_cell_color(50.0) == '#22C55E'

    def test_good(self):
        from toyomacro.voigtfit.visualization.common import psnr_cell_color
        assert psnr_cell_color(35.0) == '#EAB308'

    def test_degraded(self):
        from toyomacro.voigtfit.visualization.common import psnr_cell_color
        assert psnr_cell_color(25.0) == '#F97316'

    def test_poor(self):
        from toyomacro.voigtfit.visualization.common import psnr_cell_color
        assert psnr_cell_color(10.0) == '#EF4444'

    def test_boundary_45(self):
        from toyomacro.voigtfit.visualization.common import psnr_cell_color
        assert psnr_cell_color(45.0) == '#22C55E'

    def test_boundary_30(self):
        from toyomacro.voigtfit.visualization.common import psnr_cell_color
        assert psnr_cell_color(30.0) == '#EAB308'

    def test_boundary_20(self):
        from toyomacro.voigtfit.visualization.common import psnr_cell_color
        assert psnr_cell_color(20.0) == '#F97316'


class TestNoiseApplication:
    """Test noise injection produces correct shapes and properties."""

    def test_noise_changes_spectra(self):
        from toyomacro.voigtfit.visualization.gvrt_noise_comparison import _add_noise
        rng = np.random.default_rng(42)
        clean = np.random.rand(10, 50).astype(np.float32) * 100 + 10
        noisy = _add_noise(clean, level=1e3, rng=rng)
        assert noisy.shape == clean.shape
        assert not np.allclose(noisy, clean)

    def test_noise_preserves_shape(self):
        from toyomacro.voigtfit.visualization.gvrt_noise_comparison import _add_noise
        rng = np.random.default_rng(42)
        clean = np.ones((5, 20), dtype=np.float32) * 500
        noisy = _add_noise(clean, level=1e3, rng=rng)
        assert noisy.shape == (5, 20)
        assert noisy.dtype == np.float32 or noisy.dtype == np.float64


class TestNoisySweepData:
    """Test noisy sweep data generation."""

    def _build_dict_cache(self):
        from toyomacro.voigtfit.dictionary_solver import build_dictionary
        from toyomacro.voigtfit.param_encoder import C1S_SINGLE_PRESET
        preset = C1S_SINGLE_PRESET
        return build_dictionary(
            energy=preset.energy,
            centers=np.array([preset.element.binding_energy], dtype=np.float32),
            sigmas=np.array([preset.element.sigma], dtype=np.float32),
            gamma=preset.element.gamma,
            dE_range=(-1.2, 1.2),
            dsigma_range=(-0.2, 0.2),
        ), preset

    def test_nf_sweep_no_noise(self):
        from toyomacro.voigtfit.visualization.gvrt_noise_comparison import (
            _build_noisy_sweeps,
        )
        dc, preset = self._build_dict_cache()
        rng = np.random.default_rng(42)

        sweeps = _build_noisy_sweeps(
            preset.energy, preset.element.binding_energy,
            preset.element.sigma, preset.element.gamma,
            dc, 'NF', None, rng,
        )

        assert len(sweeps) == 3
        for ch in ['R', 'G', 'B']:
            s = sweeps[ch]
            assert s.noisy_spectra is None
            assert s.clean_spectra.shape == (7, len(preset.energy))
            assert s.fit_spectra.shape == (7, len(preset.energy))

    def test_noisy_sweep_has_noise(self):
        from toyomacro.voigtfit.visualization.gvrt_noise_comparison import (
            _build_noisy_sweeps,
        )
        dc, preset = self._build_dict_cache()
        rng = np.random.default_rng(42)

        sweeps = _build_noisy_sweeps(
            preset.energy, preset.element.binding_energy,
            preset.element.sigma, preset.element.gamma,
            dc, 'Moderate', 1e3, rng,
        )

        for ch in ['R', 'G', 'B']:
            s = sweeps[ch]
            assert s.noisy_spectra is not None
            assert s.noisy_spectra.shape == s.clean_spectra.shape
            assert not np.allclose(s.noisy_spectra, s.clean_spectra)


class TestNoisyROIData:
    """Test noisy ROI data generation."""

    def test_nf_roi_no_noise(self):
        from toyomacro.voigtfit.dictionary_solver import build_dictionary
        from toyomacro.voigtfit.param_encoder import (
            C1S_SINGLE_PRESET,
            SinglePeakEncoder,
        )
        from toyomacro.voigtfit.visualization.gvrt_noise_comparison import (
            _build_noisy_roi,
        )

        preset = C1S_SINGLE_PRESET
        image = np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)
        encoder = SinglePeakEncoder(preset)
        gt_amp, gt_dE, gt_fwhm = encoder.encode(image)
        gt_dsigma = encoder.delta_sigma(gt_fwhm)

        dc = build_dictionary(
            energy=preset.energy,
            centers=np.array([preset.element.binding_energy], dtype=np.float32),
            sigmas=np.array([preset.element.sigma], dtype=np.float32),
            gamma=preset.element.gamma,
            dE_range=(-1.2, 1.2),
            dsigma_range=(-0.2, 0.2),
        )

        flat_indices = np.arange(20)
        rng = np.random.default_rng(42)

        roi = _build_noisy_roi(
            image, preset.energy, preset.element.binding_energy,
            preset.element.sigma, preset.element.gamma,
            gt_amp, gt_dE, gt_dsigma, dc,
            'sky', '#1E88E5', (0, 0, 10, 10), flat_indices,
            'NF', None, rng,
        )

        assert roi.noisy_spectra is None
        assert roi.mean_noisy is None
        assert roi.clean_spectra.shape[0] == 20
        assert roi.mean_clean.shape == (len(preset.energy),)
        assert roi.mean_fit.shape == (len(preset.energy),)


class TestPlotSmoke:
    """Smoke tests for noise comparison figure rendering."""

    def _make_synthetic_data(self):
        """Create minimal NoiseComparisonData without running the full pipeline."""
        from toyomacro.voigtfit.param_encoder import C1S_SINGLE_PRESET
        from toyomacro.voigtfit.visualization.gvrt_noise_comparison import (
            NoiseComparisonData,
            NoiseLevelResult,
            NoisyROIData,
            NoisySweepData,
        )
        from toyomacro.voigtfit.visualization.gvrt_tracking import _make_gradient
        from toyomacro.voigtfit.voigt_jacobian import voigt_profile

        preset = C1S_SINGLE_PRESET
        energy = preset.energy
        n_E = len(energy)
        image = np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)
        nom = voigt_profile(
            energy, preset.element.binding_energy,
            preset.element.sigma, preset.element.gamma,
        ).astype(np.float32)
        spec7 = np.tile(500 * nom, (7, 1)).astype(np.float32)

        noise_results = []
        for label, level, snr, psnr_scale in [
            ('NF', None, '\u221e', 1.0),
            ('Moderate', 1e3, '~30', 0.8),
            ('Strong', 1e4, '~3', 0.5),
        ]:
            sweeps = {}
            for ch in ['R', 'G', 'B']:
                noisy = spec7 + np.random.randn(*spec7.shape).astype(np.float32) * 10 if level else None
                sweeps[ch] = NoisySweepData(
                    channel=ch, noise_label=label,
                    clean_spectra=spec7,
                    noisy_spectra=noisy,
                    fit_spectra=spec7 * 0.99,
                    nominal_spectrum=(500 * nom).astype(np.float32),
                    colors=_make_gradient(ch, 7),
                )

            n_px = 15
            roi_spec = np.tile(300 * nom, (n_px, 1)).astype(np.float32)
            noisy_roi = roi_spec + np.random.randn(*roi_spec.shape).astype(np.float32) * 10 if level else None
            roi = NoisyROIData(
                noise_label=label, name='sky', n_pixels=n_px,
                clean_spectra=roi_spec,
                noisy_spectra=noisy_roi,
                fit_spectra=roi_spec * 0.99,
                pixel_colors=np.random.rand(n_px, 3).astype(np.float32),
                mean_clean=roi_spec.mean(axis=0),
                mean_noisy=noisy_roi.mean(axis=0) if noisy_roi is not None else None,
                mean_fit=(roi_spec * 0.99).mean(axis=0),
                nominal_spectrum=(300 * nom).astype(np.float32),
            )

            noise_results.append(NoiseLevelResult(
                label=label, level=level, snr_label=snr,
                sweeps=sweeps, roi=roi,
                amp_psnr=55.0 * psnr_scale,
                dE_psnr=58.0 * psnr_scale,
                dsigma_psnr=47.0 * psnr_scale,
            ))

        return NoiseComparisonData(
            preset=preset, image=image, energy=energy,
            noise_results=noise_results,
            solver_name='Dict2D+Parabola', throughput=5.0,
        )

    def test_plot_renders_png(self, tmp_path):
        pytest.importorskip('matplotlib')
        from toyomacro.voigtfit.visualization.gvrt_noise_comparison import (
            plot_noise_comparison,
        )
        data = self._make_synthetic_data()
        out = str(tmp_path / 'test_noise.png')
        fig = plot_noise_comparison(data, figsize=(14, 12),
                                    output_path=out, dpi=72)
        assert fig is not None
        assert (tmp_path / 'test_noise.png').exists()
        assert (tmp_path / 'test_noise.png').stat().st_size > 1000

    def test_noise_results_ordering(self):
        data = self._make_synthetic_data()
        labels = [nr.label for nr in data.noise_results]
        assert labels == ['NF', 'Moderate', 'Strong']
        # PSNR should decrease with noise
        psnrs = [nr.dsigma_psnr for nr in data.noise_results]
        assert psnrs[0] > psnrs[1] > psnrs[2]
