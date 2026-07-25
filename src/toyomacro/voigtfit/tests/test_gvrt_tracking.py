"""Tests for GVRT Parameter Tracking Visualization v4."""

import numpy as np
import pytest


class TestSweepData:
    """Test synthetic parameter sweep generation."""

    def test_r_sweep_height_varies(self):
        """R sweep: peak heights should differ, positions same."""
        from toyomacro.voigtfit.dictionary_solver import build_dictionary
        from toyomacro.voigtfit.param_encoder import C1S_SINGLE_PRESET
        from toyomacro.voigtfit.visualization.gvrt_tracking import (
            _build_sweep_data,
        )

        preset = C1S_SINGLE_PRESET
        energy = preset.energy
        center = preset.element.binding_energy
        sigma = preset.element.sigma
        gamma = preset.element.gamma

        dc = build_dictionary(
            energy=energy,
            centers=np.array([center], dtype=np.float32),
            sigmas=np.array([sigma], dtype=np.float32),
            gamma=gamma,
            dE_range=(-1.2, 1.2),
            dsigma_range=(-0.2, 0.2),
        )
        sweep = _build_sweep_data(energy, center, sigma, gamma, dc, 'R')

        assert sweep.gt_spectra.shape == (7, len(energy))
        # Peak heights should increase
        peaks = sweep.gt_spectra.max(axis=1)
        assert peaks[-1] > peaks[0]
        # Peak positions should be the same
        pos = energy[np.argmax(sweep.gt_spectra, axis=1)]
        assert np.all(pos == pos[0])

    def test_g_sweep_position_varies(self):
        """G sweep: peak positions shift, heights same."""
        from toyomacro.voigtfit.dictionary_solver import build_dictionary
        from toyomacro.voigtfit.param_encoder import C1S_SINGLE_PRESET
        from toyomacro.voigtfit.visualization.gvrt_tracking import (
            _build_sweep_data,
        )

        preset = C1S_SINGLE_PRESET
        energy = preset.energy
        center = preset.element.binding_energy
        sigma = preset.element.sigma
        gamma = preset.element.gamma

        dc = build_dictionary(
            energy=energy,
            centers=np.array([center], dtype=np.float32),
            sigmas=np.array([sigma], dtype=np.float32),
            gamma=gamma,
            dE_range=(-1.2, 1.2),
            dsigma_range=(-0.2, 0.2),
        )
        sweep = _build_sweep_data(energy, center, sigma, gamma, dc, 'G')

        # Peak positions should differ
        pos = energy[np.argmax(sweep.gt_spectra, axis=1)]
        assert pos[-1] > pos[0]  # positive shift → higher energy

    def test_sweep_has_7_steps(self):
        from toyomacro.voigtfit.dictionary_solver import build_dictionary
        from toyomacro.voigtfit.param_encoder import C1S_SINGLE_PRESET
        from toyomacro.voigtfit.visualization.gvrt_tracking import (
            _build_sweep_data,
        )

        preset = C1S_SINGLE_PRESET
        dc = build_dictionary(
            energy=preset.energy,
            centers=np.array([preset.element.binding_energy], dtype=np.float32),
            sigmas=np.array([preset.element.sigma], dtype=np.float32),
            gamma=preset.element.gamma,
            dE_range=(-1.2, 1.2),
            dsigma_range=(-0.2, 0.2),
        )
        for ch in ['R', 'G', 'B']:
            sweep = _build_sweep_data(
                preset.energy, preset.element.binding_energy,
                preset.element.sigma, preset.element.gamma, dc, ch,
            )
            assert len(sweep.sweep_values) == 7
            assert len(sweep.colors) == 7
            assert sweep.gt_spectra.shape[0] == 7
            assert sweep.fit_spectra.shape[0] == 7

    def test_make_gradient_colors(self):
        from toyomacro.voigtfit.visualization.gvrt_tracking import _make_gradient
        colors = _make_gradient('R', 5)
        assert len(colors) == 5
        assert all(len(c) == 3 for c in colors)
        # First should be darker than last
        assert colors[0][0] < colors[-1][0]


class TestROISelection:
    """Test ROI selection."""

    def test_returns_3_rois(self):
        from toyomacro.voigtfit.visualization.gvrt_tracking import select_rois
        image = np.random.randint(0, 256, (100, 160, 3), dtype=np.uint8)
        rois = select_rois(image, preset_name='demo')
        assert len(rois) == 3

    def test_rois_within_bounds(self):
        from toyomacro.voigtfit.visualization.gvrt_tracking import select_rois
        H, W = 100, 160
        image = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        rois = select_rois(image, preset_name='demo')
        for name, color, (r0, c0, r1, c1), flat in rois:
            assert 0 <= r0 < r1 <= H
            assert 0 <= c0 < c1 <= W
            assert np.all(flat >= 0)
            assert np.all(flat < H * W)

    def test_roi_sample_count(self):
        from toyomacro.voigtfit.visualization.gvrt_tracking import select_rois
        image = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        rois = select_rois(image, preset_name='demo', n_sample=50)
        for _, _, _, flat in rois:
            assert len(flat) <= 50

    def test_auto_fallback(self):
        from toyomacro.voigtfit.visualization.gvrt_tracking import select_rois
        image = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
        rois = select_rois(image, preset_name='auto')
        assert len(rois) == 3

    def test_deterministic(self):
        from toyomacro.voigtfit.visualization.gvrt_tracking import select_rois
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        r1 = select_rois(image, preset_name='demo', n_sample=30)
        r2 = select_rois(image, preset_name='demo', n_sample=30)
        for (_, _, _, f1), (_, _, _, f2) in zip(r1, r2):
            np.testing.assert_array_equal(f1, f2)


class TestSpectraGeneration:
    """Test exact Voigt spectrum generation."""

    def test_nominal_spectrum_positive(self):
        from toyomacro.voigtfit.param_encoder import C1S_SINGLE_PRESET
        from toyomacro.voigtfit.voigt_jacobian import voigt_profile
        preset = C1S_SINGLE_PRESET
        energy = preset.energy
        prof = voigt_profile(
            energy, preset.element.binding_energy,
            preset.element.sigma, preset.element.gamma,
        )
        assert np.max(prof) > 0
        assert np.all(np.isfinite(prof))

    def test_generate_exact_voigt_shape(self):
        from toyomacro.voigtfit.param_encoder import C1S_SINGLE_PRESET
        from toyomacro.voigtfit.visualization.gvrt_tracking import (
            _generate_exact_voigt_spectra,
        )
        preset = C1S_SINGLE_PRESET
        energy = preset.energy
        N = 10
        spectra = _generate_exact_voigt_spectra(
            energy,
            center=preset.element.binding_energy,
            sigma_nom=preset.element.sigma,
            gamma=preset.element.gamma,
            amp=np.ones(N, dtype=np.float32) * 0.5,
            dE=np.zeros(N, dtype=np.float32),
            dsigma=np.zeros(N, dtype=np.float32),
        )
        assert spectra.shape == (N, len(energy))
        assert spectra.dtype == np.float32
        assert np.all(spectra > 0)


class TestPlotSmokeV4:
    """Smoke tests for v4 figure rendering."""

    def _make_synthetic_data(self):
        """Create minimal TrackingDataV4 for testing."""
        from toyomacro.voigtfit.param_encoder import C1S_SINGLE_PRESET
        from toyomacro.voigtfit.visualization.gvrt_tracking import (
            ROIData,
            SweepData,
            TrackingDataV4,
            _make_gradient,
        )
        from toyomacro.voigtfit.voigt_jacobian import voigt_profile

        preset = C1S_SINGLE_PRESET
        energy = preset.energy
        n_E = len(energy)
        image = np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)
        nom = voigt_profile(
            energy, preset.element.binding_energy,
            preset.element.sigma, preset.element.gamma,
        )

        sweeps = []
        for ch in ['R', 'G', 'B']:
            n = 7
            spec = np.tile(500 * nom, (n, 1)).astype(np.float32)
            sweeps.append(SweepData(
                channel=ch, param_name='test', label=f'{ch} test',
                sweep_values=np.linspace(0, 1, n).astype(np.float32),
                gt_spectra=spec, fit_spectra=spec * 0.99,
                nominal_spectrum=(500 * nom).astype(np.float32),
                colors=_make_gradient(ch, n),
            ))

        rois = []
        for name, color in [('sky', '#1E88E5'), ('trees', '#43A047'),
                             ('roof', '#E53935')]:
            n_px = 20
            spec = np.tile(300 * nom, (n_px, 1)).astype(np.float32)
            rois.append(ROIData(
                name=name, color=color, rect=(2, 2, 12, 20),
                n_pixels=n_px,
                pixel_colors=np.random.rand(n_px, 3).astype(np.float32),
                gt_spectra=spec,
                mean_gt_spectrum=spec.mean(axis=0),
                mean_fit_spectrum=spec.mean(axis=0) * 0.99,
                nominal_spectrum=(300 * nom).astype(np.float32),
            ))

        return TrackingDataV4(
            preset=preset, image=image, energy=energy,
            nominal_center=284.4, nominal_sigma=0.4247,
            nominal_gamma=0.125,
            sweeps=sweeps, rois=rois,
            amp_psnr=50.0, dE_psnr=55.0, dsigma_psnr=45.0,
            solver_name='Dict2D+Parabola', throughput=5.0,
        )

    def test_plot_renders_png(self, tmp_path):
        pytest.importorskip('matplotlib')
        from toyomacro.voigtfit.visualization.gvrt_tracking import (
            plot_gvrt_tracking,
        )
        data = self._make_synthetic_data()
        out = str(tmp_path / 'test_v4.png')
        fig = plot_gvrt_tracking(data, figsize=(12, 9),
                                 output_path=out, dpi=72)
        assert fig is not None
        assert (tmp_path / 'test_v4.png').exists()
        assert (tmp_path / 'test_v4.png').stat().st_size > 1000

    def test_plot_v4_alias(self, tmp_path):
        """Verify v4 alias works."""
        pytest.importorskip('matplotlib')
        from toyomacro.voigtfit.visualization.gvrt_tracking import (
            plot_gvrt_tracking_v4,
        )
        data = self._make_synthetic_data()
        out = str(tmp_path / 'test_v4_alias.png')
        fig = plot_gvrt_tracking_v4(data, figsize=(12, 9),
                                    output_path=out, dpi=72)
        assert fig is not None
        assert (tmp_path / 'test_v4_alias.png').exists()
