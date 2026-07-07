"""Tests for the interactive GVRT service (gvrt_service.py)."""

import numpy as np
import pytest

from toyomacro.voigtfit.gvrt_service import (
    GVRT_NOISE_LEVELS,
    GVRT_SOLVERS,
    GVRTConfig,
    GVRTService,
    make_demo_image,
    prepare_image,
)
from toyomacro.voigtfit.spectra_generator import NOISE_LEVELS


class TestPrepareImage:
    def test_rgb_passthrough(self):
        img = make_demo_image(64)
        out = prepare_image(img)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, img)

    def test_grayscale_to_rgb(self):
        gray = np.full((32, 32), 128, dtype=np.uint8)
        out = prepare_image(gray)
        assert out.shape == (32, 32, 3)

    def test_rgba_drops_alpha(self):
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        out = prepare_image(rgba)
        assert out.shape == (16, 16, 3)

    def test_downscale(self):
        big = np.zeros((200, 200, 3), dtype=np.uint8)
        out = prepare_image(big, max_pixels=10_000)
        assert out.shape[0] * out.shape[1] <= 10_000
        assert out.flags['C_CONTIGUOUS']

    def test_float_input_clipped(self):
        img = np.full((8, 8, 3), 300.0)
        out = prepare_image(img)
        assert out.dtype == np.uint8
        assert out.max() == 255


class TestDemoImage:
    def test_shape_dtype(self):
        img = make_demo_image(128)
        assert img.shape == (128, 128, 3)
        assert img.dtype == np.uint8

    def test_nonzero_amplitude(self):
        # R channel encodes amplitude; near-zero amp makes fitting degenerate
        img = make_demo_image(64)
        assert img[:, :, 0].min() > 30


class TestRoundtrip:
    @pytest.fixture(scope='class')
    def service(self):
        return GVRTService()

    @pytest.fixture(scope='class')
    def image(self):
        return make_demo_image(64)  # 4,096 spectra — fast

    def test_progress_frames(self, service, image):
        config = GVRTConfig(solver='taylor', noise_level='None', chunk_pixels=1024)
        frames = list(service.iter_roundtrip(image, config))
        assert len(frames) == 4
        assert [f.done for f in frames] == [1024, 2048, 3072, 4096]
        assert all(f.total == 4096 for f in frames)
        # Only the last frame carries the result
        assert all(f.result is None for f in frames[:-1])
        assert frames[-1].result is not None

    def test_partial_fill(self, service, image):
        config = GVRTConfig(solver='taylor', noise_level='None', chunk_pixels=1024)
        gen = service.iter_roundtrip(image, config)
        first = next(gen)
        flat = first.image.reshape(-1, 3)
        # Fitted region differs from canvas, unfitted region is still gray(40)
        assert not np.all(flat[:1024] == 40)
        np.testing.assert_array_equal(flat[1024:], 40)
        gen.close()

    def test_noisefree_taylor_psnr(self, service, image):
        config = GVRTConfig(solver='taylor', noise_level='None', chunk_pixels=4096)
        result = service.run(image, config)
        assert result.psnr.total > 25.0
        assert result.amp_corr > 0.99
        assert result.reconstructed.shape == image.shape

    def test_noisefree_dict2d_parabola_psnr(self, service, image):
        config = GVRTConfig(
            solver='dict2d_parabola', noise_level='None', chunk_pixels=4096,
        )
        result = service.run(image, config)
        assert result.psnr.total > 45.0

    def test_dict_cache_reused(self, service, image):
        # Second dict2d run must not emit the 'Building dictionary' frame
        config = GVRTConfig(
            solver='dict2d_parabola', noise_level='None', chunk_pixels=4096,
        )
        messages = [p.message for p in service.iter_roundtrip(image, config)]
        assert 'Building dictionary...' not in messages

    def test_moderate_noise_finite(self, service, image):
        config = GVRTConfig(solver='taylor', noise_level='Moderate', chunk_pixels=4096)
        result = service.run(image, config)
        assert np.isfinite(result.psnr.total)
        assert result.psnr.total > 10.0
        assert result.noise_value == NOISE_LEVELS['Moderate']

    def test_unknown_solver_raises(self, service, image):
        with pytest.raises(ValueError, match='Unknown solver'):
            next(service.iter_roundtrip(image, GVRTConfig(solver='nope')))

    def test_exposed_noise_levels_valid(self):
        for name in GVRT_NOISE_LEVELS:
            assert name in NOISE_LEVELS

    def test_exposed_solvers(self):
        assert set(GVRT_SOLVERS) == {'taylor', 'taylor6', 'dict2d_parabola'}

    def test_inspect_pixel(self, service, image):
        config = GVRTConfig(solver='taylor', noise_level='Moderate', chunk_pixels=4096)
        service.run(image, config)
        assert service.has_last_run
        assert service.last_run_shape == (64, 64)

        ins = service.inspect_pixel(10, 20)
        assert (ins.x, ins.y) == (10, 20)
        n_e = len(ins.energy)
        assert len(ins.clean) == len(ins.noisy) == len(ins.fit) == n_e
        # Fit must resemble the noisy spectrum it was fit to
        corr = np.corrcoef(ins.noisy, ins.fit)[0, 1]
        assert corr > 0.9
        # GT params come straight from the encoder ranges
        assert 0 <= ins.gt_amp <= 1000.0
        assert -1.0 <= ins.gt_dE <= 1.0
        assert 0.7 <= ins.gt_fwhm <= 1.5
        assert np.isfinite(ins.batch_amp)
        assert ins.residual_rms >= 0

    def test_inspect_pixel_clamps_coords(self, service, image):
        service.run(image, GVRTConfig(solver='taylor', noise_level='None'))
        ins = service.inspect_pixel(9999, -5)
        assert (ins.x, ins.y) == (63, 0)

    def test_inspect_before_run_raises(self):
        svc = GVRTService()
        with pytest.raises(RuntimeError, match='No completed run'):
            svc.inspect_pixel(0, 0)

    def test_multipeak_roundtrip(self, service, image):
        config = GVRTConfig(n_peaks=2, noise_level='None', chunk_pixels=4096)
        result = service.run(image, config)
        assert result.n_peaks == 2
        assert result.solver == 'multipeak_ap'
        assert result.reconstructed.shape == image.shape
        # Hilbert-packed RGB PSNR is harsh; param-space corr is the measure
        assert result.amp_corr > 0.98
        assert result.shift_corr > 0.7
        assert np.isfinite(result.psnr.total)

    def test_multipeak_inspect_pixel(self, service, image):
        service.run(image, GVRTConfig(n_peaks=2, noise_level='Weak'))
        ins = service.inspect_pixel(12, 34)
        assert len(ins.peaks) == 2
        assert ins.peaks[0]['center'] == pytest.approx(284.4)
        assert ins.peaks[1]['center'] == pytest.approx(286.4)
        assert len(ins.energy) == len(ins.fit) == len(ins.noisy)
        # Scalar mirrors == peak 0
        assert ins.gt_amp == ins.peaks[0]['gt_amp']
        corr = np.corrcoef(ins.noisy, ins.fit)[0, 1]
        assert corr > 0.8

    def test_invalid_n_peaks_raises(self, service, image):
        with pytest.raises(ValueError, match='n_peaks'):
            next(service.iter_roundtrip(image, GVRTConfig(n_peaks=3)))

    def test_exact_gen_mlx_matches_scipy(self, image):
        mlx = pytest.importorskip('mlx.core')  # noqa: F841
        from toyomacro.voigtfit.benchmarks.param_roundtrip_benchmark import (
            ParamRoundtripBenchmark,
        )
        bench = ParamRoundtripBenchmark(image=image, solver='taylor6', verbose=False)
        gen_mlx = GVRTService._make_exact_gen_mlx(bench)
        sl = slice(0, 2048)
        y_mlx = gen_mlx(bench.gt_amp[sl], bench.gt_dE[sl], bench.gt_dsigma[sl])
        y_sp = bench._generate_spectra_exact(
            bench.gt_amp[sl], bench.gt_dE[sl], bench.gt_dsigma[sl],
        )
        rel = np.abs(y_mlx - y_sp) / np.abs(y_sp).max()
        # Faddeeva table interpolation accuracy — far below any noise level
        assert rel.max() < 1e-3
