"""Tests for error statistics computation."""

from pathlib import Path

import numpy as np
import pytest

from toyomacro.voigtfit.evaluation.error_stats import compute_error_stats


class TestComputeErrorStats:
    """Test compute_error_stats with known distributions."""

    def test_zero_error(self):
        """All zeros: perfect estimation."""
        true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        est = true.copy()
        s = compute_error_stats(est, true)

        assert s.bias == pytest.approx(0.0, abs=1e-10)
        assert s.std == pytest.approx(0.0, abs=1e-10)
        assert s.rmse == pytest.approx(0.0, abs=1e-10)
        assert s.mae == pytest.approx(0.0, abs=1e-10)
        assert s.p95 == pytest.approx(0.0, abs=1e-10)
        assert len(s.errors) == 5

    def test_constant_bias(self):
        """Constant offset: bias = offset, std = 0."""
        true = np.array([1.0, 2.0, 3.0, 4.0])
        est = true + 0.5
        s = compute_error_stats(est, true)

        assert s.bias == pytest.approx(0.5, abs=1e-10)
        assert s.std == pytest.approx(0.0, abs=1e-10)
        assert s.rmse == pytest.approx(0.5, abs=1e-10)
        assert s.mae == pytest.approx(0.5, abs=1e-10)

    def test_gaussian_distribution(self):
        """Large N(μ, σ²) sample: bias ≈ μ, std ≈ σ, RMSE ≈ sqrt(μ²+σ²)."""
        rng = np.random.RandomState(42)
        mu, sigma = 0.1, 0.3
        n = 100_000
        true = rng.randn(n) * 2.0  # arbitrary true values
        errors = rng.randn(n) * sigma + mu
        est = true + errors

        s = compute_error_stats(est, true)

        assert s.bias == pytest.approx(mu, abs=0.01)
        assert s.std == pytest.approx(sigma, abs=0.01)
        expected_rmse = np.sqrt(mu ** 2 + sigma ** 2)
        assert s.rmse == pytest.approx(expected_rmse, abs=0.01)

    def test_single_pixel(self):
        """Edge case: single pixel."""
        s = compute_error_stats(np.array([1.5]), np.array([1.0]))
        assert s.bias == pytest.approx(0.5, abs=1e-10)
        assert s.std == pytest.approx(0.0, abs=1e-10)
        assert s.rmse == pytest.approx(0.5, abs=1e-10)
        assert s.mae == pytest.approx(0.5, abs=1e-10)
        assert s.p95 == pytest.approx(0.5, abs=1e-10)

    def test_large_outlier(self):
        """p95 captures tail behavior with outliers."""
        true = np.zeros(100)
        est = np.zeros(100)
        est[-1] = 100.0  # single large outlier

        s = compute_error_stats(est, true)
        assert s.rmse > s.mae  # RMSE sensitive to outlier
        assert s.p95 < 100.0   # 95th percentile below max
        assert s.p95 == pytest.approx(0.0, abs=0.01)  # 95% are zero

    def test_symmetric_errors(self):
        """Symmetric errors: bias ≈ 0."""
        true = np.zeros(1000)
        est = np.concatenate([np.ones(500) * 0.1, np.ones(500) * (-0.1)])
        s = compute_error_stats(est, true)

        assert s.bias == pytest.approx(0.0, abs=1e-10)
        assert s.std == pytest.approx(0.1, abs=1e-6)
        assert s.rmse == pytest.approx(0.1, abs=1e-6)

    def test_component_selection(self):
        """Multi-component: select component."""
        true = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        est = np.array([[1.1, 10.5], [2.1, 20.5], [3.1, 30.5]])

        s0 = compute_error_stats(est, true, component=0)
        assert s0.bias == pytest.approx(0.1, abs=1e-6)
        assert len(s0.errors) == 3

        s1 = compute_error_stats(est, true, component=1)
        assert s1.bias == pytest.approx(0.5, abs=1e-6)

    def test_errors_dtype(self):
        """Errors array is float32 for memory efficiency."""
        s = compute_error_stats(np.array([2.0]), np.array([1.0]))
        assert s.errors.dtype == np.float32


class TestBenchmarkResult:
    """Test BenchmarkResult save/load roundtrip."""

    def test_save_load_roundtrip(self, tmp_path):
        from toyomacro.voigtfit.evaluation.benchmark_result import BenchmarkResult

        rng = np.random.RandomState(0)
        n = 100
        br = BenchmarkResult(
            solver_name='test',
            noise_level='None',
            amp_est=rng.randn(n).astype(np.float32),
            dE_est=rng.randn(n).astype(np.float32),
            dsigma_est=rng.randn(n).astype(np.float32),
            amp_true=rng.randn(n).astype(np.float32),
            dE_true=rng.randn(n).astype(np.float32),
            dsigma_true=rng.randn(n).astype(np.float32),
            image_shape=(10, 10),
            throughput=1e6,
            phase2_mask=rng.rand(n) > 0.5,
        )

        path = tmp_path / 'test.npz'
        br.save_npz(path)
        loaded = BenchmarkResult.load_npz(path)

        assert loaded.solver_name == 'test'
        assert loaded.noise_level == 'None'
        assert loaded.image_shape == (10, 10)
        assert loaded.throughput == pytest.approx(1e6)
        np.testing.assert_array_almost_equal(loaded.amp_est, br.amp_est)
        np.testing.assert_array_almost_equal(loaded.dsigma_est, br.dsigma_est)
        assert loaded.phase2_mask is not None
        np.testing.assert_array_equal(loaded.phase2_mask, br.phase2_mask)

    def test_multi_save_load_roundtrip(self, tmp_path):
        from toyomacro.voigtfit.evaluation.benchmark_result import (
            BenchmarkResult,
            load_multi_results,
            save_multi_results,
        )

        rng = np.random.RandomState(1)
        n = 50

        def _make_br(sn, nl):
            return BenchmarkResult(
                solver_name=sn, noise_level=nl,
                amp_est=rng.randn(n).astype(np.float32),
                dE_est=rng.randn(n).astype(np.float32),
                dsigma_est=rng.randn(n).astype(np.float32),
                amp_true=rng.randn(n).astype(np.float32),
                dE_true=rng.randn(n).astype(np.float32),
                dsigma_true=rng.randn(n).astype(np.float32),
                image_shape=(5, 10),
                throughput=2e6,
            )

        results = {
            'None': {
                '4step': _make_br('4step', 'None'),
                'dict2d': _make_br('dict2d', 'None'),
            },
            'Strong': {
                '4step': _make_br('4step', 'Strong'),
                'dict2d': _make_br('dict2d', 'Strong'),
            },
        }

        path = tmp_path / 'multi.npz'
        save_multi_results(results, path)
        loaded = load_multi_results(path)

        assert set(loaded.keys()) == {'None', 'Strong'}
        assert set(loaded['None'].keys()) == {'4step', 'dict2d'}
        np.testing.assert_array_almost_equal(
            loaded['None']['4step'].amp_est,
            results['None']['4step'].amp_est,
        )


class TestVisualization:
    """Smoke tests for visualization functions."""

    def _make_results(self, n_pixels=100, image_shape=(10, 10)):
        """Create synthetic results for testing."""
        from toyomacro.voigtfit.evaluation.benchmark_result import BenchmarkResult

        rng = np.random.RandomState(42)

        results = {}
        for nl in ['NF', 'Strong']:
            results[nl] = {}
            noise_scale = 0.01 if nl == 'NF' else 0.2
            for sn in ['4step', 'dict2d', 'adaptive']:
                true = rng.randn(n_pixels).astype(np.float32) * 0.1
                est = true + rng.randn(n_pixels).astype(np.float32) * noise_scale
                phase2_mask = None
                if sn == 'adaptive':
                    phase2_mask = rng.rand(n_pixels) > (0.3 if nl == 'NF' else 0.8)
                results[nl][sn] = BenchmarkResult(
                    solver_name=sn,
                    noise_level=nl,
                    amp_est=rng.randn(n_pixels).astype(np.float32),
                    dE_est=rng.randn(n_pixels).astype(np.float32) * noise_scale,
                    dsigma_est=est,
                    amp_true=rng.randn(n_pixels).astype(np.float32),
                    dE_true=rng.randn(n_pixels).astype(np.float32) * 0.01,
                    dsigma_true=true,
                    image_shape=image_shape,
                    throughput=1e6,
                    phase2_mask=phase2_mask,
                )
        return results

    def test_plot_error_maps_smoke(self, tmp_path):
        """plot_error_maps runs without error and produces a file."""
        import matplotlib

        from toyomacro.voigtfit.visualization.error_maps import plot_error_maps
        matplotlib.use('Agg')

        results = self._make_results()
        out = str(tmp_path / 'test_error_maps.png')
        fig = plot_error_maps(results, param='dsigma', output_path=out)

        assert fig is not None
        assert Path(out).exists()
        assert Path(out).stat().st_size > 1000
        plt_module = __import__('matplotlib.pyplot', fromlist=['pyplot'])
        plt_module.close(fig)

    def test_clim_modes(self, tmp_path):
        """All clim_mode options produce valid figures."""
        import matplotlib

        from toyomacro.voigtfit.visualization.error_maps import plot_error_maps
        matplotlib.use('Agg')

        results = self._make_results()
        for mode in ['shared', 'per_row', 'per_panel']:
            out = str(tmp_path / f'test_{mode}.png')
            fig = plot_error_maps(
                results, param='dsigma', clim_mode=mode, output_path=out)
            assert Path(out).exists()
            plt_module = __import__('matplotlib.pyplot', fromlist=['pyplot'])
            plt_module.close(fig)

    def test_param_options(self, tmp_path):
        """All param options work."""
        import matplotlib

        from toyomacro.voigtfit.visualization.error_maps import plot_error_maps
        matplotlib.use('Agg')

        results = self._make_results()
        for param in ['dsigma', 'dE', 'amp']:
            out = str(tmp_path / f'test_{param}.png')
            fig = plot_error_maps(results, param=param, output_path=out)
            assert Path(out).exists()
            plt_module = __import__('matplotlib.pyplot', fromlist=['pyplot'])
            plt_module.close(fig)

    def test_adaptive_mask_plot(self, tmp_path):
        """Adaptive mask grid plot works."""
        import matplotlib

        from toyomacro.voigtfit.visualization.adaptive_mask import plot_adaptive_mask_grid
        matplotlib.use('Agg')

        results = self._make_results()
        adaptive_results = {
            nl: solvers['adaptive']
            for nl, solvers in results.items()
            if 'adaptive' in solvers
        }
        out = str(tmp_path / 'test_mask.png')
        fig = plot_adaptive_mask_grid(adaptive_results, output_path=out)
        assert fig is not None
        assert Path(out).exists()
        plt_module = __import__('matplotlib.pyplot', fromlist=['pyplot'])
        plt_module.close(fig)

    def test_hist_styles(self, tmp_path):
        """All histogram styles work."""
        import matplotlib

        from toyomacro.voigtfit.visualization.error_maps import plot_error_maps
        matplotlib.use('Agg')

        results = self._make_results()
        for style in ['violin', 'histogram', 'box']:
            out = str(tmp_path / f'test_hist_{style}.png')
            fig = plot_error_maps(
                results, param='dsigma', hist_style=style, output_path=out)
            assert Path(out).exists()
            plt_module = __import__('matplotlib.pyplot', fromlist=['pyplot'])
            plt_module.close(fig)
