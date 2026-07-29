"""Smoke tests for the solver comparison benchmark.

Keeps the same-problem scipy/lmfit comparison runnable so the paper's
State-of-the-field numbers can always be regenerated.
"""

import numpy as np
import pytest

from toyomacro.voigtfit.benchmarks.solver_comparison_benchmark import (
    bench_dict2d_parabola,
    bench_scipy_curve_fit,
    make_problem,
    run,
)


@pytest.fixture(scope='module')
def small_problem():
    return make_problem(n_spectra=2000, seed=1)


def test_problem_generation(small_problem):
    energy, Y, truth = small_problem
    assert Y.shape == (2000, 151)
    assert np.all(np.isfinite(Y))
    assert truth['amp'].min() >= 0.5 and truth['amp'].max() <= 2.0


def test_dict2d_recovers_parameters(small_problem):
    energy, Y, truth = small_problem
    rec = bench_dict2d_parabola(energy, Y, truth, n_common=100, repeats=1)
    # Loose sanity bounds — the committed JSON documents actual values.
    assert rec['mae_amp'] < 0.1
    assert rec['mae_dE'] < 0.05
    assert rec['throughput_spec_per_s'] > 1e4


def test_scipy_same_problem_same_accuracy_class(small_problem):
    energy, Y, truth = small_problem
    n = 25
    truth_small = {k: v[:n] for k, v in truth.items()}
    rec = bench_scipy_curve_fit(energy, Y[:n], truth_small)
    assert rec['mae_amp'] < 0.1
    assert rec['mae_dE'] < 0.05


def test_full_run_report_structure():
    report = run(n_batch=2000, n_loop=20, seed=2)
    solvers = [r['solver'] for r in report['results']]
    assert solvers == ['projection_amp_only', 'dict2d_parabola',
                       'scipy_curve_fit', 'lmfit']
    for r in report['results']:
        assert r['throughput_spec_per_s'] > 0
        assert 'timings_s' in r and len(r['timings_s']) >= 1
    env = report['environment']
    assert env['numpy'] and env['python']


class TestBackendLabel:
    """The provenance ``backend`` field must name what actually ran.

    It previously hardcoded ``mlx (Apple Silicon GPU)`` for any
    MLX-active run, which would have written a false vendor into a
    committed record the first time the benchmark was run on MLX's CUDA
    backend. Both labels are pinned here because they end up in a file
    that outlives the run.
    """

    def test_numpy_label_when_mlx_inactive(self, monkeypatch):
        from toyomacro.voigtfit.benchmarks import (
            solver_comparison_benchmark as scb,
        )
        monkeypatch.setattr(scb, '_mlx_active', lambda: False)
        assert scb._backend_label() == 'numpy (CPU)'

    def test_gpu_label_reports_device_type_not_vendor(self, monkeypatch):
        from toyomacro.voigtfit.benchmarks import (
            solver_comparison_benchmark as scb,
        )
        monkeypatch.setattr(scb, '_mlx_active', lambda: True)
        label = scb._backend_label()
        # MLX exposes a device *type* only, so Metal and CUDA are both
        # 'gpu' here; the env block is what tells them apart.
        assert label in ('mlx (gpu)', 'mlx (cpu)', 'mlx (GPU)')
        assert 'Apple' not in label

    def test_label_reaches_the_record(self, small_problem):
        from toyomacro.voigtfit.benchmarks import (
            solver_comparison_benchmark as scb,
        )
        energy, Y, truth = small_problem
        rec = scb.bench_dict2d_parabola(energy, Y, truth, n_common=100,
                                        repeats=1)
        assert rec['backend'] == scb._backend_label()


class TestReproducibility:
    """The committed JSON must be regenerable from (n, seed) alone."""

    def test_same_seed_same_spectra(self):
        from toyomacro.voigtfit.benchmarks.solver_comparison_benchmark import (
            input_sha256,
        )
        _, Y1, t1 = make_problem(500, seed=7)
        _, Y2, t2 = make_problem(500, seed=7)
        assert input_sha256(Y1) == input_sha256(Y2)
        assert np.array_equal(t1['amp'], t2['amp'])

    def test_different_seed_different_noise(self):
        from toyomacro.voigtfit.benchmarks.solver_comparison_benchmark import (
            input_sha256,
        )
        _, Y1, _ = make_problem(500, seed=7)
        _, Y2, _ = make_problem(500, seed=8)
        assert input_sha256(Y1) != input_sha256(Y2)

    def test_seeded_rng_controls_poisson_noise(self):
        from toyomacro.voigtfit.spectra_generator import add_poisson_noise
        data = np.full((1000, 8), 500.0, dtype=np.float32)
        a = add_poisson_noise(data, 1e3, rng=np.random.default_rng(3))
        b = add_poisson_noise(data, 1e3, rng=np.random.default_rng(3))
        c = add_poisson_noise(data, 1e3, rng=np.random.default_rng(4))
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)
