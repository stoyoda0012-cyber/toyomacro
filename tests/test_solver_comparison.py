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
    rec = bench_dict2d_parabola(energy, Y, truth, repeats=1)
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
