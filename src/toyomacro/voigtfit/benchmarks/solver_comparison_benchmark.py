"""Same-problem comparison: voigtfit solvers vs SciPy/lmfit LM fitting.

Answers, with recorded measurements, the question the paper raises:
how much faster is batch-first Voigt fitting than the conventional
per-spectrum Levenberg-Marquardt workflow **on the same hardware, the
same model, and the same data**?

Problem definition (identical for every contender)
--------------------------------------------------
Single Voigt peak on ``n_energy`` channels.  Ground truth per spectrum:
amplitude ~ U[0.5, 2], center shift dE ~ U[-0.3, 0.3]*sigma, width
shift dsigma ~ U[-0.1, 0.1]*sigma; gamma fixed and known.  Poisson
shot noise at severity ``level`` (peak-count SNR = 1e4/level).  Every
solver receives the same initialization information: the nominal peak
configuration (center, sigma, gamma) — no solver sees the truth.

Contenders
----------
- ``projection_amp_only``: amplitude at the *nominal* (center, sigma)
  via the precomputed weight matrix.  This solves a smaller problem
  (amplitude only, no shift/width recovery) and is reported separately
  — it is NOT an apples-to-apples rival to the full fits.
- ``dict2d_parabola``: voigtfit batch solver recovering amplitude,
  dE, and dsigma (dictionary + parabola refinement).  Dictionary
  build time is recorded separately as setup cost.
- ``scipy_curve_fit``: per-spectrum Levenberg-Marquardt
  (scipy.optimize.curve_fit), free (amplitude, center, sigma),
  gamma fixed, initialized at the nominal configuration, bounded to
  the same search ranges as the dictionary.
- ``lmfit``: the same per-spectrum problem via lmfit's Model wrapper.

Accuracy is reported as the mean absolute error of each recovered
parameter against the ground truth, so throughput cannot be traded
for silent quality loss.

Usage:
    python -m toyomacro.voigtfit.benchmarks.solver_comparison_benchmark \\
        --out paper/figures/results/solver_comparison.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..dictionary_solver import build_dictionary_2d, solve_dict2d_parabola
from ..spectra_generator import add_poisson_noise, level_to_peak_snr
from ..voigt_jacobian import voigt_profile
from ..weight_cache import WeightMatrixCache

# ── problem definition (shared by all contenders) ────────────────
N_ENERGY = 151
CENTER = 0.0
SIGMA = 0.5
GAMMA = 0.2
DE_FRAC = 0.3        # |dE| <= 0.3 * sigma
DSIGMA_FRAC = 0.1    # |dsigma| <= 0.1 * sigma
NOISE_LEVEL = 1e3    # 'Moderate': peak-count SNR = 10


def _environment() -> dict:
    env = {
        'timestamp_utc': datetime.datetime.now(datetime.UTC).isoformat(
            timespec='seconds'),
        'os': f'{platform.system()} {platform.release()}',
        'machine': platform.machine(),
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'cpu_count': os.cpu_count(),
    }
    for dist in ('scipy', 'lmfit', 'mlx'):
        try:
            import importlib.metadata as md
            env[dist] = md.version(dist)
        except Exception:
            env[dist] = None
    try:
        env['cpu'] = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        env['cpu'] = platform.processor()
    try:
        env['git_commit'] = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        env['git_commit'] = None
    return env


def make_problem(n_spectra: int, seed: int = 0):
    """Generate ground truth + noisy spectra for `n_spectra` pixels."""
    rng = np.random.default_rng(seed)
    energy = np.linspace(-3, 3, N_ENERGY).astype(np.float64)
    amp = rng.uniform(0.5, 2.0, n_spectra)
    dE = rng.uniform(-DE_FRAC * SIGMA, DE_FRAC * SIGMA, n_spectra)
    dsig = rng.uniform(-DSIGMA_FRAC * SIGMA, DSIGMA_FRAC * SIGMA, n_spectra)

    Y = np.empty((n_spectra, N_ENERGY), dtype=np.float32)
    for i in range(n_spectra):
        Y[i] = voigt_profile(energy, CENTER + dE[i], SIGMA + dsig[i],
                             GAMMA) * amp[i]
    Y = add_poisson_noise(Y, NOISE_LEVEL)
    return energy, Y, {'amp': amp, 'dE': dE, 'dsigma': dsig}


def _mae(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(estimate, dtype=np.float64)
                                - truth)))


def bench_projection_amp_only(energy, Y, truth, repeats: int = 5) -> dict:
    cache = WeightMatrixCache()
    t0 = time.perf_counter()
    Wt, Phi = cache.get_or_create(
        'X', 'bench', energy, np.array([CENTER]), np.array([SIGMA]), GAMMA)
    setup_s = time.perf_counter() - t0

    Wt_np = np.asarray(Wt, dtype=np.float32)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        A = Y @ Wt_np.T  # (n, n_comp)
        times.append(time.perf_counter() - t0)
    n = Y.shape[0]
    return {
        'solver': 'projection_amp_only',
        'problem': 'amplitude only (center/width fixed at nominal) — '
                   'NOT comparable to full fits below',
        'backend': 'numpy (BLAS)',
        'n_spectra': n,
        'setup_s': round(setup_s, 4),
        'repeats': repeats,
        'timings_s': [round(t, 6) for t in times],
        'aggregation': 'median',
        'throughput_spec_per_s': n / statistics.median(times),
        'mae_amp': _mae(A[:, 0], truth['amp']),
        'mae_dE': None,
        'mae_dsigma': None,
    }


def bench_dict2d_parabola(energy, Y, truth, repeats: int = 5) -> dict:
    t0 = time.perf_counter()
    cache = build_dictionary_2d(
        energy, np.array([CENTER]), np.array([SIGMA]), GAMMA,
        dE_range=(-DE_FRAC * SIGMA, DE_FRAC * SIGMA),
        dsigma_range=(-DSIGMA_FRAC * SIGMA, DSIGMA_FRAC * SIGMA),
        N_dE=13, N_ds=7,
    )
    setup_s = time.perf_counter() - t0

    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = solve_dict2d_parabola(Y, cache)
        times.append(time.perf_counter() - t0)
    A, chi2, dE_est, dsig_est, _ = out
    n = Y.shape[0]
    return {
        'solver': 'dict2d_parabola',
        'problem': 'amplitude + dE + dsigma (gamma fixed)',
        'backend': 'mlx-gpu' if _mlx_active() else 'numpy',
        'n_spectra': n,
        'setup_s': round(setup_s, 4),
        'repeats': repeats,
        'timings_s': [round(t, 6) for t in times],
        'aggregation': 'median',
        'throughput_spec_per_s': n / statistics.median(times),
        'mae_amp': _mae(A[0] if A.ndim == 2 else A, truth['amp']),
        'mae_dE': _mae(dE_est, truth['dE']),
        'mae_dsigma': _mae(dsig_est, truth['dsigma']),
    }


def _mlx_active() -> bool:
    from .._mlx_support import mlx_usable
    return mlx_usable()


def _voigt_model(x, amplitude, center, sigma):
    return amplitude * voigt_profile(x, center, sigma, GAMMA)


def bench_scipy_curve_fit(energy, Y, truth) -> dict:
    from scipy.optimize import curve_fit

    n = Y.shape[0]
    p0 = [1.0, CENTER, SIGMA]
    bounds = ([0.0, CENTER - DE_FRAC * SIGMA, SIGMA * 0.8],
              [5.0, CENTER + DE_FRAC * SIGMA, SIGMA * 1.2])
    est = np.empty((n, 3))
    t0 = time.perf_counter()
    for i in range(n):
        try:
            popt, _ = curve_fit(_voigt_model, energy, Y[i].astype(np.float64),
                                p0=p0, bounds=bounds)
        except RuntimeError:  # no convergence — count as elapsed time anyway
            popt = p0
        est[i] = popt
    elapsed = time.perf_counter() - t0
    return {
        'solver': 'scipy_curve_fit',
        'problem': 'amplitude + dE + dsigma (gamma fixed)',
        'backend': 'scipy per-spectrum LM (single-thread loop)',
        'n_spectra': n,
        'setup_s': 0.0,
        'repeats': 1,
        'timings_s': [round(elapsed, 3)],
        'aggregation': 'single pass',
        'throughput_spec_per_s': n / elapsed,
        'mae_amp': _mae(est[:, 0], truth['amp']),
        'mae_dE': _mae(est[:, 1] - CENTER, truth['dE']),
        'mae_dsigma': _mae(est[:, 2] - SIGMA, truth['dsigma']),
    }


def bench_lmfit(energy, Y, truth) -> dict:
    from lmfit import Model

    model = Model(_voigt_model)
    params = model.make_params(amplitude=1.0, center=CENTER, sigma=SIGMA)
    params['amplitude'].set(min=0.0, max=5.0)
    params['center'].set(min=CENTER - DE_FRAC * SIGMA,
                         max=CENTER + DE_FRAC * SIGMA)
    params['sigma'].set(min=SIGMA * 0.8, max=SIGMA * 1.2)

    n = Y.shape[0]
    est = np.empty((n, 3))
    t0 = time.perf_counter()
    for i in range(n):
        res = model.fit(Y[i].astype(np.float64), params, x=energy)
        est[i] = [res.params['amplitude'].value, res.params['center'].value,
                  res.params['sigma'].value]
    elapsed = time.perf_counter() - t0
    return {
        'solver': 'lmfit',
        'problem': 'amplitude + dE + dsigma (gamma fixed)',
        'backend': 'lmfit per-spectrum LM (single-thread loop)',
        'n_spectra': n,
        'setup_s': 0.0,
        'repeats': 1,
        'timings_s': [round(elapsed, 3)],
        'aggregation': 'single pass',
        'throughput_spec_per_s': n / elapsed,
        'mae_amp': _mae(est[:, 0], truth['amp']),
        'mae_dE': _mae(est[:, 1] - CENTER, truth['dE']),
        'mae_dsigma': _mae(est[:, 2] - SIGMA, truth['dsigma']),
    }


def run(n_batch: int = 200_000, n_loop: int = 500, seed: int = 0) -> dict:
    """Run all contenders; batch solvers on n_batch, per-spectrum on n_loop.

    The per-spectrum solvers see the FIRST n_loop spectra of the same
    dataset, so accuracy numbers are computed on identical data.
    """
    print(f'generating {n_batch:,} spectra '
          f'(peak SNR {level_to_peak_snr(NOISE_LEVEL):g}) ...')
    energy, Y, truth = make_problem(n_batch, seed)
    truth_small = {k: v[:n_loop] for k, v in truth.items()}

    results = []
    print('projection (amplitude-only) ...')
    results.append(bench_projection_amp_only(energy, Y, truth))
    print('dict2d_parabola ...')
    results.append(bench_dict2d_parabola(energy, Y, truth))
    print(f'scipy curve_fit on {n_loop} spectra ...')
    results.append(bench_scipy_curve_fit(energy, Y[:n_loop], truth_small))
    print(f'lmfit on {n_loop} spectra ...')
    results.append(bench_lmfit(energy, Y[:n_loop], truth_small))

    report = {
        'description': __doc__.split('\n')[0],
        'problem': {
            'n_energy': N_ENERGY, 'center': CENTER, 'sigma': SIGMA,
            'gamma': GAMMA, 'dE_range_sigma': DE_FRAC,
            'dsigma_range_sigma': DSIGMA_FRAC,
            'noise_level': NOISE_LEVEL,
            'peak_snr': level_to_peak_snr(NOISE_LEVEL),
            'seed': seed, 'dtype': 'float32 spectra, float64 reference fits',
        },
        'results': results,
        'environment': _environment(),
    }
    for r in results:
        rate = r['throughput_spec_per_s']
        print(f"  {r['solver']:<22s} {rate:>12,.0f} spec/s  "
              f"MAE(amp)={r['mae_amp']:.4f}"
              + (f"  MAE(dE)={r['mae_dE']:.4f}" if r['mae_dE'] else ''))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-batch', type=int, default=200_000)
    p.add_argument('--n-loop', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default=None,
                   help='write the JSON report here')
    args = p.parse_args()

    report = run(args.n_batch, args.n_loop, args.seed)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + '\n')
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
