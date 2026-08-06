"""Same-problem comparison: voigtfit solvers vs SciPy/lmfit fitting.

Answers, with recorded measurements, the question the paper raises:
how much faster is batch-first Voigt fitting than the conventional
per-spectrum optimizer workflow **on the same machine, the same
model, and the same data**?

Problem definition (identical for every contender)
--------------------------------------------------
Single Voigt peak on ``n_energy`` channels.  Ground truth per spectrum:
amplitude ~ U[0.5, 2], center shift dE ~ U[-0.3, 0.3]*sigma, width
shift dsigma ~ U[-0.1, 0.1]*sigma; gamma fixed and known.  Poisson
shot noise at severity ``level`` (peak-count SNR = 1e4/level), drawn
from a seeded generator so the exact input spectra regenerate from
``seed`` (the committed JSON also records their SHA-256).  Every
solver receives the same initialization information — the nominal
peak configuration — and the same search bounds
(|dE| <= 0.3 sigma, |dsigma| <= 0.1 sigma).  No solver sees the truth.

Fairness notes
--------------
- **Accuracy is compared on a common subset**: the first ``n_loop``
  spectra, fitted by every contender.  Batch solvers additionally
  report full-batch accuracy (informational).
- **Compute resources differ by design**: the per-spectrum tools run
  a single-thread CPU loop (their normal usage); the voigtfit batch
  path uses the GPU when usable.  The comparison is workflow vs
  workflow on one machine, not core vs core — the JSON records both
  backends explicitly.
- The two per-spectrum solvers use different optimizers, recorded
  separately in the JSON: bounded ``scipy.optimize.curve_fit`` runs
  **trust-region reflective (TRF)** (LM cannot handle bounds), while
  lmfit's default ``leastsq`` is MINPACK Levenberg-Marquardt with a
  parameter transform that maps the bounds onto an unconstrained
  problem.  They are not the same algorithm.

Contenders
----------
- ``projection_amp_only``: amplitude at the *nominal* (center, sigma)
  via the precomputed weight matrix.  Solves a smaller problem
  (amplitude only) and is reported separately — NOT an
  apples-to-apples rival to the full fits.
- ``dict2d_parabola``: voigtfit batch solver recovering amplitude,
  dE, and dsigma.  Dictionary build time recorded as setup cost.
- ``scipy_curve_fit``: per-spectrum bounded TRF.
- ``lmfit``: the same per-spectrum problem via lmfit's Model wrapper.

Usage:
    python -m toyomacro.voigtfit.benchmarks.solver_comparison_benchmark \\
        --out paper/figures/results/solver_comparison.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
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
DE_FRAC = 0.3        # |dE| <= 0.3 * sigma  (identical for all solvers)
DSIGMA_FRAC = 0.1    # |dsigma| <= 0.1 * sigma  (identical for all solvers)
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
        # Record a machine-independent invocation (argv[0] may be an
        # absolute personal path when run via -m).
        'command': ('python -m toyomacro.voigtfit.benchmarks.'
                    'solver_comparison_benchmark '
                    + ' '.join(sys.argv[1:])).strip(),
    }
    for dist in ('scipy', 'lmfit', 'mlx'):
        try:
            import importlib.metadata as md
            env[dist] = md.version(dist)
        except Exception:
            env[dist] = None
    # An installed MLX does not imply an accelerated run: the GPU path is
    # off when the default device fails the probe, or when the user
    # forces the NumPy backend.  Record both so a reader can tell which
    # happened.
    from .._mlx_support import mlx_usable
    env['mlx_usable'] = mlx_usable()
    env['TOYOMACRO_DISABLE_MLX'] = os.environ.get(
        'TOYOMACRO_DISABLE_MLX') or None
    try:
        env['cpu'] = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        env['cpu'] = platform.processor()
    repo = Path(__file__).parent
    try:
        env['git_commit'] = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=repo).stdout.strip() or None
        # Tracked-file cleanliness: measurements are reproducible from
        # git_commit only when this is empty (untracked files cannot
        # affect the installed package).
        dirty = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=no'],
            capture_output=True, text=True, timeout=5,
            cwd=repo).stdout.strip()
        env['git_tracked_dirty'] = bool(dirty)
        if dirty:
            env['git_dirty_files'] = dirty.splitlines()[:20]
    except Exception:
        env['git_commit'] = None
        env['git_tracked_dirty'] = None
    return env


def make_problem(n_spectra: int, seed: int = 0):
    """Generate ground truth + noisy spectra for `n_spectra` pixels.

    Fully deterministic: one seeded Generator drives both the
    parameter draws and the Poisson noise, so the same (n_spectra,
    seed) regenerates bit-identical spectra.
    """
    rng = np.random.default_rng(seed)
    energy = np.linspace(-3, 3, N_ENERGY).astype(np.float64)
    amp = rng.uniform(0.5, 2.0, n_spectra)
    dE = rng.uniform(-DE_FRAC * SIGMA, DE_FRAC * SIGMA, n_spectra)
    dsig = rng.uniform(-DSIGMA_FRAC * SIGMA, DSIGMA_FRAC * SIGMA, n_spectra)

    Y = np.empty((n_spectra, N_ENERGY), dtype=np.float32)
    for i in range(n_spectra):
        Y[i] = voigt_profile(energy, CENTER + dE[i], SIGMA + dsig[i],
                             GAMMA) * amp[i]
    Y = add_poisson_noise(Y, NOISE_LEVEL, rng=rng)
    return energy, Y, {'amp': amp, 'dE': dE, 'dsigma': dsig}


def input_sha256(Y: np.ndarray) -> str:
    """Hash of the exact noisy input spectra (order-sensitive)."""
    return hashlib.sha256(np.ascontiguousarray(Y).tobytes()).hexdigest()


def _mae(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(estimate, dtype=np.float64)
                                - truth)))


def _accuracy(est_amp, est_dE, est_dsig, truth, n_common: int) -> dict:
    """MAE on the common subset (first n_common spectra)."""
    out = {
        'mae_amp': _mae(est_amp[:n_common], truth['amp'][:n_common]),
    }
    out['mae_dE'] = (None if est_dE is None else
                     _mae(est_dE[:n_common], truth['dE'][:n_common]))
    out['mae_dsigma'] = (None if est_dsig is None else
                         _mae(est_dsig[:n_common], truth['dsigma'][:n_common]))
    return out


def bench_projection_amp_only(energy, Y, truth, n_common: int,
                              repeats: int = 5) -> dict:
    cache = WeightMatrixCache()
    t0 = time.perf_counter()
    Wt, Phi = cache.get_or_create(
        'X', 'bench', energy, np.array([CENTER]), np.array([SIGMA]), GAMMA)
    setup_s = time.perf_counter() - t0

    Wt_np = np.asarray(Wt, dtype=np.float32)
    times, A = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        A = Y @ Wt_np.T  # (n, n_comp)
        times.append(time.perf_counter() - t0)
    n = Y.shape[0]
    return {
        'solver': 'projection_amp_only',
        'problem': 'amplitude only (center/width fixed at nominal) — '
                   'NOT comparable to full fits below',
        'backend': 'numpy (multi-thread BLAS), CPU',
        'n_spectra': n,
        'setup_s': round(setup_s, 4),
        'repeats': repeats,
        'timings_s': [round(t, 6) for t in times],
        'aggregation': 'median',
        'throughput_spec_per_s': n / statistics.median(times),
        **_accuracy(A[:, 0], None, None, truth, n_common),
    }


def _mlx_active() -> bool:
    from .._mlx_support import mlx_usable
    return mlx_usable()


def _backend_label() -> str:
    """Name the backend that actually ran: ``mlx (<device type>)`` or
    ``numpy (CPU)``.

    Records what MLX reports rather than assuming Apple Silicon, because
    MLX has more than one GPU backend (see ``docs/CUDA_BACKEND_POC.md``)
    and a provenance record that hardcodes the vendor is wrong the first
    time it is generated somewhere else.

    Note this does **not** identify the vendor: MLX reports only a device
    type, so an accelerated run is labelled ``mlx (gpu)`` on Metal and on
    CUDA alike. The surrounding ``env`` block is what distinguishes them
    — it carries ``os``, ``machine``, ``cpu`` and the ``mlx`` version.
    """
    if not _mlx_active():
        return 'numpy (CPU)'
    try:
        import mlx.core as mx
        return f'mlx ({str(mx.default_device().type).split(".")[-1]})'
    except Exception:
        return 'mlx (GPU)'


def bench_dict2d_parabola(energy, Y, truth, n_common: int,
                          repeats: int = 5) -> dict:
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
    amp_est = A[0] if A.ndim == 2 else A
    n = Y.shape[0]
    return {
        'solver': 'dict2d_parabola',
        'problem': 'amplitude + dE + dsigma (gamma fixed)',
        'backend': _backend_label(),
        'n_spectra': n,
        'setup_s': round(setup_s, 4),
        'repeats': repeats,
        'timings_s': [round(t, 6) for t in times],
        'aggregation': 'median',
        'throughput_spec_per_s': n / statistics.median(times),
        **_accuracy(amp_est, dE_est, dsig_est, truth, n_common),
        'mae_amp_full_batch': _mae(amp_est, truth['amp']),
        'mae_dE_full_batch': _mae(dE_est, truth['dE']),
        'mae_dsigma_full_batch': _mae(dsig_est, truth['dsigma']),
    }


def _voigt_model(x, amplitude, center, sigma):
    return amplitude * voigt_profile(x, center, sigma, GAMMA)


# Identical search bounds for the per-spectrum solvers: the same
# |dE| <= DE_FRAC*sigma and |dsigma| <= DSIGMA_FRAC*sigma window the
# dictionary searches.
_P0 = [1.0, CENTER, SIGMA]
_BOUNDS_LO = [0.0, CENTER - DE_FRAC * SIGMA, SIGMA * (1 - DSIGMA_FRAC)]
_BOUNDS_HI = [5.0, CENTER + DE_FRAC * SIGMA, SIGMA * (1 + DSIGMA_FRAC)]


def bench_scipy_curve_fit(energy, Y, truth, repeats: int = 3) -> dict:
    from scipy.optimize import curve_fit

    n = Y.shape[0]
    est = np.empty((n, 3))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(n):
            try:
                popt, _ = curve_fit(
                    _voigt_model, energy, Y[i].astype(np.float64),
                    p0=_P0, bounds=(_BOUNDS_LO, _BOUNDS_HI))
            except RuntimeError:  # no convergence — time still counts
                popt = _P0
            est[i] = popt
        times.append(time.perf_counter() - t0)
    return {
        'solver': 'scipy_curve_fit',
        'problem': 'amplitude + dE + dsigma (gamma fixed)',
        'backend': 'scipy curve_fit, bounded trust-region reflective '
                   '(TRF), single-thread CPU per-spectrum loop',
        'n_spectra': n,
        'setup_s': 0.0,
        'repeats': repeats,
        'timings_s': [round(t, 3) for t in times],
        'aggregation': 'median',
        'throughput_spec_per_s': n / statistics.median(times),
        **_accuracy(est[:, 0], est[:, 1] - CENTER, est[:, 2] - SIGMA,
                    truth, n),
    }


def bench_lmfit(energy, Y, truth, repeats: int = 3) -> dict:
    from lmfit import Model

    model = Model(_voigt_model)
    params = model.make_params(amplitude=1.0, center=CENTER, sigma=SIGMA)
    params['amplitude'].set(min=_BOUNDS_LO[0], max=_BOUNDS_HI[0])
    params['center'].set(min=_BOUNDS_LO[1], max=_BOUNDS_HI[1])
    params['sigma'].set(min=_BOUNDS_LO[2], max=_BOUNDS_HI[2])

    n = Y.shape[0]
    est = np.empty((n, 3))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(n):
            res = model.fit(Y[i].astype(np.float64), params, x=energy)
            est[i] = [res.params['amplitude'].value,
                      res.params['center'].value,
                      res.params['sigma'].value]
        times.append(time.perf_counter() - t0)
    return {
        'solver': 'lmfit',
        'problem': 'amplitude + dE + dsigma (gamma fixed)',
        'backend': 'lmfit Model.fit (leastsq with bounds), '
                   'single-thread CPU per-spectrum loop',
        'n_spectra': n,
        'setup_s': 0.0,
        'repeats': repeats,
        'timings_s': [round(t, 3) for t in times],
        'aggregation': 'median',
        'throughput_spec_per_s': n / statistics.median(times),
        **_accuracy(est[:, 0], est[:, 1] - CENTER, est[:, 2] - SIGMA,
                    truth, n),
    }


def run(n_batch: int = 200_000, n_loop: int = 500, seed: int = 0) -> dict:
    """Run all contenders.

    Batch solvers fit all ``n_batch`` spectra; the per-spectrum tools
    fit the first ``n_loop`` of the same dataset.  All accuracy
    numbers in ``mae_*`` refer to that common first-``n_loop`` subset,
    computed against the same ground truth.
    """
    print(f'generating {n_batch:,} spectra '
          f'(peak SNR {level_to_peak_snr(NOISE_LEVEL):g}, seed={seed}) ...')
    energy, Y, truth = make_problem(n_batch, seed)
    truth_small = {k: v[:n_loop] for k, v in truth.items()}

    results = []
    print('projection (amplitude-only) ...')
    results.append(bench_projection_amp_only(energy, Y, truth, n_loop))
    print('dict2d_parabola ...')
    results.append(bench_dict2d_parabola(energy, Y, truth, n_loop))
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
            'bounds': {'lo': _BOUNDS_LO, 'hi': _BOUNDS_HI,
                       'note': 'identical for all full-fit solvers'},
            'noise_level': NOISE_LEVEL,
            'peak_snr': level_to_peak_snr(NOISE_LEVEL),
            'seed': seed,
            'noise_seeded': True,
            'input_sha256': input_sha256(Y),
            'n_batch': n_batch,
            'n_common_accuracy_subset': n_loop,
            'dtype': 'float32 spectra, float64 reference fits',
        },
        'fairness_note': (
            'Workflow-vs-workflow comparison on one machine: batch '
            'solver uses the GPU when usable, per-spectrum tools run '
            'their normal single-thread CPU loop. Accuracy compared '
            'on the identical first n_common spectra.'),
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
