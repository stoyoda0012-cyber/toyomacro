"""Generate Figure 1: throughput bottleneck hierarchy for the JOSS paper.

Roofline-style bar chart comparing theoretical hardware bounds against
the measured amplitude-only projection kernel.  Theoretical bounds use
the same arithmetic as ``toyomacro.voigtfit.benchmarks.theoretical_limits``.

Measurement provenance
----------------------
A bar is labeled **measured** only when it comes from an actual timing
run — either performed live (``--measure`` / ``--h5``) or loaded from a
committed machine-readable result file (``--results``, default
``results/figure1_throughput.json``).  Each measurement records the
timestamp, git commit, OS, Python/NumPy/MLX versions, hardware, dtype,
problem size, warm-up count, and every individual repetition; the
reported throughput is the **median** across repetitions, with the
min–max range stored alongside.

Without ``--measure``/``--h5`` and without a result file the script
refuses to draw measurement bars (theoretical bounds only), so a
default invocation can never mislabel a constant as a measurement.
The end-to-end bar is drawn as a hatched *projection* from the HDF5
efficiency bound unless a real end-to-end measurement is available.

Usage:
    python paper/figures/make_figure1_bottleneck.py               # from committed results
    python paper/figures/make_figure1_bottleneck.py --measure     # re-measure kernel here
    python paper/figures/make_figure1_bottleneck.py --measure \\
        --mem-bw-gbps 546 --tflops 18.4                           # e.g. on an M4 Max
    python paper/figures/make_figure1_bottleneck.py --h5 data.h5  # real e2e measurement
"""

import argparse
import datetime
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Kernel per-spectrum costs (n_energy=151, n_comp=3, float32) ──
# See benchmarks/theoretical_limits.py for the full derivation.
N_ENERGY = 151
N_COMP = 3
CHI2_SAMPLE = 1e-4
BYTES_PER_SPECTRUM_MEM = (
    N_ENERGY * 4                      # read Y
    + N_COMP * 4 + 4                  # write A, chi2
    + N_ENERGY * 4 * CHI2_SAMPLE      # sampled Y_fit (amortized)
)
FLOP_PER_SPECTRUM = N_ENERGY * N_COMP * 2 + N_ENERGY * 3 * CHI2_SAMPLE
BYTES_PER_SPECTRUM_DISK = N_ENERGY * 4

RESULTS_PATH = Path(__file__).parent / 'results' / 'figure1_throughput.json'


def _environment() -> dict:
    """Record the software/hardware environment for a measurement."""
    env = {
        'timestamp_utc': datetime.datetime.now(datetime.UTC).isoformat(
            timespec='seconds'),
        'os': f'{platform.system()} {platform.release()} '
              f'(macOS {platform.mac_ver()[0]})' if platform.system() == 'Darwin'
              else f'{platform.system()} {platform.release()}',
        'machine': platform.machine(),
        'python': sys.version.split()[0],
        'numpy': np.__version__,
    }
    try:
        import importlib.metadata as md
        env['mlx'] = md.version('mlx')
    except Exception:
        env['mlx'] = None
    try:
        env['cpu'] = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        env['cpu'] = platform.processor()
    # Machine-independent invocation (argv[0] may be an absolute path)
    env['command'] = ('python paper/figures/make_figure1_bottleneck.py '
                      + ' '.join(sys.argv[1:])).strip()
    repo = Path(__file__).parent
    try:
        env['git_commit'] = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=repo).stdout.strip() or None
        # Reproducible from git_commit only when no tracked file is
        # modified (untracked files cannot affect the package).
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


def measure_fit_only(n_spectra: int = 4_000_000, repeats: int = 9,
                     warmup: int = 2) -> dict:
    """Measure the amplitude-only projection kernel: A = Y @ W + sampled chi2.

    Matches the memory-access model used for the theoretical bound
    (read Y, write A + chi2, chi2 on a 0.01% sample).  Returns a
    provenance record with every repetition; the headline value is the
    median throughput.
    """
    import mlx.core as mx

    rng = np.random.default_rng(0)
    Y = mx.array(rng.random((n_spectra, N_ENERGY), dtype=np.float32))
    W = mx.array(rng.random((N_ENERGY, N_COMP), dtype=np.float32))
    Phi = mx.array(rng.random((N_COMP, N_ENERGY), dtype=np.float32))
    idx = mx.array(
        rng.choice(n_spectra, max(1, int(n_spectra * CHI2_SAMPLE)),
                   replace=False))
    mx.eval(Y, W, Phi, idx)

    timings_s: list[float] = []
    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        A = Y @ W
        resid = Y[idx] - A[idx] @ Phi
        chi2 = mx.sum(resid * resid, axis=1)
        mx.eval(A, chi2)
        dt = time.perf_counter() - t0
        if i >= warmup:
            timings_s.append(dt)

    rates = [n_spectra / t for t in timings_s]
    return {
        'kind': 'fit_only_kernel',
        'description': 'amplitude-only projection kernel '
                       '(A = Y @ W, chi2 on 0.01% sample)',
        'backend': 'mlx-gpu',
        'dtype': 'float32',
        'n_energy': N_ENERGY,
        'n_comp': N_COMP,
        'batch_spectra': n_spectra,
        'warmup': warmup,
        'repeats': repeats,
        'timings_s': [round(t, 6) for t in timings_s],
        'aggregation': 'median',
        'throughput_spec_per_s_median': statistics.median(rates),
        'throughput_spec_per_s_min': min(rates),
        'throughput_spec_per_s_max': max(rates),
        'environment': _environment(),
    }


def measure_e2e(h5_path: str, dataset: str | None = None) -> dict:
    """Measure disk-to-amplitude throughput on a real HDF5 file.

    The committed record stores only the dataset shape/dtype, never the
    file path, so private data locations do not leak into the repo.
    """
    import h5py
    import mlx.core as mx

    rng = np.random.default_rng(0)
    n_done = 0
    t0 = time.perf_counter()
    with h5py.File(h5_path, 'r') as f:
        if dataset is None:  # first 2-D float dataset
            def _first(name, obj):
                nonlocal dataset
                if (dataset is None and isinstance(obj, h5py.Dataset)
                        and obj.ndim == 2 and obj.dtype.kind == 'f'):
                    dataset = name
            f.visititems(_first)
        ds = f[dataset]
        shape, dtype = list(ds.shape), str(ds.dtype)
        n_energy = ds.shape[1]
        W = mx.array(rng.random((n_energy, N_COMP), dtype=np.float32))
        for start in range(0, ds.shape[0], 2_000_000):
            chunk = ds[start:start + 2_000_000].astype(np.float32)
            A = mx.array(chunk) @ W
            mx.eval(A)
            n_done += chunk.shape[0]
    elapsed = time.perf_counter() - t0
    return {
        'kind': 'end_to_end',
        'description': 'HDF5 stream -> float32 -> amplitude projection',
        'backend': 'mlx-gpu',
        'dtype': 'float32',
        'n_comp': N_COMP,
        'dataset_shape': shape,
        'dataset_dtype': dtype,
        'n_spectra': n_done,
        'warmup': 0,
        'repeats': 1,
        'timings_s': [round(elapsed, 3)],
        'aggregation': 'single-pass (dominated by I/O, not repeated)',
        'throughput_spec_per_s_median': n_done / elapsed,
        'throughput_spec_per_s_min': n_done / elapsed,
        'throughput_spec_per_s_max': n_done / elapsed,
        'environment': _environment(),
    }


def _fmt_range(rec: dict) -> str:
    lo = rec['throughput_spec_per_s_min'] / 1e6
    hi = rec['throughput_spec_per_s_max'] / 1e6
    return f'{lo:,.0f}-{hi:,.0f} M'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measure', action='store_true',
                   help='measure the projection kernel live on this host '
                        'and update the result file')
    p.add_argument('--h5', type=str, default=None,
                   help='HDF5 file for a real end-to-end measurement')
    p.add_argument('--dataset', type=str, default=None,
                   help='dataset name inside --h5 (default: first 2-D)')
    p.add_argument('--results', type=str, default=str(RESULTS_PATH),
                   help='machine-readable measurement record (JSON)')
    p.add_argument('--mem-bw-gbps', type=float, default=400.0,
                   help='memory bandwidth of this host [GB/s]')
    p.add_argument('--tflops', type=float, default=14.2,
                   help='GPU FP32 compute of this host [TFLOPS]')
    p.add_argument('--ssd-gbps', type=float, default=7.4,
                   help='SSD read bandwidth of this host [GB/s]')
    args = p.parse_args()

    results_path = Path(args.results)
    records: dict = {}
    if results_path.exists():
        records = json.loads(results_path.read_text())

    # ── measurements: live runs update the record file ────────────
    if args.measure:
        rec = measure_fit_only()
        records['fit_only_kernel'] = rec
        print(f"fit-only kernel measured: "
              f"{rec['throughput_spec_per_s_median'] / 1e6:,.0f} M spec/s "
              f"(median of {rec['repeats']}, range {_fmt_range(rec)})")
    if args.h5:
        rec = measure_e2e(args.h5, args.dataset)
        records['end_to_end'] = rec
        print(f"end-to-end measured: "
              f"{rec['throughput_spec_per_s_median'] / 1e6:.1f} M spec/s")
    if args.measure or args.h5:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(records, indent=2) + '\n')
        print(f'wrote {results_path}')

    fit_rec = records.get('fit_only_kernel')
    e2e_rec = records.get('end_to_end')
    if fit_rec is None:
        print('NOTE: no kernel measurement available (no --measure and no '
              'result file); drawing theoretical bounds only.')

    # ── assemble bars ─────────────────────────────────────────────
    hdf5_bound = 0.70 * args.ssd_gbps * 1e9 / BYTES_PER_SPECTRUM_DISK
    bounds: list[tuple[str, float, str]] = [
        ('GPU compute bound',
         args.tflops * 1e12 / FLOP_PER_SPECTRUM, 'theoretical'),
        ('Memory-bandwidth bound',
         args.mem_bw_gbps * 1e9 / BYTES_PER_SPECTRUM_MEM, 'theoretical'),
    ]
    if fit_rec is not None:
        bounds.append(
            ('Projection kernel (measured)',
             fit_rec['throughput_spec_per_s_median'], 'measured'))
    bounds += [
        ('SSD raw bound',
         args.ssd_gbps * 1e9 / BYTES_PER_SPECTRUM_DISK, 'theoretical'),
        ('HDF5 bound (~70% eff.)', hdf5_bound, 'theoretical'),
    ]
    if e2e_rec is not None:
        bounds.append(('End-to-end (measured)',
                       e2e_rec['throughput_spec_per_s_median'], 'measured'))
    else:
        bounds.append(('End-to-end (projected)', hdf5_bound, 'projected'))

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    y = range(len(bounds))[::-1]
    colors = {'theoretical': '#9ecae1', 'measured': '#2171b5',
              'projected': '#c6dbef'}
    hatches = {'theoretical': '//', 'measured': None, 'projected': 'xx'}

    for yi, (label, rate, kind) in zip(y, bounds):
        ax.barh(yi, rate / 1e6, color=colors[kind],
                hatch=hatches[kind], edgecolor='white')
        ax.text(rate / 1e6 * 1.15, yi, f'{rate / 1e6:,.0f} M',
                va='center', fontsize=9)

    ax.set_yticks(list(y))
    ax.set_yticklabels([b[0] for b in bounds], fontsize=9)
    ax.set_xscale('log')
    ax.set_xlim(1, 1e5)
    ax.set_xlabel('Amplitude-only projection throughput (million spectra / s)')
    ax.set_title(f'Bottleneck hierarchy — {args.mem_bw_gbps:.0f} GB/s, '
                 f'{args.tflops:.1f} TFLOPS, {N_ENERGY} ch '
                 f'× {N_COMP} comp', fontsize=10)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['theoretical'],
                      hatch='//', edgecolor='white',
                      label='theoretical bound'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['measured'],
                      label='measured (median)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['projected'],
                      hatch='xx', edgecolor='white',
                      label='projected from bound'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=8)
    fig.tight_layout()

    out = Path(__file__).parent / 'figure1_bottleneck.png'
    fig.savefig(out, dpi=200)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
