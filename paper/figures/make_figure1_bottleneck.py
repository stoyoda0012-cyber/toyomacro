"""Generate Figure 1: throughput bottleneck hierarchy for the JOSS paper.

Reproduces the roofline-style bar chart comparing theoretical hardware
bounds against Stage 1 (amplitude-only) throughput.  Theoretical
bounds use the same arithmetic as
``toyomacro.voigtfit.benchmarks.theoretical_limits``.

By default the two throughput bars use the documented values from the
paper (Apple M3 Max, 128 GB).  With ``--measure`` the fit-only kernel
is measured live on the current host, so the figure regenerates
itself on new hardware; pass the hardware specs of that host via
``--mem-bw-gbps/--tflops/--ssd-gbps``.  The end-to-end bar is a
projection from the HDF5 bound unless ``--h5 FILE`` points to a real
dataset, in which case it is measured by streaming that file from
disk through the kernel.

Usage:
    python paper/figures/make_figure1_bottleneck.py
    python paper/figures/make_figure1_bottleneck.py --measure
    python paper/figures/make_figure1_bottleneck.py --measure \\
        --mem-bw-gbps 546 --tflops 18.4   # e.g. on an M4 Max
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Stage 1 per-spectrum costs (n_energy=151, n_comp=3, float32) ─
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

# ── Documented values (Apple M3 Max 128 GB; see paper.md) ────────
DOC_FIT_ONLY = 400e6      # spec/s, Stage 1 amplitude-only kernel
DOC_E2E = 8.4e6           # spec/s, incl. HDF5 read from SSD


def measure_fit_only(n_spectra: int = 4_000_000, repeats: int = 5) -> float:
    """Live-measure the Stage 1 kernel: A = Y @ W + sampled chi2.

    Matches the memory-access model used for the theoretical bound
    (read Y, write A + chi2, chi2 on a 0.01% sample).
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

    best = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        A = Y @ W
        resid = Y[idx] - A[idx] @ Phi
        chi2 = mx.sum(resid * resid, axis=1)
        mx.eval(A, chi2)
        dt = time.perf_counter() - t0
        best = max(best, n_spectra / dt)
    return best


def measure_e2e(h5_path: str, dataset: str | None = None) -> float:
    """Measure disk-to-amplitude throughput on a real HDF5 file."""
    import h5py
    import mlx.core as mx

    rng = np.random.default_rng(0)
    W = None
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
        n_energy = ds.shape[1]
        W = mx.array(rng.random((n_energy, N_COMP), dtype=np.float32))
        for start in range(0, ds.shape[0], 2_000_000):
            chunk = ds[start:start + 2_000_000].astype(np.float32)
            A = mx.array(chunk) @ W
            mx.eval(A)
            n_done += chunk.shape[0]
    return n_done / (time.perf_counter() - t0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measure', action='store_true',
                   help='live-measure the fit-only kernel on this host')
    p.add_argument('--h5', type=str, default=None,
                   help='HDF5 file for a real end-to-end measurement')
    p.add_argument('--dataset', type=str, default=None,
                   help='dataset name inside --h5 (default: first 2-D)')
    p.add_argument('--mem-bw-gbps', type=float, default=400.0,
                   help='memory bandwidth of this host [GB/s]')
    p.add_argument('--tflops', type=float, default=14.2,
                   help='GPU FP32 compute of this host [TFLOPS]')
    p.add_argument('--ssd-gbps', type=float, default=7.4,
                   help='SSD read bandwidth of this host [GB/s]')
    args = p.parse_args()

    if args.measure:
        fit_only = measure_fit_only()
        fit_label = 'Fit-only (measured)'
        print(f'fit-only measured: {fit_only / 1e6:.0f} M spec/s')
    else:
        fit_only, fit_label = DOC_FIT_ONLY, 'Fit-only (measured)'

    hdf5_bound = 0.70 * args.ssd_gbps * 1e9 / BYTES_PER_SPECTRUM_DISK
    if args.h5:
        e2e = measure_e2e(args.h5, args.dataset)
        e2e_label = 'End-to-end (measured)'
        print(f'end-to-end measured: {e2e / 1e6:.1f} M spec/s')
    elif args.measure:
        e2e, e2e_label = None, None   # no honest number available live
    else:
        e2e, e2e_label = DOC_E2E, 'End-to-end (measured)'

    bounds = [
        ('GPU compute bound',
         args.tflops * 1e12 / FLOP_PER_SPECTRUM, 'theoretical'),
        ('Memory-bandwidth bound',
         args.mem_bw_gbps * 1e9 / BYTES_PER_SPECTRUM_MEM, 'theoretical'),
        (fit_label, fit_only, 'measured'),
        ('SSD raw bound',
         args.ssd_gbps * 1e9 / BYTES_PER_SPECTRUM_DISK, 'theoretical'),
        ('HDF5 bound (~70% eff.)', hdf5_bound, 'theoretical'),
    ]
    if e2e is not None:
        bounds.append((e2e_label, e2e, 'measured'))

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    y = range(len(bounds))[::-1]
    colors = {'theoretical': '#9ecae1', 'measured': '#2171b5'}
    hatches = {'theoretical': '//', 'measured': None}

    for yi, (label, rate, kind) in zip(y, bounds):
        ax.barh(yi, rate / 1e6, color=colors[kind],
                hatch=hatches[kind], edgecolor='white')
        ax.text(rate / 1e6 * 1.15, yi, f'{rate / 1e6:,.0f} M',
                va='center', fontsize=9)

    ax.set_yticks(list(y))
    ax.set_yticklabels([b[0] for b in bounds], fontsize=9)
    ax.set_xscale('log')
    ax.set_xlim(1, 1e5)
    ax.set_xlabel('Stage 1 throughput (million spectra / s)')
    ax.set_title(f'Bottleneck hierarchy — {args.mem_bw_gbps:.0f} GB/s, '
                 f'{args.tflops:.1f} TFLOPS, {N_ENERGY} ch '
                 f'× {N_COMP} comp', fontsize=10)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['theoretical'],
                      hatch='//', edgecolor='white',
                      label='theoretical bound'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['measured'],
                      label='measured'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=9)
    fig.tight_layout()

    out = Path(__file__).parent / 'figure1_bottleneck.png'
    fig.savefig(out, dpi=200)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
