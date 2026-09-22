"""Does MLX 0.32.2 still fail alternating projection above 65,535?

ml-explore/mlx#3858: the CUDA batched GEMV mapped the batch onto a single
grid dimension, so a launch above 65,535 failed. It was fixed upstream in
mlx#3929, merged 2026-08-05 and closed completed 2026-08-06. MLX 0.32.2
shipped 2026-08-25 and should contain it.

Measured 2026-09-22 on an RTX 5070 Laptop under WSL2 at MLX 0.32.2, and
on an M3 Max: all three sizes complete in a single chunk and agree with
the split run to the bit. The harness chunks unconditionally, so it
never crosses the bound on its own -- this script is how that claim gets
re-checked on a future MLX.

Each size is solved twice on the same input: once as a single chunk that
crosses the bound, and once split in two so every chunk stays under it.
A crash and a silent disagreement are then both visible.

    python probe_mlx3858_batch_limit.py
"""
import traceback

import numpy as np

from toyomacro.voigtfit.benchmarks.bench_platform import (
    GAMMA,
    SIGMA,
    _backend_label,
    make_problem_2comp,
)
from toyomacro.voigtfit.multipeak_config import ComponentConfig, MultiPeakConfig
from toyomacro.voigtfit.multipeak_solver import (
    build_multipeak_dictionaries,
    solve_multipeak_chunked,
)

SIZES = (65_535, 65_536, 65_537)
KWARGS = dict(n_iterations=3, parabola_dE=True, parabola_ds=True)


def solve(Y, dicts, chunk):
    return solve_multipeak_chunked(Y, dicts, chunk_size=chunk, **KWARGS)


print(f"backend: {_backend_label()}\n", flush=True)
for n in SIZES:
    energy, centers, Y, _ = make_problem_2comp(n)
    cfg = MultiPeakConfig(
        peaks=[ComponentConfig(center=float(c), sigma=SIGMA, gamma=GAMMA,
                               dE_range=0.15 * SIGMA, ds_range=0.05 * SIGMA,
                               n_dE=13, n_ds=7) for c in centers],
        energy_axis=energy)
    dicts = build_multipeak_dictionaries(cfg)

    print(f"n = {n:,}", flush=True)
    try:
        one = solve(Y, dicts, n)                  # a single chunk, over the bound
    except Exception:
        print("  single chunk: CRASHED -- the limit still applies\n")
        traceback.print_exc()
        print(flush=True)
        continue
    print("  single chunk: ok", flush=True)

    split = solve(Y, dicts, n // 2 + 1)           # two chunks, both under it
    worst = 0.0
    for field in ("amplitudes", "delta_E", "delta_sigma"):
        a, b = getattr(one, field), getattr(split, field)
        d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        worst = max(worst, d)
        print(f"  {field:<12} max |one - split| = {d:.3e}", flush=True)
    verdict = "AGREE" if worst < 1e-5 else "DISAGREE -- silently wrong"
    print(f"  -> {verdict}\n", flush=True)
