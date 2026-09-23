"""One throughput measurement, run the same way on every supported host.

The package has three compute paths -- MLX on Metal, MLX on CUDA, and
NumPy -- and until now each was measured by a different script, at a
different batch size, on a different date, with the results stored in a
different place. That makes the numbers incomparable by construction.
This module is the single harness: same seeded problem, same solvers,
same record schema, one command.

    python -m toyomacro.voigtfit.benchmarks.bench_platform --out record.json

The problem is the one from
:mod:`~toyomacro.voigtfit.benchmarks.solver_comparison_benchmark` --
a single Voigt peak, 151 channels, gamma fixed, free amplitude / dE /
dsigma, peak-count SNR 10, seed 0 -- so a record written here is
comparable with the committed ``solver_comparison.json`` **at the same
batch size**. Batch size is an explicit argument and is recorded,
because it moves the answer: the MLX path climbs with it and the NumPy
path peaks near 50k and then degrades.

Contenders
----------
``projection_kernel``   amplitude-only ``A = Y @ W`` on random data; the
                        headline kernel, not a fit
``amp_only_projection`` amplitude-only projection on the benchmark spectra
``taylor_4step``        4-step Taylor residual projection (amp + dE + dsigma)
``dict2d_parabola``     production single-peak solver (amp + dE + dsigma)
``multipeak_2comp``     production 2-component alternating projection
``scipy_curve_fit``     conventional per-spectrum bounded least squares
``lmfit``               the same, via lmfit

The last two fit only ``--n-loop`` spectra; they are the conventional
workflow being compared against, not a batch path.

Platform notes
--------------
- **CUDA**: MLX's CUDA backend runs float32 matmul in TF32 by default,
  which is not an acceptable default for a fitting engine. This harness
  refuses to record a CUDA run with TF32 left on unless ``--allow-tf32``
  is given, and records the setting either way.
- **All backends**: alternating projection is chunked at 65,535, once the
  CUDA ``gridDim`` limit (ml-explore/mlx#3858, first released fixed in
  MLX 0.32.1). Metal never had the limit. The chunking stays because it is
  what makes one command measure the same work everywhere, and it
  protects an older MLX. The chunk size is recorded.
- **NumPy hosts**: every contender still runs; nothing is skipped. The
  backend-parity check is skipped, since both arms would be identical.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from toyomacro.voigtfit._mlx_support import mlx_usable
from toyomacro.voigtfit.benchmarks.record import (
    SCHEMA_VERSION,
    backend_info,
    cpu_jiffies,
    environment,
    load_snapshot,
    measure_projection_kernel,
    steal_percent,
    summarize,
    write_record,
)
from toyomacro.voigtfit.benchmarks.solver_comparison_benchmark import (
    CENTER,
    DE_FRAC,
    DSIGMA_FRAC,
    GAMMA,
    N_ENERGY,
    NOISE_LEVEL,
    SIGMA,
    bench_lmfit,
    bench_scipy_curve_fit,
    input_sha256,
    make_problem,
)
from toyomacro.voigtfit.dictionary_solver import (
    build_dictionary_2d,
    solve_dict2d_parabola,
)
from toyomacro.voigtfit.multipeak_config import ComponentConfig, MultiPeakConfig
from toyomacro.voigtfit.multipeak_solver import (
    build_multipeak_dictionaries,
    solve_multipeak_chunked,
)
from toyomacro.voigtfit.pipeline import HybridPipeline
from toyomacro.voigtfit.spectra_generator import add_poisson_noise
from toyomacro.voigtfit.voigt_jacobian import voigt_profile
from toyomacro.voigtfit.weight_cache import WeightMatrixCache

PEAK_CONFIG = {"centers": np.array([CENTER]), "sigmas": np.array([SIGMA]),
               "gamma": GAMMA}
GRID = dict(N_dE=13, N_ds=7)          # identical to solver_comparison_benchmark

#: Above this, MLX's CUDA backend used to map the batch onto a grid
#: dimension untiled and the launch failed (ml-explore/mlx#3858). The fix,
#: mlx#3929, was verified on this project's CUDA host twice: on a dev build
#: on 2026-08-06, where the AP solver ran 200k spectra unchunked
#: (docs/upstream-issues/mlx-issue-1-batched-gemv-65536.md), and on the
#: released MLX 0.32.2 on 2026-09-22, where a single chunk of 65,537
#: agreed with a split one bit for bit -- reported from that host, not
#: committed. The first release carrying the fix is 0.32.1. The same
#: check on Metal (MLX 0.31.2) is the control. The chunking is applied on EVERY backend
#: anyway, because a Metal run that processed 200k in one call and a CUDA
#: run that processed it in four is not the same measurement, and
#: comparing the two would be the exact error this harness exists to
#: prevent. It also keeps an older MLX safe.
CUDA_BATCH_LIMIT = 65_535


def _mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - b)))


def _timed(fn, warmup: int, repeats: int):
    """Run ``fn`` ``warmup + repeats`` times, timing only the tail."""
    out, times = None, []
    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return out, times


def _backend_label() -> str:
    return backend_info()["backend"]


# ---------------------------------------------------------------- contenders
def bench_amp_only(energy, Y, truth, n_common, repeats, warmup) -> dict:
    """Amplitude-only projection on the benchmark spectra (n_comp = 1).

    Unlike ``projection_kernel`` this runs on the real seeded problem,
    so its accuracy column is meaningful.
    """
    cache = WeightMatrixCache()
    t0 = time.perf_counter()
    Wt, _Phi = cache.get_or_create("X", "bench", energy,
                                   PEAK_CONFIG["centers"], PEAK_CONFIG["sigmas"],
                                   GAMMA)
    setup = time.perf_counter() - t0
    W = np.ascontiguousarray(np.asarray(Wt, dtype=np.float32).T)

    if mlx_usable():
        import mlx.core as mx
        W_d, Y_d = mx.array(W), mx.array(Y)
        mx.eval(W_d, Y_d)

        def run():
            A = Y_d @ W_d
            mx.eval(A)
            return A
    else:
        def run():
            return Y @ W

    A, times = _timed(run, warmup, repeats)
    A = np.asarray(A)
    return {
        "solver": "amp_only_projection", "backend": _backend_label(),
        "dtype": "float32",
        "problem": "amplitude only at nominal (center, sigma)",
        "setup_s": setup, "warmup": warmup, "repeats": repeats,
        **summarize(times, Y.shape[0]),
        "mae_amp": _mae(A[:n_common, 0], truth["amp"][:n_common]),
        "mae_dE": None, "mae_dsigma": None,
    }


def bench_taylor_4step(energy, Y, truth, n_common, repeats, warmup) -> dict:
    """4-step Taylor residual projection: amp + dE + dsigma."""
    pipe = HybridPipeline(WeightMatrixCache(), enable_stage2=False)

    def run():
        return pipe.process_rowmajor_extended_3param(
            Y, "X", "bench", energy, PEAK_CONFIG, n_steps=4)

    res, times = _timed(run, warmup, repeats)
    amp = res.amplitudes[0] if res.amplitudes.ndim == 2 else res.amplitudes
    return {
        "solver": "taylor_4step", "backend": _backend_label(), "dtype": "float32",
        "problem": "amp + dE + dsigma via 4-step Taylor residual projection "
                   "(first-order lineshape expansion; pre-dictionary path)",
        "warmup": warmup, "repeats": repeats, **summarize(times, Y.shape[0]),
        "mae_amp": _mae(amp[:n_common], truth["amp"][:n_common]),
        "mae_dE": _mae(res.energy_shifts[:n_common], truth["dE"][:n_common]),
        "mae_dsigma": _mae(res.sigma_shifts[:n_common], truth["dsigma"][:n_common]),
        "_est": (np.asarray(amp), np.asarray(res.energy_shifts),
                 np.asarray(res.sigma_shifts)),
    }


def bench_dict2d(energy, Y, truth, n_common, repeats, warmup) -> dict:
    """Production single-peak solver: 13x7 dictionary plus parabolic refine."""
    t0 = time.perf_counter()
    cache = build_dictionary_2d(
        energy, PEAK_CONFIG["centers"], PEAK_CONFIG["sigmas"], GAMMA,
        dE_range=(-DE_FRAC * SIGMA, DE_FRAC * SIGMA),
        dsigma_range=(-DSIGMA_FRAC * SIGMA, DSIGMA_FRAC * SIGMA), **GRID)
    setup = time.perf_counter() - t0
    out, times = _timed(lambda: solve_dict2d_parabola(Y, cache), warmup, repeats)
    A, _chi2, dE, ds, _ = out
    amp = A[0] if A.ndim == 2 else A
    return {
        "solver": "dict2d_parabola", "backend": _backend_label(), "dtype": "float32",
        "problem": f"amp + dE + dsigma; 2-D dictionary {GRID['N_dE']}x{GRID['N_ds']} "
                   f"(|dE|<={DE_FRAC}sigma, |dsigma|<={DSIGMA_FRAC}sigma) "
                   "+ parabola refine",
        "setup_s": setup, "warmup": warmup, "repeats": repeats,
        **summarize(times, Y.shape[0]),
        "mae_amp": _mae(amp[:n_common], truth["amp"][:n_common]),
        "mae_dE": _mae(dE[:n_common], truth["dE"][:n_common]),
        "mae_dsigma": _mae(ds[:n_common], truth["dsigma"][:n_common]),
        "_est": (np.asarray(amp), np.asarray(dE), np.asarray(ds)),
    }


def make_problem_2comp(n: int, seed: int = 1, sep: float = 2.0):
    """Two Voigt peaks ``sep`` eV apart, same noise model as ``make_problem``."""
    rng = np.random.default_rng(seed)
    energy = np.linspace(-3, 5, N_ENERGY).astype(np.float64)
    centers = np.array([CENTER, CENTER + sep])
    amp = rng.uniform(0.5, 2.0, (n, 2))
    dE = rng.uniform(-0.15 * SIGMA, 0.15 * SIGMA, (n, 2))
    dsig = rng.uniform(-0.05 * SIGMA, 0.05 * SIGMA, (n, 2))
    Y = np.zeros((n, N_ENERGY), dtype=np.float32)
    for c in range(2):
        for s in range(0, n, 100_000):
            e = slice(s, min(n, s + 100_000))
            prof = np.stack([voigt_profile(energy, centers[c] + dE[i, c],
                                           SIGMA + dsig[i, c], GAMMA)
                             for i in range(e.start, e.stop)])
            Y[e] += (prof * amp[e, c][:, None]).astype(np.float32)
    Y = add_poisson_noise(Y, NOISE_LEVEL, rng=rng)
    return energy, centers, Y, {"amp": amp, "dE": dE, "dsigma": dsig}


def bench_multipeak(n: int, repeats: int, warmup: int, n_common: int,
                    chunk: int | None = CUDA_BATCH_LIMIT) -> dict:
    """Two-component alternating projection, chunked identically everywhere.

    ``chunk`` is applied regardless of backend and recorded. It is not
    derived from the detected device on purpose: the CUDA limit forces
    chunking there, so Metal must chunk to match, or the two records
    measure different work under the same command.
    """
    energy, centers, Y, truth = make_problem_2comp(n)
    t0 = time.perf_counter()
    cfg = MultiPeakConfig(
        peaks=[ComponentConfig(center=float(c), sigma=SIGMA, gamma=GAMMA,
                               dE_range=0.15 * SIGMA, ds_range=0.05 * SIGMA,
                               n_dE=13, n_ds=7) for c in centers],
        energy_axis=energy)
    dicts = build_multipeak_dictionaries(cfg)
    setup = time.perf_counter() - t0

    kwargs = dict(n_iterations=3, parabola_dE=True, parabola_ds=True)
    if chunk:
        kwargs["chunk_size"] = chunk

    res, times = _timed(
        lambda: solve_multipeak_chunked(Y, dicts, **kwargs), warmup, repeats)
    return {
        "solver": "multipeak_2comp", "backend": _backend_label(), "dtype": "float32",
        "problem": "2 Voigt components 4 sigma apart, per-component "
                   "amp + dE + dsigma; alternating projection x3 + parabola, "
                   "13x7 grids",
        "input_sha256": input_sha256(Y), "setup_s": setup,
        "chunk_size": chunk, "warmup": warmup, "repeats": repeats,
        **summarize(times, n),
        "mae_amp": _mae(res.amplitudes[:n_common], truth["amp"][:n_common]),
        "mae_dE": _mae(res.delta_E[:n_common], truth["dE"][:n_common]),
        "mae_dsigma": _mae(res.delta_sigma[:n_common], truth["dsigma"][:n_common]),
    }


# ------------------------------------------------------------- backend parity
def _parity_dump(in_path: str, out_path: str) -> None:
    """Subprocess entry point, run with the MLX backend disabled."""
    with np.load(in_path) as z:
        energy, Y = z["energy"], z["Y"]
        truth = {"amp": z["amp"], "dE": z["dE"], "dsigma": z["dsigma"]}
    n = Y.shape[0]
    r_d = bench_dict2d(energy, Y, truth, n, repeats=1, warmup=0)
    r_t = bench_taylor_4step(energy, Y, truth, n, repeats=1, warmup=0)
    np.savez(out_path,
             d_amp=r_d["_est"][0], d_dE=r_d["_est"][1], d_ds=r_d["_est"][2],
             t_amp=r_t["_est"][0], t_dE=r_t["_est"][1], t_ds=r_t["_est"][2],
             d_rate=r_d["rate_median"], t_rate=r_t["rate_median"])


def parity_check(n: int, accel_est: dict, energy, Y, truth) -> dict:
    """Refit the same spectra on the NumPy backend and report the difference.

    Runs in a subprocess with ``TOYOMACRO_DISABLE_MLX=1``, because the
    backend is chosen at import time. Returns per-solver absolute
    differences, not a pass/fail -- ``dict2d_parabola`` legitimately
    differs, since the NumPy path falls back to the dictionary-only
    solver with no sub-grid refinement.
    """
    # Both context managers matter on Windows: an NpzFile holds its file
    # open, and a directory with an open file in it cannot be removed there.
    with tempfile.TemporaryDirectory(prefix="toyomacro-parity-") as scratch:
        in_path = Path(scratch) / "parity_in.npz"
        path = Path(scratch) / "parity_out.npz"
        np.savez(in_path, energy=energy, Y=Y[:n], amp=truth["amp"][:n],
                 dE=truth["dE"][:n], dsigma=truth["dsigma"][:n])
        env = dict(os.environ, TOYOMACRO_DISABLE_MLX="1")
        subprocess.run([sys.executable, "-m", __spec__.name,
                        "--parity-dump", str(in_path), "--parity-out", str(path)],
                       check=True, env=env)
        with np.load(path) as npz:
            z = {k: npz[k] for k in npz.files}
    out = {
        "n_spectra": n,
        "numpy_backend": "TOYOMACRO_DISABLE_MLX=1 subprocess",
        "repeats": 1, "warmup": 0, "aggregation": "single timing",
        "rate_caveat": (
            "numpy_rate_spec_per_s is ONE cold timing of n_spectra, not a "
            "warmed median of the full batch, and n_spectra is near the "
            "NumPy path's throughput peak. It is a parity artefact, not a "
            "throughput figure -- do not compare it with rate_median above "
            "or with a record taken at a different batch size."),
    }
    for key, pre in (("dict2d_parabola", "d"), ("taylor_4step", "t")):
        a_m, e_m, s_m = accel_est[key]
        da = np.abs(a_m[:n] - z[f"{pre}_amp"])
        de = np.abs(e_m[:n] - z[f"{pre}_dE"])
        ds = np.abs(s_m[:n] - z[f"{pre}_ds"])
        out[key] = {
            "note": ("the NumPy backend falls back to the dictionary-only "
                     "solver (no parabola sub-grid refinement), so "
                     "grid-quantised dE/dsigma differences are expected"
                     if pre == "d" else "same algorithm on both backends"),
            "abs_diff_amp": {"median": float(np.median(da)),
                             "p99": float(np.percentile(da, 99)),
                             "max": float(da.max())},
            "abs_diff_dE_eV": {"median": float(np.median(de)),
                               "p99": float(np.percentile(de, 99)),
                               "max": float(de.max())},
            "abs_diff_dsigma_eV": {"median": float(np.median(ds)),
                                   "p99": float(np.percentile(ds, 99)),
                                   "max": float(ds.max())},
            "fraction_dE_diff_gt_0.01eV": float(np.mean(de > 0.01)),
            "fraction_amp_diff_gt_1e-3": float(np.mean(da > 1e-3)),
            "numpy_rate_spec_per_s": float(z[f"{pre}_rate"]),
        }
    return out


_VERSION = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?(.*)$")
_PRERELEASE = re.compile(r"(dev|rc|a|b|alpha|beta|pre)\d*", re.IGNORECASE)


def _mlx_older_than(version: str | None, fixed: tuple[int, int, int]) -> bool:
    """True unless ``version`` is known to be release ``fixed`` or later.

    A pre-release or dev build *of* ``fixed`` counts as older, because its
    version string cannot say whether it includes the fix. So does a
    missing or unparseable version. The note this gates is a warning, and
    a false one costs less than a missed one. A local label (``+abc``)
    does not make a release a pre-release.
    """
    m = _VERSION.match(version or "")
    if not m:
        return True
    release = tuple(int(g or 0) for g in m.group(1, 2, 3))
    if release != fixed:
        return release < fixed
    suffix = (m.group(4) or "").split("+", 1)[0]
    return bool(_PRERELEASE.search(suffix))


def _check_tf32(allow: bool) -> None:
    """Refuse to record a CUDA run with TF32 silently left on."""
    info = backend_info()
    if info["backend"] == "cuda" and info["tf32_enabled"] and not allow:
        raise SystemExit(
            "Refusing to record: the CUDA backend has TF32 enabled, which "
            "runs float32 matmul at roughly 1000x the error and measurably "
            "degrades fitted parameters.\n"
            "Set MLX_ENABLE_TF32=0 (or NVIDIA_TF32_OVERRIDE=0) before the "
            "first kernel call, or pass --allow-tf32 to record it anyway.")


def run(n_batch: int, n_loop: int, repeats: int, warmup: int,
        parity_n: int, do_parity: bool,
        origin: str | None = None, host: str | None = None,
        chunk: int | None = CUDA_BATCH_LIMIT) -> dict:
    """Measure every contender and return the complete record.

    Hypervisor steal is sampled across the whole measurement rather than
    at the end: on a cloud instance it is the difference between a slow
    machine and a machine that was not given the CPU, and a reading
    taken afterwards would cover the wrong window.
    """
    jiffies_before = cpu_jiffies()
    load_before = load_snapshot(exclude_self=True)
    results: list[dict] = []
    est: dict = {}

    print(f"generating {n_batch:,} single-peak spectra (seed 0) ...", flush=True)
    energy, Y, truth = make_problem(n_batch, 0)
    sha = input_sha256(Y)

    print("projection_kernel ...", flush=True)
    results.append(measure_projection_kernel(repeats=repeats, warmup=warmup))
    print("amp_only_projection ...", flush=True)
    results.append(bench_amp_only(energy, Y, truth, n_loop, repeats, warmup))
    print("taylor_4step ...", flush=True)
    r = bench_taylor_4step(energy, Y, truth, n_loop, repeats, warmup)
    est["taylor_4step"] = r.pop("_est")
    results.append(r)
    print("dict2d_parabola ...", flush=True)
    r = bench_dict2d(energy, Y, truth, n_loop, repeats, warmup)
    est["dict2d_parabola"] = r.pop("_est")
    results.append(r)
    print("multipeak_2comp ...", flush=True)
    results.append(bench_multipeak(n_batch, repeats, warmup, n_loop, chunk))

    truth_small = {k: v[:n_loop] for k, v in truth.items()}
    print(f"scipy curve_fit on {n_loop} ...", flush=True)
    results.append(bench_scipy_curve_fit(energy, Y[:n_loop], truth_small))
    print(f"lmfit on {n_loop} ...", flush=True)
    try:
        results.append(bench_lmfit(energy, Y[:n_loop], truth_small))
    except ImportError:
        print("  lmfit not installed - skipped")

    for r in results:
        rate = r.get("rate_median", r.get("throughput_spec_per_s"))
        mae = r.get("mae_amp")
        print(f"  {r['solver']:<22s} {rate:>14,.0f} spec/s"
              + (f"   MAE(amp)={mae:.4f}" if mae is not None else ""), flush=True)

    parity = None
    if do_parity and mlx_usable():
        print(f"NumPy-backend parity on first {parity_n:,} spectra ...", flush=True)
        parity = parity_check(parity_n, est, energy, Y, truth)
        for k in ("dict2d_parabola", "taylor_4step"):
            q = parity[k]
            print(f"  {k}: |dAmp| med {q['abs_diff_amp']['median']:.2e} "
                  f"max {q['abs_diff_amp']['max']:.2e}; "
                  f"numpy {q['numpy_rate_spec_per_s']:,.0f} spec/s "
                  "(one cold timing at the parity size - not a throughput "
                  "figure)", flush=True)
    elif do_parity:
        print("backend parity skipped: this host has no accelerated backend "
              "to compare against", flush=True)

    return {
        "schema_version": SCHEMA_VERSION,
        "description": "Cross-platform peak-fitting throughput: one problem, "
                       "one schema, every backend.",
        "problem": {
            "n_energy": N_ENERGY, "center": CENTER, "sigma": SIGMA,
            "gamma": GAMMA, "dE_range_sigma": DE_FRAC,
            "dsigma_range_sigma": DSIGMA_FRAC, "noise_level": NOISE_LEVEL,
            "peak_snr": 1e4 / NOISE_LEVEL, "seed": 0, "n_batch": n_batch,
            "n_common_accuracy_subset": n_loop, "input_sha256": sha,
            "dtype": "float32 spectra", "multipeak_chunk_size": chunk,
        },
        "results": results,
        "numpy_parity": parity,
        "environment": environment(
            origin=origin, host=host, load_before=load_before,
            steal=steal_percent(jiffies_before, cpu_jiffies())),
    }


# --------------------------------------------------------------- repetition
def aggregate_runs(records: list[dict]) -> dict:
    """Combine single-run records into one, and say what repetition bought.

    The number this exists to produce is ``understates_by``: the ratio of
    the spread *across* runs to the spread *within* them. A record's own
    ``rate_min``/``rate_max`` covers repetitions that shared a process, a
    memory layout and a clock state, so it cannot see what changes when
    those change. Measured on one host at 2.01x across three separate
    records against 1.03x within the tightest of them -- a record that
    looked like the most confident of three was the furthest from the
    other two.

    Each run's per-repetition timings are kept as ``per_run_timings_s``.
    A median hides a step or a drift inside a run; the repetitions show
    it, and an aggregate that discarded them would hide it twice.

    An ``understates_by`` above 1 means exactly that: the per-run range
    is optimistic by that factor, and quoting one record's range as the
    uncertainty is wrong by it.
    """
    if not records:
        raise ValueError("no run records to aggregate")

    solvers: dict[str, list[dict]] = {}
    for record in records:
        for result in record.get("results", []):
            solvers.setdefault(result["solver"], []).append(result)

    results = []
    for solver, entries in solvers.items():
        entries = [e for e in entries if e.get("rate_median")]
        medians = [e["rate_median"] for e in entries]
        if not medians:
            continue
        within = [e["rate_max"] / e["rate_min"]
                  for e in entries
                  if e.get("rate_min") and e.get("rate_max")]
        across = max(medians) / min(medians) if min(medians) > 0 else None
        within_typical = statistics.median(within) if within else None
        merged = {
            "solver": solver,
            "backend": entries[0].get("backend"),
            "problem": entries[0].get("problem"),
            "runs": len(medians),
            # `rate_median` is the median ACROSS runs, so that readers and
            # summarize_records get the repeated figure by default.
            "rate_median": statistics.median(medians),
            "rate_min": min(medians),
            "rate_max": max(medians),
            "aggregation": f"median across {len(medians)} separate runs",
            "per_run_rate_median": medians,
            "per_run_timings_s": [e.get("timings_s") for e in entries],
            "across_run_spread": across,
            "within_run_spread_typical": within_typical,
        }
        if across and within_typical and within_typical > 0:
            merged["understates_by"] = across / within_typical
            merged["understates_note"] = (
                "how much a single record's own rate_min/rate_max "
                "understates the spread across separate runs; above 1 means "
                "one record's range is optimistic by this factor")
        mae = [e["mae_amp"] for e in entries if e.get("mae_amp") is not None]
        merged["mae_amp_median"] = statistics.median(mae) if mae else None
        results.append(merged)

    problems = {json.dumps(r.get("problem"), sort_keys=True) for r in records}
    verdicts = [r.get("environment", {}).get("quality", {}).get("verdict")
                for r in records]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "repeated",
        "description": "Cross-platform peak-fitting throughput, repeated in "
                       "separate processes.",
        "runs": len(records),
        "runs_note": (
            "Each run is a separate process. Repeating inside one process "
            "would share the memory layout and clock state that make "
            "within-run repetitions agree, and would reproduce the blind "
            "spot this exists to measure."),
        "problem": records[0].get("problem"),
        "problem_identical_across_runs": len(problems) == 1,
        "results": results,
        "numpy_parity": next((r.get("numpy_parity") for r in records
                              if r.get("numpy_parity")), None),
        "run_quality_verdicts": verdicts,
        "all_runs_quiet": all(v == "quiet" for v in verdicts),
        # The aggregate inherits the first run's environment so that the
        # record has a host, a tree and a filename -- the parent process
        # measured nothing, so it has no environment of its own worth
        # recording. The verdict is replaced: an aggregate is only as
        # clean as its dirtiest run.
        "environment": {
            **(records[0].get("environment") or {}),
            "quality": {
                "verdict": ("quiet" if all(v == "quiet" for v in verdicts)
                            else "contended" if "contended" in verdicts
                            else "unknown"),
                "reasons": [f"run {i + 1} graded {v}"
                            for i, v in enumerate(verdicts) if v != "quiet"],
                "comparable": all(v == "quiet" for v in verdicts),
                "note": ("the verdict of the worst of the runs; per-run "
                         "verdicts are in run_quality_verdicts and each "
                         "run's own load is in run_environments"),
            },
        },
        "run_environments": [r.get("environment") for r in records],
    }


def run_repeated(runs: int, argv_common: list[str]) -> dict:
    """Measure `runs` times in separate processes and aggregate.

    Each child is a plain single-run invocation of this module. The
    parent only collects; it deliberately does no measuring of its own,
    so that every figure in the aggregate came from a process that did
    nothing else first.

    Parity runs once, on the first child: it is a correctness check on
    the backend, not a throughput figure, and repeating it would double
    the wall clock without adding information.

    **This is biased low as a measure of run-to-run variability.** The
    children run back to back, so they still share a thermal state, a GPU
    clock state and a warm page cache. The option was motivated by
    records separated by hours and by code generations, across which
    `amp_only_projection` moved 1.85x on one host; the record behind that
    figure was later re-taken, and the committed set moves 2.01x. Three
    back-to-back runs on the same host moved it 1.001x. Both are true:
    this catches what changes between processes, and not what changes
    between sittings.
    """
    records = []
    # Not `tmpdir`: that is often the records directory, and a child that
    # crashes would leave `_runN.json` there for `summarize_records` to
    # glob up as a real record. A TemporaryDirectory cleans up either way.
    with tempfile.TemporaryDirectory(prefix="toyomacro-runs-") as scratch:
        for index in range(runs):
            path = Path(scratch) / f"run{index}.json"
            argv = [sys.executable, "-m", __spec__.name, *argv_common,
                    "--runs", "1", "--out", str(path)]
            if index > 0:
                argv.append("--no-parity")
            print(f"\n=== run {index + 1} of {runs} "
                  "(separate process) ===", flush=True)
            subprocess.run(argv, check=True)
            records.append(json.loads(path.read_text(encoding="utf-8")))
    return aggregate_runs(records)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Cross-platform throughput record (Metal / CUDA / NumPy).")
    p.add_argument("--n-batch", type=int, default=200_000,
                   help="batch size for the batch solvers; it moves the "
                        "answer, so keep it equal across hosts "
                        "(default: %(default)s, matching solver_comparison.json)")
    p.add_argument("--n-loop", type=int, default=500,
                   help="spectra given to the per-spectrum solvers")
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=1,
                   help="measure this many times, each in a SEPARATE process, "
                        "and report the median across runs plus how much a "
                        "single run's own range understates the spread. "
                        "Repeating inside one process would share the memory "
                        "layout and clock state that make within-run "
                        "repetitions agree (default: %(default)s)")
    p.add_argument("--parity-n", type=int, default=50_000)
    p.add_argument("--no-parity", action="store_true",
                   help="skip the NumPy-backend parity check")
    p.add_argument("--chunk", type=int, default=CUDA_BATCH_LIMIT,
                   help="chunk alternating projection at this batch size on "
                        "EVERY backend (0 = no chunking). The default is the "
                        "CUDA gridDim limit, so Metal and CUDA do identical "
                        "work; changing it makes records incomparable "
                        "(default: %(default)s)")
    p.add_argument("--allow-tf32", action="store_true",
                   help="record a CUDA run even with TF32 enabled")
    p.add_argument("--origin", type=str, default=os.environ.get("TOYOMACRO_BENCH_ORIGIN"),
                   help="where this run was LAUNCHED from (mac / windows / ...). "
                        "A cloud container cannot work this out for itself, and "
                        "it is what separates 'cloud from the Mac' from 'cloud "
                        "from the PC'. Defaults to $TOYOMACRO_BENCH_ORIGIN")
    p.add_argument("--host-label", type=str, default=os.environ.get("TOYOMACRO_BENCH_HOST"),
                   help="override the hardware-derived machine label")
    p.add_argument("--out", type=str, default=None,
                   help="write the record to this exact path")
    p.add_argument("--records-dir", type=str, default=None,
                   help="write the record into this directory under an "
                        "auto-generated, conflict-free name (use this for the "
                        "shared benchmarks/records/ collection)")
    p.add_argument("--parity-dump", type=str, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--parity-out", type=str, default=None,
                   help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args.parity_dump:
        _parity_dump(args.parity_dump, args.parity_out)
        return

    _check_tf32(args.allow_tf32)

    info = backend_info()
    print(f"backend: {info['backend']} ({info['device_name']})", flush=True)

    if not args.origin:
        print("note: --origin not given; this record will be filed as "
              "'from-unknown' and will not tell you which machine launched it",
              flush=True)

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if (info["backend"] == "cuda" and args.n_batch > CUDA_BATCH_LIMIT
            and _mlx_older_than(info.get("mlx_version"), (0, 32, 1))):
        print(f"note: --n-batch {args.n_batch:,} exceeds {CUDA_BATCH_LIMIT:,}, "
              f"and MLX {info.get('mlx_version')} may predate the fix for "
              "ml-explore/mlx#3858 (first released in 0.32.1). Alternating "
              "projection is chunked, but a non-AP solver may abort the "
              "run.", flush=True)

    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.runs > 1:
        common = ["--n-batch", str(args.n_batch), "--n-loop", str(args.n_loop),
                  "--repeats", str(args.repeats), "--warmup", str(args.warmup),
                  "--parity-n", str(args.parity_n), "--chunk", str(args.chunk)]
        if args.origin:
            common += ["--origin", args.origin]
        if args.host_label:
            common += ["--host-label", args.host_label]
        if args.no_parity:
            common.append("--no-parity")
        if args.allow_tf32:
            common.append("--allow-tf32")
        report = run_repeated(args.runs, common)
    else:
        report = run(args.n_batch, args.n_loop, args.repeats, args.warmup,
                     args.parity_n, not args.no_parity,
                     origin=args.origin, host=args.host_label,
                     chunk=args.chunk)

    if report.get("kind") == "repeated":
        print(f"\n=== {report['runs']} runs, medians across them ===",
              flush=True)
        for result in report["results"]:
            line = f"  {result['solver']:<22s} {result['rate_median']:>14,.0f}"
            if result.get("understates_by"):
                line += (f"   across/within {result['understates_by']:.2f}x"
                         + ("  <-- one run's range is optimistic"
                            if result["understates_by"] > 1.5 else ""))
            print(line, flush=True)
        verdicts = report["run_quality_verdicts"]
        print(f"\nrun verdicts: {verdicts}"
              + ("" if report["all_runs_quiet"]
                 else "  <-- not every run was quiet"), flush=True)
        if args.records_dir:
            print(f"wrote {write_record(report, args.records_dir)}", flush=True)
        if out_path:
            out_path.write_text(json.dumps(report, indent=1, default=float)
                                + "\n", encoding="utf-8")
            print(f"wrote {out_path}", flush=True)
        return

    load = report["environment"]["load"]
    # Gate on the load verdict, not the composite: `looks_quiet` is also
    # set False by foreign CPU mid-run, and printing a load average as the
    # evidence for that made the console contradict the record.
    if load.get("looks_quiet_before", load.get("looks_quiet")) is False:
        # Quote the sample that decided, not the trailing one -- printing a
        # different number from the verdict is how a reader learns to
        # distrust both.
        la = (load.get("load_average_before") or load.get("load_average")
              or [None])[0]
        where = "before the run" if load.get("load_average_before") else ""
        print(f"\nWARNING: load average {la:.1f} {where} on "
              f"{load['logical_cores']} cores - this host was busy. "
              "Throughput here is not comparable with a record taken on an "
              "idle machine.", flush=True)

    q = report["environment"]["quality"]
    print(f"\nrecord quality: {q['verdict']}"
          + (f" - {'; '.join(q['reasons'])}" if q["reasons"] else ""), flush=True)

    if args.records_dir:
        print(f"wrote {write_record(report, args.records_dir)}", flush=True)
    if out_path:
        out_path.write_text(json.dumps(report, indent=1, default=float) + "\n",
                            encoding="utf-8")
        print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
