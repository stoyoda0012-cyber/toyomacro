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
- **All backends**: alternating projection is chunked at 65,535, the CUDA
  ``gridDim`` limit (ml-explore/mlx#3858, unfixed as of MLX 0.32.2).
  Metal has no such limit but chunks anyway, so that one command measures
  the same work everywhere. The chunk size is recorded.
- **NumPy hosts**: every contender still runs; nothing is skipped. The
  backend-parity check is skipped, since both arms would be identical.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from toyomacro.voigtfit._mlx_support import mlx_usable
from toyomacro.voigtfit.benchmarks.record import (
    SCHEMA_VERSION,
    backend_info,
    cpu_jiffies,
    environment,
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

#: Above this, MLX's CUDA backend maps the batch onto a grid dimension
#: untiled and the launch fails (ml-explore/mlx#3858), still unfixed as of
#: MLX 0.32.2. Metal has no such limit -- but the chunking is applied on
#: EVERY backend anyway, because a Metal run that processed 200k in one
#: call and a CUDA run that processed it in four is not the same
#: measurement, and comparing the two would be the exact error this
#: harness exists to prevent.
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
    z = np.load(in_path)
    energy, Y = z["energy"], z["Y"]
    truth = {"amp": z["amp"], "dE": z["dE"], "dsigma": z["dsigma"]}
    n = Y.shape[0]
    r_d = bench_dict2d(energy, Y, truth, n, repeats=1, warmup=0)
    r_t = bench_taylor_4step(energy, Y, truth, n, repeats=1, warmup=0)
    np.savez(out_path,
             d_amp=r_d["_est"][0], d_dE=r_d["_est"][1], d_ds=r_d["_est"][2],
             t_amp=r_t["_est"][0], t_dE=r_t["_est"][1], t_ds=r_t["_est"][2],
             d_rate=r_d["rate_median"], t_rate=r_t["rate_median"])


def parity_check(n: int, accel_est: dict, tmpdir: Path, energy, Y, truth) -> dict:
    """Refit the same spectra on the NumPy backend and report the difference.

    Runs in a subprocess with ``TOYOMACRO_DISABLE_MLX=1``, because the
    backend is chosen at import time. Returns per-solver absolute
    differences, not a pass/fail -- ``dict2d_parabola`` legitimately
    differs, since the NumPy path falls back to the dictionary-only
    solver with no sub-grid refinement.
    """
    in_path, path = tmpdir / "_parity_in.npz", tmpdir / "_parity_out.npz"
    np.savez(in_path, energy=energy, Y=Y[:n], amp=truth["amp"][:n],
             dE=truth["dE"][:n], dsigma=truth["dsigma"][:n])
    env = dict(os.environ, TOYOMACRO_DISABLE_MLX="1")
    subprocess.run([sys.executable, "-m", __spec__.name,
                    "--parity-dump", str(in_path), "--parity-out", str(path)],
                   check=True, env=env)
    z = np.load(path)
    in_path.unlink(missing_ok=True)
    out = {"n_spectra": n, "numpy_backend": "TOYOMACRO_DISABLE_MLX=1 subprocess"}
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
    path.unlink(missing_ok=True)
    return out


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
        parity_n: int, do_parity: bool, tmpdir: Path,
        origin: str | None = None, host: str | None = None,
        chunk: int | None = CUDA_BATCH_LIMIT) -> dict:
    """Measure every contender and return the complete record.

    Hypervisor steal is sampled across the whole measurement rather than
    at the end: on a cloud instance it is the difference between a slow
    machine and a machine that was not given the CPU, and a reading
    taken afterwards would cover the wrong window.
    """
    jiffies_before = cpu_jiffies()
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
        parity = parity_check(parity_n, est, tmpdir, energy, Y, truth)
        for k in ("dict2d_parabola", "taylor_4step"):
            q = parity[k]
            print(f"  {k}: |dAmp| med {q['abs_diff_amp']['median']:.2e} "
                  f"max {q['abs_diff_amp']['max']:.2e}; "
                  f"numpy {q['numpy_rate_spec_per_s']:,.0f} spec/s", flush=True)
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
            origin=origin, host=host,
            steal=steal_percent(jiffies_before, cpu_jiffies())),
    }


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
    tmpdir = (out_path.parent if out_path
              else Path(args.records_dir) if args.records_dir else Path.cwd())
    tmpdir.mkdir(parents=True, exist_ok=True)

    if info["backend"] == "cuda" and args.n_batch > CUDA_BATCH_LIMIT:
        print(f"note: --n-batch {args.n_batch:,} exceeds the CUDA gridDim "
              f"limit ({CUDA_BATCH_LIMIT:,}). Alternating projection is "
              "chunked, but if a non-AP solver trips ml-explore/mlx#3858 the "
              "run will abort -- that is a finding, not a mistake; report it.",
              flush=True)

    report = run(args.n_batch, args.n_loop, args.repeats, args.warmup,
                 args.parity_n, not args.no_parity, tmpdir,
                 origin=args.origin, host=args.host_label, chunk=args.chunk)

    load = report["environment"]["load"]
    if load.get("looks_quiet") is False:
        print(f"\nWARNING: load average {load['load_average'][0]:.1f} on "
              f"{load['logical_cores']} cores - this host was busy. "
              "Throughput here is not comparable with a record taken on an "
              "idle machine.", flush=True)

    q = report["environment"]["quality"]
    print(f"\nrecord quality: {q['verdict']}"
          + (f" - {'; '.join(q['reasons'])}" if q["reasons"] else ""), flush=True)

    if args.records_dir:
        print(f"wrote {write_record(report, args.records_dir)}", flush=True)
    if out_path:
        out_path.write_text(json.dumps(report, indent=1, default=float) + "\n")
        print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
