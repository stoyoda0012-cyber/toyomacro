"""
GP-LM hardening benchmark + exact-basis / auto-routing experiments.
=========================================================================

Compares the post-AP Newton variants on identical AP seeds:

    raw-MLX        production kernel (frozen surrogate)
    raw-CPU        _post_ap_newton_refine_numpy(mode="raw")
    gp-CPU         explicit projected columns (mode="golub_pereyra")
    gplm-CPU       compact GP + LM accept/reject (CPU)
    gplm-MLX       compact GP + LM, MLX-hybrid tensor path
    gplm-exact*    float64 exact-basis GP-LM via gp_reference.run_newton
                   (Phase-7 judgment experiment; small subset only)

plus §7.2 assembly micro-benchmark (stacked matmul vs entry-wise K) and a
§12 auto-routing experiment (raw everywhere → GP-LM only on the routed,
condition-gated subset).

Usage:
    python -m toyomacro.voigtfit.benchmarks.bench_gp_lm [--quick]
        [--big] [--out report.md]
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
from scipy.special import wofz

from toyomacro.voigtfit.gp_compact import GPLMConfig
from toyomacro.voigtfit.gp_reference import SeparableVoigtModel, run_newton
from toyomacro.voigtfit.multipeak_config import ComponentConfig, MultiPeakConfig
from toyomacro.voigtfit.multipeak_solver import (
    HAS_MLX,
    _post_ap_gp_lm_refine_numpy,
    _post_ap_newton_refine_numpy,
    build_multipeak_dictionaries,
    solve_alternating_projection,
)

warnings.filterwarnings("ignore", message="Peak separation")


# ---------------------------------------------------------------------------
# problems / data
# ---------------------------------------------------------------------------


def make_problem(n_comp, sep_sigma, sigma=0.5, gamma=0.15, bg_degree=-1,
                 small_amp=False):
    c0 = 284.0
    centers = c0 + sigma * sep_sigma * np.arange(n_comp)
    lo, hi = centers.min() - 3.0, centers.max() + 3.0
    energy = np.linspace(lo, hi, 121, dtype=np.float32)
    cfg = MultiPeakConfig(
        peaks=[ComponentConfig(center=float(c), sigma=sigma, gamma=gamma,
                               dE_range=0.6, ds_range=0.1, n_dE=7, n_ds=9)
               for c in centers],
        energy_axis=energy,
    )
    amps = 1000.0 * (1.0 - 0.3 * (np.arange(n_comp) % 2))
    if small_amp:
        amps[-1] = 0.02 * amps.max()
    return {
        "centers": centers, "sigma": sigma, "gamma": gamma,
        "energy": energy, "config": cfg, "amps": amps,
        "bg_degree": bg_degree,
    }


def voigt_np(E, c, s, g):
    z = ((E - c) + 1j * g) / (s * np.sqrt(2))
    return np.real(wofz(z)) / (s * np.sqrt(2 * np.pi))


def gen_batch(prob, n, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    E = prob["energy"].astype(np.float64)
    n_comp = len(prob["centers"])
    dE_t = rng.uniform(-0.2, 0.2, (n, n_comp))
    ds_t = rng.uniform(-0.04, 0.04, (n, n_comp))
    a_t = prob["amps"][None] * rng.uniform(0.85, 1.15, (n, n_comp))
    Y = np.zeros((n, len(E)), dtype=np.float64)
    for k in range(n_comp):
        zc = ((E[None] - (prob["centers"][k] + dE_t[:, k:k + 1]))
              + 1j * prob["gamma"])
        zz = zc / ((prob["sigma"] + ds_t[:, k:k + 1]) * np.sqrt(2))
        Y += a_t[:, k:k + 1] * np.real(wofz(zz)) / (
            (prob["sigma"] + ds_t[:, k:k + 1]) * np.sqrt(2 * np.pi))
    if prob["bg_degree"] >= 0:
        x = (E - E.mean()) / (np.ptp(E) / 2)
        Y += 120.0 + 40.0 * x
    Y += noise * Y.max() * rng.standard_normal(Y.shape)
    return Y.astype(np.float32), (dE_t, ds_t, a_t)


def rmse_vs_truth(dE, ds, truth):
    dE_t, ds_t, _ = truth
    return (float(np.sqrt(np.mean((dE - dE_t) ** 2))),
            float(np.sqrt(np.mean((ds - ds_t) ** 2))))


# ---------------------------------------------------------------------------
# solver comparison
# ---------------------------------------------------------------------------


def bench_case(lines, label, prob, batch, noise=0.01, seed=0, quick=False):
    dicts = build_multipeak_dictionaries(prob["config"])
    Y, truth = gen_batch(prob, batch, noise=noise, seed=seed)
    bg = prob["bg_degree"]

    res_ap = solve_alternating_projection(
        Y, dicts, n_iterations=3, apply_newton=False, bg_degree=bg,
    )
    seed_args = (Y, dicts, res_ap.best_indices, res_ap.delta_E,
                 res_ap.delta_sigma, res_ap.amplitudes)

    def row(name, t_newton, dE, ds, chi2, extra=""):
        cr, sr = rmse_vs_truth(dE, ds, truth)
        lines.append(
            f"| {label} | {name} | {1e3 * t_newton:8.1f} | "
            f"{batch / max(t_newton, 1e-9) / 1e6:6.2f} | "
            f"{np.median(chi2):9.3g} | {cr:.2e} | {sr:.2e} | {extra} |")
        print(lines[-1])

    cr0, sr0 = rmse_vs_truth(res_ap.delta_E, res_ap.delta_sigma, truth)
    lines.append(f"| {label} | AP-seed | - | - | "
                 f"{np.median(res_ap.chi2):9.3g} | {cr0:.2e} | {sr0:.2e} | |")
    print(lines[-1])

    # raw (MLX kernel through the public API; newton timing only)
    if HAS_MLX and bg < 0:
        res_raw = solve_alternating_projection(
            Y, dicts, n_iterations=3, apply_newton=True, newton_iter=2,
        )
        row("raw-MLX", res_raw.timing["newton"], res_raw.delta_E,
            res_raw.delta_sigma, res_raw.chi2)

    if bg < 0:
        t0 = time.perf_counter()
        a, dE, ds, chi2 = _post_ap_newton_refine_numpy(*seed_args)
        a, dE, ds, chi2 = _post_ap_newton_refine_numpy(
            Y, dicts, res_ap.best_indices, dE, ds, a)
        row("raw-CPU x2", time.perf_counter() - t0, dE, ds, chi2)

        t0 = time.perf_counter()
        a, dE, ds, chi2 = _post_ap_newton_refine_numpy(
            *seed_args, jacobian_mode="golub_pereyra")
        a, dE, ds, chi2 = _post_ap_newton_refine_numpy(
            Y, dicts, res_ap.best_indices, dE, ds, a,
            jacobian_mode="golub_pereyra")
        row("gp-CPU x2", time.perf_counter() - t0, dE, ds, chi2)

    from toyomacro.voigtfit.multipeak_solver import build_bg_design
    bg_design = (build_bg_design(prob["energy"], bg) if bg >= 0 else None)
    for name, use_mlx in (("gplm-CPU", False), ("gplm-MLX", True)):
        if use_mlx and not HAS_MLX:
            continue
        _post_ap_gp_lm_refine_numpy(  # warmup (MLX graph/dict upload)
            Y[:256], dicts, res_ap.best_indices[:256], res_ap.delta_E[:256],
            res_ap.delta_sigma[:256], res_ap.amplitudes[:256],
            res_ap.chi2[:256], n_iter=1, bg_design=bg_design, use_mlx=use_mlx)
        t0 = time.perf_counter()
        a, dE, ds, chi2, _b, diag = _post_ap_gp_lm_refine_numpy(
            *seed_args, res_ap.chi2, n_iter=2, bg_design=bg_design,
            use_mlx=use_mlx)
        row(name, time.perf_counter() - t0, dE, ds, chi2,
            extra=f"acc {100 * diag.step_accepted.mean():.0f}% "
                  f"evals {diag.n_basis_evals}")

    # Phase-7 judgment: float64 exact-basis GP-LM (gp_reference), subset
    n_sub = 200 if quick else 1000
    if batch >= n_sub:
        sub = slice(0, n_sub)
        fixed_cols = (bg_design.astype(np.float64) if bg_design is not None
                      else None)
        model = SeparableVoigtModel(
            energy=prob["energy"].astype(np.float64), gamma=prob["gamma"],
            fixed_columns=fixed_cols)
        dE_e = res_ap.delta_E[sub].astype(np.float64).copy()
        ds_e = res_ap.delta_sigma[sub].astype(np.float64).copy()
        n_comp = len(prob["centers"])
        t0 = time.perf_counter()
        for i in range(n_sub):
            th0 = np.empty(2 * n_comp)
            th0[0::2] = prob["centers"] + dE_e[i]
            th0[1::2] = prob["sigma"] + ds_e[i]
            tr = run_newton(model, th0, Y[sub][i].astype(np.float64),
                            "golub_pereyra", method="lm", max_iter=4)
            dE_e[i] = tr.theta_final[0::2] - prob["centers"]
            ds_e[i] = tr.theta_final[1::2] - prob["sigma"]
        t_exact = time.perf_counter() - t0
        truth_sub = (truth[0][sub], truth[1][sub], truth[2][sub])
        cr, sr = rmse_vs_truth(dE_e, ds_e, truth_sub)
        lines.append(
            f"| {label} | gplm-exact* | {1e3 * t_exact:8.1f} | "
            f"{n_sub / t_exact / 1e6:6.2f} | (n={n_sub}) | {cr:.2e} "
            f"| {sr:.2e} | float64 ceiling |")
        print(lines[-1])


# ---------------------------------------------------------------------------
# §7.2 assembly micro-benchmark: stacked matmul vs entry-wise K = SᵀS
# ---------------------------------------------------------------------------


def bench_assembly(lines, batch=50_000, n_E=121, quick=False):
    if quick:
        batch = 10_000
    rng = np.random.default_rng(0)
    lines.append("\n### Assembly: stacked matmul vs entry-wise (K = SᵀS)\n")
    lines.append("| p | backend | stacked ms | entrywise ms |")
    lines.append("|---|---|---|---|")
    for p in (4, 10, 20):
        cols = [rng.standard_normal((batch, n_E)).astype(np.float32)
                for _ in range(p)]
        t0 = time.perf_counter()
        S = np.stack(cols, axis=2)
        K = S.transpose(0, 2, 1) @ S
        t_stack = time.perf_counter() - t0
        t0 = time.perf_counter()
        K2 = np.empty((batch, p, p), dtype=np.float32)
        for i in range(p):
            K2[:, i, i] = np.sum(cols[i] * cols[i], axis=1)
            for j in range(i + 1, p):
                v = np.sum(cols[i] * cols[j], axis=1)
                K2[:, i, j] = v
                K2[:, j, i] = v
        t_entry = time.perf_counter() - t0
        assert np.allclose(K, K2, rtol=1e-4, atol=1e-2)
        lines.append(f"| {p} | numpy | {1e3 * t_stack:.1f} | "
                     f"{1e3 * t_entry:.1f} |")
        print(lines[-1])
        if HAS_MLX:
            import mlx.core as mx
            cols_mx = [mx.array(c) for c in cols]
            for _ in range(2):  # warmup then measure
                t0 = time.perf_counter()
                S_mx = mx.stack(cols_mx, axis=2)
                K_mx = mx.transpose(S_mx, (0, 2, 1)) @ S_mx
                mx.eval(K_mx)
                t_stack_mx = time.perf_counter() - t0
            for _ in range(2):
                t0 = time.perf_counter()
                ent = []
                for i in range(p):
                    for j in range(i, p):
                        ent.append(mx.sum(cols_mx[i] * cols_mx[j], axis=1))
                mx.eval(*ent)
                t_entry_mx = time.perf_counter() - t0
            lines.append(f"| {p} | mlx | {1e3 * t_stack_mx:.1f} | "
                         f"{1e3 * t_entry_mx:.1f} |")
            print(lines[-1])


# ---------------------------------------------------------------------------
# §12 auto-routing experiment
# ---------------------------------------------------------------------------


def bench_auto_routing(lines, quick=False):
    """raw everywhere → GP-LM only on residual-routed, condition-gated
    spectra. Mixed batch: identifiable mid-overlap + hard strong-overlap."""
    n = 2_000 if quick else 20_000
    lines.append("\n### Auto-routing experiment (mixed batch, "
                 f"n={n} per population)\n")
    prob_mid = make_problem(2, 2.0)
    prob_hard = make_problem(2, 0.8)
    rows = {}
    for tag, prob in (("mid", prob_mid), ("hard", prob_hard)):
        dicts = build_multipeak_dictionaries(prob["config"])
        Y, truth = gen_batch(prob, n, seed=3)
        res_raw = solve_alternating_projection(
            Y, dicts, n_iterations=3, apply_newton=True, newton_iter=2,
        )
        # routing signal: residual well above the batch noise floor
        thresh = np.median(res_raw.chi2) * 1.5
        route = res_raw.chi2 > thresh
        cfg = GPLMConfig(cond_gate=1e8)
        amp2 = res_raw.amplitudes.copy()
        dE2 = res_raw.delta_E.copy()
        ds2 = res_raw.delta_sigma.copy()
        chi2_2 = res_raw.chi2.copy()
        n_routed = int(route.sum())
        acc_frac = 0.0
        if n_routed:
            a_r, dE_r, ds_r, c_r, _b, diag = _post_ap_gp_lm_refine_numpy(
                Y[route], dicts, res_raw.best_indices[route],
                res_raw.delta_E[route], res_raw.delta_sigma[route],
                res_raw.amplitudes[route], res_raw.chi2[route],
                n_iter=2, config=cfg, use_mlx=HAS_MLX,
            )
            amp2[route], dE2[route], ds2[route], chi2_2[route] = (
                a_r, dE_r, ds_r, c_r)
            acc_frac = float(diag.step_accepted.mean())
        # GP-LM everywhere for comparison
        a_g, dE_g, ds_g, c_g, _b, diag_all = _post_ap_gp_lm_refine_numpy(
            Y, dicts, res_raw.best_indices, res_raw.delta_E,
            res_raw.delta_sigma, res_raw.amplitudes, res_raw.chi2,
            n_iter=2, config=cfg, use_mlx=HAS_MLX,
        )
        rows[tag] = [
            ("raw only", *rmse_vs_truth(res_raw.delta_E,
                                        res_raw.delta_sigma, truth),
             np.median(res_raw.chi2), "-"),
            ("auto (route+gate)", *rmse_vs_truth(dE2, ds2, truth),
             np.median(chi2_2),
             f"routed {100 * n_routed / n:.0f}%, acc {100 * acc_frac:.0f}%"),
            ("gp_lm all", *rmse_vs_truth(dE_g, ds_g, truth),
             np.median(c_g),
             f"acc {100 * diag_all.step_accepted.mean():.0f}%"),
        ]
    lines.append("| pop | strategy | cRMSE(eV) | sRMSE(eV) | med chi2 | note |")
    lines.append("|---|---|---|---|---|---|")
    for tag, rr in rows.items():
        for name, cr, sr, c2, note in rr:
            lines.append(f"| {tag} | {name} | {cr:.2e} | {sr:.2e} "
                         f"| {c2:.3g} | {note} |")
            print(lines[-1])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--big", action="store_true", help="add a 1M-batch case")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines = ["# GP-LM hardening benchmark\n",
             f"MLX available: {HAS_MLX}. newton_iter=2 for all variants; "
             "gplm rows show accept rate and total basis evaluations. "
             "*gplm-exact = per-spectrum float64 QR reference (quality "
             "ceiling, not a production path).\n"]
    lines.append("| case | solver | newton ms | M/s | med chi2 "
                 "| cRMSE(eV) | sRMSE(eV) | note |")
    lines.append("|---|---|---|---|---|---|---|---|")

    batches = [10_000] if args.quick else [10_000, 100_000]
    if args.big:
        batches.append(1_000_000)
    cases = [
        ("2c-sep4", make_problem(2, 4.0)),
        ("2c-mid2", make_problem(2, 2.0)),
        ("2c-strong1", make_problem(2, 1.0)),
        ("2c-small-amp", make_problem(2, 3.0, small_amp=True)),
        ("2c-bg1", make_problem(2, 3.0, bg_degree=1)),
        ("5c-1.5", make_problem(5, 1.5)),
    ]
    for batch in batches:
        for label, prob in cases:
            bench_case(lines, f"{label}/{batch // 1000}K", prob, batch,
                       quick=args.quick)

    bench_assembly(lines, quick=args.quick)
    bench_auto_routing(lines, quick=args.quick)

    text = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
