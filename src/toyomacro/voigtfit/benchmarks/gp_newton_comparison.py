"""
Raw vs Kaufman vs Golub-Pereyra Newton comparison (reference path).
===================================================================

Experiment harness for the GP Jacobian verification study. Runs the
float64 reference Gauss-Newton (`gp_reference.run_newton`) on synthetic
separable Voigt problems under identical initial values and an identical
backtracking safeguard, and tabulates convergence / accuracy / cost per
Jacobian mode.

This deliberately measures the CONTINUOUS post-dictionary problem: no
dictionary argmax, no grid, no step clipping. "Cell" units below translate
initial errors into the production dictionary's scale (cell_dE = 0.1 sigma,
cell_dsigma = 0.05 sigma).

Usage:
    python -m toyomacro.voigtfit.benchmarks.gp_newton_comparison [--quick]
        [--out report.md]

Blocks:
    A: convergence vs initial error x peak overlap (noise 1%)
    B: noise sweep (incl. Poisson) at 2-cell initial error
    C: amplitude scenarios (equal / one-small / one-zero / extra component)
    D: no-backtracking stability (native GN step quality)
    F: production-like clipped 1/3-step polish
    G: iteration-budget ablation (raw's low variance: curvature vs budget)
    E: per-iteration cost microbenchmark

Metrics per aggregate row: success rate, median iterations (successes),
median final center/sigma RMSE (eV, true components only), median relative
amplitude RMSE, negative-amplitude rate at the solution, median cond(Gram),
objective-increase count, median wall time per run.

Success criterion: final SSR <= 1.05 x SSR(theta_true with LLS amplitudes)
+ (1e-6 * max|y|)^2 * n_rows (absolute guard for noiseless cases).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace

import numpy as np

from toyomacro.voigtfit.gp_reference import (
    JACOBIAN_MODES,
    SeparableVoigtModel,
    reduced_jacobian,
    reduced_residual,
    run_newton,
)

CELL_DE_SIGMA = 0.1    # dictionary delta-E cell, in units of sigma
CELL_DS_SIGMA = 0.05   # dictionary delta-sigma cell, in units of sigma


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    name: str
    centers: np.ndarray
    sigmas: np.ndarray
    amps: np.ndarray
    gamma: float = 0.25
    so_split: np.ndarray | None = None
    partner_ratio: np.ndarray | None = None
    n_spurious: int = 0   # extra fitted components beyond the true ones

    @property
    def n_true(self) -> int:
        return len(self.centers)


def ladder(n, spacing_sigma, sigma=0.6, amp0=50_000.0):
    """n peaks spaced `spacing_sigma`*sigma apart, alternating amplitudes."""
    centers = 100.0 + sigma * spacing_sigma * np.arange(n)
    sigmas = np.full(n, sigma)
    amps = amp0 * (1.0 - 0.4 * (np.arange(n) % 2))
    return centers, sigmas, amps


def make_problem(name: str) -> Problem:
    sigma = 0.6
    if name == "1comp":
        return Problem(name, *ladder(1, 0.0, sigma))
    if name.startswith("2comp-"):
        sep = {"sep": 4.0, "mid": 2.0, "strong": 1.0, "degen": 0.35}[
            name.split("-")[1]
        ]
        return Problem(name, *ladder(2, sep, sigma))
    if name == "5comp":
        return Problem(name, *ladder(5, 1.5, sigma))
    if name == "10comp":
        return Problem(name, *ladder(10, 1.2, sigma))
    if name == "doublet2":
        c, s, a = ladder(2, 3.0, sigma)
        return Problem(
            name, c, s, a,
            so_split=np.array([0.61, 0.61]),
            partner_ratio=np.array([0.75, 0.75]),
        )
    raise ValueError(name)


def build_model_and_truth(prob: Problem):
    lo = prob.centers.min() - 4.0
    hi = prob.centers.max() + 4.0
    n_E = max(201, int((hi - lo) / 0.05) + 1)
    energy = np.linspace(lo, hi, n_E)

    n_fit = prob.n_true + prob.n_spurious
    so_split = partner_ratio = None
    if prob.so_split is not None:
        so_split = np.concatenate([prob.so_split, np.zeros(prob.n_spurious)])
        partner_ratio = np.concatenate(
            [prob.partner_ratio, np.zeros(prob.n_spurious)],
        )
    model = SeparableVoigtModel(
        energy=energy, gamma=prob.gamma,
        so_split=so_split, partner_ratio=partner_ratio,
    )

    # Fitted-model truth: true peaks (+ spurious ones parked mid-spectrum
    # with zero true amplitude).
    theta_true = np.empty(2 * n_fit)
    theta_true[0 : 2 * prob.n_true : 2] = prob.centers
    theta_true[1 : 2 * prob.n_true : 2] = prob.sigmas
    for j in range(prob.n_spurious):
        theta_true[2 * (prob.n_true + j)] = prob.centers.mean() + 0.8 * (j + 1)
        theta_true[2 * (prob.n_true + j) + 1] = prob.sigmas.mean()

    # Clean data uses ONLY the true components.
    Phi, _ = model.basis_and_derivs(theta_true)
    y_clean = Phi[:, : prob.n_true] @ prob.amps
    return model, theta_true, y_clean


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def make_noisy(y_clean, noise_kind, noise_scale, rng):
    if noise_kind == "none":
        return y_clean
    if noise_kind == "gauss":
        return y_clean + noise_scale * y_clean.max() * rng.standard_normal(
            len(y_clean),
        )
    if noise_kind == "poisson":
        return rng.poisson(np.maximum(y_clean, 0.0)).astype(np.float64)
    raise ValueError(noise_kind)


def perturb_theta(theta_true, sigmas_fit, n_cells, rng):
    """Initial guess: +-n_cells dictionary cells per parameter, random sign,
    75-100% of the nominal magnitude."""
    theta0 = theta_true.copy()
    n = len(theta0) // 2
    for k in range(n):
        s = sigmas_fit[k]
        mag = 0.75 + 0.25 * rng.random(2)
        sign = rng.choice([-1.0, 1.0], size=2)
        theta0[2 * k] += sign[0] * mag[0] * n_cells * CELL_DE_SIGMA * s
        theta0[2 * k + 1] += sign[1] * mag[1] * n_cells * CELL_DS_SIGMA * s
    return theta0


def matched_errors(theta_f, a_f, prob: Problem):
    """Component-permutation-safe errors: fitted components are assigned to
    true components by nearest center (Hungarian), so a label swap during
    optimization does not masquerade as a huge parameter error. Runaway
    components (amplitude ~ 0, center far away) still show up honestly."""
    from scipy.optimize import linear_sum_assignment

    n_fit = len(theta_f) // 2
    c_fit, s_fit = theta_f[0::2], theta_f[1::2]
    cost = np.abs(prob.centers[:, None] - c_fit[None, :])
    row, col = linear_sum_assignment(cost)
    c_rmse = float(np.sqrt(np.mean((c_fit[col] - prob.centers[row]) ** 2)))
    s_rmse = float(np.sqrt(np.mean((s_fit[col] - prob.sigmas[row]) ** 2)))
    a_matched = a_f[:n_fit][col]
    denom = prob.amps[row].copy()
    if np.any(denom == 0):
        denom[:] = prob.amps.max()
    a_rel = float(np.sqrt(np.mean(
        ((a_matched - prob.amps[row]) / denom) ** 2)))
    return c_rmse, s_rmse, a_rel


def run_trial(
    prob: Problem,
    n_cells: float,
    noise_kind: str,
    noise_scale: float,
    seed: int,
    mode: str,
    method: str = "lm",
    max_iter: int = 30,
    step_clip_cells: float | None = None,
):
    rng = np.random.default_rng(seed)
    model, theta_true, y_clean = build_model_and_truth(prob)
    y = make_noisy(y_clean, noise_kind, noise_scale, rng)

    sigmas_fit = theta_true[1::2]
    theta0 = perturb_theta(theta_true, sigmas_fit, n_cells, rng)

    step_clip = None
    if step_clip_cells is not None:
        # production-style clip: +-ratio of a dictionary cell per axis
        step_clip = np.empty_like(theta0)
        step_clip[0::2] = step_clip_cells * CELL_DE_SIGMA * sigmas_fit
        step_clip[1::2] = step_clip_cells * CELL_DS_SIGMA * sigmas_fit

    # Noise floor: SSR of the fitted model AT theta_true (LLS amplitudes).
    r_floor, _ = reduced_residual(model, theta_true, y)
    ssr_floor = float(r_floor @ r_floor)
    ssr_tol = 1.05 * ssr_floor + (1e-6 * np.abs(y).max()) ** 2 * len(y)

    t0 = time.perf_counter()
    tr = run_newton(
        model, theta0, y, mode,
        method=method, max_iter=max_iter, step_clip=step_clip,
    )
    wall = time.perf_counter() - t0

    ssr_final = tr.ssr[-1] if tr.ssr else tr.ssr0
    c_rmse, s_rmse, a_rel = matched_errors(tr.theta_final, tr.a_final, prob)

    return {
        "mode": mode,
        "success": bool(ssr_final <= ssr_tol),
        "n_iter": tr.n_iter,
        "c_rmse": c_rmse,
        "s_rmse": s_rmse,
        "a_rel": a_rel,
        "chi2": ssr_final / len(y),
        "neg_amp": bool(np.any(tr.a_final < 0)),
        "cond": float(np.median(tr.cond_gram)) if tr.cond_gram else np.nan,
        "n_increase": tr.n_increase,
        "diverged": bool(ssr_final > tr.ssr0),
        "stalled": tr.stalled,
        "wall": wall,
    }


# ---------------------------------------------------------------------------
# Aggregation / formatting
# ---------------------------------------------------------------------------


def aggregate(rows):
    out = {}
    n = len(rows)
    out["n"] = n
    out["success%"] = 100.0 * np.mean([r["success"] for r in rows])
    succ = [r for r in rows if r["success"]]
    out["iters"] = float(np.median([r["n_iter"] for r in succ])) if succ else np.nan
    for key in ("c_rmse", "s_rmse", "a_rel", "chi2", "cond", "wall"):
        pool = succ if succ and key in ("c_rmse", "s_rmse", "a_rel") else rows
        out[key] = float(np.median([r[key] for r in pool]))
    out["neg%"] = 100.0 * np.mean([r["neg_amp"] for r in rows])
    out["incr"] = float(np.mean([r["n_increase"] for r in rows]))
    out["div%"] = 100.0 * np.mean([r["diverged"] for r in rows])
    return out


HEADER = ("| case | mode | succ% | iters | cRMSE(eV) | sRMSE(eV) | ampRel "
          "| neg% | obj+ | div% | cond(G) | ms/run |")
RULE = "|---|---|---|---|---|---|---|---|---|---|---|---|"


def fmt_row(case, mode, agg):
    def g(x, f="{:.3g}"):
        return "-" if not np.isfinite(x) else f.format(x)
    return (
        f"| {case} | {mode} | {agg['success%']:.0f} | {g(agg['iters'], '{:.0f}')} "
        f"| {g(agg['c_rmse'], '{:.2e}')} | {g(agg['s_rmse'], '{:.2e}')} "
        f"| {g(agg['a_rel'], '{:.2e}')} | {agg['neg%']:.0f} | {agg['incr']:.2f} "
        f"| {agg['div%']:.0f} | {g(agg['cond'], '{:.1e}')} "
        f"| {1e3 * agg['wall']:.1f} |"
    )


def run_block(lines, title, configs, n_trials, method="lm", max_iter=30,
              step_clip_cells=None, seed0=0):
    lines.append(f"\n### {title}\n")
    lines.append(HEADER)
    lines.append(RULE)
    for label, prob, n_cells, noise_kind, noise_scale in configs:
        for mode in JACOBIAN_MODES:
            rows = [
                run_trial(
                    prob, n_cells, noise_kind, noise_scale,
                    seed=seed0 + 1000 * t, mode=mode,
                    method=method, max_iter=max_iter,
                    step_clip_cells=step_clip_cells,
                )
                for t in range(n_trials)
            ]
            lines.append(fmt_row(label, mode, aggregate(rows)))
        print("\n".join(lines[-3:]))


# ---------------------------------------------------------------------------
# Block G: iteration-budget ablation
# ---------------------------------------------------------------------------


def block_g(lines, n_trials, seed0=70):
    """Separate 'accurate descent along the chi2-flat valley' from safeguard
    artifacts: same LM safeguard, NO step clip, identical seeds — only the
    iteration budget varies. If GP's parameter RMSE grows with budget while
    its SSR keeps falling, the raw-vs-GP gap at strong overlap is dominated
    by how far each mode walks down the flat valley (estimator variance at
    the ML point), i.e. raw at practical budgets behaves like early stopping.
    A raw run at budget 200 probes whether raw eventually reaches the same
    valley floor (same destination, slower walk) or stays put (curvature
    blocks the valley direction entirely)."""
    lines.append("\n### G. Iteration-budget ablation "
                 "(LM, no clip, noise 1%, paired seeds)\n")
    lines.append(HEADER)
    lines.append(RULE)
    for pn, cells in (("2comp-strong", 0.5), ("2comp-degen", 1.0)):
        prob = make_problem(pn)
        for budget in (2, 5, 10, 20, 50, 200):
            for mode in ("raw", "golub_pereyra"):
                rows = [
                    run_trial(
                        prob, cells, "gauss", 1e-2,
                        seed=seed0 + 1000 * t, mode=mode,
                        method="lm", max_iter=budget,
                    )
                    for t in range(n_trials)
                ]
                lines.append(fmt_row(
                    f"{pn}/it{budget}", mode, aggregate(rows),
                ))
            print("\n".join(lines[-2:]))


# ---------------------------------------------------------------------------
# Block E: cost microbenchmark
# ---------------------------------------------------------------------------


def block_e(lines, n_rep=20):
    lines.append("\n### E. Per-call cost (reference implementation, "
                 "single spectrum, float64)\n")
    lines.append("| n_comp | mode | jacobian ms | vs raw |")
    lines.append("|---|---|---|---|")
    for pname in ("1comp", "2comp-mid", "5comp", "10comp"):
        prob = make_problem(pname)
        model, theta_true, y_clean = build_model_and_truth(prob)
        rng = np.random.default_rng(0)
        y = make_noisy(y_clean, "gauss", 1e-2, rng)
        theta = perturb_theta(theta_true, theta_true[1::2], 1.0, rng)
        base = None
        for mode in JACOBIAN_MODES:
            reduced_jacobian(model, theta, y, mode)  # warmup
            t0 = time.perf_counter()
            for _ in range(n_rep):
                reduced_jacobian(model, theta, y, mode)
            dt = (time.perf_counter() - t0) / n_rep * 1e3
            if mode == "raw":
                base = dt
            lines.append(
                f"| {prob.n_true} | {mode} | {dt:.3f} | {dt / base:.2f}x |",
            )
        print("\n".join(lines[-3:]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="5 trials per cell")
    ap.add_argument("--out", default=None, help="write markdown to this path")
    args = ap.parse_args()

    n_trials = 5 if args.quick else 20
    lines = ["# GP Newton comparison (reference path)\n",
             f"trials per cell: {n_trials}; blocks A-C use a COMMON "
             "Levenberg-Marquardt safeguard (max_iter=30); block D is native "
             "undamped GN; block F is the production-like clipped 1/3-step. "
             "success = SSR <= 1.05 x floor; RMSE medians over successes, "
             "component-matched by nearest center."]

    # Block A: initial error x overlap, 1% Gaussian noise
    probs_a = ["2comp-sep", "2comp-mid", "2comp-strong", "2comp-degen",
               "5comp", "doublet2"]
    cfg_a = []
    for pn in probs_a:
        prob = make_problem(pn)
        for cells in (0.5, 1.0, 2.0, 5.0, 10.0):
            cfg_a.append((f"{pn}/{cells:g}cell", prob, cells, "gauss", 1e-2))
    prob10 = make_problem("10comp")
    for cells in (1.0, 2.0, 5.0):
        cfg_a.append((f"10comp/{cells:g}cell", prob10, cells, "gauss", 1e-2))
    run_block(lines, "A. Initial error x overlap (noise 1%, LM)", cfg_a,
              n_trials, max_iter=50, seed0=10)

    # Block B: noise sweep at 2-cell initial error
    cfg_b = []
    for pn in ("2comp-mid", "5comp"):
        prob = make_problem(pn)
        for kind, scale in (("none", 0.0), ("gauss", 1e-3), ("gauss", 1e-2),
                            ("gauss", 5e-2), ("poisson", 0.0)):
            label = f"{pn}/{kind}{scale:g}" if kind == "gauss" else f"{pn}/{kind}"
            cfg_b.append((label, prob, 2.0, kind, scale))
    run_block(lines, "B. Noise sweep (2-cell init, LM)", cfg_b, n_trials,
              max_iter=50, seed0=20)

    # Block C: amplitude scenarios (2-cell init, 1% noise)
    cfg_c = []
    for pn in ("2comp-mid", "5comp"):
        base = make_problem(pn)
        small = replace(base, name=pn + "-small", amps=base.amps.copy())
        small.amps[-1] = 0.02 * small.amps.max()
        zero = replace(base, name=pn + "-zero", amps=base.amps.copy())
        zero.amps[-1] = 0.0
        extra = replace(base, name=pn + "-extra", n_spurious=1)
        for prob, tag in ((base, "equal"), (small, "one-small"),
                          (zero, "one-zero"), (extra, "extra-comp")):
            cfg_c.append((f"{pn}/{tag}", prob, 2.0, "gauss", 1e-2))
    run_block(lines, "C. Amplitude scenarios (2-cell init, noise 1%, LM)",
              cfg_c, n_trials, max_iter=50, seed0=30)

    # Block D: native undamped GN (full step always accepted)
    cfg_d = []
    for pn in ("2comp-strong", "5comp"):
        prob = make_problem(pn)
        for cells in (2.0, 5.0):
            cfg_d.append((f"{pn}/{cells:g}cell", prob, cells, "gauss", 1e-2))
    run_block(lines, "D. Native undamped GN (no safeguard)", cfg_d,
              n_trials, method="gn", seed0=40)

    # Block F: production-like polish — clipped undamped GN, 1 and 3 steps
    # (mirrors apply_newton with clip_step_ratio=0.5, newton_iter=1/3).
    for n_steps in (1, 3):
        cfg_f = []
        for pn in ("2comp-mid", "2comp-strong", "5comp", "doublet2"):
            prob = make_problem(pn)
            for cells in (0.5, 1.0):
                cfg_f.append(
                    (f"{pn}/{cells:g}cell", prob, cells, "gauss", 1e-3),
                )
        run_block(
            lines,
            f"F{n_steps}. Production-like polish: {n_steps} clipped GN "
            "step(s) (clip 0.5 cell, noise 0.1%)",
            cfg_f, n_trials, method="gn", max_iter=n_steps,
            step_clip_cells=0.5, seed0=50 + n_steps,
        )

    block_g(lines, n_trials)

    block_e(lines)

    text = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
