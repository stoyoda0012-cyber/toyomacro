"""Gate 4 synthetic benchmark for rank / model selection (Phase 4).

Runs the frozen evaluation grid of README_rank_model_selection.md §3 and
scores the acceptance criteria §4.3/§4.4 (S1, S2, S3, S4, P1, P2). S5 is
covered deterministically in tests/test_exact_k.py (truncated-k_max
residual structure); the grid records residual flags as well.

Modes (CI never runs the full product):

    python -m toyomacro.voigtfit.benchmarks.bench_rank_model_selection --mode smoke
    ... --mode full        # frozen full grid, on demand only (~30 min)
    ... --mode slice3b     # Shirley/Tougaard plug-in slice, matched vs
                           # mismatched background generation (plan 3b)
    ... --mode sliceso     # S3: structured doublet vs two free peaks

Aggregates are written to --out (JSON, committed record); per-cell rows
go alongside as .jsonl unless --no-rows.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..exact_k import ExactKConfig, run_exact_k
from ..rank_diagnostics import voigt_fwhm_approx, voigt_profile

SIGMA_TRUE = 0.40
GAMMA = 0.10
FWHM = voigt_fwhm_approx(SIGMA_TRUE, GAMMA)
ENERGY = np.linspace(0.0, 24.0, 481)
SO_SPLIT = 1.8
SO_BRANCH = 1.5
AMPLITUDES = [2.0, 1.5, 1.8, 1.3, 1.7]

FULL_GRID = {
    "true_k": [1, 2, 3, 5],
    "spacing_fwhm": [0.25, 0.5, 1.0, 2.0],
    "snr": [10, 30, 100, 1000],
    "bg": ["none", "const", "linear"],
    "so": [False, True],
    "seed": [0, 1, 2, 3, 4],
}

SMOKE_GRID = {
    "true_k": [1, 2],
    "spacing_fwhm": [0.5, 2.0],
    "snr": [30, 1000],
    "bg": ["none", "linear"],
    "so": [False],
    "seed": [0],
}


@dataclass(frozen=True)
class Cell:
    true_k: int
    spacing_fwhm: float
    snr: float
    bg: str
    so: bool
    seed: int


def _clean_signal(cell: Cell) -> tuple[np.ndarray, np.ndarray]:
    """(clean peak signal, centers). SO doubles the visible lines but the
    structured component count stays true_k."""
    span = (cell.true_k - 1) * cell.spacing_fwhm * FWHM
    c0 = 0.5 * (ENERGY[0] + ENERGY[-1]) - 0.5 * span - (SO_SPLIT / 2 if cell.so else 0.0)
    centers = c0 + np.arange(cell.true_k) * cell.spacing_fwhm * FWHM
    y = np.zeros_like(ENERGY)
    for c, a in zip(centers, AMPLITUDES[: cell.true_k]):
        y = y + a * voigt_profile(ENERGY, c, SIGMA_TRUE, GAMMA)
        if cell.so:
            y = y + (a / SO_BRANCH) * voigt_profile(
                ENERGY, c + SO_SPLIT, SIGMA_TRUE, GAMMA
            )
    return y, centers


def make_cell_data(cell: Cell) -> tuple[np.ndarray, float]:
    """(noisy spectrum, noise_std). SNR = clean peak max / noise_std."""
    y0, _ = _clean_signal(cell)
    ymax = float(np.max(y0))
    noise_std = ymax / cell.snr
    if cell.bg == "const":
        y0 = y0 + 0.3 * ymax
    elif cell.bg == "linear":
        y0 = y0 + 0.3 * ymax + 0.02 * ymax * (ENERGY - ENERGY.mean())
    rng = np.random.default_rng(cell.seed + 1000 * cell.true_k)
    return y0 + rng.normal(0.0, noise_std, ENERGY.shape), noise_std


def run_cell(cell: Cell, k_max: int = 5, n_starts: int = 3) -> dict[str, Any]:
    y, noise_std = make_cell_data(cell)
    cfg = ExactKConfig(
        sigma_init=0.45, gamma=GAMMA, k_max=k_max,
        bg_degree={"none": None, "const": 0, "linear": 1}[cell.bg],
        so_split=SO_SPLIT if cell.so else 0.0,
        branch_ratio=SO_BRANCH if cell.so else 1.0,
        n_starts=n_starts,
    )
    t0 = time.perf_counter()
    r = run_exact_k(ENERGY, y, cfg, sigma_std=noise_std)
    elapsed = time.perf_counter() - t0
    supported = list(r.supported_k)
    return {
        **asdict(cell),
        "verdict": r.verdict,
        "supported_k": supported,
        "best_aicc": r.selection.best_by["aicc"],
        "best_bic": r.selection.best_by["bic"],
        "true_in_supported": cell.true_k in supported,
        "overfit_supported_claim": bool(
            r.verdict == "supported" and supported and max(supported) > cell.true_k
        ),
        "underfit_supported_claim": bool(
            r.verdict == "supported" and supported and max(supported) < cell.true_k
        ),
        "supported_equals_truth": bool(supported == [cell.true_k]),
        "ic_criteria_disagree": bool(
            r.selection.best_by["aicc"] != r.selection.best_by["bic"]
        ),
        "rank_deficit_at_ic_best": bool(
            next(
                (
                    not e["rank_supported"]
                    for e in r.evidence
                    if e["k"] == r.selection.best_by["bic"]
                ),
                False,
            )
        ),
        "negative_amplitude_k": list(r.negative_amplitude_k),
        "residual_structured_at_best": bool(
            next(
                (
                    e.get("residual_structured", False)
                    for e in r.evidence
                    if e["k"] == (supported[0] if supported else None)
                ),
                False,
            )
        ),
        "elapsed_s": elapsed,
    }


def _cells(grid: dict[str, list]) -> list[Cell]:
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cell = Cell(**dict(zip(keys, combo)))
        if cell.true_k == 1 and cell.spacing_fwhm != grid["spacing_fwhm"][0]:
            continue  # spacing is meaningless for K=1: keep one variant
        out.append(cell)
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(sel, pred):
        subset = [r for r in rows if sel(r)]
        return {
            "n": len(subset),
            "rate": (
                float(np.mean([pred(r) for r in subset])) if subset else None
            ),
        }

    s1 = rate(
        lambda r: r["spacing_fwhm"] >= 1.0 and r["snr"] >= 100 and r["true_k"] > 1,
        lambda r: r["true_in_supported"],
    )
    s1_incl_k1 = rate(
        lambda r: (r["spacing_fwhm"] >= 1.0 or r["true_k"] == 1) and r["snr"] >= 100,
        lambda r: r["true_in_supported"],
    )
    s2 = rate(
        lambda r: r["spacing_fwhm"] == 0.25 and r["true_k"] > 1,
        lambda r: not r["overfit_supported_claim"],
    )
    s2_verdicts: dict[str, int] = {}
    for r in rows:
        if r["spacing_fwhm"] == 0.25 and r["true_k"] > 1:
            s2_verdicts[r["verdict"]] = s2_verdicts.get(r["verdict"], 0) + 1
    # Frozen S4, measured DIRECTLY: among cells where the IC layer and
    # the rank layer disagree (criteria name different best K, or the
    # IC-best K lacks nominal-dof rank support), the report must not
    # assert a single K (verdict != "supported").
    def _s4_disagreement(r):
        return r["ic_criteria_disagree"] or r["rank_deficit_at_ic_best"]

    s4_direct = rate(
        _s4_disagreement, lambda r: r["verdict"] != "supported"
    )
    # Confident-overfit rate is a DIFFERENT indicator (not frozen S4):
    # a "supported" claim with more structure than the truth.
    no_overfit = rate(lambda r: True, lambda r: not r["overfit_supported_claim"])
    underfit = rate(
        lambda r: r["verdict"] == "supported",
        lambda r: r["underfit_supported_claim"],
    )
    # Gate 4 adjudication: S1 stays as frozen (miss preserved); the
    # spacing x SO breakdown is reported alongside because "is 1 FWHM
    # well-separated?" is the dominant factor, not the population.
    s1_breakdown = {}
    for spacing in sorted({r["spacing_fwhm"] for r in rows}):
        for so in (False, True):
            cellsel = rate(
                lambda r, s=spacing, o=so: (
                    r["spacing_fwhm"] == s and r["so"] == o
                    and r["snr"] >= 100 and r["true_k"] > 1
                ),
                lambda r: r["true_in_supported"],
            )
            if cellsel["n"]:
                s1_breakdown[f"spacing={spacing}FWHM,so={so}"] = cellsel

    times = [r["elapsed_s"] for r in rows]
    return {
        "n_cells": len(rows),
        "S1_true_k_in_supported_at_sep>=1FWHM_snr>=100": s1,
        "S1_including_k1": s1_incl_k1,
        "S1_breakdown_by_spacing_and_so_at_snr>=100": s1_breakdown,
        "S2_no_overfit_supported_claim_at_0.25FWHM": s2,
        "S2_verdict_distribution_at_0.25FWHM": s2_verdicts,
        "S4_no_single_K_assertion_under_IC_rank_disagreement": s4_direct,
        "no_confident_overfit_claim_rate": no_overfit,
        "underfit_supported_claims_among_supported": underfit,
        "verdict_distribution": {
            v: sum(1 for r in rows if r["verdict"] == v)
            for v in ("supported", "ambiguous", "unsupported")
        },
        "negative_amplitude_cells": sum(
            1 for r in rows if r["negative_amplitude_k"]
        ),
        "wall_time_s": {
            "total": float(np.sum(times)),
            "per_cell_median": float(np.median(times)),
            "per_cell_max": float(np.max(times)),
        },
    }


# --------------------------------------------------------- S3 (SO slice)


def run_slice_so(seeds=range(5), snrs=(30, 100)) -> dict[str, Any]:
    """S3: data from a fixed SO doublet; the structured 1-component model
    must not be disfavored vs two independent peaks."""
    rows = []
    for snr, seed in itertools.product(snrs, seeds):
        cell = Cell(true_k=1, spacing_fwhm=1.0, snr=snr, bg="none", so=True, seed=seed)
        y, noise_std = make_cell_data(cell)
        structured = run_exact_k(
            ENERGY, y,
            ExactKConfig(
                sigma_init=0.45, gamma=GAMMA, k_max=1,
                so_split=SO_SPLIT, branch_ratio=SO_BRANCH, n_starts=3,
            ),
            sigma_std=noise_std,
        ).candidates[0]
        free = run_exact_k(
            ENERGY, y,
            ExactKConfig(sigma_init=0.45, gamma=GAMMA, k_max=2, n_starts=3),
            sigma_std=noise_std,
        ).candidates[1]
        rows.append(
            {
                "snr": snr,
                "seed": seed,
                "aicc_structured": structured.score.aicc,
                "aicc_free2": free.score.aicc,
                "bic_structured": structured.score.bic,
                "bic_free2": free.score.bic,
                "structured_not_disfavored": bool(
                    structured.score.aicc <= free.score.aicc + 2.0
                    and structured.score.bic <= free.score.bic + 2.0
                ),
            }
        )
    return {
        "n": len(rows),
        "S3_structured_not_disfavored_rate": float(
            np.mean([r["structured_not_disfavored"] for r in rows])
        ),
        "rows": rows,
    }


# --------------------------------------------------------- 3b slice


def _shirley_like_step(y0: np.ndarray, height: float) -> np.ndarray:
    tail = np.cumsum(y0[::-1])[::-1]
    return height * tail / tail[0]


def run_slice_3b(seeds=range(3)) -> dict[str, Any]:
    """Plan 3b slice: K={1,2,3} x spacing {0.5,1.0} FWHM x SNR {30,100},
    per background estimator, over NOMINAL generated backgrounds
    (shirley-like tail-cumsum vs linear ramp) — reporting the K bias a
    background-model error induces. Self-consistent matched generation
    (for either estimator) is deliberately not claimed or evaluated."""
    rows = []
    for true_k, spacing, snr, gen_bg, plugin, seed in itertools.product(
        [1, 2, 3], [0.5, 1.0], [30, 100],
        ["shirley_like", "linear"], ["shirley", "tougaard"], seeds,
    ):
        cell = Cell(
            true_k=true_k, spacing_fwhm=spacing, snr=snr,
            bg="none", so=False, seed=seed,
        )
        y0, _ = _clean_signal(cell)
        ymax = float(np.max(y0))
        noise_std = ymax / snr
        if gen_bg == "shirley_like":
            bg_true = _shirley_like_step(y0, 0.3 * ymax)
        else:
            bg_true = 0.3 * ymax + 0.02 * ymax * (ENERGY - ENERGY.mean())
        rng = np.random.default_rng(seed + 1000 * true_k)
        y = y0 + bg_true + rng.normal(0.0, noise_std, ENERGY.shape)
        r = run_exact_k(
            ENERGY, y,
            ExactKConfig(
                sigma_init=0.45, gamma=GAMMA, k_max=4, n_starts=3,
                plugin_background=plugin,
            ),
            sigma_std=noise_std,
        )
        rows.append(
            {
                "true_k": true_k, "spacing_fwhm": spacing, "snr": snr,
                "gen_bg": gen_bg, "plugin": plugin, "seed": seed,
                "nominal_match": (gen_bg == "shirley_like" and plugin == "shirley"),
                "verdict": r.verdict,
                "supported_k": list(r.supported_k),
                "best_bic": r.selection.best_by["bic"],
                "k_bias_bic": (
                    (r.selection.best_by["bic"] - true_k)
                    if r.selection.best_by["bic"] is not None else None
                ),
                "true_in_supported": true_k in r.supported_k,
                "overfit_supported_claim": bool(
                    r.verdict == "supported"
                    and r.supported_k and max(r.supported_k) > true_k
                ),
            }
        )

    def agg(sel):
        sub = [r for r in rows if sel(r)]
        biases = [r["k_bias_bic"] for r in sub if r["k_bias_bic"] is not None]
        return {
            "n": len(sub),
            "true_in_supported_rate": float(np.mean([r["true_in_supported"] for r in sub])),
            "mean_k_bias_bic": float(np.mean(biases)) if biases else None,
            "overfit_supported_claims": sum(r["overfit_supported_claim"] for r in sub),
        }

    return {
        "note": (
            "generated backgrounds are NOMINAL shapes: 'shirley_like' is "
            "the tail-cumsum of the clean peaks, NOT the self-consistent "
            "fixed point of the Shirley estimator; truly matched Shirley "
            "and matched Tougaard generation are unevaluated"
        ),
        "shirley_on_nominal_shirley_like": agg(lambda r: r["nominal_match"]),
        "shirley_on_linear": agg(
            lambda r: r["plugin"] == "shirley" and r["gen_bg"] == "linear"
        ),
        "tougaard_on_nominal_shirley_like": agg(
            lambda r: r["plugin"] == "tougaard" and r["gen_bg"] == "shirley_like"
        ),
        "tougaard_on_linear": agg(
            lambda r: r["plugin"] == "tougaard" and r["gen_bg"] == "linear"
        ),
        "rows": rows,
    }


# --------------------------------------------------------- entry point


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["smoke", "full", "slice3b", "sliceso"], default="smoke")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-rows", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.mode in ("smoke", "full"):
        grid = SMOKE_GRID if args.mode == "smoke" else FULL_GRID
        cells = _cells(grid)
        rows = []
        for i, cell in enumerate(cells):
            rows.append(run_cell(cell))
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(cells)} cells", flush=True)
        result: dict[str, Any] = {"mode": args.mode, "aggregate": aggregate(rows)}
        row_payload = rows
    elif args.mode == "sliceso":
        result = {"mode": args.mode, "aggregate": run_slice_so()}
        row_payload = result["aggregate"].pop("rows")
    else:
        result = {"mode": args.mode, "aggregate": run_slice_3b()}
        row_payload = result["aggregate"].pop("rows")

    result["wall_s"] = time.perf_counter() - t0
    print(json.dumps(result, indent=2))

    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        if not args.no_rows:
            rows_path = args.out.with_suffix(".rows.jsonl")
            with rows_path.open("w") as fh:
                for row in row_payload:
                    fh.write(json.dumps(row) + "\n")
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
