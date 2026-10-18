"""Third validation of ``_composition_uncertainty`` (private): the §8 harness.

Implements docs/design/composition-uncertainty.md §8 clause by clause:
the declared spectra D_kappa, ``_predict_r``, the count setting with its
one-sided window, the five seed series, the per-condition decision rule
and the execution-identity checks. Where this module and §8 disagree,
§8 is the registration and this module is wrong.

    python -m toyomacro._composition_validation3 run --kind pilot --out DIR
    python -m toyomacro._composition_validation3 run --kind decision --out DIR
    python -m toyomacro._composition_validation3 summarize --out DIR --record FILE

``DIR`` must lie outside the checkout (or under a path git ignores):
the run refuses to start from a checkout with a modified tracked file.
Runs are resumable: a chunk is written to a temporary name and renamed
when complete, and a resume whose configuration hash, commit or
library versions differ from the run's is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy

import toyomacro
from toyomacro._composition_validation import _PEAK, CONFIGS
from toyomacro.composition import (
    _PREDICTION_SEED_KEY,
    Conditions,
    Line,
    Matrix,
    _bootstrap_detail,
    _predict_r,
    _relative_area_noise,
    composition,
)

# ---------------------------------------------------------------------------
# Registered constants (§8, "Harness choices")
# ---------------------------------------------------------------------------

SEED = 20261014
CONFIG_NAMES = ("P", "B", "T", "G")
BACKGROUNDS = ("linear", "shirley")
LEVELS = (0.005, 0.01, 0.02, 0.04, 0.08)
RUN_KINDS = {"pilot": 0, "decision": 1}
RUN_SIZES = {"pilot": (200, 200), "decision": (2000, 1000)}  # (M, B)
CANDIDATES = (0.02, 0.01, 0.005)  # r* candidates, largest first; 0.08 and 0.04 are not
KAPPA = tuple((kp, kb) for kp in (0.5, 1.0, 2.0) for kb in (0.5, 1.0, 2.0))
B_EXPLORE = 200
B_PRED = 1000
B_DATA = 1000
N_RESAMPLE = 5000
RATE_BOUND = 0.001
AREA_1 = 1e4
MAX_STEPS = 8
WINDOW = (0.9, 1.0)  # r_pred in [0.9 x level, level]
AIM = 0.95  # the update aims at the middle of the window
GROW = 4.0
MIN_ELEMENTS, MAX_LINES = 2, 3
SHIRLEY = {"max_iter": 50, "tol": 1e-5, "auto_range": False}
GENERATOR = "numpy.random.Generator(numpy.random.PCG64(numpy.random.SeedSequence(key)))"
DESIGN = "docs/design/composition-uncertainty.md §8"

REGISTERED = {
    "seed": SEED, "configurations": CONFIG_NAMES, "backgrounds": BACKGROUNDS, "levels": LEVELS,
    "run_kinds": RUN_KINDS, "run_sizes": RUN_SIZES, "candidates": CANDIDATES, "kappa": KAPPA,
    "B_explore": B_EXPLORE, "B_pred": B_PRED, "B_data": B_DATA, "n_resample": N_RESAMPLE,
    "rate_bound": RATE_BOUND, "area_1": AREA_1, "max_steps": MAX_STEPS, "window": WINDOW,
    "aim": AIM, "grow": GROW, "min_elements": MIN_ELEMENTS, "max_lines": MAX_LINES,
    "shirley": SHIRLEY, "generator": GENERATOR, "ddof": 1,
    "configs": {k: (v[0], v[1], v[2], v[3]) for k, v in CONFIGS.items()},
    "gaussian_sigma_eV": 0.6, "window_half_eV": 10.0, "exposure_s": 1.0,
    "series": {"1": "[seed, 1, condition, candidate]", "2": "[seed, 2]",
               "3": "[seed, 3, run_kind, condition, trial]",
               "4": "[seed, 4, run_kind, condition, trial]",
               "5": "[seed, 5, run_kind, condition]"},
}


def config_hash() -> str:
    """SHA-256 of every registered item, the run's configuration hash."""
    text = json.dumps(REGISTERED, sort_keys=True, default=list)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generator(key) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(k) for k in key])))


def condition_id(config: str, background: str, level: float) -> int:
    """10 x configuration + 5 x background + level (0-39)."""
    return (10 * CONFIG_NAMES.index(config) + 5 * BACKGROUNDS.index(background)
            + LEVELS.index(level))


def condition(cid: int) -> tuple[str, str, float]:
    if not 0 <= cid < 40:
        raise ValueError(f"condition ID {cid} is not in 0..39")
    return CONFIG_NAMES[cid // 10], BACKGROUNDS[(cid % 10) // 5], LEVELS[cid % 5]


ALL_CONDITIONS = tuple(range(40))


# ---------------------------------------------------------------------------
# Expected counts and declared spectra (§8, "Conditions", "Prediction error")
# ---------------------------------------------------------------------------

def parts(cid: int, si_area: float):
    """Per line: (template Line without intensity, peak part, base part), and the Conditions.

    The §6 construction, split so that D_kappa = kappa_p x peak + kappa_b x base.
    """
    config, background, _ = condition(cid)
    spec, hv, step, ratio = CONFIGS[config]
    base_level = si_area * _PEAK / ratio
    out = []
    for el, orb, be, mult in spec:
        ke0 = hv - be
        e = np.linspace(ke0 - 12.0, ke0 + 12.0, int(round(24.0 / step)) + 1)
        peak = mult * si_area * _PEAK * np.exp(-0.5 * ((e - ke0) / 0.6) ** 2)
        base = np.full_like(e, base_level)
        template = Line(element=el, orbital=orb, energy=e, intensity=np.zeros_like(e),
                        window=(ke0 - 10.0, ke0 + 10.0), binding_energy=be, exposure_s=1.0,
                        exposure_basis="synthetic: 1 s per channel",
                        source=f"condition {cid} ({config}/{background})")
        out.append((template, peak, base))
    cond = Conditions(photon_energy=hv, table="scofield", background=background,
                      matrix=Matrix(name="SiO2", source="CompoundDB", compound="SiO2"),
                      intensity_semantics="raw_counts", transmission_applied="applied",
                      declared_elements=tuple(el for el, *_ in spec))
    return out, cond


def declared(cid: int, si_area: float, kappa=(1.0, 1.0)) -> tuple[list[Line], Conditions]:
    """D_kappa(s): kappa_p x peak + kappa_b x base, on the condition's grid."""
    kp, kb = kappa
    out, cond = parts(cid, si_area)
    return [replace(t, intensity=kp * peak + kb * base) for t, peak, base in out], cond


def expected(cid: int, si_area: float) -> tuple[list[Line], Conditions]:
    """mu(s) = D_(1,1): the counts every data set is drawn from."""
    return declared(cid, si_area, (1.0, 1.0))


# ---------------------------------------------------------------------------
# Count setting (§8, "Levels and the count setting")
# ---------------------------------------------------------------------------

def _in_window(r: float, level: float) -> bool:
    return WINDOW[0] * level <= r <= WINDOW[1] * level


def _explore(lines, cond, cid: int, k: int) -> tuple[float | None, str | None]:
    """The exploratory bootstrap (B_EXPLORE, series 1): (r, None) or (None, reason)."""
    base = composition(lines, cond)
    if base.fractions is None:
        return None, "composition(mu) refused: " + "; ".join(base.reasons)
    rng = generator([SEED, 1, cid, k])
    rows, refused, areas = _bootstrap_detail(lines, cond, B_EXPLORE, rng, base.denominator)
    if refused:
        return None, f"{refused} of {B_EXPLORE} exploratory replicates refused"
    observed = {x.element: x.area for x in base.lines}
    return _relative_area_noise(areas, [observed[el] for el in base.denominator]), None


def set_counts(cid: int, predict=_predict_r, explore=_explore) -> dict:
    """Scale every channel by one factor until r_pred(mu) lies in the window.

    Steps 1-4 of §8. Returns the trace; ``established`` is False when no
    candidate is final after MAX_STEPS, and the caller stops the run.
    """
    _, _, level = condition(cid)
    area, trace = AREA_1, []
    for k in range(1, MAX_STEPS + 1):
        lines, cond = expected(cid, area)
        entry = {"candidate": k, "si_area": area, "r_explore": None, "r_pred": None,
                 "refusal": None, "final": False}
        trace.append(entry)
        r_exp, why = explore(lines, cond, cid, k)
        if why is not None:
            entry["refusal"] = why
            area *= GROW
            continue
        entry["r_explore"] = r_exp
        if not _in_window(r_exp, level):
            area *= (r_exp / (AIM * level)) ** 2
            continue
        pred = predict(lines, cond, B_PRED)
        if pred.declared_refused or pred.n_refused or pred.r_pred is None:
            entry["refusal"] = ("composition(mu) refused in the prediction" if pred.declared_refused
                                else f"{pred.n_refused} of {B_PRED} prediction replicates refused")
            area *= GROW
            continue
        entry["r_pred"] = pred.r_pred
        if not _in_window(pred.r_pred, level):
            area *= (pred.r_pred / (AIM * level)) ** 2
            continue
        entry["final"] = True
        return {"condition": cid, "level": level, "established": True, "si_area": area,
                "candidate": k, "r_pred": pred.r_pred, "trace": trace}
    return {"condition": cid, "level": level, "established": False, "si_area": None,
            "candidate": None, "r_pred": None, "trace": trace}


class CountSettingNotEstablished(RuntimeError):
    """A condition had no final candidate after MAX_STEPS; the run stops (§8 step 4)."""


# ---------------------------------------------------------------------------
# Outer trials (§8, decision rule 1; series 3 and 4)
# ---------------------------------------------------------------------------

def trial(cid: int, kind: str, i: int, si_area: float, n_boot: int) -> dict:
    """One outer trial: its category (1, 2 or 3), cause, estimate, SE and realised r."""
    mu, cond = expected(cid, si_area)
    rk = RUN_KINDS[kind]
    rng3 = generator([SEED, 3, rk, cid, i])
    data = [replace(ln, intensity=rng3.poisson(ln.intensity).astype(np.float64)) for ln in mu]
    row = {"trial": i, "category": None, "cause": None, "estimate": None, "se": None, "r": None}
    est = composition(data, cond)
    if est.fractions is None:
        row.update(category=1, cause="composition refused: " + "; ".join(est.reasons))
        return row
    elements = est.denominator
    row["estimate"] = [est.fractions[el] for el in elements]
    # The checks _composition_uncertainty applies to observed data, in its order.
    if any(np.any(np.asarray(ln.intensity) < 0) for ln in data):
        row.update(category=2, cause="not_evaluated: negative counts")
        return row
    if any(not np.array_equal(np.asarray(ln.intensity), np.round(np.asarray(ln.intensity)))
           for ln in data):
        row.update(category=2, cause="not_evaluated: non-integer counts")
        return row
    rng4 = generator([SEED, 4, rk, cid, i])
    rows, refused, areas = _bootstrap_detail(data, cond, n_boot, rng4, elements)
    if refused:
        row.update(category=2, cause=f"withheld: {refused} of {n_boot} replicates refused")
        return row
    se = np.std(rows, axis=0, ddof=1)
    if len(rows) < 2 or not np.all(np.isfinite(se)):
        row.update(category=2, cause="non-finite standard uncertainty")
        return row
    observed = {x.element: x.area for x in est.lines}
    row.update(category=3, se=[float(v) for v in se],
               r=_relative_area_noise(areas, [observed[el] for el in elements]))
    return row


def _chunk(args) -> list[dict]:
    cid, kind, start, stop, si_area, n_boot = args
    return [trial(cid, kind, i, si_area, n_boot) for i in range(start, stop)]


# ---------------------------------------------------------------------------
# Execution identity (§8, "Execution identity")
# ---------------------------------------------------------------------------

def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def identity(root: Path | None = None) -> dict:
    """What the run records; raises unless toyomacro was imported from ``root`` and it is clean.

    ``root`` defaults to the git checkout the command is launched from
    (the working directory), not to wherever toyomacro happens to be
    imported from, so an import from another checkout or an installed
    copy is refused rather than recorded.
    """
    package = Path(toyomacro.__file__).resolve().parent
    root = Path(root or _git(Path.cwd(), "rev-parse", "--show-toplevel")).resolve()
    if not package.is_relative_to(root):
        raise RuntimeError(f"toyomacro was imported from {package}, not from the checkout {root}")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError(f"the checkout {root} has modified tracked files:\n{dirty}")
    return {"design": DESIGN, "config_hash": config_hash(), "git_commit": _git(root, "rev-parse", "HEAD"),
            "checkout": str(root), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "generator": GENERATOR, "machine": platform.machine()}


def check_output_dir(out: Path, root: Path) -> None:
    """The output directory lies outside the checkout, or under a path git ignores."""
    out = out.resolve()
    if not out.is_relative_to(root.resolve()):
        return
    rel = out.relative_to(root.resolve())
    ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", str(rel)],
                             capture_output=True).returncode == 0
    if not ignored:
        raise RuntimeError(f"{out} is inside the checkout and not ignored; the run would dirty it")


IDENTITY_KEYS = ("config_hash", "git_commit", "python", "numpy", "scipy", "machine", "generator")
RUN_KEYS = ("kind", "n_outer", "n_boot")


def _same_identity(a: dict, b: dict, keys=IDENTITY_KEYS) -> bool:
    return all(a.get(k) == b.get(k) for k in keys)


def _differences(a: dict, b: dict, keys) -> list[str]:
    return [f"{k}: {a.get(k)!r} vs {b.get(k)!r}" for k in keys if a.get(k) != b.get(k)]


def _write_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Running (count setting, then the outer trials) and summarising
# ---------------------------------------------------------------------------

def run(kind: str, out: Path, n_outer: int | None = None, n_boot: int | None = None,
        chunk: int = 50, workers: int | None = None, conditions=ALL_CONDITIONS,
        root: Path | None = None) -> None:
    """Resumable; refuses to resume a run whose identity differs."""
    M, B = RUN_SIZES[kind]
    if kind == "decision" and (n_outer, n_boot) not in ((None, None), (M, B)):
        raise RuntimeError("the decision run takes the registered M and B; no override")
    n_outer, n_boot = n_outer or M, n_boot or B
    ident = identity(root)
    check_output_dir(out, Path(ident["checkout"]))
    out.mkdir(parents=True, exist_ok=True)
    meta = {**ident, "kind": kind, "n_outer": n_outer, "n_boot": n_boot,
            "conditions": list(conditions)}
    run_file = out / "run.json"
    if run_file.exists():
        prior = json.loads(run_file.read_text())
        diff = _differences(prior, meta, IDENTITY_KEYS + RUN_KEYS)
        if diff:
            raise RuntimeError("resume refused: the run differs from this invocation in "
                               + "; ".join(diff))
    else:
        _write_atomic(run_file, meta)
    scales_file = out / "scales.json"
    if scales_file.exists():
        scales = {int(k): v for k, v in json.loads(scales_file.read_text()).items()}
    else:
        scales = dict(zip(conditions, _map(set_counts, conditions, workers=workers)))
        _write_atomic(scales_file, {str(k): v for k, v in scales.items()})
    failed = [c for c in conditions if not scales[c]["established"]]
    if failed:
        raise CountSettingNotEstablished(
            f"count setting not established for conditions {failed}; see scales.json")
    todo = []
    for cid in conditions:
        d = out / f"c{cid:02d}"
        d.mkdir(exist_ok=True)
        todo += [(cid, kind, s, min(s + chunk, n_outer), scales[cid]["si_area"], n_boot)
                 for s in range(0, n_outer, chunk) if not (d / f"chunk_{s:06d}.json").exists()]
    for n, (args, rows) in enumerate(zip(todo, _map(_chunk, todo, workers=workers))):
        path = out / f"c{args[0]:02d}" / f"chunk_{args[2]:06d}.json"
        _write_atomic(path, rows)
        if n == 0:
            print(f"first chunk written: {path}", flush=True)


def _map(fn, *iterables, workers=None):
    """``map`` in this process when ``workers == 1`` (tests, debugging), else in a pool."""
    if workers == 1:
        return list(map(fn, *iterables))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, *iterables))


def _load_trials(out: Path, cid: int, n_outer: int) -> list[dict]:
    rows = [r for f in sorted((out / f"c{cid:02d}").glob("chunk_*.json"))
            for r in json.loads(f.read_text())]
    ids = sorted(r["trial"] for r in rows)
    if ids != list(range(n_outer)):
        missing = sorted(set(range(n_outer)) - set(ids))
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise RuntimeError(f"condition {cid}: trial IDs incomplete (missing {missing[:10]}"
                           f"{'...' if len(missing) > 10 else ''}, duplicated {dup[:10]})")
    return rows


def binomial_upper_95(k: int, n: int) -> float:
    """Exact one-sided 95 % upper bound on a rate with k events in n (Clopper-Pearson)."""
    from scipy.stats import beta
    if k >= n:
        return 1.0
    return float(beta.ppf(0.95, k + 1, n - k))


def _verdict(lo: float, hi: float) -> str:
    return ("pass" if 0.9 <= lo and hi <= 1.1 else
            "fail" if hi < 0.9 or lo > 1.1 else "undecided")


def r_interval(e: np.ndarray, se: np.ndarray, rng: np.random.Generator) -> dict:
    """R_S per element with one set of resampling indices shared by the elements (series 5)."""
    n = len(e)
    idx = rng.integers(0, n, size=(N_RESAMPLE, n))

    def ratio(sel):
        return np.sqrt(np.mean(se[sel] ** 2, axis=0)) / np.std(e[sel], axis=0, ddof=1)

    # A resample with a zero spread gives an infinite ratio; rule 1 then
    # reports the condition as not evaluable, so no warning is needed.
    with np.errstate(divide="ignore", invalid="ignore"):
        point = ratio(np.arange(n))
        boots = np.array([ratio(sel) for sel in idx])  # (N_RESAMPLE, n_el)
    lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
    return {"R": [float(v) for v in point], "R_mc_95": [[float(a), float(b)] for a, b in zip(lo, hi)],
            "verdict": [_verdict(a, b) for a, b in zip(lo, hi)]}


def condition_summary(cid: int, kind: str, rows: list[dict], n_outer: int) -> dict:
    """Decision rule 1 and 2 for one condition."""
    cats = {1: [r for r in rows if r["category"] == 1], 2: [r for r in rows if r["category"] == 2],
            3: [r for r in rows if r["category"] == 3]}
    causes, cause_detail = {}, {}
    for r in cats[1] + cats[2]:
        kind_ = r["cause"].split(":")[0]
        causes[kind_] = causes.get(kind_, 0) + 1
        cause_detail[r["cause"]] = cause_detail.get(r["cause"], 0) + 1
    n_fail = len(cats[1]) + len(cats[2])
    rate = n_fail / n_outer
    s = {"condition": cid, "n_trials": n_outer, "n_category": {str(k): len(v) for k, v in cats.items()},
         "causes": causes, "cause_detail": cause_detail, "rate": rate, "rate_upper_95": binomial_upper_95(n_fail, n_outer),
         "rate_condition": rate <= RATE_BOUND, "evaluable": None, "R_S": None,
         "sd_bias_2_3": None, "sd_bias_3": None, "median_r": None, "pass": False}
    e3 = np.array([r["estimate"] for r in cats[3]], dtype=np.float64)
    se3 = np.array([r["se"] for r in cats[3]], dtype=np.float64)
    e23 = np.array([r["estimate"] for r in cats[2] + cats[3]], dtype=np.float64)
    _, cond = expected(cid, 1.0)
    s["published_digest"] = published_digest([(r["trial"], r["se"]) for r in cats[3]])
    if len(e3) >= 2 and np.all(np.std(e3, axis=0, ddof=1) > 0):
        rs = r_interval(e3, se3, generator([SEED, 5, RUN_KINDS[kind], cid]))
        finite = all(math.isfinite(v) for v in rs["R"]) and all(
            math.isfinite(a) and math.isfinite(b) for a, b in rs["R_mc_95"])
        s["R_S"] = {"elements": list(cond.declared_elements), **rs}
        s["evaluable"] = finite
        s["verdict"] = ("pass" if finite and all(v == "pass" for v in rs["verdict"]) else
                        "fail" if "fail" in rs["verdict"] else "undecided")
    else:
        s["evaluable"] = False
        s["verdict"] = "not evaluable"
    s["pass"] = bool(s["evaluable"] and s["verdict"] == "pass" and s["rate_condition"])
    s["median_r"] = float(np.median([r["r"] for r in cats[3]])) if cats[3] else None
    s["_e3"], s["_e23"] = e3, e23  # stripped before writing
    return s


def published_digest(pairs) -> str:
    """SHA-256 of the sorted (trial ID, SE) pairs a condition's admission would publish."""
    text = json.dumps(sorted((int(i), [float(v) for v in se]) for i, se in pairs))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sd_bias(e: np.ndarray, truth: list[float]) -> dict | None:
    if len(e) < 2:
        return None
    return {"sd": [float(v) for v in np.std(e, axis=0, ddof=1)],
            "bias": [float(v) for v in (np.mean(e, axis=0) - np.asarray(truth))],
            "n": int(len(e))}


PROVENANCE = "synthetic: kappa-scaled expected counts of the registered condition (planning_model)"


def admission(cid: int, si_area: float, predict=_predict_r) -> list[dict]:  # noqa: D417
    """Per kappa: r_pred, the prediction's refusals, the structural conditions, and the declared input.

    Admission conditions 1 and 2 are summarised as ``eligible``; condition
    3 as ``structural`` (two or three lines in the denominator, no
    transmission divided out, a registered background). The declared
    arrays, the conditions, their provenance and the prediction settings
    are stored, as §8 "Declared input" says they are in this round; the
    energy grid, the same for every kappa, is stored once per condition
    by :func:`declared_grid`.
    """
    out = []
    for kappa in KAPPA:
        lines, cond = declared(cid, si_area, kappa)
        p = predict(lines, cond, B_PRED)
        structural = (MIN_ELEMENTS <= len(cond.declared_elements) <= MAX_LINES
                      and cond.transmission_applied == "applied" and cond.background in BACKGROUNDS)
        out.append({
            "kappa": list(kappa), "r_pred": p.r_pred, "declared_refused": p.declared_refused,
            "n_refused": p.n_refused, "structural": bool(structural),
            "eligible": (not p.declared_refused) and p.n_refused == 0 and p.r_pred is not None,
            "declared": {
                "provenance": PROVENANCE, "fixed_before_data": True,
                "prediction": {"B_pred": B_PRED, "seed_key": list(_PREDICTION_SEED_KEY),
                               "generator": GENERATOR},
                "conditions": {"photon_energy": cond.photon_energy, "table": cond.table,
                               "background": cond.background, "matrix": cond.matrix.compound,
                               "intensity_semantics": cond.intensity_semantics,
                               "transmission_applied": cond.transmission_applied,
                               "declared_elements": list(cond.declared_elements)},
                "lines": [{"element": ln.element, "orbital": ln.orbital,
                           "energy_scale": ln.energy_scale, "window": list(ln.window),
                           "binding_energy": ln.binding_energy, "exposure_s": ln.exposure_s,
                           "exposure_basis": ln.exposure_basis,
                           "intensity": [float(v) for v in ln.intensity]} for ln in lines]}})
    return out


def declared_grid(cid: int, si_area: float) -> list[dict]:
    """The energy grid of each line, shared by every D_kappa of the condition."""
    lines, _ = expected(cid, si_area)
    return [{"element": ln.element, "orbital": ln.orbital, "energy_scale": ln.energy_scale,
             "energy": [float(v) for v in ln.energy]} for ln in lines]


def decide(conds: dict[int, dict]) -> dict:
    """Decision rules 3-5 per background, from per-condition summaries with their admissions."""
    decision = {}
    for bg in BACKGROUNDS:
        mine = {cid: s for cid, s in conds.items() if condition(cid)[1] == bg}
        trace, scope = [], None
        for r_star in CANDIDATES:
            admitted = {cid for cid, s in mine.items()
                        if any(a["eligible"] and a["r_pred"] <= r_star for a in s["admission"])}
            checks = {"non_empty": bool(admitted)}
            per_config = {}
            for cfg in CONFIG_NAMES:
                levels = sorted(condition(c)[2] for c in admitted if condition(c)[0] == cfg)
                entry = {"levels_admitted": levels, "has_level": bool(levels), "above": None,
                         "above_passes": None}
                if levels:
                    top = levels[-1]
                    if top == LEVELS[-1]:
                        entry["above"], entry["above_passes"] = None, False
                    else:
                        above = LEVELS[LEVELS.index(top) + 1]
                        c_above = condition_id(cfg, bg, above)
                        entry["above"] = above
                        entry["above_passes"] = bool(c_above in mine and mine[c_above]["pass"])
                per_config[cfg] = entry
            checks["every_configuration"] = all(v["has_level"] for v in per_config.values())
            checks["all_admitted_pass"] = all(mine[c]["pass"] for c in admitted)
            checks["level_above_passes"] = all(v["above_passes"] for v in per_config.values())
            stands = all(checks.values())
            trace.append({"r_star": r_star, "admitted": sorted(admitted), "per_config": per_config,
                          "checks": checks, "stands": stands})
            if stands and scope is None:
                scope = r_star
        decision[bg] = {"r_star": scope, "trace": trace}
    return decision


def implementation_check(rows: list[dict], summary: dict) -> dict:
    """Decision rule 6: the admission publishes exactly rule 1's category 3, same SE, same R.

    Admission is per condition, so under any admitted kappa the published
    trials are those that returned a finite standard uncertainty. This
    derives that set from the row *fields* (estimate present, SE present
    and finite), not from the category labels rule 1 used, and compares
    the digest and the R point estimate with rule 1's.
    """
    published = [(r["trial"], r["se"]) for r in rows
                 if r["estimate"] is not None and r["se"] is not None
                 and all(math.isfinite(v) for v in r["se"])]
    digest = published_digest(published)
    r_point = None
    if len(published) >= 2:
        by_id = {i: se for i, se in published}
        e = np.array([next(r["estimate"] for r in rows if r["trial"] == i) for i in sorted(by_id)])
        se = np.array([by_id[i] for i in sorted(by_id)], dtype=np.float64)
        sd = np.std(e, axis=0, ddof=1)
        r_point = [float(v) for v in np.sqrt(np.mean(se ** 2, axis=0)) / sd] if np.all(sd > 0) else None
    ok_set = digest == summary["published_digest"] and len(published) == summary["n_category"]["3"]
    ok_r = ((summary["R_S"] is None and r_point is None) or
            (summary["R_S"] is not None and r_point is not None
             and all(a == b for a, b in zip(r_point, summary["R_S"]["R"]))))
    return {"published_set_match": bool(ok_set), "R_match": bool(ok_r), "n_published": len(published),
            "ok": bool(ok_set and ok_r)}


def summarize(out: Path, workers: int | None = None) -> dict:
    """Decision rules 1-6 over a finished run; raises on an incomplete or inconsistent one."""
    meta = json.loads((out / "run.json").read_text())
    now = identity()
    diff = _differences(meta, now, IDENTITY_KEYS)
    if diff:
        raise RuntimeError("summary refused: the run and this invocation differ in " + "; ".join(diff))
    scales = {int(k): v for k, v in json.loads((out / "scales.json").read_text()).items()}
    kind, n_outer = meta["kind"], meta["n_outer"]
    conds, record = {}, []
    cids = list(meta["conditions"])
    admissions = dict(zip(cids, _map(admission, cids, [scales[c]["si_area"] for c in cids],
                                     workers=workers)))
    for cid in cids:
        rows = _load_trials(out, cid, n_outer)
        s = condition_summary(cid, kind, rows, n_outer)
        si_area = scales[cid]["si_area"]
        mu, cond = expected(cid, si_area)
        truth = composition(mu, cond).fractions
        truth_v = [truth[el] for el in cond.declared_elements]
        s["sd_bias_2_3"] = _sd_bias(s.pop("_e23"), truth_v)
        s["sd_bias_3"] = _sd_bias(s.pop("_e3"), truth_v)
        s["truth"] = dict(zip(cond.declared_elements, truth_v))
        s["admission"] = admissions[cid]
        s["declared_grid"] = declared_grid(cid, si_area)
        s["implementation_check"] = implementation_check(rows, s)
        if not s["implementation_check"]["ok"]:
            raise RuntimeError(f"condition {cid}: implementation check failed (rule 6)")
        s["count_setting"] = scales[cid]
        conds[cid] = s
    decision = decide(conds)
    for cid, s in conds.items():
        cfg, bg, level = condition(cid)
        r_star = decision[bg]["r_star"]
        for a in s["admission"]:
            a["admitted_at_r_star"] = bool(r_star is not None and a["eligible"] and a["structural"]
                                           and a["r_pred"] <= r_star)
        record.append({"condition": cid, "config": cfg, "background": bg, "level": level, **s})
    return {"design": DESIGN, "seed": SEED, "run": meta, "summary_identity": now,
            "decision": decision, "conditions": record}


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--kind", choices=sorted(RUN_KINDS), required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--outer", type=int, default=None, help="override M (tests only)")
    r.add_argument("--boot", type=int, default=None, help="override B (tests only)")
    r.add_argument("--chunk", type=int, default=50)
    r.add_argument("--workers", type=int, default=None)
    s = sub.add_parser("summarize")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--record", type=Path, default=None)
    s.add_argument("--workers", type=int, default=None)
    a = p.parse_args(argv)
    if a.cmd == "run":
        run(a.kind, a.out, a.outer, a.boot, chunk=a.chunk, workers=a.workers)
        first = sorted(a.out.glob("c*/chunk_*.json"))
        print(f"run {a.kind}: {len(first)} chunk(s) written under {a.out}")
    else:
        text = json.dumps(summarize(a.out, workers=a.workers), indent=1)
        if a.record:
            a.record.write_text(text + "\n")
        print(text if a.record is None else f"record written to {a.record}")


if __name__ == "__main__":
    main(sys.argv[1:])
