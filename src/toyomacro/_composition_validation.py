"""Synthetic validation of ``composition_uncertainty`` (private).

Implements the pre-registered check of docs/design/composition-uncertainty.md
§3. Runs are resumable: each chunk of outer data sets is written to its
own JSON file and skipped when present, so an interrupted run continues.

    python -m toyomacro._composition_validation run S1 --outer 2000 --boot 1000 --out DIR
    python -m toyomacro._composition_validation summarize DIR [DIR ...] --record FILE
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from toyomacro.composition import (
    Conditions,
    Line,
    Matrix,
    _bootstrap,
    _bootstrap_detail,
    _relative_area_noise,
    composition,
)

HV = 1486.6
SCENARIOS = {
    # name: (Si 2p area, O 1s area, base per channel, background, role)
    "S1": (30_000.0, 90_000.0, 200.0, "linear", "acceptance"),
    "S2": (3_000.0, 9_000.0, 50.0, "shirley", "acceptance"),
    "S3": (300.0, 900.0, 5.0, "linear", "reported"),
}
LINES = (("Si", "2p", 99.0), ("O", "1s", 532.0))
ELEMENTS = ("Si", "O")


def expected_lines(name: str) -> list[Line]:
    """The scenario's expected counts per channel (the Poisson means)."""
    si, o, base, _, _ = SCENARIOS[name]
    out = []
    for (el, orb, be), area in zip(LINES, (si, o)):
        ke0 = HV - be
        e = np.linspace(ke0 - 12.0, ke0 + 12.0, 241)
        mu = base + area / (0.6 * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((e - ke0) / 0.6) ** 2)
        out.append(Line(element=el, orbital=orb, energy=e, intensity=mu,
                        window=(ke0 - 10.0, ke0 + 10.0), binding_energy=be, exposure_s=1.0,
                        exposure_basis="synthetic: 1 s per channel", source=f"scenario {name}"))
    return out


def conditions(name: str) -> Conditions:
    return Conditions(photon_energy=HV, table="scofield", background=SCENARIOS[name][3],
                      matrix=Matrix(name="SiO2", source="CompoundDB", compound="SiO2"),
                      intensity_semantics="raw_counts", transmission_applied="applied",
                      declared_elements=ELEMENTS)


def _chunk(args) -> list[dict]:
    name, seed, start, stop, n_boot = args
    mu, cond = expected_lines(name), conditions(name)
    rows = []
    for m in range(start, stop):
        rng = np.random.default_rng([seed, m])
        data = [Line(**{**ln.__dict__, "intensity": rng.poisson(ln.intensity).astype(float)})
                for ln in mu]
        est = composition(data, cond)
        row = {"m": m, "estimate": None, "se": None, "refused_replicates": None}
        if est.fractions is not None:
            row["estimate"] = [est.fractions[el] for el in ELEMENTS]
            boot, refused = _bootstrap(data, cond, n_boot, rng, ELEMENTS)
            row["refused_replicates"] = refused
            if refused == 0:
                row["se"] = [float(v) for v in np.std(boot, axis=0, ddof=1)]
        rows.append(row)
    return rows


def run(name: str, n_outer: int, n_boot: int, out: Path, seed: int = 20261007,
        chunk: int = 50, workers: int | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta = {"scenario": name, "n_outer": n_outer, "n_boot": n_boot, "seed": seed}
    (out / "meta.json").write_text(json.dumps(meta))
    todo = [(name, seed, s, min(s + chunk, n_outer), n_boot)
            for s in range(0, n_outer, chunk) if not (out / f"chunk_{s:06d}.json").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for args, rows in zip(todo, pool.map(_chunk, todo)):
            (out / f"chunk_{args[2]:06d}.json").write_text(json.dumps(rows))


def summarize(out: Path, n_resample: int = 5000) -> dict:
    meta = json.loads((out / "meta.json").read_text())
    rows = [r for f in sorted(out.glob("chunk_*.json")) for r in json.loads(f.read_text())]
    name = meta["scenario"]
    truth = composition(expected_lines(name), conditions(name)).fractions
    est = np.array([r["estimate"] for r in rows if r["estimate"] is not None])
    paired = [(r["estimate"], r["se"]) for r in rows if r["se"] is not None]
    e = np.array([p[0] for p in paired])
    se = np.array([p[1] for p in paired])
    rng = np.random.default_rng(meta["seed"] + 1)
    summary = {"scenario": name, "role": SCENARIOS[name][4], "background": SCENARIOS[name][3],
               "n_outer_attempted": len(rows), "n_outer_with_estimate": int(len(est)),
               "n_outer_with_se": int(len(paired)),
               "n_outer_with_refused_replicates": sum(
                   1 for r in rows if (r["refused_replicates"] or 0) > 0),
               "n_boot": meta["n_boot"], "seed": meta["seed"], "elements": {}}
    for k, el in enumerate(ELEMENTS):
        def ratio(idx, k=k):
            return math.sqrt(np.mean(se[idx, k] ** 2)) / np.std(e[idx, k], ddof=1)
        r_hat = ratio(np.arange(len(e)))
        boots = np.array([ratio(rng.integers(0, len(e), len(e))) for _ in range(n_resample)])
        lo, hi = np.quantile(boots, [0.025, 0.975])
        verdict = ("pass" if 0.9 <= lo and hi <= 1.1 else
                   "fail" if hi < 0.9 or lo > 1.1 else "undecided")
        summary["elements"][el] = {
            "R": float(r_hat), "R_mc_95": [float(lo), float(hi)],
            "mc_half_width": float((hi - lo) / 2), "verdict": verdict,
            "sd_estimate": float(np.std(est[:, k], ddof=1)),
            "rms_se": float(math.sqrt(np.mean(se[:, k] ** 2))),
            "truth": truth[el], "bias": float(np.mean(est[:, k]) - truth[el]),
            "bias_se": float(np.std(est[:, k], ddof=1) / math.sqrt(len(est))),
        }
    verdicts = {v["verdict"] for v in summary["elements"].values()}
    summary["verdict"] = ("pass" if verdicts == {"pass"} else
                          "fail" if "fail" in verdicts else "undecided")
    return summary


def _provenance() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "numpy": np.__version__, "python": platform.python_version(),
            "machine": platform.machine(), "date": time.strftime("%Y-%m-%d")}


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("scenario", choices=sorted(SCENARIOS))
    r.add_argument("--outer", type=int, required=True)
    r.add_argument("--boot", type=int, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=None)
    r2 = sub.add_parser("run2")
    r2.add_argument("--out", type=Path, required=True)
    r2.add_argument("--workers", type=int, default=None)
    s2 = sub.add_parser("summarize2")
    s2.add_argument("out", type=Path)
    s2.add_argument("--record", type=Path, default=None)
    s = sub.add_parser("summarize")
    s.add_argument("dirs", type=Path, nargs="+")
    s.add_argument("--record", type=Path, default=None)
    a = p.parse_args()
    if a.cmd == "run":
        run(a.scenario, a.outer, a.boot, a.out, workers=a.workers)
    elif a.cmd == "run2":
        run2(a.out, workers=a.workers)
    elif a.cmd == "summarize2":
        text = json.dumps(summarize2(a.out), indent=1)
        if a.record:
            a.record.write_text(text + "\n")
        print(text)
    else:
        record = {"design": "docs/design/composition-uncertainty.md",
                  "provenance": _provenance(), "scenarios": [summarize(d) for d in a.dirs]}
        text = json.dumps(record, indent=1)
        if a.record:
            a.record.write_text(text + "\n")
        print(text)



# ---------------------------------------------------------------------------
# Second validation (design record §6)
# ---------------------------------------------------------------------------

SEED2 = 20261008
LEVELS = (0.005, 0.01, 0.02, 0.04, 0.08)
BACKGROUNDS2 = ("linear", "shirley")
#: config: (lines as (element, orbital, BE, area multiple), photon energy, channel eV,
#:          peak height / base)
CONFIGS = {
    "P": ((("Si", "2p", 99.0, 1.0), ("O", "1s", 532.0, 3.0)), HV, 0.1, 100.0),
    "B": ((("Si", "2p", 99.0, 1.0), ("O", "1s", 532.0, 3.0)), HV, 0.1, 0.1),
    "T": ((("Si", "2p", 99.0, 1.0), ("O", "1s", 532.0, 3.0), ("C", "1s", 285.0, 1.0)),
          HV, 0.5, 10.0),
    "G": ((("Si", "2p", 99.0, 1.0), ("O", "1s", 532.0, 3.0)), 9251.7, 0.1, 10.0),
}
_PEAK = 1.0 / (0.6 * math.sqrt(2 * math.pi))  # Gaussian height per unit area


def scenario2(config: str, background: str, si_area: float) -> tuple[list[Line], Conditions]:
    """Expected counts and conditions for one §6 scenario at a given Si area."""
    spec, hv, step, ratio = CONFIGS[config]
    base = si_area * _PEAK / ratio
    lines = []
    for el, orb, be, mult in spec:
        ke0 = hv - be
        e = np.linspace(ke0 - 12.0, ke0 + 12.0, int(round(24.0 / step)) + 1)
        mu = base + mult * si_area * _PEAK * np.exp(-0.5 * ((e - ke0) / 0.6) ** 2)
        lines.append(Line(element=el, orbital=orb, energy=e, intensity=mu,
                          window=(ke0 - 10.0, ke0 + 10.0), binding_energy=be, exposure_s=1.0,
                          exposure_basis="synthetic: 1 s per channel",
                          source=f"scenario {config}/{background}"))
    cond = Conditions(photon_energy=hv, table="scofield", background=background,
                      matrix=Matrix(name="SiO2", source="CompoundDB", compound="SiO2"),
                      intensity_semantics="raw_counts", transmission_applied="applied",
                      declared_elements=tuple(el for el, *_ in spec))
    return lines, cond


def _draw(lines, rng):
    return [Line(**{**ln.__dict__, "intensity": rng.poisson(ln.intensity).astype(float)})
            for ln in lines]


def _data_r(data, cond, n_boot, rng, elements):
    """(estimate dict or None, se list or None, refused, r or None) for one data set."""
    est = composition(data, cond)
    if est.fractions is None:
        return None, None, None, None
    rows, refused, areas = _bootstrap_detail(data, cond, n_boot, rng, elements)
    if refused or len(rows) < 2:
        return est, None, refused, None
    observed = {x.element: x.area for x in est.lines}
    r = _relative_area_noise(areas, [observed[el] for el in elements])
    return est, [float(v) for v in np.std(rows, axis=0, ddof=1)], refused, r


def _median_r(config, background, si_area, n_draws=50, n_boot=200, seed=SEED2):
    mu, cond = scenario2(config, background, si_area)
    elements = cond.declared_elements
    rs = []
    for d in range(n_draws):
        rng = np.random.default_rng([seed, 999_999, d])
        _, _, _, r = _data_r(_draw(mu, rng), cond, n_boot, rng, elements)
        if r is not None:
            rs.append(r)
    return float(np.median(rs)) if rs else math.inf


def find_si_area(args) -> dict:
    """Scale the counts so the median r over 50 draws is within 10 % of the level.

    r scales about as 1/sqrt(counts); start from that and correct, checking
    each candidate with the 50-draw median (R is not computed here).
    """
    config, background, level = args
    area, history = 1e4, []
    for _ in range(8):
        med = _median_r(config, background, area)
        history.append((area, med))
        if abs(med / level - 1) <= 0.10:
            return {"config": config, "background": background, "level": level,
                    "si_area": area, "median_r": med, "history": history}
        area *= (med / level) ** 2 if math.isfinite(med) else 4.0
    raise RuntimeError(f"no scale found for {args}: {history}")


def _chunk2(args) -> list[dict]:
    spec, start, stop, n_boot = args
    mu, cond = scenario2(spec["config"], spec["background"], spec["si_area"])
    elements = cond.declared_elements
    rows = []
    for m in range(start, stop):
        rng = np.random.default_rng([SEED2, spec["index"], m])
        est, se, refused, r = _data_r(_draw(mu, rng), cond, n_boot, rng, elements)
        rows.append({"m": m,
                     "estimate": None if est is None else [est.fractions[el] for el in elements],
                     "se": se, "refused_replicates": refused, "r": r})
    return rows


def run2(out: Path, n_outer: int = 2000, n_boot: int = 1000, chunk: int = 50,
         workers: int | None = None) -> None:
    """Scale every scenario, then run them all; resumable at both steps."""
    out.mkdir(parents=True, exist_ok=True)
    grid = [(c, bg, lv) for c in CONFIGS for bg in BACKGROUNDS2 for lv in LEVELS]
    scales_file = out / "scales.json"
    if scales_file.exists():
        scales = json.loads(scales_file.read_text())
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            scales = list(pool.map(find_si_area, grid))
        for i, s in enumerate(scales):
            s["index"] = i
        scales_file.write_text(json.dumps(scales, indent=1))
    todo = []
    for spec in scales:
        d = out / f"s{spec['index']:02d}"
        d.mkdir(exist_ok=True)
        (d / "meta.json").write_text(json.dumps({**spec, "n_outer": n_outer, "n_boot": n_boot}))
        todo += [(spec, s, min(s + chunk, n_outer), n_boot) for s in range(0, n_outer, chunk)
                 if not (d / f"chunk_{s:06d}.json").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for args, rows in zip(todo, pool.map(_chunk2, todo)):
            d = out / f"s{args[0]['index']:02d}"
            (d / f"chunk_{args[1]:06d}.json").write_text(json.dumps(rows))


def _r_interval(e, se, rng, n_resample=5000):
    def ratio(idx):
        return math.sqrt(np.mean(se[idx] ** 2)) / np.std(e[idx], ddof=1)
    r_hat = ratio(np.arange(len(e)))
    boots = np.array([ratio(rng.integers(0, len(e), len(e))) for _ in range(n_resample)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    verdict = ("pass" if 0.9 <= lo and hi <= 1.1 else
               "fail" if hi < 0.9 or lo > 1.1 else "undecided")
    return float(r_hat), [float(lo), float(hi)], verdict


def summarize2(out: Path) -> dict:
    """Apply the pre-registered decision rule of §6 to the runs in ``out``."""
    scales = json.loads((out / "scales.json").read_text())
    scen = []
    for spec in scales:
        d = out / f"s{spec['index']:02d}"
        rows = [r for f in sorted(d.glob("chunk_*.json")) for r in json.loads(f.read_text())]
        mu, cond = scenario2(spec["config"], spec["background"], spec["si_area"])
        truth = composition(mu, cond).fractions
        paired = [r for r in rows if r["se"] is not None]
        scen.append({"spec": spec, "rows": rows, "paired": paired, "truth": truth,
                     "elements": cond.declared_elements})

    def stats(s, admitted=None):
        """R for the first element (with two elements the others share it)."""
        rng = np.random.default_rng(SEED2 + s["spec"]["index"] + (0 if admitted is None else 7))
        use = [p for p in s["paired"] if admitted is None or p["r"] <= admitted]
        if len(use) < 20:
            return None
        e = np.array([p["estimate"] for p in use])
        se = np.array([p["se"] for p in use])
        out = {}
        for k, el in enumerate(s["elements"]):
            r_hat, ci, verdict = _r_interval(e[:, k], se[:, k], rng)
            out[el] = {"R": r_hat, "R_mc_95": ci, "verdict": verdict}
        return out

    for s in scen:
        s["all"] = stats(s)
        s["verdict"] = _combine([v["verdict"] for v in s["all"].values()])

    decision = {}
    for bg in BACKGROUNDS2:
        mine = [s for s in scen if s["spec"]["background"] == bg]
        r_star = None
        for lv in LEVELS:
            if all(s["verdict"] == "pass" for s in mine if s["spec"]["level"] <= lv):
                r_star = lv
            else:
                break
        threshold = None
        while r_star is not None:
            at = [p["r"] for s in mine if s["spec"]["level"] == r_star for p in s["paired"]]
            threshold = float(np.quantile(at, 0.95))
            checks = [stats(s, threshold) for s in mine if s["spec"]["level"] <= r_star]
            if all(c is not None and _combine([v["verdict"] for v in c.values()]) == "pass"
                   for c in checks):
                break
            lower = [lv for lv in LEVELS if lv < r_star]
            r_star = lower[-1] if lower else None
            threshold = None
        decision[bg] = {"r_star": r_star, "r_threshold": threshold}

    record = []
    for s in scen:
        thr = decision[s["spec"]["background"]]["r_threshold"]
        adm = stats(s, thr) if thr is not None else None
        est = np.array([r["estimate"] for r in s["rows"] if r["estimate"] is not None])
        rs = np.array([p["r"] for p in s["paired"]])
        record.append({
            "config": s["spec"]["config"], "background": s["spec"]["background"],
            "level": s["spec"]["level"], "si_area": s["spec"]["si_area"],
            "definition": {"lines": CONFIGS[s["spec"]["config"]][0],
                           "photon_energy": CONFIGS[s["spec"]["config"]][1],
                           "channel_eV": CONFIGS[s["spec"]["config"]][2],
                           "peak_height_over_base": CONFIGS[s["spec"]["config"]][3],
                           "gaussian_sigma_eV": 0.6, "window_half_eV": 10.0},
            "n_attempted": len(s["rows"]), "n_with_estimate": int(len(est)),
            "n_with_se": len(s["paired"]),
            "n_admitted": (int(sum(p["r"] <= thr for p in s["paired"]))
                           if thr is not None else 0),
            "median_r": float(np.median(rs)), "verdict_all": s["verdict"],
            "R_all": s["all"], "R_admitted": adm,
            "bias": {el: {"bias": float(np.mean(est[:, k]) - s["truth"][el]),
                          "se": float(np.std(est[:, k], ddof=1) / math.sqrt(len(est)))}
                     for k, el in enumerate(s["elements"])},
        })
    return {"design": "docs/design/composition-uncertainty.md §6", "seed": SEED2,
            "provenance": _provenance(), "decision": decision, "scenarios": record}


def _combine(verdicts) -> str:
    v = set(verdicts)
    return "pass" if v == {"pass"} else "fail" if "fail" in v else "undecided"


if __name__ == "__main__":
    main()
