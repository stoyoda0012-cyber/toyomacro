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

from toyomacro.composition import Conditions, Line, Matrix, _bootstrap, composition

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
    s = sub.add_parser("summarize")
    s.add_argument("dirs", type=Path, nargs="+")
    s.add_argument("--record", type=Path, default=None)
    a = p.parse_args()
    if a.cmd == "run":
        run(a.scenario, a.outer, a.boot, a.out, workers=a.workers)
    else:
        record = {"design": "docs/design/composition-uncertainty.md",
                  "provenance": _provenance(), "scenarios": [summarize(d) for d in a.dirs]}
        text = json.dumps(record, indent=1)
        if a.record:
            a.record.write_text(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
