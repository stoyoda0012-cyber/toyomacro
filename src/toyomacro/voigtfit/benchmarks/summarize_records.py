"""Read the committed benchmark records and print what they actually say.

    python -m toyomacro.voigtfit.benchmarks.summarize_records

The rule this tool exists to enforce: **group by the hardware that was
observed, never by the label somebody typed.** "cloud" is not a machine,
it is a draw from a distribution of machines -- this repository has
already seen one cloud container change CPU mid-session, from a 2.80GHz
Xeon to a 2.10GHz one, while the tracked document describing it went on
saying 2.80GHz. A summary that groups by the word "cloud" would have
averaged those two together and reported the mean of two different
computers.

So records are grouped by ``host_label``, which is derived from the CPU
and core count rather than supplied by the operator, and ``origin`` --
where the run was launched from -- is reported as a column within the
group. Whether cloud runs launched from two different machines land on
the same hardware is then something you read off the table instead of
assuming.

Records whose ``quality`` is not ``quiet`` are listed but excluded from
the headline comparison, and the exclusion is stated rather than silent.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

#: Where the shared collection lives, relative to the repository root.
DEFAULT_RECORDS_DIR = Path(__file__).resolve().parents[4] / "benchmarks" / "records"


def load_records(directory: Path) -> list[dict]:
    """Every readable record in ``directory``, newest last."""
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except Exception as exc:
            print(f"  skipped {path.name}: {type(exc).__name__}")
            continue
        rec["_file"] = path.name
        out.append(rec)
    return out


def solver_rate(rec: dict, solver: str) -> float | None:
    for r in rec.get("results", []):
        if r.get("solver") == solver:
            return r.get("rate_median") or r.get("throughput_spec_per_s")
    return None


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    if v >= 1e6:
        return f"{v / 1e6:,.1f}M"
    return f"{v / 1e3:,.0f}k"


def summarize(records: list[dict], solvers: tuple[str, ...]) -> None:
    if not records:
        print("No records found. Produce one with:\n"
              "  python -m toyomacro.voigtfit.benchmarks.bench_platform "
              "--origin <mac|windows> --records-dir benchmarks/records")
        return

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        env = rec.get("environment", {})
        groups[(env.get("host_label", "?"), env.get("backend", {}).get("backend", "?"))].append(rec)

    print(f"{len(records)} record(s) across {len(groups)} distinct machine(s).")
    print("Grouped by observed hardware, not by any supplied label.\n")

    # Records only compare if they fitted the same numbers. They may not:
    # the spectra go through scipy.special.wofz, whose last bits depend on
    # the build, so two platforms at the same (n_batch, seed) can differ.
    hashes: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        h = rec.get("problem", {}).get("input_sha256")
        if h:
            hashes[h].append(rec["_file"])
    if len(hashes) > 1:
        print(f"!! {len(hashes)} DIFFERENT input hashes across these records. "
              "Throughput still compares; accuracy (mae_*) does NOT -- these "
              "hosts fitted different numbers.")
        for h, files in sorted(hashes.items()):
            print(f"   {h[:16]}…  {len(files)} record(s): "
                  f"{', '.join(f[:34] for f in files[:3])}"
                  + (" …" if len(files) > 3 else ""))
        print()

    for (host, backend), recs in sorted(groups.items()):
        env0 = recs[0].get("environment", {})
        origins = sorted({r.get("environment", {}).get("origin") or "unknown"
                          for r in recs})
        quiet = [r for r in recs
                 if r.get("environment", {}).get("quality", {}).get("verdict") == "quiet"]
        print(f"## {host}   [{backend}]")
        print(f"   cpu      {env0.get('cpu')}  |  {env0.get('logical_cores')} cores"
              f"  |  {env0.get('memory_gb')} GB  |  {env0.get('os')}")
        print(f"   launched from: {', '.join(origins)}")
        print(f"   {len(recs)} run(s), {len(quiet)} of them quiet")

        batches = sorted({r.get("problem", {}).get("n_batch") for r in recs})
        if len(batches) > 1:
            print(f"   !! mixed batch sizes {batches} - these rows are NOT "
                  "comparable with each other")

        # Only quiet records feed the table. Falling back to contended
        # ones and then printing "excluded" in the footer would make the
        # summary state the opposite of what it did.
        if not quiet:
            print("   !! no quiet run on this machine - no headline figures. "
                  f"{len(recs)} contended record(s) listed below.")
        for solver in solvers:
            rates = [v for v in (solver_rate(r, solver) for r in quiet)
                     if v is not None]
            if not rates:
                continue
            med = statistics.median(rates)
            spread = ("" if len(rates) < 2
                      else f"   spread {_fmt(min(rates))}-{_fmt(max(rates))}"
                           f"  ({(max(rates) / min(rates) - 1) * 100:.0f}%)")
            print(f"     {solver:<22s} {_fmt(med):>10s}{spread}")

        steals = [r.get("environment", {}).get("steal_percent") for r in recs]
        steals = [s for s in steals if s is not None]
        if steals:
            worst = max(steals)
            flag = "  <-- hypervisor contention" if worst >= 2.0 else ""
            print(f"   steal time: max {worst:.1f}%{flag}")
        print()

    contended = [r for r in records
                 if r.get("environment", {}).get("quality", {}).get("verdict") != "quiet"]
    if contended:
        print(f"{len(contended)} record(s) excluded from every headline figure "
              "above, and from the spreads:")
        for r in contended:
            q = r.get("environment", {}).get("quality", {})
            rate = solver_rate(r, "projection_kernel")
            print(f"  {r['_file']}: {q.get('verdict')} "
                  f"({'; '.join(q.get('reasons') or []) or 'no reason recorded'})"
                  + (f"  [projection_kernel {_fmt(rate)}]" if rate else ""))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--records-dir", type=str, default=str(DEFAULT_RECORDS_DIR))
    p.add_argument("--solvers", type=str,
                   default="projection_kernel,dict2d_parabola,taylor_4step,"
                           "multipeak_2comp",
                   help="comma-separated solvers to tabulate")
    args = p.parse_args(argv)

    directory = Path(args.records_dir)
    if not directory.is_dir():
        print(f"no records directory at {directory}")
        return
    summarize(load_records(directory),
              tuple(s.strip() for s in args.solvers.split(",") if s.strip()))


if __name__ == "__main__":
    main()
