"""One measurement-record schema for every host a benchmark runs on.

A throughput number is only as good as what you know about the machine
that produced it. This module supplies that context in a single shape,
so a record written on Apple Silicon, on an NVIDIA box under WSL2, and
on a GPU-less cloud container can be laid side by side and diffed.

What every record carries
-------------------------
- **The backend that actually ran** — Metal, CUDA or NumPy — rather than
  "MLX was importable". These are different compute paths with different
  numerical behaviour, and on CUDA the TF32 setting changes the answer.
- **What else the machine was doing.** Load is not noise you can average
  away: a contended host can read well below its own quiet figure while
  the repetitions *within* one invocation still agree closely, because
  they share the machine state that biases them. It can also read the
  same, or higher, when the load is not competing for the same
  resource. ``docs/BENCHMARKS.md`` §5 has both cases and says which of
  them carries a committed record.
- **The tree**, as a commit and a dirty flag.

What no record carries
----------------------
These files are meant to be committed, so nothing here records a
filesystem path, an interpreter location, a repository location, or a
sibling project's version. Process names are reported as bare names,
never as the paths they were launched from.

Portability
-----------
Every field degrades to ``None`` rather than raising: a missing `git`, a
platform without ``getloadavg``, an absent MLX. A record with holes in
it is still a usable record; an exception in the middle of a benchmark
run is not.
"""
from __future__ import annotations

import datetime
import importlib.metadata as md
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

#: Bumped when a field changes meaning or disappears. Readers should
#: check it before comparing two records.
#:
#: 3 — the load block gained ``looks_quiet_before``, ``cpu_percent_others``
#: and ``others_busy_after``, and ``looks_quiet`` changed meaning: it is
#: now the BEFORE sample's verdict, where in 2 it was the trailing one.
#: A version-2 record's verdict was computed by a rule that no longer
#: exists and is not comparable with a version-3 one.
SCHEMA_VERSION = 3

#: Versions worth recording. Deliberately limited to this package and
#: its published dependencies -- a record must not disclose what else
#: is installed alongside it.
_TRACKED_DISTRIBUTIONS = (
    "toyomacro", "mlx", "numpy", "scipy", "h5py", "lmfit", "numba", "psutil",
)


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 10) -> str:
    """Run a command for its stdout, returning '' for any failure."""
    return _run_raw(cmd, cwd, timeout).strip()


def _run_raw(cmd: list[str], cwd: Path | None = None,
             timeout: float = 10) -> str:
    """As ``_run`` but without stripping.

    ``git status --porcelain`` puts the status in two fixed columns, so an
    unstaged modification begins with a space. Stripping the whole stdout
    eats it on the first line only, turning " M file" into "M  file" --
    which reads as a *staged* change, and shifts the filename.
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=cwd).stdout
    except Exception:
        return ""


def cpu_name() -> str | None:
    """The CPU's marketing name, on macOS, Linux and Windows alike."""
    system = platform.system()
    if system == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or None
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(
                    encoding="utf-8", errors="replace").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return platform.processor() or None
    return platform.processor() or None


def _os_description() -> str:
    """A one-line OS description that is correct off macOS too."""
    system, release = platform.system(), platform.release()
    if system == "Darwin":
        mac = platform.mac_ver()[0]
        return f"Darwin {release} (macOS {mac})" if mac else f"Darwin {release}"
    if system == "Linux":
        return f"Linux {release}"
    if system == "Windows":
        return f"Windows {platform.version()}"
    return f"{system} {release}"


def _memory_gb() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        return None


#: Processes whose CPU time is not work anyone is doing. Windows reports
#: idle time as a process ("System Idle Process", PID 0) which psutil
#: dutifully returns at close to 100% per core; counting it as foreign
#: load condemned every record taken on an idle Windows host. PID 0 is
#: the scheduler on Linux too and is never real work there either.
_IDLE_PROCESS_NAMES = frozenset({"system idle process", "idle"})


def _idle_pids(procs) -> set[int]:
    """PIDs that report time nobody spent."""
    pids = {0}
    for p in procs:
        try:
            name = (p.info.get("name") or "").strip().lower()
            if name in _IDLE_PROCESS_NAMES:
                pids.add(p.pid)
        except Exception:
            continue
    return pids


def _self_pids() -> set[int]:
    """This process and its children, so a sample can exclude our own work."""
    pids = {os.getpid()}
    try:
        import psutil
        me = psutil.Process()
        pids.update(c.pid for c in me.children(recursive=True))
    except Exception:
        pass
    return pids


def load_snapshot(interval: float = 0.2, top_n: int = 5,
                  exclude_self: bool = False) -> dict:
    """What else the machine is doing, sampled over ``interval`` seconds.

    ``load_average`` is the 1/5/15-minute POSIX triple, or ``None`` on
    Windows, which has no equivalent. ``cpu_percent`` is whole-machine
    utilisation over the sampling window: 0-100, and it includes us.
    ``cpu_percent_others`` is a different quantity on a different scale
    -- the per-process sum over everything that is *not* this process
    tree, where 100 means one core, so it can exceed 100 on a
    multi-core host. The two are not comparable; the second is the one
    the quality verdict uses. ``top_processes`` names the
    heaviest consumers **by process name only** -- never by path, since
    these records are committed.

    ``looks_quiet`` applies one stated convention: **every** load average
    -- 1, 5 and 15 minute -- is below a quarter of the logical core
    count. The convention is asserted, not calibrated: nothing in this
    repository establishes that a host under it keeps a kernel inside
    its own reported range. It is a hint for the reader, not a gate.
    """
    snap: dict = {"sampled_over_s": interval}
    try:
        snap["load_average"] = list(os.getloadavg())
    except (OSError, AttributeError):
        snap["load_average"] = None

    cores = os.cpu_count()
    snap["logical_cores"] = cores

    try:
        import psutil
        # cpu_percent is measured since the previous call, so prime every
        # process first, let the sampling window elapse, then read.
        procs = list(psutil.process_iter(["name"]))
        for p in procs:
            try:
                p.cpu_percent()
            except Exception:
                pass
        snap["cpu_percent"] = psutil.cpu_percent(interval=interval)
        # The idle process is excluded unconditionally: it is not our work
        # and it is not anyone else's either.
        skip = _idle_pids(procs)
        if exclude_self:
            skip |= _self_pids()
        rows, rows_pid = [], []
        for p in procs:
            try:
                pct = p.cpu_percent()
                if pct > 0:
                    rows.append((pct, p.info.get("name") or "?"))
                    rows_pid.append((pct, p.info.get("name") or "?", p.pid))
            except Exception:
                continue
        rows.sort(reverse=True)
        # A sample taken after the benchmark contains the benchmark. Only
        # the processes that are NOT us say whether the host was shared.
        others = [(v, n) for v, n, pid in rows_pid
                  if pid not in skip
                  and (n or "").strip().lower() not in _IDLE_PROCESS_NAMES]
        others.sort(reverse=True)
        snap["cpu_percent_others"] = round(sum(v for v, _ in others), 1)
        if exclude_self:
            # Must be re-sorted: `rows_pid` is in iteration order, so
            # taking the first `top_n` of it drops the heaviest consumer
            # whenever idle processes happen to come first -- in the one
            # sample the verdict is based on.
            rows = others
        # These records are committed to a public repository, so the
        # process list discloses which applications the machine runs.
        # The names are what makes a contended record diagnosable -- an
        # orphaned worker pool is identified by seeing it here -- so they
        # are kept by default and redactable for hosts where that matters.
        redact = os.environ.get("TOYOMACRO_BENCH_REDACT_PROCESSES") == "1"
        snap["processes_redacted"] = redact
        snap["top_processes"] = [
            {"name": f"process-{i + 1}" if redact else n,
             "cpu_percent": round(v, 1)}
            for i, (v, n) in enumerate(rows[:top_n])]
    except Exception:
        snap["cpu_percent"] = None
        snap["cpu_percent_others"] = None
        snap["top_processes"] = None

    la = snap.get("load_average")
    if la and cores:
        # All three averages, not just the 1-minute one. A benchmark that
        # runs for minutes is covered better by the 5- and 15-minute
        # figures, and grading on the shortest window let a host whose
        # 15-minute load was 60% of the machine record itself as quiet.
        snap["looks_quiet"] = all(v < 0.25 * cores for v in la)
        snap["quiet_convention"] = (
            "every load average (1/5/15 min) < 0.25 * logical_cores; "
            "a convention, not a calibrated threshold")
    else:
        snap["looks_quiet"] = None
        snap["quiet_convention"] = None
    return snap


def cpu_jiffies() -> tuple[int, ...] | None:
    """The aggregate ``/proc/stat`` cpu line, or None off Linux.

    Fields are user nice system idle iowait irq softirq steal ...
    """
    try:
        with open("/proc/stat", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("cpu "):
                    return tuple(int(v) for v in line.split()[1:])
    except Exception:
        pass
    return None


def steal_percent(before: tuple[int, ...] | None,
                  after: tuple[int, ...] | None) -> float | None:
    """Share of CPU time the hypervisor gave to somebody else.

    On an oversubscribed cloud instance this inflates every wall-clock
    timing while appearing nowhere in the result. It is the confound a
    cloud record most needs and a laptop record cannot have -- ``None``
    off Linux, where the counter does not exist.
    """
    if not before or not after or len(before) < 8 or len(after) < 8:
        return None
    delta = [now - was for was, now in zip(before, after)]
    total = sum(delta)
    return 100.0 * delta[7] / total if total > 0 else None


def thermal_state() -> dict | None:
    """Whether the OS is currently limiting CPU speed.

    macOS reports this through ``pmset -g therm``. A sustained benchmark
    on a laptop can throttle partway through, which looks exactly like a
    slow machine unless the record says otherwise. ``None`` where the
    platform offers no equivalent.
    """
    if platform.system() != "Darwin":
        return None
    out = _run(["pmset", "-g", "therm"])
    if not out:
        return None
    state: dict = {"raw_lines": [], "cpu_speed_limit": None, "throttled": False}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        state["raw_lines"].append(line)
        if "CPU_Speed_Limit" in line:
            try:
                state["cpu_speed_limit"] = int(line.split("=")[1].strip())
            except Exception:
                pass
    limit = state["cpu_speed_limit"]
    state["throttled"] = limit is not None and limit < 100
    return state


def host_label(override: str | None = None) -> str:
    """A stable, shareable name for the measuring machine.

    Derived from the hardware, never from the hostname -- these records
    are committed to a public repository, and a personal machine name
    does not belong there. Two runs on the same machine produce the same
    label, which is what makes them groupable; two different cloud
    instances produce different labels, which is the point.
    """
    if override:
        slug = override
    else:
        parts = [platform.system().lower(), cpu_name() or "unknown-cpu",
                 f"{os.cpu_count() or 0}c"]
        slug = "-".join(parts)
    slug = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")
    return slug[:80] or "unknown-host"


def assess_quality(load: dict, steal: float | None,
                   thermal: dict | None) -> dict:
    """Is this record fit to compare against another one?

    Deliberately advisory. A benchmark should *always* write its record;
    what it must never do is let the reader assume a number was taken on
    an idle machine. Today's failure mode was not measuring a busy host
    -- it was that nothing recorded the busyness.
    """
    reasons: list[str] = []
    # `looks_quiet` is composite: a busy start OR foreign load mid-run sets
    # it False. Only the first of those is a statement about load average,
    # so gate on the load verdict itself -- otherwise a genuinely idle
    # 0.7-on-32-cores gets printed as a reason the host was contended.
    load_verdict = load.get("looks_quiet_before",
                            load.get("looks_quiet"))
    if load_verdict is False:
        before = load.get("load_average_before")
        la = (before or load.get("load_average") or [None])[0]
        which = "before the run" if before else "after the run"
        if la is not None:
            reasons.append(f"load average {la:.1f} {which} on "
                           f"{load.get('logical_cores')} cores")
        else:
            reasons.append("host reported busy")
    if load.get("others_busy_after"):
        reasons.append(
            f"other processes took {load['cpu_percent_others']:.0f}% "
            "CPU during the run")
    if steal is not None and steal >= 2.0:
        reasons.append(f"hypervisor steal {steal:.1f}%")
    if thermal and thermal.get("throttled"):
        reasons.append(f"CPU speed limited to {thermal['cpu_speed_limit']}%")

    if reasons:
        verdict = "contended"
    elif load.get("looks_quiet") is True:
        verdict = "quiet"
    else:
        verdict = "unknown"
    return {
        "verdict": verdict,
        "reasons": reasons,
        "comparable": verdict == "quiet",
        "note": ("'quiet' means no contention signal was detected, not that "
                 "the machine was idle; 'unknown' means the platform did "
                 "not report enough to tell."),
    }


def backend_info() -> dict:
    """Which compute path will actually run, and how it is configured.

    ``backend`` is one of ``metal``, ``cuda`` or ``numpy``. The first two
    mean MLX is present *and* its default device is that accelerator;
    MLX being importable is not enough, and is not what a reader of a
    throughput number needs to know.

    On CUDA, ``tf32_enabled`` is load-bearing. MLX's CUDA backend runs
    float32 matmul in TF32 by default, which costs roughly 1000x the
    matmul error and measurably degrades fitted parameters, so a record
    taken with TF32 on is not comparable to one taken with it off.

    ``mlx_cache_limit_bytes`` is load-bearing too. On the one CUDA host
    measured, disabling MLX's buffer cache moved three solvers by 1.4x to
    7.4x, in both directions, while leaving every fitted value unchanged
    (``docs/upstream-issues/mlx-buffer-cache-ab-2026-09-23.md``). ``0``
    means the cache is off; ``None`` means MLX is not in use.
    """
    info: dict = {
        "backend": "numpy",
        "device_name": None,
        "mlx_version": None,
        "mlx_default_device": None,
        "tf32_enabled": None,
        "mlx_cache_limit_bytes": None,
        "TOYOMACRO_DISABLE_MLX": os.environ.get("TOYOMACRO_DISABLE_MLX") or None,
    }
    try:
        from toyomacro.voigtfit._mlx_support import mlx_usable
        usable = mlx_usable()
    except Exception:
        usable = False
    if not usable:
        info["device_name"] = cpu_name()
        return info

    try:
        import mlx.core as mx
        info["mlx_version"] = getattr(mx, "__version__", None)
        info["mlx_default_device"] = str(mx.default_device())
        is_cuda = bool(getattr(mx, "cuda", None) and mx.cuda.is_available())
        is_metal = bool(getattr(mx, "metal", None) and mx.metal.is_available())
        if is_cuda:
            info["backend"] = "cuda"
            # MLX_ENABLE_TF32=0 and NVIDIA_TF32_OVERRIDE=0 are measured
            # equivalents; either one disables it.
            disabled = (os.environ.get("MLX_ENABLE_TF32") == "0"
                        or os.environ.get("NVIDIA_TF32_OVERRIDE") == "0")
            info["tf32_enabled"] = not disabled
        elif is_metal:
            info["backend"] = "metal"
        # MLX has no getter for the limit. `set_cache_limit` returns the
        # previous value, and setting that straight back leaves it as it was.
        try:
            limit = mx.set_cache_limit(2 ** 62)
            mx.set_cache_limit(limit)
            info["mlx_cache_limit_bytes"] = int(limit)
        except Exception:
            pass
        try:
            dev = mx.device_info()          # mx.metal.device_info() is deprecated
        except Exception:
            dev = {}
        info["device_name"] = dev.get("device_name") or cpu_name()
    except Exception:
        pass
    return info


def git_state() -> dict:
    """The tree this ran on: a commit and a dirty flag, with no path.

    Resolved from the installed package's own location, so it reports
    ``None`` cleanly when the package was installed from a wheel rather
    than run out of a checkout.
    """
    state: dict = {"git_commit": None, "git_tracked_dirty": None,
                   "git_dirty_files": None, "harness_tracked_at_commit": None,
                   "untracked_files_present": None}
    try:
        here = Path(__file__).resolve()
        inside = _run(["git", "rev-parse", "--is-inside-work-tree"], here.parent)
        if inside != "true":
            return state
        state["git_commit"] = _run(["git", "rev-parse", "HEAD"], here.parent) or None
        # -uno would hide the harness itself when it is not yet committed,
        # so the dirty list would say "only two READMEs changed" about a
        # tree carrying three new modules. Ask for untracked files too.
        dirty = _run_raw(["git", "status", "--porcelain"], here.parent)
        lines = [ln for ln in dirty.splitlines() if ln.strip()]
        tracked = [ln for ln in lines if not ln.lstrip().startswith("??")]
        state["git_tracked_dirty"] = bool(tracked)
        state["untracked_files_present"] = any(
            ln.lstrip().startswith("??") for ln in lines)
        # Names only, and capped: a dirty list is a hint about what the
        # number means, not a diff. Porcelain status codes are two columns
        # wide, so the leading space of an unstaged change must survive.
        state["git_dirty_files"] = lines[:20]
        # Does the commit actually contain the module that produced this
        # record? If not, checking it out will not reproduce the run.
        rel = "src/toyomacro/voigtfit/benchmarks/bench_platform.py"
        listed = _run(["git", "ls-tree", "--full-tree", "HEAD", "--name-only",
                       rel], here.parent)
        state["harness_tracked_at_commit"] = bool(listed)
    except Exception:
        pass
    return state


def package_versions() -> dict:
    """Versions of this package and its published dependencies.

    These come from installed distribution metadata, which in a
    development checkout can lag the source tree -- an editable install
    keeps reporting the version it was installed at until it is
    reinstalled. Treat ``git.git_commit`` as the authoritative statement
    of what ran, and ``versions['toyomacro']`` as a hint.
    """
    out: dict = {}
    for dist in _TRACKED_DISTRIBUTIONS:
        try:
            out[dist] = md.version(dist)
        except Exception:
            out[dist] = None
    try:
        import h5py
        out["hdf5_library"] = h5py.version.hdf5_version
    except Exception:
        out["hdf5_library"] = None
    return out


def sanitize_command(argv: list[str]) -> str:
    """Render an argv as a command line with every path reduced to a name.

    A record is committed, so it must not carry the layout of the
    machine that produced it -- and paths arrive not only as ``argv[0]``
    but as option values (``--out /home/me/runs/x.json``) and as
    ``--key=/path`` pairs. Every token that looks like a path is reduced
    to its final component; everything else is passed through.

    Both separators are stripped regardless of the host, rather than
    deferring to ``PurePath``: a record produced on Windows may well be
    read, diffed or re-serialised on Linux, and the guarantee should not
    depend on where that happens.
    """
    def basename(text: str) -> str:
        """Last path component; '<dir>' for a token that ends in a separator.

        A trailing separator would otherwise basename to the empty string
        and delete the argument, leaving a command that cannot be read.
        """
        stripped = text.replace("\\", "/").rstrip("/")
        if not stripped:
            return "<root>"
        return stripped.rsplit("/", 1)[-1] or "<dir>"

    out = []
    for token in argv:
        # Reduce any token carrying a separator FIRST. Splitting on "="
        # before that leaked the key half of "--out=/a/b" forms whose
        # directory itself contained "=", e.g. "/home/me/a=b/rec.json".
        if "/" in token or "\\" in token:
            key, sep, value = token.partition("=")
            if sep and ("/" in value or "\\" in value) and not (
                    "/" in key or "\\" in key):
                out.append(f"{key}={basename(value)}")
            else:
                out.append(basename(token))
        else:
            out.append(token)
    return " ".join(out)


def environment(command: str | None = None, origin: str | None = None,
                host: str | None = None, steal: float | None = None,
                load_before: dict | None = None) -> dict:
    """The full environment block: machine, OS, backend, versions, tree.

    ``command`` should be how the run was invoked. If omitted it is
    reconstructed from ``sys.argv`` with every path-looking token
    reduced to its bare name, so the record does not disclose the
    filesystem it was produced on.

    ``origin`` says where the run was *launched from*, which a cloud
    container cannot work out for itself and which no amount of
    introspection recovers. ``host`` overrides the hardware-derived
    label. ``steal`` is the hypervisor steal measured across the
    benchmark itself -- pass it in, because a reading taken here would
    cover the wrong window.
    """
    if command is None:
        command = sanitize_command(sys.argv)
    load = load_snapshot(exclude_self=True)
    if load_before is not None:
        # **The BEFORE sample decides.** A load average taken afterwards
        # necessarily contains the benchmark that just ran, so on a CPU
        # backend it measures our own work: a NumPy run on an idle
        # 32-core host recorded load 8.4 with zero other processes and
        # graded itself contended, while a GPU run on a machine with
        # other applications open graded itself quiet. Judging on the
        # trailing sample would systematically label CPU records
        # incomparable -- in a harness whose purpose is comparing
        # backends.
        load["load_average_before"] = load_before.get("load_average")
        load["cpu_percent_before"] = load_before.get("cpu_percent")
        load["top_processes_before"] = load_before.get("top_processes")
        load["looks_quiet_after_incl_self"] = load.get("looks_quiet")
        load["looks_quiet_before"] = load_before.get("looks_quiet")
        load["looks_quiet"] = load_before.get("looks_quiet")
        # The after sample still earns a say, but only through processes
        # that are not ours: that is work which appeared mid-run.
        others = load.get("cpu_percent_others")
        cores = load.get("logical_cores")
        if others is not None and cores:
            load["others_busy_after"] = others > 25.0 * cores
            load["others_busy_convention"] = (
                "cpu_percent_others > 25 * logical_cores, where 100 = one "
                "core; a convention, not a calibrated threshold")
            if load["others_busy_after"] and load["looks_quiet"]:
                load["looks_quiet"] = False
        load["sampled"] = (
            "before and after; the verdict is the BEFORE sample, because "
            "the after sample contains this benchmark")
    else:
        load["sampled"] = "after the measurements only - the verdict may " \
                          "reflect this benchmark's own load"
    thermal = thermal_state()
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.datetime.now(
            datetime.UTC).isoformat(timespec="seconds"),
        "timestamp_local": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "host_label": host_label(host),
        "origin": origin,
        "os": _os_description(),
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "python": sys.version.split()[0],
        "command": command,
        "versions": package_versions(),
        "backend": backend_info(),
        "load": load,
        "steal_percent": steal,
        "thermal": thermal,
        "quality": assess_quality(load, steal, thermal),
        "git": git_state(),
    }


def record_filename(env: dict) -> str:
    """The name a record is stored under: time, machine, origin.

    One file per run, never an appended log. Four machines writing into
    one repository will otherwise conflict on every push; distinct
    filenames cannot.
    """
    # Normalise the offset to "Z" BEFORE stripping separators, or the
    # colons inside "+00:00" are removed first and the suffix never matches.
    stamp = (env.get("timestamp_utc") or "").replace("+00:00", "Z")
    stamp = stamp.replace("-", "").replace(":", "")
    if not stamp:
        stamp = datetime.datetime.now(datetime.UTC).strftime(
            "%Y%m%dT%H%M%SZ")
    origin = re.sub(r"[^a-z0-9]+", "-", (env.get("origin") or "unknown").lower())
    return f"{stamp}__{env.get('host_label', 'unknown-host')}__from-{origin}.json"


def write_record(report: dict, directory: str | Path) -> Path:
    """Write ``report`` under ``directory`` with a conflict-free name."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / record_filename(report.get("environment", {}))
    path.write_text(json.dumps(report, indent=1, default=float) + "\n",
                    encoding="utf-8")
    return path


def summarize(times_s: list[float], n_items: int) -> dict:
    """Rates from per-repetition timings: median, and the observed spread.

    The median is the headline. ``rate_min``/``rate_max`` bound the
    repetitions *of this invocation only* -- they share the machine
    state, so they understate the uncertainty you would see across
    sessions. Compare records by their median, and check ``load``.
    """
    if not times_s:
        raise ValueError("no timings: benchmark recorded zero repetitions")
    rates = [n_items / t for t in times_s]
    return {
        "n_items": n_items,
        "timings_s": [round(t, 6) for t in times_s],
        "time_median_s": statistics.median(times_s),
        "rate_median": statistics.median(rates),
        "rate_min": min(rates),
        "rate_max": max(rates),
        "aggregation": "median",
    }


def measure_projection_kernel(n_spectra: int = 4_000_000, repeats: int = 9,
                              warmup: int = 2, n_energy: int = 151,
                              n_comp: int = 3,
                              chi2_sample: float = 1e-4) -> dict:
    """The amplitude-only projection kernel: ``A = Y @ W`` plus sampled chi2.

    This is the package's headline throughput figure and the "fit-only"
    bar of the paper's Figure 1. It deliberately matches the memory
    access model used for the theoretical bound -- read Y, write A and
    chi2, with chi2 evaluated on a 0.01% sample -- so that the measured
    rate can be compared against the memory-bandwidth ceiling.

    It is **not** a peak fit: no position or width is recovered. Quoting
    it as a fitting rate overstates the package by two orders of
    magnitude.

    Runs on MLX where usable and on NumPy otherwise, so the same figure
    exists for every host. The backend is recorded, never assumed.
    """
    rng = np.random.default_rng(0)
    n_sample = max(1, int(n_spectra * chi2_sample))
    idx_np = rng.choice(n_spectra, n_sample, replace=False)

    try:
        from toyomacro.voigtfit._mlx_support import mlx_usable
        use_mlx = mlx_usable()
    except Exception:
        use_mlx = False

    if use_mlx:
        import mlx.core as mx
        Y = mx.array(rng.random((n_spectra, n_energy), dtype=np.float32))
        W = mx.array(rng.random((n_energy, n_comp), dtype=np.float32))
        Phi = mx.array(rng.random((n_comp, n_energy), dtype=np.float32))
        idx = mx.array(idx_np)
        mx.eval(Y, W, Phi, idx)

        def step() -> None:
            A = Y @ W
            resid = Y[idx] - A[idx] @ Phi
            chi2 = mx.sum(resid * resid, axis=1)
            mx.eval(A, chi2)
    else:
        Y = rng.random((n_spectra, n_energy), dtype=np.float32)
        W = rng.random((n_energy, n_comp), dtype=np.float32)
        Phi = rng.random((n_comp, n_energy), dtype=np.float32)
        idx = idx_np

        def step() -> None:
            A = Y @ W
            resid = Y[idx] - A[idx] @ Phi
            np.sum(resid * resid, axis=1)

    import time
    timings: list[float] = []
    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        step()
        dt = time.perf_counter() - t0
        if i >= warmup:
            timings.append(dt)

    return {
        "solver": "projection_kernel",
        "description": "amplitude-only projection kernel "
                       "(A = Y @ W, chi2 on a 0.01% sample)",
        "problem": f"random float32 data, {n_energy} channels x {n_comp} "
                   "components; no parameter recovery",
        "dtype": "float32",
        "n_energy": n_energy,
        "n_comp": n_comp,
        "warmup": warmup,
        "repeats": repeats,
        **summarize(timings, n_spectra),
    }
