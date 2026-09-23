"""The cross-platform record schema: portability, and what it must not leak.

These records are committed, so the tests that matter most here are the
negative ones -- a field that quietly starts carrying a home directory
is a repository-wide problem, not a benchmark problem.
"""
from __future__ import annotations

import json
import os
import platform

import pytest

from toyomacro.voigtfit.benchmarks import record


class TestCommandSanitisation:
    """No record may disclose the filesystem it was produced on."""

    def test_program_path_reduced_to_name(self):
        got = record.sanitize_command(["/home/someone/scratch/bench.py", "--n", "5"])
        assert got == "bench.py --n 5"

    def test_option_value_path_reduced_to_name(self):
        got = record.sanitize_command(
            ["bench.py", "--out", "/Users/someone/private/runs/rec.json"])
        assert got == "bench.py --out rec.json"

    def test_equals_form_reduced_to_name(self):
        got = record.sanitize_command(["bench.py", "--out=/var/tmp/deep/rec.json"])
        assert got == "bench.py --out=rec.json"

    def test_windows_separator_reduced(self):
        got = record.sanitize_command([r"C:\Users\someone\bench.py", "--flag"])
        assert got == "bench.py --flag"

    def test_non_paths_pass_through(self):
        argv = ["bench.py", "--n-batch", "200000", "--no-parity"]
        assert record.sanitize_command(argv) == " ".join(argv)


class TestEnvironmentBlock:
    def test_has_the_load_bearing_fields(self):
        env = record.environment()
        for key in ("schema_version", "os", "machine", "cpu", "python",
                    "command", "versions", "backend", "load", "git"):
            assert key in env, f"missing {key}"
        assert env["schema_version"] == record.SCHEMA_VERSION

    def test_carries_no_absolute_path_anywhere(self):
        """The whole block, serialised, must contain no home directory."""
        env = record.environment()
        blob = json.dumps(env)
        home = os.path.expanduser("~")
        assert home not in blob, "record leaks the home directory"
        for marker in ("/Users/", "/home/", "site-packages", "\\Users\\"):
            assert marker not in blob, f"record leaks a filesystem path: {marker}"

    def test_records_no_sibling_project_version(self):
        """Only this package and its published dependencies are recorded."""
        versions = record.package_versions()
        assert "toyomacro" in versions
        assert "deppro" not in versions

    def test_git_state_has_no_repository_path(self):
        """The point is the absence of a path, not an exact key set."""
        state = record.git_state()
        assert "repo" not in state
        assert {"git_commit", "git_tracked_dirty", "git_dirty_files"} <= set(state)
        blob = json.dumps(state)
        assert os.path.expanduser("~") not in blob
        for marker in ("/Users/", "/home/", "\\Users\\"):
            assert marker not in blob

    def test_os_description_is_not_macos_specific(self):
        """The old helper said '(macOS )' on every non-Apple host."""
        desc = record._os_description()
        assert desc
        if platform.system() != "Darwin":
            assert "macOS" not in desc


class TestBackendInfo:
    def test_reports_one_of_three_backends(self):
        info = record.backend_info()
        assert info["backend"] in {"metal", "cuda", "numpy"}

    def test_tf32_is_only_meaningful_on_cuda(self):
        info = record.backend_info()
        if info["backend"] != "cuda":
            assert info["tf32_enabled"] is None

    def test_numpy_backend_when_mlx_is_disabled(self, monkeypatch):
        monkeypatch.setenv("TOYOMACRO_DISABLE_MLX", "1")
        # _mlx_support caches its probe, so this asserts the env var is
        # recorded rather than that the backend flips within one process.
        assert record.backend_info()["TOYOMACRO_DISABLE_MLX"] == "1"


class TestLoadSnapshot:
    def test_reports_cores_and_a_quiet_verdict(self):
        snap = record.load_snapshot(interval=0.05, top_n=3)
        assert snap["logical_cores"] == os.cpu_count()
        assert snap["looks_quiet"] in (True, False, None)

    def test_load_average_absent_only_where_the_platform_lacks_it(self):
        snap = record.load_snapshot(interval=0.05)
        if hasattr(os, "getloadavg"):
            assert snap["load_average"] is not None
            assert len(snap["load_average"]) == 3
        else:
            assert snap["load_average"] is None

    def test_top_processes_carry_names_not_paths(self):
        snap = record.load_snapshot(interval=0.05, top_n=5)
        for row in snap["top_processes"] or []:
            assert "/" not in row["name"] and "\\" not in row["name"]


class TestSummarize:
    def test_median_and_bounds(self):
        out = record.summarize([1.0, 2.0, 4.0], n_items=100)
        assert out["time_median_s"] == 2.0
        assert out["rate_median"] == 50.0
        assert out["rate_min"] == 25.0
        assert out["rate_max"] == 100.0
        assert out["aggregation"] == "median"

    def test_empty_timings_is_an_error_not_a_nan(self):
        with pytest.raises(ValueError, match="zero repetitions"):
            record.summarize([], n_items=100)


class TestProjectionKernel:
    """The headline kernel must exist on every backend, not just MLX."""

    def test_runs_and_reports_a_rate(self):
        rec = record.measure_projection_kernel(
            n_spectra=20_000, repeats=2, warmup=1)
        assert rec["solver"] == "projection_kernel"
        assert rec["rate_median"] > 0
        assert rec["n_items"] == 20_000
        assert len(rec["timings_s"]) == 2

    def test_runs_on_the_numpy_path(self, monkeypatch):
        """Force the NumPy branch and check it still produces a record."""
        monkeypatch.setattr(
            "toyomacro.voigtfit._mlx_support.mlx_usable", lambda: False)
        rec = record.measure_projection_kernel(
            n_spectra=5_000, repeats=1, warmup=0)
        assert rec["rate_median"] > 0


class TestStealTime:
    """The confound a cloud record needs and a laptop cannot have."""

    def test_absent_platform_yields_none(self):
        assert record.steal_percent(None, None) is None

    def test_computed_from_the_delta_not_the_absolute(self):
        # steal is field 7; 50 of 200 jiffies elapsed => 25%
        before = (100, 0, 50, 1000, 0, 0, 0, 100)
        after = (150, 0, 50, 1100, 0, 0, 0, 150)
        assert record.steal_percent(before, after) == pytest.approx(25.0)

    def test_zero_when_the_hypervisor_took_nothing(self):
        before = (100, 0, 50, 1000, 0, 0, 0, 7)
        after = (200, 0, 50, 1000, 0, 0, 0, 7)
        assert record.steal_percent(before, after) == 0.0

    def test_no_elapsed_time_is_not_a_division_error(self):
        same = (1, 2, 3, 4, 5, 6, 7, 8)
        assert record.steal_percent(same, same) is None


class TestHostLabel:
    def test_derived_from_hardware_not_hostname(self):
        label = record.host_label()
        assert label and label == label.lower()
        assert " " not in label
        import socket
        host = socket.gethostname().split(".")[0].lower()
        if len(host) > 3:                      # ignore trivially short names
            assert host not in label, "host label leaks the machine's name"

    def test_stable_across_calls(self):
        assert record.host_label() == record.host_label()

    def test_override_is_slugified(self):
        assert record.host_label("Cloud Box #2!") == "cloud-box-2"


class TestQualityVerdict:
    def test_busy_host_is_contended_with_a_reason(self):
        load = {"looks_quiet": False, "load_average": [12.4, 0, 0],
                "logical_cores": 16}
        q = record.assess_quality(load, None, None)
        assert q["verdict"] == "contended"
        assert q["comparable"] is False
        assert q["reasons"]

    def test_steal_alone_condemns_an_otherwise_idle_host(self):
        load = {"looks_quiet": True, "load_average": [0.1, 0, 0],
                "logical_cores": 4}
        q = record.assess_quality(load, 7.5, None)
        assert q["verdict"] == "contended"
        assert any("steal" in r for r in q["reasons"])

    def test_thermal_throttle_condemns_too(self):
        load = {"looks_quiet": True, "load_average": [0.1, 0, 0],
                "logical_cores": 16}
        q = record.assess_quality(load, None,
                                  {"throttled": True, "cpu_speed_limit": 60})
        assert q["verdict"] == "contended"

    def test_quiet_when_nothing_fires(self):
        load = {"looks_quiet": True, "load_average": [0.2, 0, 0],
                "logical_cores": 16}
        q = record.assess_quality(load, 0.1, {"throttled": False})
        assert q["verdict"] == "quiet"
        assert q["comparable"] is True

    def test_unknown_is_not_silently_quiet(self):
        """A platform that cannot report load must not pass as idle."""
        q = record.assess_quality({"looks_quiet": None}, None, None)
        assert q["verdict"] == "unknown"
        assert q["comparable"] is False


class TestRecordFilename:
    def test_utc_stamp_host_and_origin(self):
        env = {"timestamp_utc": "2026-09-21T03:45:00+00:00",
               "host_label": "darwin-apple-m3-max-16c", "origin": "mac"}
        assert record.record_filename(env) == (
            "20260921T034500Z__darwin-apple-m3-max-16c__from-mac.json")

    def test_offset_normalised_before_separators_are_stripped(self):
        """Strip order matters: '+00:00' must not become '+0000'."""
        env = {"timestamp_utc": "2026-09-21T03:45:00+00:00",
               "host_label": "h", "origin": "o"}
        assert "+" not in record.record_filename(env)

    def test_missing_origin_is_named_not_omitted(self):
        env = {"timestamp_utc": "2026-09-21T03:45:00+00:00",
               "host_label": "h", "origin": None}
        assert "from-unknown" in record.record_filename(env)

    def test_two_hosts_never_collide(self):
        stamp = "2026-09-21T03:45:00+00:00"
        a = record.record_filename({"timestamp_utc": stamp, "host_label": "mac",
                                    "origin": "mac"})
        b = record.record_filename({"timestamp_utc": stamp, "host_label": "xeon",
                                    "origin": "windows"})
        assert a != b


class TestProcessRedaction:
    def test_names_kept_by_default(self, monkeypatch):
        monkeypatch.delenv("TOYOMACRO_BENCH_REDACT_PROCESSES", raising=False)
        snap = record.load_snapshot(interval=0.05, top_n=3)
        assert snap["processes_redacted"] is False

    def test_names_replaced_when_asked(self, monkeypatch):
        monkeypatch.setenv("TOYOMACRO_BENCH_REDACT_PROCESSES", "1")
        snap = record.load_snapshot(interval=0.05, top_n=3)
        assert snap["processes_redacted"] is True
        for i, row in enumerate(snap["top_processes"] or []):
            assert row["name"] == f"process-{i + 1}"
            assert isinstance(row["cpu_percent"], float)


class TestWriteRecord:
    def test_writes_under_the_generated_name(self, tmp_path):
        report = {"environment": record.environment(origin="test-origin")}
        path = record.write_record(report, tmp_path)
        assert path.parent == tmp_path
        assert path.name.endswith("__from-test-origin.json")
        assert json.loads(path.read_text(encoding="utf-8"))[
            "environment"]["origin"] == "test-origin"

    def test_written_record_leaks_no_path(self, tmp_path):
        report = {"environment": record.environment(origin="mac")}
        blob = record.write_record(report, tmp_path).read_text(
            encoding="utf-8")
        assert os.path.expanduser("~") not in blob
        for marker in ("/Users/", "/home/", "site-packages"):
            assert marker not in blob


class TestSanitiserGapsFoundInAudit:
    """Forms the first sanitiser leaked. All must reduce to bare names."""

    @pytest.mark.parametrize("argv", [
        ["bench.py", "--out", "/Users/someone/a=b/rec.json"],
        ["/Users/someone/my=proj/bench.py", "--flag"],
        ["bench.py", "--records-dir", "/Users/someone/runs/"],
        ["bench.py", "--out=/home/someone/a=b/rec.json"],
        ["/home/someone/x=y/z=w/bench.py"],
    ])
    def test_no_home_directory_survives(self, argv):
        got = record.sanitize_command(argv)
        for marker in ("/Users/", "/home/", "someone"):
            assert marker not in got, f"{argv} leaked: {got!r}"

    def test_trailing_separator_keeps_the_argument(self):
        """Basenaming a trailing slash to '' deleted the argument."""
        got = record.sanitize_command(["bench.py", "--records-dir", "/a/b/runs/"])
        assert got == "bench.py --records-dir runs"

    def test_root_is_named_not_emptied(self):
        assert record.sanitize_command(["bench.py", "--out", "/"]) == \
            "bench.py --out <root>"


class TestQualityCoversTheMeasurementWindow:
    def test_all_three_load_averages_must_be_low(self):
        """Grading on the 1-minute figure alone let a busy host pass."""
        load = {"looks_quiet": None, "load_average": [3.2, 5.1, 9.7],
                "logical_cores": 16}
        # 3.2 < 4.0 but 5.1 and 9.7 are not: this host is not quiet.
        snap = dict(load)
        snap["looks_quiet"] = all(v < 0.25 * 16 for v in snap["load_average"])
        assert snap["looks_quiet"] is False

    def test_a_busy_start_condemns_a_quiet_finish(self):
        """environment() must not grade only the moment after the work."""
        before = {"load_average": [12.0, 12.0, 12.0], "looks_quiet": False,
                  "cpu_percent": 90.0, "top_processes": []}
        env = record.environment(origin="t", load_before=before)
        assert env["load"]["looks_quiet"] is False
        assert env["quality"]["verdict"] == "contended"
        assert "before" in env["load"]["sampled"]

    def test_says_so_when_only_the_end_was_sampled(self):
        """Without a before-sample the verdict may include our own load."""
        env = record.environment(origin="t")
        sampled = env["load"]["sampled"]
        assert "after" in sampled and "own load" in sampled


class TestGitStateProvenance:
    def test_says_whether_the_commit_contains_the_harness(self):
        """A commit that lacks bench_platform.py cannot reproduce the run."""
        state = record.git_state()
        assert "harness_tracked_at_commit" in state
        assert state["harness_tracked_at_commit"] in (True, False, None)

    def test_reports_untracked_files_separately(self):
        state = record.git_state()
        assert "untracked_files_present" in state

    def test_porcelain_status_columns_survive(self):
        """stdout.strip() ate the leading space of the first entry."""
        for line in record.git_state()["git_dirty_files"] or []:
            assert len(line) > 3, line
            assert line[2] == " ", f"status column mangled: {line!r}"


class TestVerdictUsesTheBeforeSample:
    """Found on a 32-core Windows host: a CPU run condemned by its own load.

    The trailing load average necessarily contains the benchmark that
    just ran. Judging on it labelled a NumPy run on an idle machine
    `contended` (load 8.4, zero other processes) while a GPU run on a
    machine with other applications open passed as `quiet` — in a
    harness whose whole purpose is comparing backends.
    """

    IDLE_BEFORE = {"load_average": [0.7, 0.24, 0.08], "looks_quiet": True,
                   "cpu_percent": 2.0, "top_processes": []}

    @pytest.fixture
    def quiet_after(self, monkeypatch):
        """Pin the after-sample too.

        `environment()` takes its own trailing sample, so asserting a
        verdict while only controlling the before-sample asserts
        something about whatever else the test machine happens to be
        running. This test failed on a busy macOS host for exactly that
        reason after passing on a quiet one — the assertion was true by
        circumstance, not by the behaviour it names.
        """
        monkeypatch.setattr(
            record, "load_snapshot",
            lambda *a, **k: {"load_average": [0.5, 0.5, 0.5],
                             "logical_cores": 32, "cpu_percent": 1.0,
                             "cpu_percent_others": 0.0, "top_processes": [],
                             "looks_quiet": True, "processes_redacted": False})

    def test_our_own_load_does_not_condemn_the_record(self, quiet_after):
        env = record.environment(origin="windows", load_before=self.IDLE_BEFORE)
        assert env["load"]["looks_quiet"] is True
        assert env["quality"]["verdict"] == "quiet"

    def test_foreign_load_after_the_start_still_condemns(self, monkeypatch):
        """The other half: the after-sample keeps its say over non-self work."""
        monkeypatch.setattr(
            record, "load_snapshot",
            lambda *a, **k: {"load_average": [0.5, 0.5, 0.5],
                             "logical_cores": 4, "cpu_percent": 90.0,
                             "cpu_percent_others": 350.0, "top_processes": [],
                             "looks_quiet": True, "processes_redacted": False})
        env = record.environment(origin="t", load_before=self.IDLE_BEFORE)
        assert env["quality"]["verdict"] == "contended"
        assert any("other processes" in r
                   for r in env["quality"]["reasons"])

    def test_a_busy_start_still_condemns(self):
        busy = {"load_average": [12.0, 12.0, 12.0], "looks_quiet": False,
                "cpu_percent": 90.0, "top_processes": []}
        env = record.environment(origin="windows", load_before=busy)
        assert env["quality"]["verdict"] == "contended"
        assert any("before the run" in r for r in env["quality"]["reasons"])

    def test_the_trailing_verdict_is_kept_for_the_reader(self):
        env = record.environment(origin="windows", load_before=self.IDLE_BEFORE)
        assert "looks_quiet_after_incl_self" in env["load"]
        assert "BEFORE sample" in env["load"]["sampled"]

    def test_other_processes_appearing_mid_run_still_count(self):
        load = {"looks_quiet": True, "load_average_before": [0.5, 0.5, 0.5],
                "logical_cores": 4, "others_busy_after": True,
                "cpu_percent_others": 300.0}
        q = record.assess_quality(load, None, None)
        assert q["verdict"] == "contended"
        assert any("other processes" in r for r in q["reasons"])

    def test_a_sample_can_exclude_our_own_process(self):
        snap = record.load_snapshot(interval=0.05, top_n=5, exclude_self=True)
        assert "cpu_percent_others" in snap
        names = [r["name"] for r in snap["top_processes"] or []]
        assert not any("pytest" in n for n in names), names


class TestRoundTwoAuditFindings:
    """Defects the fixes for round 1 introduced. Both were real."""

    def test_excluded_sample_stays_sorted(self):
        """`exclude_self` rebuilt the list from unsorted iteration order,
        so the heaviest consumer could fall past `top_n` — in the one
        sample the verdict is based on."""
        snap = record.load_snapshot(interval=0.2, top_n=5, exclude_self=True)
        rates = [p["cpu_percent"] for p in snap["top_processes"] or []]
        assert rates == sorted(rates, reverse=True), rates

    def test_both_samples_are_sorted(self):
        for excl in (False, True):
            snap = record.load_snapshot(interval=0.1, top_n=5,
                                        exclude_self=excl)
            rates = [p["cpu_percent"] for p in snap["top_processes"] or []]
            assert rates == sorted(rates, reverse=True), (excl, rates)

    def test_idle_load_is_not_given_as_a_reason_for_contention(self):
        """Foreign load mid-run condemned the record, and the reason
        printed was an emphatically quiet load average."""
        load = {"looks_quiet": False, "looks_quiet_before": True,
                "load_average_before": [0.7, 0.5, 0.4], "logical_cores": 32,
                "others_busy_after": True, "cpu_percent_others": 3000.0}
        q = record.assess_quality(load, None, None)
        assert q["verdict"] == "contended"
        assert not any("load average" in r for r in q["reasons"]), q["reasons"]
        assert any("other processes" in r for r in q["reasons"])

    def test_a_busy_start_still_cites_its_load(self):
        load = {"looks_quiet": False, "looks_quiet_before": False,
                "load_average_before": [12.0, 12.0, 12.0],
                "logical_cores": 32}
        q = record.assess_quality(load, None, None)
        assert any("12.0 before the run" in r for r in q["reasons"])

    def test_the_before_verdict_survives_the_composite(self):
        """Overwriting looks_quiet destroyed the before-sample's own answer."""
        before = {"load_average": [0.5, 0.5, 0.5], "looks_quiet": True,
                  "cpu_percent": 1.0, "top_processes": []}
        env = record.environment(origin="t", load_before=before)
        assert env["load"]["looks_quiet_before"] is True
        assert "looks_quiet_after_incl_self" in env["load"]

    def test_the_others_threshold_is_recorded(self):
        before = {"load_average": [0.5, 0.5, 0.5], "looks_quiet": True,
                  "cpu_percent": 1.0, "top_processes": []}
        env = record.environment(origin="t", load_before=before)
        if env["load"].get("others_busy_after") is not None:
            assert "others_busy_convention" in env["load"]


class TestWindowsIdleProcess:
    """Found on native Windows: every record condemned by idle time.

    Windows reports idle time as a process — "System Idle Process",
    PID 0 — which psutil returns at close to 100% per core. Counted as
    foreign load, it tripped `others_busy_after` on a completely idle
    host, so `test_our_own_load_does_not_condemn_the_record` failed
    there and PR #26's Windows job went red.
    """

    class _FakeProc:
        def __init__(self, pid, name, pct):
            self.pid, self.info, self._pct = pid, {"name": name}, pct

        def cpu_percent(self):
            return self._pct

    def test_pid_zero_is_always_idle(self):
        assert 0 in record._idle_pids([])

    def test_the_windows_idle_process_is_named_as_idle(self):
        procs = [self._FakeProc(0, "System Idle Process", 1500.0),
                 self._FakeProc(4, "System", 2.0),
                 self._FakeProc(9, "python.exe", 100.0)]
        idle = record._idle_pids(procs)
        assert 0 in idle
        assert 4 not in idle, "the Windows kernel does real work"
        assert 9 not in idle

    def test_idle_time_is_not_foreign_load(self, monkeypatch):
        """The whole bug: 1500% of idle counted as somebody else's work."""
        procs = [self._FakeProc(0, "System Idle Process", 1500.0),
                 self._FakeProc(9999, "other.exe", 5.0)]

        class _FakePsutil:
            @staticmethod
            def process_iter(_fields):
                return procs

            @staticmethod
            def cpu_percent(interval=0.0):
                return 3.0

            class Process:
                def __init__(self, *a):
                    pass

                def children(self, recursive=False):
                    return []

        monkeypatch.setitem(__import__("sys").modules, "psutil", _FakePsutil)
        snap = record.load_snapshot(interval=0.0, top_n=5, exclude_self=True)
        assert snap["cpu_percent_others"] == 5.0, snap["cpu_percent_others"]
        names = [p["name"] for p in snap["top_processes"]]
        assert "System Idle Process" not in names

    def test_an_idle_windows_host_is_not_condemned(self, monkeypatch):
        """End to end: the assertion that failed on Windows."""
        procs = [self._FakeProc(0, "System Idle Process", 1580.0)]

        class _FakePsutil:
            @staticmethod
            def process_iter(_fields):
                return procs

            @staticmethod
            def cpu_percent(interval=0.0):
                return 1.0

            @staticmethod
            def virtual_memory():
                class _M:
                    total = 32 * 2**30
                return _M()

            class Process:
                def __init__(self, *a):
                    pass

                def children(self, recursive=False):
                    return []

        monkeypatch.setitem(__import__("sys").modules, "psutil", _FakePsutil)
        before = {"load_average": [0.7, 0.24, 0.08], "looks_quiet": True,
                  "cpu_percent": 1.0, "top_processes": []}
        env = record.environment(origin="windows", load_before=before)
        assert env["load"].get("others_busy_after") is not True
        assert env["quality"]["verdict"] == "quiet", env["quality"]["reasons"]


class TestAggregateRuns:
    """Combining separate runs, and the ratio that is the point of it."""

    @staticmethod
    def _run(rate, lo, hi, verdict="quiet", solver="k", timings=None):
        return {
            "problem": {"n_batch": 200000, "input_sha256": "abc"},
            "results": [{"solver": solver, "backend": "metal",
                         "rate_median": rate, "rate_min": lo, "rate_max": hi,
                         "timings_s": timings, "mae_amp": 0.02}],
            "environment": {"host_label": "h", "origin": "mac",
                            "quality": {"verdict": verdict}},
        }

    def test_median_is_taken_across_runs(self):
        from toyomacro.voigtfit.benchmarks import bench_platform as bp
        agg = bp.aggregate_runs([self._run(10e6, 9.9e6, 10.1e6),
                                 self._run(20e6, 19.9e6, 20.1e6),
                                 self._run(30e6, 29.9e6, 30.1e6)])
        r = agg["results"][0]
        assert r["rate_median"] == 20e6
        assert r["rate_min"] == 10e6 and r["rate_max"] == 30e6
        assert r["runs"] == 3
        assert "across" in r["aggregation"]

    def test_understates_by_flags_an_optimistic_single_record(self):
        """The Windows case: tight within-run range, wide across runs."""
        from toyomacro.voigtfit.benchmarks import bench_platform as bp
        agg = bp.aggregate_runs([
            self._run(9.44e6, 9.3e6, 9.5e6),
            self._run(9.27e6, 9.2e6, 9.4e6),
            self._run(18.65e6, 18.45e6, 19.06e6),   # within-run 1.03x
        ])
        r = agg["results"][0]
        assert r["across_run_spread"] == pytest.approx(18.65 / 9.27, rel=1e-3)
        assert r["understates_by"] > 1.5, r["understates_by"]

    def test_per_run_timings_survive_aggregation(self):
        """A median hides a step inside a run; the repetitions show it.

        The first aggregates dropped them, so the advice to read
        per-repetition timings could not be followed on exactly the
        records meant to be most trustworthy.
        """
        from toyomacro.voigtfit.benchmarks import bench_platform as bp
        step = [0.41, 0.75, 0.41, 0.41]            # one slow repetition
        flat = [0.40, 0.41, 0.40, 0.41]
        broken = self._run(None, None, None, timings=[9.9, 9.9])   # no rate
        agg = bp.aggregate_runs([self._run(9.7e6, 5.3e6, 9.8e6, timings=step),
                                 broken,
                                 self._run(9.8e6, 9.7e6, 9.9e6, timings=flat)])
        r = agg["results"][0]
        # Aligned with per_run_rate_median: the run without a rate is
        # dropped from both lists, not from one of them.
        assert r["per_run_rate_median"] == [9.7e6, 9.8e6]
        assert r["per_run_timings_s"] == [step, flat]

    def test_a_healthy_host_reports_about_one(self):
        from toyomacro.voigtfit.benchmarks import bench_platform as bp
        agg = bp.aggregate_runs([self._run(10.0e6, 9.5e6, 10.5e6),
                                 self._run(10.1e6, 9.6e6, 10.6e6),
                                 self._run(9.9e6, 9.4e6, 10.4e6)])
        assert agg["results"][0]["understates_by"] < 1.2

    def test_the_aggregate_is_only_as_clean_as_its_dirtiest_run(self):
        from toyomacro.voigtfit.benchmarks import bench_platform as bp
        agg = bp.aggregate_runs([self._run(10e6, 9e6, 11e6, "quiet"),
                                 self._run(10e6, 9e6, 11e6, "contended")])
        assert agg["all_runs_quiet"] is False
        assert agg["environment"]["quality"]["verdict"] == "contended"
        assert agg["environment"]["quality"]["comparable"] is False
        assert any("run 2" in r for r in
                   agg["environment"]["quality"]["reasons"])

    def test_all_quiet_aggregates_to_quiet(self):
        from toyomacro.voigtfit.benchmarks import bench_platform as bp
        agg = bp.aggregate_runs([self._run(10e6, 9e6, 11e6),
                                 self._run(10e6, 9e6, 11e6)])
        assert agg["environment"]["quality"]["verdict"] == "quiet"

    def test_it_notices_the_runs_solved_different_problems(self):
        from toyomacro.voigtfit.benchmarks import bench_platform as bp
        a, b = self._run(10e6, 9e6, 11e6), self._run(10e6, 9e6, 11e6)
        b["problem"]["input_sha256"] = "different"
        assert bp.aggregate_runs([a, b])["problem_identical_across_runs"] is False
        assert bp.aggregate_runs([a, a])["problem_identical_across_runs"] is True

    def test_the_aggregate_carries_a_filename_and_a_host(self):
        from toyomacro.voigtfit.benchmarks import bench_platform as bp
        agg = bp.aggregate_runs([self._run(10e6, 9e6, 11e6)])
        assert agg["environment"]["host_label"] == "h"
        assert "from-mac" in record.record_filename(agg["environment"])

    def test_no_records_is_an_error(self):
        from toyomacro.voigtfit.benchmarks import bench_platform as bp
        with pytest.raises(ValueError, match="no run records"):
            bp.aggregate_runs([])


class TestMlxVersionGate:
    """The mlx#3858 note should fire only on an MLX that predates the fix."""

    @pytest.mark.parametrize("version,older", [
        ("0.32.0", True),
        ("0.32.1", False),                       # the first release with mlx#3929
        # A dev build of the fixing release: the string cannot say whether
        # the fix is in it (the one it was verified on was), so warn.
        ("0.32.1.dev20260806+4652b008", True),
        ("0.32.1rc1", True),
        ("0.32.1+local", False),                 # a local label is not a pre-release
        ("0.32.2", False),
        ("0.33", False),                         # two-part versions pad, not fail
        ("1.0.0", False),
        (None, True),                            # unknown: warn rather than stay silent
        ("garbage", True),
    ])
    def test_version_comparison(self, version, older):
        from toyomacro.voigtfit.benchmarks import bench_platform as bp
        assert bp._mlx_older_than(version, (0, 32, 1)) is older
