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
        state = record.git_state()
        assert set(state) == {"git_commit", "git_tracked_dirty", "git_dirty_files"}
        assert "repo" not in state

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
        assert json.loads(path.read_text())["environment"]["origin"] == "test-origin"

    def test_written_record_leaks_no_path(self, tmp_path):
        report = {"environment": record.environment(origin="mac")}
        blob = record.write_record(report, tmp_path).read_text()
        assert os.path.expanduser("~") not in blob
        for marker in ("/Users/", "/home/", "site-packages"):
            assert marker not in blob
