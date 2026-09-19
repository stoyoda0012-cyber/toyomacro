"""
tests/README.md states exact counts. This keeps them exact.
===========================================================

The inventory gives a grand total, a count per location, per section and
per file, and a contract / paper-reproduction split. Each of those has
drifted at least once while every test stayed green, because nothing
compared the document with what pytest collects. This module does.

How, and why it is not circular:

- The reference is a fresh ``pytest --collect-only`` of the whole suite
  in a subprocess, not the items of the running session, so it gives the
  same answer under ``-k``, for a single file, or under xdist.
- The file counts itself: its own row is one of those checked, so adding
  a test here is caught like anywhere else.
- Collection does not depend on the MLX extra (those tests are marked,
  not removed), but it does depend on ``mcp``: ``test_mcp_server.py``
  skips at import without it, which turns its tests into one skipped
  module. The inventory is stated for an install that has ``mcp`` --
  what CI installs -- and the checks skip, rather than fail, without it.

A failure lists every disagreement with the collected value, so the fix
is to copy those numbers into the README.
"""

import importlib.util
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "tests" / "README.md"
LOCATIONS = {"tests/": "tests/", "src/toyomacro/voigtfit/tests/": "src/toyomacro/voigtfit/tests/"}


def _number(text):
    return int(text.replace(",", ""))


def _collect(*extra):
    """Node ids of the whole suite. ``addopts`` is cleared so -q prints ids."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=",
         "-p", "no:cacheprovider", *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return [line for line in result.stdout.splitlines() if "::" in line]


@pytest.fixture(scope="module")
def readme():
    if not README.exists():
        pytest.skip("tests/README.md is not part of this installation")
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def collected(readme):
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("the inventory is stated for an install with the 'mcp' extra")
    ids = _collect()
    per_file = Counter(node.split("::")[0] for node in ids)
    names = Counter(Path(path).name for path in per_file)
    assert max(names.values()) == 1, "the inventory lists bare file names; they must be unique"
    return ids, per_file


def _location(path):
    return "tests/" if path.startswith("tests/") else "src/toyomacro/voigtfit/tests/"


def test_grand_total_and_locations(readme, collected):
    ids, per_file = collected
    problems = []

    stated = re.search(r"\*\*([\d,]+) automated tests\*\* across \*\*(\d+) files\*\*", readme)
    assert stated, "the opening sentence of tests/README.md changed shape"
    if (_number(stated.group(1)), int(stated.group(2))) != (len(ids), len(per_file)):
        problems.append(
            f"opening sentence says {stated.group(1)} tests in {stated.group(2)} files; "
            f"collected {len(ids):,} in {len(per_file)}"
        )

    for location in LOCATIONS:
        files = sum(1 for path in per_file if _location(path) == location)
        tests = sum(n for path, n in per_file.items() if _location(path) == location)
        row = re.search(rf"^\| `{re.escape(location)}` \|.*\| (\d+) \| ([\d,]+) \|$", readme, re.M)
        heading = re.search(rf"^## .*`{re.escape(location)}` \(([\d,]+)\)$", readme, re.M)
        assert row and heading, f"no table row or heading for {location}"
        if (int(row.group(1)), _number(row.group(2))) != (files, tests):
            problems.append(f"table row for {location}: collected {files} files, {tests:,} tests")
        if _number(heading.group(1)) != tests:
            problems.append(f"heading for {location} says {heading.group(1)}; collected {tests:,}")

    assert not problems, "\n".join(problems)


def test_every_file_is_listed_with_its_collected_count(readme, collected):
    _, per_file = collected
    by_name = {Path(path).name: n for path, n in per_file.items()}
    rows = {name: int(n) for n, name in re.findall(r"^\| (\d+) \| `([^`]+\.py)` \|", readme, re.M)}

    problems = [f"{name}: README {rows[name]}, collected {by_name[name]}"
                for name in sorted(rows.keys() & by_name.keys()) if rows[name] != by_name[name]]
    problems += [f"{name}: collected {by_name[name]}, not listed"
                 for name in sorted(by_name.keys() - rows.keys())]
    problems += [f"{name}: listed, not collected" for name in sorted(rows.keys() - by_name.keys())]
    assert not problems, "\n".join(problems)


def test_section_headings_are_the_sum_of_their_rows(readme):
    """Needs no collection: the document must at least agree with itself."""
    problems, title, stated, total = [], None, 0, 0

    def close():
        if title is not None and stated != total:
            problems.append(f"'{title}' says {stated}, its rows sum to {total}")

    for line in readme.splitlines():
        heading = re.match(r"^### (.*) \(([\d,]+)\)\s*$", line)
        if heading or line.startswith("## "):
            close()
            title = None
        if heading:
            title, stated, total = heading.group(1), _number(heading.group(2)), 0
        row = re.match(r"^\| (\d+) \| `[^`]+\.py` \|", line)
        if row and title is not None:
            total += int(row.group(1))
    close()
    assert not problems, "\n".join(problems)


def test_contract_and_reproduction_split(readme, collected):
    """The split is defined by the ``-k`` filter the README itself prints."""
    ids, _ = collected
    keyword = re.search(r'^pytest -k "([^"]+)"\s*$', readme, re.M)
    contract = re.search(r"\*\*Contract / regression\*\* \(([\d,]+) tests, (\d+)%\)", readme)
    reproduction = re.search(r"\*\*Paper reproduction\*\* \(([\d,]+) tests, (\d+)%\)", readme)
    assert keyword and contract and reproduction, "the split paragraph changed shape"

    selected = len(_collect("-k", keyword.group(1)))
    expected = {"Contract / regression": (selected, contract),
                "Paper reproduction": (len(ids) - selected, reproduction)}
    problems = []
    for label, (count, match) in expected.items():
        share = 100.0 * count / len(ids)
        if _number(match.group(1)) != count:
            problems.append(f"{label}: README {match.group(1)}, the filter gives {count:,}")
        if abs(int(match.group(2)) - share) >= 1.0:
            problems.append(f"{label}: README {match.group(2)}%, actual {share:.1f}%")
    assert not problems, "\n".join(problems)
