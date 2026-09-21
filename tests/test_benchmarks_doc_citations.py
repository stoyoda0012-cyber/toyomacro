"""Every figure `docs/BENCHMARKS.md` attributes to a record must be in it.

This gate exists because the same mistake was made three times, by three
different edits to the same document, and caught each time only by an
independent audit reading the JSON by hand:

- a section cited `488 M` from a record that had been deleted and
  replaced by one reading `495.1 M`;
- it attributed a host's load to `fileproviderd`, `cloudd` and
  `corespotlightd` when the record's own `top_processes` said
  `mediaanalysisd`;
- it said neither record had a modified code path when the one
  supplying the headline had `record.py` — where the measured kernel is
  defined — in its dirty list.

The failure is always the same shape: prose written from memory about
records, rather than from the records. `tests/README.md` is already
guarded this way for test counts. This does the same for measurements.

**The contract.** A table row in `docs/BENCHMARKS.md` that names a
record file must quote figures that record actually contains. A row
that has no record must say so in the words "no record". There is no
third option — an unattributed number in a provenance document is the
defect this file exists to prevent.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "BENCHMARKS.md"
RECORD_DIRS = (REPO / "benchmarks" / "records", REPO / "paper" / "figures" / "results")

#: `…131732Z…__from-mac.json`, or a plain `figure1_throughput.json`.
#: Deliberately permissive: a stricter class silently failed to match
#: record names containing a hyphen, which made this gate inert for
#: exactly the rows it most needed to check.
_CITATION = re.compile(r"`([^`]*\.json)`")
#: A figure the row presents as measured: **497.6 M** or **418–431 M**.
_FIGURE = re.compile(r"\*\*([\d.]+)(?:[–-][\d.]+)?\s*M\*\*")


def _doc_rows() -> list[str]:
    return [ln for ln in DOC.read_text().splitlines()
            if ln.startswith("|") and ln.count("|") >= 3]


def _resolve(cited: str) -> Path | None:
    """Find the file a citation names, allowing the `…abbreviated…` form.

    Record filenames are long, so the document elides the middle with
    `…`. Treat each ellipsis as a wildcard: every fragment must appear,
    in order. Dropping the ellipses instead would look for a substring
    that no filename contains.
    """
    fragments = [f for f in cited.strip("`").split("…") if f]
    if not fragments:
        return None
    for directory in RECORD_DIRS:
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            pos, ok = 0, True
            for fragment in fragments:
                found = path.name.find(fragment, pos)
                if found < 0:
                    ok = False
                    break
                pos = found + len(fragment)
            if ok:
                return path
    return None


def _rates(record: dict) -> list[float]:
    """Every throughput this record reports, in spectra per second."""
    out = []
    for result in record.get("results", []):
        for key in ("rate_median", "throughput_spec_per_s",
                    "throughput_spec_per_s_median"):
            if result.get(key):
                out.append(float(result[key]))
    for block in record.values():                 # figure1-style: keyed dict
        if isinstance(block, dict) and block.get("throughput_spec_per_s_median"):
            out.append(float(block["throughput_spec_per_s_median"]))
    return out


CITING_ROWS = [ln for ln in _doc_rows() if _CITATION.search(ln)]


def test_the_document_still_cites_records_at_all():
    """A guard on the guard: if the table is restructured, fail loudly
    rather than silently testing nothing."""
    assert CITING_ROWS, (
        f"No rows in {DOC.name} cite a .json record. Either the tables "
        "changed shape or the citations were dropped; this gate is now "
        "inert and must be updated deliberately.")


@pytest.mark.parametrize("row", CITING_ROWS, ids=lambda r: r[:48])
def test_cited_record_exists(row):
    cited = _CITATION.search(row).group(1)
    assert _resolve(cited) is not None, (
        f"{DOC.name} cites `{cited}`, which is not in "
        f"{[str(d.relative_to(REPO)) for d in RECORD_DIRS]}. A record that "
        "was re-taken or deleted leaves the prose quoting numbers nothing "
        "produces.")


@pytest.mark.parametrize("row", CITING_ROWS, ids=lambda r: r[:48])
def test_cited_figure_is_in_the_cited_record(row):
    cited = _CITATION.search(row).group(1)
    path = _resolve(cited)
    if path is None:
        pytest.skip("covered by test_cited_record_exists")
    figures = _FIGURE.findall(row)
    if not figures:
        return                       # a row may cite a record without a figure
    rates = _rates(json.loads(path.read_text()))
    assert rates, f"{path.name} reports no throughput to check against"
    for figure in figures:
        want = float(figure) * 1e6
        # The document rounds to one decimal in millions, so accept the
        # rounding interval and nothing wider.
        assert any(abs(rate - want) < 0.05e6 for rate in rates), (
            f"{DOC.name} quotes {figure} M against `{cited}`, which reports "
            f"{sorted(round(r / 1e6, 2) for r in rates)} M. The prose and "
            "the record disagree.")


def test_rows_without_a_record_say_so():
    """An unattributed measurement is the defect, not a formatting nit."""
    offenders = []
    for row in _doc_rows():
        if _CITATION.search(row) or not _FIGURE.search(row):
            continue
        if "no record" in row.lower():
            continue
        offenders.append(row.strip())
    assert not offenders, (
        "These rows present a bold measurement with neither a cited record "
        "nor the words 'no record':\n  " + "\n  ".join(offenders))


def test_every_committed_record_is_readable_and_self_describing():
    directory = REPO / "benchmarks" / "records"
    records = sorted(directory.glob("*.json"))
    assert records, f"no records under {directory.relative_to(REPO)}"
    for path in records:
        record = json.loads(path.read_text())
        env = record.get("environment", {})
        assert record.get("schema_version"), f"{path.name}: no schema_version"
        assert env.get("quality", {}).get("verdict") in {
            "quiet", "contended", "unknown"}, f"{path.name}: no verdict"
        assert env.get("host_label"), f"{path.name}: no host_label"
