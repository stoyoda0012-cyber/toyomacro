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
#: A figure the row presents as measured. Matches a single value
#: (``497.6 M``), a range (``418–431 M``) and a list of separate runs
#: (``9.44 / 9.27 / 17.30 M``), bold or not.
#:
#: The first version of this required a single number immediately before
#: a bold ``M``, which silently matched none of the slash-separated rows
#: added later — the gate went inert on exactly the rows it was extended
#: to cover, and a mutation test passed because nothing was checked.
#: The unit is captured, because a row may quote ``581 k`` where another
#: quotes ``9.5 M``. Before ``k`` was accepted, every sub-megaspectrum
#: figure in the document was simply invisible to this gate -- which is
#: the same failure as the two above, waiting for the first row to use
#: one. Uppercase ``K`` is accepted too: an audit found that rewriting a
#: checked row as ``999 K`` dropped it from the gate silently. The price
#: is that a cited row mentioning a temperature (``300 K``) or an image
#: size (``4K``) would be read as a rate and fail. That fails loudly,
#: which is the right way round; today no cited row does.
_FIGURE = re.compile(
    r"((?:\d+(?:\.\d+)?\s*[/–-]\s*)*\d+(?:\.\d+)?)\s*([MkK])\b")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SCALE = {"M": 1e6, "k": 1e3, "K": 1e3}


#: "4 M spectra" is a batch size, not a rate. Only throughputs are
#: checked against a record's reported rates.
_NOT_A_RATE = re.compile(r"\b[MkK]\s+(?:spectra|fits|px|pixels)\b")


def _figures(row: str) -> list[tuple[str, float]]:
    """Every throughput a row presents, as written (to keep precision).

    Paired with its scale, so ``581 k`` and ``9.5 M`` are both checked
    against the record in spectra per second.
    """
    plain = _NOT_A_RATE.sub("", row.replace("**", ""))
    out: list[tuple[str, float]] = []
    for span, unit in _FIGURE.findall(plain):
        out.extend((n, _SCALE[unit]) for n in _NUMBER.findall(span))
    return out


#: A heading that declares its whole section record-free. §3 is exactly
#: that — "Published numbers without a committed record" — so its rows
#: are not required to repeat the disclaimer each time.
_RECORD_FREE_SECTION = "without a committed record"


def _doc_rows(skip_record_free_sections: bool = True) -> list[tuple[str, str]]:
    """``(row, record cited by its heading)`` for every table row.

    §2 names its record in the heading above the table rather than in
    each row, so a row's attribution is inherited from the nearest
    preceding heading that names a ``.json`` file.
    """
    rows, exempt, heading_records = [], False, ()
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            exempt = _RECORD_FREE_SECTION in line.lower()
            # A heading may name a matched pair, as §2 does for the MLX
            # and NumPy records; a figure below it may come from either.
            heading_records = tuple(_CITATION.findall(line))
            continue
        if line.startswith("|") and line.count("|") >= 3:
            if exempt and skip_record_free_sections:
                continue
            rows.append((line, heading_records))
    return rows


def _tolerance(figure_text: str, scale: float) -> float:
    """The rounding interval of the precision the document actually used.

    `478 M` is rounded to a whole million and must accept 477.5-478.5;
    `497.6 M` is rounded to a tenth and must accept far less. A single
    fixed tolerance either rejects honest rounding or waves through a
    figure that disagrees. `581 k` is the same rule one scale down.
    """
    if "." in figure_text:
        decimals = len(figure_text.split(".")[1])
        return 0.5 * 10 ** (-decimals) * scale
    return 0.5 * scale


def _resolve(cited: str) -> Path | None:
    """Find the file a citation names, allowing the `…abbreviated…` form.

    Record filenames are long, so the document elides the middle with
    `…`. Treat each ellipsis as a wildcard: every fragment must appear,
    in order. Dropping the ellipses instead would look for a substring
    that no filename contains.
    """
    # A citation may carry a directory ("paper/figures/results/x.json");
    # records are located by filename, so match on the last component.
    bare = cited.strip("`").rsplit("/", 1)[-1]
    fragments = [f for f in bare.split("…") if f]
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
        for key in ("rate_median", "rate_min", "rate_max",
                    "throughput_spec_per_s", "throughput_spec_per_s_median"):
            if result.get(key):
                out.append(float(result[key]))
    for block in record.values():                 # figure1-style: keyed dict
        if isinstance(block, dict) and block.get("throughput_spec_per_s_median"):
            out.append(float(block["throughput_spec_per_s_median"]))
    return out


#: Rows attributed to a record, inline or by heading.
CITING_ROWS = [(ln, tuple(_CITATION.findall(ln)) or head)
               for ln, head in _doc_rows()
               if _CITATION.search(ln) or (head and _figures(ln))]


def test_the_document_still_cites_records_at_all():
    """A guard on the guard: if the table is restructured, fail loudly
    rather than silently testing nothing."""
    assert CITING_ROWS, (
        f"No rows in {DOC.name} cite a .json record. Either the tables "
        "changed shape or the citations were dropped; this gate is now "
        "inert and must be updated deliberately.")


@pytest.mark.parametrize("row,cited", CITING_ROWS, ids=lambda r: str(r)[:40])
def test_cited_record_exists(row, cited):
    missing = [c for c in cited if _resolve(c) is None]
    assert not missing, (
        f"{DOC.name} cites {missing}, which are not in "
        f"{[str(d.relative_to(REPO)) for d in RECORD_DIRS]}. A record that "
        "was re-taken or deleted leaves the prose quoting numbers nothing "
        "produces.")


@pytest.mark.parametrize("row,cited", CITING_ROWS, ids=lambda r: str(r)[:40])
def test_cited_figure_is_in_the_cited_record(row, cited):
    paths = [p for p in (_resolve(c) for c in cited) if p is not None]
    if not paths:
        pytest.skip("covered by test_cited_record_exists")
    figures = _figures(row)
    if not figures:
        return                       # a row may cite a record without a figure
    rates = [r for path in paths
             for r in _rates(json.loads(path.read_text(encoding="utf-8")))]
    assert rates, f"{[p.name for p in paths]} report no throughput to check"
    for figure, scale in figures:
        want, tol = float(figure) * scale, _tolerance(figure, scale)
        unit = "M" if scale == 1e6 else "k"
        assert any(abs(rate - want) <= tol for rate in rates), (
            f"{DOC.name} quotes {figure} {unit} against {list(cited)}, which "
            f"report {sorted(round(r / 1e6, 3) for r in rates)} M. The prose "
            "and the record disagree.")


def test_rows_without_a_record_say_so():
    """An unattributed measurement is the defect, not a formatting nit."""
    offenders = []
    for row, heading_record in _doc_rows():
        if _CITATION.search(row) or heading_record or not _figures(row):
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
        record = json.loads(path.read_text(encoding="utf-8"))
        env = record.get("environment", {})
        assert record.get("schema_version"), f"{path.name}: no schema_version"
        assert env.get("quality", {}).get("verdict") in {
            "quiet", "contended", "unknown"}, f"{path.name}: no verdict"
        assert env.get("host_label"), f"{path.name}: no host_label"


class TestTextIOIsEncodingIndependent:
    """The gate itself broke Windows CI by reading the doc in the locale
    encoding. `docs/BENCHMARKS.md` contains em dashes and ellipses, so on
    a cp1252 host `read_text()` raised `UnicodeDecodeError` at *import*
    time and the whole module errored out of collection.

    `CHANGELOG.md` already records this class of bug against a different
    file — "one read a source file in the platform's preferred encoding
    rather than UTF-8". Enforcing it with ruff needs preview mode, which
    turns on 479 other findings, so it is enforced here instead for the
    files that write and read committed records.
    """

    #: Everything that reads or writes a record, plus this gate.
    #:
    #: `test_benchmark_record.py` was missing from this list, and a run
    #: on a Japanese-locale Windows host found two unguarded reads there
    #: that cp932 would break. A guard is only worth what it covers.
    GUARDED = (
        REPO / "src/toyomacro/voigtfit/benchmarks/record.py",
        REPO / "src/toyomacro/voigtfit/benchmarks/bench_platform.py",
        REPO / "src/toyomacro/voigtfit/benchmarks/summarize_records.py",
        REPO / "src/toyomacro/voigtfit/tests/test_benchmark_record.py",
        Path(__file__),
    )

    _TEXT_IO = re.compile(r"(?:\.read_text|\.write_text|(?<![\w.])open)\(")

    def test_the_encoding_actually_matters_here(self):
        """If the doc were pure ASCII this guard would prove nothing."""
        raw = DOC.read_bytes()
        assert any(b > 127 for b in raw), (
            f"{DOC.name} is pure ASCII; this test no longer demonstrates "
            "anything and the encoding argument is untested.")
        with pytest.raises(UnicodeDecodeError):
            raw.decode("cp1252")

    def test_the_gate_read_the_document_at_import(self):
        """Collection succeeding is the assertion; this makes it explicit."""
        assert CITING_ROWS, "the document was not read, or cites nothing"

    @pytest.mark.parametrize("path", GUARDED, ids=lambda p: p.name)
    def test_every_text_io_names_its_encoding(self, path):
        offenders = []
        source = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(source, 1):
            if not self._TEXT_IO.search(line):
                continue
            # The call may wrap; look at the next two lines for the kwarg.
            window = " ".join(source[number - 1:number + 2])
            if "encoding=" in window or '"rb"' in window or "'rb'" in window:
                continue
            offenders.append(f"{path.name}:{number}: {line.strip()}")
        assert not offenders, (
            "Text I/O without an explicit encoding reads in the platform's "
            "locale encoding, which is cp1252 on Windows and raises on any "
            "non-ASCII byte:\n  " + "\n  ".join(offenders))
