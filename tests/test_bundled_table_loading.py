"""The bundled reference tables are data, not a rebuildable cache.

``src/toyomacro/data/_cache/`` is named "cache" for historical reasons.
Its contents ship with the package, are reviewed, are documented in
``docs/DATA_SOURCES.md``, and are pinned by tests. The CSV sources they
were originally built from are not in this repository and have moved on
independently since.

Rebuilding a table from one of those CSVs therefore replaces a reference
dataset with an unreviewed one. These tests pin that it cannot happen by
accident -- only on an explicit opt-in, and loudly.
"""

from __future__ import annotations

import json
import warnings

import pytest

from toyomacro.data import paths


@pytest.fixture
def empty_cache(tmp_path, monkeypatch):
    """Point the loaders at an empty directory, leaving the real one alone."""
    monkeypatch.setattr(paths, "get_cache_dir", lambda: tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "loader,cache_name",
    [
        (paths.load_compound_data, "compounds.json"),
        (paths.load_binding_energy_data, "binding_energy.json"),
        (paths.load_cross_section_data, "cross_section.json"),
    ],
)
def test_missing_table_raises_rather_than_rebuilding(loader, cache_name, empty_cache, monkeypatch):
    """A missing table is an error, not a silent rebuild from CSV."""
    monkeypatch.delenv(paths.REGENERATE_ENV_VAR, raising=False)

    with pytest.raises(FileNotFoundError) as excinfo:
        loader()

    message = str(excinfo.value)
    assert cache_name in message
    assert paths.REGENERATE_ENV_VAR in message, "the error must say how to opt in"
    assert not (empty_cache / cache_name).exists(), "nothing may be written on the error path"


def test_rebuild_requires_opt_in_and_warns(empty_cache, monkeypatch):
    """With the opt-in set, the rebuild happens but says it is unreviewed."""
    monkeypatch.setenv(paths.REGENERATE_ENV_VAR, "1")
    monkeypatch.setattr(
        paths, "_csv_to_json_compounds", lambda path: {"Xx": {"Nv": 1.0}}
    )
    monkeypatch.setattr(paths, "get_common_data_path", lambda: empty_cache)

    with pytest.warns(UserWarning, match="unreviewed"):
        data = paths.load_compound_data()

    assert data == {"Xx": {"Nv": 1.0}}
    assert json.loads((empty_cache / "compounds.json").read_text()) == data


def test_present_table_is_returned_untouched(empty_cache, monkeypatch):
    """The opt-in never overrides a table that is present."""
    monkeypatch.setenv(paths.REGENERATE_ENV_VAR, "1")
    (empty_cache / "compounds.json").write_text(json.dumps({"Yy": {"Nv": 2.0}}))

    def fail(path):  # pragma: no cover - must not be reached
        raise AssertionError("CSV converter called although the table exists")

    monkeypatch.setattr(paths, "_csv_to_json_compounds", fail)

    assert paths.load_compound_data() == {"Yy": {"Nv": 2.0}}


def test_clear_cache_refuses_to_delete_shipped_data(empty_cache):
    """It used to unlink every table, including two nothing rebuilt."""
    (empty_cache / "compounds.json").write_text("{}")

    with pytest.raises(RuntimeError, match="bundled reference tables"):
        paths.clear_cache()

    assert (empty_cache / "compounds.json").exists()


def test_regenerate_cache_requires_the_opt_in(empty_cache, monkeypatch):
    monkeypatch.delenv(paths.REGENERATE_ENV_VAR, raising=False)

    with pytest.raises(RuntimeError, match=paths.REGENERATE_ENV_VAR):
        paths.regenerate_cache()


def test_regenerate_cache_warns_before_each_overwrite(empty_cache, monkeypatch):
    """The warning must fire before reviewed data is replaced, not after.

    ``README.md`` and the docstrings promise it; without it the opt-in is
    the only signal, and an opt-in set once in a shell persists.
    """
    monkeypatch.setenv(paths.REGENERATE_ENV_VAR, "1")
    monkeypatch.setattr(paths, "get_common_data_path", lambda: empty_cache)
    monkeypatch.setattr(paths, "_get_trzh2018_xlsx_path", lambda: empty_cache / "no-2018.xlsx")
    monkeypatch.setattr(paths, "_get_trzh2019_xlsx_path", lambda: empty_cache / "no-2019.xlsx")
    for csv_name in (
        "BindingEnergyTable.csv",
        "CompoundTable.csv",
        "CrossSectionTable_Yeh=Lindau.csv",
    ):
        (empty_cache / csv_name).write_text("")

    monkeypatch.setattr(paths, "_csv_to_json_binding_energy", lambda p: {"t": "be"})
    monkeypatch.setattr(paths, "_csv_to_json_compounds", lambda p: {"t": "compounds"})
    monkeypatch.setattr(paths, "_csv_to_json_cross_section", lambda p: {"t": "xs"})

    with pytest.warns(UserWarning) as record:
        results = paths.regenerate_cache()

    messages = [str(w.message) for w in record]
    assert len(messages) == 3, messages
    for cache_name in ("binding_energy.json", "compounds.json", "cross_section.json"):
        assert any(cache_name in m for m in messages), cache_name
        assert results[cache_name] == "rebuilt"
    assert all("unreviewed" in m for m in messages)


def test_regenerate_cache_warns_before_it_writes(empty_cache, monkeypatch):
    """Ordering, not just presence: turn the warning into an error.

    If the warning were emitted after the write, the file would already
    be on disk when the error propagates.
    """
    monkeypatch.setenv(paths.REGENERATE_ENV_VAR, "1")
    monkeypatch.setattr(paths, "get_common_data_path", lambda: empty_cache)
    (empty_cache / "BindingEnergyTable.csv").write_text("")
    monkeypatch.setattr(paths, "_csv_to_json_binding_energy", lambda p: {"t": "be"})

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning):
            paths.regenerate_cache()

    assert not (empty_cache / "binding_energy.json").exists()


def test_regenerate_cache_keeps_tables_whose_source_is_missing(empty_cache, monkeypatch):
    """A partial rebuild must not leave the package with data deleted.

    This is the failure the old pair produced together: clear_cache()
    unlinked every table, then the first loader raised, and nothing was
    restored.
    """
    monkeypatch.setenv(paths.REGENERATE_ENV_VAR, "1")
    monkeypatch.setattr(paths, "get_common_data_path", lambda: empty_cache / "no-such-dir")
    monkeypatch.setattr(paths, "_get_trzh2018_xlsx_path", lambda: empty_cache / "no-2018.xlsx")
    monkeypatch.setattr(paths, "_get_trzh2019_xlsx_path", lambda: empty_cache / "no-2019.xlsx")
    for name in ("compounds.json", "scofield.json"):
        (empty_cache / name).write_text('{"kept": true}')

    results = paths.regenerate_cache()

    assert all("skipped" in v or "not rebuilt" in v for v in results.values()), results
    for name in ("compounds.json", "scofield.json"):
        assert json.loads((empty_cache / name).read_text()) == {"kept": True}


def test_regenerate_cache_reports_the_two_it_does_not_handle(empty_cache, monkeypatch):
    """scofield/trzhaskovskaya were silently absent from the old list."""
    monkeypatch.setenv(paths.REGENERATE_ENV_VAR, "1")
    monkeypatch.setattr(paths, "get_common_data_path", lambda: empty_cache / "no-such-dir")
    monkeypatch.setattr(paths, "_get_trzh2018_xlsx_path", lambda: empty_cache / "no-2018.xlsx")
    monkeypatch.setattr(paths, "_get_trzh2019_xlsx_path", lambda: empty_cache / "no-2019.xlsx")

    results = paths.regenerate_cache()

    assert set(results) == {
        "binding_energy.json",
        "compounds.json",
        "cross_section.json",
        "trzh2018_haxpes.json",
        "trzh2019_inner.json",
        "scofield.json",
        "trzhaskovskaya.json",
    }


def test_every_shipped_table_is_present_on_disk():
    """The real directory, not a fixture: nothing above may have eaten it."""
    cache_dir = paths.get_cache_dir()
    for name in (
        "binding_energy.json",
        "compounds.json",
        "cross_section.json",
        "scofield.json",
        "trzhaskovskaya.json",
        "trzh2018_haxpes.json",
        "trzh2019_inner.json",
    ):
        assert (cache_dir / name).exists(), f"{name} missing from the bundled data"


def test_shipped_compound_table_is_the_reviewed_one():
    """Guard the specific divergence that motivated this module.

    The maintainer's current CompoundTable.csv holds 166 entries and no
    key named ``Si3N4`` (it is split by phase there). The shipped table
    holds 109 and does have it. If a rebuild ever lands silently, this
    is what it looks like.
    """
    data = paths.load_compound_data()
    assert len(data) == 109
    assert "Si3N4" in data
