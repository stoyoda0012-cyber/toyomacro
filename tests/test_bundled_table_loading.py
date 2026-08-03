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
