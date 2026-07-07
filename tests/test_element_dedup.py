"""Tests for centralized element name dedup and ElementInfo utilities."""

import pytest

from toyomacro.io.hdf5_cache import (
    ElementInfo,
    build_element_infos,
    dedup_element_names,
    dedup_names,
    parse_element_label,
)

# ---------------------------------------------------------------------------
# parse_element_label
# ---------------------------------------------------------------------------


class TestParseElementLabel:
    def test_si2p(self):
        assert parse_element_label("Si2p") == ("Si", "2p")

    def test_o1s(self):
        assert parse_element_label("O1s") == ("O", "1s")

    def test_ta4f(self):
        assert parse_element_label("Ta4f") == ("Ta", "4f")

    def test_c1s(self):
        assert parse_element_label("C1s") == ("C", "1s")

    def test_n1s(self):
        assert parse_element_label("N1s") == ("N", "1s")

    def test_al2p(self):
        assert parse_element_label("Al2p") == ("Al", "2p")

    def test_fe2p3(self):
        """Fe2p3/2 style label — orbital part is '2p3'."""
        assert parse_element_label("Fe2p3") == ("Fe", "2p3")

    def test_unknown_returns_label_empty(self):
        assert parse_element_label("unknown") == ("unknown", "")

    def test_empty_string(self):
        assert parse_element_label("") == ("", "")

    def test_just_element_symbol(self):
        assert parse_element_label("Si") == ("Si", "")


# ---------------------------------------------------------------------------
# dedup_names (core algorithm)
# ---------------------------------------------------------------------------


class TestDedupNames:
    def test_no_duplicates(self):
        assert dedup_names(["Si2p", "O1s", "C1s"]) == ["Si2p", "O1s", "C1s"]

    def test_two_duplicates(self):
        assert dedup_names(["O1s", "O1s"]) == ["O1s", "O1s_2"]

    def test_three_duplicates(self):
        assert dedup_names(["O1s", "O1s", "O1s"]) == ["O1s", "O1s_2", "O1s_3"]

    def test_mixed(self):
        result = dedup_names(["O1s", "Si2p", "O1s", "C1s", "Si2p"])
        assert result == ["O1s", "Si2p", "O1s_2", "C1s", "Si2p_2"]

    def test_empty(self):
        assert dedup_names([]) == []

    def test_single(self):
        assert dedup_names(["Si2p"]) == ["Si2p"]

    def test_preserves_order(self):
        names = ["C1s", "O1s", "Si2p"]
        result = dedup_names(names)
        assert result == names  # no duplicates, same order


# ---------------------------------------------------------------------------
# dedup_element_names (HDF5Cache wrapper) — uses mock cache
# ---------------------------------------------------------------------------


class _FakeCache:
    """Minimal mock that mimics HDF5Cache.get_metadata()."""

    def __init__(self, metadata: dict[str, object]):
        self._meta = metadata

    def get_metadata(self, path: str):
        return self._meta.get(path)


class _FakeMeta:
    def __init__(self, element: str):
        self.element = element


class TestDedupElementNames:
    def test_basic(self):
        cache = _FakeCache({
            "/a/O1s.h5": _FakeMeta("O1s"),
            "/a/Si2p.h5": _FakeMeta("Si2p"),
        })
        result = dedup_element_names(["/a/O1s.h5", "/a/Si2p.h5"], cache)
        assert result == ["O1s", "Si2p"]

    def test_duplicates(self):
        cache = _FakeCache({
            "/a/O1s.h5": _FakeMeta("O1s"),
            "/a/O1s_arpes.h5": _FakeMeta("O1s"),
        })
        result = dedup_element_names(["/a/O1s.h5", "/a/O1s_arpes.h5"], cache)
        assert result == ["O1s", "O1s_2"]

    def test_fallback_to_stem(self):
        """When metadata is missing, falls back to Path.stem."""
        cache = _FakeCache({})
        result = dedup_element_names(["/a/Si2p.h5"], cache)
        assert result == ["Si2p"]


# ---------------------------------------------------------------------------
# build_element_infos — uses mock cache
# ---------------------------------------------------------------------------


class TestBuildElementInfos:
    def test_builds_infos(self):
        cache = _FakeCache({
            "/a/O1s.h5": _FakeMeta("O1s"),
            "/a/Si2p.h5": _FakeMeta("Si2p"),
        })
        infos = build_element_infos(["/a/O1s.h5", "/a/Si2p.h5"], cache)
        assert len(infos) == 2

        assert infos[0].label == "O1s"
        assert infos[0].symbol == "O"
        assert infos[0].orbital == "1s"
        assert infos[0].dedup_key == "O1s"

        assert infos[1].label == "Si2p"
        assert infos[1].symbol == "Si"
        assert infos[1].orbital == "2p"
        assert infos[1].dedup_key == "Si2p"

    def test_dedup_infos(self):
        cache = _FakeCache({
            "/a/O1s.h5": _FakeMeta("O1s"),
            "/a/O1s_arpes.h5": _FakeMeta("O1s"),
            "/a/Si2p_arpes.h5": _FakeMeta("Si2p"),
        })
        infos = build_element_infos(
            ["/a/O1s.h5", "/a/O1s_arpes.h5", "/a/Si2p_arpes.h5"], cache
        )
        assert len(infos) == 3

        # First O1s keeps base name
        assert infos[0].dedup_key == "O1s"
        assert infos[0].label == "O1s"

        # Second O1s gets _2 suffix in dedup_key, but label stays "O1s"
        assert infos[1].dedup_key == "O1s_2"
        assert infos[1].label == "O1s"
        assert infos[1].symbol == "O"
        assert infos[1].orbital == "1s"

        # Si2p is unique
        assert infos[2].dedup_key == "Si2p"

    def test_empty(self):
        cache = _FakeCache({})
        infos = build_element_infos([], cache)
        assert infos == []


# ---------------------------------------------------------------------------
# ElementInfo dataclass
# ---------------------------------------------------------------------------


class TestElementInfo:
    def test_creation(self):
        info = ElementInfo(label="Si2p", symbol="Si", orbital="2p", dedup_key="Si2p")
        assert info.label == "Si2p"
        assert info.symbol == "Si"
        assert info.orbital == "2p"
        assert info.dedup_key == "Si2p"

    def test_equality(self):
        a = ElementInfo(label="O1s", symbol="O", orbital="1s", dedup_key="O1s")
        b = ElementInfo(label="O1s", symbol="O", orbital="1s", dedup_key="O1s")
        assert a == b

    def test_inequality(self):
        a = ElementInfo(label="O1s", symbol="O", orbital="1s", dedup_key="O1s")
        b = ElementInfo(label="O1s", symbol="O", orbital="1s", dedup_key="O1s_2")
        assert a != b
