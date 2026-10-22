"""Cells a table prints but ``lookup()`` withholds as suspected source misprints.

Yeh & Lindau, *At. Data Nucl. Data Tables* **32**, 1 (1985), Table I,
journal page 8, prints Cu 3d at 8047.8 eV as ``.32E-3`` Mb. The bundled
table holds exactly that, so the transcription is right; the printed
number is what is in doubt. Along the 3d row it falls 38-fold from
1486.6 eV where Ni falls about 300-fold and Zn 2,600-fold, and it is 30x
the bundled Scofield value where Ni is 3.0x and Zn 0.31x. The correct
digit cannot be read from the table, so the cell is withheld rather than
replaced: ``lookup()`` treats it as absent and does not extrapolate the
line across it.

These tests pin the printed value in the file, the evidence for doubting
it, the refusal, and that nothing else in any table moves.
"""

import math

import pytest

from toyomacro.data.cross_section import _WITHHELD_CELLS, AVAILABLE_TABLES, CrossSection
from toyomacro.data.paths import load_cross_section_data

CU_K_ALPHA = 8047.8
AL_K_ALPHA = 1486.6


@pytest.fixture(autouse=True)
def _pchip_default():
    before = CrossSection.get_interpolation()
    CrossSection.set_interpolation("pchip")
    yield
    CrossSection.set_interpolation(before)


def _raw_cell(element, subshell, energy):
    raw = load_cross_section_data()
    i = raw["photon_energies"].index(energy)
    return raw["data"][element][subshell]["cross_sections"][i]


def test_the_bundled_file_keeps_the_printed_value():
    assert _raw_cell("Cu", "3d", CU_K_ALPHA) == pytest.approx(3.2e-4, rel=1e-12)


def test_yeh_lindau_withholds_exactly_this_cell():
    cells = CrossSection.withheld_cells("yeh_lindau")
    assert [(c["element"], c["subshell"], c["photon_energy_eV"]) for c in cells] == [
        ("Cu", "3d", CU_K_ALPHA)
    ]
    assert cells[0]["printed_value"] == pytest.approx(3.2e-4)
    assert "p. 8" in cells[0]["reason"] and ".32E-3" in cells[0]["reason"]


@pytest.mark.parametrize("table", ["scofield", "trzhaskovskaya"])
def test_other_tables_withhold_nothing(table):
    assert CrossSection.withheld_cells(table) == []


def test_unknown_table_is_rejected():
    with pytest.raises(ValueError):
        CrossSection.withheld_cells("not_a_table")


def test_the_evidence_for_doubting_the_print():
    """Cu stands out against its neighbours in both comparisons the reason cites."""
    def yl(el, e):
        return _raw_cell(el, "3d", e)

    def scofield(el):
        return CrossSection.lookup(el, "3d", CU_K_ALPHA, table="scofield")

    fall = {el: yl(el, AL_K_ALPHA) / yl(el, CU_K_ALPHA) for el in ("Ni", "Cu", "Zn")}
    assert fall["Cu"] == pytest.approx(37.5, rel=0.01)
    assert fall["Ni"] > 250 and fall["Zn"] > 2000

    to_scofield = {el: yl(el, CU_K_ALPHA) / scofield(el) for el in ("Ni", "Cu", "Zn")}
    assert to_scofield["Cu"] == pytest.approx(30.5, rel=0.02)
    assert 0.2 < to_scofield["Zn"] < 0.5 and 2 < to_scofield["Ni"] < 4


@pytest.mark.parametrize("energy", [1600.0, 4000.0, CU_K_ALPHA, 10000.0])
@pytest.mark.parametrize("interpolation", ["pchip", "polyfit"])
def test_no_value_above_the_last_kept_cell(energy, interpolation):
    CrossSection.set_interpolation(interpolation)
    result = CrossSection.lookup_with_status("Cu", "3d", energy, table="yeh_lindau")
    assert result.value is None
    assert result.status == "outside_range_above"
    assert result.components[0].valid_range_eV == (16.7, AL_K_ALPHA)
    assert CrossSection.lookup("Cu", "3d", energy, table="yeh_lindau") is None


def test_the_last_kept_cell_still_returns_its_printed_value():
    result = CrossSection.lookup_with_status("Cu", "3d", AL_K_ALPHA, table="yeh_lindau")
    assert result.status == "tabulated"
    assert result.value == pytest.approx(0.012, rel=1e-12)


def test_scofield_still_answers_at_cu_k_alpha():
    result = CrossSection.lookup_with_status("Cu", "3d", CU_K_ALPHA, table="scofield")
    assert result.status == "tabulated" and result.value > 0


def test_to_dict_shows_the_cell_as_absent_and_marks_the_line():
    line = CrossSection.to_dict("yeh_lindau")["data"]["Cu"]["3d"]
    energies = CrossSection.to_dict("yeh_lindau")["photon_energies"]
    assert line["cross_sections"][energies.index(CU_K_ALPHA)] is None
    assert line["withheld_eV"] == [CU_K_ALPHA]


@pytest.mark.parametrize("table", AVAILABLE_TABLES)
def test_every_other_cell_is_loaded_as_stored(table):
    """Only the listed cells differ between the file and what lookups use."""
    CrossSection._data.pop(table, None)
    loaded = CrossSection.to_dict(table)
    from toyomacro.data import cross_section as cs

    if table == "yeh_lindau":
        stored = load_cross_section_data()
    elif table == "scofield":
        stored = CrossSection._load_scofield()
    else:
        stored = CrossSection._load_trzhaskovskaya()
    withheld = {(k[1], k[2], k[3]) for k in cs._WITHHELD_CELLS if k[0] == table}
    differing = set()
    for element, lines in stored["data"].items():
        for key, line in lines.items():
            for e, a, b in zip(
                stored["photon_energies"], line["cross_sections"],
                loaded["data"][element][key]["cross_sections"],
            ):
                if a != b:
                    differing.add((element, key, e))
    assert differing == withheld


def test_every_entry_names_a_printed_positive_cell():
    for (table, element, key, energy), (printed, reason) in _WITHHELD_CELLS.items():
        assert table in AVAILABLE_TABLES
        assert printed > 0 and math.isfinite(printed)
        assert reason
        if table == "yeh_lindau":
            assert _raw_cell(element, key, energy) == pytest.approx(printed, rel=1e-12)


def test_a_stale_entry_is_refused(monkeypatch):
    from toyomacro.data import cross_section as cs

    monkeypatch.setitem(
        cs._WITHHELD_CELLS, ("yeh_lindau", "Cu", "3d", CU_K_ALPHA), (9.9e-9, "stale")
    )
    with pytest.raises(RuntimeError, match="review the entry"):
        cs._withhold_cells("yeh_lindau", load_cross_section_data())
