"""Where a cross-section value comes from: ``CrossSection.lookup_with_status``.

Since v0.5.0 ``lookup()`` returns None below a line's first valid cell
(the threshold region, where a power law has no support), a line with a
single valid cell answers only at that cell, and ``lookup_with_status``
says which state applied to every j component. The range is the line's
own valid cells, never the table's energy grid.

Implementation fidelity is tested against named cases from the tables;
the invariants that tie ``lookup``, ``lookup_with_status`` and
``extrapolate_below`` together are tested over each table's whole grid.
"""

import json
import math

import pytest

from toyomacro import mcp_server
from toyomacro.data.cross_section import AVAILABLE_TABLES, CrossSection

# Yeh-Lindau Table I prints Tl 5d from 40.8 eV (53.07 Mb); v0.4.0 returned
# this power law through the first cells at He I.
TL_5D_HE_I_V040 = 796.0189856842659


@pytest.fixture(autouse=True)
def _restore_interpolation():
    method = CrossSection.get_interpolation()
    yield
    CrossSection.set_interpolation(method)


# --- named cases --------------------------------------------------------------

def test_below_the_first_cell_returns_none_and_says_why():
    assert CrossSection.lookup("Tl", "5d", 21.2, table="yeh_lindau") is None
    r = CrossSection.lookup_with_status("Tl", "5d", 21.2, "yeh_lindau")
    assert r.value is None
    assert r.status == "outside_range_below"
    (part,) = r.components
    assert part.valid_range_eV[0] == 40.8
    assert part.method is None


def test_extrapolate_below_restores_the_v040_value_for_one_call():
    v = CrossSection.lookup("Tl", "5d", 21.2, table="yeh_lindau", extrapolate_below=True)
    assert v == pytest.approx(TL_5D_HE_I_V040, rel=1e-12)
    r = CrossSection.lookup_with_status("Tl", "5d", 21.2, "yeh_lindau", extrapolate_below=True)
    # A value given on request does not become a supported one.
    assert r.status == "outside_range_below"
    assert r.components[0].method == "power_law_below"
    # And the request does not leak into the next call.
    assert CrossSection.lookup("Tl", "5d", 21.2, table="yeh_lindau") is None


def test_a_single_cell_line_answers_only_at_that_cell():
    """Yeh-Lindau Lr 4p has one valid cell, 0.012 Mb at 8047.8 eV."""
    r = CrossSection.lookup_with_status("Lr", "4p", 8047.8, "yeh_lindau")
    assert (r.value, r.status, r.components[0].method) == (0.012, "tabulated", "single_cell")
    above = CrossSection.lookup_with_status("Lr", "4p", 9251.7, "yeh_lindau")
    assert above.value is None and above.status == "outside_range_above"
    # No extrapolation is offered above a single cell, even on request.
    assert CrossSection.lookup("Lr", "4p", 9251.7, table="yeh_lindau",
                               extrapolate_below=True) is None
    below = CrossSection.lookup_with_status("Lr", "4p", 5414.7, "yeh_lindau")
    assert below.value is None and below.status == "outside_range_below"


def test_one_component_out_of_range_refuses_the_doublet_sum():
    """Zn 2p at 1048 eV on Scofield.

    2p3/2 is inside its cells; 2p1/2 is above its BindingEnergy threshold
    (1045 eV) but below Scofield's first cell for it (1057.9 eV). Adding
    2p3/2 alone would return an under-count indistinguishable from a
    complete doublet, so the sum is None.
    """
    r = CrossSection.lookup_with_status("Zn", "2p", 1048.0, "scofield")
    states = {p.orbital: p.status for p in r.components}
    assert states == {"2p1/2": "outside_range_below", "2p3/2": "tabulated"}
    assert r.value is None and r.status == "outside_range_below"
    assert CrossSection.lookup("Zn", "2p", 1048.0, table="scofield") is None

    lower = CrossSection.lookup("Zn", "2p3/2", 1048.0, table="scofield")
    upper = CrossSection.lookup("Zn", "2p1/2", 1048.0, table="scofield", extrapolate_below=True)
    both = CrossSection.lookup("Zn", "2p", 1048.0, table="scofield", extrapolate_below=True)
    assert lower > 0 and upper > 0
    assert both == pytest.approx(lower + upper, rel=1e-12)


def test_a_component_below_threshold_is_a_known_zero():
    """W 3d at 1850 eV: 3d3/2 opens at 1872 eV, 3d5/2 at 1809 eV."""
    r = CrossSection.lookup_with_status("W", "3d", 1850.0, "scofield")
    states = {p.orbital: p.status for p in r.components}
    assert states == {"3d3/2": "below_threshold", "3d5/2": "tabulated"}
    assert r.status == "tabulated"
    assert r.value == pytest.approx(
        CrossSection.lookup("W", "3d5/2", 1850.0, table="scofield"), rel=1e-12)
    (closed,) = [p for p in r.components if p.status == "below_threshold"]
    assert closed.value == 0.0 and closed.threshold_eV == 1872.0


def test_all_known_zero_is_zero_with_status_and_none_from_lookup():
    """Si 2p at 50 eV (BE 99 eV): nothing ionizable."""
    r = CrossSection.lookup_with_status("Si", "2p", 50.0, "yeh_lindau")
    assert (r.value, r.status) == (0.0, "below_threshold")
    assert CrossSection.lookup("Si", "2p", 50.0, table="yeh_lindau") is None


def test_an_unestablished_absent_component_is_not_in_table():
    """Trzhaskovskaya half-lists V 3d; the inclusion rule is unconfirmed."""
    r = CrossSection.lookup_with_status("V", "3d", 1486.6, "trzhaskovskaya")
    states = {p.orbital: p.status for p in r.components}
    assert states == {"3d3/2": "tabulated", "3d5/2": "not_in_table"}
    assert r.value is None and r.status == "not_in_table"


def test_wholly_absent_subshells_carry_no_components():
    for args in [("Ce", "1s", 1486.6, "yeh_lindau"), ("Xx", "1s", 1486.6, "scofield"),
                 ("Au", "4f7/2", 1486.6, "yeh_lindau")]:
        r = CrossSection.lookup_with_status(*args)
        assert (r.value, r.status, r.components) == (None, "not_in_table", ()), args


def test_above_the_last_cell_is_extrapolated_and_flagged():
    r = CrossSection.lookup_with_status("Si", "2p", 9251.7, "yeh_lindau")
    assert r.status == "extrapolated_above" and r.value > 0
    assert r.components[0].method == "power_law_above"


def test_trzhaskovskaya_lines_that_stop_early_extrapolate_inside_the_grid():
    """The range is per line: 91 Trzhaskovskaya lines end below 10 keV."""
    d = CrossSection.to_dict("trzhaskovskaya")
    grid_top = max(d["photon_energies"])
    early = [
        (el, o) for el, rec in d["data"].items() for o, line in rec.items()
        if max(e for e, v in zip(d["photon_energies"], line["cross_sections"]) if v) < grid_top
    ]
    assert len(early) == 91
    el, o = early[0]
    r = CrossSection.lookup_with_status(el, o, grid_top, "trzhaskovskaya")
    assert r.status == "extrapolated_above"


@pytest.mark.parametrize("table", AVAILABLE_TABLES)
def test_the_last_cell_is_tabulated_and_just_above_is_extrapolated(table):
    d = CrossSection.to_dict(table)
    element, orbital = ("Si", "2p") if table == "yeh_lindau" else ("Si", "2p3/2")
    line = d["data"][element][orbital]["cross_sections"]
    hi = max(e for e, v in zip(d["photon_energies"], line) if v)
    at = CrossSection.lookup_with_status(element, orbital, hi, table).components[0]
    assert (at.status, at.at_tabulated_cell) == ("tabulated", True)
    assert at.method != "power_law_above"
    above = CrossSection.lookup_with_status(element, orbital, hi * 1.001, table).components[0]
    assert (above.status, above.method) == ("extrapolated_above", "power_law_above")


def test_get_rsf_passes_extrapolate_below_to_both_lines():
    """Tl 5d at He I over Au 5d: Tl 5d is below its first cell, Au 5d is not."""
    assert CrossSection.get_rsf("Tl", "5d", "Au", "5d", 21.2, table="yeh_lindau") is None
    ratio = CrossSection.get_rsf("Tl", "5d", "Au", "5d", 21.2, table="yeh_lindau",
                                 extrapolate_below=True)
    assert ratio == pytest.approx(
        TL_5D_HE_I_V040 / CrossSection.lookup("Au", "5d", 21.2, table="yeh_lindau"), rel=1e-12)
    # And the reference line too.
    assert CrossSection.get_rsf("Au", "5d", "Tl", "5d", 21.2, table="yeh_lindau") is None
    assert CrossSection.get_rsf("Au", "5d", "Tl", "5d", 21.2, table="yeh_lindau",
                                extrapolate_below=True) > 0


def test_trzhaskovskaya_range_is_judged_on_the_axis_it_is_interpolated_on():
    """Documented limit, pinned so a change to it is deliberate.

    The table is gridded in photoelectron energy (first cell 100 eV) but
    read as photon energy. Au 4f at hv = 150 eV has a kinetic energy of
    about 65 eV, below the source's first cell, and is still reported
    ``tabulated`` because the range is judged on the axis as read.
    """
    r = CrossSection.lookup_with_status("Au", "4f", 150.0, "trzhaskovskaya")
    assert r.status == "tabulated"
    assert all(p.valid_range_eV[0] == 100.0 for p in r.components)
    assert all(150.0 - p.threshold_eV < 100.0 for p in r.components)


def test_method_reports_the_interpolation_in_use():
    CrossSection.set_interpolation("polyfit")
    r = CrossSection.lookup_with_status("Si", "2p", 1486.6, "yeh_lindau")
    assert r.components[0].method == "polyfit"
    CrossSection.set_interpolation("pchip")
    r = CrossSection.lookup_with_status("Si", "2p", 1486.6, "yeh_lindau")
    assert r.components[0].method == "pchip" and r.components[0].at_tabulated_cell


# --- invariants over each table's grid ----------------------------------------

@pytest.mark.parametrize("table", AVAILABLE_TABLES)
def test_lookup_and_status_agree_everywhere(table):
    """Over every element, label and (a stride of) the table's own grid.

    - lookup() is the status value, with a known zero reported as None;
    - a summed value is the sum of its components;
    - extrapolate_below changes the answer only where a component is
      outside_range_below, and nowhere returns None where lookup() has a
      value.
    """
    d = CrossSection.to_dict(table)
    energies = d["photon_energies"]
    labels = sorted({o for rec in d["data"].values() for o in rec}
                    | {o[:2] for rec in d["data"].values() for o in rec})
    checked = 0
    for element in d["data"]:
        for orbital in labels:
            for e in energies[:: max(1, len(energies) // 16)]:
                r = CrossSection.lookup_with_status(element, orbital, e, table)
                v = CrossSection.lookup(element, orbital, e, table=table)
                x = CrossSection.lookup(element, orbital, e, table=table, extrapolate_below=True)
                assert v == (r.value if r.value else None), (element, orbital, e)
                if r.value is not None:
                    assert r.value == pytest.approx(sum(p.value for p in r.components), rel=1e-12)
                if r.status != "outside_range_below":
                    assert x == v, (element, orbital, e)
                if v is not None:
                    assert x == v
                if r.status == "outside_range_below":
                    assert v is None
                checked += 1
    assert checked > 5000


# --- MCP ----------------------------------------------------------------------

def test_mcp_refuses_below_the_first_cell_and_names_the_state():
    out = json.loads(mcp_server.calculate_sensitivity(
        "Tl", "5d", photon_energy=21.2, table="yeh_lindau"))
    assert "error" in out and "first tabulated cell" in out["error"]
    assert out["cross_section_status"] == "outside_range_below"


def test_mcp_flag_is_judged_on_the_line_not_the_table_grid():
    out = json.loads(mcp_server.calculate_sensitivity(
        "Si", "2p", photon_energy=1486.6, table="yeh_lindau"))
    assert out["cross_section_status"] == "tabulated"
    assert out["cross_section_extrapolated"] is False
    (comp,) = out["cross_section_components"]
    assert comp["orbital"] == "2p" and comp["method"] == "pchip"
    assert math.isclose(comp["value"], out["cross_section"])
