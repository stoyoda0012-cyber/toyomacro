"""The binding energy a cross-section table prints, recorded but never a threshold.

The ionization threshold is the ``BindingEnergy`` value. Each component
also carries ``printed_binding_energy_eV``: the energy Yeh & Lindau (1985)
Table I or Trzhaskovskaya et al. (ADNDT 77, 82) print for every line, and
for 71 Scofield lines that ``BindingEnergy`` lacks, the subshell energies
UCRL-51326 prints per element, read from the scanned report. These are
calculated or experimental free-atom energies and can lie above a solid's edge, so
they are reported and never gate a value.
"""

import pytest

from toyomacro.data import CrossSection


def _component(*args):
    r = CrossSection.lookup_with_status(*args)
    assert len(r.components) == 1
    return r, r.components[0]


def test_binding_energy_is_the_threshold_and_the_printed_value_rides_along():
    _, c = _component("Si", "2p", 1486.6, "yeh_lindau")
    assert c.threshold_eV == 99.0
    assert c.printed_binding_energy_eV == pytest.approx(
        CrossSection.to_dict("yeh_lindau")["data"]["Si"]["2p"]["binding_energy"])


def test_a_printed_energy_does_not_gate():
    """Cu 4s: no BindingEnergy value; Table I prints 6.9 eV, a free-atom energy.

    Below it the line is still outside its range with no value, not a known
    zero: metallic Cu emits from the Fermi level above about 4.7 eV.
    """
    r, c = _component("Cu", "4s", 6.0, "yeh_lindau")
    assert (c.threshold_eV, c.printed_binding_energy_eV) == (None, 6.9)
    assert (r.status, r.value) == ("outside_range_below", None)


@pytest.mark.parametrize("table", ["yeh_lindau", "trzhaskovskaya"])
def test_every_line_of_a_table_that_prints_binding_energies_reports_one(table):
    d = CrossSection.to_dict(table)
    missing = [
        (el, key) for el, rec in d["data"].items() for key in rec
        if CrossSection.lookup_with_status(el, key, 5000.0, table)
        .components[0].printed_binding_energy_eV is None
    ]
    assert missing == []


def test_mcp_reports_the_printed_energy():
    pytest.importorskip("mcp")
    import json

    from toyomacro import mcp_server

    out = json.loads(mcp_server.calculate_sensitivity("Si", "2p", photon_energy=1486.6))
    comp = out["cross_section_components"][0]
    assert comp["threshold_eV"] == 99.0 and comp["printed_binding_energy_eV"] is not None


# --- Scofield: its printed subshell energies for Np-Fm and Pm --------------------

# (element, line, E in keV as printed, report page)
SCOFIELD_PRINTED = [
    ("Pm", "3s1/2", 1.6256, 166),
    ("Np", "2s1/2", 22.4738, 319),
    ("Np", "6d5/2", 0.0041, 319),
    ("Pu", "3d5/2", 3.7762, 325),
    ("Cf", "2p3/2", 19.9268, 349),
    ("Fm", "3d5/2", 4.4916, 361),
]


@pytest.mark.parametrize(("el", "key", "kev", "page"), SCOFIELD_PRINTED)
def test_scofield_printed_energy_is_reported_as_read(el, key, kev, page):
    c = CrossSection.lookup_with_status(el, key, 29000.0, "scofield").components[0]
    assert c.printed_binding_energy_eV == pytest.approx(kev * 1000.0), page
    assert c.threshold_eV is None, "a printed energy must not become a threshold"


def test_every_read_scofield_energy_sits_just_below_its_line_s_first_cell():
    """Independent of the reading: the report evaluates each edge at 1.0070 E.

    So a core line's first cell is 1.0070 x the printed energy (observed
    within 2e-5), or a round grid energy between the two. The check
    resolves about 2e-5 relative: it catches a misread leading digit, not
    always the last one, which rests on the reading.
    """
    from toyomacro.data.cross_section import _SCOFIELD_PRINTED_E_KEV

    d = CrossSection.to_dict("scofield")
    grid = d["photon_energies"]
    at_edge = between = 0
    for el, (_page, energies) in _SCOFIELD_PRINTED_E_KEV.items():
        for key, kev in energies.items():
            e0 = kev * 1000.0
            cells = d["data"][el][key]["cross_sections"]
            first = min(e for e, v in zip(grid, cells) if v)
            if first <= 1000.0:      # valence: the grid starts at 1 keV
                assert e0 < first
                continue
            if first / e0 == pytest.approx(1.0070, abs=3e-5):
                at_edge += 1
            else:
                assert e0 < first < 1.0071 * e0 and first % 1000.0 == 0.0, (el, key)
                between += 1
    assert (at_edge, between) == (62, 3)


def test_a_split_scofield_doublet_between_its_edges_still_refuses():
    """Np 2p at 20 keV: 2p1/2 has no BindingEnergy value, so no known zero.

    Scofield's own 2p1/2 energy (21,704.9 eV) is reported, not used: for
    Th, Pa and U its 2p1/2 edge sits 89-99 eV above BindingEnergy.
    """
    r = CrossSection.lookup_with_status("Np", "2p", 20000.0, "scofield")
    assert (r.status, r.value) == ("outside_range_below", None)
    assert r.components[0].printed_binding_energy_eV == pytest.approx(21704.9)
