"""Trzhaskovskaya is tabulated at photoelectron energy; each line is read at E + Eb.

Trzhaskovskaya, Nefedov & Yarzhemsky, ADNDT 77, 97 (2001) and 82, 257
(2002), tabulate sigma at nine photoelectron (kinetic) energies
E = 100-5000 eV, with an experimental binding energy Eb printed per
subshell; ADNDT 92, 245 (2006), Table 2, gives sigma at E = 10 keV. All
three state the unit as kb (=10^-21 cm^2) "for completely filled
subshells" (77, p. 106; 82, p. 260; 92, p. 252). Up to v0.6.0 the grid
was read as photon energy.

Implementation fidelity: printed cells come back at E + Eb, and every
line sits on its own photon axis. Plausibility: on closed core subshells,
where the filled-subshell convention does not matter, the table agrees
with Scofield at the same photon energy (kb against Mb) to a few per
cent, and the agreement no longer depends on Eb/hv, which it did when the
grid was read as photon energy.
"""

import json
import math
import statistics

import pytest

from toyomacro.data import BindingEnergy, CrossSection
from toyomacro.data.paths import get_cache_dir

GRID = [100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 10000.0]

# (element, subshell, Eb printed, E, sigma printed in kb, where)
PRINTED = [
    ("H", "1s1/2", 14.0, 100.0, 26.07, "ADNDT 77, first data block, .2607+2"),
    ("Cs", "3s1/2", 1220.0, 100.0, 60.96, "ADNDT 82, p. 260 worked example, .6096+2"),
    ("Cu", "2s1/2", 1103.0, 10000.0, 1.108, "ADNDT 92 Table 2, 1.108+0"),
    ("Cu", "2p3/2", 938.0, 10000.0, 0.4738, "ADNDT 92 Table 2, 4.738-1"),
    ("Fm", "4d5/2", None, 10000.0, 3.411, "ADNDT 92 Table 2, Z = 100, 3.411+0"),
]


@pytest.fixture(autouse=True)
def _pchip():
    before = CrossSection.get_interpolation()
    CrossSection.set_interpolation("pchip")
    yield
    CrossSection.set_interpolation(before)


def _table():
    return CrossSection.to_dict("trzhaskovskaya")


@pytest.mark.parametrize(("el", "sub", "eb", "e", "sigma", "where"), PRINTED)
def test_a_printed_cell_comes_back_at_e_plus_eb(el, sub, eb, e, sigma, where):
    line = _table()["data"][el][sub]
    if eb is not None:
        assert line["binding_energy"] == eb, where
    hv = e + line["binding_energy"]
    r = CrossSection.lookup_with_status(el, sub, hv, "trzhaskovskaya")
    assert r.status == "tabulated" and r.components[0].at_tabulated_cell, where
    assert r.value == pytest.approx(sigma, rel=1e-12), where


def test_the_grid_is_photoelectron_energy_and_every_line_has_its_own_axis():
    d = _table()
    assert d["photoelectron_energies"] == GRID
    assert "photon_energies" not in d
    n = 0
    for rec in d["data"].values():
        for line in rec.values():
            assert line["photon_energies"] == [e + line["binding_energy"] for e in GRID]
            n += 1
    assert n == 1240


def test_the_bundled_file_is_not_changed():
    """The transform is applied on load; the shipped file keeps the printed grid."""
    with open(get_cache_dir() / "trzhaskovskaya.json", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["photon_energies"] == GRID
    assert "photon_energies" not in raw["data"]["Cu"]["2p3/2"]


def test_no_shared_photon_grid_is_offered():
    with pytest.raises(ValueError, match="photoelectron energies"):
        CrossSection.get_available_photon_energies("trzhaskovskaya")
    assert CrossSection.get_available_photon_energies("scofield")


def test_no_cell_lies_below_its_binding_energy_threshold():
    """Read as photon energy, 2,304 cells fell below the BindingEnergy value."""
    below = 0
    for el, rec in _table()["data"].items():
        for key, line in rec.items():
            thr = BindingEnergy.lookup(el, key)
            if thr is None:
                continue
            below += sum(1 for e, v in zip(line["photon_energies"], line["cross_sections"])
                         if v and e < thr)
    assert below == 0


def test_below_the_first_kinetic_energy_there_is_no_value():
    """Au 4f7/2 (Eb 87 eV) at 150 eV is 63 eV kinetic energy, below E = 100 eV."""
    r = CrossSection.lookup_with_status("Au", "4f7/2", 150.0, "trzhaskovskaya")
    assert (r.status, r.value) == ("outside_range_below", None)


def _closed_core_log_ratios():
    t, s = _table()["data"], CrossSection.to_dict("scofield")["data"]
    out = []
    for el, rec in t.items():
        if el not in s:
            continue
        outer = max(int(k[0]) for k in rec)
        for key, line in rec.items():
            # Closed core: two shells below the outermost, both j listed in Scofield.
            if int(key[0]) >= outer - 1 or key not in s[el] or key[1] == "f":
                continue
            for hv in (1253.6, 1486.6, 2984.3, 4510.9, 5414.7, 8047.8, 9251.7):
                a = CrossSection.lookup_with_status(el, key, hv, "trzhaskovskaya")
                b = CrossSection.lookup_with_status(el, key, hv, "scofield")
                if a.status == b.status == "tabulated":
                    out.append((math.log(a.value / 1000.0 / b.value),
                                line["binding_energy"] / hv))
    return out


def test_closed_core_subshells_agree_with_scofield_at_the_same_photon_energy():
    """kb against Mb, closed core s, p and d subshells, the usual lab and HAXPES lines.

    Measured over 3,682 points: median |ln ratio| 0.019, and by Eb/hv
    0.014 (< 0.05), 0.020 (0.05-0.2), 0.022 (0.2-0.5), 0.034 (>= 0.5).
    Read as photon energy (v0.6.0) the same comparison gave 0.31, rising
    0.088, 0.26, 0.54, 0.92 with Eb/hv: the signature of an energy-axis
    error. The bounds leave room for the two calculations differing near
    threshold, where they differ most.
    """
    rows = _closed_core_log_ratios()
    assert len(rows) > 3000
    assert statistics.median(abs(r) for r, _ in rows) < 0.05
    for lo, hi in ((0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0)):
        band = [abs(r) for r, x in rows if lo <= x < hi]
        assert len(band) > 300 and statistics.median(band) < 0.06, (lo, hi)


def test_the_mcp_tool_reports_no_shared_grid_for_this_table():
    pytest.importorskip("mcp")
    from toyomacro import mcp_server

    out = json.loads(mcp_server.calculate_sensitivity(
        "Si", "2p", photon_energy=1486.6, table="trzhaskovskaya"))
    assert out["cross_section_table_range_eV"] is None
    assert out["cross_section_status"] == "tabulated"
