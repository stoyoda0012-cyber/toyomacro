"""The spin-orbit splitting reference.

``spin_orbit.json`` is built by ``toyomacro.data._spin_orbit_build`` from
the bundled Trzhaskovskaya 2018/2019, 2001/2002 and binding-energy tables
and the fitters' in-code splittings. These tests pin:

- the bundled file to what the build makes, so a change to any input
  shows here until the file is rebuilt;
- binding energies the two papers print, for the rows the rules turn on;
- the whole output of ``BindingEnergy.get_spin_orbit`` — every element of
  ``binding_energy.json`` and every subshell n = 1-7, l = s-f — by digest,
  with the state counts and representative rows spelled out;
- the rules of the reference: raw signs, no exception for unsettled
  levels, a pair printed equal is unresolved and not zero, one stored value
  for two members is no splitting;
- that the fitters do not read the reference;
- the j-resolved cross-section ratio at one photon energy.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from toyomacro.core import xps_database
from toyomacro.core.xps_database import get_so_info
from toyomacro.data import BindingEnergy, CrossSection, _spin_orbit_build

BUNDLED = (
    Path(__file__).parents[1] / "src" / "toyomacro" / "data" / "_cache" / "spin_orbit.json"
)

# (element, subshell): (BE j+, BE j-) as Table 1 of Trzhaskovskaya &
# Yarzhemsky 2018 prints them: j- printed less bound than j+.
NEGATIVE = {
    ("N", "2p"): (9.7, 8.9), ("O", "2p"): (7.4, 6.8), ("F", "2p"): (8.7, 8.4),
    ("P", "3p"): (10.1, 9.6), ("S", "3p"): (8.2, 7.8),
    ("Co", "3d"): (3.3, 2.7), ("Zn", "3d"): (8.0, 7.9), ("Os", "5d"): (0.9, 0.4),
}

# (element, subshell): (BE j+ in the 2018 paper, BE j- in the 2019 paper).
# The two papers divide at 1.5 keV and these doublets lie across it.
STRADDLING = {
    ("Sm", "3p"): (1419.8, 1540.7), ("Eu", "3p"): (1480.6, 1613.9),
    ("Tm", "3d"): (1467.7, 1514.6), ("Bk", "4p"): (1248.0, 1554.5),
    ("Cf", "4p"): (1291.5, 1616.5), ("Es", "4p"): (1335.5, 1680.1),
    ("Fm", "4p"): (1379.9, 1743.0),
}


def _all_answers() -> list[dict]:
    """get_spin_orbit over every element and every n = 1-7, l = s, p, d, f."""
    out = []
    for el in BindingEnergy.get_all_elements():
        for n in range(1, 8):
            for letter in "spdf":
                ref = BindingEnergy.get_spin_orbit(el, f"{n}{letter}")
                out.append(dataclasses.asdict(ref))
    return out


def _in_reference(answers: list[dict]) -> list[dict]:
    levels = BindingEnergy.spin_orbit_levels()
    return [a for a in answers if a["subshell"] in levels.get(a["element"], ())]


class TestBundledFile:
    def test_bundled_file_is_what_the_build_makes(self):
        """Byte for byte. Fails when a bundled table or an in-code splitting
        changes and the file was not rebuilt."""
        built = _spin_orbit_build.dumps(_spin_orbit_build.build())
        assert built == BUNDLED.read_text(encoding="utf-8")

    def test_a_changed_in_code_value_changes_the_build(self, monkeypatch):
        """The dependence the test above guards is real."""
        monkeypatch.setitem(xps_database._SO_SPLITS, "Si2p", 0.9)
        levels = _spin_orbit_build.build()["levels"]
        assert levels["Si"]["2p"]["state"] == "to_verify"
        assert "in-code value 0.9" in levels["Si"]["2p"]["state_reason"]

    def test_every_element_of_the_reference_is_reachable(self):
        """_all_answers walks binding_energy.json's elements; the reference
        has no element outside them."""
        assert set(BindingEnergy.spin_orbit_levels()) <= set(BindingEnergy.get_all_elements())


class TestWholeOutputSpace:
    def test_digest_of_every_answer(self):
        """Any change to any answer — value, label, state or reason — lands
        here."""
        blob = json.dumps(_all_answers(), sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        assert digest == "470ed8adba7f6f9f879dee7652c73915f71076d381d4d593c698b0407d80eac0"

    def test_state_and_source_counts(self):
        answers = _all_answers()
        in_ref = _in_reference(answers)
        assert len(in_ref) == 579
        assert len(BindingEnergy.spin_orbit_levels()) == 99
        assert Counter(a["state"] for a in in_ref) == {
            "provisional": 276, "to_verify": 216, "not_covered": 87,
        }
        assert Counter(a["source_table"] for a in in_ref) == {
            "trzh2018": 381, "trzh2019": 104, "trzh2018+trzh2019": 7,
            "binding_energy": 22, None: 65,
        }
        # Nothing is adopted: no rule of the build adopts
        assert not any(a["state"] == "adopted" for a in answers)
        # Everything outside the reference is not_covered, with no value
        rest = [a for a in answers if a not in in_ref]
        assert {a["state"] for a in rest} == {"not_covered"}
        assert all(a["split_eV"] is None and a["source_table"] is None for a in rest)

    def test_fields_follow_the_source(self):
        for a in _all_answers():
            trzh = (a["source_table"] or "").startswith("trzh")
            assert a["precision_eV"] == (0.1 if trzh else None)
            assert (a["state"] == "not_covered") == (not trzh)
            if a["split_eV"] is None:
                assert a["be_j_plus_eV"] is None and a["be_j_minus_eV"] is None
                assert a["source_table"] is None
            else:
                assert a["split_eV"] == round(a["be_j_minus_eV"] - a["be_j_plus_eV"], 2)
            assert a["unresolved_at_precision"] == (a["split_eV"] == 0.0)

    def test_a_provisional_value_is_never_negative_and_a_carried_one_never_zero(self):
        for a in _all_answers():
            if a["state"] == "provisional":
                assert a["split_eV"] >= 0
            if a["source_table"] == "binding_energy":
                assert a["split_eV"] != 0

    def test_a_positive_provisional_value_was_compared_with_something(self):
        """Passing by default is not agreement: a positive splitting no
        other bundled value can be compared with is to_verify."""
        kinds = Counter()
        for a in _in_reference(_all_answers()):
            if a["split_eV"] is None or a["split_eV"] <= 0 or a["state"] == "not_covered":
                continue
            head = a["state_reason"].split(":")[0]
            kinds[(a["state"], head)] += 1
        assert kinds == {
            ("provisional", "agrees within tolerance with"): 267,
            ("to_verify", "another value disagrees beyond tolerance"): 164,
            ("to_verify", "no other value to compare it with"): 44,
        }

    def test_statistical_ratio(self):
        for a in _all_answers():
            expected = {"s": None, "p": 0.5, "d": 2 / 3, "f": 0.75}[a["subshell"][1]]
            assert a["statistical_ratio"] == expected


class TestRepresentativeRows:
    def test_si_2p(self):
        ref = BindingEnergy.get_spin_orbit("Si", "2p")
        assert (ref.split_eV, ref.precision_eV, ref.source_table, ref.state) == (
            0.6, 0.1, "trzh2018", "provisional")
        assert (ref.be_j_plus_eV, ref.be_j_minus_eV) == (98.9, 99.5)
        assert (ref.j_plus, ref.j_minus, ref.statistical_ratio) == ("2p3/2", "2p1/2", 0.5)
        assert BindingEnergy.get_spin_orbit("Si", "2p3/2") == ref
        assert BindingEnergy.get_spin_orbit("Si", "2p1/2") == ref

    def test_hf_4d_is_to_verify_not_an_exception(self):
        ref = BindingEnergy.get_spin_orbit("Hf", "4d")
        assert (ref.split_eV, ref.source_table, ref.state) == (10.1, "trzh2018", "to_verify")
        assert "in-code value 16.8" in ref.state_reason

    def test_ga_3d_keeps_the_printed_value(self):
        ref = BindingEnergy.get_spin_orbit("Ga", "3d")
        assert (ref.be_j_plus_eV, ref.be_j_minus_eV, ref.split_eV) == (15.7, 20.7, 5.0)
        assert ref.state == "to_verify"

    @pytest.mark.parametrize(("element", "subshell"), sorted(NEGATIVE))
    def test_negative_values_keep_their_sign(self, element, subshell):
        ref = BindingEnergy.get_spin_orbit(element, subshell)
        be_plus, be_minus = NEGATIVE[(element, subshell)]
        assert (ref.be_j_plus_eV, ref.be_j_minus_eV) == (be_plus, be_minus)
        assert ref.split_eV == round(be_minus - be_plus, 2) < 0
        assert ref.state == "to_verify" and "negative as printed" in ref.state_reason

    def test_exactly_eight_negative_values(self):
        neg = {(a["element"], a["subshell"]) for a in _all_answers()
               if a["split_eV"] is not None and a["split_eV"] < 0}
        assert neg == set(NEGATIVE)

    @pytest.mark.parametrize(("element", "subshell"), sorted(STRADDLING))
    def test_a_doublet_across_the_two_papers_is_read_from_both(self, element, subshell):
        ref = BindingEnergy.get_spin_orbit(element, subshell)
        be_plus, be_minus = STRADDLING[(element, subshell)]
        assert ref.source_table == "trzh2018+trzh2019"
        assert (ref.be_j_plus_eV, ref.be_j_minus_eV) == (be_plus, be_minus)
        assert ref.split_eV == round(be_minus - be_plus, 2) > 45
        assert "straddles 1.5 keV" in ref.state_reason

    def test_exactly_seven_doublets_across_the_two_papers(self):
        both = {(a["element"], a["subshell"]) for a in _all_answers()
                if a["source_table"] == "trzh2018+trzh2019"}
        assert both == set(STRADDLING)

    def test_one_stored_value_for_two_members_gives_no_splitting(self):
        """binding_energy.json stores Md 4d3/2 and 4d5/2 as one number. That
        is not a doublet unresolved at 1 eV, and not a splitting of zero."""
        ref = BindingEnergy.get_spin_orbit("Md", "4d")
        assert BindingEnergy.get_spin_orbit_split("Md", "4d") == 0.0
        assert (ref.split_eV, ref.source_table, ref.state) == (None, None, "not_covered")
        assert not ref.unresolved_at_precision
        assert "one value for both members" in ref.state_reason

    def test_an_equal_pair_in_another_table_does_not_contradict(self):
        """Np 4p: the 2018 paper prints 1086.0 and 1327.7; binding_energy.json
        stores one value for both, which says nothing about the splitting."""
        ref = BindingEnergy.get_spin_orbit("Np", "4p")
        assert BindingEnergy.get_spin_orbit_split("Np", "4p") == 0.0
        assert (ref.be_j_plus_eV, ref.be_j_minus_eV, ref.split_eV) == (1086.0, 1327.7, 241.7)
        assert ref.state == "provisional"
        assert "binding_energy" not in ref.state_reason
        assert "trzhaskovskaya.json (2001/2002) 242" in ref.state_reason

    def test_nothing_to_compare_with_is_to_verify(self):
        """Sc 3p: 31.5 and 33.8 printed, and both other tables store one
        value for the two members."""
        ref = BindingEnergy.get_spin_orbit("Sc", "3p")
        assert (ref.be_j_plus_eV, ref.be_j_minus_eV, ref.split_eV) == (31.5, 33.8, 2.3)
        assert ref.state == "to_verify"
        assert ref.state_reason.startswith("no other value to compare it with")

    def test_the_integer_tolerance_includes_the_reference_rounding(self):
        """Rb 3p: 8.9 here, 10 in binding_energy.json. Two energies rounded
        to 1 eV and two printed to 0.1 eV can differ by that much."""
        ref = BindingEnergy.get_spin_orbit("Rb", "3p")
        assert BindingEnergy.get_spin_orbit_split("Rb", "3p") == 10.0
        assert (ref.split_eV, ref.state) == (8.9, "provisional")
        assert "binding_energy.json 10" in ref.state_reason

    def test_a_carried_pair_is_not_adopted(self):
        ref = BindingEnergy.get_spin_orbit("Lu", "2p")
        assert (ref.source_table, ref.precision_eV, ref.state) == (
            "binding_energy", None, "not_covered")
        assert ref.split_eV == 1105.0
        assert "only 2p3/2" in ref.state_reason

    def test_single_member_level_carries_no_value(self):
        """The 2018 paper prints Ti 3d3/2 alone."""
        ref = BindingEnergy.get_spin_orbit("Ti", "3d")
        assert (ref.state, ref.split_eV, ref.source_table) == ("not_covered", None, None)
        assert ref.state_reason == "only 3d3/2 is in the 2018/2019 tables"

    def test_s_level_absent_level_and_unknown_element(self):
        s = BindingEnergy.get_spin_orbit("C", "1s")
        assert s.state == "not_covered" and s.j_plus is None and s.split_eV is None
        assert "s level" in s.state_reason
        absent = BindingEnergy.get_spin_orbit("Si", "6f")
        assert absent.state == "not_covered" and absent.statistical_ratio == 0.75
        assert "no bundled table holds a member" in absent.state_reason
        for symbol in ("Xx", "si", "SI", None):
            ref = BindingEnergy.get_spin_orbit(symbol, "2p")
            assert ref.state == "not_covered" and ref.split_eV is None
            assert "no element" in ref.state_reason

    @pytest.mark.parametrize(
        "label", ["p", "2x", "2p1/2x", "22p", "2p5/2", "2p7/2", "3d1/2", "2s3/2", "8p", None]
    )
    def test_malformed_subshell_raises(self, label):
        with pytest.raises(ValueError):
            BindingEnergy.get_spin_orbit("Si", label)

    def test_printed_equal_is_flagged_unresolved(self):
        ref = BindingEnergy.get_spin_orbit("Cl", "3p")
        assert (ref.split_eV, ref.precision_eV, ref.state) == (0.0, 0.1, "provisional")
        assert ref.unresolved_at_precision
        assert "not a splitting of zero" in ref.state_reason
        assert "positive" not in ref.state_reason
        flagged = {(a["element"], a["subshell"]) for a in _all_answers()
                   if a["unresolved_at_precision"]}
        assert flagged == {
            ("Cl", "3p"), ("As", "4p"), ("Se", "4p"), ("Am", "5f"), ("Cm", "5f"),
            ("Bk", "5f"), ("Cf", "5f"), ("Es", "5f"), ("Fm", "5f"),
        }

    def test_integer_table_split_is_unchanged(self):
        """get_spin_orbit_split stays the binding_energy.json difference."""
        assert BindingEnergy.get_spin_orbit_split("Si", "2p") == 1.0
        assert BindingEnergy.get_spin_orbit_split("Ga", "3d") == 0.0


class TestFittersDoNotReadTheReference:
    """The reference is for showing and assigning a partner. A difference of
    two energies each printed to 0.1 eV is too coarse to hold a doublet at."""

    def test_get_so_info_returns_the_in_code_value_everywhere(self):
        assert len(xps_database._SO_SPLITS) == 46
        for key, value in xps_database._SO_SPLITS.items():
            assert get_so_info(key).so_split == value

    def test_levels_outside_the_in_code_values_stay_unconstrained(self):
        levels = BindingEnergy.spin_orbit_levels()
        outside = [el + nl for el, subshells in levels.items() for nl in subshells
                   if el + nl not in xps_database._SO_SPLITS]
        assert len(outside) == 579 - 46
        assert all(get_so_info(label) is None for label in outside)

    def test_the_two_differ_for_common_levels(self):
        """Ge 3d: 0.55 in the fitters, 0.7 in the reference (28.5 and 29.2
        printed). Both stand; neither is derived from the other."""
        assert get_so_info("Ge3d").so_split == 0.55
        ge = BindingEnergy.get_spin_orbit("Ge", "3d")
        assert (ge.be_j_plus_eV, ge.be_j_minus_eV, ge.split_eV) == (28.5, 29.2, 0.7)
        refs = {k: BindingEnergy.get_spin_orbit(k[:-2], k[-2:])
                for k in xps_database._SO_SPLITS}
        assert [k for k, r in refs.items() if r.split_eV is None] == ["Ti3d"]
        same = [k for k, r in refs.items() if r.split_eV == xps_database._SO_SPLITS[k]]
        assert same == ["Si2p", "Cl2p", "Co2p", "La3d"]
        gaps = {k: round(abs(r.split_eV - xps_database._SO_SPLITS[k]), 2)
                for k, r in refs.items() if r.split_eV is not None}
        assert max(gaps[k] for k, r in refs.items() if r.state == "provisional") == 0.15
        assert max(gaps.values()) == gaps["Hf4d"] == 6.7

    @pytest.mark.filterwarnings("ignore:Template peak")
    def test_fits_and_templates_run_with_the_reference_unreadable(self, monkeypatch):
        """Whatever a fitting path reads, it is not the reference: with every
        way into it raising, the fitters still give their in-code values."""
        from toyomacro.data import paths
        from toyomacro.fitting.autofitter import AutoFitter
        from toyomacro.fitting.template_bridge import template_to_configs
        from toyomacro.fitting.templates import get_template, list_templates

        from .test_fitting_templates import _make_synthetic_si2p_ke

        def refuse(*args, **kwargs):
            raise AssertionError("a fitting path read the spin-orbit reference")

        monkeypatch.setattr(BindingEnergy, "_spin_orbit_data", None)
        monkeypatch.setattr(BindingEnergy, "_get_spin_orbit_data", classmethod(refuse))
        monkeypatch.setattr(BindingEnergy, "get_spin_orbit", classmethod(refuse))
        monkeypatch.setattr(paths, "load_spin_orbit_data", refuse)

        for key, value in xps_database._SO_SPLITS.items():
            assert get_so_info(key).so_split == value
        splits = {}
        for name in list_templates():
            for config in template_to_configs(get_template(name), reference_be=100.0):
                splits[name] = config.so_split
        assert splits["Ta4f_oxide"] == 1.91  # the reference says 1.9
        result = AutoFitter().fit_with_template(_make_synthetic_si2p_ke(), "Si2p_oxide")
        assert result.success

    @pytest.mark.parametrize("module", ["toyomacro.core.xps_database",
                                        "toyomacro.fitting.autofitter",
                                        "toyomacro.fitting.template_bridge"])
    def test_no_fitting_module_names_the_reference(self, module):
        import importlib

        source = inspect.getsource(importlib.import_module(module))
        assert "get_spin_orbit(" not in source and "spin_orbit.json" not in source


class TestIntensityRatio:
    @pytest.mark.parametrize(("element", "subshell", "scofield", "trzh"), [
        ("Si", "2p", 0.510, 0.510), ("Ti", "2p", 0.516, 0.516),
        ("Ta", "4f", 0.789, 0.788), ("Au", "4f", 0.787, 0.787),
    ])
    def test_al_ka_ratios(self, element, subshell, scofield, trzh):
        s = CrossSection.get_spin_orbit_intensity_ratio(element, subshell, 1486.6,
                                                        table="scofield")
        t = CrossSection.get_spin_orbit_intensity_ratio(element, subshell, 1486.6,
                                                        table="trzhaskovskaya")
        assert (round(s.ratio, 3), s.status) == (scofield, "tabulated")
        assert (round(t.ratio, 3), t.status) == (trzh, "tabulated")

    def test_both_components_at_the_same_photon_energy(self):
        r = CrossSection.get_spin_orbit_intensity_ratio("Au", "4f", 1486.6,
                                                        table="trzhaskovskaya")
        assert r.j_minus.photon_energy == r.j_plus.photon_energy == 1486.6
        assert r.ratio == pytest.approx(r.j_minus.value / r.j_plus.value)
        assert r.j_minus.orbital == "4f5/2" and r.j_plus.orbital == "4f7/2"

    def test_bare_subshell_table_has_no_ratio(self):
        r = CrossSection.get_spin_orbit_intensity_ratio("Au", "4f", 1486.6,
                                                        table="yeh_lindau")
        assert (r.ratio, r.status) == (None, "not_in_table")

    def test_one_component_without_value_gives_no_ratio(self):
        # Zn 2p at 1048 eV on Scofield: 2p1/2 is below its first cell
        r = CrossSection.get_spin_orbit_intensity_ratio("Zn", "2p", 1048.0,
                                                        table="scofield")
        assert (r.ratio, r.status) == (None, "outside_range_below")
        assert r.j_plus.value is not None

    def test_below_threshold_gives_no_ratio(self):
        r = CrossSection.get_spin_orbit_intensity_ratio("Au", "4f", 50.0,
                                                        table="scofield")
        assert r.ratio is None and r.status == "below_threshold"

    def test_extrapolated_component_marks_the_ratio(self):
        r = CrossSection.get_spin_orbit_intensity_ratio("Au", "4f", 40000.0,
                                                        table="scofield")
        assert r.ratio is not None and r.status == "extrapolated_above"

    @pytest.mark.parametrize("label", ["2s", "4f3/2", "2p5/2", "8p", "p", "4f ", None, 4])
    def test_a_label_that_is_not_a_p_d_or_f_level_raises(self, label):
        with pytest.raises(ValueError):
            CrossSection.get_spin_orbit_intensity_ratio("Au", label, 1486.6, table="scofield")

    def test_a_j_resolved_label_is_read_as_its_subshell(self):
        bare = CrossSection.get_spin_orbit_intensity_ratio("Au", "4f", 1486.6, table="scofield")
        for label in ("4f7/2", "4f5/2"):
            assert CrossSection.get_spin_orbit_intensity_ratio(
                "Au", label, 1486.6, table="scofield") == bare
        assert (bare.table, bare.subshell, bare.element) == ("scofield", "4f", "Au")

    def test_the_status_is_that_of_the_first_component_without_a_value(self):
        """j- is looked up first. Zn 2p at 1030 eV on Scofield: both are
        below their first cell, for different reasons or the same; the
        ratio reports j-."""
        r = CrossSection.get_spin_orbit_intensity_ratio("Zn", "2p", 1048.0, table="scofield")
        assert r.j_minus.value is None and r.j_plus.value is not None
        assert r.status == r.j_minus.status
        both = CrossSection.get_spin_orbit_intensity_ratio("Au", "4f", 50.0, table="scofield")
        assert both.status == both.j_minus.status

    @pytest.mark.parametrize("energy", ["1486.6", None, True])
    def test_a_photon_energy_that_is_not_a_number_raises(self, energy):
        with pytest.raises(ValueError):
            CrossSection.get_spin_orbit_intensity_ratio("Au", "4f", energy, table="scofield")

    @pytest.mark.parametrize("energy", [math.nan, math.inf, -math.inf])
    def test_non_finite_photon_energy_raises(self, energy):
        with pytest.raises(ValueError):
            CrossSection.get_spin_orbit_intensity_ratio("Au", "4f", energy, table="scofield")
