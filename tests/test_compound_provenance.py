"""The provenance table is complete, well-formed, and kept out of the values.

``compounds_provenance.json`` records, per entry and per field, what is
known about where a number came from. Two properties matter and are
pinned here:

* it stays in step with ``compounds.json`` -- every entry, every field;
* it never leaks into the values, so a calculation sees exactly the
  numbers it saw before.

The availability state names are shared with another of the author's
projects so the two do not drift apart. That is a naming convention
only; nothing here rests on it.
"""

from __future__ import annotations

import json

import pytest

from toyomacro.data import CompoundDB
from toyomacro.data.paths import get_cache_dir, load_compound_data

AVAILABILITY_STATES = {"known", "unknown", "not_recorded", "not_applicable"}
ORIGIN_KINDS = {"derived", "cited", "asserted", "corrected"}
VALUE_FIELDS = ("Nv", "density", "Mw", "Eg")
NON_MATERIALS = {"AVERAGE", "Oxide"}


@pytest.fixture(scope="module")
def provenance():
    with open(get_cache_dir() / "compounds_provenance.json", encoding="utf-8") as f:
        return json.load(f)


# --- the values are untouched ------------------------------------------------


def test_get_properties_returns_only_numbers():
    """Provenance must not reach the mapping calculations read."""
    for name in CompoundDB.list_compounds():
        props = CompoundDB.get_properties(name)
        assert set(props) <= set(VALUE_FIELDS), name
        assert all(isinstance(v, float) for v in props.values()), name


def test_documented_return_values_are_unchanged():
    """The doctests in compound_db.py are part of the published contract."""
    assert CompoundDB.get_properties("SiO2") == {
        "Nv": 16.0,
        "density": 2.2,
        "Mw": 60.08,
        "Eg": 8.9,
    }
    avg = CompoundDB.get_average_properties()
    assert all(isinstance(v, float) for v in avg.values())


# --- key parity with the data -----------------------------------------------


def test_every_entry_has_provenance(provenance):
    assert set(provenance["entries"]) == set(load_compound_data())


def test_every_value_field_is_covered(provenance):
    """No field may be silently missing a record."""
    data = load_compound_data()
    for name, fields in data.items():
        recorded = provenance["entries"][name]
        for field in fields:
            assert field in recorded, f"{name}.{field} has no provenance"
        assert "phase" in recorded, f"{name} has no phase record"


def test_no_provenance_for_a_field_that_does_not_exist(provenance):
    data = load_compound_data()
    for name, recorded in provenance["entries"].items():
        for field in recorded:
            if field == "phase":
                continue
            assert field in data[name], f"{name}.{field} is provenance for nothing"


# --- well-formedness ---------------------------------------------------------


def _records(provenance):
    for name, fields in provenance["entries"].items():
        for field, record in fields.items():
            yield name, field, record


def test_availability_states_are_from_the_vocabulary(provenance):
    for name, field, record in _records(provenance):
        assert record["availability"] in AVAILABILITY_STATES, f"{name}.{field}"


def test_non_known_states_carry_a_reason(provenance):
    """An unexplained absence is the thing this table exists to prevent."""
    for name, field, record in _records(provenance):
        if record["availability"] != "known":
            assert record.get("reason", "").strip(), f"{name}.{field} lacks a reason"


def test_origin_kinds_are_from_the_vocabulary(provenance):
    for name, field, record in _records(provenance):
        origin = record.get("origin")
        if origin is not None:
            assert origin["kind"] in ORIGIN_KINDS, f"{name}.{field}"


def test_derived_origins_state_how_they_were_derived(provenance):
    """`derived` is a claim about authorship and must be checkable."""
    required = ("formula", "expression", "standard", "standard_version")
    found = 0
    for name, field, record in _records(provenance):
        origin = record.get("origin") or {}
        if origin.get("kind") != "derived":
            continue
        found += 1
        for key in required:
            assert origin.get(key), f"{name}.{field} derived without {key}"
    assert found, "no derived origin present; the check would be vacuous"


def _cited_origin_is_identified(origin: dict) -> bool:
    return bool(origin.get("doi") or origin.get("citation"))


def test_cited_origins_carry_an_identifier(provenance):
    """No entry cites a source yet -- every value here is uncited.

    The rule is still checked, on the shipped records and on a synthetic
    one, so it is in force before the first real citation lands rather
    than passing vacuously until then.
    """
    for name, field, record in _records(provenance):
        origin = record.get("origin") or {}
        if origin.get("kind") == "cited":
            assert _cited_origin_is_identified(origin), f"{name}.{field}"

    assert _cited_origin_is_identified({"kind": "cited", "doi": "10.1002/sia.6598"})
    assert not _cited_origin_is_identified({"kind": "cited", "name": "a paper"})


def test_asserted_origin_may_coexist_with_not_recorded(provenance):
    """The normal case: a value exists, its basis was never written down."""
    record = provenance["entries"]["SiO2"]["density"]
    assert record["availability"] == "not_recorded"
    assert record["origin"]["kind"] == "asserted"


# --- phase -------------------------------------------------------------------


def test_non_materials_are_not_applicable(provenance):
    for name in NON_MATERIALS:
        for field, record in provenance["entries"][name].items():
            assert record["availability"] == "not_applicable", f"{name}.{field}"
            assert record.get("reason")


def test_phase_records_what_it_bears_on(provenance):
    """A known phase names it; an unknown one says why it matters.

    ``applies_to`` is not a constant: for SiC the ambiguity is in the
    band gap (2.31 eV cubic against the stored 3.26 eV, a 41% spread)
    and *not* in the density, whose polytypes differ by under 1%.
    """
    geo2 = provenance["entries"]["GeO2"]["phase"]
    assert geo2["availability"] == "unknown"
    assert "value" not in geo2
    assert geo2["applies_to"] == ["density"]

    tio2 = provenance["entries"]["TiO2"]["phase"]
    assert tio2["availability"] == "known"
    assert tio2["value"] == "rutile"
    assert tio2["applies_to"] == ["density"]

    sic = provenance["entries"]["SiC"]["phase"]
    assert sic["availability"] == "unknown"
    assert sic["applies_to"] == ["Eg"], "the SiC ambiguity is not a density one"


def test_an_identified_phase_says_who_identified_it(provenance):
    """A phase read off the density is an inference authored here.

    It is held to the same standard as a value field: an entry that
    names a phase must carry an origin saying how it was concluded.
    """
    named = [
        (name, fields["phase"])
        for name, fields in provenance["entries"].items()
        if "value" in fields["phase"]
    ]
    assert named, "no identified phase present; the check would be vacuous"
    for name, phase in named:
        origin = phase.get("origin")
        assert origin, f"{name} names a phase with no origin"
        assert origin["kind"] == "derived", name
        assert origin["derived_by"], name


# --- comparisons and investigations are held apart from sources --------------


def test_comparisons_are_never_recorded_as_origins(provenance):
    """A later disagreeing publication is not where the value came from."""
    for name in provenance["comparisons"]:
        for field, record in provenance["entries"][name].items():
            origin = record.get("origin") or {}
            assert origin.get("kind") != "cited", (
                f"{name}.{field} cites a source although the entry only has a "
                f"later comparison"
            )


def test_comparison_agreement_is_reported_accurately(provenance):
    """A comparison may agree or disagree; what it must not do is misreport.

    Some agree exactly -- SESSA's GeO2 and HfO2 densities are the bundled
    numbers, most likely a shared handbook ancestor rather than
    corroboration. Recording that honestly is the point, so the fields
    are recomputed here from the values themselves.
    """
    data = load_compound_data()
    for name, comparisons in provenance["comparisons"].items():
        for comparison in comparisons:
            assert comparison["source"]["name"]
            differs = sorted(
                field
                for field, value in comparison["values"].items()
                if abs(data[name][field] - value) > 1e-9
            )
            agrees = sorted(set(comparison["values"]) - set(differs))
            assert comparison["differs_in"] == differs, name
            assert comparison["agrees_in"] == agrees, name


def test_an_exact_agreement_is_not_called_corroboration(provenance):
    """The GeO2/HfO2 SESSA matches must say why they prove nothing."""
    for name in ("GeO2", "HfO2"):
        comparison = provenance["comparisons"][name][0]
        assert comparison["differs_in"] == []
        assert "not independent" in comparison["source"]["note"]


def test_investigations_state_their_limits(provenance):
    investigations = CompoundDB.get_investigations()
    assert investigations
    for record in investigations:
        for key in ("date", "question", "scope", "result", "limits"):
            assert record.get(key), f"investigation missing {key}"


# --- the public accessors ----------------------------------------------------


def test_accessors_agree_with_the_file(provenance):
    assert CompoundDB.get_provenance("SiO2") == provenance["entries"]["SiO2"]
    assert CompoundDB.get_provenance("nonexistent") is None
    assert CompoundDB.get_comparisons("SiO2") == provenance["comparisons"]["SiO2"]
    assert CompoundDB.get_comparisons("TiO2") == []


def test_si3n4_molecular_weight_is_recorded_as_a_correction():
    """Not `derived`: the digits were already here.

    ``55d7afd`` changed 104.28346 to 140.28346, leaving the fractional
    part untouched -- a transposition repaired, not a value computed
    here. Calling it `derived` would claim an authorship the history
    refutes, and would also misattribute the number to IUPAC 2021, whose
    conventional weights give 140.283 rather than 140.28346.
    """
    record = CompoundDB.get_provenance("Si3N4")["Mw"]
    assert record["availability"] == "known"
    origin = record["origin"]
    assert origin["kind"] == "corrected"
    assert origin["previous_value"] == 104.28346
    assert origin["verified_against"]["result"] == 140.28346
    assert "2021" in origin["note"], "the standard it is NOT on must be stated"


def test_corrected_origins_say_what_they_changed_and_what_confirmed_it(provenance):
    required = ("previous_value", "corrected_in", "correction", "verified_against")
    found = 0
    for name, field, record in _records(provenance):
        origin = record.get("origin") or {}
        if origin.get("kind") != "corrected":
            continue
        found += 1
        for key in required:
            assert origin.get(key) is not None, f"{name}.{field} corrected without {key}"
        assert origin["previous_value"] != load_compound_data()[name][field]
    assert found, "no corrected origin present; the check would be vacuous"


def test_the_only_derived_origins_are_phase_identifications(provenance):
    """Guard the distinction this schema exists to keep.

    No stored *value* was computed here. If one ever is, this test
    should be updated deliberately rather than quietly passing.
    """
    derived = [
        (name, field)
        for name, field, record in _records(provenance)
        if (record.get("origin") or {}).get("kind") == "derived"
    ]
    assert derived
    assert all(field == "phase" for _, field in derived), derived
