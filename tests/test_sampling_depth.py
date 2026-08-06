"""Mean escape depth and information depth from the single-scattering albedo.

Golden values are the published slopes and the published limiting forms,
so these are implementation-fidelity checks against Jablonski & Powell,
*J. Vac. Sci. Technol. A* **27**, 253 (2009), DOI 10.1116/1.3071947:
Eqs. (22), (23), (28) and (29), and the straight-line limits Eqs. (5)
and (7). They fail on a transcribed coefficient, a lost ``cos(alpha)``,
a missing model gate, or a validity constraint dropped.

The separate concern these tests carry that the EAL ones do not: D and S
are *not* attenuation lengths and *not* each other. Several tests exist
only to hold those three apart.
"""

from __future__ import annotations

import json
import math

import pytest

from toyomacro.data.elastic_scattering import (
    INFORMATION_DEPTH_MODELS,
    MED_MODELS,
    DescribedLength,
    information_depth,
    information_depth_ratio,
    information_depth_report,
    mean_escape_depth,
    mean_escape_depth_ratio,
    mean_escape_depth_report,
    overlayer_eal,
    single_scattering_albedo,
)

# Chosen so that omega = 10/(10+40) = 0.2 exactly, inside every stated
# range. Angstrom, but the relations are unit-agnostic.
IMFP = 10.0
TRMFP = 40.0
OMEGA = 0.2

# Elastic scattering switched off: omega -> 0 as the transport mean free
# path runs away.
NO_ELASTIC_TRMFP = IMFP * 1e12


def described(value, **kw):
    kw.setdefault("unit", "angstrom")
    kw.setdefault("source", "invented for this test")
    kw.setdefault("material", "Xx")
    kw.setdefault("kinetic_energy_ev", 1000.0)
    return DescribedLength(value, **kw)


# --- the published coefficients -------------------------------------


def test_published_slopes():
    """Eqs. (22) and (23), by their coefficients."""
    assert MED_MODELS["jp2009"].slope == 0.736
    assert INFORMATION_DEPTH_MODELS["jp2009"].slope == 0.787


def test_the_three_slopes_are_distinct_and_none_is_the_eal_slope():
    """0.736, 0.787 and 0.738 are close enough to be confused."""
    from toyomacro.data.elastic_scattering import EAL_MODELS

    slopes = {
        MED_MODELS["jp2009"].slope,
        INFORMATION_DEPTH_MODELS["jp2009"].slope,
        EAL_MODELS["jp2020_unpolarized"].slope,
        EAL_MODELS["jp2020_polarized_haxpes"].slope,
    }
    assert len(slopes) == 4


def test_stated_validity_is_recorded():
    for model in (MED_MODELS["jp2009"], INFORMATION_DEPTH_MODELS["jp2009"]):
        assert model.emission_angle_deg == (0.0, 50.0)
        assert model.energy_ev == (61.0, 2016.0)
        assert model.albedo == (0.1, 0.5)
        assert "54 degrees" in model.geometry
        assert "2009" in model.equation
    # A mean percentage deviation, not an rms -- the field is named for
    # the statistic the paper actually quotes.
    assert MED_MODELS["jp2009"].mean_deviation_percent == 0.25
    assert INFORMATION_DEPTH_MODELS["jp2009"].mean_deviation_percent == 0.54


def test_ratios_follow_the_published_lines():
    assert mean_escape_depth_ratio(IMFP, TRMFP, model="jp2009") == (
        pytest.approx(1.0 - 0.736 * OMEGA, rel=1e-12))
    assert information_depth_ratio(IMFP, TRMFP, model="jp2009") == (
        pytest.approx(1.0 - 0.787 * OMEGA, rel=1e-12))


# --- the limits the relations must reproduce ------------------------


def test_med_without_elastic_scattering_is_the_sla_form():
    """Eq. (5): D_SLA = lambda cos(alpha)."""
    for angle in (0.0, 30.0, 50.0):
        assert mean_escape_depth(
            IMFP, NO_ELASTIC_TRMFP,
            emission_angle_deg=angle, model="jp2009") == pytest.approx(
                IMFP * math.cos(math.radians(angle)), rel=1e-9)


def test_information_depth_without_elastic_scattering_is_the_sla_form():
    """Eq. (7): S_SLA = lambda cos(alpha) ln[1/(1 - P/100)]."""
    for percentage in (90.0, 95.0, 99.0):
        expected = (IMFP * math.cos(math.radians(30.0))
                    * math.log(1.0 / (1.0 - percentage / 100.0)))
        assert information_depth(
            IMFP, NO_ELASTIC_TRMFP, emission_angle_deg=30.0,
            percentage=percentage, model="jp2009") == pytest.approx(
                expected, rel=1e-9)


def test_the_sla_limit_agrees_with_the_uncorrected_sampling_depth():
    """``IMFP.sampling_depth`` is S_SLA at normal emission, and says so.

    The two live in different modules and must not drift apart: with
    elastic scattering switched off, Eq. (29) has to collapse onto it.
    """
    from toyomacro.data.imfp import IMFP

    lam = IMFP.tpp2m(kinetic_energy=1000.0, compound="Si")
    assert information_depth(
        lam, lam * 1e12, emission_angle_deg=0.0, percentage=95.0,
        model="jp2009") == pytest.approx(
            IMFP.sampling_depth(kinetic_energy=1000.0, compound="Si",
                                fraction=0.95), rel=1e-9)


def test_the_full_expressions_are_eqs_28_and_29():
    angle, percentage = 40.0, 95.0
    cos = math.cos(math.radians(angle))
    assert mean_escape_depth(
        IMFP, TRMFP, emission_angle_deg=angle,
        model="jp2009") == pytest.approx(
            IMFP * cos * (1.0 - 0.736 * OMEGA), rel=1e-12)
    assert information_depth(
        IMFP, TRMFP, emission_angle_deg=angle, percentage=percentage,
        model="jp2009") == pytest.approx(
            IMFP * cos * (1.0 - 0.787 * OMEGA)
            * math.log(1.0 / (1.0 - percentage / 100.0)), rel=1e-12)


# --- what depends on the angle, and what does not -------------------


def test_the_depths_carry_cos_alpha_but_the_ratios_do_not():
    """The cos(alpha) cancels in D/D_SLA; it does not in D."""
    at_0 = mean_escape_depth(
        IMFP, TRMFP, emission_angle_deg=0.0, model="jp2009")
    at_60 = mean_escape_depth(
        IMFP, TRMFP, emission_angle_deg=60.0, model="jp2009")
    assert at_60 == pytest.approx(at_0 / 2.0, rel=1e-12)  # cos 60 = 1/2

    ratios = {mean_escape_depth_ratio(IMFP, TRMFP, model="jp2009")}
    assert len(ratios) == 1


def test_the_information_depth_ratio_does_not_depend_on_the_percentage():
    assert information_depth_ratio(IMFP, TRMFP, model="jp2009") == (
        pytest.approx(
            information_depth(IMFP, TRMFP, emission_angle_deg=0.0,
                              percentage=99.0, model="jp2009")
            / (IMFP * math.log(100.0)), rel=1e-12))


def test_the_percentage_scales_the_information_depth():
    """99% is exactly ln(100)/ln(10) = 2 times 90%."""
    s90 = information_depth(IMFP, TRMFP, emission_angle_deg=0.0,
                            percentage=90.0, model="jp2009")
    s99 = information_depth(IMFP, TRMFP, emission_angle_deg=0.0,
                            percentage=99.0, model="jp2009")
    assert s99 / s90 == pytest.approx(2.0, rel=1e-12)


def test_the_three_depths_are_different_numbers():
    """One material, one energy, one angle -- three answers."""
    angle = 45.0
    eal = overlayer_eal(IMFP, TRMFP, model="jp2020_unpolarized")
    med = mean_escape_depth(IMFP, TRMFP, emission_angle_deg=angle,
                            model="jp2009")
    idepth = information_depth(IMFP, TRMFP, emission_angle_deg=angle,
                               percentage=95.0, model="jp2009")
    assert eal != pytest.approx(med, rel=1e-3)
    assert med != pytest.approx(idepth, rel=1e-3)
    # The EAL has no angle in it at all; the MED at this angle is well
    # below it purely because of the cos.
    assert med < eal < idepth


# --- required arguments ---------------------------------------------


def test_the_model_is_required_and_has_no_default():
    with pytest.raises(TypeError):
        mean_escape_depth(IMFP, TRMFP, emission_angle_deg=0.0)
    with pytest.raises(TypeError):
        information_depth(IMFP, TRMFP, emission_angle_deg=0.0,
                          percentage=95.0)


def test_an_eal_model_name_is_not_accepted_for_a_depth():
    """Three quantities, three slopes; the names must not cross over."""
    with pytest.raises(ValueError, match="different slopes"):
        mean_escape_depth(IMFP, TRMFP, emission_angle_deg=0.0,
                          model="jp2020_unpolarized")
    with pytest.raises(ValueError, match="different slopes"):
        information_depth_ratio(IMFP, TRMFP, model="jp2020_polarized_haxpes")


def test_the_emission_angle_is_required():
    """Optional metadata for the EAL; an arithmetic input here."""
    with pytest.raises(TypeError):
        mean_escape_depth(IMFP, TRMFP, model="jp2009")
    with pytest.raises(TypeError):
        information_depth(IMFP, TRMFP, percentage=95.0, model="jp2009")


def test_the_percentage_is_required_and_bounded():
    with pytest.raises(TypeError):
        information_depth(IMFP, TRMFP, emission_angle_deg=0.0,
                          model="jp2009")
    for bad in (0.0, 100.0, -5.0, 150.0, math.nan):
        with pytest.raises(ValueError, match="percentage"):
            information_depth(IMFP, TRMFP, emission_angle_deg=0.0,
                              percentage=bad, model="jp2009")


@pytest.mark.parametrize("fraction", [0.90, 0.95, 0.99])
def test_a_fraction_passed_as_a_percentage_is_refused(fraction):
    """``IMFP.sampling_depth`` next door takes a fraction in (0, 1).

    Passing 0.95 here would otherwise return a depth 314 times too
    small, with no warning -- the silent wrong answer this module's
    docstring says a caller should not be able to get.
    """
    with pytest.raises(ValueError, match="not a fraction"):
        information_depth(IMFP, TRMFP, emission_angle_deg=0.0,
                          percentage=fraction, model="jp2009")
    with pytest.raises(ValueError, match="not a fraction"):
        information_depth_report(
            described(IMFP), described(TRMFP),
            model="jp2009", emission_angle_deg=0.0, percentage=fraction)


@pytest.mark.parametrize("angle", [-1.0, 90.1, 180.0, math.nan])
def test_an_impossible_emission_angle_is_rejected(angle):
    with pytest.raises(ValueError, match="emission_angle_deg"):
        mean_escape_depth(IMFP, TRMFP, emission_angle_deg=angle,
                          model="jp2009")


@pytest.mark.parametrize(
    "imfp, trmfp", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (math.nan, 1.0)])
def test_non_physical_lengths_are_rejected(imfp, trmfp):
    with pytest.raises(ValueError):
        mean_escape_depth(imfp, trmfp, emission_angle_deg=0.0,
                          model="jp2009")


# --- against the source's own tabulated albedos ---------------------

# Five of the 25 lines of Tables I and II of Jablonski & Powell 2009,
# quoted for verification: (label, TRMFP, IMFP, published omega). They
# span the albedo range the paper covers, from its lowest-albedo
# photoelectron line to its highest-albedo Auger line.
_TABULATED = [
    ("Si 2p3/2 (Al Ka)", 244.4, 31.6, 0.115),
    ("W N5N67N67", 16.6, 4.7, 0.220),
    ("HfO2 Hf 4d5/2 (Al Ka)", 57.8, 20.4, 0.261),
    ("Ag 3d5/2 (Al Ka)", 36.9, 16.3, 0.306),
    ("Au N7VV", 5.1, 5.3, 0.508),
]


@pytest.mark.parametrize("label, trmfp, imfp, omega", _TABULATED)
def test_the_albedo_column_of_the_source_is_reproduced(
        label, trmfp, imfp, omega):
    """Eq. (15): omega = IMFP/(IMFP + TRMFP), as the paper tabulates it.

    The tolerance is set by the paper's own rounding: it prints both
    lengths to 0.1 Angstrom, which at the shortest of them (Au N7VV,
    5.3 A) is worth 0.002 in omega.
    """
    assert single_scattering_albedo(imfp, trmfp) == pytest.approx(
        omega, abs=0.003), label


def test_the_recommended_albedo_range_is_narrower_than_the_fitted_set():
    """The paper's lines reach 0.508; its recommendation says 0.1-0.5.

    Recording the recommendation means the topmost line of its own set
    is flagged. That is the honest reading, not a transcription error.
    """
    _, trmfp, imfp, _ = _TABULATED[-1]
    report = mean_escape_depth_report(
        described(imfp), described(trmfp),
        model="jp2009", emission_angle_deg=0.0)
    assert report.albedo > 0.5
    assert report.albedo_check == "outside"
    assert any("albedo" in w for w in report.warnings)


# --- the record -----------------------------------------------------


def test_the_report_value_matches_the_scalar_function():
    report = information_depth_report(
        described(IMFP), described(TRMFP),
        model="jp2009", emission_angle_deg=30.0, percentage=95.0)
    assert report.value == pytest.approx(
        information_depth(IMFP, TRMFP, emission_angle_deg=30.0,
                          percentage=95.0, model="jp2009"), rel=1e-12)
    assert report.unit == "angstrom"
    assert report.albedo == pytest.approx(OMEGA, rel=1e-12)
    assert report.percentage == 95.0


def test_a_haxpes_energy_is_flagged_as_an_extrapolation():
    """These relations stop at 2 keV; the paper declined to claim more."""
    report = mean_escape_depth_report(
        described(IMFP, kinetic_energy_ev=7413.0),
        described(TRMFP, kinetic_energy_ev=7413.0),
        model="jp2009", emission_angle_deg=45.0)
    assert report.kinetic_energy_check == "outside"
    assert any("extrapolation" in w for w in report.warnings)


def test_grazing_emission_is_flagged():
    report = mean_escape_depth_report(
        described(IMFP), described(TRMFP),
        model="jp2009", emission_angle_deg=75.0)
    assert report.emission_angle_check == "outside"
    assert any("emission angle" in w for w in report.warnings)


def test_the_record_names_the_concept_and_its_own_baseline():
    med = mean_escape_depth_report(
        described(IMFP), described(TRMFP),
        model="jp2009", emission_angle_deg=30.0).to_dict()
    idepth = information_depth_report(
        described(IMFP), described(TRMFP),
        model="jp2009", emission_angle_deg=30.0, percentage=95.0).to_dict()

    assert med["length_concept"]["concept"] == "mean_escape_depth"
    assert idepth["length_concept"]["concept"] == "information_depth"
    assert "average depth" in med["length_concept"]["operational_definition"]
    assert "95%" in idepth["length_concept"]["operational_definition"]
    # The ratio is to the elastic-free depth, not to the IMFP, and the
    # record has to say which.
    assert "IMFP cos(alpha)" in (
        med["length_concept"]["elastic_scattering_treatment"])
    assert "ln[1/(1 - P/100)]" in (
        idepth["length_concept"]["elastic_scattering_treatment"])
    # A mean escape depth has no percentage, as against one the caller
    # failed to declare -- the module keeps those two words apart.
    assert med["inputs"]["percentage"] == "not_applicable"
    assert idepth["inputs"]["percentage"] == 95.0


def test_the_record_keeps_the_same_top_level_shape_as_the_eal_record():
    record = mean_escape_depth_report(
        described(IMFP), described(TRMFP),
        model="jp2009", emission_angle_deg=30.0).to_dict()
    assert set(record) == {
        "value", "unit", "ratio", "albedo", "model", "inputs",
        "validity", "consistency", "length_concept", "warnings"}
    assert set(record["validity"]) == {
        "kinetic_energy", "emission_angle", "albedo"}
    assert set(record["length_concept"]) == {
        "concept", "operational_definition", "angular_dependence",
        "elastic_scattering_treatment", "geometry_convention"}
    assert json.loads(json.dumps(record)) == record


def test_undeclared_facts_are_not_recorded_rather_than_passed():
    report = mean_escape_depth_report(
        DescribedLength(IMFP, "angstrom"), DescribedLength(TRMFP, "angstrom"),
        model="jp2009", emission_angle_deg=30.0)
    record = report.to_dict()
    assert record["consistency"] == {
        "material": "not_recorded",
        "kinetic_energy": "not_recorded",
        "source": "not_recorded",
    }
    assert record["validity"]["kinetic_energy"] == "not_recorded"
    assert report.warnings == ()  # nothing checked, so nothing failed


def test_the_pair_checks_are_the_same_ones_the_eal_report_applies():
    with pytest.raises(ValueError, match="one unit"):
        mean_escape_depth_report(
            described(IMFP, unit="nm"), described(TRMFP),
            model="jp2009", emission_angle_deg=0.0)
    with pytest.raises(ValueError, match="different material"):
        mean_escape_depth_report(
            described(IMFP, material="Au"), described(TRMFP, material="Si"),
            model="jp2009", emission_angle_deg=0.0)
    with pytest.raises(ValueError, match="same electron"):
        mean_escape_depth_report(
            described(IMFP, kinetic_energy_ev=1000.0),
            described(TRMFP, kinetic_energy_ev=1400.0),
            model="jp2009", emission_angle_deg=0.0)
    with pytest.raises(TypeError):
        mean_escape_depth_report(
            IMFP, TRMFP, model="jp2009", emission_angle_deg=0.0)


def test_differing_sources_warn_rather_than_raise():
    report = information_depth_report(
        described(IMFP, source="TPP-2M"),
        described(TRMFP, source="SESSA 2.2.2 (JTP)"),
        model="jp2009", emission_angle_deg=0.0, percentage=95.0)
    assert report.source_consistency == "inconsistent"
    assert any("source" in w for w in report.warnings)
