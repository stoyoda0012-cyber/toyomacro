"""Effective attenuation length from the single-scattering albedo.

Golden values are the published slopes themselves, so these tests are
implementation-fidelity checks: they fail on a transcribed coefficient,
a missing model gate, or a lost validity constraint. There is no
"physical plausibility" layer here because we ship no albedo data — the
caller supplies IMFP and TRMFP, and the relations are exact given those.
"""

from __future__ import annotations

import json
import math

import pytest

from toyomacro.data.elastic_scattering import (
    EAL_MODELS,
    DescribedLength,
    overlayer_eal,
    overlayer_eal_ratio,
    overlayer_eal_report,
    seah_gilmore_2001_overlayer_eal_ratio,
    single_scattering_albedo,
)

# Si 1s photoelectrons travelling in Au, KE 7413 eV — the case the
# sibling HAXPES work runs on. (The albedo is a property of the Au
# matrix at that kinetic energy, not of the emitting level.)
AU_OMEGA = 0.2082
# Chosen so that imfp/(imfp+trmfp) == AU_OMEGA exactly.
AU_IMFP = 5.904
AU_TRMFP = AU_IMFP * (1.0 - AU_OMEGA) / AU_OMEGA


def test_albedo_definition():
    """omega = imfp / (imfp + trmfp), dimensionless."""
    assert single_scattering_albedo(10.0, 40.0) == pytest.approx(0.2)
    assert single_scattering_albedo(AU_IMFP, AU_TRMFP) == pytest.approx(
        AU_OMEGA, rel=1e-12
    )
    # Unit-agnostic: scaling both lengths leaves omega unchanged.
    assert single_scattering_albedo(1.0, 4.0) == pytest.approx(
        single_scattering_albedo(10.0, 40.0), rel=1e-12
    )


@pytest.mark.parametrize(
    "imfp, trmfp",
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0),
     (math.nan, 1.0), (math.inf, 1.0), (1.0, math.nan)],
)
def test_non_physical_lengths_are_rejected(imfp, trmfp):
    with pytest.raises(ValueError):
        single_scattering_albedo(imfp, trmfp)


def test_published_slopes():
    """The two 2020 relations, by their coefficients."""
    assert EAL_MODELS["jp2020_unpolarized"].slope == 0.738
    assert EAL_MODELS["jp2020_polarized_haxpes"].slope == 0.836


@pytest.mark.parametrize(
    "model, expected",
    [("jp2020_unpolarized", 1.0 - 0.738 * AU_OMEGA),
     ("jp2020_polarized_haxpes", 1.0 - 0.836 * AU_OMEGA)],
)
def test_ratio_matches_the_published_linear_form(model, expected):
    got = overlayer_eal_ratio(AU_IMFP, AU_TRMFP, model=model)
    assert got == pytest.approx(expected, rel=1e-12)


def test_polarization_choice_is_not_cosmetic():
    """For Au at 7.4 keV the two models differ by ~2.4% in the EAL.

    Small enough to be dismissed as noise, large enough to matter for a
    thickness. This is why `model` has no default.
    """
    unpol = overlayer_eal_ratio(
        AU_IMFP, AU_TRMFP, model="jp2020_unpolarized")
    pol = overlayer_eal_ratio(
        AU_IMFP, AU_TRMFP, model="jp2020_polarized_haxpes")

    assert unpol == pytest.approx(0.8463, abs=1e-4)
    assert pol == pytest.approx(0.8259, abs=1e-4)
    assert (unpol - pol) / pol == pytest.approx(0.0247, abs=1e-3)


def test_zero_albedo_gives_exactly_the_imfp():
    """With no elastic scattering the EAL must equal the IMFP.

    Both 2020 relations satisfy this by construction; the 2001 one does
    not, which is one reason it is kept only for comparison.
    """
    huge_trmfp = 1e12
    for model in EAL_MODELS:
        assert overlayer_eal_ratio(
            1.0, huge_trmfp, model=model) == pytest.approx(1.0, abs=1e-9)
        assert overlayer_eal(
            2.5, huge_trmfp, model=model) == pytest.approx(2.5, rel=1e-9)

    assert seah_gilmore_2001_overlayer_eal_ratio(
        1.0, huge_trmfp, 79) == pytest.approx(0.979, abs=1e-6)


def test_model_is_keyword_only_and_required():
    with pytest.raises(TypeError):
        overlayer_eal_ratio(AU_IMFP, AU_TRMFP)          # no model
    with pytest.raises(TypeError):
        overlayer_eal_ratio(AU_IMFP, AU_TRMFP, "jp2020_unpolarized")  # noqa
    with pytest.raises(ValueError, match="no default"):
        overlayer_eal_ratio(AU_IMFP, AU_TRMFP, model="jp2009")


def test_eal_is_imfp_times_ratio_from_the_same_pair():
    """The albedo is formed from the imfp it is then applied to.

    This is the one inconsistency the signature does rule out: a second,
    different IMFP cannot enter, because there is nowhere to pass one.
    """
    for model in EAL_MODELS:
        assert overlayer_eal(AU_IMFP, AU_TRMFP, model=model) == pytest.approx(
            AU_IMFP * overlayer_eal_ratio(AU_IMFP, AU_TRMFP, model=model),
            rel=1e-12,
        )
        assert overlayer_eal(AU_IMFP, AU_TRMFP, model=model) < AU_IMFP


def test_mismatched_sources_are_accepted_not_detected():
    """Records a limitation rather than a guarantee.

    Passing an IMFP from one source with a TRMFP from another is a real
    error, and nothing here can see it: the lengths carry no provenance.
    The two-argument signature makes consistent use natural, not
    mandatory. Detecting this would need the lengths to know their source,
    material and energy.
    """
    sessa_imfp, unrelated_trmfp = 64.4, 300.0

    got = overlayer_eal(
        sessa_imfp, unrelated_trmfp, model="jp2020_unpolarized")

    assert got > 0  # accepted silently, by design of the scalar API
    assert got == pytest.approx(
        sessa_imfp * (1 - 0.738 * (sessa_imfp / (sessa_imfp + unrelated_trmfp))),
        rel=1e-12,
    )


def test_validity_metadata_is_carried_not_lost():
    """The angle and energy ranges are part of the published result."""
    unpol = EAL_MODELS["jp2020_unpolarized"]
    pol = EAL_MODELS["jp2020_polarized_haxpes"]

    assert unpol.emission_angle_deg == (0.0, 50.0)
    assert pol.emission_angle_deg == (5.0, 50.0)
    assert pol.energy_ev == (100.0, 10000.0)
    assert unpol.rms_percent == pytest.approx(1.44)
    assert pol.rms_percent == pytest.approx(1.85)
    for m in EAL_MODELS.values():
        assert "Eq." in m.equation and m.geometry


def test_seah_gilmore_2001_reproduces_its_own_equation():
    """L/IMFP = 0.979 [1 - omega (0.955 - 0.0777 ln Z)], Eq. (35)."""
    z = 79
    expected = 0.979 * (1 - AU_OMEGA * (0.955 - 0.0777 * math.log(z)))

    got = seah_gilmore_2001_overlayer_eal_ratio(AU_IMFP, AU_TRMFP, z)
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(0.8535, abs=1e-4)


def test_seah_gilmore_disagrees_with_the_polarized_model():
    """~3% apart for Au at 7.4 keV — not interchangeable for HAXPES."""
    sg = seah_gilmore_2001_overlayer_eal_ratio(AU_IMFP, AU_TRMFP, 79)
    pol = overlayer_eal_ratio(
        AU_IMFP, AU_TRMFP, model="jp2020_polarized_haxpes")

    assert (sg - pol) / pol == pytest.approx(0.0334, abs=2e-3)


@pytest.mark.parametrize("z", [0, 0.5, -1, math.nan])
def test_seah_gilmore_rejects_impossible_atomic_numbers(z):
    with pytest.raises(ValueError):
        seah_gilmore_2001_overlayer_eal_ratio(AU_IMFP, AU_TRMFP, z)


# --- sensitivity of the EAL to its inputs ----------------------------------
#
# The TRMFP is caller-supplied by policy (docs/DATA_SOURCES.md), so it
# matters how hard an error in it hits the result. These are arithmetic
# consequences of L = imfp (1 - A omega), pinned so the propagation
# behaviour is a stated property rather than folklore. Both use the
# SESSA-derived Au pair at 7413 eV (IMFP 64.40 A, TRMFP 244.90 A,
# omega = 0.2082) — one material and energy, chosen for realism, not
# claimed to bound other systems.

SESSA_AU_IMFP_A = 64.40
SESSA_AU_TRMFP_A = 244.90


def test_a_trmfp_error_is_damped_in_the_eal():
    """+10% TRMFP moves the Au@7.4keV EAL by ~+1.3%, not 10%.

    First order, d ln L = A omega (1 - omega) / (1 - A omega) *
    (d trmfp / trmfp), a factor of ~0.14 at omega ~ 0.21 for the
    unpolarized slope (0.133 realized at a finite +10%), so a
    caller-supplied TRMFP with a modest error yields an EAL error below
    the relations' own 1.44-1.85% rms.
    """
    base = overlayer_eal(
        SESSA_AU_IMFP_A, SESSA_AU_TRMFP_A, model="jp2020_unpolarized")
    bumped = overlayer_eal(
        SESSA_AU_IMFP_A, 1.10 * SESSA_AU_TRMFP_A, model="jp2020_unpolarized")

    shift = bumped / base - 1.0
    assert shift == pytest.approx(0.01332, abs=2e-4)
    assert 0.0 < shift < 0.02


def test_an_imfp_error_passes_nearly_straight_through():
    """+10% IMFP moves the Au@7.4keV EAL by ~+8.5%.

    d ln L / d ln imfp = 1 - A omega (1 - omega) / (1 - A omega) ~ 0.86
    here: the EAL inherits IMFP uncertainty almost one-to-one, so its
    error budget is dominated by whatever IMFP is fed into it.
    """
    base = overlayer_eal(
        SESSA_AU_IMFP_A, SESSA_AU_TRMFP_A, model="jp2020_unpolarized")
    bumped = overlayer_eal(
        1.10 * SESSA_AU_IMFP_A, SESSA_AU_TRMFP_A, model="jp2020_unpolarized")

    shift = bumped / base - 1.0
    assert shift == pytest.approx(0.0845, abs=5e-4)
    assert shift > 0.08


# --- DescribedLength and overlayer_eal_report ------------------------------
#
# The report layer adds no arithmetic; it records model, inputs and
# checks. Golden values therefore stay the published slopes, plus the
# vocabulary contract: declared facts are checked, undeclared facts are
# "not_recorded", and "not_recorded" is never a pass.


def _au_imfp(**kw):
    defaults = dict(
        value=SESSA_AU_IMFP_A, unit="angstrom",
        source="SESSA 2.2.2 (JTP)", material="Au",
        kinetic_energy_ev=7412.7)
    defaults.update(kw)
    return DescribedLength(**defaults)


def _au_trmfp(**kw):
    defaults = dict(
        value=SESSA_AU_TRMFP_A, unit="angstrom",
        source="SESSA 2.2.2 (JTP)", material="Au",
        kinetic_energy_ev=7412.7)
    defaults.update(kw)
    return DescribedLength(**defaults)


def test_report_numbers_equal_the_scalar_api():
    r = overlayer_eal_report(
        _au_imfp(), _au_trmfp(), model="jp2020_polarized_haxpes")

    assert r.albedo == pytest.approx(
        single_scattering_albedo(SESSA_AU_IMFP_A, SESSA_AU_TRMFP_A), rel=1e-12)
    assert r.ratio == pytest.approx(
        overlayer_eal_ratio(
            SESSA_AU_IMFP_A, SESSA_AU_TRMFP_A,
            model="jp2020_polarized_haxpes"),
        rel=1e-12)
    assert r.value == pytest.approx(
        overlayer_eal(
            SESSA_AU_IMFP_A, SESSA_AU_TRMFP_A,
            model="jp2020_polarized_haxpes"),
        rel=1e-12)
    assert r.unit == "angstrom"


def test_report_rejects_bare_floats():
    with pytest.raises(TypeError):
        overlayer_eal_report(
            SESSA_AU_IMFP_A, SESSA_AU_TRMFP_A, model="jp2020_unpolarized")


@pytest.mark.parametrize(
    "kw", [dict(value=0.0), dict(value=-1.0), dict(value=math.nan),
           dict(unit="cm"), dict(kinetic_energy_ev=-5.0),
           dict(kinetic_energy_ev=math.inf)],
)
def test_described_length_validates_its_fields(kw):
    with pytest.raises(ValueError):
        _au_imfp(**kw)


def test_report_unit_mismatch_raises():
    with pytest.raises(ValueError, match="one unit"):
        overlayer_eal_report(
            _au_imfp(unit="nm", value=6.440), _au_trmfp(),
            model="jp2020_unpolarized")


def test_report_is_unit_covariant_when_units_agree():
    in_a = overlayer_eal_report(
        _au_imfp(), _au_trmfp(), model="jp2020_unpolarized")
    in_nm = overlayer_eal_report(
        _au_imfp(unit="nm", value=SESSA_AU_IMFP_A / 10.0),
        _au_trmfp(unit="nm", value=SESSA_AU_TRMFP_A / 10.0),
        model="jp2020_unpolarized")

    assert in_nm.ratio == pytest.approx(in_a.ratio, rel=1e-12)
    assert in_nm.value == pytest.approx(in_a.value / 10.0, rel=1e-12)
    assert in_nm.unit == "nm"


def test_report_material_mismatch_raises():
    with pytest.raises(ValueError, match="different material"):
        overlayer_eal_report(
            _au_imfp(), _au_trmfp(material="Si"), model="jp2020_unpolarized")


def test_report_material_case_is_not_a_mismatch():
    r = overlayer_eal_report(
        _au_imfp(material="au"), _au_trmfp(material="Au"),
        model="jp2020_unpolarized")
    assert r.to_dict()["consistency"]["material"] == "consistent"


def test_report_kinetic_energy_mispairing_raises():
    """Au 4f vs Si 1s at Ga K-alpha: 9167.7 eV against 7412.7 eV.

    This is the real mistake the check exists for — reading each length
    at a different photoemission line. 24% apart in energy, ~19% in
    IMFP.
    """
    with pytest.raises(ValueError, match="same electron"):
        overlayer_eal_report(
            _au_imfp(kinetic_energy_ev=7412.7),
            _au_trmfp(kinetic_energy_ev=9167.7),
            model="jp2020_polarized_haxpes")


def test_report_near_degenerate_pair_is_accepted():
    """Au 4f / Si 2p kinetic energies differ by 0.2% — a genuine pair."""
    r = overlayer_eal_report(
        _au_imfp(kinetic_energy_ev=9167.7),
        _au_trmfp(kinetic_energy_ev=9152.7),
        model="jp2020_polarized_haxpes")
    assert r.to_dict()["consistency"]["kinetic_energy"] == "consistent"


def test_report_flags_the_ga_kalpha_extrapolation():
    """At 7413 eV the unpolarized relation is an extrapolation.

    Its fitted range stops at 4426 eV; only the polarized-HAXPES
    relation reaches 10 keV. This is the live case the validity check
    exists for: an unpolarized lab HAXPES source (Ga K-alpha) sits
    outside the unpolarized fit range.
    """
    unpol = overlayer_eal_report(
        _au_imfp(), _au_trmfp(), model="jp2020_unpolarized")
    pol = overlayer_eal_report(
        _au_imfp(), _au_trmfp(), model="jp2020_polarized_haxpes")

    assert unpol.kinetic_energy_check == "outside"
    assert any("extrapolation" in w for w in unpol.warnings)
    assert pol.kinetic_energy_check == "within"
    assert not pol.warnings


def test_report_checks_the_emission_angle_range():
    """Normal emission is inside the unpolarized range, outside the
    polarized one (5-50 deg)."""
    unpol = overlayer_eal_report(
        _au_imfp(kinetic_energy_ev=None), _au_trmfp(kinetic_energy_ev=None),
        model="jp2020_unpolarized", emission_angle_deg=0.0)
    pol = overlayer_eal_report(
        _au_imfp(), _au_trmfp(),
        model="jp2020_polarized_haxpes", emission_angle_deg=0.0)

    assert unpol.emission_angle_check == "within"
    assert pol.emission_angle_check == "outside"
    assert any("emission angle" in w for w in pol.warnings)


@pytest.mark.parametrize("angle", [-1.0, 90.5, math.nan])
def test_report_rejects_impossible_emission_angles(angle):
    with pytest.raises(ValueError, match="emission_angle_deg"):
        overlayer_eal_report(
            _au_imfp(), _au_trmfp(), model="jp2020_unpolarized",
            emission_angle_deg=angle)


def test_report_source_mismatch_is_recorded_not_raised():
    """Cross-source pairing is discouraged but can be deliberate."""
    r = overlayer_eal_report(
        _au_imfp(source="TPP-2M"), _au_trmfp(),
        model="jp2020_polarized_haxpes")

    assert r.source_consistency == "inconsistent"
    assert any("source" in w for w in r.warnings)
    assert r.value > 0  # still computed


def test_report_undeclared_is_not_recorded_and_not_a_pass():
    r = overlayer_eal_report(
        DescribedLength(SESSA_AU_IMFP_A, "angstrom"),
        DescribedLength(SESSA_AU_TRMFP_A, "angstrom"),
        model="jp2020_polarized_haxpes")
    d = r.to_dict()

    assert r.kinetic_energy_check == "not_recorded"
    assert r.emission_angle_check == "not_recorded"
    assert r.source_consistency == "not_recorded"
    assert d["consistency"]["material"] == "not_recorded"
    assert d["consistency"]["kinetic_energy"] == "not_recorded"
    assert not r.warnings  # nothing was checked, so nothing passed or failed
    assert r.kinetic_energy_check != "within"


@pytest.mark.parametrize("field", ["source", "material"])
@pytest.mark.parametrize("spelling", ["not_recorded", " Not_Recorded "])
def test_described_length_rejects_the_reserved_marker(field, spelling):
    """A literal 'not_recorded' source/material would be
    indistinguishable in to_dict() from an absent declaration."""
    with pytest.raises(ValueError, match="reserved"):
        _au_imfp(**{field: spelling})


def test_described_length_coerces_numpy_scalars_for_json():
    """A numpy scalar input must not break the plain-JSON promise."""
    np = pytest.importorskip("numpy")
    r = overlayer_eal_report(
        _au_imfp(value=np.float32(SESSA_AU_IMFP_A)),
        _au_trmfp(kinetic_energy_ev=np.float64(7412.7)),
        model="jp2020_unpolarized")
    json.dumps(r.to_dict())  # would raise TypeError without coercion


def test_to_dict_rederives_consistency_for_hand_built_reports():
    """The consistency block re-compares the inputs, so a report
    assembled around the factory cannot claim a consistency it lacks."""
    import dataclasses

    r = overlayer_eal_report(
        _au_imfp(), _au_trmfp(), model="jp2020_unpolarized")
    forged = dataclasses.replace(r, trmfp=_au_trmfp(material="Si"))

    assert forged.to_dict()["consistency"]["material"] == "inconsistent"


def test_report_to_dict_is_json_ready_and_complete():
    r = overlayer_eal_report(
        _au_imfp(), _au_trmfp(),
        model="jp2020_polarized_haxpes", emission_angle_deg=30.0)
    d = json.loads(json.dumps(r.to_dict()))

    assert set(d) == {
        "value", "unit", "ratio", "albedo", "model", "inputs",
        "validity", "consistency", "length_concept", "warnings"}
    assert d["model"]["slope"] == 0.836
    assert d["model"]["energy_ev"] == [100.0, 10000.0]
    assert d["inputs"]["imfp"]["material"] == "Au"
    assert d["inputs"]["emission_angle_deg"] == 30.0
    # The four descriptors an operationally defined length concept
    # needs; consumers may rely on exactly these key names.
    assert set(d["length_concept"]) == {
        "concept", "operational_definition", "angular_dependence",
        "elastic_scattering_treatment", "geometry_convention"}
    assert d["length_concept"]["concept"] == "eal"
