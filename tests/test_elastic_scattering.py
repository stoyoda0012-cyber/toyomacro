"""Effective attenuation length from the single-scattering albedo.

Golden values are the published slopes themselves, so these tests are
implementation-fidelity checks: they fail on a transcribed coefficient,
a missing model gate, or a lost validity constraint. There is no
"physical plausibility" layer here because we ship no albedo data — the
caller supplies IMFP and TRMFP, and the relations are exact given those.
"""

from __future__ import annotations

import math

import pytest

from toyomacro.data.elastic_scattering import (
    EAL_MODELS,
    overlayer_eal,
    overlayer_eal_ratio,
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
