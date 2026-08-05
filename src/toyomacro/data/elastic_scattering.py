"""Surface-sensitivity depths from the single-scattering albedo.

Elastic scattering shortens the distance over which a photoelectron signal
decays, so the inelastic mean free path (IMFP) overestimates it. The
correction is governed by one dimensionless number, the single-scattering
albedo

    omega = IMFP / (IMFP + TRMFP)

where TRMFP is the transport mean free path. Over a useful range of
emission angles each depth, divided by what it would be with elastic
scattering switched off, falls linearly in omega,

    quantity / quantity_without_elastic_scattering = 1 - A omega

with a different slope A for each quantity and geometry. For the
overlayer-thickness EAL the slope also depends on whether the exciting
radiation is polarized: for gold at 7.4 keV (omega = 0.208) the
unpolarized and polarized slopes differ by 2.4% in the resulting EAL,
which is why the model is not optional here.

**No default model, and no default albedo.** Both encode assumptions about
the experiment that only the caller knows. The predecessor of this module
applied a hardcoded EAL/IMFP ratio of 0.9 and described it as an
elastic-scattering correction; requiring omega means a caller who cannot
supply it cannot silently get a wrong answer instead.

Three different lengths
-----------------------

These are distinct quantities with distinct slopes, and they answer
distinct questions. Do not substitute one for another.

- **Overlayer-thickness EAL** (``L_TH``, :func:`overlayer_eal`): the L
  that replaces the IMFP in ``t = -L cos(alpha) ln(1 - R)`` when a film
  thickness is determined from overlayer or substrate intensities.
- **Mean escape depth** (``D``, :func:`mean_escape_depth`): the average
  depth normal to the surface that detected electrons came from.
- **Information depth** (``S``, :func:`information_depth`): the depth
  above which a stated percentage of the detected signal originates.

Marker depth and core-shell nanoparticle EALs follow yet other slopes
and are not implemented here. Only the EAL has a polarized-HAXPES
model here; see *References* for the polarized mean escape depth that
exists in the literature and why it is not shipped.

Scope and limits
----------------

- **L_TH has no angle dependence; D and S do.** The EAL relations are
  averages over their stated emission-angle range, within which L is
  treated as constant, so the angle is metadata there. D and S carry an
  explicit ``cos(alpha)`` and take the emission angle as a required
  argument. None of them is valid at grazing emission.
- **IMFP and TRMFP must come from the same source and conditions.** omega
  is a ratio of two lengths for one material at one kinetic energy, so an
  albedo derived from one dataset must not be applied to an IMFP from
  another. Taking both lengths rather than an albedo and an IMFP makes the
  consistent use the natural one — but the scalar functions do not enforce
  it. Nothing there can tell that ``overlayer_eal(64.4, 300.0, ...)``
  mixed a SESSA IMFP with a TRMFP from elsewhere. To have the mixing
  errors caught, declare what you know about each length with
  :class:`DescribedLength` and call :func:`overlayer_eal_report`: declared
  units, materials and kinetic energies are checked, and what was not
  declared is reported as ``not_recorded`` rather than silently assumed
  consistent.
- **No table lookup.** TRMFP values are the caller's responsibility; no
  TRMFP or albedo data is bundled with this package. See
  ``docs/DATA_SOURCES.md`` under *Deliberately NOT bundled* for the
  candidate source and why it is not shipped.
  :mod:`toyomacro.data.sessa` reads a consistent pair out of a SESSA
  output file the user generated themselves.

References
----------

Jablonski & Powell, *J. Phys. Chem. Ref. Data* **49**, 033102 (2020),
DOI 10.1063/5.0008576 — Eq. (20) for unpolarized x rays and Eq. (62) for
linearly polarized HAXPES. Source of the two EAL slopes.

Jablonski & Powell, *J. Vac. Sci. Technol. A* **27**, 253 (2009),
DOI 10.1116/1.3071947 — Eqs. (28) and (29). Source of the mean-escape-depth
and information-depth slopes. The 2020 review supersedes this paper's EAL
slope (0.735 there, 0.738 in its Eq. (20)) but **restates both of these
two unchanged**, as its Eqs. (A5) and (A12), citing this paper for them.
They are therefore current, and are named for the paper that derived
them rather than for the review that repeats them.

Not implemented, and why:

- **A polarized-x-ray MED does exist**: Jablonski & Powell 2020
  Eq. (A8), ``R_MED = 1 - 0.831 omega``, relative rms deviation 1.86%,
  emission angles 0-50 degrees, from non-dipolar cross sections for
  four elements and 13 photoelectron lines — the natural counterpart to
  the polarized EAL of Eq. (62). Its fitted *energy* range is not
  stated where the review presents it; the underlying paper (Jablonski,
  *Surf. Sci.* **667**, 121 (2018)) has not been read here. The review
  does say elsewhere that the same 13-line calculation ran at 100 eV to
  10 keV, but that is an inference across two sections, not a stated
  limit, and shipping a relation whose validity range was inferred is
  not something this module does. Read that paper before adding it.
- Eq. (A6) of the same review, ``D = 0.981 g_beta (1 - 0.736 omega)
  lambda cos alpha`` with ``g_beta`` a cubic in the asymmetry parameter
  (Tanuma *et al.*, *J. Electron Spectrosc. Relat. Phenom.* **190**,
  127 (2013)), is the dipole-approximation predecessor of Eq. (A8); the
  review states Eq. (A8) is believed to be the more accurate of the two.
- The review gives no polarized counterpart to the information depth.

Seah & Gilmore, *Surf. Interface Anal.* **31**, 835 (2001),
DOI 10.1002/sia.1113 — Eq. (35), provided for comparison only; see
:func:`seah_gilmore_2001_overlayer_eal_ratio`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "EALModel",
    "EAL_MODELS",
    "LengthUnit",
    "SamplingDepthModel",
    "MED_MODELS",
    "INFORMATION_DEPTH_MODELS",
    "DescribedLength",
    "OverlayerEALReport",
    "SamplingDepthReport",
    "single_scattering_albedo",
    "overlayer_eal_ratio",
    "overlayer_eal",
    "overlayer_eal_report",
    "mean_escape_depth_ratio",
    "mean_escape_depth",
    "mean_escape_depth_report",
    "information_depth_ratio",
    "information_depth",
    "information_depth_report",
    "seah_gilmore_2001_overlayer_eal_ratio",
]

EALModel = Literal["jp2020_unpolarized", "jp2020_polarized_haxpes"]

SamplingDepthModel = Literal["jp2009"]

LengthUnit = Literal["nm", "angstrom"]

#: Availability vocabulary: ``not_recorded`` means the fact exists — an
#: IMFP was computed for *some* material at *some* energy — but the
#: caller did not declare it here. It is not a pass, it is not
#: ``unknown``, and it is not ``not_applicable``; a check that could not
#: run reports this string rather than silently assuming consistency.
NOT_RECORDED = "not_recorded"

#: The other half of that vocabulary: the fact does not exist for this
#: quantity, so there was nothing to declare. A mean escape depth has no
#: signal percentage — reporting that as ``not_recorded`` would tell a
#: consumer the caller forgot to state one.
NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class _Model:
    """A published linear relation L_TH/IMFP = 1 - slope * omega."""

    slope: float
    equation: str
    emission_angle_deg: tuple[float, float]
    energy_ev: tuple[float, float]
    rms_percent: float
    geometry: str


EAL_MODELS: dict[str, _Model] = {
    "jp2020_unpolarized": _Model(
        slope=0.738,
        equation="Jablonski & Powell 2020, Eq. (20)",
        emission_angle_deg=(0.0, 50.0),
        energy_ev=(321.0, 4426.0),
        rms_percent=1.44,
        geometry=(
            "unpolarized x rays; 55 degrees between x-ray incidence and "
            "photoelectron emission. Supersedes the 0.735 slope of "
            "Jablonski & Powell 2009 Eq. (30), which used a smaller "
            "material and energy set and is numerically almost identical."
        ),
    ),
    "jp2020_polarized_haxpes": _Model(
        slope=0.836,
        equation="Jablonski & Powell 2020, Eq. (62)",
        emission_angle_deg=(5.0, 50.0),
        energy_ev=(100.0, 10000.0),
        rms_percent=1.85,
        geometry=(
            "linearly polarized x rays, for the configuration of that "
            "paper's Fig. 22: (a) the polarization vector lies in the "
            "plane perpendicular to the sample surface, (b) the analyzer "
            "axis is perpendicular to the x-ray beam, (c) x-ray incidence "
            "is near glancing. A synchrotron HAXPES arrangement that "
            "violates these does not obey this slope."
        ),
    ),
}


@dataclass(frozen=True)
class _DepthModel:
    """A published linear relation quantity/quantity_SLA = 1 - slope * omega.

    Separate from :class:`_Model` on purpose. The fit quality quoted for
    these relations is a *mean* percentage deviation, not an rms, and
    the paper states an albedo range of validity that the EAL entries
    here do not carry. Reusing one dataclass would have meant labelling
    one statistic with the other's name.
    """

    slope: float
    equation: str
    emission_angle_deg: tuple[float, float]
    energy_ev: tuple[float, float]
    albedo: tuple[float, float]
    mean_deviation_percent: float
    geometry: str


# Jablonski & Powell 2009 derived the MED, ID and EAL relations from one
# set of calculations, so the three share a validity envelope: 25
# photoelectron and Auger lines of five elemental solids (Z 14-79) and
# four inorganic compounds, emission angles 0-50 degrees, and an XPS
# configuration with psi = 54 degrees between the analyzer axis and the
# x-ray direction. (The 2020 review, restating these as its Eqs. (A5)
# and (A12), calls the same configuration psi = 55 degrees. The value
# recorded here is the one in the paper that derived the relations; the
# review's own EAL work is at 55 degrees, and the difference is inside
# the ~2% the 2009 paper reports for a 40-to-70-degree change in psi.)
# The energy range below is the union of the
# photoelectron (156-1455 eV) and Auger (61-2016 eV) energies covered;
# the paper says the relations may well hold above 2 keV but explicitly
# did not test that, so HAXPES energies come back flagged as an
# extrapolation rather than silently accepted.
#
# The albedo range below is the 0.1-0.5 the paper states when it
# recommends these expressions, not the 0.103-0.508 its own lines
# actually span. The recommendation is the narrower claim, so it is the
# one recorded; a line at the top of the fitted set therefore comes back
# marginally "outside".
_JP2009_ANGLE_RANGE = (0.0, 50.0)
_JP2009_ENERGY_RANGE = (61.0, 2016.0)
_JP2009_ALBEDO_RANGE = (0.1, 0.5)
_JP2009_GEOMETRY = (
    "conventional XPS or AES with psi = 54 degrees between the analyzer "
    "axis and the x-ray direction, close to the magic angle (the 2020 "
    "review calls the same configuration 55 degrees when it restates "
    "these relations; 54 is the value in the paper that derived them). "
    "The paper "
    "reports that the slope changes by about 2% for psi = 40 or 70 "
    "degrees, and that psi is between 49 and 60 degrees on many "
    "commercial instruments. For AES the angle between the primary beam "
    "and the analyzer does not enter. Unpolarized excitation: for the "
    "mean escape depth a linearly polarized counterpart exists in the "
    "literature (Jablonski & Powell 2020 Eq. (A8), slope 0.831) but is "
    "not implemented here, and for the information depth none is known; "
    "see the module docstring."
)

MED_MODELS: dict[str, _DepthModel] = {
    "jp2009": _DepthModel(
        slope=0.736,
        equation="Jablonski & Powell 2009, Eqs. (22) and (28)",
        emission_angle_deg=_JP2009_ANGLE_RANGE,
        energy_ev=_JP2009_ENERGY_RANGE,
        albedo=_JP2009_ALBEDO_RANGE,
        mean_deviation_percent=0.25,
        geometry=_JP2009_GEOMETRY,
    ),
}

INFORMATION_DEPTH_MODELS: dict[str, _DepthModel] = {
    "jp2009": _DepthModel(
        slope=0.787,
        equation="Jablonski & Powell 2009, Eqs. (23) and (29)",
        emission_angle_deg=_JP2009_ANGLE_RANGE,
        energy_ev=_JP2009_ENERGY_RANGE,
        albedo=_JP2009_ALBEDO_RANGE,
        mean_deviation_percent=0.54,
        geometry=_JP2009_GEOMETRY,
    ),
}


def _check_length(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def _resolve(model: str) -> _Model:
    try:
        return EAL_MODELS[model]
    except KeyError:
        raise ValueError(
            f"unknown model {model!r}; available: {sorted(EAL_MODELS)}. "
            "There is no default: the slope depends on whether the x rays "
            "are polarized and on the measurement geometry."
        ) from None


def single_scattering_albedo(imfp: float, trmfp: float) -> float:
    """Return omega = imfp / (imfp + trmfp).

    Both lengths must be in the *same* unit — nm or Angstrom, either is
    fine, since omega is dimensionless — and must describe the same
    material at the same kinetic energy.

    Args:
        imfp: Inelastic mean free path.
        trmfp: Transport mean free path.

    Returns:
        The albedo, in (0, 1). It approaches 0 as elastic scattering
        becomes negligible (trmfp much larger than imfp).

    Raises:
        ValueError: If either length is not finite and positive.
    """
    imfp = _check_length("imfp", imfp)
    trmfp = _check_length("trmfp", trmfp)
    return imfp / (imfp + trmfp)


def overlayer_eal_ratio(imfp: float, trmfp: float, *, model: EALModel) -> float:
    """Return L_TH / IMFP for overlayer-thickness measurements.

    Args:
        imfp: Inelastic mean free path.
        trmfp: Transport mean free path, same unit and source as ``imfp``.
        model: Keyword-only and required. ``"jp2020_unpolarized"`` for a
            conventional lab source, ``"jp2020_polarized_haxpes"`` for
            linearly polarized synchrotron HAXPES in the geometry
            described in :data:`EAL_MODELS`.

    Returns:
        The dimensionless ratio, exactly 1 when omega is 0.

    Raises:
        ValueError: If a length is invalid or the model name is unknown.
    """
    return 1.0 - _resolve(model).slope * single_scattering_albedo(imfp, trmfp)


def overlayer_eal(imfp: float, trmfp: float, *, model: EALModel) -> float:
    """Return L_TH in the unit of ``imfp``.

    The albedo is formed from ``imfp`` and ``trmfp`` and applied to that
    same ``imfp``, so no second IMFP can enter. Whether the two lengths
    themselves come from one source, material and energy is not something
    this function can check — see the module docstring.

    Args:
        imfp: Inelastic mean free path.
        trmfp: Transport mean free path, same unit and source as ``imfp``.
        model: See :func:`overlayer_eal_ratio`. Keyword-only, required.

    Returns:
        L_TH, shorter than ``imfp`` whenever elastic scattering is present.
    """
    return imfp * overlayer_eal_ratio(imfp, trmfp, model=model)


def _sampling_depth_report(
    imfp: DescribedLength,
    trmfp: DescribedLength,
    *,
    concept: str,
    m: _DepthModel,
    model: str,
    emission_angle_deg: float,
    percentage: float | None,
) -> SamplingDepthReport:
    if not isinstance(imfp, DescribedLength) or not isinstance(
            trmfp, DescribedLength):
        raise TypeError(
            "the report functions take DescribedLength inputs; for bare "
            "numbers use mean_escape_depth / information_depth")
    angle = _check_emission_angle(emission_angle_deg)
    checks = _check_pair(
        imfp, trmfp,
        model_name=model,
        energy_ev=m.energy_ev,
        emission_angle_range=m.emission_angle_deg,
        emission_angle_deg=angle,
        albedo_range=m.albedo,
    )
    ratio = 1.0 - m.slope * checks.albedo
    value = imfp.value * math.cos(math.radians(angle)) * ratio
    if percentage is not None:
        value *= math.log(1.0 / (1.0 - percentage / 100.0))
    return SamplingDepthReport(
        value=value,
        unit=imfp.unit,
        ratio=ratio,
        albedo=checks.albedo,
        concept=concept,
        model_name=model,
        imfp=imfp,
        trmfp=trmfp,
        emission_angle_deg=angle,
        percentage=percentage,
        kinetic_energy_check=checks.kinetic_energy_check,
        emission_angle_check=checks.emission_angle_check,
        albedo_check=checks.albedo_check,
        source_consistency=checks.source_consistency,
        warnings=checks.warnings,
    )


def mean_escape_depth_report(
    imfp: DescribedLength,
    trmfp: DescribedLength,
    *,
    model: SamplingDepthModel,
    emission_angle_deg: float,
) -> SamplingDepthReport:
    """Compute D and return it with its inputs, model and checks.

    Same arithmetic as :func:`mean_escape_depth`; what this adds is the
    record, on the same terms as :func:`overlayer_eal_report` — declared
    unit, material and kinetic-energy mismatches raise, being outside a
    model's stated range is recorded and warned about, and an undeclared
    fact is ``"not_recorded"`` rather than a pass. (Stated, not fitted:
    the energy and angle ranges are what the relation was fitted over,
    but the albedo range is what the paper recommends, and is narrower
    than the set it fitted.)

    Args:
        imfp: Inelastic mean free path with its declarations.
        trmfp: Transport mean free path with its declarations.
        model: Keyword-only and required; see :data:`MED_MODELS`.
        emission_angle_deg: Emission angle from the surface normal, in
            degrees. Keyword-only and required — unlike the EAL, it
            enters the arithmetic.

    Returns:
        A :class:`SamplingDepthReport` with ``concept ==
        "mean_escape_depth"``.

    Raises:
        TypeError: If the lengths are not :class:`DescribedLength`.
        ValueError: On a declared unit, material or kinetic-energy
            mismatch, an invalid length, an unknown model, or an
            emission angle outside [0, 90] degrees.
    """
    return _sampling_depth_report(
        imfp, trmfp,
        concept="mean_escape_depth",
        m=_resolve_depth(MED_MODELS, model, "mean escape depth"),
        model=model,
        emission_angle_deg=emission_angle_deg,
        percentage=None,
    )


def information_depth_report(
    imfp: DescribedLength,
    trmfp: DescribedLength,
    *,
    model: SamplingDepthModel,
    emission_angle_deg: float,
    percentage: float,
) -> SamplingDepthReport:
    """Compute S and return it with its inputs, model and checks.

    Same arithmetic as :func:`information_depth`, and the same
    reporting terms as :func:`mean_escape_depth_report`. The percentage
    is recorded in the result, because an information depth without one
    means nothing.

    Args:
        imfp: Inelastic mean free path with its declarations.
        trmfp: Transport mean free path with its declarations.
        model: Keyword-only and required; see
            :data:`INFORMATION_DEPTH_MODELS`.
        emission_angle_deg: Emission angle from the surface normal, in
            degrees. Keyword-only and required.
        percentage: Percentage of the detected signal, in (0, 100).
            Keyword-only and required.

    Returns:
        A :class:`SamplingDepthReport` with ``concept ==
        "information_depth"``.

    Raises:
        TypeError: If the lengths are not :class:`DescribedLength`.
        ValueError: As :func:`mean_escape_depth_report`, plus a
            percentage outside (0, 100).
    """
    return _sampling_depth_report(
        imfp, trmfp,
        concept="information_depth",
        m=_resolve_depth(
            INFORMATION_DEPTH_MODELS, model, "information depth"),
        model=model,
        emission_angle_deg=emission_angle_deg,
        percentage=_check_percentage(percentage),
    )


def _resolve_depth(
    models: dict[str, _DepthModel], model: str, quantity: str
) -> _DepthModel:
    try:
        return models[model]
    except KeyError:
        raise ValueError(
            f"unknown {quantity} model {model!r}; available: "
            f"{sorted(models)}. There is no default, and an EAL model "
            "name is not accepted here: the three quantities have "
            "different slopes."
        ) from None


def _check_emission_angle(emission_angle_deg: float) -> float:
    angle = float(emission_angle_deg)
    if not math.isfinite(angle) or not 0.0 <= angle <= 90.0:
        raise ValueError(
            f"emission_angle_deg must be in [0, 90], got {angle!r}")
    return angle


#: Below this, a "percentage" is almost certainly a fraction. The
#: relations were evaluated at 90%, 95% and 99%, and an information
#: depth holding under 1% of the signal is not a quantity anyone wants,
#: so the interval (0, 1) is given up in exchange for catching the one
#: confusion that would otherwise pass silently — ``IMFP.sampling_depth``
#: next door takes a *fraction*, and ``percentage=0.95`` there would
#: return a depth 314 times too small.
_MIN_PERCENTAGE = 1.0


def _check_percentage(percentage: float) -> float:
    p = float(percentage)
    if not math.isfinite(p) or not 0.0 < p < 100.0:
        raise ValueError(
            f"percentage must be in (0, 100), got {p!r}. It is the "
            "percentage of the detected signal that originates above "
            "the information depth; 100% is at infinite depth.")
    if p < _MIN_PERCENTAGE:
        raise ValueError(
            f"percentage={p!r} is a percentage, not a fraction: 95% is "
            "written 95.0, not 0.95. Values below "
            f"{_MIN_PERCENTAGE:g}% are refused because they are far more "
            "often that confusion than a wanted information depth "
            "(IMFP.sampling_depth, the elastic-free counterpart, takes a "
            "fraction in (0, 1) instead).")
    return p


def mean_escape_depth_ratio(
    imfp: float, trmfp: float, *, model: SamplingDepthModel
) -> float:
    """Return D / (IMFP cos alpha), the MED relative to no elastic scattering.

    This ratio is what the published relation fits, and it does not
    depend on the emission angle: the ``cos alpha`` cancels between the
    MED and its straight-line-approximation counterpart. Use
    :func:`mean_escape_depth` for the depth itself.

    Args:
        imfp: Inelastic mean free path.
        trmfp: Transport mean free path, same unit and source as ``imfp``.
        model: Keyword-only and required; see :data:`MED_MODELS`.

    Returns:
        The dimensionless ratio, exactly 1 when omega is 0.

    Raises:
        ValueError: If a length is invalid or the model name is unknown.
    """
    m = _resolve_depth(MED_MODELS, model, "mean escape depth")
    return 1.0 - m.slope * single_scattering_albedo(imfp, trmfp)


def mean_escape_depth(
    imfp: float,
    trmfp: float,
    *,
    emission_angle_deg: float,
    model: SamplingDepthModel,
) -> float:
    """Return the mean escape depth D, in the unit of ``imfp``.

    The MED is the average depth, normal to the surface, that the
    detected electrons originated from. It is *not* an attenuation
    length: it is a property of the depth distribution of the signal,
    and it is roughly ``IMFP cos alpha`` rather than ``IMFP``.

    Unlike the overlayer EAL, this depends on the emission angle
    explicitly, through ``cos alpha`` — the relation is not an average
    over an angle range. The angle is therefore required, and passing
    the wrong one scales the answer directly.

    Computed for a specimen thicker than the information depth.

    **This function does not check the model's validity ranges.** It
    accepts any angle in [0, 90] and any albedo, although the relation
    is stated only for 0-50 degrees, albedos of 0.1-0.5 and energies to
    2016 eV; above 50 degrees the paper reports the ratio can exceed
    unity. Use :func:`mean_escape_depth_report` to have the range
    checked and the extrapolation reported.

    Args:
        imfp: Inelastic mean free path.
        trmfp: Transport mean free path, same unit and source as ``imfp``.
        emission_angle_deg: Emission angle from the surface normal, in
            degrees. Keyword-only and required.
        model: Keyword-only and required; see :data:`MED_MODELS`.

    Returns:
        D, in the unit of ``imfp``.

    Raises:
        ValueError: If a length is invalid, the angle is outside
            [0, 90] degrees, or the model name is unknown.
    """
    angle = _check_emission_angle(emission_angle_deg)
    return (_check_length("imfp", imfp)
            * math.cos(math.radians(angle))
            * mean_escape_depth_ratio(imfp, trmfp, model=model))


def information_depth_ratio(
    imfp: float, trmfp: float, *, model: SamplingDepthModel
) -> float:
    """Return S / S_SLA, the information depth relative to no elastic scattering.

    As with :func:`mean_escape_depth_ratio`, this is the quantity the
    published relation fits, and it depends on neither the emission
    angle nor the percentage: both cancel against the straight-line
    counterpart.

    Args:
        imfp: Inelastic mean free path.
        trmfp: Transport mean free path, same unit and source as ``imfp``.
        model: Keyword-only and required; see
            :data:`INFORMATION_DEPTH_MODELS`.

    Returns:
        The dimensionless ratio, exactly 1 when omega is 0.

    Raises:
        ValueError: If a length is invalid or the model name is unknown.
    """
    m = _resolve_depth(INFORMATION_DEPTH_MODELS, model, "information depth")
    return 1.0 - m.slope * single_scattering_albedo(imfp, trmfp)


def information_depth(
    imfp: float,
    trmfp: float,
    *,
    emission_angle_deg: float,
    percentage: float,
    model: SamplingDepthModel,
) -> float:
    """Return the information depth S, in the unit of ``imfp``.

    S is the depth above which ``percentage`` of the detected signal
    originates, so it grows without bound as the percentage approaches
    100 and is meaningless without one. The published relation was
    evaluated at 90%, 95% and 99%; the fit quality quoted in
    :data:`INFORMATION_DEPTH_MODELS` pools those three.

    Computed for a specimen thicker than the information depth itself.

    **This function does not check the model's validity ranges**; see
    :func:`mean_escape_depth` and use :func:`information_depth_report`
    to have them checked.

    Args:
        imfp: Inelastic mean free path.
        trmfp: Transport mean free path, same unit and source as ``imfp``.
        emission_angle_deg: Emission angle from the surface normal, in
            degrees. Keyword-only and required.
        percentage: Percentage of the detected signal, in (0, 100), and
            a *percentage*: 95% is ``95.0``. Keyword-only and required —
            there is no conventional default, and the answer scales with
            the choice: 99% is 2.0 times 90%. Values below 1% are
            refused as a likely fraction; note that the elastic-free
            counterpart ``IMFP.sampling_depth`` takes a fraction in
            (0, 1) instead.
        model: Keyword-only and required; see
            :data:`INFORMATION_DEPTH_MODELS`.

    Returns:
        S, in the unit of ``imfp``.

    Raises:
        ValueError: If a length is invalid, the angle is outside
            [0, 90] degrees, the percentage is outside (0, 100), or the
            model name is unknown.
    """
    angle = _check_emission_angle(emission_angle_deg)
    p = _check_percentage(percentage)
    return (_check_length("imfp", imfp)
            * math.cos(math.radians(angle))
            * math.log(1.0 / (1.0 - p / 100.0))
            * information_depth_ratio(imfp, trmfp, model=model))


def seah_gilmore_2001_overlayer_eal_ratio(
    imfp: float,
    trmfp: float,
    atomic_number: float,
) -> float:
    """Return L/IMFP from Seah & Gilmore (2001) Eq. (35).

    Provided for comparison with :func:`overlayer_eal_ratio`, not as a
    recommended path. Two reasons. Its leading coefficient leaves
    L/IMFP = 0.979 rather than 1 as omega goes to 0, which is unphysical
    when elastic scattering vanishes. And it carries an explicit Z
    dependence that the later Monte Carlo analyses do not reproduce.

    Derived from the Monte Carlo simulations of Cumpson & Seah — a
    different dataset from the Jablonski tables behind that paper's Q and
    beta_eff equations. Stated validity: 0 < alpha < 58 degrees emission, with the
    angle between the x rays and the detected electrons above 45 degrees.

    Args:
        imfp: Inelastic mean free path.
        trmfp: Transport mean free path, same unit and source as ``imfp``.
        atomic_number: Z of the matrix. For compounds the paper averages
            the *ratio* over atomic fractions (its Eq. 39) rather than
            substituting a mean Z; do that at the call site.

    Returns:
        The dimensionless ratio.

    Raises:
        ValueError: If a length is invalid or ``atomic_number`` is below 1.
    """
    z = float(atomic_number)
    if not math.isfinite(z) or z < 1.0:
        raise ValueError(f"atomic_number must be >= 1, got {z!r}")
    omega = single_scattering_albedo(imfp, trmfp)
    return 0.979 * (1.0 - omega * (0.955 - 0.0777 * math.log(z)))


# Tolerance for a declared kinetic-energy pair. Reading one length at a
# 1% wrong energy perturbs it by ~0.8% (IMFP, ~KE^0.8 here) to ~1.4%
# (TRMFP, ~KE^1.4 for SESSA's Au values between 7.4 and 9.2 keV), hence
# the albedo by at most ~(1 - omega) * 1.4%, which moves L_TH/IMFP by
# ~0.2% at omega ~ 0.2 — far under the 1.44-1.85% rms of the relations
# themselves. The mispairings this check exists to catch are orders
# larger: reading the IMFP at one line's kinetic energy and the TRMFP
# at another's — Au 4f against Si 1s at Ga K-alpha is a 24% energy
# mismatch and a 19% IMFP mismatch — while a legitimate near-degenerate
# pair such as Au 4f / Si 2p differs by 0.2%.
_KE_CONSISTENCY_RTOL = 0.01


@dataclass(frozen=True)
class DescribedLength:
    """A path length together with what the caller can say about it.

    ``value`` and ``unit`` are required; ``source``, ``material`` and
    ``kinetic_energy_ev`` are declarations. :func:`overlayer_eal_report`
    checks whatever is declared on both lengths and reports what is not
    as ``"not_recorded"`` — absence of a declaration is never treated as
    consistency.

    Args:
        value: The length, finite and positive.
        unit: ``"nm"`` or ``"angstrom"``. Required because a unit
            mismatch between IMFP and TRMFP corrupts the albedo silently.
        source: Where the number came from, e.g. ``"TPP-2M"``. Free
            text, compared verbatim (case-insensitively) between the two
            lengths, so it can be made as specific as the pairing needs:
            :meth:`toyomacro.data.sessa.SessaInteractionParameters.row_source`
            puts the SESSA run *and the row* in it, because two rows of
            one run agree on everything else that is checked.
        material: The matrix the electron travels through, e.g. ``"Au"``.
            Compared verbatim (case-insensitively); the same material
            must be spelled the same way on both lengths.
        kinetic_energy_ev: Electron kinetic energy the length was
            evaluated at, in eV.
    """

    value: float
    unit: LengthUnit
    source: str | None = None
    material: str | None = None
    kinetic_energy_ev: float | None = None

    def __post_init__(self) -> None:
        # Coerce to builtin float so to_dict() stays JSON-serializable
        # even when the caller passes e.g. a numpy scalar.
        object.__setattr__(self, "value", _check_length("value", self.value))
        if self.unit not in ("nm", "angstrom"):
            raise ValueError(
                f"unit must be 'nm' or 'angstrom', got {self.unit!r}")
        if self.kinetic_energy_ev is not None:
            ke = float(self.kinetic_energy_ev)
            if not math.isfinite(ke) or ke <= 0.0:
                raise ValueError(
                    f"kinetic_energy_ev must be finite and positive, got {ke!r}")
            object.__setattr__(self, "kinetic_energy_ev", ke)
        # "not_recorded" marks an absent declaration in to_dict(); a
        # literal source or material spelled that way would be
        # indistinguishable from one that was never declared.
        for field_name in ("source", "material"):
            declared = getattr(self, field_name)
            if (declared is not None
                    and declared.strip().casefold() == NOT_RECORDED):
                raise ValueError(
                    f"{field_name}={declared!r} collides with the reserved "
                    f"{NOT_RECORDED!r} marker for an absent declaration; "
                    "omit the field instead")

    def to_dict(self) -> dict[str, object]:
        """Plain-JSON form. Undeclared fields become ``"not_recorded"``.

        The string ``"not_recorded"`` is reserved: it marks an absent
        declaration, and the constructor rejects it (in any casing) as a
        literal source or material name so the two cases can never be
        confused.
        """
        return {
            "value": self.value,
            "unit": self.unit,
            "source": self.source if self.source is not None else NOT_RECORDED,
            "material": (
                self.material if self.material is not None else NOT_RECORDED),
            "kinetic_energy_ev": (
                self.kinetic_energy_ev
                if self.kinetic_energy_ev is not None else NOT_RECORDED),
        }


@dataclass(frozen=True)
class OverlayerEALReport:
    """An overlayer-thickness EAL with its inputs, model and checks.

    Produced by :func:`overlayer_eal_report`. The numbers are identical
    to what the scalar functions return; what this adds is the record —
    which model, which inputs, which declared facts were checked and
    which were never declared. Consumers that keep results (a depth
    profile, an archive) should keep :meth:`to_dict` next to the value.

    ``kinetic_energy_check`` and ``emission_angle_check`` are
    ``"within"`` / ``"outside"`` / ``"not_recorded"`` against the model's
    fitted ranges; ``source_consistency`` is ``"consistent"`` /
    ``"inconsistent"`` / ``"not_recorded"`` between the two lengths.
    ``"outside"`` and ``"inconsistent"`` also append to ``warnings``
    rather than raising: an extrapolation or a cross-source pair can be a
    deliberate, stated choice, unlike a unit or material mismatch.
    """

    value: float
    unit: LengthUnit
    ratio: float
    albedo: float
    model_name: str
    imfp: DescribedLength
    trmfp: DescribedLength
    emission_angle_deg: float | None
    kinetic_energy_check: str
    emission_angle_check: str
    source_consistency: str
    warnings: tuple[str, ...]

    @property
    def model(self) -> _Model:
        """The published relation used, with its validity metadata."""
        return EAL_MODELS[self.model_name]

    def to_dict(self) -> dict[str, object]:
        """Plain-JSON record of the value, model, inputs and checks.

        The ``length_concept`` block states what kind of length the
        value is, in the four descriptors an operationally defined
        length needs to be interpretable downstream: operational
        definition, angular dependence, elastic-scattering treatment,
        geometry convention. Consumers may rely on exactly those key
        names.
        """
        m = self.model
        consistency = _declared_consistency(self.imfp, self.trmfp)
        return {
            "value": self.value,
            "unit": self.unit,
            "ratio": self.ratio,
            "albedo": self.albedo,
            "model": {
                "name": self.model_name,
                "equation": m.equation,
                "slope": m.slope,
                "emission_angle_deg": list(m.emission_angle_deg),
                "energy_ev": list(m.energy_ev),
                "rms_percent": m.rms_percent,
                "geometry": m.geometry,
            },
            "inputs": {
                "imfp": self.imfp.to_dict(),
                "trmfp": self.trmfp.to_dict(),
                "emission_angle_deg": (
                    self.emission_angle_deg
                    if self.emission_angle_deg is not None else NOT_RECORDED),
            },
            "validity": {
                "kinetic_energy": self.kinetic_energy_check,
                "emission_angle": self.emission_angle_check,
            },
            "consistency": {**consistency, "source": self.source_consistency},
            "length_concept": {
                "concept": "eal",
                "operational_definition": (
                    "overlayer-thickness effective attenuation length "
                    "L_TH: the constant that replaces the IMFP in "
                    "t = -L cos(alpha) ln(1 - R) when a film thickness "
                    "is determined from overlayer or substrate "
                    "intensities"),
                "angular_dependence": (
                    "none within the model's stated emission-angle "
                    "range: L_TH is that range's average, not an "
                    "angle-resolved quantity; grazing emission is "
                    "outside the range"),
                "elastic_scattering_treatment": (
                    "transport approximation through the "
                    "single-scattering albedo "
                    "omega = IMFP / (IMFP + TRMFP); "
                    "L_TH/IMFP = 1 - slope * omega with a published, "
                    "geometry-specific slope"),
                "geometry_convention": m.geometry,
            },
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SamplingDepthReport:
    """A mean escape depth or information depth with its inputs and checks.

    Produced by :func:`mean_escape_depth_report` and
    :func:`information_depth_report`. The counterpart of
    :class:`OverlayerEALReport` for the two quantities that are *not*
    attenuation lengths, and the ``concept`` field is what tells them
    apart downstream: three different depths for one material at one
    energy differ by more than their uncertainties, so a stored number
    without its concept is not interpretable.

    ``ratio`` is relative to the same depth with elastic scattering
    switched off — ``IMFP cos(alpha)`` for the MED, and that times
    ``ln[1/(1 - P/100)]`` for the information depth — not relative to
    the IMFP.

    ``percentage`` is the P of the information depth, and ``None`` for
    a mean escape depth — which :meth:`to_dict` records as
    ``"not_applicable"``, not ``"not_recorded"``: a mean escape depth
    has no percentage, rather than an undeclared one.
    """

    value: float
    unit: LengthUnit
    ratio: float
    albedo: float
    concept: str
    model_name: str
    imfp: DescribedLength
    trmfp: DescribedLength
    emission_angle_deg: float
    percentage: float | None
    kinetic_energy_check: str
    emission_angle_check: str
    albedo_check: str
    source_consistency: str
    warnings: tuple[str, ...]

    @property
    def model(self) -> _DepthModel:
        """The published relation used, with its validity metadata."""
        models = (MED_MODELS if self.concept == "mean_escape_depth"
                  else INFORMATION_DEPTH_MODELS)
        return models[self.model_name]

    def to_dict(self) -> dict[str, object]:
        """Plain-JSON record of the value, model, inputs and checks.

        Same shape as :meth:`OverlayerEALReport.to_dict`, with an added
        ``albedo`` entry under ``validity`` — these relations state an
        albedo range of validity and the EAL ones here do not — and an
        added ``percentage`` under ``inputs``.
        """
        m = self.model
        consistency = _declared_consistency(self.imfp, self.trmfp)
        if self.concept == "mean_escape_depth":
            operational_definition = (
                "mean escape depth D: the average depth normal to the "
                "surface from which the detected electrons originated. "
                "Not an attenuation length, and computed for a specimen "
                "thicker than the information depth")
            without_elastic = "IMFP cos(alpha)"
        else:
            operational_definition = (
                f"information depth S: the depth above which "
                f"{self.percentage:g}% of the detected signal "
                "originates. Computed for a specimen thicker than S "
                "itself")
            without_elastic = "IMFP cos(alpha) ln[1/(1 - P/100)]"
        return {
            "value": self.value,
            "unit": self.unit,
            "ratio": self.ratio,
            "albedo": self.albedo,
            "model": {
                "name": self.model_name,
                "equation": m.equation,
                "slope": m.slope,
                "emission_angle_deg": list(m.emission_angle_deg),
                "energy_ev": list(m.energy_ev),
                "albedo": list(m.albedo),
                "mean_deviation_percent": m.mean_deviation_percent,
                "geometry": m.geometry,
            },
            "inputs": {
                "imfp": self.imfp.to_dict(),
                "trmfp": self.trmfp.to_dict(),
                "emission_angle_deg": self.emission_angle_deg,
                # not_applicable, not not_recorded: a mean escape depth
                # has no percentage to declare in the first place.
                "percentage": (
                    self.percentage if self.percentage is not None
                    else NOT_APPLICABLE),
            },
            "validity": {
                "kinetic_energy": self.kinetic_energy_check,
                "emission_angle": self.emission_angle_check,
                "albedo": self.albedo_check,
            },
            "consistency": {**consistency, "source": self.source_consistency},
            "length_concept": {
                "concept": self.concept,
                "operational_definition": operational_definition,
                "angular_dependence": (
                    "explicit: the depth carries a cos(alpha) factor at "
                    "the stated emission angle. The albedo correction "
                    "itself is an average over the model's "
                    "emission-angle range; grazing emission is outside "
                    "it"),
                "elastic_scattering_treatment": (
                    "transport approximation through the "
                    "single-scattering albedo "
                    "omega = IMFP / (IMFP + TRMFP); "
                    f"depth / [{without_elastic}] = 1 - slope * omega "
                    "with a published, quantity-specific slope"),
                "geometry_convention": m.geometry,
            },
            "warnings": list(self.warnings),
        }


def _declared_pair_or_raise(
    name: str, a: str | None, b: str | None
) -> bool:
    """True if both declared and equal after normalisation; raise if both
    declared and different; False if either is undeclared."""
    if a is None or b is None:
        return False
    if a.strip().casefold() != b.strip().casefold():
        raise ValueError(
            f"imfp and trmfp declare different {name}s: {a!r} vs {b!r}. "
            "An albedo formed from lengths for two different "
            f"{name}s is not a physical quantity. If these name the "
            f"same {name}, spell them identically.")
    return True


def _range_check(value: float | None, lo: float, hi: float) -> str:
    if value is None:
        return NOT_RECORDED
    return "within" if lo <= value <= hi else "outside"


@dataclass(frozen=True)
class _PairChecks:
    """What the two declared lengths jointly say, and what that fails."""

    albedo: float
    kinetic_energy_ev: float | None
    kinetic_energy_check: str
    emission_angle_check: str
    albedo_check: str
    source_consistency: str
    warnings: tuple[str, ...]


def _check_pair(
    imfp: DescribedLength,
    trmfp: DescribedLength,
    *,
    model_name: str,
    energy_ev: tuple[float, float],
    emission_angle_range: tuple[float, float],
    emission_angle_deg: float | None,
    albedo_range: tuple[float, float] | None,
) -> _PairChecks:
    """Validate a described pair against a model's stated envelope.

    Shared by every report builder here, so that the three quantities
    police their inputs identically. Mismatches that have no legitimate
    reading raise; being outside a stated range is recorded and warned
    about, and an undeclared fact is ``not_recorded``. The energy and
    angle ranges are fitted ranges; the albedo range is a recommendation
    narrower than the fitted set, which is why only its warning says so.
    """
    if not isinstance(imfp, DescribedLength) or not isinstance(
            trmfp, DescribedLength):
        raise TypeError(
            "this function takes DescribedLength inputs; for bare "
            "numbers use the scalar functions in this module")

    if imfp.unit != trmfp.unit:
        raise ValueError(
            f"imfp is in {imfp.unit!r} but trmfp is in {trmfp.unit!r}; "
            "the albedo needs both lengths in one unit")
    _declared_pair_or_raise("material", imfp.material, trmfp.material)

    ke_i, ke_t = imfp.kinetic_energy_ev, trmfp.kinetic_energy_ev
    if ke_i is not None and ke_t is not None:
        if abs(ke_i - ke_t) > _KE_CONSISTENCY_RTOL * max(ke_i, ke_t):
            raise ValueError(
                f"imfp is declared at {ke_i} eV but trmfp at {ke_t} eV "
                f"(more than {_KE_CONSISTENCY_RTOL:.0%} apart). Both "
                "lengths must describe the same electron; this mismatch "
                "is the size of reading them at two different "
                "photoemission lines.")

    albedo = single_scattering_albedo(imfp.value, trmfp.value)
    ke = ke_i if ke_i is not None else ke_t
    ke_check = _range_check(ke, *energy_ev)
    angle_check = _range_check(emission_angle_deg, *emission_angle_range)
    albedo_check = (
        NOT_RECORDED if albedo_range is None
        else _range_check(albedo, *albedo_range))

    warnings: list[str] = []
    if ke_check == "outside":
        warnings.append(
            f"kinetic energy {ke} eV is outside the "
            f"{energy_ev[0]:g}-{energy_ev[1]:g} eV range the "
            f"{model_name} relation was fitted over; the result is an "
            "extrapolation")
    if angle_check == "outside":
        warnings.append(
            f"emission angle {emission_angle_deg:g} deg is outside the "
            f"{emission_angle_range[0]:g}-{emission_angle_range[1]:g} "
            f"deg range the {model_name} relation averages over")
    if albedo_check == "outside" and albedo_range is not None:
        # "recommended", not "fitted": the 2009 relations are recommended
        # for omega in 0.1-0.5 while their own lines span 0.103-0.508, so
        # a point just outside this range may still be inside the fitted
        # set. Calling it "fitted" here would contradict that.
        warnings.append(
            f"albedo {albedo:.4g} is outside the "
            f"{albedo_range[0]:g}-{albedo_range[1]:g} range the "
            f"{model_name} relation is recommended for")

    if imfp.source is not None and trmfp.source is not None:
        if imfp.source.strip().casefold() == trmfp.source.strip().casefold():
            source_consistency = "consistent"
        else:
            source_consistency = "inconsistent"
            warnings.append(
                f"imfp source {imfp.source!r} and trmfp source "
                f"{trmfp.source!r} differ; an albedo mixes cleanly only "
                "when both lengths come from one dataset")
    else:
        source_consistency = NOT_RECORDED

    return _PairChecks(
        albedo=albedo,
        kinetic_energy_ev=ke,
        kinetic_energy_check=ke_check,
        emission_angle_check=angle_check,
        albedo_check=albedo_check,
        source_consistency=source_consistency,
        warnings=tuple(warnings),
    )


def _declared_consistency(
    imfp: DescribedLength, trmfp: DescribedLength
) -> dict[str, str]:
    """Re-derive the material and energy agreement of a described pair.

    The report builders raise on either mismatch, so a report they
    produced can only carry ``"consistent"`` or ``"not_recorded"``
    here. It is still re-derived rather than assumed, so that a
    hand-assembled report cannot claim a consistency it does not have.
    """
    if imfp.material is None or trmfp.material is None:
        material = NOT_RECORDED
    elif (imfp.material.strip().casefold()
            == trmfp.material.strip().casefold()):
        material = "consistent"
    else:
        material = "inconsistent"

    ke_i, ke_t = imfp.kinetic_energy_ev, trmfp.kinetic_energy_ev
    if ke_i is None or ke_t is None:
        kinetic_energy = NOT_RECORDED
    elif abs(ke_i - ke_t) <= _KE_CONSISTENCY_RTOL * max(ke_i, ke_t):
        kinetic_energy = "consistent"
    else:
        kinetic_energy = "inconsistent"

    return {"material": material, "kinetic_energy": kinetic_energy}


def overlayer_eal_report(
    imfp: DescribedLength,
    trmfp: DescribedLength,
    *,
    model: EALModel,
    emission_angle_deg: float | None = None,
) -> OverlayerEALReport:
    """Compute L_TH and return it with its inputs, model and checks.

    Same arithmetic as :func:`overlayer_eal`; the difference is that the
    inputs are :class:`DescribedLength` and the declared facts are used.
    Three kinds of declared mismatch raise, because no reading of them is
    legitimate:

    - different ``unit`` on the two lengths (the albedo would be wrong
      by a silent factor);
    - different declared ``material`` (an albedo across two materials is
      not a physical quantity);
    - declared kinetic energies more than 1% apart (the pairing error
      this catches — reading each length at a different photoemission
      line — shifts the IMFP by tens of percent, while a genuine pair
      agrees to well under 1%; see ``_KE_CONSISTENCY_RTOL``).

    Everything else is recorded, not policed. A declared kinetic energy
    or emission angle outside the model's fitted range yields
    ``"outside"`` plus a warning — extrapolation can be a stated choice.
    Differing declared sources yield ``source_consistency ==
    "inconsistent"`` plus a warning — mixing sources is discouraged (see
    the module docstring) but is sometimes deliberate. An undeclared
    fact yields ``"not_recorded"``: the check did not run, which is not
    the same as passing it.

    Args:
        imfp: Inelastic mean free path with its declarations.
        trmfp: Transport mean free path with its declarations. Same
            requirements as the scalar API: same material, same kinetic
            energy, ideally the same source as ``imfp``.
        model: Keyword-only and required, as in
            :func:`overlayer_eal_ratio`.
        emission_angle_deg: The emission angle (from the surface normal)
            the result will be used at, in degrees, if known. Checked
            against the model's fitted angle range; it does not enter
            the arithmetic, because the models are range averages.

    Returns:
        An :class:`OverlayerEALReport`; its ``value`` is in the unit the
        lengths were declared in.

    Raises:
        TypeError: If the lengths are not :class:`DescribedLength`.
        ValueError: On a declared unit, material or kinetic-energy
            mismatch, an invalid length, an unknown model, or an
            emission angle outside [0, 90] degrees.
    """
    if not isinstance(imfp, DescribedLength) or not isinstance(
            trmfp, DescribedLength):
        raise TypeError(
            "overlayer_eal_report takes DescribedLength inputs; for bare "
            "numbers use overlayer_eal / overlayer_eal_ratio")
    m = _resolve(model)
    if emission_angle_deg is not None:
        emission_angle_deg = _check_emission_angle(emission_angle_deg)

    checks = _check_pair(
        imfp, trmfp,
        model_name=model,
        energy_ev=m.energy_ev,
        emission_angle_range=m.emission_angle_deg,
        emission_angle_deg=emission_angle_deg,
        albedo_range=None,
    )
    ratio = 1.0 - m.slope * checks.albedo

    return OverlayerEALReport(
        value=imfp.value * ratio,
        unit=imfp.unit,
        ratio=ratio,
        albedo=checks.albedo,
        model_name=model,
        imfp=imfp,
        trmfp=trmfp,
        emission_angle_deg=emission_angle_deg,
        kinetic_energy_check=checks.kinetic_energy_check,
        emission_angle_check=checks.emission_angle_check,
        source_consistency=checks.source_consistency,
        warnings=checks.warnings,
    )
