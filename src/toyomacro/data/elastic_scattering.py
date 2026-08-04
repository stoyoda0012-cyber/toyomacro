"""Effective attenuation length from the single-scattering albedo.

Elastic scattering shortens the distance over which a photoelectron signal
decays, so the inelastic mean free path (IMFP) overestimates it. The
correction is governed by one dimensionless number, the single-scattering
albedo

    omega = IMFP / (IMFP + TRMFP)

where TRMFP is the transport mean free path. Over a useful range of
emission angles the ratio of the effective attenuation length to the IMFP
falls linearly in omega,

    L_TH / IMFP = 1 - A omega

and the slope A depends on the measurement geometry — in particular on
whether the exciting radiation is polarized. For gold at 7.4 keV
(omega = 0.208) the unpolarized and polarized slopes differ by 2.4% in
the resulting EAL, which is why the model is not optional here.

**No default model, and no default albedo.** Both encode assumptions about
the experiment that only the caller knows. The predecessor of this module
applied a hardcoded EAL/IMFP ratio of 0.9 and described it as an
elastic-scattering correction; requiring omega means a caller who cannot
supply it cannot silently get a wrong answer instead.

Scope and limits
----------------

- **Overlayer-thickness EALs only** (``L_TH``): the L that replaces the
  IMFP in ``t = -L cos(alpha) ln(1 - R)`` when a film thickness is
  determined from overlayer or substrate intensities. Mean escape depth,
  information depth, marker depth and core-shell nanoparticle EALs follow
  different slopes and are not implemented here.
- **No angle dependence.** Each relation is an average over its stated
  emission-angle range, within which L is treated as constant. Neither is
  valid at grazing emission.
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

References
----------

Jablonski & Powell, *J. Phys. Chem. Ref. Data* **49**, 033102 (2020),
DOI 10.1063/5.0008576 — Eq. (20) for unpolarized x rays and Eq. (62) for
linearly polarized HAXPES.

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
    "DescribedLength",
    "OverlayerEALReport",
    "single_scattering_albedo",
    "overlayer_eal_ratio",
    "overlayer_eal",
    "overlayer_eal_report",
    "seah_gilmore_2001_overlayer_eal_ratio",
]

EALModel = Literal["jp2020_unpolarized", "jp2020_polarized_haxpes"]

LengthUnit = Literal["nm", "angstrom"]

#: Availability vocabulary: ``not_recorded`` means the fact exists — an
#: IMFP was computed for *some* material at *some* energy — but the
#: caller did not declare it here. It is not a pass, it is not
#: ``unknown``, and it is not ``not_applicable``; a check that could not
#: run reports this string rather than silently assuming consistency.
NOT_RECORDED = "not_recorded"


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
        source: Where the number came from, e.g. ``"TPP-2M"`` or
            ``"SESSA 2.2.2 (JTP)"``. Free text, compared verbatim
            (case-insensitively) between the two lengths.
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
        # overlayer_eal_report raises on a declared material or
        # kinetic-energy mismatch, so reports it builds only ever carry
        # "consistent" or "not_recorded" here; the comparison is still
        # re-derived rather than assumed, so a hand-assembled report
        # cannot claim a consistency it does not have.
        if self.imfp.material is None or self.trmfp.material is None:
            material_consistency = NOT_RECORDED
        elif (self.imfp.material.strip().casefold()
                == self.trmfp.material.strip().casefold()):
            material_consistency = "consistent"
        else:
            material_consistency = "inconsistent"
        ke_i, ke_t = self.imfp.kinetic_energy_ev, self.trmfp.kinetic_energy_ev
        if ke_i is None or ke_t is None:
            ke_consistency = NOT_RECORDED
        elif abs(ke_i - ke_t) <= _KE_CONSISTENCY_RTOL * max(ke_i, ke_t):
            ke_consistency = "consistent"
        else:
            ke_consistency = "inconsistent"
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
            "consistency": {
                "material": material_consistency,
                "kinetic_energy": ke_consistency,
                "source": self.source_consistency,
            },
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

    if emission_angle_deg is not None:
        a = float(emission_angle_deg)
        if not math.isfinite(a) or not 0.0 <= a <= 90.0:
            raise ValueError(
                f"emission_angle_deg must be in [0, 90], got {a!r}")
        emission_angle_deg = a

    albedo = single_scattering_albedo(imfp.value, trmfp.value)
    ratio = 1.0 - m.slope * albedo

    ke = ke_i if ke_i is not None else ke_t
    ke_check = _range_check(ke, *m.energy_ev)
    angle_check = _range_check(emission_angle_deg, *m.emission_angle_deg)

    warnings: list[str] = []
    if ke_check == "outside":
        warnings.append(
            f"kinetic energy {ke} eV is outside the "
            f"{m.energy_ev[0]:g}-{m.energy_ev[1]:g} eV range the "
            f"{model} relation was fitted over; the result is an "
            "extrapolation")
    if angle_check == "outside":
        warnings.append(
            f"emission angle {emission_angle_deg:g} deg is outside the "
            f"{m.emission_angle_deg[0]:g}-{m.emission_angle_deg[1]:g} "
            f"deg range the {model} relation averages over")

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

    return OverlayerEALReport(
        value=imfp.value * ratio,
        unit=imfp.unit,
        ratio=ratio,
        albedo=albedo,
        model_name=model,
        imfp=imfp,
        trmfp=trmfp,
        emission_angle_deg=emission_angle_deg,
        kinetic_energy_check=ke_check,
        emission_angle_check=angle_check,
        source_consistency=source_consistency,
        warnings=tuple(warnings),
    )
