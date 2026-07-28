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
  consistent use the natural one — but it does not enforce it. Nothing
  here can tell that ``overlayer_eal(64.4, 300.0, ...)`` mixed a SESSA
  IMFP with a TRMFP from elsewhere; **that remains the caller's
  responsibility.** Enforcement would need the lengths to carry their
  source, material and energy, which is a larger change than this module.
- **No table lookup.** TRMFP values are the caller's responsibility.
  Jablonski & Powell (2020) tabulate albedo and TRMFP for 41 elemental
  solids and 42 inorganic compounds from 50 eV to 30 keV in their
  supplementary material; bundling that is a separate decision, not least
  because its redistribution terms need checking.

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
    "single_scattering_albedo",
    "overlayer_eal_ratio",
    "overlayer_eal",
    "seah_gilmore_2001_overlayer_eal_ratio",
]

EALModel = Literal["jp2020_unpolarized", "jp2020_polarized_haxpes"]


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
