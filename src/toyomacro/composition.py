"""Homogeneous-equivalent composition from XPS peak areas, with its terms.

Experimental (docs/API.md, "API stability"): not re-exported from any
``__init__``; the API may change between versions.

What the number is. The atomic fraction of each element the sample
would have **if it were homogeneous** over the analysed depth, from
background-subtracted peak areas divided by an *intrinsic* sensitivity
σ × λ (photoionization cross-section times TPP-2M inelastic mean free
path) and, where stated or assumed, an analyzer transmission T. It is
not a measured composition of a layered or segregated sample, and σ × λ
is not a complete AMRSF: elastic scattering, the photoelectron angular
distribution, polarization and geometry are not in it. Instrument RSF
tables are a different route and are not mixed into this one.

What is returned. A :class:`CompositionResult` keeps four things apart
and never combines them into one "±":

- the estimate (``fractions``),
- the statistical uncertainty — not evaluated in this version,
- condition dependence — how the estimate moves when a choice is changed
  (:func:`condition_dependence`), a difference between conditions and not
  an uncertainty,
- what was not evaluated (``not_evaluated``), which does not mean "zero".

Its ``status`` is ``"refused"`` (no number: a required fact is missing
and no assumption was given, or the numbers do not determine a result),
``"conditional"`` (a number that rests on the listed assumptions or on
an extrapolated input) or ``"supported"`` (every input is a stated fact
within its range). Nothing is filled in by default: an unknown exposure,
transmission state or intensity meaning refuses unless the caller names
an assumption, and an element missing from the declared set is never
dropped from the denominator silently.

Area. Each line's area is the trapezoidal integral over its window of
``I/(t·T) − B`` in eV, where ``t`` is the line's exposure (only for
integrated counts), ``T`` the transmission curve (only when it is to be
divided out) and ``B`` the background, estimated on that same corrected
scale. It is the area inside the window, not the integral of a fitted
model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from toyomacro.background import Linear, Shirley
from toyomacro.data.cross_section import CrossSection, CrossSectionLookup
from toyomacro.data.imfp import IMFP, TPP2M_FITTED_RANGE_EV

Background = Literal["shirley", "linear"]
Status = Literal["supported", "conditional", "refused"]
IntensityAssumption = Literal["integrated_counts", "count_rate"]
TransmissionAssumption = Literal["divide_by_curve", "equal_across_lines"]

_BACKGROUNDS = {"shirley": Shirley, "linear": Linear}


@dataclass(frozen=True)
class Matrix:
    """The matrix whose IMFP is used for every line (homogeneous sample).

    Give ``compound`` (a :class:`toyomacro.data.CompoundDB` name) or all
    four TPP-2M parameters, and say where they come from.
    """

    name: str
    source: str
    compound: str | None = None
    Nv: float | None = None
    density: float | None = None
    Mw: float | None = None
    Eg: float | None = None

    def imfp_nm(self, kinetic_energy: float) -> float:
        if self.compound is not None:
            return IMFP.tpp2m(kinetic_energy=kinetic_energy, compound=self.compound)
        return IMFP.tpp2m(kinetic_energy=kinetic_energy, Nv=self.Nv, density=self.density,
                          Mw=self.Mw, Eg=self.Eg)


@dataclass(frozen=True, eq=False)
class Line:
    """One element's peak region.

    Attributes:
        element, orbital: The line, as for :meth:`CrossSection.lookup` (a
            bare label such as ``"2p"`` is the whole doublet).
        energy: Energy axis in eV, kinetic or binding (``energy_scale``).
        intensity: Stored intensity on that axis.
        window: (low, high) energy limits of the peak region, same scale.
        binding_energy: The line's binding energy in eV, for its kinetic
            energy (the IMFP argument).
        exposure_s: Seconds of acquisition behind each point, for
            integrated counts; None when not known.
        exposure_basis: How ``exposure_s`` was obtained (e.g. "file:
            collection time 0.2 s x 4 scans, assumed per scan").
        transmission: Transmission curve aligned with ``energy``, or None.
        source: Where the data came from (file, block), for the record.
    """

    element: str
    orbital: str
    energy: np.ndarray
    intensity: np.ndarray
    window: tuple[float, float]
    binding_energy: float
    energy_scale: Literal["kinetic", "binding"] = "kinetic"
    exposure_s: float | None = None
    exposure_basis: str = ""
    transmission: np.ndarray | None = None
    source: str = ""


@dataclass(frozen=True)
class Conditions:
    """The analysis choices; none has a default that stands in for a fact.

    Attributes:
        photon_energy: eV.
        table: Cross-section table name.
        background: Background under every line.
        matrix: The IMFP matrix.
        intensity_semantics: What the stored intensity is, from the file's
            provenance (``SpectrumMetadata.intensity_semantics``).
        transmission_applied: Whether the intensity is already divided by
            the transmission (``SpectrumMetadata.transmission_applied``).
        assume_intensity: Read an intensity whose meaning is not stated as
            this. Makes the result conditional.
        assume_transmission: When ``transmission_applied`` is unknown:
            ``"divide_by_curve"`` (the curves are not applied and share a
            normalisation) or ``"equal_across_lines"`` (T is the same for
            every line, so it cancels). Makes the result conditional.
        declared_elements: Every element the sample is taken to contain;
            the denominator. Defaults to the lines' elements.
        normalise_over: A subset to normalise over instead. Elements left
            out are recorded with ``exclusion_reason``; conditional.
        exclusion_reason: Required with ``normalise_over``.
    """

    photon_energy: float
    table: str
    background: Background
    matrix: Matrix
    intensity_semantics: str
    transmission_applied: str
    assume_intensity: IntensityAssumption | None = None
    assume_transmission: TransmissionAssumption | None = None
    declared_elements: tuple[str, ...] | None = None
    normalise_over: tuple[str, ...] | None = None
    exclusion_reason: str = ""


@dataclass(frozen=True)
class LineReport:
    """What went into one line's share."""

    element: str
    orbital: str
    source: str
    area: float | None
    area_definition: str
    cross_section: CrossSectionLookup
    kinetic_energy_eV: float
    imfp_nm: float | None
    imfp_extrapolated: bool
    transmission_handling: str
    exposure_s: float | None
    exposure_basis: str
    problems: tuple[str, ...]


@dataclass(frozen=True)
class CompositionResult:
    """Estimate, its conditions and its gaps — never one combined "±"."""

    status: Status
    quantity: str
    fractions: dict[str, float] | None
    lines: tuple[LineReport, ...]
    denominator: tuple[str, ...]
    declared_elements: tuple[str, ...]
    excluded: dict[str, str]
    assumptions: tuple[str, ...]
    reasons: tuple[str, ...]
    not_evaluated: tuple[str, ...]
    statistical_uncertainty: None = None
    conditions: Conditions | None = field(default=None, repr=False)


QUANTITY = (
    "homogeneous-equivalent atomic fraction: area / (cross-section x TPP-2M IMFP), "
    "normalised over the denominator; not a complete AMRSF"
)


def _line_area(line: Line, background: str, divide_t: bool, exposure: float | None):
    """(area, definition, problem or None) for one line."""
    e = np.asarray(line.energy, dtype=np.float64)
    y = np.asarray(line.intensity, dtype=np.float64)
    lo, hi = sorted(line.window)
    inside = (e >= lo) & (e <= hi)
    if inside.sum() < 3:
        return None, "", "fewer than 3 points inside the window"
    e, y = e[inside], y[inside]
    terms = ["I"]
    if exposure is not None:
        y = y / exposure
        terms.append("/t")
    if divide_t:
        t = np.asarray(line.transmission, dtype=np.float64)[inside]
        if np.any(~np.isfinite(t)) or np.any(t <= 0):
            return None, "", "transmission curve is not positive and finite in the window"
        y = y / t
        terms.append("/T")
    bg = _BACKGROUNDS[background]().calculate(e, y)
    net = y - bg
    area = float(abs(np.trapezoid(net, e)))
    definition = (f"trapezoidal integral over [{lo:g}, {hi:g}] eV of "
                  f"{''.join(terms)} - {background} background, in eV x intensity unit")
    if not np.isfinite(area) or area <= 0:
        return None, definition, "area is not positive"
    return area, definition, None


def composition(lines: Sequence[Line], conditions: Conditions) -> CompositionResult:
    """Homogeneous-equivalent composition, refusing what it cannot support.

    See the module docstring for what the number is and is not.
    """
    c = conditions
    reasons: list[str] = []
    assumptions: list[str] = []
    not_evaluated = [
        "statistical uncertainty: not evaluated in this version",
        "elastic scattering, angular distribution, polarization and geometry: "
        "not in the sensitivity",
    ]

    elements = [ln.element for ln in lines]
    if len(set(elements)) != len(elements):
        reasons.append("more than one line per element is not supported")
    declared = tuple(c.declared_elements) if c.declared_elements else tuple(elements)
    missing = [el for el in declared if el not in elements]
    if missing:
        reasons.append(f"declared elements without a line: {missing}")
    excluded: dict[str, str] = {}
    denominator = declared
    if c.normalise_over is not None:
        if not c.exclusion_reason:
            reasons.append("normalise_over needs an exclusion_reason")
        bad = [el for el in c.normalise_over if el not in declared]
        if bad:
            reasons.append(f"normalise_over includes undeclared elements: {bad}")
        denominator = tuple(c.normalise_over)
        excluded = {el: c.exclusion_reason for el in declared if el not in denominator}
        if excluded:
            assumptions.append(
                f"normalised over {list(denominator)} only, out of {list(declared)}")

    # What the stored intensity is, and so whether exposure divides it.
    semantics = c.intensity_semantics
    if semantics not in ("raw_counts", "count_rate"):
        if c.assume_intensity is None:
            reasons.append(f"intensity meaning is {semantics!r}; name assume_intensity "
                           "('integrated_counts' or 'count_rate') to proceed")
        else:
            assumptions.append(f"intensity ({semantics}) read as {c.assume_intensity}")
            semantics = "raw_counts" if c.assume_intensity == "integrated_counts" else "count_rate"
    use_exposure = semantics == "raw_counts"

    # Transmission: never applied twice, never assumed silently.
    divide_t = False
    if c.transmission_applied == "applied":
        handling = "already applied to the intensity (stated)"
    elif c.transmission_applied == "not_applied":
        divide_t = True
        handling = "divided by the curve (stated not applied)"
    elif c.assume_transmission == "divide_by_curve":
        divide_t = True
        handling = "divided by the curve (assumed not applied, common normalisation)"
        assumptions.append("transmission curves not yet applied and on a common normalisation")
    elif c.assume_transmission == "equal_across_lines":
        handling = "not divided (assumed equal across lines)"
        assumptions.append("transmission equal for every line, so it cancels")
        not_evaluated.append("transmission: assumed equal across lines")
    else:
        handling = "unknown"
        reasons.append("transmission state is unknown; name assume_transmission to proceed")

    if c.background not in _BACKGROUNDS:
        reasons.append(f"background {c.background!r} is not one of {sorted(_BACKGROUNDS)}")

    reports: list[LineReport] = []
    extrapolated = False
    for ln in lines:
        problems: list[str] = []
        exposure = ln.exposure_s if use_exposure else None
        if use_exposure and (ln.exposure_s is None or not ln.exposure_s > 0):
            problems.append("exposure unknown for integrated counts")
        if divide_t and ln.transmission is None:
            problems.append("no transmission curve to divide by")
        area, definition, area_problem = (None, "", None)
        if not problems and c.background in _BACKGROUNDS:
            area, definition, area_problem = _line_area(ln, c.background, divide_t, exposure)
            if area_problem:
                problems.append(area_problem)
        sigma = CrossSection.lookup_with_status(ln.element, ln.orbital, c.photon_energy, c.table)
        if sigma.value is None or sigma.value <= 0:
            problems.append(f"no cross-section ({sigma.status})")
        elif sigma.status != "tabulated":
            extrapolated = True
        ke = c.photon_energy - ln.binding_energy
        lam: float | None = None
        try:
            lam = c.matrix.imfp_nm(ke)
        except Exception as err:  # noqa: BLE001 - reported, not raised
            problems.append(f"IMFP unavailable: {type(err).__name__}")
        lam_extrap = not (TPP2M_FITTED_RANGE_EV[0] <= ke <= TPP2M_FITTED_RANGE_EV[1])
        if lam_extrap:
            extrapolated = True
        if problems and ln.element in denominator:
            reasons += [f"{ln.element} {ln.orbital}: {p}" for p in problems]
        reports.append(LineReport(
            element=ln.element, orbital=ln.orbital, source=ln.source, area=area,
            area_definition=definition, cross_section=sigma, kinetic_energy_eV=ke,
            imfp_nm=lam, imfp_extrapolated=lam_extrap, transmission_handling=handling,
            exposure_s=exposure, exposure_basis=ln.exposure_basis if use_exposure else "",
            problems=tuple(problems)))

    if extrapolated:
        assumptions.append("an input lies outside its tabulated or fitted range")
    base = dict(quantity=QUANTITY, lines=tuple(reports), denominator=denominator,
                declared_elements=declared, excluded=excluded,
                assumptions=tuple(assumptions), not_evaluated=tuple(not_evaluated),
                conditions=c)
    if reasons:
        return CompositionResult(status="refused", fractions=None,
                                 reasons=tuple(reasons), **base)
    weights = {r.element: r.area / (r.cross_section.value * r.imfp_nm)
               for r in reports if r.element in denominator}
    total = math.fsum(weights.values())
    fractions = {el: weights[el] / total for el in denominator}
    status: Status = "conditional" if assumptions else "supported"
    return CompositionResult(status=status, fractions=fractions, reasons=(), **base)


@dataclass(frozen=True)
class ConditionChange:
    """The estimate under one alternative condition, against the baseline."""

    table: str
    background: str
    status: Status
    fractions: dict[str, float] | None
    change: dict[str, float] | None  # this minus the baseline, per element


@dataclass(frozen=True)
class ConditionDependence:
    """How the estimate moves across a small grid of conditions.

    These are differences between conditions, not an uncertainty: two
    tables or two backgrounds are not a sample of an error distribution.
    ``interaction`` is the two-factor term for a 2 x 2 grid, per element:
    (b, y) - (b, x) - (a, y) + (a, x).
    """

    baseline: CompositionResult
    changes: tuple[ConditionChange, ...]
    interaction: dict[str, float] | None


def condition_dependence(
    lines: Sequence[Line],
    conditions: Conditions,
    tables: Sequence[str],
    backgrounds: Sequence[Background],
) -> ConditionDependence:
    """Recompute the composition for every table x background pair.

    The baseline is ``conditions`` itself; each pair changes only the
    table and the background. A pair that refuses is reported as such.
    """
    baseline = composition(lines, conditions)
    changes = []
    grid: dict[tuple[str, str], dict[str, float] | None] = {}
    for table in tables:
        for bg in backgrounds:
            r = composition(lines, replace(conditions, table=table, background=bg))
            grid[(table, bg)] = r.fractions
            change = None
            if r.fractions is not None and baseline.fractions is not None:
                change = {el: r.fractions[el] - baseline.fractions[el] for el in r.fractions}
            changes.append(ConditionChange(table=table, background=bg, status=r.status,
                                           fractions=r.fractions, change=change))
    interaction = None
    if len(tables) == 2 and len(backgrounds) == 2:
        (a, b), (x, y) = tables, backgrounds
        cells = [grid[(a, x)], grid[(a, y)], grid[(b, x)], grid[(b, y)]]
        if all(cell is not None for cell in cells):
            ax, ay, bx, by = cells
            interaction = {el: by[el] - bx[el] - ay[el] + ax[el] for el in ax}
    return ConditionDependence(baseline=baseline, changes=tuple(changes),
                               interaction=interaction)
