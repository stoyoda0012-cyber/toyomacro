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
- the statistical uncertainty — withheld in this version: the scope in
  which it was validated on synthetic data is empty
  (docs/design/composition-uncertainty.md §7),
- condition dependence — how the estimate moves when a choice is changed
  (:func:`condition_dependence`), a difference between conditions and not
  an uncertainty,
- what was not evaluated (``not_evaluated``), which does not mean "zero".

Its ``status`` is ``"refused"`` (no number: a required fact is missing
and no assumption was given, or the numbers do not determine a result),
``"conditional"`` (a number that rests on the listed assumptions, on a
subset denominator or on an input outside its tabulated or fitted range)
or ``"inputs_stated"`` (no named assumption, no subset and no
extrapolated input). ``"inputs_stated"`` is not "validated": the number
still rests on the homogeneous σ × λ model and on the chosen matrix,
table, background and window. Nothing is filled in by default: the
declared elements, the exposure and its basis, the transmission state,
the intensity meaning and the matrix with its source are required; an
unknown one refuses unless the caller names an assumption, and an element
is never dropped from the denominator silently.

The expression is the standard one for a homogeneous sample,
``x_i = (A_i/S_i) / Σ_j (A_j/S_j)`` (the subject of ISO 18118, a guide
to experimentally determined relative sensitivity factors S for
homogeneous materials; 2015 edition, superseded by ISO 18118:2024), here
with the
intrinsic S = σ × λ. It is not validated in this version against a
measured stoichiometry.

Limits. All lines share one intensity meaning and one transmission state
(split the call otherwise). Shirley weights points, not eV, so it is
refused on a grid whose steps differ by more than 1 % (a float32 axis,
as the HDF5 cache stores it, stays well inside that). The window must lie inside the
data and contain the line.

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
Status = Literal["inputs_stated", "conditional", "refused"]
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
            the denominator. Required, and every line's element must be
            in it.
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
    extrapolated_inputs: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    statistical_uncertainty: None = None
    conditions: Conditions | None = field(default=None, repr=False)


QUANTITY = (
    "homogeneous-equivalent atomic fraction: area / (cross-section x TPP-2M IMFP), "
    "normalised over the denominator; not a complete AMRSF"
)


def _matrix_problem(m: Matrix) -> str | None:
    if not m.name.strip() or not m.source.strip():
        return "matrix needs a name and a source"
    if m.compound is None and any(v is None for v in (m.Nv, m.density, m.Mw, m.Eg)):
        return "matrix needs a compound or all four TPP-2M parameters (Nv, density, Mw, Eg)"
    return None


def _line_area(line: Line, background: str, divide_t: bool, exposure: float | None,
               photon_energy: float):
    """(area, definition, problem or None) for one line."""
    e = np.asarray(line.energy, dtype=np.float64)
    y = np.asarray(line.intensity, dtype=np.float64)
    t = None if line.transmission is None else np.asarray(line.transmission, dtype=np.float64)
    if e.ndim != 1 or y.shape != e.shape or (t is not None and t.shape != e.shape):
        return None, "", "energy, intensity and transmission differ in shape"
    if not np.all(np.isfinite(e)):
        return None, "", "non-finite energy"
    order = np.argsort(e)  # integrate on an ascending axis: the sign then means something
    e, y = e[order], y[order]
    t = None if t is None else t[order]
    lo, hi = sorted(line.window)
    if lo < e[0] or hi > e[-1]:
        return None, "", f"window [{lo:g}, {hi:g}] eV extends beyond the data"
    position = (line.binding_energy if line.energy_scale == "binding"
                else photon_energy - line.binding_energy)
    if not lo <= position <= hi:
        return None, "", f"the line ({position:g} eV) is outside its window"
    inside = (e >= lo) & (e <= hi)
    if inside.sum() < 3:
        return None, "", "fewer than 3 points inside the window"
    e, y = e[inside], y[inside]
    if not np.all(np.isfinite(y)):
        return None, "", "non-finite intensity inside the window"
    steps = np.diff(e)
    if background == "shirley" and np.ptp(steps) > 1e-2 * np.mean(steps):
        return None, "", "Shirley here weights points, not eV; the grid is not uniform"
    terms = ["I"]
    if exposure is not None:
        y = y / exposure
        terms.append("/t")
    if divide_t:
        t = t[inside]
        if np.any(~np.isfinite(t)) or np.any(t <= 0):
            return None, "", "transmission curve is not positive and finite in the window"
        y = y / t
        terms.append("/T")
    net = y - _BACKGROUNDS[background]().calculate(e, y)
    area = float(np.trapezoid(net, e))
    definition = (f"trapezoidal integral over [{lo:g}, {hi:g}] eV of "
                  f"{''.join(terms)} - {background} background, in eV x intensity unit")
    if not area > 0:
        return None, definition, f"net area is not positive ({area:.4g})"
    return area, definition, None


def composition(lines: Sequence[Line], conditions: Conditions) -> CompositionResult:
    """Homogeneous-equivalent composition, refusing what it cannot support.

    See the module docstring for what the number is and is not.
    """
    c = conditions
    reasons: list[str] = []
    assumptions: list[str] = []
    notes: list[str] = []
    extrapolated: list[str] = []
    not_evaluated = [
        ("statistical uncertainty: withheld; the scope validated on synthetic data is "
         "empty in this version (docs/design/composition-uncertainty.md §7)")
        if c.intensity_semantics == "raw_counts" else
        ("statistical uncertainty: not evaluated; the intensity is not stated raw counts, "
         "so the Poisson noise model does not apply (and no scope is validated in this "
         "version: docs/design/composition-uncertainty.md §7)"),
        "elastic scattering, angular distribution, polarization and geometry: "
        "not in the sensitivity",
    ]

    elements = [ln.element for ln in lines]
    if len(set(elements)) != len(elements):
        reasons.append("more than one line per element is not supported")
    declared = tuple(c.declared_elements or ())
    if not declared:
        reasons.append("declared_elements is required (the denominator is never implied)")
    elif len(set(declared)) != len(declared):
        reasons.append("declared_elements has duplicates")
    missing = [el for el in declared if el not in elements]
    if missing:
        reasons.append(f"declared elements without a line: {missing}")
    undeclared = [el for el in elements if el not in declared]
    if declared and undeclared:
        reasons.append(f"lines for undeclared elements: {undeclared}; declare them "
                       "(and exclude with normalise_over if intended)")
    excluded: dict[str, str] = {}
    denominator = declared
    if c.normalise_over is not None:
        subset = tuple(c.normalise_over)
        if not subset:
            reasons.append("normalise_over is empty: there is no denominator")
        elif len(set(subset)) != len(subset):
            reasons.append("normalise_over has duplicates")
        bad = [el for el in subset if el not in declared]
        if bad:
            reasons.append(f"normalise_over includes undeclared elements: {bad}")
        denominator = subset
        excluded = {el: c.exclusion_reason for el in declared if el not in subset}
        if excluded:
            if not c.exclusion_reason:
                reasons.append("normalise_over excludes elements; give an exclusion_reason")
            assumptions.append(
                f"normalised over {list(subset)} only, out of {list(declared)}")

    problem = _matrix_problem(c.matrix)
    if problem:
        reasons.append(problem)

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
        handling = "divided by the curve (assumed not applied)"
        assumptions.append("transmission curves not yet applied to the intensity")
    elif c.assume_transmission == "equal_across_lines":
        handling = "not divided (assumed equal across lines)"
        assumptions.append("transmission equal for every line, so it cancels")
        not_evaluated.append("transmission: assumed equal across lines")
    else:
        handling = "unknown"
        reasons.append("transmission state is unknown; name assume_transmission to proceed")
    if divide_t:
        # Nothing in a file states that two lines' curves share a scale.
        assumptions.append("transmission curves share one normalisation across lines "
                           "(not checked)")

    if c.background not in _BACKGROUNDS:
        reasons.append(f"background {c.background!r} is not one of {sorted(_BACKGROUNDS)}")

    reports: list[LineReport] = []
    for ln in lines:
        problems: list[str] = []
        exposure = ln.exposure_s if use_exposure else None
        if use_exposure:
            if ln.exposure_s is None:
                problems.append("exposure unknown for integrated counts")
            elif not (math.isfinite(ln.exposure_s) and ln.exposure_s > 0):
                problems.append(f"exposure must be positive, got {ln.exposure_s!r}")
            elif not ln.exposure_basis.strip():
                problems.append("exposure_basis is required (how exposure_s was obtained)")
        elif ln.exposure_s is not None:
            notes.append(f"{ln.element} {ln.orbital}: exposure_s ignored, the intensity is a "
                         "count rate")
        if divide_t and ln.transmission is None:
            problems.append("no transmission curve to divide by")
        area, definition = None, ""
        if not problems and c.background in _BACKGROUNDS:
            area, definition, area_problem = _line_area(
                ln, c.background, divide_t, exposure, c.photon_energy)
            if area_problem:
                problems.append(area_problem)
        sigma = CrossSection.lookup_with_status(ln.element, ln.orbital, c.photon_energy, c.table)
        if sigma.value is None or sigma.value <= 0:
            problems.append(f"no cross-section ({sigma.status})")
        elif sigma.status != "tabulated" and ln.element in denominator:
            extrapolated.append(f"{ln.element} {ln.orbital}: cross-section {sigma.status}")
        ke = c.photon_energy - ln.binding_energy
        lam: float | None = None
        if not problem:
            try:
                lam = c.matrix.imfp_nm(ke)
            except Exception as err:  # noqa: BLE001 - reported, not raised
                problems.append(f"IMFP unavailable: {type(err).__name__}")
        lam_extrap = not (TPP2M_FITTED_RANGE_EV[0] <= ke <= TPP2M_FITTED_RANGE_EV[1])
        if lam_extrap and ln.element in denominator:
            extrapolated.append(f"{ln.element} {ln.orbital}: IMFP at {ke:g} eV, outside "
                                "TPP-2M's 50-2000 eV")
        if problems and ln.element in denominator:
            reasons += [f"{ln.element} {ln.orbital}: {p}" for p in problems]
        reports.append(LineReport(
            element=ln.element, orbital=ln.orbital, source=ln.source, area=area,
            area_definition=definition, cross_section=sigma, kinetic_energy_eV=ke,
            imfp_nm=lam, imfp_extrapolated=lam_extrap, transmission_handling=handling,
            exposure_s=exposure, exposure_basis=ln.exposure_basis if use_exposure else "",
            problems=tuple(problems)))

    base = dict(quantity=QUANTITY, lines=tuple(reports), denominator=denominator,
                declared_elements=declared, excluded=excluded,
                assumptions=tuple(assumptions), not_evaluated=tuple(not_evaluated),
                extrapolated_inputs=tuple(extrapolated), notes=tuple(notes), conditions=c)
    if reasons:
        return CompositionResult(status="refused", fractions=None,
                                 reasons=tuple(reasons), **base)
    weights = {r.element: r.area / (r.cross_section.value * r.imfp_nm)
               for r in reports if r.element in denominator}
    total = math.fsum(weights.values())
    fractions = {el: weights[el] / total for el in denominator}
    status: Status = "conditional" if (assumptions or extrapolated) else "inputs_stated"
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
    if len(set(tables)) != len(tables) or len(set(backgrounds)) != len(backgrounds):
        raise ValueError("tables and backgrounds must not repeat")
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


# ---------------------------------------------------------------------------
# Statistical uncertainty (docs/design/composition-uncertainty.md)
# ---------------------------------------------------------------------------

_NOISE_MODEL = "independent Poisson counts per channel (intensity stated as raw counts)"
_RESAMPLING = ("each channel drawn from Poisson(observed count) before exposure, "
              "transmission and background; background, areas and composition recomputed")

#: The scope in which the reported standard uncertainty is published:
#: ``{"r_threshold": {background: r}, "max_lines": n, "record": path}``.
#: None — nothing is published in this version. The second, pre-registered
#: validation (design record §6–7) found the standard uncertainty itself
#: calibrated in 39 of 40 scenarios, but admitting data sets by their
#: measured relative area noise r miscalibrates the admitted subset
#: (R about 1.2), so its decision rule gave no scope.
_VALIDATED_SCOPE: dict | None = None


@dataclass(frozen=True)
class _CompositionUncertainty:
    """Standard uncertainty of each atomic fraction, or why there is none.

    ``status`` is ``"evaluated"``, ``"withheld"`` (computed or computable,
    but outside the validated scope or with refused replicates) or
    ``"not_evaluated"`` (the noise model does not apply). The standard
    uncertainty is one standard deviation of the estimator under
    :data:`_NOISE_MODEL`; it is not a confidence interval and does not
    cover condition dependence.
    """

    status: Literal["evaluated", "withheld", "not_evaluated"]
    reasons: tuple[str, ...]
    elements: tuple[str, ...]
    standard_uncertainty: dict[str, float] | None
    covariance: tuple[tuple[float, ...], ...] | None
    n_boot: int
    n_refused: int
    seed: int | None
    noise_model: str = _NOISE_MODEL
    resampling: str = _RESAMPLING
    validated_scope: dict | None = None


def _resampled(lines: Sequence[Line], rng: np.random.Generator) -> list[Line]:
    return [replace(ln, intensity=rng.poisson(np.asarray(ln.intensity, dtype=np.float64))
                    .astype(np.float64)) for ln in lines]


def _bootstrap_detail(lines: Sequence[Line], conditions: Conditions, n_boot: int,
                      rng: np.random.Generator, elements: Sequence[str]):
    """(fractions (n_ok, n_el), n_refused, areas (n_ok, n_el)) over Poisson replicates.

    The areas are each replicate's line areas in the order of ``elements``,
    on the corrected scale composition() uses (a constant exposure does not
    change their relative spread).
    """
    rows, areas, refused = [], [], 0
    for _ in range(n_boot):
        r = composition(_resampled(lines, rng), conditions)
        if r.fractions is None:
            refused += 1
        else:
            rows.append([r.fractions[el] for el in elements])
            by_el = {x.element: x.area for x in r.lines}
            areas.append([by_el[el] for el in elements])
    shape = (-1, len(elements))
    return (np.asarray(rows, dtype=np.float64).reshape(shape), refused,
            np.asarray(areas, dtype=np.float64).reshape(shape))


def _bootstrap(lines: Sequence[Line], conditions: Conditions, n_boot: int,
               rng: np.random.Generator, elements: Sequence[str]):
    """(fractions array (n_ok, n_el), n_refused) over Poisson replicates."""
    rows, refused, _ = _bootstrap_detail(lines, conditions, n_boot, rng, elements)
    return rows, refused


def _relative_area_noise(areas: np.ndarray, observed: Sequence[float]) -> float:
    """r: the largest, over lines, of SD over replicates / observed net area."""
    return float(max(np.std(areas[:, k], ddof=1) / observed[k]
                     for k in range(areas.shape[1])))


def _composition_uncertainty(
    lines: Sequence[Line],
    conditions: Conditions,
    n_boot: int = 1000,
    seed: int | None = None,
) -> _CompositionUncertainty:
    """Bootstrap standard uncertainty of the composition, inside its validated scope.

    Private while :data:`_VALIDATED_SCOPE` is empty: a public function that
    always withholds would only invite reading its output as an answer.
    In this version the scope is empty (:data:`_VALIDATED_SCOPE` is None):
    the result is always ``withheld`` or ``not_evaluated``, with the reason.
    The calculation and its checks are in place for a later scope.

    Each replicate redraws every channel at the count stage and recomputes
    background, areas and composition (design record §2). Reported only
    when the intensity is stated as raw counts, every replicate succeeds,
    and the case lies in :data:`_VALIDATED_SCOPE`; otherwise ``withheld``
    or ``not_evaluated`` with the reason.
    """
    c = conditions
    base = composition(lines, c)
    elements = base.denominator

    def result(status, reasons, se=None, cov=None, refused=0):
        return _CompositionUncertainty(
            status=status, reasons=tuple(reasons), elements=tuple(elements),
            standard_uncertainty=se, covariance=cov, n_boot=n_boot, n_refused=refused,
            seed=seed, validated_scope=_VALIDATED_SCOPE)

    if base.fractions is None:
        return result("not_evaluated", ["the composition itself is refused"])
    if c.intensity_semantics != "raw_counts":
        return result("not_evaluated", [
            f"intensity is {c.intensity_semantics!r}, not stated raw counts; the Poisson "
            "noise model does not apply"])
    if any(np.any(np.asarray(ln.intensity) < 0) for ln in lines):
        return result("not_evaluated", ["negative counts"])
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")

    if any(not np.array_equal(np.asarray(ln.intensity), np.round(np.asarray(ln.intensity)))
           for ln in lines):
        return result("not_evaluated", ["intensity stated as raw counts is not integer"])
    rows, refused, areas = _bootstrap_detail(
        lines, c, n_boot, np.random.default_rng(seed), elements)
    if refused:
        return result("withheld", [f"{refused} of {n_boot} replicates were refused; a "
                                   "standard deviation over the rest would be conditional "
                                   "on success"], refused=refused)
    cov = np.atleast_2d(np.cov(rows, rowvar=False, ddof=1))
    se = {el: float(math.sqrt(cov[i, i])) for i, el in enumerate(elements)}
    cov_t = tuple(tuple(float(v) for v in row) for row in cov)

    reasons = []
    scope = _VALIDATED_SCOPE
    if scope is None:
        reasons.append("no validated scope in this version (design record "
                       "docs/design/composition-uncertainty.md §7)")
    else:
        threshold = scope["r_threshold"].get(c.background)
        if threshold is None:
            reasons.append(f"background {c.background!r} was not validated")
        else:
            observed = {x.element: x.area for x in base.lines}
            r = _relative_area_noise(areas, [observed[el] for el in elements])
            if r > threshold:
                reasons.append(f"relative area noise {r:.3g} above the validated {threshold:g}")
        if len(elements) > scope["max_lines"]:
            reasons.append(f"{len(elements)} lines; validated up to {scope['max_lines']}")
        if any(x.transmission_handling.startswith("divided") for x in base.lines):
            reasons.append("transmission divided out: not validated")
    if reasons:
        return result("withheld", reasons, se=None, cov=None)
    return result("evaluated", [], se=se, cov=cov_t)
