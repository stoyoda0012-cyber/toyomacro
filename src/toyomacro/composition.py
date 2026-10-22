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
- the statistical uncertainty — not computed by :func:`composition`
  itself; :func:`composition_uncertainty` computes it from a
  :class:`DeclaredSpectrum` fixed before the data set was looked at,
  inside the scope of docs/design/composition-uncertainty.md §8–10,
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

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

import numpy as np

import toyomacro
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
        ("statistical uncertainty: not computed in this call; composition_uncertainty with "
         "a declared spectrum computes it (docs/design/composition-uncertainty.md §10)")
        if c.intensity_semantics == "raw_counts" else
        ("statistical uncertainty: not computed in this call, and the intensity is not stated "
         "raw counts, so the Poisson noise model does not apply; composition_uncertainty "
         "with a declared spectrum computes it for stated raw counts "
         "(docs/design/composition-uncertainty.md §10)"),
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


#: Series 2 of docs/design/composition-uncertainty.md §8: the key the
#: prediction's generator is seeded with at every call. It names no
#: condition, level, kappa or run kind, so the same declared spectrum and
#: conditions always give the same prediction.
_PREDICTION_SEED_KEY: tuple[int, ...] = (20261014, 2)


@dataclass(frozen=True)
class _Prediction:
    """What :func:`_predict_r` returns (design record §8, "The prediction")."""

    r_pred: float | None
    declared_refused: bool
    declared_reasons: tuple[str, ...]
    n_refused: int | None  # None: the bootstrap was not run (composition(D) refused)
    n_boot: int
    elements: tuple[str, ...]


def _predict_r(
    declared: Sequence[Line],
    conditions: Conditions,
    n_boot: int = 1000,
    key: Sequence[int] = _PREDICTION_SEED_KEY,
) -> _Prediction:
    """Predicted relative area noise of a declared expected spectrum (§8).

    ``declared`` carries, per line, the expected counts on the line's own
    grid (base included) in place of the observed intensity, and every
    other ``Line`` field as the data would have it. The arrays must be
    non-negative, finite and real; anything else raises ``ValueError``
    (a run error in the validation; what a public function would return
    is not decided here). They are passed as Poisson means to
    :func:`_bootstrap_detail` **without** rounding, from a generator
    freshly seeded with ``key`` at every call, and r is as in §6 with
    ``ddof = 1`` over the replicates and the declared spectrum's own net
    area (corrected scale) as the divisor. The integer check that
    :func:`composition_uncertainty` applies to observed data is not
    applied here.
    """
    for ln in declared:
        raw = np.asarray(ln.intensity)
        if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
            raise ValueError(f"{ln.element} {ln.orbital}: a declared spectrum must be real-valued")
        arr = raw.astype(np.float64)
        if arr.ndim != 1 or not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"{ln.element} {ln.orbital}: a declared spectrum must be a 1-D "
                             "array of non-negative finite reals")
    base = composition(declared, conditions)
    if base.fractions is None:
        # The bootstrap is not run: n_refused is None, not 0.
        return _Prediction(r_pred=None, declared_refused=True, declared_reasons=base.reasons,
                           n_refused=None, n_boot=n_boot, elements=())
    elements = base.denominator
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(key))))
    rows, refused, areas = _bootstrap_detail(declared, conditions, n_boot, rng, elements)
    # r is defined over the full set of replicates only: with a refused
    # replicate it is None (as round 2 did), and the case is not admitted.
    r = None
    if refused == 0 and len(rows) >= 2:
        observed = {x.element: x.area for x in base.lines}
        r = _relative_area_noise(areas, [observed[el] for el in elements])
    return _Prediction(r_pred=r, declared_refused=False, declared_reasons=(), n_refused=refused,
                       n_boot=n_boot, elements=tuple(elements))


# ---------------------------------------------------------------------------
# The published scope and the public function (design record §10)
# ---------------------------------------------------------------------------

#: The scope in which the standard uncertainty is returned, from the third
#: validation (design record §8–9): admission by the predicted relative area
#: noise of the declared spectrum, r_pred, at or below the background's
#: threshold. Tied to the record's decision by a test.
_VALIDATED_SCOPE: dict = {
    "r_pred_threshold": {"linear": 0.02, "shirley": 0.01},
    "max_lines": 3,
    "min_elements": 2,
    "record": "docs/design/composition-uncertainty-results-3.json",
}
#: The data-side bootstrap's key in the public function (§10): fixed, so the
#: same data and declaration always give the same result. The validation
#: keyed the data side per trial ([seed, 4, run kind, condition, trial]).
_PUBLIC_DATA_KEY: tuple[int, ...] = (20261014, 4)
_B_DATA = 1000
_B_PRED = 1000

#: The user's assumptions, by name (§10, "The declared input"). The first
#: three are required; the fourth is recorded when given and never required
#: by the code, since no numerical rule can tell when it applies.
ASSUMPTIONS = ("channel_independence", "no_prior_correction", "transfer",
               "more_counts_extrapolation")
REQUIRED_ASSUMPTIONS = ASSUMPTIONS[:3]
Provenance = Literal["planning_model", "other_measurement"]
Applicability = Literal["assumed", "registered"]

#: Values the §8 conditions were registered at, for the ``registered`` report.
_REGISTERED_VALUES = {
    "table": ("scofield",), "background": ("linear", "shirley"),
    "photon_energy_eV": (1486.6, 9251.7), "matrix_compound": ("SiO2",),
    "lines": ((("Si", "2p"), ("O", "1s")), (("Si", "2p"), ("O", "1s"), ("C", "1s"))),
    "window_half_width_eV": (10.0,), "channel_step_eV": (0.1, 0.5), "exposure_s": (1.0,),
}


def _real_array(name: str, x, shape=None) -> np.ndarray:
    raw = np.asarray(x)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name}: must be real-valued")
    arr = raw.astype(np.float64)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: must be a 1-D array of finite reals")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name}: shape {arr.shape} differs from the energy grid {shape}")
    return arr.copy()


@dataclass(frozen=True, eq=False)
class DeclaredLine:
    """One line of a declared spectrum: the expected counts on the line's own grid.

    The fields are those of :class:`Line` with ``expected`` in place of
    ``intensity``: expected counts per channel, base included, as
    non-negative finite reals (not rounded). Arrays are copied on
    construction, so later changes to the caller's arrays do not reach
    the declaration.
    """

    element: str
    orbital: str
    energy: np.ndarray
    expected: np.ndarray
    window: tuple[float, float]
    binding_energy: float
    energy_scale: Literal["kinetic", "binding"] = "kinetic"
    exposure_s: float | None = None
    exposure_basis: str = ""

    def __post_init__(self):
        e = _real_array(f"{self.element} {self.orbital} energy", self.energy)
        x = _real_array(f"{self.element} {self.orbital} expected", self.expected, e.shape)
        if np.any(x < 0):
            raise ValueError(f"{self.element} {self.orbital} expected: negative values")
        object.__setattr__(self, "energy", e)
        object.__setattr__(self, "expected", x)
        object.__setattr__(self, "window", (float(self.window[0]), float(self.window[1])))


@dataclass(frozen=True, eq=False)
class DeclaredSpectrum:
    """What the user declares before looking at the data set (design record §10).

    Attributes:
        lines: One :class:`DeclaredLine` per observed line, in the same order.
        conditions: The :class:`Conditions` the declaration was fixed with;
            the call's conditions must equal them.
        provenance: ``"planning_model"`` (a pre-measurement estimate) or
            ``"other_measurement"`` (another data set sharing nothing with
            the one evaluated). Anything derived from the data set being
            evaluated — a fit, a smoothing, its areas — is ``"this_data"``
            and is refused.
        provenance_basis: Where the expectation came from, in words.
        fixed_before_data: The user's statement that the expected counts
            **and** the analysis conditions were fixed before the data set
            was looked at. Not verifiable here; recorded.
        assumptions: Names from :data:`ASSUMPTIONS`. The first three are
            required for a result; ``more_counts_extrapolation`` is recorded
            when given.
        notes: Free text per assumption name.
    """

    lines: tuple[DeclaredLine, ...]
    conditions: Conditions
    provenance: str
    provenance_basis: str
    fixed_before_data: bool
    assumptions: tuple[str, ...] = ()
    notes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.provenance == "this_data":
            raise ValueError("a declaration derived from the data set being evaluated "
                             "(this_data) is refused: it would select by the data's own "
                             "fluctuations")
        if self.provenance not in ("planning_model", "other_measurement"):
            raise ValueError(f"provenance must be planning_model or other_measurement, "
                             f"got {self.provenance!r}")
        if not str(self.provenance_basis).strip():
            raise ValueError("provenance_basis is required: say where the expectation came from")
        if not self.lines:
            raise ValueError("a declaration needs at least one line")
        unknown = [a for a in self.assumptions if a not in ASSUMPTIONS]
        if unknown:
            raise ValueError(f"unknown assumption name(s) {unknown}; known: {list(ASSUMPTIONS)}")
        if len(set(self.assumptions)) != len(self.assumptions):
            raise ValueError("assumptions has duplicates")
        if self.fixed_before_data is not True and self.fixed_before_data is not False:
            raise ValueError("fixed_before_data must be True or False: it is your statement")
        object.__setattr__(self, "lines", tuple(self.lines))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "notes", dict(self.notes))

    def as_lines(self) -> list[Line]:
        """The declaration as :class:`Line` objects, for the prediction."""
        return [Line(element=d.element, orbital=d.orbital, energy=d.energy.copy(),
                     intensity=d.expected.copy(), window=tuple(d.window),
                     binding_energy=d.binding_energy, energy_scale=d.energy_scale,
                     exposure_s=d.exposure_s, exposure_basis=d.exposure_basis,
                     source=f"declared ({self.provenance})") for d in self.lines]

    def snapshot(self) -> dict:
        """A plain-data copy for the record."""
        return {
            "lines": [{"element": d.element, "orbital": d.orbital,
                       "energy": [float(v) for v in d.energy],
                       "expected": [float(v) for v in d.expected],
                       "window": [float(d.window[0]), float(d.window[1])],
                       "binding_energy": d.binding_energy, "energy_scale": d.energy_scale,
                       "exposure_s": d.exposure_s, "exposure_basis": d.exposure_basis}
                      for d in self.lines],
            "conditions": _conditions_snapshot(self.conditions),
            "provenance": self.provenance, "provenance_basis": self.provenance_basis,
            "fixed_before_data": self.fixed_before_data,
            "assumptions": list(self.assumptions), "notes": dict(self.notes),
        }


def _conditions_snapshot(c: Conditions) -> dict:
    d = asdict(c)
    d["matrix"] = asdict(c.matrix)
    return d


@dataclass(frozen=True)
class CompositionUncertainty:
    """The standard uncertainty of a composition, or why there is none (§10).

    Two fields are kept apart. ``status`` says what the code did:
    ``"evaluated"`` (a standard uncertainty is returned), ``"withheld"``
    (computable in principle, not returned, with the reason) or
    ``"not_evaluated"`` (the noise model does not apply, or the
    composition itself is refused). ``applicability`` says on what ground
    the calibration is applied: ``"assumed"`` — always, from the public
    function — with the named assumptions the result rests on;
    ``"registered"`` only from the tests, on data sets drawn from a
    registered condition. The standard uncertainty is one standard
    deviation of the estimator under :data:`_NOISE_MODEL` (sample
    standard deviation over the replicates, ddof = 1); it is not a
    confidence interval and does not cover condition dependence.
    """

    composition: CompositionResult
    status: Literal["evaluated", "withheld", "not_evaluated"]
    reasons: tuple[str, ...]
    applicability: Applicability
    assumptions: tuple[str, ...]
    elements: tuple[str, ...]
    standard_uncertainty: dict[str, float] | None
    covariance: tuple[tuple[float, ...], ...] | None
    prediction: dict | None
    data_side: dict | None
    declared: dict
    registered: dict
    versions: dict
    noise_model: str = _NOISE_MODEL
    resampling: str = _RESAMPLING
    scope: dict = field(default_factory=lambda: copy.deepcopy(_VALIDATED_SCOPE))


def _check_declaration(lines: Sequence[Line], conditions: Conditions,
                       declared: DeclaredSpectrum) -> None:
    """§10 check 6 and the input inconsistencies: a mismatch is a ValueError."""
    if conditions.intensity_semantics == "raw_counts" and conditions.assume_intensity is not None:
        raise ValueError("assume_intensity given with intensity stated as raw counts: an input "
                         "inconsistency, not an assumption")
    for ln in lines:
        raw = np.asarray(ln.intensity)
        if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
            raise ValueError(f"{ln.element} {ln.orbital}: the observed intensity must be "
                             "real-valued")
    if len(lines) != len(declared.lines):
        raise ValueError(f"the declaration has {len(declared.lines)} lines, the data "
                         f"{len(lines)}")
    for ln, d in zip(lines, declared.lines):
        where = f"{ln.element} {ln.orbital}"
        if (ln.element, ln.orbital) != (d.element, d.orbital):
            raise ValueError(f"line order differs: data {where}, declaration "
                             f"{d.element} {d.orbital}")
        if not np.array_equal(np.asarray(ln.energy, dtype=np.float64), d.energy):
            raise ValueError(f"{where}: the declared energy grid differs from the data's")
        if (tuple(ln.window) != tuple(d.window) or ln.binding_energy != d.binding_energy
                or ln.energy_scale != d.energy_scale or ln.exposure_s != d.exposure_s
                or ln.exposure_basis != d.exposure_basis):
            raise ValueError(f"{where}: window, binding energy, energy scale, exposure or "
                             "exposure basis differ between the data and the declaration")
    if conditions != declared.conditions:
        raise ValueError("the conditions passed differ from the ones the declaration was "
                         "fixed with")


def _registered_report(lines: Sequence[Line], c: Conditions) -> dict:
    """Which registered values the case matches, and which it does not (§10)."""
    steps = [float(np.mean(np.diff(np.sort(np.asarray(ln.energy, dtype=np.float64)))))
             for ln in lines]
    halves = [float((max(ln.window) - min(ln.window)) / 2) for ln in lines]
    found = {
        "table": c.table, "background": c.background, "photon_energy_eV": c.photon_energy,
        "matrix_compound": c.matrix.compound,
        "lines": tuple((ln.element, ln.orbital) for ln in lines),
        "window_half_width_eV": halves, "channel_step_eV": steps,
        "exposure_s": [ln.exposure_s for ln in lines],
    }
    reg = _REGISTERED_VALUES
    close = lambda v, ref: any(math.isclose(v, r, rel_tol=1e-6) for r in ref)  # noqa: E731
    matches = {
        "table": found["table"] in reg["table"],
        "background": found["background"] in reg["background"],
        "photon_energy_eV": close(found["photon_energy_eV"], reg["photon_energy_eV"]),
        "matrix_compound": found["matrix_compound"] in reg["matrix_compound"],
        "lines": found["lines"] in reg["lines"],
        "window_half_width_eV": all(close(h, reg["window_half_width_eV"]) for h in halves),
        "channel_step_eV": all(close(s, reg["channel_step_eV"]) for s in steps),
        "exposure_s": all(e is not None and close(e, reg["exposure_s"]) for e in found["exposure_s"]),
    }
    return {"found": found, "registered": {k: list(v) for k, v in reg.items()},
            "matches": matches,
            "differs": sorted(k for k, ok in matches.items() if not ok),
            "assumed_only": ["peak shapes and widths", "background shape",
                             "relative intensities of the lines"],
            "scope": copy.deepcopy(_VALIDATED_SCOPE)}


def _uncertainty(lines: Sequence[Line], conditions: Conditions, declared: DeclaredSpectrum,
                 *, data_key: Sequence[int], registered: bool) -> CompositionUncertainty:
    """The steps of §10 "What the code checks", in order."""
    if not isinstance(declared, DeclaredSpectrum):
        raise TypeError("declared must be a DeclaredSpectrum")
    _check_declaration(lines, conditions, declared)
    c = conditions
    base = composition(lines, c)
    elements = tuple(base.denominator)
    versions = {"toyomacro": toyomacro.__version__, "numpy": np.__version__,
                "scipy": __import__("scipy").__version__,
                "data": [ln.source for ln in lines]}
    common = dict(composition=base, applicability="registered" if registered else "assumed",
                  assumptions=tuple(declared.assumptions), elements=elements,
                  declared=declared.snapshot(), registered=_registered_report(lines, c),
                  versions=versions)

    def out(status, reasons, se=None, cov=None, prediction=None, data_side=None):
        return CompositionUncertainty(status=status, reasons=tuple(reasons),
                                      standard_uncertainty=se, covariance=cov,
                                      prediction=prediction, data_side=data_side, **common)

    if base.fractions is None:
        return out("not_evaluated", ["the composition itself is refused: " + "; ".join(base.reasons)])
    # check 1: the noise model applies
    if c.intensity_semantics != "raw_counts":
        return out("not_evaluated", [f"intensity is {c.intensity_semantics!r}, not stated raw "
                                     "counts; the Poisson noise model does not apply"])
    for ln in lines:
        x = np.asarray(ln.intensity, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            return out("not_evaluated", [f"{ln.element} {ln.orbital}: non-finite counts"])
        if np.any(x < 0):
            return out("not_evaluated", [f"{ln.element} {ln.orbital}: negative counts"])
        if not np.array_equal(x, np.round(x)):
            return out("not_evaluated", [f"{ln.element} {ln.orbital}: intensity stated as raw "
                                         "counts is not integer"])
    scope = _VALIDATED_SCOPE
    # check 2: table, background, denominator
    reasons = []
    if c.table != "scofield":
        reasons.append(f"table {c.table!r}: validated for scofield only")
    if c.background not in scope["r_pred_threshold"]:
        reasons.append(f"background {c.background!r} was not validated")
    if c.normalise_over is not None:
        reasons.append("normalise_over: validated for the declared elements as denominator only")
    if not scope["min_elements"] <= len(elements) <= scope["max_lines"]:
        reasons.append(f"{len(elements)} elements in the denominator; validated for "
                       f"{scope['min_elements']} to {scope['max_lines']}")
    if reasons:
        return out("withheld", reasons)
    # check 3: no transmission correction before storage or in the calculation
    admitted_t = c.transmission_applied == "unknown" and c.assume_transmission == "equal_across_lines"
    if registered and c.transmission_applied == "applied" and c.assume_transmission is None:
        admitted_t = True  # the harness's label on synthetic data: the same computation path
    if not admitted_t:
        why = ("the stored intensity was already divided by a transmission curve "
               "(transmission_applied = 'applied'); it is no longer raw counts"
               if c.transmission_applied == "applied" else
               "a transmission curve would be divided out in the calculation; not validated"
               if c.transmission_applied == "not_applied" or c.assume_transmission == "divide_by_curve"
               else "transmission state must be 'unknown' with assume_transmission = "
                    "'equal_across_lines' (nothing divided)")
        return out("withheld", [why])
    # the required assumptions
    missing = [a for a in REQUIRED_ASSUMPTIONS if a not in declared.assumptions]
    if not declared.fixed_before_data:
        missing.insert(0, "fixed_before_data")
    if missing:
        return out("withheld", [f"applicability not declared: {m}" for m in missing])
    # check 4 (prediction side) and check 5: the admission, from the declaration alone
    pred = _predict_r(declared.as_lines(), c, n_boot=_B_PRED)
    threshold = scope["r_pred_threshold"][c.background]
    prediction = {"r_pred": pred.r_pred, "r_star": threshold, "declared_refused": pred.declared_refused,
                  "declared_reasons": list(pred.declared_reasons), "n_refused": pred.n_refused,
                  "n_boot": _B_PRED, "key": list(_PREDICTION_SEED_KEY)}
    if pred.declared_refused:
        return out("withheld", ["the composition of the declared spectrum is refused: "
                                + "; ".join(pred.declared_reasons)], prediction=prediction)
    if pred.n_refused:
        return out("withheld", [f"{pred.n_refused} of {_B_PRED} prediction replicates refused"],
                   prediction=prediction)
    if pred.r_pred is None:
        return out("withheld", ["the prediction returned no r_pred"], prediction=prediction)
    if pred.r_pred > threshold:
        return out("withheld", [f"predicted relative area noise {pred.r_pred:.4g} above the "
                                f"validated {threshold:g} for the {c.background} background"],
                   prediction=prediction)
    # check 4 (data side): the bootstrap of the observed data
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(k) for k in data_key])))
    rows, refused, _ = _bootstrap_detail(lines, c, _B_DATA, rng, elements)
    data_side = {"n_boot": _B_DATA, "n_refused": refused, "key": [int(k) for k in data_key]}
    if refused:
        return out("withheld", [f"{refused} of {_B_DATA} replicates were refused; a standard "
                                "deviation over the rest would be conditional on success"],
                   prediction=prediction, data_side=data_side)
    sd = np.std(rows, axis=0, ddof=1)
    cov = np.atleast_2d(np.cov(rows, rowvar=False, ddof=1))
    if not (np.all(np.isfinite(sd)) and np.all(np.isfinite(cov))):
        return out("withheld", ["non-finite standard uncertainty"], prediction=prediction,
                   data_side=data_side)
    se = {el: float(v) for el, v in zip(elements, sd)}
    cov_t = tuple(tuple(float(v) for v in row) for row in cov)
    return out("evaluated", [], se=se, cov=cov_t, prediction=prediction, data_side=data_side)


def composition_uncertainty(lines: Sequence[Line], conditions: Conditions, *,
                            declared: DeclaredSpectrum) -> CompositionUncertainty:
    """Standard uncertainty of the composition, admitted by a declaration made beforehand.

    Experimental. Implements docs/design/composition-uncertainty.md §10:
    the declaration (expected counts and the analysis conditions, fixed
    before the data set was looked at) decides whether the uncertainty is
    computed, through the predicted relative area noise of the declared
    spectrum against the background's validated threshold; nothing
    computed from the observed data enters that admission. B = 1,000 on
    both sides, with fixed keys; there is no ``n_boot``, no seed and no
    retry. ``applicability`` is always ``"assumed"``: the result rests on
    the named assumptions the declaration carries, and the registered
    calibration (§9) transfers to this case only under them.
    """
    return _uncertainty(lines, conditions, declared, data_key=_PUBLIC_DATA_KEY, registered=False)


def _composition_uncertainty_registered(lines: Sequence[Line], conditions: Conditions,
                                        declared: DeclaredSpectrum, *,
                                        data_key: Sequence[int] = _PUBLIC_DATA_KEY,
                                        ) -> CompositionUncertainty:
    """The tests' entry point: ``applicability = "registered"`` and a data-side key.

    Only for data sets drawn from a registered condition (§8); the key
    lets a test reproduce the validation's per-trial draws. The public
    function has neither.
    """
    return _uncertainty(lines, conditions, declared, data_key=data_key, registered=True)
