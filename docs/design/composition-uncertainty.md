# Statistical uncertainty of a composition — design record

Dated 2026-10-07. Records what the composition bootstrap (private,
`toyomacro.composition._composition_uncertainty`; first drafted as the
public `composition_uncertainty`) computes, under which noise model, and how it was checked
before being published. Every figure here comes from simulation with a
fixed seed; none is a measurement.

Sections 1–4 were written and committed **before** any validation run,
and the scenarios and the acceptance rule in §3 are not changed after
seeing results. §5 is filled in from the committed record
(`composition-uncertainty-results.json`).

## 1. What is estimated

The estimator is `composition()`: background-subtracted window areas
divided by σ × λ and normalised over the declared elements. The
uncertainty reported is the **standard uncertainty of each atomic
fraction** — one standard deviation of the estimator under the noise
model below — together with the covariance matrix of the fraction
vector. It is not a confidence interval, and it does not cover the
choice of table, background, window or matrix (that is condition
dependence) or anything outside the model (elastic scattering,
homogeneity).

## 2. Noise model and resampling

- **Independent Poisson counts per channel.** Only when the intensity is
  stated to be raw counts (`intensity_semantics == "raw_counts"` in the
  conditions, not reached through `assume_intensity`). Otherwise the
  uncertainty is not evaluated. Raw counts do not by themselves make the
  channels independent: on one laboratory detector adjacent channels of
  the *stored* intensity correlated at +0.16 (v0.4.0 known issue), and
  where that holds this standard uncertainty is too small.
- **Resampling at the count stage.** Each replicate draws every channel
  from a Poisson distribution whose mean is the **observed** count
  (no peak model is fitted, so there is no fitted expectation to draw
  from), before the exposure division, the transmission division and the
  background estimate, and then recomputes the background, the areas and
  the composition.
- **Failures.** A replicate that `composition()` refuses (a net area that
  is not positive, for instance) is counted. If any replicate is refused
  the standard uncertainty is **withheld**: a standard deviation over the
  surviving replicates would be conditional on success, and no threshold
  on the failure rate has been validated.

## 3. Validation (fixed before the runs)

For a scenario with known expected counts μ (Gaussian peaks of given
area on a linear base, per channel), draw M outer data sets from
Poisson(μ); for each, compute the estimate and its bootstrap standard
uncertainty with B replicates. Then

    R = sqrt(mean over outer sets of SE²) / SD over outer sets of the estimate

per element. R = 1 means the reported standard uncertainty matches the
spread it claims to describe.

- **Monte Carlo error.** The 95 % interval of R is taken from 5,000
  resamples of the M outer (estimate, SE) pairs, recomputing numerator
  and denominator together each time.
- **Acceptance.** A scenario passes when, for every element, the whole
  95 % interval of R lies inside [0.9, 1.1]. If the interval crosses a
  bound the scenario is undecided; if it lies outside, it fails. The
  ±10 % tolerance and the target half-width of 0.05 are this project's
  choice, not a value from the literature.
- **Bookkeeping.** For each scenario: M attempted, M with an estimate,
  M with a standard uncertainty, refused counts, R and its interval, and
  the bias of the estimate against the composition of μ itself.
- **Runs.** A pilot with M = 200, B = 200 is used only to check cost and
  Monte Carlo error. The decision uses M = 2,000, B = 1,000.
- **Scenarios.** 241 channels 0.1 eV apart over 24 eV around each line,
  window ±10 eV, Gaussian σ = 0.6 eV, Si 2p and O 1s at Al Kα, Scofield
  table, SiO₂ matrix, exposure 1 s. Areas are integrals in counts × eV
  (a 30,000 area holds 300,000 counts at 0.1 eV per channel); the base
  is counts per channel:

  | name | Si 2p area | O 1s area | base per channel | background | role |
  |---|---|---|---|---|---|
  | S1 high | 30,000 | 90,000 | 200 | linear | acceptance |
  | S2 medium | 3,000 | 9,000 | 50 | Shirley | acceptance |
  | S3 low | 300 | 900 | 5 | linear | reported, not used to widen the scope |

## 4. What is published

`composition_uncertainty` reports the standard uncertainty only inside
the scope that passed in §5: the noise model above, a background type
whose scenario passed, and every line's raw net area at or above the
smallest area of a passing scenario. Outside it the result is
`withheld` with the reason. The scope constants in the module are tied
to the record by a test.

## 5. Results

From `composition-uncertainty-results.json` (runs at commit c298bcd,
M = 2,000 outer data sets, B = 1,000 replicates each, seed 20261007).
With two elements the fractions sum to one, so Si and O share one R;
one row per scenario.

| scenario | role | background | R | 95 % MC interval | verdict |
|---|---|---|---|---|---|
| S1 high | acceptance | linear | 1.003 | [0.973, 1.036] | pass |
| S2 medium | acceptance | Shirley | 1.005 | [0.975, 1.036] | pass |
| S3 low | reported | linear | 1.051 | [1.021, 1.084] | inside the band, not used |

- Monte Carlo half-widths are 0.030–0.032, inside the 0.05 target.
- In S1 and S2 every outer data set had a standard uncertainty (no
  refused replicate). In S3, 3 of 2,000 had refused replicates; the
  published function would withhold those, and R is over the other
  1,997, so S3's R is conditional on success.
- S3 overstates the spread by about 5 %. It lies inside the ±10 % band,
  but by §3 it was registered as "reported", so it does not widen the
  scope.
- Bias of the Si fraction against the composition of μ (the estimator
  applied to the expected counts — the estimator's own bias, not a model
  error): S1 −0.7 × 10⁻⁵ ± 4.1 × 10⁻⁵, S2 −3.7 × 10⁻⁴ ± 2.0 × 10⁻⁴
  (1.9 standard errors), S3 −3.5 × 10⁻⁴ ± 6.4 × 10⁻⁴. The standard
  uncertainty in S2 is about 9 × 10⁻³, so this bias is about 4 % of it.

**Scope adopted, then withdrawn.** (The independent audit of this
section found the guard below on the wrong axis and selecting the
validated scenarios' own data sets; §6–7 replace it, and no scope is
published.) §4 says "a background type whose scenario passed,
and every line's raw net area at or above the smallest area of a
passing scenario". Read across backgrounds, that would let a linear
background through at 3,000 although linear passed only at 30,000. The
module reads it per background, the narrower reading: linear at
≥ 30,000 counts × eV, Shirley at ≥ 3,000. This is a necessary guard and
not a sufficient one: the scenarios also fix the peak shape, the window,
the channel spacing, the base level and two lines at Al Kα, none of
which the check sees. Outside these synthetic conditions the reported
value is the same calculation without a validation behind it.

**Reading the first record.** In `composition-uncertainty-results.json`,
`R` is over the data sets that have a standard uncertainty (all 2,000 in
S1 and S2, 1,997 in S3) while `sd_estimate` and `bias` are over every
data set with an estimate, so R is not `rms_se / sd_estimate` in S3. Its
`verdict` is the band test of §3 for every scenario; S3's "pass" there
does not change its "reported" role. Commit 807ec13, which states the
area unit in §3, also carries the first version of
`composition_uncertainty`; no run had been made at that point.

**Not validated.** Real detectors (see §2 on channel correlation); more
than two elements; other photon energies, tables and matrices; Tougaard;
transmission division in the resampled pipeline; and whether the
estimate is right — this checks only that the reported spread matches
the spread of the estimator.

## 6. Second validation: a scope on the right axis (pre-registered)

*Written after the independent audit of §5 and before any of the
runs below. Committed before they start; not changed after.*

The audit found the §4 guard on the wrong axis. Net area in counts × eV
is not how hard a case is: at a fixed area, a larger background or wider
channels make the bootstrap worse, and a case with the peak at 11 % of a
large base gave R = 1.11 [1.06, 1.17] while the guard said "evaluated".
It also cut half of the validated scenarios' own data sets, because the
threshold sat at their expected area. §3's mechanism (resampling about
the observed counts) overstates by an amount that grows with the
relative noise of the net areas (+5 % at S3). So the guard moves to that
quantity, measured by the bootstrap itself.

**Guard quantity.** For each data set, r = the largest, over the lines
in the denominator, of (standard deviation over the replicates of the
line's raw net area) / (its raw net area). It is computed from the same
replicates as the standard uncertainty, so it costs nothing extra and
needs no expected value.

**Scenarios.** Si 2p and O 1s (and C 1s at 285 eV where stated), Gaussian
σ = 0.6 eV, window ±10 eV, Scofield, SiO₂ matrix, exposure 1 s, no
transmission division. Four configurations × two backgrounds (linear,
Shirley) × five noise levels = 40 scenarios:

| config | lines | photon energy | channel | peak height / base |
|---|---|---|---|---|
| P | Si, O | Al Kα | 0.1 eV | about 100 |
| B | Si, O | Al Kα | 0.1 eV | about 0.1 (background-dominated) |
| T | Si, O, C | Al Kα | 0.5 eV | about 10 |
| G | Si, O | Ga Kα (9251.7 eV) | 0.1 eV | about 10 |

Noise levels: r = 0.005, 0.01, 0.02, 0.04, 0.08. For each scenario the
counts are scaled (all channels by one factor, keeping the shape) so
that the median r over 50 draws is within 10 % of the level. This sets
the counts only; R is not computed at this step.

**Runs.** M = 2,000 data sets, B = 1,000 replicates, as in §3, with a new
seed (20261008).

**Decision rule.**

1. Each scenario gets R over all data sets with a standard uncertainty,
   its 95 % Monte Carlo interval and a verdict, as in §3.
2. For each background, r\* is the highest level at which every
   configuration passes at that level and at every lower level. If the
   lowest level does not pass in every configuration, that background
   has no scope.
3. The published guard admits a data set when r ≤ the 95th percentile of
   r over all data sets of the scenarios at level r\* (pooled over
   configurations), and the case is structurally inside what was run:
   at most three lines in the denominator, no transmission divided out,
   a validated background, intensities that are integers (raw counts
   that are not integers are not evaluated).
4. **Published-population check.** For every scenario at levels ≤ r\*, R
   is recomputed over the data sets the guard admits; its interval must
   also lie in [0.9, 1.1]. If any does not, r\* drops one level and steps
   3–4 repeat.
5. Reported per scenario: data sets attempted, with an estimate, with a
   standard uncertainty, admitted by the guard; R over all and over the
   admitted; both intervals; bias against the composition of μ with its
   standard error; the realised median r.

**What this does not cover.** Real detectors (channel correlation), more
than three lines, Tougaard, transmission divided in the resampled
pipeline, other tables or matrices, peak shapes other than a Gaussian.
These stay withheld or not evaluated.

## 7. Second validation: results

From `composition-uncertainty-results-2.json` (runs at commit ead01d4;
summary and decision trace at 59c5332; seed 20261008; M = 2,000, B =
1,000 for each of the 40 scenarios). The realised median r was within
10 % of every level.

**The standard uncertainty is calibrated.** 39 of the 40 scenarios pass
the §3 rule over all their data sets with a standard uncertainty, in
every configuration (peak- and background-dominated, three lines with
0.5 eV channels, Ga Kα) and both backgrounds, up to r = 0.08. The one
failure is peak-dominated, Shirley, r = 0.08: R = 1.24 [1.20, 1.28],
with 301 of 2,000 data sets withheld for refused replicates.

**Admitting data sets by their own r is not.** Step 4 recomputes R over
the data sets the guard would admit. At every level tried, for both
backgrounds, the peak-dominated configuration does not pass it (fail, or
undecided at the lower levels): R over the admitted subset is 1.25
(r\* = 0.08), 1.23, 1.21, 1.11 and 1.07 (r\* = 0.005) with a linear
background, and 1.07–1.25 with Shirley; at r\* = 0.02 with a linear
background the Ga Kα configuration does not pass either (1.07,
undecided). R above 1 means the reported standard uncertainty is larger
than the spread of the estimates it is reported for: the admitted
subset is miscalibrated, conservatively here, not optimistically. In
that configuration r is measured on the same peak as the estimate, so a
cut on r removes the data sets with the smallest areas and narrows the
spread of the estimate the standard uncertainty is compared with. This is
the selection the first audit found for the area cut (§5), on another
quantity. By the rule, r\* falls through every level: **no scope for
either background, and nothing is published in this version.**
The scope is None, `composition()` reports the statistical uncertainty
as withheld with this reason, and the bootstrap stays private. Step 3
puts the threshold at the 95th percentile, so about 5 % of the pooled
data sets at r\* are cut by construction; wherever r follows the
estimated line, step 4 was likely to fail whatever the calibration —
which is the point of having step 4.

**Choices the harness made that §6 did not state** (none changed the
outcome): the scaling step used B = 200 replicates, a starting Si area of
10⁴ counts × eV, the update area × (median r / level)² and at most 8
steps, with draws seeded [20261008, 999999, d]; the 95th percentile is
`numpy.quantile` with linear interpolation, over the data sets that have
an r (those with refused replicates have none — at most one per level
used for a threshold); an admitted subset under 20 data sets would count
as not passing (none was). The record names the summary commit
(59c5332); the runs were at ead01d4, and their per-data-set code is
unchanged between the two.

**What would open a scope.** A guard whose admission does not depend on
the data set's own fluctuations — r of the expected counts supplied by
the user, or a threshold set so far above the operating range that it
admits essentially every data set of a passing scenario — evaluated the
same way, on the admitted population, in a new pre-registered round.
This is not attempted here: changing the rule after these results would
make it post hoc.

## 8. Third validation: admission from declared expected counts (pre-registered)

*Written after §7 and before any of the runs below. Committed before
they start; not changed after. Where this section and the harness
disagree, this section is the registration and the harness is wrong.*

**Aim.** To find out whether a scope for the statistical uncertainty
can be opened with an admission rule that does not depend on the
realised data set at all. §7 named the rule: admission from the
relative area noise of expected counts supplied by the user, evaluated
the same way as before on the admitted population. This round runs
that rule. Nothing in §1–7 changes; this round reuses §2 (noise model
and resampling), §3 (R, its Monte Carlo interval and the acceptance
band [0.9, 1.1]) and the §6 conditions.

**What rounds 1 and 2 established.** On the data sets that returned a
standard uncertainty, the calibration held: S1 and S2 passed, S3 lay
inside the band at R = 1.051 (over the 1,997 of 2,000 without refused
replicates), and 39 of the 40 §6 scenarios passed; the one failure was
P, Shirley, r = 0.08 (R = 1.24, 301 of 2,000 withheld). R has been seen
to rise with r (+5 % at S3, +24 % at that scenario), but not
monotonically (B, linear, by level: about 0.978, 1.020, 0.990, 1.009,
0.996), so monotonicity is not assumed. Admitting data sets by their
own fluctuations broke the calibration twice (§5, §7).

**Framing.** Write T for the estimate, û for the reported standard
uncertainty, and S for the event that a data set returns one (an
estimate exists and a finite standard uncertainty was computed). What
the harness measures, per condition s and declared spectrum D, is

    R_S = sqrt(E[û² | s, D, S]) / sqrt(Var(T | s, D, S))

- An admission decided from s, D, the prediction's own random numbers
  (one fixed seed, below) and the analysis conditions (window,
  background, denominator elements, exposure handling), none of which
  depends on the realised data set, is the same for every data set of
  the condition. Adding it changes nothing in R_S.
- The conditioning on S remains: `composition()` refuses a line whose
  net area is not positive; `_composition_uncertainty` returns
  `not_evaluated` for negative or non-integer counts and `withheld`
  when any replicate is refused. A small rate of these does not bound
  their effect on the variance, so the pass criterion is R_S itself,
  the published claim is conditional on S ("calibrated when a standard
  uncertainty is returned"), the rate is an auxiliary condition, and
  the size of the remaining selection is recorded, not corrected. A
  failure of R_S over ③ is a failure: it cannot be rescued afterwards
  as a matter of the claim's scope ("the selection did it").
- What is checked is the agreement, per condition, between the
  root-mean-square reported standard uncertainty and the standard
  deviation of the estimate. Not checked: that each data set's own
  standard uncertainty is accurate, that the estimate is unbiased, or
  that any interval covers.
- Two questions: (1) with no added selection, up to which predicted r
  does R_S stay in the band; (2) when the user's prediction is off,
  does the realised r of an admitted condition still lie in that range.
  The direction that matters: a prediction that overstates the peak or
  understates the base predicts a smaller r than the true one (with a
  linear background, r_pred / r_true ≈ sqrt(κ_p V_p + κ_b V_b) /
  (κ_p sqrt(V_p + V_b)) for peak and base variances V_p, V_b; about
  0.354 at (κ_p, κ_b) = (2, 0.5) when the base dominates, 1/sqrt(2)
  when both are doubled). This is an explanation of direction only;
  the decision uses the prediction computed through the estimator.

**Declared input.** A declared spectrum D is, per line, the expected
counts on that line's energy grid (base included): a non-negative,
finite, real-valued array that is not rounded to integers. D carries
the same fields as the data's `Line` except `intensity` — `element`,
`orbital`, the energy grid, `energy_scale`, `window`, `binding_energy`,
`exposure_s`, `exposure_basis`, `transmission`, `source` — and these, with the `Conditions`
(background, table, matrix, declared and denominator elements,
intensity semantics, transmission state), are the analysis conditions,
fixed before the data set is looked at. The provenance of D is
declared as one of:

- `planning_model`: a pre-measurement estimate (exposure, count rate,
  assumed composition);
- `other_measurement`: another data set that shares nothing with the
  one evaluated;
- `this_data`: anything derived from the data set being evaluated (a
  fit, a smoothing, its areas). **Refused.**

With `planning_model` or `other_measurement` the user also declares
that D and the analysis conditions were fixed before the data set was
looked at. Without a declared D the uncertainty is withheld, as now.
The package cannot tell whether D came from the same data; no partial
detection is attempted. In this round the declared arrays, conditions,
provenance and prediction settings are stored in the results record;
whether and how they go into `/provenance` (HDF5), and the vocabulary
there, is decided in the publication design after the round.

**The prediction, `predict_r(D, conditions)`.** One function, used for
the count setting and for admission alike; no other seed or B is used
for either.

- Checks that every array of D is non-negative, finite and real; in
  this round a D that fails the check is a run error that stops the
  run (the synthetic D_κ cannot fail it). What a public function
  returns for such a D is decided in the publication design.
- Computes `composition(D)` and records whether it is refused.
- Passes D unchanged (not integerised) as the Poisson means to
  `_bootstrap_detail` with B_pred = 1,000 replicates, from a generator
  freshly seeded with series 2 (below) at every call — a key that does
  not depend on the condition, level, κ or run kind — so the same D and
  conditions always give the same r_pred and the same refusal counts,
  whatever was computed before. The integer check that
  `_composition_uncertainty` applies to observed data is not applied
  to D.
- r is as in §6: over the lines in the denominator, the largest value
  of (standard deviation over the replicates of the line's net area,
  ddof = 1) / (the net area of D itself on the same corrected scale —
  what §6 calls the raw net area; the ratio does not change with the
  constant exposure factor).
- Returns r_pred, whether `composition(D)` was refused, and the number
  of refused replicates.

**Admission.** A case is admitted when all four hold:

1. r_pred ≤ r\*_bg, the threshold of its background (decided below);
2. `composition(D)` is not refused and the prediction has 0 refused
   replicates;
3. the structural conditions of §6 rule 3 — at most three lines in the
   denominator, no transmission divided out, a validated background,
   integer intensities in the data — and at least two elements in the
   denominator (with one, the fraction is always 1 and R is undefined);
4. the case lies within the registered synthetic conditions (see
   "Publication").

On the data side the bootstrap uses B = 1,000, and one refused
replicate withholds the result, as now. Other B, or re-drawing with
another seed until it succeeds, are outside what is validated.

**Conditions.** The 40 of §6 unchanged: configurations P, B, T, G ×
backgrounds linear, Shirley × levels 0.005, 0.01, 0.02, 0.04, 0.08,
with the same lines, photon energies, channel widths, peak-height-to-
base ratios, Gaussian σ = 0.6 eV, window ±10 eV, Scofield, SiO₂
matrix, exposure 1 s, and no transmission division. μ(s) is the
expected counts of condition s; data sets are always drawn from μ(s).

**Levels and the count setting.** A level is the target for r_pred of
the declared spectrum with κ = (1, 1), i.e. D = μ(s). The median
realised r over data sets is recorded only as bookkeeping. The counts
of a condition are set by scaling every channel by one factor; the
candidate is the Si 2p area A, and "one step" is the evaluation of one
candidate:

1. A₁ = 10⁴ counts × eV.
2. For candidate A_k, run the exploratory bootstrap (B = 200, series 1)
   on μ(A_k). If `composition(μ)` is refused or any replicate is
   refused, A_{k+1} = 4 A_k. Else if the exploratory r is outside
   [0.9 × level, level], A_{k+1} = A_k × (r / (0.95 × level))².
3. Else evaluate the same A_k with `predict_r` (B = 1,000, series 2).
   If it reports a refusal, A_{k+1} = 4 A_k. Else if r_pred is outside
   [0.9 × level, level], A_{k+1} = A_k × (r_pred / (0.95 × level))².
   Else A_k is final.
4. If no candidate is final after 8 steps, the condition's count
   setting **has not been established**: the run stops and reports
   it, as a record distinct from a calibration failure or an empty
   scope. The condition is not dropped; changing anything here is a
   revision of this registration.

The acceptance window is one-sided so that the κ = (1, 1) spectrum of
every established condition satisfies r_pred ≤ level and is admitted
at its own level; the update aims at the middle of the window. The
final A, every candidate, its exploratory r, its r_pred and the
refusals are recorded.

**Prediction error, κ.** D_κ(s) = κ_p × (peak part of μ(s)) + κ_b ×
(base part of μ(s)), with (κ_p, κ_b) ∈ {0.5, 1, 2}², nine points;
(1, 1) is μ(s) itself. Data are always drawn from μ(s); κ changes only
what `predict_r` is given, so the outer runs do not depend on κ and
the cost is that of round 2. The nine points are fixed sensitivity
points, not a guarantee over the interval between them or over any
user's actual prediction error.

**Reference value.** As in §3 and §6, bias is against `composition(μ)`,
the estimator applied to the expected counts — not a physical truth.
The nominal-versus-true IMFP is not varied.

**Shirley.** The estimator as it is, with `max_iter = 50`, `tol = 1e-5`
and `auto_range = False` (the implementation's defaults, checked
before registration). Refusal counts count `composition()`'s refusals,
not whether the Shirley iteration converged.

**Runs.** A pilot with M = 200 and B = 200 checks cost and Monte Carlo
error only. The decision uses M = 2,000 and B = 1,000.

**Decision rule.**

1. *Per condition.* Over the M structurally eligible outer trials,
   each trial is in exactly one of: ① no estimate (`composition()`
   refused); ② estimate, no standard uncertainty (`not_evaluated`,
   `withheld` for refused replicates, or a non-finite standard
   uncertainty); ③ a finite standard uncertainty. Counts per cause are
   kept. A missing file or an exception is a run error that stops the
   run; it is never counted in ① or ②. R_S, its 95 % Monte Carlo
   interval and the §3 verdict are computed over ③. A condition with
   at most one trial in ③, a zero standard deviation of the estimate,
   or a non-finite R or interval is **not evaluable and does not
   pass**; no other minimum count is imposed. The standard deviation
   and bias of the estimate are recorded over ② ∪ ③ and over ③ alone,
   and not used in the decision.
2. *A condition passes* when R_S passes the §3 verdict (the main
   condition) and (|①| + |②|) / M ≤ 0.1 % (the auxiliary rate
   condition). The counts, the denominator and the exact one-sided
   95 % upper binomial bound on the underlying rate are reported with
   it; with M = 2,000 and 0 events that bound is 0.1497 %, so a pass
   does not show the rate to be at most 0.1 %, and the record says so.
   Under the rate condition |①| + |②| ≤ 2, so |③| ≥ 1,998.
3. *Admission sets.* The candidates for r\* are, from the largest,
   {0.02, 0.01, 0.005}; 0.08 and 0.04 are not candidates. 0.04 was
   removed before any run, from the design's logic alone: in
   configuration B the level 0.08 has r_pred ≈ 0.028 at κ = (2, 0.5)
   and so enters A_bg(0.04), and since no level lies above 0.08, rule 4
   can never let r\* = 0.04 stand. For background
   bg and candidate r\*, A_bg(r\*) is the set of conditions s with
   that background for which at least one of the nine κ satisfies
   admission conditions 1 and 2.
4. *A candidate stands* when all four hold: A_bg(r\*) is not empty;
   every configuration P, B, T, G has at least one level in A_bg(r\*);
   every condition in A_bg(r\*) passes rule 2; and for every
   configuration, the level one above its highest level in A_bg(r\*)
   also passes rule 2 — if that highest level is 0.08 there is no
   level above, and the candidate does not stand.
5. *The scope* of a background is the largest standing candidate; if
   none stands, that background has no scope.
6. *Implementation check* (the counterpart of §6 rule 4). For every
   condition and every admitted κ, the set of trial IDs that the
   admission would publish and their standard uncertainties are the
   same as rule 1's ③, and R's point estimate is the same; the Monte
   Carlo interval is computed once in rule 1 and never recomputed. A
   mismatch is an implementation error and stops the run. This checks
   that no further selection crept in; it does not re-examine the
   selection by S.

**Harness choices, registered here.**

- *Numbering.* Configuration P = 0, B = 1, T = 2, G = 3; background
  linear = 0, Shirley = 1; level 0.005 = 0, 0.01 = 1, 0.02 = 2,
  0.04 = 3, 0.08 = 4. Condition ID = 10 × configuration + 5 ×
  background + level (0–39). Run kind pilot = 0, decision = 1. Element
  order is the line order of the condition (Si, O, C). Candidate
  numbers k count from 1 (A₁ is candidate 1); outer trial IDs run from
  0 to M − 1.
- *Generator.* `numpy.random.Generator(numpy.random.PCG64(
  numpy.random.SeedSequence(key)))` with an integer-list key; chunk
  boundaries and worker counts never enter a key. Base seed 20261014.
  - series 1, count-setting exploration (B = 200):
    [20261014, 1, condition ID, candidate k];
  - series 2, `predict_r` (B = 1,000): [20261014, 2] — no condition,
    level, κ or run kind;
  - series 3, outer data generation: [20261014, 3, run kind,
    condition ID, trial ID] — no κ;
  - series 4, the data-side bootstrap: [20261014, 4, run kind,
    condition ID, trial ID] — no κ;
  - series 5, the Monte Carlo interval of R: [20261014, 5, run kind,
    condition ID]; the 5,000 resampling index sets are drawn once and
    shared by the elements.
- *Other fixed values.* Data-side B = 1,000; B_pred = 1,000; the κ
  grid {0.5, 1, 2}²; the rate bound 0.1 %; the candidate list {0.02,
  0.01, 0.005}; 5,000 Monte Carlo resamples; sample standard
  deviations with ddof = 1 throughout (as the §3 and §6 harness
  computes them, and in r).
- *Execution identity.* The runs are made from a fixed worktree. At
  start-up the harness checks that `toyomacro` was imported from that
  checkout and that the checkout is clean — no tracked file modified;
  all outputs (chunks, metadata, scales, the record) are written
  outside the checkout, or under a path git ignores, so they never
  make it dirty — and stops otherwise. It records, at run time, a hash
  of every item of this subsection, the implementation commit, the
  Python, numpy and scipy versions and the generator's name. A resume
  whose hash, commit or versions differ is refused. Before summarising,
  every planned trial ID must be present exactly once. A chunk is
  written to a temporary name and renamed when complete; only renamed
  chunks count. The pilot and the decision run have separate output
  directories.
- *Procedure.* The run writes its record outside the checkout (above);
  the summary is copied into `docs/design/` as
  `composition-uncertainty-results-3.json` when it is committed;
  shells use `set -o pipefail`; the first chunk's existence is checked
  right after launch (round 2's first launch was a no-op).
- *Not used.* Round 2's 95th percentile and its minimum admitted count:
  the decision is per condition.

**Reported per condition** (§6 rule 5 continued): the counts of ①, ②,
③ with causes, the rate (|①| + |②|) / M and its one-sided 95 % upper
bound; standard deviation and bias over ② ∪ ③ and over ③; r_pred for
each κ and whether the condition is admitted under it; the prediction's
refusal of `composition(D)` and its refused-replicate count; the count
setting's trace (each candidate's area, exploratory r, r_pred,
refusals, and the final candidate number); the median realised r
(bookkeeping only); the result of rule 6; the configuration hash, the
commit and the versions.

**Publication, if a scope opens.** The claim is "calibrated, when a
standard uncertainty is returned, within the registered synthetic
conditions in which it held" and nothing wider. The scope is recorded
as the pairs (registered condition, κ): the target level set at
κ = (1, 1); the μ actually established at that level and its r_pred
(inside [0.9 × level, level]); the nine D_κ and their r_pred; and what
is outside — unregistered μ, shapes, line sets, tables, matrices and
photon energies. No single number (0.005, say) decides whether a real
data set is inside: the κ = (1, 1) point carries a −10 % allowance,
and at κ ≠ (1, 1) r_pred can fall below the level. The 40 conditions,
the nine κ and the candidate list are fixed here; a later publication
design that allows use outside them does not widen what this round
established. Acceptance conditions split into those the code checks
(denominator of two or three elements, background type, transmission
not divided, intensity semantics, B, r_pred and the prediction's
refusals) and those the user declares (the provenance of D and that it
was fixed beforehand, channel independence, peak shape); whether the
table, matrix, photon energies, window and channel width are enforced
or declared is left to that design. Use below the lowest registered
level (more counts) is not validated; if it is published at all it is
labelled as an assumption, "extrapolation to more counts", never as
established, and where the boundary lies is decided in the
publication design from the (condition, κ) record. Between levels
nothing beyond rule 4's bracketing is shown. A scope that opens
adds public API and ships in the next minor version; an empty scope
adds nothing, not even the input.

**What this does not cover.** Unconditional calibration including the
cases that return no standard uncertainty; the accuracy of any single
data set's standard uncertainty, the bias of the estimate, interval
coverage; the nominal-versus-true IMFP; errors in the shape of D (κ
scales amplitudes only); points between or beyond the nine κ, or any
distribution of real prediction errors; analysis conditions chosen
after seeing the data; real detectors (adjacent-channel correlation
+0.16, gain, non-integer intensities); data-side B ≠ 1,000 or re-seeded
retries; Shirley convergence; four or more lines, one element,
Tougaard, transmission division, other tables, matrices, photon
energies or peak shapes; behaviour between levels beyond the
bracketing, and extrapolation to more counts than the lowest level.

## 9. Third validation: results

From `composition-uncertainty-results-3.json`: the decision run and its
summary at commit 063a0c7 (the §8 registration is 51be653; the harness
of #59 and #60), seed 20261014, M = 2,000 data sets and B = 1,000
replicates for each of the 40 conditions, 1 h 50 min of wall time on
one 16-core machine. The run's and the summary's identities
(configuration hash, commit, Python 3.12.12, numpy 2.3.5, scipy 1.17.0,
an x86_64 interpreter, generator) are the same, and rule 6 holds in 40
of 40 conditions. The two `checkout` fields, which held the path of the
worktree, were removed from the committed copy of the record; the
harness no longer records that path. Before the decision run, the
registered pilot (M = 200, B = 200) was run three times, at 3ace680,
467c41d and def5c6b, identical in every number that enters the
decision (3ace680 still reported an r_pred for the four refused κ
points that later versions withhold): cost and Monte Carlo error only
(half-widths 0.084–0.129), no decision. A first launch of
the decision run, from 6cc4e1e, was stopped after about half an hour
(#60): the harness then collected every chunk before writing any, so
the first chunk could not be checked after launch and the run was not
resumable; #60 made the chunks stream, and nothing from that launch is
used.

**Count setting.** All 40 conditions established their counts, in 1 to
8 candidates (mean 3.25; B/Shirley/0.005 needed all 8), with
r_pred(κ = (1, 1)) at 0.904–0.988 of the level, inside the one-sided
window. The realised median r over the data sets was 0.894–1.033 of
the level.

**Rules 1 and 2.** Of 80,000 data sets, 0 were refused by
`composition()`, 144 returned no standard uncertainty (all withheld
for refused replicates, in three conditions) and 79,856 returned one.
Monte Carlo half-widths of R_S are 0.029–0.036 (target 0.05). 38
conditions pass; the two that do not are both at r = 0.08 with a
Shirley background:

| condition | ③ | R_S | 95 % MC interval | rate (① + ②)/M, upper 95 % | result |
|---|---|---|---|---|---|
| P, Shirley, 0.08 | 1,861 | 1.189 | [1.154, 1.226] | 6.95 %, 7.96 % | fails R_S and the rate |
| G, Shirley, 0.08 | 1,997 | 1.075 | [1.041, 1.111] | 0.15 %, 0.39 % | undecided; fails the rate |
| T, Shirley, 0.08 | 1,998 | 1.008 | [0.977, 1.043] | 0.10 %, 0.31 % | passes (rate at the bound) |

The 38 passing conditions have R_S between 0.959 (G/linear/0.01,
[0.931, 0.989]) and 1.052 (P/linear/0.08, [1.023, 1.083]); every one
of their intervals lies inside [0.9, 1.1] (over every element; the
tables give the Si element, and the O and C entries of the T
configuration are in the record). Per condition (configuration /
background / level: R_S [interval]):

| level | P linear | P Shirley | B linear | B Shirley |
|---|---|---|---|---|
| 0.005 | 1.004 [0.975, 1.035] | 1.000 [0.970, 1.032] | 0.993 [0.963, 1.025] | 0.996 [0.966, 1.028] |
| 0.01 | 1.007 [0.979, 1.039] | 0.970 [0.942, 1.000] | 0.983 [0.953, 1.016] | 1.015 [0.984, 1.050] |
| 0.02 | 1.014 [0.983, 1.049] | 1.010 [0.978, 1.043] | 0.984 [0.955, 1.016] | 1.030 [1.000, 1.063] |
| 0.04 | 0.986 [0.954, 1.020] | 1.018 [0.988, 1.053] | 0.992 [0.962, 1.023] | 0.994 [0.965, 1.027] |
| 0.08 | 1.052 [1.023, 1.083] | **1.189 [1.154, 1.226]** | 1.009 [0.980, 1.040] | 1.030 [1.000, 1.062] |

| level | T linear | T Shirley | G linear | G Shirley |
|---|---|---|---|---|
| 0.005 | 1.010 [0.981, 1.042] | 0.995 [0.967, 1.026] | 0.999 [0.969, 1.031] | 0.986 [0.956, 1.018] |
| 0.01 | 1.000 [0.970, 1.032] | 0.975 [0.946, 1.007] | 0.959 [0.931, 0.989] | 1.004 [0.975, 1.034] |
| 0.02 | 1.018 [0.987, 1.053] | 1.005 [0.973, 1.039] | 0.982 [0.953, 1.014] | 0.974 [0.946, 1.004] |
| 0.04 | 0.968 [0.938, 1.001] | 0.984 [0.954, 1.017] | 1.013 [0.982, 1.045] | 0.989 [0.960, 1.020] |
| 0.08 | 0.998 [0.969, 1.030] | 1.008 [0.977, 1.043] | 1.022 [0.993, 1.055] | *1.075 [1.041, 1.111]* |

Bias of the Si fraction against `composition(μ)`, over ③, is within
±1.2 × 10⁻³ in every condition, at most 2.8 standard errors of the
mean; the standard deviation and bias over ② ∪ ③ and over ③ are in the
record.

**Rules 3–5: the scope.** r\* = 0.02 for the linear background and
r\* = 0.01 for Shirley.

- *Linear.* A_linear(0.02) is the 16 conditions at levels 0.005–0.04 of
  all four configurations: the level-0.04 conditions enter through
  κ = (2, 0.5) and (2, 1), whose r_pred is 0.33–0.55 of κ = (1, 1)'s;
  no level-0.08 condition enters at any candidate. Every admitted
  condition passes, and the level above each configuration's highest
  admitted level, 0.08, passes too. r\* = 0.02 stands (so do 0.01 and
  0.005; the largest is taken).
- *Shirley.* A_Shirley(0.02) is likewise 16 conditions, but the level
  above, 0.08, does not pass for P (R_S = 1.189, 139 withheld) or for G
  (undecided, rate 0.15 %), so 0.02 does not stand. A_Shirley(0.01) is
  the 12 conditions at levels 0.005–0.02 (the level-0.02 conditions
  enter through κ = (2, 0.5), and (2, 1) except for P); all pass, and
  the level above, 0.04, passes in every configuration. r\* = 0.01
  stands.

Four κ points were ineligible (admission condition 2) because the
prediction refused 1–8 replicates: P/Shirley/0.08 at κ = (0.5, 1) and
(0.5, 2), T/Shirley/0.08 at (0.5, 2), G/Shirley/0.08 at (0.5, 1); none
of them touches the admission sets above.

**What the scope means.** Admission is by r_pred of the declared
spectrum, and r_pred is sensitive to the declaration in the direction
§8 "Framing" gave: over the 40 conditions, r_pred at κ = (2, 0.5) is
0.33–0.41 of its value at κ = (1, 1), at (2, 1) 0.44–0.55, at (2, 2)
0.68–0.75, and at (0.5, 2) 2.5–3.1 (over the 38 conditions that have
an r_pred there). So a data set admitted at r_pred ≤ r\* whose
declaration overstated the peak twofold and understated the base
twofold has a true r of up to about 3 r\*, between the highest admitted
level (2 r\*) and the level above it (4 r\*); rule 4's bracketing shows
a pass at both of those levels and, as §8 says, nothing between them.
Within the registered conditions, then: with a linear
background and r_pred ≤ 0.02, or a Shirley background and r_pred ≤
0.01, the standard uncertainty returned by the bootstrap matches the
spread of the estimate to within the ±10 % band, on the data sets that
return one, for declarations within the nine κ points. The published
guard admits exactly that; the rest of §8 "Publication" (what the
code checks, what the user declares, the many-counts side as an
extrapolation) is implemented in the publication design that follows.

**Not shown here.** Everything in §8 "What this does not cover". In
particular, the rate of data sets that return no standard uncertainty
is bounded only as the record states (0 of 2,000 gives an upper bound
of 0.15 %, not a demonstration of 0.1 %), T/Shirley/0.08 passes the
rate condition exactly at the bound, and the failing Shirley
conditions at r = 0.08 are the second round's failure again (P) and a
new undecided one (G): the mechanism §6 names — resampling about the
observed counts — is consistent with R_S rising there, but its cause
was not isolated in this round either.

## 10. Publication design: what the code returns, and under what assumption

*Written after §9, before the public function exists. It does not
change §8 or §9; it says how their result reaches a user, and keeps two
things apart that an API can fold together: the conditions under which
the code returns a standard uncertainty, and the conditions under which
its calibration was shown.*

**The two statements.** §9 showed, for the registered (condition, κ)
pairs, that on the data sets that return a standard uncertainty its
root-mean-square agrees with the spread of the estimate to within
±10 %, when admission is by r_pred ≤ r\* (0.02 linear, 0.01 Shirley).
That is a statement about synthetic conditions. A real data set is
never one of them: its true expected counts are unknown, and its
lines, shapes, background shape, relative intensities, exposure and
detector are all outside what was registered. So the public result
carries two fields, never one:

- `status` — what the code did: `evaluated` (a standard uncertainty is
  returned), `withheld` (computable in principle, not returned, with
  the reason) or `not_evaluated` (the noise model does not apply, or
  the composition itself is refused);
- `applicability` — on what ground the calibration is applied. The
  public function always returns `assumed`, with the list of
  assumptions the result rests on: the package cannot tell whether an
  observed data set was drawn from the expected counts of a registered
  condition, and the registered declared spectra are themselves in the
  record, so nothing in a declaration's content can establish
  `registered`. `registered` is set only by a private entry point the
  tests use on data sets they draw from a registered condition.

`conditional` in `CompositionResult` is not reused for this: it says
that some assumption was made, not which. The uncertainty result names
them.

**What the code checks (mechanical).** In this order: the declaration
is validated (an invalid array, an unknown assumption name or a
mismatch with the observed input is a `ValueError`, whatever else
holds); `composition(lines, conditions)` runs, and a refusal gives
`not_evaluated` with its reasons and ends the call; check 1 failing
gives `not_evaluated` and ends the call; checks 2 and 3, then the
required assumptions, then the prediction (check 4's prediction side
and check 5) are evaluated in that order, the first failure giving
`withheld` with its reason and ending the call, so the prediction
runs only when checks 2–3 and the assumptions hold and the data-side
bootstrap runs only when the case is admitted; `prediction` and
`data_side` are None when not run.

1. The intensity is stated to be raw counts (`intensity_semantics ==
   "raw_counts"`); every observed intensity is non-negative, finite
   and integer. (An `assume_intensity` given with stated raw counts is
   an input inconsistency, refused as a `ValueError` at the
   declaration step, not ignored.) Integers
   alone do not make the counts Poisson: the user's declaration (below)
   carries the channel-independence assumption.
2. The table is Scofield and the background linear or Shirley; the
   denominator is `declared_elements` itself (no `normalise_over`), with
   two or three elements, one line each.
3. No transmission correction, before storage or in the calculation.
   In the package's vocabulary `transmission_applied = "applied"` means
   the stored intensity was already divided by a curve, so it is
   refused (the counts are no longer counts); `"not_applied"` makes
   `composition()` divide the curves out, so it is refused too. The
   admitted state is `transmission_applied = "unknown"` with
   `assume_transmission = "equal_across_lines"`: nothing is divided,
   the composition carries that assumption, and the declaration adds
   that the stored counts carry no prior correction (which the package
   cannot verify). The validation's synthetic conditions were labelled
   `"applied"` as a stand-in for "no transmission anywhere"; the
   computation path is the same, and the registered comparison (below)
   compares the path, not the label.
4. B = 1,000 on the data side and B_pred = 1,000 in the prediction,
   the prediction seeded with [20261014, 2] (series 2); no replicate
   refused on either side; the standard uncertainty and covariance
   finite. The data-side bootstrap is seeded with the fixed key
   [20261014, 4]: the same data and declaration always give the same
   result, and there is no argument to re-draw. §8's validation keyed
   the data side per trial; the public key is a convention of this
   section, not of the registration, and the record says which was
   used.
5. r_pred ≤ r\*_bg from `_VALIDATED_SCOPE`. **No quantity computed
   from the observed data enters the admission** — not its areas, not
   its own bootstrap's relative noise. The observed data decide only
   whether an estimate and a standard uncertainty exist.
6. The declared input matches the observed input: the same elements
   and orbitals in the same order; per line, an energy grid equal to
   the observed one element for element (no tolerance: the declaration
   is made on that grid), the same window, binding energy, exposure
   and energy scale; and a `Conditions` equal, as a dataclass, to the
   one the declaration was fixed with — every field, including
   `assume_intensity`, `assume_transmission`, `normalise_over` and
   `exclusion_reason`. A mismatch is a `ValueError`: the declaration
   does not describe this data set.

**What the code checks and records, and the user applies by assumption.**
The matrix, the photon energy, the window, the channel width, the
elements and orbitals and the exposure are read from the inputs and
recorded in the result, against their registered values; the peak
shapes and widths, the background shape and the relative intensities
of the lines are implicit in the arrays and are recorded only as the
user's assumption. None of them is a reason to withhold, and none of
them is covered by §9 except at the registered values. r_pred does not absorb them: it is the relative
noise of the net areas, so a different matrix or photon energy changes
σλ and not r_pred, and a different window or grid changes r_pred
without any evidence that equal r_pred means equal calibration. The
calibration is applied to them under the user's declaration that the
registered result transfers; the result lists each as an assumption.
Likewise the many-counts side: use at r_pred below the lowest level is
an extrapolation that the declaration must state (the code does not
check it, below); r_pred < 0.005 alone
does not detect it (the registered κ = (1, 1) spectra at level 0.005
themselves have r_pred 0.0045–0.0049, and admitted pairs go down to
0.0016), so no numerical lower bound is imposed.

**The declared input.** A new dataclass, `DeclaredSpectrum`, separate
from `Line` (observed facts) and `Conditions` (analysis choices):

- per line: `element`, `orbital`, the energy grid (`energy`,
  `energy_scale`), `window`, `binding_energy`, `exposure_s`,
  `exposure_basis`, and `expected` — the expected counts, base
  included, non-negative finite reals, not rounded; the arrays are
  copied on construction so that later changes to the caller's arrays
  do not reach the declaration;
- the `Conditions` the declaration was fixed with (stored, and
  compared with the ones passed at the call);
- `provenance`: `planning_model` or `other_measurement`, with
  `provenance_basis`, a free-text statement of where the expectation
  came from; `this_data` (anything derived from the data set being
  evaluated: a fit, a smoothing, its areas) is refused with a
  `ValueError`;
- `fixed_before_data: True` — the user's declaration that the expected
  counts **and** the analysis conditions were fixed before the data set
  was looked at; the package cannot verify it and records it;
- `assumptions`: a tuple of named statements from a fixed vocabulary,
  each the user's own: `channel_independence` (the counts are
  independent Poisson variables), `no_prior_correction` (the stored
  counts were not divided, smoothed or rescaled before storage),
  `transfer` (the registered calibration transfers to this case's
  matrix, photon energy, window, channel width, lines, shapes and
  exposure), and `more_counts_extrapolation` (use at more counts than
  the lowest registered level). The first three are required; a
  missing one gives `withheld` with the reason "applicability not
  declared: <name>". The fourth is never required by the code — no
  numerical rule can tell when it applies (above) — it is recorded
  when given, and the user judges from the (condition, κ) record
  whether it applies. An unknown name is a `ValueError`. Free text may
  accompany each.

An invalid array (negative, non-finite, complex, wrong shape, grid
mismatch) is a `ValueError` at construction or at the call: the input
is wrong. A valid declaration whose prediction is refused (the
composition of the declared spectrum is refused, or a prediction
replicate is) gives `withheld` with the reason: the input is right and
the method does not apply.

**The function.**

    composition_uncertainty(lines, conditions, *, declared) -> CompositionUncertainty

No `n_boot`, no seed, no retry: B = 1,000 and the keys above are the
only validated setting. It runs the steps in the order given under
"What the code checks" and returns:

- `composition`: the `CompositionResult` the uncertainty belongs to;
- `status`, `reasons`, `applicability`, `assumptions`;
- `standard_uncertainty` (per element, computed as the harness does:
  the sample standard deviation over the replicates with ddof = 1) and
  `covariance` (element order stated, ddof = 1), or None;
- `prediction`: r_pred, r\* applied, the prediction's refusal flags and
  refused-replicate count (None when the bootstrap did not run), B_pred
  and its key;
- `data_side`: B, refused-replicate count (None when the bootstrap
  did not run), key;
- `declared`: a snapshot of the declaration (arrays, grid, provenance,
  the fixed-before-data statement, the conditions);
- `registered`: which registered values the case matches and which it
  does not (table, background, line set, photon energy, matrix, window,
  channel width), the scope constant used and the record it comes from;
- `versions`: the package, numpy and scipy versions, and an identifier
  of the observed data (the lines' `source` fields).

`composition()` alone still returns no uncertainty; both variants of
its `not_evaluated` note (for stated raw counts, "withheld; the scope
validated on synthetic data is empty in this version", and otherwise)
change to "not computed in this call; `composition_uncertainty` with
a declared spectrum computes it", keeping the clause that the Poisson
model does not apply when the intensity is not stated raw counts.
The private `_composition_uncertainty` of v0.5.x, which admitted by the
observed data's own bootstrap noise when given a scope — the rule §7
rejected — is removed, not renamed.

**The scope constant.** `_VALIDATED_SCOPE` becomes
`{"r_pred_threshold": {"linear": 0.02, "shirley": 0.01}, "max_lines":
3, "min_elements": 2, "record": "composition-uncertainty-results-3.json"}`
and a test ties it to the record's decision, as the v0.5.x tests tied
the empty scope to the second record.

**The implementation check in the public path.** The counterpart of
§8 rule 6, as tests that run in CI:

The tests use an admitted (condition, κ) pair — one whose κ = (1, 1)
r_pred is at or below its background's r\*, e.g. P/linear/0.02 — so
that the public path reaches the data-side bootstrap; they relabel the
harness's conditions to the admitted transmission state (`"unknown"`
with `assume_transmission = "equal_across_lines"`, the same
computation path) and supply the three required assumptions, so the
public function is exercised, not short-circuited at check 3.

- *No observed quantity in the admission.* Over a handful of data sets
  of that condition drawn with the harness (series 3), the prediction part of
  the result (r_pred, the refusal flags, the admission) is identical
  for every data set, and is identical to `_predict_r` called on the
  declaration alone.
- *Same bootstrap as the validation.* The private entry point takes
  the data-side key as a parameter; with §8's per-trial key
  [20261014, 4, 1, condition, trial] it reproduces, for a handful of
  trials of that condition, the harness's `trial()`: the same
  category, the same standard uncertainty bit for bit (both compute
  the sample standard deviation with ddof = 1 over the same
  replicates), the same refused-replicate count. The public function
  has no such parameter; the test is the only place the per-trial
  keys are used.
- *The published set.* With the public key, over a handful of data
  sets of that condition, those that return
  a standard uncertainty are exactly those whose data-side bootstrap
  refuses no replicate, and `status` is `withheld` for the others with
  that reason — the ③ / ② split of §8 under the public key, which can
  differ from the record's split (another key draws other
  replicates), so the record's digest is not compared.
- `applicability` is `registered` only through the private entry
  point and `assumed` from the public one, for the same inputs. The
  private entry point also accepts the harness's `"applied"` label on
  its synthetic conditions (check 3 compares the computation path,
  which is the same); the public function does not.

**What the guide says** (its section on the error bar; the text is
rendered in the guide, not here):

1. Whether a statistical uncertainty is computed is decided from the
   expected counts and the analysis conditions fixed before the data
   set was looked at; the data set's own areas or relative noise never
   re-select it. It is still not returned when the composition cannot
   be estimated, or when any of the 1,000 bootstrap replicates is
   refused.
2. What is returned is the standard uncertainty and covariance of the
   composition estimate under independent Poisson counts. What was
   validated, per registered condition, is the agreement between the
   root-mean-square reported standard uncertainty and the standard
   deviation of the estimate over the data sets that return one. Not
   guaranteed: the accuracy of any one data set's uncertainty, the
   bias of the estimate, the coverage of any interval, the overall
   accuracy of the composition.
3. In the registered sensitivity analysis a declaration with twice the
   peak and half the base gives an r_pred of about 0.33–0.41 of the
   correct declaration's, so the true relative noise of a case
   admitted at the threshold can reach about 3 r\*. That is a ratio
   observed in the registered conditions, not a bound for any
   declaration error; the levels 2 r\* and 4 r\* passed, the range
   between them was not validated.
4. κ is nine discrete points on the peak and base amplitudes. Between
   the points, wrong shapes, different errors per line, and real data
   are separate assumptions. Use at more counts than the lowest
   registered level is an extrapolation too, and r_pred below 0.005
   alone does not detect it.

followed by the guide's existing paragraph on condition dependence
and two added ones, on detector correlation (+0.16 between adjacent
channels on one laboratory detector, §2) and on what the sensitivity
factor leaves out (elastic scattering, angular distribution,
polarisation, geometry — the `not_evaluated` list of
`composition()`). The refusal rate is described as §9 does (a
finite-sample bound), never as "at most 0.1 %".

**Version and record.** A public function and a public dataclass are
API additions: the next minor version, with a `[Unreleased]` entry, in
the Experimental tier with the rest of `toyomacro.composition`. Seed
keys are integer lists of the lengths registered in §8 and here;
`SeedSequence` pads a short key with zeros, so [20261014, 4] is the
same state as [20261014, 4, 0], and no key of another length may be
introduced that could coincide with one in use.
Saving the snapshot to `/provenance` (HDF5) is not part of this
design; the result object is serialisable, which is enough for a
record. §8 "What this does not cover" stands unchanged; the
independent audit AGENTS.md requires for a public-boundary and
meaning-of-a-number change runs after the implementation, against
this section.
