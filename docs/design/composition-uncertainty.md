# Statistical uncertainty of a composition — design record

Dated 2026-10-07. Records what `toyomacro.composition.composition_uncertainty`
(experimental) computes, under which noise model, and how it was checked
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

**Scope adopted.** §4 says "a background type whose scenario passed,
and every line's raw net area at or above the smallest area of a
passing scenario". Read across backgrounds, that would let a linear
background through at 3,000 although linear passed only at 30,000. The
module reads it per background, the narrower reading: linear at
≥ 30,000 counts × eV, Shirley at ≥ 3,000. This is a necessary guard and
not a sufficient one: the scenarios also fix the peak shape, the window,
the channel spacing, the base level and two lines at Al Kα, none of
which the check sees. Outside these synthetic conditions the reported
value is the same calculation without a validation behind it.

**Not validated.** Real detectors (see §2 on channel correlation); more
than two elements; other photon energies, tables and matrices; Tougaard;
transmission division in the resampled pipeline; and whether the
estimate is right — this checks only that the reported spread matches
the spread of the estimator.
