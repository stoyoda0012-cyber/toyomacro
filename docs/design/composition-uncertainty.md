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
- **Scenarios.** 241 channels over 24 eV around each line, window ±10 eV,
  Si 2p and O 1s at Al Kα, Scofield table, SiO₂ matrix, exposure 1 s:

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

To be filled from the committed record after the runs.
