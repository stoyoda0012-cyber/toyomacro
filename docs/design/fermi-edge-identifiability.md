# Identifiability of the resolution and the temperature on a Fermi edge — design record

Dated 2026-09-21. Records why `toyomacro.fitting.fermi_edge_identifiability`
(experimental) is shaped the way it is, which numbers are conventions,
which estimator each standard deviation belongs to, and what is left
open. Every figure below is computed from a model or is a simulation
with a fixed seed; none is a measurement. Its companion for the Voigt
widths is [`voigt-width-identifiability.md`](voigt-width-identifiability.md),
and the two share a private core (`toyomacro._identifiability`).

The question the module answers is narrow and worth stating first: given
a counting spectrum of a metal Fermi edge, how much does it say about
the instrumental Gaussian variance `v = σ²` and the thermal scale
`τ = (kT)²`, once E_F, the amplitude, the density of states and the
background are estimated from the same spectrum?

## 1. Why the two widths are hard to separate

Convolution adds cumulants. The derivative of the Fermi–Dirac occupation
is a logistic density of scale kT, whose second and fourth cumulants are

    κ₂ = (π²/3) τ        κ₄ = (2π⁴/15) τ²

and the Gaussian adds `v` to the second and nothing to the fourth. So
the data see

    κ₂ = v + (π²/3) τ

strongly — it is the width of the edge, and everything measures it —
while **only the fourth cumulant separates `v` from `τ`**, and it
carries no `v` at all. The information available on the separating
direction is therefore of order τ², and vanishes as τ → 0 for that
reason, not because of any numerical difficulty. A test pins the rate
at 1.998–2.000 between 5 K and 2.5 K, over three combinations of DOS
form and coefficients.

Two consequences shape the whole module.

**Use `v` and `τ`, not `σ` and `T`.** In σ and T both first-order
sensitivities vanish at zero (`d/dσ = 2σ d/dv`), which makes a Fisher
matrix singular at a point where nothing physical is degenerate. In
`v` and `τ` they do not. Results are reported in σ, T and FWHM because
that is how a measurement is quoted, but every matrix is built in
`(v, τ)`.

**Split the width, then judge the split.** The module reports the
well-determined combination κ₂ and the share `x_v = v/κ₂` separately.
A spectrum can fix the edge width to a fraction of a percent while
saying nothing about which half of it is instrumental.

## 2. Derivatives at the corner

Every column of the Jacobian is an integral against the Gaussian of a
function that is smooth except at y = 0. Three of them need care.

**The variance column, via the heat equation.** `∂μ/∂v = ½ ∂²μ/∂u²`
keeps the integrand bounded and free of cancellation as v → 0. The
density of states is flat below E_F and polynomial above it, so `h' `
jumps by `c1 f(0) = c1/2` at y = 0 and the second derivative carries a
delta there:

    ∂μ/∂v = ½ [ G_v ⊗ h''  +  (c1/2) G_v(u) ]

Dropping that term is caught by 22 tests. (Counts like this one come
from a deliberate-defect run: the defect was introduced and the suite
run, so it is how many tests *noticed*, not how many are named for
it. Sixteen defects were tried; §7 records the one that survived.)

**Two routes, and where they change over.** A Gauss–Legendre quadrature
(16-point panels, breakpoints at y = 0, graded in kT around it and in σ
around each energy point, cut at 10σ) works everywhere except when kT
is far below σ, where the `d/dτ` integrand is odd about E_F with
magnitude 1/kT and loses about `eps·σ/kT`. A Sommerfeld-type series
covers that limit. Measured against an adaptive quadrature of the same
integrals, over kT/σ from 0.02 to 3, both DOS forms and five sets of
coefficients: the quadrature route agrees to 2.6e-13 and the series to
6e-13 where it is used.

**The series is asymptotic, not convergent.** Its terms shrink only
while n < (σ/kT)². `auto` switches at kT/σ = 0.07, where the two routes
agree to 2e-13 and (σ/kT)² = 204, and `_SERIES_TERMS = 30` stays well
under that; a test asserts the inequality rather than trusting the
comment. Above about kT/σ = 0.15 the series fails outright; a test pins it at
order one by kT/σ = 0.3, where (σ/kT)² = 11 is below the 30 terms
taken. The switch point and the term count are **choices calibrated on
these settings**, not bounds.

At τ = 0 everything is closed form, including
`∂μ/∂τ → A[(π²/6) G_v'(u) − (π²/12) c1 G_v(u)]`. At v = 0 the
convolution is the identity. **`v = τ = 0` is a bare step and nothing
here is defined there** — the module raises rather than returning a
number.

## 3. The Fisher matrix, and what a fixed temperature means

For Poisson counts with mean μ and exposure `t`, `I = t·Jᵀ diag(1/μ) J`.
The background belongs in μ even when it is known, because its shot
noise raises the variance under the edge; leaving it out of the weights
is caught by 21 tests. A test checks the matrix against the Hessian of
the expected deviance, to 1e-7…2.7e-6.

The temperature enters in one of three ways, and the choice changes the
answer more than anything else in the module:

| mode | what it assumes |
|---|---|
| `free` | τ is estimated from the same spectrum |
| `fixed_temperature` | τ is known exactly; its row and column are dropped |
| `temperature_prior` | a normal prior on T, `1/sd_τ²` added to the τ diagonal, with `sd_τ = 2k²T·sd_T` |

`temperature_prior` is a Bayesian-CRB-style addition and nothing more:
no van Trees bound is computed, and the result is not a posterior. Its
two limits are pinned by tests — as `sd_T → 0` it tends to
`fixed_temperature`, as `sd_T → ∞` to `free`, both at the expected rate.

Over the default scan (window ±6√κ₂, DOS rising 30% per edge width,
background estimated, 1000 counts per channel on the plateau) **freeing
the temperature inflates sd(σ)/σ by 1.3× to 78×** depending on
kT/σ and the window, and sd(σ)/σ with T fixed runs from 1.6% to 107%.
At kT/σ = 0.1 the inflation is 70× and the temperature is essentially
undetermined. This is the module's single most practical output: on
most real edges the temperature is not something the spectrum can tell
you, and fitting it anyway spends the resolution's precision on it.

## 4. Effective information, and the scale it is measured in

For one parameter among many, the quantity that matters is the Schur
complement — the information left after the others are profiled out —
not the diagonal element, which assumes everything else known. The two
differ by orders of magnitude here. Substituting one for the other is
caught by 17 tests.

Rank is judged on a **unit-diagonal** matrix against `n·eps`, never on
the raw one: the raw Fisher matrix mixes units (counts, eV, eV²), so
its eigenvalues change with the energy unit while the physics does not.
Un-normalising is caught by 15 tests, the first of them the one that
asserts the scan does not depend on the energy unit.

Reported standard deviations are made dimensionless against the edge's
own width (convention, §5 of the Voigt record has the same one):
positions and widths by √κ₂, variance-type quantities by κ₂. For
display, `width_1090` — the 10–90% width of the edge rebuilt on a flat
DOS — is used instead, because it is what an experimentalist reads off
a plot.

## 5. The labels, and the numbers in them that are conventions

`assess_edge_identifiability` sorts the parameters into
`identified` / `weakly_identified` / `rank_deficient` / `undersampled`,
and the width split into `separable` / `not_separable` / `assumed` /
`undersampled`. Three thresholds are **working rules, not results**:

| threshold | value | meaning |
|---|---|---|
| `weak_relative_sd` | 0.1 | a parameter whose sd exceeds a tenth of its reference scale is weak |
| `boundary_sd` | 3.0 | a width within 3 sd of zero is "near the boundary" |
| `separation_sd` | 0.1 | sd(x_v) above this means the split is `not_separable` |

They were chosen to put the labels where the behaviour changes on the
scan, and a user who disagrees passes their own `EdgeThresholds`. The
0.1 on the share is the one to argue with: it says that knowing the
instrumental fraction of the width to ±10% is the least that counts as
having separated the two.

**The boundary is two-dimensional.** `v = 0` (a purely thermal edge)
and `τ = 0` (a zero-temperature edge) are both edges of the parameter
space, and `v = τ = 0` is the corner where the model does not exist.
The report flags each boundary separately (`near_boundary_v`,
`near_boundary_tau`) rather than collapsing them, and where the split
is `not_separable` **τ's standard deviation is returned as `None`**, not
as a large number: a number there invites a reader to use it. With the
temperature fixed the verdict is `assumed`, which is a statement about
the analysis and not about the data. `undersampled` — fewer than about
two channels across the edge — takes precedence over `rank_deficient`
for the width parameters, because the cure is different.

Near a boundary the inverse Fisher matrix is not the variance of a
constrained estimator (Self & Liang 1987), and the module says so
rather than reporting a bound that does not apply.

## 6. Whose standard deviation is it

This is the question that most often goes wrong, so the result type
names the estimator rather than leaving it to be inferred.

`fitting.fermi_edge` minimises a **weighted sum of squares**, with unit
weights by default. Its covariance is the sandwich
`(JᵀWJ)⁻¹ JᵀW diag(μ) WJ (JᵀWJ)⁻¹`, which equals the inverse Fisher
matrix only when `W = 1/μ`. `InstrumentalResolution` therefore carries
two standard deviations and a label:

- `sd_sigma_bound` — the Cramér–Rao bound. A diagnostic: what any
  unbiased estimator of a correctly specified model could achieve.
- `sd_estimator` — the sandwich covariance of the least-squares
  estimator the package actually ships, for the weights given.

They are not interchangeable. Simulated at 2000 counts per channel over
a background of 50, 200 draws, temperature fixed: the measured spread
of the fitted σ is 1.896e-3 eV, the sandwich says 1.876e-3 (ratio
1.011) and the bound says 1.424e-3 (ratio 1.331). **Quoting the bound
as the error bar of that fit would understate it by a third.** Across a
range of conditions the ratio runs 1.25 to 2.55.

The same distinction reached `fit_fermi_edge` itself: its `*_err`
fields are the least-squares covariance scaled by the reduced
chi-squared, which assumes one variance for every channel. On an edge
the Poisson mean spans a factor of 53 from background to plateau, and
that assumption makes `ef_err` 8 to 21% smaller than the actual scatter
of E_F while leaving the resolution's error within 20%. `poisson_err`
was added as a second field carrying the sandwich; the default was left
alone so that no existing number moves.

**The temperature sensitivity is reported because a thermocouple is not
the electron temperature.** `temperature_sensitivity` is dσ/dT in eV per
K: how far the fitted resolution moves if the temperature held fixed is
wrong by one kelvin. On a 2 eV window in 10 meV steps with 2000 counts
per channel:

| FWHM | T | kT/σ | sd(σ) bound | sandwich (unit w.) | dσ/dT | a 10 K error moves σ by |
|---|---|---|---|---|---|---|
| 300 meV | 300 K | 0.20 | 1.67 µeV·10³ | 2.28 | −57 µeV/K | 0.45% |
| 100 meV | 300 K | 0.61 | 1.35 | 1.75 | −160 µeV/K | 3.8% |
| 50 meV | 300 K | 1.22 | 1.87 | 2.34 | −303 µeV/K | 14% |
| 30 meV | 30 K | 0.20 | 0.41 | 0.55 | −57 µeV/K | 4.5% |

(standard deviations in meV). The last two rows are labelled
`undersampled` on a 10 meV grid, which is the point of that label.

## 7. Resampling: a Monte Carlo and a bootstrap are the same machinery

`toyomacro._bootstrap` draws Poisson counts from a mean and fits them
back. Drawing from a model's mean at stated parameters is the Monte
Carlo check of a bound; drawing from a *fit's* mean is the parametric
bootstrap; drawing from the observed counts is the nonparametric one.
Only the mean differs. What comes back is described as **the
distribution of an estimator** — never as a posterior, since no prior
enters.

**Pass or fail only at an interior point**, where the bound is supposed
to be attained. At exposure 50 and kT/σ = 1, over 400 draws, the
measured spread is 0.96 to 1.03 of the bound for every parameter, with
nothing at a bound (the sampling error of a standard deviation from 400
draws is 3.5%, so the band is ±10%).

Elsewhere the distribution is **recorded, not judged**:

- at `v = 0`, 54% of draws come back with v held at zero — the half-mass
  Self & Liang predict for one parameter at its boundary — and the
  spread of the rest is 3.7 times the bound;
- where the split is `not_separable`, the spread of v and τ is 0.82 and
  0.81 of the bound, because the likelihood is not quadratic there.

**Coverage is a different question from spread**, and A9 did not answer
it. Nesting a bootstrap inside each Monte Carlo replica and asking
whether the 95% percentile interval holds the truth, pooled over three
independent sets of trials at the interior point (1200 parametric, 1000
nonparametric, 200 draws each, standard error 0.7–0.8%):

| | E_F | amplitude | DOS slope | v | τ | background |
|---|---|---|---|---|---|---|
| parametric | 93.6 | 95.1 | 95.1 | 93.6 | 93.6 | 94.2 |
| nonparametric | 92.5 | 94.5 | 93.7 | 93.8 | 93.5 | 95.0 |

Eleven of the twelve are within three standard errors of 95%, and E_F
nonparametric sits exactly on the edge at −3.00. Ten of the twelve are
below 95%, so a deficit of one to two and a half points is real rather
than a property of which trials were drawn — single runs of 400 and 200
gave 91.5% and 95.5% for the same cell, four standard errors apart,
which is why nothing here is quoted from one run. It is **not** the
number of bootstrap draws (holding the trials fixed, 200 → 1000 moves
nothing downward) and not width alone (the bootstrap sd is 0.98 to 1.05
of the Monte Carlo sd). A percentile interval is first-order accurate
and this is the size of error that buys; separating that from any other
cause needs a bias-corrected interval, which is §10.

At the boundary, 98.5% of the intervals for v begin at the bound and are
4.5 times the Cramér–Rao width — coverage bought with width. Where the
split is `not_separable`, v and τ cover 95.4% parametric against 86.2%
nonparametric, the one place the two kinds part company. With 25% of
channels empty the two agree within 3% on every parameter that is
determined.

**Anti-circularity.** The fitter shares its model with what it tests, so
four things are pinned independently: the constrained fit against a
different optimiser, a replica held at a bound against the
Karush–Kuhn–Tucker condition, the drawn counts against the Poisson law,
and the deviance against its own value. That last one exists because of
a defect this record should keep: a deliberate-defect test found a
`poisson_deviance` missing its saturated term — off by a data-dependent
constant, returning negative deviances — passing the entire suite,
*including* the cross-check against an independent optimiser. The
cross-check minimised the same mutated function, so the constant
cancelled on both sides. **A cross-check that shares a part with what
it checks is not independent**, however different the other parts are.
The same run showed the active-set logic untested for a related reason:
the cross-check runs at an interior point, where nothing is held.

## 8. The density of states: two separate faults

The fitted DOS is flat below E_F and polynomial above it, so its slope
changes at E_F — inside the thermal tail, where the temperature
information lives. That kink is an assumption about the sample, and it
is not free.

**What it is worth when it is right.** Over the scan, the kinked form
gives sd(x_v) smaller by 1.6× to 3.0× than the same polynomial
continued smoothly through E_F — a factor of 2.6 to 9 in information —
at the window and background a test pins, and 1.4× to 3.3× across the
wider scan. At zero DOS slope the two forms coincide exactly, which the
same test checks. So
a real part of any apparent separation of the two widths rests on the
DOS having a corner exactly at E_F.

**What it costs when it is wrong.** Simulated edges whose true DOS runs
smoothly through E_F, fitted with the shipped model at its true
temperature (window ±0.6 eV in 10 meV steps, 2000 counts per channel
over a background of 50, κ₂ = (0.1 eV)²). The bias is exact, not a
Monte Carlo estimate: for unit-weight least squares the minimiser of
the expected sum of squares is the fit to the noiseless mean.

| kT/σ | DOS change across the window | σ bias / its error bar |
|---|---|---|
| 0.1 | 0.1 … 0.9 | −0.00 … −0.01 |
| 1.0 | 0.1 / 0.3 / 0.9 | −0.19 / −0.55 / −1.55 |
| 3.0 | 0.1 / 0.3 / 0.9 | −0.30 / −1.09 / −1.67 (on its bound) |

At kT/σ = 0.1 the assumption is free; by kT/σ = 1 it costs more than an
error bar. **Whether the assumption matters is itself set by the
measurement.** At the strongest case the resolution collapses onto its
lower bound and the fit reports itself a failure naming the parameter —
there the success gate, not the number, is what protects a user. With
the temperature fitted instead, the fit buys the missing slope with a
wider Gaussian and a colder sample: σ 18.1 → 40.2 meV and T 629 → 538 K
at kT/σ = 3.

**The kink and the curvature are different axes.** A DOS with smooth
curvature through E_F moved E_F by up to 49 meV — an *energy-axis
calibration* error, not a resolution one, and the same phenomenon as
the 0.05–0.14 eV that `flat`, `linear` and `quadratic` gave on two
measured Au edges. Refitting with `dos='quadratic'`, which keeps the
kink and gains the curvature, removes essentially all of it (E_F to
+0.14 meV). The complement holds: a quadratic term does not recover the
resolution against a DOS with no kink. **Curvature is a DOS-order
problem and belongs to the energy scale; the kink is a DOS-form problem
and belongs to the width split.**

### Why `dos_form` is a comparison and not a choice

The two forms put the same polynomial on different domains, so they can
differ only where the occupation is neither 0 nor 1. The difference
peaks two or three kT below E_F and is 5.6e-4 of the step height at
kT/σ = 0.2 against 1.5e-2 at kT/σ = 1, and is below 2e-4 ten kT out.

That is the conclusion this section exists for: **the region where a
fit can tell the two forms apart is the region where the choice between
them moves the estimates.** Where a spectrum contains enough
information to prefer one form, that same information is what the
choice is spending. Selecting by goodness of fit therefore writes
whichever form won straight into the result as a systematic, invisibly.

So `compare_dos_forms` fits both and returns both, with the spread
between them. It does not average the fits, select one, or rank them.
The spread is **a guide to the systematic the DOS-form assumption
carries**, to be read beside the statistical error and never added to
it in quadrature.

How far the guide can be trusted was measured, not assumed. The bias is
nearly antisymmetric in which form is wrong — a kinked model on a
smooth truth and a smooth model on a kinked truth give −2.068/+2.018,
−6.584/+6.006 and −17.650/+16.280 meV — so the spread between two fits
of *one* spectrum stands in for a bias nobody can measure without
knowing the true DOS. Over six conditions it recovered **63 to 110%** of
it. Two limits on that claim: it is one model family over one range of
conditions, and a linear DOS through E_F reaches zero at the window
edge once its change across the window reaches 1, which capped what
could be tested at half the slope the scan uses.

## 9. The scan

`scan_edge_identifiability` sweeps seven axes — kT/σ, window half-width,
exposure, DOS slope, background treatment, temperature mode, DOS form —
holding κ₂ fixed so that a cell is a measurement condition rather than a
different edge. Its outputs are unit-free by construction and a test
asserts that describing the same spectrum in eV or meV changes nothing.

Two things the scan is not. It is not a recommendation: no cell is
labelled good. And its `temperature_sensitivity` is the dimensionless
`(dσ/dT)·T/σ`, while the report's field of the same name is dσ/dT in
eV/K — the same quantity in two conventions, which is a wart worth
knowing about.

## 10. Scope, and what is open

**In scope.** A Gaussian instrument function, no lifetime broadening, a
low-order polynomial DOS, Poisson counts, one spectrum at a time.

**Known open, in the order they are likely to matter.**

1. **Bias-corrected intervals.** The percentile interval under-covers by
   one to two and a half points at an interior point (§7). The cause is
   consistent with a percentile interval's first-order accuracy, but
   ruling out anything else needs a BCa interval, which is not
   implemented. Deferred past v0.3.0.
2. **`poisson_err` is an additional field, not a replacement.** The
   default `*_err` was left exactly as it was so that no released
   number moves; whether the sandwich should become the default is a
   later decision.
3. **Lifetime broadening.** A Lorentzian component would add a third
   width to the same κ₂ and is not modelled. On a clean metal edge it is
   small; this is not checked here.
4. **Real data.** Everything above is simulation. The bound, the
   sandwich, the coverage and the DOS biases have not been measured
   against a spectrometer whose resolution is independently known.
5. **The series term count and switch point** (§2) are calibrated on the
   settings tested, not derived.

**Not in scope, deliberately.** Sweep-level resampling, block bootstraps
over a map, prior weighting of the resampled draws, and gain checks on
real data were all considered for this release and excluded.

## References

- S. G. Self and K.-Y. Liang, *Asymptotic properties of maximum likelihood
  estimators and likelihood ratio tests under nonstandard conditions*,
  J. Am. Stat. Assoc. **82**, 605 (1987) —
  [doi:10.1080/01621459.1987.10478472](https://doi.org/10.1080/01621459.1987.10478472)
- B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*,
  Chapman & Hall (1993) — percentile intervals and their accuracy
- H. L. Van Trees, *Detection, Estimation, and Modulation Theory, Part I*,
  Wiley (1968) — the Bayesian Cramér–Rao form §3 borrows from without
  computing the bound itself
