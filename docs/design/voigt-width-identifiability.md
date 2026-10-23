# Identifiability of the Voigt widths — design record

Dated 2026-09-19. Records why `toyomacro.voigtfit.identifiability`
(experimental) is shaped the way it is, which numbers are conventions,
and what is left open. Every figure below is computed from a model or is
a simulation with a fixed seed; none is a measurement.

## 1. Five difficulties that are usually blamed on each other

Separating the Gaussian width σ from the Lorentzian half-width γ of a
Voigt profile is hard for reasons of different kinds. The module exists
to show them separately.

1. **A degenerate coordinate.** The profile is a Lorentzian convolved
   with a Gaussian, so it depends on σ only through the variance
   v = σ² and obeys ∂V/∂v = ½ ∂²V/∂x². In the σ coordinate the
   first-order sensitivity ∂V/∂σ = 2σ ∂V/∂v vanishes at σ = 0 and
   I_σσ ∝ σ². Taking v as the coordinate removes this completely: at
   v = 0 the Fisher matrix in (A, c, v, γ) is positive definite.
2. **Precision of a small component.** sd(σ)/σ = ½ sd(v)/v. A finite
   bound on v is still an unbounded *relative* error on a vanishing
   Gaussian component. No re-parameterisation removes this.
3. **A boundary.** v = 0 (or v = σ_inst² when the instrument width is
   calibrated) is the edge of the parameter space. A finite Fisher
   matrix there does not make the inverse Fisher matrix the variance of
   a constrained estimator (Self & Liang 1987).
4. **A finite window.** The Lorentzian width lives in the tails. With no
   background at all, the condition number of the unit-diagonal Fisher
   matrix at σ/γ = 1/30 is 6.18 on ±33.3γ, 9.4×10² on ±γ and 5.5×10⁶ on
   ±0.3γ. The last two are limits of fine sampling; 21 points across the
   window give 7.2×10² and 4.0×10⁶, lower by 23 % and 27 %.
5. **A background.** Its shot noise raises the mean under the peak; if
   it also has to be estimated from the same spectrum it competes with
   the widths for the same counts, and that second effect does not need
   a large background. At ±10γ, σ/γ = 1/3, a flat background of 0.05 %
   of the peak height leaves the least determined width direction 99 %
   of its information when known and 61 % when estimated; at 10 % of
   the peak height the figures are 57 % and 36 %. The direction is
   taken in the scaled coordinates of §5. The smallest eigenvalue of
   the raw (eV², eV) block is not a physical quantity — for the first
   case it gives 43 %, 63 % or 41 % according to whether the same
   spectrum is described in eV, meV or keV — and the test suite pins
   the share as unit-free.

Items 4 and 5 are treated as *practical weak identification*: more
counts or a wider window change them. They are not structural
non-identifiability and the labels below never claim so.

## 2. Derivatives that survive σ → 0

The engine's Faddeeva-based Jacobian cannot be used here. In w″(z) a
term of order 4z cancels down to 2/z³, so the relative error of the
variance derivative grows like |z|⁴ with |z| = |x + iγ|/(σ√2). The wings
go first and the Fisher matrix follows: on a ±33γ window the variance
element built from `voigt_with_hessian` (or from `voigt_with_jacobian`
by the chain rule) is off by 1.5×10⁻³ at σ/γ = 10⁻³, by a factor of about
6 at 3×10⁻⁴ (5.2 to 6.5 depending on the grid) and by four orders of
magnitude at 10⁻⁴.

`voigt_derivatives` therefore switches *per energy point*: for |z| ≥ 7 it
sums the large-|z| expansion

    V = L + (v/2) L″ + (v²/8) L⁗ + …

to 24 terms in closed form from powers of 1/(x − iγ) — no division by v,
exact at v = 0 — and uses the Faddeeva closed form below that. Two
decisions in there are not obvious.

**Second order is not enough.** Truncating after (v²/8)L⁗ leaves 4.5×10⁻⁷
in ∂V/∂v at σ/γ = 0.01 although the profile itself is good to 10⁻¹¹; the
switch was chosen on the derivatives and on the Fisher matrix, not on the
profile.

**The expansion is asymptotic.** Successive terms are in the ratio
(2n+1)/(2|z|²): they shrink while n < |z|² and grow without limit after
it, and the smallest term, of order exp(−|z|²), is the best accuracy
available at that |z|. Two separate constraints follow:

- N < |z_c|² at the switch |z_c| (24 < 49), so every term summed is on
  the shrinking side wherever the expansion is used;
- the switch cannot be lowered to buy back the Faddeeva error: on ∂V/∂v,
  scanning every N from 2 to 70 at σ/γ = 1, the best reached is 1×10⁻⁸
  at |z| = 5 (N = 25) and 3×10⁻⁵ at |z| = 4 (N = 12), and past the
  turning point more terms make it worse (N = 40 at |z| = 4: 3×10²).

Which test catches which violation (by mutation):

| change | constraint | caught by |
|---|---|---|
| switch 7 → 4 | broken (24 > 16) | 4 accuracy tests + the assertion |
| switch 7 → 4.9 | holds (24 < 24.01) | 3 accuracy tests — the floor exp(−24) fails, not the term count |
| terms 24 → 60 | broken (60 > 49) | the assertion only |
| terms 24 → 110 | broken | 9 accuracy tests + the assertion |

The third row is why N < |z_c|² is asserted on the constants: at |z| = 7
the terms near n = 49 are of order exp(−49) and the measured error stays
at rounding level for every N from 24 to 60.

**Measured accuracy**, against a trapezoid quadrature of the convolution
integral (an independent route: no Faddeeva function, no expansion), as
the largest error in a band of |z| over the largest magnitude in that
band, worst over 31 shapes σ/γ = 0.01…10 on ±33.3γ: value 6×10⁻¹⁴,
∂/∂v 3×10⁻¹¹, ∂/∂γ 5×10⁻¹², ∂/∂c 7×10⁻¹³. The worst case sits just below
the switch on the Faddeeva side, which is why the switch is the lowest
|z| at which the expansion is at rounding level. These are band maxima
over a population. A single point per |z| is not a substitute: random
points in 6 ≤ |z| < 7 have a median error 25 times below the band
maximum. The test suite asserts the population figure at 10⁻¹⁰.

NumPy float64 is the reference implementation and the only one. The MLX
Jacobian in the engine covers σ/γ ≥ 0.71 only and is not used here.

## 3. The Fisher matrix

μ(E) = exposure × (Σ_k A_k V_k(E) + background(E)), I = Jᵀ diag(1/μ) J.
`amplitude × V(E)` is counts per channel at unit exposure; the matrix is
exactly linear in `exposure`. A channel with no expected counts raises
instead of being floored: the information there is undefined, not large.

The background is linear in its coefficients and each term is either
known (it enters μ only) or estimated (it is a parameter): constant,
linear, or a Shirley-type step. The Shirley step has a *fixed shape*, the
running integral of the peaks at the stated parameters; its dependence
on the peak parameters is not differentiated, which understates the
coupling of an iteratively determined Shirley background to the widths.

Width coordinates `sigma_gamma`, `var_gamma`, `fwhm_shape` (the
Olivero–Longbothum total width and the Lorentzian share of it) and
`fixed_instrument` with a floor are coordinates on one statistical model,
related by I_φ = Tᵀ I_θ T; T is singular only for `sigma_gamma`, and only
at σ = 0. `fixed_instrument` without a floor is a sub-model of it, one
parameter per peak fewer. `pvoigt` is a
different lineshape model evaluated at the Thompson–Cox–Hastings point,
not a re-parameterisation. `fwhm_shape` is itself poorly conditioned at
the Lorentzian end: v is encoded in the last digits of the share.

`fixed_instrument` with `variance_floor = σ_inst²` estimates the excess
v − σ_inst². In XPS the instrument resolution is usually known and
whether the sample broadens beyond it is the actual question. The matrix
equals the `var_gamma` one; what moves is the boundary.

**Position is orthogonal to the rest only if the mean is even.** A
symmetric window is not enough: the weights 1/μ must be even too. The
center is odd; amplitude, widths and a flat level are even; a background
*slope* is odd like the center. With a true slope of zero the center
couples to the slope and to nothing else. With a non-zero true slope the
center recouples to every even parameter: in the pinned case its
correlation is 4×10⁻⁴ with the background level, 2×10⁻⁴ with γ, 1×10⁻⁴
with the amplitude and 5×10⁻⁵ with v.

## 4. Effective versus conditional information

For widths u and everything else q (amplitude, position, background),

    I_eff = I_uu − I_uq I_qq⁺ I_qu

is what is left for the widths once q is estimated from the same
spectrum; its inverse is the width block of the full inverse. The plain
sub-block I_uu is the information *if q were known* — what
`fisher_information.analyze_sigma_gamma_axes` reports. I_uu − I_eff is
positive semi-definite, so the sub-block is always the optimistic one,
and the gap is large exactly where it matters: at ±γ, σ/γ = 1/3, with
an estimated flat background at a tenth of the peak height, the bound on
γ from I_eff is about 3×10² times the one from I_uu: 318 on 61 points,
338 with fine sampling. The ratio does not depend on the exposure; it
does depend on the shape (94 at σ/γ = 0.1, 6×10³ at σ/γ = 1, same
window). The bounds themselves scale with the counts: 0.24 and 7×10⁻⁴ of
the FWHM for 5.5×10⁶ counts in the window, 0.75 and 2×10⁻³ for a tenth
of that. Both are reported side by side.

The nuisance block is inverted on the unit-diagonal matrix with the n·ε
rank rule, the gauge `crlb.compute_multipeak_fisher` already fixes: the
Fisher matrix mixes counts, eV and eV², so a rank decision on the raw
matrix is a statement about units.

## 5. The labels

Judged in `var_gamma`, from the full inverse, so item 1 of §1 cannot
raise a flag.

- `rank_deficient` — the unit-diagonal matrix is numerically singular
  and this parameter's axis projects onto its null space. The same
  decision as `crlb.compute_multipeak_fisher`: its n·ε rank rule,
  re-implemented here in one place, and its projection tolerance,
  imported.
- `weakly_identified` — the bound exceeds a fraction of a reference
  scale.
- `identified` — neither.
- `near_boundary` — an independent flag: the variance lies within k
  bounds of its lower limit. `near_boundary_gamma` is the same for γ
  at 0 (§9.2, v0.7.0).

**Reference scales**: |A| for an amplitude; the total FWHM for the center
and for γ; FWHM²/(8 ln 2) for v. Not the parameter's own value, which is
undefined at v = 0 and, for a position, not invariant under a shift of
the energy origin. The variance is reported against itself as well
(`variance_relative_sd`), because the FWHM scale stays finite as the
Gaussian component vanishes: at σ/γ = 0.03 (±10γ, 1.6×10⁷ counts in
the window) the width block is `identified` at 2×10⁻³ while
sd(v)/v = 1.6. With more counts sd(v)/v falls like any other bound; the
point is that the FWHM-scaled number says nothing about it. The test
suite checks that
the same spectrum described in meV instead of eV gives the same labels.
Background coefficients have no natural scale and are `not_assessed`
unless rank deficient.

**Defaults, and what they are.**

- `weak_relative_sd = 0.1`: a tenth of the total FWHM. A working rule of
  thumb for when a width has stopped being a usable number. It has no
  physical necessity; change it to suit the question.
- `boundary_sd = 3`: for an unconstrained, normally distributed
  estimator the chance of falling below the limit would be 0.13 % at 3
  and 2.3 % at 2. But that normal approximation is exactly what fails
  near a boundary, so 3 is a margin of caution, not a probability. The
  simulation in §6 does not contradict it — the Fisher number is not the
  estimator's spread at 0 and 1 bounds from the floor and is at 2 and 4
  — but it has no row at 3 and is one configuration.

**What the labels are not.** `identified` on the FWHM scale is not
detection of a small Gaussian component, and the same holds at the other
end: γ = 0 is a boundary too, γ is judged against the FWHM, and a nearly
Gaussian peak can have γ `identified` with sd(γ)/γ of order one (0.58 at
σ/γ = 100 for the default scan peak on ±5 FWHM). Since v0.7.0
`gamma_relative_sd` and `near_boundary_gamma` report it (§9.2); before,
no flag was raised for that boundary. `weakly_identified` is not
structural non-identifiability — the tests show more exposure lifting
it. Near the boundary the reported bounds are still the Fisher numbers;
they are not corrected. Nothing here is a statement about a fit.

## 6. Monte Carlo (simulation)

The check uses a test-side fitter, not shipped, that minimises the
*exact* Poisson deviance under v ≥ floor: Fisher scoring, whose fixed
points are the zeros of the exact score; steps accepted on the exact
deviance; convergence on the exact score; the constraint as an active
set. It is cross-checked against L-BFGS-B on the same deviance. A
weighted least-squares fit would have a different fixed point; swapping
in Neyman weights fails every test.

All of it is one configuration: a single peak, γ = 0.3 eV on ±3 eV at a
0.02 eV step, 2×10⁴ counts·eV of area on a flat background of 300 counts
per channel. It checks the module against an estimator there; it is not
a statement about other windows, intensities or several peaks.

Interior, pass/fail: σ/γ = 1, estimated flat background, 10⁴ replicas, a
regime the module itself labels identified and away from the boundary.
Replica sd over Fisher sd 1.002…1.010 with the recorded seed; other
seeds deviate by up to 1.7 %, the sampling error being 0.7 %.
Correlations within 0.006, bias under 0.01 sd. Tolerance 4 %: a
Fisher matrix wrong by 10 % in either direction moves the ratio by 5 %.

Near the boundary, *recorded, not judged* (4000 replicas per row):

| true v, in Fisher sd | near_boundary | share held at the floor | replica sd / Fisher sd (v) |
|---|---|---|---|
| 0 | yes | 0.4975 | 0.579 |
| 1 | yes | 0.1623 | 0.863 |
| 2 | yes | 0.0260 | 1.007 |
| 4 | no | 0.0000 | 0.986 |

For orientation, not as a criterion: with one parameter on the boundary
and the rest interior the asymptotic law is half a point mass at the
floor and half a half-normal (Self & Liang 1987), giving ½ and
√(½ − 1/(2π)) = 0.584 in the first row. With the floor at a calibrated instrument variance and no
sample broadening the share at the floor is 0.4988.

## 7. The scan

`scan_identifiability` evaluates window half-width, intensity,
σ/γ at fixed total FWHM, background (none / known / estimated) and the
spacing of a second peak, with lengths in units of the FWHM. Two
normalisations: *exposure* keeps the energy step and the exposure per
point fixed, so a wider window has more points and more counts, the way
a measurement trades; *total counts* gives every window the same total
and isolates the shape of the window. Under the second, the bound on γ
has a minimum near ±7 FWHM for the example configuration and rises again
beyond it, because a wider window spends its counts on background.

Every cell is evaluated in `var_gamma` and again in `sigma_gamma`.
Towards σ = 0 the variance bound does not move (it changes by under
0.1 % from σ/γ = 10⁻² down to 0, at 0.009 of its scale for the example's
counts) while the σ bound grows as 1/σ and the information
on σ falls as σ²; towards a narrow window both grow. That is how item 1
of §1 is told from items 4 and 5. `examples/07_width_identifiability_map.py`
draws the maps; its output is not tracked.

A pair of peaks is given a window wider by its separation, so at narrow
half-widths a pair can come out better than a single peak in the same
cell; compare across separations at wide windows only.

## 8. Scope, and what is open

Out of scope by decision: changing any solver's default or any existing
public API; an MLX path; exporting a covariance for downstream weighting;
singular learning theory (real log canonical threshold, WBIC).

Open questions:

- Bounds near the boundary are reported uncorrected. A chi-bar-squared
  style treatment would give calibrated statements there; whether it
  belongs in this module is undecided.
- The Shirley step ignores its own dependence on the peak parameters
  (§3). How much that understates the loss has not been measured.
- γ = 0 is a boundary as well. *Closed in v0.7.0 (§9.2)*: it gets the
  same flag and relative-precision field as the variance.
- The Doniach–Šunjić asymmetry is not covered. It adds a parameter that
  competes with γ for the same tail.
- The Olivero–Longbothum width is used as the reference scale. It is the
  simpler of the two fits in that paper, stated there as accurate to
  about 0.02 %; their 0.01 % is the other, more elaborate expression.
  Its largest error against the exact half-maximum width is 2.37×10⁻⁴ (at
  f_L/f_G = 0.29), the figure Wang et al. (2022) also report: the right
  size, not a bound. Irrelevant for a scale; stated because the package
  said "within 0.02 %" before.

## 9. Modes, the estimate, and the shape of the profile (v0.7.0)

*The definitions of §9.4, the checks of §9.6, and §9.1's notation,
null-space split and sign rule as far as those use them, are the
registration: written
before the code and before the registered runs (§9.6 lists what was
computed earlier), and not changed after. For them, where this section
and the code disagree, the code is wrong.
The rest of §9 records the design as ruled; where implementation or an
audit changes it, a dated amendment goes below the paragraph it
changes, never a rewrite. Results go to §9.8 only.*

§1–§8 report how well each parameter is determined: a label and a bound
per parameter, and for the two widths of one peak the least determined
direction of that pair. This section adds three things, all
Experimental like the rest of the module. P1, *modes*: which
combinations of peak parameters the spectrum determines worst, and what
each looks like in the spectrum. P2, *the estimate*: the same
diagnostics at a fitted estimate — through the `exact_k` adapter, for
the sub-model that fit used.
P3, *the profile*: for a single peak, whether the Fisher ellipse
describes the likelihood along the worst direction, or whether
curvature and boundaries take over first.

Like everything above, these are statements computed from a model at
stated parameters. The modes are a **diagnostic, not a set of
weights**: nothing here is meant to be fed to a downstream fit as a
covariance (§8's exclusion of exporting a covariance stands), and
nothing here is a confidence interval.

### 9.1 Modes (P1)

`identifiability_modes(energy, peaks, background=None, *, exposure=1.0,
k=3, ...)`.

**Notation.** θ_p are the peak parameters (amplitude, center, variance
and γ of every peak) in `var_gamma` coordinates and θ₀ their stated
values. I_eff is the effective information on θ_p with every estimated
background term profiled out — §4's Schur complement with the whole
background as the nuisance — so it already contains the cost of
estimating the background. S is the diagonal of the §5 reference
scales of θ_p (|A|, FWHM, FWHM²/(8 ln 2), FWHM) and I_s = S I_eff S. A
mode is a unit vector w with I_s w = w/λ; √λ is its relative
standard-deviation bound. The k with the largest λ are reported,
softest first. Mode i in this text is index i − 1 in the API.

**Why the background is not in the eigenvectors.** It was the first
design, and it needed a reference scale for every background
coefficient. The candidate (the peak sum's maximum count per channel,
divided by the basis function's maximum) failed two checks made before
this registration. The modes changed when the same linear background
was written on [1, E] instead of [1, E − E_mid], i.e. with the energy
origin, which §5 chose the FWHM to avoid; and a tenth of the peak height
was a different amount of background in different windows. Profiling
the background out needs no scale for it, and the modes do not change
under any rewriting of the background basis that keeps its span.

**What each mode reports**: its relative SD and a status, `identified`
or `weakly_identified` against the same `weak_relative_sd` as the
per-parameter labels; the weights on the peak parameters, by name, in
scaled coordinates; how the background follows — the change of each
background coefficient, in its own units, when the peak parameters move
one SD along the mode and the background is re-estimated; the
signature, the change of the expected counts per channel for that move,
background included, and its whitened norm; and the ratio of its λ to
the next one's, since where two are close only the plane they span is
determined, not the vectors. For every parameter, background
coefficients included, it also reports the share of that parameter's
variance bound lying along the mode, i.e. corr² between the parameter
and the mode coordinate, which does not depend on how the mode's own
weights are scaled. The share is `None` for a rank-deficient parameter
(its variance is unbounded while its covariance with an estimable mode
is not, so the limit is 0, and a finite number there would be an
artefact). Over all modes the shares of a peak parameter add to 1; those
of a background coefficient b add to 1 − (I_bb⁻¹)_bb/Var(b), with I_bb
the background block of the Fisher matrix; the rest is its own
uncertainty with the peaks known.

**What depends on the convention, and what does not.** The eigenvectors
and their order depend on the reference scales: another scaling of the
peak parameters lists other directions. The null space, as a subspace,
does not. Nor does the size of any signature: the whitened signature of
a one-SD move along any reported mode, background re-estimated, has
squared norm exactly 1 (Σ δμ²/μ = λ·wᵀ I_s w = 1, with μ and the
Jacobian both at the stated exposure). Every mode is one unit of
counting noise in the data; the convention decides which directions are
listed, not how visible they are.

*Amendment, 2026-10-23, same audit.* "Exactly 1" holds in exact
arithmetic. The covariance side keeps the soft end, which is what is
reported by default, except in a design singular to within rounding. At
the stiff end of a near-singular design, in such a design, and where the
background is itself nearly singular, a mode loses digits, and the
departure of its whitened norm from 1 measures how many. A mode for
which rounding leaves no positive bound is reported with status
`not_resolved` and a NaN bound, never as 0.

With `variance_floor_sd = 0` in `assess_identifiability`, the relative
SD of the softest mode is never smaller than any bounded peak
parameter's `relative_sd` or than `worst_relative_sd`: the largest
eigenvalue bounds every diagonal element and every principal sub-block.
The converse does not hold. A mode can be `weakly_identified` while no
single parameter is, because 0.1 is then applied to a norm over
fractions of different scales. That is what a joint direction is for,
and the docstring says so.

**Null space and weak directions are kept apart**, because they mean
different things (§5). The number of null directions of the peak block
is `null_space_dim − nuisance_null_dim`, both taken from the existing
decisions (the n·ε rule on the full unit-diagonal matrix, and on its
background block), so that the labels and the modes always agree. Those
directions are spanned by that many eigenvectors of I_eff rescaled to
unit diagonal, those with the smallest eigenvalues. Counting by the n·ε
rule on that matrix itself agrees in exact arithmetic but can differ by
one near the threshold; it is not used. Null modes go in their own field, with an
infinite bound, status `rank_deficient` and their whitened signature
norm as a numerical check; only the subspace they span means anything,
the vectors chosen inside it are arbitrary. The soft modes are taken on
the complement of the null space in the scaled coordinates, which makes
them combinations the data do determine: with two coincident peaks,
A₀ + A₁ lies in their span while every individual peak parameter is
`rank_deficient`. `crlb._NULL_PROJECTION_TOL` is not used: it decides
whether one parameter's axis touches the null space, it was calibrated
on crlb's background-free design, and the modes do not need it.

*Amendment, 2026-10-23, from the implementation's independent audit,
before any registered run (neither configuration of §9.6 has a null
space).* The count above does not keep the labels and the modes in
agreement. The two n·ε decisions are made on matrices of different size
and largest eigenvalue, so a dependency lying wholly within the
background can be null for the full matrix and not for its block. Two
background terms [1, 1 + δE] then gave a null mode for 8 of 81 values
of δ between 10⁻¹⁰ and 10⁻⁶ — the ordinary softest direction, plainly
visible in the data — while every peak parameter was labelled
`identified`. The number of null modes
is instead the rank of the full null space's peak-parameter rows,
counting singular values whose square exceeds `_NULL_PROJECTION_TOL`: the
tolerance that labels a single parameter `rank_deficient` — and 0 when
no peak parameter's bound is infinite, since the largest singular value
can exceed the tolerance where no single row does (it is at least the
largest row norm and at most √n times it). There is then a null mode
exactly when some peak parameter carries that label.
`nuisance_null_dim` is the rest of the null space. So the tolerance is
used after all, to keep the modes on the labels' decision.

**Computing route.** Invert on the unit-diagonal matrix as
`covariance_bound` does, apply the reference scales, then project out
the null space. Diagonalising the scaled matrix directly loses accuracy
where its condition exceeds 1/ε and the unit-diagonal one's does not.

**`variance_floor_sd` is not an input.** The floor's own uncertainty is
not information carried by the spectrum. Added to the bound it would
break the identity above — a mode would be drawn as invisible when the
data show it — and, the floor being shared by every peak, it would enter
as a rank-one term, not on the diagonal. It stays in `WidthAssessment`;
neither `identifiability_modes` nor `mode_profile` takes it. The floor's
location, `variance_floor`, is accepted: it moves a boundary, not the
matrix.

**Sign.** Each vector is oriented so that its largest-magnitude
component is positive; components within 10⁻⁶ of the largest, relative,
are a tie, broken by the first in parameter order. No rule is continuous
over a scan; a scan that tracks a mode aligns each vector with its
predecessor and says so.

**Lineshape independence.** The decomposition, the null-space split, the
background re-estimation, the shares and the signatures take a
Jacobian, parameter names, the peak/nuisance partition and the scales,
and live in `toyomacro._identifiability`, which imports nothing from the
package. The Voigt entry supplies the Voigt Jacobian and the §5 scales;
another lineshape needs its own Jacobian and scales and nothing more.

`to_dict` holds no matrix. k is not capped: k = n recovers the bound, and
`IdentifiabilityReport.fisher` is public anyway, so a cap would protect
nothing. What keeps the modes from being used as weights is the wording
of the docstring and of this record.

### 9.2 The γ = 0 boundary

The open item of §8 is closed in favour of symmetric treatment.
`WidthAssessment` gains `gamma_relative_sd`, sd(γ)/γ (infinite at
γ = 0), and `near_boundary_gamma`, γ < `boundary_sd`·sd(γ) (always true
when γ is rank deficient, as for the variance); `scan_identifiability`
gains the same flag as an array. The existing `near_boundary` keeps its
meaning, the Gaussian variance, now said in its docstring. The names
follow `near_boundary_v` and `near_boundary_tau` of the Fermi-edge
module, which already flags its two boundaries separately.

Reason: γ judged against the FWHM can read `identified` while sd(γ)/γ is
of order one (the §5 example, 0.58). That is the defect `near_boundary`
and `variance_relative_sd` exist to show on the Gaussian side, and the
fix costs two numbers that are already computed.

This adds fields to a public dataclass. No existing value changes, but
its repr, equality and `to_dict` keys do, and §8 lists "changing any
existing public API" as out of scope. It is recorded here as a
deliberate, additive exception and in the CHANGELOG under Changed;
`variance_floor_sd` was added to the same dataclass additively, in
v0.3.0.
The sentences of §5 and §8 that say no flag is raised are updated when
the flag ships.

### 9.3 The estimate (P2)

`identifiability_at_estimate(energy, peaks, background, *, intensity)`
evaluates §9.1 and the per-parameter labels at a fitted estimate. It is
a plug-in at a noisy estimate, not a confidence interval, and the
docstring says both. The fitters in this package that produce such
estimates are least squares, not Poisson maximum likelihood. If the
fit's estimator is unbiased and the model — including the fit's ties
and independent Poisson channels — is correct, its variance is at least
the bound at the true parameters; this report is a plug-in estimate of
that bound.

**The intensity is declared, never guessed.** The bound needs
`amplitude × V(E)` to be counts per channel as recorded (§3), and no fit
result in the package records what its intensities are. `intensity` is
a required keyword with no default, in the package's existing
vocabulary (`IntensitySemantics` of the readers). Only `"raw_counts"`
proceeds; `"count_rate"`, `"corrected_intensity"`, `"arbitrary"` and
`"unknown"` return a *withheld* report — the reason, and no numbers; a
string outside that vocabulary raises `ValueError`. With
raw counts the exposure is 1, so `exposure` is not taken: it would count
the acquisition twice.

The generic entry evaluates every peak parameter free. For a fit that
tied or fixed parameters it answers another question, and its
docstring says so.

**One adapter, for `exact_k`.** It takes an `ExactKReport`, a K, the
energy axis and `intensity` (the report stores neither the axis nor the
intensity convention), and not a bare candidate, because only the report
records whether the background was a plug-in. It rebuilds the
polynomial background on `build_bg_design`'s own basis, cast to
float64, and withholds
when the background was a plug-in Shirley or Tougaard (only a summary is
stored), when an amplitude is not positive (`exact_k` allows negative
ones by design), when the fitted mean is not positive in every channel,
or when the candidate carries `sigma_at_bound` or a
`center_*_at_energy_bound` flag (the estimate is not a stationary point).

*Amendment, 2026-10-23, at implementation.* The adapter is
`identifiability_from_exact_k(report, k_structured, energy, *,
intensity, k=3, thresholds=None)`. Both entries return an
`EstimateReport`, which holds:

- the per-parameter rows of the model actually evaluated;
- the flags of its width parameters (the variance against its floor, γ
  against 0);
- its `ModeReport`;
- for the free model only, the full `IdentifiabilityReport`;
- a one-sentence caveat.

The adapter also withholds when the candidate did not converge, and when
its background coefficients are missing or not finite. It raises on a K
the report has no candidate for, on an energy axis of another length or
one that does not contain the fitted centers, and on a non-positive
branching ratio. Because `exact_k` constrains σ to its own
`sigma_bounds`, the adapter also flags a shared variance within
`boundary_sd` bounds of either bound squared
(`near_boundary["var_shared_fit_bounds"]`). The flag against 0 does not
see that bound, and it is where the constrained fit stops. The generic entry raises, rather than withholds, on a mean that
is not positive: its inputs are the caller's, as for
`assess_identifiability`.

**The model the fit used.** `exact_k` shares one σ across the peaks,
fixes γ and can fix a doublet's splitting and branching ratio. All of
these are affine constraints in `var_gamma` coordinates, so the adapter
evaluates the sub-model, with Jacobian J·L, not the model with every
parameter free. The free model answers another question: it can name
var₀ − var₁ as the softest direction of a fit that has one shared
variance. In the core, a shared variance is scaled by FWHM²/(8 ln 2) of
the narrowest peak that shares it, the smallest candidate scale and so
the largest relative bound. (In `exact_k` every component has the same σ
and the same fixed γ, so the choice does not arise there.)
Per-parameter labels for a sub-model come from its own matrix;
`assess_identifiability` assumes four parameters per peak and is not
used for it.

**Not received in v0.7.0**, with the reason: `MultiPeakResult` (a
float32 batch, and without the Newton step its amplitude, center and σ
are not one stationary point); `core.FittingResult` (its area/height
columns mean different things for different producers, and the
background survives only as a curve); `FastFitResult` and
`pipeline.FitResult` (no background).

P2 has no stochastic check of its own; it is covered by deterministic
tests.

### 9.4 The profile (P3) — registered

`mode_profile(energy, peak, background=None, *, mode=0, t=...,
exposure=1.0, variance_floor=0.0)`, for a single peak only (it raises
otherwise). It has no `variance_floor_sd` argument (§9.1).

**Data and Δ.** The data are the expected counts y = μ(θ₀) at the
stated parameters — no noise, no seed; the "Asimov" data set of Cowan
et al. (2011, §3.2). Δ(t) is the minimum of the Poisson deviance
D = 2 Σ [μ − y + y ln(y/μ)] subject to the profile constraint below, so
Δ(0) = 0. At θ₀ the Hessian of D is exactly 2× the Fisher matrix, so the
profile departs from the Fisher prediction by curvature and boundaries
alone. This is a property of the model like everything else here, not a
coverage statement about any interval.

**Coordinate.** With w the chosen non-null mode of §9.1 and λ_w its
eigenvalue, the constraint is wᵀ S⁻¹ (θ_p − θ₀,p) = t·√λ_w, and D is
minimised over everything else: the coordinates along the other
eigenvectors of I_s, null ones included, and the background
coefficients. Both routes below work in these rotated, scaled
coordinates. The Fisher prediction is Δ(t) = t². A null mode is
refused: it has no finite SD in which to measure t.

**Constraints.** v ≥ `variance_floor` and γ ≥ 0 are held as bounds, as
the §6 estimator holds v (and as it would hold γ, instead of rejecting
γ = 0). Background coefficients are not constrained in sign, as the
estimator does not constrain them. μ > 0 is the domain of D, which is
infinite outside it. No boundary correction is applied (§8's chi-bar
item stays open).

**Active bounds and the KKT test**, the same for both routes. Let z be
the free coordinates of the coordinate paragraph above, so that each
width bound is an affine inequality g_k(z) ≥ 0 in z. A bound is active
when the solution lies within 10⁻⁶ of it in units of that parameter's
reference scale; the solution is projected onto its active bounds, and
D, Δ and the KKT test are evaluated there. With I_z the Fisher matrix in
z at that point, take the multipliers ν_k of
½∇_z D = Σ_active ν_k ∇_z g_k by ordinary least squares in z (the
minimum-norm solution if the active gradients are dependent) and the
residual r = ½∇_z D − Σ ν_k ∇_z g_k. The test passes when every inactive
g_k(z) ≥ 0, every coordinate has |r_i|/√(I_z)_ii < 10⁻⁶ (the rule of
`_bootstrap.fit_poisson_mle`, with the multipliers taken out), and every
multiplier, normalised as ν_k·(∇g_kᵀ I_z⁻¹ ∇g_k)^½, is above −10⁻⁶; a point
where I_z is singular fails it. The recorded KKT residual is
max_i |r_i|/√(I_z)_ii; for a route that fails, that of its
lowest-deviance candidate.

**Events.** Events are read from the solution the summaries use
(below). The stop is found in the outward pass; if the inward pass
finds, in the solution the summaries use, *amplitude at zero* or
*corner* at an earlier grid point, that point becomes the stop. A failed
inward iterate never moves the stop. Other
events are read after the inward pass. *Amplitude at zero* and *corner*
are limits at which no route can be expected to succeed: when neither
route succeeds at a point, the point is the stop — not "not obtained" —
if any final iterate of either route, from any start or active set,
shows one of them.

- *bound active* — for each side and each bound, the first grid point
  (not refined) whose solution has that bound active. Reported with the
  bound and Δ there. The profile continues past it; it is the
  constrained minimum.
- *background negative* — a solution whose re-estimated background
  curve is negative in some channel. Recorded per point.
- *amplitude at zero* — a solution with A ≤ 10⁻⁶|A₀|. At A = 0 the
  center and the widths leave the mean, and when w has a component on
  the center any t is then met at the background-only deviance.
- *corner* — with `variance_floor` 0, a solution with both width bounds
  active at once. The Voigt does not exist there. With a positive floor,
  v at the floor and γ = 0 is a valid Voigt (route A's fourth active
  set) and the side continues.

**Stops.** A side's stop is the first grid point from t = 0 that has
*amplitude at zero* or *corner*. The side is not solved past it, and the
inward pass on that side starts at the point before the stop. The stop
and every later point are reported as the event, not as a number, and
so are Δ(±1) and Δ(±2) at or beyond it. A stop reports its t, Δ at the
point before it, and the background-only deviance (D minimised over the
background coefficients with A = 0).

**Two routes.** Route A enumerates the admissible sets of active bounds
on {v, γ} — three when `variance_floor` is 0 (both at zero is the
corner), four otherwise — solves each by Fisher scoring, and keeps the
lowest-deviance point that passes the KKT test, with `max_iter=60`; it
succeeds when one does. Route B is SLSQP (`ftol=1e-12`,
`maxiter=1000`); it succeeds when SciPy reports success and the KKT
test passes. Grid points are solved outward from t = 0 on each side,
starting from the nearest obtained neighbour on the t = 0 side and from
the stated point, then once inward from the ends, each point starting
from its own outward solution and, where its outer neighbour was
obtained and is not the stop, from that neighbour's solution;
each route keeps its lowest-deviance successful result. Bisection points
are solved from their bracketing solutions. Every grid point solved
records both routes' Δ, success and KKT residual. A point is flagged
when |Δ_A − Δ_B| > 10⁻⁶·max(1, min(Δ_A, Δ_B)) or when either route
fails. The summaries use the lower successful Δ of the two and name the
route it came from. Infeasibility is not tested: a point where neither
route succeeds, and which is not a stop, is "not obtained", and so is
every summary that needs it. Numerical details not fixed here — damping
in the scoring, SLSQP's internals beyond the settings above — are the
code's: fixed before the registered run and recorded with its results.
Nothing is retuned after the registered profiles have been seen, and a
disagreement between the routes is reported, never resolved.

**Summaries.** Δ on the grid; Δ(±1) and Δ(±2); and the crossings. t₊ is
found in the first grid interval with t_j ≥ 0, both ends before the
stop, and Δ(t_j) < 1 ≤ Δ(t_{j+1}), refined by bisection to a bracket of
10⁻⁶ in t, whose midpoint is reported. If
Δ < 1 at every grid point before the stop, t₊ is reported as the stop's
event, and with no stop, as "beyond the grid". t₋ likewise for t ≤ 0.
t± needs every grid point from 0 to the far end of its bracket and every
bisection point; when it is the stop's event or "beyond the grid", every
grid point before the stop, or on that side. If any of those is "not
obtained", so is t±. The ratio to the
Fisher crossing is |t±| itself. 1.00 is the one-parameter 68.3 %
convention, used as a yardstick only; there is no 2-D profile, so 2.30
is not used.

### 9.5 Vocabulary

Words used for these ideas elsewhere, mapped with their limits:

| here | there | limit |
|---|---|---|
| `rank_deficient`, null mode | gauge direction | only for a redundancy of description: a background term with the peak's own shape (a null mode), or two background terms with the same shape (`rank_deficient` background coefficients, not a mode). Coincident peaks are distinct components the data cannot separate, a degeneracy of the measurement, not a gauge. §4's "gauge" — the unit-diagonal rescaling — is the same idea applied to units |
| `weakly_identified` mode | sloppy direction | related, not equivalent. Sloppy models (Gutenkunst et al. 2007) have eigenvalues spread over many decades; here a threshold is applied to one direction |
| a known background, `fixed_instrument`, `exact_k`'s fixed γ | datum | a value chosen, not learnt from the spectrum |

### 9.6 Checks — registered

Both checks take seconds and run as tests, as §6 does; the long-run
harness of the composition validation (resumable chunks, run identity
files) is for runs that take hours and is not used. What carries over is
that seeds, settings and criteria are fixed here before the code exists,
and that each test asserts that the computation ran. Each result is
entered in §9.8 when its check lands, with the commit and the Python,
NumPy and SciPy versions it ran on (NumPy does not promise identical
Poisson streams across versions).

**P1, pass/fail.** One peak at center 0, γ = 0.3 eV, σ/γ = 1, area
2×10⁴ counts·eV, on `linear_background(E, 300, 10)` with both terms
estimated; E = −3…3 eV in 0.02 eV steps (301 channels); exposure 1;
`variance_floor` 0. This is §6's configuration with a linear instead of
a flat background. The data are
`_bootstrap.draw_poisson(expected_counts, 10_000, rng)` with
`rng = default_rng(SeedSequence([20261023, 1]))`, fitted by
`tests/_poisson_mle.fit_in_chunks` (the §6 estimator: exact constrained
Poisson maximum likelihood) with the linear basis, started at the truth,
`max_iter=60`, `score_tol=1e-6`.

Before the replicas, the test asserts that the configuration is interior
along each of the k = 3 modes, which are computed from the model at the
truth, not from the replicas: θ₀,p ± 3√λᵢ·S wᵢ has A > 0, v > 0 and γ > 0
for i = 1, 2, 3 (3 = `boundary_sd`).

*Criterion.* For each mode i = 1, 2, 3, the sample standard deviation
(ddof = 1) over the replicas of wᵢᵀ S⁻¹ (θ̂_p − θ₀,p), divided by √λᵢ, lies
within 4 % of 1. All three must pass. The tolerance and its reasons are
§6's: the sampling error of an SD from 10⁴ replicas is 0.71 %, so 4 % is
5.7 standard errors, ample for three comparisons, and a Fisher matrix
wrong by 10 % moves the ratio by about 5 %.

*Not evaluable.* If any replica of seed key 1 does not converge or ends
at the floor, or the interior check fails, P1 is not evaluable and does
not pass. The estimator settings and the seed are not changed to rescue
it. A fail or a not-evaluable result is entered in §9.8, and
`identifiability_modes` does not ship until its cause is found. After a
cause is found and fixed, P1 is run once more with seed key 1, and both
results are entered.

*Recorded, not judged:* the angle, folded to 0°–90°, between mode 1 and
the first principal axis of the scaled replicas S⁻¹(θ̂_p − θ₀,p) (the
leading eigenvector of their sample covariance), with λ₁/λ₂; the mean of
each projection in units of √λᵢ, signed by §9.1's sign rule; and the same ratios, with their
convergence counts, for seed keys [20261023, 2] and [20261023, 3], which
decide nothing.

*What this checks and what it does not.* The variance of a fixed linear
combination of the estimates agrees with the bound along any direction
fixed in advance, whatever the scaling, so a pass confirms the bound
along the reported directions and says nothing about the scale
convention, which is a choice, not a fact. The decomposition itself is
covered by deterministic tests: the eigen-relations; the whitened-norm
identity; the softest mode against every bounded `relative_sd`, and
`worst_relative_sd` at `variance_floor_sd = 0` (to rounding); invariance
under the energy unit and under rewriting the background basis; the
null-space count on constructed cases (coincident peaks 4, a duplicated
background term 0, a background term with the peak's own shape 1);
relative SDs scaling as 1/√exposure.

**P3, recorded only.** No prior claim of agreement or disagreement. Along
mode 1 (`mode=0`), on the grid t = k/10, k = −30…30, with the
definitions of §9.4. Every configuration has the peak at center 0, a
0.02 eV step, an estimated `linear_background(E, 300, 10)`, exposure 1
and `variance_floor` 0:

| name | σ/γ | window | area (counts·eV) |
|---|---|---|---|
| soft | 1/3 (γ = 0.3 eV) | ±γ: −0.3…0.3 eV, ends included, 31 channels | 2×10⁴ |
| soft ×100 | 1/3 (γ = 0.3 eV) | ±γ: −0.3…0.3 eV, ends included, 31 channels | 2×10⁶ |
| hard | 1 (γ = 0.3 eV) | −3…3 eV, ends included, 301 channels (the P1 configuration) | 2×10⁴ |

Recorded per configuration: the summaries of §9.4, the events and
stops with what each stop reports, and the per-point agreement of the
two routes.

**Disclosure.** Some of these quantities were computed before this
registration, while the design was being checked. The existing §6
per-parameter check was run once on this window with
`linear_background(E, 300, 10)` and §6's own seed, to see whether the
estimator copes with a slope. Single-route profiles along mode 1 were
computed in the hard and soft configurations, and the soft ×100 row was
added because the soft profile met a bound before Δ = 1 on both sides.
None of those numbers is a result; the registered checks use a new seed
and the definitions above.

### 9.7 Not in this version, and what stays open

The Doniach–Šunjić asymmetry α (P4) is not in this version. The core of
§9.1 is lineshape-independent, so what it lacks is the DS ⊗ Gauss
lineshape and its Jacobian — a lineshape change first. Two things have
to be decided before it: the amplitude convention, since the area
diverges for α > 0 (height, or area truncated to the window), and the
window width as a scan axis, since the information on α sits in the
high-binding-energy tail that the window cuts. The expectation, not a
result, is that the first mode mixes α, γ and the Shirley height and
stays finite, a monotone step and a one-sided power-law tail differing
in shape. Fixing α at a literature value is a datum.

Families of models (P5) — multistart, enumerating the distinct
solutions (peak count, component layout) and weighting each by a
Laplace evidence log L − ½ log det(F/2π) — belong with `model_selection`
and `exact_k` (docs/RANK_MODEL_SELECTION.md), not here.

Still open: §8's chi-bar item, and the Shirley step's own dependence on
the peaks.

### 9.8 Results

Each part gets an independent audit in a context that did not write it.
Results of §9.6 are entered here as each check lands; nothing above
changes because of them.

**P1 — pass.** Run once, at commit dc05fb2 with a clean tree, on
Python 3.12.11, NumPy 2.3.5 and SciPy 1.17.0 (macOS, arm64). Seed key
1: all 10 000 replicas converged and none ended at the floor; the
interior check passed.

| mode | relative SD bound | replica SD / √λ | replica mean / √λ |
|---|---|---|---|
| 1 | 0.00708 | 0.991 | +0.004 |
| 2 | 0.00155 | 1.002 | −0.006 |
| 3 | 0.000663 | 0.985 | −0.001 |

All three ratios lie within 4 % of 1. Recorded, not judged:

- The first principal axis of the scaled replicas is 0.13° from mode 1,
  with λ₁/λ₂ = 21.0.
- Seed keys 2 and 3, all replicas converged and none at the floor, give
  0.998, 0.993, 0.993 and 0.977, 1.005, 1.003.
- Over the nine ratios the mean is 0.994. Seven lie within two sampling
  errors (0.71 % each). The largest departure, 2.3 % (key 3, mode 1),
  is 3.3 of them.

As §9.6 says, this confirms the bound along the reported directions in
one single-peak configuration. It says nothing about the scale
convention.

## References

- J. J. Olivero and R. L. Longbothum, *J. Quant. Spectrosc. Radiat.
  Transfer* **17**, 233–236 (1977), doi:10.1016/0022-4073(77)90161-3
- P. Thompson, D. E. Cox and J. B. Hastings, *J. Appl. Cryst.* **20**,
  79–83 (1987), doi:10.1107/S0021889887087090
- Y. Wang, B. Zhou, R. Zhao, B. Wang, Q. Liu and M. Dai, *Mathematics*
  **10**, 210 (2022), doi:10.3390/math10020210
- S. G. Self and K.-Y. Liang, *J. Am. Stat. Assoc.* **82**, 605–610
  (1987), doi:10.1080/01621459.1987.10478472
- G. Cowan, K. Cranmer, E. Gross and O. Vitells, *Eur. Phys. J. C*
  **71**, 1554 (2011), doi:10.1140/epjc/s10052-011-1554-0
- R. N. Gutenkunst, J. J. Waterfall, F. P. Casey, K. S. Brown, C. R.
  Myers and J. P. Sethna, *PLoS Comput. Biol.* **3**, e189 (2007),
  doi:10.1371/journal.pcbi.0030189
