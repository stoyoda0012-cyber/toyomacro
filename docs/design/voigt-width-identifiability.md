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
   matrix at σ/γ = 1/30 is 6.18 on ±33.3γ, about 9×10² on ±γ and about
   5×10⁶ on ±0.3γ (the last two move by several percent with the
   sampling).
5. **A background.** Its shot noise raises the mean under the peak; if
   it also has to be estimated from the same spectrum it competes with
   the widths for the same counts, and that second effect does not need
   a large background. At ±10γ, σ/γ = 1/3, a flat background of 0.05 %
   of the peak height leaves the least determined width direction 99 %
   of its information when known and 43 % when estimated; at 10 % of
   the peak height the figures are 44 % and 20 %.

Items 4 and 5 are treated as *practical weak identification*: more
counts or a wider window change them. They are not structural
non-identifiability and the labels below never claim so.

## 2. Derivatives that survive σ → 0

The engine's Faddeeva-based Jacobian cannot be used here. In w″(z) a
term of order 4z cancels down to 2/z³, so the relative error of the
variance derivative grows like |z|⁴ with |z| = |x + iγ|/(σ√2); on a
±33γ window `voigt_with_jacobian` and `voigt_with_hessian` have lost
every digit by σ/γ ≈ 10⁻³.

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
- the switch cannot be lowered to buy back the Faddeeva error: on ∂V/∂v
  the best any N achieves is 4×10⁻⁸ at |z| = 5 and 3×10⁻⁵ at |z| = 4,
  and more terms make it worse (N = 40 at |z| = 4: 3×10²).

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
the switch on the Faddeeva side. An earlier draft quoted 10⁻¹¹ from one
point per |z|; the population measurement replaced that number and moved
the switch from 8 to 7, the lowest |z| at which the expansion is at
rounding level.

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
`fixed_instrument` describe one statistical model and are related by
I_φ = Tᵀ I_θ T; only `sigma_gamma` has a singular T. `pvoigt` is a
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
center recouples to the widths (correlation 4×10⁻⁴ in the pinned case).

## 4. Effective versus conditional information

For widths u and everything else q (amplitude, position, background),

    I_eff = I_uu − I_uq I_qq⁺ I_qu

is what is left for the widths once q is estimated from the same
spectrum; its inverse is the width block of the full inverse. The plain
sub-block I_uu is the information *if q were known* — what
`fisher_information.analyze_sigma_gamma_axes` reports. I_uu − I_eff is
positive semi-definite, so the sub-block is always the optimistic one,
and the gap is large exactly where it matters: at ±γ with an estimated
flat background the bound on γ is 0.24 of the FWHM from I_eff and 7×10⁻⁴
from I_uu. Both are reported side by side.

The nuisance block is inverted on the unit-diagonal matrix with the n·ε
rank rule, the gauge `crlb.compute_multipeak_fisher` already fixes: the
Fisher matrix mixes counts, eV and eV², so a rank decision on the raw
matrix is a statement about units.

## 5. The labels

Judged in `var_gamma`, from the full inverse, so item 1 of §1 cannot
raise a flag.

- `rank_deficient` — the unit-diagonal matrix is numerically singular
  and this parameter's axis projects onto its null space. Rank rule and
  both tolerances are imported from `crlb`, not copied.
- `weakly_identified` — the bound exceeds a fraction of a reference
  scale.
- `identified` — neither.
- `near_boundary` — an independent flag: the variance lies within k
  bounds of its lower limit.

**Reference scales**: |A| for an amplitude; the total FWHM for the center
and for γ; FWHM²/(8 ln 2) for v. Not the parameter's own value, which is
undefined at v = 0 and, for a position, not invariant under a shift of
the energy origin. The variance is reported against itself as well
(`variance_relative_sd`), because the FWHM scale stays finite as the
Gaussian component vanishes: at σ/γ = 0.03 the width block is
`identified` at 2×10⁻³ while sd(v)/v = 1.6. The test suite checks that
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
  simulation in §6 is the evidence that 3 is a sensible margin.

**What the labels are not.** `identified` on the FWHM scale is not
detection of a small Gaussian component. `weakly_identified` is not
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

Interior, pass/fail: σ/γ = 1, estimated flat background, 10⁴ replicas, a
regime the module itself labels identified and away from the boundary.
Replica sd over Fisher sd 1.002…1.010, correlations within 0.006, bias
under 0.01 sd; tolerance 5 %.

Near the boundary, *recorded, not judged* (4000 replicas per row):

| true v, in Fisher sd | near_boundary | share held at the floor | replica sd / Fisher sd (v) |
|---|---|---|---|
| 0 | yes | 0.4975 | 0.579 |
| 1 | yes | 0.1623 | 0.863 |
| 2 | yes | 0.0260 | 1.007 |
| 4 | no | 0.0000 | 0.986 |

For orientation, not as a criterion: with one parameter on the boundary
and the rest interior the asymptotic law is half a point mass at the
floor and half a half-normal, giving ½ and √(½ − 1/(2π)) = 0.584 in the
first row. With the floor at a calibrated instrument variance and no
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
Towards σ = 0 the variance bound does not move (0.009 of its scale from
σ/γ = 10⁻² down to 0) while the σ bound grows as 1/σ and the information
on σ falls as σ²; towards a narrow window both grow. That is how item 1
of §1 is told from items 4 and 5. `examples/05_width_identifiability_map.py`
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
- The Doniach–Šunjić asymmetry is not covered. It adds a parameter that
  competes with γ for the same tail.
- The Olivero–Longbothum width is used as the reference scale. Its
  largest error against the exact half-maximum width is 2.37×10⁻⁴ (at
  f_L/f_G = 0.29), slightly above the nominal 0.02 %; irrelevant for a
  scale, stated because the package said "within 0.02 %" before.

## References

- J. J. Olivero and R. L. Longbothum, *J. Quant. Spectrosc. Radiat.
  Transfer* **17**, 233–236 (1977), doi:10.1016/0022-4073(77)90161-3
- P. Thompson, D. E. Cox and J. B. Hastings, *J. Appl. Cryst.* **20**,
  79–83 (1987), doi:10.1107/S0021889887087090
- S. G. Self and K.-Y. Liang, *J. Am. Stat. Assoc.* **82**, 605–610
  (1987), doi:10.1080/01621459.1987.10478472
