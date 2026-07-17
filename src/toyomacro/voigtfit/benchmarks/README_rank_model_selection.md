# Rank diagnostics & model selection — evaluation design

**Status:** evaluation design frozen BEFORE implementation (Phase 0 /
Gate 0 artifact, 2026-07-17). Code that this document evaluates does
not exist yet; changing the success criteria after implementation
starts requires an explicit note in this file's history section.

**Scope guard:** research/diagnostic feature, developed strictly
outside the v0.1 public surface — no CLI, no GUI, no Web API, no
`__init__` re-export until the API is frozen (a separate, later
decision). Reference implementation is SciPy/NumPy only; MLX/JAX ports
are out of scope.

## 1. What is being evaluated

Given peak placement, widths, background family, and spin-orbit (SO)
constraints, the feature must diagnose — reproducibly, not assert —

1. how many independent directions the data supports (rank layer),
2. which candidate peak count `K` the information criteria and
   residuals support (IC layer),
3. and report `supported / ambiguous / unsupported` with a candidate
   range instead of a single "estimated peak count" when the layers
   disagree.

The primary result of the whole effort is the **map of conditions
under which K is NOT identifiable**, not a peak-counting success rate.

## 2. Frozen conventions

### 2.1 Model

```text
y = Phi(theta) a + B beta + epsilon
```

`Phi`: Voigt/SO-structured peak basis (one structured amplitude column
per component — a fixed SO doublet is ONE column, not two);
`a`: linear amplitudes; `B beta`: linear background; `theta`:
nonlinear parameters (centers, widths).

### 2.2 Four ranks, always reported together

1. data-matrix rank `rank(W^(1/2) Y)` (multi-spectrum case)
2. linear basis rank `rank(W^(1/2) [Phi | B])`
3. local full-parameter rank `rank(W^(1/2) [Phi | B | S D])`
4. profiled nonlinear rank `rank(W^(1/2) (I - P_A) S D)` (VARPRO
   residual projection), with `S = d(Phi(theta)a)/dtheta`,
   `A = [Phi | B]`, `D` the physical parameter-scale matrix.

### 2.3 Rank reporting — no single number

Persist all of: full singular-value vector; effective rank at relative
threshold `s_i/s_1 > 1e-2`; reference rank at `> 1e-3`; entropy
effective rank `exp(-Σ p_i ln p_i)` with **Fisher-eigenvalue weights
`p_i = s_i² / Σ_j s_j²`** (consistent with the M5 identity
`λ_i(Fisher) = s_i²`); condition number (`inf` whenever
`n_cols > n_rows` — a wide matrix always has a parameter-space null
space, so it is singular as an identifiability diagnostic); Fisher
eigenvalues and CRLB where computable. Threshold ranks are never treated as ground truth; absolute
detectability claims use CRLB / physically scaled Fisher eigenvalues
(uniform-variance whitening only rescales singular values uniformly, so
SNR does not move relative ranks).

### 2.4 Parameter scales (defaults, frozen)

Jacobian column units change ranks, so scales are fixed now and echoed
into every result object:

| parameter | default scale |
|---|---|
| amplitude | 10 % of representative amplitude (safe floor near zero) |
| center | 10 % of representative FWHM |
| Gaussian width | 10 % of representative FWHM |
| Lorentzian width | 10 % of representative FWHM |

Scales are configurable, but no conclusion may rest on an unrecorded
automatic normalization.

### 2.5 Information criteria (Gaussian, unknown common variance)

```text
AIC  = n log(RSS/n) + 2k
AICc = AIC + 2k(k+1)/(n-k-1)     (inf + warning when n <= k+1)
BIC  = n log(RSS/n) + k log(n)
```

Poisson data use the Poisson log-likelihood / deviance; non-positive
model predictions raise or warn explicitly — never silently clipped.

`k` = actual free parameters: fixed/shared parameters and fixed SO
splitting / branch ratio do not count; free background coefficients do;
an estimated Gaussian variance is counted explicitly. Nominal `k` is
reported side by side with effective (SVD) rank — never silently
replaced by it.

### 2.6 Exact-K guarantee

Every candidate model is refit with **exactly K structured
components**. `max_peaks=K` ("at most K") is not an acceptable
surrogate and its use in the candidate loop is a test failure
(see 4.2-E5).

### 2.7 Constraint scope (v1)

The v1 reference problem is unconstrained linear LS. NNLS / simplex
constraints / active-set switching are NOT treated as the derivative of
unconstrained VARPRO. If regularization is added, data-derived rank and
post-regularization invertibility are reported separately.

## 3. Synthetic evaluation grid (frozen)

| axis | values |
|---|---|
| true structured components | 1, 2, 3, 5 |
| peak spacing | 0.25, 0.5, 1.0, 2.0 × FWHM |
| SNR | 10, 30, 100, 1000 |
| background | none, constant, linear |
| SO | off, fixed doublet (known splitting & branch ratio) |
| random seeds | ≥ 5 per cell |

- Lineshape: Voigt only. Candidate range `K = 1..5`.
- Full grid runs on demand only; CI gets a smoke subset
  (see 4.4-P2).
- Data generation reuses the existing synthetic infrastructure
  (`crlb.generate_ncomp_spectra` and friends) where possible.

## 4. Acceptance criteria (frozen — Gate 4 judges against these)

### 4.1 Mathematical sanity (Gate 1)

- M1: two columns at identical position and width → linear rank 1.
- M2: K well-separated peaks → linear basis rank K.
- M3: a fixed SO doublet counts as ONE structured amplitude column.
- M4: adding an independent constant / linear background raises the
  maximal linear rank by exactly 1 each.
- M5: squared singular values of the whitened scaled Jacobian match
  Fisher eigenvalues within numerical error (cross-check against
  `crlb.compute_multipeak_fisher`).
- M6: changing parameter units while changing the corresponding scale
  leaves the diagnostics invariant.

### 4.2 Information-criterion correctness (Gate 2)

- E1: Gaussian AIC/AICc/BIC match hand-computed values on a small
  fixed example.
- E2: Poisson likelihood matches a known small example.
- E3: `n <= k+1` yields `AICc = inf` + warning, never a finite value.
- E4: free-parameter count is correct under fixed/shared/SO
  constraints.
- E5: every candidate contains exactly K structured components —
  asserted by test, not by convention.
- E6: a failed candidate fit (non-convergence, boundary hit) is never
  selected silently; failure states propagate into the report.

### 4.3 Synthetic behavior (Gate 4)

- S1: well-separated peaks at SNR ≥ 100 → true K inside the supported
  range in ≥ 90 % of seeds.
- S2: at 0.25 FWHM spacing, forcing the correct K is NOT the success
  condition; reporting non-identifiable/ambiguous IS success.
- S3: on data generated from a fixed SO doublet, at sufficient SNR the
  structured model is not disfavored by AICc/BIC against the
  overparameterized two-independent-peaks model.
- S4: when AIC/AICc/BIC and rank disagree, the report warns and does
  not assert a single K.
- S5: systematic residual structure blocks a "sufficient" verdict even
  at the lowest IC.

### 4.4 Performance

- P1: IC computation itself is negligible (<< fit time).
- P2: CI smoke subset ≤ 60 s on CPU.
- P3: no per-pixel K sweep over full maps — representative spectra /
  means / cluster centroids only.

## 5. Per-candidate output contract

```text
K_structured, n_visible_peaks, n_free_parameters,
fit_success / termination_reason,
log_likelihood / RSS / deviance,
AIC / AICc / BIC / delta_AICc / delta_BIC,
linear_rank_1e-2 / linear_rank_1e-3,
profiled_nonlinear_rank, entropy_effective_rank, condition_number,
Fisher_eigenvalues / selected_CRLB,
boundary_flags, residual_diagnostics, warnings
```

Global result additionally carries: supported candidate range, verdict
state, noise model, whitening, parameter scales, rank thresholds, seed.
Everything JSON-serializable.

## 6. Verdict rules (frozen wording)

- ICs agree AND local rank supports the nominal dof → `supported`
- small IC differences OR IC/rank disagreement → `ambiguous`
- added peak direction non-identifiable AND no IC improvement →
  `unsupported`

Thresholds behind "small" are implementation parameters that must be
echoed in the output — the verdict is never hard-coded from an IC
threshold alone, and the evidence behind each verdict is part of the
output contract.

## 7. Phase 0 baseline record (2026-07-17)

- HEAD: `75a82e0` (docs(upstream-issues): filed as ml-explore/mlx#3858/#3859/#3860/#3861)
- Working tree at start: only `?? docs/RANK_MODEL_SELECTION_IMPLEMENTATION_PLAN.md`
  (untracked planning doc; deliberately left untouched and uncommitted).
- `gp_reference.py`: tracked, clean, last touched by `d7eaf22`
  (2026-07-16) → stable; the Phase 0 contingency (build only on
  low-level Voigt/Jacobian + local QR) is NOT needed.
- Reuse candidates present and clean: `weight_cache.py` (671 L),
  `voigt_jacobian.py` (667 L), `crlb.py` (1009 L),
  `multipeak_config.py` (180 L), `multipeak_solver.py` (3521 L,
  `build_bg_design` at :514).
- No name collisions with planned files (`rank_diagnostics.py`,
  `model_selection.py`, their tests, this benchmark pair).
- Baseline tests: `test_crlb.py` + `test_voigt_jacobian.py` +
  `test_gp_lm.py` + `test_gp_compact.py` + `test_gp_jacobian.py`
  → **124 passed, 0 failed, 2.87 s**.
  (Known cross-machine note: `test_gp_lm_background_end_to_end[2]` is
  marginal on the CUDA/WSL2 box's NumPy baseline; it passes here.)
- Environment: Apple M3 Max, Python 3.12.11, NumPy 2.3.5,
  SciPy 1.17.0, MLX 0.31.2 (Metal; irrelevant to the SciPy/NumPy
  reference implementation but recorded for completeness).

## 8. Change history (clarifications only — criteria unchanged)

- 2026-07-17 (Phase 1, M6-driven): two clarifications of §2.4, found by
  the unit-invariance test before any acceptance criterion changed.
  (a) The background coefficient scale defaults to the typical-height
  proxy `amplitude_scale / representative_fwhm` — background columns
  are dimensionless normalized monomials (counts), NOT area-unit
  amplitude columns (counts·energy), so reusing the amplitude scale
  would break unit invariance. (b) "scaled before any SVD" applies to
  the linear layer too: `rank(W^(1/2)[Phi|B])` is computed on scaled
  columns; the unscaled mix of 1/energy-unit peak columns with
  dimensionless background columns is intrinsically unit-dependent.
- 2026-07-17 (Gate 1 review): (a) entropy effective rank fixed to
  Fisher-eigenvalue weighting `p_i = s_i²/Σs_j²` (was `s_i/Σs_j`);
  formula now stated in §2.3, docstring, and tests. (b) condition
  number defined as `inf` for wide matrices (`n_cols > n_rows`).
  (c) Phase 3 handoff: candidate models K = 1..Kmax must share ONE
  `ParameterScales` instance — per-candidate auto-derived scales would
  shift rank layers and make them incomparable across K.
- 2026-07-17 (Gate 2 review): (a) Gaussian RSS scoring requires
  `variance_estimated=True` — the §2.5 formulas are the profile
  likelihood with the variance MLE substituted in, so the variance is
  always an estimated parameter there; a known-variance Gaussian path
  is a separate (unimplemented) log-likelihood. (b) Non-finite IC
  selection semantics fixed: all-`inf` criterion → best `None` +
  deltas `None` + warning; two or more `-inf` exact fits → tie
  reported, not resolved; a single `-inf` wins with delta 0 / `+inf`;
  NaN is never produced. (c) `poisson_deviance` validates shape and
  `y >= 0` like `poisson_loglik`.
- 2026-07-17 (Gate 3 review): (a) rank diagnostics gained
  `shared_sigma=True` — a shared-width fit is diagnosed on its actual
  K + 1 nonlinear directions (`df/d(sigma_shared) = Σ_k df/d(sigma_k)`),
  not 2K independent ones; the independent-sigma diagnosis stays the
  default. (b) verdict rules tightened: `best_by["aicc"] !=
  best_by["bic"]` is `ambiguous` even for a single-K support range, and
  a selected candidate carrying negative amplitudes is `ambiguous`,
  never `supported`. (c) the multi-start winner is the minimum-RSS
  CONVERGED start; per-start convergence is recorded; a candidate fails
  only when no start converges. (d) descending (binding-energy) axes
  are normalized to ascending internally; non-monotonic axes are
  rejected.
- 2026-07-17 (Phase 3b): plug-in Shirley/Tougaard backgrounds per the
  plan — estimated ONCE from the observed spectrum before the K loop,
  subtracted and held fixed for every candidate, never a linear column
  or a free parameter; `background_mode` + settings + curve summary
  recorded; the report warns that ICs are heuristics conditioned on a
  data-estimated background and must not be compared across background
  families. Shirley is fixed-endpoint (`auto_range=False`); Tougaard
  uses the universal `C = 1643 eV²`. Also generalized during 3b
  testing: the AICc/BIC-disagreement check now runs BEFORE the
  support-set-size check — when the criteria agree their common best K
  has both deltas 0, so an empty support set is itself a disagreement
  symptom (observed: AICc chasing noise with a negative-amplitude
  extra component while BIC rejects it) and is labelled `ambiguous`,
  not `unsupported`.
- 2026-07-17 (Phase 4): (a) the §5 output contract's
  `residual_diagnostics` implemented — whitened-residual lag-1
  autocorrelation with a z-gate (default 4.0, echoed in config); a
  selected candidate leaving structured residual is `ambiguous`, never
  `supported` (S5). (b) `bench_rank_model_selection.py` runs the
  frozen grid (`--mode smoke|full|slice3b|sliceso`); the committed
  record is `results_rank_model_selection.json`; CI runs only the
  smoke subset. (c) The naive full-window equal-spacing start lost
  clustered peaks and was replaced by signal-mass quantile placement
  (measured on the grid: true-K-in-supported at K=3/1-FWHM/high-SNR
  went 5/60 → 30/60; overall S1 77.2 % → 82.5 %). Gate 4 outcome:
  S2/S3/S5/P1-P3 met; S4 99.55 % (7/1560 overfit claims, all
  low-SNR); S1 at 82.5 % vs the 90 % target — dominated by SO cells
  and 1-FWHM spacing, with 62/63 failures reporting `ambiguous`
  (honest refusals, not wrong claims). Recorded as a miss with
  analysis, per the Gate 4 contract.
- 2026-07-17 (Gate 4 adjudication + report corrections): decision (a) —
  the S1 miss is ACCEPTED and preserved as-is; no SO-exclusion
  redefinition (the dominant factor is whether 1 FWHM counts as
  "well-separated": at SNR≥100, 87.8 % no-SO vs 48.9 % SO at 1 FWHM,
  93.3 %/100 % at 2 FWHM); AICc scan-statistic correction deferred to
  future work. Three report corrections, rules unchanged: (1) frozen
  S4 is now measured DIRECTLY
  (`S4_no_single_K_assertion_under_IC_rank_disagreement`); the former
  99.55 % metric is renamed `no_confident_overfit_claim_rate` — it is
  a different indicator. (2) The overfit-claim note "all low-SNR" was
  wrong: six at SNR 30, one at SNR 100 (SO, 1 FWHM). (3) The 3b slice
  generation is a NOMINAL Shirley-like shape (tail cumsum of clean
  peaks), not self-consistent with the estimator — aggregates renamed
  (`shirley_on_nominal_shirley_like` etc.) and matched Shirley/Tougaard
  generation explicitly marked unevaluated. S1 is additionally
  reported broken down by spacing × SO.

## 9. Non-goals (frozen)

- SVD rank is never reported as "the number of chemical states".
- Minimum AIC/BIC alone never fixes the physical model.
- Nominal `k` is never silently replaced by effective rank.
- No dependency on `peakanalysis`; the small formulas are implemented
  here independently.
- No JAX/MLX unified API, no map-wide per-pixel sweeps, no GUI/Web/CLI
  exposure of interim versions.
- The 2026-08-01 public release does not wait for this feature.
