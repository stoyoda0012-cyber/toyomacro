# Golub–Pereyra Jacobian verification — raw vs Kaufman vs full GP

Verification study of the post-AP Newton refinement Jacobian in
`toyomacro.voigtfit.multipeak_solver`, comparing the production approximation
("raw") against the Kaufman projection and the full Golub–Pereyra
reduced-residual Jacobian for the separable (VarPro) Voigt problem

    a*(θ) = argmin_a ‖y − Φ(θ)a‖²,   r(θ) = y − Φ(θ)a*(θ).

Date: 2026-07-16. Machine: M3 Max. All numbers reproducible from the
generator commands listed below. The production default was NOT changed.

---

## 1. Audit of the current implementation

Source: `_post_ap_newton_refine_chunk` / `_post_ap_newton_refine_mlx`
(`multipeak_solver.py`).

* **Jacobian**: model-side columns `S_(2k) = a_k·J_c,k`, `S_(2k+1) = a_k·J_σ,k`
  — the **raw** mode of this study. The amplitude-variation (projection) term
  of VarPro is absent in both the frozen-surrogate and the exact
  (`newton_exact=True`) paths. Terminology: "raw" names this JACOBIAN
  approximation, not a Hessian variant — in particular it is unrelated to the
  `H_raw` variable in the MLX diagnostics path (the Hessian captured before
  the Tikhonov load). Because `ΦᵀR = 0` at the LLS optimum, all three modes
  share the same gradient; they differ only in the Gauss-Newton curvature.
* **Sign system**: residual `R = Y − Σ_k a_k Φ̂_k`; `g = SᵀR`;
  `H = SᵀS + 1e-6·tr(H)·I`; solve `H·Δ = g`; `θ ← θ + Δ`. Equivalent to
  Gauss–Newton in the reduced convention with `J = ∂r/∂θ = −S`
  (verified: `test_raw_step_equals_production_normal_equations`).
* **Amplitude solve**: unconstrained LLS (Gram + jitter 1e-10, Cramer for
  n≤2). Held FIXED during the step; re-solved on the updated basis after it.
* **Safeguard**: per-axis step clip `±clip_step_ratio(=0.5)·grid step`.
  No line search, no LM damping.
* **`newton_exact=True`** = re-evaluate exact basis + analytic Jacobian each
  iteration (removes the frozen-surrogate linearization error). It is NOT a
  GP Jacobian — the projection and second GP terms are still absent.
* **Background** (`bg_degree≥0`): MLX path only; `[Φ|B]` joint LLS re-solved
  every iteration, i.e. the BG columns are handled by re-solving, not by a
  projected Jacobian.
* **SO doublets**: composite structured column
  `Φ_k = V(c) + (1/BR)·V(c+so_split)` with composite derivatives — already
  the correct structured treatment; mirrored in the reference implementation.

## 2. What was built

| artifact | purpose |
|---|---|
| `gp_reference.py` | float64 QR-based reference: raw / kaufman / golub_pereyra reduced Jacobians, FD Jacobian, GN with common LM / backtracking / native safeguards. No `inv`, no explicit projector (`Pv = Q(Qᵀv)`, `(Φ⁺)ᵀq = QR⁻ᵀq`). Doublets, fixed background columns and Tikhonov (explicit row augmentation) supported and tested. |
| `tests/test_gp_jacobian.py` | 23 tests: FD correctness battery + structural identities + production-hook regression (all pass). |
| `benchmarks/gp_newton_comparison.py` | Phase-4 experiment harness (blocks A–G below). `python -m toyomacro.voigtfit.benchmarks.gp_newton_comparison [--quick] [--out f.md]` |
| `multipeak_solver.py` | EXPERIMENTAL `newton_jacobian_mode="raw"|"kaufman"|"golub_pereyra"` on `solve_alternating_projection` (default `"raw"` = byte-identical fast path; non-raw runs the frozen-surrogate step on CPU). |

## 3. Correctness (Phase 3) — all criteria met

Central differences of the FULL reduced residual (complete amplitude re-solve
on both sides; no dictionary argmax, no clipping inside the differencing):

* **GP vs FD relative Frobenius error < 1e-5** on: 1-peak (center, sigma),
  2-peak separated and strongly overlapped, SO doublet, 5-component,
  non-zero residual, near-solution. With background columns and with
  Tikhonov (augmented system) as well.
* Near-rank-deficient case (0.008σ separation, cond(Gram) ≈ 1.8e5):
  agreement < 1e-3 (FD cancellation + a*(θ) sensitivity dominate; documented).
* **h-refinement**: FD error drops ≳20× when h shrinks 10× (O(h²) regime
  confirmed) — the 1e-5 agreement is truncation-limited, not coincidental.
* **Gradient identity**: `JᵀR` identical for raw / kaufman / GP to 1e-8·‖g‖
  (consequence of ΦᵀR = 0 — the three modes differ only in curvature).
* **Kaufman↔GP gap linear in r for fixed basis and derivatives** (the GP
  second term is linear in the residual when Φ, a* and ∂Φ are held fixed;
  confirmed exactly under the controlled residual-scaling experiment —
  same noise realization, scaled amplitude); gap = 0 at zero residual.
  Across ITERATIONS the basis and derivatives move too, so the gap is not
  governed by ‖r‖ alone.
* **Production hook cross-check**: the float32 batched chunk implementation
  matches a per-spectrum float64 explicit-projector reimplementation, and
  kaufman ≡ golub_pereyra on data built exactly from the surrogate basis.

## 4. Newton comparison (Phase 4) — key findings

Full tables: run the harness (20 trials/cell shown here; identical seeds
across modes → paired comparisons). Blocks A–C use a COMMON LM safeguard;
D is native undamped GN; F mirrors the production clipped 1/3-step polish.

### 4.1 Where Kaufman/GP clearly win

* **Convergence rate**: at ≥2σ separation they converge in 4–9 LM iterations
  where raw needs 30–50+ (and often still fails the SSR-floor criterion):
  noiseless 2comp-mid: kaufman/GP reach machine precision (RMSE ~1e-13) in 6
  iterations; raw is still at 4e-2 eV after 50. This is the classic VarPro
  result: raw omits the amplitude-response terms of the reduced-residual
  curvature (it remains a legitimate approximate curvature of the JOINT
  (a, θ) model), so along the amplitude-coupled directions its GN model
  mismatches the reduced objective, giving
  slow linear convergence near the solution.
* **Convergence basin** (moderate overlap): 2comp-mid at 10-cell (≈1σ) initial
  error: GP 85–100% vs raw 20%. doublet2 at 10-cell: GP 100% vs raw 40%.
* **Production-like 3-step polish (block F3)**: 2comp-mid and doublet2:
  kaufman/GP 100% success and 2–3× lower parameter RMSE vs raw 25–55%.
  Even the single clipped step (F1) contracts the error visibly faster.
  In the production-path measurement the single GP step also reached a lower
  median χ² than the raw step (99.43 vs 100.0 at 2comp; 94.61 vs 95.59 at
  5comp, same AP seed).
* **Weak/zero components (block C)**: GP is the most stable of the three
  (2comp one-small: amp error 0.41 vs raw 1.21 / kaufman 1.97; 5comp
  one-zero: GP 0.76 vs kaufman 4.29). The second GP term matters exactly
  when the residual is non-negligible — Kaufman alone is the fragile variant.

### 4.2 Where raw is better — and why production survived without GP

* **Strong overlap (≤1σ) at realistic noise**: all modes reach the SSR floor,
  but kaufman/GP walk along the χ²-flat valley to the (high-variance) ML
  minimizer: parameter RMSE 2.0e-1 vs raw 2.9e-2 at 2comp-strong/0.5cell.
  Accurate optimization of the least-squares objective is NOT the same thing
  as a low-variance physical estimate in this regime. raw's behavior is
  consistent with an implicit regularization; the iteration-budget ablation
  (§4.3) narrows down its origin. Negative-amplitude rates rise
  correspondingly for kaufman/GP.
* **Many overlapping components without damping**: undamped (block D) and
  clipped-undamped (block F, 5comp/10comp) projected steps oscillate or
  diverge: 5comp F3: raw 80–100% vs kaufman 0% (div 60–65%), GP 0–10%
  (div 15–20%). Kaufman occasionally runs away completely (parameter error
  1e5–1e9 with cond(Gram) 1e24+ — component escape/collapse). **Projected
  Jacobians require LM/trust-region globalization; the production
  clip-only single step is exactly the regime where raw is the safe choice
  at n_comp ≥ ~5 overlapping.**
* raw never produced an objective increase in blocks A–C (obj+ = 0
  throughout); its failures are "too slow", not "explodes".

### 4.3 Iteration-budget ablation (block G) — where raw's low variance comes from

Candidate explanations for §4.2 were confounded a priori: raw curvature
itself, step clipping, few iterations, proximity to the dictionary init,
the stopping rule, or the LM schedule. Block G fixes everything except the
iteration budget (common LM, NO step clip, paired seeds; note blocks A–C
also ran without clipping, which already excludes the clip):

| case (noise 1%) | mode | it=2 | it=10 | it=50 | it=200 |
|---|---|---|---|---|---|
| 2comp-strong cRMSE (eV) | raw | 2.8e-2 | 3.2e-2 | 3.2e-2 | 3.2e-2 |
| | golub_pereyra | 7.3e-2 | 2.2e-1 | 2.55e-1 (conv. ~19 it) | 2.55e-1 |
| 2comp-degen cRMSE (eV) | raw | 5.0e-2 | 1.0e-1 | 1.0e-1 | 1.0e-1 |
| | golub_pereyra | 8.6e-2 | 1.4e-1 | 1.7e-1 (conv. ~42 it) | 1.7e-1 |

Both modes sit within 1.05× of the SSR floor at every budget (success 100%);
the parameter gap is purely placement along the χ²-flat valley.

* GP's parameter error **grows monotonically with budget** until it converges
  at the ML point — it is accurately descending the flat valley.
* raw's parameter error is **budget-independent from 2 to 200 iterations** —
  its along-valley motion is glacially slow (it exhausts even 200 LM
  iterations without leaving the dictionary-seeded neighborhood). This rules
  out "few iterations" and "clipping" as the cause and points at the raw
  curvature itself: `SᵀS` overestimates stiffness in exactly the directions
  the amplitude re-solve would compensate, freezing the valley coordinate.
  Functionally this IS an early-stopping-type regularizer, but one induced
  by the curvature model, not by the iteration limit or safeguards.
* Remaining unresolved confound: both modes shared one LM schedule
  (λ₀=1e-3, ×4/÷3); a trust-region or different schedule could shift the
  quantitative gap, though not the monotonic-vs-flat qualitative split.

### 4.4 Prior hypotheses vs outcomes

1. "1–2 comp with good dict2D init → differences small": **partially true**
   for final accuracy at ≥3.5σ separation, but kaufman/GP still cut the
   3-step polish error 2–3× and the iteration count 5–8×.
2. "Kaufman more stable than raw on overlapped multi-component": **rejected**
   — the opposite. Kaufman is the least stable mode in every hard block.
3. "GP second term small near solution": **confirmed** (linear in r at
   fixed basis/derivatives, verified by the controlled residual-scaling
   experiment; §3).
4. "Coarse init → GP widens the basin": **confirmed at moderate overlap**
   (2comp-mid, doublet2), **rejected at strong overlap / many components**
   where the wider steps land in the flat valley or diverge.
5. "GP as fallback rather than default": **supported** — see recommendation.

## 5. Cost

Reference (float64, single spectrum, per-Jacobian call): kaufman 1.0–1.3×,
GP 1.4–2.2× raw (n_comp 1→10).

Production path (float32 batched, 121 energies, newton step only):

| n_comp | batch | raw (MLX) | kaufman (CPU) | GP (CPU) | raw (CPU) |
|---|---|---|---|---|---|
| 2 | 100K | 0.37 s | 0.65 s | 0.74 s | 0.48 s |
| 5 | 50K  | 0.76 s | 1.00 s | 1.09 s | 0.94 s |

CPU-vs-CPU the GP overhead is 1.25–1.6×: one batched (n×n) Gram solve with
2–3·n_params right-hand sides plus 2–3 batched (n_E×n) matvecs per column.
The same structure would map to MLX matmuls if ever promoted.

## 6. Known limitations

* The production-hook (non-raw) step operates on the frozen grid-cell Taylor
  surrogate, so it verifies the projection/second-term physics but inherits
  the surrogate's linearization ceiling; the exact-model comparison lives in
  `gp_reference` (float64, scipy wofz).
* Non-raw modes are CPU-only and incompatible with `newton_exact`,
  `fit_gamma`, `bg_degree≥0`, `quality_flags` (guarded by ValueError).
* Success in the harness is SSR-based; on flat valleys a run can "succeed"
  with poor parameters — the matched RMSE columns carry that information,
  and negative-amplitude rates are reported, not clipped (NNLS is
  deliberately out of scope).
* LM is the only globalization tested; a trust region could change the
  block-D/F balance for the projected modes.
* fixed per-block seeds → paired but not exhaustive sampling (20 trials/cell).

## 7. Recommendation

**Keep `raw` as the production default.** The evidence says the choice is
regime-dependent, which fits a fallback/tiered design rather than a swap:

1. **Worth keeping (done)**: `newton_jacobian_mode` flag as a RESEARCH HOOK —
   zero cost when unused, default byte-identical. It should stay experimental
   and undocumented as a user-facing feature: it is CPU-only and incompatible
   with `newton_exact` / `fit_gamma` / `bg_degree≥0` / `quality_flags`.
2. **GP is a fallback for *identifiable-but-slow* cases, not for "hard cases"
   in general.** A blanket "anomaly → GP" rule would be wrong: when the
   anomaly stems from non-identifiability (χ²-flat valley), GP descends that
   valley *more* accurately and makes the parameter estimate worse (§4.2–4.3).
   The routing should gate on identifiability first:

   | condition | route |
   |---|---|
   | moderate cond(Gram) + raw stalled / slow | GP + LM damping |
   | coarse initialization, peaks identifiable | GP + LM damping |
   | high cond(Gram) / strong overlap | NOT GP — hierarchical grouping (`auto_group` / `solve_grouped`), ridge/prior, tighter constraints |
   | zero or surplus components suspected | component pruning or NNLS (out of scope of this study) |
   | background mismatch | re-estimate background (`bg_degree`), not a Jacobian change |

   cond(Gram) is already computed per spectrum by the `quality_flags`
   diagnostics, so the gate is cheap. With that gate, GP+LM costs 1.3–1.6×
   the raw step on the routed subset only.
3. **Kaufman alone is NOT worth promoting**: it shares GP's fragility in the
   hard regimes without GP's robustness for weak components; the second GP
   term is cheap (one extra Gram RHS per parameter) — if projecting at all,
   use full GP.
4. **Do not switch n_comp ≥ ~5 overlapping ladders to projected modes without
   adding LM/trust-region globalization** to the production Newton step; the
   current clip-only single step is exactly where raw is safest — and per
   §4.3 raw's stability there is a property of its curvature, not of the
   clip, so it survives even with the safeguard removed.
5. MLX promotion of GP is justified only if (2) is adopted for large routed
   fractions; the batched-matmul structure is ready but the measured CPU cost
   on anomaly-sized subsets does not require it.

## Reproduction

```bash
# correctness
uv run pytest src/toyomacro/voigtfit/tests/test_gp_jacobian.py -q
# comparison tables (20 trials/cell; --quick for 5)
uv run python -m toyomacro.voigtfit.benchmarks.gp_newton_comparison --out /tmp/gp_full.md
```
