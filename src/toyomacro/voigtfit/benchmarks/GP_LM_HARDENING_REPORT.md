# Full-GP hardening: compact Hessian + LM safeguard + MLX hybrid

Phase report on hardening the full Golub-Pereyra (GP) Newton path of the
multipeak solver: a compact-Hessian formulation, a Levenberg-Marquardt
safeguard, and an MLX-hybrid batched layer. Date: 2026-07-16, M3 Max.
Production default (`raw`) unchanged and byte-identical; every non-raw
path is opt-in via `newton_jacobian_mode` (internal experimental API —
no public API reshuffle yet).

## Deliverable map

| work item | artifact | status |
|---|---|---|
| Phase 0 audit | this report §1 | done |
| compact Hessian reference | `gp_compact.py` | done |
| compact vs explicit tests | `tests/test_gp_compact.py` (35) | pass |
| LM/trust-region + tests | `gp_compact.py` + `_gp_lm_chunk` + `tests/test_gp_lm.py` (15) | pass |
| frozen-surrogate MLX compact GP | `_gp_lm_chunk_mlx` (hybrid) | done |
| MLX vs NumPy tests | parity tests in `test_gp_lm.py` | pass |
| fixed linear background | `_gp_lm_chunk(bg_design=…)` + API `bg_degree` with `gp_lm` | done |
| quality diagnostics | `GPLMDiagnostics` (+ `MultiPeakResult.gp_diagnostics`) | done |
| exact-basis GP (Phase 7) | float64 experiment only (see §5) | judged — not promoted |
| `fit_gamma` GP (Phase 8) | not attempted (gated on Phase 7) | deferred |
| auto routing | experiment in `bench_gp_lm.py` | measured |
| benchmark | `benchmarks/bench_gp_lm.py` | done |

## 1. Phase 0 audit (delta vs the previous verification session)

* The existing non-raw hook (`_post_ap_newton_refine_chunk` kaufman/GP
  modes) materialises the projected columns M_p — correct but O(p) extra
  (batch, n_E) buffers. It remains as the explicit reference; the new
  compact path never builds those columns.
* Sign system everywhere: `H·Δ = g`, `g = SᵀR`, `θ ← θ+Δ`. The compact raw
  Hessian is literally `K = SᵀS`.
* Numerical jitter (Gram 1e-10/1e-8) vs solve stabilisation (1e-6·tr on H)
  vs objective terms (γ prior) are distinct in the existing code and stay
  distinct here: `compact_normal_equations` returns the un-stabilised DATA
  Hessian; the LM load is applied only inside `lm_step`.
* `Aᵀr = 0` holds to ~1e-12 (relative) with the jitter-free QR reference
  and to float32 accuracy in the production layer — the shared-gradient
  identity is enforced by re-solving amplitudes on the current basis.
* Reusable solvers: `_spd_cholesky_solve_entries` (unrolled GPU Cholesky,
  m ≤ 24), `_gram_solve_mx`. The Hessian cache is frozen-surrogate/raw-only
  and is correctly bypassed by GP (its curvature is residual-dependent).

## 2. Compact formulation (Phases 1-2) — verified

    G = AᵀA,  C = AᵀS,  K = SᵀS,  Q[:,j] = D_jᵀr
    H_K  = K − CᵀG⁻¹C
    H_GP = H_K + QᵀG⁻¹Q          (orthogonal split of M_j; cross terms 0)
    g    = Sᵀr                    (mode-independent at the LLS optimum)

One multi-RHS solve `solve(G, [C|Q])`, no inverse, no projector,
`H ← (H+Hᵀ)/2` at the end.

Test results (35 tests): compact H, g, Newton step and predicted reduction
match the explicit projected-Jacobian reference to **≤ 1e-9 relative**
(float64, well-conditioned) across 1/2/5 components, strong overlap, SO
doublet, background columns, small/zero amplitudes, near-zero residual and
the Tikhonov-augmented system. Near-rank-deficient (cond(G) ≈ 2e5):
agreement scales with conditioning as expected (tolerance `1e-9 +
1e-14·cond`, reported not hidden). `H_GP − H_K ⪰ 0` verified;
Kaufman ≡ GP at zero residual; raw ≡ K.

## 3. LM safeguard (Phase 3) — semantics verified

Scaled-diagonal LM (`d_i = max(H_ii, ε·mean diag)` — Marquardt, not
identity-λ), `pred = gᵀΔ − ½ΔᵀHΔ` recomputed after the production-style
clip, `ared` from a FULL re-evaluation (basis → joint LLS → residual),
accept iff `pred>0 ∧ ared>0 ∧ ρ>η ∧ finite ∧ cond-gate`; λ ×4 on rejection
/ ρ<0.25, ÷3 on ρ>0.75; ≤ `max_retries`(3) escalations; rejected spectra
keep the AP seed **bit-for-bit** (tested). Per-spectrum λ and accept masks;
trials are evaluated full-batch and committed under masks (a deliberate
simplicity trade; see §9). All thresholds live in `GPLMConfig`, no magic numbers.

Robustness: non-finite input spectra are quarantined before the batched
LAPACK calls (they would otherwise poison the whole chunk), never accepted,
outputs remain the finite seed, `fallback_reason=3`. Condition gate
(`cond(G) ≥ cond_gate`) refuses to run GP and reports `fallback_reason=1`.
`step_accepted ⇔ fallback_reason==0` is asserted as a contract.

## 4. MLX hybrid (Phase 4)

Division of labour: GPU does the O(batch·n_E) work (surrogate basis, Gram,
S/C/K/Q/g reductions, trial residuals as batched matmuls); CPU float64 does
the (n_lin)/(p) linear algebra and the LM bookkeeping — identical logic to
the CPU layer, so only (batch × small) buffers cross the boundary and the
LM semantics cannot drift between backends. Parity test: accept-mask
disagreement ≤ 2% (razor-edge ρ in float32), accepted states match to
float32 tolerance, commonly-rejected seeds identical.

§7.2 stacked-vs-entry-wise and end-to-end timings: see benchmark tables
(§6). Speed verdict in §7.

## 5. Phase 7 (exact basis) and Phase 8 (fit_gamma) — stop-gate decisions

**Phase 7**: instead of building the full MLX exact-GP path first, the
benchmark runs the float64 exact-basis GP-LM (`gp_reference.run_newton`,
QR, LM, unclipped, max_iter=4) on a 1000-spectrum subset as the quality
ceiling. Findings (full tables in §6):

* At separable/moderate overlap the exact basis clearly improves δσ over
  the frozen surrogate (the known surrogate δσ ceiling) — a factor ~2–4 in
  sRMSE at 2comp sep/mid.
* At weak-amplitude components the UNCLIPPED exact GP walks the flat
  valley and is substantially WORSE than the clipped frozen GP-LM — the
  Block-G phenomenon again; an exact-GP production tier would need the
  same clip/gate discipline (this comparison confounds clip with basis
  exactness; noted honestly).
* Verdict: exact GP stays a **reference/experiment mode**. The MLX
  exact-GP integration (per-retry Faddeeva re-evaluation) is NOT built
  now: its value concentrates in a routed, identifiable subset where the
  frozen GP-LM already captures most of the χ² gain, and the added cost
  is per-retry stacked Faddeeva evaluations. Revisit if the routed tier
  is adopted and δσ accuracy on that tier becomes the binding constraint.

**Phase 8** (`fit_gamma` + GP): not attempted, by the planned phase
ordering — γ has no frozen-grid Jacobian, so GP-γ REQUIRES the exact path that
Phase 7 declined to promote. The existing raw fit_gamma machinery (γ
prior + χ² gate) remains the production route. Deferred, not failed.

## 6. Benchmark results

(model M3 Max; newton_iter=2; medians; generator:
`python -m toyomacro.voigtfit.benchmarks.bench_gp_lm --out …`)

### 6.1 Solver comparison (identical AP seeds, newton_iter=2)

Reading guide: raw rows are the production kernel; gplm rows add the LM
safeguard; `evals` counts full trial re-evaluations (basis+LLS+residual).
The gplm-exact* rows are float64/QR/UNCLIPPED — quality ceiling only.

| case | solver | newton ms | M/s | med chi2 | cRMSE(eV) | sRMSE(eV) | note |
|---|---|---|---|---|---|---|---|
| 2c-sep4/10K | AP-seed | - | - |       371 | 2.69e-02 | 3.32e-02 | |
| 2c-sep4/10K | raw-MLX |    151.5 |   0.07 |      67.9 | 1.37e-02 | 2.17e-02 |  |
| 2c-sep4/10K | raw-CPU x2 |     77.3 |   0.13 |      67.9 | 1.37e-02 | 2.17e-02 |  |
| 2c-sep4/10K | gp-CPU x2 |    130.4 |   0.08 |      67.1 | 8.33e-03 | 1.76e-02 |  |
| 2c-sep4/10K | gplm-CPU |    208.0 |   0.05 |      67.1 | 8.33e-03 | 1.76e-02 | acc 100% evals 7 |
| 2c-sep4/10K | gplm-MLX |    107.4 |   0.09 |      67.1 | 8.33e-03 | 1.76e-02 | acc 100% evals 7 |
| 2c-sep4/10K | gplm-exact* |    813.7 |   0.00 | (n=1000) | 3.81e-03 | 4.44e-03 | float64 ceiling |
| 2c-mid2/10K | AP-seed | - | - |  2.01e+03 | 1.33e-01 | 9.52e-02 | |
| 2c-mid2/10K | raw-MLX |      6.8 |   1.48 |       375 | 8.16e-02 | 7.21e-02 |  |
| 2c-mid2/10K | raw-CPU x2 |     80.0 |   0.12 |       375 | 8.16e-02 | 7.21e-02 |  |
| 2c-mid2/10K | gp-CPU x2 |    132.6 |   0.08 |       408 | 9.15e-02 | 7.21e-02 |  |
| 2c-mid2/10K | gplm-CPU |    205.0 |   0.05 |       408 | 9.14e-02 | 7.21e-02 | acc 100% evals 7 |
| 2c-mid2/10K | gplm-MLX |    114.5 |   0.09 |       408 | 9.14e-02 | 7.21e-02 | acc 100% evals 7 |
| 2c-mid2/10K | gplm-exact* |    816.8 |   0.00 | (n=1000) | 6.28e-02 | 2.38e-02 | float64 ceiling |
| 2c-strong1/10K | AP-seed | - | - |  3.24e+03 | 2.85e-01 | 1.03e-01 | |
| 2c-strong1/10K | raw-MLX |      8.6 |   1.17 |       244 | 1.52e-01 | 7.84e-02 |  |
| 2c-strong1/10K | raw-CPU x2 |     82.8 |   0.12 |       244 | 1.52e-01 | 7.84e-02 |  |
| 2c-strong1/10K | gp-CPU x2 |    131.9 |   0.08 |       292 | 1.70e-01 | 7.86e-02 |  |
| 2c-strong1/10K | gplm-CPU |    146.6 |   0.07 |       291 | 1.70e-01 | 7.86e-02 | acc 100% evals 4 |
| 2c-strong1/10K | gplm-MLX |     73.6 |   0.14 |       291 | 1.70e-01 | 7.86e-02 | acc 100% evals 4 |
| 2c-strong1/10K | gplm-exact* |    947.4 |   0.00 | (n=1000) | 1.75e-01 | 5.45e-02 | float64 ceiling |
| 2c-small-amp/10K | AP-seed | - | - |  1.59e+03 | 4.49e-01 | 9.84e-02 | |
| 2c-small-amp/10K | raw-MLX |      8.5 |   1.17 |       192 | 5.69e-01 | 7.59e-02 |  |
| 2c-small-amp/10K | raw-CPU x2 |     79.5 |   0.13 |       192 | 5.69e-01 | 7.59e-02 |  |
| 2c-small-amp/10K | gp-CPU x2 |    132.1 |   0.08 |       193 | 5.44e-01 | 7.63e-02 |  |
| 2c-small-amp/10K | gplm-CPU |    206.8 |   0.05 |       193 | 5.45e-01 | 7.63e-02 | acc 100% evals 7 |
| 2c-small-amp/10K | gplm-MLX |    105.9 |   0.09 |       193 | 5.45e-01 | 7.63e-02 | acc 100% evals 7 |
| 2c-small-amp/10K | gplm-exact* |   1052.9 |   0.00 | (n=1000) | 6.26e-01 | 3.37e-01 | float64 ceiling |
| 2c-bg1/10K | AP-seed | - | - |   1.6e+03 | 1.28e-01 | 6.90e-02 | |
| 2c-bg1/10K | gplm-CPU |    306.3 |   0.03 |       140 | 2.95e-02 | 5.26e-02 | acc 100% evals 7 |
| 2c-bg1/10K | gplm-MLX |    145.2 |   0.07 |       140 | 2.95e-02 | 5.26e-02 | acc 100% evals 7 |
| 2c-bg1/10K | gplm-exact* |    851.1 |   0.00 | (n=1000) | 1.05e-02 | 8.81e-03 | float64 ceiling |
| 5c-1.5/10K | AP-seed | - | - |       313 | 1.13e-01 | 5.12e-02 | |
| 5c-1.5/10K | raw-MLX |     36.1 |   0.28 |       223 | 1.13e-01 | 5.15e-02 |  |
| 5c-1.5/10K | raw-CPU x2 |    324.0 |   0.03 |       223 | 1.13e-01 | 5.15e-02 |  |
| 5c-1.5/10K | gp-CPU x2 |    422.6 |   0.02 |       449 | 1.46e-01 | 4.86e-02 |  |
| 5c-1.5/10K | gplm-CPU |    692.0 |   0.01 |       227 | 1.11e-01 | 5.06e-02 | acc 100% evals 10 |
| 5c-1.5/10K | gplm-MLX |    342.2 |   0.03 |       227 | 1.11e-01 | 5.06e-02 | acc 100% evals 10 |
| 5c-1.5/10K | gplm-exact* |   2203.3 |   0.00 | (n=1000) | 1.79e-01 | 2.70e-01 | float64 ceiling |
| 2c-sep4/100K | AP-seed | - | - |       373 | 2.70e-02 | 3.31e-02 | |
| 2c-sep4/100K | raw-MLX |     46.3 |   2.16 |      69.1 | 1.38e-02 | 2.18e-02 |  |
| 2c-sep4/100K | raw-CPU x2 |    850.3 |   0.12 |      69.1 | 1.38e-02 | 2.18e-02 |  |
| 2c-sep4/100K | gp-CPU x2 |   1388.4 |   0.07 |      68.3 | 8.30e-03 | 1.75e-02 |  |
| 2c-sep4/100K | gplm-CPU |   2267.3 |   0.04 |      68.3 | 8.30e-03 | 1.75e-02 | acc 100% evals 14 |
| 2c-sep4/100K | gplm-MLX |    896.5 |   0.11 |      68.3 | 8.30e-03 | 1.75e-02 | acc 100% evals 14 |
| 2c-sep4/100K | gplm-exact* |    829.6 |   0.00 | (n=1000) | 3.85e-03 | 4.60e-03 | float64 ceiling |
| 2c-mid2/100K | AP-seed | - | - |  2.02e+03 | 1.32e-01 | 9.54e-02 | |
| 2c-mid2/100K | raw-MLX |     36.3 |   2.76 |       381 | 8.17e-02 | 7.23e-02 |  |
| 2c-mid2/100K | raw-CPU x2 |    863.8 |   0.12 |       381 | 8.17e-02 | 7.23e-02 |  |
| 2c-mid2/100K | gp-CPU x2 |   1389.5 |   0.07 |       414 | 9.19e-02 | 7.23e-02 |  |
| 2c-mid2/100K | gplm-CPU |   2287.3 |   0.04 |       414 | 9.17e-02 | 7.23e-02 | acc 100% evals 14 |
| 2c-mid2/100K | gplm-MLX |    896.6 |   0.11 |       414 | 9.17e-02 | 7.23e-02 | acc 100% evals 14 |
| 2c-mid2/100K | gplm-exact* |    833.1 |   0.00 | (n=1000) | 6.65e-02 | 2.50e-02 | float64 ceiling |
| 2c-strong1/100K | AP-seed | - | - |   3.2e+03 | 2.84e-01 | 1.03e-01 | |
| 2c-strong1/100K | raw-MLX |     35.2 |   2.84 |       250 | 1.52e-01 | 7.85e-02 |  |
| 2c-strong1/100K | raw-CPU x2 |    858.3 |   0.12 |       250 | 1.52e-01 | 7.85e-02 |  |
| 2c-strong1/100K | gp-CPU x2 |   1393.3 |   0.07 |       296 | 1.70e-01 | 7.87e-02 |  |
| 2c-strong1/100K | gplm-CPU |   1594.5 |   0.06 |       296 | 1.69e-01 | 7.87e-02 | acc 100% evals 8 |
| 2c-strong1/100K | gplm-MLX |    635.7 |   0.16 |       296 | 1.69e-01 | 7.87e-02 | acc 100% evals 8 |
| 2c-strong1/100K | gplm-exact* |    959.9 |   0.00 | (n=1000) | 1.83e-01 | 9.09e-02 | float64 ceiling |
| 2c-small-amp/100K | AP-seed | - | - |  1.55e+03 | 4.48e-01 | 9.86e-02 | |
| 2c-small-amp/100K | raw-MLX |     35.8 |   2.79 |       192 | 5.69e-01 | 7.60e-02 |  |
| 2c-small-amp/100K | raw-CPU x2 |    856.8 |   0.12 |       192 | 5.69e-01 | 7.60e-02 |  |
| 2c-small-amp/100K | gp-CPU x2 |   1385.0 |   0.07 |       193 | 5.43e-01 | 7.64e-02 |  |
| 2c-small-amp/100K | gplm-CPU |   2057.6 |   0.05 |       193 | 5.43e-01 | 7.64e-02 | acc 100% evals 12 |
| 2c-small-amp/100K | gplm-MLX |    803.1 |   0.12 |       193 | 5.43e-01 | 7.64e-02 | acc 100% evals 12 |
| 2c-small-amp/100K | gplm-exact* |   1066.2 |   0.00 | (n=1000) | 5.81e-01 | 5.54e-01 | float64 ceiling |
| 2c-bg1/100K | AP-seed | - | - |  1.61e+03 | 1.28e-01 | 6.91e-02 | |
| 2c-bg1/100K | gplm-CPU |   3315.4 |   0.03 |       146 | 2.97e-02 | 5.27e-02 | acc 100% evals 14 |
| 2c-bg1/100K | gplm-MLX |   1208.1 |   0.08 |       146 | 2.97e-02 | 5.27e-02 | acc 100% evals 14 |
| 2c-bg1/100K | gplm-exact* |    868.1 |   0.00 | (n=1000) | 1.13e-02 | 9.43e-03 | float64 ceiling |
| 5c-1.5/100K | AP-seed | - | - |       319 | 1.13e-01 | 5.12e-02 | |
| 5c-1.5/100K | raw-MLX |    145.1 |   0.69 |       228 | 1.14e-01 | 5.15e-02 |  |
| 5c-1.5/100K | raw-CPU x2 |   3448.6 |   0.03 |       228 | 1.14e-01 | 5.15e-02 |  |
| 5c-1.5/100K | gp-CPU x2 |   4477.7 |   0.02 |       461 | 1.47e-01 | 4.86e-02 |  |
| 5c-1.5/100K | gplm-CPU |   7120.7 |   0.01 |       233 | 1.11e-01 | 5.06e-02 | acc 100% evals 20 |
| 5c-1.5/100K | gplm-MLX |   3114.5 |   0.03 |       233 | 1.11e-01 | 5.06e-02 | acc 100% evals 20 |
| 5c-1.5/100K | gplm-exact* |   2243.8 |   0.00 | (n=1000) | 6.81e-01 | 2.73e-01 | float64 ceiling |

### Assembly: stacked matmul vs entry-wise (K = SᵀS)

| p | backend | stacked ms | entrywise ms |
|---|---|---|---|
| 4 | numpy | 51.2 | 29.7 |
| 4 | mlx | 8.7 | 3.7 |
| 10 | numpy | 113.9 | 183.9 |
| 10 | mlx | 21.9 | 14.0 |
| 20 | numpy | 200.3 | 743.3 |
| 20 | mlx | 35.9 | 54.9 |

### Auto-routing experiment (mixed batch, n=20000 per population)

| pop | strategy | cRMSE(eV) | sRMSE(eV) | med chi2 | note |
|---|---|---|---|---|---|
| mid | raw only | 8.15e-02 | 7.21e-02 | 375 | - |
| mid | auto (route+gate) | 8.50e-02 | 6.90e-02 | 354 | routed 12%, acc 100% |
| mid | gp_lm all | 9.16e-02 | 5.11e-02 | 202 | acc 100% |
| hard | raw only | 1.97e-01 | 7.83e-02 | 317 | - |
| hard | auto (route+gate) | 2.03e-01 | 7.55e-02 | 315 | routed 16%, acc 100% |
| hard | gp_lm all | 2.08e-01 | 5.84e-02 | 196 | acc 100% |


### 6.2 Background head-to-head (2c-bg1, 10K, public API, newton_iter=2)

| solver | newton ms | med chi2 | cRMSE(eV) | sRMSE(eV) |
|---|---|---|---|---|
| raw-MLX (bg re-solve) | 19.6 | 152 | 3.36e-02 | 5.91e-02 |
| gp_lm-MLX (bg in projection) | 131.7 | 140 | 2.95e-02 | 5.26e-02 |

### 6.3 Reading of the numbers

* **Cost**: gplm-MLX ≈ 2.5× faster than gplm-CPU, but 15–25× the raw-MLX
  kernel (0.6–1.2 s vs 36–46 ms per 100K at 2 comp). Fine for a routed
  subset (at ≤16% routed the pipeline overhead is ≤4×), not a full-batch
  replacement.
* **Quality**: GP-LM beats raw where components are identifiable
  (sep4: cRMSE 8.3e-3 vs 1.38e-2) and with background in the projection
  (§6.2). At strong overlap the two clipped raw steps still reach a lower
  χ² and cRMSE than two GP-LM iterations from the same seed (250 vs 296;
  0.152 vs 0.170) — the LM safeguard prevents divergence but does not
  make GP the better 2-step polisher there.
* **Auto-routing caveat**: in §6.1's auto table the gp_lm rows are seeded
  from the RAW-refined state (raw + 2 GP iterations = 4 refine passes) —
  that is the actual `auto` composition, not a same-budget comparison.
  Within it, GP passes cut χ² (375→354 routed-only, →202 everywhere) and
  σ error (7.2e-2→5.1e-2) while center error stays raw-level or slightly
  worse (valley effect).
* **Assembly (§7.2)**: entry-wise reductions win at p ≤ 4 (the n_comp ≤ 2
  production case, both backends); stacked matmuls win from p ≈ 10
  (numpy) / p ≈ 20 (MLX). The gp layer currently always stacks —
  switching to entry-wise for n_comp ≤ 2 is a cheap future optimization.
* **gplm-exact\*** (unclipped float64): large δσ gains on identifiable
  cases (sep4 sRMSE 4.4e-3 vs 1.76e-2), catastrophic on weak-amplitude
  and 5-comp cases (Block-G valley descent without clip) — see §5
  verdict.

## 7. Design questions and answers

1. **compact Hessian式は正しいか** — yes: ≤1e-9 vs explicit MᵀM (float64),
   float32 production layer ≤2e-3, PSD/zero-residual identities hold.
2. **MLX化の実測速度向上** — see §6; on this machine the hybrid's gain
   over CPU is modest at n_E=121 (the small-matrix CPU work and per-retry
   sync dominate); the MLX win grows with n_E·n_comp. Honest verdict: for
   anomaly-scale subsets the CPU path is already sufficient.
3. **GP-LMは安全になったか** — yes by construction and test: no accepted
   step increases the surrogate objective, rejects preserve the seed
   bit-for-bit, retries bounded, NaN quarantined, cond-gated.
4. **背景対応の効果** — fixed polynomial designs verified against the
   augmented reference; end-to-end bg cases converge with correct
   background recovery. Shirley/Tougaard are explicitly OUT of scope of
   this projection (documented in code).
5. **exact GPの追加価値** — real for δσ on identifiable spectra, negative
   for weak components without clipping; reference mode only (§5).
6. **fit_gamma対応の価値** — untested by design (gated on Phase 7);
   existing raw γ machinery unchanged.
7. **推奨fallback率・condition gate** — from the auto-routing experiment
   (§6.1 auto table, n=20,000 per population): route on
   `chi2 > 1.5×median` with `cond_gate=1e8`; observed routed fractions
   12% (identifiable pop.) / 16% (hard pop.), ~100% accept on routed
   spectra. Caveat: the same rule on the n=2,000 quick run routed 9% /
   36% — a median-relative threshold's routed fraction depends on the
   χ² distribution tail, which is itself dataset-dependent. This
   variability is part of why `auto` stays experimental; thresholds live
   in `GPLMConfig`, tune per dataset.
8. **rawをデフォルト維持すべきか** — yes. Unchanged, byte-identical,
   zero overhead when GP unused (all new work is behind the mode flag).
9. **`auto`を公開APIにする価値** — promising but premature: the routing
   signal (residual vs floor) and the gate work in the experiment, yet
   the measured end-to-end gain vs "gp_lm everywhere" is not decisive and
   the thresholds are dataset-dependent. Keep as an experiment recipe;
   revisit with real anomaly-tier data.

## 8. Non-negotiables checklist

raw default unchanged / no extra cost when unused (mode flag branch only)
/ no unconditional GP / no amplitude clipping, nothing called NNLS / no
explicit projector or inverse anywhere / correctness before optimization
(Phases 1-3 gated Phase 4) / rejected LM steps never adopted / objective
increase never reported as convergence / high-cond spectra gated out, not
forced / fit_gamma not attempted first.

## 9. Known limitations

* Non-raw modes remain incompatible with `newton_exact`, `fit_gamma`,
  `quality_flags` (ValueError-guarded). `bg_degree≥0` is now supported for
  `gp_lm` only (fixed designs).
* The MLX layer is a hybrid (small-matrix solves on CPU float64); a fully
  on-GPU LM loop (masked retries, `_spd_cholesky_solve_entries` for H)
  is the next optimization step IF a routed tier ever needs >1M-spectrum
  GP throughput.
* Trials are recomputed full-batch during retries (accepted spectra are
  masked out of commits but still computed) — a deliberate simplicity
  trade.
* The exact-GP experiment is float64/unclipped: its weak-amplitude failure
  overstates what a clipped exact tier would do.
* `param_std` is NOT computed by GP-LM; `cond_hessian`/`soft` reliability
  flags are the honest subset (a Gaussian-approximation std under a flat
  valley would be noise — see the GPLMDiagnostics docstring).
