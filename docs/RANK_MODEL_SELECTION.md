# Rank diagnostics & exact-K model selection (experimental)

**Status: experimental diagnostic API.** This layer answers *"how many
peak components does this spectrum support, and how sure can we be?"*
— it is **not an automatic peak-count decider**, and it is deliberately
not exported from `toyomacro.voigtfit.__init__`, not wired to the CLI,
GUI, or Web API. Import the modules explicitly (paths below). The
frozen evaluation design and its change history live in
`src/toyomacro/voigtfit/benchmarks/README_rank_model_selection.md`;
the measured acceptance record is
`benchmarks/results_rank_model_selection.json`.

The primary output is a **supported range plus evidence**, never a
single K: `supported / ambiguous / unsupported` with the reasons
(IC deltas, rank deficits, residual structure, negative amplitudes,
initialization dependence) attached.

## The three layers

| module | role |
|---|---|
| `toyomacro.voigtfit.rank_diagnostics` | four ranks of one structured problem: scaled linear basis, local full-parameter, VARPRO-profiled nonlinear, optional data-matrix — with full singular-value spectra, s²-weighted entropy rank, condition numbers |
| `toyomacro.voigtfit.model_selection` | pure IC scoring of already-fitted candidates: Gaussian profile-likelihood AIC/AICc/BIC, exact Poisson log-likelihood, strict free-parameter counting, NaN-free selection |
| `toyomacro.voigtfit.exact_k` | the discrete inverse problem: outer comparison over K = 1..k_max, inner VARPRO fit with exactly K structured components per candidate |

## Quick start

```python
import numpy as np
from toyomacro.voigtfit.exact_k import ExactKConfig, run_exact_k

report = run_exact_k(
    energy, intensity,                    # descending BE axes are fine
    ExactKConfig(
        sigma_init=0.45,                  # prior Gaussian width (eV)
        gamma=0.10,                       # fixed Lorentzian HWHM (eV)
        k_max=5,
        bg_degree=1,                      # None | 0 (const) | 1 (linear)
        so_split=1.8, branch_ratio=1.5,   # 0.0 = no SO doublet
    ),
    sigma_std=noise_std,                  # noise STANDARD DEVIATION
)

print(report.verdict)          # "supported" | "ambiguous" | "unsupported"
print(report.supported_k)      # IC-supported range, e.g. (2,)
print(report.verdict_reasons)  # why — always populated
json_blob = report.to_dict()   # fully JSON-serializable
```

Standalone rank diagnostics for a hypothesised placement:

```python
from toyomacro.voigtfit.multipeak_config import ComponentConfig
from toyomacro.voigtfit.rank_diagnostics import compute_rank_diagnostics

diag = compute_rank_diagnostics(
    energy,
    [ComponentConfig(center=99.5, sigma=0.4, gamma=0.1),
     ComponentConfig(center=100.6, sigma=0.4, gamma=0.1)],
    amplitudes=np.array([2.0, 1.0]),
    bg_degree=1,
    sigma_std=0.05,          # or variance= / expected_counts= (Poisson)
    shared_sigma=True,       # diagnose K+1 directions, matching a shared-width fit
)
print(diag.linear.rank_primary, diag.profiled_nonlinear.singular_values)
```

## What the report contains

Per candidate K (`report.candidates[i]`): the exact-K model and its
free-parameter count, AIC/AICc/BIC, fitted centers (always ascending) /
shared sigma / amplitudes / background coefficients, per-start RSS and
convergence, boundary flags (including per-component negative
amplitudes), rank diagnostics computed with ONE `ParameterScales`
shared across all K, and whitened-residual lag-1 autocorrelation.

Globally: `selection` (delta tables, best-by-criterion), `supported_k`,
`verdict` + `verdict_reasons`, per-K `evidence`,
`negative_amplitude_k`, the echoed configuration (every threshold is in
the output), and the background record.

A `supported` verdict requires ALL of: AICc and BIC naming the same
best K; that K being the single both-deltas-within-threshold candidate;
nominal-dof rank support; no negative amplitudes; no systematic
residual structure. Anything less is `ambiguous` with reasons.

## Conventions that matter

- **Noise**: Gaussian, parameterized by the standard deviation
  (`sigma_std`), matching `crlb.compute_multipeak_fisher(noise_std=…)`.
  A `variance=` entry point exists and converts explicitly. The IC
  layer also scores Poisson fits (exact log-likelihood; non-positive
  predictions raise), but the exact-K orchestrator is Gaussian-only in
  v1.
- **SO degrees of freedom**: a fixed spin-orbit doublet is ONE
  structured component — one amplitude, one center; the fixed splitting
  and branch ratio contribute nothing to `k`. It shows as two visible
  peaks (`n_visible_peaks`).
- **Gaussian dof**: the RSS formulas are the profile likelihood with
  the variance MLE substituted in, so `variance_estimated=True` is
  mandatory and the variance counts as one parameter.
- **Unconstrained LS**: amplitudes may come out negative; they are
  flagged, they block `supported`, and their frequency is collected as
  evidence for a future NNLS variant. They are never clipped.

## Measured limitations (Gate 4 record, 1,560-cell frozen grid)

- **S1 = 82.5 %** against a 90 % target — accepted as a miss. The
  dominant factor is whether 1 FWHM spacing counts as
  "well-separated": at SNR ≥ 100, true-K-in-supported is 87.8 %
  (1 FWHM, no SO) vs 48.9 % (1 FWHM, SO), but 93.3 % / 100 % at
  2 FWHM. 62/63 failures report `ambiguous` — honest refusals, not
  wrong claims.
- **Conservatism is deliberate**: AICc's argmin drifts to K+1 under
  free-center scanning (a scan-statistic effect), which the
  criteria-agreement rule converts into `ambiguous` rather than an
  overfit claim. Confident overfit claims: 7/1,560 (0.45 %; six at
  SNR 30, one at SNR 100 with SO at 1 FWHM). An AICc correction for
  adaptive center search is future work.
- **Plug-in Shirley/Tougaard backgrounds are NOT a recommended
  default.** On the 3b slice the background-model error inflates the
  BIC-preferred K by +1.3 to +2.0 and true-K support drops to ≤ 0.19
  (Shirley on a nominal Shirley-like shape) and ~0 elsewhere; the
  verdict layer held confident overclaims at 0/144, so reports remain
  honest — but expect `ambiguous`, not answers. Prefer polynomial
  backgrounds (`bg_degree`) where physically defensible, and treat
  plug-in-background ICs as heuristics conditioned on an estimated
  background (never compare them across background families). The
  slice used nominal generated shapes; self-consistent matched
  Shirley/Tougaard generation is unevaluated.
- v1 scope: Voigt lineshape only, shared Gaussian width, fixed gamma,
  Gaussian noise, no joint peak-background estimation, no per-pixel
  map sweeps (use representative spectra).

## Reproducing the record

```bash
uv run pytest src/toyomacro/voigtfit/tests/test_rank_diagnostics.py \
    src/toyomacro/voigtfit/tests/test_model_selection.py \
    src/toyomacro/voigtfit/tests/test_exact_k.py \
    src/toyomacro/voigtfit/tests/test_bench_rank_model_selection.py

python -m toyomacro.voigtfit.benchmarks.bench_rank_model_selection --mode smoke
python -m toyomacro.voigtfit.benchmarks.bench_rank_model_selection --mode full   # ~7 min
```
