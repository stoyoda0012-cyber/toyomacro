# Related software survey

Documented search backing the paper's *State of the field* section.

**Method.** Searched PyPI, GitHub, and Zenodo (2026-05 through
2026-07) for: `XPS fitting`, `photoelectron spectroscopy python`,
`Voigt fit batch`, `XPS peak fit open source`; followed citations
from the KherveFitting paper and the lmfit ecosystem. Inclusion
criteria: open-source, installable, actively usable for XPS peak
fitting. The comparison benchmark
(`toyomacro.voigtfit.benchmarks.solver_comparison_benchmark`) runs
the general-purpose optimizers on this package's benchmark problem;
GUI tools are characterized by architecture, not re-benchmarked.

## General-purpose fitting engines

| Tool | Architecture | Measured on our benchmark problem |
|---|---|---|
| `scipy.optimize.curve_fit` | per-spectrum Levenberg-Marquardt | 819 spectra/s (M3 Max, committed JSON) |
| `lmfit` | per-spectrum LM with parameter objects | 820 spectra/s (same) |

Both are excellent at what they target: flexible per-spectrum models
with rich constraint systems. Throughput is bounded by per-spectrum
Python dispatch and model re-evaluation, not by the optimizer math.

## XPS-specific open tools

| Tool | Type | Relation to voigtfit |
|---|---|---|
| **KherveFitting** (Kerherve et al., *Surf. Interface Anal.* 2026, doi:10.1002/sia.70032) | GUI application (Python) | Interactive per-spectrum workflow: backgrounds, constraints, survey analysis. Complementary — not designed for 10⁶–10⁹-spectrum batches. |
| **lmfitxps** (Hochhaus, Zenodo doi:10.5281/zenodo.17086095) | lmfit extension library | Adds XPS lineshapes (Gaussian×Doniach-Sunjic convolution) and backgrounds (Tougaard/Shirley/slope) to lmfit's per-spectrum architecture. |
| **LG4X / LG4X-V2** | GUI wrapper around lmfit | Streamlines interactive lmfit workflows; same per-spectrum core. |
| **CasaXPS** (commercial, closed) | GUI application | De-facto standard for interactive XPS analysis; per-spectrum processing; not open source, so excluded from the open-tool comparison. |

## Niche claimed by voigtfit

Batch-first architecture: one shared peak configuration amortized as
precomputed weight/dictionary matrices, all solver stages expressed
as dense linear algebra (GPU or BLAS). Recorded same-problem result:
`dict2d_parabola` at 5.8 M spectra/s vs. 819-820 spectra/s for the
LM tools, at matching accuracy — see
`paper/figures/results/solver_comparison.json` for the full record
(environment, seeds, per-parameter MAE).

*This survey reflects the state at the time of writing. If you know
of another open batch-oriented XPS fitting package, please open an
issue — this document and the paper's claims should track reality.*
