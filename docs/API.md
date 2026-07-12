# Core API reference

The entry points a researcher needs, with array shapes, units, and
backend behavior stated explicitly. Everything below is importable
from a plain `pip install toyomacro` (NumPy backend) — the MLX GPU
path activates automatically on Apple Silicon when usable.

**Conventions.** Energy in eV; `sigma` is the Gaussian standard
deviation (eV); `gamma` the Lorentzian half-width (eV);
FWHM ≈ 0.5346·(2γ) + √(0.2166·(2γ)² + (2.355σ)²). A "spectrum" is a
1-D intensity array of `n_energy` channels. **Batch APIs are
row-major**: `Y` has shape `(n_spectra, n_energy)` unless a method
name or docstring says otherwise (the lower-level
`HybridPipeline.stage1_screening` is column-major,
`(n_energy, n_spectra)` — its docstring states this).

## Backend selection

```python
from toyomacro.voigtfit import mlx_installed, mlx_usable, require_mlx
```

- `mlx_usable()` — True only if a probe kernel actually executed on a
  Metal device. Installed-but-headless MLX returns False and every
  solver silently uses the NumPy backend (numerically identical to
  ~1e-3 relative in float32).
- `require_mlx()` — raises an actionable `RuntimeError` when the GPU
  path is unavailable; call it first if your workload depends on GPU
  throughput.
- `TOYOMACRO_DISABLE_MLX=1` — environment variable forcing the NumPy
  backend (useful for A/B validation).

## 1. Lineshape

```python
from toyomacro.voigtfit import voigt_profile
y = voigt_profile(energy, center, sigma, gamma)   # (n_energy,)
```

Exact Voigt via the Faddeeva function (scipy.wofz). Unit peak-area
normalization: multiply by an amplitude to scale.

## 2. Batch fitting, high level

```python
from toyomacro.fitting import FastVoigtFitter, FastFitConfig
fitter = FastVoigtFitter(FastFitConfig(mode="balanced"))
result = fitter.fit_batch_rowmajor(
    spectra,          # (n_spectra, n_energy) float32
    energy,           # (n_energy,) eV
    centers, sigmas,  # (n_comp,) nominal configuration
    gamma,            # scalar
)
```

| mode | solver | recovers | typical rate* |
|---|---|---|---|
| `preview` | amplitude-only projection | amplitude | ~10⁸ spec/s |
| `fast` | 4-step Taylor projection | + δE, δσ (small shifts) | ~10⁶ spec/s |
| `balanced` | dictionary + parabola | + δE, δσ (wide range) | ~5×10⁶ spec/s |
| `precise` | adaptive | best quality | ~4×10⁵ spec/s |

*Recorded on Apple M3 Max (MLX); see committed records under
`paper/figures/results/` and regenerate with the benchmarks below.

**Solver selection.** Start with `balanced`. Use `preview` when
centers/widths are known and only amplitudes are needed (composition
maps). Use `fast` when shifts are guaranteed small (|δE| ≲ 0.3σ —
Taylor validity). Escalate to `precise` for publication-grade
parameter maps or difficult overlaps.

## 3. Batch fitting, low level

```python
from toyomacro.voigtfit.dictionary_solver import (
    build_dictionary_2d, solve_dict2d_parabola)

cache = build_dictionary_2d(energy, centers, sigmas, gamma,
                            dE_range=(-0.15, 0.15),
                            dsigma_range=(-0.05, 0.05),
                            N_dE=13, N_ds=7)          # setup cost, once
amp, chi2, dE, dsigma, idx = solve_dict2d_parabola(Y, cache)
```

- Setup vs. steady state: dictionary construction is a one-time cost
  (seconds) amortized over the batch; benchmark records report it
  separately (`setup_s`).
- Returns per-spectrum arrays `(n_spectra,)`; `dE`/`dsigma` are
  offsets from the nominal configuration, in eV.

Multipeak clusters (overlapping components fitted jointly):

```python
from toyomacro.voigtfit.multipeak_solver import process_multipeak
```

runs alternating projection with joint block deflation; a pure-NumPy
implementation is used automatically when MLX is not usable.

## 4. Quality flags and failure modes

- Every solver returns per-spectrum `chi2`; anomalously high values
  flag pixels that need escalation to a slower solver stage or manual
  inspection.
- Shift estimates at the edge of the dictionary range indicate the
  search window was too narrow — rebuild with a wider `dE_range`.
- Taylor solvers (`fast`) degrade smoothly but systematically beyond
  |δE| ≈ σ: two-lobed residuals and underestimated widths are the
  signature. Prefer `balanced` when in doubt.
- At peak-count SNR ≲ 1 (peak signal equals shot noise), parameter
  recovery collapses regardless of solver — the CRLB utilities
  (`toyomacro.voigtfit.crlb`) quantify this limit for your
  configuration, and Figure 2 of the paper shows the collapse
  ordering (width → amplitude → position).

## 5. Synthetic data and GVRT

```python
from toyomacro.voigtfit.spectra_generator import (
    SpectraGenerator, add_poisson_noise,
    level_to_peak_snr, level_to_peak_lambda)
from toyomacro.voigtfit.gvrt_service import GVRTService, GVRTConfig
```

Noise severity `level` is dimensionless: peak-count SNR = 10⁴/level,
peak Poisson mean = (10⁴/level)². `GVRTService.run(image, config)`
executes the image → spectra → fit → image round trip and returns
per-channel PSNR — the package's end-to-end accuracy check.

## 6. Reproducible benchmarking

```bash
# kernel throughput with full provenance (JSON committed under
# paper/figures/results/)
python paper/figures/make_figure1_bottleneck.py --measure

# same-problem comparison vs scipy/lmfit
python -m toyomacro.voigtfit.benchmarks.solver_comparison_benchmark \
    --out my_comparison.json

# GVRT round trip from the CLI
python -m toyomacro.voigtfit.cli gvrt --solver dict2d_parabola
```

Every benchmark writes machine-readable records (timestamp, git
commit, OS, library versions, problem size, individual timings,
median + range) so numbers in the paper can be re-derived on any
host.
