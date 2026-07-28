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

## API stability

The package is larger than its supported surface. Three tiers, and the
rule that separates them:

**Supported.** Every name in the `__all__` list of `toyomacro`,
`toyomacro.core`, `toyomacro.lineshape`, `toyomacro.background`,
`toyomacro.io`, `toyomacro.data`, `toyomacro.fitting`, and
`toyomacro.voigtfit`, plus the submodule entry points given with a
call signature in the sections below (`dictionary_solver`,
`multipeak_solver`, `gvrt_service`, `spectra_generator`). These are
covered by the test suite, exercised by `examples/`, and will not
change signature or semantics without a minor-version bump and a
changelog entry.

Two mechanical checks: `python -c "import toyomacro.voigtfit as v;
print(v.__all__)"` enumerates the exported surface, and a submodule that
you had to name explicitly (`from toyomacro.voigtfit.rank_diagnostics
import …`) is not exported — see the next tier before depending on it.

**Experimental.** Research modules that are deliberately *not*
re-exported from any `__init__`, reachable only by explicit submodule
import. They are tested, but their API may change or disappear in any
release, and no CLI or documented workflow depends on them:

| Module | What it is |
|---|---|
| `voigtfit.rank_diagnostics` | numerical rank / conditioning of structured Voigt designs |
| `voigtfit.model_selection` | AIC / AICc / BIC scoring of already-fitted candidates |
| `voigtfit.exact_k` | component-count (model order) selection |
| `voigtfit.auto_grouping` | Fisher-justified component grouping for hierarchical fits |
| `voigtfit.fisher_*`, `voigtfit.crlb` | Fisher information, coordinate transforms, CRLB bounds (referenced in §4 as a diagnostic, not a stable entry point) |
| `voigtfit.gp_compact`, `voigtfit.gp_reference` | Golub-Pereyra / LM reference layer behind `newton_jacobian_mode` |
| `voigtfit.dictionary_solver_3d` | δE × δσ × δγ dictionary (2-D version is the supported path) |
| `voigtfit.matlab_bridge`, `voigtfit.prefetch_pipeline`, `voigtfit.simulation` | workflow adapters and validation harnesses |
| `data.transmission` | Scienta analyzer transmission adapter (§5); reads user-supplied vendor data, no data bundled |

Likewise `multipeak_solver`'s `newton_jacobian_mode` is experimental for
any value other than the default `"raw"`.

**Internal.** Underscore-prefixed names, `voigtfit.benchmarks.*`, and
anything not listed above. Benchmarks are runnable and their record
format is stable enough to compare across hosts, but they are tools,
not a library API.

**Not in this repository.** The desktop GUI (PySide6), the FastAPI/React
web front end, and the ARXPS depth-profile solver are developed
separately and are not part of the published package — `toyomacro.gui`,
`toyomacro.api`, and `toyomacro.depth` do not exist in an installed
copy. Nothing in the supported surface imports them, so the boundary
above does not move if they are released later.

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
Without a GPU the same code runs on NumPy at roughly a quarter of these
rates — on the identical seeded problem, `balanced` measures 6.6 M
spec/s with MLX and 1.6 M spec/s with `TOYOMACRO_DISABLE_MLX=1`, at
indistinguishable accuracy (paired records:
`solver_comparison.json` / `solver_comparison_numpy.json`).

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
- **Batch size matters, and the two backends want opposite things.**
  On one M3 Max measurement of `solve_dict2d_parabola` (151 channels,
  13×7 dictionary), MLX climbs with batch size — 0.8 M spec/s at 2 K,
  6.1 M at 200 K, 7.3 M at 1 M and still rising, since small batches
  are dominated by dispatch overhead. NumPy runs the opposite way: it
  peaks near **50 K spectra** (1.9 M spec/s) and then *degrades* as the
  working set outgrows cache (1.4 M at 1 M). Feed the GPU path the
  largest batch that fits; chunk the NumPy path into tens of thousands
  of spectra. Both are the same algorithm, so the ~4× backend gap holds
  at either side's best size.
- Returns per-spectrum arrays `(n_spectra,)`; `dE`/`dsigma` are
  offsets from the nominal configuration, in eV.

Multipeak clusters (overlapping components fitted jointly):

```python
from toyomacro.voigtfit.multipeak_solver import process_multipeak
```

runs alternating projection with joint block deflation; a pure-NumPy
implementation is used automatically when MLX is not usable.

### Per-spectrum reference fitter

```python
from toyomacro.voigtfit import VarProFitter
```

Classical variable projection [Golub & Pereyra 1973] on SciPy's
optimizer: centers and widths are optimized numerically while
amplitudes are eliminated analytically. It fits **one spectrum at a
time** and is not part of the high-throughput path — nothing in
`HybridPipeline`, the dictionary solvers, or `FastVoigtFitter` calls
it. Use it as an independent oracle when you want a conventional
optimizer's answer for a handful of spectra, and as the estimator whose
misspecification bias is characterized in
`examples/04_projection_law_validation.py` (its effectively
unregularized inner solve is what makes it match the projection law's
assumptions; solvers with other regularizers are only directionally
described — see `docs/projection_law_validation.md`).

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

## 5. Quantification inputs

```python
from toyomacro.data import CrossSection, IMFP, BindingEnergy
```

Photoionization cross-sections (Scofield, Yeh–Lindau, Trzhaskovskaya)
ship as tables; TPP-2M inelastic mean free paths are computed from the
published formula. Provenance and licensing for each is stated in
[`DATA_SOURCES.md`](DATA_SOURCES.md).

**Analyzer transmission** is an adapter, not shipped data:

```python
from toyomacro.data.transmission import TransmissionFunction

TransmissionFunction.available()        # is user data present?
T = TransmissionFunction.load(slit=0.5) # Scienta XOP TSV, log-log interp
T = TransmissionFunction.from_power_law(alpha=-0.35)  # fallback
T(ek_ep)                                # transmission in mSr
```

Pass energy is **not** a table dimension. A curve is selected by
`(analyzer, mode, slit, spot)` — slit and spot are physical apertures in
**mm** — and evaluated at the dimensionless ratio KE/Ep using whatever
Ep you acquired at. Each curve covers a finite ratio range
(`ek_ep_range`); outside it the value is clamped to the endpoint, so a
wide kinetic-energy sweep can silently run flat at the ends. Check
`ek_ep_range` against `KE_max/Ep` before trusting a correction.

Vendor-measured curves are **not redistributed** — point
`TOYOMACRO_SCIENTA_DATA_DIR` at your own
`ScientaXOP/Transmission/data` folder (read at call time, so setting it
mid-session works) and `load()` will find the matching `(slit, spot)`
configuration. Start from `list_analyzers()` / `list_configs()`: a slit
width that is not on the grid raises rather than rounding to the
nearest file, since silently applying the neighbouring calibration is
the expensive kind of mistake.

Without that directory, `from_power_law()` gives a smooth
T ∝ (Ek/Ep)^α stand-in. That is a *shape* assumption, not a
calibration — across one real EW4000/Hipp3/R3000 set the fitted
exponent ranges from about −0.71 to −0.03 (median ≈ −0.26), so no
single default stands in for a specific slit/spot/mode. Absolute
quantification from the fallback is not traceable; use it for relative
trends only.

This module is experimental in the sense of the stability section — it
is not re-exported from `toyomacro.data` and no other part of the
package calls it.

## 6. Synthetic data and GVRT

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

## 7. Reproducible benchmarking

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
