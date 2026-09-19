# toyomacro

[![CI](https://github.com/stoyoda0012-cyber/toyomacro/actions/workflows/ci.yml/badge.svg)](https://github.com/stoyoda0012-cyber/toyomacro/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

> High-throughput Voigt-fitting for X-ray Photoelectron Spectroscopy (XPS).
> The bundled engine [`toyomacro.voigtfit`](src/toyomacro/voigtfit/)
> runs its **amplitude-only projection kernel** at a measured median
> of **478 M spectrum-vectors per second** (range 453–490 over nine
> repetitions; full provenance record committed under
> [`paper/figures/results/`](paper/figures/results/)) on an Apple
> M3 Max with MLX, and previously completed the full GVRT (Giga Voigt
> Round Trip: image → Voigt spectra → noise → fit → reconstructed
> image) benchmark on an 8K UHD source
> — ≈265 M Voigt fits across eight shot-noise severities —
> in about ten minutes on the same hardware. A pure-NumPy fallback
> runs the identical algorithms anywhere CPython runs.
>
> See [How fast](#how-fast) below for what these numbers actually
> measure and how the MLX and NumPy backends compare.

## What's in the box

| Subpackage | What it does | Install extra |
|---|---|---|
| `toyomacro.voigtfit` | The dictionary + Newton Voigt solvers, MLX backends, CRLB analysis, benchmarks. Standalone (numpy/scipy/h5py/matplotlib). | (core) |
| `toyomacro.{core, lineshape, background, data, fitting, io}` | XPS analysis foundation: lineshapes, Shirley/Tougaard backgrounds, cross-sections (Scofield, Yeh-Lindau, Trzhaskovskaya 2018/2019), IMFP (TPP-2M), an analyzer-transmission loader for user-supplied vendor curves (no transmission data is bundled), HDF5/MAT readers. Also an **experimental** overlayer-thickness EAL correction computed from a caller-supplied IMFP/TRMFP pair — no TRMFP or albedo tables are bundled (see [`docs/API.md`](docs/API.md#5-quantification-inputs) and [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)). | (core) |

The Voigt engine is the part the project is being prepared for JOSS
submission as a standalone tool.

### Backend support status

| Backend | Status | Installation | Notes |
|---|---|---|---|
| NumPy / CPU | **Supported** | base install | Runs the identical algorithms anywhere CPython runs. This is the path CI covers — the hosted runners are headless, so no CI job exercises a GPU. |
| MLX / Apple Metal | **Supported accelerator** | `[mlx]` | Accelerates the Faddeeva kernel, the legacy compatibility fallback (Stage 2) and the multipeak solver. Every MLX number quoted below was recorded here; correctness rests on the MLX↔NumPy parity tests, which run only where `mlx_usable()` is True. |
| MLX / CUDA | **Experimental validation** — not a supported install target | manual setup; no extra ships for it | The MLX↔NumPy parity suite passed once on an RTX 5070 Laptop under WSL2. Requires `MLX_ENABLE_TF32=0` (TF32 is on by default and silently costs precision) and multipeak batches ≤ 65,535 (upstream crash). No CUDA CI, no committed performance record. See [`docs/CUDA_BACKEND_POC.md`](docs/CUDA_BACKEND_POC.md). |

The backend probe (`mlx_usable()`) is device-agnostic — it checks
whether MLX can execute work on its default device, not whether that
device is Metal. Supported packaging and published benchmarks currently
stop at Apple.

## How fast

> A **"spectrum"** here means one 1-D intensity array of `n_energy`
> floats — typically what is measured at a single pixel of an XPS
> map for a single chemical element. Rates below are per such array,
> processed in batch. All numbers were recorded on an Apple
> **M3 Max (128 GB)** with the **MLX** backend unless otherwise
> noted — the kernel headline and the scipy/lmfit comparison carry
> committed machine-readable provenance records
> ([`paper/figures/results/`](paper/figures/results/)); the other
> rows are previously recorded values, regenerable with the bundled
> benchmarks. The pure-NumPy fallback runs the same algorithms with
> numerically equivalent results (≲10⁻³ relative agreement on the
> shared solver paths) at roughly **4× lower throughput** on the same
> hardware — measured, see
> [without Apple Silicon](#without-apple-silicon-numpy-backend) below.

### Per-solver throughput (MLX, M3 Max)

What "throughput" measures here is the wall-clock rate of the
solver step itself — given a batch of `n` spectra and a peak
configuration, how many spectra per second the solver can return
amplitudes (and δE / δσ, where applicable) for.

| Solver | Output | Throughput |
|---|---|---:|
| **Amplitude-only projection** (`Y @ W` matmul kernel) | amplitude only, no shift/width recovery | **478 M spec/s** median, 453–490 range (memory-bandwidth bound; committed record) |
| 4-step Taylor residual projection | amp + δE + δσ | ~1.5 M spec/s |
| `parabola` (Dict2D + parabolic refine) | amp + δE + δσ, no Jacobian | ~5 M spec/s |
| `gamma_calibrated` (Dict3D → Dict2D γ-cal, warm cache) | amp + δE + δσ + γ correction | ~2 M spec/s |
| `multipeak` 2-component (alternating projection) | per-component amp + δE + δσ | ~3 M spec/s |
| `multipeak` 10-component | per-component amp + δE + δσ | ~40 K spec/s |

The amplitude-only kernel is essentially memory bandwidth on the
GPU; it is **not** a full peak fit on its own. The other rows are
full per-spectrum recoveries for the listed parameters.

The multipeak solver additionally exposes an **experimental**
`newton_jacobian_mode` flag (Kaufman / Golub-Pereyra / GP-LM
safeguarded refinement with per-spectrum diagnostics). The default
(`"raw"`) is the supported production path; the non-raw modes are
research features outside the API stability guarantee — see
`src/toyomacro/voigtfit/benchmarks/GP_LM_HARDENING_REPORT.md`.

### Same problem, same machine: vs scipy / lmfit

`python -m toyomacro.voigtfit.benchmarks.solver_comparison_benchmark`
fits the identical seeded dataset (single Voigt peak, 151 channels,
free amplitude/position/width, peak-count SNR 10) with identical
initialization and identical bounds; accuracy is scored on a common
subset fitted by every solver. This is a workflow-vs-workflow
comparison on one machine: the per-spectrum tools run their normal
single-thread CPU loop, the batch solver uses the GPU when usable.
Committed record (exact figures, input hash, environment):
[`paper/figures/results/solver_comparison.json`](paper/figures/results/solver_comparison.json).

| Solver | Throughput | Notes |
|---|---:|---|
| voigtfit `dict2d_parabola` | **6.6 M spec/s** | MLX GPU batch |
| `scipy.optimize.curve_fit` | 845 spec/s | bounded TRF, single-thread CPU |
| `lmfit` | 801 spec/s | bounded, single-thread CPU |

Matching accuracy on the common subset (MAE amplitude 0.022 vs
0.022), a factor of about 8,000 in throughput — that ratio, not the kernel headline, is
the relevant comparison with the conventional workflow.

#### Without the GPU (NumPy backend)

The GPU is an accelerator here, not the source of the speedup. Rerun
the same benchmark with `TOYOMACRO_DISABLE_MLX=1` and it takes the
pure-NumPy path through the same solver code; the record is committed
next to the accelerated one
([`solver_comparison_numpy.json`](paper/figures/results/solver_comparison_numpy.json)),
same host, same day, **same input hash** — the backend is the only
difference.

| Solver | MLX (GPU) | NumPy (CPU) | MAE amplitude |
|---|---:|---:|---|
| voigtfit `dict2d_parabola` | 6.6 M spec/s | **1.6 M spec/s** | 0.0222 / 0.0222 |
| `scipy.optimize.curve_fit` | 845 spec/s | 822 spec/s | 0.0219 |
| `lmfit` | 801 spec/s | 808 spec/s | 0.0219 |

So the batch formulation alone — no GPU — is worth about **2,000×**
over the per-spectrum workflow, and MLX contributes a further ~4×.
Accuracy is unchanged (MAE δE 0.0134 GPU vs 0.0146 CPU, both against
a peak-count SNR of 10).

Read this as isolating the *backend*, not the vendor: both columns come
from the same M3 Max, so the NumPy row is what that machine's CPU and
BLAS deliver. Another host will land elsewhere in absolute terms — but
it runs the identical code path, and `--out` writes the same record
format, so you can generate the comparable row on your own machine in
about a minute:

```bash
TOYOMACRO_DISABLE_MLX=1 python -m toyomacro.voigtfit.benchmarks.solver_comparison_benchmark --out my_host.json
```

> The amplitude-only projection row is BLAS-on-CPU in *both* records
> (that kernel is not routed through MLX in this benchmark), so its
> two columns differ only by run-to-run noise. The MLX headline for
> that kernel is the separate `figure1_throughput.json` record.

### End-to-end image roundtrip

The GVRT (Giga Voigt Round Trip) benchmarks measure the complete
pipeline (RGB amplitudes → Voigt spectra → Poisson noise → fit →
reconstructed RGB) on real images. Numbers below count the total
generation + fit time, not the solver alone.

| Image | Noise sweep | Total fits | Wall time (MLX, M3 Max) | PSNR (noise-free) δa / δc / δσ |
|---|---|---:|---:|---|
| 540p demo image (960×540, 6 components) | none | ~0.52 M × 5 = **2.6 M** | ~0.9 s | 59.8 dB (image-level) |
| 4K UHD | 8 Poisson levels | **66 M** | 1.4 min | 64.1 / 61.3 / 56.9 dB |
| 8K UHD (7680×4320) | 8 Poisson levels | **265 M** | ~10.4 min | 60.8 / 61.3 / 55.1 dB |

On the same 540p image the pure-NumPy backend completes in ~1.3 s
with identical PSNR (59.77 dB), demonstrating that "MLX-only"
performance gains do not come at the cost of correctness.

See [`src/toyomacro/voigtfit/benchmarks/`](src/toyomacro/voigtfit/benchmarks/)
for the runnable scripts behind each number.

## Installation

Requires Python **3.11** or newer. We recommend
[`uv`](https://docs.astral.sh/uv/) for installation.

```bash
# Voigt engine only (numpy / scipy / h5py / matplotlib / lmfit / lz4 / psutil)
uv sync

# + Apple Silicon GPU backend
uv sync --extra mlx

# Development tooling (pytest, ruff, mypy)
uv sync --extra dev

# Everything
uv sync --extra all
```

Pip works equivalently:

```bash
pip install -e .                     # core
pip install -e ".[mlx]"              # + MLX
pip install -e ".[all]"              # everything
```

## Quick start

### Fit a single Voigt profile

```python
import numpy as np
from toyomacro.voigtfit import voigt_profile

energy = np.linspace(97, 107, 256)             # eV (Si 2p region)
y = voigt_profile(energy, center=99.3, sigma=0.45, gamma=0.10)
# y is the noise-free Voigt for Si⁰ evaluated on `energy`.
```

### Fit a batch of spectra with the hybrid pipeline

```python
import numpy as np
from toyomacro.voigtfit import (
    HybridPipeline, WeightMatrixCache, voigt_profile,
)

# 1. Build a synthetic Si 2p sub-oxide spectrum (Si⁰, Si¹⁺, Si⁴⁺),
#    10,000 pixels.
energy = np.linspace(97, 107, 256, dtype=np.float32)
centers = np.array([99.3, 100.3, 103.4])             # Si⁰, Si¹⁺, Si⁴⁺
sigmas  = np.array([0.45, 0.50, 0.60])
gamma   = 0.10

basis = np.stack([
    voigt_profile(energy, c, s, gamma) for c, s in zip(centers, sigmas)
])                                                    # (3, n_energy)

rng = np.random.default_rng(0)
true_amps = rng.uniform(0.3, 1.2, size=(3, 10_000)).astype(np.float32)
Y = (basis.T @ true_amps).astype(np.float32)          # (n_energy, n_spectra)
Y += rng.normal(0.0, 0.005, size=Y.shape).astype(np.float32)

# 2. Fit.
pipeline = HybridPipeline(WeightMatrixCache(), use_mlx=False)
result   = pipeline.process(
    Y=Y, element="Si", orbital="2p", energy=energy,
    peak_config=dict(centers=centers, sigmas=sigmas, gamma=gamma),
)

# 3. Inspect.
print(result.amplitudes.shape)                        # (3, 10_000)
recovery_error = np.median(np.abs(result.amplitudes - true_amps))
print(f"median |Δamp| = {recovery_error:.4f}")        # ~0.0011
```

Switch the pipeline to `use_mlx=True` on an Apple Silicon machine for
GPU acceleration; the API is unchanged.

### More examples

The [`examples/`](examples/) directory contains seven self-contained,
runnable scripts (synthetic data, no measurement files needed): template
peak decomposition with `AutoFitter`, batch chemical-state mapping with
`FastVoigtFitter`, quantification with the bundled cross-section /
IMFP reference data, and a numerical validation of the fitted-center
bias under systematic lineshape error (a first-order projection law,
verified against the production `VarProFitter`; see
[`docs/projection_law_validation.md`](docs/projection_law_validation.md)),
fitting a map read from a file — the step to your own measurement —
calibrating the energy axis on a metal Fermi edge, and mapping how well
a spectrum can tell the Gaussian from the Lorentzian width of a Voigt
peak (model bounds, not a fit; see
[`docs/design/voigt-width-identifiability.md`](docs/design/voigt-width-identifiability.md)).
They run in CI on every commit.

### GVRT roundtrip demo

GVRT (Giga Voigt Round Trip) encodes an image as per-pixel Voigt
parameters (R=amplitude, G=energy shift, B=FWHM), generates one
spectrum per pixel, adds Poisson noise, fits every spectrum, and
reconstructs the image. Per-channel PSNR gives a direct, visual
accuracy metric for each solver:

```bash
python -m toyomacro.voigtfit.cli gvrt                        # demo image, Moderate noise
python -m toyomacro.voigtfit.cli gvrt --solver dict2d_parabola --noise None
python -m toyomacro.voigtfit.cli gvrt photo.png --inspect 128,128
```

Saves an original / reconstructed / difference figure and prints PSNR
per channel; `--inspect X,Y` adds a spectrum-space plot of one pixel
(ground truth, noisy measurement, and fit).

### Use from AI agents (MCP)

The engine is exposed as an [MCP](https://modelcontextprotocol.io)
server so AI agents (Claude Code, Claude Desktop, ...) can drive it —
look up binding energies, compute simplified intrinsic sensitivities
(cross-section × IMFP, with no instrument response), fit spectrum
files, and run GVRT accuracy experiments:

```bash
pip install -e ".[mcp]"
python -m toyomacro.mcp_server
```

Tools: `lookup_binding_energy`, `calculate_sensitivity`,
`list_fitting_templates`, `fit_spectrum_file`, `gvrt_run`, `gvrt_sweep`.


## Scope: what this repository provides

Everything documented here installs and runs from this repository
alone — `pip install`, the examples, the benchmarks, and the full
test suite need no external tools or data.

- **`toyomacro.voigtfit`** — the batch-first Voigt-fitting engine.
  This is the subject of the JOSS paper; all performance claims and
  benchmarks refer to it.
- `toyomacro.{core, lineshape, background, data, io}` — foundation
  modules the engine uses: lineshapes, backgrounds, quantification
  reference tables, and file readers (PXT/VAMAS/NPL/two-column text).
- `toyomacro.fitting` — high-level peak-fitting templates that wrap
  the engine for common XPS analyses.

### Quantification boundary in v0.1

The v0.1 release includes quantification-oriented reference utilities: binding-energy and
photoionization cross-section lookup, TPP-2M IMFP calculation, and interpolation of a
user-supplied, authorized analyzer-transmission curve. Their outputs retain the selected source,
units, energy range, extrapolation status, and instrument-specific assumptions where available.

These utilities provide peak observables and declared sensitivity terms for relative comparisons.
They do not claim traceable absolute composition from a universal sensitivity factor. In
particular, `cross-section × IMFP`, even with transmission applied, omits factors such as
elastic-scattering/EAL, detector response, angular distribution, polarization, geometry, and
matrix assumptions. A future composition workflow will require an explicit input contract,
uncertainty/assumption reporting, and redistributable validation data.

GUI front-ends and a depth-profiling solver built on this engine are
maintained separately; nothing in this repository depends on them.
If you are evaluating the JOSS submission, the entry point is
`toyomacro.voigtfit` and the runnable examples above; the test
inventory in [tests/README.md](tests/README.md) maps what each test
file guards.

## Architecture

```
toyomacro/
├── src/toyomacro/
│   ├── voigtfit/        # Voigt-fitting engine (the JOSS target)
│   │   ├── pipeline.py            # HybridPipeline (Stage 1 screening + legacy compatibility fallback)
│   │   ├── dictionary_solver.py   # Dict1D / Dict2D / parabola
│   │   ├── dictionary_solver_3d.py # δE × δσ × δγ joint
│   │   ├── multipeak_solver.py    # Alternating projection
│   │   ├── crlb.py                # Cramer-Rao Lower Bound
│   │   ├── benchmarks/            # GVRT 1B, multi-image, noise sweep
│   │   └── tests/                 # 839 unit tests
│   ├── core/            # Spectrum, FittingResult
│   ├── lineshape/       # Voigt, Gaussian, Lorentzian, PseudoVoigt, Doniach-Sunjic
│   ├── background/      # Shirley / Tougaard / Linear
│   ├── data/            # Cross sections, binding energies, IMFP, transmission
│   ├── fitting/         # Templates, AutoFitter, FastVoigtFitter
│   └── io/              # HDF5 cache, readers (PXT/IBW/VAMAS/NPL/SES), writers
├── tests/               # Integration / top-level tests
└── docs/                # Data provenance (DATA_SOURCES.md)
```

The engine has **zero circular dependencies** between subpackages and
each layer can be imported in isolation.

The principles behind these choices — throughput as a design
constraint, honest benchmark numbers, round-trip validation, models
as data — are written up in
[`docs/DESIGN_PHILOSOPHY.md`](docs/DESIGN_PHILOSOPHY.md).

## Compatibility

- **HDF5**: ships streaming/chunked writers compatible with the
  DepthProfiler depth-estimate format ([`toyomacro.io.writers`](src/toyomacro/io/writers/)).
- **MATLAB `.mat`**: read via [`toyomacro.io.readers.mat_reader`](src/toyomacro/io/readers/mat_reader.py).
- **XPS analyzer files**: PXT, IBW, VAMAS, NPL, SES TXT — all under
  [`toyomacro.io.readers`](src/toyomacro/io/readers/).

## Development

```bash
uv sync --extra dev --extra mlx
uv run pytest                                    # 1,608 tests (see tests/README.md)
uv run pytest src/toyomacro/voigtfit/tests/      # voigtfit unit tests only
uv run ruff check src/ tests/                    # lint
```

CI (GitHub Actions) runs the full suite on Ubuntu and macOS for
Python 3.11 / 3.12 on every push. See
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Citing

Citation metadata lives in [CITATION.cff](CITATION.cff) — GitHub's
"Cite this repository" button renders it as BibTeX/APA. Every tagged
release is archived on Zenodo:

- **All versions** (resolves to the latest release):
  [10.5281/zenodo.22092076](https://doi.org/10.5281/zenodo.22092076)
- **v0.1.0**:
  [10.5281/zenodo.22092077](https://doi.org/10.5281/zenodo.22092077)

A Zenodo download is a snapshot of one release, without git history.
To follow development between releases, clone this repository instead.
Release history is tracked in [CHANGELOG.md](CHANGELOG.md).

## License

Released under the [MIT License](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Support and maintenance

This project is maintained on a **best-effort basis** by a single
maintainer. Bug reports and feature requests are welcome through the
GitHub issue templates; there is no guaranteed response time, and
well-scoped reports with a minimal reproduction are triaged first.
Pull requests are the fastest route to getting a change in — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgments

Successor to the original MATLAB Toyomacro suite, rewritten in Python
with a heavy focus on GPU-friendly batched algorithms and a clean
public API.
