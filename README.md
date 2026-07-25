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
| `toyomacro.{core, lineshape, background, data, fitting, io}` | XPS analysis foundation: lineshapes, Shirley/Tougaard backgrounds, cross-sections (Scofield, Yeh-Lindau, Trzhaskovskaya 2018/2019), IMFP (TPP-2M), analyzer-transmission loader, HDF5/MAT readers. | (core) |
| MLX GPU backend | Apple-Silicon accelerated Faddeeva, Stage 2 refinement, multipeak solver. | `[mlx]` |

The Voigt engine is the part the project is being prepared for JOSS
submission as a standalone tool.

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
> shared solver paths) at roughly **3–5× lower throughput**
> on the same hardware (and 5–10× lower on a typical Linux x86_64
> CI runner).

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
| voigtfit `dict2d_parabola` | **6.5 M spec/s** | MLX GPU batch |
| `scipy.optimize.curve_fit` | 798 spec/s | bounded TRF, single-thread CPU |
| `lmfit` | 783 spec/s | bounded, single-thread CPU |

Matching accuracy on the common subset (MAE amplitude 0.022 vs
0.022), a factor of about 8,000 in throughput — that ratio, not the kernel headline, is
the relevant comparison with the conventional workflow.

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

The [`examples/`](examples/) directory contains four self-contained,
runnable scripts (synthetic data, no measurement files needed): template
peak decomposition with `AutoFitter`, batch chemical-state mapping with
`FastVoigtFitter`, quantification with the bundled cross-section /
IMFP reference data, and a numerical validation of the fitted-center
bias under systematic lineshape error (a first-order projection law,
verified against the production `VarProFitter`; see
[`docs/projection_law_validation.md`](docs/projection_law_validation.md)).
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
look up binding energies and sensitivity factors, fit spectrum files,
and run GVRT accuracy experiments:

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

GUI front-ends and a depth-profiling solver built on this engine are
maintained separately; nothing in this repository depends on them.
If you are evaluating the JOSS submission, the entry point is
`toyomacro.voigtfit` and the runnable examples above; the test
inventory in [tests/README.md](tests/README.md) maps what each test
file guards.

## Architecture

```
toyomacro-python/
├── src/toyomacro/
│   ├── voigtfit/        # Voigt-fitting engine (the JOSS target)
│   │   ├── pipeline.py            # HybridPipeline (Stage1 + Stage2)
│   │   ├── dictionary_solver.py   # Dict1D / Dict2D / parabola
│   │   ├── dictionary_solver_3d.py # δE × δσ × δγ joint
│   │   ├── multipeak_solver.py    # Alternating projection
│   │   ├── crlb.py                # Cramer-Rao Lower Bound
│   │   ├── benchmarks/            # GVRT 1B, multi-image, noise sweep
│   │   └── tests/                 # 701 unit tests
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
uv run pytest                                    # 1,146+ tests (see tests/README.md)
uv run pytest src/toyomacro/voigtfit/tests/      # voigtfit unit tests only
uv run ruff check src/ tests/                    # lint
```

CI (GitHub Actions) runs the full suite on Ubuntu and macOS for
Python 3.11 / 3.12 on every push. See
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Citing

Citation metadata lives in [CITATION.cff](CITATION.cff) — GitHub's
"Cite this repository" button renders it as BibTeX/APA. There is no
DOI yet; the first tagged release will be archived on Zenodo and the
DOI added here and to `CITATION.cff`. Release history is tracked in
[CHANGELOG.md](CHANGELOG.md).

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
