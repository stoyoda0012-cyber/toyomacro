# CUDA backend proof-of-concept (MLX on Linux / WSL2)

**Status:** planned — not yet attempted.
**Scope:** correctness validation only. Performance work and any public
CUDA support claim come later, if the PoC passes.

## Purpose

`toyomacro.voigtfit` currently has two verified backends:

1. **MLX (Metal)** on Apple Silicon — the accelerated path.
2. **NumPy** everywhere else — CI-tested on Ubuntu and macOS; selected
   automatically when MLX is absent or unusable, or forced with
   `TOYOMACRO_DISABLE_MLX=1`.

MLX itself now ships an official CUDA backend for Linux
(`pip install "mlx[cuda13]"`), which means the existing MLX kernels may
run on NVIDIA GPUs **without a rewrite**. This document is a
self-contained recipe to test that hypothesis on a fresh machine. The
governing rule: a new backend is adopted only after its required
operations and numerical behavior are verified against the existing
reference tests — GPU support is a result to validate, not a goal that
overrides correctness.

## Target environment

- Windows 11 + WSL2 (Ubuntu 24.04) — MLX has no native Windows support;
  WSL2 is the supported route. NVIDIA officially supports CUDA in WSL2.
- NVIDIA driver **on the Windows side only** (>= 580 for CUDA 13).
  Do NOT install a Linux GPU driver inside WSL2 — the Windows driver is
  exposed inside WSL as a stub `libcuda.so`.
- RTX 50-series (Blackwell, `sm_120`) requires a recent CUDA stack
  (CUDA 12.8+ / 13); older wheels do not support `sm_120`.
- First test machine: RTX 5070 Laptop (8 GB VRAM).

## Setup

```bash
# Inside WSL2 Ubuntu. Verify the GPU is visible first:
nvidia-smi

git clone <private remote> toyomacro && cd toyomacro
uv sync                          # core (NumPy backend) deps
uv pip install "mlx[cuda13]"     # MLX CUDA backend
```

Notes:

- Most voigtfit tests are synthetic and self-contained. Tests that need
  measured data resolve it via `TOYOMACRO_TESTDATA` and skip cleanly
  when unset — no data transfer is required for the PoC.
- The capability probe (`toyomacro.voigtfit._mlx_support.mlx_usable`)
  runs a tiny kernel on the **default device** and is device-agnostic:
  it is expected to return True on CUDA unchanged. Its error messages
  still say "Apple Silicon only" — update them once CUDA is confirmed.

## Step 0 — NumPy baseline (should already pass)

```bash
TOYOMACRO_DISABLE_MLX=1 uv run pytest src/toyomacro/voigtfit/tests -q
```

This is the CI-covered configuration. Any failure here is an
environment problem, not a CUDA problem — fix it before proceeding.

## Step 1 — probe

```bash
uv run python -c "
import mlx.core as mx
print('default device:', mx.default_device())
mx.eval(mx.zeros((4,)) + 1)
from toyomacro.voigtfit._mlx_support import mlx_usable
print('mlx_usable:', mlx_usable())
"
```

Expected: `Device(gpu, 0)` and `mlx_usable: True`.

## Step 2 — acceptance suite

The existing MLX↔NumPy parity tests are the acceptance harness; no new
tests are needed to judge the PoC.

```bash
uv run pytest src/toyomacro/voigtfit/tests -q
```

Highest-signal files if triaging selectively:

| file | what it certifies on CUDA |
|---|---|
| `test_pipeline_mlx.py` | Stage-1 weight-matrix pipeline (matmul kernel) |
| `test_multipeak_solver.py` | alternating projection, `requires_mlx` blocks, MLX-vs-NumPy parity |
| `test_gp_lm.py` | GP-LM hybrid chunk CPU/MLX parity |
| `test_dictionary_solver.py` | dictionary gather/argmax kernels |
| `test_extended_svd.py` | `mx.compile`d 4-step solver kernels |

## Known risk map

Expected to be **safe** (verified against the current code):

- **FFT**: the MLX CUDA backend lacks FFTs, but the only FFT in
  voigtfit (Doniach-Sunjic convolution, `ds_profile.py`) runs on
  `np.fft` — never on MLX. Not hit.
- **LAPACK-style ops**: `mx.linalg.solve` appears at 2 call sites,
  both already routed to `stream=mx.cpu` (it is CPU-only on Metal
  too). The GP-LM small-matrix float64 work is plain NumPy. Not hit,
  provided the CPU stream works on the Linux build (verify early).

Needs **watching**:

- **Missing ops crash — no automatic CPU fallback.** If a kernel uses
  an unimplemented op, MLX raises instead of falling back. Record the
  op + stack trace; the fix is `stream=mx.cpu` routing at that call
  site.
- **`mx.compile` on CUDA**: `spectra_generator.py`, `pipeline.py`, and
  `LazyCompiled` kernels rely on it. Functionality and compile latency
  are unverified on the CUDA backend.
- **uint16 direct transfer** (`h5io` specdata path): dtype support is
  standard, but transfer semantics differ on a discrete GPU.
- **Threading**: the Metal backend had a thread-local compile-cache
  teardown crash under Qt threads (fixed by avoiding worker-thread
  kernel calls). The CUDA backend's threading behavior is unknown —
  keep kernel calls on the main thread for the PoC.

## Pass / fail criteria

**PoC passes** when the parity suite passes on CUDA within existing
tolerances, or every failure is attributed to a documented missing op
with a working `stream=mx.cpu` workaround.

**Performance is explicitly not a PoC criterion.** Record benchmark
medians (`benchmarks/`) for the report, but judge them later:

- The amplitude-only streaming kernel is memory-bandwidth bound and
  benefits from Apple's unified memory; on a discrete GPU the PCIe hop
  (plus WSL2 overhead) becomes the bottleneck. Expect it to fall short
  of the Apple-Silicon headline numbers.
- Compute-dense paths with GPU-resident dictionaries (multipeak AP,
  many-component fits) are the most likely to port well.

## Reporting

Append results here (environment, `mlx.__version__`, driver/CUDA
versions, pass/fail counts, failing-op list, benchmark medians). The
outcome feeds the next milestone review; productionizing CUDA support
is out of scope of the current release plan, and no CUDA performance
claim goes into the README or paper until it is reproducible from a
committed record.
