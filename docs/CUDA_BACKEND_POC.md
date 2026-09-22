# CUDA backend proof-of-concept (MLX on Linux / WSL2)

**Status:** **PASSED** — first run 2026-07-17 on `3aedcf3`, re-verified
2026-09-21 on `3a13de5`, both on the same RTX 5070 Laptop. See
[Results](#results-2026-07-17--rtx-5070-laptop) and
[Re-verification](#re-verification-2026-09-21--rtx-5070-laptop) below.
Two setup fixes were required that this recipe did not anticipate
(CUDA headers, TF32); both still apply.
**Scope:** correctness validation only. Performance work and any public
CUDA support claim come later, if the PoC passes.

## Purpose

`toyomacro.voigtfit` currently has two verified backends:

1. **MLX (Metal)** on Apple Silicon — the accelerated path.
2. **NumPy** everywhere else — CI-tested on Ubuntu, macOS and Windows;
   selected automatically when MLX is absent or unusable, or forced with
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

- Windows 11 + WSL2 (Ubuntu 24.04) — WSL2 is the only route to MLX's
  CUDA backend on a Windows box. MLX does publish `win_amd64` wheels
  (since 0.32.0), but they declare no backend at all: every CUDA extra
  in its metadata is gated on `platform_system == "Linux"`. A native
  Windows install is CPU-only, not a faster CUDA. NVIDIA officially
  supports CUDA in WSL2.
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
  it is expected to return True on CUDA unchanged. (Its error messages
  said "Apple Silicon only" when this recipe was written; they no
  longer do — `_mlx_support` describes the probe as running on MLX's
  default device, and `tests/test_mlx_support.py` pins that.)

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

Append results here (**the commit SHA verified**, environment,
`mlx.__version__`, driver/CUDA versions, pass/fail counts, failing-op
list, benchmark medians). Record the SHA even when it is the branch tip
at the time: without it a later reader cannot diff what changed since a
run, which is exactly the gap the 2026-07-17 entry left. The
outcome feeds the next milestone review; productionizing CUDA support
is out of scope of the current release plan, and no CUDA performance
claim goes into the README or paper until it is reproducible from a
committed record.

## Results (2026-07-17) — RTX 5070 Laptop

**Verdict: PASS.** The parity suite passes on CUDA within existing
tolerances. Every remaining failure is a speed assertion, which the
pass/fail criteria explicitly exclude. No missing-op was hit, so no
`stream=mx.cpu` workaround was needed.

**Tree:** `3aedcf3`. The run itself recorded no SHA — this is
reconstructed, and the reasoning is in
[Correction](#correction-the-2026-07-17-poc-tree-is-3aedcf3-and-it-is-in-this-history)
at the end of this document. Record the SHA at the time from now on.

### Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop (Blackwell, `sm_120`), 8151 MiB |
| CPU | AMD Ryzen 9 8940HX |
| Driver | 596.13 (Windows-side only, as prescribed) |
| Host | Windows 11 + WSL2, Ubuntu 24.04.4 LTS, kernel 6.18.33.2-microsoft-standard-WSL2 |
| Python / NumPy | 3.12.3 / 2.3.5 |
| MLX | 0.32.0 (`mlx-cuda-13` 0.32.0) |
| CUDA wheels | cublas 13.6.0.2, nvrtc 13.3.33, cuda-runtime 13.3.29, cuda-cccl 13.3.3.4.1, cudnn 9.24.0.43, cufft 12.3.0.29, nvjitlink 13.3.33 |

No system CUDA Toolkit was installed — `nvcc` is absent and was never
needed. The pip wheels supply the whole stack.

### Two setup fixes this recipe was missing

**1. `mlx[cuda13]` does not pull the CUDA headers it needs to JIT.**
The extra installs cuBLAS/cuFFT/NVRTC headers only (13 files); the CUDA
runtime and CCCL headers are absent, so NVRTC cannot compile any kernel.
The first MLX GPU op dies with:

```
RuntimeError: Can not find locations of CUDA headers, please set
environment variable CUDA_HOME or CUDA_PATH.
```

Inside pytest this surfaces as a bare `Fatal Python error: Aborted`
with no message — the RuntimeError is only visible when the op is run
outside the test harness. Fix:

```bash
uv pip install nvidia-cuda-runtime nvidia-cuda-cccl   # NOT the *-cu13 names
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
```

(The `nvidia-cuda-runtime-cu13` / `nvidia-cuda-cccl-cu13` spellings are
deprecated stubs that fail to build. Use the unsuffixed names, matching
the `nvidia-cublas` / `nvidia-cufft` already pulled in.) This raises the
include dir from 13 headers to 93.

**2. MLX's CUDA backend runs float32 matmul in TF32 by default.**
This is the important finding. Measured relative Frobenius error of a
512×512 float32 matmul against a float64 reference:

| backend | rel. error |
|---|---|
| NumPy fp32 | 2.877e-07 |
| MLX CPU stream | 4.081e-07 |
| **MLX GPU (default)** | **2.930e-04** |
| MLX GPU + `NVIDIA_TF32_OVERRIDE=0` | 2.077e-07 |

~1000× the error of true fp32 — the TF32 10-bit-mantissa signature.
Metal has no such mode, so this is a CUDA-only divergence and it is
silent: nothing warns, results are just less precise.

It broke exactly what you would predict — the near-tie-sensitive paths:

- `test_dictionary_solver.py::TestSortedSolver::test_sorted_matches_hybrid` — argmax indices differ on 7/500 (1.4%) spectra
- `test_memory.py::TestChunkedParabola::test_chunked_matches_full` — argmax indices differ on 25/1000 (2.5%)
- `test_gvrt_multipeak_1b.py::TestEndToEndPSNR::test_psnr_targets_met` — δσ PSNR 29.4 < 30 dB
- `test_split_encoder_e2e.py::TestSeparationSweep::test_wide_separation_high_psnr` — δσ PSNR 28.8 < 30 dB

The mismatched indices land on *neighbouring* dictionary entries (max
index distance 15 and 11) — near-ties resolved the other way, not
garbage. Setting `NVIDIA_TF32_OVERRIDE=0` cleared all four.

### Counts

| run | passed | failed | skipped | wall |
|---|---|---|---|---|
| Step 0 — NumPy baseline (`TOYOMACRO_DISABLE_MLX=1`) | 690 | 3 | 81 | 2:02 |
| Step 2 — CUDA, TF32 on (default) | 765 | 8 | 1 | 9:00 |
| **Step 2 — CUDA, `NVIDIA_TF32_OVERRIDE=0`** | **769** | **4** | **1** | **7:49** |

Skips collapsing 81 → 1 is the load-bearing number: the `requires_mlx`
blocks are genuinely executing on CUDA, so the parity assertions are
real, not vacuous.

Step 1 probe returned `Device(gpu, 0)` / `mlx_usable: True` **unchanged**
— the capability probe is device-agnostic as predicted. Its
"Apple Silicon only" error strings are now wrong and should be reworded.

### Remaining failures (all speed assertions — excluded by the criteria)

| test | measured | target |
|---|---|---|
| `test_compression.py::TestFitparaCodec::test_decode_speed` | 3.4M spec/s | 20M |
| `test_compression.py::TestArrayLZ4Compression::test_decode_speed` | 8.5M spec/s | 50M |
| `test_compression.py::TestSpecdataUint16::test_mlx_throughput` | 27.3M spec/s | 50M |
| `test_gvrt_multipeak_1b.py::TestThroughput::test_fit_throughput_target` | 0.07 M/s | 0.5 M/s |

The two `test_decode_speed` failures are **not CUDA-related** — they fail
identically on the Step 0 NumPy baseline. They are pure-CPU codec paths;
`skip_in_ci` hid them on CI runners at the time of this run, and this
machine was likewise below what was then taken for a dev-hardware
threshold. (Both statements are historical: PR #22 removed that guard
and retired the threshold.) Worth a separate look (~~a Ryzen 9 8940HX
should not be 6× under a 20M target — suspect WSL2 CPU allocation or
laptop power policy~~ — superseded; the figures it rests on are not
comparable across hosts, see
[Third data point](#third-data-point-2026-09-21--every-figure-with-its-command)),
but not a CUDA signal.

`test_fit_throughput_target` at 0.07 M/s vs 0.5 M/s confirms this
document's own prediction: the streaming kernel is bandwidth-bound and
loses Apple's unified memory to the PCIe hop plus WSL2 overhead.
Judge later, per the criteria.

Note `test_gp_lm.py::test_gp_lm_background_end_to_end[2]` fails on the
NumPy baseline here (bg mean |28.8 - 120| vs a < 25.0 tolerance) yet
**passes** on CUDA. It is a known-marginal experimental path
(`newton_jacobian_mode="gp_lm"`, degree≥2 trades the constant against the
x² column). Unrelated to CUDA; flagged for the baseline, not this PoC.

### Risk map: how the predictions held

| predicted | outcome |
|---|---|
| FFT not hit (Doniach-Sunjic runs on `np.fft`) | ✅ correct — never hit |
| `mx.linalg.solve` safe via `stream=mx.cpu`; "verify early" | ✅ **CPU stream works on the Linux build** — no failures |
| Missing ops crash with no CPU fallback | not hit — no missing op encountered |
| `mx.compile` on CUDA unverified | ✅ works — compiled kernels pass parity |
| uint16 direct transfer | ✅ correct — only the throughput target missed |
| Threading unknown | untested — kernels kept on the main thread, per the recipe |

Unanticipated: the two setup fixes above. Neither is an MLX op problem.

### Reproducing

```bash
uv sync --extra dev                              # NOT --extra mlx (pins mlx-metal, Apple-only)
uv pip install "mlx[cuda13]"
uv pip install nvidia-cuda-runtime nvidia-cuda-cccl
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
export NVIDIA_TF32_OVERRIDE=0
uv run pytest src/toyomacro/voigtfit/tests -q
```

## Throughput & precision baseline (2026-07-17, follow-up investigation)

Correctness passed, so this section judges the part the criteria deferred:
what the CUDA backend is actually worth on this machine. Configs measured:
**NumPy** (`TOYOMACRO_DISABLE_MLX=1`), **CUDA TF32on** (backend default),
**CUDA TF32off** (`NVIDIA_TF32_OVERRIDE=0`). Same synthetic workloads,
same seeds.

### Precision: what TF32 costs end-to-end (`bench_dict2d`, 200k spectra)

| config | amp PSNR | FWHM PSNR | δσ RMSE | throughput |
|---|---|---|---|---|
| NumPy (`Dict2D_ultra+amp_only`) | 60.1 dB | 44.7 dB | 0.00174 | 0.17 M/s |
| CUDA TF32off | **60.1 dB** | **44.7 dB** | **0.00174** | 0.08 M/s |
| CUDA TF32on | 50.8 dB | 35.3 dB | 0.00513 | 0.14 M/s |

TF32off reproduces the NumPy accuracy **to every reported digit**, on all
four solver configs. TF32on costs **~9.4 dB of PSNR and 3× the δσ RMSE**
on the ultra-dense dictionary, and buys only ~1.7× within the pipeline.
For a fitting engine the trade is indefensible: **`NVIDIA_TF32_OVERRIDE=0`
should be considered mandatory on CUDA**, pending an in-process fix.

(TF32 = NVIDIA **TensorFloat-32**, unrelated to TensorFlow: fp32 range
[8-bit exponent] with an fp16-class 10-bit mantissa, used silently by
tensor cores for fp32 matmul. Measured matmul error 2.9e-04 vs 2.1e-07 —
the ~2^13 mantissa-bit gap, as expected.)

### NEW BUG: batch > 65,535 crashes the AP solver on CUDA

`solve_alternating_projection` (multipeak_solver.py:2853, `mx.eval`) dies at
production scale:

```
RuntimeError: cudaGraphAddKernelNode(...) failed: invalid argument      # graphs on
RuntimeError: cudaLaunchKernelExC(&config, func, params) failed: ...    # MLX_USE_CUDA_GRAPHS=0
```

Bisected: **65,535 spectra OK, 65,536 crashes** — exactly the CUDA
`gridDim.y/z` limit. Distilled to a one-line pure-MLX repro: the
**batched matrix-vector** kernel maps batch to a grid dimension untiled
(square batched matmul at the same batch size is fine):

```python
mx.eval(mx.zeros((65536, 225, 4)) @ mx.zeros((65536, 4, 1)))  # crashes; 65535 OK
```

`MLX_USE_CUDA_GRAPHS=0` only moves the error (`cudaLaunchKernelExC`),
and after the failure the CUDA context is poisoned — subsequent evals
abort. Metal has no such limit, so only CUDA trips it; the parity suite
never saw it (test batches are small). **Upstream MLX bug** (same class
as fixed mlx#2267/#3659/#2724); call-site workaround: chunk AP batches
to ≤ 65,535 (the chunked infrastructure from `test_memory.py` already
exists for this). Draft report: `docs/upstream-issues/mlx-issue-1-*.md`.

**Fixed upstream; measured here 2026-09-22.** `mlx#3929` split the batch
across `grid.y`/`grid.z` and was merged 2026-08-05;
[mlx#3858](https://github.com/ml-explore/mlx/issues/3858) closed
completed the next day, and MLX 0.32.2 shipped 2026-08-25. On this
machine at 0.32.2, `solve_multipeak_chunked` at 65,535 / 65,536 /
65,537 completes in a **single chunk** and agrees with the same problem
split in two to the **bit** on `amplitudes`, `delta_E` and
`delta_sigma`. Metal gives the same bit-identical agreement, which is
the control: chunking does not itself change the answer, so a CUDA
disagreement would have been the bug and there was none.

The chunking stays in the harness. It is what makes a Metal record and
a CUDA record the same measurement, and it protects anyone on an older
MLX. What changed is that it is no longer a workaround for a live
crash.

### Throughput: MLX CUDA GEMM is ~36× below the hardware on `sm_120`

Raw 4096³ fp32 GEMM, same GPU, same WSL2 session, same cuBLAS libraries:

| path | median | effective |
|---|---|---|
| **CuPy → cuBLAS direct** | **11.5 ms** | **11.96 TFLOP/s** |
| NumPy (Ryzen 9 8940HX) | 208.5 ms | 0.66 TFLOP/s |
| MLX `mx.matmul` (TF32off) | 435.2 ms | 0.32 TFLOP/s |
| MLX `mx.matmul` (TF32on) | 729.7 ms | 0.19 TFLOP/s |

The GPU itself is healthy — CuPy proves 12 TFLOP/s through the very same
wheels. MLX's matmul is the problem, and it decomposes as **~17× slow
kernel** (chained GEMMs still 196 ms each) **plus ~220 ms fixed dispatch
overhead per `mx.eval`**. During the slow runs `nvidia-smi` shows
2805 MHz / P0 / 100 % util at only **40 W** — SMs busy executing bad code.
`libmlx.so` ships CUTLASS kernels for **SM90 (Hopper)** only; on consumer
Blackwell (`sm_120`) it evidently falls back to something JIT-degraded.
MLX 0.32.0 is current at the time of writing, so this is a live upstream
limitation, not a stale install. TF32on being *slower* than TF32off in
the microbench is consistent with a mis-selected tensor-core code path.

### Pipeline throughput on this machine (median rates)

| workload | NumPy (CPU) | CUDA TF32off | CUDA TF32on | GPU verdict |
|---|---|---|---|---|
| Stage-1 streaming, 100k | **1.93 M/s** | 0.51 M/s | 0.82 M/s | CPU wins ~2–4× (doc predicted: PCIe/WSL2) |
| Stage-1 streaming, 1M | **1.38 M/s** | — | 0.82 M/s | CPU wins; GPU flat at 0.82 |
| `amp_only` n_comp=4, 65k | **1.09 M/s** | 0.23 M/s | 0.37 M/s | CPU wins ~3× |
| `4-step` n_comp=4, 65k | 0.26 M/s | 0.24 M/s | 0.22 M/s | parity |
| `dict2d` (AP) n_comp=4, 65k | 8.9 k/s | 36.2 k/s | 56.1 k/s | **GPU wins 4–6×** (doc predicted: compute-dense) |

Both of this document's predictions were correct: streaming loses to the
PCIe hop, compute-dense dictionary AP wins even on the crippled backend.
The headline is the upside: dict2d wins 4–6× while the GEMM engine runs
at ~3 % of the hardware. If upstream MLX ships `sm_120`-tuned kernels
(or routes matmul to cuBLASLt properly), a further order of magnitude is
on the table without touching voigtfit code.

### The same machine re-measured (2026-09-21/22, records committed)

The table above is 2026-07, read off a prose report. The one below is
the committed harness at `n_batch=200,000`, single-peak, on the same
laptop, pooled over eight runs — three CUDA and five NumPy, spanning
three harness generations. **It is a different workload, not a
correction of the table above**: the 2026-07 rows measure `n_comp=4` at
65 k and Stage-1 streaming at 100 k/1 M. Both stay.

Ranges are min–max of `rate_median` across runs, not one run's own
spread. Records are in `benchmarks/records/`, filenames beginning
`20260921T1011`, `…T1129`, `…T1134` and `20260922T0117`, `…T0123`.

| solver | CUDA TF32off | NumPy (same host) | verdict |
|---|---|---|---|
| `projection_kernel` | 4.68–9.67 M/s | **18.76–20.11 M/s** | CPU wins ~2–4× |
| `amp_only_projection` | 9.27–18.65 M/s | **33.42–88.95 M/s** | CPU wins ~2–7× |
| `multipeak_2comp` | 31.7–40.6 k/s | **111.7–117.8 k/s** | CPU wins ~3.5× |
| `dict2d_parabola` | **1.37–1.48 M/s** | 0.63–0.74 M/s | GPU wins ~2× |
| `taylor_4step` | **0.43–0.50 M/s** | 0.35–0.40 M/s | GPU wins, narrowly |

No solver's ranges overlap, so the direction of each row holds across
every run taken. `taylor_4step` is the one with little margin — 0.43
against 0.40 — and nothing more than "it does not overlap today"
should be read into it. Whether that margin survives the dispersion is
answerable by `--runs N`, which reports `understates_by` per solver:

```bash
NVIDIA_TF32_OVERRIDE=0 python -m toyomacro.voigtfit.benchmarks.bench_platform \
    --origin windows --runs 3 --records-dir benchmarks/records
```

#### What the CPU-wins rows are measuring

Not the GPU. `projection_kernel` reads 151 float32 channels per
spectrum and writes little, so its rate converts straight to a
bandwidth: 604 bytes × spectra/s.

| host, backend | rate | effective bandwidth |
|---|---|---|
| M3 Max, Metal | 496.3 M/s | 300 GB/s |
| Ryzen, NumPy | 18.8–20.1 M/s | 11.3–12.1 GB/s |
| RTX 5070 Laptop, CUDA | 4.68–9.67 M/s | **2.8–5.8 GB/s** |

The first two are those machines' memory bandwidths — unified memory on
the M3 Max, DDR5 on the Ryzen — which is what a memory-bound kernel
should report. The third is two orders of magnitude under this card's
memory bandwidth, and sits in the range of its PCIe link.

That is [`mlx#3861`](https://github.com/ml-explore/mlx/issues/3861),
confirmed on this machine: WSL2 reports `concurrentManagedAccess == 0`,
so MLX's unified allocator falls back to `cudaMallocHost`, and an array
built from host data — which is what `mx.array(numpy_array)` does, and
what this kernel does — stays in pinned host memory for life. Every
step streams its operand across PCIe.

So "CPU wins ~2–4×" is a true measurement of this software stack on
this host, and not a statement about the GPU. Read it as: under WSL2,
MLX's memory-bound paths do not reach the card. The rows the GPU still
wins are the compute-dense ones, which is consistent with a per-byte
penalty rather than a per-flop one — though that split has not been
measured separately, and nothing here establishes it.

**The fix is not on the Windows side.** There is no MLX CUDA backend
for Windows to move to (see Target environment). Measuring CUDA without
this fault means native Linux on this hardware, or waiting for the
#3861 patch and re-measuring here — the second is the better
experiment, because it moves the allocator while holding the hardware,
the driver and WSL2 fixed.

#### The link generation moves mid-run, and halves the rate

Measured 2026-09-22 on this machine: `nvidia-smi.exe` sampled from the
**Windows side** at 0.5 s while a probe timed each repetition of the
projection kernel.

| | rate | effective bandwidth | link |
|---|---|---|---|
| repetitions 1-2 | 10.6 M/s | 6.4 GB/s | `gen4 x8` |
| repetitions 3-9 | 5.4 M/s | 3.28 GB/s | `gen3 x8` |

Not a warm-up transient: it held for seven repetitions, which is enough
to move the median of nine. `gen4` to `gen3` is exactly a factor two
per lane, and **the width never moved off `x8` in any sample** — the
halving needs no width change. The efficiency is the same either side
of the step, 40.6% of `gen4 x8`'s 15.75 GB/s and 41.6% of `gen3 x8`'s
7.88 GB/s, which is what a fixed-fraction PCIe stream looks like when
the line rate halves under it. Across the window the box visited
`gen1`, `gen2`, `gen3` and `gen4`, all at `x8`.

**The same step is in the committed records.** `summarize` stores every
repetition in `timings_s`. Converted to bandwidth,
`…011702…from-windows.json` reads:

    5.11  3.19  5.87  5.84  5.87  5.87  5.84  5.87  5.84   GB/s

Seven repetitions sit at 5.84–5.87 GB/s, one drops to 3.19, and the
first reads 5.11. The two low-mode records hold eight of their nine
repetitions at 2.80–2.92 GB/s; their first repetitions read 3.53 and
3.50.

**Nobody sampled the link during those runs**, so the generation each
repetition ran at is inferred, not observed. The inference: the two
plateaus are a factor two apart, which is the `gen4`/`gen3` per-lane
ratio, and each sits at about the same fraction of its line rate (~37%),
as the probe's two did. The harness runs a few points less efficiently
than the probe, which is why its absolute values sit below the probe's.
That is strong circumstantial support. The only run in which generation
and rate were recorded together is the probe above.

Nothing had to be added to the schema to see the step — it had to be
read.

**`nvidia-smi` inside WSL2 cannot see it.** It returns `gen.max` for
`pcie.link.gen.current`: `4`, always, idle or loaded. Only the
Windows-side `nvidia-smi.exe` reports the live value. A sampling run
driven from inside WSL2 shows a perfectly steady `gen4` while the link
steps underneath it, and 78 such samples say no more than one does.

So the 2.07× spread between this host's CUDA records is neither
measurement noise nor a property of the solver. It is which PCIe
generation the link happened to be in, and the record's own
`timings_s` shows which.

**The GPU's one decisive 2026-07 win is the row it loses here.** That
table has dictionary AP 4–6× ahead on the GPU at `n_comp=4, 65k`;
`multipeak_2comp` at 200 k puts the CPU 3.5× ahead. The CUDA side
barely moved — 36.2 k/s then, 31.7–40.6 k/s now — and the NumPy side
went from 8.9 k/s to 111.7–117.8 k/s. Component count, batch size and
five months of solver work all differ between the two, so this is not
one number contradicting another; it is a caution that "compute-dense
wins on the GPU" was established at one operating point and does not
survive being moved off it. The upside stated above — a GEMM engine at
~3 % of the hardware — is unchanged either way.

### Operating guidance for this machine (RTX 5070 Laptop, WSL2)

- Always: `MLX_ENABLE_TF32=0` (MLX-native; measured equivalent to the
  driver-level `NVIDIA_TF32_OVERRIDE=0`, and settable in-process via
  `os.environ` before the first kernel call). AP batch chunks ≤ 65,535
  are no longer required at MLX 0.32.2 (mlx#3858 is fixed, measured); the
  harness keeps chunking so that Metal and CUDA measure the same work.
- Streaming / amp-only paths: **stay on NumPy** — the Ryzen wins today.
- Multipeak AP / dict2d at scale: CUDA is already a real 4–6× win.
- Re-benchmark on each MLX release; the 36× GEMM gap is the number to watch.
- `cupy-cuda13x` is installed in the venv as the ground-truth harness
  (`uv pip install cupy-cuda13x`); keep it for regression comparisons.

### Escape hatch: routing hot GEMMs through CuPy (measured)

The 36× gap is fixable without waiting for upstream. A thin shim that
hands the hot matmuls to CuPy/cuBLAS recovers most of the hardware:

| path | 4096³ GEMM | vs pure MLX |
|---|---|---|
| pure MLX | 435 ms | 1× |
| MLX→host→CuPy GEMM→host→MLX (full roundtrip) | 37 ms | **12×** |
| dictionary resident in CuPy, stream spectra only | 15 ms (9.1 TFLOP/s) | **29×** |

Caveat: **zero-copy DLPack MLX→CuPy is blocked** — MLX's CUDA arrays
export `__dlpack_device__` as CPU, so CuPy refuses the direct import
and the roundtrip must go through host memory (`np.from_dlpack` is at
least a zero-copy host view). Still 12–29× ahead. The dictionary-
resident variant maps naturally onto the dict2d/AP solvers, whose
dictionaries are constant across the batch loop.

### Upstream MLX issues filed from this PoC

Four, all accepted by the tracker; the drafts and repro scripts are in
`docs/upstream-issues/`.

| issue | subject |
|---|---|
| [mlx#3858](https://github.com/ml-explore/mlx/issues/3858) | batched-GEMV crashes for batch > 65,535 — **fixed** in mlx#3929, verified here at 0.32.2 |
| [mlx#3859](https://github.com/ml-explore/mlx/issues/3859) | the `[cuda13]` extra is missing runtime/CCCL headers |
| [mlx#3860](https://github.com/ml-explore/mlx/issues/3860) | TF32 on by default, undocumented |
| [mlx#3861](https://github.com/ml-explore/mlx/issues/3861) | `sm_120` GEMM ~36× under cuBLAS |

### Where `MLX_ENABLE_TF32` lives

The operating guidance above prefers MLX's own flag to the driver
variable. It is easy to miss: `MLX_ENABLE_TF32` is declared in
`mlx/utils.h` with default `1`, and selects `CUBLAS_COMPUTE_32F` over
`CUBLAS_COMPUTE_32F_FAST_TF32` in `cublas_gemm.cpp`. Nothing outside
`utils.h` and `mlx_tests.py` mentions it, which is why the earlier
recipes in this document reach for `NVIDIA_TF32_OVERRIDE=0` instead —
and why mlx#3860 is filed as "undocumented bad default" rather than
"no opt-out API".

MLX reads the flag lazily, on first use, so setting
`os.environ["MLX_ENABLE_TF32"] = "0"` before the first kernel call
takes effect in-process; the driver variable is process-wide and has to
be exported before Python starts.

Measured against the driver override on this machine: the MLX flag
reproduces the 2.08e-07 that the precision table above records for
`NVIDIA_TF32_OVERRIDE=0`, and the four tests that fail under TF32 pass
with the MLX flag alone. GEMM time is 421 ms with the MLX flag against
420 ms with the driver override — which also rules out one reading of
the `sm_120` result above: the 36× gap is not an artifact of how TF32
was disabled.

### What this PoC did not wire into the package

Stated as limits, not plans — none of these is scheduled here.

- `toyomacro` does not set `MLX_ENABLE_TF32` at CUDA-backend init.
  Anyone on this path sets one of the two variables themselves, and
  a run that forgets loses about three decimal digits silently.
- `solve_alternating_projection` is not chunked against the 65,535
  batch limit on the CUDA path. This was a live crash when written; it
  is not one at MLX 0.32.2, where a single chunk of 65,537 agrees with
  a split one bit for bit. Left unwired deliberately: adding a guard
  now would work around a fixed bug.
- The `[cuda13]` header install and `CUDA_HOME` are manual, as the
  setup recipe above spells out; the package does neither.

## Second data point (2026-09-21) — GPU-less x86 control

The 2026-07 run left open whether the NumPy-baseline `decode_speed`
gap was specific to the RTX 5070 box. It is not. The same assertions were run on
an unrelated cloud container with **no GPU at all** — a control for the
pure-CPU codec paths, not a CUDA run. Nothing here bears on the CUDA
verdict above.

### Environment

| | |
|---|---|
| CPU | Intel Xeon @ 2.80GHz, 4 vCPU |
| RAM | 15 GB |
| Host | Linux 6.18.44-fc-v37; no NVIDIA device (`nvidia-smi`, `nvcc`, `libcuda` all absent) |
| Python / NumPy / SciPy | 3.12.3 / 2.3.5 / 1.17.0 |
| BLAS | scipy-openblas 0.3.30 |
| lz4 | 4.4.5 |
| toyomacro | 0.2.0 |

### Measured

| test | RTX 5070 box | this control | target |
|---|---|---|---|
| `TestFitparaCodec::test_decode_speed` | 3.4M spec/s | 3.9–4.0M spec/s | 20M |
| `TestArrayLZ4Compression::test_decode_speed` | 8.5M spec/s | 16.1–17.0M spec/s | 50M |

Ranges are the min and max of five consecutive runs of the two tests in
isolation; the spread within that set is under 6%. The other two
speed entries in the Step 2 table are MLX-gated and skip cleanly with no
GPU, so this control says nothing about them.

~~**The WSL2/power-policy hypothesis is refuted.** A stock 4-vCPU cloud
Xeon is *faster* than the Ryzen 9 8940HX on both codecs — roughly 2× on
the LZ4 array path — and still lands 5× under the 20M target and 3×
under the 50M one. Two unrelated hosts failing the same way points at
the thresholds, not at either machine.~~ **Struck 2026-09-21 — wrong on
a fact, and too broad.** See
[Third data point](#third-data-point-2026-09-21--every-figure-with-its-command)
below: the comparison above was against the Ryzen *under WSL2* and
under a different pytest command, so it reads as a claim about the
silicon that its figures cannot support. What can be said without
crossing commands is narrower: the Ryzen's 35.8M is a bare-`pytest`
figure and this Xeon's direct-call figures are 19.8–33.3M, so they are
not comparable; under a bare `pytest` the Xeon cleared 50M, which is
the only like-for-like pair available and does not put it behind.
`git log -S "rate > 20"`
traces the assertion only to the squashed import commit, so neither
target has recorded provenance, nor a machine where it was ever met —
that part stands.

### The assertions also depend on pytest's collection scope

`TestArrayLZ4Compression::test_decode_speed` **fails at ~16.7M** under
`pytest src/toyomacro/voigtfit/tests` (the Step 2 command) but **passes
— clearing 50M** under a bare `pytest`, which also collects the
top-level `tests/` root. Both outcomes reproduce. `TestFitparaCodec`
measured 3.8M and 6.0M on two bare-`pytest` runs against 3.9–4.0M
isolated, i.e. a wider spread under the larger collection.

Allocator or page-cache state is the obvious suspect, and the cause was
not chased further. The point stands without it: an assertion whose
verdict flips with the set of *other* tests collected alongside it is
not measuring the codec, and a green result from it is not evidence.

### Consequence

~~These two assertions currently certify the host, not the code, and
`skip_in_ci` means nothing re-derives their thresholds. Decide either a
target with recorded provenance, or move them to
`src/toyomacro/voigtfit/benchmarks/` where a number without a pass/fail
gate is the expected artifact.~~ **Settled 2026-09-21** — both are now
ratio gates; see the next section.

## Third data point (2026-09-21) — every figure with its command

PR #15 measured both codecs on Windows 11 natively. **Provenance
caveat:** that host is believed to be the same Ryzen 9 8940HX as the
RTX 5070 box above, but no CPU is recorded for it anywhere — PR #15's
body and the CHANGELOG entry at `800f439` both say only "Windows 11
(Python 3.12.14)". The identification comes from the maintainer, not
from the repository. Nothing below depends on it; an earlier draft of
this section did, and was wrong to.

Every published figure for these two gates, with the command that
produced it — because the command turns out to matter more than the
hardware:

| host | command | fitpara | array |
|---|---|---|---|
| Ryzen 9 8940HX, WSL2 | `pytest src/toyomacro/voigtfit/tests` | 3.4M spec/s | 8.5M spec/s |
| Ryzen 9 8940HX, native Win11 | bare `pytest` (both roots) | 4.9M spec/s | 35.8M spec/s |
| Xeon @2.80GHz, 4 vCPU | two tests in isolation, ×5 | 3.9–4.0M spec/s | 16.1–17.0M spec/s |
| Xeon @2.80GHz, 4 vCPU | bare `pytest` (both roots) | 3.8M, 6.0M | > 50M (passed) |
| Xeon @2.80GHz, 4 vCPU | direct call, ×5 | 5.2–6.7M spec/s | 22.7–33.3M spec/s |
| Xeon @2.10GHz, 4 vCPU | direct call, ×8 | 4.7–5.7M spec/s | 19.8–23.4M spec/s |
| Apple M3 Max, load ≈11 | `pytest` | 16.4M spec/s | — |
| Apple M3 Max, load ≈11 | direct call, ×7 | 20.1–20.8M spec/s | — |
| target | | 20M | 50M |

Two of those Xeon rows are not the same machine: the container was
re-provisioned mid-session from a 2.80GHz part to a 2.10GHz one, which
is why its figures drop. Both are 4-vCPU virtualised instances with
hypervisor steal measured at 0.00–0.01% over the measurement windows,
so contention is not what separates them. The M3 Max rows were taken
under a 1-minute load average of about 11 on 16 cores and are therefore
lower bounds on that host, not clean figures.

**No WSL2 cost can be read off this table.** An earlier draft asserted
1.44× and 4.21× from the first two rows; those rows differ in the
command as well as the environment, and the section above already
records that the command alone moves the array figure from ~16.7M to
over 50M on one host. The difference-of-differences is not attributable.

What survives:

- **Only an Apple host has met the 20M target.** A contended M3 Max
  reaches 20.1–20.8M by direct call; the best x86 figure on any host,
  command or clock is 6.7M, 3.0× under. An earlier draft called 6.0M
  the best and computed 3.3× from it, while a row of this same table
  recorded 6.7M — the superlative was updated without being re-derived
  against the data added beside it.
- **The 50M target has been met once**, by this Xeon under a bare
  `pytest` (> 50M, passed). Comparing like with like is only possible
  within a command: by direct call the Ryzen has no figure, the two
  Xeons give 19.8–33.3M, and under a bare `pytest` the Ryzen gives
  35.8M against the Xeon's > 50M. An earlier draft compared the Ryzen's
  bare-`pytest` figure to a Xeon direct call and read "rough parity"
  off it; that is a cross-command comparison, the thing this section
  exists to stop.
- **The spread is dominated by measurement context, not by silicon.**
  On one Xeon the array figure spans 16.1M to over 50M on collection
  scope alone, and the same pattern appears on Apple hardware: 16.4M
  under `pytest` against 20.1–20.8M by direct call. That is the finding
  this table is for, and it is the one that holds across every host.

### What was done about it

Neither target survives as an absolute rate, for a reason the table
above makes sharper than the earlier argument did: **one machine, one
build, one codec, one command apart — 16.1M to over 50M.** An earlier
draft put the headline at "4.2× apart" from the WSL2/native pair, but
that computation is retracted above and those two differ in build and
command as well. The collection-scope spread needs no such caveat, and
it is larger. No absolute spec/s can be right on both sides of it, and
picking either turns the gate into a statement about where the suite
happens to run.

Both assertions are now ratios against a plain copy of the array decode
produces. **The two sides are not symmetric, and an earlier draft of
this section was wrong to say they were.** A copy is bandwidth-bound;
decode is not. Decomposed on the array codec here, 73.5% of decode is
`lz4.frame.decompress` running at 1.44 GB/s out, against 15.0 GB/s for
memcpy — an order of magnitude below it per byte. Counting traffic
rather than writes, decode moves ~120 MB (LZ4 writes 40, then
`frombuffer().copy()` reads 40 and writes 40) against the copy's 80 MB,
so a bandwidth-only floor is ~1.5 and the measured ratio is ~7–8. For
the fitpara codec the int16 → float32 dequantise dominates instead.

So the ratio is compute over bandwidth, and the two track each other
only loosely across microarchitectures. What it does buy is immunity to
the measurement-context spread above, which is what made an absolute
rate uncalibratable. The bound is 40× a copy against ratios measured at
8.4–13.3 (fitpara) and 6.9–8.7 (array) on 4-vCPU Xeons — session
extrema of a noisy statistic rather than a settled range, and one host
class rather than a cross-machine span. It is set far above them rather
than tight, precisely because the ratio's stability across hardware is
an argument and not yet a measurement. A regression has to be roughly
3.7× (fitpara) or 4.9× (array) before the gate fires: this catches a
kernel that stopped being vectorised, not a 20% slowdown.

`skip_in_ci` is gone with them, and the helper it needed. Both gates now
run everywhere, including CI, and the pytest-collection-scope
sensitivity in the section above no longer flips a verdict: it moves a
ratio whose bound is nowhere near either value. Encode was considered as
the baseline and rejected — for the array codec decode is *slower* than
encode (a near-constant payload compresses almost for free but still
writes 80 MB on the way out — 40 MB out of LZ4, then 40 MB again through
`np.frombuffer(...).copy()`), so the comparison would have been
backwards. That second copy is the very operation the gate compares
against, which is what puts a structural floor near 1 under the ratio.

A `decode_speed` failure on a new machine now says something narrower
than a rate did, and something broader: decode has moved relative to a
memcpy on the same host. That is not by itself proof of a regression —
the ratio is compute over bandwidth and is only loosely stable across
microarchitectures — but it is no longer a statement about how fast the
machine is.

## Re-verification (2026-09-21) — RTX 5070 Laptop

**Verdict: PASS.** The parity suite passes on CUDA within existing
tolerances once TF32 is disabled. Every remaining failure is a speed
assertion, which the pass/fail criteria explicitly exclude. No
missing-op failure appeared, so no `stream=mx.cpu` workaround was
needed — the same outcome as 2026-07-17, reached on a tree, a driver
and a CUDA stack that have all moved since.

### Environment

| | |
|---|---|
| **Commit SHA verified** | **`3a13de5ee56b5f5e084dee2772081d5955eec87d`** (clean tree) |
| GPU | NVIDIA GeForce RTX 5070 Laptop (Blackwell, `sm_120`), 8151 MiB |
| Driver | 610.71 (was 596.13 in 2026-07) |
| CPU / host | AMD Ryzen 9 8940HX — Windows 11 + WSL2, Ubuntu 24.04, kernel 6.18.33.2-microsoft-standard-WSL2 |
| Python / NumPy / SciPy | 3.12.3 / 2.3.5 / 1.17.0 |
| MLX | 0.32.2 (`mlx-cuda-13` 0.32.2), default device `Device(gpu, 0)` |
| CUDA wheels | cublas 13.8.0.4, nvrtc 13.4.92, cuda-runtime 13.4.92, cuda-cccl 13.3.4.3.1, cudnn 9.26.0.51, cufft 12.4.0.43 |
| `CUDA_HOME` | `.venv/lib/python3.12/site-packages/nvidia/cu13` — 93 headers present |
| TF32 disabled via | `NVIDIA_TF32_OVERRIDE=0` |

`src/` is unchanged between the verified `3a13de5` and `02095d8`, main
at the time of writing: the seven files that differ are CI config,
`README.md`, `AGENTS.md`, `CHANGELOG.md`, `CITATION.cff`,
`pyproject.toml` and `uv.lock`. This run therefore covers main's engine
code as it stands, not only the commit named above.

### Counts

Whole suite, `pytest src/toyomacro/voigtfit/tests -q -rs`; 981 tests
collected in every run.

| run | passed | failed | skipped | wall |
|---|--:|--:|--:|--:|
| Step 0 — NumPy baseline (`TOYOMACRO_DISABLE_MLX=1`) | 920 | 2 | 59 | 177 s |
| Step 2 — CUDA, TF32 on (default) | 973 | 8 | 0 | 559 s |
| Step 2 — CUDA, TF32 disabled | 976 | 5 | 0 | 481 s |

Step 1 probe: `Device(gpu, 0)`, `mlx_installed: True`, `mlx_usable:
True`. The capability probe is one of the changed files and is
unchanged in behaviour on CUDA.

### 1. The skips collapsed — completely

All 59 skips in the NumPy baseline are MLX-gated, and all 59 executed
on CUDA. The 2026-07 collapse was 81 → 1; this one is 59 → 0.

| skip reason (Step 0) | count |
|---|--:|
| `MLX not available` | 50 |
| `mx.compile not available` | 2 |
| `MLX unavailable` | 2 |
| `MLX not usable (not installed, or the default device failed the probe)` | 2 |
| `tolerance calibrated for the MLX path (fails on the NumPy backend)` | 1 |
| `compares the MLX exact-Voigt path against scipy` | 1 |
| `MLX throughput target not meaningful on CI or CPU fallback` | 1 |
| **total** | **59** |

A green Step 2 therefore asserts something: 53 more tests ran with TF32
on than in the baseline, and 56 more with it off.

### 2. The four TF32 canaries behaved exactly as in 2026-07

With TF32 on, these four fail and nothing else correctness-related
does. With TF32 disabled, all four pass:

- `test_dictionary_solver.py::TestSortedSolver::test_sorted_matches_hybrid`
- `test_memory.py::TestChunkedParabola::test_chunked_matches_full`
- `test_gvrt_multipeak_1b.py::TestEndToEndPSNR::test_psnr_targets_met`
- `test_split_encoder_e2e.py::TestSeparationSweep::test_wide_separation_high_psnr`

The canary set is neither larger nor smaller than it was, across a
driver bump, an MLX bump and 193 commits. That is the finding.

### 3. Every remaining failure is a speed assertion

With TF32 disabled, five fail:

| test | file | asserts |
|---|---|---|
| `TestFitparaCodec::test_decode_speed` | `test_compression.py` | ≥ 20M spec/s |
| `TestArrayLZ4Compression::test_decode_speed` | `test_compression.py` | ≥ 50M spec/s |
| `TestSpecdataUint16::test_mlx_throughput` | `test_compression.py` | throughput target |
| `TestThroughput::test_extended_throughput` | `test_extended_svd.py` | extended pipeline within 3× of standard |
| `TestThroughput::test_fit_throughput_target` | `test_gvrt_multipeak_1b.py` | fit throughput target |

The two `decode_speed` assertions are pure-CPU codec paths that fail on
every host measured so far — see
[Second data point](#second-data-point-2026-09-21--gpu-less-x86-control).
`test_extended_throughput` failed only in the TF32-off run and passed
with TF32 on; it compares two timings against a 3× ratio, so it moves
with scheduling noise and a slower GEMM. None of the five asserts a
numerical result.

**Superseded for the two `decode_speed` rows by PR #22**, which is not in
`3a13de5`: both became ratio gates against a copy of the array decode
produces, so they no longer carry an absolute rate and are expected green
here. The record above stands as measured on the tree that was run.

### Performance, which is not a criterion

The CUDA suite took 559 s (TF32 on) and 481 s (off) against 177 s for
the NumPy baseline on the same host — 2.7× to 3.2× slower in wall
clock. The `sm_120` GEMM shortfall recorded in 2026-07 is not fixed by
driver 610.71 or MLX 0.32.2.

### The batch > 65,535 limit, tested 2026-09-22

This 2026-09 suite completed without hitting the crash
([mlx#3858](https://github.com/ml-explore/mlx/issues/3858)) only
because its batches stay at or below that size, so it was **not**
evidence either way. It has since been tested directly: the limit is
gone at MLX 0.32.2, bit-identically. See the resolution note under
"NEW BUG: batch > 65,535" above.

### Two traps that cost 40 minutes here

Both produce a run that looks fine and is not.

- **`$VAR` in a `wsl.exe -- bash -lc '...'` string is expanded by the
  Windows shell first.** `CUDA_HOME="$PWD/..."` silently became a
  Windows path that does not exist, i.e. `CUDA_HOME` unset. Most of the
  suite still passed, and the failure that did appear was in
  `test_extended_svd.py`, which is exactly where a missing NVRTC header
  would show up — so the artefact imitated a real finding. Write the
  script to a file with a quoted heredoc instead, and assert
  `[ -d "$CUDA_HOME/include" ]` before running anything.
- **WSL2's `/tmp` did not survive between invocations here.** Logs
  written there were gone when the run finished. Keep run logs under
  `$HOME`.

### Correction: the 2026-07-17 PoC tree is `3aedcf3`, and it is in this history

The frame this section replaces asserted that the PoC "validated a tree
that no longer exists", and measured the delta from `97f444d`
(2026-07-18) as the earliest visible commit. Both are wrong; they came
from a shallow clone. The history runs back to `fe2e7ef`
(2026-07-07) and held 187 commits when this run started.

The tree the PoC ran on is **`3aedcf3`** (2026-07-16,
`feat(cli): restrict public command surface to import/convert`):

- `0bb1d14`, which recorded the PoC results, is dated 2026-07-17 11:15
  JST and changes five files, all under `docs/` — no code. Its parent
  is `3aedcf3`.
- Nothing outside this history is needed to place it. `3aedcf3` is
  reachable from `main`, and the 2026-07-17 Environment table above
  records the toolchain that run used — MLX 0.32.0, cublas 13.6.0.2,
  driver 596.13 — none of which the 2026-09-21 re-verification below
  shares. The two are separate runs on one machine, not one record
  read twice.

Correcting the base changes the numbers, and the conclusion partly. The
real delta `3aedcf3..3a13de5` is **90 files, +11,807 / −4,264 across
193 commits** — the frame's figure missed 17 commits and 3,909 lines.
Those 17 add `exact_k.py`, `model_selection.py`, `rank_diagnostics.py`,
their tests and a benchmark; every one is a new file and none touches
an MLX path, so the frame's judgement that they do not matter here
holds. But the corrected base lists **five** files on the certified
path, not three:

| file | change since `3aedcf3` |
|---|--:|
| `pipeline.py` | 49 |
| `_mlx_support.py` | 38 |
| `multipeak_solver.py` (incl. `d989712`) | 21 |
| `weight_cache.py` | 6 |
| `dictionary_solver.py` | 2 |

`weight_cache.py` and `dictionary_solver.py` were invisible from the
wrong base. All five are covered by the runs above.
