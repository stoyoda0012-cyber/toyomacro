# CUDA backend proof-of-concept (MLX on Linux / WSL2)

**Status:** **PASSED** — first run 2026-07-17 (RTX 5070 Laptop). See
[Results](#results-2026-07-17--rtx-5070-laptop) below. Two setup fixes were
required that this recipe did not anticipate (CUDA headers, TF32).
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
  it is expected to return True on CUDA unchanged. (Its error messages
  said "Apple Silicon only" when this recipe was written; they were
  reworded after the run below — see follow-up 2.)

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

## Results (2026-07-17) — RTX 5070 Laptop

**Verdict: PASS.** The parity suite passes on CUDA within existing
tolerances. Every remaining failure is a speed assertion, which the
pass/fail criteria explicitly exclude. No missing-op was hit, so no
`stream=mx.cpu` workaround was needed.

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
`skip_in_ci` hides them on CI runners, and this machine is likewise below
the dev-hardware threshold. Worth a separate look (~~a Ryzen 9 8940HX
should not be 6× under a 20M target — suspect WSL2 CPU allocation or
laptop power policy~~ — that hypothesis is refuted; see
[Second data point](#second-data-point-2026-09-21--gpu-less-x86-control)),
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

### Operating guidance for this machine (RTX 5070 Laptop, WSL2)

- Always: `MLX_ENABLE_TF32=0` (MLX-native; measured equivalent to the
  driver-level `NVIDIA_TF32_OVERRIDE=0`, and settable in-process via
  `os.environ` before the first kernel call) and AP batch chunks ≤ 65,535.
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

### Suggested follow-ups

1. **Decide the TF32 policy.** `NVIDIA_TF32_OVERRIDE=0` is a blunt
   process-wide env var and an easy thing to forget — a silent 1000×
   precision loss is a bad default for a fitting engine. Prefer setting
   it in-process at CUDA-backend init, or gate the argmax paths, and
   measure what TF32 actually buys before trading accuracy for it.
2. ~~Reword the `_mlx_support` "Apple Silicon only" messages.~~ —
   **DONE.** `_mlx_support` now describes the probe as running on MLX's
   default device, and neither `require_mlx()` message claims MLX is
   Apple-only or that the probe looks for a Metal device. The same
   correction was applied to the 33 `HAS_MLX = _mlx_usable()` comments
   that repeated the claim, to the `test_compression.py` skip messages,
   and to the `solver_comparison_benchmark` provenance label, which had
   hardcoded `"mlx (Apple Silicon GPU)"` for any accelerated run.
   `tests/test_mlx_support.py` now pins the messages against a
   regression. Detection behavior is unchanged.
3. Fold the header install + `CUDA_HOME` into the setup recipe above.
4. ~~Investigate the NumPy-baseline `decode_speed` gap on this machine.~~
   — **answered 2026-09-21: not machine-specific.** A GPU-less x86
   control reproduces both shortfalls while running *faster* than this
   box, and the assertions additionally flip with pytest's collection
   scope; see
   [Second data point](#second-data-point-2026-09-21--gpu-less-x86-control).
   What is left is a decision — give the two targets recorded
   provenance, or move them to `benchmarks/`.
5. ~~Benchmark medians not yet recorded~~ — done; see the baseline section above.
6. ~~**File upstream MLX issues**~~ — **FILED 2026-07-17** from the Mac
   side, order 1→4→3→2, all four accepted by the tracker:
   [mlx#3858](https://github.com/ml-explore/mlx/issues/3858) (batched-GEMV
   batch>65,535 crash), [mlx#3859](https://github.com/ml-explore/mlx/issues/3859)
   (`[cuda13]` missing headers), [mlx#3860](https://github.com/ml-explore/mlx/issues/3860)
   (TF32 default, undocumented), [mlx#3861](https://github.com/ml-explore/mlx/issues/3861)
   (`sm_120` GEMM 36×). Cross-referenced (#3860 ↔ #3861). **Watch GitHub
   notifications for maintainer follow-ups** — verification requests run
   on the CUDA box. Original draft/dedup record below:

   Drafts in `docs/upstream-issues/`:
   (1) batched-GEMV batch>65,535 crash (one-line repro); (2) `sm_120` matmul
   ~36× under cuBLAS; (3) TF32-by-default, undocumented; (4) `[cuda13]`
   extra missing runtime/CCCL headers. Deduped against the tracker 2026-07-17
   (related closed: mlx#2267, #3659, #2724).

   **Second dedup pass (2026-07-17, Mac side, tracker + source):** MLX
   *does* have a native opt-out — `MLX_ENABLE_TF32` (`mlx/utils.h`,
   default `1`) gates `CUBLAS_COMPUTE_32F_FAST_TF32` vs
   `CUBLAS_COMPUTE_32F` in `cublas_gemm.cpp`. It is undocumented (only
   `utils.h` + `mlx_tests.py`), which is why we missed it; draft 3 was
   rewritten around "undocumented bad default", its original "no
   opt-out API" claim was wrong. Because the flag is read lazily on
   first use, setting `os.environ["MLX_ENABLE_TF32"]="0"` at
   backend-init time works in-process — follow-up 1 is implementable
   without the process-wide driver var. ~~Re-measure the precision
   table with `MLX_ENABLE_TF32=0` before filing draft 3~~ — **done
   (2026-07-17, CUDA box):** `MLX_ENABLE_TF32=0` measures 2.08e-07
   (identical to the driver override), GEMM perf 421 ms vs 420 ms
   (identical — the 36× gap is not an artifact of the disable method),
   the in-process `os.environ` set works (2.08e-07), and the four
   formerly TF32-failing tests pass with the MLX flag alone. Drafts 2
   and 3 updated with the measured numbers; all four are ready to file.
   Also folded into the drafts: #3666 (same gridDim class, fixed for
   binary/copy/unary but not batched GEMV) → draft 1; #3056 (consumer
   Blackwell graph limits) → draft 2; #2906/#2842/#2357/#2382 (header
   search exists but misses the `nvidia/cu13` layout) → draft 4.
7. Chunk `solve_alternating_projection` batches to ≤ 65,535 on the CUDA path
   (existing chunked infrastructure applies).

## Second data point (2026-09-21) — GPU-less x86 control

Follow-up 4 asked whether the NumPy-baseline `decode_speed` gap was
specific to the RTX 5070 box. It is not. The same assertions were run on
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

**The WSL2/power-policy hypothesis is refuted.** A stock 4-vCPU cloud
Xeon is *faster* than the Ryzen 9 8940HX on both codecs — roughly 2× on
the LZ4 array path — and still lands 5× under the 20M target and 3×
under the 50M one. Two unrelated hosts failing the same way points at
the thresholds, not at either machine. `git log -S "rate > 20"` traces
the assertion only to the squashed import commit, so neither target has
recorded provenance, nor a machine where it was ever met.

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

These two assertions currently certify the host, not the code, and
`skip_in_ci` means nothing re-derives their thresholds. Decide either a
target with recorded provenance, or move them to
`src/toyomacro/voigtfit/benchmarks/` where a number without a pass/fail
gate is the expected artifact. Until then, treat a `decode_speed`
failure on a new machine as uninformative — as the Step 2 criteria
already do.
