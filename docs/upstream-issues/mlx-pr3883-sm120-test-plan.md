# mlx#3883 sm_120 verification plan (WSL2 / RTX 5070)

Verifies the CUDA half of ml-explore/mlx#3883 (one-time TF32 warning),
which the author could only compile-check (no NVIDIA hardware). We
promised this run in ml-explore/mlx#3860.

**Outcome (2026-08-03): ml-explore/mlx#3883 was closed unmerged.** Not
on technical grounds — none of the verification below was disputed —
but on policy: warning on TF32 was judged not to be common practice,
with documentation and a programmable switch preferred instead. The
measurements here stand as a record of what was run and found; they are
not a description of current MLX behaviour, and nothing in this
repository depends on the warning existing.

## Review summary (Mac-side code read, 2026-07-19)

Design is sound: every call site reorders the existing short-circuit
`(env::enable_tf32() || dtype != float32)` to
`(dtype != float32 || env::tf32_active_for_fp32())`, so non-fp32 never
reaches the helper and fp32 warns exactly at the reduced-route decision.
The helper is thread-safe (magic static), prints once per process, and
only when `MLX_ENABLE_TF32` is unset. No behavioral change beyond the
log — verified across all 8 files of the diff.

**The one open question this run must answer (backend asymmetry):**
Metal's gates sit at NAX dispatch, which is shape-aware — matvec never
reaches them (author measured 0 warnings for matvec-only on M5). CUDA's
main gate is `dtype_to_compute_type()` in cublas_gemm.cpp, which is
shape-blind: if plain fp32 matvec routes through cuBLAS with
`CUBLAS_COMPUTE_32F_FAST_TF32`, the warning may fire for matvec-only
workloads even though (per our sm_120 measurements in mlx#3860) matvec
results stay fp32-exact. That would be a mild false positive unique to
CUDA — worth reporting either way, with a suggested shape guard if it
fires.

## Setup (WSL2 box)

```bash
# CUDA toolkit (nvcc) is required for a source build — the pip-wheel
# route from the PoC is NOT enough here. sm_120 needs CUDA >= 12.8/13.
sudo apt install cuda-toolkit-13-0   # NVIDIA repo; driver stays Windows-side

# GATE BEFORE BUILDING (do not skip): an older toolkit rejects
# CMAKE_CUDA_ARCHITECTURES=120 — and that failure can surface LATE in
# the build, burning the whole time budget. Verify nvcc accepts sm_120
# with a 10-second smoke first:
nvcc --version
printf '__global__ void k(){}\nint main(){return 0;}\n' > /tmp/sm120.cu
nvcc -arch=sm_120 /tmp/sm120.cu -o /tmp/sm120 && echo "sm_120 OK" \
  || echo "STOP: toolkit too old for sm_120 — upgrade before building"

git clone https://github.com/ml-explore/mlx.git && cd mlx   # or reuse a clone
# NOTE: the PR gained a second commit (d3d6c38, per-family warning text).
# If pr3883 was fetched before, DELETE and re-fetch to get the new head:
git branch -D pr3883 2>/dev/null; git fetch origin pull/3883/head:pr3883
git checkout pr3883 && git log --oneline -2   # expect d3d6c38 on top
CMAKE_ARGS="-DMLX_BUILD_CUDA=ON" pip install . --no-build-isolation -v
# (if the build flag differs, follow docs/src/install.rst in the checkout)
```

## Test matrix

Run each scenario as a SEPARATE process (the warning is once-per-
process). Since d3d6c38 the message is per-family:
`[mlx] float32 <family> ops are running at reduced (TF32) ...` with
family in {matmul, convolution, attention, quantized matmul, grouped
matmul}, named after whichever route engaged FIRST in the process.
Count stderr lines containing the family-agnostic substring
`running at reduced (TF32)` and ALSO record the family word — it is a
new checkable: the CUDA conv site is compile-only upstream, so
scenario 6 printing "convolution" is itself a result.

**Isolation is load-bearing, not hygiene**: any fp32 GEMM that runs
first — framework warm-up, an import side effect, a capability probe —
consumes the single warning and silently invalidates the scenario.
Every cell must be its own interpreter process with stderr captured
whole. And a "0 warnings" result is ambiguous on its own (genuinely
not triggered vs consumed earlier), so scenario 2 carries a built-in
control: after the matvec calls, run one fp32 GEMM *in the same
process*. If matvec printed nothing and the trailing GEMM then prints
the warning, the 0 is proven genuine — the machinery works and matvec
really did not trigger it. Report scenario 2 as the pair
(matvec-phase count, post-GEMM count).

| # | scenario (env / workload) | expected |
|---|---|---|
| 1 | env unset / fp32 GEMM x2 (512x512, twice in one process) | exactly 1, family "matmul" |
| 2 | env unset / fp32 matvec, then one fp32 GEMM as in-process control | **report pair** (Metal analog = 0; control GEMM must warn — proves a matvec 0 is genuine, not consumed) |
| 3 | env unset / bf16 GEMM | 0 |
| 4 | MLX_ENABLE_TF32=0 / fp32 GEMM | 0 |
| 5 | MLX_ENABLE_TF32=1 (explicit) / fp32 GEMM | 0 |
| 6 | env unset / fp32 conv2d (small NCHW) | expect 1, family "convolution" (cuDNN gate; wording untested upstream) |
| 7 | numerics: 512x512 fp32 GEMM rel err vs fp64 ref — default and =0 | ~2.9e-4 / ~2.1e-7 (matches mlx#3860) |
| 8 | bonus: voigtfit parity suite against the branch build | pass, warning appears at most once per process |

Driver script sketch (each cell via `subprocess.run([sys.executable, "-c", ...], env=...)`,
capture stderr, `count = stderr.count("running at reduced (TF32)")`):

```python
GEMM   = "import mlx.core as mx; a=mx.ones((512,512)); mx.eval(a@a); mx.eval(a@a)"
MATVEC = ("import mlx.core as mx; a=mx.ones((512,512)); v=mx.ones((512,1));"
          "mx.eval(a@v); mx.eval(v.T@a)")
BF16   = "import mlx.core as mx; a=mx.ones((512,512), dtype=mx.bfloat16); mx.eval(a@a)"
CONV   = ("import mlx.core as mx; x=mx.ones((1,32,32,8)); w=mx.ones((8,3,3,8));"
          "mx.eval(mx.conv2d(x,w))")
```

## Reporting

Paste the filled table as a comment on ml-explore/mlx#3883, including
the matvec answer (and, if it fires, whether we suggest a shape guard
vs accepting requested-mode semantics). Keep the same format as the
author's M5 table for side-by-side comparison.

## Results (2026-07-22, RTX 5070 Laptop / sm_120)

Branch build `0.32.0.dev20260722+d3d6c38a`, CUDA toolkit 13.3
(nvcc 13.3.73), `CMAKE_CUDA_ARCHITECTURES=120`, driver 596.13,
WSL2 Ubuntu 24.04, Python 3.12.3.

| # | scenario | count | family | verdict |
|---|---|---|---|---|
| 1 | env unset / fp32 GEMM ×2 | **1** | matmul | ✅ once per process |
| 2 | env unset / fp32 matvec → control GEMM | **pair (1, 0)** | matmul | ⚠️ **fires on matvec** — see below |
| 3 | env unset / bf16 GEMM | 0 | – | ✅ |
| 4 | `MLX_ENABLE_TF32=0` / fp32 GEMM | 0 | – | ✅ |
| 5 | `MLX_ENABLE_TF32=1` explicit / fp32 GEMM | 0 | – | ✅ |
| 6 | env unset / fp32 conv2d | **1** | **convolution** | ✅ cuDNN gate fires, correct family (first hardware exercise — upstream was compile-only) |
| 7 | numerics, default / `=0` | rel err **2.930e-04 / 2.077e-07** | matmul / – | ✅ matches mlx#3860 |
| 8a | voigtfit parity suite (774 tests), env unset | **1 warning**; 8 failed / 765 passed | matmul | ✅ = wheel-0.32.0 TF32-on failure set exactly; no behavior change |
| 8b | suite, `MLX_ENABLE_TF32=0` | **0 warnings**; 4 failed / 769 passed (all speed assertions) | – | ✅ = wheel TF32-off exactly |

**The backend-asymmetry answer (scenario 2): the warning is a false
positive for matvec on CUDA.** fp32 matvec fires it (the control GEMM
then stays silent — the single warning was consumed by the matvec
phase, proving the pair semantics), yet the matvec *result* is
fp32-exact: rel err **9.718e-08** vs a float64 reference, measured in
the same env-unset configuration that fired the warning. So CUDA warns
about a precision reduction that does not occur for that shape, where
Metal (author's M5 run) correctly stays silent. Cause as predicted:
`dtype_to_compute_type()` in `cublas_gemm.cpp` is shape-blind.
Suggested upstream: shape-gate the CUDA call site (warn only when
`n > 1` and `m > 1` for the output), or accept requested-mode
semantics and document the asymmetry.

**Bonus finding — pytest swallows the warning.** Under pytest's
default fd-level capture, the whole 774-test suite *appeared* to emit
0 warnings; the single warning only became visible with
`--capture=no`. It fires inside whichever test happens to run the
process's first fp32 GEMM — usually a passing test whose captured
stderr is discarded. Not a defect in the PR, but worth noting in the
discussion: in test-driven workflows the once-per-process warning can
be invisible in exactly the runs where TF32 is corrupting results.

### Build notes for this box (differences from the plan's sketch)

- `cuda-toolkit-13-3` (newer than the plan's 13-0; matches the pip
  wheel generation). sm_120 nvcc gate passed.
- cuDNN: NVIDIA's wsl-ubuntu apt repo carries no cudnn9 packages —
  pointed CMake at the venv's pip wheel instead via a shim dir with
  unversioned symlinks: `-DCUDNN_INCLUDE_PATH=$HOME/cudnn-shim/include
  -DCUDNN_LIBRARY_PATH=$HOME/cudnn-shim/lib`.
- `libopenblas-dev liblapacke-dev` required (CPU backend).
- Runtime needs `LD_LIBRARY_PATH` covering the venv's
  `nvidia/{cudnn,cu13,nccl}/lib` (source build has no wheel rpath).
- **`CMAKE_BUILD_PARALLEL_LEVEL=6`**, not nproc: a 32-way nvcc build
  OOM-crashed the entire WSL2 VM (31 GB host / ~15.6 GB WSL default).
- Restore the wheel afterwards with `uv pip install "mlx[cuda13]"`
  (the branch build replaced it in the toyomacro venv).

### Addendum for mlx#3861 (measured on the same branch build)

The native-arch build **does not change GEMM performance**: 4096³ fp32
GEMM on the `CMAKE_CUDA_ARCHITECTURES=120` source build measures
425.5 ms TF32off / 750.5 ms TF32on — statistically identical to the
wheel (435.2 / 729.7). This falsifies the "SM90-only CUTLASS falling
back to JIT-degraded code" hypothesis floated in #3861: compiling
natively for sm_120 changes nothing, so the 36× gap vs CuPy/cuBLAS
(11.5 ms on the same library files) lives in how MLX invokes the GEMM
(algo heuristic / workspace / dispatch), not in cross-arch JIT.
Suggested one-line comment for #3861:

> Datapoint: rebuilt from source with CMAKE_CUDA_ARCHITECTURES=120
> (CUDA 13.3, RTX 5070 Laptop) — 4096³ fp32 GEMM is unchanged
> (425 ms native vs 435 ms wheel, TF32 off both). So the gap is not
> PTX JIT for a missing arch; CuPy hits 11.5 ms through the same
> cublas wheel, which points at the MLX-side cublasLt invocation
> (heuristic/workspace/dispatch) rather than kernel codegen.

### PR comment draft (paste to ml-explore/mlx#3883)

> Ran the CUDA half on real sm_120 hardware (RTX 5070 Laptop, WSL2
> Ubuntu 24.04, CUDA 13.3, branch @ d3d6c38): warning behaves as
> designed for GEMM (fires once per process, silent for bf16 and for
> either explicit setting), the cuDNN conv site fires with family
> "convolution", and numerics match the measurements in #3860
> (2.9e-04 default / 2.1e-07 with `MLX_ENABLE_TF32=0`). Full 774-test
> parity suite: exactly 1 warning, failure set identical to the
> 0.32.0 wheel — no behavioral change.
>
> One asymmetry vs the M5 measurement: **fp32 matvec fires the warning
> on CUDA** (your Metal run had 0), while the matvec result itself
> stays fp32-exact (rel err 9.7e-08 vs fp64, same process). The CUDA
> gate in `dtype_to_compute_type()` is shape-blind, so matvec-only
> workloads get warned about a reduction that doesn't happen for that
> shape. Suggest either shape-gating that call site or documenting the
> requested-mode semantics.
>
> Minor observation for the discussion: under pytest's default capture
> the once-per-process warning is invisible (swallowed with whichever
> test ran the first fp32 GEMM); `--capture=no` shows it. May be worth
> a line in the docs.

## Round 2 — shape-gate re-verification (requested by author, commit 57466a7)

The warning moved from `dtype_to_compute_type()` into the `CublasGemm`
constructor, gated on `M_ > 1 && N_ > 1` (per our scenario-2 finding).
Author remains compile-only on CUDA; four checks requested.

Setup (incremental — reuse the existing clone and venv):

```bash
cd mlx
git branch -D pr3883 2>/dev/null; git fetch origin pull/3883/head:pr3883
git checkout pr3883 && git log --oneline -3   # expect 57466a7 on top
CMAKE_ARGS="-DMLX_BUILD_CUDA=ON" pip install . --no-build-isolation -v
# CMake cache from round 1 makes this a short rebuild; same
# LD_LIBRARY_PATH / cudnn-shim / CMAKE_BUILD_PARALLEL_LEVEL=6 notes apply.
```

| # | scenario (separate processes, stderr captured whole) | expected |
|---|---|---|
| R1 | env unset / fp32 matvec, then control GEMM in-process | **pair (0, 1)** — was (1, 0) in round 1; the control now proves the gate, not consumption |
| R2 | env unset / fp32 GEMM x2 | exactly 1, family "matmul" (unchanged) |
| R3 | env unset / fp32 `mx.addmm(c, a, b, beta=1.0)` (512x512) | exactly 1, family "matmul" — exercises the second, direct `CublasGemm` construction the author gated at the constructor for |
| R4a/b | 774-test parity suite, env unset / `MLX_ENABLE_TF32=0` | identical to round 1: (1 warning; 8f/765p) / (0 warnings; 4f/769p) |
| R5 | optional: fp32 conv2d | still 1, family "convolution" (site untouched by 57466a7) |

R3 snippet:

```python
import mlx.core as mx
a = mx.ones((512, 512)); b = mx.ones((512, 512)); c = mx.ones((512, 512))
mx.eval(mx.addmm(c, a, b, beta=1.0))
```

Report the five rows on ml-explore/mlx#3883 in the same table format;
that should be the last hardware gate before merge.

## Round 2 results (2026-07-23, RTX 5070 Laptop / sm_120)

Branch build `0.32.0.dev20260723+57466a72` (incremental rebuild, same
toolchain/notes as round 1). **All five rows pass — no deviations.**

| # | scenario | count | family | verdict |
|---|---|---|---|---|
| R1 | fp32 matvec → control GEMM, env unset | **pair (0, 1)** | – / matmul | ✅ **shape gate works** — matvec now silent (was (1, 0)), control GEMM still warns, so the 0 is a genuine gate, not consumption |
| R2 | fp32 GEMM ×2, env unset | 1 | matmul | ✅ unchanged |
| R3 | fp32 `addmm`, env unset | 1 | matmul | ✅ direct `CublasGemm` constructor path covered |
| R4a | 774-test parity suite, env unset | 1 warning; 8 failed / 765 passed | matmul | ✅ identical to round 1 |
| R4b | suite, `MLX_ENABLE_TF32=0` | 0 warnings; 4 failed / 769 passed (all speed) | – | ✅ identical to round 1 |
| R5 | fp32 conv2d, env unset | 1 | convolution | ✅ unchanged (site untouched by 57466a7) |

### PR comment draft (round 2, paste to ml-explore/mlx#3883)

> Re-ran the matrix on real sm_120 hardware against 57466a7
> (RTX 5070 Laptop, WSL2, CUDA 13.3, incremental rebuild):
>
> | scenario | result |
> |---|---|
> | fp32 matvec → in-process control GEMM | **(0, 1)** — matvec silent, control still warns |
> | fp32 GEMM ×2 | 1, family "matmul" |
> | fp32 `addmm` | 1, family "matmul" |
> | 774-test parity suite (env unset / `MLX_ENABLE_TF32=0`) | 1 warning, failures identical to wheel / 0 warnings, failures identical to wheel |
> | fp32 conv2d | 1, family "convolution" |
>
> The shape gate resolves the matvec false positive from my earlier
> comment — matvec-only workloads are now silent while the in-process
> control GEMM proves the machinery still engages, and the addmm
> constructor path warns as intended. No behavioral change in either
> suite configuration. LGTM from the hardware side.
