# mlx#3883 sm_120 verification plan (WSL2 / RTX 5070)

Verifies the CUDA half of ml-explore/mlx#3883 (one-time TF32 warning),
which the author could only compile-check (no NVIDIA hardware). We
promised this run in ml-explore/mlx#3860.

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
