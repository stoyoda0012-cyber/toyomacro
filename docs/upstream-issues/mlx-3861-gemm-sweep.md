# mlx#3861 — N sweep: fixed launch cost vs kernel throughput

Run on the WSL2 box (RTX 5070 Laptop) to answer the question raised by
zcbenz's 2026-08-07 comment on
[ml-explore/mlx#3861](https://github.com/ml-explore/mlx/issues/3861):

> I can reproduce with 5090 on native Windows build, but not with DGX on
> Linux. I think this is pure Windows issue that something has super high
> overhead when launching the kernel, we need some profiling to be sure
> though.

The sweep distinguishes the two candidate explanations without a
profiler: a **fixed per-launch cost** is constant in absolute ms (so it
dominates small N and washes out at large N), whereas a **slow kernel**
grows with N³ (so the ratio stays flat at every N).

Method: three timings per size — `mlx_single` (one matmul per `mx.eval`,
what the issue originally reported), `mlx_chained` (CHAIN dependent
matmuls in one `mx.eval`, divided by CHAIN — this cuts the per-eval cost
by 1/CHAIN without removing it, so the column is an *upper bound* on
per-kernel time), and `cupy` (cuBLAS reference). The summary splits
single vs chained under a `per-GEMM = G + E/k` model to estimate the two
parts separately.

Driver script: `mlx3861_gemm_sweep.py` (authored Mac-side; not tracked
here — it is a one-off measurement harness, not project code).

Official wheel `mlx 0.32.0` — the source build from the #3883/#3929 work
was reinstalled back to the wheel first, so nothing here is specific to a
locally compiled MLX.

## Headline finding: MLX and CuPy do not call the same library

The environment block prints every mapped `libcublas*`, and only
`libcublasLt.so.13` appeared. Checking each backend separately:

```
after MLX only:
    .../site-packages/nvidia/cu13/lib/libcublasLt.so.13
newly mapped by CuPy:
    .../site-packages/cupy_backends/cuda/libs/cublas.cpython-312-...so
    .../site-packages/nvidia/cu13/lib/libcublas.so.13
```

**MLX goes through cuBLASLt; CuPy goes through classic cuBLAS.** Same
wheel directory, different entry point. Earlier reports from this box
(including the #3861 comment of 2026-07-22) said "the same cublas wheel",
which is true of the *wheel* but overstates the sameness of the call
path — that wording should be corrected upstream.

This matters because it makes the rest coherent. cuBLASLt requires the
caller to pick the algorithm and workspace; classic cuBLAS dispatches on
NVIDIA's own heuristic. If MLX selects a poor algorithm on `sm_120`,
both the flat ~17× gap below and the TF32 result (slower with TF32 *on*)
follow from one cause.

## ① Main sweep, TF32 forced off (chain=8, repeats=5)

```
platform      : Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
kernel        : 6.18.33.2-microsoft-standard-WSL2   (WSL2)
MLX_ENABLE_TF32: 0
NVIDIA_TF32_OVERRIDE: 0
CUDA_HOME     : /home/ytoyo/toyomacro/.venv/lib/python3.12/site-packages/nvidia/cu13
tf32 arm      : forced off
mlx           : 0.32.0   default_device=Device(gpu, 0)
cupy          : 14.1.1
gpu           : NVIDIA GeForce RTX 5070 Laptop GPU   cc=12.0
cuda runtime  : 13020
libcublas     : /home/ytoyo/toyomacro/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcublasLt.so.13

     N  mlx_single  mlx_chained      cupy        gap   ratio  mlx TF/s  cupy TF/s
---------------------------------------------------------------------------------
   512       1.65m        1.20m     0.24m      0.96m    5.1x     0.22      1.14
  1024       9.32m        6.31m     0.31m      5.99m   20.1x     0.34      6.84
  1536      15.75m       11.76m     0.66m     11.10m   17.7x     0.62     10.93
  2048      51.96m       24.90m     1.47m     23.43m   16.9x     0.69     11.66
  3072     172.92m       82.46m     4.70m     77.76m   17.6x     0.70     12.35
  4096     408.90m      194.82m    30.32m    164.50m    6.4x     0.71      4.53
  6144    1377.99m      684.68m    41.36m    643.32m   16.6x     0.68     11.22
  8192    2367.26m     1469.41m    79.92m   1389.48m   18.4x     0.75     13.76

     N  mlx_chained  mlx_kernel  per-eval E  ratio(kernel)
   512        1.20m       1.13m       0.51m           4.8x
  1024        6.31m       5.88m       3.44m          18.7x
  1536       11.76m      11.20m       4.56m          16.9x
  2048       24.90m      21.03m      30.93m          14.3x
  3072       82.46m      69.53m     103.39m          14.8x
  4096      194.82m     164.24m     244.66m           5.4x
  6144      684.68m     585.63m     792.36m          14.2x
  8192     1469.41m    1341.14m    1026.12m          16.8x

Any fixed per-kernel cost is <= 1.20 ms (the whole per-GEMM time at N=512).
The mlx-cupy gap at N=8192 is 1389.48 ms, 1159x that bound.
From N=512 to N=8192 the gap grew 1442.8x, against 4096x for N^3 and 256x for N^2.
```

All eight sizes completed; N=8192 did not exhaust the 8 GB of VRAM.

**Caveat on one cell.** The `cupy` figure at N=4096 (30.32 ms) is an
outlier — runs ② and ③ measured 9.80 ms and 11.69 ms for the same shape.
The derived `ratio` (6.4×) and `cupy TF/s` (4.53) on that row should be
disregarded. Every other size puts CuPy at 11–14 TF/s.

## ② Convergence check (N=4096, chain=32, repeats=3)

```
     N  mlx_single  mlx_chained      cupy        gap   ratio  mlx TF/s  cupy TF/s
  4096     443.12m      182.75m     9.80m    172.95m   18.7x     0.75     14.03
```

`mlx_chained` falls monotonically as the chain lengthens — 194.82 ms at
chain=8, 182.75 ms at chain=32 — approaching the 164.24 ms that ①'s
two-parameter split predicted for the kernel alone. Re-solving `G + E/k`
from the chain=8 and chain=32 points gives **G ≈ 179 ms, E ≈ 129 ms**,
about 9% away from ①'s estimate (G=164, E=245).

The disagreement is itself informative: a single-`mx.eval` GEMM costs
more than a constant per-eval term would predict, so the single arm
carries something extra (a full synchronising round trip is the obvious
candidate). Either estimate puts per-kernel time at 165–180 ms against
CuPy's 10–15 ms, so the conclusion is unchanged whether the number is
read as an upper bound or as an estimate.

## ③ TF32 arm (MLX default, sizes 1024/2048/4096)

```
     N  mlx_single  mlx_chained      cupy        gap   ratio  mlx TF/s  cupy TF/s
  1024      16.99m        8.14m     0.35m      7.79m   23.2x     0.26      6.11
  2048      97.32m       51.68m     1.39m     50.28m   37.1x     0.33     12.34
  4096     736.51m      397.62m    11.69m    385.93m   34.0x     0.35     11.76
```

**Enabling TF32 makes MLX roughly 2× slower** (0.35 TF/s against 0.71
with it off), while CuPy is unaffected at ~12 TF/s. The gap widens from
~17× to ~34×. A feature that trades precision for speed is costing
double the time here — consistent with a heuristic picking an even worse
cuBLASLt algorithm once `CUBLAS_COMPUTE_32F_FAST_TF32` is requested.

## Reading

**The gap is kernel throughput, not a fixed per-launch cost.**

1. `gap` is not constant in absolute time — it grows from 0.96 ms at
   N=512 to 1389 ms at N=8192.
2. Any fixed per-kernel cost is bounded by 1.20 ms (the entire per-GEMM
   time at N=512). The N=8192 gap is 1159× that bound, so a per-launch
   toll cannot account for it.
3. A submission cost scaling with the working set would track N²; the
   gap grew 1443× where N² predicts 256× and N³ predicts 4096×. It sits
   near the N³ end, and the residual is explained by both backends being
   below their asymptotic throughput at N=512.
4. Once both reach plateau (N≥2048), MLX holds ~0.70 TF/s and CuPy
   ~12 TF/s — a flat ~17× ratio.

This does not match the "super high overhead when launching the kernel"
hypothesis as the dominant term.

**A second, separate effect** is visible in the single-vs-chained
columns: `mlx_single` costs ~2× `mlx_chained` per GEMM, and that
difference scales with N rather than being constant. Something
per-`mx.eval` is roughly as expensive as the GEMM itself. Worth
isolating separately when profiling.

## Datapoint on the "pure Windows issue" framing

This measurement is **WSL2**: Linux kernel, Linux userspace, Linux MLX
and CuPy wheels. No MSVC-built MLX is involved. It nonetheless
reproduces the gap.

That narrows the search: the cause is not the native-Windows *build* of
MLX (codegen, compiler, Windows-specific source paths), since none of
that is present here. What WSL2 *does* share with native Windows is the
GPU driver stack — the Windows WDDM driver, exposed into WSL through
`/dev/dxg` and the stub `libcuda.so`. CuPy reaches ~12 TF/s through that
same stack, so WDDM is not slow per se.

Consistent with the DGX-Linux non-reproduction: DGX has neither WDDM nor
the WSL translation layer. Combined with the cuBLASLt-vs-cuBLAS finding
above, the most economical hypothesis is that MLX's cuBLASLt algorithm
selection lands badly on this driver stack, in a way NVIDIA's own
in-cuBLAS heuristic does not.
