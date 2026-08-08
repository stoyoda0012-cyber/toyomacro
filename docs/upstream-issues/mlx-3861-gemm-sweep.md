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

The API split itself is real: cuBLASLt requires the caller to pick the
algorithm and workspace; classic cuBLAS dispatches on NVIDIA's own
heuristic.

> **Superseded — see Round 3 below.** This section originally continued:
> *"If MLX selects a poor algorithm on `sm_120`, both the flat ~17× gap
> below and the TF32 result follow from one cause."* That inference was
> tested in Round 3 by calling cuBLASLt directly with MLX's own
> parameters, and it is **wrong**: top-1 at MLX's workspace setting runs
> at 12.93 TF/s, and asking for eight candidates is no better. The
> library split is a true observation; the causal story built on it was
> not.

## ① Main sweep, TF32 forced off (chain=8, repeats=5)

```
platform      : Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
kernel        : 6.18.33.2-microsoft-standard-WSL2   (WSL2)
MLX_ENABLE_TF32: 0
NVIDIA_TF32_OVERRIDE: 0
CUDA_HOME     : <venv>/lib/python3.12/site-packages/nvidia/cu13
tf32 arm      : forced off
mlx           : 0.32.0   default_device=Device(gpu, 0)
cupy          : 14.1.1
gpu           : NVIDIA GeForce RTX 5070 Laptop GPU   cc=12.0
cuda runtime  : 13020
libcublas     : <venv>/lib/python3.12/site-packages/nvidia/cu13/lib/libcublasLt.so.13

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
outlier — other runs measured 9.80, 11.69 and 10.61 ms for the same
shape. The derived `ratio` (6.4×) and `cupy TF/s` (4.53) on that row
should be disregarded. Every other size puts CuPy at 11–14 TF/s.
Round 3 traces the outlier to co-residency with MLX in one process, not
to CuPy or to thermals; the clean reference at N=4096 is ~9–10 ms.

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
double the time here.

> **Superseded — see Round 3 below.** This section originally attributed
> that to *"a heuristic picking an even worse cuBLASLt algorithm once
> `CUBLAS_COMPUTE_32F_FAST_TF32` is requested."* Measured directly,
> cuBLASLt with that compute type is **faster**, not slower
> (17.7 vs 12.9 TF/s), so the slowdown cannot be a cuBLASLt selection
> effect. The measurement stands; the explanation does not.

A rerun at `--repeats 5` reproduced the effect: `mlx_chained`
8.47 / 51.68 / 407.72 ms at N=1024/2048/4096.

## Reading

**The gap is kernel throughput, not a fixed per-launch cost.**

1. `gap` is not constant in absolute time — it grows from 0.96 ms at
   N=512 to 1389 ms at N=8192.
2. Any fixed per-kernel cost is bounded by the *whole* per-GEMM time at
   the smallest size: 1.20 ms at N=512. The N=8192 gap is 1159× that
   bound, so a per-launch toll cannot account for it.
3. A submission cost scaling with the working set would track N²; the
   gap grew 1443× where N² predicts 256× and N³ predicts 4096×. It sits
   near the N³ end, and the residual is explained by both backends being
   below their asymptotic throughput at N=512.
4. Once both reach plateau (N≥2048), MLX holds ~0.70 TF/s and CuPy
   ~12 TF/s — a flat ~17× ratio.

(Evidence #2 is the direct bound, not a fitted intercept. The audit of
the first round established that the least-squares intercept was a fit
artifact — unweighted over sizes spanning 4000× in FLOPs, it absorbed
the small-N throughput ramp and exceeded the smallest measured total.
The harness used here prints the direct bound instead and no longer
reports a fit, so that objection does not apply to these numbers.)

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

---

# Round 3 (2026-08-09): the cuBLASLt causal claim is refuted

Audit round 8 blocked the draft on two readings of MLX's source. Both
were checked here by measurement rather than by re-reading. The audit
was right to block: **the "MLX picks a bad cuBLASLt algorithm" story
does not survive.**

## Blocker 2 — CUDA graph capture: eliminated

MLX wraps its `cublasLtMatmul` in a stream capture
(`cublas_utils.cpp`, `encoder.capture_context()`); CuPy issues directly.
`MLX_USE_CUDA_GRAPHS=0` turns the capture off without a rebuild.

| N | `mlx_chained`, graphs on | graphs off |
|---|---|---|
| 1024 | 6.31 ms | 6.36 ms |
| 2048 | 24.90 ms | 24.84 ms |
| 4096 | 194.82 ms | 196.12 ms |

Identical within ~1%. The flag is genuinely honoured — `device.cpp:18`
reads it via `env::get_var("MLX_USE_CUDA_GRAPHS", true)` and it gates
six or more sites in that file — so this is a real elimination, not a
no-op flag.

## The decisive test: call cuBLASLt directly, the way MLX calls it

Reading MLX v0.32.0 rather than guessing at it:

- workspace is `cc_major >= 9 ? 32 MiB : 4 MiB` (`cublas_utils.cpp:70`);
  `sm_120` is cc 12, so **32 MiB**
- `cublasLtMatmulAlgoGetHeuristic(..., 1, &heuristic_, &ret)` — literal
  `requestedAlgoCount = 1` (`cublas_utils.cpp:179`)
- operands are swapped rather than transposed ("a and b are swapped for
  row-major layout")

Driving `libcublasLt.so.13` from ctypes with the same compute type,
workspace and algo count, N=4096 (median of 5):

| workspace | top-1 (count=1) | best of 8 candidates |
|---|---|---|
| 0 | 18.49 ms — 7.43 TF/s | 10.48 ms — 13.11 TF/s |
| 1 MiB | 14.35 ms — 9.58 TF/s | 10.84 ms — 12.68 TF/s |
| **32 MiB** (MLX's setting) | **10.63 ms — 12.93 TF/s** | 11.71 ms — 11.74 TF/s |
| 256 MiB | 11.37 ms — 12.09 TF/s | 11.33 ms — 12.13 TF/s |

With `CUBLAS_COMPUTE_32F_FAST_TF32`, same N: 7.78–9.02 ms (15.2–17.7
TF/s). CuPy reference in the same process: 11.66 ms.

**MLX at N=4096 takes ~180–195 ms.** cuBLASLt asked the same way takes
10.63 ms. That is the whole gap, and it is not in cuBLASLt.

Three specific claims die here:

1. *"`requestedAlgoCount = 1` gets a bad algorithm."* Best-of-8 is not
   better than top-1 (10.5–11.7 ms either way). Asking for more
   candidates would not help.
2. *"No good `sm_120` kernel exists in the pool."* One does, and top-1
   finds it: 12.93 TF/s at MLX's own workspace setting.
3. *"TF32 is slower in MLX because a worse algorithm gets selected."*
   Through raw cuBLASLt, TF32 is **faster** (17.7 vs 12.9 TF/s). MLX's
   TF32 slowdown therefore cannot be a cuBLASLt selection effect. The
   audit's question 2 was exactly right: that observation was merely
   *not contradicting* the story, never supporting it.

## Blocker 1 — the per-call heuristic query: measured, negligible

`CublasGemm` is a stack local (`matmul.cpp:116`, `:338`), so the
heuristic is re-queried per GEMM rather than cached across calls. That
is a real structural difference from CuPy's `cublasGemmEx`, which has no
such query. It is also far too cheap to matter:

| N | heuristic query | matmul | query share |
|---|---|---|---|
| 1024 | 0.031 ms | 0.36 ms | 7.9% |
| 2048 | 0.032 ms | 1.91 ms | 1.7% |
| 4096 | 0.059 ms | 18.31 ms | 0.3% |
| 6144 | 0.055 ms | 38.37 ms | 0.1% |

Tens of microseconds against a ~170 ms gap.

## Which library each backend actually calls

An `LD_PRELOAD` interposer confirms the split at runtime:

- CuPy calls **`cublasGemmEx`** (4 calls for 4 GEMMs) — never
  `cublasLtMatmul`.
- MLX calls **`cublasLtMatmulAlgoGetHeuristic`**, i.e. it does take the
  cuBLASLt path. (The shim could not forward MLX's call — MLX resolves
  cuBLASLt in a local scope, so `dlsym(RTLD_NEXT)` finds no next
  definition — so only the entry point is established, not a count.)

## The CuPy N=4096 outlier: an artifact of co-residency, not of CuPy

Five readings existed: 30.32 / 9.80 / 11.69 / 10.61 / 30.99 ms —
bimodal, not noise. Resolved:

- **Not thermal.** 24 back-to-back iterations alone: 9.87–10.49 ms,
  13.1–13.9 TF/s, 55→59 °C, SM clock 1635–1732 MHz throughout.
- **Co-residency reproduces it.** In one process — CuPy first
  (median 9.42 ms, max 9.85), then the MLX arm, then CuPy again — the
  median stays 9.05 ms but a single iteration spikes to **27.91 ms**.

So the ~30 ms readings come from measuring CuPy in a process where MLX
has run, not from CuPy. The clean reference at N=4096 is **~9–10 ms
(13.5 TF/s)**. Note the direction: quoting 30 ms *understated* the gap.

## Where this leaves the explanation

Eliminated so far, each by measurement on this box:

| candidate | how ruled out |
|---|---|
| PTX JIT for a missing arch | native `sm_120` build unchanged (425 vs 435 ms) |
| CUDA graph capture | `MLX_USE_CUDA_GRAPHS=0` unchanged |
| cuBLASLt algorithm selection | direct call, same parameters, 10.63 ms |
| too-small workspace | MLX already requests 32 MiB; 0→256 MiB spans only 18.5→11.4 ms |
| per-call heuristic query | 0.06 ms at N=4096 |
| fixed per-launch cost | bounded by 1.20 ms at N=512; gap is 1159× that |
| thermal throttling | clocks and temperature logged, flat |

What remains is ~170 ms per 4096³ GEMM inside MLX, scaling with N³ —
i.e. compute-proportional, not a per-call toll and not data movement
(which would track N²).

**Caveat on the comparison.** The ctypes probe matched MLX's compute
type, workspace size and algo count, and used plain N×N column-major
layouts on the default stream. It did not replicate every descriptor
attribute MLX may set (transpose flags, epilogue) nor MLX's non-default
stream. So the honest claim is bounded: *a straightforward cuBLASLt call
with the same compute type, workspace and algo count runs ~17× faster
than MLX's GEMM on this machine* — which locates the cost inside MLX's
setup or execution rather than in cuBLASLt's choices, without yet naming
the line. Naming it needs a profiler, which is what the issue asked for
in the first place.

---

# Round 4 (2026-08-09): operand residency — hypothesis refuted

Proposed next: the harness builds operands with `mx.array(numpy_array)`,
so perhaps they sit somewhere the GEMM reaches over PCIe on every pass,
which would be N³-proportional and would *be* the answer rather than
another exclusion. Tested three ways.

## The construction path makes no difference

N=4096, same process, `mlx_chained` per GEMM:

| operands built by | single | chained | TF/s |
|---|---|---|---|
| `mx.array(numpy)` | 410.53 ms | 196.43 ms | 0.70 |
| `mx.random.normal` (device-generated) | 165.86 ms | 195.81 ms | 0.70 |

Ratio 0.997. Whatever the cost is, it does not depend on how the array
was produced.

## What MLX's allocator does, and what the pointer says

MLX v0.32.0 allocates with **`cudaMallocManaged`**
(`mlx/backend/cuda/allocator.cpp:58`) — managed/unified memory, not
`cudaMalloc`.

`cudaPointerGetAttributes` on an MLX buffer (address taken through
`np.from_dlpack`, which succeeds — MLX exports these buffers as
host-visible):

```
status=0 type=1 (host, pinned/mapped) device=0
devicePointer=0x205000000  hostPointer=0x205000000   SAME (mapped)
```

Control, separate process, same method: a CuPy buffer reports
`type=2 (device)`, `cudaMemGetInfo` free −0.25 GiB, `nvidia-smi` +256 MiB
for a 256 MiB array. So the method does report "device" when the memory
is on device.

A host view of an MLX array reads back `1.0` everywhere after a GPU-side
`+ 1.0`, with no explicit copy — one mapped allocation, as the identical
pointers imply.

**This looked like the answer and it is not.** Read on.

## The decisive measurement: achieved bandwidth

Pointer attributes can be argued about; a memory-bound kernel's achieved
rate cannot. If operands were really being fetched across PCIe, a
reduction would run at roughly 10–25 GB/s. If they are in VRAM, several
hundred.

| backend | 64 MiB | 256 MiB | 1024 MiB |
|---|---|---|---|
| `mx.sum` | 208.0 GB/s | 293.4 GB/s | **328.2 GB/s** |
| `cupy sum` | 216.1 GB/s | 302.9 GB/s | **321.6 GB/s** |

Identical, and both at VRAM speed. **MLX's managed allocations are
device-backed in practice on this box; the residency hypothesis is
refuted.** The `type=1` flag reflects how `cudaMallocManaged` memory is
described here, not where the kernel actually reads from.

Recording this explicitly because it is the same failure mode audit
round 8 caught: an attribute flag and a plausible mechanism made a
compelling story, and the story was wrong. The bandwidth number is what
settles it.

## Updated elimination list

| candidate | how ruled out |
|---|---|
| PTX JIT for a missing arch | native `sm_120` build unchanged |
| CUDA graph capture | `MLX_USE_CUDA_GRAPHS=0` unchanged |
| cuBLASLt algorithm selection | direct call, MLX's parameters: 10.63 ms |
| too-small workspace | MLX requests 32 MiB; 0→256 MiB spans 18.5→11.4 ms |
| per-call heuristic query | 0.06 ms at N=4096 |
| fixed per-launch cost | bounded by 1.20 ms at N=512 |
| thermal throttling | clocks and temperature logged flat |
| **operand residency / PCIe** | **328 GB/s on a streaming reduction** |
| operand construction path | `mx.random.normal` vs `mx.array`: ratio 0.997 |

Memory bandwidth is healthy, the library is fast when called the same
way, and the cost is compute-proportional. The remaining ~170 ms per
4096³ GEMM is specific to MLX's GEMM invocation and is not any of the
above. Naming it needs a profiler — which is what the issue asked for.

---

# Round 5 (2026-08-09): found it — buffers are host-resident, and the cache spreads it

Round 4 concluded "residency refuted". **That conclusion was wrong**, and
the error was mine: the bandwidth test that refuted it happened to use a
*device-resident* array, so it measured the fast case and generalised.
Rerunning the comparison the way it was originally proposed — one
construction per fresh process — gives the opposite answer.

## The switch is how the array was built

N=4096 fp32 GEMM, one variant per process, nothing else differing:

| operands built by | time | rate |
|---|---|---|
| `mx.random.normal` (device-side) | **10.58 ms** | 12.99 TF/s |
| `mx.array(numpy_array)` (host data) | **425.07 ms** | 0.32 TF/s |

40×. Reproduced across six clean processes for the fast case
(10.58 / 10.63 / 10.75 / 10.88 / 11.09 / 11.21 ms) and three for the slow
one (417.75 / 425.07 / 442.32 ms). Whether a chain of dependent matmuls
ran first makes no difference (1.0× before vs after).

## Where the memory actually is

1 GiB array, measured without `np.from_dlpack` (which itself migrates the
buffer and invalidated an earlier reading):

| built by | `cudaMemGetInfo` free | `nvidia-smi` used | streaming read | verdict |
|---|---|---|---|---|
| `mx.random.normal` | **−1.00 GiB** | **+1024 MiB** | 346.5 GB/s | device |
| `mx.array(numpy)` | −0.00 GiB | −1 MiB | **12.6 GB/s** | host, over PCIe |

Three independent signals agree in each row.

## The mechanism, in MLX's own source

`mlx/backend/cuda/allocator.cpp`:

- `supports_managed_memory()` returns **false** when
  `concurrent_managed_access()` is 0, with a comment citing NVIDIA's
  "Limited unified memory support" for Windows, WSL and Tegra. This box
  reports `cudaDevAttrConcurrentManagedAccess = 0`.
- `unified_malloc()` therefore calls **`cudaMallocHost`** — pinned host
  memory — rather than `cudaMallocManaged`.
- `malloc()` routes `device == -1` to `unified_malloc()`, and device-side
  allocations to `cudaMallocAsync` / `cudaMalloc`.

So anything ingested from host data — i.e. anything loaded from NumPy,
HDF5 or disk — lands in pinned host memory and stays there.

Confirmed independently by driving cuBLASLt directly with the same call,
same algorithm, same workspace, changing only the allocator:

| operand allocator | time | rate |
|---|---|---|
| `cudaMalloc` | 8.48 ms | 16.21 TF/s |
| `cudaMallocManaged` | 422.45 ms | 0.33 TF/s |
| **`cudaMallocHost`** (what MLX uses here) | **410.66 ms** | **0.33 TF/s** |

410.66 ms against MLX's measured ~410–425 ms. The workspace allocator is
nearly irrelevant by comparison (8.48 → 8.45 ms when it too is host).

## The part that makes it spread: the buffer cache ignores residency

`buffer_cache_.reuse_from_cache(size)` (allocator.cpp:185) is keyed on
**size alone**, and it runs *before* the `device == -1` branch that
chooses host versus device allocation. A freed pinned-host buffer is
therefore handed back to a later device-side request of the same size.

Same GEMM, device-generated operands throughout; only the fate of two
prior host-sourced buffers differs:

| what happened to the earlier host buffers | GEMM | rate |
|---|---|---|
| kept alive (never enter the cache) | 10.75 ms | 12.78 TF/s |
| dropped **and** `mx.clear_cache()` | 10.63 ms | 12.93 TF/s |
| dropped, cache left populated | **446.71 ms** | **0.31 TF/s** |

42× for a difference that is invisible in user code. Any program that
loads data from NumPy and then allocates working arrays will hit this: the
working arrays silently inherit host memory.

## Why this explains everything, including the platform split

- **Windows/WSL only** — `concurrentManagedAccess` is 0 exactly there.
  On DGX Linux it is 1, `cudaMallocManaged` is used, and pages migrate to
  the device on first touch, so nothing is slow and nothing reproduces.
- **N³ scaling** — a GEMM re-reads each operand O(N) times; every pass
  crosses PCIe.
- **Immune to graph capture, arch rebuild, algorithm choice, workspace
  size** — none of them changes where the operands live.
- **CuPy unaffected** — it allocates with `cudaMalloc`.
- **TF32 "slower"** — a TF32 kernel is more compute-efficient, so it is
  starved harder by the same PCIe bottleneck.

## Corrections to earlier rounds

- Round 4's "operand residency refuted" is **withdrawn**. Its 328 GB/s
  reading was taken on a `mx.random.normal` array, which is device-
  resident; it never tested the NumPy-sourced case it claimed to refute.
- Round 4's `cudaPointerGetAttributes` reading of `type=1 (host)` was
  **correct** for host-sourced arrays. It was dismissed on the strength of
  the flawed bandwidth test.
- Round 3's elimination of cuBLASLt algorithm selection stands, and is
  now explained: the library was never the problem.
- The original hypothesis in this investigation — that operand residency
  was the answer and `mx.random.normal` would drop 195 ms to ~10 ms — was
  right. The measurement that appeared to refute it (ratio 0.997) ran both
  arms in one process, where the cache had already been populated with
  host buffers, so both arms used host memory.

## What upstream should change

1. Key the buffer cache on residency as well as size, so a pinned-host
   buffer is never handed to a device-side allocation.
2. Promote host-sourced arrays to device memory when they are consumed by
   device computation, rather than leaving them pinned for their lifetime.

Workaround for users on affected platforms, though a fragile one:
`mx.clear_cache()` after dropping host-sourced arrays, and prefer
device-side construction for anything hot.
