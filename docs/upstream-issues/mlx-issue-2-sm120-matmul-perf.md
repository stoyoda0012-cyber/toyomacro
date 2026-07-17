# [PERF] [CUDA] fp32 matmul ~36× slower than cuBLAS on sm_120 (RTX 50 / consumer Blackwell)

> **Filed 2026-07-17 as [ml-explore/mlx#3861](https://github.com/ml-explore/mlx/issues/3861).** This file is the archived draft; the live discussion is upstream.

## Describe the issue

On an RTX 5070 Laptop (sm_120), `mx.matmul` sustains **0.32 TFLOP/s**
on a 4096³ fp32 GEMM. CuPy calling cuBLAS **through the exact same
`nvidia-cublas` wheel, on the same GPU, in the same WSL2 session**
sustains **11.96 TFLOP/s** — a ~36× gap. The host CPU (Ryzen 9 8940HX,
NumPy/OpenBLAS) beats the GPU at 0.66 TFLOP/s.

| path | 4096³ fp32 GEMM (median) | effective |
|---|---|---|
| CuPy → cuBLAS | 11.5 ms | 11.96 TFLOP/s |
| NumPy on host CPU | 208.5 ms | 0.66 TFLOP/s |
| `mx.matmul` (`NVIDIA_TF32_OVERRIDE=0`) | 435.2 ms | 0.32 TFLOP/s |
| `mx.matmul` (default / TF32) | 729.7 ms | 0.19 TFLOP/s |

Two oddities that may help localize it:

1. **TF32-on is *slower* than TF32-off** in this microbench, consistent
   with a mis-selected tensor-core code path on this arch. Measured
   both off-paths: MLX-native `MLX_ENABLE_TF32=0` (requests
   `CUBLAS_COMPUTE_32F` outright) gives 421 ms and driver-level
   `NVIDIA_TF32_OVERRIDE=0` (driver forcing fp32 under the default
   `CUBLAS_COMPUTE_32F_FAST_TF32` request) gives 420 ms — identical, so
   the 36× gap is not an artifact of how TF32 is disabled. The default
   TF32 path is 734 ms (0.19 TFLOP/s) either way.
2. During the slow runs `nvidia-smi` reports 2805 MHz (near max), P0,
   100 % GPU utilization — at only **40 W**. The SMs are busy executing
   very low-efficiency code, not idle and not throttled.

Decomposition: chaining 10 GEMMs into a single `mx.eval` still costs
~196 ms per GEMM, i.e. the kernel itself is ~17× slow, plus ~220 ms of
per-`mx.eval` dispatch overhead on top for this graph.

`libmlx.so` in the `mlx-cuda-13` wheel contains CUTLASS kernel
instantiations for **SM90 only** (e.g. `qmm_sm90`,
`MMA_64x16x8_F32TF32TF32`); nothing for sm_120. If matmul on
unrecognized archs falls back to a generic JIT path, that would explain
both the throughput and the wattage signature.

## To Reproduce

```python
import time
import numpy as np
import mlx.core as mx

n = 4096
A = mx.array(np.random.default_rng(0).standard_normal((n, n)).astype(np.float32))
B = mx.array(np.random.default_rng(1).standard_normal((n, n)).astype(np.float32))
mx.eval(mx.matmul(A, B))  # warmup / JIT

ts = []
for _ in range(5):
    t0 = time.perf_counter()
    mx.eval(mx.matmul(A, B))
    ts.append(time.perf_counter() - t0)
med = sorted(ts)[2]
print(f"{med*1e3:.1f} ms  {2*n**3/med/1e12:.2f} TFLOP/s")
```

Control (same wheels, same session): `pip install cupy-cuda13x`, same
shapes through `cupy` → 11.5 ms.

## Expected behavior

fp32 GEMM within, say, 2× of cuBLAS on supported hardware — or routing
matmul to cuBLASLt on archs without tuned CUTLASS kernels.

## Desktop

- OS: Windows 11 + WSL2 (Ubuntu 24.04), kernel 6.18.33.2-microsoft-standard-WSL2
- GPU: NVIDIA GeForce RTX 5070 Laptop (Blackwell, sm_120), 8 GB, driver 596.13
- MLX: 0.32.0 (`mlx-cuda-13` 0.32.0; `nvidia-cublas` 13.6.0.2)
- Python 3.12.3, NumPy 2.3.5, CuPy 14.1.1 (control only)

## Additional context

Possibly related: #3056 ("Better support consumer CUDA GPUs") extended
CUDA-graph limits that were tuned for data-center GPUs and could not
saturate consumer Blackwell — same theme of consumer-arch parameters,
though that PR addressed graph limits, not the GEMM kernel selection
itself.

Found while porting an MLX-based (Metal-first) XPS spectral-fitting
engine to CUDA. Correctness parity with the NumPy reference passes
fully (with TF32 disabled); throughput is the remaining blocker — the
GPU currently loses to the host CPU on GEMM-heavy paths despite the
hardware being demonstrably capable of ~12 TFLOP/s through cuBLAS.
