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
what the issue originally reported), `mlx_chained` (8 dependent matmuls
in one `mx.eval`, divided by 8, so per-eval graph cost is amortised and
what remains is per-kernel), and `cupy` (cuBLAS reference). Least-squares
fit of `t = fixed + per_flop · 2N³` on the chained arm.

Driver script: `mlx3861_gemm_sweep.py` (authored Mac-side; not tracked
here — it is a one-off measurement harness, not project code).

## Result (2026-08-08)

Official wheel `mlx 0.32.0` — the source build from the #3883/#3929 work
was reinstalled back to the wheel first, so nothing here is specific to a
locally compiled MLX.

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

     N  mlx_single  mlx_chained      cupy        gap   ratio  mlx TF/s  cupy TF/s
---------------------------------------------------------------------------------
   512       1.78m        1.35m     0.17m      1.18m    7.7x     0.20      1.54
  1024       9.85m        6.78m     0.49m      6.29m   14.0x     0.32      4.42
  1536      16.95m       12.33m     1.02m     11.30m   12.0x     0.59      7.08
  2048      53.44m       25.59m     2.11m     23.48m   12.1x     0.67      8.16
  3072     176.51m       83.76m     7.41m     76.36m   11.3x     0.69      7.83
  4096     432.37m      213.21m    15.56m    197.65m   13.7x     0.64      8.83
  6144    1452.67m      689.16m    59.58m    629.58m   11.6x     0.67      7.78
  8192    2415.60m     1573.95m   126.33m   1447.62m   12.5x     0.70      8.70

(times are per GEMM, in ms; gap and ratio use the chained arm,
 i.e. with per-mx.eval graph cost already amortised)

mlx (chained)  t =     5.64 ms +   0.70 TFLOP/s
               at N=8192, the size-independent part is 0.4% of the time
cupy           t =     0.67 ms +   8.61 TFLOP/s
               at N=8192, the size-independent part is 0.5% of the time
```

All eight sizes completed; N=8192 did not exhaust the 8 GB of VRAM.

## Reading

**The gap is kernel throughput, not a fixed per-launch cost.** Three
independent signs:

1. `gap` is not constant in absolute time — it grows from 1.18 ms at
   N=512 to 1448 ms at N=8192, i.e. with N³.
2. The fitted size-independent term is negligible for *both* backends
   (5.64 ms for MLX, 0.67 ms for CuPy — 0.4% and 0.5% of the N=8192
   time respectively). If MLX were paying a large fixed launch cost,
   this term is exactly where it would appear, and it does not.
3. The ratio is flat: 11–14× at every size from N=1024 up. MLX plateaus
   at ~0.70 TFLOP/s while CuPy reaches ~8.6 TFLOP/s on the same GPU,
   same session, same `nvidia-cublas` wheel.

This does not match the "super high overhead when launching the kernel"
hypothesis, at least not as the dominant term.

**A second, separate effect is visible in the single-vs-chained columns.**
`mlx_single` costs roughly 2× `mlx_chained` per GEMM (N=4096: 432 vs
213 ms; N=8192: 2416 vs 1574 ms). That difference also scales with N
rather than being a constant, so it too is proportional work rather than
a fixed cost — something per-`mx.eval` that is roughly as expensive as
the GEMM itself. Worth separating from the throughput question when
profiling.

## Datapoint on the "pure Windows issue" framing

This measurement is **WSL2**: Linux kernel, Linux userspace, Linux MLX
and CuPy wheels. No MSVC-built MLX is involved. It nonetheless
reproduces the ~12× gap.

That narrows the search: the cause is not the native-Windows *build* of
MLX (codegen, compiler, Windows-specific source paths), since none of
that is present here. What WSL2 *does* share with native Windows is the
GPU driver stack — the Windows WDDM driver, exposed into WSL through
`/dev/dxg` and the stub `libcuda.so`. CuPy reaches 8.6 TFLOP/s through
that same stack, so WDDM is not slow per se; the interaction is
MLX-specific.

Consistent with the DGX-Linux non-reproduction: DGX has neither WDDM nor
the WSL translation layer.
