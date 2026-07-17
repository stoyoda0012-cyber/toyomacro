# [BUG] [CUDA] fp32 matmul silently defaults to TF32 (`MLX_ENABLE_TF32=1`); undocumented, diverges from Metal

> **Filed 2026-07-17 as [ml-explore/mlx#3860](https://github.com/ml-explore/mlx/issues/3860).** This file is the archived draft; the live discussion is upstream.

## Describe the bug

On the CUDA backend, fp32 `mx.matmul` runs in TF32 (10-bit mantissa)
by default: `enable_tf32()` in `mlx/utils.h` defaults `MLX_ENABLE_TF32`
to `1`, and `dtype_to_compute_type()` in
`mlx/backend/cuda/gemms/cublas_gemm.cpp` then selects
`CUBLAS_COMPUTE_32F_FAST_TF32` for `float32` (and `complex64`) inputs.

The switch exists, but nothing tells the user about it: there is no
warning, and `MLX_ENABLE_TF32` appears nowhere in the documentation —
the only occurrences in the repo are `mlx/utils.h` and
`python/tests/mlx_tests.py`. We spent a full-pipeline bisection tracing
a ~9 dB accuracy regression before finding it, and initially worked
around it with NVIDIA's driver-level `NVIDIA_TF32_OVERRIDE=0` because
we could not find any MLX-side control.

Relative Frobenius error of a 512×512 fp32 GEMM against a float64
reference:

| backend | rel. error |
|---|---|
| NumPy fp32 | 2.9e-07 |
| MLX CPU stream | 4.1e-07 |
| MLX Metal (M-series) | fp32-class |
| **MLX CUDA (default, `MLX_ENABLE_TF32=1`)** | **2.9e-04** |
| MLX CUDA + `MLX_ENABLE_TF32=0` | 2.1e-07 |
| MLX CUDA + `NVIDIA_TF32_OVERRIDE=0` (driver-level) | 2.1e-07 |

~1000× the error of fp32 — the expected TF32 mantissa signature.
`MLX_ENABLE_TF32=0` is measured equivalent to the driver-level
override (2.08e-07 vs 2.077e-07 on the same seed). Because the flag is
read lazily on first use, `os.environ["MLX_ENABLE_TF32"] = "0"` before
the first kernel call also works in-process (measured: 2.08e-07) —
useful, but only if the user knows the flag exists.

The practical problem is the **silent divergence from Metal**: the same
MLX program produces fp32-accurate results on Apple silicon and
TF32-accurate results on CUDA. In our fitting engine this flipped
near-tie `argmax` decisions over correlation scores (1.4–2.5 % of
spectra picked a neighbouring dictionary entry) and cost ~9 dB of
end-to-end PSNR — while every op-level parity test still passed, since
the corruption enters through the accumulated matmul, not any single
op. It took a full-pipeline bisection to trace it back to TF32.

## To Reproduce

```python
import numpy as np
import mlx.core as mx

rng = np.random.default_rng(0)
A = rng.standard_normal((512, 512)).astype(np.float32)
B = rng.standard_normal((512, 512)).astype(np.float32)
ref = A.astype(np.float64) @ B.astype(np.float64)

g = np.array(mx.matmul(mx.array(A), mx.array(B)))
err = np.linalg.norm(g.astype(np.float64) - ref) / np.linalg.norm(ref)
print(f"rel Frobenius err: {err:.1e}")   # ~3e-04 default; ~2e-07 with MLX_ENABLE_TF32=0
```

## Expected behavior

Any of these would resolve it (in order of preference):

1. Default `MLX_ENABLE_TF32` to `0` — true fp32 for fp32 inputs,
   matching Metal and NumPy semantics — and make TF32 opt-in.
2. Expose a runtime switch (analogous to
   `torch.backends.cuda.matmul.allow_tf32`) instead of a
   read-once-at-first-use environment variable.
3. At minimum, document `MLX_ENABLE_TF32` and the TF32-by-default
   behavior prominently for the CUDA backend.

(Related: #3235 asked what `MLX_ENABLE_TF32` actually does on the
Metal/NAX path — the flag is doing double duty across backends, which
is another reason to document it.)

Notably, TF32 wasn't even buying speed in our measurements on sm_120
(TF32-on was *slower* than TF32-off in a 4096³ GEMM microbench —
reported separately), so on this arch the default trades ~3 decimal
digits of precision for nothing.

## Desktop

- OS: Windows 11 + WSL2 (Ubuntu 24.04)
- GPU: NVIDIA GeForce RTX 5070 Laptop (Blackwell, sm_120), driver 596.13
- MLX: 0.32.0 (`mlx-cuda-13` 0.32.0)
- Python 3.12.3, NumPy 2.3.5
