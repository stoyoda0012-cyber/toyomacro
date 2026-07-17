# [BUG] [CUDA] float32 matmul silently uses TF32; no API to opt out, diverges from Metal

## Describe the bug

On the CUDA backend, fp32 `mx.matmul` runs in TF32 (10-bit mantissa)
by default. There is no warning, no documentation of the behavior, and
no MLX API to disable it — the only opt-out we found is NVIDIA's
driver-level `NVIDIA_TF32_OVERRIDE=0` environment variable.

Relative Frobenius error of a 512×512 fp32 GEMM against a float64
reference:

| backend | rel. error |
|---|---|
| NumPy fp32 | 2.9e-07 |
| MLX CPU stream | 4.1e-07 |
| MLX Metal (M-series) | fp32-class |
| **MLX CUDA (default)** | **2.9e-04** |
| MLX CUDA + `NVIDIA_TF32_OVERRIDE=0` | 2.1e-07 |

~1000× the error of fp32 — the expected TF32 mantissa signature.

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
print(f"rel Frobenius err: {err:.1e}")   # ~3e-04 default; ~2e-07 with NVIDIA_TF32_OVERRIDE=0
```

## Expected behavior

Any of these would resolve it (in order of preference):

1. Default to true fp32 for fp32 inputs, matching Metal and NumPy
   semantics; make TF32 opt-in.
2. Expose a runtime switch (analogous to
   `torch.backends.cuda.matmul.allow_tf32`).
3. At minimum, document the behavior prominently for the CUDA backend.

Notably, TF32 wasn't even buying speed in our measurements on sm_120
(TF32-on was *slower* than TF32-off in a 4096³ GEMM microbench —
reported separately), so on this arch the default trades ~3 decimal
digits of precision for nothing.

## Desktop

- OS: Windows 11 + WSL2 (Ubuntu 24.04)
- GPU: NVIDIA GeForce RTX 5070 Laptop (Blackwell, sm_120), driver 596.13
- MLX: 0.32.0 (`mlx-cuda-13` 0.32.0)
- Python 3.12.3, NumPy 2.3.5
