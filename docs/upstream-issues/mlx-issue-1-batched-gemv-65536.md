# [BUG] [CUDA] Batched matrix-vector product crashes when batch > 65,535 (gridDim limit)

> **Filed 2026-07-17 as [ml-explore/mlx#3858](https://github.com/ml-explore/mlx/issues/3858).** This file is the archived draft; the live discussion is upstream.

## Describe the bug

On the CUDA backend, a batched matrix-vector `matmul` crashes with
`cudaGraphAddKernelNode(...) failed: invalid argument` when the batch
dimension exceeds 65,535. Bisected exactly: **B = 65,535 works,
B = 65,536 crashes** — the CUDA `gridDim.y/z` limit, suggesting the
batched-GEMV kernel maps the batch axis to a grid dimension without
tiling.

Square batched matmul at the same batch size is fine
(`(65536, 8, 8) @ (65536, 8, 8)` works), so this is specific to the
matvec (n=1 output column) path.

With `MLX_USE_CUDA_GRAPHS=0` the same shape still fails, just later:
`cudaLaunchKernelExC(&config, func, params) failed: invalid argument` —
so it is the kernel launch configuration itself, not the graph capture.

After the failure the CUDA context appears unusable: subsequent
unrelated `mx.eval` calls abort the process.

Metal has no equivalent grid limit, so code ported from Apple silicon
hits this only on CUDA, and only at production batch sizes (our parity
test suite passed everywhere because test batches are small).

## To Reproduce

```python
import mlx.core as mx

for B in (65535, 65536):
    A = mx.zeros((B, 225, 4))
    v = mx.zeros((B, 4, 1))
    try:
        mx.eval(A @ v)
        print(f"B={B}: OK")
    except RuntimeError as e:
        print(f"B={B}: {e}")
```

Output:

```
B=65535: OK
B=65536: cudaGraphAddKernelNode(&node, graph_, NULL, 0, &params) failed: invalid argument
```

## Expected behavior

Batched matmul should either tile the batch axis across grid launches
(as large-size elementwise ops already do) or raise a clear
shape-limit error at trace time.

## Desktop

- OS: Windows 11 + WSL2 (Ubuntu 24.04), kernel 6.18.33.2-microsoft-standard-WSL2
- GPU: NVIDIA GeForce RTX 5070 Laptop (Blackwell, sm_120), 8 GB, driver 596.13
- MLX: 0.32.0 (`pip install "mlx[cuda13]"`, `mlx-cuda-13` 0.32.0)
- Python 3.12.3, NumPy 2.3.5

## Additional context

Looks like the same class as the (fixed) #3666 ("fix grid for large
uncontiguous input" — the identical `gridDim.y <= 65,535` limit, fixed
for binary/copy/unary ops in June 2026 but evidently not for batched
GEMV), #2267 (`mx.random.uniform` large-size launch failure), #3659
(rope `cudaGraphAddKernelNode`), and #2724 (CUDA random large sizes) —
one more kernel with an untiled grid-dimension mapping.

Found while validating an XPS spectral-fitting engine (per-spectrum
4-component Gram solve over 10⁵–10⁶ spectra); the workaround is
chunking batches to ≤ 65,535.

---

## Fix landed and verified (2026-08-06)

Merged as [ml-explore/mlx#3929](https://github.com/ml-explore/mlx/pull/3929)
("Fix CUDA batched GEMV grid overflow", commit `4652b00`). The fix
splits the batch dimension across `grid.y` **and** `grid.z` (both
capped at 65,535) in `gemv.cu`, reconstructs the logical batch index
in both `gemv_batched` and `gemv_gather`, and bounds-checks the padded
tail slot. A related edge case (`GatherMM` with a zero-sized output)
was bundled into the same PR.

The author had no CUDA hardware to test on and left verification to
CI (`pull/3929`: *"This machine does not have a CUDA toolkit or
NVIDIA GPU"*) — same situation as the original report. Verified on
this box (RTX 5070 Laptop, sm_120), build `0.32.1.dev20260806+4652b008`
(main, includes #3929):

| check | result |
|---|---|
| Original one-line repro, B ∈ {65535, 65536, 65537, 131072} | ✅ all pass (was: crash at 65536) |
| Numerical correctness at/around the boundary (65534–200003) vs fp64 reference | ✅ max abs err ~1e-6 (fp32-normal), no seam artifacts |
| `gather_mv` path (also touched by the fix) | ✅ B=65536 OK |
| Empty-output `GatherMM` edge case | ✅ OK |
| toyomacro AP solver (`bench_ncomp_scaling`), 200k spectra, n_comp=4 — the exact workload that motivated this report | ✅ completes without chunking (was: crash) |
| Full 774-test voigtfit parity suite, `MLX_ENABLE_TF32=0` | ✅ 769 passed / 4 failed — failure set byte-identical to pre-fix (all speed assertions, unrelated) |

**Verdict: confirmed fixed, no regressions.** The chunk-to-≤65,535
workaround in `multipeak_solver.py` is no longer necessary once the
project's MLX dependency is bumped past this commit (follow-up 7 in
`docs/CUDA_BACKEND_POC.md` can be closed/reverted at that point).
