# [PACKAGING] `mlx[cuda13]` missing CUDA runtime/CCCL headers — first GPU op fails; silent abort under pytest

## Describe the bug

A fresh `pip install "mlx[cuda13]"` cannot execute any GPU op. The
extra pulls `nvidia-cublas`, `nvidia-cuda-nvrtc`, `nvidia-cudnn-cu13`,
`nvidia-cufft`, `nvidia-nccl-cu13`, `nvidia-nvjitlink` — whose combined
`include/` holds only 13 headers (cuBLAS/cuFFT/NVRTC API headers). The
CUDA runtime headers (`cuda_runtime.h`, ...) and CCCL (`cub`/`thrust`)
that NVRTC needs to JIT MLX's kernels are absent, so the first kernel
compile fails:

```
RuntimeError: Can not find locations of CUDA headers, please set
environment variable CUDA_HOME or CUDA_PATH.
```

Two aggravations:

1. On a machine with no system CUDA toolkit (the presumable target
   audience of a pip-wheel backend), there is no valid `CUDA_HOME` to
   point at — the fix is discovering that two more pip packages exist.
2. **Under pytest the error is invisible**: the suite dies with a bare
   `Fatal Python error: Aborted` and a C-level stack dump, no mention
   of headers. We only found the real error by running a single op
   outside the test harness.

## To Reproduce

On a machine with no system CUDA toolkit:

```bash
python -m venv v && . v/bin/activate
pip install "mlx[cuda13]" numpy
python -c "import mlx.core as mx; mx.eval(mx.zeros((4,)) + 1)"
# RuntimeError: Can not find locations of CUDA headers ...
```

## Workaround / suggested fix

```bash
pip install nvidia-cuda-runtime nvidia-cuda-cccl
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
```

(Note the unsuffixed package names; `nvidia-cuda-runtime-cu13` /
`nvidia-cuda-cccl-cu13` are deprecated stubs that fail to build.)

Suggested fixes, any of:

1. Add `nvidia-cuda-runtime` and `nvidia-cuda-cccl` to the `[cuda13]`
   extra, and default the header search to the venv's
   `site-packages/nvidia/cu13` before requiring `CUDA_HOME`.
2. Vendor the needed headers in the `mlx-cuda-13` wheel.
3. At minimum, extend the error message to name the pip packages.

Note there is prior art for exactly this: #2906 (closing #2842) added a
search of `../../nvidia/cuda_runtime/include` for the `cuda-toolkit`
pip package's layout, and #2357/#2382 addressed CCCL discovery. The
cu13-generation wheels install to a different layout
(`site-packages/nvidia/cu13/include`), which that search does not
cover — so the existing mechanism just needs the new path (and the
`[cuda13]` extra needs to pull the two packages).

## Desktop

- OS: Windows 11 + WSL2 (Ubuntu 24.04); no system CUDA toolkit installed
- GPU: NVIDIA GeForce RTX 5070 Laptop (sm_120), driver 596.13
- MLX: 0.32.0 (`mlx-cuda-13` 0.32.0)
- Python 3.12.3
