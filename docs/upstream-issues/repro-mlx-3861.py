"""Minimal reproduction for ml-explore/mlx#3861.

On a device with concurrentManagedAccess == 0 (Windows, WSL2, Tegra),
MLX arrays built from host data are allocated with cudaMallocHost and
stay in pinned host memory for life, so every kernel reading them
crosses PCIe. The buffer cache is keyed on size alone, so once such a
buffer is freed it is also handed out to later device-side allocations.

Expected on an affected machine: three timings that should be identical
differ by ~40x. Expected on Linux with concurrentManagedAccess == 1:
all three the same.

    python repro_3861.py            # runs all three cases as subprocesses
    python repro_3861.py device     # one case only
"""
import subprocess
import sys
import time

N = 4096
FLOPS = 2.0 * N**3


def run_case(case):
    import numpy as np
    import mlx.core as mx

    if case == "device":
        # operands generated on the device
        a = mx.random.normal((N, N))
        b = mx.random.normal((N, N))
    elif case == "host":
        # operands built from host data -- what any data-loading code does
        a = mx.array(np.zeros((N, N), dtype=np.float32))
        b = mx.array(np.zeros((N, N), dtype=np.float32))
    elif case == "device_after_freed_host":
        # device-generated operands, but the cache holds freed host buffers
        h1 = mx.array(np.zeros((N, N), dtype=np.float32))
        h2 = mx.array(np.zeros((N, N), dtype=np.float32))
        mx.eval(h1, h2)
        del h1, h2                     # returned to MLX's buffer cache
        a = mx.random.normal((N, N))
        b = mx.random.normal((N, N))
    else:
        raise SystemExit(f"unknown case {case}")

    mx.eval(a, b)
    mx.eval(mx.matmul(a, b))           # warm up
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        mx.eval(mx.matmul(a, b))
        ts.append(time.perf_counter() - t0)
    t = sorted(ts)[2]
    print(f"{case:>26} : {t*1e3:8.2f} ms   {FLOPS/t/1e12:6.2f} TFLOP/s")


if len(sys.argv) > 1:
    run_case(sys.argv[1])
else:
    # each case needs a fresh process: the cache is per-process state
    import mlx.core as mx
    print(f"mlx {mx.__version__}, {N}x{N} float32 matmul\n", flush=True)
    for case in ("device", "host", "device_after_freed_host"):
        subprocess.run([sys.executable, __file__, case])
    print("\nIf these differ, arrays built from host data are not on the "
          "device.\nCheck with: cudaDeviceGetAttribute("
          "cudaDevAttrConcurrentManagedAccess)", flush=True)
