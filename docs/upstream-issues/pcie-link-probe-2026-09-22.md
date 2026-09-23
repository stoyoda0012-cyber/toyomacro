# PCIe link generation and the CUDA streaming rate — probe evidence

Raw outputs from the RTX 5070 Laptop host behind the PCIe-link statements
in `docs/CUDA_BACKEND_POC.md` and `docs/BENCHMARKS.md` §5. This file
carries the observations; the interpretation lives in those documents.

The central observation is §2 with §3: the only run in which the
streaming rate and the link generation were recorded at the same time.

## Environment

| | 2026-09-22 (original) | 2026-09-23 (re-runs) |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU | same |
| CPU | AMD Ryzen 9 8940HX | same |
| Host | Windows 11 + WSL2 Ubuntu 24.04, kernel `6.18.33.2-microsoft-standard-WSL2` | same |
| Driver | 610.71 (read from the WSL2 side) | 610.71 from both sides (§4b) |
| MLX | 0.32.2 (printed by `repro-mlx-3861.py`, §6a) | 0.32.2 (§5b, §6b) |
| Repository tree | `aa2a2f7` | `c8b941a` |

The tree matters only for §5 and §6, which import this package or run a
tracked script. §1's probe imports nothing but `mlx` and `numpy`.

## Provenance of each block

Not every output here was written to a file by the run that produced
it. Where it was not, the block is the terminal capture of that run,
reproduced unedited, and says so.

| § | source | timestamps |
|---|---|---|
| 2 | terminal capture; the probe writes no log | in-band, UTC, WSL2 clock |
| 3 | log file written by the sampler during §2 | in-band, UTC, Windows clock |
| 4a, 5a, 6a | terminal capture, 2026-09-22 | none in the output |
| 4b, 5b, 6b, 6c | re-run 2026-09-23 with version and time printed in-band | in-band, UTC |

§4b–§6c exist because §4a–§6a carry no timestamp. They are separate
runs, not the same ones re-read.

## 1. The probe

`probe_projection_link.py`, run from the repository root. It mirrors
`record.measure_projection_kernel` — same shapes, seed, step, warmup and
repeat counts — but prints one line per repetition, with wall-clock
stamps, so the rate can be lined up against a link log.

```python
"""Per-repetition timings for the projection kernel, with wall-clock stamps.

Mirrors `record.measure_projection_kernel` exactly -- same shapes, same
seed, same step, same warmup and repeat counts -- but prints one line per
repetition so the timings can be lined up against a PCIe link sample log
taken from the Windows side.

The harness reports only min/median/max, which is how a single slow
repetition inside an otherwise fast run stayed invisible. This prints
every one.

    python probe_projection_link.py

Align with the sampler by the STEP, not the clock: WSL2's clock can drift
from Windows'. If the rate steps and the link generation steps, they are
the same event whatever the timestamps say.
"""
import time
from datetime import UTC, datetime

import mlx.core as mx
import numpy as np

N, E, C = 4_000_000, 151, 3
WARMUP, REPEATS = 2, 9
BYTES = N * E * 4


def now() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S.%f")[:-3]


rng = np.random.default_rng(0)
idx_np = rng.choice(N, max(1, int(N * 1e-4)), replace=False)

print(f"building operands ({BYTES / 1e9:.2f} GB) ...", flush=True)
Y = mx.array(rng.random((N, E), dtype=np.float32))
W = mx.array(rng.random((E, C), dtype=np.float32))
Phi = mx.array(rng.random((C, E), dtype=np.float32))
idx = mx.array(idx_np)
mx.eval(Y, W, Phi, idx)
print(f"built at {now()} UTC\n", flush=True)

print("start_utc     end_utc        iter   seconds    Mspec/s    GB/s", flush=True)
for i in range(WARMUP + REPEATS):
    start = now()
    t0 = time.perf_counter()
    A = Y @ W
    resid = Y[idx] - A[idx] @ Phi
    mx.eval(A, mx.sum(resid * resid, axis=1))
    dt = time.perf_counter() - t0
    tag = "warm" if i < WARMUP else str(i - WARMUP + 1)
    print(f"{start}  {now()}  {tag:>5}  {dt:8.4f}  {N / dt / 1e6:9.2f}  "
          f"{BYTES / dt / 1e9:6.2f}", flush=True)
```

## 2. Probe output — three runs, one invocation

Run three times back to back from a shell loop
(`for i in 1 2 3; do echo "########## PROBE RUN $i"; python probe_projection_link.py; done`),
with the sampler in §3 running throughout. Columns: repetition start
and end (UTC, WSL2 clock), repetition, seconds, million spectra per
second, GB/s over the 2.42 GB operand.

```
########## PROBE RUN 1
building operands (2.42 GB) ...
built at 10:39:07.177 UTC

start_utc     end_utc        iter   seconds    Mspec/s    GB/s
10:39:07.178  10:39:12.773   warm    5.5954       0.71    0.43
10:39:12.773  10:39:13.146   warm    0.3728      10.73    6.48
10:39:13.146  10:39:13.525      1    0.3785      10.57    6.38
10:39:13.525  10:39:13.899      2    0.3742      10.69    6.46
10:39:13.899  10:39:14.637      3    0.7373       5.43    3.28
10:39:14.637  10:39:15.372      4    0.7349       5.44    3.29
10:39:15.372  10:39:16.111      5    0.7388       5.41    3.27
10:39:16.111  10:39:16.849      6    0.7383       5.42    3.27
10:39:16.849  10:39:17.584      7    0.7347       5.44    3.29
10:39:17.584  10:39:18.323      8    0.7385       5.42    3.27
10:39:18.323  10:39:19.060      9    0.7373       5.43    3.28
########## PROBE RUN 2
building operands (2.42 GB) ...
built at 10:39:32.714 UTC

start_utc     end_utc        iter   seconds    Mspec/s    GB/s
10:39:32.714  10:39:33.632   warm    0.9174       4.36    2.63
10:39:33.632  10:39:34.025   warm    0.3934      10.17    6.14
10:39:34.025  10:39:34.423      1    0.3979      10.05    6.07
10:39:34.423  10:39:34.816      2    0.3929      10.18    6.15
10:39:34.816  10:39:33.442      3    0.3961      10.10    6.10
10:39:33.442  10:39:33.834      4    0.3924      10.19    6.16
10:39:33.834  10:39:34.227      5    0.3927      10.19    6.15
10:39:34.227  10:39:34.625      6    0.3974      10.06    6.08
10:39:34.625  10:39:35.017      7    0.3924      10.19    6.16
10:39:35.018  10:39:35.410      8    0.3927      10.19    6.15
10:39:35.410  10:39:35.806      9    0.3956      10.11    6.11
########## PROBE RUN 3
building operands (2.42 GB) ...
built at 10:39:51.772 UTC

start_utc     end_utc        iter   seconds    Mspec/s    GB/s
10:39:51.772  10:39:52.942   warm    1.1702       3.42    2.06
10:39:52.943  10:39:53.336   warm    0.3935      10.16    6.14
10:39:53.336  10:39:53.732      1    0.3954      10.12    6.11
10:39:53.732  10:39:54.130      2    0.3977      10.06    6.08
10:39:54.130  10:39:54.505      3    0.3753      10.66    6.44
10:39:54.505  10:39:54.872      4    0.3667      10.91    6.59
10:39:54.872  10:39:55.237      5    0.3645      10.97    6.63
10:39:55.237  10:39:55.601      6    0.3637      11.00    6.64
10:39:55.601  10:39:55.967      7    0.3664      10.92    6.59
10:39:55.968  10:39:56.332      8    0.3643      10.98    6.63
10:39:56.332  10:39:56.696      9    0.3641      10.99    6.63
```

Two facts about the timestamps, without interpretation:

- RUN 2, repetition 3 ends before it starts (`10:39:34.816` →
  `10:39:33.442`): the WSL2 clock stepped back 1.374 s during it.
- The WSL2 and Windows clocks are not synchronised with each other, so
  §2 and §3 line up by where each changes, not by equal timestamps.

## 3. Windows-side link samples during §2

Taken from PowerShell on the Windows side, because WSL2's `nvidia-smi`
does not report the current link generation (§4). Loop:

```powershell
while ($true) {
  $t = (Get-Date).ToUniversalTime().ToString("HH:mm:ss.fff")
  $v = nvidia-smi.exe --query-gpu=pcie.link.gen.current,pcie.link.width.current,power.draw,clocks.current.sm,utilization.gpu --format=csv,noheader
  "$t, $v" | Out-File -Append -Encoding utf8 <log>
  Start-Sleep -Milliseconds 500
}
```

The requested interval is 0.5 s; each `nvidia-smi.exe` call adds to it,
and the log's actual spacing is about 0.59 s. Columns: time (UTC,
Windows clock), link generation, link width, power draw, SM clock, GPU
utilisation. Complete log, 109 lines, covering all three runs of §2:

```
10:38:54.190, 1, 8, 8.61 W, 630 MHz, 2 %
10:38:54.800, 1, 8, 8.72 W, 630 MHz, 2 %
10:38:55.380, 1, 8, 9.06 W, 630 MHz, 2 %
10:38:55.971, 1, 8, 8.61 W, 645 MHz, 2 %
10:38:56.558, 1, 8, 8.84 W, 645 MHz, 2 %
10:38:57.157, 1, 8, 8.84 W, 645 MHz, 2 %
10:38:57.744, 1, 8, 8.66 W, 645 MHz, 2 %
10:38:58.330, 1, 8, 8.67 W, 645 MHz, 2 %
10:38:58.916, 4, 8, 11.75 W, 1837 MHz, 0 %
10:38:59.504, 4, 8, 13.56 W, 1837 MHz, 0 %
10:39:00.089, 4, 8, 13.71 W, 1837 MHz, 0 %
10:39:00.679, 4, 8, 13.45 W, 1837 MHz, 0 %
10:39:01.264, 4, 8, 13.38 W, 1837 MHz, 0 %
10:39:01.852, 4, 8, 13.53 W, 1837 MHz, 0 %
10:39:02.440, 4, 8, 13.56 W, 1755 MHz, 1 %
10:39:03.026, 4, 8, 13.59 W, 1755 MHz, 0 %
10:39:03.612, 4, 8, 13.48 W, 1755 MHz, 0 %
10:39:04.201, 4, 8, 13.64 W, 1755 MHz, 0 %
10:39:04.793, 4, 8, 13.60 W, 1755 MHz, 0 %
10:39:05.381, 4, 8, 13.49 W, 1755 MHz, 1 %
10:39:05.968, 2, 8, 12.14 W, 645 MHz, 1 %
10:39:06.554, 2, 8, 11.69 W, 645 MHz, 2 %
10:39:07.140, 2, 8, 11.13 W, 645 MHz, 2 %
10:39:07.729, 2, 8, 11.23 W, 645 MHz, 2 %
10:39:08.316, 1, 8, 11.18 W, 645 MHz, 2 %
10:39:08.902, 1, 8, 10.19 W, 645 MHz, 18 %
10:39:09.493, 1, 8, 6.47 W, 660 MHz, 2 %
10:39:10.081, 1, 8, 8.85 W, 660 MHz, 2 %
10:39:10.666, 1, 8, 8.79 W, 660 MHz, 2 %
10:39:11.256, 1, 8, 8.89 W, 660 MHz, 2 %
10:39:11.845, 1, 8, 8.79 W, 660 MHz, 2 %
10:39:12.429, 1, 8, 8.94 W, 637 MHz, 2 %
10:39:13.014, 1, 8, 8.99 W, 637 MHz, 2 %
10:39:13.601, 4, 8, 11.84 W, 637 MHz, 17 %
10:39:14.190, 4, 8, 26.86 W, 637 MHz, 71 %
10:39:14.778, 4, 8, 35.04 W, 637 MHz, 100 %
10:39:15.365, 3, 8, 35.06 W, 637 MHz, 100 %
10:39:15.956, 3, 8, 12.95 W, 1485 MHz, 100 %
10:39:16.547, 3, 8, 12.93 W, 1485 MHz, 100 %
10:39:17.135, 3, 8, 12.92 W, 1485 MHz, 100 %
10:39:17.713, 3, 8, 12.89 W, 1485 MHz, 100 %
10:39:18.303, 3, 8, 12.87 W, 1485 MHz, 100 %
10:39:18.891, 3, 8, 12.92 W, 1485 MHz, 99 %
10:39:19.482, 3, 8, 12.95 W, 1485 MHz, 100 %
10:39:20.072, 3, 8, 12.28 W, 1485 MHz, 1 %
10:39:20.661, 4, 8, 10.88 W, 1485 MHz, 2 %
10:39:21.254, 4, 8, 9.70 W, 1485 MHz, 2 %
10:39:21.841, 2, 8, 9.77 W, 1485 MHz, 2 %
10:39:22.423, 4, 8, 11.52 W, 1537 MHz, 0 %
10:39:23.011, 4, 8, 11.26 W, 1537 MHz, 0 %
10:39:23.600, 4, 8, 11.29 W, 1537 MHz, 0 %
10:39:24.185, 4, 8, 11.35 W, 1537 MHz, 0 %
10:39:24.772, 4, 8, 11.28 W, 1537 MHz, 0 %
10:39:25.378, 4, 8, 11.28 W, 1537 MHz, 0 %
10:39:25.966, 4, 8, 11.22 W, 1537 MHz, 0 %
10:39:26.551, 4, 8, 11.27 W, 1537 MHz, 0 %
10:39:27.138, 4, 8, 11.30 W, 1537 MHz, 0 %
10:39:27.725, 4, 8, 11.28 W, 1537 MHz, 0 %
10:39:28.313, 4, 8, 11.28 W, 1537 MHz, 0 %
10:39:28.898, 3, 8, 11.37 W, 1537 MHz, 0 %
10:39:29.483, 3, 8, 9.71 W, 412 MHz, 1 %
10:39:30.071, 3, 8, 9.70 W, 412 MHz, 1 %
10:39:30.660, 3, 8, 9.69 W, 412 MHz, 2 %
10:39:31.250, 2, 8, 9.60 W, 412 MHz, 2 %
10:39:31.835, 2, 8, 9.61 W, 412 MHz, 2 %
10:39:32.422, 2, 8, 9.65 W, 1425 MHz, 8 %
10:39:33.010, 4, 8, 13.26 W, 1425 MHz, 100 %
10:39:33.602, 4, 8, 13.36 W, 1425 MHz, 100 %
10:39:34.188, 4, 8, 13.28 W, 1425 MHz, 99 %
10:39:34.776, 4, 8, 13.31 W, 1425 MHz, 100 %
10:39:35.362, 4, 8, 13.27 W, 1425 MHz, 100 %
10:39:35.951, 4, 8, 13.37 W, 1485 MHz, 100 %
10:39:36.537, 4, 8, 13.31 W, 1485 MHz, 99 %
10:39:37.121, 4, 8, 12.63 W, 1485 MHz, 1 %
10:39:37.706, 4, 8, 11.21 W, 1485 MHz, 2 %
10:39:38.290, 4, 8, 9.83 W, 1485 MHz, 2 %
10:39:38.874, 4, 8, 9.77 W, 1485 MHz, 2 %
10:39:39.458, 4, 8, 11.39 W, 1537 MHz, 0 %
10:39:40.044, 4, 8, 11.47 W, 1537 MHz, 0 %
10:39:40.625, 4, 8, 11.49 W, 1537 MHz, 0 %
10:39:41.213, 4, 8, 11.35 W, 1537 MHz, 0 %
10:39:41.799, 4, 8, 11.35 W, 1537 MHz, 0 %
10:39:42.383, 4, 8, 11.41 W, 1537 MHz, 0 %
10:39:42.973, 4, 8, 11.36 W, 1537 MHz, 0 %
10:39:43.558, 4, 8, 11.38 W, 1537 MHz, 0 %
10:39:44.139, 4, 8, 11.34 W, 1537 MHz, 0 %
10:39:44.727, 4, 8, 11.40 W, 1537 MHz, 0 %
10:39:45.316, 4, 8, 11.35 W, 1537 MHz, 0 %
10:39:45.905, 2, 8, 11.38 W, 1537 MHz, 1 %
10:39:46.488, 2, 8, 10.13 W, 412 MHz, 1 %
10:39:47.075, 2, 8, 9.61 W, 412 MHz, 1 %
10:39:47.663, 2, 8, 9.70 W, 412 MHz, 2 %
10:39:48.252, 2, 8, 9.68 W, 412 MHz, 2 %
10:39:48.839, 1, 8, 9.78 W, 412 MHz, 2 %
10:39:49.425, 1, 8, 7.55 W, 1410 MHz, 2 %
10:39:50.013, 1, 8, 7.53 W, 1410 MHz, 2 %
10:39:50.600, 1, 8, 7.58 W, 1410 MHz, 2 %
10:39:51.189, 1, 8, 7.59 W, 1410 MHz, 2 %
10:39:51.776, 1, 8, 7.89 W, 1410 MHz, 7 %
10:39:52.378, 4, 8, 7.58 W, 1410 MHz, 2 %
10:39:52.963, 4, 8, 13.34 W, 1485 MHz, 100 %
10:39:53.550, 4, 8, 13.43 W, 1485 MHz, 100 %
10:39:54.140, 4, 8, 13.38 W, 1485 MHz, 99 %
10:39:54.731, 4, 8, 13.41 W, 1485 MHz, 100 %
10:39:55.322, 4, 8, 13.36 W, 1485 MHz, 100 %
10:39:55.913, 4, 8, 13.58 W, 1485 MHz, 100 %
10:39:56.499, 4, 8, 13.10 W, 1485 MHz, 66 %
10:39:57.085, 4, 8, 11.13 W, 1485 MHz, 1 %
10:39:57.675, 4, 8, 10.01 W, 1485 MHz, 2 %
```

For locating RUN 1 in it: the probe's rate changes between repetitions
2 and 3, and the sampled generation changes from 4 to 3 between
`10:39:14.778` and `10:39:15.365`.

## 4. WSL2 and Windows `nvidia-smi` read side by side

### 4a. 2026-09-22 (terminal capture, no timestamps)

At idle. Columns: current generation, maximum generation, current
width, maximum width, power, SM clock. The `Windows`/`WSL2` labels are
added here; the values are as printed.

```
Windows : 1, 4, 8, 16, 8.88 W, 712 MHz
WSL2    : 4, 4, 8, 16, 8.37 W, 690 MHz
```

Under sustained device-side load (the script below), about 20 s in.
Columns: current generation, current width, power, SM clock,
utilisation.

```
Windows : 4, 8, 86.90 W, 2100 MHz, 96 %
Windows : 4, 8, 85.24 W, 2122 MHz, 96 %
Windows : 4, 8, 84.99 W, 2115 MHz, 96 %
Windows : 4, 8, 84.75 W, 2100 MHz, 96 %
Windows : 4, 8, 82.37 W, 2047 MHz, 97 %
WSL2    : 4, 8, 83.07 W, 2047 MHz, 97 %
matmuls: 5817
```

Load generator, run in WSL2 for 45 s. The first attempt at a load used
`repro-mlx-3861.py`'s `device` case alone; at ~7 ms it finished before a
sample could see it, so it is not included.

```python
"""Sustained device-side GPU load: ~45 s of back-to-back matmuls."""
import time
import mlx.core as mx

N = 4096
a = mx.random.normal((N, N))
b = mx.random.normal((N, N))
mx.eval(a, b)
end = time.time() + 45
n = 0
while time.time() < end:
    c = a @ b
    mx.eval(c)
    n += 1
print("matmuls:", n)
```

### 4b. 2026-09-23 (re-run, stamped)

The same two readings, plus the driver version from each side, followed
by §6c. Printed by one PowerShell script; the labels are its own.

```
##### versions
utc              : 2026-09-23T03:55:04.067Z
driver (Windows) : 610.71, NVIDIA GeForce RTX 5070 Laptop GPU
driver (WSL2)    : 610.71, NVIDIA GeForce RTX 5070 Laptop GPU

##### idle   columns: gen.current, gen.max, width.current, width.max, power.draw, clocks.sm
utc     : 2026-09-23T03:55:07.621Z
Windows : 2, 4, 8, 8, 15.08 W, 1275 MHz
WSL2    : 4, 4, 8, 8, 15.15 W, 1275 MHz

##### sustained load   columns: gen.current, width.current, power.draw, clocks.sm, utilization.gpu
2026-09-23T03:55:29.070Z  Windows : 4, 8, 79.99 W, 2092 MHz, 98 %
2026-09-23T03:55:33.160Z  Windows : 4, 8, 82.93 W, 2107 MHz, 97 %
2026-09-23T03:55:37.249Z  Windows : 4, 8, 81.09 W, 2092 MHz, 96 %
2026-09-23T03:55:41.339Z  Windows : 4, 8, 83.16 W, 2092 MHz, 97 %
2026-09-23T03:55:45.427Z  Windows : 4, 8, 83.14 W, 2085 MHz, 98 %
2026-09-23T03:55:49.508Z  WSL2    : 4, 8, 82.87 W, 2077 MHz, 96 %
load generator: matmuls: 5498

##### repro-mlx-3861.py with the Windows-side link sampled every 0.5 s
utc_start : 2026-09-23T03:56:00.877Z
mlx 0.32.2, 4096x4096 float32 matmul

                    device :     7.17 ms    19.17 TFLOP/s
                      host :   733.52 ms     0.19 TFLOP/s
   device_after_freed_host :   733.68 ms     0.19 TFLOP/s

If these differ, arrays built from host data are not on the device.
Check with: cudaDeviceGetAttribute(cudaDevAttrConcurrentManagedAccess)
utc_end   : 2026-09-23T03:56:18.981Z

##### link samples during repro (Windows clock, UTC)   columns: gen.current, width.current, power.draw, clocks.sm, utilization.gpu
03:56:01.229, 4, 8, 18.70 W, 1455 MHz, 3 %
03:56:01.956, 4, 8, 18.32 W, 1455 MHz, 3 %
03:56:02.555, 4, 8, 18.02 W, 1957 MHz, 2 %
03:56:03.139, 4, 8, 20.58 W, 1957 MHz, 5 %
03:56:03.727, 4, 8, 19.30 W, 1957 MHz, 2 %
03:56:04.345, 4, 8, 17.89 W, 1957 MHz, 2 %
03:56:04.934, 4, 8, 19.75 W, 1957 MHz, 6 %
03:56:05.522, 4, 8, 36.87 W, 2805 MHz, 100 %
03:56:06.108, 4, 8, 36.94 W, 2805 MHz, 100 %
03:56:06.692, 4, 8, 37.01 W, 2805 MHz, 100 %
03:56:07.282, 4, 8, 37.04 W, 2805 MHz, 100 %
03:56:07.871, 4, 8, 37.07 W, 2805 MHz, 100 %
03:56:08.458, 4, 8, 37.12 W, 2805 MHz, 100 %
03:56:09.037, 4, 8, 37.05 W, 2805 MHz, 100 %
03:56:09.612, 4, 8, 37.04 W, 2805 MHz, 100 %
03:56:10.211, 4, 8, 19.01 W, 2805 MHz, 3 %
03:56:10.812, 4, 8, 18.60 W, 2805 MHz, 5 %
03:56:11.694, 4, 8, 38.85 W, 2805 MHz, 100 %
03:56:12.330, 4, 8, 38.95 W, 2805 MHz, 100 %
03:56:13.012, 4, 8, 38.70 W, 2805 MHz, 100 %
03:56:13.723, 4, 8, 38.94 W, 2805 MHz, 100 %
03:56:14.307, 4, 8, 35.65 W, 2805 MHz, 2 %
03:56:14.892, 4, 8, 31.25 W, 2805 MHz, 100 %
03:56:15.470, 4, 8, 37.57 W, 2805 MHz, 97 %
03:56:16.062, 4, 8, 37.63 W, 2805 MHz, 100 %
03:56:16.652, 4, 8, 37.65 W, 2805 MHz, 99 %
03:56:17.240, 4, 8, 37.64 W, 2805 MHz, 99 %
03:56:17.824, 4, 8, 37.20 W, 2805 MHz, 100 %
03:56:18.410, 4, 8, 36.60 W, 2805 MHz, 100 %
03:56:18.998, 4, 8, 36.22 W, 2805 MHz, 100 %
03:56:19.583, 4, 8, 16.75 W, 1215 MHz, 4 %
```

`width.max` reads 16 on both sides in §4a and 8 on both sides in §4b.

## 5. mlx#3858 — a single chunk across 65,535

### 5a. 2026-09-22 (terminal capture, no timestamp)

Run with `NVIDIA_TF32_OVERRIDE=0`. The script was an untracked copy
that differs from the tracked
`docs/upstream-issues/verify-mlx-3858-batch-limit.py` in one respect: it
reduced the per-field differences with a running `max()`, which reads a
NaN as zero, where the tracked script rejects non-finite values. Each
per-field value below is printed directly and would show as `nan` if it
were one.

```
backend: cuda

n = 65,535
  single chunk: ok
  amplitudes   max |one - split| = 0.000e+00
  delta_E      max |one - split| = 0.000e+00
  delta_sigma  max |one - split| = 0.000e+00
  -> AGREE

n = 65,536
  single chunk: ok
  amplitudes   max |one - split| = 0.000e+00
  delta_E      max |one - split| = 0.000e+00
  delta_sigma  max |one - split| = 0.000e+00
  -> AGREE

n = 65,537
  single chunk: ok
  amplitudes   max |one - split| = 0.000e+00
  delta_E      max |one - split| = 0.000e+00
  delta_sigma  max |one - split| = 0.000e+00
  -> AGREE
```

### 5b and 6b. 2026-09-23 (re-run, stamped)

`repro-mlx-3861.py` and then the tracked
`verify-mlx-3858-batch-limit.py`, in that order, each preceded by the
time, tree, MLX version, driver and kernel.

```
##### repro-mlx-3861.py
utc       : 2026-09-23T03:52:53Z
commit    : c8b941a
mlx       : 0.32.2
driver    : 610.71
gpu       : NVIDIA GeForce RTX 5070 Laptop GPU
kernel    : 6.18.33.2-microsoft-standard-WSL2

mlx 0.32.2, 4096x4096 float32 matmul

                    device :     6.96 ms    19.75 TFLOP/s
                      host :   740.51 ms     0.19 TFLOP/s
   device_after_freed_host :   735.53 ms     0.19 TFLOP/s

If these differ, arrays built from host data are not on the device.
Check with: cudaDeviceGetAttribute(cudaDevAttrConcurrentManagedAccess)

##### verify-mlx-3858-batch-limit.py  (NVIDIA_TF32_OVERRIDE=0)
utc       : 2026-09-23T03:53:14Z
commit    : c8b941a
mlx       : 0.32.2
driver    : 610.71
gpu       : NVIDIA GeForce RTX 5070 Laptop GPU
kernel    : 6.18.33.2-microsoft-standard-WSL2

backend: cuda

n = 65,535
  single chunk: ok
  amplitudes   max |one - split| = 0.000e+00
  delta_E      max |one - split| = 0.000e+00
  delta_sigma  max |one - split| = 0.000e+00
  -> AGREE

n = 65,536
  single chunk: ok
  amplitudes   max |one - split| = 0.000e+00
  delta_E      max |one - split| = 0.000e+00
  delta_sigma  max |one - split| = 0.000e+00
  -> AGREE

n = 65,537
  single chunk: ok
  amplitudes   max |one - split| = 0.000e+00
  delta_E      max |one - split| = 0.000e+00
  delta_sigma  max |one - split| = 0.000e+00
  -> AGREE

exit=0
```

## 6. `repro-mlx-3861.py`

### 6a. 2026-09-22 (terminal capture, no timestamp)

```
mlx 0.32.2, 4096x4096 float32 matmul

                    device :     6.36 ms    21.60 TFLOP/s
                      host :   733.64 ms     0.19 TFLOP/s
   device_after_freed_host :   937.58 ms     0.15 TFLOP/s

If these differ, arrays built from host data are not on the device.
Check with: cudaDeviceGetAttribute(cudaDevAttrConcurrentManagedAccess)
```

6b is in the block above §6. 6c is at the end of §4b, run with the §3
sampler loop active so the link generation is recorded alongside it.

`device_after_freed_host` reads 937.58 ms in 6a, 735.53 ms in 6b and
733.68 ms in 6c. The link was not sampled during 6a or 6b; during 6c it
read generation 4, width 8, throughout.
