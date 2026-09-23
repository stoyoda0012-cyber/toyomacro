# MLX buffer cache on and off — A/B evidence on CUDA

Raw outputs from the RTX 5070 Laptop host comparing `bench_platform`
with MLX's buffer cache at its default (A) and disabled with
`mx.set_cache_limit(0)` (B). This file carries the observations; the
interpretation belongs in `docs/CUDA_BACKEND_POC.md` and
`docs/BENCHMARKS.md` §5.

Two sessions: one A/B pair (§2), then four runs alternating A, B, A, B
in one session with every stdout line time-stamped on the same clock
as a PCIe link log, so each solver's interval can be matched to the
link generations sampled during it (§3).

## Environment

| | |
|---|---|
| GPU / CPU | NVIDIA GeForce RTX 5070 Laptop GPU / AMD Ryzen 9 8940HX |
| Host | Windows 11 + WSL2 Ubuntu 24.04, kernel `6.18.33.2-microsoft-standard-WSL2` |
| Driver | 610.71 |
| MLX | 0.32.2 |
| Repository tree | `c8b941a` |
| Run | `bench_platform --origin windows --no-parity`, default `--n-batch` (200,000), `NVIDIA_TF32_OVERRIDE=0` |
| Date | 2026-09-23 |

`--no-parity` keeps each run in a single process, so the cache setting
reaches every kernel measured. `--runs` was left at 1 for the same
reason: above 1 the harness measures in child processes.

## Provenance of each block

| § | source |
|---|---|
| 2a | terminal capture, filtered at capture to the lines shown |
| 2b, 2c, 3c | printed from the runs' output JSON by the scripts named there |
| 3a | stdout of each run, one file per run, stamped as it arrived |
| 3b | derived from 3a and §4 by `alt_analyse.py`, reproduced below it |
| 4 | the link log file, complete |

The output JSONs themselves are not committed, and not to
`benchmarks/records/` in particular: the record schema has no field for
the cache setting, so a cache-disabled run would be indistinguishable
there from an ordinary CUDA record.

In 3a the `--out` path printed by `bench_platform` is replaced with
`<out>/`; the console line carries it absolute, unlike the record's
sanitised `command` field. Nothing else is edited.

## 1. The wrapper

`cachewrap.py`. The two cases differ only in the one call before
`bench_platform` is imported.

```python
"""Run bench_platform in-process with MLX's buffer cache left alone or disabled.

    python cachewrap.py default <out.json>
    python cachewrap.py off     <out.json>

The only difference between the two is the one call below, made before
bench_platform is imported. `--no-parity` keeps everything in this one
process, so the setting reaches every kernel measured.
"""
import sys

mode, out = sys.argv[1], sys.argv[2]
import mlx.core as mx  # noqa: E402

if mode == "off":
    mx.set_cache_limit(0)
print(f"cache mode: {mode}", flush=True)

from toyomacro.voigtfit.benchmarks import bench_platform  # noqa: E402

bench_platform.main(["--origin", "windows", "--no-parity", "--out", out])
print(f"cache memory at exit: {mx.get_cache_memory()} bytes", flush=True)
```

## 2. Pair 1 — one A, one B (04:04–04:11 UTC)

A ran first, then 90 s idle, then B. The Windows-side link was sampled
throughout each case but the output lines were not time-stamped, so the
samples cannot be matched to solvers; only the per-case counts were
printed.

### 2a. Console

```
########## case default
utc_start: 04:04:47.891
cache mode: default
  projection_kernel           4,535,922 spec/s
  amp_only_projection         8,651,469 spec/s   MAE(amp)=0.0294
  taylor_4step                  474,149 spec/s   MAE(amp)=0.0219
  dict2d_parabola             1,353,750 spec/s   MAE(amp)=0.0222
  multipeak_2comp                34,083 spec/s   MAE(amp)=0.0290
  scipy_curve_fit                   176 spec/s   MAE(amp)=0.0219
  lmfit                             186 spec/s   MAE(amp)=0.0219
record quality: quiet
cache memory at exit: 5369512264 bytes
utc_end:   04:07:29.721
link generations sampled during case default: gen1 x51, gen2 x70, gen3 x24, gen4 x129
settling 90 s
########## case off
utc_start: 04:09:09.637
cache mode: off
  projection_kernel           4,585,389 spec/s
  amp_only_projection        17,183,721 spec/s   MAE(amp)=0.0294
  taylor_4step                  702,053 spec/s   MAE(amp)=0.0219
  dict2d_parabola               456,870 spec/s   MAE(amp)=0.0222
  multipeak_2comp               180,314 spec/s   MAE(amp)=0.0290
  scipy_curve_fit                   171 spec/s   MAE(amp)=0.0219
  lmfit                             200 spec/s   MAE(amp)=0.0219
record quality: quiet
cache memory at exit: 0 bytes
utc_end:   04:11:10.297
link generations sampled during case off: gen1 x29, gen2 x92, gen3 x46, gen4 x50
```

### 2b. Per-repetition rates

The nine timed repetitions of each solver, from `timings_s`.

```
=== default
  projection_kernel    M/s: 9.37, 4.93, 4.54, 4.54, 4.54, 4.53, 4.53, 4.53, 4.53
  amp_only_projection  M/s: 8.63, 8.61, 8.79, 8.69, 8.85, 8.65, 8.63, 8.81, 8.53
  taylor_4step         k/s: 497, 474, 479, 460, 437, 418, 399, 856, 852
  dict2d_parabola      M/s: 1.33, 1.38, 1.36, 1.35, 1.35, 1.33, 1.35, 1.36, 1.36
  multipeak_2comp      k/s: 53, 73, 38, 39, 33, 34, 32, 29, 32
=== off
  projection_kernel    M/s: 5.44, 4.55, 4.54, 4.56, 4.60, 4.60, 4.59, 4.59, 4.58
  amp_only_projection  M/s: 16.80, 16.88, 17.36, 17.18, 17.17, 17.21, 17.24, 16.85, 17.32
  taylor_4step         k/s: 727, 737, 764, 702, 640, 695, 701, 691, 766
  dict2d_parabola      k/s: 461, 460, 457, 436, 462, 464, 456, 435, 456
  multipeak_2comp      k/s: 183, 177, 177, 180, 181, 180, 186, 184, 179
```

### 2c. Answers

Input hash and every `mae_*` field, A against B.

```
input_sha256 default: 212b959b22568c16662ad51506e2b356ebbc4038457748eab4b09866eaeef091
input_sha256 off    : 212b959b22568c16662ad51506e2b356ebbc4038457748eab4b09866eaeef091
identical: True

mae keys: ['mae_amp', 'mae_dE', 'mae_dsigma']
solver              key                        default                   off  equal
amp_only_projection mae_amp       0.029441094598883437  0.029441094598883437  True
taylor_4step        mae_amp       0.021891153550741064  0.021891153550741064  True
taylor_4step        mae_dE        0.013760101116344499  0.013760101116344499  True
taylor_4step        mae_dsigma    0.013913020205371825  0.013913020205371825  True
dict2d_parabola     mae_amp        0.02223129842563654   0.02223129842563654  True
dict2d_parabola     mae_dE        0.013351282999338876  0.013351282999338876  True
dict2d_parabola     mae_dsigma    0.012833833646284884  0.012833833646284884  True
multipeak_2comp     mae_amp        0.02896153203955796   0.02896153203955796  True
multipeak_2comp     mae_dE        0.019570656657920116  0.019570656657920116  True
multipeak_2comp     mae_dsigma    0.017360326730129032  0.017360326730129032  True
scipy_curve_fit     mae_amp        0.02194642850864147   0.02194642850864147  True
scipy_curve_fit     mae_dE        0.013086001140407763  0.013086001140407763  True
scipy_curve_fit     mae_dsigma    0.012530243611882345  0.012530243611882345  True
lmfit               mae_amp        0.02194630510576088   0.02194630510576088  True
lmfit               mae_dE         0.01308597789456242   0.01308597789456242  True
lmfit               mae_dsigma     0.01253023695195868   0.01253023695195868  True
```

## 3. Four runs alternating A, B, A, B (10:36–10:49 UTC)

One session, 60 s idle between runs. A guard aborted the sequence
unless no benchmark process was already running. The link was sampled
for the whole session into one log (§4), and each stdout line was
stamped on arrival by the same Windows-side process, so §3a and §4
share a clock.

### 3a. Stdout

Each solver announces itself with a line ending in `...`; its interval
runs to the next such line. Two do not lead with their own name:
`scipy curve_fit on 500 ...` and `lmfit on 500 ...`.

`default` #1:

```
10:36:59.355 cache mode: default
10:37:01.578 backend: cuda (NVIDIA GeForce RTX 5070 Laptop GPU)
10:37:01.809 generating 200,000 single-peak spectra (seed 0) ...
10:37:13.600 projection_kernel ...
10:37:32.235 amp_only_projection ...
10:37:32.673 taylor_4step ...
10:37:42.137 dict2d_parabola ...
10:37:56.561 multipeak_2comp ...
10:39:02.942 scipy curve_fit on 500 ...
10:39:10.983 lmfit on 500 ...
10:39:20.900   projection_kernel           9,508,932 spec/s
10:39:20.901   amp_only_projection        17,928,633 spec/s   MAE(amp)=0.0294
10:39:20.901   taylor_4step                  499,180 spec/s   MAE(amp)=0.0219
10:39:20.902   dict2d_parabola             1,464,653 spec/s   MAE(amp)=0.0222
10:39:20.902   multipeak_2comp                45,999 spec/s   MAE(amp)=0.0290
10:39:20.902   scipy_curve_fit                   182 spec/s   MAE(amp)=0.0219
10:39:20.902   lmfit                             160 spec/s   MAE(amp)=0.0219
10:39:21.182 
10:39:21.182 record quality: quiet
10:39:21.183 wrote <out>/diag_default_1.json
10:39:21.183 cache memory at exit: 5807259976 bytes
```

`off` #1:

```
10:40:30.110 cache mode: off
10:40:32.308 backend: cuda (NVIDIA GeForce RTX 5070 Laptop GPU)
10:40:32.557 generating 200,000 single-peak spectra (seed 0) ...
10:40:43.394 projection_kernel ...
10:41:03.069 amp_only_projection ...
10:41:04.554 taylor_4step ...
10:41:12.871 dict2d_parabola ...
10:41:29.733 multipeak_2comp ...
10:42:06.137 scipy curve_fit on 500 ...
10:42:14.983 lmfit on 500 ...
10:42:24.456   projection_kernel           8,915,915 spec/s
10:42:24.457   amp_only_projection        16,832,744 spec/s   MAE(amp)=0.0294
10:42:24.457   taylor_4step                  707,688 spec/s   MAE(amp)=0.0219
10:42:24.458   dict2d_parabola               508,279 spec/s   MAE(amp)=0.0222
10:42:24.458   multipeak_2comp               253,770 spec/s   MAE(amp)=0.0290
10:42:24.459   scipy_curve_fit                   163 spec/s   MAE(amp)=0.0219
10:42:24.459   lmfit                             165 spec/s   MAE(amp)=0.0219
10:42:24.741 
10:42:24.742 record quality: quiet
10:42:24.743 wrote <out>/diag_off_1.json
10:42:24.743 cache memory at exit: 0 bytes
```

`default` #2:

```
10:43:31.274 cache mode: default
10:43:33.555 backend: cuda (NVIDIA GeForce RTX 5070 Laptop GPU)
10:43:33.784 generating 200,000 single-peak spectra (seed 0) ...
10:43:45.402 projection_kernel ...
10:44:04.146 amp_only_projection ...
10:44:04.561 taylor_4step ...
10:44:13.597 dict2d_parabola ...
10:44:27.447 multipeak_2comp ...
10:45:47.411 scipy curve_fit on 500 ...
10:45:55.426 lmfit on 500 ...
10:46:05.331   projection_kernel           9,210,698 spec/s
10:46:05.331   amp_only_projection        17,492,163 spec/s   MAE(amp)=0.0294
10:46:05.332   taylor_4step                  483,961 spec/s   MAE(amp)=0.0219
10:46:05.332   dict2d_parabola             1,393,243 spec/s   MAE(amp)=0.0222
10:46:05.333   multipeak_2comp                34,303 spec/s   MAE(amp)=0.0290
10:46:05.333   scipy_curve_fit                   187 spec/s   MAE(amp)=0.0219
10:46:05.333   lmfit                             157 spec/s   MAE(amp)=0.0219
10:46:05.609 
10:46:05.610 record quality: quiet
10:46:05.611 wrote <out>/diag_default_2.json
10:46:05.611 cache memory at exit: 5364965704 bytes
```

`off` #2:

```
10:47:14.553 cache mode: off
10:47:16.809 backend: cuda (NVIDIA GeForce RTX 5070 Laptop GPU)
10:47:17.036 generating 200,000 single-peak spectra (seed 0) ...
10:47:27.982 projection_kernel ...
10:47:47.285 amp_only_projection ...
10:47:48.813 taylor_4step ...
10:47:57.642 dict2d_parabola ...
10:48:14.703 multipeak_2comp ...
10:48:50.190 scipy curve_fit on 500 ...
10:48:59.519 lmfit on 500 ...
10:49:08.282   projection_kernel           9,337,232 spec/s
10:49:08.301   amp_only_projection        17,275,385 spec/s   MAE(amp)=0.0294
10:49:08.302   taylor_4step                  735,953 spec/s   MAE(amp)=0.0219
10:49:08.302   dict2d_parabola               495,107 spec/s   MAE(amp)=0.0222
10:49:08.302   multipeak_2comp               249,989 spec/s   MAE(amp)=0.0290
10:49:08.304   scipy_curve_fit                   145 spec/s   MAE(amp)=0.0219
10:49:08.304   lmfit                             184 spec/s   MAE(amp)=0.0219
10:49:08.557 
10:49:08.558 record quality: quiet
10:49:08.559 wrote <out>/diag_off_2.json
10:49:08.559 cache memory at exit: 0 bytes
```

### 3b. Link generations by solver

For each solver, its interval from §3a, its median rate, and the count
of §4 samples in that interval by link generation (`g4:9` = nine
samples at generation 4).

```
=== default#1   quality=quiet
  projection_kernel      18.6 s      9.51 M   link g2:23 g3:2 g4:7
  amp_only_projection     0.4 s     17.93 M   link g4:1
  taylor_4step            9.5 s     499.2 k   link g1:1 g2:6 g4:9
  dict2d_parabola        14.4 s      1.46 M   link g2:15 g4:9
  multipeak_2comp        66.4 s      46.0 k   link g1:22 g2:5 g3:13 g4:72
  scipy_curve_fit         8.0 s  -   link g2:10 g4:3
  lmfit                  10.2 s  -   link g1:14 g2:4

=== off#1   quality=quiet
  projection_kernel      19.7 s      8.92 M   link g2:24 g3:2 g4:7
  amp_only_projection     1.5 s     16.83 M   link g4:3
  taylor_4step            8.3 s     707.7 k   link g2:7 g4:7
  dict2d_parabola        16.9 s     508.3 k   link g2:14 g4:15
  multipeak_2comp        36.4 s     253.8 k   link g2:32 g3:3 g4:27
  scipy_curve_fit         8.8 s  -   link g2:7 g4:8
  lmfit                   9.8 s  -   link g2:17

=== default#2   quality=quiet
  projection_kernel      18.7 s      9.21 M   link g2:24 g4:8
  amp_only_projection     0.4 s     17.49 M   link (no sample)
  taylor_4step            9.0 s     484.0 k   link g2:6 g4:10
  dict2d_parabola        13.8 s      1.39 M   link g1:1 g2:14 g4:8
  multipeak_2comp        80.0 s      34.3 k   link g1:26 g2:7 g3:8 g4:94
  scipy_curve_fit         8.0 s  -   link g2:14
  lmfit                  10.2 s  -   link g1:15 g2:2

=== off#2   quality=quiet
  projection_kernel      19.3 s      9.34 M   link g2:24 g3:2 g4:7
  amp_only_projection     1.5 s     17.28 M   link g4:3
  taylor_4step            8.8 s     736.0 k   link g1:1 g4:14
  dict2d_parabola        17.1 s     495.1 k   link g2:15 g4:14
  multipeak_2comp        35.5 s     250.0 k   link g2:33 g3:3 g4:24
  scipy_curve_fit         9.3 s  -   link g2:9 g4:7
  lmfit                   9.0 s  -   link g2:15

input_sha256 identical across all four: True
mae_* identical across all four (18 values each): True
```

Two limits on reading this:

- An interval includes the solver's CPU-side setup — for
  `projection_kernel`, generating the 2.42 GB operand — during which the
  GPU is idle and the link steps down. The counts are the link over the
  whole interval, not while the GPU was busy.
- `amp_only_projection` runs for 0.4–1.5 s and holds 0–3 samples.

Produced by:

```python
"""Per-solver link generations for the four alternating cache runs.

Both the stdout stamps and the link log use the Windows clock, so an
interval taken from one can be looked up in the other directly.
"""
import collections
import json
import os
import re

LOGDIR = "<directory holding link.log and stdout_*.log>"
DIAG = os.path.expanduser("~/bench-logs")
SOLVERS = ["projection_kernel", "amp_only_projection", "taylor_4step",
           "dict2d_parabola", "multipeak_2comp", "scipy_curve_fit", "lmfit"]
RUNS = [("default", 1), ("off", 1), ("default", 2), ("off", 2)]


def secs(hms):
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def read(path):
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        return [ln.rstrip("\r\n") for ln in fh if ln.strip()]


link = []
for ln in read(f"{LOGDIR}/link.log"):
    p = [x.strip() for x in ln.split(",")]
    if len(p) >= 2 and re.match(r"\d\d:\d\d:\d\d\.\d+$", p[0]):
        link.append((secs(p[0]), p[1]))

hashes, maes = {}, {}
for mode, idx in RUNS:
    tag = f"{mode}#{idx}"
    lines = read(f"{LOGDIR}/stdout_{mode}_{idx}.log")
    stamped = []
    for ln in lines:
        m = re.match(r"(\d\d:\d\d:\d\d\.\d+) (.*)$", ln)
        if m:
            stamped.append((secs(m.group(1)), m.group(2)))
    # Every solver announces itself with a line ending in "...". Two of
    # them do not print their own name first ("scipy curve_fit on 500 ...",
    # "lmfit on 500 ..."), so map the first token; without these as
    # boundaries the solver before them would absorb their time.
    alias = {"scipy": "scipy_curve_fit", "lmfit": "lmfit"}
    starts = []
    for t, txt in stamped:
        body = txt.strip()
        if not body.endswith("...") or body.startswith("generating"):
            continue
        name = alias.get(body.split()[0], body.split()[0])
        if name in SOLVERS:
            starts.append((t, name))
    tail = next((t for t, txt in stamped if txt.strip().startswith("record quality")),
                stamped[-1][0])
    d = json.load(open(f"{DIAG}/diag_{mode}_{idx}.json", encoding="utf-8"))
    rates = {r["solver"]: r.get("rate_median") for r in d["results"]}
    hashes[tag] = d["problem"]["input_sha256"]
    maes[tag] = {(r["solver"], k): r[k] for r in d["results"] for k in r if k.startswith("mae")}
    print(f"=== {tag}   quality={d['environment']['quality']['verdict']}")
    for i, (t0, s) in enumerate(starts):
        t1 = starts[i + 1][0] if i + 1 < len(starts) else tail
        gens = collections.Counter(g for t, g in link if t0 <= t < t1)
        gtxt = " ".join(f"g{g}:{n}" for g, n in sorted(gens.items())) or "(no sample)"
        r = rates.get(s)
        rtxt = (f"{r / 1e6:8.2f} M" if r and r >= 1e6 else f"{r / 1e3:8.1f} k") if r else "-"
        print(f"  {s:<20} {t1 - t0:6.1f} s  {rtxt}   link {gtxt}")
    print()

print("input_sha256 identical across all four:", len(set(hashes.values())) == 1)
ref = maes["default#1"]
same = all(maes[k] == ref for k in maes)
print(f"mae_* identical across all four ({len(ref)} values each):", same)
```

### 3c. Per-repetition rates, `taylor_4step` and `multipeak_2comp`

```
default#1  taylor_4step     k/s: 549, 522, 499, 471, 453, 442, 402, 847, 849
default#1  multipeak_2comp  k/s: 75, 62, 65, 42, 51, 46, 43, 45, 41
off#1  taylor_4step     k/s: 766, 670, 583, 708, 741, 688, 716, 735, 705
off#1  multipeak_2comp  k/s: 178, 217, 255, 254, 253, 254, 255, 257, 247
default#2  taylor_4step     k/s: 527, 506, 484, 451, 384, 401, 391, 834, 842
default#2  multipeak_2comp  k/s: 60, 66, 54, 38, 34, 26, 27, 29, 27
off#2  taylor_4step     k/s: 756, 756, 792, 708, 599, 666, 771, 568, 736
off#2  multipeak_2comp  k/s: 176, 210, 243, 261, 244, 262, 261, 260, 250
```

## 4. Link log

Windows side, every 0.5 s requested (actual spacing about 0.6 s, each
`nvidia-smi.exe` call adding to it), 10:36:57–10:49:09 UTC. Columns:
time (UTC, Windows clock), link generation, link width, power draw, SM
clock, GPU utilisation. Complete, 1,246 lines.

```
10:36:57.886, 2, 8, 14.29 W, 1327 MHz, 3 %
10:36:58.610, 2, 8, 14.65 W, 1297 MHz, 3 %
10:36:59.194, 4, 8, 14.71 W, 1297 MHz, 3 %
10:36:59.782, 4, 8, 15.71 W, 1297 MHz, 2 %
10:37:00.366, 4, 8, 16.10 W, 1297 MHz, 2 %
10:37:00.970, 4, 8, 15.97 W, 1297 MHz, 2 %
10:37:01.554, 4, 8, 15.89 W, 1950 MHz, 2 %
10:37:02.139, 4, 8, 15.99 W, 1950 MHz, 2 %
10:37:02.724, 4, 8, 15.90 W, 1950 MHz, 2 %
10:37:03.310, 4, 8, 16.00 W, 1950 MHz, 2 %
10:37:03.901, 4, 8, 15.88 W, 1950 MHz, 2 %
10:37:04.490, 4, 8, 15.90 W, 1950 MHz, 2 %
10:37:05.078, 4, 8, 16.44 W, 1957 MHz, 2 %
10:37:05.662, 4, 8, 15.16 W, 1957 MHz, 3 %
10:37:06.246, 2, 8, 14.92 W, 1957 MHz, 3 %
10:37:06.833, 2, 8, 14.63 W, 1957 MHz, 3 %
10:37:07.423, 2, 8, 14.92 W, 1957 MHz, 3 %
10:37:08.011, 2, 8, 14.73 W, 1282 MHz, 3 %
10:37:08.597, 2, 8, 14.58 W, 1282 MHz, 3 %
10:37:09.183, 2, 8, 14.52 W, 1282 MHz, 3 %
10:37:09.766, 2, 8, 14.36 W, 1282 MHz, 3 %
10:37:10.354, 2, 8, 14.46 W, 1282 MHz, 3 %
10:37:10.945, 2, 8, 14.45 W, 1282 MHz, 3 %
10:37:11.528, 2, 8, 14.57 W, 1260 MHz, 3 %
10:37:12.119, 2, 8, 14.62 W, 1260 MHz, 3 %
10:37:12.706, 2, 8, 14.85 W, 1260 MHz, 3 %
10:37:13.290, 2, 8, 15.00 W, 1260 MHz, 3 %
10:37:13.881, 2, 8, 15.05 W, 1260 MHz, 3 %
10:37:14.467, 2, 8, 15.44 W, 1260 MHz, 3 %
10:37:15.053, 2, 8, 14.93 W, 1282 MHz, 3 %
10:37:15.641, 2, 8, 15.08 W, 1282 MHz, 3 %
10:37:16.226, 2, 8, 15.01 W, 1282 MHz, 3 %
10:37:16.814, 2, 8, 15.19 W, 1282 MHz, 3 %
10:37:17.399, 2, 8, 15.16 W, 1282 MHz, 3 %
10:37:17.983, 2, 8, 14.76 W, 1282 MHz, 3 %
10:37:18.567, 2, 8, 14.19 W, 1267 MHz, 3 %
10:37:19.151, 2, 8, 14.48 W, 1267 MHz, 3 %
10:37:19.741, 2, 8, 14.50 W, 1267 MHz, 3 %
10:37:20.328, 2, 8, 15.10 W, 1267 MHz, 3 %
10:37:20.913, 2, 8, 14.98 W, 1267 MHz, 3 %
10:37:21.504, 2, 8, 14.76 W, 1267 MHz, 3 %
10:37:22.091, 2, 8, 15.74 W, 1357 MHz, 9 %
10:37:22.679, 2, 8, 14.59 W, 1357 MHz, 3 %
10:37:23.268, 2, 8, 15.51 W, 1357 MHz, 3 %
10:37:23.855, 2, 8, 14.87 W, 1357 MHz, 3 %
10:37:24.441, 2, 8, 14.99 W, 1357 MHz, 3 %
10:37:25.025, 2, 8, 14.85 W, 1327 MHz, 3 %
10:37:25.611, 2, 8, 14.75 W, 1327 MHz, 3 %
10:37:26.197, 2, 8, 15.03 W, 1327 MHz, 3 %
10:37:26.787, 2, 8, 14.83 W, 1327 MHz, 3 %
10:37:27.376, 4, 8, 19.55 W, 1327 MHz, 95 %
10:37:27.957, 3, 8, 34.09 W, 1327 MHz, 67 %
10:37:28.545, 3, 8, 13.11 W, 1492 MHz, 99 %
10:37:29.139, 4, 8, 27.26 W, 1492 MHz, 100 %
10:37:29.726, 4, 8, 35.47 W, 1492 MHz, 100 %
10:37:30.317, 4, 8, 35.49 W, 1492 MHz, 100 %
10:37:30.909, 4, 8, 34.49 W, 1492 MHz, 99 %
10:37:31.495, 4, 8, 35.54 W, 1492 MHz, 99 %
10:37:32.083, 4, 8, 35.58 W, 2820 MHz, 100 %
10:37:32.670, 4, 8, 23.09 W, 2820 MHz, 3 %
10:37:33.261, 4, 8, 15.14 W, 2820 MHz, 3 %
10:37:33.847, 4, 8, 15.28 W, 2820 MHz, 4 %
10:37:34.437, 4, 8, 14.95 W, 2820 MHz, 3 %
10:37:35.021, 2, 8, 15.91 W, 1432 MHz, 3 %
10:37:35.604, 2, 8, 15.52 W, 1432 MHz, 3 %
10:37:36.190, 2, 8, 14.74 W, 1432 MHz, 3 %
10:37:36.778, 2, 8, 15.08 W, 1432 MHz, 3 %
10:37:37.361, 2, 8, 15.34 W, 1432 MHz, 3 %
10:37:37.948, 2, 8, 15.21 W, 1432 MHz, 3 %
10:37:38.536, 1, 8, 22.89 W, 1050 MHz, 3 %
10:37:39.123, 4, 8, 22.89 W, 1050 MHz, 57 %
10:37:39.711, 4, 8, 21.76 W, 1050 MHz, 22 %
10:37:40.297, 4, 8, 23.89 W, 1050 MHz, 2 %
10:37:40.882, 4, 8, 15.33 W, 1050 MHz, 3 %
10:37:41.468, 4, 8, 20.60 W, 1050 MHz, 51 %
10:37:42.052, 4, 8, 26.52 W, 2700 MHz, 99 %
10:37:42.639, 4, 8, 14.73 W, 2700 MHz, 4 %
10:37:43.215, 4, 8, 14.27 W, 2700 MHz, 4 %
10:37:43.803, 4, 8, 14.83 W, 2700 MHz, 3 %
10:37:44.391, 4, 8, 15.65 W, 2700 MHz, 3 %
10:37:44.981, 4, 8, 16.16 W, 2700 MHz, 3 %
10:37:45.568, 4, 8, 15.90 W, 1440 MHz, 3 %
10:37:46.156, 2, 8, 15.71 W, 1440 MHz, 3 %
10:37:46.741, 2, 8, 13.82 W, 1440 MHz, 10 %
10:37:47.329, 2, 8, 15.00 W, 1440 MHz, 3 %
10:37:47.905, 2, 8, 15.96 W, 1440 MHz, 3 %
10:37:48.494, 2, 8, 15.26 W, 1440 MHz, 3 %
10:37:49.081, 2, 8, 15.25 W, 1312 MHz, 3 %
10:37:49.664, 2, 8, 15.06 W, 1312 MHz, 3 %
10:37:50.251, 2, 8, 15.11 W, 1312 MHz, 3 %
10:37:50.837, 2, 8, 15.33 W, 1312 MHz, 3 %
10:37:51.429, 2, 8, 15.22 W, 1312 MHz, 3 %
10:37:52.003, 2, 8, 14.71 W, 1312 MHz, 3 %
10:37:52.588, 2, 8, 15.10 W, 1305 MHz, 3 %
10:37:53.170, 2, 8, 15.20 W, 1305 MHz, 3 %
10:37:53.759, 2, 8, 15.08 W, 1305 MHz, 3 %
10:37:54.349, 2, 8, 14.72 W, 1305 MHz, 3 %
10:37:54.933, 4, 8, 14.64 W, 1305 MHz, 3 %
10:37:55.515, 4, 8, 17.85 W, 1305 MHz, 3 %
10:37:56.098, 4, 8, 30.15 W, 2287 MHz, 59 %
10:37:56.688, 4, 8, 30.70 W, 2287 MHz, 72 %
10:37:57.274, 4, 8, 17.47 W, 2287 MHz, 3 %
10:37:57.863, 4, 8, 14.00 W, 2287 MHz, 4 %
10:37:58.449, 4, 8, 15.34 W, 2287 MHz, 3 %
10:37:59.036, 4, 8, 15.32 W, 1312 MHz, 3 %
10:37:59.619, 4, 8, 14.74 W, 1312 MHz, 3 %
10:38:00.205, 4, 8, 15.12 W, 1312 MHz, 3 %
10:38:00.788, 4, 8, 14.90 W, 1312 MHz, 3 %
10:38:01.378, 4, 8, 14.83 W, 1312 MHz, 3 %
10:38:01.967, 3, 8, 15.15 W, 1312 MHz, 3 %
10:38:02.555, 2, 8, 9.85 W, 1215 MHz, 5 %
10:38:03.145, 2, 8, 9.89 W, 1215 MHz, 4 %
10:38:03.729, 2, 8, 9.75 W, 1215 MHz, 4 %
10:38:04.318, 2, 8, 9.80 W, 1215 MHz, 4 %
10:38:04.906, 1, 8, 7.73 W, 1215 MHz, 4 %
10:38:05.488, 1, 8, 7.85 W, 1215 MHz, 5 %
10:38:06.079, 1, 8, 7.87 W, 1252 MHz, 4 %
10:38:06.666, 1, 8, 7.74 W, 1252 MHz, 4 %
10:38:07.248, 1, 8, 7.80 W, 1252 MHz, 4 %
10:38:07.827, 1, 8, 7.74 W, 1252 MHz, 5 %
10:38:08.417, 1, 8, 7.63 W, 1252 MHz, 4 %
10:38:09.005, 1, 8, 7.53 W, 1252 MHz, 4 %
10:38:09.596, 1, 8, 7.57 W, 1192 MHz, 4 %
10:38:10.185, 1, 8, 7.55 W, 1192 MHz, 4 %
10:38:10.776, 1, 8, 7.54 W, 1192 MHz, 4 %
10:38:11.363, 1, 8, 7.55 W, 1192 MHz, 4 %
10:38:11.949, 1, 8, 7.56 W, 1192 MHz, 4 %
10:38:12.537, 1, 8, 7.60 W, 1155 MHz, 4 %
10:38:13.125, 1, 8, 7.58 W, 1155 MHz, 4 %
10:38:13.710, 1, 8, 7.43 W, 1155 MHz, 4 %
10:38:14.297, 1, 8, 7.50 W, 1155 MHz, 4 %
10:38:14.882, 1, 8, 7.55 W, 1155 MHz, 4 %
10:38:15.468, 1, 8, 7.57 W, 1155 MHz, 4 %
10:38:16.055, 1, 8, 7.52 W, 1102 MHz, 4 %
10:38:16.643, 1, 8, 7.39 W, 1102 MHz, 4 %
10:38:17.233, 2, 8, 8.72 W, 1102 MHz, 4 %
10:38:17.820, 1, 8, 8.80 W, 1102 MHz, 5 %
10:38:18.407, 3, 8, 8.14 W, 1102 MHz, 4 %
10:38:18.991, 3, 8, 14.84 W, 1102 MHz, 99 %
10:38:19.576, 3, 8, 9.75 W, 1080 MHz, 4 %
10:38:20.161, 3, 8, 9.88 W, 1080 MHz, 4 %
10:38:20.747, 3, 8, 9.91 W, 1080 MHz, 4 %
10:38:21.334, 3, 8, 9.93 W, 1080 MHz, 4 %
10:38:21.917, 3, 8, 9.89 W, 1080 MHz, 4 %
10:38:22.494, 3, 8, 9.90 W, 1080 MHz, 4 %
10:38:23.080, 3, 8, 9.98 W, 1125 MHz, 5 %
10:38:23.666, 3, 8, 9.94 W, 1125 MHz, 4 %
10:38:24.243, 3, 8, 12.55 W, 1125 MHz, 60 %
10:38:24.826, 3, 8, 13.60 W, 1125 MHz, 99 %
10:38:25.432, 4, 8, 13.62 W, 1125 MHz, 92 %
10:38:26.019, 4, 8, 13.43 W, 1125 MHz, 99 %
10:38:26.610, 4, 8, 13.38 W, 1185 MHz, 65 %
10:38:27.200, 4, 8, 12.88 W, 1185 MHz, 99 %
10:38:27.803, 4, 8, 12.60 W, 1185 MHz, 20 %
10:38:28.412, 4, 8, 13.11 W, 1185 MHz, 99 %
10:38:29.018, 4, 8, 13.55 W, 1185 MHz, 99 %
10:38:29.593, 4, 8, 24.64 W, 2820 MHz, 100 %
10:38:30.180, 4, 8, 33.79 W, 2820 MHz, 93 %
10:38:30.771, 4, 8, 33.11 W, 2820 MHz, 100 %
10:38:31.363, 4, 8, 32.68 W, 2820 MHz, 100 %
10:38:31.971, 4, 8, 31.31 W, 2820 MHz, 17 %
10:38:32.561, 4, 8, 35.49 W, 2812 MHz, 98 %
10:38:33.171, 4, 8, 34.26 W, 2812 MHz, 99 %
10:38:33.747, 4, 8, 32.98 W, 2812 MHz, 98 %
10:38:34.323, 4, 8, 33.47 W, 2812 MHz, 99 %
10:38:34.941, 4, 8, 33.70 W, 2812 MHz, 100 %
10:38:35.548, 4, 8, 34.02 W, 2812 MHz, 99 %
10:38:36.128, 4, 8, 34.46 W, 2820 MHz, 99 %
10:38:36.719, 4, 8, 33.80 W, 2820 MHz, 100 %
10:38:37.357, 4, 8, 33.87 W, 2820 MHz, 100 %
10:38:37.993, 4, 8, 33.37 W, 2820 MHz, 100 %
10:38:38.570, 4, 8, 34.19 W, 2820 MHz, 100 %
10:38:39.176, 4, 8, 33.86 W, 2820 MHz, 100 %
10:38:39.764, 4, 8, 34.83 W, 2820 MHz, 99 %
10:38:40.355, 4, 8, 34.46 W, 2820 MHz, 99 %
10:38:40.966, 4, 8, 33.60 W, 2820 MHz, 99 %
10:38:41.571, 4, 8, 33.82 W, 2820 MHz, 100 %
10:38:42.158, 4, 8, 32.84 W, 2820 MHz, 99 %
10:38:42.794, 4, 8, 33.00 W, 2820 MHz, 98 %
10:38:43.419, 4, 8, 32.67 W, 2820 MHz, 8 %
10:38:44.024, 4, 8, 33.06 W, 2820 MHz, 99 %
10:38:44.611, 4, 8, 34.48 W, 2752 MHz, 93 %
10:38:45.215, 4, 8, 34.41 W, 2752 MHz, 99 %
10:38:45.855, 4, 8, 34.43 W, 2752 MHz, 99 %
10:38:46.506, 4, 8, 34.83 W, 2752 MHz, 99 %
10:38:47.098, 4, 8, 34.16 W, 2805 MHz, 100 %
10:38:47.704, 4, 8, 34.65 W, 2805 MHz, 100 %
10:38:48.330, 4, 8, 35.57 W, 2805 MHz, 100 %
10:38:48.905, 4, 8, 34.28 W, 2805 MHz, 99 %
10:38:49.511, 4, 8, 33.85 W, 2805 MHz, 100 %
10:38:50.132, 4, 8, 34.60 W, 2812 MHz, 91 %
10:38:50.755, 4, 8, 34.23 W, 2812 MHz, 100 %
10:38:51.349, 4, 8, 34.36 W, 2812 MHz, 100 %
10:38:51.926, 4, 8, 30.80 W, 2812 MHz, 100 %
10:38:52.518, 4, 8, 33.02 W, 2812 MHz, 100 %
10:38:53.173, 4, 8, 34.54 W, 2812 MHz, 99 %
10:38:53.766, 4, 8, 32.69 W, 2812 MHz, 99 %
10:38:54.367, 4, 8, 34.47 W, 2812 MHz, 100 %
10:38:54.958, 4, 8, 34.22 W, 2812 MHz, 90 %
10:38:55.536, 4, 8, 34.22 W, 2812 MHz, 100 %
10:38:56.143, 4, 8, 34.56 W, 2812 MHz, 99 %
10:38:56.750, 4, 8, 34.87 W, 2805 MHz, 99 %
10:38:57.326, 4, 8, 35.00 W, 2805 MHz, 100 %
10:38:57.914, 4, 8, 34.70 W, 2805 MHz, 91 %
10:38:58.506, 4, 8, 34.47 W, 2805 MHz, 100 %
10:38:59.111, 4, 8, 34.75 W, 2805 MHz, 99 %
10:38:59.719, 4, 8, 34.04 W, 2812 MHz, 99 %
10:39:00.324, 4, 8, 35.59 W, 2812 MHz, 100 %
10:39:00.962, 4, 8, 31.57 W, 2812 MHz, 45 %
10:39:01.565, 4, 8, 32.97 W, 2812 MHz, 99 %
10:39:02.172, 4, 8, 34.38 W, 2805 MHz, 99 %
10:39:02.793, 4, 8, 29.88 W, 2805 MHz, 3 %
10:39:03.375, 4, 8, 15.14 W, 2805 MHz, 4 %
10:39:03.964, 4, 8, 15.56 W, 2805 MHz, 3 %
10:39:04.546, 4, 8, 16.47 W, 2805 MHz, 3 %
10:39:05.134, 2, 8, 16.66 W, 2805 MHz, 3 %
10:39:05.724, 2, 8, 16.34 W, 1455 MHz, 4 %
10:39:06.310, 2, 8, 15.09 W, 1455 MHz, 4 %
10:39:06.894, 2, 8, 15.30 W, 1455 MHz, 4 %
10:39:07.478, 2, 8, 14.60 W, 1455 MHz, 3 %
10:39:08.067, 2, 8, 15.15 W, 1455 MHz, 3 %
10:39:08.656, 2, 8, 15.13 W, 1455 MHz, 3 %
10:39:09.239, 2, 8, 14.87 W, 1230 MHz, 3 %
10:39:09.831, 2, 8, 14.86 W, 1230 MHz, 3 %
10:39:10.418, 2, 8, 15.06 W, 1230 MHz, 3 %
10:39:11.007, 2, 8, 15.08 W, 1230 MHz, 3 %
10:39:11.597, 2, 8, 15.05 W, 1230 MHz, 3 %
10:39:12.189, 2, 8, 14.88 W, 1245 MHz, 3 %
10:39:12.773, 2, 8, 14.87 W, 1245 MHz, 3 %
10:39:13.358, 1, 8, 8.23 W, 1245 MHz, 7 %
10:39:13.943, 1, 8, 8.12 W, 1245 MHz, 5 %
10:39:14.530, 1, 8, 7.98 W, 1245 MHz, 5 %
10:39:15.120, 1, 8, 7.91 W, 1245 MHz, 5 %
10:39:15.701, 1, 8, 8.02 W, 1350 MHz, 5 %
10:39:16.289, 1, 8, 7.94 W, 1350 MHz, 5 %
10:39:16.888, 1, 8, 8.06 W, 1350 MHz, 5 %
10:39:17.475, 1, 8, 7.95 W, 1350 MHz, 5 %
10:39:18.066, 1, 8, 7.86 W, 1350 MHz, 5 %
10:39:18.652, 1, 8, 7.84 W, 1350 MHz, 5 %
10:39:19.236, 1, 8, 7.79 W, 1012 MHz, 5 %
10:39:19.825, 1, 8, 7.99 W, 1012 MHz, 5 %
10:39:20.410, 1, 8, 7.75 W, 1012 MHz, 4 %
10:39:20.996, 1, 8, 7.80 W, 1012 MHz, 4 %
10:39:21.584, 1, 8, 7.90 W, 1012 MHz, 4 %
10:39:22.174, 1, 8, 7.95 W, 1012 MHz, 5 %
10:39:22.759, 1, 8, 7.93 W, 1245 MHz, 6 %
10:39:23.361, 1, 8, 7.99 W, 1245 MHz, 4 %
10:39:23.948, 1, 8, 7.85 W, 1245 MHz, 4 %
10:39:24.534, 1, 8, 7.74 W, 1245 MHz, 5 %
10:39:25.124, 1, 8, 7.70 W, 1245 MHz, 5 %
10:39:25.707, 1, 8, 7.79 W, 1275 MHz, 5 %
10:39:26.294, 1, 8, 7.77 W, 1275 MHz, 5 %
10:39:26.884, 2, 8, 7.85 W, 1275 MHz, 5 %
10:39:27.466, 1, 8, 8.66 W, 1275 MHz, 4 %
10:39:28.045, 1, 8, 8.45 W, 1275 MHz, 4 %
10:39:28.627, 1, 8, 8.11 W, 1275 MHz, 4 %
10:39:29.212, 1, 8, 8.19 W, 1350 MHz, 5 %
10:39:29.797, 1, 8, 8.06 W, 1350 MHz, 4 %
10:39:30.382, 1, 8, 8.08 W, 1350 MHz, 4 %
10:39:30.963, 1, 8, 7.88 W, 1350 MHz, 4 %
10:39:31.550, 1, 8, 8.03 W, 1350 MHz, 5 %
10:39:32.131, 1, 8, 7.98 W, 1350 MHz, 4 %
10:39:32.722, 1, 8, 7.85 W, 1260 MHz, 4 %
10:39:33.303, 2, 8, 15.11 W, 1260 MHz, 3 %
10:39:33.886, 2, 8, 14.75 W, 1260 MHz, 3 %
10:39:34.472, 2, 8, 14.86 W, 1260 MHz, 3 %
10:39:35.061, 2, 8, 15.23 W, 1260 MHz, 3 %
10:39:35.646, 2, 8, 15.25 W, 1260 MHz, 3 %
10:39:36.230, 2, 8, 15.39 W, 1267 MHz, 3 %
10:39:36.820, 2, 8, 14.94 W, 1267 MHz, 3 %
10:39:37.406, 2, 8, 15.00 W, 1267 MHz, 3 %
10:39:37.985, 2, 8, 15.17 W, 1267 MHz, 3 %
10:39:38.566, 2, 8, 16.23 W, 1267 MHz, 3 %
10:39:39.146, 2, 8, 15.52 W, 1267 MHz, 3 %
10:39:39.734, 2, 8, 15.34 W, 1305 MHz, 3 %
10:39:40.316, 2, 8, 15.37 W, 1305 MHz, 3 %
10:39:40.900, 2, 8, 14.80 W, 1305 MHz, 3 %
10:39:41.483, 2, 8, 16.26 W, 1305 MHz, 3 %
10:39:42.067, 2, 8, 15.07 W, 1305 MHz, 3 %
10:39:42.642, 2, 8, 15.48 W, 1305 MHz, 3 %
10:39:43.227, 2, 8, 15.24 W, 1327 MHz, 3 %
10:39:43.806, 2, 8, 15.21 W, 1327 MHz, 3 %
10:39:44.391, 2, 8, 15.11 W, 1327 MHz, 3 %
10:39:44.970, 2, 8, 15.36 W, 1327 MHz, 3 %
10:39:45.552, 2, 8, 15.50 W, 1327 MHz, 3 %
10:39:46.131, 2, 8, 15.10 W, 1327 MHz, 3 %
10:39:46.715, 2, 8, 14.81 W, 1282 MHz, 3 %
10:39:47.295, 2, 8, 16.02 W, 1282 MHz, 3 %
10:39:47.875, 2, 8, 15.31 W, 1282 MHz, 3 %
10:39:48.455, 2, 8, 15.32 W, 1282 MHz, 3 %
10:39:49.035, 2, 8, 15.05 W, 1282 MHz, 3 %
10:39:49.613, 2, 8, 15.02 W, 1282 MHz, 3 %
10:39:50.195, 2, 8, 14.86 W, 1282 MHz, 3 %
10:39:50.778, 2, 8, 15.27 W, 1290 MHz, 3 %
10:39:51.360, 2, 8, 14.85 W, 1290 MHz, 3 %
10:39:51.940, 2, 8, 14.87 W, 1290 MHz, 3 %
10:39:52.521, 2, 8, 15.07 W, 1290 MHz, 3 %
10:39:53.101, 2, 8, 15.22 W, 1290 MHz, 3 %
10:39:53.679, 2, 8, 15.25 W, 1290 MHz, 3 %
10:39:54.263, 2, 8, 14.71 W, 1237 MHz, 3 %
10:39:54.847, 2, 8, 14.70 W, 1237 MHz, 3 %
10:39:55.428, 2, 8, 14.97 W, 1237 MHz, 3 %
10:39:56.015, 2, 8, 15.26 W, 1237 MHz, 3 %
10:39:56.603, 2, 8, 14.85 W, 1237 MHz, 3 %
10:39:57.183, 2, 8, 14.85 W, 1237 MHz, 3 %
10:39:57.765, 2, 8, 14.88 W, 1290 MHz, 3 %
10:39:58.352, 2, 8, 14.66 W, 1290 MHz, 3 %
10:39:58.934, 2, 8, 14.77 W, 1290 MHz, 3 %
10:39:59.516, 2, 8, 14.99 W, 1290 MHz, 3 %
10:40:00.093, 2, 8, 15.14 W, 1290 MHz, 3 %
10:40:00.679, 2, 8, 15.16 W, 1290 MHz, 3 %
10:40:01.260, 2, 8, 15.66 W, 1327 MHz, 3 %
10:40:01.846, 2, 8, 15.31 W, 1327 MHz, 3 %
10:40:02.427, 2, 8, 16.74 W, 1327 MHz, 3 %
10:40:03.004, 2, 8, 15.09 W, 1327 MHz, 3 %
10:40:03.582, 2, 8, 14.79 W, 1327 MHz, 3 %
10:40:04.168, 2, 8, 15.15 W, 1327 MHz, 3 %
10:40:04.757, 2, 8, 15.31 W, 1207 MHz, 3 %
10:40:05.346, 2, 8, 15.14 W, 1207 MHz, 3 %
10:40:05.929, 2, 8, 15.37 W, 1207 MHz, 3 %
10:40:06.510, 2, 8, 15.42 W, 1207 MHz, 3 %
10:40:07.086, 2, 8, 15.21 W, 1207 MHz, 3 %
10:40:07.668, 2, 8, 15.36 W, 1207 MHz, 3 %
10:40:08.250, 2, 8, 16.17 W, 1237 MHz, 3 %
10:40:08.830, 2, 8, 14.79 W, 1237 MHz, 3 %
10:40:09.407, 2, 8, 14.80 W, 1237 MHz, 3 %
10:40:09.992, 2, 8, 15.17 W, 1237 MHz, 3 %
10:40:10.577, 2, 8, 15.22 W, 1237 MHz, 3 %
10:40:11.158, 2, 8, 15.36 W, 1237 MHz, 3 %
10:40:11.738, 2, 8, 15.32 W, 1305 MHz, 3 %
10:40:12.321, 2, 8, 15.22 W, 1305 MHz, 3 %
10:40:12.901, 2, 8, 15.19 W, 1305 MHz, 3 %
10:40:13.477, 2, 8, 15.25 W, 1305 MHz, 3 %
10:40:14.058, 2, 8, 15.13 W, 1305 MHz, 3 %
10:40:14.641, 2, 8, 15.06 W, 1305 MHz, 3 %
10:40:15.219, 2, 8, 15.13 W, 1245 MHz, 3 %
10:40:15.805, 2, 8, 15.09 W, 1245 MHz, 3 %
10:40:16.389, 2, 8, 15.27 W, 1245 MHz, 3 %
10:40:16.968, 2, 8, 15.23 W, 1245 MHz, 3 %
10:40:17.553, 2, 8, 15.16 W, 1245 MHz, 3 %
10:40:18.137, 2, 8, 15.22 W, 1245 MHz, 3 %
10:40:18.718, 2, 8, 15.77 W, 1305 MHz, 3 %
10:40:19.294, 2, 8, 15.24 W, 1305 MHz, 3 %
10:40:19.879, 2, 8, 15.16 W, 1305 MHz, 3 %
10:40:20.461, 2, 8, 14.91 W, 1305 MHz, 3 %
10:40:21.047, 2, 8, 14.83 W, 1305 MHz, 3 %
10:40:21.632, 2, 8, 14.87 W, 1305 MHz, 3 %
10:40:22.215, 2, 8, 15.15 W, 1177 MHz, 3 %
10:40:22.803, 2, 8, 15.16 W, 1177 MHz, 3 %
10:40:23.383, 2, 8, 15.16 W, 1177 MHz, 3 %
10:40:23.968, 2, 8, 15.10 W, 1177 MHz, 3 %
10:40:24.560, 2, 8, 15.18 W, 1177 MHz, 3 %
10:40:25.143, 2, 8, 15.08 W, 1177 MHz, 3 %
10:40:25.729, 2, 8, 16.17 W, 1305 MHz, 7 %
10:40:26.313, 2, 8, 15.43 W, 1305 MHz, 3 %
10:40:26.900, 2, 8, 14.83 W, 1305 MHz, 3 %
10:40:27.490, 2, 8, 15.30 W, 1305 MHz, 3 %
10:40:28.074, 2, 8, 14.97 W, 1305 MHz, 3 %
10:40:28.661, 2, 8, 15.23 W, 1305 MHz, 3 %
10:40:29.246, 2, 8, 14.85 W, 1320 MHz, 3 %
10:40:29.835, 2, 8, 14.96 W, 1320 MHz, 3 %
10:40:30.421, 4, 8, 16.28 W, 1320 MHz, 2 %
10:40:31.008, 4, 8, 16.53 W, 1320 MHz, 3 %
10:40:31.586, 4, 8, 16.46 W, 1320 MHz, 2 %
10:40:32.171, 4, 8, 16.33 W, 1320 MHz, 2 %
10:40:32.761, 4, 8, 16.22 W, 1965 MHz, 2 %
10:40:33.346, 4, 8, 16.31 W, 1965 MHz, 2 %
10:40:33.932, 4, 8, 16.64 W, 1965 MHz, 2 %
10:40:34.517, 4, 8, 16.23 W, 1965 MHz, 2 %
10:40:35.104, 4, 8, 16.06 W, 1965 MHz, 2 %
10:40:35.686, 4, 8, 16.17 W, 1965 MHz, 2 %
10:40:36.271, 2, 8, 16.22 W, 1950 MHz, 2 %
10:40:36.855, 2, 8, 14.81 W, 1950 MHz, 3 %
10:40:37.445, 2, 8, 15.19 W, 1950 MHz, 3 %
10:40:38.021, 2, 8, 14.95 W, 1950 MHz, 3 %
10:40:38.610, 2, 8, 15.01 W, 1950 MHz, 3 %
10:40:39.195, 2, 8, 15.17 W, 1950 MHz, 3 %
10:40:39.782, 2, 8, 15.00 W, 1320 MHz, 3 %
10:40:40.369, 2, 8, 14.78 W, 1320 MHz, 3 %
10:40:40.953, 2, 8, 14.81 W, 1320 MHz, 3 %
10:40:41.530, 2, 8, 14.34 W, 1320 MHz, 3 %
10:40:42.116, 2, 8, 14.71 W, 1320 MHz, 3 %
10:40:42.699, 2, 8, 14.99 W, 1320 MHz, 3 %
10:40:43.286, 2, 8, 14.82 W, 1327 MHz, 3 %
10:40:43.875, 2, 8, 14.84 W, 1327 MHz, 3 %
10:40:44.461, 2, 8, 15.62 W, 1327 MHz, 3 %
10:40:45.047, 2, 8, 14.51 W, 1327 MHz, 3 %
10:40:45.632, 2, 8, 14.65 W, 1327 MHz, 3 %
10:40:46.222, 2, 8, 14.59 W, 1267 MHz, 3 %
10:40:46.808, 2, 8, 14.72 W, 1267 MHz, 3 %
10:40:47.397, 2, 8, 14.90 W, 1267 MHz, 3 %
10:40:47.984, 2, 8, 14.76 W, 1267 MHz, 3 %
10:40:48.571, 2, 8, 14.24 W, 1267 MHz, 3 %
10:40:49.155, 2, 8, 14.11 W, 1267 MHz, 3 %
10:40:49.742, 2, 8, 14.39 W, 1252 MHz, 3 %
10:40:50.330, 2, 8, 14.76 W, 1252 MHz, 3 %
10:40:50.916, 2, 8, 14.45 W, 1252 MHz, 3 %
10:40:51.503, 2, 8, 14.41 W, 1252 MHz, 3 %
10:40:52.084, 2, 8, 14.40 W, 1252 MHz, 3 %
10:40:52.672, 2, 8, 14.33 W, 1252 MHz, 3 %
10:40:53.261, 2, 8, 14.99 W, 1252 MHz, 3 %
10:40:53.849, 2, 8, 14.83 W, 1252 MHz, 3 %
10:40:54.435, 2, 8, 14.65 W, 1252 MHz, 3 %
10:40:55.022, 2, 8, 14.52 W, 1252 MHz, 3 %
10:40:55.606, 2, 8, 14.84 W, 1252 MHz, 3 %
10:40:56.192, 2, 8, 14.76 W, 1252 MHz, 3 %
10:40:56.774, 2, 8, 14.73 W, 1267 MHz, 3 %
10:40:57.361, 2, 8, 15.12 W, 1267 MHz, 3 %
10:40:57.939, 4, 8, 15.49 W, 1267 MHz, 4 %
10:40:58.528, 4, 8, 36.01 W, 1267 MHz, 100 %
10:40:59.118, 3, 8, 35.95 W, 1267 MHz, 99 %
10:40:59.709, 3, 8, 13.33 W, 1267 MHz, 99 %
10:41:00.296, 4, 8, 35.64 W, 2527 MHz, 100 %
10:41:00.881, 4, 8, 35.70 W, 2527 MHz, 100 %
10:41:01.469, 4, 8, 35.72 W, 2527 MHz, 100 %
10:41:02.057, 4, 8, 35.83 W, 2527 MHz, 100 %
10:41:02.646, 4, 8, 35.82 W, 2527 MHz, 99 %
10:41:03.231, 4, 8, 26.95 W, 2820 MHz, 3 %
10:41:03.822, 4, 8, 14.98 W, 2820 MHz, 4 %
10:41:04.411, 4, 8, 16.06 W, 2820 MHz, 4 %
10:41:04.998, 4, 8, 18.02 W, 2820 MHz, 21 %
10:41:05.582, 4, 8, 15.33 W, 2820 MHz, 4 %
10:41:06.166, 2, 8, 15.29 W, 2820 MHz, 3 %
10:41:06.757, 2, 8, 14.76 W, 1425 MHz, 3 %
10:41:07.343, 2, 8, 14.75 W, 1425 MHz, 3 %
10:41:07.930, 2, 8, 14.85 W, 1425 MHz, 3 %
10:41:08.507, 2, 8, 14.44 W, 1425 MHz, 3 %
10:41:09.092, 2, 8, 14.75 W, 1425 MHz, 3 %
10:41:09.678, 2, 8, 14.80 W, 1425 MHz, 3 %
10:41:10.262, 4, 8, 35.66 W, 1725 MHz, 99 %
10:41:10.847, 4, 8, 15.79 W, 1725 MHz, 3 %
10:41:11.467, 4, 8, 19.44 W, 1725 MHz, 3 %
10:41:12.054, 4, 8, 17.94 W, 1725 MHz, 34 %
10:41:12.641, 4, 8, 19.59 W, 1725 MHz, 35 %
10:41:13.231, 4, 8, 15.12 W, 1267 MHz, 3 %
10:41:13.816, 4, 8, 15.32 W, 1267 MHz, 3 %
10:41:14.396, 4, 8, 14.50 W, 1267 MHz, 3 %
10:41:14.994, 4, 8, 15.24 W, 1267 MHz, 3 %
10:41:15.580, 4, 8, 16.45 W, 1267 MHz, 3 %
10:41:16.165, 4, 8, 16.50 W, 1267 MHz, 3 %
10:41:16.752, 4, 8, 16.39 W, 1492 MHz, 3 %
10:41:17.340, 4, 8, 16.32 W, 1492 MHz, 3 %
10:41:17.926, 2, 8, 15.34 W, 1492 MHz, 3 %
10:41:18.512, 2, 8, 23.47 W, 1492 MHz, 9 %
10:41:19.102, 2, 8, 14.56 W, 1492 MHz, 3 %
10:41:19.705, 2, 8, 14.88 W, 1492 MHz, 3 %
10:41:20.290, 2, 8, 14.60 W, 1275 MHz, 3 %
10:41:20.875, 2, 8, 14.90 W, 1275 MHz, 3 %
10:41:21.461, 2, 8, 14.60 W, 1275 MHz, 3 %
10:41:22.043, 2, 8, 14.56 W, 1275 MHz, 3 %
10:41:22.631, 2, 8, 14.40 W, 1275 MHz, 3 %
10:41:23.215, 2, 8, 14.28 W, 1275 MHz, 3 %
10:41:23.803, 2, 8, 14.60 W, 1222 MHz, 3 %
10:41:24.391, 2, 8, 14.70 W, 1222 MHz, 3 %
10:41:24.980, 2, 8, 14.76 W, 1222 MHz, 3 %
10:41:25.567, 2, 8, 15.11 W, 1222 MHz, 3 %
10:41:26.142, 4, 8, 20.84 W, 1222 MHz, 9 %
10:41:26.730, 4, 8, 19.71 W, 1680 MHz, 3 %
10:41:27.317, 4, 8, 20.85 W, 1680 MHz, 43 %
10:41:27.908, 4, 8, 20.66 W, 1680 MHz, 19 %
10:41:28.494, 4, 8, 20.07 W, 1680 MHz, 2 %
10:41:29.083, 4, 8, 19.50 W, 1680 MHz, 7 %
10:41:29.671, 4, 8, 21.20 W, 1680 MHz, 44 %
10:41:30.248, 4, 8, 17.97 W, 1020 MHz, 2 %
10:41:30.837, 4, 8, 14.88 W, 1020 MHz, 3 %
10:41:31.412, 4, 8, 15.01 W, 1020 MHz, 3 %
10:41:31.998, 4, 8, 16.09 W, 1020 MHz, 3 %
10:41:32.586, 4, 8, 16.42 W, 1020 MHz, 3 %
10:41:33.174, 2, 8, 16.10 W, 1020 MHz, 3 %
10:41:33.760, 2, 8, 10.03 W, 1207 MHz, 4 %
10:41:34.344, 2, 8, 13.46 W, 1207 MHz, 3 %
10:41:34.926, 2, 8, 14.21 W, 1207 MHz, 3 %
10:41:35.502, 2, 8, 15.13 W, 1207 MHz, 3 %
10:41:36.089, 2, 8, 15.31 W, 1207 MHz, 3 %
10:41:36.673, 2, 8, 15.08 W, 1207 MHz, 3 %
10:41:37.265, 2, 8, 14.77 W, 1230 MHz, 3 %
10:41:37.847, 2, 8, 14.62 W, 1230 MHz, 3 %
10:41:38.432, 2, 8, 15.16 W, 1230 MHz, 3 %
10:41:39.019, 2, 8, 14.57 W, 1230 MHz, 3 %
10:41:39.602, 2, 8, 14.55 W, 1230 MHz, 3 %
10:41:40.192, 2, 8, 14.56 W, 1230 MHz, 3 %
10:41:40.775, 2, 8, 14.59 W, 1200 MHz, 3 %
10:41:41.363, 2, 8, 14.64 W, 1200 MHz, 3 %
10:41:41.954, 2, 8, 14.52 W, 1200 MHz, 3 %
10:41:42.536, 2, 8, 14.55 W, 1200 MHz, 3 %
10:41:43.124, 2, 8, 14.47 W, 1200 MHz, 3 %
10:41:43.698, 2, 8, 14.50 W, 1200 MHz, 3 %
10:41:44.283, 2, 8, 15.11 W, 1215 MHz, 3 %
10:41:44.870, 2, 8, 14.48 W, 1215 MHz, 3 %
10:41:45.456, 2, 8, 14.72 W, 1215 MHz, 3 %
10:41:46.042, 2, 8, 11.90 W, 1215 MHz, 3 %
10:41:46.626, 2, 8, 13.92 W, 1215 MHz, 3 %
10:41:47.211, 2, 8, 14.83 W, 1215 MHz, 3 %
10:41:47.796, 2, 8, 15.03 W, 1132 MHz, 3 %
10:41:48.380, 2, 8, 14.75 W, 1132 MHz, 3 %
10:41:48.962, 2, 8, 14.55 W, 1132 MHz, 3 %
10:41:49.544, 2, 8, 14.69 W, 1132 MHz, 3 %
10:41:50.129, 2, 8, 14.69 W, 1132 MHz, 3 %
10:41:50.716, 2, 8, 14.78 W, 1132 MHz, 3 %
10:41:51.299, 2, 8, 15.12 W, 1222 MHz, 3 %
10:41:51.883, 4, 8, 15.59 W, 1222 MHz, 13 %
10:41:52.472, 4, 8, 20.91 W, 1222 MHz, 3 %
10:41:53.058, 4, 8, 14.45 W, 1222 MHz, 3 %
10:41:53.644, 4, 8, 14.94 W, 1222 MHz, 2 %
10:41:54.231, 4, 8, 16.25 W, 1447 MHz, 3 %
10:41:54.816, 4, 8, 16.37 W, 1447 MHz, 3 %
10:41:55.399, 4, 8, 16.15 W, 1447 MHz, 3 %
10:41:55.982, 4, 8, 16.17 W, 1447 MHz, 3 %
10:41:56.572, 4, 8, 15.05 W, 1447 MHz, 3 %
10:41:57.157, 4, 8, 15.72 W, 1447 MHz, 6 %
10:41:57.733, 4, 8, 30.29 W, 2317 MHz, 86 %
10:41:58.320, 3, 8, 26.47 W, 2317 MHz, 64 %
10:41:58.907, 3, 8, 16.51 W, 2317 MHz, 82 %
10:41:59.496, 3, 8, 14.35 W, 2317 MHz, 99 %
10:42:00.084, 4, 8, 12.89 W, 2317 MHz, 50 %
10:42:00.670, 4, 8, 29.73 W, 2317 MHz, 80 %
10:42:01.256, 4, 8, 39.98 W, 2325 MHz, 100 %
10:42:01.842, 4, 8, 31.82 W, 2325 MHz, 100 %
10:42:02.431, 4, 8, 29.50 W, 2325 MHz, 99 %
10:42:03.019, 4, 8, 39.97 W, 2325 MHz, 79 %
10:42:03.610, 4, 8, 31.56 W, 2325 MHz, 74 %
10:42:04.199, 4, 8, 31.24 W, 2325 MHz, 41 %
10:42:04.786, 4, 8, 23.87 W, 2295 MHz, 41 %
10:42:05.376, 4, 8, 30.57 W, 2295 MHz, 32 %
10:42:05.964, 4, 8, 31.17 W, 2295 MHz, 33 %
10:42:06.548, 4, 8, 13.59 W, 2295 MHz, 4 %
10:42:07.134, 4, 8, 13.38 W, 2295 MHz, 4 %
10:42:07.718, 4, 8, 13.67 W, 2295 MHz, 3 %
10:42:08.303, 4, 8, 13.80 W, 997 MHz, 3 %
10:42:08.889, 4, 8, 13.60 W, 997 MHz, 3 %
10:42:09.470, 4, 8, 13.69 W, 997 MHz, 3 %
10:42:10.055, 4, 8, 13.73 W, 997 MHz, 3 %
10:42:10.639, 4, 8, 13.79 W, 997 MHz, 3 %
10:42:11.226, 2, 8, 13.84 W, 997 MHz, 3 %
10:42:11.811, 2, 8, 15.05 W, 1035 MHz, 3 %
10:42:12.397, 2, 8, 15.89 W, 1035 MHz, 3 %
10:42:12.983, 2, 8, 15.60 W, 1035 MHz, 3 %
10:42:13.569, 2, 8, 15.10 W, 1035 MHz, 3 %
10:42:14.158, 2, 8, 14.70 W, 1035 MHz, 3 %
10:42:14.743, 2, 8, 14.73 W, 1275 MHz, 3 %
10:42:15.333, 2, 8, 14.52 W, 1275 MHz, 3 %
10:42:15.918, 2, 8, 14.53 W, 1275 MHz, 3 %
10:42:16.505, 2, 8, 14.62 W, 1275 MHz, 3 %
10:42:17.090, 2, 8, 14.31 W, 1275 MHz, 3 %
10:42:17.675, 2, 8, 14.48 W, 1275 MHz, 3 %
10:42:18.265, 2, 8, 14.63 W, 1237 MHz, 3 %
10:42:18.855, 2, 8, 14.59 W, 1237 MHz, 3 %
10:42:19.442, 2, 8, 15.05 W, 1237 MHz, 3 %
10:42:20.025, 2, 8, 14.55 W, 1237 MHz, 3 %
10:42:20.600, 2, 8, 14.64 W, 1237 MHz, 3 %
10:42:21.186, 2, 8, 14.68 W, 1237 MHz, 3 %
10:42:21.775, 2, 8, 14.63 W, 1282 MHz, 3 %
10:42:22.360, 2, 8, 14.63 W, 1282 MHz, 3 %
10:42:22.944, 2, 8, 14.65 W, 1282 MHz, 3 %
10:42:23.525, 2, 8, 14.66 W, 1282 MHz, 3 %
10:42:24.109, 2, 8, 14.65 W, 1282 MHz, 3 %
10:42:24.696, 2, 8, 14.62 W, 1282 MHz, 3 %
10:42:25.280, 2, 8, 14.70 W, 1282 MHz, 3 %
10:42:25.864, 2, 8, 14.65 W, 1282 MHz, 3 %
10:42:26.441, 2, 8, 15.92 W, 1282 MHz, 3 %
10:42:27.020, 2, 8, 15.23 W, 1282 MHz, 3 %
10:42:27.605, 2, 8, 16.68 W, 1282 MHz, 3 %
10:42:28.185, 2, 8, 15.78 W, 1282 MHz, 3 %
10:42:28.764, 2, 8, 14.81 W, 1297 MHz, 3 %
10:42:29.349, 2, 8, 14.80 W, 1297 MHz, 3 %
10:42:29.929, 2, 8, 14.24 W, 1297 MHz, 3 %
10:42:30.518, 1, 8, 11.72 W, 1297 MHz, 4 %
10:42:31.102, 1, 8, 7.97 W, 1297 MHz, 5 %
10:42:31.686, 1, 8, 7.92 W, 1297 MHz, 5 %
10:42:32.267, 1, 8, 8.03 W, 1305 MHz, 5 %
10:42:32.852, 1, 8, 7.95 W, 1305 MHz, 5 %
10:42:33.433, 1, 8, 8.06 W, 1305 MHz, 4 %
10:42:34.020, 1, 8, 7.97 W, 1305 MHz, 4 %
10:42:34.606, 1, 8, 7.97 W, 1305 MHz, 4 %
10:42:35.190, 1, 8, 7.87 W, 1305 MHz, 5 %
10:42:35.773, 1, 8, 7.97 W, 1267 MHz, 5 %
10:42:36.354, 2, 8, 12.05 W, 1267 MHz, 3 %
10:42:36.933, 2, 8, 14.75 W, 1267 MHz, 3 %
10:42:37.515, 2, 8, 15.40 W, 1267 MHz, 3 %
10:42:38.097, 2, 8, 15.08 W, 1267 MHz, 3 %
10:42:38.681, 2, 8, 15.33 W, 1267 MHz, 3 %
10:42:39.264, 2, 8, 15.05 W, 1207 MHz, 3 %
10:42:39.846, 2, 8, 14.99 W, 1207 MHz, 3 %
10:42:40.430, 2, 8, 15.13 W, 1207 MHz, 3 %
10:42:41.012, 2, 8, 15.18 W, 1207 MHz, 3 %
10:42:41.596, 2, 8, 15.73 W, 1207 MHz, 3 %
10:42:42.179, 2, 8, 15.11 W, 1207 MHz, 3 %
10:42:42.765, 2, 8, 17.82 W, 1305 MHz, 3 %
10:42:43.343, 2, 8, 15.27 W, 1305 MHz, 3 %
10:42:43.929, 2, 8, 15.18 W, 1305 MHz, 3 %
10:42:44.509, 2, 8, 15.35 W, 1305 MHz, 3 %
10:42:45.088, 2, 8, 15.43 W, 1305 MHz, 3 %
10:42:45.674, 2, 8, 15.41 W, 1305 MHz, 3 %
10:42:46.271, 2, 8, 15.12 W, 1230 MHz, 3 %
10:42:46.852, 2, 8, 15.39 W, 1230 MHz, 3 %
10:42:47.439, 2, 8, 15.28 W, 1230 MHz, 3 %
10:42:48.025, 2, 8, 15.15 W, 1230 MHz, 3 %
10:42:48.607, 2, 8, 14.86 W, 1230 MHz, 3 %
10:42:49.190, 2, 8, 15.21 W, 1230 MHz, 3 %
10:42:49.773, 2, 8, 15.73 W, 1230 MHz, 3 %
10:42:50.356, 2, 8, 15.05 W, 1230 MHz, 3 %
10:42:50.936, 2, 8, 15.18 W, 1230 MHz, 3 %
10:42:51.522, 2, 8, 16.10 W, 1230 MHz, 3 %
10:42:52.105, 2, 8, 15.35 W, 1230 MHz, 3 %
10:42:52.687, 2, 8, 15.15 W, 1230 MHz, 3 %
10:42:53.267, 2, 8, 15.37 W, 1260 MHz, 3 %
10:42:53.846, 2, 8, 15.07 W, 1260 MHz, 3 %
10:42:54.431, 2, 8, 15.20 W, 1260 MHz, 3 %
10:42:55.015, 2, 8, 15.09 W, 1260 MHz, 3 %
10:42:55.596, 2, 8, 15.12 W, 1260 MHz, 3 %
10:42:56.176, 2, 8, 15.11 W, 1260 MHz, 3 %
10:42:56.764, 2, 8, 15.11 W, 1230 MHz, 3 %
10:42:57.342, 2, 8, 15.86 W, 1230 MHz, 3 %
10:42:57.922, 2, 8, 14.93 W, 1230 MHz, 3 %
10:42:58.509, 2, 8, 14.90 W, 1230 MHz, 3 %
10:42:59.092, 2, 8, 14.21 W, 1230 MHz, 3 %
10:42:59.668, 2, 8, 15.01 W, 1230 MHz, 3 %
10:43:00.249, 2, 8, 15.09 W, 1200 MHz, 3 %
10:43:00.834, 2, 8, 15.06 W, 1200 MHz, 3 %
10:43:01.416, 2, 8, 15.03 W, 1200 MHz, 3 %
10:43:01.996, 2, 8, 14.77 W, 1200 MHz, 3 %
10:43:02.577, 2, 8, 15.08 W, 1200 MHz, 3 %
10:43:03.155, 2, 8, 15.19 W, 1200 MHz, 3 %
10:43:03.743, 2, 8, 15.22 W, 1200 MHz, 3 %
10:43:04.330, 2, 8, 15.17 W, 1252 MHz, 3 %
10:43:04.913, 2, 8, 14.49 W, 1252 MHz, 3 %
10:43:05.498, 2, 8, 14.85 W, 1252 MHz, 3 %
10:43:06.081, 2, 8, 14.63 W, 1252 MHz, 3 %
10:43:06.660, 2, 8, 14.76 W, 1252 MHz, 3 %
10:43:07.247, 2, 8, 14.91 W, 1252 MHz, 3 %
10:43:07.830, 2, 8, 14.91 W, 1275 MHz, 3 %
10:43:08.417, 2, 8, 15.52 W, 1275 MHz, 3 %
10:43:09.005, 2, 8, 15.15 W, 1275 MHz, 3 %
10:43:09.583, 2, 8, 15.51 W, 1275 MHz, 3 %
10:43:10.165, 2, 8, 15.12 W, 1275 MHz, 3 %
10:43:10.751, 2, 8, 14.82 W, 1275 MHz, 3 %
10:43:11.329, 2, 8, 15.19 W, 1275 MHz, 3 %
10:43:11.909, 2, 8, 15.17 W, 1275 MHz, 3 %
10:43:12.494, 2, 8, 15.15 W, 1275 MHz, 3 %
10:43:13.080, 2, 8, 15.21 W, 1275 MHz, 3 %
10:43:13.660, 2, 8, 15.21 W, 1275 MHz, 3 %
10:43:14.245, 2, 8, 14.86 W, 1282 MHz, 3 %
10:43:14.842, 2, 8, 14.83 W, 1282 MHz, 3 %
10:43:15.423, 2, 8, 14.97 W, 1282 MHz, 3 %
10:43:16.006, 2, 8, 14.79 W, 1282 MHz, 3 %
10:43:16.593, 2, 8, 15.10 W, 1282 MHz, 3 %
10:43:17.171, 2, 8, 14.96 W, 1282 MHz, 3 %
10:43:17.767, 2, 8, 14.99 W, 1252 MHz, 3 %
10:43:18.344, 2, 8, 15.26 W, 1252 MHz, 3 %
10:43:18.929, 2, 8, 14.98 W, 1252 MHz, 3 %
10:43:19.514, 2, 8, 15.06 W, 1252 MHz, 3 %
10:43:20.095, 2, 8, 15.21 W, 1252 MHz, 3 %
10:43:20.677, 2, 8, 15.06 W, 1252 MHz, 3 %
10:43:21.260, 2, 8, 14.75 W, 1290 MHz, 3 %
10:43:21.842, 2, 8, 14.68 W, 1290 MHz, 3 %
10:43:22.422, 2, 8, 14.73 W, 1290 MHz, 3 %
10:43:23.004, 2, 8, 14.83 W, 1290 MHz, 3 %
10:43:23.589, 2, 8, 14.85 W, 1290 MHz, 3 %
10:43:24.176, 2, 8, 14.81 W, 1290 MHz, 3 %
10:43:24.762, 2, 8, 14.76 W, 1312 MHz, 3 %
10:43:25.344, 2, 8, 14.76 W, 1312 MHz, 3 %
10:43:25.924, 2, 8, 14.72 W, 1312 MHz, 3 %
10:43:26.506, 2, 8, 15.48 W, 1312 MHz, 3 %
10:43:27.093, 2, 8, 15.33 W, 1312 MHz, 3 %
10:43:27.682, 2, 8, 16.06 W, 1312 MHz, 3 %
10:43:28.271, 2, 8, 14.76 W, 1185 MHz, 3 %
10:43:28.860, 2, 8, 14.89 W, 1185 MHz, 3 %
10:43:29.444, 2, 8, 14.58 W, 1185 MHz, 3 %
10:43:30.032, 2, 8, 14.76 W, 1185 MHz, 3 %
10:43:30.617, 2, 8, 14.90 W, 1185 MHz, 3 %
10:43:31.202, 4, 8, 15.01 W, 1185 MHz, 3 %
10:43:31.788, 4, 8, 15.91 W, 1980 MHz, 2 %
10:43:32.375, 4, 8, 16.30 W, 1980 MHz, 2 %
10:43:32.958, 4, 8, 16.06 W, 1980 MHz, 2 %
10:43:33.545, 4, 8, 16.20 W, 1980 MHz, 2 %
10:43:34.132, 4, 8, 16.06 W, 1980 MHz, 2 %
10:43:34.717, 4, 8, 17.19 W, 1980 MHz, 2 %
10:43:35.309, 4, 8, 16.09 W, 1950 MHz, 2 %
10:43:35.895, 4, 8, 16.15 W, 1950 MHz, 2 %
10:43:36.481, 4, 8, 16.28 W, 1950 MHz, 2 %
10:43:37.067, 4, 8, 16.16 W, 1950 MHz, 2 %
10:43:37.644, 4, 8, 16.13 W, 1950 MHz, 2 %
10:43:38.228, 4, 8, 14.79 W, 1950 MHz, 3 %
10:43:38.815, 4, 8, 14.85 W, 1035 MHz, 3 %
10:43:39.403, 2, 8, 15.44 W, 1035 MHz, 3 %
10:43:39.987, 2, 8, 15.41 W, 1035 MHz, 3 %
10:43:40.577, 2, 8, 14.70 W, 1035 MHz, 3 %
10:43:41.165, 2, 8, 14.63 W, 1035 MHz, 3 %
10:43:41.752, 2, 8, 14.91 W, 1035 MHz, 3 %
10:43:42.341, 2, 8, 14.51 W, 1237 MHz, 3 %
10:43:42.929, 2, 8, 14.54 W, 1237 MHz, 3 %
10:43:43.515, 2, 8, 14.57 W, 1237 MHz, 3 %
10:43:44.105, 2, 8, 15.47 W, 1237 MHz, 3 %
10:43:44.689, 2, 8, 15.21 W, 1237 MHz, 3 %
10:43:45.278, 2, 8, 14.96 W, 1207 MHz, 3 %
10:43:45.863, 2, 8, 14.99 W, 1207 MHz, 3 %
10:43:46.448, 2, 8, 15.01 W, 1207 MHz, 3 %
10:43:47.037, 2, 8, 14.97 W, 1207 MHz, 3 %
10:43:47.622, 2, 8, 14.72 W, 1207 MHz, 3 %
10:43:48.211, 2, 8, 14.94 W, 1207 MHz, 3 %
10:43:48.801, 2, 8, 15.20 W, 1237 MHz, 3 %
10:43:49.389, 2, 8, 15.41 W, 1237 MHz, 3 %
10:43:49.974, 2, 8, 14.85 W, 1237 MHz, 3 %
10:43:50.563, 2, 8, 16.45 W, 1237 MHz, 3 %
10:43:51.151, 2, 8, 14.86 W, 1237 MHz, 3 %
10:43:51.738, 2, 8, 14.72 W, 1237 MHz, 3 %
10:43:52.325, 2, 8, 14.69 W, 1252 MHz, 3 %
10:43:52.905, 2, 8, 14.50 W, 1252 MHz, 3 %
10:43:53.489, 2, 8, 14.81 W, 1252 MHz, 3 %
10:43:54.072, 2, 8, 14.81 W, 1252 MHz, 3 %
10:43:54.658, 2, 8, 15.27 W, 1252 MHz, 9 %
10:43:55.241, 2, 8, 14.94 W, 1252 MHz, 3 %
10:43:55.828, 2, 8, 14.97 W, 1237 MHz, 3 %
10:43:56.417, 2, 8, 14.85 W, 1237 MHz, 3 %
10:43:57.002, 2, 8, 14.73 W, 1237 MHz, 3 %
10:43:57.589, 2, 8, 15.07 W, 1237 MHz, 3 %
10:43:58.176, 2, 8, 15.37 W, 1237 MHz, 3 %
10:43:58.754, 2, 8, 15.34 W, 1237 MHz, 3 %
10:43:59.338, 2, 8, 14.73 W, 1245 MHz, 3 %
10:43:59.927, 4, 8, 26.95 W, 1245 MHz, 100 %
10:44:00.509, 4, 8, 39.25 W, 1245 MHz, 100 %
10:44:01.094, 4, 8, 35.76 W, 1245 MHz, 100 %
10:44:01.670, 4, 8, 35.81 W, 1245 MHz, 99 %
10:44:02.248, 4, 8, 35.82 W, 1245 MHz, 100 %
10:44:02.841, 4, 8, 35.96 W, 2820 MHz, 100 %
10:44:03.426, 4, 8, 35.99 W, 2820 MHz, 100 %
10:44:04.010, 4, 8, 35.98 W, 2820 MHz, 99 %
10:44:04.593, 4, 8, 15.72 W, 2820 MHz, 2 %
10:44:05.176, 4, 8, 16.22 W, 2820 MHz, 3 %
10:44:05.763, 4, 8, 15.46 W, 1350 MHz, 4 %
10:44:06.352, 2, 8, 14.98 W, 1350 MHz, 3 %
10:44:06.937, 2, 8, 14.68 W, 1350 MHz, 3 %
10:44:07.527, 2, 8, 14.62 W, 1350 MHz, 3 %
10:44:08.114, 2, 8, 16.22 W, 1350 MHz, 3 %
10:44:08.697, 2, 8, 15.93 W, 1350 MHz, 3 %
10:44:09.285, 2, 8, 14.97 W, 1230 MHz, 3 %
10:44:09.871, 4, 8, 29.83 W, 1230 MHz, 18 %
10:44:10.458, 4, 8, 15.31 W, 1230 MHz, 2 %
10:44:11.045, 4, 8, 22.66 W, 1230 MHz, 80 %
10:44:11.636, 4, 8, 22.37 W, 1230 MHz, 36 %
10:44:12.223, 4, 8, 16.86 W, 1230 MHz, 3 %
10:44:12.828, 4, 8, 20.63 W, 1492 MHz, 4 %
10:44:13.416, 4, 8, 32.08 W, 1492 MHz, 99 %
10:44:14.004, 4, 8, 28.28 W, 1492 MHz, 35 %
10:44:14.589, 4, 8, 14.80 W, 1492 MHz, 3 %
10:44:15.177, 4, 8, 14.84 W, 1492 MHz, 4 %
10:44:15.761, 4, 8, 16.34 W, 1492 MHz, 3 %
10:44:16.349, 4, 8, 16.17 W, 1447 MHz, 3 %
10:44:16.934, 2, 8, 16.19 W, 1447 MHz, 3 %
10:44:17.518, 2, 8, 15.95 W, 1447 MHz, 3 %
10:44:18.101, 2, 8, 15.71 W, 1447 MHz, 3 %
10:44:18.686, 2, 8, 15.76 W, 1447 MHz, 8 %
10:44:19.273, 2, 8, 14.45 W, 1200 MHz, 3 %
10:44:19.856, 2, 8, 14.50 W, 1200 MHz, 3 %
10:44:20.442, 2, 8, 14.62 W, 1200 MHz, 3 %
10:44:21.028, 2, 8, 14.58 W, 1200 MHz, 3 %
10:44:21.611, 2, 8, 14.41 W, 1200 MHz, 3 %
10:44:22.196, 1, 8, 14.42 W, 1200 MHz, 3 %
10:44:22.780, 2, 8, 14.65 W, 1192 MHz, 3 %
10:44:23.363, 2, 8, 14.58 W, 1192 MHz, 3 %
10:44:23.948, 2, 8, 14.59 W, 1192 MHz, 3 %
10:44:24.534, 2, 8, 14.57 W, 1192 MHz, 3 %
10:44:25.118, 2, 8, 14.53 W, 1192 MHz, 3 %
10:44:25.709, 4, 8, 13.89 W, 1192 MHz, 3 %
10:44:26.297, 4, 8, 23.07 W, 2250 MHz, 69 %
10:44:26.883, 4, 8, 29.91 W, 2250 MHz, 50 %
10:44:27.468, 4, 8, 30.01 W, 2250 MHz, 60 %
10:44:28.050, 4, 8, 15.44 W, 2250 MHz, 3 %
10:44:28.640, 4, 8, 14.70 W, 2250 MHz, 4 %
10:44:29.230, 4, 8, 14.90 W, 2250 MHz, 3 %
10:44:29.810, 4, 8, 15.35 W, 1260 MHz, 3 %
10:44:30.387, 4, 8, 15.35 W, 1260 MHz, 3 %
10:44:30.971, 4, 8, 14.99 W, 1260 MHz, 3 %
10:44:31.562, 2, 8, 15.17 W, 1260 MHz, 3 %
10:44:32.147, 2, 8, 14.86 W, 1260 MHz, 4 %
10:44:32.735, 2, 8, 15.85 W, 1260 MHz, 3 %
10:44:33.318, 2, 8, 15.97 W, 1455 MHz, 3 %
10:44:33.908, 2, 8, 15.53 W, 1455 MHz, 3 %
10:44:34.499, 1, 8, 12.85 W, 1455 MHz, 4 %
10:44:35.084, 1, 8, 7.78 W, 1455 MHz, 5 %
10:44:35.672, 1, 8, 7.72 W, 1455 MHz, 5 %
10:44:36.260, 1, 8, 8.14 W, 1455 MHz, 5 %
10:44:36.839, 1, 8, 7.69 W, 1170 MHz, 4 %
10:44:37.429, 1, 8, 7.64 W, 1170 MHz, 4 %
10:44:38.021, 1, 8, 7.76 W, 1170 MHz, 5 %
10:44:38.607, 1, 8, 7.56 W, 1170 MHz, 5 %
10:44:39.195, 1, 8, 8.02 W, 1170 MHz, 5 %
10:44:39.782, 1, 8, 7.71 W, 1072 MHz, 5 %
10:44:40.358, 1, 8, 7.75 W, 1072 MHz, 4 %
10:44:40.944, 1, 8, 7.78 W, 1072 MHz, 4 %
10:44:41.532, 1, 8, 7.76 W, 1072 MHz, 4 %
10:44:42.121, 1, 8, 7.71 W, 1072 MHz, 4 %
10:44:42.707, 1, 8, 7.73 W, 1072 MHz, 4 %
10:44:43.298, 1, 8, 7.76 W, 1200 MHz, 4 %
10:44:43.887, 1, 8, 7.75 W, 1200 MHz, 4 %
10:44:44.472, 1, 8, 7.75 W, 1200 MHz, 4 %
10:44:45.058, 1, 8, 7.73 W, 1200 MHz, 4 %
10:44:45.648, 1, 8, 7.72 W, 1200 MHz, 4 %
10:44:46.226, 1, 8, 7.81 W, 1200 MHz, 4 %
10:44:46.815, 1, 8, 7.61 W, 1087 MHz, 4 %
10:44:47.398, 1, 8, 7.61 W, 1087 MHz, 4 %
10:44:47.981, 1, 8, 7.74 W, 1087 MHz, 5 %
10:44:48.567, 1, 8, 9.21 W, 1087 MHz, 5 %
10:44:49.154, 1, 8, 7.82 W, 1087 MHz, 5 %
10:44:49.745, 3, 8, 9.98 W, 1087 MHz, 79 %
10:44:50.320, 3, 8, 11.19 W, 1087 MHz, 4 %
10:44:50.906, 3, 8, 9.88 W, 1087 MHz, 4 %
10:44:51.489, 3, 8, 10.12 W, 1087 MHz, 4 %
10:44:52.074, 3, 8, 10.13 W, 1087 MHz, 4 %
10:44:52.660, 3, 8, 10.10 W, 1087 MHz, 4 %
10:44:53.245, 2, 8, 10.09 W, 1087 MHz, 4 %
10:44:53.829, 2, 8, 9.85 W, 1140 MHz, 4 %
10:44:54.414, 3, 8, 10.39 W, 1140 MHz, 13 %
10:44:55.001, 3, 8, 13.87 W, 1140 MHz, 99 %
10:44:55.587, 4, 8, 14.00 W, 1140 MHz, 97 %
10:44:56.172, 4, 8, 15.64 W, 1140 MHz, 61 %
10:44:56.759, 4, 8, 13.73 W, 1140 MHz, 99 %
10:44:57.349, 4, 8, 13.93 W, 1477 MHz, 98 %
10:44:57.951, 4, 8, 13.14 W, 1477 MHz, 46 %
10:44:58.557, 4, 8, 20.82 W, 1477 MHz, 99 %
10:44:59.174, 4, 8, 35.04 W, 1477 MHz, 100 %
10:44:59.761, 4, 8, 34.18 W, 2820 MHz, 100 %
10:45:00.399, 4, 8, 32.58 W, 2820 MHz, 100 %
10:45:00.988, 4, 8, 33.84 W, 2820 MHz, 100 %
10:45:01.574, 4, 8, 34.58 W, 2820 MHz, 100 %
10:45:02.166, 4, 8, 34.62 W, 2820 MHz, 98 %
10:45:02.753, 4, 8, 32.74 W, 2820 MHz, 17 %
10:45:03.347, 4, 8, 35.00 W, 2820 MHz, 100 %
10:45:03.951, 4, 8, 34.54 W, 2820 MHz, 98 %
10:45:04.540, 4, 8, 32.71 W, 2820 MHz, 99 %
10:45:05.129, 4, 8, 34.88 W, 2820 MHz, 99 %
10:45:05.717, 4, 8, 34.50 W, 2820 MHz, 100 %
10:45:06.306, 4, 8, 28.83 W, 2820 MHz, 17 %
10:45:06.908, 4, 8, 35.41 W, 2812 MHz, 100 %
10:45:07.496, 4, 8, 34.22 W, 2812 MHz, 99 %
10:45:08.084, 4, 8, 34.67 W, 2812 MHz, 100 %
10:45:08.690, 4, 8, 34.74 W, 2812 MHz, 100 %
10:45:09.280, 4, 8, 34.00 W, 2812 MHz, 99 %
10:45:09.884, 4, 8, 33.83 W, 2812 MHz, 99 %
10:45:10.472, 4, 8, 34.62 W, 2812 MHz, 100 %
10:45:11.148, 4, 8, 28.78 W, 2812 MHz, 34 %
10:45:11.781, 4, 8, 32.42 W, 2812 MHz, 100 %
10:45:12.367, 4, 8, 34.42 W, 2812 MHz, 100 %
10:45:12.984, 4, 8, 34.28 W, 2812 MHz, 99 %
10:45:13.567, 4, 8, 33.60 W, 2812 MHz, 100 %
10:45:14.158, 4, 8, 34.20 W, 2812 MHz, 99 %
10:45:14.748, 4, 8, 32.67 W, 2812 MHz, 97 %
10:45:15.353, 4, 8, 34.78 W, 2812 MHz, 100 %
10:45:15.946, 4, 8, 34.59 W, 2812 MHz, 98 %
10:45:16.550, 4, 8, 34.30 W, 2812 MHz, 100 %
10:45:17.132, 4, 8, 32.20 W, 2812 MHz, 99 %
10:45:17.723, 4, 8, 33.17 W, 2812 MHz, 100 %
10:45:18.312, 4, 8, 34.73 W, 2812 MHz, 100 %
10:45:18.915, 4, 8, 34.68 W, 2812 MHz, 99 %
10:45:19.519, 4, 8, 34.87 W, 2812 MHz, 100 %
10:45:20.105, 4, 8, 34.69 W, 2812 MHz, 98 %
10:45:20.708, 4, 8, 34.53 W, 2812 MHz, 100 %
10:45:21.344, 4, 8, 31.84 W, 2812 MHz, 99 %
10:45:21.930, 4, 8, 34.66 W, 2812 MHz, 100 %
10:45:22.505, 4, 8, 34.74 W, 2812 MHz, 99 %
10:45:23.122, 4, 8, 34.83 W, 2812 MHz, 99 %
10:45:23.726, 4, 8, 34.66 W, 2812 MHz, 100 %
10:45:24.315, 4, 8, 34.54 W, 2812 MHz, 99 %
10:45:24.899, 4, 8, 44.99 W, 2812 MHz, 98 %
10:45:25.504, 4, 8, 34.90 W, 2812 MHz, 99 %
10:45:26.105, 4, 8, 32.91 W, 2812 MHz, 86 %
10:45:26.706, 4, 8, 33.01 W, 2812 MHz, 100 %
10:45:27.340, 4, 8, 34.91 W, 2812 MHz, 99 %
10:45:27.925, 4, 8, 35.20 W, 2812 MHz, 99 %
10:45:28.525, 4, 8, 33.39 W, 2812 MHz, 100 %
10:45:29.114, 4, 8, 33.19 W, 2812 MHz, 99 %
10:45:29.744, 4, 8, 34.89 W, 2812 MHz, 100 %
10:45:30.342, 4, 8, 33.02 W, 2812 MHz, 99 %
10:45:31.006, 4, 8, 34.99 W, 2812 MHz, 100 %
10:45:31.595, 4, 8, 34.93 W, 2812 MHz, 100 %
10:45:32.196, 4, 8, 35.04 W, 2812 MHz, 100 %
10:45:32.784, 4, 8, 34.97 W, 2812 MHz, 99 %
10:45:33.373, 4, 8, 25.08 W, 2812 MHz, 54 %
10:45:34.023, 4, 8, 33.68 W, 2812 MHz, 100 %
10:45:34.641, 4, 8, 34.54 W, 2812 MHz, 99 %
10:45:35.240, 4, 8, 34.22 W, 2812 MHz, 99 %
10:45:35.843, 4, 8, 34.28 W, 2812 MHz, 99 %
10:45:36.442, 4, 8, 34.95 W, 2812 MHz, 100 %
10:45:37.062, 4, 8, 34.28 W, 2812 MHz, 100 %
10:45:37.709, 4, 8, 34.54 W, 2812 MHz, 99 %
10:45:38.331, 4, 8, 35.08 W, 2812 MHz, 100 %
10:45:38.930, 4, 8, 35.04 W, 2812 MHz, 100 %
10:45:39.533, 4, 8, 34.99 W, 2812 MHz, 99 %
10:45:40.114, 4, 8, 32.50 W, 2812 MHz, 99 %
10:45:40.701, 4, 8, 33.82 W, 2812 MHz, 100 %
10:45:41.307, 4, 8, 34.74 W, 2812 MHz, 100 %
10:45:41.888, 4, 8, 34.60 W, 2812 MHz, 99 %
10:45:42.491, 4, 8, 34.48 W, 2745 MHz, 99 %
10:45:43.079, 4, 8, 34.98 W, 2745 MHz, 99 %
10:45:43.668, 4, 8, 34.85 W, 2745 MHz, 99 %
10:45:44.256, 4, 8, 34.89 W, 2745 MHz, 100 %
10:45:44.844, 4, 8, 34.92 W, 2745 MHz, 93 %
10:45:45.434, 4, 8, 34.86 W, 2745 MHz, 99 %
10:45:46.019, 4, 8, 35.12 W, 2812 MHz, 99 %
10:45:46.622, 4, 8, 34.67 W, 2812 MHz, 99 %
10:45:47.207, 4, 8, 16.60 W, 2812 MHz, 20 %
10:45:47.793, 2, 8, 15.89 W, 2812 MHz, 3 %
10:45:48.378, 2, 8, 14.98 W, 2812 MHz, 5 %
10:45:48.964, 2, 8, 15.78 W, 1260 MHz, 4 %
10:45:49.553, 2, 8, 15.92 W, 1260 MHz, 4 %
10:45:50.142, 2, 8, 16.46 W, 1260 MHz, 4 %
10:45:50.727, 2, 8, 15.97 W, 1260 MHz, 4 %
10:45:51.302, 2, 8, 16.28 W, 1260 MHz, 4 %
10:45:51.889, 2, 8, 16.05 W, 1260 MHz, 4 %
10:45:52.480, 2, 8, 15.93 W, 1335 MHz, 4 %
10:45:53.068, 2, 8, 15.99 W, 1335 MHz, 4 %
10:45:53.655, 2, 8, 15.85 W, 1335 MHz, 4 %
10:45:54.240, 2, 8, 15.72 W, 1335 MHz, 3 %
10:45:54.829, 2, 8, 15.38 W, 1335 MHz, 3 %
10:45:55.418, 2, 8, 15.23 W, 1335 MHz, 3 %
10:45:56.002, 2, 8, 15.38 W, 1252 MHz, 3 %
10:45:56.585, 2, 8, 15.43 W, 1252 MHz, 3 %
10:45:57.172, 1, 8, 14.98 W, 1252 MHz, 5 %
10:45:57.755, 1, 8, 11.55 W, 1252 MHz, 6 %
10:45:58.339, 1, 8, 8.27 W, 1252 MHz, 4 %
10:45:58.925, 1, 8, 8.23 W, 1252 MHz, 5 %
10:45:59.512, 1, 8, 8.25 W, 1260 MHz, 5 %
10:46:00.103, 1, 8, 8.17 W, 1260 MHz, 5 %
10:46:00.691, 1, 8, 8.16 W, 1260 MHz, 5 %
10:46:01.276, 1, 8, 8.17 W, 1260 MHz, 4 %
10:46:01.856, 1, 8, 8.13 W, 1260 MHz, 4 %
10:46:02.440, 1, 8, 8.06 W, 1260 MHz, 4 %
10:46:03.027, 1, 8, 8.11 W, 1312 MHz, 4 %
10:46:03.615, 1, 8, 8.16 W, 1312 MHz, 4 %
10:46:04.204, 1, 8, 8.15 W, 1312 MHz, 4 %
10:46:04.783, 1, 8, 8.16 W, 1312 MHz, 4 %
10:46:05.374, 1, 8, 8.45 W, 1312 MHz, 4 %
10:46:05.963, 1, 8, 8.05 W, 1312 MHz, 4 %
10:46:06.547, 1, 8, 7.99 W, 1290 MHz, 4 %
10:46:07.130, 1, 8, 8.11 W, 1290 MHz, 4 %
10:46:07.716, 1, 8, 8.46 W, 1290 MHz, 5 %
10:46:08.306, 1, 8, 7.67 W, 1290 MHz, 4 %
10:46:08.892, 1, 8, 8.00 W, 1290 MHz, 4 %
10:46:09.488, 1, 8, 7.94 W, 1387 MHz, 4 %
10:46:10.073, 1, 8, 8.06 W, 1387 MHz, 4 %
10:46:10.657, 1, 8, 8.64 W, 1387 MHz, 4 %
10:46:11.238, 1, 8, 7.92 W, 1387 MHz, 4 %
10:46:11.817, 1, 8, 8.08 W, 1387 MHz, 4 %
10:46:12.397, 1, 8, 7.90 W, 1387 MHz, 5 %
10:46:12.980, 1, 8, 8.09 W, 847 MHz, 5 %
10:46:13.561, 1, 8, 7.76 W, 847 MHz, 5 %
10:46:14.147, 1, 8, 7.71 W, 847 MHz, 5 %
10:46:14.726, 1, 8, 7.78 W, 847 MHz, 5 %
10:46:15.304, 1, 8, 7.95 W, 847 MHz, 5 %
10:46:15.886, 1, 8, 7.89 W, 847 MHz, 5 %
10:46:16.465, 1, 8, 7.82 W, 847 MHz, 5 %
10:46:17.053, 2, 8, 9.89 W, 1117 MHz, 4 %
10:46:17.645, 2, 8, 15.92 W, 1117 MHz, 3 %
10:46:18.225, 2, 8, 16.04 W, 1117 MHz, 3 %
10:46:18.811, 2, 8, 15.93 W, 1117 MHz, 3 %
10:46:19.391, 2, 8, 15.97 W, 1117 MHz, 3 %
10:46:19.978, 2, 8, 16.15 W, 1455 MHz, 3 %
10:46:20.561, 2, 8, 16.20 W, 1455 MHz, 3 %
10:46:21.144, 2, 8, 16.08 W, 1455 MHz, 3 %
10:46:21.722, 2, 8, 16.12 W, 1455 MHz, 3 %
10:46:22.302, 2, 8, 16.13 W, 1455 MHz, 3 %
10:46:22.880, 2, 8, 13.46 W, 1455 MHz, 3 %
10:46:23.462, 2, 8, 16.13 W, 1455 MHz, 3 %
10:46:24.049, 2, 8, 15.91 W, 1425 MHz, 3 %
10:46:24.637, 2, 8, 15.39 W, 1425 MHz, 3 %
10:46:25.220, 2, 8, 16.74 W, 1425 MHz, 3 %
10:46:25.801, 2, 8, 16.09 W, 1425 MHz, 3 %
10:46:26.383, 2, 8, 16.25 W, 1425 MHz, 3 %
10:46:26.972, 2, 8, 16.10 W, 1425 MHz, 3 %
10:46:27.553, 2, 8, 16.09 W, 1440 MHz, 3 %
10:46:28.136, 2, 8, 16.08 W, 1440 MHz, 3 %
10:46:28.716, 2, 8, 15.96 W, 1440 MHz, 3 %
10:46:29.304, 2, 8, 16.26 W, 1440 MHz, 3 %
10:46:29.882, 2, 8, 16.02 W, 1440 MHz, 3 %
10:46:30.460, 2, 8, 15.92 W, 1440 MHz, 3 %
10:46:31.043, 2, 8, 15.82 W, 1410 MHz, 3 %
10:46:31.625, 2, 8, 15.70 W, 1410 MHz, 3 %
10:46:32.210, 2, 8, 16.31 W, 1410 MHz, 3 %
10:46:32.789, 2, 8, 15.80 W, 1410 MHz, 3 %
10:46:33.374, 2, 8, 15.91 W, 1410 MHz, 3 %
10:46:33.958, 2, 8, 16.08 W, 1410 MHz, 3 %
10:46:34.539, 2, 8, 16.16 W, 1485 MHz, 3 %
10:46:35.122, 2, 8, 15.55 W, 1485 MHz, 3 %
10:46:35.705, 2, 8, 15.87 W, 1485 MHz, 3 %
10:46:36.288, 2, 8, 15.71 W, 1485 MHz, 3 %
10:46:36.862, 2, 8, 15.67 W, 1485 MHz, 3 %
10:46:37.438, 2, 8, 15.99 W, 1485 MHz, 3 %
10:46:38.017, 2, 8, 15.78 W, 1440 MHz, 3 %
10:46:38.602, 2, 8, 15.51 W, 1440 MHz, 3 %
10:46:39.184, 2, 8, 15.83 W, 1440 MHz, 3 %
10:46:39.778, 2, 8, 15.67 W, 1440 MHz, 3 %
10:46:40.355, 2, 8, 15.80 W, 1440 MHz, 3 %
10:46:40.943, 2, 8, 15.87 W, 1440 MHz, 3 %
10:46:41.523, 2, 8, 16.00 W, 1470 MHz, 3 %
10:46:42.103, 2, 8, 15.95 W, 1470 MHz, 3 %
10:46:42.677, 2, 8, 16.45 W, 1470 MHz, 3 %
10:46:43.258, 2, 8, 15.73 W, 1470 MHz, 3 %
10:46:43.840, 2, 8, 15.73 W, 1470 MHz, 3 %
10:46:44.421, 2, 8, 15.61 W, 1470 MHz, 3 %
10:46:45.001, 2, 8, 16.09 W, 1515 MHz, 3 %
10:46:45.581, 2, 8, 15.95 W, 1515 MHz, 3 %
10:46:46.170, 2, 8, 15.28 W, 1515 MHz, 3 %
10:46:46.754, 2, 8, 15.26 W, 1515 MHz, 3 %
10:46:47.338, 2, 8, 14.72 W, 1515 MHz, 3 %
10:46:47.919, 2, 8, 14.65 W, 1515 MHz, 3 %
10:46:48.504, 2, 8, 14.74 W, 1260 MHz, 3 %
10:46:49.087, 2, 8, 14.73 W, 1260 MHz, 3 %
10:46:49.676, 2, 8, 14.71 W, 1260 MHz, 3 %
10:46:50.253, 2, 8, 14.48 W, 1260 MHz, 3 %
10:46:50.839, 2, 8, 14.36 W, 1260 MHz, 3 %
10:46:51.417, 2, 8, 14.33 W, 1260 MHz, 3 %
10:46:51.999, 2, 8, 14.57 W, 1275 MHz, 3 %
10:46:52.581, 2, 8, 14.43 W, 1275 MHz, 3 %
10:46:53.163, 2, 8, 14.43 W, 1275 MHz, 3 %
10:46:53.752, 2, 8, 14.46 W, 1275 MHz, 3 %
10:46:54.330, 2, 8, 14.41 W, 1275 MHz, 3 %
10:46:54.911, 2, 8, 14.52 W, 1275 MHz, 3 %
10:46:55.496, 2, 8, 14.51 W, 1222 MHz, 3 %
10:46:56.078, 2, 8, 14.49 W, 1222 MHz, 3 %
10:46:56.658, 2, 8, 14.47 W, 1222 MHz, 3 %
10:46:57.238, 2, 8, 14.75 W, 1222 MHz, 3 %
10:46:57.814, 2, 8, 14.68 W, 1222 MHz, 3 %
10:46:58.390, 2, 8, 14.68 W, 1222 MHz, 3 %
10:46:58.975, 2, 8, 14.76 W, 1222 MHz, 3 %
10:46:59.557, 2, 8, 14.73 W, 1297 MHz, 3 %
10:47:00.148, 2, 8, 14.81 W, 1297 MHz, 3 %
10:47:00.732, 2, 8, 21.54 W, 1297 MHz, 3 %
10:47:01.312, 2, 8, 14.84 W, 1297 MHz, 3 %
10:47:01.895, 2, 8, 14.71 W, 1297 MHz, 3 %
10:47:02.478, 2, 8, 12.96 W, 1297 MHz, 3 %
10:47:03.060, 2, 8, 14.66 W, 1252 MHz, 3 %
10:47:03.638, 2, 8, 11.99 W, 1252 MHz, 3 %
10:47:04.221, 2, 8, 14.78 W, 1252 MHz, 3 %
10:47:04.808, 2, 8, 14.77 W, 1252 MHz, 3 %
10:47:05.394, 2, 8, 14.49 W, 1252 MHz, 3 %
10:47:05.976, 2, 8, 15.08 W, 1252 MHz, 3 %
10:47:06.569, 2, 8, 14.24 W, 1260 MHz, 3 %
10:47:07.154, 2, 8, 14.56 W, 1260 MHz, 3 %
10:47:07.738, 2, 8, 15.26 W, 1260 MHz, 3 %
10:47:08.314, 2, 8, 14.42 W, 1260 MHz, 3 %
10:47:08.895, 2, 8, 14.55 W, 1260 MHz, 3 %
10:47:09.480, 2, 8, 14.86 W, 1260 MHz, 3 %
10:47:10.065, 2, 8, 15.11 W, 1290 MHz, 3 %
10:47:10.651, 2, 8, 15.11 W, 1290 MHz, 3 %
10:47:11.237, 2, 8, 14.60 W, 1290 MHz, 3 %
10:47:11.822, 2, 8, 14.43 W, 1290 MHz, 3 %
10:47:12.410, 2, 8, 15.94 W, 1290 MHz, 3 %
10:47:12.995, 2, 8, 14.85 W, 1335 MHz, 3 %
10:47:13.581, 2, 8, 14.99 W, 1335 MHz, 3 %
10:47:14.172, 2, 8, 14.19 W, 1335 MHz, 3 %
10:47:14.757, 4, 8, 16.07 W, 1335 MHz, 3 %
10:47:15.359, 4, 8, 16.32 W, 1335 MHz, 2 %
10:47:15.944, 4, 8, 16.44 W, 1335 MHz, 2 %
10:47:16.532, 4, 8, 16.21 W, 1957 MHz, 2 %
10:47:17.120, 4, 8, 16.23 W, 1957 MHz, 2 %
10:47:17.707, 4, 8, 16.54 W, 1957 MHz, 2 %
10:47:18.292, 4, 8, 16.15 W, 1957 MHz, 2 %
10:47:18.877, 4, 8, 16.17 W, 1957 MHz, 2 %
10:47:19.465, 4, 8, 16.11 W, 1957 MHz, 2 %
10:47:20.048, 4, 8, 16.00 W, 1980 MHz, 2 %
10:47:20.636, 2, 8, 16.25 W, 1980 MHz, 2 %
10:47:21.218, 2, 8, 15.34 W, 1980 MHz, 3 %
10:47:21.805, 2, 8, 14.84 W, 1980 MHz, 3 %
10:47:22.392, 2, 8, 14.84 W, 1980 MHz, 3 %
10:47:22.978, 2, 8, 15.55 W, 1980 MHz, 3 %
10:47:23.567, 2, 8, 15.02 W, 1192 MHz, 3 %
10:47:24.151, 2, 8, 14.49 W, 1192 MHz, 3 %
10:47:24.735, 2, 8, 14.46 W, 1192 MHz, 3 %
10:47:25.322, 2, 8, 14.55 W, 1192 MHz, 3 %
10:47:25.909, 2, 8, 14.75 W, 1192 MHz, 3 %
10:47:26.495, 2, 8, 15.78 W, 1327 MHz, 3 %
10:47:27.112, 2, 8, 23.90 W, 1327 MHz, 3 %
10:47:27.695, 2, 8, 15.56 W, 1327 MHz, 3 %
10:47:28.282, 2, 8, 14.80 W, 1327 MHz, 3 %
10:47:28.870, 2, 8, 14.70 W, 1327 MHz, 3 %
10:47:29.457, 2, 8, 15.90 W, 1327 MHz, 3 %
10:47:30.047, 2, 8, 15.15 W, 1207 MHz, 3 %
10:47:30.637, 2, 8, 14.98 W, 1207 MHz, 3 %
10:47:31.223, 2, 8, 15.19 W, 1207 MHz, 3 %
10:47:31.799, 2, 8, 15.24 W, 1207 MHz, 3 %
10:47:32.385, 2, 8, 15.23 W, 1207 MHz, 3 %
10:47:32.970, 2, 8, 15.15 W, 1207 MHz, 3 %
10:47:33.558, 2, 8, 14.75 W, 1215 MHz, 3 %
10:47:34.140, 2, 8, 15.18 W, 1215 MHz, 3 %
10:47:34.723, 2, 8, 15.16 W, 1215 MHz, 3 %
10:47:35.313, 2, 8, 15.18 W, 1215 MHz, 3 %
10:47:35.899, 2, 8, 15.39 W, 1215 MHz, 3 %
10:47:36.488, 2, 8, 15.09 W, 1215 MHz, 3 %
10:47:37.075, 2, 8, 15.74 W, 1192 MHz, 6 %
10:47:37.660, 2, 8, 15.34 W, 1192 MHz, 3 %
10:47:38.246, 2, 8, 14.95 W, 1192 MHz, 3 %
10:47:38.834, 2, 8, 15.01 W, 1192 MHz, 3 %
10:47:39.420, 2, 8, 15.18 W, 1192 MHz, 3 %
10:47:40.004, 2, 8, 14.89 W, 1170 MHz, 3 %
10:47:40.592, 2, 8, 15.36 W, 1170 MHz, 3 %
10:47:41.181, 2, 8, 15.44 W, 1170 MHz, 3 %
10:47:41.769, 2, 8, 15.09 W, 1170 MHz, 3 %
10:47:42.347, 4, 8, 14.67 W, 1170 MHz, 3 %
10:47:42.938, 3, 8, 30.74 W, 1170 MHz, 99 %
10:47:43.526, 3, 8, 13.39 W, 1485 MHz, 99 %
10:47:44.113, 4, 8, 23.66 W, 1485 MHz, 99 %
10:47:44.704, 4, 8, 35.86 W, 1485 MHz, 100 %
10:47:45.292, 4, 8, 35.90 W, 1485 MHz, 100 %
10:47:45.879, 4, 8, 37.23 W, 1485 MHz, 100 %
10:47:46.461, 4, 8, 36.09 W, 1485 MHz, 100 %
10:47:47.051, 4, 8, 36.08 W, 2820 MHz, 100 %
10:47:47.639, 4, 8, 18.79 W, 2820 MHz, 3 %
10:47:48.226, 4, 8, 14.67 W, 2820 MHz, 4 %
10:47:48.810, 4, 8, 15.40 W, 2820 MHz, 4 %
10:47:49.400, 4, 8, 15.98 W, 2820 MHz, 3 %
10:47:49.983, 4, 8, 15.34 W, 2820 MHz, 3 %
10:47:50.570, 4, 8, 14.07 W, 1245 MHz, 4 %
10:47:51.154, 4, 8, 14.73 W, 1245 MHz, 3 %
10:47:51.742, 4, 8, 14.76 W, 1245 MHz, 3 %
10:47:52.330, 4, 8, 14.42 W, 1245 MHz, 3 %
10:47:52.914, 4, 8, 14.98 W, 1245 MHz, 3 %
10:47:53.501, 4, 8, 14.68 W, 1260 MHz, 3 %
10:47:54.088, 4, 8, 14.80 W, 1260 MHz, 3 %
10:47:54.674, 1, 8, 35.90 W, 1260 MHz, 99 %
10:47:55.273, 4, 8, 28.45 W, 1260 MHz, 36 %
10:47:55.857, 4, 8, 21.17 W, 1260 MHz, 34 %
10:47:56.443, 4, 8, 19.54 W, 1260 MHz, 27 %
10:47:57.046, 4, 8, 19.97 W, 1680 MHz, 35 %
10:47:57.624, 4, 8, 18.39 W, 1680 MHz, 3 %
10:47:58.210, 4, 8, 15.47 W, 1680 MHz, 3 %
10:47:58.791, 4, 8, 14.86 W, 1680 MHz, 2 %
10:47:59.380, 4, 8, 14.76 W, 1680 MHz, 3 %
10:48:00.002, 4, 8, 16.59 W, 1440 MHz, 3 %
10:48:00.590, 4, 8, 16.75 W, 1440 MHz, 3 %
10:48:01.178, 4, 8, 16.37 W, 1440 MHz, 3 %
10:48:01.761, 4, 8, 16.47 W, 1440 MHz, 3 %
10:48:02.350, 2, 8, 16.72 W, 1440 MHz, 3 %
10:48:02.950, 2, 8, 17.99 W, 1440 MHz, 3 %
10:48:03.524, 2, 8, 16.08 W, 1470 MHz, 3 %
10:48:04.104, 2, 8, 16.06 W, 1470 MHz, 3 %
10:48:04.686, 2, 8, 16.58 W, 1470 MHz, 3 %
10:48:05.276, 2, 8, 16.40 W, 1470 MHz, 3 %
10:48:05.854, 2, 8, 15.87 W, 1470 MHz, 3 %
10:48:06.442, 2, 8, 16.21 W, 1470 MHz, 3 %
10:48:07.027, 2, 8, 14.36 W, 1350 MHz, 3 %
10:48:07.613, 2, 8, 14.71 W, 1350 MHz, 3 %
10:48:08.202, 2, 8, 14.78 W, 1350 MHz, 3 %
10:48:08.792, 2, 8, 14.76 W, 1350 MHz, 3 %
10:48:09.379, 2, 8, 14.84 W, 1350 MHz, 3 %
10:48:09.956, 2, 8, 14.90 W, 1350 MHz, 3 %
10:48:10.548, 2, 8, 18.03 W, 1252 MHz, 3 %
10:48:11.140, 4, 8, 32.94 W, 1252 MHz, 60 %
10:48:11.724, 4, 8, 25.77 W, 1252 MHz, 43 %
10:48:12.312, 4, 8, 15.72 W, 1252 MHz, 32 %
10:48:12.918, 4, 8, 19.20 W, 1252 MHz, 3 %
10:48:13.507, 4, 8, 16.01 W, 1207 MHz, 3 %
10:48:14.092, 4, 8, 19.43 W, 1207 MHz, 8 %
10:48:14.680, 4, 8, 19.89 W, 1207 MHz, 3 %
10:48:15.265, 4, 8, 14.92 W, 1207 MHz, 4 %
10:48:15.853, 4, 8, 15.06 W, 1207 MHz, 3 %
10:48:16.440, 4, 8, 15.05 W, 1207 MHz, 3 %
10:48:17.025, 2, 8, 13.57 W, 1455 MHz, 4 %
10:48:17.602, 2, 8, 11.64 W, 1455 MHz, 5 %
10:48:18.187, 2, 8, 12.87 W, 1455 MHz, 4 %
10:48:18.773, 2, 8, 14.41 W, 1455 MHz, 3 %
10:48:19.361, 2, 8, 16.05 W, 1455 MHz, 3 %
10:48:19.946, 2, 8, 16.47 W, 1455 MHz, 4 %
10:48:20.533, 2, 8, 15.72 W, 1417 MHz, 3 %
10:48:21.116, 2, 8, 15.78 W, 1417 MHz, 3 %
10:48:21.704, 2, 8, 15.84 W, 1417 MHz, 3 %
10:48:22.281, 2, 8, 15.99 W, 1417 MHz, 3 %
10:48:22.860, 2, 8, 15.36 W, 1417 MHz, 3 %
10:48:23.446, 2, 8, 15.84 W, 1417 MHz, 3 %
10:48:24.037, 2, 8, 16.14 W, 1455 MHz, 3 %
10:48:24.617, 2, 8, 16.06 W, 1455 MHz, 3 %
10:48:25.206, 2, 8, 15.96 W, 1455 MHz, 3 %
10:48:25.793, 2, 8, 15.98 W, 1455 MHz, 3 %
10:48:26.382, 2, 8, 16.02 W, 1455 MHz, 3 %
10:48:26.970, 2, 8, 16.07 W, 1455 MHz, 3 %
10:48:27.558, 2, 8, 15.95 W, 1425 MHz, 3 %
10:48:28.144, 2, 8, 15.82 W, 1425 MHz, 3 %
10:48:28.734, 2, 8, 15.78 W, 1425 MHz, 3 %
10:48:29.322, 2, 8, 15.86 W, 1425 MHz, 3 %
10:48:29.909, 2, 8, 16.07 W, 1425 MHz, 3 %
10:48:30.495, 2, 8, 16.12 W, 1425 MHz, 3 %
10:48:31.078, 2, 8, 16.18 W, 1485 MHz, 3 %
10:48:31.669, 2, 8, 15.82 W, 1485 MHz, 3 %
10:48:32.255, 2, 8, 15.85 W, 1485 MHz, 3 %
10:48:32.838, 2, 8, 15.87 W, 1485 MHz, 3 %
10:48:33.424, 2, 8, 15.85 W, 1485 MHz, 3 %
10:48:34.002, 2, 8, 15.90 W, 1485 MHz, 3 %
10:48:34.585, 2, 8, 16.74 W, 1552 MHz, 3 %
10:48:35.173, 2, 8, 15.36 W, 1552 MHz, 3 %
10:48:35.752, 2, 8, 15.22 W, 1552 MHz, 3 %
10:48:36.327, 4, 8, 29.95 W, 1552 MHz, 3 %
10:48:36.912, 4, 8, 22.66 W, 1552 MHz, 3 %
10:48:37.500, 4, 8, 15.38 W, 1552 MHz, 3 %
10:48:38.084, 4, 8, 15.86 W, 1230 MHz, 3 %
10:48:38.661, 4, 8, 16.53 W, 1230 MHz, 3 %
10:48:39.249, 4, 8, 16.22 W, 1230 MHz, 3 %
10:48:39.837, 4, 8, 16.47 W, 1230 MHz, 3 %
10:48:40.425, 4, 8, 16.49 W, 1230 MHz, 3 %
10:48:41.003, 4, 8, 16.35 W, 1230 MHz, 3 %
10:48:41.588, 4, 8, 33.08 W, 2362 MHz, 68 %
10:48:42.175, 3, 8, 28.93 W, 2362 MHz, 99 %
10:48:42.763, 3, 8, 11.87 W, 2362 MHz, 65 %
10:48:43.349, 3, 8, 14.74 W, 2362 MHz, 88 %
10:48:43.937, 4, 8, 15.37 W, 2362 MHz, 99 %
10:48:44.527, 4, 8, 29.71 W, 2122 MHz, 20 %
10:48:45.114, 4, 8, 31.74 W, 2122 MHz, 65 %
10:48:45.702, 4, 8, 31.48 W, 2122 MHz, 89 %
10:48:46.290, 4, 8, 37.30 W, 2122 MHz, 91 %
10:48:46.877, 4, 8, 31.19 W, 2122 MHz, 100 %
10:48:47.465, 4, 8, 31.36 W, 2122 MHz, 99 %
10:48:48.052, 4, 8, 32.49 W, 2265 MHz, 98 %
10:48:48.643, 4, 8, 32.14 W, 2265 MHz, 77 %
10:48:49.232, 4, 8, 31.90 W, 2265 MHz, 58 %
10:48:49.809, 4, 8, 28.78 W, 2265 MHz, 31 %
10:48:50.397, 4, 8, 14.43 W, 2265 MHz, 3 %
10:48:50.982, 4, 8, 13.64 W, 2265 MHz, 4 %
10:48:51.567, 4, 8, 13.64 W, 1042 MHz, 3 %
10:48:52.152, 4, 8, 13.71 W, 1042 MHz, 3 %
10:48:52.735, 4, 8, 13.71 W, 1042 MHz, 3 %
10:48:53.320, 4, 8, 13.71 W, 1042 MHz, 3 %
10:48:53.911, 4, 8, 14.61 W, 1042 MHz, 3 %
10:48:54.494, 2, 8, 15.83 W, 1042 MHz, 3 %
10:48:55.071, 2, 8, 16.04 W, 1440 MHz, 3 %
10:48:55.658, 2, 8, 15.56 W, 1440 MHz, 3 %
10:48:56.251, 2, 8, 14.71 W, 1440 MHz, 3 %
10:48:56.834, 2, 8, 14.70 W, 1440 MHz, 3 %
10:48:57.419, 2, 8, 14.64 W, 1440 MHz, 3 %
10:48:58.005, 2, 8, 14.73 W, 1440 MHz, 3 %
10:48:58.593, 2, 8, 14.95 W, 1275 MHz, 3 %
10:48:59.178, 2, 8, 14.94 W, 1275 MHz, 3 %
10:48:59.763, 2, 8, 14.70 W, 1275 MHz, 3 %
10:49:00.350, 2, 8, 14.73 W, 1275 MHz, 3 %
10:49:00.938, 2, 8, 15.19 W, 1275 MHz, 3 %
10:49:01.526, 2, 8, 15.97 W, 1432 MHz, 3 %
10:49:02.110, 2, 8, 15.91 W, 1432 MHz, 3 %
10:49:02.689, 2, 8, 16.04 W, 1432 MHz, 3 %
10:49:03.273, 2, 8, 16.49 W, 1432 MHz, 3 %
10:49:03.863, 2, 8, 16.13 W, 1432 MHz, 3 %
10:49:04.454, 2, 8, 16.11 W, 1432 MHz, 3 %
10:49:05.040, 2, 8, 16.17 W, 1500 MHz, 3 %
10:49:05.622, 2, 8, 16.09 W, 1500 MHz, 3 %
10:49:06.207, 2, 8, 16.08 W, 1500 MHz, 3 %
10:49:06.797, 2, 8, 16.25 W, 1500 MHz, 3 %
10:49:07.385, 2, 8, 15.92 W, 1500 MHz, 3 %
10:49:07.974, 2, 8, 16.02 W, 1500 MHz, 3 %
10:49:08.565, 2, 8, 16.06 W, 1417 MHz, 3 %
10:49:09.153, 2, 8, 15.80 W, 1417 MHz, 3 %
```
