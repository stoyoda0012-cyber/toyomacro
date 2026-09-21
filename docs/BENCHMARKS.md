# Where the published numbers come from

Every throughput figure in `README.md`, `paper/paper.md` and
`docs/API.md` was measured on one specific machine, on one specific
tree. This file is the index: for each published number, which machine,
which commit, whether a machine-readable record is committed, and how to
regenerate it.

It records provenance only. It does not introduce a number that is not
already published elsewhere, and it is not a results document — the
results live in the records and in the documents that quote them.

## The rule

**A number is published only when a record of it is committed, or when
the document quoting it says plainly that it is not.** Two of the
published figures carry a committed machine-readable record; the rest
are previously recorded values, regenerable from the bundled
benchmarks. §2 and §3 separate them, and `README.md` marks the
distinction inline as well.

The consequence worth stating: a newer measurement on the maintainer's
machine does not change a published number. It changes it when its
record lands in `paper/figures/results/` and the quoting document is
updated in the same change set. Until then the committed record is the
published value, however old it is.

## 1. The machines

| # | Machine | Backend | Role | Covered by CI |
|---|---|---|---|---|
| **M1** | Apple M3 Max, 128 GB, macOS | MLX (Metal) | The published-benchmark machine. Every number in §2 and §3 is from here. | Metal is not; the macOS job runs MLX force-disabled |
| **M2** | RTX 5070 Laptop / Ryzen 9 8940HX, Windows 11 + WSL2 | MLX (CUDA) | Correctness validation of the CUDA backend. **No published performance claim.** | No |
| **M3** | Intel Xeon @ 2.80 GHz, 4 vCPU, no GPU, Linux | NumPy | A control, used once to refute a host-specific hypothesis about two codec speed assertions. | No |
| — | GitHub Actions: `ubuntu-26.04`, `macos-latest`, Python 3.11/3.12 | NumPy | Correctness only. **Runs no benchmarks**; speed assertions skip when `CI=true` (`skip_in_ci` in `test_compression.py`, `skipif(IN_CI, …)` elsewhere). | — |

M2 and M3 measurements are recorded as prose in
[`CUDA_BACKEND_POC.md`](CUDA_BACKEND_POC.md), not as JSON records —
see §4 for what that means when reading them.

## 2. Published numbers with a committed record

Both records were produced on **M1** and carry a full `environment`
block (timestamp, OS, CPU, Python/NumPy/SciPy/MLX versions, the exact
command, the git commit, and whether the tree was dirty).

### `paper/figures/results/figure1_throughput.json`

| | |
|---|---|
| Measured | 2026-07-12, commit `743c379`, clean tree |
| What | Amplitude-only projection kernel (`A = Y @ W`), 151 channels, 3 components, 4 M spectra, float32, MLX GPU |
| Method | 2 warmup runs discarded, 9 repetitions, median |
| Value | **478 M spec/s** median, 453–490 range |
| Quoted by | `README.md` headline and per-solver table; `paper/paper.md` Summary; `docs/API.md` `preview` row (as ~10⁸) |
| Regenerate | `python paper/figures/make_figure1_bottleneck.py --measure` |

Figure 1 of the paper is drawn from this record, and CI asserts that it
still regenerates (`build + clean-install smoke + metadata` job).

### `paper/figures/results/solver_comparison.json` and `…_numpy.json`

| | |
|---|---|
| Measured | 2026-07-28, commit `534d709` |
| What | Same-problem comparison: one Voigt peak, 151 channels, free amplitude/δE/δσ, peak-count SNR 10, 200 K spectra, seed 0 — input SHA-256 recorded in both files |
| Values | `dict2d_parabola` **6.6 M spec/s** (MLX) / **1.6 M spec/s** (NumPy); `scipy.optimize.curve_fit` 845 / 822; `lmfit` 801 / 808 |
| Quoted by | `README.md` §"Same problem, same machine" and §"Without the GPU"; `paper/paper.md` (as "about 8,000×"); `docs/API.md` `balanced` row |
| Regenerate | `python -m toyomacro.voigtfit.benchmarks.solver_comparison_benchmark --out <file>` (prefix `TOYOMACRO_DISABLE_MLX=1` for the NumPy record) |

The two records are a matched pair: same host, same day, same input
hash, differing only in the backend.

**Provenance caveat.** Both records were taken on a dirty tree. The
modified file was the benchmark script itself in both cases — the
`solver_comparison_numpy.json` run additionally saw the already-written
MLX record as modified. Each file's `git_dirty_files` names them. The
measured code path is therefore `534d709` plus an unrecorded edit to
the harness, not `534d709` exactly.

**One row in these records is not what its name suggests.** The
amplitude-only projection entry is BLAS-on-CPU in *both* files — that
kernel is not routed through MLX in this benchmark — so its two columns
differ only by run-to-run noise. The MLX headline for that kernel is
`figure1_throughput.json`, above. `README.md` carries this warning
inline; `docs/API.md` does not.

## 3. Published numbers without a committed record

All from **M1**, none carrying a record file. The quoting documents
describe them as previously recorded values, regenerable from the
bundled benchmarks — which holds for the first four rows and not for
the two marked below.

| Number | Quoted by | Regenerate with |
|---|---|---|
| 4-step Taylor ~1.5 M spec/s | `README.md` per-solver table; `API.md` `fast` (~10⁶) | `benchmarks.bench_ncomp_scaling` (4-step row, 100 K spectra) |
| `parabola` ~5 M spec/s | `README.md` per-solver table; `API.md` `balanced` (~5×10⁶) | `benchmarks.bench_dict2d`; `bench_ncomp_scaling` (dict2d row) |
| `multipeak` 2-component ~3 M, 10-component ~40 K spec/s | `README.md` per-solver table | `benchmarks.bench_ncomp_scaling` (`N_COMP_LIST = [1, 2, 3, 5, 10]`) |
| `precise` ~4×10⁵ spec/s | `API.md` mode table | `benchmarks.bench_dict2d_adaptive_fused` (`precise` maps to the adaptive solver) |
| `gamma_calibrated` ~2 M spec/s | `README.md` per-solver table | **No throughput benchmark.** Traces to a module docstring (`fitting/fast_voigt.py:9`). `bench_dict3d_psnr` and `bench_gamma_perturbation` exercise this solver but are accuracy studies. |
| Batch-size scaling: MLX 0.8 M @ 2 K → 7.3 M @ 1 M; NumPy peaks 1.9 M @ 50 K, 1.4 M @ 1 M | `API.md` §3 | **No benchmark reproduces this.** A one-off measurement, stated as such ("one M3 Max measurement"); the figures appear nowhere else in the repository. `bench_dict2d_sweep` sweeps dictionary resolution, not batch size. |
| GVRT: 540p 2.6 M fits / ~0.9 s; 4K 66 M / 1.4 min; 8K UHD 265 M / ~10.4 min | `README.md` §"End-to-end image roundtrip"; `paper/paper.md` Summary | `benchmarks.param_roundtrip_benchmark`; `bench_gvrt_1b` for the synthetic 10⁹ variant. 4K/8K need an image under `VOIGTFIT_DATA_ROOT`; the 540p demo image is synthetic (`voigtfit.gvrt_service`) |

The last two rows are the weak points of the published set: one figure
whose only source is a docstring, and one that no bundled benchmark
regenerates. Both are small claims in context, but neither can be
checked by running something.

Two readability notes, since the same solver appears at two rates:

- `parabola ~5 M` (§3) and `dict2d_parabola 6.6 M` (§2) are the same
  solver measured under different benchmarks and batch sizes, not a
  disagreement. §2's figure is the one with a record, and the one the
  paper quotes.
- The per-solver table measures the solver step alone; the GVRT rows
  count generation + fit for a whole image. They are not comparable.

## 4. The CUDA and GPU-less machines

Neither M2 nor M3 produced a JSON record. Their measurements are prose
tables in [`CUDA_BACKEND_POC.md`](CUDA_BACKEND_POC.md), which is the
authoritative text for both; this section only says what is there and
how far it travels.

**M2 — RTX 5070 / WSL2 (CUDA).** The MLX↔NumPy parity suite has been
run end to end against MLX's CUDA backend and passed, with TF32
disabled; `CUDA_BACKEND_POC.md` carries the dated runs and their
criteria. Throughput and precision were also measured there, but
**nothing from M2 is a published performance claim**: `README.md` and
`docs/API.md` both state that CUDA has no CI and no committed
performance record. Read those measurements as findings about MLX's
CUDA backend on `sm_120`, not as the package's performance.

Two operating constraints from M2 do carry into user-facing
documentation, because they are correctness matters rather than
performance ones: TF32 must be disabled (`MLX_ENABLE_TF32=0`), and
multipeak alternating-projection batches must stay at or below 65,535
(an upstream MLX limit).

**M3 — GPU-less Xeon (NumPy).** Used once, for one question: whether
the `test_decode_speed` assertions fail on M2 because of something
about M2. They do not — a second, unrelated host missed the same
thresholds, which points at the thresholds. No performance claim
anywhere derives from M3.

**A Linux-only measurement bug closed in 2026-09.** `ru_maxrss` is in
kibibytes on Linux; `memory.get_rss_gb` converted it at 1000 bytes per
kilobyte, and four benchmark modules treated it as bytes outright.
Peak-RSS figures measured on M2 or M3 before that fix are wrong by
1000× or 1024× depending on the call site. No published number is
affected — the figures come from M1 (macOS, where `ru_maxrss` is
already in bytes), and `CUDA_BACKEND_POC.md` records no RSS-derived
value. See the `[Unreleased]` section of `CHANGELOG.md`.

## 5. Reading a record

Every record written by these benchmarks carries an `environment`
block. When comparing two numbers, check these before concluding
anything:

- `cpu` and `mlx_usable` — which machine and which backend.
- `git_commit` and `git_tracked_dirty` — which tree. A dirty tree means
  the record does not identify the code that produced it.
- `input_sha256`, where present — whether two records fitted the same
  data. The MLX/NumPy pair in §2 does.
- `aggregation`, `warmup`, `repeats` — a median over repetitions after
  discarded warmups is not comparable to a single cold timing. Cold
  first runs absorb MLX compilation and read low.

The last point is not hypothetical: two ratio-based tests in this
repository once compared a cold run against a warm one and reported
that the pipeline doing more work was faster. Ratios need warmup on
both arms and a median over repetitions.

### What the record does not capture: machine load

The `environment` block identifies the machine, the software and the
tree. It does not record how busy the machine was, and the amplitude-only
kernel is sensitive to that at a scale larger than its own reported
range.

Measured on M1: with the machine otherwise idle the committed record
gives **478 M spec/s** (range 453–490 over 9 repetitions). Re-running
the identical command while unrelated CPU-bound work saturated the
machine (load average ~12) gave **418–431 M** across four independent
invocations — each internally consistent to about 3%, all of them
9–13% below the committed median, and none of them overlapping it.

So the within-run range printed in a record understates the real
uncertainty: repetitions inside one invocation share the machine state
that biases them. Two consequences:

- **A number measured on a loaded machine is not a regression.** Before
  concluding that throughput has changed, check that the host was quiet
  — `uptime` and the top CPU consumers, at the time of the run.
- **Comparing two records across dates compares two unrecorded machine
  states as well.** Until load is captured in the `environment` block,
  a cross-date difference under roughly 10% on this kernel is not
  evidence of anything.

Schema v2 closes this: `load` is now part of every record written by
`benchmarks.record`, and the cross-platform harness warns when it
measures a busy host. Records in §2 predate it and carry no load field,
which is itself worth knowing when comparing them against anything new.

## 6. Measuring a new host

`benchmarks.bench_platform` is the one harness meant to run on all
three machine classes — Metal, CUDA and NumPy — producing records in a
single schema that can be laid side by side:

```bash
python -m toyomacro.voigtfit.benchmarks.bench_platform --out record.json
```

The things that make two records comparable, and that this harness
fixes rather than leaves to the operator:

- **The same problem.** It reuses `solver_comparison_benchmark`'s seeded
  single-Voigt problem and records the input SHA-256, so a record is
  comparable with `solver_comparison.json` at the same batch size.
- **The same batch size.** `--n-batch` defaults to 200,000, matching the
  committed record. It is recorded, because it moves the answer: the MLX
  path climbs with batch size and the NumPy path peaks near 50 K.
- **A backend that is named, not assumed.** `metal` / `cuda` / `numpy`,
  from the default device rather than from whether MLX imported.
- **TF32 off on CUDA.** The harness refuses to write a CUDA record with
  TF32 enabled unless explicitly overridden.
- **Batches within the CUDA `gridDim` limit**, chunked automatically
  there and left alone on Metal.
- **The machine's load**, captured at measurement time, with a warning
  when the host was busy.

Records are committed, so the schema deliberately carries no filesystem
path, no interpreter location, no repository path and no sibling
project's version; process names in the load snapshot are bare names.
`record.py` states this as its contract and
`src/toyomacro/voigtfit/tests/test_benchmark_record.py` asserts it.

### When a record disagrees with a published number

It will, and that is not a contradiction to fix by editing one of them.
A record is what one machine measured on one tree on one day. A
published number is a claim this project stands behind. They are
different objects with different burdens of proof.

The rule from §"The rule" decides it: **the committed record that
`README.md` and `paper/paper.md` quote is the published value until a
change set updates both the record and the quoting document together.**
A newer record under `benchmarks/records/` showing a higher figure is
evidence toward such a change, not the change itself — and because
altering a published performance claim is one of the changes
`AGENTS.md` requires an independent audit for, a single run is not
enough to make it.

So a reader comparing the two should expect the records to run ahead,
and should read a gap as work not yet done rather than as an error.
