# Benchmark records

One JSON file per benchmark run, committed. This directory is the
database: there is no service to query and no dashboard to log into, and
a record is readable five years from now with `cat`.

## Producing one

```bash
python -m toyomacro.voigtfit.benchmarks.bench_platform \
    --origin mac \
    --records-dir benchmarks/records
```

`--origin` says where the run was **launched from**, not where it ran. A
cloud container cannot work that out for itself, and it is the only
field here that no amount of introspection recovers. Everything else —
CPU, cores, memory, OS, backend, load, thermal state, hypervisor steal,
git commit — is detected.

Keep `--n-batch` at its default unless you mean to change it. It moves
the answer, and records taken at different batch sizes are not
comparable; the summary tool says so when it sees them mixed.

## Reading them

```bash
python -m toyomacro.voigtfit.benchmarks.summarize_records
```

## The four configurations

| # | launched from | runs on | what it isolates |
|---|---|---|---|
| 1 | Mac | Mac (Metal) | Apple Silicon, the published-benchmark machine |
| 2 | Mac | cloud | — |
| 3 | Windows PC | Windows PC (CUDA) | NVIDIA, and the WSL2 layer |
| 4 | Windows PC | cloud | — |

2 and 4 are the interesting pair: they should land on the same kind of
hardware, so agreement between them validates the cloud baseline and
disagreement says the allocation depends on the caller. Either way it is
a control that most benchmark suites do not have.

**Cloud arms need repetition; local arms do not.** "Cloud" is not a
machine, it is a draw from a distribution of machines. This repository
has already watched one container change CPU mid-session — a 2.80GHz
Xeon became a 2.10GHz one, while the tracked document describing it went
on saying 2.80GHz. A single run from each origin cannot separate an
origin effect from the instance lottery.

## Filenames

```
<UTC timestamp>__<hardware label>__from-<origin>.json
20260921T034500Z__darwin-apple-m3-max-16c__from-mac.json
```

The hardware label is derived from the CPU and core count, never from
the hostname — these files are public, and a personal machine name does
not belong in them. One file per run rather than an appended log, so
four machines pushing to one repository never conflict.

## What a record deliberately does not contain

No filesystem path, no interpreter location, no repository location, no
sibling project's version. Paths in the recorded command line are
reduced to bare filenames.
`src/toyomacro/voigtfit/tests/test_benchmark_record.py` asserts this;
treat a failure there as a repository-wide problem rather than a
benchmark one. Run it by that full path — `pytest` on a non-existent
path exits 0.

Process names in the load snapshot **are** recorded, because they are
what makes a contended record diagnosable — an orphaned worker pool is
identified by seeing it in that list. On a machine where the running
applications should not be published, set
`TOYOMACRO_BENCH_REDACT_PROCESSES=1` and the names become
`process-1`, `process-2`, … with their CPU shares intact.

## Quality, and why nothing is refused

Every run writes its record, including runs on a machine that was busy.
The failure this design exists to prevent is not *measuring* a
contended host — it is measuring one and having nothing say so
afterwards. Each record carries a verdict:

- `quiet` — no contention signal was detected. Not a claim that the
  machine was idle.
- `contended` — load, hypervisor steal, or thermal throttling was seen,
  and the reasons are listed.
- `unknown` — the platform did not report enough to tell.

The summary tool excludes non-`quiet` records from headline figures and
prints what it excluded. Comparison is where the filtering belongs;
recording is not.
