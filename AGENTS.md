# AGENTS.md

Instructions for any AI coding agent working in this repository.
Tool-agnostic, and deliberately short — this is not a development log.
`README.md` and `CONTRIBUTING.md` are the source of record for anything
user-facing and win wherever this file disagrees with them.

## What this repository is

`toyomacro` is a Python package for high-throughput Voigt fitting of
X-ray photoelectron spectroscopy (XPS) data, together with the analysis
foundation it needs. The batch-first fitting engine
(`toyomacro.voigtfit`) is the primary artifact.

## Layout

These are the subpackages that exist here:

| Path | Contents |
|---|---|
| `src/toyomacro/voigtfit/` | batch Voigt solvers, MLX and NumPy backends, benchmarks |
| `src/toyomacro/core/` | `Spectrum`, `FittingResult`, `PeakComponent` |
| `src/toyomacro/lineshape/` | Voigt, Gaussian, Lorentzian, pseudo-Voigt, Doniach-Sunjic |
| `src/toyomacro/background/` | Shirley, Tougaard, linear |
| `src/toyomacro/fitting/` | templates, `AutoFitter`, `FastVoigtFitter` |
| `src/toyomacro/data/` | cross-sections, binding energies, IMFP, transmission, elastic scattering |
| `src/toyomacro/io/` | HDF5 cache, readers, writers |
| `src/toyomacro/cli/` | command-line entry points |
| `src/toyomacro/mcp_server.py` | MCP server over the public layer |
| `tests/` and `src/toyomacro/voigtfit/tests/` | both are collected by pytest |
| `docs/`, `examples/`, `paper/` | documentation, runnable examples, JOSS paper |

A desktop GUI, a web front end and the ARXPS depth-profile solver are
developed in separate repositories. `toyomacro.gui`, `toyomacro.api` and
`toyomacro.depth` do not exist here: do not import them, do not add them
back, and do not describe them as part of this package. The Scope
section of `README.md` is the public statement of that boundary.

## Commands

```bash
uv sync --extra dev --extra mcp             # what CI installs
uv sync --extra dev --extra mlx             # + Apple Silicon GPU backend
uv run pytest                               # collects both test roots
uv run ruff check src/ tests/ examples/     # what CI lints
```

A bare `uv sync` does not install pytest — it lives in the `dev` extra.
Always include `--extra dev` before running tests, or you will silently
run some other interpreter's pytest.

## Conventions

- Python 3.11+; `ruff` with line length 100 (configured in `pyproject.toml`).
- Prefer small, composable functions. Avoid speculative abstraction.
- New public APIs need a docstring and at least one test.
- New solvers on a performance-critical path need a benchmark under
  `src/toyomacro/voigtfit/benchmarks/`.
- Match the surrounding code's comment density and naming rather than
  importing a different house style.

## Changelog

Significant user-visible changes must update `[Unreleased]` in
`CHANGELOG.md` **in the same change set**. The definition of
"significant" lives in the Changelog section of `CONTRIBUTING.md` and is
authoritative — read it there rather than reproducing it here.

If you judge that no entry is needed, say so and why in your final
report. Do not defer the changelog to a later task, and do not add
unreleased changes to an already-released section.

## Changing scientific content

Changes to a scientific model, a coefficient, a reference dataset, or
the numerical meaning of a result carry a higher bar than ordinary code:

- Cite the primary source, not a secondary description of it. Where the
  source states a quantity explicitly — a table of values, a worked
  example — reproduce it as a test.
- Separate implementation fidelity from physical plausibility. The first
  tests against what the source states; the second uses tolerances whose
  justification is written down.
- A formula transcribed from a paper is not verified by "the code runs",
  nor by agreement at one convenient energy. Exercise the regime where
  the disputed term dominates.
- State validity limits in the docstring: the range fitted, the geometry
  assumed, the estimator properties required. A bound that holds only
  for an unbiased, correctly specified model must say so.
- Never present a bound, a fitted coefficient, or a simulation result as
  a measurement.

## Roles

Two roles, defined by function rather than by tool, vendor or model. One
agent may hold both on ordinary work.

- **Implementer** — makes the change and its tests.
- **Independent auditor** — checks the change against its evidence
  without having produced it.

An independent audit is **required** before a change is considered
finished when it touches any of:

- a scientific model, coefficient, or reference dataset
- the meaning of a number: units, estimator, or what a quantity claims
  to measure
- the public/private boundary — what ships, what is bundled, what is
  merely readable
- a published performance or accuracy claim
- a release candidate

Everything else — ordinary user-visible changes, CLI work,
compatibility fixes — requires tests and the implementer's own review.
An independent audit there is useful, not mandatory; a task brief may
still call for one.

Rules:

- Where an independent audit is required, it must run in a context that
  did not produce the change. A re-read by the implementer in the same
  context is a self-check; it must not be recorded as an independent
  audit.
- Grade findings **Blocker / Major / Minor / Optional**.
- A change carrying an unresolved Blocker or Major finding is not
  release-ready. Resolved means fixed, or refuted with evidence — not
  acknowledged.
- Which agent fills which role is not recorded here. It changes with
  availability and cost, and pinning it in a tracked file makes the file
  wrong. Task briefs and untracked local notes carry the assignment.

## Public boundary

This repository is published. Tracked files must not contain:

- absolute paths from a developer's machine, or the layout of sibling
  checkouts
- locations of local backups or archives
- the internal structure of private repositories
- credentials, tokens, or local agent/MCP launch paths
- session handoffs, dated development logs, or task lists
- roadmaps the project is not committed to
- AI-provider/model routing, usage limits, or billing context

Machine-specific and session-specific material belongs in
`CLAUDE.local.md` or another untracked file.

`AGENTS.md` and `CLAUDE.md` are repository instructions, not package
content: they are excluded from the built sdist and wheel, and CI
asserts that the archives contain neither.

## Before you change things

- Public behavior, install paths, supported platforms → `README.md`
- Contribution rules and the changelog definition → `CONTRIBUTING.md`
- API surface and stability tiers → `docs/API.md`
- Provenance and redistribution of bundled data → `docs/DATA_SOURCES.md`
- Why the code is shaped this way → `docs/DESIGN_PHILOSOPHY.md`
