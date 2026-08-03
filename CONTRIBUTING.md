# Contributing to toyomacro

Thank you for your interest in contributing! This project welcomes bug
reports, feature suggestions, documentation improvements, and pull
requests.

## Reporting issues

- Search [existing issues](../../issues) before opening a new one.
- Include: a minimal reproducer, expected vs actual behavior, Python
  version, OS, and relevant package versions (`pip list` excerpt).
- For numerical / fitting issues, attach the input array shape, dtype,
  and a small synthetic dataset (or describe how to generate one) so
  the problem can be reproduced without your private data.

## Development setup

```bash
git clone https://github.com/<your-fork>/toyomacro.git
cd toyomacro
uv sync --extra dev          # or: pip install -e ".[dev]"
pytest                       # run the test suite
```

The Voigt-fitting engine (`toyomacro.voigtfit`) is intentionally
lightweight: `numpy`, `scipy`, `h5py`, `matplotlib`, and optional
`mlx` (Apple Silicon GPU). Desktop/web front-ends are separate
closed-source companion tools and are not part of this repository
(see the Scope section of the README).

## Pull requests

1. Fork the repository and create a topic branch from `main`.
2. Make focused commits — one logical change per commit, present-tense
   imperative subject line (e.g. `Fix Voigt amplitude overflow at
   large gamma`).
3. Add or update tests for the change. Existing tests must keep passing.
4. Run `ruff check .` and `pytest` before pushing.
5. Open a PR describing what changed and why; link any related issue.
6. Be patient — review is best-effort, not real-time.

## Changelog

Significant user-visible changes must update the `[Unreleased]` section
of [CHANGELOG.md](CHANGELOG.md) in the same pull request. Treat these as
significant:

- adding, changing, or removing a public API
- changing a scientific model, coefficient, data source, or the
  numerical meaning of a result
- changing a backend, a supported platform, or how the package installs
- changing the CLI or any other user-facing behavior
- changing a published performance or accuracy claim
- fixing a bug that affects compatibility or previously published results

Internal refactors, test-only changes, and typo fixes may be omitted —
but be ready to say why in the PR. Add entries under `[Unreleased]`, not
under an already-released version.

## Style

- Python: PEP 8 via `ruff` (configured in `pyproject.toml`,
  line length 100).
- Prefer small, composable functions. Avoid speculative abstraction.
- New public APIs need a docstring and at least one test.
- Performance-critical paths in `voigtfit` should include a benchmark
  in `src/toyomacro/voigtfit/benchmarks/` when introducing new solvers.

## Dev-log and design-note references

Docstrings and comments occasionally cite `dev-log NN`. These refer to
entries in the maintainer's (non-public) development log, recording
when a numerical result, threshold, or design decision was
established — read them like issue-tracker references: the number
identifies the experiment, the surrounding text states its
conclusion.

A few modules likewise cite a **design note** that is not published —
the `/provenance` HDF5 schema is the main one. These are dated working
records built on other non-public material, so they are kept under
`docs/_internal/` rather than shipped. Nothing in this repository
requires one in order to be understood: where such a note fixes a rule
that the code implements, the rule is stated at the point of use, and
the tests assert it. Cite them the same way — as provenance for a
decision, not as required reading. If you find a place where the note
is load-bearing and the rule is not stated locally, that is a bug in
the comment; please report it.

## Code of Conduct

This project adheres to the [Contributor Covenant
v2.1](CODE_OF_CONDUCT.md). By participating, you agree to uphold its
terms.

## License

By contributing, you agree that your contributions will be licensed
under the [MIT License](LICENSE).
