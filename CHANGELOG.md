# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).
Releases are tagged `vX.Y.Z` on `main`; each tagged release is
archived on Zenodo for a citable DOI.

## [Unreleased]

### Added
- **Experimental** `newton_jacobian_mode` on the multipeak solver
  (`"raw"` | `"kaufman"` | `"golub_pereyra"` | `"gp_lm"`): compact
  Golub-Pereyra normal equations (no projector, no inverse; one
  multi-RHS Gram solve) with a scaled-diagonal Levenberg-Marquardt
  safeguard and per-spectrum accept/reject (`gp_lm`), plus a float64
  QR-based Jacobian verification layer and diagnostics
  (`MultiPeakResult.gp_diagnostics`). The default `"raw"` path is
  unchanged and byte-identical; non-raw modes are experimental and
  outside the public API stability guarantee — their names, defaults,
  and diagnostics may change in any release. See
  `src/toyomacro/voigtfit/benchmarks/GP_LM_HARDENING_REPORT.md`.
- Machine-readable benchmark provenance records
  (`paper/figures/results/`): kernel throughput (median + range +
  environment + per-repetition timings) and a same-problem
  scipy/lmfit comparison benchmark
  (`toyomacro.voigtfit.benchmarks.solver_comparison_benchmark`).
- MLX capability API: `mlx_installed()` / `mlx_usable()` /
  `require_mlx()`; the package now distinguishes "MLX installed"
  from "MLX can execute work" and falls back to NumPy on headless
  or virtualized macOS. `TOYOMACRO_DISABLE_MLX=1` forces the NumPy
  backend.
- Noise-semantics helpers `level_to_peak_snr` / `level_to_peak_lambda`
  with tests pinning the sampler's statistics to the documented
  conversions.
- Core API reference (`docs/API.md`), related-software survey
  (`docs/RELATED_SOFTWARE.md`), measured-data schema
  (`examples/data/README.md`), test inventory (`tests/README.md`),
  `CITATION.cff`, and this changelog.

### Changed
- Public CLI surface reduced to implemented, distributed commands:
  `gui` (private companion layer, not installed by this package) and
  `fit` (placeholder) are no longer registered; `import` and `convert`
  are unchanged. Running `toyomacro` with no arguments now prints help
  and exits successfully instead of attempting to launch a GUI. This is
  a public-boundary decision, not a feature removal: fitting remains
  fully available through the Python APIs (`AutoFitter`, `BatchFitter`,
  `FastVoigtFitter`, `VarProFitter`); a `fit` command may return once
  its data schema is defined, and a `gui` command if a distributable
  companion product exists.
- Single version source: `pyproject.toml` — `toyomacro.__version__`,
  `toyomacro.voigtfit.__version__`, and the CLI `--version` all
  derive from package metadata (previously the voigtfit subpackage
  reported an independent 0.6.0).
- Figure 1 (paper) no longer labels reference constants as measured;
  measurement bars require a live run or a committed record.
- Figure 2 (paper) x-axis changed from the misleading "Poisson λ" to
  peak-count SNR; the collapse criterion is now stated physically
  (peak signal = shot noise at SNR ≈ 1).
- `mx.compile` calls moved from import time to first call
  (`LazyCompiled`), so importing the package never touches the Metal
  compiler.
- JOSS paper restructured (State of the field, Software design,
  Research impact statement, AI usage disclosure sections added).

## [0.1.0] - 2026-07-07

Initial public code (squashed history; no tagged release yet — the
`v0.1.0` tag and Zenodo archive are pending release approval).

- `toyomacro.voigtfit`: batch-first Voigt solvers (amplitude-only
  projection, Taylor residual projection, dictionary + parabola,
  multipeak alternating projection), MLX GPU backend with pure-NumPy
  fallback, CRLB utilities, GVRT round-trip benchmark suite.
- Foundation modules: lineshapes (Voigt, Doniach-Sunjic, ...),
  Shirley/Tougaard backgrounds, quantification reference tables
  (Scofield, Yeh-Lindau, Trzhaskovskaya, TPP-2M), file readers
  (PXT/VAMAS/NPL/two-column text).
- Runnable examples, MCP server, CI on Linux/macOS × Python 3.11/3.12.
