# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).
Releases are tagged `vX.Y.Z` on `main`; each tagged release is
archived on Zenodo for a citable DOI.

## [Unreleased]

### Added

- The Scofield summation convention is now checked against a number the
  source states rather than against our own arithmetic. Table A2 of
  UCRL-51326 prints TOTAL and K/L/M SHELL columns beside the individual
  subshells; those columns are **not** in the bundled JSON, which stores
  j-resolved subshells only, so summing our stored Si subshells and
  comparing against them cannot be satisfied by the data agreeing with
  itself. They agree to ~1.4×10⁻⁵ relative — the source's own five-figure
  rounding — at both 1.0 and 1.5 keV, and the sum is verified to reject
  returning one j component (19% off), dropping a doublet (0.9%), a
  0.1% drift in a single subshell, and re-weighting by degeneracy on top
  of Scofield's already-included fractional occupancies (130%). No
  implementation change was needed; this records agreement that was
  previously untested.
- `CrossSection.unit_info(table)` reports the unit *and how well
  established it is*: `unit`, `inferred_unit`, `status`,
  `values_rescaled`, plus a `note` where inferred. The three tables are
  not on a common scale — Trzhaskovskaya runs ~10³ above the megabarn
  tables — and its unit has not been read from the ADNDT 77/82 table
  heading, so `unit` stays `None`, the candidate sits in
  `inferred_unit`, and **no conversion factor is offered**. Stored values
  are never rescaled on an inference. The 2018/2019 tables were checked
  separately rather than inheriting that inference: the UCL
  digitization's own sheet states kb explicitly, and that those σ are
  for completely filled subshells. Recorded in `docs/DATA_SOURCES.md`.
- The MCP `calculate_sensitivity` result carries the unit with its
  status intact — `cross_section_unit`, `cross_section_inferred_unit`,
  `cross_section_unit_status`, `sensitivity_unit`,
  `sensitivity_inferred_unit` — so an inferred unit cannot be collapsed
  into a single confirmed-looking string at the layer where an agent
  would act on it.
- **Experimental** `toyomacro.data.elastic_scattering`: overlayer-thickness
  effective attenuation length from the single-scattering albedo
  ω = IMFP/(IMFP+TRMFP), with a required `model` keyword selecting the
  published slope (Jablonski & Powell 2020 Eq. 20 for unpolarized x rays,
  Eq. 62 for linearly polarized HAXPES — a 2.4% difference for gold at
  7.4 keV, so there is no default). Both lengths are caller-supplied; no
  TRMFP or albedo tables are bundled. Scope is overlayer thickness only —
  mean escape depth, information depth and marker depth follow different
  slopes and are not implemented.
- CUDA backend proof-of-concept and validation record
  (`docs/CUDA_BACKEND_POC.md`): the MLX↔NumPy parity suite run against
  MLX's CUDA backend on an RTX 5070 Laptop under WSL2, plus the
  throughput/precision baseline, the TF32-by-default finding, the
  batch > 65,535 multipeak crash, and the four upstream MLX issues filed
  from it (`docs/upstream-issues/`). Validation only — CUDA is **not** a
  supported installation target and no `[cuda]` extra ships.
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
- **TPP-2M IMFP provenance and validity limits are now stated**
  (`data/imfp.py`, `docs/API.md`, `docs/DATA_SOURCES.md`). The formula
  itself is unchanged and every returned value inside the documented
  domain is bit-identical. What was missing was the surrounding
  contract: the source paper fits Eqn (3) over **50–2000 eV** and states
  it should not be used above 2000 eV, with the largest deviations below
  200 eV — yet the package's own examples apply it at Ga Kα HAXPES
  energies with no caveat. Now documented, together with the equation
  numbers used, the units (Eqn (3) is in Ångströms, the API returns nm),
  the definition of the energy argument (kinetic energy in the solid;
  nothing subtracted internally), `Eg = 0` for conductors, and the fact
  that `CompoundDB` material parameters are a separate input with
  separate provenance. `data.imfp.TPP2M_FITTED_RANGE_EV` exports the
  bounds; they are **not** enforced, since HAXPES extrapolation is a
  legitimate use — the same authors later reported TPP-2M "useful for
  energies between 50 eV and 30 keV" (Shinotsuka et al., *Surf.
  Interface Anal.* **47**, 871 (2015)). What that paper adds and this
  module does *not* implement is the relativistic form for higher
  energies; omitting its α(T) factor overestimates the IMFP by ~1.8% at
  7.4 keV and ~7% at 30 keV, one-sided and growing with energy. Now
  documented and pinned by test.
- `IMFP.tpp2m()` and `IMFP.sampling_depth()` now reject non-physical
  input with a `ValueError` naming the offending argument, instead of
  surfacing a bare `math domain error` (E ≤ 0, ρ ≤ 0) or
  `ZeroDivisionError` (M_w = 0). `sampling_depth()` also rejects
  `fraction` outside [0, 1). A non-positive modified-Bethe bracket —
  reachable below 38.4 eV for three very high density entries (Sg, Bh,
  Hs), where Eqn (3) would otherwise hand back a *negative* mean free
  path — now raises rather than returning it. The guard is one-sided:
  just above that root the formula returns absurdly large positive
  values instead, which is documented but not caught. Both regimes are
  well below the 50 eV fit floor.
- The MCP `calculate_sensitivity` tool now returns `imfp_model`,
  `imfp_fitted_range_eV` and `imfp_extrapolated` alongside `imfp_nm`.
  Its default photon energy is a HAXPES line, so its *default* call was
  returning a TPP-2M value four times beyond the source's stated
  ceiling with nothing in the response to say so.
- **Sensitivity-factor terminology is now stated precisely**
  (`data/cross_section.py`, `data/transmission.py`, `mcp_server.py`,
  `docs/API.md`, `examples/03_quantification.py`, README). No formula,
  coefficient or returned number changed. What changed is what the
  package claims those numbers are. `CrossSection.get_rsf()` returns a
  photoionization-cross-section ratio σ₁/σ₂ against a reference line;
  despite the historical method name it is not a complete relative
  sensitivity factor and not an average-matrix RSF. The `σ × IMFP`
  product from `calculate_sensitivity` and
  `examples/03_quantification.py` is likewise a *simplified intrinsic*
  sensitivity — the example previously called it "AMRSF-equivalent",
  which it is not. Both omit analyzer transmission, detector response,
  elastic-scattering/EAL corrections, the photoelectron angular
  distribution, x-ray polarization and the source/analyzer geometry.
  Subshell occupancy is **not** among the omissions — it is already
  carried by the tabulated cross-sections — and is named as such, so
  that no one adds it a second time. No complete AMRSF is assembled and
  no third-party AMRSF table is bundled. `docs/API.md` §5 now gives the
  AMRSF composition, and a table of which factors have implementations
  here and at what stability tier — σ and `IMFP` supported,
  `AngularCorrection` and `elastic_scattering` experimental — versus
  what composing them still would not reproduce. The method name is
  retained for backward compatibility; renaming is deferred.
- Analyzer transmission documentation now warns against applying the
  correction twice (`data/transmission.py`, `docs/API.md` §5). The two
  equivalent workflows — correcting the spectrum, or correcting the
  sensitivity — are spelled out with the explicit statement that doing
  both applies T twice, an energy-dependent error that does not cancel
  between lines. The `× 1000` factor is identified as the mSr→Sr
  conversion for these particular curves rather than a universal
  constant, and the endpoint clamp is flagged as extrapolation in all
  but name at HAXPES ratios. Transmission data remains instrument- and
  configuration-specific and is **not** bundled; the module is an
  adapter for curves the user is authorized to use.
- The MCP `calculate_sensitivity` result now names its own provenance:
  `cross_section_table` (the table actually used — the docstring
  claimed Scofield while the call resolves the process default, Yeh &
  Lindau, a 43% difference for Si 2p at Al Kα and 30–39% in the O/Si
  *ratio* that quantification actually uses) and `sensitivity_model` (a
  one-line statement of what the number excludes).
  `examples/03_quantification.py` carried the same incorrect Scofield
  attribution and is corrected. Also added
  `cross_section_table_range_eV` and `cross_section_extrapolated`: the
  IMFP already carried an extrapolation flag, and flagging only the IMFP
  implied the cross-section was on firmer ground at the same energy. It
  is not — Yeh & Lindau stops at 8047.8 eV, so the tool's own 9251.7 eV
  default extrapolates **both** factors. The new flag is one-sided and
  says so: out-of-range is certainly extrapolated, in-range is not a
  guarantee for a sparsely tabulated orbital. Numerical values are
  unaffected.
- Every magnitude `docs/API.md` §5 quotes for the cross-section tables is
  now pinned by `tests/test_cross_section_spin_orbit_limits.py`, and
  pinned **over the population** the claim describes rather than over
  exemplars. Five audit rounds found numbers in that section that had
  been measured on a few lines and written as general — one of them
  inside the test written to prevent the previous one. The rule going
  forward is that a magnitude in §5 is pinned by a test over its own
  population, or it does not appear there. **Known gap**: the §5
  magnitudes inherited from the IMFP change — the 1.8%/7% relativistic
  α(T) figures, the 18.9% RMS for inorganic compounds, and the
  −0.71/−0.03/−0.26 transmission power-law exponents — are *not* pinned.
  They were audited in their own change set; extending the rule to them
  is deferred rather than silently skipped.
- `AngularCorrection` is now documented, and documented as
  **experimental** (`docs/API.md` §5). It was a public export named
  nowhere in the docs while §5 listed the angular/polarization factor it
  implements as an AMRSF ingredient. It does not meet the Supported bar:
  no tests, no example. Its β/γ/δ grid also starts at 1500 eV and
  lookups below that clamp to the edge value rather than refusing, so
  Al Kα at 1486.6 eV silently returns the 1500 eV parameters — now
  stated, and unflagged in the return value.
- MLX capability messaging is now device-neutral. `mlx_usable()` probes
  MLX's default device and always did; the docstrings and
  `require_mlx()` errors wrongly described it as looking for a Metal
  device and told callers MLX was "Apple Silicon only" — false, and a
  dead end for non-Apple users. Detection behavior is unchanged.
- README and `docs/API.md` now state the backend boundary explicitly:
  NumPy and Apple Metal supported, CUDA experimentally validated with
  its caveats (TF32 default, the 65,535-batch multipeak crash, no CI)
  rather than left unmentioned.
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

### Fixed
- **The solvers no longer warn on their own first iteration.** Both
  Gauss-Newton refiners seed the previous residual norm with infinity and
  divided by it on iteration 0, so `inf/inf` raised "invalid value
  encountered in scalar divide" (and the array form in the vectorized
  path). It surfaced on the first line of output from the README quick
  start. The nan never reached control flow — `nan < tol` is False and an
  `iteration > 0` guard already suppressed the break — so **no computed
  value changes**: iteration counts, convergence flags, residuals and
  recovered centers are identical before and after, verified across six
  tolerance/iteration regimes. The ratio is simply not formed until there
  is a previous norm. Regression tests fail on the pre-fix code
  (`test_convergence_warnings.py`), which nothing else did.
- **Documented commands that could not run.** 45 `Usage:` lines across
  19 modules invoked `python -m voigtfit.…` — the import path from
  before the engine moved under `toyomacro`, so every one raised
  `ModuleNotFoundError`. Seven carried a `PYTHONPATH=python` prefix from
  a repository layout that no longer exists. The `HybridPipeline`
  docstring and `roundtrip_benchmark`'s usage block imported from
  `voigtfit` and from a top-level `benchmark` module respectively;
  `test_compression` reached `h5io` through a `sys.path` insertion that
  assumed a POSIX path separator. All now use the installed paths.
- **`src/toyomacro/voigtfit/scripts/` was excluded from linting.** The
  `.gitignore` rule for the repository's private `scripts/` directory
  was unanchored, so it also matched the tracked package directory of
  the same name — and ruff, which respects `.gitignore`, skipped it. The
  rule is now `/scripts/`; the previously hidden import-order error is
  fixed. New files added under any nested `scripts/` are no longer
  silently untracked.
- Test inventory (`tests/README.md`) reported 1,149 tests across 50
  files; the suite collects **1,578 across 67**. Fifteen files were
  missing from the tables, including the TPP-2M IMFP, elastic-scattering
  and provenance-schema suites. Every file is now listed and every
  subtotal sums to the stated count. `README.md`'s own two test counts,
  and its architecture tree's stale `toyomacro-python/` root, are
  corrected with it.
- **A bare orbital label now returns the whole spin-orbit doublet**
  (`data/cross_section.py`). It previously returned a single j
  component: the suffix search tried `3/2, 5/2, 7/2, 1/2` and took the
  first key that existed, never reaching the doublet-summing path. σ
  came out low by roughly the ratio of the shell's degeneracy to that
  component's — ~1.5× for p, ~2.5× for d, ~2.33× for f — so `'3d'` and
  `'4f'` were worse than the p case, and Ag 3d and Au 4f are standard
  calibration lines. **This changes returned numbers.** The Scofield
  O 1s / Si 2p ratio from the Scofield table moves from 5.40 to 3.57 —
  the summed doublet rather than one component. No published sensitivity
  ratio is cited for it: none was verified against a primary source, so
  the value is pinned as a regression on the bundled table rather than
  as a literature check.
  - Each component is now evaluated at the requested energy and then
    added. Summing the stored arrays and interpolating once is not
    equivalent: the interpolator fits one polynomial across an element's
    whole range, so where the components sit on different grids — routine
    just above a split threshold — the two disagree by up to ~27%
    (Re 3d at 2 keV). The invariant `bare == sum of ionizable
    components` is pinned over every non-s subshell in both j-resolved
    tables at six energies.
  - Thresholds are applied **per component**. Between the two thresholds
    of a split doublet only the lower-BE member is ionizable and the
    bare label returns it alone, which is what the spectrum contains
    there (W 3d at 1850 eV, between 1809 and 1872 eV).
  - Scofield cross sections already incorporate the report's assumed
    fractional subshell occupations — UCRL-51326 Table A1: B 2p is
    NE 0.33 + 0.67 = 1 electron, C 0.67 + 1.33 = 2, N 1 + 2 = 3,
    O 1.33 + 2.67 = 4 — so components are summed as stored and are not
    weighted by occupancy again.
  - A missing component is never guessed at, on either table, and the
    evidence for the two is kept separate. Trzhaskovskaya half-lists 38
    non-s subshells — all open valence shells whose upper-j component
    would be empty, with the configuration exceptions Cr 3d⁵, Mo 4d⁵,
    Eu 4f⁷ and Re 5d⁵ carrying both. That is consistent with "listed iff
    occupied", but the rule has not been read from ADNDT 77/82, so a
    half-listed bare subshell returns `None` rather than treating the
    absence as a zero; the listed component stays available as a
    j-resolved request. Scofield lists both components of everything it
    carries, so an absent one there would be a data gap and the subshell
    likewise returns `None`. The Trzhaskovskaya count is pinned, so the rule must
    be re-established if the data changes.
  - A component omitted from a covered subshell is no longer filled in
    by cross-element extrapolation. Where the table carries the subshell
    and leaves one component out, fitting that component from other
    elements would replace a known gap with a fabricated number that is
    indistinguishable from a tabulated one. Where the table carries no
    component of the subshell at all, the extrapolation still applies.
- **The CI archive-content guards never failed** (`.github/workflows/ci.yml`).
  They were written as `! grep <pattern> …` under `set -e`, and `set -e`
  explicitly does not exit when a command's status is inverted with `!` —
  so a match printed the offending path and the step passed anyway, with
  the trailing `echo` setting the exit status. Every guard was inert,
  including the pre-existing ones for the private `gui`/`api`/`agent`/
  `piar`/`depth` layers and for `AGENTS.md` / `CLAUDE*`. The claim in
  `AGENTS.md` that "CI asserts that the archives contain neither" was
  therefore false for as long as the check existed. Rewritten as an
  explicit `if grep …; then fail=1; fi` helper with a final
  `[ "$fail" -eq 0 ] || exit 1`, plus a check that the two contents
  listings are non-empty — an empty listing would otherwise satisfy every
  pattern. Verified to exit 1 on each denied pattern and 0 on a clean
  archive. The archives themselves were, and are, clean.
- **TPP-2M IMFP transcription error** (`data/imfp.py`): the C and D
  terms of Tanuma, Powell & Penn substituted `Ep/1000` for the parameter
  `U = Nv·ρ/M = Ep²/829.4`. The error grows as kinetic energy falls —
  about 1% at 1 keV, +34% at 100 eV, +900% for Au at 50 eV — and stayed
  hidden because the examples and the quantification demo all sit near
  1 keV. The corrected implementation reproduces all 14 compounds of the
  source paper's Table 5 to within 0.3%. Two layers of regression tests
  (implementation fidelity against quantities the paper states
  explicitly; physical plausibility with justified tolerances) added in
  `tests/test_imfp_tpp2m.py`, where there had been none.

### Removed
- **Breaking:** a j-resolved request against a table that stores only
  bare subshells now returns `None` instead of a fabricated value.
  Yeh–Lindau is the **default** table and carries no j-resolved key for
  any of its 105 elements, so `CrossSection.lookup('Au', '4f7/2', hv)`
  with no `table=` argument is now `None`; previously it fell through to
  the cross-element log(Z) fit and returned roughly the *whole* shell,
  so summing two components double-counted by ~2×. Splitting a total by
  an assumed branching ratio is not available: the statistical ratio is
  not exact (Scofield's own Si 2p3/2 : 2p1/2 is 1.966) and a split value
  would be indistinguishable from a tabulated one. Ask `scofield` or
  `trzhaskovskaya` for j-resolved values, or use the bare label.
- `CrossSection.unit()` was **not** published, deliberately. A method
  returning a bare `'kb'` string would let a caller convert on an
  inference by reading one field; `unit_info()` is the only accessor.
- `IMFP.attenuation_length()`, which multiplied the TPP-2M IMFP by a
  hardcoded 0.9 and described that as accounting for elastic scattering.
  It had no callers and was never part of the documented surface; use
  `data.elastic_scattering` with an explicit IMFP/TRMFP pair instead.
- `ReconstructionBenchmark.run_with_noise()`, reachable from the
  `toyomacro.voigtfit` surface, accepted a `noise_levels` list and
  ignored it: it always returned `{0.0: baseline}` and printed a note
  that noise injection needed a MATLAB routine not in this repository.
  A caller reading the return value got a silently truncated sweep. It
  had no callers. Working noise sweeps live in
  `RoundtripBenchmark.run_noise_sweep()`, the module-level
  `run_noise_sweep()`, and `spectra_generator.generate_noise_sweep()`.

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
