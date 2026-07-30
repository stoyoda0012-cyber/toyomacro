# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).
Releases are tagged `vX.Y.Z` on `main`; each tagged release is
archived on Zenodo for a citable DOI.

## [Unreleased]

### Added

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
- `docs/API.md` §5 now warns about two pre-existing cross-section
  lookup defects, because the section's new advice to pass `table=`
  explicitly is what exposes them. Documented, not yet fixed — fixing
  either changes returned numbers and belongs in its own change set with
  its own audit.
  - Spin-orbit-split orbitals resolve differently per table, and no
    single request is correct everywhere. On Scofield and Trzhaskovskaya
    a bare orbital never reaches the doublet-summing path: the suffix
    search tries `3/2, 5/2, 7/2, 1/2` in that order and returns the
    first key that exists, so σ is low by *approximately*
    `2(2l+1)/(2j+1)` — ≈1.50× for p, 2.50× for d, 2.33× for f. Which key
    is returned is exact; the magnitude is a statistical branching-ratio
    estimate, and over all 832 doublets in the two tables only 71% fall
    within 3% of it (88% within 10%, all within 30%, worst case +29.6%
    for Scofield Pu 3d), so §5 states it as an approximation for
    recognizing the defect rather than correcting it. It is *not* always
    the j = 3/2 component (`4f3/2` does not exist, so `'4f'` yields
    `4f5/2`) and the shortfall is ~1.6× worse for d and f than for p,
    which matters because Ag 3d and Au 4f are standard calibration
    lines. The "sum the components" remedy also has a precondition: 38
    non-s subshells in `trzhaskovskaya` are tabulated with only one j
    component (`scofield` has none), and requesting the absent partner
    resolves through the log(Z) fit rather than raising — so the sum
    adds a fabricated value to a real one and roughly doubles σ
    (Si 3p 2.51×, Ti 3d 1.94×, C 2p 2.25×). On the
    default Yeh–Lindau table the mirror applies — no element in its 105
    entries carries a j-resolved key — so a bare orbital is correct and
    requesting components sends both through the cross-element log(Z)
    fallback, each returning roughly the whole shell: summing them
    double-counts by 1.979× (2p), 2.184× (Ag 3d), 2.258× (Au 4f). §5
    now gives the mechanism and the per-table remedy rather than one
    blanket instruction.
  - The tables are not reconcilable by any scale factor. At 1486.6 eV
    the Trzhaskovskaya values run 141× (Zn 2p) to 553× (C 1s) the
    Yeh–Lindau ones across twelve lines sampled — a 3.9× spread, so the
    unconditional "Megabarn" in the `CrossSection` docstring cannot hold
    for all three tables, and the discrepancy is units *plus* the
    photon-vs-photoelectron axis convention already recorded in
    `DATA_SOURCES.md`, not a single unit error. Part of the spread is the
    component-selection defect above leaking in: the d lines sit below
    the trend for their orbital type, and multiplying them by the 2.45×
    d-orbital shortfall puts them back on it.
  - Every magnitude quoted in §5 for these two defects is now pinned by
    `tests/test_cross_section_spin_orbit_limits.py`, and pinned **over
    the whole population** rather than over exemplars: the degeneracy
    test iterates all 832 doublets in both tables and asserts the
    distribution (including that a hard 3% tolerance would *fail*, so
    the approximation cannot quietly be restated as an equality). Four
    audit rounds found numbers in this section that had been measured on
    a few lines and written as general — the fourth found it inside the
    test written to prevent the third. The rule going forward is that a
    magnitude in §5 is pinned by a test over the population it claims to
    describe, or it does not appear there.
  - **Known gap**: the §5 magnitudes inherited from the IMFP change — the
    1.8%/7% relativistic α(T) figures, the 18.9% RMS for inorganic
    compounds, and the −0.71/−0.03/−0.26 transmission power-law
    exponents — are *not* pinned by tests. They were audited in their own
    change set; extending the rule to them is deferred rather than
    silently skipped.
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
- `IMFP.attenuation_length()`, which multiplied the TPP-2M IMFP by a
  hardcoded 0.9 and described that as accounting for elastic scattering.
  It had no callers and was never part of the documented surface; use
  `data.elastic_scattering` with an explicit IMFP/TRMFP pair instead.

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
