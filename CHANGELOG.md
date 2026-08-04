# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).
Releases are tagged `vX.Y.Z` on `main`; each tagged release is
archived on Zenodo for a citable DOI.

## [Unreleased]

### Fixed

- **`clear_cache()` deleted shipped reference data, and
  `regenerate_cache()` could not put it back.** `clear_cache()` unlinked
  every `*.json` under `data/_cache/` — the package's own bundled
  tables. `regenerate_cache()` called it first and then rebuilt only
  five of the seven: `scofield.json` and `trzhaskovskaya.json` were
  absent from its list, so they were deleted with nothing to restore
  them, after which `CrossSection` silently fell back to an **empty
  table** rather than reporting the loss. The loader change above made
  it worse, turning a partial loss into a total one — the first loader
  now raises, so all seven files were gone and none rebuilt. Both
  functions are Supported-tier, so the names stay: `clear_cache()` now
  raises and explains why deleting shipped data has no correct use, and
  `regenerate_cache()` requires the same `TOYOMACRO_REGENERATE_DATA`
  opt-in, **never deletes first**, writes each table in place only if
  its source is available, keeps the shipped copy otherwise, and returns
  a per-table report naming the two it does not handle instead of
  omitting them silently. It also warns immediately **before** each
  overwrite, which the README and docstrings promised but the code did
  not do — the opt-in alone is a poor signal, since an environment
  variable set once in a shell persists; a test turns the warning into
  an error and asserts nothing was written.
  `src/toyomacro/data/_cache/README.md`
  described these files as regenerable build artifacts that should not
  be hand-edited, which was the opposite of the policy; it now says what
  they are.

- **Deleting a bundled reference table no longer rebuilds it, silently,
  from a CSV that is not in this repository.** The tables under
  `data/_cache/` are shipped, reviewed data; the loaders treated them as
  a regenerable cache, and `docs/DATA_SOURCES.md` documented deleting
  one as the way to rebuild it. On the maintainer's machine that path
  returns a **different dataset**: 166 compound entries instead of 109,
  **no `Si3N4`** — the CSV splits it by phase under other names, so
  `CompoundDB.get_properties("Si3N4")` would begin returning `None` —
  and different parameters for Al2O3, GaAs, SiC and SiO2, the four with
  published counterparts. Any of those may be the right change; none
  should arrive because a file was deleted, least of all in the one
  category the project requires an independent audit for. A missing
  table now raises `FileNotFoundError` naming the file and the opt-in;
  setting `TOYOMACRO_REGENERATE_DATA=1` rebuilds it and warns that the
  result is unreviewed. Public installs are unaffected either way — the
  CSVs have never shipped, so that path could only ever fail. Measured
  while checking the scope: `cross_section.json` and
  `binding_energy.json` still match their CSVs exactly, so
  `compounds.json` was the only table that had actually diverged.
  `tests/test_bundled_table_loading.py` covers all three.

- **`compounds.json` gave Si3N4 a molecular weight of 104.28346 g/mol;
  the correct value is 140.28346.** A digit transposition, and the only
  one in the table — the other ten compounds agree with their formula
  weights to better than 0.011%, and the 85 elements that have a
  standard atomic weight agree with it to better than 0.1%. It was not
  cosmetic: TPP-2M combines the parameters as `U = N_v·ρ/M`, so an M
  that is 26% low inflates U by 34% and the reported IMFP by **+25.8% at
  50 eV, +20% at 100 eV and +11.5% at 2 keV** — the whole range the
  formula was fitted over. Anything that took Si3N4 parameters from
  `CompoundDB` and passed them to `IMFP.tpp2m()` or `sampling_depth()`
  was affected, including the `calculate_sensitivity` tool of the
  bundled MCP server; explicitly passed parameters were not. The 1991
  paper corroborates the correction independently: inverting the
  `E_p = 28.8·√(N_v·ρ/M)` printed with its Table 5 gives M ≈ 140.4 g/mol
  for Si3N4. The project's upstream table already carried 140.2833, so
  only the bundled cache was stale.
  `tests/test_compound_parameters.py` (25 tests) now derives every
  compound's molecular weight from its chemical formula, checks the
  element entries against IUPAC values, and checks that N_v is counted
  per molecule to match — the coherence a mistyped constant breaks;
  reintroducing 104.28346 fails two of them. No other bundled value
  changed.

- **`voigtfit.crlb` reported a small bound where it should have
  reported no bound at all.** `compute_multipeak_fisher()` never
  inverted the Fisher matrix; it built a pseudo-inverse with eigenvalues
  at or below `1e-12·λ_max` sent to zero, so a direction carrying no
  information contributed **nothing** to the diagonal instead of
  diverging. The reported `crlb` was therefore *smaller* than the true
  bound exactly where the configuration was hardest, and it fell as the
  problem got worse: four peaks at σ=0.5, γ=0.3 go from a 1-dimensional
  null space at overlap 0.3 to a 3-dimensional one at overlap 0.2, and
  the reported bound drops by a factor of 26 across that step. The
  consequence reached users: `process_multipeak()` attaches a
  `SolvabilityInfo` to every result, and at SNR 1e6 that four-peak,
  overlap-0.2 configuration — where the individual centres are not
  identifiable at all — was classified **EASY**, "meV precision,
  well-resolved". It is now IMPOSSIBLE. `crlb` entries are `inf` for any
  parameter whose axis projects onto the numerical null space, flagged
  in the new `unbounded` mask and counted by `null_space_dim`; the
  previous diagonal remains available as `crlb_pseudo`, documented as a
  regularized diagnostic and explicitly not a bound. Full-rank
  configurations are unaffected: `crlb` and `crlb_pseudo` are identical
  there, and every existing test passes unchanged. Downstream, an
  infinite bound propagates to `classify_solvability()` (IMPOSSIBLE) and
  to `efficiency_*` (`inf`), which is the correct answer rather than a
  failure — there is no efficiency to quote against an unbounded target.

- **`voigtfit.crlb` efficiency is now the mean of per-component
  efficiencies, and the oracle step behind it is visible.** The ratio
  was `mean_k(CRLB_k) / (mean_k RMSE_k)²` — a mean of variances over the
  square of a mean of standard deviations — which by Jensen exceeds 1
  for an estimator that attains the bound in every component, unless the
  per-component bounds happen to be equal. At σ=0.5, γ=0.3 that ideal
  value is 1.16 for three peaks at overlap 0.3 and 1.51 for five at
  overlap 0.5, i.e. the artefact is largest exactly where the harness is
  used, and it was indistinguishable from the effects being measured.
  The equal-bounds case returns 1.000, which is why the well-separated
  two-peak smoke test never saw it. `mean_efficiency()` now computes
  `mean_k(CRLB_k / RMSE_k²)`; it is a module-level function so the
  property can be asserted directly rather than only through a solver
  run. Separately, the RMSE behind these numbers is computed after
  `_correct_swaps_ncomp()`, which chooses the component permutation
  minimising error **against the ground truth** — an oracle no estimator
  has, applied to 16–96% of spectra at the overlaps this harness sweeps.
  It remains the primary figure, because without it the RMSE at high
  overlap is dominated by label permutation rather than estimation
  error, but `efficiency_*_unmatched` now reports the same comparison in
  the solver's own component order, and `swap_fraction` sits beside it.
  At the module's own defaults the aggregation change is worth −3% to
  +1%, but the oracle is not small: it raises `efficiency_dE` by 98.7%
  for two peaks at overlap 0.5, 13.6% for five at overlap 0.5, and 6.2%
  for two at overlap 0.8. An earlier draft of this entry said "0–12%"
  and called both "corrections of meaning more than of magnitude"; that
  range came from a reduced grid and does not hold. It survives in the
  message of commit `4b480b6`, which was already published and is left
  as written.
  `EfficiencyResult` also exposes `rmse_*_per` so no averaging is
  hidden. The remaining upstream defect — singular directions inverted
  to zero rather than diverging — is unchanged.

- **`voigtfit.crlb.compute_efficiency_point()` measured its spectra and
  its bound in different amplitude conventions.** The synthetic spectra
  were generated peak-normalised (`generate_ncomp_spectra` defaults to
  `peak_normalize=True`, so amplitude 1.0 means peak height 1) while
  both the Fisher matrix and the solver's own basis are area-normalised
  — `Re[w]/(σ√(2π))`, i.e. amplitude 1.0 means unit integrated area.
  Two things followed. The recorded ground-truth amplitude was wrong by
  `1/V_max - 1`, so `rmse_amp` measured that offset rather than any
  estimator error: 0.917 where 0.0025 was right, at σ=0.5, γ=0.3,
  n_comp=2, SNR 500. And `noise_std` was taken from the stronger
  generated spectra while the Fisher matrix was built from the weaker
  modelled one, inflating every CRLB by the square of the amplitude
  ratio — 2.99× measured over that ensemble, against a single-peak
  nominal `(1/V_max)²` of 3.64; the two differ because the ensemble
  jitters σ, and the figure is configuration-dependent in any case. The
  generator call now passes
  `peak_normalize=False`; nothing else moved. **Every `efficiency_*`
  number this function has ever reported was affected**, in both
  directions: the same well-separated case returned 2.04 before the fix
  and 0.598 after, so the ratio had been sitting above the bound it is
  measured against. The guard was `assert efficiency_dE > 0.01`, which
  is one-sided and could not see an inflated value; it is now a
  two-sided bracket plus an amplitude-bias assertion, both set from
  measurement and verified to fail when the defect is reintroduced
  (0.917 against a 0.05 ceiling, 2.036 against a 1.5 one). The margins
  are not equal: 19.8× on the amplitude assertion, 2.51× on the upper
  end of the efficiency bracket, which is deliberately loose while the
  artefacts in that ratio remain. Two further artefacts in the same ratio — the
  aggregation formula and an oracle relabelling step — remain
  documented and unfixed.

### Added

- **`CompoundDB` now reports where each bundled value came from.**
  `get_provenance(name)` returns a record per field, `get_comparisons(name)`
  the published values found later that disagree with it, and
  `get_investigations()` what was searched and what the search did not
  cover. The data lives in a new bundled table,
  `data/_cache/compounds_provenance.json`, deliberately **separate from
  the values**: `get_properties()` still returns the four numbers and
  nothing else, so no calculation sees a difference, and the CSV rebuild
  path — which knows only the numeric columns — cannot silently drop the
  provenance. The file is hand-authored and has no rebuild source.
  Three distinctions are kept apart because collapsing any of them
  states something untrue. **`availability` vs `origin`**: what is known
  about the basis for a value is not how the value came to be what it
  is. Almost every field is `not_recorded` with an `asserted` origin —
  a number was typed in and the basis was never written down — and a
  test being able to re-derive that number now does not make its origin
  `derived`. **No stored value is `derived`**: the one field this change
  set touched, the Si3N4 molecular weight, is `corrected`, because the
  commit repaired a digit transposition and left the fractional part
  untouched — the value is the one the original entry intended, not one
  authored here. Its record carries the previous value, the commit, and
  the check that settled it, including that the check used the pre-2009
  atomic weights those digits imply rather than IUPAC 2021, which gives
  140.283 (3 ppm away, immaterial to `U = N_v·ρ/M`). **A source vs a later
  comparison**: Shinotsuka *et al.* 2019 is not where SiO2's 2.2 g/cm³
  came from, so it appears under `comparisons`, never as an origin; a
  test fails if any entry with only a comparison acquires a `cited`
  origin. **A property of the material vs a fact about a search**: "no
  published counterpart" describes which three compilations were read,
  and is recorded with its limits under `investigation`. A `phase`
  record sits beside the value fields because density depends on it and
  the table has no phase label — `rutile` for TiO2 and `monoclinic` for
  HfO2 and ZrO2, each recording that the identification is an inference
  made here, but **`unknown` for GeO2**, whose two forms differ by about
  47% in density and which is therefore the largest unrecorded
  uncertainty in the table. What the ambiguity bears on is recorded and
  is not always density: for SiC the polytype densities differ by under
  1% while the band gaps span 2.31–3.26 eV. `tests/test_compound_provenance.py` (18
  tests) pins key parity with `compounds.json`, coverage of every field,
  a mandatory reason on every non-`known` state, the required members of
  a `derived`, `cited` or `corrected` origin, that a comparison reports
  agreement and disagreement accurately — some agree exactly, and an
  exact agreement with a table that carries no sources of its own is not
  corroboration — and that the values themselves are unchanged.

- **`AngularCorrection` now warns when the x-ray incidence angle was
  never stated.** `angular_distribution()` and
  `angular_distribution_unpolarized()` default `xray_from_normal_deg` to
  88°, which is a placeholder for a grazing-incidence geometry and not a
  property of anyone's instrument. Leaving it in place emits a
  `UserWarning`; passing `88.0` explicitly does not, so "I checked" and
  "I never thought about it" are no longer indistinguishable. The
  default is **unchanged** and no returned number moves. It matters
  because the assumption sets the *shape* of the angular dependence, not
  its scale: over one 51°–9° emission fan the peak-to-peak spread of
  `L_dipole` relative to its mean is 85% at 88° incidence, 194% at 60°
  and 216% at 55°.
- The angle conventions are now written down where they are used — a
  geometry sketch and a table of θ (measured), `xray_from_normal`
  (instrument) and ψ (derived) in `docs/API.md` and the module
  docstring, plus the warning that `L_dipole`/`L_unpolarized` take the
  *derived* angle. Handing them θ returns a plausible wrong number
  instead of raising: for β = 1.9255 and θ = 8.78° that is 0.0709 where
  1.4309 is correct, a factor of 20 between two unremarkable values.
  `angular_distribution()` already takes θ and derives ψ itself; it is
  now documented as the entry point rather than a variant.
  `tests/test_angular_geometry.py` (16 tests) pins all of it.
- **The CRLB utilities now state when their bound is a bound.**
  `voigtfit.crlb` presented `Var(θ̂) ≥ [g⁻¹]ᵢᵢ` as unconditional; the
  words *unbiased*, *bias* and *misspecification* did not appear in the
  file. The inequality needs an unbiased estimator of a correctly
  specified model and a non-singular Fisher matrix, and the module also
  conditions on everything outside θ — no background parameter enters
  the Fisher matrix, `mode='3d'` treats the Lorentzian widths as known,
  and the component count is assumed known. Three consequences are now
  written down at the point of use rather than left to be rediscovered:
  eigenvalues at or below `1e-12·λ_max` are inverted to **zero**, so in
  a strongly overlapped configuration the reported `crlb` *understates*
  the bound and can fall as the problem gets harder; `classify_solvability()`
  — which `process_multipeak()` attaches to every result — computes its
  meV figure from a default SNR of 100 and unit amplitudes, not from the
  data being fitted; and `efficiency_*` is moved by an aggregation
  artefact (variances averaged over squared averages of standard
  deviations), by an oracle relabelling step, and by a profile-normalisation
  mismatch between the generated spectra and the Fisher matrix, before
  any property of the solver enters. No returned number changes; the
  three defects named here are documented, not fixed.
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

- **The `/provenance` HDF5 schema design record is no longer published.**
  `docs/hdf5_provenance_phase_b1_design.md` was a dated internal working
  document — an approval and phase log, in the maintainer's working
  language, built on a non-public audit note it cited throughout — which
  the repository's own publication policy excludes. It moves to
  `docs/_internal/`. Five shipped modules cited it as their design
  contract; they now cite it the way this project already cites its
  non-public development log, and `CONTRIBUTING.md` documents that
  convention. The rules the note fixes are stated where they are
  implemented and asserted by `tests/test_provenance_schema.py`, so no
  behaviour, schema or test changed. Internal documents are now excluded
  by directory rather than by listing each filename in `.gitignore`,
  which had put those filenames in a public file and did not catch new
  ones.

- **Documentation only: `docs/DATA_SOURCES.md` now states where the
  `compounds.json` values actually came from.** The entry read "in-house
  curation" and "project-original selection", which described the
  *choice* of entries but implied a curation process the file never had.
  The values were hand-entered in 2020–2021 from Wikipedia and commonly
  quoted literature figures, with no per-entry source recorded — the
  absence of citations was already flagged, but not its cause. A new
  section separates the two questions the entry had run together: the
  rights position is clean (the set reproduces no published table, and
  is not derived from the TPP-series parameter compilations), while
  accuracy is weak and uneven. Five of the eleven compounds appear
  somewhere in that series — Al2O3, GaAs, SiC and SiO2 in Shinotsuka
  *et al.* 2019, and Si3N4 in Tanuma, Powell &amp; Penn 1991 — and where a
  published set exists the hand-entered parameters give λ within 2.6% of
  it. TiO2, HfO2, ZrO2, Ta2O5, SrTiO3 and GeO2 appear in none of the
  three compilations checked, so no published parameter set is available
  there. A measured sensitivity table (TPP-2M at 1 keV, inside the
  fitted range) shows
  where this costs anything: ρ ±10% moves λ by 2–3% for most oxides but
  5% for HfO2, whose bundled 9.68 g/cm³ is the bulk monoclinic density
  and overestimates amorphous ALD films by 8–14%, biasing λ 3.5–6% in
  one direction. No bundled value changed.

- **Documentation only: "Stage 2" now says what it is and what it is
  for.** Two unrelated constructions here have two phases, and both were
  written "2-Stage": the Stage 1 / Stage 2 screening pipeline, and the
  γ-calibrated Dict3D → Dict2D solver, which contains no Stage 2. The
  latter is now "two-phase" throughout, with the distinction stated
  where a reader meets either. Stage 2 itself is described consistently
  as a **legacy compatibility fallback** that may change in a future
  release: `enable_stage2` still defaults to `True` so existing callers
  keep their behaviour — a bare `HybridPipeline(cache)` does route its
  anomalous spectra through it — while the 4-step and adaptive
  dictionary solvers are the recommended route for new code. No
  identifier, default, or executable line changed; the parsed source is
  identical once docstrings are removed.
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
- **`AngularCorrection.L_full`'s convention is flagged as unresolved.**
  Its dipole term is the polarization-*averaged* coefficient
  `1 − (β/2)P₂` applied to an angle documented as being from the photon
  direction, while its non-dipole term is the linearly polarized form,
  which the module docstring writes with the angle measured from the
  polarization vector. The two conventions cannot both apply, and the
  module's own statement that `L_unpolarized` is `L_full` averaged over
  ε ⊥ k does not hold — that average sends cos φ to zero and returns
  `L_dipole`. Settling it needs the printed formula in Trzhaskovskaya &
  Yarzhemsky, ADNDT **119**, 99 (2018), which has not been read, so
  **nothing was changed**: the docstring and `docs/API.md` say the
  convention is under review, and the current values are pinned as a
  record — explicitly not as a claim of correctness — so that resolving
  it produces a visible diff. Not academic: for Si 1s at 9.25 keV the
  non-dipole term runs from 4.5% of the dipole value to 132% across one
  emission fan. `L_dipole` and `L_unpolarized` are unaffected; each
  matches its form in the module docstring.
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
