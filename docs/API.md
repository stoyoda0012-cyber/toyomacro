# Core API reference

The entry points a researcher needs, with array shapes, units, and
backend behavior stated explicitly. Everything below is importable
from a plain `pip install toyomacro` (NumPy backend) — the MLX GPU
path activates automatically on Apple Silicon when usable.

**Conventions.** Energy in eV; `sigma` is the Gaussian standard
deviation (eV); `gamma` the Lorentzian half-width (eV);
FWHM ≈ 0.5346·(2γ) + √(0.2166·(2γ)² + (2.355σ)²). A "spectrum" is a
1-D intensity array of `n_energy` channels. **Batch APIs are
row-major**: `Y` has shape `(n_spectra, n_energy)` unless a method
name or docstring says otherwise (the lower-level
`HybridPipeline.stage1_screening` is column-major,
`(n_energy, n_spectra)` — its docstring states this).

**"Stage 2" vs "two-phase".** Two unrelated things here have two
phases. **Stage 2** always means one thing: the legacy compatibility
fallback — a Gauss-Newton refinement that `HybridPipeline` runs on the
spectra Stage 1 flags as anomalous. The 4-step and adaptive dictionary
solvers are the recommended route for new code: every benchmark in this
repository sets `enable_stage2=False`, and `FastVoigtFitter` disables
it. The constructor default remains `True` so existing callers keep
their behaviour, so a bare `HybridPipeline(cache)` still routes
anomalies through it. Being a compatibility path, it **may change in a
future release**. The
**γ-calibrated two-phase solver** (Dict3D fixes a global γ, then
Dict2D fits precisely) is a separate construction and contains no
Stage 2.

## API stability

The package is larger than its supported surface. Three tiers, and the
rule that separates them:

**Supported.** Every name in the `__all__` list of `toyomacro`,
`toyomacro.core`, `toyomacro.lineshape`, `toyomacro.background`,
`toyomacro.io`, `toyomacro.data`, `toyomacro.fitting`, and
`toyomacro.voigtfit`, plus the submodule entry points given with a
call signature in the sections below (`dictionary_solver`,
`multipeak_solver`, `gvrt_service`, `spectra_generator`). These are
covered by the test suite, exercised by `examples/`, and will not
change signature or semantics without a minor-version bump and a
changelog entry.

Two mechanical checks: `python -c "import toyomacro.voigtfit as v;
print(v.__all__)"` enumerates the exported surface, and a submodule that
you had to name explicitly (`from toyomacro.voigtfit.rank_diagnostics
import …`) is not exported — see the next tier before depending on it.

**Experimental.** Research modules that are deliberately *not*
re-exported from any `__init__`, reachable only by explicit submodule
import. They are tested, but their API may change or disappear in any
release, and no CLI or documented workflow depends on them:

| Module | What it is |
|---|---|
| `voigtfit.rank_diagnostics` | numerical rank / conditioning of structured Voigt designs |
| `voigtfit.model_selection` | AIC / AICc / BIC scoring of already-fitted candidates |
| `voigtfit.exact_k` | component-count (model order) selection |
| `voigtfit.auto_grouping` | Fisher-justified component grouping for hierarchical fits |
| `voigtfit.fisher_*`, `voigtfit.crlb` | Fisher information, coordinate transforms, CRLB bounds (referenced in §4 as a diagnostic, not a stable entry point) |
| `voigtfit.gp_compact`, `voigtfit.gp_reference` | Golub-Pereyra / LM reference layer behind `newton_jacobian_mode` |
| `voigtfit.dictionary_solver_3d` | δE × δσ × δγ dictionary (2-D version is the supported path) |
| `voigtfit.matlab_bridge`, `voigtfit.prefetch_pipeline`, `voigtfit.simulation` | workflow adapters and validation harnesses |
| `data.transmission` | Scienta analyzer transmission adapter (§5); reads user-supplied vendor data, no data bundled |
| `data.elastic_scattering` | overlayer-thickness effective attenuation length from the single-scattering albedo; requires caller-supplied IMFP *and* TRMFP, no TRMFP or albedo data bundled |

Likewise `multipeak_solver`'s `newton_jacobian_mode` is experimental for
any value other than the default `"raw"`.

**Internal.** Underscore-prefixed names, `voigtfit.benchmarks.*`, and
anything not listed above. Benchmarks are runnable and their record
format is stable enough to compare across hosts, but they are tools,
not a library API.

**Not in this repository.** The desktop GUI (PySide6), the FastAPI/React
web front end, and the ARXPS depth-profile solver are developed
separately and are not part of the published package — `toyomacro.gui`,
`toyomacro.api`, and `toyomacro.depth` do not exist in an installed
copy. Nothing in the supported surface imports them, so the boundary
above does not move if they are released later.

## Backend selection

```python
from toyomacro.voigtfit import mlx_installed, mlx_usable, require_mlx
```

- `mlx_usable()` — True only if a probe kernel actually executed. The
  probe runs on **MLX's default device, whatever that is**; it answers
  "can MLX execute work here", not "is this a Metal device". Installed
  MLX that cannot execute (headless or virtualized host, broken
  driver/toolchain) returns False and every solver silently uses the
  NumPy backend (numerically identical to ~1e-3 relative in float32).
- `require_mlx()` — raises an actionable `RuntimeError` when the GPU
  path is unavailable; call it first if your workload depends on GPU
  throughput.
- `TOYOMACRO_DISABLE_MLX=1` — environment variable forcing the NumPy
  backend (useful for A/B validation).

### What is distributed, and what merely runs

The device-agnostic probe and the distribution boundary are separate
questions, and the answer differs:

| Backend | Status | Installation |
|---|---|---|
| NumPy / CPU | Supported | base install; runs anywhere CPython runs |
| MLX / Apple Metal | Supported accelerator | `pip install "toyomacro[mlx]"` |
| MLX / CUDA | Experimentally validated, **not a supported install target** | manual; no extra ships for it |

Apple Metal is the accelerator this project distributes and quotes
numbers for. The `[mlx]` extra pins `mlx-metal` and is Apple-only.

**What CI actually covers.** The hosted runners are headless, so no CI
job exercises an accelerator. The Ubuntu/macOS × 3.11/3.12 matrix
installs without the `[mlx]` extra and tests the NumPy path; one macOS
job installs MLX and runs the whole suite with `TOYOMACRO_DISABLE_MLX=1`,
which certifies the *fallback*, not the GPU. The accelerated Metal path
is covered by the MLX↔NumPy parity tests, which run only on a machine
where `mlx_usable()` is True — the maintainer's, and (once)
the CUDA box.

MLX's own CUDA backend also satisfies `mlx_usable()`, and the MLX↔NumPy
parity suite has been run end to end against it once — on an RTX 5070
Laptop under WSL2, where the `requires_mlx` blocks genuinely executed
(81 skips collapsed to 1) and passed within existing tolerances. That is
a validation result, not a support claim. Before relying on it, read
[`CUDA_BACKEND_POC.md`](CUDA_BACKEND_POC.md); the load-bearing caveats:

- **No `[cuda]` extra.** Setup is manual, and `mlx[cuda13]` does not
  pull the CUDA headers it needs to JIT — see the recipe in that
  document.
- **TF32 must be disabled.** MLX's CUDA backend runs float32 matmul in
  TF32 by default, silently: ~1000× the matmul error of true fp32,
  costing ~9.4 dB PSNR and 3× the δσ RMSE end to end, and flipping
  near-tie dictionary argmax indices. Set `MLX_ENABLE_TF32=0`.
- **Batches above 65,535 crash the multipeak alternating-projection
  solver** (upstream MLX batched-GEMV grid-dimension limit,
  [mlx#3858](https://github.com/ml-explore/mlx/issues/3858)). Chunk to
  ≤ 65,535. Metal has no such limit.
- **No CUDA CI**, and no committed performance record. On the one
  machine measured, streaming/amplitude-only paths lost to that host's
  CPU while compute-dense dictionary AP won 4–6×; MLX's `sm_120` GEMM
  measured ~36× below what cuBLAS delivers on the same GPU. Do not
  generalize any of that.

## 1. Lineshape

```python
from toyomacro.voigtfit import voigt_profile
y = voigt_profile(energy, center, sigma, gamma)   # (n_energy,)
```

Exact Voigt via the Faddeeva function (scipy.wofz). Unit peak-area
normalization: multiply by an amplitude to scale.

## 2. Batch fitting, high level

```python
from toyomacro.fitting import FastVoigtFitter, FastFitConfig
fitter = FastVoigtFitter(FastFitConfig(mode="balanced"))
result = fitter.fit_batch_rowmajor(
    spectra,          # (n_spectra, n_energy) float32
    energy,           # (n_energy,) eV
    centers, sigmas,  # (n_comp,) nominal configuration
    gamma,            # scalar
)
```

| mode | solver | recovers | typical rate* |
|---|---|---|---|
| `preview` | amplitude-only projection | amplitude | ~10⁸ spec/s |
| `fast` | 4-step Taylor projection | + δE, δσ (small shifts) | ~10⁶ spec/s |
| `balanced` | dictionary + parabola | + δE, δσ (wide range) | ~5×10⁶ spec/s |
| `precise` | adaptive | best quality | ~4×10⁵ spec/s |

*Recorded on Apple M3 Max (MLX); see committed records under
`paper/figures/results/` and regenerate with the benchmarks below.
Without a GPU the same code runs on NumPy at roughly a quarter of these
rates — on the identical seeded problem, `balanced` measures 6.6 M
spec/s with MLX and 1.6 M spec/s with `TOYOMACRO_DISABLE_MLX=1`, at
indistinguishable accuracy (paired records:
`solver_comparison.json` / `solver_comparison_numpy.json`).

**Solver selection.** Start with `balanced`. Use `preview` when
centers/widths are known and only amplitudes are needed (composition
maps). Use `fast` when shifts are guaranteed small (|δE| ≲ 0.3σ —
Taylor validity). Escalate to `precise` for publication-grade
parameter maps or difficult overlaps.

## 3. Batch fitting, low level

```python
from toyomacro.voigtfit.dictionary_solver import (
    build_dictionary_2d, solve_dict2d_parabola)

cache = build_dictionary_2d(energy, centers, sigmas, gamma,
                            dE_range=(-0.15, 0.15),
                            dsigma_range=(-0.05, 0.05),
                            N_dE=13, N_ds=7)          # setup cost, once
amp, chi2, dE, dsigma, idx = solve_dict2d_parabola(Y, cache)
```

- Setup vs. steady state: dictionary construction is a one-time cost
  (seconds) amortized over the batch; benchmark records report it
  separately (`setup_s`).
- **Batch size matters, and the two backends want opposite things.**
  On one M3 Max measurement of `solve_dict2d_parabola` (151 channels,
  13×7 dictionary), MLX climbs with batch size — 0.8 M spec/s at 2 K,
  6.1 M at 200 K, 7.3 M at 1 M and still rising, since small batches
  are dominated by dispatch overhead. NumPy runs the opposite way: it
  peaks near **50 K spectra** (1.9 M spec/s) and then *degrades* as the
  working set outgrows cache (1.4 M at 1 M). Feed the GPU path the
  largest batch that fits; chunk the NumPy path into tens of thousands
  of spectra. Both are the same algorithm, so the ~4× backend gap holds
  at either side's best size.
- Returns per-spectrum arrays `(n_spectra,)`; `dE`/`dsigma` are
  offsets from the nominal configuration, in eV.

Multipeak clusters (overlapping components fitted jointly):

```python
from toyomacro.voigtfit.multipeak_solver import process_multipeak
```

runs alternating projection with joint block deflation; a pure-NumPy
implementation is used automatically when MLX is not usable.

### Per-spectrum reference fitter

```python
from toyomacro.voigtfit import VarProFitter
```

Classical variable projection [Golub & Pereyra 1973] on SciPy's
optimizer: centers and widths are optimized numerically while
amplitudes are eliminated analytically. It fits **one spectrum at a
time** and is not part of the high-throughput path — nothing in
`HybridPipeline`, the dictionary solvers, or `FastVoigtFitter` calls
it. Use it as an independent oracle when you want a conventional
optimizer's answer for a handful of spectra, and as the estimator whose
misspecification bias is characterized in
`examples/04_projection_law_validation.py` (its effectively
unregularized inner solve is what makes it match the projection law's
assumptions; solvers with other regularizers are only directionally
described — see `docs/projection_law_validation.md`).

## 4. Quality flags and failure modes

- Every solver returns per-spectrum `chi2`; anomalously high values
  flag pixels that need escalation to a slower solver stage or manual
  inspection.
- Shift estimates at the edge of the dictionary range indicate the
  search window was too narrow — rebuild with a wider `dE_range`.
- Taylor solvers (`fast`) degrade smoothly but systematically beyond
  |δE| ≈ σ: two-lobed residuals and underestimated widths are the
  signature. Prefer `balanced` when in doubt.
- At peak-count SNR ≲ 1 (peak signal equals shot noise), parameter
  recovery collapses regardless of solver — the CRLB utilities
  (`toyomacro.voigtfit.crlb`) quantify this limit for your
  configuration, and Figure 2 of the paper shows the collapse
  ordering (width → amplitude → position).

## 5. Quantification inputs

```python
from toyomacro.data import CrossSection, IMFP, BindingEnergy
```

Photoionization cross-sections (Scofield, Yeh–Lindau, Trzhaskovskaya)
ship as tables; TPP-2M inelastic mean free paths are computed from the
published formula. Provenance and licensing for each is stated in
[`DATA_SOURCES.md`](DATA_SOURCES.md). `CrossSection.lookup()` uses the
process default table — Yeh–Lindau unless `set_default_table()` changed
it — so pass `table=` explicitly whenever the choice matters, and read
`unit_info()` before comparing anything across tables.

**A bare orbital is the whole doublet.** `lookup('Si', '2p', hv)` returns
σ(2p1/2) + σ(2p3/2) — what a measured peak envelope contains. Each
component is evaluated at the requested energy and then added; summing
the stored arrays and interpolating once is not equivalent, because the
interpolator fits one polynomial across an element's whole tabulated
range. Pass `'2p3/2'` for a single component.

`BindingEnergy` uses the opposite convention for bare labels (the main
line, j = l+1/2), deliberately: a binding energy is a position, of which
a doublet has two, whereas a cross-section adds over the subshells that
make up the envelope.

Three consequences worth knowing before you trust a number:

- **Each component carries its own threshold.** Between the two
  thresholds of a well-split doublet only the lower-BE member is
  ionizable, and the bare label returns that component alone — which is
  what the spectrum contains there. W 3d5/2 opens at 1809 eV and 3d3/2
  at 1872 eV, so `lookup('W', '3d', 1850)` is the 3d5/2 value.
- **A j-resolved request against a bare-keyed table returns `None`.**
  Yeh–Lindau — the *default* — stores no j-resolved key for any of its
  105 elements, so `lookup('Au', '4f7/2', hv)` with no `table=` is
  `None` rather than the total split by an assumed branching ratio. The
  statistical ratio is not exact (Scofield's own Si 2p3/2 : 2p1/2 is
  1.966, not 2) and a split value would be indistinguishable from a
  tabulated one.
- **A missing component is not guessed at.** Trzhaskovskaya
  half-lists 38 non-s subshells, all open valence shells whose upper-j
  component would be empty, and the configuration exceptions (Cr 3d⁵,
  Mo 4d⁵, Eu 4f⁷, Re 5d⁵) carry both. That is consistent with "listed
  iff occupied" — but the inclusion rule has **not** been read from
  ADNDT 77/82, so a half-listed bare subshell returns `None` rather than
  treating the absence as a zero. The component that *is* listed remains
  available as a j-resolved request. Scofield lists both
  components of every subshell it carries, so an absent one there would
  be a data gap and the subshell refuses rather than under-counting.
  These rules rest on different evidence and are pinned separately by
  `tests/test_cross_section_spin_orbit_limits.py`.

Scofield cross sections **already incorporate the report's assumed
fractional subshell occupations** (UCRL-51326 Table A1: B 2p is
NE 0.33 + 0.67 = 1 electron, C 0.67 + 1.33 = 2, N 1 + 2 = 3,
O 1.33 + 2.67 = 4). Sum the components as stored; weighting them by
occupancy again double-corrects.

**The tables are not on a common scale, and one unit is not
established.** Ask rather than assume:

```python
CrossSection.unit_info('scofield')
# {'table': 'scofield', 'unit': 'Mb', 'inferred_unit': None,
#  'status': 'confirmed', 'values_rescaled': False}
CrossSection.unit_info('trzhaskovskaya')
# {'unit': None, 'inferred_unit': 'kb', 'status': 'inferred', ...}
```

Yeh–Lindau and Scofield are megabarn, each read from its source.
Trzhaskovskaya values run ~10³ larger; the evidence points to kilobarn
but the ADNDT 77/82 table heading has not been read, so `unit` stays
empty, the candidate travels in `inferred_unit`, and no conversion
factor is offered. Nothing stored is rescaled on an inference. Ratios
within one table cancel the unknown *unit* factor — and only that. For
`trzhaskovskaya` a second, independent problem survives any ratio: the
table is gridded in photoelectron energy but interpolated as photon
energy, so two lines with different binding energies are read at
different effective points. Absolute cross-sections, σ×λ sensitivities
and cross-table comparisons carry both. See
[`DATA_SOURCES.md`](DATA_SOURCES.md), which also records why the
2018/2019 tables' unit was checked separately rather than inherited.

`CrossSection.get_rsf()` returns a **cross-section ratio**
σ₁(hν)/σ₂(hν) against a reference line, and nothing more. Despite the
name it is not a complete relative sensitivity factor and not an
average-matrix RSF: no IMFP or EAL, no elastic scattering, no angular
distribution, no polarization or geometry, no analyzer transmission, no
matrix averaging. (Subshell occupancy is *not* a missing factor — it is
already inside the tabulated σ, which is why the `2p3/2` : `2p1/2` ratio
comes out near 2:1.) A vendor RSF table is a different quantity; the two
are not interchangeable. The name is kept for backward compatibility and
may be revisited after v0.1.0.

The same caveat applies one level up. What this package can assemble is
an **intrinsic** sensitivity,

```
S_intrinsic = σ(hν) × λ(KE)
```

which is what the MCP `calculate_sensitivity` tool and
[`examples/03_quantification.py`](../examples/03_quantification.py)
compute. A physical AMRSF contains, at minimum,

```
AMRSF ∝ σ(hν) × L(θ) × λ_average-matrix(KE) × Q_elastic
```

where σ already carries the subshell occupancy, `L(θ)` is the
angular-asymmetry factor at the source/analyzer angle θ for the given
x-ray polarization, `λ_average-matrix` is an IMFP or EAL for the
averaged matrix rather than the specific compound, and `Q_elastic`
corrects for elastic scattering. (θ here is the geometric angle —
`psi_deg` / `alpha_deg` / `phi_deg` in `AngularCorrection`. Do not
confuse it with the non-dipole asymmetry *parameters* β, γ and δ that
`L(θ)` is computed **from**; γ means the parameter, never the angle.)
The analyzer transmission function is a further, **separate** instrument
response, not part of the physical AMRSF.

**No complete AMRSF is assembled here**, and no third-party AMRSF table
is bundled. Two of the four factors do have implementations you can
compose yourself:

| Factor | Where | Status |
|---|---|---|
| σ(hν) | `CrossSection.lookup()` | supported, with the two table caveats above |
| angular/polarization | `AngularCorrection` (`data.angular_correction`) — dipole, unpolarized and full β/γ/δ forms from the bundled Trzhaskovskaya 2018/2019 tables. **only j-resolved input is defined** | **experimental** — see below |
| λ | `IMFP.tpp2m()` — for the *specific* compound, not an averaged matrix | supported |
| `Q_elastic` | `data.elastic_scattering` — needs a caller-supplied IMFP/TRMFP pair | experimental |

**Angles: which one you have, and which one the formulas want.** Three
angles are involved, and only the first is something you measured.

```
 hν                        normal        e⁻
                              ^
   \                          |        /
      \                       |       /
         \.-------- ψ = 83° --|-----./
            \                 |     /
               \              |    /
                 α_xray = 56° | θ = 27°
                     \        |  /
                        \     | /
                           \  |/
    --------------------------+-----------------------
                       sample surface
```

Angles in the sketch are magnitudes, as drawn. The values are
representative, not any particular instrument. (Apparent angles depend
on the font's character-cell aspect ratio; the drawing assumes 2:1.)

| Symbol | Definition | Where it comes from |
|---|---|---|
| `theta` | Electron emission angle from the surface normal | **Your data** — the analyzer's angle axis. Often built as `C − analyzer_axis_value` for a stated centre `C`, so pin down `C` too |
| `xray_from_normal` | X-ray incidence angle from the surface normal | **Your instrument** — a value to confirm, never to inherit |
| `psi` / `alpha` | Angle from the photon direction to the emission direction | **Derived**. As magnitudes: `psi = alpha_xray + theta` when source and analyzer sit on opposite sides of the normal, as drawn (56° + 27° = 83°); `psi = \|alpha_xray − theta\|` on the same side. This is what `L_dipole` / `L_full` / `L_unpolarized` take |

The API takes a **signed** incidence angle instead:
`angular_distribution()` computes `psi = xray - theta`, so pass a
negative `xray_from_normal_deg` for the geometry drawn above
(`psi = -83°` here). `L_dipole` and `L_unpolarized` are even in `psi`,
so the sign cannot reach them. `L_full` is not even: its non-dipole
term flips with the sign. Over the full range it is negative for 23% of
`psi` — on −178.8°…−137.8° and −42.2°…−1.2° for the Si 1s 9.25 keV
parameters — where it is not a correction factor at all. The drawn
geometry sits outside those bands, but nothing in the API checks, so
verify the sign before using `L_full`.

Use `angular_distribution()` or `angular_distribution_unpolarized()`:
they take `theta` and derive the rest. Calling `L_dipole` with `theta`
returns a plausible wrong number rather than raising — for β = 1.9255
and θ = 8.78° it gives 0.0709 where 1.4309 is correct, a factor of 20,
with both values sitting unremarkably between 0 and 2.

Leaving `xray_from_normal_deg` at its 88° default now emits a
`UserWarning`. The default is a placeholder, not an instrument value,
and the assumption sets the *shape* of the angular dependence: over one
51°–9° emission fan the peak-to-peak spread of `L_dipole` relative to
its mean is 85% at 88° incidence, 194% at 60° and 216% at 55°. The
default is unchanged for compatibility; the warning exists so that no
result is produced without the geometry having been stated once.

**Sign and reference axis.** `L_dipole` is `1 − (β/2)·P₂(cos ψ)` — the
**unpolarized** form, with the angle taken from the beam **k**. The form
quoted throughout the synchrotron literature, `1 + β·P₂(cos θ_ε)`,
differs in *both* sign and reference axis: θ_ε is measured from the
**polarization vector ε**. Transcribing one for the other is a real
error. `L_full` currently mixes the two conventions — see immediately
below.

`AngularCorrection` is experimental for four reasons.

**`L_full`'s convention is unresolved.** Its dipole term is the
polarization-*averaged* coefficient `1 − (β/2)P₂` applied to an angle
documented as being from the photon direction, while its non-dipole term
is the linearly polarized form, which the module docstring writes with
the angle taken from ε. The two cannot both be right, and the module's
own claim that `L_unpolarized` is `L_full` averaged over ε ⊥ k does not
hold: that average sends cos φ to zero and returns `L_dipole`. Settling
it requires the printed formula in Trzhaskovskaya & Yarzhemsky, ADNDT
**119**, 99 (2018), which has not been read. **Nothing has been changed**
— `tests/test_angular_geometry.py` pins the present values as a record,
explicitly not as a claim that they are right, so that resolving the
question produces a visible diff. `L_dipole` and `L_unpolarized` are
unaffected; each matches its form in the module docstring. Prefer them
until this is closed.

**Only j-resolved input is defined.** A bare label such as `'2p'` is
not rejected at runtime yet — it is simply undefined: it resolves to whichever j component the
internal suffix search finds first, which is the same miscounting
`CrossSection` was fixed for. It has not been given the same treatment
because the fix is not analogous — the 2018/2019 tables give σ for
*completely filled* subshells (stated on the digitization's own
explanation sheet), and β, γ and δ are per-component quantities that do
not add. Use `AngularCorrection.lookup('Si', '2p3/2', hv)`. Bare input is deprecated;
rejecting or warning on it, and deciding what it should mean, is
deferred rather than guessed at.

**Its β/γ/δ grid starts at 1500 eV** (2018 outer shells 1.5–10 keV; 2019
inner shells 2–18 keV), and lookups below that clamp to the edge value
instead of refusing, so `lookup('Si', '2p3/2', 1486.6)` returns the
1500 eV parameters unchanged. Al Kα — the most common lab source — sits
just under that floor. That is the same failure mode flagged for σ, λ
and analyzer transmission elsewhere in this section, and it is not
signalled in the return value.

**And it has no example.** Its tests pin the limits and conventions
described above — the grid clamp, the magic-angle identity, the θ↔ψ
conversion and its cost, the geometry warning, and `L_full`'s present
values — which constrains the behaviour without making the surface
stable.

Composing these factors is the caller's responsibility, and doing so
still does not reproduce a vendor AMRSF: the averaging convention, the
reference line, and the normalization basis all have to match the table
you want to compare against. Treat `S_intrinsic` as a relative figure of
merit; absolute composition from it alone is not traceable.

`IMFP.tpp2m()` takes the electron's kinetic energy **in the solid**, in
eV, and returns nanometres (Eqn (3) of the paper is in Ångströms). No
work function or photon energy is subtracted internally — convert from
binding energy before calling. The paper defines `Eg` "for
non-conductors" and prescribes nothing for conductors; 0 is this
package's default and the usual convention. Pass it for anything else.

Tanuma, Powell & Penn fit the formula over **50–2000 eV** and state that
"Eqns (3) and (4) should not be used for energies greater than 2000 eV",
with the largest deviations from
optically-derived IMFPs below 200 eV. `tpp2m()` does not refuse energies
outside that window — HAXPES estimates are a legitimate use — and
`data.imfp.TPP2M_FITTED_RANGE_EV` exports the bounds so callers can
decide for themselves.

What is returned above 2 keV is the **non-relativistic** form — which
the same authors later reported "useful for energies between 50 eV and
30 keV" (Shinotsuka et al., *Surf. Interface Anal.* **47**, 871 (2015)),
on the evidence of 41 elemental solids, so that reassurance does not
extend to compounds on its own.
That paper also gives a relativistic form for higher energies, using the
same parameters plus a factor α(T); it is **not** implemented here, and
omitting it overestimates the IMFP by roughly 1.8% at 7.4 keV and 7% at
30 keV. Small against the formula's ~10–19% RMS, but one-sided.

Material parameters from `CompoundDB` are curated data with their own
provenance and their own errors, independent of the formula's fidelity.
`CompoundDB.get_provenance(name)` reports, per field, what is known
about where each value came from; `get_comparisons(name)` gives
published values found later that disagree with it, and
`get_investigations()` says which compilations were searched and what
the search did not cover. `get_properties()` is unaffected — it still
returns the four numbers and nothing else.
Note also that the 18.9% RMS figure for inorganic compounds — the group
SiO₂ belongs to — is for a group that was *excluded* from the TPP-2M
fit, on the authors' judgement that its optical data was less reliable.

`IMFP.sampling_depth()` is likewise IMFP-based: the straight-line
approximation at normal emission, excluding elastic scattering. Do not
correct it with the module below — those slopes are for the
overlayer-thickness attenuation length, and sampling/information depth is
a separately defined quantity with a different slope.

For that attenuation length see `data.elastic_scattering`
(experimental). It takes an IMFP/TRMFP pair and a required `model`,
because the L/IMFP slope differs between unpolarized XPS (0.738) and
linearly polarized HAXPES (0.836) — a 2.4% difference in the result for
gold at 7.4 keV. Supplying both lengths keeps the albedo tied to the IMFP
it is applied to, but nothing checks that the two came from the same
source, material and energy; that stays with the caller.

There is no `IMFP.attenuation_length()` — a method that multiplied the
IMFP by a fixed 0.9 under that name was removed rather than published.

**Analyzer transmission** is an adapter, not shipped data:

```python
from toyomacro.data.transmission import TransmissionFunction

TransmissionFunction.available()        # is user data present?
T = TransmissionFunction.load(slit=0.5) # Scienta XOP TSV, log-log interp
T = TransmissionFunction.from_power_law(alpha=-0.35)  # fallback
T(ek_ep)                                # transmission in mSr
```

Pass energy is **not** a table dimension. A curve is selected by
`(analyzer, mode, slit, spot)` — slit and spot are physical apertures in
**mm** — and evaluated at the dimensionless ratio KE/Ep using whatever
Ep you acquired at. Each curve covers a finite ratio range
(`ek_ep_range`); outside it the value is clamped to the endpoint, so a
wide kinetic-energy sweep can silently run flat at the ends. Check
`ek_ep_range` against `KE_max/Ep` before trusting a correction. HAXPES
kinetic energies reach ratios far above those of curves measured for
soft-x-ray work, where that clamp is extrapolation under another name.

Apply transmission on **one** side of the quantification. Either
correct the spectrum,

```
I_corrected  = I_measured / T(KE/Ep) × 1000
quantity     ∝ I_corrected / S_intrinsic
```

or correct the sensitivity and leave the spectrum raw,

```
S_instrument = S_intrinsic × T(KE/Ep) / 1000
quantity     ∝ I_measured / S_instrument
```

These are algebraically identical. Doing both — dividing a
transmission-corrected spectrum by a transmission-corrected sensitivity
— applies T twice, and because T is energy-dependent the error does not
cancel between lines. The factor 1000 is the mSr→Sr conversion for
these particular curves, not a constant that transfers to transmission
data stored in other units.

Vendor-measured curves are **not redistributed** — point
`TOYOMACRO_SCIENTA_DATA_DIR` at your own
`ScientaXOP/Transmission/data` folder (read at call time, so setting it
mid-session works) and `load()` will find the matching `(slit, spot)`
configuration. Start from `list_analyzers()` / `list_configs()`: a slit
width that is not on the grid raises rather than rounding to the
nearest file, since silently applying the neighbouring calibration is
the expensive kind of mistake.

Without that directory, `from_power_law()` gives a smooth
T ∝ (Ek/Ep)^α stand-in. That is a *shape* assumption, not a
calibration — across one real EW4000/Hipp3/R3000 set the fitted
exponent ranges from about −0.71 to −0.03 (median ≈ −0.26), so no
single default stands in for a specific slit/spot/mode. Absolute
quantification from the fallback is not traceable; use it for relative
trends only.

This module is experimental in the sense of the stability section — it
is not re-exported from `toyomacro.data` and no other part of the
package calls it.

## 6. Synthetic data and GVRT

```python
from toyomacro.voigtfit.spectra_generator import (
    SpectraGenerator, add_poisson_noise,
    level_to_peak_snr, level_to_peak_lambda)
from toyomacro.voigtfit.gvrt_service import GVRTService, GVRTConfig
```

Noise severity `level` is dimensionless: peak-count SNR = 10⁴/level,
peak Poisson mean = (10⁴/level)². `GVRTService.run(image, config)`
executes the image → spectra → fit → image round trip and returns
per-channel PSNR — the package's end-to-end accuracy check.

## 7. Reproducible benchmarking

```bash
# kernel throughput with full provenance (JSON committed under
# paper/figures/results/)
python paper/figures/make_figure1_bottleneck.py --measure

# same-problem comparison vs scipy/lmfit
python -m toyomacro.voigtfit.benchmarks.solver_comparison_benchmark \
    --out my_comparison.json

# GVRT round trip from the CLI
python -m toyomacro.voigtfit.cli gvrt --solver dict2d_parabola
```

Every benchmark writes machine-readable records (timestamp, git
commit, OS, library versions, problem size, individual timings,
median + range) so numbers in the paper can be re-derived on any
host.
