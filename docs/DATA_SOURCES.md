# Bundled reference data — provenance and data-rights review

The reference tables under `src/toyomacro/data/_cache/` are published
scientific data, bundled and redistributed here as machine-readable JSON.

The project applies a **proportionate, source-specific data-rights
review**. Absence of an explicit open-data licence does not by itself make
factual scientific values non-redistributable; equally, factuality alone
does not settle the question. Where a source carries an express rights
notice, it is reviewed on its own terms.

What is bundled is numerical fact transformed into this project's own
machine-readable schema. No publisher typography, explanatory text,
figures, or vendor databases are reproduced. Sources with explicit
restrictive redistribution terms are not bundled (see *Deliberately NOT
bundled*).

Criteria applied:

- factual numerical values only
- project-specific machine-readable transformation
- full source citation and provenance
- no publisher typography, prose, figures, or binaries
- explicit vendor/database redistribution restrictions are respected
- source-specific review where an express rights notice exists

This document records provenance and the review the project has carried
out. It is not legal advice, and it does not assert that permission has
been granted where none is stated.

| File | Contents | Source | Data-rights rationale |
|---|---|---|---|
| `scofield.json` | Photoionization cross sections σ(hν), 1–30 keV, Z = 1–101 | Scofield, *Theoretical photoionization cross sections from 1 to 1500 keV*, UCRL-51326, Lawrence Livermore Laboratory (1973). DOI: 10.2172/4545040 | U.S. government-sponsored scientific report; factual numerical data transformed and fully cited. The OSTI record (biblio/4545040) displays no rights, copyright or access-limitation statement, and gives DOE contract W-7405-ENG-48. Bundled cache truncated at 30 keV (covers all lab / HAXPES sources). |
| `cross_section.json` | σ(hν) at 16 photon energies, 10.2 eV–8.05 keV | Yeh & Lindau, *At. Data Nucl. Data Tables* **32**, 1 (1985). DOI: 10.1016/0092-640X(85)90016-6 | Factual numerical values, transformed into this project's schema and fully cited. No express restrictive notice applies to the values themselves. (The same dataset is redistributed by other open-source projects, e.g. `galore`, JOSS 2018 — noted as context, not as a grant of permission.) |
| `trzhaskovskaya.json` | Relativistic σ, 10-energy grid 0.1–10 keV, Z = 1–100 | Hand-digitized (in the MATLAB Toyomacro era) from Trzhaskovskaya, Nefedov & Yarzhemsky, *At. Data Nucl. Data Tables* **77**, 97 (2001), DOI: 10.1006/adnd.2000.0849 (Z = 1–54) and **82**, 257 (2002), DOI: 10.1006/adnd.2002.0886 (Z = 55–100); the 10 keV column comes from a later extension of the same series. **Caveat:** these compilations tabulate against *photoelectron* (kinetic) energy, but the lookup — inherited from the original MATLAB implementation — interpolates the grid as *photon* energy, a systematic energy-axis offset of order BE/hν. Negligible for shallow levels at HAXPES energies; use the 2018/2019 tables below (photon-energy grid, σ included) for deep core levels. | Factual numerical values, hand-transcribed in the MATLAB era and re-expressed in this project's schema, fully cited. No publisher typography or prose is reproduced. |
| `trzh2018_haxpes.json` | σ, β, γ, δ (outer shells, hν = 1.5–10 keV) | Trzhaskovskaya & Yarzhemsky, *At. Data Nucl. Data Tables* **119**, 99 (2018). DOI: 10.1016/j.adt.2017.04.003. Converted from the digitization by J. Willis, C. Kalha, M. B. Trzhaskovskaya, V. G. Yarzhemsky, D. O. Scanlon, A. Regoutz (UCL — Scanlon Materials Theory Group / Applied X-ray Spectroscopy Group). | Strongest basis in this table: the upstream UCL digitization records the lead author's approval — *"The reproduction of this data is approved by the lead author of the original paper, Malvina Trzhaskovskaya."* Retain a copy or permanent reference to that statement alongside this file. |
| `trzh2019_inner.json` | σ, β, γ, δ (inner shells, hν = 2–18 keV) | Trzhaskovskaya & Yarzhemsky, *At. Data Nucl. Data Tables* **129–130**, 101280 (2019). DOI: 10.1016/j.adt.2019.05.001. Same UCL digitization team as above. | Same recorded author approval as above; same evidence retention applies. |
| `binding_energy.json` | Elemental core-level binding energies (integer eV), per subshell | Standard elemental BE compilation — values match the LBNL X-ray Data Booklet "Electron binding energies" table (after Bearden & Burr 1967; Fuggle & Mårtensson 1980), e.g. Au 1s = 80725, Si 2p3/2 = 99, C 1s = 284. | Standard experimental constants, editorially compiled into this project's own schema. The values appear identically across compilations and derive from the primary literature cited (Bearden & Burr 1967; Fuggle & Mårtensson 1980); the LBNL X-Ray Data Booklet is given as a convenient cross-check, and its own presentation (which carries "©2000") is not reproduced. |
| `compounds.json` | Compound and element properties for IMFP (N_v, density, M_w, E_g), 109 entries: 96 elements — including 11 transactinides (Rf, Db, Sg, Bh, Hs, Mt, Fl, Mc, Lv, Ts, Og) — 11 compounds, and two non-materials (`AVERAGE`, `Oxide`) | **Hand-entered by the author in 2020–2021 from Wikipedia and commonly quoted literature values.** Per-entry sources were not recorded at the time and cannot now be reconstructed. See *`compounds.json` — where these numbers came from* below. | Project-original selection and machine-readable arrangement, released under this package's licence. No published table is reproduced: the entry set is this project's own, and every entry that overlaps a TPP-series parameter table differs from it. **The constants are asserted, not cited** — that is a limitation of the data, not a rights question. Pass parameters explicitly where the value matters. |

## Cross-section units — one table's is not established

The three tables reachable through `CrossSection.lookup` are **not on a
common scale**, and the evidence for each unit is different. Query it
rather than assuming: `CrossSection.unit_info(table)`.

| Table | Unit | Evidence |
|---|---|---|
| `yeh_lindau` | Mb | **Confirmed.** Yeh & Lindau tabulate in Mb. |
| `scofield` | Mb | **Confirmed.** UCRL-51326 Table A2 is headed "PHOTOELECTRIC CROSS SECTIONS(BARNS)"; the bundled cache reproduces those entries after barn → Mb. |
| `trzhaskovskaya` | *inferred* kb | **Not confirmed.** Values run ~10³ above the megabarn tables. Comparison with Scofield after correcting the energy axis, and the same authors' later tables, both point to kb — but the ADNDT 77 (2001) / 82 (2002) table heading itself has not been read. |

`unit_info()` therefore reports `unit=None` and `inferred_unit="kb"` for
that table, so no caller can convert by reading a single field, and no
conversion factor is offered. **The stored values are not rescaled** on
an inference. A ratio within one table cancels the unknown *unit*
factor, and only that — for `trzhaskovskaya` the photoelectron-vs-photon
energy axis error recorded above survives any ratio between lines of
different binding energy. Absolute cross-sections, σ×λ sensitivities,
and any comparison that mixes tables carry both problems.

The 2018/2019 tables are a **separate** question and were checked
separately — the 2001/2002 inference is not inherited. The UCL
digitization's own "Explanation of Tables" sheet states that column (B)
lists the cross section "in kb (=10⁻²¹ cm²) **for completely filled
subshells**". So for `trzh2018_haxpes.json` and `trzh2019_inner.json`
the unit is confirmed as kb, and the σ values are full-subshell
quantities rather than occupancy-weighted ones. Those two files feed
`AngularCorrection`, not `CrossSection.lookup`.

Two conventions that are easy to conflate, and are not the same:

- **Scofield** incorporates the report's assumed *fractional* subshell
  occupations. UCRL-51326 Table A1 lists them per state: B 2p is
  NE 0.33 + 0.67 = 1 electron, C 0.67 + 1.33 = 2, N 1 + 2 = 3,
  O 1.33 + 2.67 = 4. Components are summed as stored; weighting them by
  occupancy again double-corrects.
- **Trzhaskovskaya 2018/2019** gives full-subshell values, per the sheet
  quoted above.

Whether the 2001/2002 table follows one convention or the other has not
been established here, and `CrossSection` does not assume an answer. Its
38 half-listed subshells are consistent with "listed iff occupied", but
until the inclusion rule is read from the source a half-listed bare
subshell returns `None` rather than summing the listed component alone
and calling the absent one zero. The listed component is still available
as a j-resolved request, and the 38-count is pinned by test so the
evidence remains actionable if the rule is later confirmed.

## `compounds.json` — where these numbers came from

The other bundled tables are transcriptions of a named publication. This
one is not, and the difference matters when the values feed TPP-2M.

The entries were typed in by hand in 2020–2021 from Wikipedia and from
figures commonly quoted in the literature, as a working parameter set for
the IMFP formula. No per-entry source was recorded, and the record cannot
be reconstructed after the fact.

Two consequences follow, and they pull in opposite directions.

**On rights, the position is clean.** The set is not an extract of any
compilation. The TPP-series parameter tables — Tanuma, Powell & Penn,
*Surf. Interface Anal.* **17**, 927 (1991) (15 inorganic compounds,
Table 5); Shinotsuka *et al.*, **51**, 427
(2019), DOI 10.1002/sia.6598 (42 inorganic compounds, Table 1); and
**54**, 534 (2022), DOI 10.1002/sia.7064 (organics and water) — are the
obvious upstream candidate, and they are **not** the source of these
values. Every entry that overlaps one of those tables differs from it in
at least one field. Nothing here reproduces a published table.

This file is the one bundled table that does not satisfy the "full
source citation and provenance" criterion listed at the top of this
document, and the exception is deliberate rather than overlooked. The
criteria exist to establish that what is redistributed is not someone
else's compilation; here the *absence* of an upstream compilation is
what the record shows. Missing citations remain a defect — an accuracy
defect, described below — but they are not the defect the criteria are
screening for. (Wikipedia, named above as one origin, licenses its prose
under CC BY-SA. None of it is reproduced here: what was taken is bare
physical constants, which carry no such claim.)

**On accuracy, the position is weaker, and uneven.** Five of the eleven
compounds appear somewhere in the TPP series — Al2O3, GaAs, SiC and SiO2
in the 2019 set, and Si3N4 in the 1991 set (the 2019 study dropped Si3N4
and LiF, whose energy-loss functions gave large sum-rule errors, and
recommends the TPP-2M formula for those two instead). Where a published
parameter set exists the hand-entered one is close to it, though not
identical.

The table below compares **parameter sets, not IMFPs**. Both columns are
TPP-2M evaluated here at 1 keV kinetic energy — the left from the
bundled entry, the right from the parameters the cited paper tabulates.
Neither is a value the papers publish.

| Entry | λ from bundled parameters (Å) | λ from published parameters (Å) | Δ | differs in |
|---|---:|---:|---:|---|
| GaAs | 22.44 | 22.44 | −0.0% | ρ 5.316 vs 5.32, E_g 1.424 vs 1.47 |
| SiO2 | 30.08 | 30.39 | −1.0% | ρ 2.2 vs 2.19, E_g 8.9 vs 9.1 |
| SiC | 22.63 | 22.38 | +1.1% | E_g 3.26 vs 2.31 |
| Si3N4 | 23.98 | 23.59 | +1.7% | ρ 3.2 vs 3.44 |
| Al2O3 | 24.26 | 24.91 | −2.6% | E_g 7.6 vs 8.63 |

Two caveats on that table. The 1991 paper's Table 5 has no molecular
weight column, so the Si3N4 comparison uses this project's own M for
both columns and isolates ρ and E_g. (Neither Si3N4 density is obviously
the better one: the α and β phases have crystallographic densities of
3.18 and 3.20 g/cm³ respectively, so the bundled 3.2 sits on top of them
and the tabulated 3.44 is ~8% above both. The 1991 table states no
phase, and where its value comes from is not recorded there.) And a
small Δ here is **not** a
claim of agreement with the literature: the 2019 paper also publishes
directly calculated IMFPs (its Table 5), and those differ from TPP-2M by
more than any of these parameter differences do — at 1 keV it gives
1.93 nm for SiC where TPP-2M on its own parameters gives 2.24 nm, a 16%
gap, and 2.13 nm for GaAs against 2.24 nm. That gap is the formula's,
not the parameters' — the paper reports an average RMS deviation of
10.7% between TPP-2M and its calculated values — and it is the larger
error for anyone using `IMFP.tpp2m()` on these five materials.

**The other six — TiO2, HfO2, ZrO2, Ta2O5, SrTiO3 and GeO2 — appear in
none of the three compilations named above**, so there is no published
parameter set here to check them against; later parts of the series have
not been surveyed. They are uncited and will stay uncited until sourced
individually. Note what the table above does *not* license: agreement
within 2.6% on five entries that happen to have published counterparts
says nothing about the six that do not.

One class of error is checkable without any source, and one instance was
found: `Si3N4` carried M = 104.28346 g/mol, a digit transposition of the
correct 140.28346, which inflated λ by 11–26% across the fitted range
(+25.8% at 50 eV, +11.5% at 2 keV).
`tests/test_compound_parameters.py` now derives every compound's
molecular weight from its formula and checks that N_v is counted over
the same unit, so a mistyped constant can no longer sit in the table
looking plausible.

How much a *sourcing* error costs depends on the material. Perturbing
one input of
TPP-2M at 1 keV kinetic energy, inside the 50–2000 eV range the formula
was fitted over, and reading λ off the bundled entries:

| Entry | λ (Å) | ρ ±10% → λ | E_g +1 eV → λ |
|---|---:|---:|---:|
| SiO2 | 30.1 | ∓2.1–2.4% | +4.8% |
| SiC | 22.6 | ∓2.1–2.3% | +1.6% |
| TiO2 | 21.2 | ∓2.6% | +1.2% |
| Al2O3 | 24.3 | ∓2.7–2.9% | +2.8% |
| ZrO2 | 19.7 | ∓3.8–4.2% | +1.9% |
| Ta2O5 | 17.3 | ∓4.3–4.8% | +1.3% |
| HfO2 | 17.0 | ∓4.8–5.3% | +1.6% |

For most oxides a 10% density error moves λ by 2–3%, which is small
against the other terms in a quantification. **HfO2 is the entry to
watch**: it is both the most density-sensitive of the set and the one
where the stored value is least likely to describe the sample. The
bundled 9.68 g/cm³ is the bulk monoclinic density, and ALD-grown films
are frequently amorphous and lower. Taking 8.5–9.0 g/cm³ as an
illustrative film range — itself an uncited figure, quoted here to size
the effect and not as a reference value — the stored density is 8–14%
high and λ comes out 3.5–6% short: a one-directional bias, not a
scatter. ZrO2 and Ta2O5 sit in the same category to a lesser degree.
Band gap is the weaker lever for every entry except SiO2.

The 11 transactinide entries are a separate case: those densities are
theoretical predictions rather than measurements, and TPP-2M was never
fitted to anything resembling those materials. They are placeholders,
not reference data.

The remedy in all of these cases is the same — pass `Nv`, `density`,
`Mw` and `Eg` to `IMFP.tpp2m()` explicitly, from a source appropriate to
the sample, whenever the value affects a reported result.

## In-code constants

Two modules carry small curated sets of scalar physical constants
directly in source (no external database is extracted or shipped):

- `core/xps_database.py`: ~50 spin-orbit splitting values (one number
  per element/orbital, e.g. Au 4f = 3.67 eV) plus branch ratios from
  the quantum-mechanical degeneracy formula (2l)/(2l+2). These are
  element-level physical constants reported consistently across the
  primary XPS literature, intended as fitting initial guesses.
- `fitting/templates.py`: chemical-shift starting values for the three
  bundled fitting templates (Si 2p oxide, Ta 4f oxide, C 1s organic),
  likewise widely published reference values used as initial guesses.

## Deliberately NOT bundled

- **Scienta analyzer transmission functions** (`data/transmission.py`):
  vendor-measured curves are read from a user-supplied directory
  (`TOYOMACRO_SCIENTA_DATA_DIR`); a power-law fallback is used when absent.
  Vendor data is not redistributed.
- **TPP-2M IMFP** (`data/imfp.py`): implemented as the published formula,
  no data tables shipped. Model TPP-2M; source S. Tanuma, C. J. Powell &
  D. R. Penn, *Surf. Interface Anal.* **21**, 165 (1994), DOI
  10.1002/sia.740210302; equations (3), (4b), (4c), (4d), (4e) and (8).
  Inputs: kinetic energy in eV (the electron's energy in the solid —
  nothing is subtracted internally), N_v valence electrons per atom
  (elements) or per molecule (compounds), density in g/cm³, molecular
  weight in g/mol, band gap in eV (0 for conductors). Eqn (3) yields
  Ångströms; the public functions return nanometres. The quantity is the
  *inelastic* mean free path — elastic scattering is not included, so it
  is not an EAL. Fitted by the authors over 50–2000 eV; they state the
  equations should not be used above 2000 eV, and that deviations are
  largest below 200 eV. The material parameters in `compounds.json` are
  a separate input with separate provenance (see the table above): the
  fidelity of this formula implies nothing about the accuracy of any
  particular entry.
- **NIST SRD compound chemical-shift data**: not bundled and not loaded by
  this package (NIST Standard Reference Data carries redistribution terms
  of its own; only the elemental BE table above is shipped).
- **Transport mean free paths and single-scattering albedos**
  (`data/elastic_scattering.py`): the effective-attenuation-length helpers
  require a caller-supplied IMFP/TRMFP pair, and no TRMFP or albedo data
  is shipped. Jablonski & Powell, *J. Phys. Chem. Ref. Data* **49**,
  033102 (2020), DOI: 10.1063/5.0008576, tabulate both for 41 elemental
  solids and 42 inorganic compounds from 50 eV to 30 keV in their
  supplementary material, which is the obvious candidate and reaches
  HAXPES energies. The publication carries the notice "© 2020 by the U.S.
  Secretary of Commerce on behalf of the United States. All rights
  reserved." No licence or permission covering redistribution of the
  supplementary tables has yet been documented, so they are deliberately
  not bundled.

## Regenerating the caches

The JSON caches are loaded cache-first (`CrossSection._load_with_cache`);
the original `Common/data/*.csv` sources are only needed to regenerate them
and are not part of this repository. Deleting a cache file and performing a
lookup with the CSV sources present rebuilds it.
