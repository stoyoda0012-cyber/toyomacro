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
| `compounds.json` | Compound properties for IMFP (N_v, density, M_w, E_g), ~20 entries | In-house curation of standard physical constants (densities, molecular weights, band gaps) for TPP-2M input. | Project-original selection and machine-readable arrangement, released under this package's licence. The underlying physical constants are factual values; their sources should be recorded in the dataset metadata. |

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
- **TPP-2M IMFP** (`data/imfp.py`): implemented as the published formula
  (Tanuma, Powell & Penn), no data tables shipped.
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
