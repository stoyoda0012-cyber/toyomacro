# Bundled reference data — provenance and licensing

The reference tables under `src/toyomacro/data/_cache/` are published
scientific data, **bundled and redistributed** here as machine-readable
JSON. What is redistributed is the table — its selection and arrangement
included — not merely the numbers in it, so the basis is given per source
rather than by a general argument.

Individual factual values are generally not protected by copyright as
such. This does not by itself establish a right to redistribute a table
or dataset: its selection and arrangement, database rights in some
jurisdictions, contractual terms, and source-specific rights notices
must be considered separately.

This document records the project's provenance and redistribution due
diligence; it is not legal advice. A dataset marked "under verification"
or "not yet established" must not be included in a public distribution.

No publisher-typeset tables, PDFs, or vendor binaries are included.

| File | Contents | Source | Redistribution basis / status |
|---|---|---|---|
| `scofield.json` | Photoionization cross sections σ(hν), 1–30 keV, Z = 1–101 | Scofield, *Theoretical photoionization cross sections from 1 to 1500 keV*, UCRL-51326, Lawrence Livermore Laboratory (1973). DOI: 10.2172/4545040 | **Redistribution basis under verification — partially checked (2026-07-29).** The OSTI record (biblio/4545040) displays no rights, copyright or access-limitation statement, and gives DOE contract W-7405-ENG-48. Absence of an assertion is not a grant, so this is not yet a basis. Remaining check: whether the report itself carries a copyright notice — a 1973 U.S. publication without notice was governed by the 1909 Act, which is a materially different position. Bundled cache truncated at 30 keV (covers all lab / HAXPES sources). |
| `cross_section.json` | σ(hν) at 16 photon energies, 10.2 eV–8.05 keV | Yeh & Lindau, *At. Data Nucl. Data Tables* **32**, 1 (1985). DOI: 10.1016/0092-640X(85)90016-6 | **Redistribution basis not yet established.** The values were format-transformed and fully cited, but citation and redistribution by other open-source projects (e.g. `galore`, JOSS 2018) do not constitute permission. Exclude from the distributed package unless an applicable licence or permission is documented. |
| `trzhaskovskaya.json` | Relativistic σ, 10-energy grid 0.1–10 keV, Z = 1–100 | Hand-digitized (in the MATLAB Toyomacro era) from Trzhaskovskaya, Nefedov & Yarzhemsky, *At. Data Nucl. Data Tables* **77**, 97 (2001), DOI: 10.1006/adnd.2000.0849 (Z = 1–54) and **82**, 257 (2002), DOI: 10.1006/adnd.2002.0886 (Z = 55–100); the 10 keV column comes from a later extension of the same series. **Caveat:** these compilations tabulate against *photoelectron* (kinetic) energy, but the lookup — inherited from the original MATLAB implementation — interpolates the grid as *photon* energy, a systematic energy-axis offset of order BE/hν. Negligible for shallow levels at HAXPES energies; use the 2018/2019 tables below (photon-energy grid, σ included) for deep core levels. | **Redistribution basis not yet established.** Hand transcription, format transformation, and citation do not by themselves grant redistribution rights. Exclude from the distributed package unless an applicable licence or permission is documented. |
| `trzh2018_haxpes.json` | σ, β, γ, δ (outer shells, hν = 1.5–10 keV) | Trzhaskovskaya & Yarzhemsky, *At. Data Nucl. Data Tables* **119**, 99 (2018). DOI: 10.1016/j.adt.2017.04.003. Converted from the digitization by J. Willis, C. Kalha, M. B. Trzhaskovskaya, V. G. Yarzhemsky, D. O. Scanlon, A. Regoutz (UCL — Scanlon Materials Theory Group / Applied X-ray Spectroscopy Group). | The upstream digitization records approval by the lead author for reproduction of the data: *"The reproduction of this data is approved by the lead author of the original paper, Malvina Trzhaskovskaya."* Before release, retain a copy or permanent reference to that approval and verify that its scope covers redistribution of the digitized dataset in a software package. |
| `trzh2019_inner.json` | σ, β, γ, δ (inner shells, hν = 2–18 keV) | Trzhaskovskaya & Yarzhemsky, *At. Data Nucl. Data Tables* **129–130**, 101280 (2019). DOI: 10.1016/j.adt.2019.05.001. Same UCL digitization team as above. | Same recorded approval as above; the same scope verification and evidence retention applies. |
| `binding_energy.json` | Elemental core-level binding energies (integer eV), per subshell | Standard elemental BE compilation — values match the LBNL X-ray Data Booklet "Electron binding energies" table (after Bearden & Burr 1967; Fuggle & Mårtensson 1980), e.g. Au 1s = 80725, Si 2p3/2 = 99, C 1s = 284. | **Action required — checked 2026-07-29.** The LBNL X-Ray Data Booklet asserts "©2000" and states no reuse or redistribution terms, so it cannot serve as the basis and the earlier "freely redistributed" claim was wrong. These are, however, standard experimental constants that appear identically across compilations; re-source the cache to the primary literature it derives from (Bearden & Burr 1967; Fuggle & Mårtensson 1980) and cite those, rather than the Booklet. |
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
