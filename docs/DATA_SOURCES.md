# Bundled reference data — provenance and licensing

All reference tables shipped under `src/toyomacro/data/_cache/` are
machine-readable JSON conversions of published scientific data. The numeric
values are uncopyrightable facts; what we redistribute is our own
transformed representation (interpolation-ready JSON), with full citations
below. No publisher-typeset tables, PDFs, or vendor binaries are included.

| File | Contents | Source | Licensing rationale |
|---|---|---|---|
| `scofield.json` | Photoionization cross sections σ(hν), 1–30 keV, Z = 1–101 | Scofield, *Theoretical photoionization cross sections from 1 to 1500 keV*, UCRL-51326, Lawrence Livermore Laboratory (1973). DOI: 10.2172/4545040 | U.S. government laboratory report — public domain. Bundled cache truncated at 30 keV (covers all lab / HAXPES sources). |
| `cross_section.json` | σ(hν) at 16 photon energies, 10.2 eV–8.05 keV | Yeh & Lindau, *At. Data Nucl. Data Tables* **32**, 1 (1985). DOI: 10.1016/0092-640X(85)90016-6 | Factual data tables, format-transformed, fully cited. Same dataset is redistributed by other open-source packages (e.g. `galore`, JOSS 2018). |
| `trzhaskovskaya.json` | Relativistic σ, 10-energy grid 0.1–10 keV, Z = 1–100 | Hand-digitized (in the MATLAB Toyomacro era) from Trzhaskovskaya, Nefedov & Yarzhemsky, *At. Data Nucl. Data Tables* **77**, 97 (2001), DOI: 10.1006/adnd.2000.0849 (Z = 1–54) and **82**, 257 (2002), DOI: 10.1006/adnd.2002.0886 (Z = 55–100); the 10 keV column comes from a later extension of the same series. **Caveat:** these compilations tabulate against *photoelectron* (kinetic) energy, but the lookup — inherited from the original MATLAB implementation — interpolates the grid as *photon* energy, a systematic energy-axis offset of order BE/hν. Negligible for shallow levels at HAXPES energies; use the 2018/2019 tables below (photon-energy grid, σ included) for deep core levels. | Factual data, hand-transcribed and format-transformed, fully cited. |
| `trzh2018_haxpes.json` | σ, β, γ, δ (outer shells, hν = 1.5–10 keV) | Trzhaskovskaya & Yarzhemsky, *At. Data Nucl. Data Tables* **119**, 99 (2018). DOI: 10.1016/j.adt.2017.04.003. Converted from the digitization by J. Willis, C. Kalha, M. B. Trzhaskovskaya, V. G. Yarzhemsky, D. O. Scanlon, A. Regoutz (UCL — Scanlon Materials Theory Group / Applied X-ray Spectroscopy Group). | The digitization states: *"The reproduction of this data is approved by the lead author of the original paper, Malvina Trzhaskovskaya."* Fully cited. |
| `trzh2019_inner.json` | σ, β, γ, δ (inner shells, hν = 2–18 keV) | Trzhaskovskaya & Yarzhemsky, *At. Data Nucl. Data Tables* **129–130**, 101280 (2019). DOI: 10.1016/j.adt.2019.05.001. Same UCL digitization team as above. | Same explicit reproduction approval as above. |
| `binding_energy.json` | Elemental core-level binding energies (integer eV), per subshell | Standard elemental BE compilation — values match the LBNL X-ray Data Booklet "Electron binding energies" table (after Bearden & Burr 1967; Fuggle & Mårtensson 1980), e.g. Au 1s = 80725, Si 2p3/2 = 99, C 1s = 284. | U.S. national-laboratory publication of factual data — freely redistributed. |
| `compounds.json` | Compound properties for IMFP (N_v, density, M_w, E_g), ~20 entries | In-house curation of standard physical constants (densities, molecular weights, band gaps) for TPP-2M input. | Small original compilation of public physical constants — no restriction. |

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

## Regenerating the caches

The JSON caches are loaded cache-first (`CrossSection._load_with_cache`);
the original `Common/data/*.csv` sources are only needed to regenerate them
and are not part of this repository. Deleting a cache file and performing a
lookup with the CSV sources present rebuilds it.
