# `_cache/` — pre-generated reference tables

The JSON files in this directory are a **pre-generated cache**, committed to the
repository so that `toyomacro` works offline without the upstream source tables.

They are **regenerable** and should be treated as build artifacts, not
hand-edited data. Two mechanisms produce them, both cache-first (regenerate only
when the JSON is missing):

- `toyomacro.data.paths` — binding energies, compounds, Yeh–Lindau and
  Trzhaskovskaya HAXPES tables. Force a rebuild with:

  ```python
  from toyomacro.data.paths import regenerate_cache
  regenerate_cache()
  ```

- `toyomacro.data.cross_section` — Scofield and Trzhaskovskaya–Yarzhemsky
  cross sections, rebuilt automatically on first use from their CSVs.

## Sources

| File | Generated from | Contents |
|------|----------------|----------|
| `binding_energy.json` | `Common/data/BindingEnergyTable.csv` | Core-level binding energies |
| `compounds.json` | `Common/data/CompoundTable.csv` | Compound properties (Nv, density, Mw, Eg) |
| `cross_section.json` | `Common/data/CrossSectionTable_Yeh=Lindau.csv` | Yeh–Lindau photoionization cross sections |
| `scofield.json` | `Scofield_1973_cross_sections.csv` | Scofield 1973 cross sections (≤ 30 keV) |
| `trzhaskovskaya.json` | `CrossSectionTable_Trzhaskovskaya=Yarzhemsky.csv` | Trzhaskovskaya–Yarzhemsky cross sections |
| `trzh2018_haxpes.json` | SESSAAnalyser Trzhaskovskaya 2018 xlsx | HAXPES σ/β/γ/δ, Z=1–100, 1.5–10 keV |
| `trzh2019_inner.json` | SESSAAnalyser Trzhaskovskaya 2019 xlsx | Inner-shell σ/β/γ/δ, Z=13–100 |

Source paths are resolved by `toyomacro.data.paths` and can be overridden with
the `TOYOMACRO_COMMON_PATH` environment variable.
