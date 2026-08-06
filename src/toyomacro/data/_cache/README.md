# `_cache/` — bundled reference tables

The directory name is historical. The JSON files in it are **shipped
reference data**, not a cache and not a build artifact: they are reviewed,
their provenance and data-rights position are recorded in
[`docs/DATA_SOURCES.md`](../../../../docs/DATA_SOURCES.md), and their
contents are pinned by tests. Editing one is a change to a reference
dataset, which this project treats as requiring an independent audit
(see `AGENTS.md`) — not something to avoid, but not a build step either.

The upstream CSV/XLSX sources they were built from are **not part of this
repository** and have since moved on independently: rebuilding
`compounds.json` from the maintainer's current CSV yields a different
table (166 entries instead of 109, no `Si3N4`, different parameters for
four compounds).

So a missing file is **not** rebuilt on the quiet. Loading one that is
absent raises, naming the file. To rebuild deliberately:

```python
# TOYOMACRO_REGENERATE_DATA=1 must be set in the environment
from toyomacro.data.paths import regenerate_cache
regenerate_cache()  # writes in place, warns, returns per-table results
```

Nothing is deleted first, and a table whose source is unavailable keeps
its shipped copy rather than disappearing. `clear_cache()` no longer
deletes anything: it raises, because deleting shipped data had no
correct use.

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
| `compounds_provenance.json` | hand-authored | Per-field provenance for `compounds.json` (no source to rebuild from) |

"Generated from" records where a table originally came from, not a
current equivalence: see the `compounds.json` section of
`docs/DATA_SOURCES.md` for what has diverged since.

`scofield.json` and `trzhaskovskaya.json` are rebuilt through
`CrossSection`'s own loader rather than by `regenerate_cache()`, which
reports them as such instead of silently omitting them.

Source paths are resolved by `toyomacro.data.paths` and can be overridden with
the `TOYOMACRO_COMMON_PATH` environment variable.
