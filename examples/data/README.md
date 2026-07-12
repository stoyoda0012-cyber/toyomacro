# Measured XPS example data

This directory is the designated location for small, redistributable
**measured** XPS spectra used by the examples and tests. It currently
contains no measured data: publication of measured spectra requires
an explicit authorization decision by the author
(see *Status* below). All examples and CI runs work without it —
they use synthetic data and say so.

## Status

<!-- TODO(author): decide whether one or two small measured spectra
(e.g. a Si 2p map tile) can be published here. Requirements before
committing any file:
  1. employer approval for the specific dataset,
  2. no sample/customer-identifying metadata (anonymize acquisition
     fields; keep only what the fit needs),
  3. a license statement (CC0 or CC-BY recommended for data),
  4. a reference fit result committed alongside for regression.
Until then this README documents the schema only. -->

**No measured data is published yet.** Tests that exercise measured
maps skip when the files are absent; nothing in the repository
depends on them.

## Schema

Two accepted formats:

### 1. Two-column text (single spectrum)

```
# energy_eV    intensity_counts
97.00    1523.0
97.10    1498.2
...
```

- Energy in eV (binding or kinetic — state which in a `# BE` / `# KE`
  header comment), intensity in raw counts.
- Loadable with `toyomacro.io` readers and directly with `np.loadtxt`.

### 2. HDF5 map (`specdata` layout)

```
/specdata   float32 (n_spectra + 1, n_energy)
            row 0        = energy axis (eV)
            rows 1..N    = one spectrum per pixel, row-major pixel order
attributes: element (str), orbital (str), energy_type ('BE'|'KE'),
            source ('anonymized measured'|'synthetic'), license (str)
```

- This is the same layout the batch pipeline consumes
  (`toyomacro.voigtfit.h5io.read_spectra`).

### Accompanying files (required for measured data)

- `<name>.reference_fit.json` — committed fit result (solver name,
  peak configuration, recovered parameters) so any regression is
  visible in review.
- License and provenance stated in this README's table:

| File | Element/orbital | Points/pixels | Source | License |
|---|---|---|---|---|
| *(none yet)* | | | | |
