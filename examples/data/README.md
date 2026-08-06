# Example data

The scripts in `examples/` need nothing from this directory — each one
synthesizes its data in-process, so a fresh clone runs them immediately.
This directory is for the other case: a *file* on disk, in one of the
two formats the readers and the batch pipeline consume, to exercise the
I/O path or to see the expected layout before pointing the pipeline at
your own measurement.

Generate them (nothing is downloaded; both are synthetic):

```bash
python examples/data/make_example_data.py
```

| File | Content | Committed | Format |
|---|---|---|---|
| `si2p_single.txt` | one Si 2p spectrum, 151 channels, Poisson counts | yes (1.9 kB) | two-column text |
| `si2p_map_8x8.h5` | 64-pixel map with a metal→oxide ramp | no, generated | HDF5 `specdata` |

Both are **synthetic** — a Si 2p metal/oxide pair of spin-orbit doublets
on a flat background, not a measurement — and fully determined by the
seed in the generator, so regeneration is byte-identical. Verify the
committed file with:

```bash
python examples/data/make_example_data.py --check
```

(Byte-exactness is tied to the NumPy version's `Generator.poisson`
stream, so `--check` is a local reproducibility tool; CI asserts the
file's structure and peak positions instead — `tests/test_example_data.py`.)

Loading them:

```python
import numpy as np
from toyomacro.voigtfit import read_spectra

data = np.loadtxt("examples/data/si2p_single.txt")   # (151, 2): eV, counts
xps = read_spectra("examples/data/si2p_map_8x8.h5")  # xps.spectra (64, 151)
```

They carry the repository's MIT license.

## Measured data

**No measured spectra are published here.** Doing so requires an
explicit authorization decision by the author, and nothing in the
repository depends on it: examples and CI use synthetic data and say so,
and tests that would exercise measured maps skip when the files are
absent.

## Schema

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

### Accompanying files (required for any measured data added later)

- `<name>.reference_fit.json` — committed fit result (solver name, peak
  configuration, recovered parameters) so any regression is visible in
  review.
- An entry in the table above stating element/orbital, size, source, and
  license, plus confirmation that no sample- or customer-identifying
  acquisition metadata survives in the file.
