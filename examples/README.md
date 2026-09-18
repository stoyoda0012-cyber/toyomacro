# Examples

Self-contained, runnable examples. Each script generates its own
synthetic data (no measurement files needed), prints a summary, and
saves a figure to `examples/output/`.

Run from the repository root:

```bash
python examples/01_fit_single_spectrum.py
python examples/02_chemical_state_map.py
python examples/03_quantification.py
python examples/04_projection_law_validation.py   # ~1-2 min
python examples/05_fit_map_from_file.py           # or: ... my_map.h5 --state ...
```

| Script | What it shows | Key API |
|--------|---------------|---------|
| `01_fit_single_spectrum.py` | Template-based peak decomposition of a single Si 2p spectrum (spin-orbit doublets, Shirley background) | `AutoFitter`, `Spectrum`, `create_lineshape` |
| `02_chemical_state_map.py` | Batch-fitting a 64x64 imaging-XPS map (4,096 spectra in one call) and rendering chemical-state maps | `FastVoigtFitter`, `FastFitConfig` |
| `03_quantification.py` | Peak areas to composition: Scofield cross-sections and TPP-2M IMFP at Al Ka vs. Ga Ka (HAXPES) | `BindingEnergy`, `CrossSection`, `IMFP` |
| `04_projection_law_validation.py` | Systematic-error sensitivity: validates the first-order projection law for the fitted-center bias under lineshape misspecification, incl. the production solver — details in [`docs/projection_law_validation.md`](../docs/projection_law_validation.md) | `VarProFitter`, `voigt_profile` |
| `05_fit_map_from_file.py` | From a file to chemical-state maps: reads an HDF5 `specdata` map or a reader-supported file (`.pxt`, `.vms`, `.npl`, SES `.txt`; columns become pixels), subtracts a linear background, batch-fits one Voigt per state. Defaults to the synthetic map in `data/` and reports recovery against its ground truth | `read_spectra`, `create_reader`, `FastVoigtFitter` |

All examples run on the pure-numpy backend. Installing the `mlx` extra
(`pip install toyomacro[mlx]`) moves the batch fit in example 02 onto the
Apple Silicon GPU, same code — but at 4,096 spectra a single call is
dominated by one-off setup, so the time it prints barely changes. The
GPU pays off on larger maps: from tens of thousands of spectra up, the
same call runs about an order of magnitude faster than on NumPy.

## Data

Nothing to download: every script above synthesizes its own data, so a
fresh clone runs them as-is. If you want a data *file* — to exercise the
readers, or to see the layout the batch pipeline expects — generate the
small synthetic pair described in
[`data/README.md`](data/README.md):

```bash
python examples/data/make_example_data.py
```
