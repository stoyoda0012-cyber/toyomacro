# Examples

Self-contained, runnable examples. Each script generates its own
synthetic data (no measurement files needed), prints a summary, and
saves a figure to `examples/output/`.

Run from the repository root:

```bash
python examples/01_fit_single_spectrum.py
python examples/02_chemical_state_map.py
python examples/03_quantification.py
python examples/04_projection_law_validation.py
python examples/05_fit_map_from_file.py           # or: ... my_map.h5 --state ...
python examples/06_fermi_edge_calibration.py      # or: ... data.vms --vb-region 2 ...
python examples/07_width_identifiability_map.py
python examples/08_map_viewer_frontend.py          # opens a window; --smoke to skip it
python examples/09_fermi_edge_identifiability.py
python examples/10_composition_ptfe.py            # or: ... --data "Degradation Polymers.zip"
```

| Script | What it shows | Key API |
|--------|---------------|---------|
| `01_fit_single_spectrum.py` | Template-based peak decomposition of a single Si 2p spectrum (spin-orbit doublets, Shirley background) | `AutoFitter`, `Spectrum`, `create_lineshape` |
| `02_chemical_state_map.py` | Batch-fitting a 64x64 imaging-XPS map (4,096 spectra in one call) and rendering chemical-state maps | `FastVoigtFitter`, `FastFitConfig` |
| `03_quantification.py` | Peak areas to composition: Scofield cross-sections and TPP-2M IMFP at Al Ka vs. Ga Ka (HAXPES) | `BindingEnergy`, `CrossSection`, `IMFP` |
| `04_projection_law_validation.py` | Systematic-error sensitivity: validates the first-order projection law for the fitted-center bias under lineshape misspecification, incl. the production solver — details in [`docs/projection_law_validation.md`](../docs/projection_law_validation.md) | `VarProFitter`, `voigt_profile` |
| `05_fit_map_from_file.py` | From a file to chemical-state maps: reads an HDF5 `specdata` map or a reader-supported file (`.pxt`, `.vms`, `.npl`, SES `.txt`; columns become pixels), subtracts a linear background, batch-fits one Voigt per state. Defaults to the synthetic map in `data/` and reports recovery against its ground truth | `read_spectra`, `create_reader`, `FastVoigtFitter` |
| `06_fermi_edge_calibration.py` | Energy-axis calibration on a metal Fermi edge: E_F and resolution for three DOS models (their spread is a sensitivity check on the model choice, not an uncertainty), axis shifted to E_F = 0, Au 4f7/2 read on the calibrated axis. Synthetic by default, with a known offset recovered; takes a file with a VB and an Au 4f region | `fermi_edge.fit_fermi_edge`, `to_binding_energy` |
| `07_width_identifiability_map.py` | How well a spectrum can tell the Gaussian from the Lorentzian width, mapped over window, shape and background — separating the degenerate σ coordinate from real loss of information. Model bounds, not a fit — details in [`docs/design/voigt-width-identifiability.md`](../docs/design/voigt-width-identifiability.md) | `scan_identifiability`, `ScanGrid` (experimental) |
| `08_map_viewer_frontend.py` | A minimal front end on the batch engine: click a pixel of a fitted map to see its spectrum and fit. The back end fits the map once and hands over parameter maps and one pixel at a time, never fitted curves for the whole map; everything runs on one thread (see "Threads" in [`docs/API.md`](../docs/API.md#threads)). The pattern for a viewer or GUI of your own | `FastVoigtFitter`, `voigt_profile` |
| `09_fermi_edge_identifiability.py` | Whether a Fermi edge can tell the instrumental resolution from the temperature at all: the edge width is one number and its split into an instrumental and a thermal half is a separate question, fitting the temperature multiplies the resolution's uncertainty by up to 78, the Cramér–Rao bound is not the shipped fitter's error bar, and the DOS assumption moves σ further than the noise does. Model bounds plus one simulated spectrum — details in [`docs/design/fermi-edge-identifiability.md`](../docs/design/fermi-edge-identifiability.md) | `assess_edge_identifiability`, `scan_edge_identifiability`, `compare_dos_forms` (experimental) |
| `10_composition_ptfe.py` | What a v0.5.0 composition answer looks like, on public data (the first F 1s / C 1s pair of the Kratos PTFE file in Zenodo 7074887, not bundled; synthetic stand-in offline): the homogeneous-equivalent F:C, how far the cross-section table, the background, the assumptions about the stored data (count rate or integrated counts; transmission applied or not) and the band gap move it (the two assumptions about the stored data move it most), the assumptions, windows, exposure basis and matrix source it rests on, and the statistical uncertainty not evaluated, with its reason. Not a check against PTFE's stoichiometry | `composition`, `condition_dependence` (experimental), `VAMASReader` |

All examples run on the pure-numpy backend. Installing the `mlx` extra
(`pip install -e ".[mlx]"` from a clone) moves the batch fit in example 02 onto the
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
