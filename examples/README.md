# Examples

Self-contained, runnable examples. Each script generates its own
synthetic data (no measurement files needed), prints a summary, and
saves a figure to `examples/output/`.

Run from the repository root:

```bash
python examples/01_fit_single_spectrum.py
python examples/02_chemical_state_map.py
python examples/03_quantification.py
```

| Script | What it shows | Key API |
|--------|---------------|---------|
| `01_fit_single_spectrum.py` | Template-based peak decomposition of a single Si 2p spectrum (spin-orbit doublets, Shirley background) | `AutoFitter`, `Spectrum`, `create_lineshape` |
| `02_chemical_state_map.py` | Batch-fitting a 64x64 imaging-XPS map (4,096 spectra in one call) and rendering chemical-state maps | `FastVoigtFitter`, `FastFitConfig` |
| `03_quantification.py` | Peak areas to composition: Scofield cross-sections and TPP-2M IMFP at Al Ka vs. Ga Ka (HAXPES) | `BindingEnergy`, `CrossSection`, `IMFP` |

All examples run on the pure-numpy backend. Installing the `mlx` extra
(`pip install toyomacro[mlx]`) accelerates example 02 on Apple Silicon,
same code.
