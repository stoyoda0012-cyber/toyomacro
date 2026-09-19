"""A minimal front end on the batch engine: click a pixel, see its fit.

The pattern for putting your own viewer or GUI on toyomacro:

- **Back end** (``MapBackend``). Holds the spectra, fits the whole map in
  one batch call, and keeps only the fitted parameters — a few numbers
  per pixel. It hands the front end two kinds of thing: parameter maps
  (images), and one pixel's spectrum with its model rebuilt on demand.
  It never builds fitted curves for the whole map: those are
  n_pixels × n_energy numbers, as large as the data itself, while the
  parameters are a few per pixel.
- **Front end** (``MapViewer``). A matplotlib figure; clicking the map
  asks the back end for that pixel. Replace it with Qt, a web page or
  anything else — the back end does not change.
- **One thread.** The batch fit runs once at start-up and each click
  only rebuilds curves in NumPy, all on the main thread. With the MLX
  backend, do not start a new thread per fit: see "Threads" in
  ``docs/API.md``.

The data are a synthetic Si 2p map like example 02's: two chemical
states, each fitted as a single Voigt. Where a state is nearly absent, its
fitted position and width are not determined by the data and should not
be read.

Run from the repository root::

    python examples/08_map_viewer_frontend.py          # opens a window
    python examples/08_map_viewer_frontend.py --smoke  # no window

``--smoke`` clicks two pixels programmatically and saves the figure to
``examples/output/08_map_viewer_frontend.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from toyomacro.fitting import FastFitConfig, FastVoigtFitter
from toyomacro.voigtfit import voigt_profile

OUT_DIR = Path(__file__).parent / "output"

STATES = {"Si0": (99.3, 0.45), "Si4+": (103.4, 0.60)}  # name: (center, sigma), eV
GAMMA = 0.10
ENERGY = np.linspace(96.0, 107.0, 111)
MAP_SIZE = 64
NOISE_SIGMA = 0.02


def make_map(n: int = MAP_SIZE, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic map: (n*n, n_energy) spectra and the (n, n) true oxide fraction."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    frac = np.exp(-((x - 0.35) ** 2 + (y - 0.40) ** 2) / 0.018)
    frac = np.clip(frac + np.exp(-((x - 0.72) ** 2 + (y - 0.68) ** 2) / 0.045), 0.0, 1.0)
    basis = np.stack([voigt_profile(ENERGY, c, s, GAMMA) for c, s in STATES.values()])
    areas = np.stack([1.0 - frac.ravel(), frac.ravel()], axis=1)
    rng = np.random.default_rng(seed)
    spectra = areas @ basis + rng.normal(0.0, NOISE_SIGMA, (n * n, ENERGY.size))
    return spectra.astype(np.float32), frac


class MapBackend:
    """Owns the data and the fit; answers with maps and single pixels."""

    def __init__(self, energy: np.ndarray, spectra: np.ndarray, shape: tuple[int, int]):
        self.energy = energy
        self.spectra = spectra  # (n_pixels, n_energy), row-major pixels
        self.shape = shape
        self.names = list(STATES)
        self.result = None

    def fit(self) -> dict[str, np.ndarray]:
        """Fit every pixel in one call; return parameter maps, not curves."""
        centers = np.array([c for c, _ in STATES.values()])
        sigmas = np.array([s for _, s in STATES.values()])
        self.result = FastVoigtFitter(FastFitConfig(mode="fast")).fit_batch_rowmajor(
            self.spectra, self.energy, centers, sigmas, GAMMA)
        areas = self.result.amplitudes  # (n_states, n_pixels), integrated areas
        total = np.clip(areas.sum(axis=0), 1e-6, None)
        maps = {f"area {name}": a.reshape(self.shape) for name, a in zip(self.names, areas)}
        maps["oxide fraction"] = (areas[1] / total).reshape(self.shape)
        return maps

    def pixel(self, row: int, col: int) -> dict:
        """One pixel: its spectrum and the model rebuilt from its parameters."""
        r, p = self.result, row * self.shape[1] + col
        components = np.stack([
            r.amplitudes[k, p] * voigt_profile(self.energy, r.centers[k, p], r.sigmas[k, p], r.gamma)
            for k in range(len(self.names))
        ])
        params = {name: (float(r.amplitudes[k, p]), float(r.centers[k, p]), float(r.sigmas[k, p]))
                  for k, name in enumerate(self.names)}
        return {"energy": self.energy, "data": self.spectra[p], "components": components,
                "model": components.sum(axis=0), "params": params}


class MapViewer:
    """A map and a spectrum panel. Knows nothing about fitting."""

    def __init__(self, backend: MapBackend, maps: dict[str, np.ndarray],
                 show: str = "oxide fraction"):
        import matplotlib.pyplot as plt

        self.backend = backend
        self.fig, (self.ax_map, self.ax_spec) = plt.subplots(1, 2, figsize=(11, 4.2))
        im = self.ax_map.imshow(maps[show], origin="lower", cmap="viridis", vmin=0, vmax=1)
        self.fig.colorbar(im, ax=self.ax_map, fraction=0.046)
        self.ax_map.set_title(f"{show} — click a pixel", fontsize=9)
        (self.marker,) = self.ax_map.plot([], [], "+", color="red", ms=14, mew=2)
        (self.data_line,) = self.ax_spec.plot([], [], ".", ms=3, color="0.4", label="data")
        (self.model_line,) = self.ax_spec.plot([], [], "-", color="tab:red", label="fit")
        self.comp_lines = [self.ax_spec.plot([], [], "--", lw=1, label=name)[0]
                           for name in backend.names]
        self.ax_spec.set_xlabel("Binding energy (eV)")
        self.ax_spec.set_xlim(backend.energy.max(), backend.energy.min())
        self.ax_spec.legend(fontsize=8)
        self.selected: tuple[int, int] | None = None
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.tight_layout()

    def on_click(self, event) -> None:
        if event.inaxes is not self.ax_map or event.xdata is None:
            return
        row, col = int(round(event.ydata)), int(round(event.xdata))
        n_rows, n_cols = self.backend.shape
        if 0 <= row < n_rows and 0 <= col < n_cols:
            self.show_pixel(row, col)

    def show_pixel(self, row: int, col: int) -> None:
        px = self.backend.pixel(row, col)
        self.selected = (row, col)
        self.marker.set_data([col], [row])
        self.data_line.set_data(px["energy"], px["data"])
        self.model_line.set_data(px["energy"], px["model"])
        for line, comp in zip(self.comp_lines, px["components"]):
            line.set_data(px["energy"], comp)
        top = float(max(px["data"].max(), px["model"].max()))
        self.ax_spec.set_ylim(-0.1 * top, 1.15 * top)
        text = "  ".join(f"{name}: area {a:.2f}, {c:.2f} eV" for name, (a, c, _) in px["params"].items())
        self.ax_spec.set_title(f"pixel (row {row}, col {col})   {text}", fontsize=8)
        self.fig.canvas.draw_idle()


def main(argv: list[str] | None = None) -> MapViewer:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="no window: click two pixels and save the figure")
    args = parser.parse_args(argv)
    import matplotlib

    if args.smoke:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    spectra, frac_true = make_map()
    backend = MapBackend(ENERGY, spectra, frac_true.shape)
    maps = backend.fit()
    r = backend.result
    rmse = float(np.sqrt(np.mean((maps["oxide fraction"] - frac_true) ** 2)))
    print(f"fitted {r.n_spectra} spectra in {r.elapsed_ms:.0f} ms; "
          f"oxide-fraction RMSE against the truth {rmse:.3f}")

    viewer = MapViewer(backend, maps)
    if not args.smoke:
        viewer.show_pixel(*np.unravel_index(np.argmax(maps["oxide fraction"]), frac_true.shape))
        plt.show()
        return viewer

    # Drive the same click handler a user would, then save what it drew.
    for row, col in [(25, 22), (5, 55)]:
        viewer.on_click(SimpleNamespace(inaxes=viewer.ax_map, xdata=col + 0.2, ydata=row - 0.3))
        print(f"clicked pixel {viewer.selected}: " + ", ".join(
            f"{k} area {a:.2f}" for k, (a, _, _) in backend.pixel(row, col)["params"].items()))
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "08_map_viewer_frontend.png"
    viewer.fig.savefig(out, dpi=130)
    print(f"saved: {out}")
    return viewer


if __name__ == "__main__":
    main()
