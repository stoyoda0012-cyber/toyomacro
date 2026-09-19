"""examples/08_map_viewer_frontend.py: a click reaches the right pixel.

CI runs the example with ``--smoke``. These tests pin what a user
copying the pattern relies on: that a click in data coordinates selects
the pixel under the cursor, that the model the back end rebuilds for a
pixel is that pixel's fit, and that the interactive path, which CI does
not run, opens on the most oxidised pixel.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

from toyomacro.fitting.fast_voigt import VOIGTFIT_AVAILABLE

matplotlib.use("Agg", force=True)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "08_map_viewer_frontend.py"

pytestmark = pytest.mark.skipif(not VOIGTFIT_AVAILABLE, reason="voigtfit not available")


@pytest.fixture(scope="module")
def ex():
    spec = importlib.util.spec_from_file_location("_example_08", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        del sys.modules["_example_08"]


@pytest.fixture(scope="module")
def fitted(ex):
    spectra, frac = ex.make_map()
    backend = ex.MapBackend(ex.ENERGY, spectra, frac.shape)
    maps = backend.fit()
    return backend, maps


def test_a_click_selects_the_pixel_under_the_cursor(ex, fitted):
    import matplotlib.pyplot as plt

    backend, maps = fitted
    viewer = ex.MapViewer(backend, maps)
    try:
        # imshow puts pixel (row, col) at x = col, y = row, each ±0.5
        viewer.on_click(SimpleNamespace(inaxes=viewer.ax_map, xdata=12.4, ydata=40.6))
        assert viewer.selected == (41, 12)
        # outside the map axes, or outside the image: nothing changes
        viewer.on_click(SimpleNamespace(inaxes=viewer.ax_spec, xdata=3.0, ydata=3.0))
        viewer.on_click(SimpleNamespace(inaxes=viewer.ax_map, xdata=70.0, ydata=3.0))
        viewer.on_click(SimpleNamespace(inaxes=viewer.ax_map, xdata=None, ydata=None))
        assert viewer.selected == (41, 12)
    finally:
        plt.close(viewer.fig)


def test_pixel_returns_that_pixels_data_and_fit_on_a_non_square_map(ex):
    """Every pixel has its own random areas, and the map is 5 x 8, so a
    transposed index lands on a different pixel with different data and
    areas (the example's own map is nearly symmetric under transposition
    and would not show it). Noise-free: over 10 seeds x 40 pixels the
    fitted areas were within 0.0065 of the truth and each component
    peaked on its state's centre (grid step 0.1 eV)."""
    rng = np.random.default_rng(3)
    shape = (5, 8)
    areas = rng.uniform(0.2, 1.0, (shape[0] * shape[1], 2))
    centers = [c for c, _ in ex.STATES.values()]
    basis = np.stack([ex.voigt_profile(ex.ENERGY, c, s, ex.GAMMA) for c, s in ex.STATES.values()])
    spectra = (areas @ basis).astype(np.float32)
    backend = ex.MapBackend(ex.ENERGY, spectra, shape)
    backend.fit()
    for r in range(shape[0]):
        for c in range(shape[1]):
            p = r * shape[1] + c
            px = backend.pixel(r, c)
            np.testing.assert_array_equal(px["data"], spectra[p])
            fitted_areas = [px["params"][name][0] for name in backend.names]
            np.testing.assert_allclose(fitted_areas, areas[p], atol=0.02)
            for comp, center in zip(px["components"], centers):
                assert abs(ex.ENERGY[np.argmax(comp)] - center) < 0.15
            np.testing.assert_allclose(px["components"].sum(axis=0), px["model"])


def test_rebuilt_pixel_models_follow_their_spectra_over_the_whole_map(fitted):
    """Over all 4,096 pixels of the example's map. The bound: noise 0.02
    combined with the 0.009 rms the batch fit leaves on the noise-free
    map (measured, seeds 0-4) gives 0.022; observed 0.0212-0.0213."""
    backend, _ = fitted
    n_rows, n_cols = backend.shape
    residual = np.stack([
        (lambda px: px["model"] - px["data"])(backend.pixel(r, c))
        for r in range(n_rows) for c in range(n_cols)
    ])
    assert np.sqrt(np.mean(residual ** 2)) < 0.025


def test_parameter_maps_recover_the_true_oxide_fraction(ex, fitted):
    _, maps = fitted
    _, frac_true = ex.make_map()
    rmse = np.sqrt(np.mean((maps["oxide fraction"] - frac_true) ** 2))
    # observed 0.014; the single-Voigt fit of each state sets the floor
    assert rmse < 0.03


def test_smoke_run_writes_the_figure(ex, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(ex, "OUT_DIR", tmp_path)
    viewer = ex.main(["--smoke"])
    try:
        assert (tmp_path / "08_map_viewer_frontend.png").exists()
        assert viewer.selected == (5, 55)
    finally:
        plt.close(viewer.fig)


def test_interactive_path_opens_on_the_most_oxidised_pixel(ex, monkeypatch):
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    viewer = ex.main([])
    try:
        assert shown == [True]
        _, frac_true = ex.make_map()
        backend_frac = viewer.backend.fit()["oxide fraction"]
        expected = np.unravel_index(np.argmax(backend_frac), frac_true.shape)
        assert viewer.selected == tuple(int(i) for i in expected)
    finally:
        plt.close(viewer.fig)
