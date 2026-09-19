"""examples/05_fit_map_from_file.py: loading, background, and recovery.

The example is run end to end in CI; these tests pin the pieces a user
would rely on when pointing it at their own file — the column-to-pixel
mapping, the kinetic/binding conversion, the energy sort — and that the
default synthetic map is actually recovered.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from toyomacro.fitting.fast_voigt import VOIGTFIT_AVAILABLE

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "05_fit_map_from_file.py"
GEN_PATH = REPO / "examples" / "data" / "make_example_data.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ex():
    mod = _load(EXAMPLE, "_example_05")
    try:
        yield mod
    finally:
        del sys.modules["_example_05"]


@pytest.fixture(scope="module")
def synthetic_h5(tmp_path_factory):
    gen = _load(GEN_PATH, "_make_example_data_05")
    try:
        path = tmp_path_factory.mktemp("data") / "si2p_map_8x8.h5"
        gen.write_h5(path)
        yield path
    finally:
        del sys.modules["_make_example_data_05"]


def test_background_sorts_descending_axis_and_removes_a_line(ex):
    energy = np.linspace(110.0, 100.0, 51)  # descending, as BE files often are
    line = 3.0 * energy + 7.0
    spectra = np.stack([line, 2.0 * line]).astype(np.float32)

    e_sorted, signal, _bg = ex.subtract_linear_background(energy, spectra)

    assert np.all(np.diff(e_sorted) > 0)
    np.testing.assert_allclose(signal, 0.0, atol=1e-3)


def test_reader_columns_become_spectra_and_sweeps_are_summed(ex, monkeypatch):
    from toyomacro.io.readers import RawSpectrumData, SpectrumMetadata

    n_e, n_col, n_sweep = 20, 3, 2
    block = np.arange(n_e * n_col * n_sweep, dtype=float).reshape(n_e, n_col, n_sweep)
    raw = RawSpectrumData(
        specdata=block,
        energy=np.linspace(1380.0, 1390.0, n_e),
        angle=np.arange(n_col, dtype=float),
        metadata=SpectrumMetadata(energy_scale="Kinetic", excitation_energy=0.0),
    )

    class FakeReader:
        def read(self, region_index=0):
            return raw

    import toyomacro.io.readers as readers
    monkeypatch.setattr(readers, "create_reader", lambda path: FakeReader())

    energy, spectra, scale = ex.load_spectra(Path("fake.pxt"))
    assert spectra.shape == (n_col, n_e)
    np.testing.assert_allclose(spectra[1], block[:, 1, :].sum(axis=1))
    # 0 means "not recorded": the axis stays kinetic rather than going negative
    assert scale.startswith("KE")
    np.testing.assert_allclose(energy, raw.energy)

    energy, _spectra, scale = ex.load_spectra(Path("fake.pxt"), photon_energy=1486.6)
    assert scale.startswith("BE")
    np.testing.assert_allclose(energy, 1486.6 - raw.energy)


@pytest.mark.skipif(not VOIGTFIT_AVAILABLE, reason="voigtfit not available")
def test_synthetic_map_fractions_are_recovered(ex, synthetic_h5):
    from toyomacro.fitting import FastFitConfig, FastVoigtFitter

    energy, spectra, _ = ex.load_spectra(synthetic_h5)
    energy, signal, _bg = ex.subtract_linear_background(energy, spectra)
    states = [ex.parse_state(s) for s in ex.DEFAULT_STATES]
    result = FastVoigtFitter(FastFitConfig(mode="fast")).fit_batch_rowmajor(
        signal, energy,
        np.array([s[1] for s in states]),
        np.array([s[2] for s in states]) / ex.FWHM_PER_SIGMA,
        0.05,
    )
    areas = np.clip(result.amplitudes, 0.0, None)
    truth = ex.synthetic_truth(signal.shape[0])
    rmse = np.sqrt(np.mean((areas / areas.sum(0) - truth / truth.sum(0)) ** 2))
    # Observed 0.011; the single-Voigt doublet approximation sets the floor.
    assert rmse < 0.03


@pytest.mark.skipif(not VOIGTFIT_AVAILABLE, reason="voigtfit not available")
def test_end_to_end_writes_figure(ex, synthetic_h5, tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "OUT_DIR", tmp_path)
    ex.main([str(synthetic_h5), "--shape", "8", "8"])
    assert (tmp_path / "05_map_from_file.png").exists()


@pytest.mark.skipif(not VOIGTFIT_AVAILABLE, reason="voigtfit not available")
def test_missing_default_map_is_generated_and_then_fitted(ex, tmp_path, monkeypatch):
    """A fresh clone has no map file. The example must write one and carry
    on; it used to stop, with exit code 0, right after writing it, because
    the generator it ran ends in sys.exit(). CI never saw that: it runs the
    generator first."""
    missing = tmp_path / "data" / "si2p_map_8x8.h5"
    missing.parent.mkdir()
    monkeypatch.setattr(ex, "DEFAULT_MAP", missing)
    monkeypatch.setattr(ex, "OUT_DIR", tmp_path / "output")
    ex.main([])
    assert missing.exists()
    assert (tmp_path / "output" / "05_map_from_file.png").exists()


def test_center_outside_window_stops_before_fitting(ex, synthetic_h5):
    with pytest.raises(SystemExit, match="outside the energy window"):
        ex.main([str(synthetic_h5), "--state", "Ta:26.0:0.6"])
