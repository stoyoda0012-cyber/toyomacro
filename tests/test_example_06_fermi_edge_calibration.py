"""examples/06_fermi_edge_calibration.py recovers a known axis offset.

The synthetic spectra put E_F at a known offset and Au 4f7/2 at 84 eV on
the true axis; calibrating on the Fermi edge must bring 4f7/2 back.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "06_fermi_edge_calibration.py"


@pytest.fixture(scope="module")
def ex():
    spec = importlib.util.spec_from_file_location("_example_06", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod
    finally:
        del sys.modules[spec.name]


def test_offset_and_core_level_are_recovered(ex):
    vb, vb_counts, core, core_counts = ex.synthetic_spectra()
    fits = ex.fit_ef_models(vb, vb_counts, ex.default_window(vb, vb_counts), 300.0)
    lin = fits["linear"]  # the generating model
    assert lin.success
    assert abs(lin.ef - ex.SYNTHETIC_OFFSET) < 4 * lin.ef_err
    params, _model, _data = ex.fit_doublet(ex.to_binding_energy(core, lin.ef), core_counts)
    c_72 = min(params[1], params[4])
    assert c_72 == pytest.approx(84.0, abs=0.03)


def test_end_to_end_writes_figure(ex, tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "OUT_DIR", tmp_path)
    ex.main([])
    assert (tmp_path / "06_fermi_edge_calibration.png").exists()


def test_default_window_finds_the_fermi_edge_not_a_steeper_deeper_band(ex):
    """A valence band whose d-band rise is steeper than the Fermi edge.

    Measured Au spectra look like this; picking the steepest rise put the
    window on the 5d onset ~2 eV below E_F.
    """
    import numpy as np

    be = np.arange(-3.0, 8.0 + 1e-9, 0.05)
    edge = ex.fermi_edge(be, ef=-0.4, amplitude=3000.0, fwhm_g=0.4, dos_c1=0.3,
                         bg_const=300.0, bg_ref=0.0)
    d_band = ex.fermi_edge(be, ef=1.6, amplitude=40000.0, fwhm_g=0.4, bg_ref=0.0)
    counts = np.random.default_rng(0).poisson(edge + d_band).astype(float)
    lo, hi = ex.default_window(be, counts)
    assert lo < -0.4 < hi
    assert abs(0.5 * (lo + hi) - (-0.4)) < 0.3
