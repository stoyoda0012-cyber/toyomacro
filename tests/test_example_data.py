"""The committed example data file parses and matches its generator.

Deliberately checks structure and physics rather than bytes: the
generator's ``--check`` mode does the byte-exact comparison, but that is
tied to the NumPy version's ``Generator.poisson`` stream, so it is a
manual/local tool rather than a CI gate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GEN_PATH = REPO / "examples" / "data" / "make_example_data.py"
TXT_PATH = REPO / "examples" / "data" / "si2p_single.txt"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location(
        "_make_example_data", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod
    finally:
        del sys.modules[spec.name]


def test_committed_text_file_parses(gen):
    data = np.loadtxt(TXT_PATH)
    assert data.shape == (gen.ENERGY.size, 2)
    np.testing.assert_allclose(data[:, 0], gen.ENERGY)
    assert (data[:, 1] >= 0).all(), "Poisson counts must be non-negative"


def test_committed_text_file_has_both_chemical_states(gen):
    """Peaks sit where the generator's Si0 / Si4+ states put them."""
    energy, counts = np.loadtxt(TXT_PATH).T
    for _name, center, _area, _fwhm in gen.STATES:
        window = np.abs(energy - center) < 0.4
        # The 2p3/2 line dominates its neighbourhood; require a clear
        # excess over the background floor rather than an exact height.
        assert counts[window].max() > 3 * np.median(counts), (
            f"no peak near {center} eV")


def test_generated_map_has_expected_layout_and_ramp(gen, tmp_path):
    h5py = pytest.importorskip("h5py")
    from toyomacro.voigtfit import read_spectra

    path = tmp_path / "map.h5"
    gen.write_h5(path)

    with h5py.File(path) as f:
        attrs = dict(f["specdata"].attrs)
    assert attrs["element"] == "Si"
    assert attrs["energy_type"] == "BE"
    assert attrs["source"] == "synthetic"

    xps = read_spectra(path)
    assert xps.spectra.shape == (64, gen.ENERGY.size)
    np.testing.assert_allclose(xps.energy, gen.ENERGY, rtol=1e-6)

    # Oxide fraction ramps along the fast axis, so the Si4+ line grows
    # monotonically across the first row of pixels.
    i_ox = int(np.abs(gen.ENERGY - 103.40).argmin())
    row = xps.spectra[:8, i_ox]
    assert np.all(np.diff(row) > 0), f"oxide ramp not monotonic: {row}"


def test_regeneration_is_deterministic(gen):
    np.testing.assert_array_equal(gen.build_single(), gen.build_single())
    np.testing.assert_array_equal(gen.build_map(), gen.build_map())
