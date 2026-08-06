"""Bundled cross-section tables must work on a clean install.

The JSON caches under ``toyomacro/data/_cache`` ship with the package;
``Common/data`` CSV sources are only needed to regenerate them. These tests
simulate a clean clone (no Common/data) and require every advertised table
to resolve lookups from the bundled caches alone.
"""

import pytest

import toyomacro.data.cross_section as cs_mod
from toyomacro.data.cross_section import AVAILABLE_TABLES, CrossSection


@pytest.fixture()
def no_common_data(monkeypatch):
    """Simulate a clean clone: Common/data unavailable, memo cleared."""

    def _raise():
        raise FileNotFoundError("simulated clean clone (no Common/data)")

    monkeypatch.setattr(cs_mod, "get_common_data_path", _raise)
    CrossSection._data.clear()
    yield
    CrossSection._data.clear()


@pytest.mark.parametrize("table", AVAILABLE_TABLES)
def test_si2p_al_kalpha_from_bundled_cache(no_common_data, table):
    sigma = CrossSection.lookup("Si", "2p", 1486.6, table=table)
    assert sigma is not None and sigma > 0, (
        f"table '{table}' must resolve Si 2p @ Al K-alpha from the bundled cache"
    )


def test_scofield_haxpes_range(no_common_data):
    """Scofield cache covers the HAXPES range (truncated at 30 keV)."""
    sigma = CrossSection.lookup("Si", "2p", 9251.7, table="scofield")
    assert sigma is not None and sigma > 0
    energies = CrossSection.get_available_photon_energies(table="scofield")
    assert energies and max(energies) <= 30_000.0


def test_rsf_from_bundled_cache(no_common_data):
    rsf = CrossSection.get_rsf("Si", "2p", "C", "1s", 1486.6, table="scofield")
    assert rsf is not None and rsf > 0


def test_get_rsf_is_only_a_cross_section_ratio(no_common_data):
    """`get_rsf` is sigma1/sigma2 and nothing else.

    The name invites reading it as a complete relative sensitivity
    factor. Pin the arithmetic so that anything folded in later — IMFP,
    elastic scattering, transmission — has to break this test and be
    argued for, rather than arriving as a silent change of meaning.
    """
    ratio = CrossSection.get_rsf("Si", "2p", "C", "1s", 1486.6, table="scofield")
    sigma_si = CrossSection.lookup("Si", "2p", 1486.6, table="scofield")
    sigma_c = CrossSection.lookup("C", "1s", 1486.6, table="scofield")

    assert ratio == pytest.approx(sigma_si / sigma_c)
