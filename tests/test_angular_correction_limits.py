"""Validity limits of the non-dipole angular correction.

`AngularCorrection` is an **experimental** API (see `docs/API.md` §5).
These tests do not promote it: they pin the two limits that section
states, so the documentation cannot drift away from the behavior.

The β/γ/δ parameters come from Trzhaskovskaya tables gridded from
1500 eV up (2018 outer shells 1.5–10 keV; 2019 inner shells 2–18 keV).
Below the grid floor the lookup clamps to the edge value instead of
refusing, which matters because Al Kα (1486.6 eV) — the most common
laboratory source — sits just underneath it.
"""

import pytest

from toyomacro.data import AngularCorrection

AL_KA_EV = 1486.6
GRID_FLOOR_EV = 1500.0


def _params(hv):
    got = AngularCorrection.lookup("Si", "2p", hv)
    if got is None:
        pytest.skip("bundled Trzhaskovskaya angular tables unavailable")
    return {k: got[k] for k in ("beta", "gamma", "delta")}


def test_below_grid_floor_clamps_rather_than_extrapolating():
    """Al Ka returns the 1500 eV parameters unchanged.

    Not an interpolation and not an extrapolation — a clamp. A caller
    who reads the returned beta/gamma/delta as applying *at* 1486.6 eV
    is being handed values for a different energy with nothing in the
    result to say so.
    """
    assert _params(AL_KA_EV) == _params(GRID_FLOOR_EV)


def test_the_clamp_is_not_flagged_in_the_return_value():
    """The documented gap: nothing distinguishes clamped from measured.

    `CrossSection` and `IMFP` both carry range/extrapolation
    information through the MCP layer. This module carries none, so
    `docs/API.md` has to say so in prose. If a flag is ever added,
    this test should fail and the prose should be updated with it.
    """
    got = AngularCorrection.lookup("Si", "2p", AL_KA_EV)
    if got is None:
        pytest.skip("bundled Trzhaskovskaya angular tables unavailable")

    assert not any(
        "clamp" in k or "extrapolat" in k or "range" in k for k in got
    ), f"return value gained range metadata: {sorted(got)}"


def test_above_the_floor_the_parameters_actually_vary():
    """Guard against the clamp test passing for the wrong reason.

    If the table were degenerate — one energy, or constant parameters —
    the equality above would hold trivially and prove nothing.
    """
    assert _params(GRID_FLOOR_EV) != _params(3000.0)
