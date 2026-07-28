"""Smoke tests for the Scienta transmission adapter.

No vendor data is bundled, so these exercise the paths that work
without it: the power-law fallback, discovery returning empty rather
than raising, and the log-log interpolation contract. `load()` itself
is only reachable with a user-supplied
``TOYOMACRO_SCIENTA_DATA_DIR``.
"""

from __future__ import annotations

import numpy as np
import pytest

from toyomacro.data.transmission import TransmissionFunction


def test_discovery_is_quiet_without_vendor_data(tmp_path):
    """Absent data is a False/empty answer, never an exception."""
    assert TransmissionFunction.available(tmp_path) is False
    assert TransmissionFunction.list_analyzers(tmp_path) == []
    assert TransmissionFunction.list_configs(data_root=tmp_path) == []
    assert TransmissionFunction.list_configs(
        analyzer="NoSuchAnalyzer", data_root=tmp_path) == []


def test_power_law_fallback_matches_its_own_formula():
    alpha, amp = -0.35, 45.0
    t = TransmissionFunction.from_power_law(alpha=alpha, A=amp)

    lo, hi = t.ek_ep_range
    assert (lo, hi) == (1.0, 200.0)
    for ratio in (1.0, 10.0, 100.0):
        assert t(ratio) == pytest.approx(amp * ratio ** alpha, rel=1e-6)

    # Monotonically decreasing for negative alpha.
    probe = np.geomspace(lo, hi, 25)
    assert np.all(np.diff(t(probe)) < 0)


def test_call_is_vectorized_and_clamps_outside_the_data_range():
    t = TransmissionFunction.from_power_law()
    lo, hi = t.ek_ep_range

    vector = t(np.array([lo, 10.0, hi]))
    assert vector.shape == (3,)
    assert np.isscalar(t(10.0)) or np.ndim(t(10.0)) == 0

    # Extrapolation is held flat at the endpoints, not extended.
    assert t(lo / 10) == pytest.approx(t(lo), rel=1e-6)
    assert t(hi * 10) == pytest.approx(t(hi), rel=1e-6)


def test_interpolation_reproduces_supplied_points():
    ek_ep = np.geomspace(1.0, 100.0, 12)
    values = 30.0 * ek_ep ** -0.5
    t = TransmissionFunction(ek_ep, values, analyzer="EW4000", slit=0.2)

    np.testing.assert_allclose(t(ek_ep), values, rtol=1e-6)
    assert "EW4000" in t.label


# --- regressions found by running the adapter against real vendor data ---

def test_unsorted_input_is_sorted_not_silently_misinterpolated():
    """Descending input used to return the wrong branch of the curve."""
    ek_ep = np.array([10.0, 5.0, 1.0])
    values = np.array([1.0, 2.0, 3.0])
    t = TransmissionFunction(ek_ep, values)

    assert np.all(np.diff(t.ek_ep) > 0)
    assert t.ek_ep_range == (1.0, 10.0)
    assert t(5.0) == pytest.approx(2.0)


@pytest.mark.parametrize("values", [
    [5.0, 0.0, 1.0],            # zero -> log(0)
    [5.0, -1.0, 1.0],           # negative
    [5.0, np.nan, 1.0],         # nan
])
def test_non_positive_transmission_raises(values):
    """Used to yield alpha=nan and T=0 with only a RuntimeWarning."""
    with pytest.raises(ValueError):
        TransmissionFunction(np.array([1.0, 2.0, 3.0]), np.array(values))


def test_degenerate_input_raises():
    with pytest.raises(ValueError):
        TransmissionFunction(np.array([1.0]), np.array([5.0]))
    with pytest.raises(ValueError):
        TransmissionFunction(np.array([1.0, 2.0]), np.array([5.0]))
    with pytest.raises(ValueError):
        TransmissionFunction(np.array([2.0, 2.0]), np.array([5.0, 6.0]))


def test_off_grid_slit_refuses_instead_of_rounding(tmp_path):
    """slit=0.54 used to silently load the 0.5 eV calibration."""
    d = tmp_path / "EW4000" / "Transmission"
    d.mkdir(parents=True)
    (d / "Slit0p5_spot0p1x0p1.txt").write_text("1.0\t10.0\r\n2.0\t8.0\r\n")

    ok = TransmissionFunction.load(slit=0.5, spot="0.1x0.1", data_root=tmp_path)
    assert ok.slit == 0.5

    with pytest.raises(ValueError, match="not an available slit width"):
        TransmissionFunction.load(slit=0.54, spot="0.1x0.1",
                                  data_root=tmp_path)


def test_missing_file_error_is_short_and_leaks_no_paths(tmp_path):
    """The message used to embed every absolute path in the directory."""
    d = tmp_path / "EW4000" / "Transmission"
    d.mkdir(parents=True)
    for i in range(20):
        (d / f"Slit0p5_spot{i}x1.txt").write_text("1.0\t10.0\n2.0\t8.0\n")

    with pytest.raises(FileNotFoundError) as excinfo:
        TransmissionFunction.load(slit=4.0, spot="9x9", data_root=tmp_path)
    msg = str(excinfo.value)
    assert len(msg) < 600
    assert str(tmp_path) not in msg
    assert "Slit4p0_spot9x9.txt" in msg


def test_analyzer_is_listed_when_only_a_secondary_mode_has_data(tmp_path):
    """EW4000 used to be invisible without its Transmission subdirectory."""
    for mode in ("Angular45", "Angular56"):
        d = tmp_path / "EW4000" / mode
        d.mkdir(parents=True)
        (d / "Slit0p5_spot0p1x0p1.txt").write_text("1.0\t10.0\n2.0\t8.0\n")

    assert TransmissionFunction.available(tmp_path) is True
    assert TransmissionFunction.list_analyzers(tmp_path) == ["EW4000"]
    assert TransmissionFunction.list_configs(
        "EW4000", "Angular45", tmp_path) == [(0.5, "0.1x0.1")]


def test_env_var_is_honoured_when_set_after_import(tmp_path, monkeypatch):
    """The root used to be frozen into a constant at import time."""
    d = tmp_path / "R3000" / "Transmission"
    d.mkdir(parents=True)
    (d / "Slit0p5_spot0p1x0p1.txt").write_text("1.0\t10.0\n2.0\t8.0\n")

    monkeypatch.setenv("TOYOMACRO_SCIENTA_DATA_DIR", str(tmp_path))
    assert TransmissionFunction.available() is True
    assert TransmissionFunction.list_analyzers() == ["R3000"]

    monkeypatch.delenv("TOYOMACRO_SCIENTA_DATA_DIR")
    assert TransmissionFunction.available(tmp_path) is True
