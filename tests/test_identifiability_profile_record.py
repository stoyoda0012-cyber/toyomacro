"""
The registered profiles of the design record (§9.6, P3)
=======================================================

``mode_profile`` along the softest mode, in the three configurations the
design record registered before the code existed. Recorded, not judged:
no prior claim of agreement or disagreement with the Fisher prediction.
The only assertions are that the computation ran -- every grid point
produced an entry -- and that each entry is one of the defined outcomes.
The record is printed, shown with ``-rP``; the full profile also goes to
``record_property`` (in a junit report).

Every setting below is the registration's: the peak at center 0, a 0.02
eV step, an estimated ``linear_background(E, 300, 10)``, exposure 1,
``variance_floor`` 0, mode 1 (``mode=0``), and the grid t = k/10 for k =
-30..30. Expected counts, so there is no seed. Everything here is
computed from a model; none of it is a measurement.
"""

import platform

import numpy as np
import pytest
import scipy

from toyomacro.voigtfit.identifiability import (
    PROFILE_GRID,
    VoigtPeak,
    linear_background,
    mode_profile,
)

SOFT = np.arange(-0.3, 0.3 + 1e-9, 0.02)
HARD = np.arange(-3.0, 3.0 + 1e-9, 0.02)
CONFIGURATIONS = {
    "soft": (SOFT, VoigtPeak(2.0e4, 0.0, 0.1, 0.3)),
    "soft x100": (SOFT, VoigtPeak(2.0e6, 0.0, 0.1, 0.3)),
    "hard": (HARD, VoigtPeak(2.0e4, 0.0, 0.3, 0.3)),
}
OUTCOMES = ("ok", "not_obtained", "amplitude_at_zero", "corner")


def _blas():
    try:
        return np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception:  # the layout of show_config is not a stable API
        return "unknown"


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_registered_profile_is_recorded(name, record_property):
    energy, peak = CONFIGURATIONS[name]
    assert energy.size == (31 if name.startswith("soft") else 301)
    profile = mode_profile(energy, peak, linear_background(energy, 300.0, 10.0), mode=0,
                           t=PROFILE_GRID, exposure=1.0, variance_floor=0.0)

    assert [p.t for p in profile.points] == list(PROFILE_GRID)
    assert all(p.outcome in OUTCOMES for p in profile.points)

    mode = profile.mode
    lines = [
        f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, "
        f"{platform.system()} {platform.machine()}, BLAS {_blas()}",
        f"{name}: mode 1 relative SD {mode.relative_sd:.6g}, weights "
        + ", ".join(f"{k} {v:+.4f}" for k, v in mode.weights.items()),
    ]
    for label, side in (("t > 0", profile.plus), ("t < 0", profile.minus)):
        lines.append(
            f"  {label}: crossing {_fmt(side.crossing)}, Delta(1) {_fmt(side.delta_1)}, "
            f"Delta(2) {_fmt(side.delta_2)}, activations {side.activations}, "
            f"stop {side.stop}, background negative at {list(side.background_negative)}"
        )
    lines.append("     t  outcome            Delta        route  Delta_A      Delta_B      "
                 "kkt_A      kkt_B      flag  active")
    for p in profile.points:
        lines.append(
            f"  {p.t:4.1f}  {p.outcome:17s}  {_fmt(p.delta):11s}  {_fmt(p.route):5s}  "
            f"{_fmt(p.delta_a):11s}  {_fmt(p.delta_b):11s}  {_fmt(p.kkt_a):9s}  "
            f"{_fmt(p.kkt_b):9s}  {'yes' if p.flagged else 'no':4s}  {','.join(p.active)}"
        )
    record_property(f"profile_{name}", profile.to_dict())
    print("\n".join(lines))
