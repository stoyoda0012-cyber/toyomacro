"""Angle conventions in ``AngularCorrection``, and what they cost.

Three angles are involved and only one is measured: θ from the surface
normal (the analyzer's axis), α_xray from the normal (an instrument
property), and the derived angle from the photon direction to the
emission direction that every ``L_*`` primitive actually wants. Passing
θ where the derived angle belongs does not raise — it returns another
number between 0 and 2.

That happened in real analysis: β = 1.9255, θ = 8.78°, L = 0.0709 where
the answer was 1.4309. The tests below pin the size of that trap so it
stays visible, pin the identity that makes the convention self-checking,
and pin the geometry warning.

They also record — as a *record*, not as a correctness claim — the
present values of ``L_full``, whose convention is unresolved. See its
docstring: the module documents the linearly polarized form with the
angle taken from the polarization vector, while the implementation
applies the polarization-*averaged* dipole coefficient to an angle
documented as being from the photon direction. Pinning the current
numbers means that when the primary source is read, the diff is visible
rather than silent.
"""

import warnings

import numpy as np
import pytest

from toyomacro.data import AngularCorrection

# Si 1s at the Ga Kα HAXPES energy, the case the trap was found on.
SI_1S_HV = 9251.74
BETA = 1.925483
GAMMA = 1.872726
DELTA = 3e-6

# The analyzer's angle axis for that measurement, and the incidence
# angle that goes with it.
THETA_FAN = np.array([51.59, 45.47, 39.36, 33.24, 27.13, 21.01, 14.90, 8.78])
XRAY_FROM_NORMAL = 88.0

# arccos(1/sqrt(3)) exactly — 54.7356 rounded leaves a 4e-7 residue,
# which would make an exact assertion here fail for the wrong reason.
MAGIC_ANGLE_DEG = np.degrees(np.arccos(1.0 / np.sqrt(3.0)))


def _params():
    got = AngularCorrection.lookup("Si", "1s", SI_1S_HV)
    if got is None:
        pytest.skip("bundled Trzhaskovskaya angular tables unavailable")
    return got


# ---------------------------------------------------------------------------
# The convention is self-checking at one angle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, BETA, 2.0, 3.0])
def test_magic_angle_gives_unity_for_every_beta(beta):
    """P₂(cos ψ) = 0 at arccos(1/√3), so L = 1 independent of β.

    This is the diagnostic for the dipole convention: it holds for
    ``1 - (β/2)P₂`` and for ``1 + βP₂`` alike, but only when the
    argument really is the angle those forms are written in. It fails
    immediately if the P₂ argument is rescaled or offset.
    """
    assert AngularCorrection.L_dipole(beta, MAGIC_ANGLE_DEG) == pytest.approx(
        1.0, abs=1e-12
    )


def test_the_magic_angle_is_not_where_rounding_puts_it():
    """Guard the constant above against being 'simplified' to 54.7356.

    The rounded value is off by 4e-7 in L, which is harmless in use and
    fatal to an exact assertion — the kind of edit that looks like
    tidying.
    """
    rounded = AngularCorrection.L_dipole(BETA, 54.7356)
    assert rounded != pytest.approx(1.0, abs=1e-9)
    assert rounded == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# The measured angle and the derived angle
# ---------------------------------------------------------------------------


def test_angular_distribution_derives_psi_from_the_measured_angle():
    """The high-level entry point is the one that takes θ.

    Equivalently: a caller who reaches for ``L_dipole`` must do this
    subtraction themselves, and that is the step that went wrong.
    """
    _params()
    result = AngularCorrection.angular_distribution(
        "Si", "1s", SI_1S_HV, THETA_FAN, xray_from_normal_deg=XRAY_FROM_NORMAL
    )
    expected_psi = XRAY_FROM_NORMAL - THETA_FAN
    assert result["psi"] == pytest.approx(expected_psi)
    assert result["L_dipole"] == pytest.approx(
        AngularCorrection.L_dipole(result["beta"], expected_psi)
    )


def test_passing_theta_where_psi_belongs_costs_a_factor_of_twenty():
    """Both numbers are plausible; that is the whole problem.

    Pinned side by side deliberately. If someone later makes the
    primitives reject out-of-range input, this test should fail and be
    rewritten — it is not asserting that the wrong call *should* work,
    only recording what it currently returns.
    """
    theta = 8.78
    psi = XRAY_FROM_NORMAL - theta

    wrong = float(AngularCorrection.L_dipole(BETA, theta))
    right = float(AngularCorrection.L_dipole(BETA, psi))

    assert wrong == pytest.approx(0.0709, abs=5e-5)
    assert right == pytest.approx(1.4309, abs=5e-5)
    assert right / wrong == pytest.approx(20.2, rel=0.02)
    # Neither is out of range, so no bounds check could have caught it.
    assert 0.0 < wrong < 2.0 and 0.0 < right < 2.0


# ---------------------------------------------------------------------------
# The incidence angle has no safe default
# ---------------------------------------------------------------------------


def test_leaving_the_incidence_angle_unstated_warns():
    _params()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        AngularCorrection.angular_distribution("Si", "1s", SI_1S_HV, THETA_FAN)
    assert [w for w in record if issubclass(w.category, UserWarning)]


def test_stating_the_same_value_explicitly_does_not_warn():
    """An explicit 88.0 means the geometry was considered. The sentinel
    exists to tell that apart from inheriting a placeholder."""
    _params()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        AngularCorrection.angular_distribution(
            "Si", "1s", SI_1S_HV, THETA_FAN, xray_from_normal_deg=88.0
        )
    assert not [w for w in record if issubclass(w.category, UserWarning)]


def test_the_warning_does_not_change_any_number():
    _params()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        defaulted = AngularCorrection.angular_distribution(
            "Si", "1s", SI_1S_HV, THETA_FAN
        )
    stated = AngularCorrection.angular_distribution(
        "Si", "1s", SI_1S_HV, THETA_FAN, xray_from_normal_deg=88.0
    )
    assert defaulted["L_dipole"] == pytest.approx(stated["L_dipole"])
    assert defaulted["psi"] == pytest.approx(stated["psi"])


def test_the_incidence_angle_changes_the_shape_not_just_the_scale():
    """Why the default cannot be waved through.

    If the assumption only rescaled the correction it could be absorbed
    downstream. It does not: the peak-to-peak variation of L across the
    emission fan, relative to its mean, more than doubles between 88°
    and 55° incidence. The same quantity is quoted in the warning text,
    so the two cannot drift apart.
    """
    _params()

    def spread(xray):
        """Peak-to-peak L over the fan, as a fraction of the mean."""
        L = AngularCorrection.angular_distribution(
            "Si", "1s", SI_1S_HV, THETA_FAN, xray_from_normal_deg=xray
        )["L_dipole"]
        return (L.max() - L.min()) / L.mean()

    assert spread(88.0) == pytest.approx(0.854, abs=0.01)
    assert spread(60.0) == pytest.approx(1.940, abs=0.01)
    assert spread(55.0) == pytest.approx(2.163, abs=0.01)


# ---------------------------------------------------------------------------
# L_full: recorded, not endorsed
# ---------------------------------------------------------------------------


def test_l_full_current_values_are_pinned_pending_the_primary_source():
    """A record of present behaviour, **not** a correctness claim.

    ``L_full``'s convention is unresolved (see its docstring). These
    numbers are pinned so that resolving it produces a visible diff
    instead of a silent change in every downstream result.
    """
    _params()
    result = AngularCorrection.angular_distribution(
        "Si", "1s", SI_1S_HV, THETA_FAN, xray_from_normal_deg=XRAY_FROM_NORMAL
    )
    expected = np.array(
        [1.2660, 1.3846, 1.4646, 1.5098, 1.5268, 1.5241, 1.5108, 1.4952]
    )
    assert result["L_full"] == pytest.approx(expected, abs=5e-4)

    # The non-dipole term is not a small correction here: it runs from
    # 4.5% of the dipole value at grazing emission to 132% at 51.6°.
    ratio = (result["L_full"] - result["L_dipole"]) / result["L_dipole"]
    assert ratio.min() == pytest.approx(0.045, abs=0.005)
    assert ratio.max() == pytest.approx(1.319, abs=0.01)


def test_l_unpolarized_is_not_the_polarization_average_of_l_full():
    """Pin the inconsistency itself, so it cannot be lost track of.

    ``L_unpolarized``'s docstring states it is obtained by averaging
    ``L_full`` over all ε ⊥ k. For the current implementation the
    average over φ sends cos φ to zero, which leaves ``L_dipole``
    instead. This test passing means the discrepancy is still there;
    when the convention is settled it should fail and be replaced by
    the relation that does hold.
    """
    phi = np.linspace(0.0, 360.0, 721)[:-1]
    for angle in (36.41, 79.22, 120.0):
        averaged = np.mean(
            [AngularCorrection.L_full(BETA, GAMMA, DELTA, angle, p) for p in phi]
        )
        assert averaged == pytest.approx(
            float(AngularCorrection.L_dipole(BETA, angle)), abs=1e-6
        )
        assert averaged != pytest.approx(
            float(AngularCorrection.L_unpolarized(BETA, GAMMA, DELTA, angle)),
            rel=0.02,
        )


def test_l_dipole_and_l_unpolarized_each_match_the_module_docstring():
    """The two forms that are *not* in question, pinned as the contrast.

    L_unpol = 1 − (β/2)P₂(cos α) + (δ + γ/2 sin²α) cos α, and L_dipole
    is its dipole part. Both are reproduced here from the formula
    rather than called, so a change to either implementation shows up.
    """
    for angle in (0.0, 36.41, 79.22, 120.0):
        a = np.radians(angle)
        p2 = (3.0 * np.cos(a) ** 2 - 1.0) / 2.0
        dipole = 1.0 - (BETA / 2.0) * p2
        unpol = dipole + (DELTA + GAMMA / 2.0 * np.sin(a) ** 2) * np.cos(a)

        assert AngularCorrection.L_dipole(BETA, angle) == pytest.approx(dipole)
        assert AngularCorrection.L_unpolarized(
            BETA, GAMMA, DELTA, angle
        ) == pytest.approx(unpol)
