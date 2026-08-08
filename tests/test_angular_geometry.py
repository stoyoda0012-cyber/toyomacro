"""Angle conventions in ``AngularCorrection``, and what they cost.

Several angles are involved and only one is measured: θ from the
surface normal (the analyzer's axis), the angle at which the source
sits (an instrument property), and a derived angle that every ``L_*``
primitive actually wants. Passing θ where the derived angle belongs
does not raise — it returns another number between 0 and 2.

That happened in real analysis: β = 1.9255, θ = 8.78°, L = 0.0709 where
the answer was 1.4309. The tests below pin the size of that trap so it
stays visible, pin the identity that makes the convention self-checking,
and pin the geometry warning.

A second, quieter confusion sits inside the derived angle itself: the
angle from the direction *toward the source* and the angle from the
beam's *propagation direction* are supplements, and ``L_unpolarized``
wants the latter. ``L_dipole`` cannot distinguish them, so the usual
diagnostic is blind to it. The last section pins that convention from
vector geometry and from the sign of the forward-emission asymmetry —
no instrument value, no measured number.

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

# An arbitrary 8-channel emission fan, and the module's own placeholder
# incidence angle. The assertions below are about the *size of an
# error*, so these only have to stay fixed — not to be right, and not
# to describe anything. Do not read an instrument geometry off them.
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


# ---------------------------------------------------------------------------
# Which axis the unpolarized angle is measured from
# ---------------------------------------------------------------------------
#
# ``L_unpolarized`` wants α, from the beam's propagation direction k.
# The angle from the direction *toward the source*, ψ, is its supplement.
# ``L_dipole`` cannot tell the two apart, so the dipole part of a result
# is no evidence that the convention was applied correctly — which is
# what makes the mistake silent.
#
# Everything below is fixed by geometry and by the sign of the
# forward-emission asymmetry. No instrument value and no measurement is
# used, so these hold for any source position.


def _unit(angle_deg):
    """Direction at a signed angle from the outward normal, in the plane."""
    a = np.radians(angle_deg)
    return np.array([np.sin(a), 0.0, np.cos(a)])


def _kappa(source_angle_deg):
    """Propagation direction κ for a source sitting at σ_s.

    Written as a branch rather than ``σ_s − 180·sign(σ_s)`` so that
    normal incidence works: ``sign(0) = 0`` would leave κ = 0, which
    points out of the sample.
    """
    return source_angle_deg + 180.0 if source_angle_deg <= 0.0 else source_angle_deg - 180.0


def _fold(angle_deg):
    """Fold a signed angle difference into the 0°–180° range.

    κ − θ is the argument the cosine wants and is correct unfolded; this
    is only for comparing against ``arccos``, whose range is 0°–180°.
    """
    a = angle_deg % 360.0
    return 360.0 - a if a > 180.0 else a


@pytest.mark.parametrize(
    "source_deg", [-90.0, -84.0, -47.0, -33.0, -12.0, 0.0, 21.0, 47.0, 71.0, 90.0]
)
@pytest.mark.parametrize("theta", [0.0, 7.0, 24.0, 41.0, 62.0])
def test_kappa_minus_theta_is_the_angle_between_beam_and_emission(source_deg, theta):
    """κ − θ reproduces the vector angle; the source's own angle does not.

    This is the definition ``L_unpolarized`` documents, checked against
    the geometry it describes rather than against a stored number.

    Executable documentation of the geometry, not coverage: the
    assertions are trig identities among this file's own helpers and
    hold against any implementation. What they protect is the *meaning*
    assigned to ``xray_from_normal_deg``, which the module cannot state
    in code. The angles are arbitrary and deliberately not any
    instrument's.
    """
    kappa = _kappa(source_deg)
    k = _unit(kappa)  # propagation direction
    e = _unit(theta)  # emission direction
    from_vectors = np.degrees(np.arccos(np.clip(np.dot(k, e), -1.0, 1.0)))

    assert _fold(kappa - theta) == pytest.approx(from_vectors, abs=1e-9)
    # The cosine is what the formula uses, and it needs no folding.
    assert np.cos(np.radians(kappa - theta)) == pytest.approx(
        float(np.dot(k, e)), abs=1e-12
    )

    # The source's own angle gives the supplement, not the same number,
    # except where the two coincide at exactly 90°.
    psi = _fold(source_deg - theta)
    if abs(from_vectors - 90.0) > 1e-9:
        assert psi != pytest.approx(from_vectors, abs=1e-6)
        assert psi + from_vectors == pytest.approx(180.0, abs=1e-9)


def test_exact_grazing_is_the_boundary_and_is_not_treated_as_entering():
    """σ_s = ±90° is the degenerate case, and the check excludes it.

    A source in the surface plane gives κ = ∓90°: the beam is tangent,
    entering neither the sample nor the vacuum. ``|κ| > 90`` is strict,
    so this warns — which is the conservative side, since a tangent
    beam does not illuminate the surface either.
    """
    _params()
    for source_deg in (-90.0, 90.0):
        kappa = _kappa(source_deg)
        assert abs(kappa) == pytest.approx(90.0)
        assert _unit(kappa)[2] == pytest.approx(0.0, abs=1e-15)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            AngularCorrection.angular_distribution_unpolarized(
                "Si", "1s", SI_1S_HV, np.array([30.0]), xray_from_normal_deg=kappa
            )
        assert [w for w in record if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize("source_deg", [-84.0, -47.0, -33.0, 0.0, 21.0, 71.0])
def test_a_physical_beam_always_has_kappa_beyond_ninety(source_deg):
    """The check a caller can apply without knowing the instrument.

    A beam that enters the sample has a negative outward-normal
    component, and that is exactly ``|κ| > 90°``. Executable
    documentation, as above — see the note there.
    """
    kappa = _kappa(source_deg)
    assert abs(kappa) > 90.0
    assert _unit(kappa)[2] < 0.0
    # The source itself is on the vacuum side, which is why passing it
    # in place of κ is the mistake worth guarding.
    assert _unit(source_deg)[2] > 0.0


@pytest.mark.parametrize("seed", range(8))
def test_the_polarization_average_puts_alpha_on_the_beam_direction(seed):
    """What actually fixes the axis, for arbitrary β, γ, δ.

    The module states ``L_unpolarized`` is ``L_pol`` averaged over all
    ε ⊥ k, with ``L_pol = 1 + β P₂(cos θ_ε) + (δ + γ cos²θ_ε) sin θ_ε
    cos φ`` and θ_ε measured from ε. Carrying that average out here —
    from the docstring's formula, not from the implementation — lands
    on ``L_unpolarized`` evaluated at the angle from **k**, and not at
    the angle from the direction toward the source.

    Signs are randomised because the point is that this holds for any
    parameters, unlike "forward emission is enhanced", which fails on a
    third of the bundled table — see
    ``test_the_documented_negative_fractions_come_from_the_bundled_tables``.
    """
    rng = np.random.default_rng(seed)
    beta = rng.uniform(-1.0, 2.0)
    gamma = rng.uniform(-2.0, 2.0)
    delta = rng.uniform(-0.5, 0.5)
    alpha_deg = rng.uniform(5.0, 175.0)

    # Emission direction p̂, with k̂ = ẑ, at α from the beam.
    a = np.radians(alpha_deg)
    p = np.array([np.sin(a), 0.0, np.cos(a)])

    # Average L_pol over ε ⊥ k, i.e. ε(η) = (cos η, sin η, 0).
    eta = np.linspace(0.0, 2.0 * np.pi, 20001)[:-1]
    eps = np.stack([np.cos(eta), np.sin(eta), np.zeros_like(eta)], axis=1)
    cos_theta_eps = eps @ p
    # φ is the azimuth about ε from the (ε, k) plane, with φ = 0 taken in
    # the half containing +k. That choice is the whole question: the other
    # half gives -k̂·p̂ and lands the average on ψ instead of α. It is a
    # convention of the source formula, not something this average derives,
    # and randomising β, γ and δ does not exercise it.
    sin_cos_phi = float(p[2])

    p2 = (3.0 * cos_theta_eps**2 - 1.0) / 2.0
    averaged = np.mean(
        1.0 + beta * p2 + (delta + gamma * cos_theta_eps**2) * sin_cos_phi
    )

    from_k = float(AngularCorrection.L_unpolarized(beta, gamma, delta, alpha_deg))
    from_source = float(
        AngularCorrection.L_unpolarized(beta, gamma, delta, 180.0 - alpha_deg)
    )

    assert averaged == pytest.approx(from_k, abs=1e-9)
    # And it is the angle from k specifically: the other choice differs
    # by twice the non-dipole term, except where that term vanishes.
    if abs(alpha_deg - 90.0) > 1.0:
        assert averaged != pytest.approx(from_source, abs=1e-6)


def test_the_non_dipole_term_is_odd_about_ninety_degrees():
    """The exact invariant, stated without appeal to forward peaking.

    Swapping α for its supplement negates the non-dipole term and
    leaves the dipole part alone — for any parameters, whatever the
    sign of ``δ + γ/2 sin²α``. At 90° the term vanishes outright.
    """
    for beta, gamma, delta in [
        (BETA, GAMMA, DELTA),
        (0.4, -1.8, 0.3),  # γ < 0: forward suppressed
        (1.2, 0.02, -0.4),  # γ > 0 but δ dominates and reverses it
    ]:
        for alpha in (10.0, 45.0, 89.0, 130.0):
            up = float(AngularCorrection.L_unpolarized(beta, gamma, delta, alpha))
            down = float(
                AngularCorrection.L_unpolarized(beta, gamma, delta, 180.0 - alpha)
            )
            dip = float(AngularCorrection.L_dipole(beta, alpha))
            assert (up - dip) == pytest.approx(-(down - dip), abs=1e-12)

        assert AngularCorrection.L_unpolarized(
            beta, gamma, delta, 90.0
        ) == pytest.approx(float(AngularCorrection.L_dipole(beta, 90.0)))


def test_the_documented_negative_fractions_come_from_the_bundled_tables():
    """Recompute the percentages the module docstring quotes.

    The docstring warns against "forward emission is enhanced" as a
    geometry check and backs it with three fractions. Those are claims
    about the bundled data, so they are recomputed here rather than
    trusted — an earlier draft quoted 13.8%, which is not any of them.

    ``δ + γ/2·sin²α`` is linear in ``sin²α ∈ (0, 1)`` over
    0° < α < 90°, so it goes negative somewhere in that range exactly
    when ``min(δ, δ + γ/2) < 0``. That equivalence is the definition
    being counted; it is stated so the numbers below mean something.
    """
    from toyomacro.data.paths import load_trzh2018_data, load_trzh2019_data

    rows = []
    for data in (load_trzh2018_data(), load_trzh2019_data()):
        for element, orbitals in data.get("data", {}).items():
            for orbital, record in orbitals.items():
                gammas = record.get("gamma") or []
                deltas = record.get("delta") or []
                for g, d in zip(gammas, deltas):
                    if g is None or d is None:
                        continue
                    rows.append((element, orbital, float(g), float(d)))
    if not rows:
        pytest.skip("bundled Trzhaskovskaya angular tables unavailable")

    gamma = np.array([r[2] for r in rows])
    delta = np.array([r[3] for r in rows])
    dips = np.minimum(delta, delta + gamma / 2.0) < 0.0

    assert dips.mean() * 100 == pytest.approx(35.3, abs=0.1)
    assert ((gamma > 0) & dips).mean() * 100 == pytest.approx(21.4, abs=0.1)

    per_subshell = {}
    for element, orbital, g, d in rows:
        per_subshell.setdefault((element, orbital), []).append(
            min(d, d + g / 2.0) < 0.0
        )
    assert (
        np.mean([any(v) for v in per_subshell.values()]) * 100
        == pytest.approx(71.9, abs=0.1)
    )


def test_forward_peaking_is_an_expectation_not_a_diagnostic():
    """Guard the docstring's warning against being 'simplified' away.

    It would be tempting to check a geometry by asserting that emission
    along k is enhanced. The bundled tables refuse: Tm 2p3/2 at the
    Ga Kα energy is backward-peaked, so that check would condemn a
    correct κ. Pinned so the caveat cannot be dropped as pedantry.
    """
    p = AngularCorrection.lookup("Tm", "2p3/2", SI_1S_HV)
    if p is None:
        pytest.skip("bundled Trzhaskovskaya angular tables unavailable")
    forward = float(
        AngularCorrection.L_unpolarized(p["beta"], p["gamma"], p["delta"], 45.0)
    )
    backward = float(
        AngularCorrection.L_unpolarized(p["beta"], p["gamma"], p["delta"], 135.0)
    )
    dip = float(AngularCorrection.L_dipole(p["beta"], 45.0))
    assert forward < dip < backward


def test_supplying_psi_instead_of_alpha_is_invisible_in_the_dipole_part():
    """Why this had to be pinned: the wrong call looks entirely normal.

    Swapping α for its supplement leaves ``L_dipole`` unchanged to
    within rounding — P₂ is even under ``cos → −cos``, and the residue
    is a last-ULP artefact of computing two different cosines, not a
    real difference. The non-dipole term, meanwhile, inverts exactly.
    Nothing is out of range, and the dipole-only diagnostic that catches
    the θ-for-α trap is blind here.
    """
    for alpha in (66.7, 97.0, 109.5):
        psi = 180.0 - alpha
        assert float(AngularCorrection.L_dipole(BETA, psi)) == pytest.approx(
            float(AngularCorrection.L_dipole(BETA, alpha)), rel=1e-12
        )

        d = float(AngularCorrection.L_dipole(BETA, alpha))
        right = float(AngularCorrection.L_unpolarized(BETA, GAMMA, DELTA, alpha))
        wrong = float(AngularCorrection.L_unpolarized(BETA, GAMMA, DELTA, psi))
        assert (right - d) == pytest.approx(-(wrong - d), abs=1e-12)
        assert 0.0 < wrong < 2.0 and 0.0 < right < 2.0


def test_the_entry_point_forms_alpha_from_kappa():
    """``angular_distribution_unpolarized`` applies κ − θ, as documented."""
    _params()
    theta = np.array([10.0, 25.0, 40.0, 55.0])
    kappa = 133.0  # arbitrary valid κ; not any instrument's
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a valid κ must not warn
        got = AngularCorrection.angular_distribution_unpolarized(
            "Si", "1s", SI_1S_HV, theta, xray_from_normal_deg=kappa
        )
    assert got["alpha"] == pytest.approx(kappa - theta)
    assert got["L_unpolarized"] == pytest.approx(
        AngularCorrection.L_unpolarized(
            got["beta"], got["gamma"], got["delta"], kappa - theta
        )
    )


@pytest.mark.parametrize("bad", [47.0, -47.0, 88.0, 0.0, 90.0, 350.0, -710.0])
def test_a_beam_leaving_the_sample_warns(bad):
    """The source's own angle is the natural wrong input; it warns.

    ``350.0`` and ``-710.0`` exercise the wrap: both reduce to 10° from
    the normal, still on the vacuum side.
    """
    _params()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        AngularCorrection.angular_distribution_unpolarized(
            "Si", "1s", SI_1S_HV, np.array([30.0]), xray_from_normal_deg=bad
        )
    issued = [w for w in record if issubclass(w.category, UserWarning)]
    assert issued
    # The remedy the message offers must not be the form the module
    # documents as wrong at normal incidence: sign(0) = 0 leaves κ = 0.
    text = str(issued[0].message)
    assert "sign(σ_s)" not in text and "sign(sigma_s)" not in text
    assert "σ_s + 180°" in text


@pytest.mark.parametrize("good", [90.5, 91.0, 133.0, -133.0, 180.0, 220.0, -220.0])
def test_a_beam_entering_the_sample_does_not_warn(good):
    """The complement of the above, including wrapped-but-valid values."""
    _params()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        AngularCorrection.angular_distribution_unpolarized(
            "Si", "1s", SI_1S_HV, np.array([30.0]), xray_from_normal_deg=good
        )
    assert not [w for w in record if issubclass(w.category, UserWarning)]


def test_the_unstated_default_warns_once_and_only_about_being_unstated():
    """The sentinel path: the placeholder 88° is not a valid κ either.

    It still reports exactly one warning, because the beam-direction
    check defers to the more specific placeholder message rather than
    stacking a second one on the same value.
    """
    _params()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        AngularCorrection.angular_distribution_unpolarized(
            "Si", "1s", SI_1S_HV, np.array([30.0])
        )
    issued = [w for w in record if issubclass(w.category, UserWarning)]
    assert len(issued) == 1
    assert "placeholder" in str(issued[0].message)


def test_the_beam_direction_warning_does_not_change_any_number():
    """Warning only — every returned value matches an identical unwarned call.

    Named for what it must show, so it compares the two runs rather than
    re-deriving alpha. 47° trips the check (|κ| < 90°); 137° is its
    supplement and does not, so the pair differs only in whether the
    warning fires.
    """
    _params()
    theta = np.array([10.0, 30.0, 50.0])

    def run(xray):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out = AngularCorrection.angular_distribution_unpolarized(
                "Si", "1s", SI_1S_HV, theta, xray_from_normal_deg=xray
            )
        fired = [w for w in record if issubclass(w.category, UserWarning)]
        return out, fired

    warned, warned_issues = run(47.0)
    quiet, quiet_issues = run(-47.0 + 180.0)  # same |κ| geometry, no warning

    assert len(warned_issues) == 1, "47 deg should trip the beam-direction check"
    assert quiet_issues == [], "133 deg is a beam entering the sample"

    assert warned["alpha"] == pytest.approx(47.0 - theta)
    for key, value in warned.items():
        if isinstance(value, np.ndarray):
            assert np.all(np.isfinite(value)), key
