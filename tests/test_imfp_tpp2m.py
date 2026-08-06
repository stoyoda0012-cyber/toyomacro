"""TPP-2M inelastic mean free path.

Two layers, deliberately separated:

1. **Implementation fidelity** — does the code compute the equations of
   Tanuma, Powell & Penn, *Surf. Interface Anal.* **21**, 165 (1994)?
   Checked against quantities the paper states directly, so a failure
   means a transcription error, not a physics disagreement.
2. **Physical plausibility** — does the result sit where independently
   computed IMFPs sit? These use looser, *documented* tolerances,
   because TPP-2M is a predictive formula fitted to optical-data IMFPs
   and is not expected to reproduce them exactly. Table 9 of the paper
   gives average RMS deviations of 10.2% (elements), 8.5% (organic
   compounds) and 18.9% (inorganic compounds), and the text notes the
   largest deviations occur below 200 eV. So a plausibility test that
   demanded better than ~10% would be testing luck, not the formula.

The first layer is what catches the 2026-07-29 bug: `U` in Eqns (4c)/(4d)
had been coded as `Ep/1000` instead of `Nv*rho/M`, inflating the IMFP by
up to ~900% at 50 eV while staying under 1% at 1 keV.
"""

from __future__ import annotations

import math

import pytest

from toyomacro.data import IMFP
from toyomacro.data.imfp import IMFP as _IMFP
from toyomacro.data.imfp import TPP2M_FITTED_RANGE_EV

# --- Table 5 of TPP 1994: material parameters and the paper's own Ep ------
# (rho g/cm^3, M g/mol, Nv, Ep_paper eV, Eg eV)
TPP_TABLE5 = {
    "26-n-paraffin": (0.99, 366.7, 158, 18.8, 6.0),
    "adenine": (1.35, 135.1, 50, 20.4, 3.8),
    "beta-carotene": (0.993, 536.9, 216, 18.2, 0.0),
    "BPA": (1.14, 66000.0, 25700, 19.2, 5.5),
    "DNA": (1.35, 664.0, 238, 20.0, 4.0),
    "diphenyl-hexatriene": (0.986, 232.3, 88, 17.6, 0.0),
    "guanine": (1.58, 151.1, 56, 22.0, 2.5),
    "kapton": (1.36, 382.3, 138, 20.2, 3.5),
    "polyacetylene": (1.13, 26.0, 10, 19.0, 1.2),
    "poly(butene-1-sulfone)": (1.39, 120.2, 42, 20.1, 6.0),
    "polyethylene": (0.93, 28.1, 12, 18.2, 7.0),
    "PMMA": (1.19, 100.1, 40, 19.9, 5.0),
    "polystyrene": (1.05, 104.2, 40, 18.3, 4.5),
    "poly(2-vinylpyridine)": (1.01, 105.1, 40, 17.9, 4.0),
}


# ====================== 1. implementation fidelity =======================

@pytest.mark.parametrize("name", sorted(TPP_TABLE5))
def test_U_identity_holds(name):
    """Eqn (4e): U = Nv*rho/M = Ep^2/829.4.

    Exact against 28.8^2; the paper's 829.4 is that rounded, so the
    published form agrees only to ~5e-5. Asserting 1e-12 against 829.4
    would fail for a correct implementation — the looser bound is the
    paper's rounding, not our slack. Coding U as anything else
    (`Ep/1000`, say) misses by a factor of 20-36 and fails either way.
    """
    rho, M, Nv, _, _ = TPP_TABLE5[name]
    U = Nv * rho / M
    Ep = 28.8 * math.sqrt(U)

    assert math.isclose(U, Ep**2 / 28.8**2, rel_tol=1e-12)
    assert math.isclose(U, Ep**2 / 829.4, rel_tol=1e-4)


@pytest.mark.parametrize("name", sorted(TPP_TABLE5))
def test_plasmon_energy_matches_paper_table5(name):
    """Ep = 28.8 sqrt(Nv rho / M) against the paper's tabulated Ep.

    Table 5 quotes Ep to one decimal, so agreement is limited by its own
    rounding, not by us — 1% is generous for that.
    """
    rho, M, Nv, Ep_paper, _ = TPP_TABLE5[name]

    assert 28.8 * math.sqrt(Nv * rho / M) == pytest.approx(Ep_paper, rel=0.01)


def test_C_and_D_use_U_not_plasmon_energy():
    """Regression: Eqns (4c)/(4d) take U, whose scale is ~0.1-1 mol/cm^3.

    For Au, U = 1.08 while Ep/1000 = 0.030 — a factor of 36. Pin the
    coefficients so the substitution cannot silently return.
    """
    Nv, rho, M = 11, 19.3, 196.97
    U = Nv * rho / M

    assert U == pytest.approx(1.0778, rel=1e-3)
    assert 1.97 - 0.91 * U == pytest.approx(0.98918, rel=1e-4)   # C, Eqn (4c)
    assert 53.4 - 20.8 * U == pytest.approx(30.98115, rel=1e-4)  # D, Eqn (4d)
    # The value the buggy version used, kept as a negative control.
    assert 28.8 * math.sqrt(U) / 1000 == pytest.approx(0.02989, rel=1e-3)
    assert U / (28.8 * math.sqrt(U) / 1000) == pytest.approx(36.1, rel=0.01)


def test_beta_is_eqn8_not_eqn4a():
    """TPP-2M differs from TPP-2 *only* in using Eqn (8) for beta (p. 172).

    Deliberately not compared against Table 4: those beta/gamma/C/D are
    fitted separately for each compound, and the predictive equations
    scatter about them by design (Figs. 14-16). Table 4 is not a golden
    source for the predictive formula.
    """
    rho, M, Nv, _, Eg = TPP_TABLE5["PMMA"]
    Ep = 28.8 * math.sqrt(Nv * rho / M)
    U = Nv * rho / M
    gamma = 0.191 * rho**-0.5
    C, D = 1.97 - 0.91 * U, 53.4 - 20.8 * U

    beta_2m = -0.10 + 0.944 / math.sqrt(Ep**2 + Eg**2) + 0.069 * rho**0.1
    beta_2 = -0.0216 + 0.944 / math.sqrt(Ep**2 + Eg**2) + 7.39e-4 * rho
    assert not math.isclose(beta_2m, beta_2, rel_tol=1e-2)

    def imfp_with(beta):
        E = 1000.0
        return E / (Ep**2 * (beta * math.log(gamma * E) - C / E + D / E**2)) / 10

    got = _IMFP._calculate_tpp2m(1000.0, Nv, rho, M, Eg)
    assert got == pytest.approx(imfp_with(beta_2m), rel=1e-12)
    assert not math.isclose(got, imfp_with(beta_2), rel_tol=1e-2)


def test_gamma_is_eqn4b():
    """Eqn (4b): gamma = 0.191 rho^-0.5, density in g/cm^3."""
    for rho in (0.93, 1.19, 2.33, 19.3):
        assert 0.191 * rho**-0.5 == pytest.approx(0.191 / math.sqrt(rho), rel=1e-12)


# The canonical regression grid. Si (semiconductor), SiO2 (insulator,
# and the group with the paper's worst RMS) and Au (metal, Eg=0, and the
# material where the `Ep/1000` bug was largest). Energies span the low
# end where C/E and D/E^2 dominate, the middle of the fitted range, the
# Al Ka line, and a Ga Ka HAXPES energy far outside it.
GOLDEN_MATERIALS = {
    "Si":   (4, 2.33, 28.09, 1.12),
    "SiO2": (16, 2.2, 60.08, 8.9),
    "Au":   (11, 19.3, 196.97, 0.0),
}
GOLDEN_ENERGIES = (50.0, 100.0, 500.0, 1000.0, 1386.6, 7413.0)


def _reference_imfp_angstrom(E, Nv, rho, M, Eg):
    """TPP-2M transcribed independently from the paper, in Angstroms.

    Eqns (3), (4b), (4c), (4d), (4e), (8). Deliberately does not import
    anything from `imfp.py`, so a refactor there cannot drag the
    expectation along with it. Returns Angstroms — the unit Eqn (3)
    itself is in — so the nm conversion is tested rather than assumed.
    """
    Ep = 28.8 * math.sqrt(Nv * rho / M)
    U = Nv * rho / M
    beta = -0.10 + 0.944 / math.sqrt(Ep**2 + Eg**2) + 0.069 * rho**0.1
    gamma = 0.191 * rho**-0.5
    C, D = 1.97 - 0.91 * U, 53.4 - 20.8 * U
    return E / (Ep**2 * (beta * math.log(gamma * E) - C / E + D / E**2))


@pytest.mark.parametrize("energy_ev", GOLDEN_ENERGIES)
@pytest.mark.parametrize("material", sorted(GOLDEN_MATERIALS))
def test_golden_values_from_an_independent_implementation(material, energy_ev):
    """Whole-formula check against values computed straight from the paper."""
    Nv, rho, M, Eg = GOLDEN_MATERIALS[material]

    got_nm = _IMFP._calculate_tpp2m(energy_ev, Nv, rho, M, Eg)

    assert got_nm == pytest.approx(
        _reference_imfp_angstrom(energy_ev, Nv, rho, M, Eg) / 10.0, rel=1e-12
    )


@pytest.mark.parametrize("energy_ev", GOLDEN_ENERGIES)
@pytest.mark.parametrize("material", sorted(GOLDEN_MATERIALS))
def test_public_api_returns_nm_for_the_angstrom_formula(material, energy_ev):
    """Eqn (3) is in Angstroms; `tpp2m()` is documented as nm.

    Pinned as a ratio rather than a magnitude, so this fails on a unit
    slip and on nothing else.
    """
    Nv, rho, M, Eg = GOLDEN_MATERIALS[material]

    got_nm = IMFP.tpp2m(energy_ev, Nv=Nv, density=rho, Mw=M, Eg=Eg)

    assert got_nm * 10.0 == pytest.approx(
        _reference_imfp_angstrom(energy_ev, Nv, rho, M, Eg), rel=1e-12
    )


# Ep, U and the four Eqn (3) parameters for the golden set, evaluated
# from Eqns (4b), (4c), (4d), (4e) and (8) outside this file. Frozen as
# literals: they are what pins `_reference_imfp_angstrom`, which is
# otherwise just a second copy of the formula and could drift with it.
GOLDEN_INTERMEDIATES = {
    #          Ep          U           beta         gamma        C           D
    "Si":   (16.589167, 0.33179067, 0.031865804, 0.12512826, 1.6680705, 46.498754),
    "SiO2": (22.044429, 0.58588549, 0.014369095, 0.12877217, 1.4368442, 41.213582),
    "Au":   (29.899742, 1.0778291,  0.024341594, 0.043476514, 0.98917551, 30.981154),
}


@pytest.mark.parametrize("material", sorted(GOLDEN_INTERMEDIATES))
def test_intermediate_quantities(material):
    """Ep, U and the four Eqn (3) parameters, each pinned individually.

    Without this, a sign error in one parameter compensated by another
    would pass the whole-formula check. U in particular is the quantity
    the 2026-07-29 bug got wrong, and its scale (~0.1-1 mol/cm^3) is
    what distinguishes it from every other candidate.
    """
    Nv, rho, M, Eg = GOLDEN_MATERIALS[material]
    Ep, U, beta, gamma, C, D = GOLDEN_INTERMEDIATES[material]

    assert 28.8 * math.sqrt(Nv * rho / M) == pytest.approx(Ep, rel=1e-7)
    assert Nv * rho / M == pytest.approx(U, rel=1e-7)
    assert -0.10 + 0.944 / math.sqrt(Ep**2 + Eg**2) + 0.069 * rho**0.1 == (
        pytest.approx(beta, rel=1e-6)
    )
    assert 0.191 * rho**-0.5 == pytest.approx(gamma, rel=1e-7)
    assert 1.97 - 0.91 * U == pytest.approx(C, rel=1e-7)
    assert 53.4 - 20.8 * U == pytest.approx(D, rel=1e-7)


@pytest.mark.parametrize("energy_ev", GOLDEN_ENERGIES)
@pytest.mark.parametrize("material", sorted(GOLDEN_INTERMEDIATES))
def test_imfp_is_eqn3_of_the_pinned_intermediates(material, energy_ev):
    """Close the loop: `imfp.py` must equal Eqn (3) built from those.

    This is the one assertion that ties the frozen intermediates above
    to the shipped function, rather than to a second transcription of
    the formula living in this file.
    """
    Nv, rho, M, Eg = GOLDEN_MATERIALS[material]
    _, _, beta, gamma, C, D = GOLDEN_INTERMEDIATES[material]
    Ep = GOLDEN_INTERMEDIATES[material][0]

    expected_angstrom = energy_ev / (
        Ep**2
        * (beta * math.log(gamma * energy_ev) - C / energy_ev + D / energy_ev**2)
    )

    got_nm = _IMFP._calculate_tpp2m(energy_ev, Nv, rho, M, Eg)

    assert got_nm * 10.0 == pytest.approx(expected_angstrom, rel=1e-7)


def test_metals_take_eg_zero_by_default():
    """The paper defines Eg for non-conductors; conductors use 0.

    Au carries Eg=0 in `compounds.json`, so omitting it must not change
    the answer — and must match passing it explicitly.
    """
    au = dict(Nv=11, density=19.3, Mw=196.97)

    assert IMFP.tpp2m(1000, **au) == pytest.approx(
        IMFP.tpp2m(1000, Eg=0.0, **au), rel=1e-12
    )
    # A non-zero gap lengthens the IMFP, so the default is not inert.
    assert IMFP.tpp2m(1000, Eg=5.0, **au) > IMFP.tpp2m(1000, **au)


# ======================= 1b. domain and bad input ========================

@pytest.mark.parametrize("energy_ev", [0.0, -100.0, float("nan"), float("inf")])
def test_non_physical_energy_is_rejected(energy_ev):
    """`math.log` used to raise a bare 'math domain error' for E <= 0."""
    with pytest.raises(ValueError, match="kinetic energy"):
        IMFP.tpp2m(energy_ev, compound="Si")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(Nv=0, density=2.33, Mw=28.09), "Nv"),
        (dict(Nv=-4, density=2.33, Mw=28.09), "Nv"),
        (dict(Nv=4, density=0.0, Mw=28.09), "density"),
        (dict(Nv=4, density=-2.33, Mw=28.09), "density"),
        (dict(Nv=4, density=2.33, Mw=0.0), "molecular weight"),
        (dict(Nv=4, density=2.33, Mw=-28.09), "molecular weight"),
        (dict(Nv=4, density=2.33, Mw=28.09, Eg=-1.0), "band gap"),
    ],
)
def test_non_physical_material_parameters_are_rejected(kwargs, message):
    """Mw=0 used to raise ZeroDivisionError, and rho<0 a domain error."""
    with pytest.raises(ValueError, match=message):
        IMFP.tpp2m(1000.0, **kwargs)


def test_a_non_positive_bethe_bracket_raises_rather_than_returning_a_length():
    """Below the fit floor, Eqn (3) can return a *negative* IMFP.

    Reachable for a few very high density entries in the compound table
    at ~27 eV — far below the paper's 50 eV floor, but silently handing
    back a negative mean free path is worse than refusing. Seaborgium is
    used because it is one of the cases that actually triggers it.
    """
    sg = dict(Nv=6, density=35.0, Mw=271.0)

    with pytest.raises(ValueError, match="out of domain"):
        IMFP.tpp2m(27.0, **sg)

    # Well inside the fitted range the same material is fine.
    assert IMFP.tpp2m(1000.0, **sg) > 0


def test_unknown_compound_is_rejected():
    with pytest.raises(ValueError, match="not found"):
        IMFP.tpp2m(1000.0, compound="Unobtainium")


def test_missing_material_parameters_are_rejected():
    with pytest.raises(ValueError, match="Required parameters missing"):
        IMFP.tpp2m(1000.0, Nv=4, density=2.33)


def test_docstring_example_is_the_actual_return_value():
    """The docstring claimed 2.34 nm while the code returned 3.90 nm."""
    assert IMFP.tpp2m(1386.6, compound="SiO2") == pytest.approx(3.88, abs=0.01)
    assert IMFP.tpp2m(
        1386.6, Nv=16, density=2.2, Mw=60.08, Eg=8.9
    ) == pytest.approx(3.88, abs=0.01)


# ====================== 2. physical plausibility =========================

def test_monotonic_and_positive_over_the_fitted_range():
    """IMFP rises with energy above the ~60-70 eV minimum (paper, Fig. 9)."""
    prev = None
    for E in (100, 200, 500, 1000, 2000):
        val = IMFP.tpp2m(E, compound="SiO2")
        assert val > 0
        if prev is not None:
            assert val > prev
        prev = val


@pytest.mark.parametrize(
    "energy_ev, srd71_angstrom",
    [(1101, 26.4), (1155, 27.4), (1385, 31.6), (93, 5.1), (1616, 35.8)],
)
def test_elemental_silicon_tracks_optical_data_imfps(energy_ev, srd71_angstrom):
    """Sanity check for Si only, against Table I of Jablonski & Powell,
    *J. Vac. Sci. Technol. A* **27**, 253 (2009).

    Those are IMFPs from optical data — that paper attributes them to
    its own refs [5, 10, 11], not to NIST SRD 71 as an earlier version
    of this docstring said — and they are *not* TPP-2M outputs, so this
    is not a correctness gate for the formula. TPP-2M scatters
    ~10% RMS against such values across elements generally, and silicon
    simply happens to agree closely. Kept because it is an independent
    source and it fails on the `Ep/1000` bug at the low-energy point.
    """
    got_angstrom = IMFP.tpp2m(
        energy_ev, Nv=4, density=2.33, Mw=28.09, Eg=1.12
    ) * 10

    assert got_angstrom == pytest.approx(srd71_angstrom, rel=0.05)


def test_low_energy_is_where_the_bug_lived():
    """At 50 eV the C/E and D/E^2 terms dominate; Au was ~10x too large."""
    au = dict(Nv=11, density=19.3, Mw=196.97, Eg=0.0)

    assert IMFP.tpp2m(50, **au) == pytest.approx(0.486, abs=0.01)
    assert IMFP.tpp2m(50, **au) < IMFP.tpp2m(1000, **au)


def test_sampling_depth_is_three_imfps_at_95_percent():
    """d = -lambda ln(1-F); F=0.95 gives 3.00 lambda, normal emission."""
    imfp = IMFP.tpp2m(1000, compound="SiO2")

    assert IMFP.sampling_depth(1000, compound="SiO2") == pytest.approx(
        -imfp * math.log(1 - 0.95), rel=1e-12
    )
    assert IMFP.sampling_depth(
        1000, compound="SiO2", fraction=0.95
    ) == pytest.approx(2.996 * imfp, rel=1e-3)


def test_the_fitted_range_is_the_papers_and_is_not_enforced():
    """50-2000 eV is where the authors fit Eqn (3).

    The constant is exported so callers can decide for themselves;
    `tpp2m()` deliberately does not refuse energies outside it, because
    HAXPES estimates are a legitimate (if extrapolated) use. Pin both
    halves: the numbers, and the fact that nothing enforces them.
    """
    assert TPP2M_FITTED_RANGE_EV == (50.0, 2000.0)

    assert IMFP.tpp2m(30.0, compound="Si") > 0     # below the floor
    assert IMFP.tpp2m(10000.0, compound="Si") > 0  # above the ceiling


@pytest.mark.parametrize(
    "energy_ev, expected_pct",
    [(1386.6, -0.33), (7413.0, -1.82), (9251.7, -2.28), (30000.0, -7.20)],
)
def test_the_omitted_relativistic_correction_is_the_size_documented(
    energy_ev, expected_pct
):
    """`imfp.py` publishes what omitting alpha(T) costs. Pin the numbers.

    Shinotsuka, Tanuma, Powell & Penn, *Surf. Interface Anal.* **47**,
    871 (2015) give the relativistic TPP-2M as their Eqn (26): the same
    beta, gamma, C, D with T replaced by alpha(T)T in the numerator and
    inside the logarithm, C/T and D/T^2 keeping the plain T, where

        alpha(T) = (1 + T/2 m_e c^2) / (1 + T/m_e c^2)^2

    Eqn (26) is *not* implemented — the module ships the 1994
    non-relativistic form — so this reconstructs it here purely to check
    that the percentages quoted in the docstring stay true. Without
    this, a documented quantitative claim has nothing holding it.
    """
    m_e_c2 = 510998.95  # eV, CODATA electron rest energy

    def relativistic_angstrom(E, Nv, rho, M, Eg):
        Ep = 28.8 * math.sqrt(Nv * rho / M)
        U = Nv * rho / M
        beta = -0.10 + 0.944 / math.sqrt(Ep**2 + Eg**2) + 0.069 * rho**0.1
        gamma = 0.191 * rho**-0.5
        C, D = 1.97 - 0.91 * U, 53.4 - 20.8 * U
        alpha = (1 + E / (2 * m_e_c2)) / (1 + E / m_e_c2) ** 2
        return (alpha * E) / (
            Ep**2 * (beta * math.log(gamma * alpha * E) - C / E + D / E**2)
        )

    for material, (Nv, rho, M, Eg) in GOLDEN_MATERIALS.items():
        shipped = _reference_imfp_angstrom(energy_ev, Nv, rho, M, Eg)
        relativistic = relativistic_angstrom(energy_ev, Nv, rho, M, Eg)
        pct = 100.0 * (relativistic - shipped) / shipped

        # The docstring says "roughly", and it has to: the correction is
        # material-dependent through gamma, so at 30 keV Au sits at
        # -7.06% against Si's -7.20%. 0.2 points covers the observed
        # spread and still fails on a wrong alpha or a dropped factor.
        assert pct == pytest.approx(expected_pct, abs=0.2), material
        # One-sided, as documented — never a wash, never positive.
        assert pct < 0.0, material


def test_the_bracket_root_sits_below_the_documented_38_4_eV():
    """The docstring commits to "below 38.4 eV" for every bundled material.

    The excluded region is a *band*, not a floor: at 1 eV the D/E^2 term
    dominates and the bracket is large and positive, it dips negative in
    the twenties-to-thirties for a few very dense materials, then turns
    positive again. So find the band's upper edge, which is the number
    the docstring actually commits to, rather than assuming the formula
    fails monotonically below some energy.

    Bisected rather than spot-checked, so a future compound-table entry
    with a higher density cannot quietly invalidate the claim.
    """
    from toyomacro.data import CompoundDB

    worst = 0.0
    affected = []
    for name in CompoundDB.list_compounds():
        props = CompoundDB.get_properties(name)
        args = (props["Nv"], props["density"], props["Mw"], props.get("Eg", 0.0))

        # Locate any energy inside the excluded band.
        inside = None
        for i in range(1, 600):
            probe = i * 0.1
            try:
                _IMFP._calculate_tpp2m(probe, *args)
            except ValueError:
                inside = probe
                break
        if inside is None:
            continue
        affected.append(name)

        # Bisect the band's upper edge between `inside` and a known-good E.
        lo, hi = inside, 60.0
        for _ in range(60):
            mid = (lo + hi) / 2
            try:
                _IMFP._calculate_tpp2m(mid, *args)
            except ValueError:
                lo = mid
            else:
                hi = mid
        worst = max(worst, hi)

    assert affected == ["Sg", "Bh", "Hs"], affected
    assert 0.0 < worst < 38.4
    assert worst < TPP2M_FITTED_RANGE_EV[0]  # and below the fit floor


@pytest.mark.parametrize("fraction", [1.0, 1.5, -0.1, float("nan")])
def test_sampling_depth_rejects_impossible_signal_fractions(fraction):
    """F=1 is reached only at infinite depth; ln(0) is not an answer."""
    with pytest.raises(ValueError, match="fraction"):
        IMFP.sampling_depth(1000, compound="SiO2", fraction=fraction)


def test_no_eal_helper_is_exposed():
    """`attenuation_length()` applied a hardcoded 0.9 and was removed.

    It was never called anywhere, and shipping a fixed ratio labelled as
    an elastic-scattering correction is worse than shipping nothing: the
    real ratio depends on material, energy and emission angle. A
    physics-based replacement belongs in its own module, with the
    single-scattering albedo as a required input.
    """
    assert not hasattr(IMFP, "attenuation_length")
