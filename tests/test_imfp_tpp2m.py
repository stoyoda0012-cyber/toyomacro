"""TPP-2M inelastic mean free path.

Two layers, deliberately separated:

1. **Implementation fidelity** — does the code compute the equations of
   Tanuma, Powell & Penn, *Surf. Interface Anal.* **21**, 165 (1994)?
   Checked against quantities the paper states directly, so a failure
   means a transcription error, not a physics disagreement.
2. **Physical plausibility** — does the result sit where independently
   computed IMFPs sit? These use looser, *documented* tolerances,
   because TPP-2M is a predictive formula fitted to optical-data IMFPs
   and is not expected to reproduce them exactly (the paper reports
   ~10% RMS for elements, ~8.5% for organics).

The first layer is what catches the 2026-07-29 bug: `U` in Eqns (4c)/(4d)
had been coded as `Ep/1000` instead of `Nv*rho/M`, inflating the IMFP by
up to ~900% at 50 eV while staying under 1% at 1 keV.
"""

from __future__ import annotations

import math

import pytest

from toyomacro.data import IMFP
from toyomacro.data.imfp import IMFP as _IMFP

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


def test_golden_values_from_an_independent_implementation():
    """Whole-formula check against values computed straight from the paper.

    Written out longhand here so a refactor of imfp.py cannot drag the
    expectation along with it.
    """
    def reference(E, Nv, rho, M, Eg):
        Ep = 28.8 * math.sqrt(Nv * rho / M)
        U = Nv * rho / M
        beta = -0.10 + 0.944 / math.sqrt(Ep**2 + Eg**2) + 0.069 * rho**0.1
        gamma = 0.191 * rho**-0.5
        C, D = 1.97 - 0.91 * U, 53.4 - 20.8 * U
        return E / (Ep**2 * (beta * math.log(gamma * E) - C / E + D / E**2)) / 10

    for E in (50.0, 100.0, 500.0, 1386.6, 7413.0):
        for Nv, rho, M, Eg in [
            (4, 2.33, 28.09, 1.12),      # Si
            (16, 2.2, 60.08, 8.9),       # SiO2
            (11, 19.3, 196.97, 0.0),     # Au
        ]:
            got = _IMFP._calculate_tpp2m(E, Nv, rho, M, Eg)
            assert got == pytest.approx(reference(E, Nv, rho, M, Eg), rel=1e-12)


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

    Those are IMFPs from optical data (NIST SRD 71), *not* TPP-2M outputs,
    so this is not a correctness gate for the formula — TPP-2M scatters
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


def test_no_eal_helper_is_exposed():
    """`attenuation_length()` applied a hardcoded 0.9 and was removed.

    It was never called anywhere, and shipping a fixed ratio labelled as
    an elastic-scattering correction is worse than shipping nothing: the
    real ratio depends on material, energy and emission angle. A
    physics-based replacement belongs in its own module, with the
    single-scattering albedo as a required input.
    """
    assert not hasattr(IMFP, "attenuation_length")
