"""VarProFitter options: joint background columns, fixed widths, batch layout.

Covers the ``background_design`` and ``fixed_shape`` options of
``fit_single``/``fit_batch`` and the ``energy_axis`` batch layout. All fixtures
are synthetic and noise-free; nothing here is instrument data.
"""
import numpy as np
import pytest

from toyomacro.voigtfit import VarProFitter

ENERGY = np.linspace(95.0, 105.0, 201)
GAMMA = 0.25
CENTERS = np.array([98.8, 101.2])
SIGMAS = np.array([0.55, 0.7])
AMPS = np.array([3.0, 1.7])


def _background(energy: np.ndarray) -> np.ndarray:
    """Constant, linear ramp and a smooth step: a generic caller-built design."""
    x = (energy - energy.mean()) / np.ptp(energy)
    step = 0.5 * (1.0 + np.tanh((energy - 100.0) / 1.5))
    return np.column_stack([np.ones_like(x), x, step])


BG = _background(ENERGY)
BG_COEF = np.array([0.4, -0.3, 0.8])


def _clean(fitter: VarProFitter, amps=AMPS, bg_coef=BG_COEF) -> np.ndarray:
    phi = fitter.voigt_basis(ENERGY, CENTERS, SIGMAS, GAMMA)
    return phi @ amps + BG @ bg_coef


# ---------------------------------------------------------------- T1: background

def test_background_design_exact_recovery_noise_free():
    """With the shape at truth, amplitudes and background coefficients are exact."""
    fitter = VarProFitter(reg_lambda=0.0, bounds_center_delta=0.0)
    y = _clean(fitter)

    res = fitter.fit_single(y, ENERGY, CENTERS, SIGMAS, GAMMA,
                            background_design=BG, fixed_shape=True)

    assert res["success"]
    assert res["n_iter"] == 0
    np.testing.assert_allclose(res["amplitudes"], AMPS, rtol=0, atol=1e-10)
    np.testing.assert_allclose(res["background"], BG_COEF, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(res["centers"], CENTERS)
    np.testing.assert_array_equal(res["sigmas"], SIGMAS)
    assert res["chi2"] < 1e-20


def test_background_design_joint_solve_matches_lstsq():
    """The inner solve is one joint least-squares problem over [Φ | B]."""
    fitter = VarProFitter(reg_lambda=0.0, bounds_center_delta=0.0)
    rng = np.random.default_rng(3)
    y = _clean(fitter) + 0.05 * rng.standard_normal(len(ENERGY))

    res = fitter.fit_single(y, ENERGY, CENTERS, SIGMAS, GAMMA,
                            background_design=BG, fixed_shape=True)

    design = np.hstack([fitter.voigt_basis(ENERGY, CENTERS, SIGMAS, GAMMA), BG])
    oracle = np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(res["amplitudes"], oracle[:2], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(res["background"], oracle[2:], rtol=1e-10, atol=1e-12)


def test_background_design_free_fit_recovers_noise_free():
    """Free centers and widths still converge with background columns present."""
    fitter = VarProFitter()
    y = _clean(fitter)

    res = fitter.fit_single(y, ENERGY, CENTERS + np.array([0.3, -0.2]),
                            SIGMAS + 0.15, GAMMA, background_design=BG)

    assert res["success"]
    np.testing.assert_allclose(res["centers"], CENTERS, atol=1e-4)
    np.testing.assert_allclose(res["sigmas"], SIGMAS, atol=1e-3)
    np.testing.assert_allclose(res["amplitudes"], AMPS, rtol=1e-3)
    np.testing.assert_allclose(res["background"], BG_COEF, atol=1e-3)


def test_no_background_reports_empty_coefficients_and_is_unchanged():
    """Without a design the result is the pre-existing fit plus an empty key."""
    fitter = VarProFitter()
    y = fitter.voigt_basis(ENERGY, CENTERS, SIGMAS, GAMMA) @ AMPS

    res = fitter.fit_single(y, ENERGY, CENTERS + 0.2, SIGMAS + 0.1, GAMMA)
    res_none = fitter.fit_single(y, ENERGY, CENTERS + 0.2, SIGMAS + 0.1, GAMMA,
                                 background_design=None)

    assert res["background"].shape == (0,)
    for key in ("centers", "sigmas", "amplitudes"):
        np.testing.assert_array_equal(res[key], res_none[key])


@pytest.mark.parametrize("bad", [
    np.ones(len(ENERGY)),                 # 1-D
    np.ones((len(ENERGY) - 1, 2)),        # wrong number of rows
    np.ones((2, len(ENERGY))),            # transposed
    np.ones((len(ENERGY), 2, 1)),         # 3-D
])
def test_background_design_wrong_shape_raises(bad):
    fitter = VarProFitter()
    y = _clean(fitter)
    with pytest.raises(ValueError, match="background_design"):
        fitter.fit_single(y, ENERGY, CENTERS, SIGMAS, GAMMA, background_design=bad)
    with pytest.raises(ValueError, match="background_design"):
        fitter.fit_batch(np.column_stack([y, y]), ENERGY, CENTERS, SIGMAS, GAMMA,
                         background_design=bad)


def test_background_design_non_finite_raises():
    fitter = VarProFitter()
    bad = BG.copy()
    bad[5, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fitter.fit_single(_clean(fitter), ENERGY, CENTERS, SIGMAS, GAMMA,
                          background_design=bad)


# ---------------------------------------------------------------- T2: fixed shape

def test_fixed_shape_equals_free_fit_noise_free():
    """At the true widths, the fixed-width fit agrees with the free fit."""
    fitter = VarProFitter(tol=1e-12, max_iter=500)
    y = _clean(fitter)
    start = CENTERS + np.array([0.25, -0.2])

    free = fitter.fit_single(y, ENERGY, start, SIGMAS + 0.1, GAMMA, background_design=BG)
    fixed = fitter.fit_single(y, ENERGY, start, SIGMAS, GAMMA,
                              background_design=BG, fixed_shape=True)

    assert free["success"] and fixed["success"]
    np.testing.assert_array_equal(fixed["sigmas"], SIGMAS)
    np.testing.assert_allclose(fixed["sigmas"], free["sigmas"], atol=1e-4)
    np.testing.assert_allclose(fixed["centers"], free["centers"], atol=1e-5)
    np.testing.assert_allclose(fixed["amplitudes"], free["amplitudes"], rtol=1e-4)
    np.testing.assert_allclose(fixed["background"], free["background"], atol=1e-4)
    # both recover the truth
    np.testing.assert_allclose(fixed["centers"], CENTERS, atol=1e-5)
    np.testing.assert_allclose(fixed["amplitudes"], AMPS, rtol=1e-5)


def test_fixed_shape_holds_widths_at_wrong_values():
    """Fixed widths are not refined, even when they are wrong."""
    fitter = VarProFitter()
    y = _clean(fitter)
    wrong = SIGMAS * 1.3
    res = fitter.fit_single(y, ENERGY, CENTERS, wrong, GAMMA,
                            background_design=BG, fixed_shape=True)
    np.testing.assert_array_equal(res["sigmas"], wrong)
    assert res["chi2"] > 1e-6


def test_fixed_shape_centers_respect_bounds():
    fitter = VarProFitter(bounds_center_delta=0.1)
    y = _clean(fitter)
    start = CENTERS + 0.5
    res = fitter.fit_single(y, ENERGY, start, SIGMAS, GAMMA,
                            background_design=BG, fixed_shape=True)
    assert np.all(np.abs(res["centers"] - start) <= 0.1 + 1e-12)


def test_fixed_shape_needs_one_sigma_per_component():
    fitter = VarProFitter()
    with pytest.raises(ValueError, match="one sigma per component"):
        fitter.fit_single(_clean(fitter), ENERGY, CENTERS, SIGMAS[:1], GAMMA,
                          fixed_shape=True)


# ---------------------------------------------------------------- T2: batch layout

def _stack(fitter: VarProFitter, k: int = 3, m: int = 2) -> np.ndarray:
    """(k, m, n_energy) stack of distinct clean spectra."""
    rng = np.random.default_rng(11)
    out = np.empty((k, m, len(ENERGY)))
    for i in range(k):
        for j in range(m):
            out[i, j] = _clean(fitter, amps=AMPS * rng.uniform(0.5, 1.5, 2),
                               bg_coef=BG_COEF * rng.uniform(0.5, 1.5, 3))
    return out


@pytest.mark.parametrize("fixed_shape", [False, True])
def test_leading_axis_batch_equals_python_loop(fixed_shape):
    fitter = VarProFitter()
    Y = _stack(fitter)
    start = CENTERS + np.array([0.1, -0.1])

    batch = fitter.fit_batch(Y, ENERGY, start, SIGMAS, GAMMA, background_design=BG,
                             fixed_shape=fixed_shape, energy_axis=-1)

    assert batch["centers"].shape == (3, 2, 2)
    assert batch["background"].shape == (3, 2, 3)
    assert batch["chi2"].shape == (3, 2)
    assert batch["success"].shape == (3, 2)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            ref = fitter.fit_single(Y[i, j], ENERGY, start, SIGMAS, GAMMA,
                                    background_design=BG, fixed_shape=fixed_shape)
            for key in ("centers", "sigmas", "amplitudes", "background"):
                np.testing.assert_allclose(batch[key][i, j], ref[key], rtol=0, atol=1e-12)
            np.testing.assert_allclose(batch["chi2"][i, j], ref["chi2"], rtol=0, atol=1e-12)
            assert batch["success"][i, j] == ref["success"]


def test_default_column_layout_unchanged_and_matches_leading_layout():
    """energy_axis=0 keeps (n_energy, n_spectra) -> (n_components, n_spectra)."""
    fitter = VarProFitter()
    Y_lead = _stack(fitter, k=4, m=1)[:, 0, :]          # (K, n_energy)
    start = CENTERS + np.array([0.1, -0.1])

    cols = fitter.fit_batch(Y_lead.T, ENERGY, start, SIGMAS, GAMMA, background_design=BG)
    lead = fitter.fit_batch(Y_lead, ENERGY, start, SIGMAS, GAMMA, background_design=BG,
                            energy_axis=-1)

    assert cols["centers"].shape == (2, 4)
    assert cols["background"].shape == (3, 4)
    assert cols["chi2"].shape == (4,)
    for key in ("centers", "sigmas", "amplitudes", "background"):
        np.testing.assert_array_equal(cols[key], lead[key].T)
    np.testing.assert_array_equal(cols["chi2"], lead["chi2"])


def test_batch_energy_length_mismatch_raises():
    fitter = VarProFitter()
    Y = _stack(fitter, k=2, m=1)[:, 0, :]
    with pytest.raises(ValueError, match="energy_axis"):
        fitter.fit_batch(Y, ENERGY, CENTERS, SIGMAS, GAMMA)   # energy is axis -1 here
