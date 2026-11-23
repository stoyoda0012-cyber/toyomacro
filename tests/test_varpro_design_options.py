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


def test_no_background_reports_empty_coefficients():
    """Without a design the result carries an empty ``background`` key."""
    fitter = VarProFitter()
    y = fitter.voigt_basis(ENERGY, CENTERS, SIGMAS, GAMMA) @ AMPS

    res = fitter.fit_single(y, ENERGY, CENTERS + 0.2, SIGMAS + 0.1, GAMMA)
    assert res["background"].shape == (0,)
    batch = fitter.fit_batch(np.column_stack([y, y, y]), ENERGY, CENTERS + 0.2,
                             SIGMAS + 0.1, GAMMA)
    assert batch["background"].shape == (0, 3)


# Default-argument fit_batch output on a fixed noisy fixture, computed with the
# VarProFitter that preceded the background/fixed-shape/energy-axis options.
# It pins the default path: any change to it fails here. Tolerances allow
# BLAS-level roundoff in the L-BFGS-B path, not a change of estimator.
_DEFAULT_PIN = {
    "centers": np.array([[98.79785563362022, 98.79802784074295, 98.80080230692137],
                         [101.20443060960395, 101.18721830035409, 101.19915004947507]]),
    "sigmas": np.array([[0.5504621055895663, 0.5558431829687132, 0.5467071099388755],
                        [0.697649836554396, 0.7188400943150302, 0.7021185243354192]]),
    "amplitudes": np.array([[3.0094324250038316, 2.083361660348194, 3.902230584650578],
                            [1.6957942606042093, 1.2010597202134787, 2.2158633008638624]]),
    "chi2": np.array([0.00044873214901548, 0.00047120213282459, 0.00044978672556238]),
}


def _default_fixture(fitter: VarProFitter) -> np.ndarray:
    phi = fitter.voigt_basis(ENERGY, CENTERS, SIGMAS, GAMMA)
    rng = np.random.default_rng(20261105)
    return (np.column_stack([phi @ (AMPS * s) for s in (1.0, 0.7, 1.3)])
            + 0.02 * rng.standard_normal((len(ENERGY), 3)))


def test_default_path_matches_pinned_pre_change_output():
    fitter = VarProFitter()
    Y = _default_fixture(fitter)
    res = fitter.fit_batch(Y, ENERGY, CENTERS + np.array([0.1, -0.1]), SIGMAS + 0.05,
                           GAMMA)

    assert res["success"].all()
    np.testing.assert_allclose(res["centers"], _DEFAULT_PIN["centers"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(res["sigmas"], _DEFAULT_PIN["sigmas"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(res["amplitudes"], _DEFAULT_PIN["amplitudes"], rtol=1e-6)
    np.testing.assert_allclose(res["chi2"], _DEFAULT_PIN["chi2"], rtol=1e-6)
    # independent oracle at the returned shape: amplitudes are the (tiny-ridge)
    # least-squares solution and chi2 is its mean squared residual
    for j in range(Y.shape[1]):
        phi = fitter.voigt_basis(ENERGY, res["centers"][:, j], res["sigmas"][:, j], GAMMA)
        amps = np.linalg.lstsq(phi, Y[:, j], rcond=None)[0]
        np.testing.assert_allclose(res["amplitudes"][:, j], amps, rtol=1e-7)
        np.testing.assert_allclose(res["chi2"][j],
                                   np.mean((Y[:, j] - phi @ amps) ** 2), rtol=1e-7)


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


def test_background_design_linearly_dependent_columns_raise():
    """Duplicate columns are refused rather than split by the regularizer."""
    fitter = VarProFitter()
    dup = np.column_stack([BG, BG[:, 0]])
    for reg in (fitter, VarProFitter(reg_lambda=0.0)):
        with pytest.raises(ValueError, match="linearly dependent"):
            reg.fit_single(_clean(fitter), ENERGY, CENTERS, SIGMAS, GAMMA,
                           background_design=dup)
    combo = np.column_stack([BG, BG[:, 0] - 2.0 * BG[:, 1]])
    with pytest.raises(ValueError, match="linearly dependent"):
        fitter.fit_batch(np.column_stack([_clean(fitter)] * 2), ENERGY, CENTERS,
                         SIGMAS, GAMMA, background_design=combo)


def test_default_reg_bias_is_negligible_for_unit_scale_columns():
    """With O(1) columns the default absolute ridge biases nothing visibly."""
    fitter = VarProFitter(bounds_center_delta=0.0)          # reg_lambda=1e-8
    res = fitter.fit_single(_clean(fitter), ENERGY, CENTERS, SIGMAS, GAMMA,
                            background_design=BG, fixed_shape=True)
    np.testing.assert_allclose(res["amplitudes"], AMPS, rtol=1e-7)
    np.testing.assert_allclose(res["background"], BG_COEF, rtol=0, atol=1e-7)


def test_default_reg_biases_small_norm_columns():
    """The ridge is absolute: a tiny-norm column is shrunk and biases the peaks.

    This pins the documented limitation (scale the columns), not a feature.
    """
    fitter = VarProFitter(bounds_center_delta=0.0)
    scale = 1e-5
    y = _clean(fitter)
    small = BG.copy()
    small[:, 2] *= scale
    res = fitter.fit_single(y, ENERGY, CENTERS, SIGMAS, GAMMA,
                            background_design=small, fixed_shape=True)
    # true coefficient on the scaled column is BG_COEF[2] / scale
    assert abs(res["background"][2] * scale - BG_COEF[2]) > 1e-2
    assert np.max(np.abs(res["amplitudes"] - AMPS)) > 1e-3
    exact = VarProFitter(reg_lambda=0.0, bounds_center_delta=0.0).fit_single(
        y, ENERGY, CENTERS, SIGMAS, GAMMA, background_design=small, fixed_shape=True)
    np.testing.assert_allclose(exact["background"][2] * scale, BG_COEF[2], rtol=1e-6)


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


@pytest.mark.parametrize("energy_axis", [2, -3, 5])
def test_batch_energy_axis_out_of_range_raises(energy_axis):
    """An out-of-range axis is refused, not wrapped round."""
    fitter = VarProFitter()
    Y = np.column_stack([_clean(fitter)] * 3)                 # (n_energy, 3)
    with pytest.raises(ValueError, match="out of range"):
        fitter.fit_batch(Y, ENERGY, CENTERS, SIGMAS, GAMMA, energy_axis=energy_axis)


@pytest.mark.parametrize("energy_axis", [0, 1, 2, -1, -2])
def test_three_d_batch_every_energy_axis_matches_single_fit(energy_axis):
    """Parameter axis sits where the energy axis was, for first, middle and last."""
    fitter = VarProFitter()
    base = _stack(fitter, k=2, m=3)                           # (2, 3, n_energy)
    Y = np.moveaxis(base, -1, energy_axis)
    start = CENTERS + np.array([0.1, -0.1])

    res = fitter.fit_batch(Y, ENERGY, start, SIGMAS, GAMMA, background_design=BG,
                           fixed_shape=True, energy_axis=energy_axis)

    axis = energy_axis % 3
    expect = list((2, 3))
    expect.insert(axis, 2)
    assert res["centers"].shape == tuple(expect)
    expect[axis] = 3
    assert res["background"].shape == tuple(expect)
    assert res["chi2"].shape == (2, 3)
    for i in range(2):
        for j in range(3):
            ref = fitter.fit_single(base[i, j], ENERGY, start, SIGMAS, GAMMA,
                                    background_design=BG, fixed_shape=True)
            for key in ("centers", "amplitudes", "background"):
                got = np.moveaxis(res[key], axis, -1)[i, j]
                np.testing.assert_allclose(got, ref[key], rtol=0, atol=1e-12)
            assert res["chi2"][i, j] == ref["chi2"]
