"""SciPy oracle tests for VarProFitter.

Guards the two contracts the estimator class rests on: the inner amplitude
solve matches its closed form (Φ'Φ + λI)⁻¹ Φ' y, and ``fit_single`` recovers
the parameters of a clean synthetic single peak.
"""
import numpy as np
from scipy import linalg

from toyomacro.voigtfit import VarProFitter


def test_solve_amplitudes_matches_closed_form():
    rng = np.random.default_rng(7)
    energy = np.linspace(-5.0, 5.0, 121)
    fitter = VarProFitter()
    Phi = fitter.voigt_basis(energy, np.array([-0.8, 0.6]), np.array([0.5, 0.7]), 0.3)
    y = rng.standard_normal(len(energy))

    a = fitter.solve_amplitudes(Phi, y)
    n = Phi.shape[1]
    oracle = linalg.solve(Phi.T @ Phi + fitter.reg_lambda * np.eye(n), Phi.T @ y,
                          assume_a='pos')
    np.testing.assert_allclose(a, oracle, rtol=1e-10, atol=1e-12)

    # unregularized limit agrees with lstsq on a well-conditioned basis
    lstsq_sol = linalg.lstsq(Phi, y)[0]
    np.testing.assert_allclose(a, lstsq_sol, rtol=1e-5)


def test_fit_single_recovers_synthetic_peak():
    energy = np.linspace(95.0, 105.0, 201)
    true_center, true_sigma, true_amp, gamma = 99.5, 0.6, 3.0, 0.25
    fitter = VarProFitter()
    Phi = fitter.voigt_basis(energy, np.array([true_center]), np.array([true_sigma]), gamma)
    y = true_amp * Phi[:, 0]

    result = fitter.fit_single(y, energy, np.array([99.0]), np.array([0.8]), gamma)

    assert result['success']
    assert abs(result['centers'][0] - true_center) < 1e-4
    assert abs(result['sigmas'][0] - true_sigma) < 1e-3
    assert abs(result['amplitudes'][0] - true_amp) < 1e-3
    assert result['chi2'] < 1e-10
