"""
Voigt Profile Analytical Jacobian
=================================

Analytical derivatives of Voigt profile with respect to:
- center (c): peak position
- sigma (σ): Gaussian width
- gamma (γ): Lorentzian width

Based on Faddeeva function w(z) and its derivative:
    dw/dz = -2z·w(z) + 2i/√π

Reference:
    Abrarov & Quine (2011) "Efficient algorithmic implementation of the Voigt/complex error function"
"""


import numpy as np
from scipy import special as sps

SQRT2 = np.sqrt(2.0)
SQRT2PI = np.sqrt(2.0 * np.pi)
SQRTPI = np.sqrt(np.pi)


def voigt_profile(
    energy: np.ndarray,
    center: float,
    sigma: float,
    gamma: float,
) -> np.ndarray:
    """
    Compute Voigt profile using Faddeeva function.

    V(x; c, σ, γ) = Re[w(z)] / (σ√(2π))
    where z = (x - c + iγ) / (σ√2)

    Args:
        energy: Energy axis (n_energy,)
        center: Peak center position
        sigma: Gaussian width (standard deviation)
        gamma: Lorentzian half-width

    Returns:
        V: Voigt profile values (n_energy,)
    """
    z = ((energy - center) + 1j * gamma) / (sigma * SQRT2)
    w = sps.wofz(z)
    return np.real(w) / (sigma * SQRT2PI)


def voigt_with_jacobian(
    energy: np.ndarray,
    center: float,
    sigma: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Voigt profile and its analytical Jacobian.

    Uses the Faddeeva function derivative:
        dw/dz = -2z·w(z) + 2i/√π

    Args:
        energy: Energy axis (n_energy,)
        center: Peak center position
        sigma: Gaussian width
        gamma: Lorentzian half-width

    Returns:
        V: Voigt profile (n_energy,)
        dV_dc: ∂V/∂center (n_energy,)
        dV_dsigma: ∂V/∂sigma (n_energy,)
        dV_dgamma: ∂V/∂gamma (n_energy,)
    """
    # Complex argument
    z = ((energy - center) + 1j * gamma) / (sigma * SQRT2)

    # Faddeeva function and its derivative
    w = sps.wofz(z)
    dw_dz = -2 * z * w + 2j / SQRTPI

    # Voigt profile
    V = np.real(w) / (sigma * SQRT2PI)

    # Partial derivatives of z
    dz_dc = -1 / (sigma * SQRT2)
    dz_dsigma = -z / sigma
    dz_dgamma = 1j / (sigma * SQRT2)

    # Chain rule: ∂V/∂θ = Re[dw/dz · ∂z/∂θ] / (σ√(2π)) + correction terms

    # ∂V/∂c
    dV_dc = np.real(dw_dz * dz_dc) / (sigma * SQRT2PI)

    # ∂V/∂σ (includes -V/σ term from the 1/σ prefactor)
    dV_dsigma = np.real(dw_dz * dz_dsigma) / (sigma * SQRT2PI) - V / sigma

    # ∂V/∂γ
    dV_dgamma = np.real(dw_dz * dz_dgamma) / (sigma * SQRT2PI)

    return V, dV_dc, dV_dsigma, dV_dgamma


def voigt_with_hessian(
    energy: np.ndarray,
    center: float,
    sigma: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Voigt profile, Jacobian, and d²V/dc² Hessian.

    Uses the second derivative of the Faddeeva function:
        d²w/dz² = (4z² - 2)·w(z) - 4iz/√π

    Args:
        energy: Energy axis (n_energy,)
        center: Peak center position
        sigma: Gaussian width
        gamma: Lorentzian half-width

    Returns:
        V: Voigt profile (n_energy,)
        dV_dc: ∂V/∂center (n_energy,)
        dV_dsigma: ∂V/∂sigma (n_energy,)
        dV_dgamma: ∂V/∂gamma (n_energy,)
        d2V_dc2: ∂²V/∂center² (n_energy,)
    """
    z = ((energy - center) + 1j * gamma) / (sigma * SQRT2)
    w = sps.wofz(z)
    dw_dz = -2 * z * w + 2j / SQRTPI
    d2w_dz2 = (4 * z**2 - 2) * w - 4j * z / SQRTPI

    norm = sigma * SQRT2PI
    V = np.real(w) / norm

    dz_dc = -1 / (sigma * SQRT2)
    dz_dsigma = -z / sigma
    dz_dgamma = 1j / (sigma * SQRT2)

    dV_dc = np.real(dw_dz * dz_dc) / norm
    dV_dsigma = np.real(dw_dz * dz_dsigma) / norm - V / sigma
    dV_dgamma = np.real(dw_dz * dz_dgamma) / norm

    # d²V/dc²: d²z/dc² = 0, so only the (dz/dc)² term
    d2V_dc2 = np.real(d2w_dz2 * dz_dc**2) / norm

    return V, dV_dc, dV_dsigma, dV_dgamma, d2V_dc2


def voigt_with_sigma_hessian(
    energy: np.ndarray,
    center: float,
    sigma: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Voigt profile, Jacobian, and sigma-related Hessians.

    Extends voigt_with_hessian() with:
        - d²V/dσ²: sigma self-curvature (for 8-step solver correction)
        - d²V/dcdσ: cross-Hessian (for cross-curvature correction)

    Derivation:
        d²V/dσ² = {Re[d²w/dz²·z²] + 4·Re[dw/dz·z] + 2·Re[w]} / (σ²·norm)
        d²V/dcdσ = Re[d²w/dz²·z + 2·dw/dz] / (σ²·√2·norm)

    where norm = σ·√(2π), z = (x-c+iγ)/(σ√2)

    Returns:
        V: Voigt profile (n_energy,)
        dV_dc: ∂V/∂c (n_energy,)
        dV_dsigma: ∂V/∂σ (n_energy,)
        dV_dgamma: ∂V/∂γ (n_energy,)
        d2V_dc2: ∂²V/∂c² (n_energy,)
        d2V_dsigma2: ∂²V/∂σ² (n_energy,)
        d2V_dcdsigma: ∂²V/∂c∂σ (n_energy,)
    """
    z = ((energy - center) + 1j * gamma) / (sigma * SQRT2)
    w = sps.wofz(z)
    dw_dz = -2 * z * w + 2j / SQRTPI
    d2w_dz2 = (4 * z**2 - 2) * w - 4j * z / SQRTPI

    norm = sigma * SQRT2PI
    V = np.real(w) / norm

    dz_dc = -1 / (sigma * SQRT2)
    dz_dsigma = -z / sigma
    dz_dgamma = 1j / (sigma * SQRT2)

    dV_dc = np.real(dw_dz * dz_dc) / norm
    dV_dsigma = np.real(dw_dz * dz_dsigma) / norm - V / sigma
    dV_dgamma = np.real(dw_dz * dz_dgamma) / norm

    # d²V/dc²
    d2V_dc2 = np.real(d2w_dz2 * dz_dc**2) / norm

    # d²V/dσ²: full second derivative including 1/σ prefactor terms
    # Uses: (dz/dσ)² = z²/σ², d²z/dσ² = 2z/σ², d(1/norm)/dσ = -1/(σ·norm),
    #        d²(1/norm)/dσ² = 2/(σ²·norm)
    d2V_dsigma2 = (
        np.real(d2w_dz2 * z**2)
        + 4 * np.real(dw_dz * z)
        + 2 * np.real(w)
    ) / (sigma**2 * norm)

    # d²V/dcdσ: cross-Hessian
    # Uses: dz/dc·dz/dσ = z/(σ²·√2), d²z/dcdσ = 1/(σ²·√2)
    d2V_dcdsigma = np.real(
        d2w_dz2 * z + 2 * dw_dz
    ) / (sigma**2 * SQRT2 * norm)

    return V, dV_dc, dV_dsigma, dV_dgamma, d2V_dc2, d2V_dsigma2, d2V_dcdsigma


def voigt_sigma_hessian_batch(
    energy: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gammas: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch version of voigt_with_sigma_hessian for multiple components.

    Returns:
        Phi: Basis matrix (n_energy, n_comp)
        J_c: ∂Phi/∂centers (n_energy, n_comp)
        J_sigma: ∂Phi/∂sigmas (n_energy, n_comp)
        J_gamma: ∂Phi/∂gammas (n_energy, n_comp)
        H_cc: ∂²Phi/∂c² (n_energy, n_comp)
        H_ss: ∂²Phi/∂σ² (n_energy, n_comp)
        H_cs: ∂²Phi/∂c∂σ (n_energy, n_comp)
    """
    n_energy = len(energy)
    n_comp = len(centers)

    if np.isscalar(gammas):
        gammas = np.full(n_comp, gammas)

    Phi = np.zeros((n_energy, n_comp))
    J_c = np.zeros((n_energy, n_comp))
    J_sigma = np.zeros((n_energy, n_comp))
    J_gamma = np.zeros((n_energy, n_comp))
    H_cc = np.zeros((n_energy, n_comp))
    H_ss = np.zeros((n_energy, n_comp))
    H_cs = np.zeros((n_energy, n_comp))

    for k in range(n_comp):
        (V, dV_dc, dV_dsigma, dV_dgamma,
         d2V_dc2, d2V_dsigma2, d2V_dcdsigma) = voigt_with_sigma_hessian(
            energy, centers[k], sigmas[k], gammas[k]
        )
        Phi[:, k] = V
        J_c[:, k] = dV_dc
        J_sigma[:, k] = dV_dsigma
        J_gamma[:, k] = dV_dgamma
        H_cc[:, k] = d2V_dc2
        H_ss[:, k] = d2V_dsigma2
        H_cs[:, k] = d2V_dcdsigma

    return Phi, J_c, J_sigma, J_gamma, H_cc, H_ss, H_cs


def voigt_hessian_batch(
    energy: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gammas: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Voigt basis matrix, Jacobian, and d²V/dc² Hessian for multiple components.

    Args:
        energy: Energy axis (n_energy,)
        centers: Peak centers (n_comp,)
        sigmas: Gaussian widths (n_comp,)
        gammas: Lorentzian widths, scalar or (n_comp,)

    Returns:
        Phi: Basis matrix (n_energy, n_comp)
        J_c: ∂Phi/∂centers (n_energy, n_comp)
        J_sigma: ∂Phi/∂sigmas (n_energy, n_comp)
        J_gamma: ∂Phi/∂gammas (n_energy, n_comp)
        H_cc: ∂²Phi/∂centers² (n_energy, n_comp)
    """
    n_energy = len(energy)
    n_comp = len(centers)

    if np.isscalar(gammas):
        gammas = np.full(n_comp, gammas)

    Phi = np.zeros((n_energy, n_comp))
    J_c = np.zeros((n_energy, n_comp))
    J_sigma = np.zeros((n_energy, n_comp))
    J_gamma = np.zeros((n_energy, n_comp))
    H_cc = np.zeros((n_energy, n_comp))

    for k in range(n_comp):
        V, dV_dc, dV_dsigma, dV_dgamma, d2V_dc2 = voigt_with_hessian(
            energy, centers[k], sigmas[k], gammas[k]
        )
        Phi[:, k] = V
        J_c[:, k] = dV_dc
        J_sigma[:, k] = dV_dsigma
        J_gamma[:, k] = dV_dgamma
        H_cc[:, k] = d2V_dc2

    return Phi, J_c, J_sigma, J_gamma, H_cc


def voigt_jacobian_batch(
    energy: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gammas: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Voigt basis matrix and Jacobian for multiple components.

    Args:
        energy: Energy axis (n_energy,)
        centers: Peak centers (n_comp,)
        sigmas: Gaussian widths (n_comp,)
        gammas: Lorentzian widths, scalar or (n_comp,)

    Returns:
        Phi: Basis matrix (n_energy, n_comp)
        J_c: ∂Phi/∂centers (n_energy, n_comp)
        J_sigma: ∂Phi/∂sigmas (n_energy, n_comp)
        J_gamma: ∂Phi/∂gammas (n_energy, n_comp)
    """
    n_energy = len(energy)
    n_comp = len(centers)

    # Handle scalar gamma
    if np.isscalar(gammas):
        gammas = np.full(n_comp, gammas)

    Phi = np.zeros((n_energy, n_comp))
    J_c = np.zeros((n_energy, n_comp))
    J_sigma = np.zeros((n_energy, n_comp))
    J_gamma = np.zeros((n_energy, n_comp))

    for k in range(n_comp):
        V, dV_dc, dV_dsigma, dV_dgamma = voigt_with_jacobian(
            energy, centers[k], sigmas[k], gammas[k]
        )
        Phi[:, k] = V
        J_c[:, k] = dV_dc
        J_sigma[:, k] = dV_dsigma
        J_gamma[:, k] = dV_dgamma

    return Phi, J_c, J_sigma, J_gamma


def numerical_jacobian(
    energy: np.ndarray,
    center: float,
    sigma: float,
    gamma: float,
    eps: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute numerical Jacobian using central differences.

    For verification of analytical derivatives.

    Args:
        energy: Energy axis
        center, sigma, gamma: Voigt parameters
        eps: Finite difference step size

    Returns:
        dV_dc_num, dV_dsigma_num, dV_dgamma_num: Numerical derivatives
    """
    # ∂V/∂c
    Vp = voigt_profile(energy, center + eps, sigma, gamma)
    Vm = voigt_profile(energy, center - eps, sigma, gamma)
    dV_dc_num = (Vp - Vm) / (2 * eps)

    # ∂V/∂σ
    Vp = voigt_profile(energy, center, sigma + eps, gamma)
    Vm = voigt_profile(energy, center, sigma - eps, gamma)
    dV_dsigma_num = (Vp - Vm) / (2 * eps)

    # ∂V/∂γ
    Vp = voigt_profile(energy, center, sigma, gamma + eps)
    Vm = voigt_profile(energy, center, sigma, gamma - eps)
    dV_dgamma_num = (Vp - Vm) / (2 * eps)

    return dV_dc_num, dV_dsigma_num, dV_dgamma_num


def verify_jacobian(
    sigma_range: tuple[float, float] = (0.3, 2.0),
    gamma_range: tuple[float, float] = (0.1, 1.0),
    n_tests: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Verify analytical Jacobian against numerical derivatives.

    Tests multiple (σ, γ) combinations and reports max relative error.

    Args:
        sigma_range: Range of sigma values to test
        gamma_range: Range of gamma values to test
        n_tests: Number of random test cases
        verbose: Print results

    Returns:
        Dictionary with verification results
    """
    energy = np.linspace(-10, 10, 201)
    center = 0.0

    np.random.seed(42)
    sigmas = np.random.uniform(*sigma_range, n_tests)
    gammas = np.random.uniform(*gamma_range, n_tests)

    errors_c = []
    errors_sigma = []
    errors_gamma = []

    for sigma, gamma in zip(sigmas, gammas):
        # Analytical
        V, dV_dc, dV_dsigma, dV_dgamma = voigt_with_jacobian(energy, center, sigma, gamma)

        # Numerical
        dV_dc_num, dV_dsigma_num, dV_dgamma_num = numerical_jacobian(
            energy, center, sigma, gamma
        )

        # Relative errors (avoid division by zero)
        def rel_error(ana, num):
            mask = np.abs(ana) > 1e-10
            if mask.sum() == 0:
                return 0.0
            return np.max(np.abs(ana[mask] - num[mask]) / np.abs(ana[mask]))

        errors_c.append(rel_error(dV_dc, dV_dc_num))
        errors_sigma.append(rel_error(dV_dsigma, dV_dsigma_num))
        errors_gamma.append(rel_error(dV_dgamma, dV_dgamma_num))

    results = {
        "max_error_center": np.max(errors_c),
        "max_error_sigma": np.max(errors_sigma),
        "max_error_gamma": np.max(errors_gamma),
        "mean_error_center": np.mean(errors_c),
        "mean_error_sigma": np.mean(errors_sigma),
        "mean_error_gamma": np.mean(errors_gamma),
        "all_passed": all(
            e < 1e-5 for e in [np.max(errors_c), np.max(errors_sigma), np.max(errors_gamma)]
        ),
    }

    if verbose:
        print("=" * 50)
        print("Voigt Jacobian Verification")
        print("=" * 50)
        print(f"Number of tests: {n_tests}")
        print(f"Sigma range: {sigma_range}")
        print(f"Gamma range: {gamma_range}")
        print("-" * 50)
        print(f"∂V/∂c:  max rel error = {results['max_error_center']:.2e}, "
              f"mean = {results['mean_error_center']:.2e}")
        print(f"∂V/∂σ:  max rel error = {results['max_error_sigma']:.2e}, "
              f"mean = {results['mean_error_sigma']:.2e}")
        print(f"∂V/∂γ:  max rel error = {results['max_error_gamma']:.2e}, "
              f"mean = {results['mean_error_gamma']:.2e}")
        print("-" * 50)
        print(f"All passed (< 1e-5): {results['all_passed']}")
        print("=" * 50)

    return results


def verify_hessian(
    sigma_range: tuple[float, float] = (0.3, 2.0),
    gamma_range: tuple[float, float] = (0.1, 1.0),
    n_tests: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Verify analytical d²V/dc² against numerical second derivative.

    Uses central differences of the first derivative dV/dc.
    """
    energy = np.linspace(-10, 10, 201)
    center = 0.0
    eps = 1e-5

    np.random.seed(42)
    sigmas = np.random.uniform(*sigma_range, n_tests)
    gammas = np.random.uniform(*gamma_range, n_tests)

    errors = []

    for sigma, gamma in zip(sigmas, gammas):
        _, _, _, _, d2V_dc2 = voigt_with_hessian(energy, center, sigma, gamma)

        # Numerical: central difference of dV/dc
        _, dV_dc_p, _, _ = voigt_with_jacobian(energy, center + eps, sigma, gamma)
        _, dV_dc_m, _, _ = voigt_with_jacobian(energy, center - eps, sigma, gamma)
        d2V_dc2_num = (dV_dc_p - dV_dc_m) / (2 * eps)

        mask = np.abs(d2V_dc2) > 1e-10
        if mask.sum() > 0:
            errors.append(np.max(np.abs(d2V_dc2[mask] - d2V_dc2_num[mask]) / np.abs(d2V_dc2[mask])))
        else:
            errors.append(0.0)

    results = {
        "max_error": np.max(errors),
        "mean_error": np.mean(errors),
        "all_passed": np.max(errors) < 1e-4,
    }

    if verbose:
        print("=" * 50)
        print("Voigt Hessian d²V/dc² Verification")
        print("=" * 50)
        print(f"Number of tests: {n_tests}")
        print(f"∂²V/∂c²: max rel error = {results['max_error']:.2e}, "
              f"mean = {results['mean_error']:.2e}")
        print(f"All passed (< 1e-4): {results['all_passed']}")
        print("=" * 50)

    return results


def verify_sigma_hessian(
    sigma_range: tuple[float, float] = (0.3, 2.0),
    gamma_range: tuple[float, float] = (0.1, 1.0),
    n_tests: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Verify analytical d²V/dσ² and d²V/dcdσ against numerical second derivatives.

    Uses central differences of first derivatives dV/dσ and dV/dc.
    """
    energy = np.linspace(-10, 10, 201)
    center = 0.0
    eps = 1e-5

    np.random.seed(42)
    sigmas = np.random.uniform(*sigma_range, n_tests)
    gammas = np.random.uniform(*gamma_range, n_tests)

    errors_ss = []
    errors_cs = []

    for sigma, gamma in zip(sigmas, gammas):
        _, _, _, _, _, d2V_dsigma2, d2V_dcdsigma = voigt_with_sigma_hessian(
            energy, center, sigma, gamma
        )

        # d²V/dσ²: central difference of dV/dσ
        _, _, dV_dsigma_p, _, = voigt_with_jacobian(energy, center, sigma + eps, gamma)
        _, _, dV_dsigma_m, _, = voigt_with_jacobian(energy, center, sigma - eps, gamma)
        d2V_dsigma2_num = (dV_dsigma_p - dV_dsigma_m) / (2 * eps)

        mask = np.abs(d2V_dsigma2) > 1e-10
        if mask.sum() > 0:
            errors_ss.append(np.max(np.abs(
                d2V_dsigma2[mask] - d2V_dsigma2_num[mask]
            ) / np.abs(d2V_dsigma2[mask])))
        else:
            errors_ss.append(0.0)

        # d²V/dcdσ: central difference of dV/dc w.r.t. σ
        _, dV_dc_p, _, _ = voigt_with_jacobian(energy, center, sigma + eps, gamma)
        _, dV_dc_m, _, _ = voigt_with_jacobian(energy, center, sigma - eps, gamma)
        d2V_dcdsigma_num = (dV_dc_p - dV_dc_m) / (2 * eps)

        mask = np.abs(d2V_dcdsigma) > 1e-10
        if mask.sum() > 0:
            errors_cs.append(np.max(np.abs(
                d2V_dcdsigma[mask] - d2V_dcdsigma_num[mask]
            ) / np.abs(d2V_dcdsigma[mask])))
        else:
            errors_cs.append(0.0)

    results = {
        "max_error_d2V_dsigma2": np.max(errors_ss),
        "mean_error_d2V_dsigma2": np.mean(errors_ss),
        "max_error_d2V_dcdsigma": np.max(errors_cs),
        "mean_error_d2V_dcdsigma": np.mean(errors_cs),
        "all_passed": all(
            e < 1e-4 for e in [np.max(errors_ss), np.max(errors_cs)]
        ),
    }

    if verbose:
        print("=" * 50)
        print("Voigt Sigma Hessian Verification")
        print("=" * 50)
        print(f"Number of tests: {n_tests}")
        print(f"∂²V/∂σ²:   max rel error = {results['max_error_d2V_dsigma2']:.2e}, "
              f"mean = {results['mean_error_d2V_dsigma2']:.2e}")
        print(f"∂²V/∂c∂σ:  max rel error = {results['max_error_d2V_dcdsigma']:.2e}, "
              f"mean = {results['mean_error_d2V_dcdsigma']:.2e}")
        print(f"All passed (< 1e-4): {results['all_passed']}")
        print("=" * 50)

    return results


if __name__ == "__main__":
    # Run verification
    results = verify_jacobian(verbose=True)

    # Visual check
    import matplotlib.pyplot as plt

    energy = np.linspace(-5, 5, 201)
    center, sigma, gamma = 0.0, 0.8, 0.3

    V, dV_dc, dV_dsigma, dV_dgamma = voigt_with_jacobian(energy, center, sigma, gamma)
    dV_dc_num, dV_dsigma_num, dV_dgamma_num = numerical_jacobian(energy, center, sigma, gamma)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Profile
    axes[0, 0].plot(energy, V, 'b-', lw=2, label='Voigt')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('V(x)')
    axes[0, 0].set_title(f'Voigt Profile (σ={sigma}, γ={gamma})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # ∂V/∂c
    axes[0, 1].plot(energy, dV_dc, 'b-', lw=2, label='Analytical')
    axes[0, 1].plot(energy, dV_dc_num, 'r--', lw=1.5, label='Numerical')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('∂V/∂c')
    axes[0, 1].set_title('Derivative w.r.t. center')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ∂V/∂σ
    axes[1, 0].plot(energy, dV_dsigma, 'b-', lw=2, label='Analytical')
    axes[1, 0].plot(energy, dV_dsigma_num, 'r--', lw=1.5, label='Numerical')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('∂V/∂σ')
    axes[1, 0].set_title('Derivative w.r.t. sigma')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ∂V/∂γ
    axes[1, 1].plot(energy, dV_dgamma, 'b-', lw=2, label='Analytical')
    axes[1, 1].plot(energy, dV_dgamma_num, 'r--', lw=1.5, label='Numerical')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('∂V/∂γ')
    axes[1, 1].set_title('Derivative w.r.t. gamma')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('voigt_jacobian_verification.png', dpi=150)
    plt.show()

    print("\nFigure saved: voigt_jacobian_verification.png")
