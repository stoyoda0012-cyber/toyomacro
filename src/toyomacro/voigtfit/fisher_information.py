"""
Fisher Information Matrix for Voigt Spectral Parameters
========================================================

Computes the Fisher Information Matrix (FIM) for Voigt profile parameters
under Poisson noise, enabling information-geometric analysis of parameter
estimation limits.

The FIM quantifies how much information a spectrum carries about each
parameter. For Poisson-distributed photon counts with expected value
f(E; theta):

    g_ij(theta) = sum_E  (df/dtheta_i)(df/dtheta_j) / f(E; theta)

This is the Riemannian metric tensor on parameter space, where geodesic
distance approximates KL divergence between nearby spectra.

Key properties:
    - Diagonal g_ii: Fisher information per parameter (larger = easier to estimate)
    - Off-diagonal g_ij: parameter coupling (non-zero = correlated estimation)
    - Eigenvalues: principal information axes
    - Condition number: anisotropy of estimation difficulty

Supports:
    - 3D: theta = (amplitude, delta_E, delta_sigma), gamma fixed
    - 4D: theta = (amplitude, delta_E, delta_sigma, delta_gamma)

Uses analytical Jacobians from voigt_jacobian.py.

Reference:
    Rao (1945), Amari & Nagaoka (2000) "Methods of Information Geometry"
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .voigt_jacobian import voigt_with_jacobian

# Parameter ordering conventions
PARAM_NAMES_3D = ('amplitude', 'delta_E', 'delta_sigma')
PARAM_NAMES_4D = ('amplitude', 'delta_E', 'delta_sigma', 'delta_gamma')

# FWHM <-> sigma/gamma conversions
FWHM_TO_SIGMA = 1.0 / (2 * np.sqrt(2 * np.log(2)))  # FWHM_G -> sigma
# gamma = FWHM_L / 2


@dataclass
class FisherResult:
    """Result of Fisher Information Matrix computation.

    Attributes:
        g: Fisher Information Matrix (n_params, n_params)
        correlation: Correlation matrix r_ij = g_ij / sqrt(g_ii * g_jj)
        eigenvalues: Eigenvalues in ascending order
        eigenvectors: Corresponding eigenvectors (columns)
        condition_number: lambda_max / lambda_min
        param_names: Parameter labels
        params: Dictionary of parameter values used
    """
    g: np.ndarray
    correlation: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    param_names: tuple[str, ...]
    params: dict


def compute_fisher_matrix(
    amplitude: float,
    center: float,
    sigma: float,
    gamma: float,
    energy: np.ndarray,
    mode: Literal['3d', '4d'] = '3d',
    floor_eps: float = 1e-30,
) -> FisherResult:
    """
    Compute Fisher Information Matrix for a single Voigt peak.

    For Poisson noise with expected counts f(E) = A * V(E; c, sigma, gamma):

        g_ij = sum_E (df/dtheta_i)(df/dtheta_j) / f(E)

    The Jacobians df/dtheta are:
        df/d(amp)    = V(E)                     (profile itself)
        df/d(delta_E) = A * dV/dc               (center shift)
        df/d(delta_sigma) = A * dV/dsigma       (width change)
        df/d(delta_gamma) = A * dV/dgamma       (4D only)

    Args:
        amplitude: Peak amplitude (expected count at peak maximum ~ A * V_max)
        center: Peak center position (eV)
        sigma: Gaussian width parameter (standard deviation, eV)
        gamma: Lorentzian half-width (eV)
        energy: Energy axis (eV), shape (n_energy,)
        mode: '3d' for (amp, dE, dsigma), '4d' adds dgamma
        floor_eps: Floor value to avoid division by zero in 1/f

    Returns:
        FisherResult with g_ij, correlation, eigenanalysis
    """
    # Compute Voigt profile and all Jacobians
    V, dV_dc, dV_dsigma, dV_dgamma = voigt_with_jacobian(energy, center, sigma, gamma)

    # Expected Poisson counts: f(E) = amplitude * V(E)
    f = amplitude * V
    f_safe = np.maximum(f, floor_eps)

    # Build Jacobian matrix: df/dtheta_i for each parameter
    # Row i = df/dtheta_i evaluated at each energy point
    if mode == '3d':
        n_params = 3
        param_names = PARAM_NAMES_3D
        # df/d(amp) = V, df/d(dE) = amp * dV/dc, df/d(dsigma) = amp * dV/dsigma
        J = np.stack([
            V,                        # d/d(amplitude)
            amplitude * dV_dc,        # d/d(delta_E)
            amplitude * dV_dsigma,    # d/d(delta_sigma)
        ])  # shape: (3, n_energy)
    elif mode == '4d':
        n_params = 4
        param_names = PARAM_NAMES_4D
        J = np.stack([
            V,                        # d/d(amplitude)
            amplitude * dV_dc,        # d/d(delta_E)
            amplitude * dV_dsigma,    # d/d(delta_sigma)
            amplitude * dV_dgamma,    # d/d(delta_gamma)
        ])  # shape: (4, n_energy)
    else:
        raise ValueError(f"mode must be '3d' or '4d', got '{mode}'")

    # Fisher Information Matrix: g_ij = sum_E J_i(E) * J_j(E) / f(E)
    # Efficient: weight each energy point by 1/f, then outer product
    w = 1.0 / f_safe  # (n_energy,)
    Jw = J * np.sqrt(w)[np.newaxis, :]  # (n_params, n_energy), weighted
    g = Jw @ Jw.T  # (n_params, n_params)

    # Correlation matrix
    diag = np.sqrt(np.diag(g))
    diag_safe = np.maximum(diag, 1e-30)
    correlation = g / np.outer(diag_safe, diag_safe)

    # Eigenanalysis (ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(g)

    # Condition number
    eig_positive = eigenvalues[eigenvalues > 0]
    if len(eig_positive) >= 2:
        condition_number = eig_positive[-1] / eig_positive[0]
    else:
        condition_number = np.inf

    return FisherResult(
        g=g,
        correlation=correlation,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        condition_number=condition_number,
        param_names=param_names,
        params={
            'amplitude': amplitude,
            'center': center,
            'sigma': sigma,
            'gamma': gamma,
            'n_energy': len(energy),
            'energy_range': (energy[0], energy[-1]),
        },
    )


def compute_fisher_matrix_batch(
    amplitudes: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gammas: np.ndarray,
    energy: np.ndarray,
    mode: Literal['3d', '4d'] = '3d',
) -> list[FisherResult]:
    """Compute Fisher matrix for multiple parameter sets.

    Useful for exploring parameter dependence of g_ij.
    """
    results = []
    for amp, c, s, g in zip(amplitudes, centers, sigmas, gammas):
        results.append(compute_fisher_matrix(amp, c, s, g, energy, mode=mode))
    return results


# ---------------------------------------------------------------------------
# Eta (mixing ratio) utilities
# ---------------------------------------------------------------------------

def eta_from_sigma_gamma(sigma: float, gamma: float) -> float:
    """Compute Voigt mixing ratio eta = f_L / f_V.

    Uses Thompson et al. (1987) approximation for Voigt FWHM:
        f_V = 0.5346 * f_L + sqrt(0.2166 * f_L^2 + f_G^2)

    where f_G = 2*sqrt(2*ln(2))*sigma, f_L = 2*gamma.
    """
    f_G = sigma / FWHM_TO_SIGMA  # = 2*sqrt(2*ln2) * sigma
    f_L = 2 * gamma
    f_V = 0.5346 * f_L + np.sqrt(0.2166 * f_L**2 + f_G**2)
    if f_V < 1e-30:
        return 0.0
    return f_L / f_V


def sigma_gamma_from_eta(eta: float, fwhm_total: float = 1.0) -> tuple[float, float]:
    """Compute (sigma, gamma) from mixing ratio eta and total Voigt FWHM.

    Given eta = f_L / f_V and target f_V:
        f_L = eta * f_V
        Solve Thompson formula for f_G:
            f_V = 0.5346 * f_L + sqrt(0.2166 * f_L^2 + f_G^2)
            => f_G = sqrt((f_V - 0.5346*f_L)^2 - 0.2166*f_L^2)

    Args:
        eta: Lorentzian fraction [0, 1]. 0 = pure Gaussian, 1 = pure Lorentzian.
        fwhm_total: Target total Voigt FWHM (eV)

    Returns:
        (sigma, gamma) in eV
    """
    f_V = fwhm_total
    f_L = eta * f_V

    # From Thompson: f_V = 0.5346*f_L + sqrt(0.2166*f_L^2 + f_G^2)
    remainder = f_V - 0.5346 * f_L
    f_G_sq = remainder**2 - 0.2166 * f_L**2

    if f_G_sq < 0:
        # Near pure Lorentzian: numerical precision issue
        f_G = 0.0
    else:
        f_G = np.sqrt(f_G_sq)

    sigma = f_G * FWHM_TO_SIGMA
    gamma = f_L / 2.0

    return sigma, gamma


def eta_sweep(
    etas: np.ndarray | None = None,
    amplitude: float = 1000.0,
    center: float = 284.4,
    fwhm_total: float = 1.0,
    energy: np.ndarray | None = None,
    mode: Literal['3d', '4d'] = '4d',
) -> dict:
    """Sweep eta from Gaussian-dominated to Lorentzian-dominated.

    Computes Fisher matrix at each eta value with fixed total Voigt FWHM.
    This shows how the information geometry changes across elements:
        - C 1s (eta ~ 0.15): Gaussian dominated
        - O 1s (eta ~ 0.3): Mixed
        - Au 4f (eta ~ 0.5): Balanced
        - Cu 2p (eta ~ 0.8): Lorentzian dominated

    Args:
        etas: Array of eta values to sweep. Default: 20 points from 0.05 to 0.90
        amplitude: Peak amplitude (Poisson counts at peak)
        center: Peak center (eV)
        fwhm_total: Total Voigt FWHM (eV), held constant across sweep
        energy: Energy axis. Default: center +/- 5*fwhm_total, 201 points
        mode: '3d' or '4d'

    Returns:
        Dictionary with:
            etas: (n_eta,) eta values
            sigmas, gammas: (n_eta,) corresponding width parameters
            fisher_results: list of FisherResult
            diagonal: (n_eta, n_params) diagonal elements g_ii
            correlations: (n_eta, n_params, n_params) correlation matrices
            eigenvalues: (n_eta, n_params) eigenvalues (ascending)
            condition_numbers: (n_eta,) condition numbers
            info_retention_3d: (n_eta,) fraction of info in top-3 eigenvalues (4D only)
    """
    if etas is None:
        etas = np.linspace(0.05, 0.90, 20)

    if energy is None:
        half_range = 5 * fwhm_total
        energy = np.linspace(center - half_range, center + half_range, 201)

    n_eta = len(etas)
    n_params = 4 if mode == '4d' else 3

    sigmas = np.zeros(n_eta)
    gammas = np.zeros(n_eta)
    results = []
    diagonal = np.zeros((n_eta, n_params))
    correlations = np.zeros((n_eta, n_params, n_params))
    eigenvalues = np.zeros((n_eta, n_params))
    condition_numbers = np.zeros(n_eta)
    info_retention_3d = np.zeros(n_eta) if mode == '4d' else None

    for i, eta in enumerate(etas):
        s, g = sigma_gamma_from_eta(eta, fwhm_total)
        sigmas[i] = s
        gammas[i] = g

        result = compute_fisher_matrix(
            amplitude, center, s, g, energy, mode=mode
        )
        results.append(result)

        diagonal[i] = np.diag(result.g)
        correlations[i] = result.correlation
        eigenvalues[i] = result.eigenvalues
        condition_numbers[i] = result.condition_number

        if mode == '4d':
            total_info = result.eigenvalues.sum()
            if total_info > 0:
                info_retention_3d[i] = result.eigenvalues[-3:].sum() / total_info

    return {
        'etas': etas,
        'sigmas': sigmas,
        'gammas': gammas,
        'fisher_results': results,
        'diagonal': diagonal,
        'correlations': correlations,
        'eigenvalues': eigenvalues,
        'condition_numbers': condition_numbers,
        'info_retention_3d': info_retention_3d,
        'mode': mode,
        'amplitude': amplitude,
        'fwhm_total': fwhm_total,
        'param_names': PARAM_NAMES_4D[:n_params],
    }


# ---------------------------------------------------------------------------
# Amplitude dependence analysis
# ---------------------------------------------------------------------------

def amplitude_sweep(
    amplitudes: np.ndarray | None = None,
    sigma: float = 0.4247,
    gamma: float = 0.125,
    center: float = 284.4,
    energy: np.ndarray | None = None,
    mode: Literal['3d', '4d'] = '3d',
) -> dict:
    """Sweep amplitude to show Fisher scaling with signal level.

    Fisher information scales as g_ij ~ amplitude for Poisson noise.
    This verifies the scaling and shows how dark pixels lose information.

    Args:
        amplitudes: Array of amplitude values. Default: logspace from 10 to 10000
        sigma: Gaussian width (default: C 1s nominal)
        gamma: Lorentzian half-width (default: C 1s nominal)
        center: Peak center
        energy: Energy axis
        mode: '3d' or '4d'

    Returns:
        Dictionary with sweep results
    """
    if amplitudes is None:
        amplitudes = np.logspace(1, 4, 20)

    if energy is None:
        energy = np.linspace(center - 5, center + 5, 201)

    n_amp = len(amplitudes)
    n_params = 4 if mode == '4d' else 3

    results = []
    diagonal = np.zeros((n_amp, n_params))
    condition_numbers = np.zeros(n_amp)

    for i, amp in enumerate(amplitudes):
        result = compute_fisher_matrix(
            amp, center, sigma, gamma, energy, mode=mode
        )
        results.append(result)
        diagonal[i] = np.diag(result.g)
        condition_numbers[i] = result.condition_number

    return {
        'amplitudes': amplitudes,
        'fisher_results': results,
        'diagonal': diagonal,
        'condition_numbers': condition_numbers,
        'mode': mode,
        'param_names': PARAM_NAMES_4D[:n_params],
    }


# ---------------------------------------------------------------------------
# Sigma-Gamma principal axis analysis
# ---------------------------------------------------------------------------

@dataclass
class SigmaGammaAxes:
    """Principal axes of the (delta_sigma, delta_gamma) Fisher sub-block.

    Physical interpretation:
        v_width: eigenvector for "total width change" (sigma and gamma co-move)
                 → high eigenvalue → easy to estimate
        v_ratio: eigenvector for "G/L ratio change" (sigma up, gamma down)
                 → low eigenvalue → hard to estimate (the Voigt identifiability problem)

    Attributes:
        g_sub: (2, 2) Fisher sub-matrix for (delta_sigma, delta_gamma)
        eigenvalues: (2,) ascending order [lambda_ratio, lambda_width]
        eigenvectors: (2, 2) columns are eigenvectors in (sigma, gamma) space
        v_ratio: (2,) "G/L ratio" direction (hard mode)
        v_width: (2,) "total width" direction (easy mode)
        lambda_ratio: eigenvalue for ratio direction (small)
        lambda_width: eigenvalue for width direction (large)
        rotation_angle: angle (degrees) of principal axes from (sigma, gamma) axes
        info_ratio: lambda_width / lambda_ratio (anisotropy)
        bias_coefficient: g_sigma_gamma / g_sigma_sigma (omitted variable bias)
        correlation: r(sigma, gamma) from full Fisher matrix
    """
    g_sub: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    v_ratio: np.ndarray
    v_width: np.ndarray
    lambda_ratio: float
    lambda_width: float
    rotation_angle: float
    info_ratio: float
    bias_coefficient: float
    correlation: float


def analyze_sigma_gamma_axes(fisher_result: FisherResult) -> SigmaGammaAxes:
    """Extract and analyze the (delta_sigma, delta_gamma) sub-block.

    The 4x4 Fisher matrix has a 2x2 sub-block for the width parameters:
        g_sub = [[g_sigma_sigma, g_sigma_gamma],
                 [g_sigma_gamma, g_gamma_gamma]]

    Its eigenvectors define the "natural" axes for width estimation:
        - Large eigenvalue: "total width" direction (both widen or both narrow)
        - Small eigenvalue: "G/L ratio" direction (one widens, other narrows)

    The small eigenvalue direction is the Voigt identifiability problem:
    changing the Gaussian/Lorentzian ratio while keeping total width constant
    produces minimal spectral change → hardest to estimate.

    Args:
        fisher_result: Must be from a 4D computation (mode='4d')

    Returns:
        SigmaGammaAxes with full decomposition
    """
    if len(fisher_result.param_names) < 4:
        raise ValueError("Need 4D Fisher result (mode='4d') for sigma-gamma analysis")

    g = fisher_result.g

    # Extract 2x2 sub-block: indices 2 (delta_sigma) and 3 (delta_gamma)
    g_sub = g[2:4, 2:4].copy()

    # Eigendecomposition (ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(g_sub)

    # Identify which eigenvector is "width" (co-moving) vs "ratio" (counter-moving)
    # Convention: v_width has same-sign components (sigma↑ gamma↑)
    # v_ratio has opposite-sign components (sigma↑ gamma↓)
    v0 = eigenvectors[:, 0]  # smallest eigenvalue
    v1 = eigenvectors[:, 1]  # largest eigenvalue

    # Ensure consistent sign: v_width should have both positive components
    if v1[0] * v1[1] < 0:
        # v1 is counter-moving → swap
        v_ratio, v_width = v1, v0
        lambda_ratio, lambda_width = eigenvalues[1], eigenvalues[0]
    else:
        v_ratio, v_width = v0, v1
        lambda_ratio, lambda_width = eigenvalues[0], eigenvalues[1]

    # Normalize sign: v_width should have positive sigma component
    if v_width[0] < 0:
        v_width = -v_width
    # v_ratio: convention is sigma>0, gamma<0 (increasing G/L ratio)
    if v_ratio[0] < 0:
        v_ratio = -v_ratio

    # Rotation angle from (sigma, gamma) axes to principal axes
    # Angle of the "width" eigenvector from the sigma axis
    rotation_angle = np.degrees(np.arctan2(v_width[1], v_width[0]))

    # Anisotropy
    info_ratio = lambda_width / lambda_ratio if lambda_ratio > 0 else np.inf

    # Omitted variable bias: if gamma is fixed at wrong value,
    # bias(sigma_hat) = -(g_sigma_sigma)^-1 * g_sigma_gamma * delta_gamma
    # bias_coefficient = |g_sigma_gamma / g_sigma_sigma|
    bias_coefficient = abs(g_sub[0, 1]) / g_sub[0, 0] if g_sub[0, 0] > 0 else np.inf

    # Correlation
    correlation = g_sub[0, 1] / np.sqrt(g_sub[0, 0] * g_sub[1, 1])

    return SigmaGammaAxes(
        g_sub=g_sub,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        v_ratio=v_ratio,
        v_width=v_width,
        lambda_ratio=lambda_ratio,
        lambda_width=lambda_width,
        rotation_angle=rotation_angle,
        info_ratio=info_ratio,
        bias_coefficient=bias_coefficient,
        correlation=correlation,
    )


def sigma_gamma_sweep(
    etas: np.ndarray | None = None,
    amplitude: float = 1000.0,
    center: float = 284.4,
    fwhm_total: float = 1.0,
    energy: np.ndarray | None = None,
) -> dict:
    """Compute sigma-gamma principal axis decomposition across eta values.

    Args:
        etas: Mixing ratios to sweep
        amplitude, center, fwhm_total: Voigt parameters
        energy: Energy axis

    Returns:
        Dictionary with per-eta decomposition results
    """
    if etas is None:
        etas = np.linspace(0.05, 0.90, 30)

    if energy is None:
        half_range = 5 * fwhm_total
        energy = np.linspace(center - half_range, center + half_range, 201)

    n_eta = len(etas)

    axes_list = []
    lambda_ratios = np.zeros(n_eta)
    lambda_widths = np.zeros(n_eta)
    rotation_angles = np.zeros(n_eta)
    info_ratios = np.zeros(n_eta)
    bias_coefficients = np.zeros(n_eta)
    correlations_sg = np.zeros(n_eta)
    v_ratios = np.zeros((n_eta, 2))
    v_widths = np.zeros((n_eta, 2))

    for i, eta in enumerate(etas):
        s, g = sigma_gamma_from_eta(eta, fwhm_total)
        result = compute_fisher_matrix(amplitude, center, s, g, energy, mode='4d')
        axes = analyze_sigma_gamma_axes(result)

        axes_list.append(axes)
        lambda_ratios[i] = axes.lambda_ratio
        lambda_widths[i] = axes.lambda_width
        rotation_angles[i] = axes.rotation_angle
        info_ratios[i] = axes.info_ratio
        bias_coefficients[i] = axes.bias_coefficient
        correlations_sg[i] = axes.correlation
        v_ratios[i] = axes.v_ratio
        v_widths[i] = axes.v_width

    return {
        'etas': etas,
        'axes_list': axes_list,
        'lambda_ratio': lambda_ratios,
        'lambda_width': lambda_widths,
        'rotation_angles': rotation_angles,
        'info_ratios': info_ratios,
        'bias_coefficients': bias_coefficients,
        'correlations': correlations_sg,
        'v_ratios': v_ratios,
        'v_widths': v_widths,
        'amplitude': amplitude,
        'fwhm_total': fwhm_total,
    }


# ---------------------------------------------------------------------------
# Cholesky decomposition for Fisher coordinates (Phase 2 prep)
# ---------------------------------------------------------------------------

def fisher_cholesky(g: np.ndarray) -> np.ndarray:
    """Compute Cholesky factor L such that g = L @ L.T.

    In Fisher coordinates xi = L @ (theta - theta_ref), the metric
    becomes the identity: ||xi_1 - xi_2||^2 ~ KL(p_1 || p_2).

    Args:
        g: Fisher Information Matrix (n, n), positive definite

    Returns:
        L: Lower triangular Cholesky factor (n, n)

    Raises:
        np.linalg.LinAlgError: if g is not positive definite
    """
    return np.linalg.cholesky(g)


def fisher_transform(
    theta: np.ndarray,
    theta_ref: np.ndarray,
    L: np.ndarray,
) -> np.ndarray:
    """Transform from parameter space to Fisher coordinates.

    xi = L @ (theta - theta_ref)

    Args:
        theta: Parameters (n_params,) or (n_points, n_params)
        theta_ref: Reference point (n_params,)
        L: Cholesky factor from fisher_cholesky()

    Returns:
        xi: Fisher coordinates, same shape as theta
    """
    delta = theta - theta_ref
    if delta.ndim == 1:
        return L @ delta
    else:
        return (L @ delta.T).T


def fisher_inverse_transform(
    xi: np.ndarray,
    theta_ref: np.ndarray,
    L: np.ndarray,
) -> np.ndarray:
    """Transform from Fisher coordinates back to parameter space.

    theta = L^{-1} @ xi + theta_ref

    Args:
        xi: Fisher coordinates (n_params,) or (n_points, n_params)
        theta_ref: Reference point
        L: Cholesky factor

    Returns:
        theta: Parameters in original space
    """
    L_inv = np.linalg.inv(L)
    if xi.ndim == 1:
        return L_inv @ xi + theta_ref
    else:
        return (L_inv @ xi.T).T + theta_ref


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------

def print_fisher_summary(result: FisherResult, title: str = '') -> None:
    """Print a formatted summary of a FisherResult."""
    n = len(result.param_names)
    header = f"Fisher Information Matrix ({n}D)"
    if title:
        header += f" — {title}"
    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")

    # Parameters
    print(f"  amplitude = {result.params['amplitude']:.1f}")
    print(f"  sigma     = {result.params['sigma']:.4f} eV")
    print(f"  gamma     = {result.params['gamma']:.4f} eV")
    eta = eta_from_sigma_gamma(result.params['sigma'], result.params['gamma'])
    print(f"  eta       = {eta:.3f} (Lorentzian fraction)")

    # Diagonal (information per parameter)
    print(f"\n{'Diagonal g_ii (information per parameter)':}")
    for i, name in enumerate(result.param_names):
        print(f"  g_{name:12s} = {result.g[i,i]:12.2f}")

    # Correlation matrix
    print("\nCorrelation matrix r_ij:")
    header_str = "          " + "".join(f"{name:>12s}" for name in result.param_names)
    print(header_str)
    for i, name_i in enumerate(result.param_names):
        row = f"  {name_i:8s}"
        for j in range(n):
            row += f"{result.correlation[i,j]:12.4f}"
        print(row)

    # Eigenvalues
    print("\nEigenvalues (ascending):")
    for i, ev in enumerate(result.eigenvalues):
        pct = 100 * ev / result.eigenvalues.sum() if result.eigenvalues.sum() > 0 else 0
        print(f"  lambda_{i+1} = {ev:12.2f}  ({pct:5.1f}%)")

    print(f"\nCondition number: {result.condition_number:.1f}")

    if n == 4:
        total = result.eigenvalues.sum()
        top3 = result.eigenvalues[-3:].sum()
        print(f"3D retention: {100*top3/total:.2f}% (top-3 eigenvalues)")

    print(f"{'='*60}")
