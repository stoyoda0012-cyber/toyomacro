"""
Gauss-Newton Refiner for Voigt Profiles
=======================================

Stage 2 refinement using analytical Jacobian.
Optimizes non-linear parameters (center, sigma, gamma) while
analytically solving for amplitudes at each step.

Modes:
- Light: center only (1D per component)
- Medium: center + sigma (2D per component)
- Full: center + sigma + gamma (3D per component)

Reference:
    Golub & Pereyra (2003) "Separable nonlinear least squares"
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

from .voigt_jacobian import voigt_jacobian_batch


class RefineMode(Enum):
    """Refinement mode for Stage 2."""
    LIGHT = "light"      # center only
    MEDIUM = "medium"    # center + sigma
    FULL = "full"        # center + sigma + gamma


@dataclass
class RefineResult:
    """Result of Gauss-Newton refinement."""
    centers: np.ndarray       # Refined centers (n_comp,)
    sigmas: np.ndarray        # Refined sigmas (n_comp,)
    gammas: np.ndarray        # Refined gammas (n_comp,)
    amplitudes: np.ndarray    # Optimal amplitudes (n_comp,)
    residual_norm: float      # ||y - Φ@A||
    chi2: float               # residual_norm² / n_energy
    n_iter: int               # Iterations used
    converged: bool           # Convergence flag
    delta_history: list       # History of parameter updates


class GaussNewtonRefiner:
    """
    Gauss-Newton optimizer for Voigt profile refinement.

    Uses Variable Projection: at each step, amplitudes are solved
    analytically, and non-linear parameters are updated via Gauss-Newton.

    Usage:
        refiner = GaussNewtonRefiner(mode='medium', max_iter=10)
        result = refiner.refine(
            y=spectrum,
            energy=energy_axis,
            center0=np.array([707.0]),
            sigma0=np.array([0.8]),
            gamma0=np.array([0.3])
        )
    """

    def __init__(
        self,
        mode: Literal["light", "medium", "full"] = "medium",
        max_iter: int = 10,
        tol: float = 1e-6,
        damping: float = 0.5,
        reg_lambda: float = 1e-8,
        bounds_center: tuple[float, float] | None = None,
        bounds_sigma: tuple[float, float] = (0.1, 5.0),
        bounds_gamma: tuple[float, float] = (0.01, 2.0),
    ):
        """
        Initialize refiner.

        Args:
            mode: 'light' (center), 'medium' (center+σ), 'full' (center+σ+γ)
            max_iter: Maximum Gauss-Newton iterations
            tol: Convergence tolerance (relative change in residual)
            damping: Step size damping factor (0 < damping <= 1)
            reg_lambda: Tikhonov regularization for amplitude solve
            bounds_center: (min_delta, max_delta) from initial center
            bounds_sigma: (min, max) for sigma
            bounds_gamma: (min, max) for gamma
        """
        self.mode = RefineMode(mode)
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.reg_lambda = reg_lambda
        self.bounds_center = bounds_center
        self.bounds_sigma = bounds_sigma
        self.bounds_gamma = bounds_gamma

    def _solve_amplitudes(
        self,
        Phi: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Solve for optimal amplitudes: A* = (Φ'Φ + λI)⁻¹ Φ' y

        Args:
            Phi: Basis matrix (n_energy, n_comp)
            y: Spectrum (n_energy,)

        Returns:
            A: Amplitudes (n_comp,)
        """
        n_comp = Phi.shape[1]
        AtA = Phi.T @ Phi + self.reg_lambda * np.eye(n_comp)
        Aty = Phi.T @ y
        return np.linalg.solve(AtA, Aty)

    def _build_jacobian(
        self,
        J_c: np.ndarray,
        J_sigma: np.ndarray,
        J_gamma: np.ndarray,
        A: np.ndarray,
    ) -> np.ndarray:
        """
        Build Jacobian matrix for current mode.

        The full model is: y = Φ(θ) @ A
        Jacobian of residual w.r.t. θ: J = -∂Φ/∂θ @ A

        Args:
            J_c, J_sigma, J_gamma: Partial derivatives (n_energy, n_comp)
            A: Current amplitudes (n_comp,)

        Returns:
            J: Jacobian matrix (n_energy, n_params)
        """
        n_energy, n_comp = J_c.shape

        # J_param[:, k] = ∂Φ[:, k]/∂param_k * A[k]
        # Weighted by amplitude

        if self.mode == RefineMode.LIGHT:
            # Only center: (n_energy, n_comp)
            J = J_c * A  # broadcast: (n_energy, n_comp) * (n_comp,)

        elif self.mode == RefineMode.MEDIUM:
            # Center + sigma: (n_energy, 2*n_comp)
            J = np.zeros((n_energy, 2 * n_comp))
            J[:, :n_comp] = J_c * A
            J[:, n_comp:] = J_sigma * A

        else:  # FULL
            # Center + sigma + gamma: (n_energy, 3*n_comp)
            J = np.zeros((n_energy, 3 * n_comp))
            J[:, :n_comp] = J_c * A
            J[:, n_comp:2*n_comp] = J_sigma * A
            J[:, 2*n_comp:] = J_gamma * A

        return J

    def _pack_params(
        self,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gammas: np.ndarray,
    ) -> np.ndarray:
        """Pack parameters into vector according to mode."""
        if self.mode == RefineMode.LIGHT:
            return centers.copy()
        elif self.mode == RefineMode.MEDIUM:
            return np.concatenate([centers, sigmas])
        else:
            return np.concatenate([centers, sigmas, gammas])

    def _unpack_params(
        self,
        theta: np.ndarray,
        n_comp: int,
        sigmas0: np.ndarray,
        gammas0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unpack parameter vector according to mode."""
        if self.mode == RefineMode.LIGHT:
            return theta, sigmas0.copy(), gammas0.copy()
        elif self.mode == RefineMode.MEDIUM:
            return theta[:n_comp], theta[n_comp:], gammas0.copy()
        else:
            return theta[:n_comp], theta[n_comp:2*n_comp], theta[2*n_comp:]

    def _apply_bounds(
        self,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gammas: np.ndarray,
        centers0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply parameter bounds."""
        # Center bounds (relative to initial)
        if self.bounds_center is not None:
            delta_min, delta_max = self.bounds_center
            centers = np.clip(centers, centers0 + delta_min, centers0 + delta_max)

        # Sigma bounds
        sigmas = np.clip(sigmas, *self.bounds_sigma)

        # Gamma bounds
        gammas = np.clip(gammas, *self.bounds_gamma)

        return centers, sigmas, gammas

    def refine(
        self,
        y: np.ndarray,
        energy: np.ndarray,
        center0: np.ndarray,
        sigma0: np.ndarray,
        gamma0: np.ndarray,
    ) -> RefineResult:
        """
        Refine Voigt parameters using Gauss-Newton.

        Args:
            y: Spectrum (n_energy,)
            energy: Energy axis (n_energy,)
            center0: Initial centers (n_comp,)
            sigma0: Initial sigmas (n_comp,)
            gamma0: Initial gammas (n_comp,)

        Returns:
            RefineResult with optimized parameters
        """
        n_comp = len(center0)
        n_energy = len(energy)

        # Initialize
        centers = center0.copy()
        sigmas = sigma0.copy()
        gammas = gamma0.copy()

        delta_history = []
        prev_residual_norm = np.inf

        for iteration in range(self.max_iter):
            # Compute basis and Jacobian
            Phi, J_c, J_sigma, J_gamma = voigt_jacobian_batch(
                energy, centers, sigmas, gammas
            )

            # Solve for amplitudes
            A = self._solve_amplitudes(Phi, y)

            # Compute residual
            r = y - Phi @ A
            residual_norm = np.linalg.norm(r)

            # Check convergence. Skipped on the first pass: there is no
            # previous norm to compare against, and computing the ratio
            # from the inf sentinel gives inf/inf = nan. `rel_change` is
            # read again after the loop, under the same `iteration > 0`
            # guard, so it is always bound where it is used.
            if iteration > 0:
                rel_change = (
                    abs(prev_residual_norm - residual_norm)
                    / (prev_residual_norm + 1e-10)
                )
                if rel_change < self.tol:
                    break
            prev_residual_norm = residual_norm

            # Build Jacobian for Gauss-Newton
            J = self._build_jacobian(J_c, J_sigma, J_gamma, A)

            # Gauss-Newton step: δ = (J'J)⁻¹ J' r
            # Note: r = y - Φ@A, and we want to minimize ||r||
            # The update should move params in direction that reduces r
            JtJ = J.T @ J
            Jtr = J.T @ r

            # Add small regularization for stability
            JtJ += 1e-8 * np.eye(JtJ.shape[0])

            try:
                delta = np.linalg.solve(JtJ, Jtr)
            except np.linalg.LinAlgError:
                # Singular matrix, use pseudoinverse
                delta = np.linalg.lstsq(JtJ, Jtr, rcond=None)[0]

            delta_history.append(delta.copy())

            # Apply damped update
            theta = self._pack_params(centers, sigmas, gammas)
            theta_new = theta + self.damping * delta

            # Unpack and apply bounds
            centers, sigmas, gammas = self._unpack_params(
                theta_new, n_comp, sigma0, gamma0
            )
            centers, sigmas, gammas = self._apply_bounds(
                centers, sigmas, gammas, center0
            )

        # Final computation
        Phi, _, _, _ = voigt_jacobian_batch(energy, centers, sigmas, gammas)
        A = self._solve_amplitudes(Phi, y)
        r = y - Phi @ A
        residual_norm = np.linalg.norm(r)
        chi2 = residual_norm**2 / n_energy

        return RefineResult(
            centers=centers,
            sigmas=sigmas,
            gammas=gammas,
            amplitudes=A,
            residual_norm=residual_norm,
            chi2=chi2,
            n_iter=iteration + 1,
            converged=(rel_change < self.tol) if iteration > 0 else True,
            delta_history=delta_history,
        )


def demo_gauss_newton():
    """Demonstration of Gauss-Newton refiner."""
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Gauss-Newton Refiner Demo")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    energy = np.linspace(95, 110, 151)

    # True parameters
    true_center = np.array([99.5, 103.0])
    true_sigma = np.array([0.6, 0.8])
    true_gamma = np.array([0.25, 0.30])
    true_amp = np.array([1000.0, 600.0])

    # Generate clean spectrum
    y_clean = np.zeros_like(energy)
    for c, s, g, a in zip(true_center, true_sigma, true_gamma, true_amp):
        from .voigt_jacobian import voigt_profile
        y_clean += a * voigt_profile(energy, c, s, g)

    # Add Poisson-like noise (SNR ~ 30)
    noise_level = np.sqrt(y_clean.max()) / 30
    y = y_clean + noise_level * np.random.randn(len(energy)) * np.sqrt(np.maximum(y_clean, 1))

    print("\nTrue parameters:")
    print(f"  Centers: {true_center}")
    print(f"  Sigmas:  {true_sigma}")
    print(f"  Gammas:  {true_gamma}")
    print(f"  Amps:    {true_amp}")

    # Perturbed initial guess
    init_center = true_center + np.array([0.3, -0.2])  # shift
    init_sigma = true_sigma * np.array([1.2, 0.9])     # scale
    init_gamma = true_gamma * np.array([0.8, 1.1])     # scale

    print("\nInitial guess (perturbed):")
    print(f"  Centers: {init_center}")
    print(f"  Sigmas:  {init_sigma}")
    print(f"  Gammas:  {init_gamma}")

    # Test all three modes
    results = {}
    for mode in ["light", "medium", "full"]:
        refiner = GaussNewtonRefiner(
            mode=mode,
            max_iter=20,
            damping=0.7,
            bounds_center=(-2.0, 2.0),
        )
        result = refiner.refine(y, energy, init_center, init_sigma, init_gamma)
        results[mode] = result

        print(f"\n--- Mode: {mode.upper()} ---")
        print(f"  Iterations: {result.n_iter}, Converged: {result.converged}")
        print(f"  Chi2: {result.chi2:.6f}")
        print(f"  Centers: {result.centers} (true: {true_center})")
        print(f"  Sigmas:  {result.sigmas} (true: {true_sigma})")
        print(f"  Gammas:  {result.gammas} (true: {true_gamma})")
        print(f"  Amps:    {result.amplitudes.astype(int)} (true: {true_amp.astype(int)})")

        # Errors
        print(f"  Center error: {np.abs(result.centers - true_center).mean():.4f} eV")
        if mode != "light":
            print(f"  Sigma error:  {np.abs(result.sigmas - true_sigma).mean():.4f}")
        if mode == "full":
            print(f"  Gamma error:  {np.abs(result.gammas - true_gamma).mean():.4f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Data and fits
    ax = axes[0, 0]
    ax.plot(energy, y, 'ko', ms=3, alpha=0.5, label='Data')
    ax.plot(energy, y_clean, 'k-', lw=1, alpha=0.3, label='True')

    colors = {'light': 'C0', 'medium': 'C1', 'full': 'C2'}
    for mode, result in results.items():
        from .voigt_jacobian import voigt_profile
        y_fit = np.zeros_like(energy)
        for c, s, g, a in zip(result.centers, result.sigmas, result.gammas, result.amplitudes):
            y_fit += a * voigt_profile(energy, c, s, g)
        ax.plot(energy, y_fit, '-', color=colors[mode], lw=2,
                label=f'{mode} (χ²={result.chi2:.4f})')

    ax.set_xlabel('Binding Energy [eV]')
    ax.set_ylabel('Intensity')
    ax.set_title('Gauss-Newton Refinement')
    ax.legend()
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # Center convergence
    ax = axes[0, 1]
    for mode, result in results.items():
        if result.delta_history:
            n_comp = len(true_center)
            deltas = np.array([d[:n_comp] for d in result.delta_history])
            for k in range(n_comp):
                ax.plot(np.abs(deltas[:, k]), 'o-', color=colors[mode],
                        alpha=0.7, label=f'{mode} comp{k+1}' if k == 0 else None)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('|Δcenter| [eV]')
    ax.set_title('Center Update Convergence')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residuals
    ax = axes[1, 0]
    for mode, result in results.items():
        from .voigt_jacobian import voigt_profile
        y_fit = np.zeros_like(energy)
        for c, s, g, a in zip(result.centers, result.sigmas, result.gammas, result.amplitudes):
            y_fit += a * voigt_profile(energy, c, s, g)
        residual = y - y_fit
        ax.plot(energy, residual, '-', color=colors[mode], lw=1.5,
                label=f'{mode}', alpha=0.7)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('Binding Energy [eV]')
    ax.set_ylabel('Residual')
    ax.set_title('Fit Residuals')
    ax.legend()
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # Parameter errors
    ax = axes[1, 1]
    modes = list(results.keys())
    x = np.arange(len(modes))
    width = 0.25

    center_err = [np.abs(results[m].centers - true_center).mean() for m in modes]
    sigma_err = [np.abs(results[m].sigmas - true_sigma).mean() for m in modes]
    gamma_err = [np.abs(results[m].gammas - true_gamma).mean() for m in modes]

    ax.bar(x - width, center_err, width, label='Center', color='C0')
    ax.bar(x, sigma_err, width, label='Sigma', color='C1')
    ax.bar(x + width, gamma_err, width, label='Gamma', color='C2')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modes])
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Parameter Estimation Errors')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('gauss_newton_demo.png', dpi=150)
    print("\nFigure saved: gauss_newton_demo.png")

    return results


if __name__ == "__main__":
    demo_gauss_newton()
