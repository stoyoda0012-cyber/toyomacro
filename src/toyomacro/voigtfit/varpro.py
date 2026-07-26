"""
Variable Projection (VarPro) Fitter for Voigt Profiles
========================================================

VarPro separates linear and non-linear parameters:
    - Non-linear: peak centers, widths (optimize numerically)
    - Linear: amplitudes (solve analytically given non-linear params)

This eliminates amplitudes from the optimization, reducing dimensionality
and improving convergence.

For Voigt fitting:
    Y = Φ(θ) @ A + ε

    where θ = (centers, sigmas) are non-linear parameters
    and A = amplitudes are linear parameters

VarPro minimizes:
    ||Y - Φ(θ) @ A*(θ)||²

    where A*(θ) = (Φ'Φ)⁻¹ Φ' Y is the optimal amplitude for given θ

Reference:
    Golub & Pereyra (1973) "The Differentiation of Pseudo-Inverses..."
"""


import numpy as np
from scipy import optimize
from scipy import special as sps

SQRT2 = np.sqrt(2.0)
SQRT2PI = np.sqrt(2.0 * np.pi)


class VarProFitter:
    """
    Variable Projection fitter for Voigt profiles.

    Optimizes (center, sigma) while analytically eliminating amplitudes.

    Usage:
        fitter = VarProFitter()
        result = fitter.fit_single(
            y=spectrum,
            energy=energy_axis,
            initial_centers=np.array([707.0, 709.5]),
            initial_sigmas=np.array([0.8, 0.9]),
            gamma=0.2
        )
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        bounds_sigma: tuple[float, float] = (0.1, 5.0),
        bounds_center_delta: float = 2.0,
        reg_lambda: float = 1e-8,
    ):
        """
        Initialize VarPro fitter.

        Args:
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
            bounds_sigma: (min, max) bounds for sigma
            bounds_center_delta: Max shift from initial center position
            reg_lambda: Tikhonov regularization for amplitude solve
        """
        self.max_iter = max_iter
        self.tol = tol
        self.bounds_sigma = bounds_sigma
        self.bounds_center_delta = bounds_center_delta
        self.reg_lambda = reg_lambda

    def voigt_basis(
        self,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        """
        Compute Voigt basis matrix using Faddeeva function.

        Args:
            energy: Energy axis (n_energy,)
            centers: Peak positions (n_components,)
            sigmas: Gaussian widths (n_components,)
            gamma: Lorentzian width (scalar)

        Returns:
            Phi: Basis matrix (n_energy, n_components)
        """
        n_energy = len(energy)
        n_components = len(centers)
        Phi = np.zeros((n_energy, n_components), dtype=np.float64)

        for k in range(n_components):
            z = ((energy - centers[k]) + 1j * gamma) / (sigmas[k] * SQRT2)
            Phi[:, k] = np.real(sps.wofz(z)) / (sigmas[k] * SQRT2PI)

        return Phi

    def solve_amplitudes(
        self,
        Phi: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Solve for optimal amplitudes given basis matrix.

        A* = (Φ'Φ + λI)⁻¹ Φ' y

        Args:
            Phi: Basis matrix (n_energy, n_components)
            y: Spectrum (n_energy,)

        Returns:
            A: Optimal amplitudes (n_components,)
        """
        n_components = Phi.shape[1]
        AtA = Phi.T @ Phi + self.reg_lambda * np.eye(n_components)
        Aty = Phi.T @ y
        return np.linalg.solve(AtA, Aty)

    def residual(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        energy: np.ndarray,
        gamma: float,
        n_components: int,
    ) -> float:
        """
        VarPro residual: ||y - Φ(θ) @ A*(θ)||²

        Args:
            theta: Non-linear params [centers..., sigmas...]
            y: Spectrum (n_energy,)
            energy: Energy axis (n_energy,)
            gamma: Lorentzian width
            n_components: Number of peaks

        Returns:
            Squared residual norm
        """
        centers = theta[:n_components]
        sigmas = theta[n_components:]

        Phi = self.voigt_basis(energy, centers, sigmas, gamma)
        A = self.solve_amplitudes(Phi, y)

        residuals = y - Phi @ A
        return np.sum(residuals**2)

    def fit_single(
        self,
        y: np.ndarray,
        energy: np.ndarray,
        initial_centers: np.ndarray,
        initial_sigmas: np.ndarray,
        gamma: float = 0.2,
        initial_amplitudes: np.ndarray | None = None,
    ) -> dict:
        """
        Fit single spectrum using VarPro.

        Args:
            y: Spectrum (n_energy,)
            energy: Energy axis (n_energy,)
            initial_centers: Initial peak positions
            initial_sigmas: Initial Gaussian widths
            gamma: Lorentzian width (fixed)
            initial_amplitudes: Optional initial amplitudes (unused, for interface)

        Returns:
            Dict with centers, sigmas, amplitudes, chi2, success. ``chi2`` is
            the unweighted mean squared residual (not a variance-weighted χ²).
        """
        n_components = len(initial_centers)

        # Pack initial parameters
        theta0 = np.concatenate([initial_centers, initial_sigmas])

        # Set bounds
        bounds_lower = np.concatenate([
            initial_centers - self.bounds_center_delta,
            np.full(n_components, self.bounds_sigma[0]),
        ])
        bounds_upper = np.concatenate([
            initial_centers + self.bounds_center_delta,
            np.full(n_components, self.bounds_sigma[1]),
        ])
        bounds = optimize.Bounds(bounds_lower, bounds_upper)

        # Optimize
        result = optimize.minimize(
            self.residual,
            theta0,
            args=(y, energy, gamma, n_components),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        # Extract results
        centers_opt = result.x[:n_components]
        sigmas_opt = result.x[n_components:]

        # Get final amplitudes
        Phi = self.voigt_basis(energy, centers_opt, sigmas_opt, gamma)
        amplitudes = self.solve_amplitudes(Phi, y)

        # Fit quality: unweighted mean squared residual (reported as "chi2")
        residuals = y - Phi @ amplitudes
        chi2 = np.sum(residuals**2) / len(y)

        return {
            "centers": centers_opt,
            "sigmas": sigmas_opt,
            "amplitudes": amplitudes,
            "chi2": chi2,
            "success": result.success,
            "n_iter": result.nit,
        }

    def fit_batch(
        self,
        Y: np.ndarray,
        energy: np.ndarray,
        initial_centers: np.ndarray,
        initial_sigmas: np.ndarray,
        gamma: float = 0.2,
    ) -> dict:
        """
        Fit a batch of spectra with a sequential loop over ``fit_single``.

        For large batches use ``FastVoigtFitter`` (the batch-first,
        matmul-shaped pipeline); this method is a convenience wrapper.

        Args:
            Y: Spectra matrix (n_energy, n_spectra)
            energy: Energy axis (n_energy,)
            initial_centers: Initial peak positions (shared)
            initial_sigmas: Initial Gaussian widths (shared)
            gamma: Lorentzian width

        Returns:
            Dict with arrays of results (``chi2`` as in ``fit_single``:
            unweighted mean squared residual)
        """
        n_spectra = Y.shape[1]
        n_components = len(initial_centers)

        results_list = []
        for i in range(n_spectra):
            results_list.append(self.fit_single(
                y=Y[:, i],
                energy=energy,
                initial_centers=initial_centers,
                initial_sigmas=initial_sigmas,
                gamma=gamma,
            ))

        # Collect results
        centers_all = np.zeros((n_components, n_spectra))
        sigmas_all = np.zeros((n_components, n_spectra))
        amplitudes_all = np.zeros((n_components, n_spectra))
        chi2_all = np.zeros(n_spectra)
        success_all = np.zeros(n_spectra, dtype=bool)

        for i, result in enumerate(results_list):
            centers_all[:, i] = result["centers"]
            sigmas_all[:, i] = result["sigmas"]
            amplitudes_all[:, i] = result["amplitudes"]
            chi2_all[i] = result["chi2"]
            success_all[i] = result["success"]

        return {
            "centers": centers_all,
            "sigmas": sigmas_all,
            "amplitudes": amplitudes_all,
            "chi2": chi2_all,
            "success": success_all,
        }


class VarProFitterMLX:
    """
    MLX-accelerated VarPro fitter (for future implementation).

    Uses MLX for batch gradient computation and optimization.
    Expected performance: 10-100K spectra/s
    """

    def __init__(self):
        raise NotImplementedError(
            "MLX-accelerated VarPro not yet implemented. "
            "Use VarProFitter for CPU-based fitting."
        )
