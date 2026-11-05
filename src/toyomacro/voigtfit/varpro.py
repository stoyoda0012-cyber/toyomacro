"""
Variable Projection (VarPro) Fitter for Voigt Profiles
========================================================

VarPro separates linear and non-linear parameters:
    - Non-linear: peak centers, widths (optimize numerically)
    - Linear: amplitudes, plus any caller-supplied background columns
      (solve analytically given non-linear params)

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

    @staticmethod
    def _check_background_design(
        background_design: np.ndarray | None,
        n_energy: int,
    ) -> np.ndarray | None:
        """Validate an optional background design matrix.

        Returns it as a float64 array of shape (n_energy, n_background),
        or None. Raises ValueError on any other shape or on non-finite
        entries; nothing is reshaped or repaired.
        """
        if background_design is None:
            return None
        B = np.asarray(background_design, dtype=np.float64)
        if B.ndim != 2 or B.shape[0] != n_energy:
            raise ValueError(
                f"background_design must have shape (n_energy, n_background) = "
                f"({n_energy}, n_background), got {B.shape}"
            )
        if not np.all(np.isfinite(B)):
            raise ValueError("background_design contains non-finite values")
        return B

    def _design(
        self,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float,
        background_design: np.ndarray | None,
    ) -> np.ndarray:
        """Joint linear design [Φ(θ) | B]: Voigt columns, then background columns."""
        Phi = self.voigt_basis(energy, centers, sigmas, gamma)
        if background_design is None:
            return Phi
        return np.hstack([Phi, background_design])

    def residual(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        energy: np.ndarray,
        gamma: float,
        n_components: int,
        background_design: np.ndarray | None = None,
        fixed_sigmas: np.ndarray | None = None,
    ) -> float:
        """
        VarPro residual: ||y - D(θ) @ c*(θ)||², with D = [Φ(θ) | B].

        Args:
            theta: Non-linear params [centers..., sigmas...], or
                [centers...] when ``fixed_sigmas`` is given
            y: Spectrum (n_energy,)
            energy: Energy axis (n_energy,)
            gamma: Lorentzian width
            n_components: Number of peaks
            background_design: Optional extra linear columns B
                (n_energy, n_background), solved jointly with the amplitudes
            fixed_sigmas: Optional Gaussian widths held fixed (n_components,)

        Returns:
            Squared residual norm
        """
        centers = theta[:n_components]
        sigmas = fixed_sigmas if fixed_sigmas is not None else theta[n_components:]

        D = self._design(energy, centers, sigmas, gamma, background_design)
        c = self.solve_amplitudes(D, y)

        residuals = y - D @ c
        return np.sum(residuals**2)

    def fit_single(
        self,
        y: np.ndarray,
        energy: np.ndarray,
        initial_centers: np.ndarray,
        initial_sigmas: np.ndarray,
        gamma: float = 0.2,
        initial_amplitudes: np.ndarray | None = None,
        background_design: np.ndarray | None = None,
        fixed_shape: bool = False,
    ) -> dict:
        """
        Fit single spectrum using VarPro.

        Args:
            y: Spectrum (n_energy,)
            energy: Energy axis (n_energy,)
            initial_centers: Initial peak positions
            initial_sigmas: Initial Gaussian widths; with ``fixed_shape`` these
                are the fixed widths
            gamma: Lorentzian width (fixed)
            initial_amplitudes: Optional initial amplitudes (unused, for interface)
            background_design: Optional extra linear columns B of shape
                (n_energy, n_background), for example a constant, a linear ramp
                or a caller-supplied background curve. Their coefficients are
                solved jointly with the peak amplitudes in the same inner linear
                solve (the Tikhonov term ``reg_lambda`` applies to every column
                of the joint design). The caller supplies the columns; none are
                built here. A wrong shape raises ``ValueError``.
            fixed_shape: If True, the Gaussian widths are held at
                ``initial_sigmas`` and only the centers are optimized (within
                ``bounds_center_delta``). With ``bounds_center_delta=0`` no
                non-linear parameter is free and the fit is the linear solve
                alone.

        Returns:
            Dict with centers, sigmas, amplitudes, background, chi2, success,
            n_iter. ``background`` holds the coefficients of the
            ``background_design`` columns (shape (0,) when none is given).
            ``chi2`` is the unweighted mean squared residual of the full model,
            peaks plus background (not a variance-weighted χ²).
        """
        initial_centers = np.asarray(initial_centers, dtype=np.float64)
        initial_sigmas = np.asarray(initial_sigmas, dtype=np.float64)
        n_components = len(initial_centers)
        B = self._check_background_design(background_design, len(energy))
        if fixed_shape and initial_sigmas.shape != (n_components,):
            raise ValueError(
                f"fixed_shape needs one sigma per component: got {initial_sigmas.shape} "
                f"for {n_components} centers"
            )
        fixed_sigmas = initial_sigmas if fixed_shape else None

        # Pack initial parameters and bounds
        center_lower = initial_centers - self.bounds_center_delta
        center_upper = initial_centers + self.bounds_center_delta
        if fixed_shape:
            theta0 = initial_centers.copy()
            bounds = optimize.Bounds(center_lower, center_upper)
        else:
            theta0 = np.concatenate([initial_centers, initial_sigmas])
            bounds = optimize.Bounds(
                np.concatenate([center_lower, np.full(n_components, self.bounds_sigma[0])]),
                np.concatenate([center_upper, np.full(n_components, self.bounds_sigma[1])]),
            )

        # Optimize
        result = optimize.minimize(
            self.residual,
            theta0,
            args=(y, energy, gamma, n_components, B, fixed_sigmas),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        # Extract results
        centers_opt = result.x[:n_components]
        sigmas_opt = initial_sigmas.copy() if fixed_shape else result.x[n_components:]

        # Get final linear coefficients
        D = self._design(energy, centers_opt, sigmas_opt, gamma, B)
        coef = self.solve_amplitudes(D, y)
        amplitudes = coef[:n_components]
        background = coef[n_components:]

        # Fit quality: unweighted mean squared residual (reported as "chi2")
        residuals = y - D @ coef
        chi2 = np.sum(residuals**2) / len(y)

        return {
            "centers": centers_opt,
            "sigmas": sigmas_opt,
            "amplitudes": amplitudes,
            "background": background,
            "chi2": chi2,
            "success": result.success,
            # SciPy omits ``nit`` when every variable is fixed by its bounds
            "n_iter": getattr(result, "nit", 0),
        }

    def fit_batch(
        self,
        Y: np.ndarray,
        energy: np.ndarray,
        initial_centers: np.ndarray,
        initial_sigmas: np.ndarray,
        gamma: float = 0.2,
        background_design: np.ndarray | None = None,
        fixed_shape: bool = False,
        energy_axis: int = 0,
    ) -> dict:
        """
        Fit a batch of spectra with a sequential loop over ``fit_single``.

        For large batches use ``FastVoigtFitter`` (the batch-first,
        matmul-shaped pipeline); this method is a convenience wrapper. Each
        spectrum is fitted independently from the same starting values, so
        every result equals the corresponding ``fit_single`` call.

        Args:
            Y: Spectra. By default (``energy_axis=0``) a matrix
                (n_energy, n_spectra). With ``energy_axis=-1`` the energy axis
                is last and any leading axes are batch axes, e.g.
                (K, n_spectra, n_energy) for K stacked replicates.
            energy: Energy axis (n_energy,)
            initial_centers: Initial peak positions (shared)
            initial_sigmas: Initial Gaussian widths (shared)
            gamma: Lorentzian width
            background_design: Optional (n_energy, n_background) background
                columns, shared by every spectrum (see ``fit_single``)
            fixed_shape: Hold the widths at ``initial_sigmas`` (see
                ``fit_single``)
            energy_axis: Axis of ``Y`` that runs over energy; every other
                axis is a batch axis

        Returns:
            Dict with arrays of results. ``centers``, ``sigmas``, ``amplitudes``
            and ``background`` carry their parameter axis where ``Y`` has its
            energy axis: (n_components, n_spectra) for the default layout,
            batch_shape + (n_components,) for ``energy_axis=-1``. ``chi2`` and
            ``success`` have the batch shape. ``chi2`` is as in ``fit_single``:
            unweighted mean squared residual.
        """
        Y = np.asarray(Y)
        if Y.ndim < 2:
            raise ValueError(f"Y must have an energy axis and at least one batch axis, "
                             f"got shape {Y.shape}")
        axis = energy_axis % Y.ndim
        Y_last = np.moveaxis(Y, axis, -1)
        n_energy = Y_last.shape[-1]
        if n_energy != len(energy):
            raise ValueError(
                f"Y has {n_energy} points along energy_axis={energy_axis}, "
                f"but energy has {len(energy)}"
            )
        B = self._check_background_design(background_design, n_energy)
        batch_shape = Y_last.shape[:-1]
        flat = Y_last.reshape(-1, n_energy)

        results_list = [
            self.fit_single(
                y=flat[i],
                energy=energy,
                initial_centers=initial_centers,
                initial_sigmas=initial_sigmas,
                gamma=gamma,
                background_design=B,
                fixed_shape=fixed_shape,
            )
            for i in range(flat.shape[0])
        ]

        def stack(key: str) -> np.ndarray:
            arr = np.stack([r[key] for r in results_list])
            arr = arr.reshape(batch_shape + arr.shape[1:])
            return np.moveaxis(arr, -1, axis)

        return {
            "centers": stack("centers"),
            "sigmas": stack("sigmas"),
            "amplitudes": stack("amplitudes"),
            "background": stack("background"),
            "chi2": np.array([r["chi2"] for r in results_list]).reshape(batch_shape),
            "success": np.array([r["success"] for r in results_list],
                                dtype=bool).reshape(batch_shape),
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
