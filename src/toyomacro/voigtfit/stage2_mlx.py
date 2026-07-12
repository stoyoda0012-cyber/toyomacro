"""
Stage 2: MLX-accelerated Gauss-Newton Refinement
=================================================

GPU-accelerated batch processing of anomalous spectra using MLX.
Uses table-based Voigt profile computation for speed.

Key optimizations:
1. Pre-computed Faddeeva lookup table on GPU
2. Batched linear solves via MLX
3. Vectorized Jacobian computation via numerical differentiation
4. All operations stay on GPU until final result

Target performance: 50-200K spectra/s (vs ~4K on CPU)
"""

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    import mlx.core as mx

    from ._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND a Metal device works
except ImportError:
    HAS_MLX = False
    mx = None

from .faddeeva_mlx import (
    get_faddeeva_table,
    voigt_basis_batch_mlx_vectorized,
    voigt_basis_with_gradient_mlx,
)

# Constants for numerical differentiation
EPS_CENTER = 0.001  # eV
EPS_SIGMA = 0.001   # eV
EPS_GAMMA = 0.001   # eV


@dataclass
class Stage2MLXConfig:
    """Configuration for MLX Stage 2 refinement."""
    mode: Literal["light", "medium", "full"] = "medium"
    max_iter: int = 15
    tol: float = 1e-5
    damping: float = 0.5
    bounds_center: tuple[float, float] = (-2.0, 2.0)
    bounds_sigma: tuple[float, float] = (0.3, 3.0)
    bounds_gamma: tuple[float, float] = (0.05, 1.0)


@dataclass
class Stage2MLXResult:
    """Result of MLX Stage 2 batch refinement."""
    centers: np.ndarray       # (n_comp, n_spectra)
    sigmas: np.ndarray        # (n_comp, n_spectra)
    gammas: np.ndarray        # (n_comp, n_spectra)
    amplitudes: np.ndarray    # (n_comp, n_spectra)
    chi2: np.ndarray          # (n_spectra,)
    converged: np.ndarray     # (n_spectra,) bool
    n_iter: np.ndarray        # (n_spectra,) int
    indices: np.ndarray       # Original indices


def voigt_basis_batch_mlx(energy, centers, sigmas, gamma):
    """
    Compute Voigt basis for batch of spectra with per-spectrum centers.

    Uses vectorized implementation for better GPU performance.

    Args:
        energy: Energy axis (n_energy,) MLX array
        centers: Peak centers (n_comp, n_spectra) MLX array
        sigmas: Gaussian widths (n_comp,) - shared
        gamma: Lorentzian width (scalar)

    Returns:
        Phi: Basis matrix (n_energy, n_comp, n_spectra)
    """
    if not HAS_MLX:
        raise ImportError("MLX not available")

    # Convert centers to numpy for the vectorized function
    if isinstance(centers, mx.array):
        centers_np = np.array(centers)
    else:
        centers_np = np.asarray(centers, dtype=np.float32)

    if isinstance(sigmas, mx.array):
        sigmas_np = np.array(sigmas)
    else:
        sigmas_np = np.asarray(sigmas, dtype=np.float32)

    # Use the optimized vectorized implementation
    return voigt_basis_batch_mlx_vectorized(energy, centers_np, sigmas_np, float(gamma))


def solve_amplitudes_batch_mlx(Phi, Y):
    """
    Solve for amplitudes given basis and observations.

    Solves: A = (Φ'Φ)⁻¹ Φ' Y for each spectrum in batch.

    Args:
        Phi: Basis matrix (n_energy, n_comp, n_spectra)
        Y: Observations (n_energy, n_spectra)

    Returns:
        A: Amplitudes (n_comp, n_spectra)
    """
    n_energy, n_comp, n_spectra = Phi.shape

    # Optimized paths for common component counts
    if n_comp == 1:
        # 1-component: simple least squares A = Φ'Y / Φ'Φ
        phi0 = Phi[:, 0, :]  # (n_energy, n_spectra)
        p00 = mx.sum(phi0 * phi0, axis=0) + 1e-10
        ptY0 = mx.sum(phi0 * Y, axis=0)
        a0 = ptY0 / p00
        A = mx.expand_dims(a0, axis=0)  # (1, n_spectra)

    elif n_comp == 2:
        # 2-component: explicit 2x2 inverse
        phi0 = Phi[:, 0, :]
        phi1 = Phi[:, 1, :]

        p00 = mx.sum(phi0 * phi0, axis=0)
        p01 = mx.sum(phi0 * phi1, axis=0)
        p11 = mx.sum(phi1 * phi1, axis=0)

        ptY0 = mx.sum(phi0 * Y, axis=0)
        ptY1 = mx.sum(phi1 * Y, axis=0)

        det = p00 * p11 - p01 * p01 + 1e-10
        a0 = (p11 * ptY0 - p01 * ptY1) / det
        a1 = (p00 * ptY1 - p01 * ptY0) / det

        A = mx.stack([a0, a1], axis=0)

    elif n_comp == 3:
        # 3-component: explicit 3x3 inverse (Cramer's rule)
        phi0 = Phi[:, 0, :]
        phi1 = Phi[:, 1, :]
        phi2 = Phi[:, 2, :]

        # Gram matrix elements
        p00 = mx.sum(phi0 * phi0, axis=0)
        p01 = mx.sum(phi0 * phi1, axis=0)
        p02 = mx.sum(phi0 * phi2, axis=0)
        p11 = mx.sum(phi1 * phi1, axis=0)
        p12 = mx.sum(phi1 * phi2, axis=0)
        p22 = mx.sum(phi2 * phi2, axis=0)

        # Φ'Y
        ptY0 = mx.sum(phi0 * Y, axis=0)
        ptY1 = mx.sum(phi1 * Y, axis=0)
        ptY2 = mx.sum(phi2 * Y, axis=0)

        # 3x3 determinant
        det = (p00 * (p11 * p22 - p12 * p12)
               - p01 * (p01 * p22 - p12 * p02)
               + p02 * (p01 * p12 - p11 * p02)) + 1e-10

        # Cofactors for inverse
        c00 = p11 * p22 - p12 * p12
        c01 = -(p01 * p22 - p02 * p12)
        c02 = p01 * p12 - p02 * p11
        c11 = p00 * p22 - p02 * p02
        c12 = -(p00 * p12 - p02 * p01)
        c22 = p00 * p11 - p01 * p01

        # A = adj(P) * b / det
        a0 = (c00 * ptY0 + c01 * ptY1 + c02 * ptY2) / det
        a1 = (c01 * ptY0 + c11 * ptY1 + c12 * ptY2) / det
        a2 = (c02 * ptY0 + c12 * ptY1 + c22 * ptY2) / det

        A = mx.stack([a0, a1, a2], axis=0)

    else:
        # General case: use NumPy batch solve (fallback)
        mx.eval(Phi, Y)
        Phi_np = np.array(Phi)
        Y_np = np.array(Y)

        # Build Gram matrices: PhiTPhi[j, l, k] = sum_e Phi[e, j, k] * Phi[e, l, k]
        PhiTPhi = np.einsum('ejk,elk->jlk', Phi_np, Phi_np)
        PhiTPhi += 1e-8 * np.eye(n_comp)[:, :, np.newaxis]  # regularization

        # Build PhiTY: [j, k] = sum_e Phi[e, j, k] * Y[e, k]
        PhiTY = np.einsum('ejk,ek->jk', Phi_np, Y_np)

        # Batch solve - need to reshape for np.linalg.solve
        # np.linalg.solve expects (batch, n, n) and (batch, n) or (batch, n, k)
        PhiTPhi_T = PhiTPhi.transpose(2, 0, 1)  # (n_spectra, n_comp, n_comp)
        PhiTY_T = PhiTY.T[:, :, np.newaxis]  # (n_spectra, n_comp, 1)

        try:
            A_T = np.linalg.solve(PhiTPhi_T, PhiTY_T)[:, :, 0]  # (n_spectra, n_comp)
        except np.linalg.LinAlgError:
            A_T = np.zeros((n_spectra, n_comp), dtype=np.float32)
            for k in range(n_spectra):
                try:
                    A_T[k] = np.linalg.solve(PhiTPhi_T[k], PhiTY_T[k, :, 0])
                except np.linalg.LinAlgError:
                    pass

        A = mx.array(A_T.T.astype(np.float32))

    return A


def compute_residuals_batch_mlx(Phi, A, Y):
    """
    Compute residuals and chi-squared for batch.

    Args:
        Phi: Basis matrix (n_energy, n_comp, n_spectra)
        A: Amplitudes (n_comp, n_spectra)
        Y: Observations (n_energy, n_spectra)

    Returns:
        chi2: Reduced chi-squared (n_spectra,)
    """
    n_energy = Y.shape[0]

    # Y_fit = sum_j Phi[:, j, :] * A[j, :]
    # For 2 components:
    if Phi.shape[1] == 2:
        Y_fit = Phi[:, 0, :] * A[0:1, :] + Phi[:, 1, :] * A[1:2, :]
    else:
        # General case
        Y_fit = mx.sum(Phi * A[None, :, :], axis=1)

    residuals = Y - Y_fit
    chi2 = mx.sum(residuals * residuals, axis=0) / n_energy

    return chi2


def refine_center_batch_mlx(
    Y: "mx.array",
    energy: "mx.array",
    centers: "mx.array",
    sigmas: np.ndarray,
    gamma: float,
    config: Stage2MLXConfig,
) -> tuple["mx.array", "mx.array", "mx.array", "mx.array"]:
    """
    Refine centers for batch of spectra using vectorized Gauss-Newton (LIGHT mode).

    Optimized implementation using voigt_basis_with_gradient_mlx to compute
    basis and gradients in a single pass (reduces Faddeeva lookups by ~3x).

    Args:
        Y: Spectra (n_energy, n_spectra)
        energy: Energy axis (n_energy,)
        centers: Initial centers (n_comp, n_spectra)
        sigmas: Gaussian widths (n_comp,) - shared
        gamma: Lorentzian width (scalar)
        config: Refinement configuration

    Returns:
        centers: Refined centers (n_comp, n_spectra)
        amplitudes: Final amplitudes (n_comp, n_spectra)
        chi2: Final chi-squared (n_spectra,)
        n_iter: Iterations taken (n_spectra,)
    """
    n_energy, n_spectra = Y.shape
    n_comp = centers.shape[0]

    # Convert to numpy for internal use
    if isinstance(centers, mx.array):
        centers_init = np.array(centers).astype(np.float32)
    else:
        centers_init = np.asarray(centers, dtype=np.float32)

    centers_curr = centers_init.copy()
    sigmas_np = np.asarray(sigmas, dtype=np.float32)

    eps = 0.01  # Finite difference step

    # Track convergence
    prev_chi2 = mx.ones((n_spectra,)) * 1e10
    n_iter_tracker = mx.zeros((n_spectra,), dtype=mx.int32)

    for iteration in range(config.max_iter):
        # Compute basis AND gradient in one pass (optimized)
        Phi, dPhi_dc = voigt_basis_with_gradient_mlx(energy, centers_curr, sigmas_np, gamma, eps)

        # Solve for amplitudes
        A = solve_amplitudes_batch_mlx(Phi, Y)

        # Compute residuals and chi2
        if n_comp == 2:
            Y_fit = Phi[:, 0, :] * A[0:1, :] + Phi[:, 1, :] * A[1:2, :]
        else:
            Y_fit = mx.sum(Phi * A[None, :, :], axis=1)

        residuals = Y - Y_fit  # (n_energy, n_spectra)
        chi2 = mx.sum(residuals * residuals, axis=0) / n_energy

        # Check convergence
        rel_change = mx.abs(prev_chi2 - chi2) / (prev_chi2 + 1e-10)
        not_converged = rel_change >= config.tol
        n_iter_tracker = mx.where(not_converged, iteration + 1, n_iter_tracker)
        prev_chi2 = chi2

        # Compute Gauss-Newton update using pre-computed gradients
        # J[:, j, :] = A[j, :] * dPhi_dc[:, j, :]
        if n_comp == 1:
            # 1-component: simple scalar solve delta = J'r / J'J
            J0 = A[0:1, :] * dPhi_dc[:, 0, :]  # (n_energy, n_spectra)
            JtJ_00 = mx.sum(J0 * J0, axis=0) + 1e-6
            Jtr_0 = mx.sum(J0 * residuals, axis=0)
            delta_0 = Jtr_0 / JtJ_00
            delta = mx.expand_dims(delta_0, axis=0)  # (1, n_spectra)

        elif n_comp == 2:
            J0 = A[0:1, :] * dPhi_dc[:, 0, :]  # (n_energy, n_spectra)
            J1 = A[1:2, :] * dPhi_dc[:, 1, :]

            # J'J elements
            JtJ_00 = mx.sum(J0 * J0, axis=0)
            JtJ_01 = mx.sum(J0 * J1, axis=0)
            JtJ_11 = mx.sum(J1 * J1, axis=0)

            # J'r elements
            Jtr_0 = mx.sum(J0 * residuals, axis=0)
            Jtr_1 = mx.sum(J1 * residuals, axis=0)

            # Solve 2x2 system
            reg = 1e-6
            det = (JtJ_00 + reg) * (JtJ_11 + reg) - JtJ_01 * JtJ_01 + 1e-10

            delta_0 = ((JtJ_11 + reg) * Jtr_0 - JtJ_01 * Jtr_1) / det
            delta_1 = ((JtJ_00 + reg) * Jtr_1 - JtJ_01 * Jtr_0) / det

            delta = mx.stack([delta_0, delta_1], axis=0)

        elif n_comp == 3:
            # 3-component: explicit 3x3 solve
            J0 = A[0:1, :] * dPhi_dc[:, 0, :]
            J1 = A[1:2, :] * dPhi_dc[:, 1, :]
            J2 = A[2:3, :] * dPhi_dc[:, 2, :]

            # J'J elements (symmetric)
            JtJ_00 = mx.sum(J0 * J0, axis=0)
            JtJ_01 = mx.sum(J0 * J1, axis=0)
            JtJ_02 = mx.sum(J0 * J2, axis=0)
            JtJ_11 = mx.sum(J1 * J1, axis=0)
            JtJ_12 = mx.sum(J1 * J2, axis=0)
            JtJ_22 = mx.sum(J2 * J2, axis=0)

            # J'r elements
            Jtr_0 = mx.sum(J0 * residuals, axis=0)
            Jtr_1 = mx.sum(J1 * residuals, axis=0)
            Jtr_2 = mx.sum(J2 * residuals, axis=0)

            # Add regularization
            reg = 1e-6
            JtJ_00 = JtJ_00 + reg
            JtJ_11 = JtJ_11 + reg
            JtJ_22 = JtJ_22 + reg

            # 3x3 determinant
            det = (JtJ_00 * (JtJ_11 * JtJ_22 - JtJ_12 * JtJ_12)
                   - JtJ_01 * (JtJ_01 * JtJ_22 - JtJ_12 * JtJ_02)
                   + JtJ_02 * (JtJ_01 * JtJ_12 - JtJ_11 * JtJ_02)) + 1e-10

            # Cofactors
            c00 = JtJ_11 * JtJ_22 - JtJ_12 * JtJ_12
            c01 = -(JtJ_01 * JtJ_22 - JtJ_02 * JtJ_12)
            c02 = JtJ_01 * JtJ_12 - JtJ_02 * JtJ_11
            c11 = JtJ_00 * JtJ_22 - JtJ_02 * JtJ_02
            c12 = -(JtJ_00 * JtJ_12 - JtJ_02 * JtJ_01)
            c22 = JtJ_00 * JtJ_11 - JtJ_01 * JtJ_01

            delta_0 = (c00 * Jtr_0 + c01 * Jtr_1 + c02 * Jtr_2) / det
            delta_1 = (c01 * Jtr_0 + c11 * Jtr_1 + c12 * Jtr_2) / det
            delta_2 = (c02 * Jtr_0 + c12 * Jtr_1 + c22 * Jtr_2) / det

            delta = mx.stack([delta_0, delta_1, delta_2], axis=0)

        else:
            # General n_comp case - use NumPy batch solve
            J = A[None, :, :] * dPhi_dc  # (n_energy, n_comp, n_spectra)

            JtJ = mx.einsum('ejk,elk->jlk', J, J)  # (n_comp, n_comp, n_spectra)
            JtR = mx.einsum('ejk,ek->jk', J, residuals)  # (n_comp, n_spectra)

            # Add regularization
            reg = 1e-6
            eye = mx.eye(n_comp)[:, :, None]
            JtJ = JtJ + reg * eye

            # Batch solve via NumPy
            mx.eval(JtJ, JtR)
            JtJ_np = np.array(JtJ).transpose(2, 0, 1)  # (n_spectra, n_comp, n_comp)
            JtR_np = np.array(JtR).T[:, :, np.newaxis]  # (n_spectra, n_comp, 1)

            try:
                delta_np = np.linalg.solve(JtJ_np, JtR_np)[:, :, 0]  # (n_spectra, n_comp)
            except np.linalg.LinAlgError:
                delta_np = np.zeros((n_spectra, n_comp), dtype=np.float32)
                for k in range(n_spectra):
                    try:
                        delta_np[k] = np.linalg.solve(JtJ_np[k], JtR_np[k, :, 0])
                    except np.linalg.LinAlgError:
                        pass

            delta = mx.array(delta_np.T.astype(np.float32))

        # Damped update with step clipping
        delta = config.damping * delta
        delta = mx.clip(delta, -0.3, 0.3)

        # Update centers (need to convert to numpy for next iteration)
        mx.eval(delta)
        delta_np = np.array(delta)
        centers_curr = centers_curr + delta_np

        # Apply bounds
        delta_from_init = centers_curr - centers_init
        delta_clipped = np.clip(delta_from_init, config.bounds_center[0], config.bounds_center[1])
        centers_curr = centers_init + delta_clipped

    # Final evaluation
    Phi = voigt_basis_batch_mlx(energy, centers_curr, sigmas_np, gamma)
    A = solve_amplitudes_batch_mlx(Phi, Y)
    chi2 = compute_residuals_batch_mlx(Phi, A, Y)

    centers_out = mx.array(centers_curr)
    mx.eval(centers_out, A, chi2, n_iter_tracker)

    return centers_out, A, chi2, n_iter_tracker


def _set_row(arr, idx, values):
    """Helper to set a row in MLX array (workaround for no item assignment)."""
    # Convert to numpy, modify, convert back
    arr_np = np.array(arr)
    arr_np[idx] = np.array(values)
    return mx.array(arr_np)


class Stage2MLXRefiner:
    """
    MLX-accelerated Stage 2 refiner.

    Usage:
        refiner = Stage2MLXRefiner(config=Stage2MLXConfig(mode='light'))
        result = refiner.refine_batch(Y_anomaly, energy, centers0, sigmas0, gammas0, indices)
    """

    def __init__(self, config: Stage2MLXConfig | None = None):
        self.config = config or Stage2MLXConfig()

        if not HAS_MLX:
            raise ImportError("MLX not available for Stage2MLXRefiner")

        # Initialize Faddeeva table
        _ = get_faddeeva_table()

    def refine_batch(
        self,
        Y_anomaly: np.ndarray,
        energy: np.ndarray,
        centers0: np.ndarray,
        sigmas0: np.ndarray,
        gammas0: np.ndarray,
        anomaly_indices: np.ndarray,
    ) -> Stage2MLXResult:
        """
        Refine batch of anomalous spectra using MLX.

        Args:
            Y_anomaly: Anomalous spectra (n_energy, n_anomaly)
            energy: Energy axis (n_energy,)
            centers0: Initial centers (n_comp,)
            sigmas0: Initial sigmas (n_comp,)
            gammas0: Initial gammas (n_comp,)
            anomaly_indices: Original indices

        Returns:
            Stage2MLXResult
        """
        n_energy, n_anomaly = Y_anomaly.shape
        n_comp = len(centers0)

        # Convert to MLX
        Y_mx = mx.array(Y_anomaly.astype(np.float32))
        energy_mx = mx.array(energy.astype(np.float32))

        # Broadcast initial parameters to all spectra
        centers = mx.array(np.tile(centers0[:, np.newaxis], (1, n_anomaly)).astype(np.float32))
        sigmas = sigmas0.astype(np.float32)
        gamma = float(gammas0[0])  # Assume shared gamma

        # Run refinement based on mode
        if self.config.mode == "light":
            centers_out, A_out, chi2_out, n_iter_out = refine_center_batch_mlx(
                Y_mx, energy_mx, centers, sigmas, gamma, self.config
            )
            sigmas_out = mx.array(np.tile(sigmas[:, np.newaxis], (1, n_anomaly)))
            gammas_out = mx.array(np.full((n_comp, n_anomaly), gamma, dtype=np.float32))

        else:
            # For medium/full modes, fall back to simpler implementation
            # (or extend refine_center_batch_mlx to handle sigma/gamma)
            centers_out, A_out, chi2_out, n_iter_out = refine_center_batch_mlx(
                Y_mx, energy_mx, centers, sigmas, gamma, self.config
            )
            sigmas_out = mx.array(np.tile(sigmas[:, np.newaxis], (1, n_anomaly)))
            gammas_out = mx.array(np.full((n_comp, n_anomaly), gamma, dtype=np.float32))

        # Evaluate and convert to numpy
        mx.eval(centers_out, sigmas_out, gammas_out, A_out, chi2_out, n_iter_out)

        return Stage2MLXResult(
            centers=np.array(centers_out),
            sigmas=np.array(sigmas_out),
            gammas=np.array(gammas_out),
            amplitudes=np.array(A_out),
            chi2=np.array(chi2_out),
            converged=np.array(n_iter_out) < self.config.max_iter,
            n_iter=np.array(n_iter_out),
            indices=anomaly_indices,
        )


def benchmark_mlx_stage2():
    """Benchmark MLX Stage 2 performance."""
    print("=" * 60)
    print("MLX Stage 2 Benchmark")
    print("=" * 60)

    if not HAS_MLX:
        print("MLX not available")
        return

    np.random.seed(42)

    # Setup
    n_energy = 151
    energy = np.linspace(95, 110, n_energy).astype(np.float32)

    # True parameters
    true_centers = np.array([99.5, 103.0], dtype=np.float32)
    true_sigmas = np.array([0.6, 0.8], dtype=np.float32)
    true_gamma = 0.25
    true_amps = np.array([1000.0, 600.0], dtype=np.float32)

    from .faddeeva_mlx import voigt_profile_numpy

    # Generate test data with varying center shifts
    for n_spectra in [100, 1000, 10000]:
        print(f"\n--- {n_spectra} spectra ---")

        # Create spectra with random center shifts
        center_shifts = np.random.uniform(-0.5, 0.5, n_spectra).astype(np.float32)

        Y = np.zeros((n_energy, n_spectra), dtype=np.float32)
        for i in range(n_spectra):
            for c, s, a in zip(true_centers + center_shifts[i], true_sigmas, true_amps):
                Y[:, i] += a * voigt_profile_numpy(energy, c, s, true_gamma)

        # Add noise
        noise_level = Y.max() / 100
        Y += noise_level * np.random.randn(n_energy, n_spectra).astype(np.float32)

        # Initialize refiner
        config = Stage2MLXConfig(mode="light", max_iter=30)
        refiner = Stage2MLXRefiner(config)

        # Warm up
        _ = refiner.refine_batch(
            Y[:, :10], energy, true_centers, true_sigmas,
            np.array([true_gamma, true_gamma]), np.arange(10)
        )

        # Benchmark
        t0 = time.perf_counter()
        result = refiner.refine_batch(
            Y, energy, true_centers, true_sigmas,
            np.array([true_gamma, true_gamma]), np.arange(n_spectra)
        )
        elapsed = time.perf_counter() - t0

        # Analyze
        detected_shifts = result.centers[0, :] - true_centers[0]
        shift_error = detected_shifts - center_shifts
        rmse = np.sqrt(np.mean(shift_error**2))

        print(f"Time: {elapsed*1000:.1f}ms ({n_spectra/elapsed:.0f} spec/s)")
        print(f"Avg iter: {result.n_iter.mean():.1f}")
        print(f"Shift RMSE: {rmse:.4f} eV")
        print(f"Final chi2: {result.chi2.mean():.2f}")


if __name__ == "__main__":
    benchmark_mlx_stage2()
