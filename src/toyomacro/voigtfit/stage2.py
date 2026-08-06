"""
Stage 2: Batch Refinement with Gauss-Newton
===========================================

Batch processing of anomalous spectra using analytical Jacobian.

Features:
- Three refinement modes: LIGHT (center), MEDIUM (center+σ), FULL (center+σ+γ)
- GPU-friendly batch processing
- Adaptive mode selection based on anomaly characteristics

Performance target: 50-100K spectra/s
"""

from dataclasses import dataclass

import numpy as np

from .gauss_newton import GaussNewtonRefiner, RefineResult
from .voigt_jacobian import voigt_jacobian_batch, voigt_profile


@dataclass
class Stage2Config:
    """Configuration for Stage 2 refinement."""
    mode: str = "medium"           # light, medium, full
    max_iter: int = 15             # Max Gauss-Newton iterations per spectrum
    damping: float = 0.7           # Step damping factor
    tol: float = 1e-5              # Convergence tolerance
    bounds_center: tuple[float, float] = (-2.0, 2.0)  # Center shift bounds
    bounds_sigma: tuple[float, float] = (0.1, 5.0)    # Sigma bounds
    bounds_gamma: tuple[float, float] = (0.01, 2.0)   # Gamma bounds
    parallel_batch_size: int = 1000  # Batch size for parallel processing


@dataclass
class Stage2Result:
    """Result of Stage 2 batch refinement."""
    centers: np.ndarray       # (n_comp, n_spectra)
    sigmas: np.ndarray        # (n_comp, n_spectra)
    gammas: np.ndarray        # (n_comp, n_spectra)
    amplitudes: np.ndarray    # (n_comp, n_spectra)
    chi2: np.ndarray          # (n_spectra,)
    converged: np.ndarray     # (n_spectra,) bool
    n_iter: np.ndarray        # (n_spectra,) int
    indices: np.ndarray       # Original indices in full dataset


class Stage2Refiner:
    """
    Stage 2 batch refiner using Gauss-Newton optimization.

    Processes anomalous spectra identified by Stage 1, refining
    non-linear parameters (center, σ, γ) while analytically solving
    for amplitudes.

    Usage:
        refiner = Stage2Refiner(config=Stage2Config(mode='medium'))
        result = refiner.refine_batch(
            Y_anomaly=spectra[:, anomaly_mask],
            energy=energy_axis,
            centers0=np.array([707.0, 709.5]),
            sigmas0=np.array([0.8, 0.9]),
            gammas0=np.array([0.2, 0.2]),
            anomaly_indices=np.where(anomaly_mask)[0]
        )
    """

    def __init__(self, config: Stage2Config | None = None):
        """Initialize Stage 2 refiner."""
        self.config = config or Stage2Config()
        self._refiner = GaussNewtonRefiner(
            mode=self.config.mode,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            damping=self.config.damping,
            bounds_center=self.config.bounds_center,
            bounds_sigma=self.config.bounds_sigma,
            bounds_gamma=self.config.bounds_gamma,
        )

    def refine_single(
        self,
        y: np.ndarray,
        energy: np.ndarray,
        centers0: np.ndarray,
        sigmas0: np.ndarray,
        gammas0: np.ndarray,
    ) -> RefineResult:
        """Refine single spectrum."""
        return self._refiner.refine(y, energy, centers0, sigmas0, gammas0)

    def refine_batch(
        self,
        Y_anomaly: np.ndarray,
        energy: np.ndarray,
        centers0: np.ndarray,
        sigmas0: np.ndarray,
        gammas0: np.ndarray,
        anomaly_indices: np.ndarray,
    ) -> Stage2Result:
        """
        Refine batch of anomalous spectra.

        Currently uses sequential processing per spectrum.
        Future: GPU batch parallelization.

        Args:
            Y_anomaly: Anomalous spectra (n_energy, n_anomaly)
            energy: Energy axis (n_energy,)
            centers0: Initial centers (n_comp,) - shared for all
            sigmas0: Initial sigmas (n_comp,) - shared for all
            gammas0: Initial gammas (n_comp,) - shared for all
            anomaly_indices: Original indices of anomalous spectra

        Returns:
            Stage2Result with refined parameters for all anomalous spectra
        """
        n_energy, n_anomaly = Y_anomaly.shape
        n_comp = len(centers0)

        # Output arrays
        centers_all = np.zeros((n_comp, n_anomaly))
        sigmas_all = np.zeros((n_comp, n_anomaly))
        gammas_all = np.zeros((n_comp, n_anomaly))
        amplitudes_all = np.zeros((n_comp, n_anomaly))
        chi2_all = np.zeros(n_anomaly)
        converged_all = np.zeros(n_anomaly, dtype=bool)
        n_iter_all = np.zeros(n_anomaly, dtype=int)

        # Process each spectrum
        for i in range(n_anomaly):
            y = Y_anomaly[:, i]
            result = self.refine_single(y, energy, centers0, sigmas0, gammas0)

            centers_all[:, i] = result.centers
            sigmas_all[:, i] = result.sigmas
            gammas_all[:, i] = result.gammas
            amplitudes_all[:, i] = result.amplitudes
            chi2_all[i] = result.chi2
            converged_all[i] = result.converged
            n_iter_all[i] = result.n_iter

        return Stage2Result(
            centers=centers_all,
            sigmas=sigmas_all,
            gammas=gammas_all,
            amplitudes=amplitudes_all,
            chi2=chi2_all,
            converged=converged_all,
            n_iter=n_iter_all,
            indices=anomaly_indices,
        )

    def refine_batch_vectorized(
        self,
        Y_anomaly: np.ndarray,
        energy: np.ndarray,
        centers0: np.ndarray,
        sigmas0: np.ndarray,
        gammas0: np.ndarray,
        anomaly_indices: np.ndarray,
    ) -> Stage2Result:
        """
        Vectorized batch refinement (experimental).

        All spectra share the same initial parameters and are updated
        simultaneously. This is faster but may be less accurate for
        spectra with very different optimal parameters.

        Best for: Uniform charging shift correction (LIGHT mode)
        """
        n_energy, n_anomaly = Y_anomaly.shape
        n_comp = len(centers0)

        # Broadcast initial parameters to all spectra
        centers = np.tile(centers0[:, np.newaxis], (1, n_anomaly))
        sigmas = np.tile(sigmas0[:, np.newaxis], (1, n_anomaly))
        gammas = np.tile(gammas0[:, np.newaxis], (1, n_anomaly))

        prev_residual_norm = np.full(n_anomaly, np.inf)

        for iteration in range(self.config.max_iter):
            # Compute basis and Jacobian for all spectra
            # (Using shared parameters for simplicity - could be per-spectrum)
            Phi, J_c, J_sigma, J_gamma = voigt_jacobian_batch(
                energy, centers[:, 0], sigmas[:, 0], gammas[:, 0]
            )

            # Solve for amplitudes: A = (Φ'Φ)⁻¹ Φ' Y
            PhiTPhi = Phi.T @ Phi + 1e-8 * np.eye(n_comp)
            PhiTY = Phi.T @ Y_anomaly
            amplitudes = np.linalg.solve(PhiTPhi, PhiTY)

            # Compute residuals for all spectra
            Y_fit = Phi @ amplitudes
            residuals = Y_anomaly - Y_fit
            residual_norms = np.linalg.norm(residuals, axis=0)

            # Check convergence (global). Skipped on the first pass for
            # the same reason as the scalar path in gauss_newton.py: the
            # inf sentinel would make every ratio nan.
            if iteration > 0:
                rel_change = (
                    np.abs(prev_residual_norm - residual_norms)
                    / (prev_residual_norm + 1e-10)
                )
                if np.median(rel_change) < self.config.tol:
                    break
            prev_residual_norm = residual_norms

            # Build Jacobian and compute updates for each spectrum
            # Simplified: use average update direction
            if self.config.mode == "light":
                # Average center shift across all spectra
                J = J_c  # (n_energy, n_comp)
                JtJ = J.T @ J + 1e-8 * np.eye(n_comp)
                JtR = J.T @ residuals  # (n_comp, n_anomaly)
                delta_c = np.linalg.solve(JtJ, JtR)  # (n_comp, n_anomaly)
                centers = centers + self.config.damping * delta_c

                # Apply bounds
                delta_from_init = centers - centers0[:, np.newaxis]
                delta_clipped = np.clip(delta_from_init, *self.config.bounds_center)
                centers = centers0[:, np.newaxis] + delta_clipped

        # Final chi2
        Phi, _, _, _ = voigt_jacobian_batch(energy, centers[:, 0], sigmas[:, 0], gammas[:, 0])
        PhiTPhi = Phi.T @ Phi + 1e-8 * np.eye(n_comp)
        PhiTY = Phi.T @ Y_anomaly
        amplitudes = np.linalg.solve(PhiTPhi, PhiTY)
        Y_fit = Phi @ amplitudes
        residuals = Y_anomaly - Y_fit
        chi2 = np.sum(residuals**2, axis=0) / n_energy

        return Stage2Result(
            centers=centers,
            sigmas=sigmas,
            gammas=gammas,
            amplitudes=amplitudes,
            chi2=chi2,
            converged=np.ones(n_anomaly, dtype=bool),
            n_iter=np.full(n_anomaly, iteration + 1),
            indices=anomaly_indices,
        )


def select_mode_adaptive(
    chi2_anomaly: np.ndarray,
    chi2_normal_median: float,
    spatial_variance: float | None = None,
) -> str:
    """
    Adaptively select refinement mode based on anomaly characteristics.

    Heuristics:
    - If anomaly chi2 is only slightly elevated: LIGHT (charging shift)
    - If anomaly chi2 is moderately elevated: MEDIUM (instrument variation)
    - If anomaly chi2 is very high: FULL (unknown peaks)

    Args:
        chi2_anomaly: Chi2 values of anomalous spectra
        chi2_normal_median: Median chi2 of normal spectra
        spatial_variance: Variance of anomaly positions (optional)

    Returns:
        Recommended mode: 'light', 'medium', or 'full'
    """
    chi2_ratio = np.median(chi2_anomaly) / chi2_normal_median

    if chi2_ratio < 2.0:
        return "light"
    elif chi2_ratio < 5.0:
        return "medium"
    else:
        return "full"


def demo_stage2_batch():
    """Demonstration of Stage 2 batch processing."""
    import time

    print("=" * 60)
    print("Stage 2 Batch Processing Demo")
    print("=" * 60)

    np.random.seed(42)

    # Setup
    n_energy = 151
    n_spectra = 1000
    n_anomaly = 50  # 5% anomalies
    energy = np.linspace(95, 110, n_energy)

    # True parameters
    true_centers = np.array([99.5, 103.0])
    true_sigmas = np.array([0.6, 0.8])
    true_gammas = np.array([0.25, 0.30])
    true_amps = np.array([1000.0, 600.0])

    # Generate normal spectra
    Y_clean = np.zeros((n_energy, n_spectra))
    for c, s, g, a in zip(true_centers, true_sigmas, true_gammas, true_amps):
        for i in range(n_spectra):
            Y_clean[:, i] += a * voigt_profile(energy, c, s, g)

    # Add noise (SNR ~100)
    noise_level = Y_clean.max() / 100
    Y = Y_clean + noise_level * np.random.randn(n_energy, n_spectra)

    # Create anomalies (shifted centers)
    anomaly_indices = np.random.choice(n_spectra, n_anomaly, replace=False)
    center_shifts = np.random.uniform(-0.5, 0.5, n_anomaly)

    for i, (idx, shift) in enumerate(zip(anomaly_indices, center_shifts)):
        Y[:, idx] = 0
        for c, s, g, a in zip(true_centers + shift, true_sigmas, true_gammas, true_amps):
            Y[:, idx] += a * voigt_profile(energy, c, s, g)
        Y[:, idx] += noise_level * np.random.randn(n_energy)

    Y_anomaly = Y[:, anomaly_indices]

    print(f"\nDataset: {n_spectra} spectra, {n_anomaly} anomalies ({100*n_anomaly/n_spectra:.1f}%)")
    print(f"True center shifts: {center_shifts.min():.2f} to {center_shifts.max():.2f} eV")

    # Test different modes
    for mode in ["light", "medium"]:
        print(f"\n--- Mode: {mode.upper()} ---")

        config = Stage2Config(mode=mode, max_iter=15)
        refiner = Stage2Refiner(config)

        t0 = time.perf_counter()
        result = refiner.refine_batch(
            Y_anomaly, energy,
            true_centers, true_sigmas, true_gammas,
            anomaly_indices
        )
        elapsed = time.perf_counter() - t0

        # Analyze results
        detected_shifts = result.centers[0, :] - true_centers[0]
        shift_error = detected_shifts - center_shifts

        print(f"  Time: {elapsed:.3f}s ({n_anomaly/elapsed:.0f} spec/s)")
        print(f"  Converged: {result.converged.sum()}/{n_anomaly}")
        print(f"  Mean iterations: {result.n_iter.mean():.1f}")
        print(f"  Shift detection error: {np.abs(shift_error).mean():.4f} ± {np.std(shift_error):.4f} eV")
        print(f"  Chi2 reduction: {result.chi2.mean():.2f}")

    # Vectorized version (LIGHT mode only)
    print("\n--- Vectorized LIGHT mode ---")
    config = Stage2Config(mode="light", max_iter=15)
    refiner = Stage2Refiner(config)

    t0 = time.perf_counter()
    result_vec = refiner.refine_batch_vectorized(
        Y_anomaly, energy,
        true_centers, true_sigmas, true_gammas,
        anomaly_indices
    )
    elapsed = time.perf_counter() - t0

    detected_shifts = result_vec.centers[0, :] - true_centers[0]
    shift_error = detected_shifts - center_shifts

    print(f"  Time: {elapsed:.3f}s ({n_anomaly/elapsed:.0f} spec/s)")
    print(f"  Shift detection error: {np.abs(shift_error).mean():.4f} ± {np.std(shift_error):.4f} eV")

    return result


if __name__ == "__main__":
    demo_stage2_batch()
