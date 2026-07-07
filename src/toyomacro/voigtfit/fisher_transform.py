"""
Fisher Coordinate Transform for Voigt Parameters
=================================================

Transforms Voigt profile deviation parameters (δE, δσ, δγ) from Euclidean
parameter space to Fisher information coordinates where:

    ||ξ₁ - ξ₂||² ≈ KL(p_θ₁ || p_θ₂)

This makes Euclidean distance in the transformed space equal to the
information-theoretic distance between spectra.

The transform uses the Cholesky decomposition of the Fisher Information
Matrix: ξ = L · θ, where g = L · Lᵀ.

Key properties:
    - σ direction stretched (g_σσ > g_EE for C 1s) → finer σ quantization
    - σ-γ correlation decorrelated → no correlated encoder error
    - Amplitude excluded (Fisher null space, ~4 orders smaller than g_EE)

dev-log 74 additions:
    - FisherWhitenedTransform: g^{1/2} symmetric factorization
    - bbox_aspect_ratio(): diagnostic for Hilbert grid isotropy

Reference:
    dev-log 61-62 (Fisher information geometry)
    dev-log 72 (δσ correlated error discovery)
    dev-log 74 (neighbor preservation improvement)
"""

from dataclasses import dataclass

import numpy as np

from .fisher_information import (
    compute_fisher_matrix,
    eta_from_sigma_gamma,
    sigma_gamma_from_eta,
)


@dataclass
class FisherTransformInfo:
    """Diagnostic information about the Fisher coordinate transform."""
    g: np.ndarray           # 3×3 Fisher sub-matrix for (δE, δσ, δγ)
    L: np.ndarray           # Cholesky factor
    L_inv: np.ndarray       # Inverse of L
    sigma: float
    gamma: float
    eta: float
    scale_factors: np.ndarray  # Diagonal of L
    condition_number: float


class FisherCoordinateTransform:
    """Transform Voigt deviation parameters to Fisher information coordinates.

    Converts (δE, δσ, δγ) → (ξ₁, ξ₂, ξ₃) where Euclidean distance
    in ξ-space approximates KL divergence between spectra.

    Uses the 4D Fisher Information Matrix computed at nominal parameters,
    with the amplitude row/column removed (amplitude is in the Fisher
    null space — g_amp ≈ 0.02 vs g_EE ≈ 90,000).

    Args:
        sigma: Nominal Gaussian width (eV). Mutually exclusive with eta.
        gamma: Nominal Lorentzian half-width (eV). Mutually exclusive with eta.
        eta: Voigt mixing ratio. If provided, (sigma, gamma) computed via
             Thompson approximation with fwhm_total.
        fwhm_total: Total Voigt FWHM (eV), used with eta. Default: 1.0
        amplitude: Nominal peak amplitude for Fisher computation.
        center: Nominal peak center (eV). Default: 284.4 (C 1s)
        energy: Energy axis. Default: center ± 5 eV, 201 points.
    """

    def __init__(
        self,
        sigma: float | None = None,
        gamma: float | None = None,
        eta: float | None = None,
        fwhm_total: float = 1.0,
        amplitude: float = 1000.0,
        center: float = 284.4,
        energy: np.ndarray | None = None,
    ):
        if eta is not None:
            self._sigma, self._gamma = sigma_gamma_from_eta(eta, fwhm_total)
            self._eta = eta
        elif sigma is not None and gamma is not None:
            self._sigma = sigma
            self._gamma = gamma
            self._eta = eta_from_sigma_gamma(sigma, gamma)
        else:
            raise ValueError("Provide either eta or (sigma, gamma)")

        if energy is None:
            energy = np.linspace(center - 5, center + 5, 201)

        # 4D Fisher matrix: (amplitude, δE, δσ, δγ)
        result = compute_fisher_matrix(
            amplitude, center, self._sigma, self._gamma, energy, mode='4d'
        )

        # 3×3 sub-block for (δE, δσ, δγ) — exclude amplitude (index 0)
        self._g = result.g[1:4, 1:4].copy()

        # Cholesky: g = L Lᵀ
        self._L = np.linalg.cholesky(self._g)
        self._L_inv = np.linalg.inv(self._L)

        eigs = np.linalg.eigvalsh(self._g)
        self._condition_number = eigs[-1] / max(eigs[0], 1e-30)

    @property
    def info(self) -> FisherTransformInfo:
        return FisherTransformInfo(
            g=self._g.copy(),
            L=self._L.copy(),
            L_inv=self._L_inv.copy(),
            sigma=self._sigma,
            gamma=self._gamma,
            eta=self._eta,
            scale_factors=np.diag(self._L).copy(),
            condition_number=self._condition_number,
        )

    def to_fisher(self, theta: np.ndarray) -> np.ndarray:
        """(δE, δσ, δγ) → Fisher coordinates.

        Args:
            theta: (3,) or (N, 3) deviations
        Returns:
            xi: same shape, Fisher coordinates
        """
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 1:
            return self._L @ theta
        return (self._L @ theta.T).T

    def from_fisher(self, xi: np.ndarray) -> np.ndarray:
        """Fisher coordinates → (δE, δσ, δγ).

        Args:
            xi: (3,) or (N, 3) Fisher coordinates
        Returns:
            theta: same shape, parameter deviations
        """
        xi = np.asarray(xi, dtype=np.float64)
        if xi.ndim == 1:
            return self._L_inv @ xi
        return (self._L_inv @ xi.T).T

    def fisher_distance(self, theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
        """Information distance between parameter pairs.

        Args:
            theta1, theta2: (3,) or (N, 3)
        Returns:
            scalar or (N,) distances
        """
        diff = self.to_fisher(theta1) - self.to_fisher(theta2)
        if diff.ndim == 1:
            return float(np.sqrt(np.sum(diff ** 2)))
        return np.sqrt(np.sum(diff ** 2, axis=1))

    def fisher_range(
        self,
        dE_range: tuple[float, float],
        dsigma_range: tuple[float, float],
        dgamma_range: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute bounding box of parameter ranges in Fisher space.

        Transforms all 8 corners of the parameter box to Fisher space.

        Returns:
            (xi_min, xi_max): each (3,) arrays
        """
        corners = np.array([
            [dE, ds, dg]
            for dE in dE_range
            for ds in dsigma_range
            for dg in dgamma_range
        ])
        xi = self.to_fisher(corners)
        return xi.min(axis=0), xi.max(axis=0)

    def bbox_aspect_ratio(
        self,
        dE_range: tuple[float, float],
        dsigma_range: tuple[float, float],
        dgamma_range: tuple[float, float],
    ) -> float:
        """Aspect ratio of the bounding box in Fisher space.

        Ratio of longest to shortest axis. Lower is better for Hilbert
        curve locality (1.0 = cubic = optimal).
        """
        xi_min, xi_max = self.fisher_range(dE_range, dsigma_range, dgamma_range)
        ranges = xi_max - xi_min
        return float(ranges.max() / max(ranges.min(), 1e-30))


class FisherWhitenedTransform(FisherCoordinateTransform):
    """Fisher transform using symmetric positive square root g^{1/2}.

    Uses W = V diag(√λ) Vᵀ instead of Cholesky L, where (λ, V) are
    eigenvalues/vectors of the 3×3 Fisher sub-matrix.

    Properties:
        - W is symmetric (W = Wᵀ) — no preferred triangular structure
        - Wᵀ W = g — same Fisher metric as Cholesky
        - Fisher distances are preserved exactly
        - Bounding box shape may differ from Cholesky (better or worse
          depending on parameter ranges)

    The whitened transform rotates to the Fisher eigenbasis, scales by
    √λ, then rotates back. This distributes the Fisher scaling more
    evenly across all axes compared to Cholesky's sequential structure.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Eigendecomposition of Fisher sub-matrix
        evals, evecs = np.linalg.eigh(self._g)
        sqrt_eig = np.sqrt(np.maximum(evals, 1e-30))
        # Override L with symmetric g^{1/2} = V sqrt(Λ) Vᵀ
        self._L = evecs @ np.diag(sqrt_eig) @ evecs.T
        # g^{-1/2} = V 1/sqrt(Λ) Vᵀ
        self._L_inv = evecs @ np.diag(1.0 / sqrt_eig) @ evecs.T
        # Diagnostics
        self._whitened_eigenvalues = evals
        self._whitened_eigenvectors = evecs

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the Fisher sub-matrix (ascending)."""
        return self._whitened_eigenvalues.copy()

    @property
    def eigenvectors(self) -> np.ndarray:
        """Eigenvectors of the Fisher sub-matrix (columns)."""
        return self._whitened_eigenvectors.copy()
