"""Model-independent parts of the identifiability diagnostics (private).

What here does not depend on the lineshape: the Poisson Fisher matrix
from a Jacobian, the effective information of a parameter block with the
rest profiled out, the rank decision and covariance bound on a
unit-diagonal matrix, the thresholds of the labels, and backgrounds
that are linear in their coefficients.

``toyomacro.voigtfit.identifiability`` builds its Voigt diagnostics on
these and re-exports the public names, which is where they are
documented and supported (Experimental tier). This module sits outside
``voigtfit`` and imports nothing from the package, so a model that is
not a Voigt peak can use it without depending on the Voigt engine.

Everything here is NumPy float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Background",
    "EffectiveInformation",
    "IdentifiabilityThresholds",
    "constant_background",
    "covariance_bound",
    "effective_information",
    "linear_background",
    "null_directions",
    "poisson_fisher_matrix",
]


# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Background:
    """Background that is linear in its coefficients.

    ``counts = coefficients @ basis`` per channel at unit exposure. Terms
    flagged in ``estimated`` become parameters of the Fisher matrix; the
    others enter the Poisson mean only, i.e. they are treated as known.
    Backgrounds add: ``constant_background(...) + shirley_background(...)``.

    Attributes:
        basis: Shape functions, shape (n_terms, n_energy)
        coefficients: Shape (n_terms,)
        names: Parameter name per term
        estimated: Boolean per term
    """

    basis: np.ndarray
    coefficients: np.ndarray
    names: tuple[str, ...]
    estimated: tuple[bool, ...]

    def __post_init__(self) -> None:
        n = len(self.names)
        if self.basis.shape[0] != n or self.coefficients.shape != (n,) or len(self.estimated) != n:
            raise ValueError("basis, coefficients, names and estimated disagree on n_terms")

    def counts(self) -> np.ndarray:
        return self.coefficients @ self.basis

    def __add__(self, other: Background) -> Background:
        if self.basis.shape[1] != other.basis.shape[1]:
            raise ValueError("backgrounds are on different energy axes")
        return Background(
            basis=np.vstack([self.basis, other.basis]),
            coefficients=np.concatenate([self.coefficients, other.coefficients]),
            names=self.names + other.names,
            estimated=self.estimated + other.estimated,
        )


def constant_background(
    energy: np.ndarray, level: float, *, estimated: bool = True
) -> Background:
    """Flat background of ``level`` counts per channel at unit exposure."""
    energy = np.asarray(energy, dtype=np.float64)
    return Background(
        basis=np.ones((1, energy.size)),
        coefficients=np.array([float(level)]),
        names=("bg_level",),
        estimated=(estimated,),
    )


def linear_background(
    energy: np.ndarray, level: float, slope: float, *, estimated: bool = True
) -> Background:
    """``level + slope * (E - E_mid)`` with E_mid the window midpoint.

    Referring the slope to the midpoint only changes the coordinates of
    the background block; the information left for the peak parameters
    depends on the span of the basis, not on this choice.
    """
    energy = np.asarray(energy, dtype=np.float64)
    e_mid = 0.5 * (energy[0] + energy[-1])
    return Background(
        basis=np.vstack([np.ones(energy.size), energy - e_mid]),
        coefficients=np.array([float(level), float(slope)]),
        names=("bg_level", "bg_slope"),
        estimated=(estimated, estimated),
    )


# ---------------------------------------------------------------------------
# Fisher matrix, rank decision, effective information
# ---------------------------------------------------------------------------


def poisson_fisher_matrix(
    jacobian_unit: np.ndarray, mean_unit: np.ndarray, exposure: float
) -> np.ndarray:
    """``exposure * J^T diag(1/mu) J`` for Poisson counts.

    Args:
        jacobian_unit: d(expected counts)/d(parameters) at unit exposure,
            (n_energy, n_params)
        mean_unit: Expected counts per channel at unit exposure, all
            positive; the caller checks that
        exposure: Multiplier on the whole mean; the matrix is linear in it
    """
    weighted = jacobian_unit / np.sqrt(mean_unit)[:, np.newaxis]
    return exposure * (weighted.T @ weighted)


def null_directions(eigenvalues: np.ndarray) -> np.ndarray:
    """Mask of numerically null eigenvalues of a unit-diagonal matrix.

    The ``n * eps`` rule of ``np.linalg.matrix_rank``, as used by
    ``crlb.compute_multipeak_fisher``; re-implemented here, not imported.
    """
    n = eigenvalues.size
    return eigenvalues <= max(eigenvalues[-1] * n * np.finfo(np.float64).eps, 1e-300)


def covariance_bound(
    matrix: np.ndarray, projection_tol: float
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Inverse of a Fisher matrix with the rank decision of ``crlb``.

    Returns the pseudo-inverse taken on the unit-diagonal matrix (``n *
    eps`` rule), a mask of the axes that project onto its null space and
    whose bound is therefore infinite, the null-space dimension and the
    condition number of the unit-diagonal matrix. Entries of the
    pseudo-inverse on masked axes are not bounds and must not be read.

    ``projection_tol`` decides when an axis counts as projecting onto the
    null space. It is a calibrated constant, not a universal one: the
    caller passes the value it has justified for its own model
    (``crlb._NULL_PROJECTION_TOL`` for Voigt designs, with its evidence
    in that module).
    """
    d = np.sqrt(np.maximum(np.diag(matrix), 0.0))
    d_safe = np.where(d > 0.0, d, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix / np.outer(d_safe, d_safe))
    null = null_directions(eigenvalues)
    inverse = np.where(null, 0.0, 1.0 / np.where(null, 1.0, eigenvalues))
    covariance = (eigenvectors * inverse[np.newaxis, :]) @ eigenvectors.T
    covariance /= np.outer(d_safe, d_safe)
    unbounded = (eigenvectors[:, null] ** 2).sum(axis=1) > projection_tol
    positive = eigenvalues[eigenvalues > 0.0]
    condition = float(positive[-1] / positive[0]) if positive.size >= 2 and not null.any() else np.inf
    return covariance, unbounded, int(null.sum()), condition


@dataclass
class EffectiveInformation:
    """Information about a target block, with and without the rest known.

    Attributes:
        target_index: Positions of the target parameters in the full matrix
        effective: ``I_uu - I_uq pinv(I_qq) I_qu``: what is left for the
            targets ``u`` once the other parameters ``q`` are estimated
            from the same data. Where it is invertible its inverse is the
            target block of the full inverse Fisher matrix.
        conditional: ``I_uu``: the information if ``q`` were known
            exactly. ``conditional - effective`` is positive
            semi-definite, so this is never the smaller of the two; it is
            what ``fisher_information.analyze_sigma_gamma_axes`` reports.
        nuisance_null_dim: Number of numerically null directions in the
            nuisance block, by the ``n * eps`` rule on its correlation
            matrix. Non-zero means some combination of nuisance
            parameters is itself not estimable; ``effective`` is then
            taken with those directions dropped.
    """

    target_index: tuple[int, ...]
    effective: np.ndarray
    conditional: np.ndarray
    nuisance_null_dim: int


def effective_information(
    fisher: np.ndarray, target_index: tuple[int, ...] | list[int]
) -> EffectiveInformation:
    """Schur complement of the nuisance block, in a scale-free metric.

    The Fisher matrix mixes units (counts, eV, eV**2), so the nuisance
    block is inverted after rescaling the whole matrix to unit diagonal
    and the result is scaled back. The rank decision inside the
    pseudo-inverse therefore does not move with the units of any
    parameter -- the gauge ``crlb.compute_multipeak_fisher`` fixes, with
    the same ``n * eps`` rule (re-implemented here).

    Args:
        fisher: Symmetric positive semi-definite matrix, (n, n)
        target_index: Positions of the target parameters ``u``; every
            other parameter is a nuisance parameter ``q``

    Returns:
        EffectiveInformation
    """
    fisher = np.asarray(fisher, dtype=np.float64)
    n = fisher.shape[0]
    u = np.array(sorted(target_index), dtype=int)
    if u.size == 0 or u.size != len(set(target_index)) or u.min() < 0 or u.max() >= n:
        raise ValueError(f"invalid target_index {tuple(target_index)} for a {n}x{n} matrix")
    q = np.array([i for i in range(n) if i not in set(u.tolist())], dtype=int)

    conditional = fisher[np.ix_(u, u)].copy()
    if q.size == 0:
        return EffectiveInformation(tuple(u.tolist()), conditional.copy(), conditional, 0)

    d = np.sqrt(np.maximum(np.diag(fisher), 0.0))
    d_safe = np.where(d > 0.0, d, 1.0)
    scaled = fisher / np.outer(d_safe, d_safe)

    eigenvalues, eigenvectors = np.linalg.eigh(scaled[np.ix_(q, q)])
    null = null_directions(eigenvalues)
    inverse = np.where(null, 0.0, 1.0 / np.where(null, 1.0, eigenvalues))
    c_uq = scaled[np.ix_(u, q)] @ eigenvectors
    schur = scaled[np.ix_(u, u)] - (c_uq * inverse[np.newaxis, :]) @ c_uq.T
    schur = 0.5 * (schur + schur.T)

    return EffectiveInformation(
        target_index=tuple(u.tolist()),
        effective=schur * np.outer(d_safe[u], d_safe[u]),
        conditional=conditional,
        nuisance_null_dim=int(null.sum()),
    )


# ---------------------------------------------------------------------------
# Label thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentifiabilityThresholds:
    """Where 'identified' ends. Conventions, not physics.

    Attributes:
        weak_relative_sd: A parameter is 'weakly_identified' when its
            standard-deviation bound exceeds this fraction of its
            reference scale. The default 0.1 means "a tenth of the total
            FWHM" for Voigt positions and widths: a working rule of thumb
            for when a width has stopped being a usable number, with no
            deeper justification. Change it to suit the question.
        boundary_sd: A parameter with a lower limit (the Gaussian
            variance of a Voigt peak) is 'near_boundary' when it lies
            within this many standard-deviation bounds of that limit. For
            an unconstrained, normally distributed estimator the chance of
            falling below the limit would be 0.13 % at 3 and 2.3 % at 2
            -- but that normal approximation is exactly what fails near a
            boundary (Self & Liang 1987), so read the default 3 as a
            margin of caution, not as a probability.
    """

    weak_relative_sd: float = 0.1
    boundary_sd: float = 3.0

    def __post_init__(self) -> None:
        if not (self.weak_relative_sd > 0.0 and self.boundary_sd > 0.0):
            raise ValueError("thresholds must be positive")
