"""Model-independent parts of the identifiability diagnostics (private).

What here does not depend on the lineshape: the Poisson Fisher matrix
from a Jacobian, the effective information of a parameter block with the
rest profiled out, the rank decision and covariance bound on a
unit-diagonal matrix, the thresholds of the labels, backgrounds that are
linear in their coefficients, and the modes of a parameter block (its
least determined directions, with the rest profiled out).

``toyomacro.voigtfit.identifiability`` builds its Voigt diagnostics on
these and re-exports the public names, which is where they are
documented and supported (Experimental tier). This module sits outside
``voigtfit`` and imports nothing from the package, so a model that is
not a Voigt peak can use it without depending on the Voigt engine.

Everything here is NumPy float64.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Background",
    "EffectiveInformation",
    "IdentifiabilityThresholds",
    "Mode",
    "ModeDecomposition",
    "constant_background",
    "covariance_bound",
    "effective_information",
    "linear_background",
    "mode_decomposition",
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


# ---------------------------------------------------------------------------
# Modes of the target block
# ---------------------------------------------------------------------------

# Components within this fraction of the largest one tie for the sign.
_SIGN_TIE = 1e-6


@dataclass(frozen=True)
class Mode:
    """One direction in the target block, in scaled coordinates.

    Attributes:
        index: Position in the report, 0 for the softest
        relative_sd: ``sqrt(lambda)``, the standard-deviation bound along
            the direction in units of the reference scales; ``inf`` for a
            null mode
        status: 'rank_deficient' (null mode), 'weakly_identified' when
            ``relative_sd`` exceeds the threshold, 'not_resolved' where
            rounding leaves no positive bound (``relative_sd`` nan), else
            'identified'
        weights: Target parameter name -> component of the unit vector
            ``w`` in scaled coordinates (each parameter divided by its
            reference scale). The largest component is positive.
        background_shift: Name of each parameter outside the target block
            (the background coefficients, for the Voigt entry) -> its
            change, in its own units, when the targets move one SD along
            ``w`` and those parameters are re-estimated. For a null mode
            the move is one unit of the scaled coordinates instead.
        signature: Change of the expected counts per channel for that
            move, at the stated exposure, shape (n_energy,)
        whitened_norm: ``sqrt(sum(signature**2 / mu))``. 1 for every
            non-null mode in exact arithmetic -- a one-SD move is one unit
            of counting noise, whatever the scaling -- and numerically 0
            for a null one; a departure from 1 measures the mode's own
            numerical error
        shares: Every parameter name -> corr**2 between that parameter and
            the mode coordinate ``w^T S^-1 theta``, i.e. the share of its
            variance bound lying along the mode. None for a parameter
            whose bound is infinite, and for every parameter of a null mode
        eigenvalue_ratio: ``lambda`` over the next mode's; None for the
            last non-null mode and for null modes. Close to 1 means only
            the plane of the two is determined, not the vectors.
    """

    index: int
    relative_sd: float
    status: str
    weights: dict[str, float]
    background_shift: dict[str, float]
    signature: np.ndarray = field(repr=False, compare=False)
    whitened_norm: float
    shares: dict[str, float | None]
    eigenvalue_ratio: float | None


@dataclass(frozen=True)
class ModeDecomposition:
    """Result of ``mode_decomposition``.

    Attributes:
        modes: Up to ``k`` non-null modes, softest first
        null_modes: An orthonormal basis of the target block's null space,
            in scaled coordinates. Only the span means anything; the
            vectors chosen inside it are arbitrary.
        null_space_dim: Null directions of the full matrix
        nuisance_null_dim: Null directions of the full matrix with no
            component on the targets (by ``projection_tol``);
            ``null_space_dim - nuisance_null_dim`` is the number of null
            modes
        target_names: Names of the target parameters, in order
        scales: Their reference scales
    """

    modes: tuple[Mode, ...]
    null_modes: tuple[Mode, ...]
    null_space_dim: int
    nuisance_null_dim: int
    target_names: tuple[str, ...]
    scales: tuple[float, ...]


def _unit_diagonal(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.sqrt(np.maximum(np.diag(matrix), 0.0))
    d_safe = np.where(d > 0.0, d, 1.0)
    return 0.5 * (matrix + matrix.T) / np.outer(d_safe, d_safe), d_safe


def _pinv_unit_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Pseudo-inverse with the n*eps rank rule on the unit-diagonal matrix."""
    scaled, d = _unit_diagonal(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(scaled)
    null = null_directions(eigenvalues)
    inverse = np.where(null, 0.0, 1.0 / np.where(null, 1.0, eigenvalues))
    return ((eigenvectors * inverse[np.newaxis, :]) @ eigenvectors.T) / np.outer(d, d)


def _oriented(vector: np.ndarray) -> np.ndarray:
    """Sign convention: the largest-magnitude component is positive; within
    ``_SIGN_TIE`` of the largest, relative, the first in order decides."""
    magnitude = np.abs(vector)
    first = int(np.flatnonzero(magnitude >= (1.0 - _SIGN_TIE) * magnitude.max())[0])
    return vector if vector[first] > 0.0 else -vector


def mode_decomposition(
    fisher: np.ndarray,
    jacobian: np.ndarray,
    expected_counts: np.ndarray,
    names: tuple[str, ...] | list[str],
    target_index: tuple[int, ...] | list[int],
    scales: np.ndarray | list[float],
    *,
    k: int,
    weak_relative_sd: float,
    projection_tol: float,
) -> ModeDecomposition:
    """Least determined directions of a target block, the rest profiled out.

    The target block of the inverse Fisher matrix -- the inverse of the
    effective information with every other parameter estimated from the
    same data -- is divided by the reference ``scales`` and diagonalised.
    The eigenvectors and their order depend on those scales, which are a
    convention; the null space as a subspace, and the whitened norm of
    every signature, do not.

    The null space of the full matrix comes from the ``n * eps`` rule on
    its unit-diagonal form. The number of null modes is the rank of that
    null space's target rows, counting singular values whose square
    exceeds ``projection_tol`` -- the tolerance that marks a single
    parameter's bound infinite in ``covariance_bound`` -- and is 0 when no
    target parameter's bound is infinite. So there is a null mode exactly
    when some target parameter is 'rank_deficient'. The rest of the null
    space lies wholly outside the target block (``nuisance_null_dim``). The null
    modes are the eigenvectors of the unit-diagonal effective information
    with the smallest eigenvalues; the soft modes are taken on their
    complement in the scaled coordinates, so they are combinations the
    data determine. ``projection_tol`` also decides which ``shares`` are
    None.

    ``whitened_norm`` is 1 in exact arithmetic. The covariance side is
    accurate at the soft end, which is what is reported by default; at
    the stiff end of a near-singular design a mode can lose digits, and so
    can every mode of a design singular to within rounding or of a nearly
    singular background. A departure of ``whitened_norm`` from 1 measures
    by how much. Where rounding leaves no positive bound at all, the mode
    is reported as 'not_resolved' with ``relative_sd`` nan. An eigenvalue
    the null-space count keeps but that sits at rounding level is floored
    at ``n * eps`` of the largest, so it gets a large finite bound, never
    a negative one.

    Args:
        fisher: Fisher matrix, (n, n)
        jacobian: d(expected counts)/d(parameters) at the stated exposure,
            (n_energy, n)
        expected_counts: Poisson mean per channel at the stated exposure
        names: Parameter names, length n
        target_index: Positions of the target parameters; every other
            parameter is profiled out
        scales: Positive reference scale of each target, in the order of
            ``sorted(target_index)``
        k: Number of non-null modes to report, at most
        weak_relative_sd: Threshold for 'weakly_identified'
        projection_tol: As in ``covariance_bound``

    Returns:
        ModeDecomposition
    """
    fisher = np.asarray(fisher, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    mean = np.asarray(expected_counts, dtype=np.float64)
    n = fisher.shape[0]
    names = tuple(names)
    target = np.array(sorted(target_index), dtype=int)
    scales = np.asarray(scales, dtype=np.float64)
    if len(names) != n or jacobian.shape != (mean.size, n):
        raise ValueError("fisher, jacobian, expected_counts and names disagree on sizes")
    if scales.shape != target.shape or not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("need one positive, finite scale per target parameter")
    try:
        k = operator.index(k)
    except TypeError:
        raise ValueError(f"k must be a positive integer, got {k!r}") from None
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    nuisance = np.array([i for i in range(n) if i not in set(target.tolist())], dtype=int)

    unit, _ = _unit_diagonal(fisher)
    full_eigenvalues, full_vectors = np.linalg.eigh(unit)
    null = null_directions(full_eigenvalues)
    null_space_dim = int(null.sum())
    # The rank of the null space's target rows, with the same projection
    # tolerance that marks a parameter's bound infinite.
    target_rows = full_vectors[np.ix_(target, np.flatnonzero(null))]
    n_null = int(np.sum(np.linalg.svd(target_rows, compute_uv=False) ** 2 > projection_tol)
                 if target_rows.size else 0)
    covariance, unbounded, _, _ = covariance_bound(fisher, projection_tol)
    if not unbounded[target].any():
        n_null = 0  # the rank can exceed the tolerance where no single row does
    info = effective_information(fisher, target.tolist())

    # Inverse on the unit-diagonal effective information, its n_null
    # smallest directions dropped, then into scaled coordinates.
    unit_eff, d_eff = _unit_diagonal(info.effective)
    eigenvalues, eigenvectors = np.linalg.eigh(unit_eff)
    kept = eigenvalues[n_null:]
    if kept.size:  # directions the rule kept can still sit at rounding level
        kept = np.maximum(kept, kept[-1] * kept.size * np.finfo(np.float64).eps)
    inverse = (eigenvectors[:, n_null:] / kept) @ eigenvectors[:, n_null:].T
    scaled_cov = inverse / np.outer(d_eff * scales, d_eff * scales)

    null_scaled = eigenvectors[:, :n_null] / (d_eff * scales)[:, np.newaxis]
    basis, _ = np.linalg.qr(
        np.hstack([null_scaled, np.eye(target.size)]) if n_null else np.eye(target.size)
    )
    null_basis, complement = basis[:, :n_null], basis[:, n_null:]
    if complement.shape[1]:
        lam, vectors = np.linalg.eigh(complement.T @ scaled_cov @ complement)
        order = np.argsort(lam)[::-1]
        lam, soft = lam[order], complement @ vectors[:, order]
    else:
        lam, soft = np.zeros(0), complement

    f_qq_pinv = _pinv_unit_diagonal(fisher[np.ix_(nuisance, nuisance)]) if nuisance.size else None

    def move(step_targets: np.ndarray) -> tuple[np.ndarray, dict[str, float], np.ndarray, float]:
        step = np.zeros(n)
        step[target] = step_targets
        if nuisance.size:
            step[nuisance] = -f_qq_pinv @ (fisher[np.ix_(nuisance, target)] @ step_targets)
        signature = jacobian @ step
        shift = {names[j]: float(step[j]) for j in nuisance}
        return step, shift, signature, float(np.sqrt(np.sum(signature**2 / mean)))

    target_names = tuple(names[i] for i in target)
    modes = []
    for i in range(min(k, lam.size)):
        w = _oriented(soft[:, i])
        if not lam[i] > 0.0:
            # Rounding left no bound: only at the stiff end of a
            # near-singular design, where the covariance side has no digits.
            nan_step = np.full(target.size, np.nan)
            _, shift, signature, norm = move(nan_step)
            modes.append(Mode(
                index=i, relative_sd=math.nan, status="not_resolved",
                weights=dict(zip(target_names, (float(x) for x in w))),
                background_shift=shift, signature=signature, whitened_norm=norm,
                shares={name: None for name in names}, eigenvalue_ratio=None,
            ))
            continue
        relative_sd = float(np.sqrt(lam[i]))
        _, shift, signature, norm = move(relative_sd * scales * w)
        coordinate_cov = covariance[:, target] @ (w / scales)
        shares = {
            names[j]: None if unbounded[j] or not covariance[j, j] > 0.0 else float(
                coordinate_cov[j] ** 2 / (covariance[j, j] * lam[i])
            )
            for j in range(n)
        }
        modes.append(Mode(
            index=i, relative_sd=relative_sd,
            status="weakly_identified" if relative_sd > weak_relative_sd else "identified",
            weights=dict(zip(target_names, (float(x) for x in w))),
            background_shift=shift, signature=signature, whitened_norm=norm, shares=shares,
            eigenvalue_ratio=(float(lam[i] / lam[i + 1])
                              if i + 1 < lam.size and lam[i + 1] > 0.0 else None),
        ))

    null_modes = []
    for i in range(n_null):
        q = _oriented(null_basis[:, i])
        _, shift, signature, norm = move(scales * q)
        null_modes.append(Mode(
            index=i, relative_sd=np.inf, status="rank_deficient",
            weights=dict(zip(target_names, (float(x) for x in q))),
            background_shift=shift, signature=signature, whitened_norm=norm,
            shares={name: None for name in names}, eigenvalue_ratio=None,
        ))

    return ModeDecomposition(
        modes=tuple(modes), null_modes=tuple(null_modes), null_space_dim=null_space_dim,
        nuisance_null_dim=null_space_dim - n_null, target_names=target_names,
        scales=tuple(float(s) for s in scales),
    )
