"""
Non-uniform grid generators for Dict2D solvers.

Provides sinh, Chebyshev, and quantile grid generators for the δE and δσ
axes, plus spacing utility functions for non-uniform parabola interpolation.

dev-log 56 (Phase 3 Part 1).
"""


import numpy as np

# ---------------------------------------------------------------------------
# Grid generators
# ---------------------------------------------------------------------------

def uniform_grid(n_points: int, range_max: float) -> np.ndarray:
    """Uniform grid on [-range_max, +range_max].

    Equivalent to the existing ``np.arange`` grid but with exact endpoint
    control and guaranteed symmetry.

    Args:
        n_points: Number of grid points (should be odd for a center point).
        range_max: Grid spans [-range_max, +range_max].

    Returns:
        (n_points,) float64 array, sorted ascending.
    """
    return np.linspace(-range_max, range_max, n_points)


def sinh_grid(n_points: int, range_max: float, alpha: float = 2.0) -> np.ndarray:
    """Sinh-transformed grid: dense at center, sparse at edges.

    Maps uniform t ∈ [-1, +1] through sinh to produce a grid on
    [-range_max, +range_max].  The parameter α controls density:
    larger α → more points near center.  α = 0 degenerates to uniform.

    Args:
        n_points: Number of grid points.
        range_max: Grid spans [-range_max, +range_max].
        alpha: Density parameter (>0).  Typical: 1.0–3.0.

    Returns:
        (n_points,) float64 array, sorted ascending.
    """
    if alpha < 1e-10:
        return np.linspace(-range_max, range_max, n_points)
    t = np.linspace(-1.0, 1.0, n_points)
    grid = range_max * np.sinh(alpha * t) / np.sinh(alpha)
    return grid


def chebyshev_grid(n_points: int, range_max: float) -> np.ndarray:
    """Chebyshev points (Type I): dense at edges and center.

    The Chebyshev distribution minimizes the Runge phenomenon in polynomial
    interpolation.  Points cluster at both endpoints and the center.

    Args:
        n_points: Number of grid points.
        range_max: Grid spans [-range_max, +range_max].

    Returns:
        (n_points,) float64 array, sorted ascending.
    """
    k = np.arange(n_points)
    # cos(π·k/(n-1)) gives points in [+1, -1]; negate to sort ascending
    grid = -range_max * np.cos(np.pi * k / (n_points - 1))
    return grid


def quantile_grid(distribution: np.ndarray, n_points: int,
                  range_max: float | None = None) -> np.ndarray:
    """Quantile grid from an empirical distribution.

    Places grid points at equally-spaced percentiles of the observed
    parameter distribution.  Dense where data is abundant.

    Args:
        distribution: 1D array of observed parameter values.
        n_points: Number of grid points.
        range_max: If given, clip grid to [-range_max, +range_max].

    Returns:
        (n_points,) float64 array, sorted ascending.
    """
    percentiles = np.linspace(0, 100, n_points)
    grid = np.percentile(distribution, percentiles)
    if range_max is not None:
        grid = np.clip(grid, -range_max, range_max)
    return grid


# ---------------------------------------------------------------------------
# Grid spacing utilities for non-uniform parabola interpolation
# ---------------------------------------------------------------------------

def precompute_grid_spacings(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute left and right spacings for non-uniform parabola.

    For each grid point i:
        h_L[i] = grid[i] - grid[i-1]   (left spacing)
        h_R[i] = grid[i+1] - grid[i]   (right spacing)

    Boundary points (i=0, i=n-1) get h_L=0 / h_R=0 respectively,
    and should be masked out during parabola interpolation.

    Args:
        grid: (n,) sorted 1D grid values.

    Returns:
        (h_L, h_R) each (n,) float64.
    """
    n = len(grid)
    diffs = np.diff(grid)                    # (n-1,)
    h_L = np.empty(n, dtype=np.float64)
    h_R = np.empty(n, dtype=np.float64)
    h_L[0] = 0.0
    h_L[1:] = diffs
    h_R[:-1] = diffs
    h_R[-1] = 0.0
    return h_L, h_R


def is_uniform(grid: np.ndarray, rtol: float = 1e-6) -> bool:
    """Check whether a grid is effectively uniform.

    Args:
        grid: (n,) sorted 1D grid values.
        rtol: Relative tolerance for spacing uniformity.

    Returns:
        True if all spacings are equal within rtol.
    """
    if len(grid) < 2:
        return True
    diffs = np.diff(grid)
    mean_diff = np.mean(diffs)
    if mean_diff == 0:
        return True
    return bool(np.all(np.abs(diffs - mean_diff) / mean_diff < rtol))
