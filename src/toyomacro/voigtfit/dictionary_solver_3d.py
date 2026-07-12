"""
3D Dictionary Solver (δE × δσ × δγ)
====================================

Extends Dict2D + parabola to jointly estimate gamma (Lorentzian
half-width) alongside energy shift and Gaussian width.

dev-log 63 showed that gamma bias is the dominant structural limitation of
δσ accuracy — all solvers exhibit identical bias slope (+0.635 vs Fisher
+0.870).  Including γ as an estimation parameter eliminates this bias.

Architecture:
    Phase 0: Build 3D dictionary of Voigt profiles at discrete (δE, δσ, δγ) grid
    Phase 1: GPU matmul + argmax to find nearest dictionary element
    Phase 1.5: Parabola sub-grid interpolation on 3 independent axes
    Phase 2: LLS amplitude at best grid point

Grid sizing (default): 21 × 11 × 9 = 2,079 entries (~8 MB total).
"""

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

try:
    import mlx.core as mx

    from ._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND a Metal device works
except ImportError:
    HAS_MLX = False

from .dictionary_solver import (
    _PHASE1_MAX_BYTES,
    _group_by_index,
    _parabola_vertex,
    _parabola_vertex_nonuniform,
)

# ---------------------------------------------------------------------------
# Dict3D LRU Cache — amortizes build cost across repeated calls
# ---------------------------------------------------------------------------

def _dict3d_cache_key(
    energy: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gamma: float,
    dE_range: tuple[float, float],
    dE_step: float | None,
    dsigma_range: tuple[float, float],
    dsigma_step: float | None,
    dgamma_range: tuple[float, float],
    dgamma_step: float | None,
    dE_grid_override: np.ndarray | None,
    dsigma_grid_override: np.ndarray | None,
    dgamma_grid_override: np.ndarray | None,
) -> str:
    """Compute a stable hash key for Dict3D build parameters."""
    h = hashlib.md5(usedforsecurity=False)
    # Energy axis: shape + endpoints (avoids hashing full array)
    h.update(f"e:{len(energy)}:{float(energy[0]):.6f}:{float(energy[-1]):.6f}".encode())
    # Peak parameters
    h.update(centers.astype(np.float32).tobytes())
    h.update(sigmas.astype(np.float32).tobytes())
    h.update(f"g:{gamma:.8f}".encode())
    # Grid parameters
    h.update(f"dE:{dE_range}:{dE_step}".encode())
    h.update(f"ds:{dsigma_range}:{dsigma_step}".encode())
    h.update(f"dg:{dgamma_range}:{dgamma_step}".encode())
    # Grid overrides
    for name, arr in [("dEo", dE_grid_override), ("dso", dsigma_grid_override), ("dgo", dgamma_grid_override)]:
        if arr is not None:
            h.update(f"{name}:".encode())
            h.update(np.asarray(arr, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


class _Dict3DLRUCache:
    """Thread-safe LRU cache for DictionaryCache3D objects.

    Typical usage: 1-2 entries (one per gamma_nom value).
    max_size is auto-determined from available memory if not specified:
        16GB  → 1, 32GB → 2, 64GB → 3, 128GB → 4
    """

    def __init__(self, max_size: int | None = None):
        if max_size is None:
            from .memory import optimal_dict3d_cache_size
            max_size = optimal_dict3d_cache_size()
        self._cache: OrderedDict[str, DictionaryCache3D] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional['DictionaryCache3D']:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: 'DictionaryCache3D') -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def info(self) -> dict:
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(1, self._hits + self._misses),
            }


# Module-level singleton
_dict3d_cache = _Dict3DLRUCache()


def get_or_build_dictionary_3d(
    energy: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gamma: float,
    dE_range: tuple[float, float] = (-2.0, 2.0),
    dE_step: float | None = None,
    dsigma_range: tuple[float, float] = (-0.3, 0.3),
    dsigma_step: float | None = None,
    dgamma_range: tuple[float, float] = (-0.05, 0.05),
    dgamma_step: float | None = None,
    reg_lambda: float = 1e-8,
    dE_grid_override: np.ndarray | None = None,
    dsigma_grid_override: np.ndarray | None = None,
    dgamma_grid_override: np.ndarray | None = None,
) -> 'DictionaryCache3D':
    """Build or retrieve a cached Dict3D.

    Same signature as build_dictionary_3d but with LRU caching.
    First call builds the dictionary (~2K grid points, expensive);
    subsequent calls with identical parameters return the cached object.

    Returns:
        DictionaryCache3D (may be shared — do not mutate).
    """
    energy = np.asarray(energy, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    sigmas = np.asarray(sigmas, dtype=np.float32)

    key = _dict3d_cache_key(
        energy, centers, sigmas, gamma,
        dE_range, dE_step, dsigma_range, dsigma_step,
        dgamma_range, dgamma_step,
        dE_grid_override, dsigma_grid_override, dgamma_grid_override,
    )

    cached = _dict3d_cache.get(key)
    if cached is not None:
        return cached

    result = build_dictionary_3d(
        energy=energy, centers=centers, sigmas=sigmas, gamma=gamma,
        dE_range=dE_range, dE_step=dE_step,
        dsigma_range=dsigma_range, dsigma_step=dsigma_step,
        dgamma_range=dgamma_range, dgamma_step=dgamma_step,
        reg_lambda=reg_lambda,
        dE_grid_override=dE_grid_override,
        dsigma_grid_override=dsigma_grid_override,
        dgamma_grid_override=dgamma_grid_override,
    )
    _dict3d_cache.put(key, result)
    return result


def clear_dict3d_cache() -> None:
    """Clear the Dict3D LRU cache."""
    _dict3d_cache.clear()


def dict3d_cache_info() -> dict:
    """Return cache statistics."""
    return _dict3d_cache.info()


@dataclass
class DictionaryCache3D:
    """Pre-computed 3D dictionary and per-grid-point matrices.

    Flat index convention:
        flat = dE_idx * (n_ds * n_dg) + ds_idx * n_dg + dg_idx

    Attributes:
        D: L2-normalized dictionary matrix (n_energy, n_dict)
        dE_grid: δE grid values (n_dE,)
        dsigma_grid: δσ grid values (n_ds,)
        dgamma_grid: δγ grid values (n_dg,)
        grid_shape: (n_dE, n_ds, n_dg)
        Wt_per_grid: Weight matrices (n_dict, n_comp, n_energy)
        Phi_per_grid: Basis matrices (n_dict, n_energy, n_comp)
        nominal_index: Index of the (δE≈0, δσ≈0, δγ≈0) grid point
        mean_sigma: Mean σ of the peak components
    """
    D: np.ndarray
    dE_grid: np.ndarray
    dsigma_grid: np.ndarray
    dgamma_grid: np.ndarray
    grid_shape: tuple[int, int, int]
    Wt_per_grid: np.ndarray
    Phi_per_grid: np.ndarray
    nominal_index: int = 0
    mean_sigma: float = 1.0
    # MLX lazy cache
    _D_mx: Any | None = field(default=None, repr=False)
    _Wt_mx: Any | None = field(default=None, repr=False)
    _Phi_mx: Any | None = field(default=None, repr=False)

    @property
    def n_dict(self) -> int:
        return self.D.shape[1]

    @property
    def n_dE(self) -> int:
        return self.grid_shape[0]

    @property
    def n_dsigma(self) -> int:
        return self.grid_shape[1]

    @property
    def n_dgamma(self) -> int:
        return self.grid_shape[2]

    def to_mlx(self):
        """Lazy convert to MLX arrays."""
        if not HAS_MLX:
            return
        if self._D_mx is None:
            self._D_mx = mx.array(self.D)
            self._Wt_mx = mx.array(self.Wt_per_grid)
            self._Phi_mx = mx.array(self.Phi_per_grid)


def build_dictionary_3d(
    energy: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gamma: float,
    dE_range: tuple[float, float] = (-2.0, 2.0),
    dE_step: float | None = None,
    dsigma_range: tuple[float, float] = (-0.3, 0.3),
    dsigma_step: float | None = None,
    dgamma_range: tuple[float, float] = (-0.05, 0.05),
    dgamma_step: float | None = None,
    reg_lambda: float = 1e-8,
    dE_grid_override: np.ndarray | None = None,
    dsigma_grid_override: np.ndarray | None = None,
    dgamma_grid_override: np.ndarray | None = None,
) -> DictionaryCache3D:
    """Build 3D Voigt dictionary over (δE, δσ, δγ) grid.

    Args:
        energy: Energy axis (n_energy,)
        centers: Nominal peak centers (n_comp,)
        sigmas: Nominal Gaussian widths (n_comp,)
        gamma: Nominal Lorentzian width (shared)
        dE_range: δE search range in eV (default ±2.0)
        dE_step: δE grid spacing (default 0.1 × mean_sigma)
        dsigma_range: δσ search range in eV (default ±0.3)
        dsigma_step: δσ grid spacing (default 0.06)
        dgamma_range: δγ search range in eV (default ±0.05)
        dgamma_step: δγ grid spacing (default 0.0125)
        reg_lambda: Tikhonov regularization
        dE_grid_override: Pre-built δE grid
        dsigma_grid_override: Pre-built δσ grid
        dgamma_grid_override: Pre-built δγ grid

    Returns:
        DictionaryCache3D with all pre-computed matrices
    """
    from .voigt_jacobian import voigt_jacobian_batch

    energy = np.asarray(energy, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    sigmas = np.asarray(sigmas, dtype=np.float32)
    n_energy = len(energy)
    n_comp = len(centers)

    mean_sigma = float(np.mean(sigmas))
    if dE_step is None:
        dE_step = 0.1 * mean_sigma
    if dsigma_step is None:
        dsigma_step = 0.06
    if dgamma_step is None:
        dgamma_step = 0.0125

    # Build grids (or use overrides)
    if dE_grid_override is not None:
        dE_grid = np.asarray(dE_grid_override, dtype=np.float32)
    else:
        dE_grid = np.arange(dE_range[0], dE_range[1] + dE_step / 2, dE_step,
                            dtype=np.float32)

    if dsigma_grid_override is not None:
        dsigma_grid = np.asarray(dsigma_grid_override, dtype=np.float32)
    else:
        dsigma_grid = np.arange(dsigma_range[0], dsigma_range[1] + dsigma_step / 2,
                                dsigma_step, dtype=np.float32)

    if dgamma_grid_override is not None:
        dgamma_grid = np.asarray(dgamma_grid_override, dtype=np.float32)
    else:
        dgamma_grid = np.arange(dgamma_range[0], dgamma_range[1] + dgamma_step / 2,
                                dgamma_step, dtype=np.float32)

    # Clip grids to ensure sigma and gamma stay positive
    min_sigma = float(np.min(sigmas))
    dsigma_grid = dsigma_grid[dsigma_grid > -0.9 * min_sigma]
    dgamma_grid = dgamma_grid[dgamma_grid > -0.9 * gamma]

    n_dE = len(dE_grid)
    n_ds = len(dsigma_grid)
    n_dg = len(dgamma_grid)
    n_dict = n_dE * n_ds * n_dg

    # Allocate per-grid-point arrays
    D = np.zeros((n_energy, n_dict), dtype=np.float32)
    Wt_per_grid = np.zeros((n_dict, n_comp, n_energy), dtype=np.float32)
    Phi_per_grid = np.zeros((n_dict, n_energy, n_comp), dtype=np.float32)

    eye_comp = np.eye(n_comp, dtype=np.float32)

    for i, dE in enumerate(dE_grid):
        for j, ds in enumerate(dsigma_grid):
            for k, dg in enumerate(dgamma_grid):
                flat_idx = i * (n_ds * n_dg) + j * n_dg + k

                shifted_centers = centers + dE
                shifted_sigmas = sigmas + ds
                shifted_gamma = gamma + dg

                # Build Voigt basis at this grid point (only need Phi)
                Phi, _, _, _ = voigt_jacobian_batch(
                    energy, shifted_centers, shifted_sigmas, shifted_gamma
                )
                Phi = Phi.astype(np.float32)

                # Dictionary column: composite profile (L2-normalized)
                composite = Phi.sum(axis=1)
                norm = np.linalg.norm(composite)
                if norm > 1e-10:
                    D[:, flat_idx] = composite / norm
                else:
                    D[:, flat_idx] = 0.0

                # Weight matrix Wt = (Phi'Phi + λI)^-1 Phi'
                AtA = Phi.T @ Phi + reg_lambda * eye_comp
                Wt = np.linalg.solve(AtA, Phi.T).astype(np.float32)

                Wt_per_grid[flat_idx] = Wt
                Phi_per_grid[flat_idx] = Phi

    # Find nominal index (δE≈0, δσ≈0, δγ≈0)
    nom_dE_idx = int(np.argmin(np.abs(dE_grid)))
    nom_ds_idx = int(np.argmin(np.abs(dsigma_grid)))
    nom_dg_idx = int(np.argmin(np.abs(dgamma_grid)))
    nominal_index = nom_dE_idx * (n_ds * n_dg) + nom_ds_idx * n_dg + nom_dg_idx

    return DictionaryCache3D(
        D=D,
        dE_grid=dE_grid,
        dsigma_grid=dsigma_grid,
        dgamma_grid=dgamma_grid,
        grid_shape=(n_dE, n_ds, n_dg),
        Wt_per_grid=Wt_per_grid,
        Phi_per_grid=Phi_per_grid,
        nominal_index=nominal_index,
        mean_sigma=mean_sigma,
    )


def parabola_refine_3d(
    scores: np.ndarray,
    best_idx: np.ndarray,
    grid_shape: tuple[int, int, int],
    dE_grid: np.ndarray,
    dsigma_grid: np.ndarray,
    dgamma_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-grid parabola interpolation on a 3D (dE × dσ × dγ) score grid.

    Fits independent 1D parabolas along each axis. This is justified by
    the moderate Fisher off-diagonal coupling (r(σ,γ) ≈ 0.68).

    Args:
        scores: (n_spectra, n_dict) similarity scores.
        best_idx: (n_spectra,) int32 flat index of argmax.
        grid_shape: (n_dE, n_ds, n_dg).
        dE_grid: (n_dE,) float32.
        dsigma_grid: (n_ds,) float32.
        dgamma_grid: (n_dg,) float32.

    Returns:
        (dE_refined, dsigma_refined, dgamma_refined) each (n_spectra,) float32.
    """
    n_dE, n_ds, n_dg = grid_shape
    n_ds_dg = n_ds * n_dg
    n_spectra = len(best_idx)

    # Unflatten to 3D indices
    dE_idx = best_idx // n_ds_dg
    remainder = best_idx % n_ds_dg
    ds_idx = remainder // n_dg
    dg_idx = remainder % n_dg

    # Interior masks
    dE_interior = (dE_idx > 0) & (dE_idx < n_dE - 1)
    ds_interior = (ds_idx > 0) & (ds_idx < n_ds - 1)
    dg_interior = (dg_idx > 0) & (dg_idx < n_dg - 1)

    # Center score
    spec_range = np.arange(n_spectra)
    s0 = scores[spec_range, best_idx]

    # --- dE axis neighbours ---
    dE_m_flat = np.clip(dE_idx - 1, 0, n_dE - 1) * n_ds_dg + ds_idx * n_dg + dg_idx
    dE_p_flat = np.clip(dE_idx + 1, 0, n_dE - 1) * n_ds_dg + ds_idx * n_dg + dg_idx
    sm_dE = scores[spec_range, dE_m_flat]
    sp_dE = scores[spec_range, dE_p_flat]

    # --- dsigma axis neighbours ---
    ds_m_flat = dE_idx * n_ds_dg + np.clip(ds_idx - 1, 0, n_ds - 1) * n_dg + dg_idx
    ds_p_flat = dE_idx * n_ds_dg + np.clip(ds_idx + 1, 0, n_ds - 1) * n_dg + dg_idx
    sm_ds = scores[spec_range, ds_m_flat]
    sp_ds = scores[spec_range, ds_p_flat]

    # --- dgamma axis neighbours ---
    dg_m_flat = dE_idx * n_ds_dg + ds_idx * n_dg + np.clip(dg_idx - 1, 0, n_dg - 1)
    dg_p_flat = dE_idx * n_ds_dg + ds_idx * n_dg + np.clip(dg_idx + 1, 0, n_dg - 1)
    sm_dg = scores[spec_range, dg_m_flat]
    sp_dg = scores[spec_range, dg_p_flat]

    from .grids import is_uniform, precompute_grid_spacings

    # --- dE axis parabola ---
    if is_uniform(dE_grid):
        dE_step = dE_grid[1] - dE_grid[0] if n_dE > 1 else 1.0
        delta_dE = _parabola_vertex(sm_dE, s0, sp_dE)
        delta_dE = np.where(dE_interior, delta_dE, 0.0)
        dE_refined = dE_grid[dE_idx] + delta_dE * dE_step
    else:
        hL_dE, hR_dE = precompute_grid_spacings(dE_grid)
        shift_dE = _parabola_vertex_nonuniform(
            sm_dE, s0, sp_dE, hL_dE[dE_idx], hR_dE[dE_idx])
        shift_dE = np.where(dE_interior, shift_dE, 0.0)
        dE_refined = dE_grid[dE_idx] + shift_dE

    # --- dsigma axis parabola ---
    if is_uniform(dsigma_grid):
        ds_step = dsigma_grid[1] - dsigma_grid[0] if n_ds > 1 else 1.0
        delta_ds = _parabola_vertex(sm_ds, s0, sp_ds)
        delta_ds = np.where(ds_interior, delta_ds, 0.0)
        dsigma_refined = dsigma_grid[ds_idx] + delta_ds * ds_step
    else:
        hL_ds, hR_ds = precompute_grid_spacings(dsigma_grid)
        shift_ds = _parabola_vertex_nonuniform(
            sm_ds, s0, sp_ds, hL_ds[ds_idx], hR_ds[ds_idx])
        shift_ds = np.where(ds_interior, shift_ds, 0.0)
        dsigma_refined = dsigma_grid[ds_idx] + shift_ds

    # --- dgamma axis parabola ---
    if is_uniform(dgamma_grid):
        dg_step = dgamma_grid[1] - dgamma_grid[0] if n_dg > 1 else 1.0
        delta_dg = _parabola_vertex(sm_dg, s0, sp_dg)
        delta_dg = np.where(dg_interior, delta_dg, 0.0)
        dgamma_refined = dgamma_grid[dg_idx] + delta_dg * dg_step
    else:
        hL_dg, hR_dg = precompute_grid_spacings(dgamma_grid)
        shift_dg = _parabola_vertex_nonuniform(
            sm_dg, s0, sp_dg, hL_dg[dg_idx], hR_dg[dg_idx])
        shift_dg = np.where(dg_interior, shift_dg, 0.0)
        dgamma_refined = dgamma_grid[dg_idx] + shift_dg

    return (
        dE_refined.astype(np.float32),
        dsigma_refined.astype(np.float32),
        dgamma_refined.astype(np.float32),
    )


def solve_dict3d_parabola(
    Y: np.ndarray,
    dict_cache: DictionaryCache3D,
    min_group_for_mlx: int = 64,
    chunk_bytes: int = _PHASE1_MAX_BYTES,
    use_fp16_scores: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """3D dictionary solver with parabola sub-grid interpolation.

    Phase 1: Dictionary lookup (matmul + argmax) → coarse (δE, δσ, δγ)
    Phase 1.5: Parabola interpolation on 3 axes → refined (δE, δσ, δγ)
    Phase 2: LLS amplitude at best grid point

    Added memory-aware chunking and fp16 support.

    Args:
        Y: Spectra (n_spectra, n_energy) float32
        dict_cache: Pre-built 3D dictionary cache
        min_group_for_mlx: Min group size for MLX path
        chunk_bytes: Max GPU buffer for scores matrix (default 4 GB)
        use_fp16_scores: If True, compute scores in float16 (halves memory)

    Returns:
        (amplitudes, chi2, energy_shifts, sigma_shifts, gamma_shifts, dict_indices)
    """
    if not HAS_MLX:
        # CPU fallback
        return _solve_dict3d_parabola_cpu(Y, dict_cache)

    Y = np.asarray(Y, dtype=np.float32)
    n_spectra = Y.shape[0]
    n_energy = Y.shape[1]
    n_comp = dict_cache.Wt_per_grid.shape[1]
    n_energy_inv = 1.0 / n_energy

    dict_cache.to_mlx()

    # Phase 1: Dictionary lookup on GPU
    Y_mx = mx.array(Y)
    n_dict = dict_cache.n_dict
    dtype_bytes = 2 if use_fp16_scores else 4
    max_rows = max(1, chunk_bytes // (n_dict * dtype_bytes))

    if max_rows >= n_spectra:
        D_mx = dict_cache._D_mx
        if use_fp16_scores:
            scores_mx = Y_mx.astype(mx.float16) @ D_mx.astype(mx.float16)
        else:
            scores_mx = Y_mx @ D_mx
        best_idx_mx = mx.argmax(scores_mx, axis=1)
        mx.eval(best_idx_mx)
    else:
        # Chunked lookup
        best_idx_parts = []
        scores_parts = []
        for start in range(0, n_spectra, max_rows):
            end = min(start + max_rows, n_spectra)
            chunk_Y = Y_mx[start:end]
            D_mx = dict_cache._D_mx
            if use_fp16_scores:
                chunk_scores = chunk_Y.astype(mx.float16) @ D_mx.astype(mx.float16)
            else:
                chunk_scores = chunk_Y @ D_mx
            chunk_best = mx.argmax(chunk_scores, axis=1)
            mx.eval(chunk_best)
            best_idx_parts.append(np.array(chunk_best).astype(np.int32))
            scores_parts.append(chunk_scores)
        best_idx = np.concatenate(best_idx_parts)
        # For parabola we need scores_mx — reconstruct or do chunked parabola
        # For simplicity, process chunks separately
        return _solve_dict3d_chunked(
            Y, Y_mx, dict_cache, best_idx_parts, scores_parts,
            n_energy, n_comp, n_energy_inv, min_group_for_mlx
        )

    best_idx = np.array(best_idx_mx).astype(np.int32)

    # Phase 1.5: Extract neighbour scores on GPU, then parabola on CPU
    n_dE, n_ds, n_dg = dict_cache.grid_shape
    n_ds_dg = n_ds * n_dg

    dE_idx = best_idx // n_ds_dg
    remainder = best_idx % n_ds_dg
    ds_idx = remainder // n_dg
    dg_idx = remainder % n_dg

    spec_idx_mx = mx.arange(n_spectra)
    best_flat_mx = mx.array(best_idx)

    s0 = scores_mx[spec_idx_mx, best_flat_mx]

    # dE neighbours
    dE_m = np.clip(dE_idx - 1, 0, n_dE - 1) * n_ds_dg + ds_idx * n_dg + dg_idx
    dE_p = np.clip(dE_idx + 1, 0, n_dE - 1) * n_ds_dg + ds_idx * n_dg + dg_idx
    s_dE_m = scores_mx[spec_idx_mx, mx.array(dE_m)]
    s_dE_p = scores_mx[spec_idx_mx, mx.array(dE_p)]

    # dsigma neighbours
    ds_m = dE_idx * n_ds_dg + np.clip(ds_idx - 1, 0, n_ds - 1) * n_dg + dg_idx
    ds_p = dE_idx * n_ds_dg + np.clip(ds_idx + 1, 0, n_ds - 1) * n_dg + dg_idx
    s_ds_m = scores_mx[spec_idx_mx, mx.array(ds_m)]
    s_ds_p = scores_mx[spec_idx_mx, mx.array(ds_p)]

    # dgamma neighbours
    dg_m = dE_idx * n_ds_dg + ds_idx * n_dg + np.clip(dg_idx - 1, 0, n_dg - 1)
    dg_p = dE_idx * n_ds_dg + ds_idx * n_dg + np.clip(dg_idx + 1, 0, n_dg - 1)
    s_dg_m = scores_mx[spec_idx_mx, mx.array(dg_m)]
    s_dg_p = scores_mx[spec_idx_mx, mx.array(dg_p)]

    mx.eval(s0, s_dE_m, s_dE_p, s_ds_m, s_ds_p, s_dg_m, s_dg_p)

    # Parabola on CPU
    s0_np = np.array(s0)
    sm_dE_np = np.array(s_dE_m)
    sp_dE_np = np.array(s_dE_p)
    sm_ds_np = np.array(s_ds_m)
    sp_ds_np = np.array(s_ds_p)
    sm_dg_np = np.array(s_dg_m)
    sp_dg_np = np.array(s_dg_p)

    dE_interior = (dE_idx > 0) & (dE_idx < n_dE - 1)
    ds_interior = (ds_idx > 0) & (ds_idx < n_ds - 1)
    dg_interior = (dg_idx > 0) & (dg_idx < n_dg - 1)

    from .grids import is_uniform, precompute_grid_spacings

    # dE parabola
    if is_uniform(dict_cache.dE_grid):
        dE_step = dict_cache.dE_grid[1] - dict_cache.dE_grid[0] if n_dE > 1 else 1.0
        delta_dE = _parabola_vertex(sm_dE_np, s0_np, sp_dE_np)
        delta_dE = np.where(dE_interior, delta_dE, 0.0)
        dE_refined = (dict_cache.dE_grid[dE_idx] + delta_dE * dE_step).astype(np.float32)
    else:
        hL_dE, hR_dE = precompute_grid_spacings(dict_cache.dE_grid)
        shift_dE = _parabola_vertex_nonuniform(
            sm_dE_np, s0_np, sp_dE_np, hL_dE[dE_idx], hR_dE[dE_idx])
        shift_dE = np.where(dE_interior, shift_dE, 0.0)
        dE_refined = (dict_cache.dE_grid[dE_idx] + shift_dE).astype(np.float32)

    # dsigma parabola
    if is_uniform(dict_cache.dsigma_grid):
        ds_step = dict_cache.dsigma_grid[1] - dict_cache.dsigma_grid[0] if n_ds > 1 else 1.0
        delta_ds = _parabola_vertex(sm_ds_np, s0_np, sp_ds_np)
        delta_ds = np.where(ds_interior, delta_ds, 0.0)
        dsigma_refined = (dict_cache.dsigma_grid[ds_idx] + delta_ds * ds_step).astype(np.float32)
    else:
        hL_ds, hR_ds = precompute_grid_spacings(dict_cache.dsigma_grid)
        shift_ds = _parabola_vertex_nonuniform(
            sm_ds_np, s0_np, sp_ds_np, hL_ds[ds_idx], hR_ds[ds_idx])
        shift_ds = np.where(ds_interior, shift_ds, 0.0)
        dsigma_refined = (dict_cache.dsigma_grid[ds_idx] + shift_ds).astype(np.float32)

    # dgamma parabola
    if is_uniform(dict_cache.dgamma_grid):
        dg_step = dict_cache.dgamma_grid[1] - dict_cache.dgamma_grid[0] if n_dg > 1 else 1.0
        delta_dg = _parabola_vertex(sm_dg_np, s0_np, sp_dg_np)
        delta_dg = np.where(dg_interior, delta_dg, 0.0)
        dgamma_refined = (dict_cache.dgamma_grid[dg_idx] + delta_dg * dg_step).astype(np.float32)
    else:
        hL_dg, hR_dg = precompute_grid_spacings(dict_cache.dgamma_grid)
        shift_dg = _parabola_vertex_nonuniform(
            sm_dg_np, s0_np, sp_dg_np, hL_dg[dg_idx], hR_dg[dg_idx])
        shift_dg = np.where(dg_interior, shift_dg, 0.0)
        dgamma_refined = (dict_cache.dgamma_grid[dg_idx] + shift_dg).astype(np.float32)

    del scores_mx  # Free GPU memory

    # Phase 2: Amplitude-only at best grid point
    groups = _group_by_index(best_idx, dict_cache.n_dict)

    amp_out = np.zeros((n_comp, n_spectra), dtype=np.float32)
    chi2_out = np.zeros(n_spectra, dtype=np.float32)

    mlx_groups = []
    mlx_outputs = []

    for orig_indices, gidx in groups:
        group_size = len(orig_indices)

        if group_size >= min_group_for_mlx:
            idx_mx = mx.array(orig_indices.astype(np.int32))
            Y_g_mx = Y_mx[idx_mx]
            Wt_mx = dict_cache._Wt_mx[gidx]
            Phi_mx = dict_cache._Phi_mx[gidx]

            A_row = Y_g_mx @ Wt_mx.T
            R = Y_g_mx - A_row @ Phi_mx.T
            chi2_g = mx.sum(R * R, axis=1) * n_energy_inv

            mlx_groups.append((orig_indices, gidx))
            mlx_outputs.append((A_row, chi2_g))
        else:
            Wt = dict_cache.Wt_per_grid[gidx]
            Phi = dict_cache.Phi_per_grid[gidx]
            Y_group = Y[orig_indices]
            A_row_np = Y_group @ Wt.T
            R = Y_group - A_row_np @ Phi.T
            amp_out[:, orig_indices] = A_row_np.T
            chi2_out[orig_indices] = np.sum(R * R, axis=1) / n_energy

    if mlx_outputs:
        all_lazy = []
        for out in mlx_outputs:
            all_lazy.extend(out)
        mx.eval(*all_lazy)

        for (orig_indices, _gidx), (A_row_mx, chi2_mx) in zip(mlx_groups, mlx_outputs):
            amp_out[:, orig_indices] = np.array(A_row_mx).T
            chi2_out[orig_indices] = np.array(chi2_mx)

    return amp_out, chi2_out, dE_refined, dsigma_refined, dgamma_refined, best_idx


def _solve_dict3d_parabola_cpu(
    Y: np.ndarray,
    dict_cache: DictionaryCache3D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CPU fallback for solve_dict3d_parabola."""
    Y = np.asarray(Y, dtype=np.float32)
    n_spectra = Y.shape[0]
    n_energy = Y.shape[1]
    n_comp = dict_cache.Wt_per_grid.shape[1]

    # Phase 1: Dictionary lookup
    scores = Y @ dict_cache.D
    best_idx = np.argmax(scores, axis=1).astype(np.int32)

    # Phase 1.5: Parabola
    dE_refined, dsigma_refined, dgamma_refined = parabola_refine_3d(
        scores, best_idx, dict_cache.grid_shape,
        dict_cache.dE_grid, dict_cache.dsigma_grid, dict_cache.dgamma_grid,
    )

    # Phase 2: LLS amplitude
    groups = _group_by_index(best_idx, dict_cache.n_dict)
    amp_out = np.zeros((n_comp, n_spectra), dtype=np.float32)
    chi2_out = np.zeros(n_spectra, dtype=np.float32)

    for orig_indices, gidx in groups:
        Wt = dict_cache.Wt_per_grid[gidx]
        Phi = dict_cache.Phi_per_grid[gidx]
        Y_group = Y[orig_indices]
        A_row_np = Y_group @ Wt.T
        R = Y_group - A_row_np @ Phi.T
        amp_out[:, orig_indices] = A_row_np.T
        chi2_out[orig_indices] = np.sum(R * R, axis=1) / n_energy

    return amp_out, chi2_out, dE_refined, dsigma_refined, dgamma_refined, best_idx


def _solve_dict3d_chunked(
    Y: np.ndarray,
    Y_mx: 'mx.array',
    dict_cache: DictionaryCache3D,
    best_idx_parts: list,
    scores_parts: list,
    n_energy: int,
    n_comp: int,
    n_energy_inv: float,
    min_group_for_mlx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Handle chunked case when score matrix doesn't fit in single GPU pass."""
    n_spectra = Y.shape[0]
    n_dict = dict_cache.n_dict
    max_rows = max(1, _PHASE1_MAX_BYTES // (n_dict * 4))

    all_dE = np.empty(n_spectra, dtype=np.float32)
    all_ds = np.empty(n_spectra, dtype=np.float32)
    all_dg = np.empty(n_spectra, dtype=np.float32)
    all_best = np.concatenate(best_idx_parts)

    offset = 0
    for chunk_best, chunk_scores_mx in zip(best_idx_parts, scores_parts):
        chunk_size = len(chunk_best)
        # Transfer chunk scores to CPU for parabola
        chunk_scores = np.array(chunk_scores_mx)
        dE_r, ds_r, dg_r = parabola_refine_3d(
            chunk_scores, chunk_best, dict_cache.grid_shape,
            dict_cache.dE_grid, dict_cache.dsigma_grid, dict_cache.dgamma_grid,
        )
        all_dE[offset:offset + chunk_size] = dE_r
        all_ds[offset:offset + chunk_size] = ds_r
        all_dg[offset:offset + chunk_size] = dg_r
        offset += chunk_size
        del chunk_scores_mx  # Free GPU

    # Phase 2: LLS amplitude (same as non-chunked)
    groups = _group_by_index(all_best, n_dict)
    amp_out = np.zeros((n_comp, n_spectra), dtype=np.float32)
    chi2_out = np.zeros(n_spectra, dtype=np.float32)

    mlx_groups = []
    mlx_outputs = []

    for orig_indices, gidx in groups:
        group_size = len(orig_indices)

        if group_size >= min_group_for_mlx:
            idx_mx = mx.array(orig_indices.astype(np.int32))
            Y_g_mx = Y_mx[idx_mx]
            Wt_mx = dict_cache._Wt_mx[gidx]
            Phi_mx = dict_cache._Phi_mx[gidx]

            A_row = Y_g_mx @ Wt_mx.T
            R = Y_g_mx - A_row @ Phi_mx.T
            chi2_g = mx.sum(R * R, axis=1) * n_energy_inv

            mlx_groups.append((orig_indices, gidx))
            mlx_outputs.append((A_row, chi2_g))
        else:
            Wt = dict_cache.Wt_per_grid[gidx]
            Phi = dict_cache.Phi_per_grid[gidx]
            Y_group = Y[orig_indices]
            A_row_np = Y_group @ Wt.T
            R = Y_group - A_row_np @ Phi.T
            amp_out[:, orig_indices] = A_row_np.T
            chi2_out[orig_indices] = np.sum(R * R, axis=1) / n_energy

    if mlx_outputs:
        all_lazy = []
        for out in mlx_outputs:
            all_lazy.extend(out)
        mx.eval(*all_lazy)

        for (orig_indices, _gidx), (A_row_mx, chi2_mx) in zip(mlx_groups, mlx_outputs):
            amp_out[:, orig_indices] = np.array(A_row_mx).T
            chi2_out[orig_indices] = np.array(chi2_mx)

    return amp_out, chi2_out, all_dE, all_ds, all_dg, all_best


# ---------------------------------------------------------------------------
# 2-Stage Hybrid: Dict3D (global γ) → Dict2D (precision fit)
# ---------------------------------------------------------------------------

@dataclass
class TwoStageResult:
    """Result of 2-stage hybrid γ calibration.

    Stage 1: Dict3D estimates per-spectrum δγ → mean = global Δγ.
    Stage 2: Dict2D with γ_corrected = γ_nom + Δγ_global.

    Attributes:
        amplitudes: (n_comp, n_spectra)
        chi2: (n_spectra,)
        energy_shifts: (n_spectra,) parabola-refined δE from Stage 2
        sigma_shifts: (n_spectra,) parabola-refined δσ from Stage 2
        global_dgamma: Scalar — estimated global γ shift
        dgamma_per_spectrum: (n_spectra,) Stage 1 per-spectrum δγ estimates
        dgamma_std: Std of per-spectrum δγ (noise floor indicator)
        n_calibration: Number of spectra used for γ calibration
        corrected_gamma: γ_nom + global_dgamma
        timing: Dict of stage timings
    """
    amplitudes: np.ndarray
    chi2: np.ndarray
    energy_shifts: np.ndarray
    sigma_shifts: np.ndarray
    global_dgamma: float
    dgamma_per_spectrum: np.ndarray
    dgamma_std: float
    n_calibration: int
    corrected_gamma: float
    timing: dict = field(default_factory=dict)


def solve_2stage_hybrid(
    Y: np.ndarray,
    energy: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gamma: float,
    dE_range: tuple[float, float] = (-2.0, 2.0),
    dsigma_range: tuple[float, float] = (-0.3, 0.3),
    dgamma_range: tuple[float, float] = (-0.05, 0.05),
    dgamma_step: float | None = None,
    stage1_dE_step: float | None = None,
    stage1_dsigma_step: float | None = None,
    n_calibration: int | None = None,
    aggregation: str = "mean",
) -> TwoStageResult:
    """2-stage hybrid solver: Dict3D γ-calibration → Dict2D precision fit.

    Stage 1: Run Dict3D on calibration spectra to estimate global Δγ.
             Per-spectrum δγ has high variance (Fisher v_ratio), but
             averaging over N spectra gives √N precision improvement.
             Uses coarser dE/dσ grid (don't need per-spectrum precision)
             and finer dγ grid (this is what we're estimating).

    Stage 2: Build Dict2D with corrected γ (γ_nom + Δγ_global) and
             run parabola solver for high-precision δE, δσ recovery.

    This combines Dict3D's systematic bias removal (93% γ-bias reduction)
    with Dict2D's precision (30+ dB δσ PSNR), eliminating the accuracy
    tradeoff discovered in dev-log 64.

    Args:
        Y: Spectra (n_spectra, n_energy) float32
        energy: Energy axis (n_energy,) float32
        centers: Peak centers (n_comp,) float32
        sigmas: Gaussian widths (n_comp,) float32
        gamma: Nominal Lorentzian half-width
        dE_range: δE search range for both stages
        dsigma_range: δσ search range for Stage 2
        dgamma_range: δγ search range for Stage 1
        dgamma_step: Stage 1 δγ grid step (default 0.005 = fine resolution)
        stage1_dE_step: Stage 1 δE grid step (default 2× normal = coarser)
        stage1_dsigma_step: Stage 1 δσ grid step (default 2× normal = coarser)
        n_calibration: Number of spectra for Stage 1 γ estimation.
            None = use all spectra. Set to limit cost when N is very large.
        aggregation: "mean" or "median" for global Δγ from per-spectrum estimates.

    Returns:
        TwoStageResult with amplitudes, shifts, and γ-calibration diagnostics.
    """
    import time

    from .dictionary_solver import build_dictionary, solve_dict2d_parabola

    Y = np.asarray(Y, dtype=np.float32)
    energy = np.asarray(energy, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    sigmas = np.asarray(sigmas, dtype=np.float32)
    n_spectra = Y.shape[0]
    timing = {}

    mean_sigma = float(np.mean(sigmas))

    # Stage 1 grid: fine dσ + fine dγ for reliable γ estimation.
    # σ-γ Fisher coupling (r≈0.68-0.76) means the argmax can only resolve
    # γ when σ has adequate grid resolution to separate the two dimensions.
    # dsigma_step=0.04 verified in dev-log 64 (test_gamma_recovery).
    if dgamma_step is None:
        dgamma_step = 0.01  # Finer than Dict3D default (0.0125)
    if stage1_dE_step is None:
        stage1_dE_step = None  # Use Dict3D default (0.1 * mean_sigma)
    if stage1_dsigma_step is None:
        stage1_dsigma_step = 0.04  # Finer than default 0.06; needed for σ-γ separation

    # --- Stage 1: Dict3D → global Δγ ---
    t0 = time.perf_counter()

    # Select calibration subset
    if n_calibration is not None and n_calibration < n_spectra:
        # Use evenly-spaced indices for representative sampling
        cal_idx = np.linspace(0, n_spectra - 1, n_calibration, dtype=int)
        Y_cal = Y[cal_idx]
        n_cal = len(cal_idx)
    else:
        Y_cal = Y
        cal_idx = None
        n_cal = n_spectra

    # Build Dict3D with optimized grid for γ estimation (cached)
    dict3d = get_or_build_dictionary_3d(
        energy=energy, centers=centers, sigmas=sigmas, gamma=gamma,
        dE_range=dE_range, dE_step=stage1_dE_step,
        dsigma_range=dsigma_range, dsigma_step=stage1_dsigma_step,
        dgamma_range=dgamma_range, dgamma_step=dgamma_step,
    )

    # Solve Dict3D on calibration spectra
    _, _, _, _, dg_per_spectrum, _ = solve_dict3d_parabola(Y_cal, dict3d)

    # Aggregate to global Δγ
    if aggregation == "median":
        global_dg = float(np.median(dg_per_spectrum))
    else:
        global_dg = float(np.mean(dg_per_spectrum))

    dg_std = float(np.std(dg_per_spectrum))
    corrected_gamma = gamma + global_dg

    timing["stage1_dict_build_and_solve"] = time.perf_counter() - t0
    timing["stage1_n_calibration"] = n_cal

    # dict3d is cached — don't delete (shared across calls)

    # --- Stage 2: Dict2D with corrected γ ---
    t0 = time.perf_counter()

    dict2d = build_dictionary(
        energy=energy, centers=centers, sigmas=sigmas,
        gamma=corrected_gamma,
        dE_range=dE_range, dsigma_range=dsigma_range,
    )

    amplitudes, chi2, energy_shifts, sigma_shifts, _ = solve_dict2d_parabola(Y, dict2d)

    timing["stage2_dict_build_and_solve"] = time.perf_counter() - t0
    timing["total"] = sum(v for v in timing.values() if isinstance(v, (int, float)))
    timing["rate"] = n_spectra / timing["total"] if timing["total"] > 0 else float("inf")

    # Expand per-spectrum dg to full size if subset was used
    if cal_idx is not None:
        dg_full = np.full(n_spectra, np.nan, dtype=np.float32)
        dg_full[cal_idx] = dg_per_spectrum
    else:
        dg_full = dg_per_spectrum

    return TwoStageResult(
        amplitudes=amplitudes,
        chi2=chi2,
        energy_shifts=energy_shifts,
        sigma_shifts=sigma_shifts,
        global_dgamma=global_dg,
        dgamma_per_spectrum=dg_full,
        dgamma_std=dg_std,
        n_calibration=n_cal,
        corrected_gamma=corrected_gamma,
        timing=timing,
    )
