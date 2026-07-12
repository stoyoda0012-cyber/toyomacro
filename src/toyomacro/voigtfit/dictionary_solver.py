"""
Dictionary + 4-Step Hybrid Solver
=================================

Combines dictionary-based coarse search (Phase 1) with 4-step residual
projection (Phase 2) for accurate parameter recovery beyond the Taylor
expansion limit.

Architecture:
    Phase 0: Build dictionary of Voigt profiles at discrete (δE, δσ) grid
             + pre-compute Wt, Φ, J_c, J_σ, C at each grid point
    Phase 1: GPU matmul + argmax to find nearest dictionary element
    Phase 2: 4-step residual projection from the nearest grid point
             (Taylor error limited to half-grid spacing)

Key insight: Dictionary absorbs nonlinearity on a discrete grid, then
4-step handles sub-grid residuals where Taylor is always valid.
Grid spacing 0.1σ → Taylor residual < 0.25% (vs 3% at full range).

Performance:
    Phase 1 (dictionary lookup): 1 matmul (N, n_E) @ (n_E, n_dict) + argmax
    Phase 2 (4-step per group): 4 matmuls per grid point group
    Expected: ~10M spec/s total (Phase 1 adds ~1 matmul cost)
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import mlx.core as mx

    from ._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND a Metal device works
except ImportError:
    HAS_MLX = False

# Max GPU buffer for Phase 1 matmul (bytes). Metal limit ~80 GB on M3 Max;
# keep well below to leave room for Phase 2.  Default 4 GB → ~1M spectra
# per chunk at 2610 dict entries.
_PHASE1_MAX_BYTES = 4 * 1024**3


def _chunked_dictionary_lookup(
    Y_mx: 'mx.array',
    D_mx: 'mx.array',
    chunk_bytes: int = _PHASE1_MAX_BYTES,
) -> 'mx.array':
    """Chunked matmul + argmax to avoid Metal OOM on large inputs.

    Returns best_idx (n_spectra,) int32 on CPU (numpy).
    """
    n_spectra = Y_mx.shape[0]
    n_dict = D_mx.shape[1]
    # scores would be (n_spectra, n_dict) float32 → 4 bytes each
    max_rows = max(1, chunk_bytes // (n_dict * 4))
    if max_rows >= n_spectra:
        scores = Y_mx @ D_mx
        best_idx_mx = mx.argmax(scores, axis=1)
        mx.eval(best_idx_mx)
        return np.array(best_idx_mx).astype(np.int32)
    # Chunked path
    best_idx = np.empty(n_spectra, dtype=np.int32)
    for start in range(0, n_spectra, max_rows):
        end = min(start + max_rows, n_spectra)
        chunk = Y_mx[start:end]
        scores = chunk @ D_mx
        idx = mx.argmax(scores, axis=1)
        mx.eval(idx)
        best_idx[start:end] = np.array(idx).astype(np.int32)
    return best_idx


def _group_by_index(
    best_idx: np.ndarray,
    n_dict: int,
) -> list[tuple[np.ndarray, int]]:
    """Group spectrum indices by their dictionary assignment using argsort.

    Replaces repeated ``np.where(best_idx == gidx)`` calls (O(N) each,
    ~50 groups → O(50·N)) with a single O(N log N) argsort + split.

    Args:
        best_idx: (n_spectra,) int32 — dictionary index per spectrum
        n_dict: Total number of dictionary entries (for bincount minlength)

    Returns:
        List of (orig_indices, gidx) for each non-empty group,
        where orig_indices is int32 array of spectrum positions.
    """
    sort_order = np.argsort(best_idx, kind='mergesort').astype(np.int32)
    sorted_idx = best_idx[sort_order]

    # Find group boundaries via bincount + cumsum
    counts = np.bincount(best_idx, minlength=n_dict)
    offsets = np.empty(n_dict + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    groups = []
    active = np.where(counts > 0)[0]
    for gidx in active:
        gidx = int(gidx)
        start = int(offsets[gidx])
        end = int(offsets[gidx + 1])
        groups.append((sort_order[start:end], gidx))
    return groups


# ---------------------------------------------------------------------------
# Parabola sub-grid interpolation
# ---------------------------------------------------------------------------

def _parabola_vertex(s_m: np.ndarray, s_0: np.ndarray, s_p: np.ndarray) -> np.ndarray:
    """Compute parabola vertex offset from center sample.

    Given three equally-spaced samples s_m, s_0, s_p (at positions -1, 0, +1),
    fit a parabola and return the fractional offset of the peak from position 0.

    Returns δ ∈ [-0.5, +0.5] (clamped).  If denominator ≈ 0 returns 0.
    """
    num = s_p - s_m                         # 2a·δ numerator
    den = 2.0 * (2.0 * s_0 - s_m - s_p)    # 2a denominator (= -2a·2)
    # Avoid division by zero (flat or concave neighbourhood)
    safe = np.abs(den) > 1e-30
    with np.errstate(invalid='ignore', divide='ignore'):
        delta = np.where(safe, num / den, 0.0)
    # Clamp to ±0.5 — beyond that the vertex is outside [i-1, i+1]
    return np.clip(delta, -0.5, 0.5)


def _parabola_vertex_nonuniform(
    s_m: np.ndarray, s_0: np.ndarray, s_p: np.ndarray,
    h_L: np.ndarray, h_R: np.ndarray,
) -> np.ndarray:
    """Parabola vertex for non-uniform grid spacing.

    Fits a quadratic through three points at positions (-h_L, 0, +h_R)
    with values (s_m, s_0, s_p).  Returns the vertex offset in real
    coordinate units (not fractional), clamped to [-h_L, +h_R].

    Lagrange interpolation on unequal spacing:
        s(x) = s_m·L_m(x) + s_0·L_0(x) + s_p·L_p(x)

    Vertex at x* = -b/(2a) where a, b are the quadratic coefficients.

    When h_L == h_R (uniform), this reduces to the standard formula
    with shift = h * (s_p - s_m) / (2*(2*s_0 - s_m - s_p)).

    Args:
        s_m, s_0, s_p: Score values at left, center, right neighbours.
        h_L: Left spacing  grid[i] - grid[i-1], per spectrum.
        h_R: Right spacing grid[i+1] - grid[i], per spectrum.

    Returns:
        shift: Vertex offset in real coordinates, clamped to [-h_L, +h_R].
    """
    # Quadratic coefficients from Lagrange interpolation on (-h_L, 0, +h_R)
    #   a = [h_L·s_p - (h_L+h_R)·s_0 + h_R·s_m] / [h_L·h_R·(h_L+h_R)]
    #   b = [h_L²·s_p - (h_L²-h_R²)·s_0 - h_R²·s_m] / [h_L·h_R·(h_L+h_R)]
    # Vertex: x* = -b/(2a)
    h_sum = h_L + h_R
    denom_common = h_L * h_R * h_sum  # common denominator

    # Numerator of -b/(2a) simplifies to:
    #   num = -(h_L²·s_p - (h_L²-h_R²)·s_0 - h_R²·s_m)
    #   den = 2·(h_L·s_p - (h_L+h_R)·s_0 + h_R·s_m)
    # (denom_common cancels out in the ratio)
    h_L2 = h_L * h_L
    h_R2 = h_R * h_R
    num = -(h_L2 * s_p - (h_L2 - h_R2) * s_0 - h_R2 * s_m)
    den = 2.0 * (h_L * s_p - h_sum * s_0 + h_R * s_m)

    safe = np.abs(den) > 1e-30
    with np.errstate(invalid='ignore', divide='ignore'):
        shift = np.where(safe, num / den, 0.0)

    # Clamp to [-h_L, +h_R]
    shift = np.maximum(shift, -h_L)
    shift = np.minimum(shift, h_R)
    return shift


def parabola_refine_2d(
    scores: np.ndarray,
    best_idx: np.ndarray,
    grid_shape: tuple[int, int],
    dE_grid: np.ndarray,
    dsigma_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-grid parabola interpolation on a 2D (dE × dσ) score grid.

    For each spectrum, takes the argmax grid point and fits independent 1D
    parabolas along the dE and dσ axes to refine the coarse grid values.

    Args:
        scores: (n_spectra, n_dict) similarity scores from Y @ D.
        best_idx: (n_spectra,) int32 flat index of argmax.
        grid_shape: (n_dE, n_dsigma).
        dE_grid: (n_dE,) float32 grid values.
        dsigma_grid: (n_dsigma,) float32 grid values.

    Returns:
        (dE_refined, dsigma_refined) each (n_spectra,) float32.
    """
    n_dE, n_ds = grid_shape
    n_spectra = len(best_idx)

    # Unflatten to 2D indices
    dE_idx = best_idx // n_ds
    ds_idx = best_idx % n_ds

    # Interior masks
    dE_interior = (dE_idx > 0) & (dE_idx < n_dE - 1)
    ds_interior = (ds_idx > 0) & (ds_idx < n_ds - 1)

    # Neighbour flat indices
    spec_range = np.arange(n_spectra)
    s0_dE = scores[spec_range, best_idx]
    dE_m_idx = np.clip(dE_idx - 1, 0, n_dE - 1) * n_ds + ds_idx
    dE_p_idx = np.clip(dE_idx + 1, 0, n_dE - 1) * n_ds + ds_idx
    sm_dE = scores[spec_range, dE_m_idx]
    sp_dE = scores[spec_range, dE_p_idx]

    ds_m_idx = dE_idx * n_ds + np.clip(ds_idx - 1, 0, n_ds - 1)
    ds_p_idx = dE_idx * n_ds + np.clip(ds_idx + 1, 0, n_ds - 1)
    sm_ds = scores[spec_range, ds_m_idx]
    sp_ds = scores[spec_range, ds_p_idx]

    from .grids import is_uniform, precompute_grid_spacings

    # --- dE axis parabola ---
    if is_uniform(dE_grid):
        dE_step = dE_grid[1] - dE_grid[0] if n_dE > 1 else 1.0
        delta_dE = _parabola_vertex(sm_dE, s0_dE, sp_dE)
        delta_dE = np.where(dE_interior, delta_dE, 0.0)
        dE_refined = dE_grid[dE_idx] + delta_dE * dE_step
    else:
        hL_dE, hR_dE = precompute_grid_spacings(dE_grid)
        shift_dE = _parabola_vertex_nonuniform(
            sm_dE, s0_dE, sp_dE, hL_dE[dE_idx], hR_dE[dE_idx])
        shift_dE = np.where(dE_interior, shift_dE, 0.0)
        dE_refined = dE_grid[dE_idx] + shift_dE

    # --- dsigma axis parabola ---
    if is_uniform(dsigma_grid):
        ds_step = dsigma_grid[1] - dsigma_grid[0] if n_ds > 1 else 1.0
        delta_ds = _parabola_vertex(sm_ds, s0_dE, sp_ds)
        delta_ds = np.where(ds_interior, delta_ds, 0.0)
        dsigma_refined = dsigma_grid[ds_idx] + delta_ds * ds_step
    else:
        hL_ds, hR_ds = precompute_grid_spacings(dsigma_grid)
        shift_ds = _parabola_vertex_nonuniform(
            sm_ds, s0_dE, sp_ds, hL_ds[ds_idx], hR_ds[ds_idx])
        shift_ds = np.where(ds_interior, shift_ds, 0.0)
        dsigma_refined = dsigma_grid[ds_idx] + shift_ds

    return dE_refined.astype(np.float32), dsigma_refined.astype(np.float32)


@dataclass
class DictionaryCache:
    """Pre-computed dictionary and per-grid-point matrices.

    Attributes:
        D: L2-normalized dictionary matrix (n_energy, n_dict)
        dE_grid: δE grid values (n_dE,)
        dsigma_grid: δσ grid values (n_dsigma,)
        grid_shape: (n_dE, n_dsigma)
        Wt_per_grid: Weight matrices (n_dict, n_comp, n_energy)
        Phi_per_grid: Basis matrices (n_dict, n_energy, n_comp)
        Jc_per_grid: Center Jacobians (n_dict, n_energy, n_comp)
        Jsigma_per_grid: Sigma Jacobians (n_dict, n_energy, n_comp)
        C_per_grid: Coupling matrices C_σ = (Φ'Φ)⁻¹Φ'J_σ (n_dict, n_comp, n_comp)
        Hcc_per_grid: Hessian ∂²V/∂c² matrices (n_dict, n_energy, n_comp) or None
        Cc_per_grid: Center coupling C_c = (Φ'Φ)⁻¹Φ'J_c (n_dict, n_comp, n_comp) or None
        nominal_index: Index of the (δE=0, δσ=0) grid point
        mean_sigma: Mean σ of the peak components (for Taylor threshold)
    """
    D: np.ndarray
    dE_grid: np.ndarray
    dsigma_grid: np.ndarray
    grid_shape: tuple[int, int]
    Wt_per_grid: np.ndarray
    Phi_per_grid: np.ndarray
    Jc_per_grid: np.ndarray
    Jsigma_per_grid: np.ndarray
    C_per_grid: np.ndarray
    nominal_index: int = 0
    mean_sigma: float = 1.0
    Hcc_per_grid: np.ndarray | None = None
    Cc_per_grid: np.ndarray | None = None
    # Nominal model parameters (for exact-Jacobian Newton re-evaluation).
    # The absolute peak parameters of a refined fit are:
    #     center_k = nominal_centers[k] + δE,  sigma_k = nominal_sigmas[k] + δσ
    # gamma is shared; so_split / branch_ratio describe an optional SO doublet
    # partner peak (partner at center+so_split, intensity = main / branch_ratio).
    energy: np.ndarray | None = None
    nominal_centers: np.ndarray | None = None
    nominal_sigmas: np.ndarray | None = None
    gamma: float = 0.0
    so_split: float = 0.0
    branch_ratio: float = 1.0
    # MLX versions (lazy, created on first use)
    _D_mx: Any | None = None
    _Wt_mx: Any | None = None
    _Phi_mx: Any | None = None
    _Jc_mx: Any | None = None
    _Jsigma_mx: Any | None = None
    _Hcc_mx: Any | None = None

    @property
    def n_dict(self) -> int:
        return self.D.shape[1]

    @property
    def n_dE(self) -> int:
        return self.grid_shape[0]

    @property
    def n_dsigma(self) -> int:
        return self.grid_shape[1]

    def is_within_taylor(self, dict_indices: np.ndarray,
                         taylor_threshold: float = 0.35) -> np.ndarray:
        """Check if spectra are within Taylor-valid range of nominal.

        A spectrum is within Taylor range if its coarse δE and δσ satisfy
        |δE|/σ < threshold AND |δσ|/σ < threshold.

        Args:
            dict_indices: (n_spectra,) grid indices from Phase 1
            taylor_threshold: Max |δ/σ| for Taylor validity (default 0.35)

        Returns:
            Boolean mask (n_spectra,) — True if within Taylor range
        """
        n_dsigma = self.grid_shape[1]
        dE_idx = dict_indices // n_dsigma
        ds_idx = dict_indices % n_dsigma
        dE = self.dE_grid[dE_idx]
        ds = self.dsigma_grid[ds_idx]
        return (np.abs(dE) / self.mean_sigma < taylor_threshold) & \
               (np.abs(ds) / self.mean_sigma < taylor_threshold)

    @property
    def has_hessian(self) -> bool:
        return self.Hcc_per_grid is not None

    def to_mlx(self):
        """Convert arrays to MLX (lazy, cached)."""
        if not HAS_MLX:
            return
        if self._D_mx is None:
            self._D_mx = mx.array(self.D)
            self._Wt_mx = mx.array(self.Wt_per_grid)
            self._Phi_mx = mx.array(self.Phi_per_grid)
            self._Jc_mx = mx.array(self.Jc_per_grid)
            self._Jsigma_mx = mx.array(self.Jsigma_per_grid)
            if self.Hcc_per_grid is not None and self._Hcc_mx is None:
                self._Hcc_mx = mx.array(self.Hcc_per_grid)


def build_dictionary(
    energy: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gamma: float,
    dE_range: tuple[float, float] = (-2.0, 2.0),
    dE_step: float | None = None,
    dsigma_range: tuple[float, float] = (-0.3, 0.3),
    dsigma_step: float | None = None,
    reg_lambda: float = 1e-8,
    include_hessian: bool = False,
    dE_grid_override: np.ndarray | None = None,
    dsigma_grid_override: np.ndarray | None = None,
    so_split: float = 0.0,
    branch_ratio: float = 1.0,
) -> DictionaryCache:
    """Build Voigt profile dictionary and per-grid-point matrices.

    For multi-component peaks, the dictionary uses a composite profile
    (sum of all components) for matching, but each grid point has its
    own per-component Φ, J_c, J_σ, W, C matrices.

    Args:
        energy: Energy axis (n_energy,)
        centers: Nominal peak centers (n_comp,)
        sigmas: Nominal Gaussian widths (n_comp,)
        gamma: Lorentzian width (shared)
        dE_range: δE search range in eV (default ±2.0)
        dE_step: δE grid spacing in eV (default 0.1 * mean_sigma)
        dsigma_range: δσ search range in eV (default ±0.3)
        dsigma_step: δσ grid spacing in eV (default 0.1 * mean_sigma)
        reg_lambda: Tikhonov regularization
        dE_grid_override: Pre-built δE grid (overrides dE_range/dE_step).
            Can be non-uniform (e.g. from ``grids.sinh_grid``).
        dsigma_grid_override: Pre-built δσ grid (overrides dsigma_range/dsigma_step).

    Returns:
        DictionaryCache with all pre-computed matrices
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
        dsigma_step = 0.1 * mean_sigma

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

    # Clip grids to ensure sigma stays positive
    min_sigma = float(np.min(sigmas))
    dsigma_grid = dsigma_grid[dsigma_grid > -0.9 * min_sigma]

    n_dE = len(dE_grid)
    n_dsigma = len(dsigma_grid)
    n_dict = n_dE * n_dsigma

    # Import Hessian if needed
    if include_hessian:
        from .voigt_jacobian import voigt_hessian_batch

    # Allocate per-grid-point arrays
    D = np.zeros((n_energy, n_dict), dtype=np.float32)
    Wt_per_grid = np.zeros((n_dict, n_comp, n_energy), dtype=np.float32)
    Phi_per_grid = np.zeros((n_dict, n_energy, n_comp), dtype=np.float32)
    Jc_per_grid = np.zeros((n_dict, n_energy, n_comp), dtype=np.float32)
    Jsigma_per_grid = np.zeros((n_dict, n_energy, n_comp), dtype=np.float32)
    C_per_grid = np.zeros((n_dict, n_comp, n_comp), dtype=np.float32)
    Hcc_per_grid = np.zeros((n_dict, n_energy, n_comp), dtype=np.float32) if include_hessian else None
    Cc_per_grid = np.zeros((n_dict, n_comp, n_comp), dtype=np.float32) if include_hessian else None

    is_doublet = so_split != 0.0

    for i, dE in enumerate(dE_grid):
        for j, ds in enumerate(dsigma_grid):
            idx = i * n_dsigma + j

            # Shifted parameters
            shifted_centers = centers + dE
            shifted_sigmas = sigmas + ds

            if include_hessian:
                # Build Voigt basis + Jacobian + Hessian at this grid point
                Phi, J_c, J_sigma, _, H_cc = voigt_hessian_batch(
                    energy, shifted_centers, shifted_sigmas, gamma
                )
                H_cc = H_cc.astype(np.float32)
            else:
                # Build Voigt basis + Jacobian at this grid point
                Phi, J_c, J_sigma, _ = voigt_jacobian_batch(
                    energy, shifted_centers, shifted_sigmas, gamma
                )
            Phi = Phi.astype(np.float32)
            J_c = J_c.astype(np.float32)
            J_sigma = J_sigma.astype(np.float32)

            if is_doublet:
                # SO doublet: add partner peak contribution to each component.
                # Partner at center + so_split, intensity = main / branch_ratio.
                partner_centers = shifted_centers + so_split
                partner_sigmas = shifted_sigmas  # shared width
                if include_hessian:
                    Phi_p, Jc_p, Js_p, _, Hcc_p = voigt_hessian_batch(
                        energy, partner_centers, partner_sigmas, gamma
                    )
                    Hcc_p = Hcc_p.astype(np.float32)
                    H_cc = H_cc + Hcc_p / branch_ratio
                else:
                    Phi_p, Jc_p, Js_p, _ = voigt_jacobian_batch(
                        energy, partner_centers, partner_sigmas, gamma
                    )
                r = np.float32(1.0 / branch_ratio)
                Phi = Phi + r * Phi_p.astype(np.float32)
                J_c = J_c + r * Jc_p.astype(np.float32)
                J_sigma = J_sigma + r * Js_p.astype(np.float32)

            # Dictionary column: composite profile (sum of all components)
            composite = Phi.sum(axis=1)  # (n_energy,)
            norm = np.linalg.norm(composite)
            if norm > 1e-10:
                D[:, idx] = composite / norm
            else:
                D[:, idx] = 0.0

            # Weight matrix at this grid point
            AtA = Phi.T @ Phi + reg_lambda * np.eye(n_comp, dtype=np.float32)
            Wt = np.linalg.solve(AtA, Phi.T).astype(np.float32)

            # Coupling matrix C_σ = (Φ'Φ)⁻¹Φ'J_σ
            C = np.linalg.solve(AtA, Phi.T @ J_sigma).astype(np.float32)

            Wt_per_grid[idx] = Wt
            Phi_per_grid[idx] = Phi
            Jc_per_grid[idx] = J_c
            Jsigma_per_grid[idx] = J_sigma
            C_per_grid[idx] = C

            if include_hessian:
                Hcc_per_grid[idx] = H_cc
                # Center coupling C_c = (Φ'Φ)⁻¹Φ'J_c
                Cc_per_grid[idx] = np.linalg.solve(AtA, Phi.T @ J_c).astype(np.float32)

    # Find nominal index (δE≈0, δσ≈0)
    nom_dE_idx = int(np.argmin(np.abs(dE_grid)))
    nom_ds_idx = int(np.argmin(np.abs(dsigma_grid)))
    nominal_index = nom_dE_idx * n_dsigma + nom_ds_idx

    return DictionaryCache(
        D=D,
        dE_grid=dE_grid,
        dsigma_grid=dsigma_grid,
        grid_shape=(n_dE, n_dsigma),
        Wt_per_grid=Wt_per_grid,
        Phi_per_grid=Phi_per_grid,
        Jc_per_grid=Jc_per_grid,
        Jsigma_per_grid=Jsigma_per_grid,
        C_per_grid=C_per_grid,
        Hcc_per_grid=Hcc_per_grid,
        Cc_per_grid=Cc_per_grid,
        nominal_index=nominal_index,
        mean_sigma=mean_sigma,
        energy=energy,
        nominal_centers=centers,
        nominal_sigmas=sigmas,
        gamma=float(gamma),
        so_split=float(so_split),
        branch_ratio=float(branch_ratio),
    )


def dictionary_lookup_numpy(
    D: np.ndarray,
    Y: np.ndarray,
    grid_shape: tuple[int, int],
    dE_grid: np.ndarray,
    dsigma_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find nearest dictionary element for each spectrum (NumPy path).

    Args:
        D: L2-normalized dictionary (n_energy, n_dict)
        Y: Spectra (n_spectra, n_energy)
        grid_shape: (n_dE, n_dsigma)
        dE_grid: δE values (n_dE,)
        dsigma_grid: δσ values (n_dsigma,)

    Returns:
        (best_indices, dE_coarse, dsigma_coarse)
    """
    scores = Y @ D  # (n_spectra, n_dict)
    best_flat = np.argmax(scores, axis=1)  # (n_spectra,)

    # Unflatten to (dE_idx, dsigma_idx)
    n_dsigma = grid_shape[1]
    dE_idx = best_flat // n_dsigma
    ds_idx = best_flat % n_dsigma

    dE_coarse = dE_grid[dE_idx]
    dsigma_coarse = dsigma_grid[ds_idx]

    return best_flat.astype(np.int32), dE_coarse, dsigma_coarse


def dictionary_lookup_mlx(
    D_mx: 'mx.array',
    Y_mx: 'mx.array',
    grid_shape: tuple[int, int],
    dE_grid: np.ndarray,
    dsigma_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find nearest dictionary element (MLX path, chunked for large inputs)."""
    best_flat_np = _chunked_dictionary_lookup(Y_mx, D_mx)

    n_dsigma = grid_shape[1]
    dE_idx = best_flat_np // n_dsigma
    ds_idx = best_flat_np % n_dsigma

    dE_coarse = dE_grid[dE_idx]
    dsigma_coarse = dsigma_grid[ds_idx]

    return best_flat_np, dE_coarse, dsigma_coarse


def _four_step_for_group(
    Y_group: np.ndarray,
    Wt: np.ndarray,
    Phi: np.ndarray,
    J_c: np.ndarray,
    J_sigma: np.ndarray,
    C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run 4-step residual projection on a group of spectra (NumPy).

    All spectra in the group share the same grid point matrices.

    Returns:
        (amplitudes, chi2, dE_fine, dsigma_fine, amp_corrected)
        where amplitudes is (n_comp, n_group) AFTER correction.
    """
    n_group = Y_group.shape[0]
    n_energy = Y_group.shape[1]
    n_comp = Wt.shape[0]

    # Step 1: amplitude
    A_row = Y_group @ Wt.T  # (n_group, n_comp)

    # Step 2: shift
    R = Y_group - A_row @ Phi.T  # (n_group, n_energy)
    S_c = A_row @ J_c.T          # (n_group, n_energy)
    num_c = np.sum(R * S_c, axis=1)
    den_c = np.sum(S_c * S_c, axis=1)
    safe_den_c = np.where(den_c > 1e-20, den_c, 1.0)
    dE_fine = np.where(den_c > 1e-20, num_c / safe_den_c, 0.0).astype(np.float32)

    # Step 3: width (cross-talk corrected)
    S_s = A_row @ J_sigma.T
    RS_s = np.sum(R * S_s, axis=1)
    ScSs = np.sum(S_c * S_s, axis=1)
    den_s = np.sum(S_s * S_s, axis=1)
    num_s = RS_s - dE_fine * ScSs
    safe_den_s = np.where(den_s > 1e-20, den_s, 1.0)
    dsigma_fine = np.where(den_s > 1e-20, num_s / safe_den_s, 0.0).astype(np.float32)

    # Chi2
    chi2 = (np.sum(R * R, axis=1) / n_energy).astype(np.float32)

    # Step 4: amplitude correction
    amplitudes = A_row.T  # (n_comp, n_group)
    if n_comp == 1:
        correction = 1.0 + dsigma_fine * C[0, 0]
        correction = np.where(np.abs(correction) > 1e-6, correction, 1.0)
        amplitudes = amplitudes / correction[np.newaxis, :]
    elif n_comp == 2:
        ds = dsigma_fine
        d = (1 + ds * C[0, 0]) * (1 + ds * C[1, 1]) - ds * ds * C[0, 1] * C[1, 0]
        d = np.where(np.abs(d) > 1e-12, d, 1.0)
        inv_d = 1.0 / d
        a0 = amplitudes[0, :].copy()
        a1 = amplitudes[1, :].copy()
        amplitudes[0, :] = ((1 + ds * C[1, 1]) * a0 - ds * C[0, 1] * a1) * inv_d
        amplitudes[1, :] = (-ds * C[1, 0] * a0 + (1 + ds * C[0, 0]) * a1) * inv_d
    else:
        I_mat = np.eye(n_comp, dtype=np.float32)
        M = I_mat[np.newaxis, :, :] + dsigma_fine[:, np.newaxis, np.newaxis] * C[np.newaxis, :, :]
        a_col = amplitudes.T[:, :, np.newaxis]
        a_corr = np.linalg.solve(M, a_col)
        amplitudes = a_corr[:, :, 0].T

    return amplitudes, chi2, dE_fine, dsigma_fine


def _six_step_for_group(
    Y_group: np.ndarray,
    Wt: np.ndarray,
    Phi: np.ndarray,
    J_c: np.ndarray,
    J_sigma: np.ndarray,
    H_cc: np.ndarray,
    C_sigma: np.ndarray,
    C_c: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run 6-step residual projection on a group of spectra (NumPy).

    Extends 4-step with Hessian d²V/dc² correction for δσ (Step 2.5)
    and optional C_c augmented amplitude correction (Step 3.5).

    Returns:
        (amplitudes, chi2, dE_fine, dsigma_fine)
        where amplitudes is (n_comp, n_group) AFTER correction.
    """
    n_group = Y_group.shape[0]
    n_energy = Y_group.shape[1]
    n_comp = Wt.shape[0]

    # Step 1: amplitude
    A_row = Y_group @ Wt.T

    # Step 2: shift
    R = Y_group - A_row @ Phi.T
    S_c = A_row @ J_c.T
    num_c = np.sum(R * S_c, axis=1)
    den_c = np.sum(S_c * S_c, axis=1)
    safe_den_c = np.where(den_c > 1e-20, den_c, 1.0)
    dE_fine = np.where(den_c > 1e-20, num_c / safe_den_c, 0.0).astype(np.float32)

    # Step 2.5 + Step 3: Hessian-corrected width
    S_s = A_row @ J_sigma.T
    H = A_row @ H_cc.T
    RS_s = np.sum(R * S_s, axis=1)
    ScSs = np.sum(S_c * S_s, axis=1)
    den_s = np.sum(S_s * S_s, axis=1)
    hcc_ss = np.sum(H * S_s, axis=1)
    correction = 0.5 * dE_fine * dE_fine * hcc_ss
    num_s = RS_s - dE_fine * ScSs - correction
    safe_den_s = np.where(den_s > 1e-20, den_s, 1.0)
    dsigma_fine = np.where(den_s > 1e-20, num_s / safe_den_s, 0.0).astype(np.float32)

    # Chi2
    chi2 = (np.sum(R * R, axis=1) / n_energy).astype(np.float32)

    # Step 3.5 + Step 4: amplitude correction with optional C_c
    amplitudes = A_row.T
    if n_comp == 1:
        corr = 1.0 + dsigma_fine * C_sigma[0, 0]
        corr = np.where(np.abs(corr) > 1e-6, corr, 1.0)
        amplitudes = amplitudes / corr[np.newaxis, :]
    elif n_comp == 2:
        ds = dsigma_fine
        M00 = 1 + ds * C_sigma[0, 0]
        M01 = ds * C_sigma[0, 1]
        M10 = ds * C_sigma[1, 0]
        M11 = 1 + ds * C_sigma[1, 1]
        if C_c is not None:
            de = dE_fine
            M00 = M00 + de * C_c[0, 0]
            M01 = M01 + de * C_c[0, 1]
            M10 = M10 + de * C_c[1, 0]
            M11 = M11 + de * C_c[1, 1]
        d = M00 * M11 - M01 * M10
        d = np.where(np.abs(d) > 1e-12, d, 1.0)
        inv_d = 1.0 / d
        a0 = amplitudes[0, :].copy()
        a1 = amplitudes[1, :].copy()
        amplitudes[0, :] = (M11 * a0 - M01 * a1) * inv_d
        amplitudes[1, :] = (-M10 * a0 + M00 * a1) * inv_d
    else:
        I_mat = np.eye(n_comp, dtype=np.float32)
        M = I_mat[np.newaxis, :, :] + dsigma_fine[:, np.newaxis, np.newaxis] * C_sigma[np.newaxis, :, :]
        if C_c is not None:
            M = M + dE_fine[:, np.newaxis, np.newaxis] * C_c[np.newaxis, :, :]
        a_col = amplitudes.T[:, :, np.newaxis]
        a_corr = np.linalg.solve(M, a_col)
        amplitudes = a_corr[:, :, 0].T

    return amplitudes, chi2, dE_fine, dsigma_fine


def solve_dictionary_only(
    Y: np.ndarray,
    dict_cache: DictionaryCache,
    use_mlx: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dictionary-only solver: lookup + linear fit at grid point (no 4-step).

    Fastest path: 1 matmul (dictionary) + argmax + 1 matmul (LLS at grid point).
    Skips the 4-step refinement — δE/δσ are quantized to grid resolution.

    For n_dict=68 (shift-only), ~2-3x faster than full hybrid.

    Args:
        Y: Spectra (n_spectra, n_energy) float32
        dict_cache: Pre-built dictionary cache
        use_mlx: Use MLX for lookup

    Returns:
        (amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices)
    """
    Y = np.asarray(Y, dtype=np.float32)
    n_spectra = Y.shape[0]
    n_energy = Y.shape[1]
    n_comp = dict_cache.Wt_per_grid.shape[1]

    # Phase 1: Dictionary lookup
    if use_mlx and HAS_MLX:
        dict_cache.to_mlx()
        Y_mx = mx.array(Y)
        best_idx, dE_coarse, dsigma_coarse = dictionary_lookup_mlx(
            dict_cache._D_mx, Y_mx, dict_cache.grid_shape,
            dict_cache.dE_grid, dict_cache.dsigma_grid,
        )
    else:
        best_idx, dE_coarse, dsigma_coarse = dictionary_lookup_numpy(
            dict_cache.D, Y, dict_cache.grid_shape,
            dict_cache.dE_grid, dict_cache.dsigma_grid,
        )

    # Sort + contiguous group LLS (Step 1 only, no 4-step refinement)
    sort_order = np.argsort(best_idx, kind='mergesort')
    sorted_idx = best_idx[sort_order]
    Y_sorted = Y[sort_order]

    amplitudes_sorted = np.zeros((n_comp, n_spectra), dtype=np.float32)
    chi2_sorted = np.zeros(n_spectra, dtype=np.float32)

    change_pts = np.where(np.diff(sorted_idx) != 0)[0] + 1
    starts = np.concatenate([[0], change_pts])
    ends = np.concatenate([change_pts, [n_spectra]])

    for s, e in zip(starts, ends):
        grid_idx = sorted_idx[s]
        Wt = dict_cache.Wt_per_grid[grid_idx]
        Phi = dict_cache.Phi_per_grid[grid_idx]
        Y_group = Y_sorted[s:e]
        A_row = Y_group @ Wt.T
        R = Y_group - A_row @ Phi.T
        amplitudes_sorted[:, s:e] = A_row.T
        chi2_sorted[s:e] = np.sum(R * R, axis=1) / n_energy

    unsort = np.argsort(sort_order)
    amplitudes = amplitudes_sorted[:, unsort]
    chi2 = chi2_sorted[unsort]

    return amplitudes, chi2, dE_coarse, dsigma_coarse, best_idx


def _solve_hybrid_mlx_gather(
    Y_mx: 'mx.array',
    best_idx_mx: 'mx.array',
    dict_cache: DictionaryCache,
) -> tuple['mx.array', 'mx.array', 'mx.array', 'mx.array']:
    """GPU-native Phase 2: gather per-spectrum Wt/Phi rows + batched ops.

    Instead of grouping by grid index on CPU, we gather matrices from
    the pre-computed cache using the dictionary index, then do element-wise
    operations. For n_comp=1, this avoids explicit matmul by using
    element-wise multiply + sum.

    Returns:
        (amplitudes_row, chi2, num_c, den_c) all as mx.array
    """
    n_spectra = Y_mx.shape[0]
    n_energy = Y_mx.shape[1]

    # Gather per-spectrum matrices: Wt[idx] shape (N, n_comp, n_E)
    Wt_batch = dict_cache._Wt_mx[best_idx_mx]   # (N, n_comp, n_E)
    Phi_batch = dict_cache._Phi_mx[best_idx_mx]  # (N, n_E, n_comp)
    Jc_batch = dict_cache._Jc_mx[best_idx_mx]    # (N, n_E, n_comp)

    # Step 1: amplitude — a_i = Wt_i @ y_i
    # (N, n_comp, n_E) @ (N, n_E, 1) → (N, n_comp, 1) → (N, n_comp)
    Y_col = mx.expand_dims(Y_mx, axis=2)  # (N, n_E, 1)
    A_row = mx.squeeze(Wt_batch @ Y_col, axis=2)  # (N, n_comp)

    # Step 2: residual + shift projection
    # recon = sum_k a_k * Phi_k[e] = (N,1,n_comp) @ (N,n_comp,n_E) → but Phi is (N,n_E,n_comp)
    # R = Y - A @ Phi.T per spectrum
    A_row_ex = mx.expand_dims(A_row, axis=1)  # (N, 1, n_comp)
    Phi_T = mx.transpose(Phi_batch, axes=(0, 2, 1))  # (N, n_comp, n_E)
    recon = mx.squeeze(A_row_ex @ Phi_T, axis=1)  # (N, n_E)
    R = Y_mx - recon

    # S_c = A @ J_c.T
    Jc_T = mx.transpose(Jc_batch, axes=(0, 2, 1))
    S_c = mx.squeeze(A_row_ex @ Jc_T, axis=1)  # (N, n_E)

    num_c = mx.sum(R * S_c, axis=1)
    den_c = mx.sum(S_c * S_c, axis=1)
    chi2 = mx.sum(R * R, axis=1) * (1.0 / n_energy)

    return A_row, chi2, num_c, den_c


def solve_hybrid_mlx(
    Y: np.ndarray,
    dict_cache: DictionaryCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GPU-native dictionary + 2-step hybrid solver.

    All computation on MLX GPU via gather + batched matmul.
    No CPU for-loop over groups, no sort/unsort.

    Phase 1: Dictionary lookup (matmul + argmax) on GPU.
    Phase 2: Gather per-spectrum Wt/Phi/J_c, batched matmul for
             amplitude + shift recovery.

    Note: Only does 2-step (amplitude + shift), not full 4-step
    (width + amplitude correction). For full 4-step, use solve_hybrid().

    Returns:
        (amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices)
    """
    if not HAS_MLX:
        raise RuntimeError("MLX required for solve_hybrid_mlx")

    Y = np.asarray(Y, dtype=np.float32)
    n_spectra = Y.shape[0]
    n_comp = dict_cache.Wt_per_grid.shape[1]

    dict_cache.to_mlx()
    Y_mx = mx.array(Y)

    # Phase 1: Dictionary lookup
    scores = Y_mx @ dict_cache._D_mx
    best_idx_mx = mx.argmax(scores, axis=1)
    mx.eval(best_idx_mx)
    best_idx_np = np.array(best_idx_mx).astype(np.int32)

    n_dsigma = dict_cache.grid_shape[1]
    dE_idx = best_idx_np // n_dsigma
    ds_idx = best_idx_np % n_dsigma
    dE_coarse = dict_cache.dE_grid[dE_idx]
    dsigma_coarse = dict_cache.dsigma_grid[ds_idx]

    # Phase 2: GPU gather + batched matmul
    A_row, chi2_mx, num_c, den_c = _solve_hybrid_mlx_gather(
        Y_mx, best_idx_mx, dict_cache,
    )
    mx.eval(A_row, chi2_mx, num_c, den_c)

    # CPU post-processing
    A_np = np.array(A_row)        # (N, n_comp)
    chi2 = np.array(chi2_mx)
    num_c_np = np.array(num_c)
    den_c_np = np.array(den_c)

    safe_den = np.where(den_c_np > 1e-20, den_c_np, 1.0)
    dE_fine = np.where(den_c_np > 1e-20, num_c_np / safe_den, 0.0).astype(np.float32)

    amplitudes = A_np.T  # (n_comp, N)
    energy_shifts = (dE_coarse + dE_fine).astype(np.float32)
    sigma_shifts = dsigma_coarse.astype(np.float32)  # no fine σ correction in 2-step

    return amplitudes, chi2, energy_shifts, sigma_shifts, best_idx_np


def solve_hybrid(
    Y: np.ndarray,
    dict_cache: DictionaryCache,
    use_mlx: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dictionary + 4-step hybrid solver.

    Phase 1: Dictionary lookup (1 matmul + argmax) for coarse (δE, δσ).
    Phase 2: 4-step refinement at nearest grid point for fine residuals.

    Optimization: spectra are sorted by grid index so each group is a
    contiguous slice — no fancy indexing or np.where per group.

    All steps are direct (no iteration). Total: ~5 matmuls + argmax.

    Args:
        Y: Spectra (n_spectra, n_energy) float32
        dict_cache: Pre-built dictionary cache
        use_mlx: Use MLX for Phase 1 dictionary lookup

    Returns:
        (amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices)
        - amplitudes: (n_comp, n_spectra)
        - chi2: (n_spectra,)
        - energy_shifts: (n_spectra,) total δE = coarse + fine
        - sigma_shifts: (n_spectra,) total δσ = coarse + fine
        - dict_indices: (n_spectra,) which grid point was used
    """
    Y = np.asarray(Y, dtype=np.float32)
    n_spectra = Y.shape[0]
    n_comp = dict_cache.Wt_per_grid.shape[1]

    # Phase 1: Dictionary lookup
    if use_mlx and HAS_MLX:
        dict_cache.to_mlx()
        Y_mx = mx.array(Y)
        best_idx, dE_coarse, dsigma_coarse = dictionary_lookup_mlx(
            dict_cache._D_mx, Y_mx, dict_cache.grid_shape,
            dict_cache.dE_grid, dict_cache.dsigma_grid,
        )
    else:
        best_idx, dE_coarse, dsigma_coarse = dictionary_lookup_numpy(
            dict_cache.D, Y, dict_cache.grid_shape,
            dict_cache.dE_grid, dict_cache.dsigma_grid,
        )

    # Phase 2: Sort by grid index for contiguous group processing
    sort_order = np.argsort(best_idx, kind='mergesort')
    sorted_idx = best_idx[sort_order]
    Y_sorted = Y[sort_order]

    amplitudes_sorted = np.zeros((n_comp, n_spectra), dtype=np.float32)
    chi2_sorted = np.zeros(n_spectra, dtype=np.float32)
    dE_fine_sorted = np.zeros(n_spectra, dtype=np.float32)
    dsigma_fine_sorted = np.zeros(n_spectra, dtype=np.float32)

    # Find group boundaries in sorted array
    change_pts = np.where(np.diff(sorted_idx) != 0)[0] + 1
    starts = np.concatenate([[0], change_pts])
    ends = np.concatenate([change_pts, [n_spectra]])

    for s, e in zip(starts, ends):
        grid_idx = sorted_idx[s]
        Y_group = Y_sorted[s:e]  # contiguous slice, no copy

        amp_g, chi2_g, dE_g, ds_g = _four_step_for_group(
            Y_group,
            dict_cache.Wt_per_grid[grid_idx],
            dict_cache.Phi_per_grid[grid_idx],
            dict_cache.Jc_per_grid[grid_idx],
            dict_cache.Jsigma_per_grid[grid_idx],
            dict_cache.C_per_grid[grid_idx],
        )

        amplitudes_sorted[:, s:e] = amp_g
        chi2_sorted[s:e] = chi2_g
        dE_fine_sorted[s:e] = dE_g
        dsigma_fine_sorted[s:e] = ds_g

    # Unsort back to original order
    unsort = np.argsort(sort_order)
    dE_fine = dE_fine_sorted[unsort]
    dsigma_fine = dsigma_fine_sorted[unsort]
    amplitudes = amplitudes_sorted[:, unsort]
    chi2 = chi2_sorted[unsort]

    # Total = coarse + fine
    energy_shifts = (dE_coarse + dE_fine).astype(np.float32)
    sigma_shifts = (dsigma_coarse + dsigma_fine).astype(np.float32)

    return amplitudes, chi2, energy_shifts, sigma_shifts, best_idx


def build_dictionary_shift_only(
    energy: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gamma: float,
    dE_range: tuple[float, float] = (-2.0, 2.0),
    dE_step: float | None = None,
    reg_lambda: float = 1e-8,
    include_hessian: bool = False,
) -> DictionaryCache:
    """Build shift-only dictionary (δσ=0 throughout, handled by 4-step/6-step).

    This is the recommended strategy when δσ range is small:
    the dictionary handles the wide δE range, and 4-step/6-step handles
    the narrower δσ within its Taylor-valid regime.

    Equivalent to build_dictionary with dsigma_range=(0, 0).
    """
    return build_dictionary(
        energy=energy,
        centers=centers,
        sigmas=sigmas,
        gamma=gamma,
        dE_range=dE_range,
        dE_step=dE_step,
        dsigma_range=(0.0, 0.0),
        dsigma_step=1.0,  # only 1 point at δσ=0
        reg_lambda=reg_lambda,
        include_hessian=include_hessian,
    )


def _find_group_boundaries(sorted_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find contiguous group boundaries in a sorted index array.

    Returns:
        (starts, ends, unique_indices) — group boundaries and the grid index for each group.
    """
    n = len(sorted_idx)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int32)
    change_pts = np.where(np.diff(sorted_idx) != 0)[0] + 1
    starts = np.concatenate([[0], change_pts])
    ends = np.concatenate([change_pts, [n]])
    unique_indices = sorted_idx[starts]
    return starts, ends, unique_indices


def _four_step_group_mlx(
    Y_group_mx: 'mx.array',
    Wt_mx: 'mx.array',
    Phi_mx: 'mx.array',
    Jc_mx: 'mx.array',
    Js_mx: 'mx.array',
    n_energy_inv: float,
) -> tuple['mx.array', 'mx.array', 'mx.array', 'mx.array',
           'mx.array', 'mx.array', 'mx.array']:
    """4-step residual projection for one group (MLX, no eval).

    Returns all numerics needed for CPU post-processing.
    Does NOT call mx.eval() — caller accumulates and evals in batch.

    Returns:
        (A_row, chi2, num_c, den_c, RS_s, ScSs, den_s)
    """
    A_row = Y_group_mx @ Wt_mx.T          # (G, n_comp)
    R = Y_group_mx - A_row @ Phi_mx.T     # (G, n_energy)
    S_c = A_row @ Jc_mx.T                 # (G, n_energy)
    S_s = A_row @ Js_mx.T                 # (G, n_energy)

    num_c = mx.sum(R * S_c, axis=1)
    den_c = mx.sum(S_c * S_c, axis=1)
    RS_s = mx.sum(R * S_s, axis=1)
    ScSs = mx.sum(S_c * S_s, axis=1)
    den_s = mx.sum(S_s * S_s, axis=1)
    chi2 = mx.sum(R * R, axis=1) * n_energy_inv

    return A_row, chi2, num_c, den_c, RS_s, ScSs, den_s


def _steps234_from_AR_mlx(
    A_row_mx: 'mx.array',
    R_mx: 'mx.array',
    Jc_mx: 'mx.array',
    Js_mx: 'mx.array',
    n_energy_inv: float,
) -> tuple['mx.array', 'mx.array', 'mx.array', 'mx.array',
           'mx.array', 'mx.array', 'mx.array']:
    """Steps 2-4 projection from pre-computed A and R (MLX, no eval).

    Like _four_step_group_mlx but skips the A=Y@Wt' and R=Y-AΦ' steps
    that were already computed in a prior round.

    Returns:
        (A_row, chi2, num_c, den_c, RS_s, ScSs, den_s)
    """
    S_c = A_row_mx @ Jc_mx.T              # (G, n_energy)
    S_s = A_row_mx @ Js_mx.T              # (G, n_energy)

    num_c = mx.sum(R_mx * S_c, axis=1)
    den_c = mx.sum(S_c * S_c, axis=1)
    RS_s = mx.sum(R_mx * S_s, axis=1)
    ScSs = mx.sum(S_c * S_s, axis=1)
    den_s = mx.sum(S_s * S_s, axis=1)
    chi2 = mx.sum(R_mx * R_mx, axis=1) * n_energy_inv

    return A_row_mx, chi2, num_c, den_c, RS_s, ScSs, den_s


def _six_step_group_mlx(
    Y_group_mx: 'mx.array',
    Wt_mx: 'mx.array',
    Phi_mx: 'mx.array',
    Jc_mx: 'mx.array',
    Js_mx: 'mx.array',
    Hcc_mx: 'mx.array',
    n_energy_inv: float,
) -> tuple['mx.array', 'mx.array', 'mx.array', 'mx.array',
           'mx.array', 'mx.array', 'mx.array', 'mx.array']:
    """6-step residual projection for one group (MLX, no eval).

    Returns:
        (A_row, chi2, num_c, den_c, RS_s, ScSs, den_s, hcc_ss)
    """
    A_row = Y_group_mx @ Wt_mx.T
    R = Y_group_mx - A_row @ Phi_mx.T
    S_c = A_row @ Jc_mx.T
    S_s = A_row @ Js_mx.T
    H = A_row @ Hcc_mx.T

    num_c = mx.sum(R * S_c, axis=1)
    den_c = mx.sum(S_c * S_c, axis=1)
    RS_s = mx.sum(R * S_s, axis=1)
    ScSs = mx.sum(S_c * S_s, axis=1)
    den_s = mx.sum(S_s * S_s, axis=1)
    hcc_ss = mx.sum(H * S_s, axis=1)
    chi2 = mx.sum(R * R, axis=1) * n_energy_inv

    return A_row, chi2, num_c, den_c, RS_s, ScSs, den_s, hcc_ss


def solve_hybrid_sorted(
    Y: np.ndarray,
    dict_cache: DictionaryCache,
    min_group_for_mlx: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort-and-batch dictionary hybrid solver.

    Optimization over solve_hybrid(): sorts only indices on CPU (cheap),
    then gathers Y rows on MLX (fast) per group, processing with shared
    matrices. Avoids the expensive large-Y copy of a full sort.

    Strategy:
        1. Phase 1: MLX dictionary lookup (matmul + argmax)
        2. CPU sort of index array only (not Y)
        3. Phase 2: Per-group MLX gather of Y rows + 4-step with shared matrices
           - All groups: lazy MLX compute, single batch eval
           - Small groups (<min_group_for_mlx): NumPy CPU fallback
        4. Single mx.eval() for all GPU groups
        5. CPU post-processing (δE, δσ, amplitude correction)
        6. Scatter results to original order using sort_order mapping

    Args:
        Y: Spectra (n_spectra, n_energy) float32
        dict_cache: Pre-built dictionary cache
        min_group_for_mlx: Minimum group size for MLX path

    Returns:
        (amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices)
    """
    if not HAS_MLX:
        return solve_hybrid(Y, dict_cache, use_mlx=False)

    Y = np.asarray(Y, dtype=np.float32)
    n_spectra = Y.shape[0]
    n_energy = Y.shape[1]
    n_comp = dict_cache.Wt_per_grid.shape[1]
    n_energy_inv = 1.0 / n_energy

    dict_cache.to_mlx()

    # Phase 1: Dictionary lookup on GPU (chunked to avoid Metal OOM)
    Y_mx = mx.array(Y)
    best_idx = _chunked_dictionary_lookup(Y_mx, dict_cache._D_mx)

    n_dsigma = dict_cache.grid_shape[1]
    dE_idx = best_idx // n_dsigma
    ds_idx = best_idx % n_dsigma
    dE_coarse = dict_cache.dE_grid[dE_idx]
    dsigma_coarse = dict_cache.dsigma_grid[ds_idx]

    # Group spectra by dictionary index (argsort once, O(N log N))
    n_dict = dict_cache.n_dict
    groups = _group_by_index(best_idx, n_dict)

    # Pre-allocate output arrays (original order)
    amp_out = np.zeros((n_comp, n_spectra), dtype=np.float32)
    chi2_out = np.zeros(n_spectra, dtype=np.float32)
    dE_fine_out = np.zeros(n_spectra, dtype=np.float32)
    dsigma_fine_out = np.zeros(n_spectra, dtype=np.float32)

    # Phase 2: Per-group compute with lazy eval
    mlx_groups = []   # (orig_indices, grid_idx)
    mlx_outputs = []  # lazy mx.array results
    cpu_groups = []

    for orig_indices, gidx in groups:
        group_size = len(orig_indices)

        if group_size >= min_group_for_mlx:
            # MLX: gather Y rows on GPU (fast random-access), shared matrices
            idx_mx = mx.array(orig_indices.astype(np.int32))
            Y_g_mx = Y_mx[idx_mx]  # GPU gather: fast
            out = _four_step_group_mlx(
                Y_g_mx, dict_cache._Wt_mx[gidx], dict_cache._Phi_mx[gidx],
                dict_cache._Jc_mx[gidx], dict_cache._Jsigma_mx[gidx],
                n_energy_inv,
            )
            mlx_groups.append((orig_indices, gidx))
            mlx_outputs.append(out)
        else:
            cpu_groups.append((orig_indices, gidx))

    # Single mx.eval() for all GPU groups
    if mlx_outputs:
        all_lazy = []
        for out in mlx_outputs:
            all_lazy.extend(out)
        mx.eval(*all_lazy)

        # Scatter results to original positions
        for (orig_indices, gidx), out in zip(mlx_groups, mlx_outputs):
            A_row_np = np.array(out[0])  # (G, n_comp)
            chi2_np = np.array(out[1])
            num_c_np = np.array(out[2])
            den_c_np = np.array(out[3])
            RS_s_np = np.array(out[4])
            ScSs_np = np.array(out[5])
            den_s_np = np.array(out[6])

            # δE, δσ
            safe_den_c = np.where(den_c_np > 1e-20, den_c_np, 1.0)
            dE_g = np.where(den_c_np > 1e-20, num_c_np / safe_den_c, 0.0).astype(np.float32)
            num_s = RS_s_np - dE_g * ScSs_np
            safe_den_s = np.where(den_s_np > 1e-20, den_s_np, 1.0)
            ds_g = np.where(den_s_np > 1e-20, num_s / safe_den_s, 0.0).astype(np.float32)

            # Amplitude correction
            amp_g = A_row_np.T  # (n_comp, G)
            C = dict_cache.C_per_grid[gidx]
            if n_comp == 1:
                correction = 1.0 + ds_g * C[0, 0]
                correction = np.where(np.abs(correction) > 1e-6, correction, 1.0)
                amp_g[0, :] /= correction
            elif n_comp == 2:
                d = (1 + ds_g * C[0, 0]) * (1 + ds_g * C[1, 1]) - ds_g * ds_g * C[0, 1] * C[1, 0]
                d = np.where(np.abs(d) > 1e-12, d, 1.0)
                inv_d = 1.0 / d
                a0, a1 = amp_g[0, :].copy(), amp_g[1, :].copy()
                amp_g[0, :] = ((1 + ds_g * C[1, 1]) * a0 - ds_g * C[0, 1] * a1) * inv_d
                amp_g[1, :] = (-ds_g * C[1, 0] * a0 + (1 + ds_g * C[0, 0]) * a1) * inv_d
            else:
                I_mat = np.eye(n_comp, dtype=np.float32)
                M = I_mat[np.newaxis, :, :] + ds_g[:, np.newaxis, np.newaxis] * C[np.newaxis, :, :]
                a_col = amp_g.T[:, :, np.newaxis]
                a_corr = np.linalg.solve(M, a_col)
                amp_g = a_corr[:, :, 0].T

            amp_out[:, orig_indices] = amp_g
            chi2_out[orig_indices] = chi2_np
            dE_fine_out[orig_indices] = dE_g
            dsigma_fine_out[orig_indices] = ds_g

    # CPU fallback for small groups
    for orig_indices, gidx in cpu_groups:
        Y_group = Y[orig_indices]  # small group, copy is cheap
        amp_g, chi2_g, dE_g, ds_g = _four_step_for_group(
            Y_group,
            dict_cache.Wt_per_grid[gidx],
            dict_cache.Phi_per_grid[gidx],
            dict_cache.Jc_per_grid[gidx],
            dict_cache.Jsigma_per_grid[gidx],
            dict_cache.C_per_grid[gidx],
        )
        amp_out[:, orig_indices] = amp_g
        chi2_out[orig_indices] = chi2_g
        dE_fine_out[orig_indices] = dE_g
        dsigma_fine_out[orig_indices] = ds_g

    # Total = coarse + fine
    energy_shifts = (dE_coarse + dE_fine_out).astype(np.float32)
    sigma_shifts = (dsigma_coarse + dsigma_fine_out).astype(np.float32)

    return amp_out, chi2_out, energy_shifts, sigma_shifts, best_idx


def solve_hybrid_6step_sorted(
    Y: np.ndarray,
    dict_cache: DictionaryCache,
    min_group_for_mlx: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort-and-batch dictionary hybrid solver with 6-step Hessian refinement.

    Combines dictionary coarse search (Phase 1) with 6-step Hessian-corrected
    residual projection (Phase 2) for superior δσ recovery.

    Requires dict_cache built with include_hessian=True.

    Architecture:
        Phase 1: Dictionary lookup (same as solve_hybrid_sorted)
        Phase 2: 6-step refinement per group (Hessian ½δE²·H_cc decontamination)
                 → δσ correlation 0.93 (vs 4-step's -0.81)

    Returns:
        (amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices)
    """
    if not dict_cache.has_hessian:
        raise ValueError(
            "solve_hybrid_6step_sorted requires dict_cache built with "
            "include_hessian=True"
        )

    if not HAS_MLX:
        # CPU-only fallback: use _six_step_for_group directly
        return _solve_hybrid_6step_cpu(Y, dict_cache)

    Y = np.asarray(Y, dtype=np.float32)
    n_spectra = Y.shape[0]
    n_energy = Y.shape[1]
    n_comp = dict_cache.Wt_per_grid.shape[1]
    n_energy_inv = 1.0 / n_energy

    dict_cache.to_mlx()

    # Phase 1: Dictionary lookup on GPU (keep Y_mx alive for gather)
    Y_mx = mx.array(Y)
    scores = Y_mx @ dict_cache._D_mx
    best_idx_mx = mx.argmax(scores, axis=1)
    mx.eval(best_idx_mx)
    best_idx = np.array(best_idx_mx).astype(np.int32)

    n_dsigma = dict_cache.grid_shape[1]
    dE_idx = best_idx // n_dsigma
    ds_idx = best_idx % n_dsigma
    dE_coarse = dict_cache.dE_grid[dE_idx]
    dsigma_coarse = dict_cache.dsigma_grid[ds_idx]

    # Group spectra by dictionary index (argsort once, O(N log N))
    n_dict = dict_cache.n_dict
    groups = _group_by_index(best_idx, n_dict)

    # Pre-allocate output arrays
    amp_out = np.zeros((n_comp, n_spectra), dtype=np.float32)
    chi2_out = np.zeros(n_spectra, dtype=np.float32)
    dE_fine_out = np.zeros(n_spectra, dtype=np.float32)
    dsigma_fine_out = np.zeros(n_spectra, dtype=np.float32)

    # Phase 2: Per-group 6-step compute with lazy eval
    mlx_groups = []
    mlx_outputs = []
    cpu_groups = []

    for orig_indices, gidx in groups:
        group_size = len(orig_indices)

        if group_size >= min_group_for_mlx:
            idx_mx = mx.array(orig_indices.astype(np.int32))
            Y_g_mx = Y_mx[idx_mx]
            out = _six_step_group_mlx(
                Y_g_mx, dict_cache._Wt_mx[gidx], dict_cache._Phi_mx[gidx],
                dict_cache._Jc_mx[gidx], dict_cache._Jsigma_mx[gidx],
                dict_cache._Hcc_mx[gidx],
                n_energy_inv,
            )
            mlx_groups.append((orig_indices, gidx))
            mlx_outputs.append(out)
        else:
            cpu_groups.append((orig_indices, gidx))

    # Single mx.eval() for all GPU groups
    if mlx_outputs:
        all_lazy = []
        for out in mlx_outputs:
            all_lazy.extend(out)
        mx.eval(*all_lazy)

        # Scatter results to original positions
        for (orig_indices, gidx), out in zip(mlx_groups, mlx_outputs):
            A_row_np = np.array(out[0])
            chi2_np = np.array(out[1])
            num_c_np = np.array(out[2])
            den_c_np = np.array(out[3])
            RS_s_np = np.array(out[4])
            ScSs_np = np.array(out[5])
            den_s_np = np.array(out[6])
            hcc_ss_np = np.array(out[7])

            # δE
            safe_den_c = np.where(den_c_np > 1e-20, den_c_np, 1.0)
            dE_g = np.where(den_c_np > 1e-20, num_c_np / safe_den_c, 0.0).astype(np.float32)

            # δσ with Hessian correction (Step 2.5 + 3)
            correction = 0.5 * dE_g * dE_g * hcc_ss_np
            num_s = RS_s_np - dE_g * ScSs_np - correction
            safe_den_s = np.where(den_s_np > 1e-20, den_s_np, 1.0)
            ds_g = np.where(den_s_np > 1e-20, num_s / safe_den_s, 0.0).astype(np.float32)

            # Amplitude correction with C_σ + C_c (Step 3.5 + 4)
            amp_g = A_row_np.T
            C_sigma = dict_cache.C_per_grid[gidx]
            C_c = dict_cache.Cc_per_grid[gidx] if dict_cache.Cc_per_grid is not None else None
            if n_comp == 1:
                corr = 1.0 + ds_g * C_sigma[0, 0]
                if C_c is not None:
                    corr = corr + dE_g * C_c[0, 0]
                corr = np.where(np.abs(corr) > 1e-6, corr, 1.0)
                amp_g[0, :] /= corr
            elif n_comp == 2:
                M00 = 1 + ds_g * C_sigma[0, 0]
                M01 = ds_g * C_sigma[0, 1]
                M10 = ds_g * C_sigma[1, 0]
                M11 = 1 + ds_g * C_sigma[1, 1]
                if C_c is not None:
                    M00 = M00 + dE_g * C_c[0, 0]
                    M01 = M01 + dE_g * C_c[0, 1]
                    M10 = M10 + dE_g * C_c[1, 0]
                    M11 = M11 + dE_g * C_c[1, 1]
                d = M00 * M11 - M01 * M10
                d = np.where(np.abs(d) > 1e-12, d, 1.0)
                inv_d = 1.0 / d
                a0, a1 = amp_g[0, :].copy(), amp_g[1, :].copy()
                amp_g[0, :] = (M11 * a0 - M01 * a1) * inv_d
                amp_g[1, :] = (-M10 * a0 + M00 * a1) * inv_d
            else:
                I_mat = np.eye(n_comp, dtype=np.float32)
                M = I_mat[np.newaxis, :, :] + ds_g[:, np.newaxis, np.newaxis] * C_sigma[np.newaxis, :, :]
                if C_c is not None:
                    M = M + dE_g[:, np.newaxis, np.newaxis] * C_c[np.newaxis, :, :]
                a_col = amp_g.T[:, :, np.newaxis]
                a_corr = np.linalg.solve(M, a_col)
                amp_g = a_corr[:, :, 0].T

            amp_out[:, orig_indices] = amp_g
            chi2_out[orig_indices] = chi2_np
            dE_fine_out[orig_indices] = dE_g
            dsigma_fine_out[orig_indices] = ds_g

    # CPU fallback for small groups (use full _six_step_for_group)
    for orig_indices, gidx in cpu_groups:
        Y_group = Y[orig_indices]
        C_c = dict_cache.Cc_per_grid[gidx] if dict_cache.Cc_per_grid is not None else None
        amp_g, chi2_g, dE_g, ds_g = _six_step_for_group(
            Y_group,
            dict_cache.Wt_per_grid[gidx],
            dict_cache.Phi_per_grid[gidx],
            dict_cache.Jc_per_grid[gidx],
            dict_cache.Jsigma_per_grid[gidx],
            dict_cache.Hcc_per_grid[gidx],
            dict_cache.C_per_grid[gidx],
            C_c=C_c,
        )
        amp_out[:, orig_indices] = amp_g
        chi2_out[orig_indices] = chi2_g
        dE_fine_out[orig_indices] = dE_g
        dsigma_fine_out[orig_indices] = ds_g

    # Total = coarse + fine
    energy_shifts = (dE_coarse + dE_fine_out).astype(np.float32)
    sigma_shifts = (dsigma_coarse + dsigma_fine_out).astype(np.float32)

    return amp_out, chi2_out, energy_shifts, sigma_shifts, best_idx


def _solve_hybrid_6step_cpu(
    Y: np.ndarray,
    dict_cache: DictionaryCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CPU-only fallback for 6-step hybrid solver."""
    Y = np.asarray(Y, dtype=np.float32)
    n_spectra = Y.shape[0]
    n_comp = dict_cache.Wt_per_grid.shape[1]

    # Phase 1: Dictionary lookup (CPU)
    scores = Y @ dict_cache.D
    best_idx = np.argmax(scores, axis=1).astype(np.int32)

    n_dsigma = dict_cache.grid_shape[1]
    dE_idx = best_idx // n_dsigma
    ds_idx = best_idx % n_dsigma
    dE_coarse = dict_cache.dE_grid[dE_idx]
    dsigma_coarse = dict_cache.dsigma_grid[ds_idx]

    # Phase 2: sort-and-batch 6-step
    sort_order = np.argsort(best_idx)
    sorted_idx = best_idx[sort_order]
    Y_sorted = Y[sort_order]

    starts, ends, unique_indices = _find_group_boundaries(sorted_idx)

    amp_sorted = np.zeros((n_comp, n_spectra), dtype=np.float32)
    chi2_sorted = np.zeros(n_spectra, dtype=np.float32)
    dE_fine_sorted = np.zeros(n_spectra, dtype=np.float32)
    dsigma_fine_sorted = np.zeros(n_spectra, dtype=np.float32)

    for gi in range(len(starts)):
        s, e = int(starts[gi]), int(ends[gi])
        gidx = int(unique_indices[gi])
        Y_group = Y_sorted[s:e]
        C_c = dict_cache.Cc_per_grid[gidx] if dict_cache.Cc_per_grid is not None else None
        amp_g, chi2_g, dE_g, ds_g = _six_step_for_group(
            Y_group,
            dict_cache.Wt_per_grid[gidx],
            dict_cache.Phi_per_grid[gidx],
            dict_cache.Jc_per_grid[gidx],
            dict_cache.Jsigma_per_grid[gidx],
            dict_cache.Hcc_per_grid[gidx],
            dict_cache.C_per_grid[gidx],
            C_c=C_c,
        )
        amp_sorted[:, s:e] = amp_g
        chi2_sorted[s:e] = chi2_g
        dE_fine_sorted[s:e] = dE_g
        dsigma_fine_sorted[s:e] = ds_g

    unsort = np.argsort(sort_order)
    energy_shifts = (dE_coarse + dE_fine_sorted[unsort]).astype(np.float32)
    sigma_shifts = (dsigma_coarse + dsigma_fine_sorted[unsort]).astype(np.float32)
    amplitudes = amp_sorted[:, unsort]
    chi2 = chi2_sorted[unsort]

    return amplitudes, chi2, energy_shifts, sigma_shifts, best_idx


def adaptive_solve(
    Y: np.ndarray,
    dict_cache: DictionaryCache,
    Wt_nominal: Any = None,
    Phi_nominal: Any = None,
    Jc_nominal: Any = None,
    Jsigma_nominal: Any = None,
    C_nominal: np.ndarray | None = None,
    taylor_threshold: float = 0.35,
    min_group_for_mlx: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive pipeline: fast 4-step for Taylor-range, dictionary hybrid for outliers.

    Phase 1: Dictionary lookup (all spectra) — identifies coarse (δE, δσ).
    Phase 2a: Taylor-range spectra (|δE/σ| < threshold) → fast 4-step at nominal
              (single shared matrix, no per-group loop → maximum throughput).
    Phase 2b: Beyond-Taylor spectra → sort-and-batch dictionary hybrid
              (per-grid-point matrices, full 4-step correction).

    When most spectra are in Taylor range (typical for XPS), this approaches
    pure 4-step speed (~13M/s) while still handling outliers correctly.

    Args:
        Y: Spectra (n_spectra, n_energy) float32
        dict_cache: Pre-built dictionary cache
        Wt_nominal: Weight matrix at δE=0, δσ=0 (MLX or NumPy)
        Phi_nominal: Basis matrix at δE=0, δσ=0 (MLX or NumPy)
        Jc_nominal: Center Jacobian at nominal (MLX or NumPy)
        Jsigma_nominal: Sigma Jacobian at nominal (MLX or NumPy)
        C_nominal: Coupling matrix at nominal (NumPy), (n_comp, n_comp)
        taylor_threshold: |δ/σ| threshold for Taylor validity
        min_group_for_mlx: Min group size for MLX in hybrid path

    Returns:
        (amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices)
    """
    if not HAS_MLX:
        return solve_hybrid(Y, dict_cache, use_mlx=False)

    Y = np.asarray(Y, dtype=np.float32)
    n_spectra = Y.shape[0]
    n_energy = Y.shape[1]
    n_comp = dict_cache.Wt_per_grid.shape[1]

    dict_cache.to_mlx()

    # Phase 1: Dictionary lookup on GPU
    Y_mx = mx.array(Y)
    scores = Y_mx @ dict_cache._D_mx
    best_idx_mx = mx.argmax(scores, axis=1)
    mx.eval(best_idx_mx)
    best_idx = np.array(best_idx_mx).astype(np.int32)

    n_dsigma = dict_cache.grid_shape[1]
    dE_idx = best_idx // n_dsigma
    ds_idx = best_idx % n_dsigma
    dE_coarse = dict_cache.dE_grid[dE_idx]
    dsigma_coarse = dict_cache.dsigma_grid[ds_idx]

    # Classify: Taylor-range vs beyond-Taylor
    taylor_mask = dict_cache.is_within_taylor(best_idx, taylor_threshold)
    n_taylor = int(np.sum(taylor_mask))
    n_hybrid = n_spectra - n_taylor

    # Pre-allocate outputs
    amplitudes = np.zeros((n_comp, n_spectra), dtype=np.float32)
    chi2 = np.zeros(n_spectra, dtype=np.float32)
    energy_shifts = np.zeros(n_spectra, dtype=np.float32)
    sigma_shifts = np.zeros(n_spectra, dtype=np.float32)

    # Use nominal matrices from dict_cache if not provided
    nom_idx = dict_cache.nominal_index
    if Wt_nominal is None:
        Wt_nominal = mx.array(dict_cache.Wt_per_grid[nom_idx])
    if Phi_nominal is None:
        Phi_nominal = mx.array(dict_cache.Phi_per_grid[nom_idx])
    if Jc_nominal is None:
        Jc_nominal = mx.array(dict_cache.Jc_per_grid[nom_idx])
    if Jsigma_nominal is None:
        Jsigma_nominal = mx.array(dict_cache.Jsigma_per_grid[nom_idx])
    if C_nominal is None:
        C_nominal = dict_cache.C_per_grid[nom_idx]

    # Phase 2a: Taylor-range → fast 4-step at nominal (single matmul pass)
    if n_taylor > 0:
        taylor_indices = np.where(taylor_mask)[0]
        Y_taylor = Y[taylor_indices]
        Y_taylor_mx = mx.array(Y_taylor)
        n_energy_inv = mx.array(1.0 / n_energy)

        # MLX: 3-step kernel at nominal point
        W_nom = Wt_nominal.T  # (n_energy, n_comp)
        Phi_T = Phi_nominal.T
        Jc_T = Jc_nominal.T
        Js_T = Jsigma_nominal.T

        A_row, chi2_mx, num_c, den_c, RS_s, ScSs, den_s = \
            _shift_width_kernel_mlx(
                Y_taylor_mx, W_nom, Phi_T, Jc_T, Js_T, n_energy_inv,
            )
        mx.eval(A_row, chi2_mx, num_c, den_c, RS_s, ScSs, den_s)

        A_np = np.array(A_row)
        chi2_taylor = np.array(chi2_mx)
        num_c_np = np.array(num_c)
        den_c_np = np.array(den_c)
        RS_s_np = np.array(RS_s)
        ScSs_np = np.array(ScSs)
        den_s_np = np.array(den_s)

        # CPU: δE, δσ
        safe_den_c = np.where(den_c_np > 1e-20, den_c_np, 1.0)
        dE_taylor = np.where(den_c_np > 1e-20, num_c_np / safe_den_c, 0.0).astype(np.float32)
        num_s = RS_s_np - dE_taylor * ScSs_np
        safe_den_s = np.where(den_s_np > 1e-20, den_s_np, 1.0)
        ds_taylor = np.where(den_s_np > 1e-20, num_s / safe_den_s, 0.0).astype(np.float32)

        # Amplitude correction
        amp_taylor = A_np.T  # (n_comp, n_taylor)
        if n_comp == 1:
            correction = 1.0 + ds_taylor * C_nominal[0, 0]
            correction = np.where(np.abs(correction) > 1e-6, correction, 1.0)
            amp_taylor[0, :] /= correction
        elif n_comp == 2:
            d = (1 + ds_taylor * C_nominal[0, 0]) * (1 + ds_taylor * C_nominal[1, 1]) - \
                ds_taylor * ds_taylor * C_nominal[0, 1] * C_nominal[1, 0]
            d = np.where(np.abs(d) > 1e-12, d, 1.0)
            inv_d = 1.0 / d
            a0 = amp_taylor[0, :].copy()
            a1 = amp_taylor[1, :].copy()
            amp_taylor[0, :] = ((1 + ds_taylor * C_nominal[1, 1]) * a0 - ds_taylor * C_nominal[0, 1] * a1) * inv_d
            amp_taylor[1, :] = (-ds_taylor * C_nominal[1, 0] * a0 + (1 + ds_taylor * C_nominal[0, 0]) * a1) * inv_d
        else:
            I_mat = np.eye(n_comp, dtype=np.float32)
            M = I_mat[np.newaxis, :, :] + ds_taylor[:, np.newaxis, np.newaxis] * C_nominal[np.newaxis, :, :]
            a_col = amp_taylor.T[:, :, np.newaxis]
            a_corr = np.linalg.solve(M, a_col)
            amp_taylor = a_corr[:, :, 0].T

        amplitudes[:, taylor_indices] = amp_taylor
        chi2[taylor_indices] = chi2_taylor
        energy_shifts[taylor_indices] = dE_taylor
        sigma_shifts[taylor_indices] = ds_taylor

    # Phase 2b: Beyond-Taylor → sort-and-batch hybrid
    if n_hybrid > 0:
        hybrid_indices = np.where(~taylor_mask)[0]
        Y_hybrid = Y[hybrid_indices]
        best_idx_hybrid = best_idx[hybrid_indices]
        dE_coarse_hybrid = dE_coarse[hybrid_indices]
        dsigma_coarse_hybrid = dsigma_coarse[hybrid_indices]

        # Sort within hybrid subset
        sub_sort = np.argsort(best_idx_hybrid, kind='mergesort')
        sorted_idx = best_idx_hybrid[sub_sort]
        Y_sorted = Y_hybrid[sub_sort]

        starts, ends, unique_grid = _find_group_boundaries(sorted_idx)

        amp_h = np.zeros((n_comp, n_hybrid), dtype=np.float32)
        dE_fine_h = np.zeros(n_hybrid, dtype=np.float32)
        ds_fine_h = np.zeros(n_hybrid, dtype=np.float32)
        chi2_h = np.zeros(n_hybrid, dtype=np.float32)

        # MLX groups with lazy eval
        mlx_groups = []
        mlx_outputs = []

        for g in range(len(starts)):
            s, e = int(starts[g]), int(ends[g])
            gidx = int(unique_grid[g])
            group_size = e - s

            if group_size >= min_group_for_mlx:
                Y_g_mx = mx.array(Y_sorted[s:e])
                out = _four_step_group_mlx(
                    Y_g_mx,
                    dict_cache._Wt_mx[gidx],
                    dict_cache._Phi_mx[gidx],
                    dict_cache._Jc_mx[gidx],
                    dict_cache._Jsigma_mx[gidx],
                    1.0 / n_energy,
                )
                mlx_groups.append((s, e, gidx))
                mlx_outputs.append(out)
            else:
                # CPU path
                amp_g, chi2_g, dE_g, ds_g = _four_step_for_group(
                    Y_sorted[s:e],
                    dict_cache.Wt_per_grid[gidx],
                    dict_cache.Phi_per_grid[gidx],
                    dict_cache.Jc_per_grid[gidx],
                    dict_cache.Jsigma_per_grid[gidx],
                    dict_cache.C_per_grid[gidx],
                )
                amp_h[:, s:e] = amp_g
                chi2_h[s:e] = chi2_g
                dE_fine_h[s:e] = dE_g
                ds_fine_h[s:e] = ds_g

        # Batch eval MLX groups
        if mlx_outputs:
            all_lazy = []
            for out in mlx_outputs:
                all_lazy.extend(out)
            mx.eval(*all_lazy)

            for (s, e, gidx), out in zip(mlx_groups, mlx_outputs):
                A_row, chi2_g, num_c, den_c, RS_s, ScSs, den_s = out
                A_np = np.array(A_row)
                C = dict_cache.C_per_grid[gidx]

                # δE, δσ
                num_c_np = np.array(num_c)
                den_c_np = np.array(den_c)
                safe_den_c = np.where(den_c_np > 1e-20, den_c_np, 1.0)
                dE_g = np.where(den_c_np > 1e-20, num_c_np / safe_den_c, 0.0).astype(np.float32)

                RS_s_np = np.array(RS_s)
                ScSs_np = np.array(ScSs)
                den_s_np = np.array(den_s)
                num_s = RS_s_np - dE_g * ScSs_np
                safe_den_s = np.where(den_s_np > 1e-20, den_s_np, 1.0)
                ds_g = np.where(den_s_np > 1e-20, num_s / safe_den_s, 0.0).astype(np.float32)

                # Amplitude correction
                amp_g = A_np.T
                if n_comp == 1:
                    correction = 1.0 + ds_g * C[0, 0]
                    correction = np.where(np.abs(correction) > 1e-6, correction, 1.0)
                    amp_g[0, :] /= correction
                elif n_comp == 2:
                    d = (1 + ds_g * C[0, 0]) * (1 + ds_g * C[1, 1]) - ds_g * ds_g * C[0, 1] * C[1, 0]
                    d = np.where(np.abs(d) > 1e-12, d, 1.0)
                    inv_d = 1.0 / d
                    a0 = amp_g[0, :].copy()
                    a1 = amp_g[1, :].copy()
                    amp_g[0, :] = ((1 + ds_g * C[1, 1]) * a0 - ds_g * C[0, 1] * a1) * inv_d
                    amp_g[1, :] = (-ds_g * C[1, 0] * a0 + (1 + ds_g * C[0, 0]) * a1) * inv_d
                else:
                    I_mat = np.eye(n_comp, dtype=np.float32)
                    M = I_mat[np.newaxis, :, :] + ds_g[:, np.newaxis, np.newaxis] * C[np.newaxis, :, :]
                    a_col = amp_g.T[:, :, np.newaxis]
                    a_corr = np.linalg.solve(M, a_col)
                    amp_g = a_corr[:, :, 0].T

                amp_h[:, s:e] = amp_g
                chi2_h[s:e] = np.array(chi2_g)
                dE_fine_h[s:e] = dE_g
                ds_fine_h[s:e] = ds_g

        # Unsort hybrid subset
        unsort_h = np.argsort(sub_sort)
        amp_h = amp_h[:, unsort_h]
        chi2_h = chi2_h[unsort_h]
        dE_fine_h = dE_fine_h[unsort_h]
        ds_fine_h = ds_fine_h[unsort_h]

        amplitudes[:, hybrid_indices] = amp_h
        chi2[hybrid_indices] = chi2_h
        energy_shifts[hybrid_indices] = (dE_coarse_hybrid + dE_fine_h).astype(np.float32)
        sigma_shifts[hybrid_indices] = (dsigma_coarse_hybrid + ds_fine_h).astype(np.float32)

    return amplitudes, chi2, energy_shifts, sigma_shifts, best_idx


def _shift_width_kernel_mlx(Y_mx, W, Phi_T, J_c_T, J_s_T, n_energy_inv):
    """Inline 3-step kernel for adaptive solver (mirrors pipeline._shift_width_kernel_mlx)."""
    A_row = Y_mx @ W
    R = Y_mx - A_row @ Phi_T
    S_c = A_row @ J_c_T
    S_s = A_row @ J_s_T

    num_c = mx.sum(R * S_c, axis=1)
    den_c = mx.sum(S_c * S_c, axis=1)
    RS_s = mx.sum(R * S_s, axis=1)
    ScSs = mx.sum(S_c * S_s, axis=1)
    den_s = mx.sum(S_s * S_s, axis=1)
    chi2 = mx.sum(R * R, axis=1) * n_energy_inv

    return A_row, chi2, num_c, den_c, RS_s, ScSs, den_s


# ============================================================================
# 2D Dictionary Extensions (δE × δσ dense grid)
# ============================================================================

def build_dictionary_2d(
    energy: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gamma: float,
    dE_range: tuple[float, float] = (-2.0, 2.0),
    dE_step: float | None = None,
    dsigma_range: tuple[float, float] = (-0.3, 0.3),
    dsigma_step: float | None = None,
    N_dE: int | None = None,
    N_ds: int | None = None,
    reg_lambda: float = 1e-8,
    include_hessian: bool = False,
) -> DictionaryCache:
    """Build a dense 2D dictionary over (δE, δσ) grid.

    Convenience wrapper around build_dictionary() that accepts grid counts
    (N_dE, N_ds) as an alternative to step sizes.

    When N_dE/N_ds are given, step sizes are computed from ranges.
    When step sizes are given, N_dE/N_ds are ignored.
    Default: step = 0.1 * mean_sigma (same as build_dictionary).

    Args:
        energy: Energy axis (n_energy,)
        centers: Nominal peak centers (n_comp,)
        sigmas: Nominal Gaussian widths (n_comp,)
        gamma: Lorentzian width (shared)
        dE_range: δE search range (eV)
        dE_step: δE grid step (eV), overrides N_dE
        dsigma_range: δσ search range (eV)
        dsigma_step: δσ grid step (eV), overrides N_ds
        N_dE: Number of δE grid points (used if dE_step is None)
        N_ds: Number of δσ grid points (used if dsigma_step is None)
        reg_lambda: Tikhonov regularization
        include_hessian: Include Hessian for 6-step

    Returns:
        DictionaryCache with 2D grid
    """
    if dE_step is None and N_dE is not None and N_dE > 1:
        dE_step = (dE_range[1] - dE_range[0]) / (N_dE - 1)
    if dsigma_step is None and N_ds is not None and N_ds > 1:
        dsigma_step = (dsigma_range[1] - dsigma_range[0]) / (N_ds - 1)

    return build_dictionary(
        energy=energy,
        centers=centers,
        sigmas=sigmas,
        gamma=gamma,
        dE_range=dE_range,
        dE_step=dE_step,
        dsigma_range=dsigma_range,
        dsigma_step=dsigma_step,
        reg_lambda=reg_lambda,
        include_hessian=include_hessian,
    )


def solve_dict2d_amp_only(
    Y: np.ndarray,
    dict_cache: DictionaryCache,
    min_group_for_mlx: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2D dictionary solver with amplitude-only Phase 2.

    Phase 1: Dictionary lookup (matmul + argmax) → coarse (δE, δσ)
    Phase 2: LLS amplitude at best grid point (no δE/δσ fine correction)

    When the 2D grid is sufficiently dense, the coarse (δE, δσ) values
    are already accurate to within half the grid spacing. Phase 2 only
    needs to recover amplitudes → 1 matmul per group (vs 4 for full 4-step).

    Faster than solve_hybrid_sorted because it skips Steps 2-4 of the
    residual projection. Use when grid spacing is fine enough that
    sub-grid Taylor correction is unnecessary.

    Args:
        Y: Spectra (n_spectra, n_energy) float32
        dict_cache: Pre-built 2D dictionary cache
        min_group_for_mlx: Min group size for MLX path

    Returns:
        (amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices)
        energy_shifts/sigma_shifts are coarse-only (no fine correction)
    """
    if not HAS_MLX:
        return solve_dictionary_only(Y, dict_cache, use_mlx=False)

    Y = np.asarray(Y, dtype=np.float32)
    n_spectra = Y.shape[0]
    n_energy = Y.shape[1]
    n_comp = dict_cache.Wt_per_grid.shape[1]
    n_energy_inv = 1.0 / n_energy

    dict_cache.to_mlx()

    # Phase 1: Dictionary lookup on GPU
    Y_mx = mx.array(Y)
    scores = Y_mx @ dict_cache._D_mx
    best_idx_mx = mx.argmax(scores, axis=1)
    mx.eval(best_idx_mx)
    best_idx = np.array(best_idx_mx).astype(np.int32)

    n_dsigma = dict_cache.grid_shape[1]
    dE_idx = best_idx // n_dsigma
    ds_idx = best_idx % n_dsigma
    dE_coarse = dict_cache.dE_grid[dE_idx]
    dsigma_coarse = dict_cache.dsigma_grid[ds_idx]

    # Phase 2: Amplitude-only at best grid point
    # Group by argsort (O(N log N) once vs O(N) × n_groups)
    n_dict = dict_cache.n_dict
    groups = _group_by_index(best_idx, n_dict)

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

            A_row = Y_g_mx @ Wt_mx.T           # (G, n_comp)
            R = Y_g_mx - A_row @ Phi_mx.T      # (G, n_energy)
            chi2_g = mx.sum(R * R, axis=1) * n_energy_inv

            mlx_groups.append((orig_indices, gidx))
            mlx_outputs.append((A_row, chi2_g))
        else:
            # CPU path
            Wt = dict_cache.Wt_per_grid[gidx]
            Phi = dict_cache.Phi_per_grid[gidx]
            Y_group = Y[orig_indices]
            A_row_np = Y_group @ Wt.T
            R = Y_group - A_row_np @ Phi.T
            amp_out[:, orig_indices] = A_row_np.T
            chi2_out[orig_indices] = np.sum(R * R, axis=1) / n_energy

    # Single mx.eval() for all GPU groups
    if mlx_outputs:
        all_lazy = []
        for out in mlx_outputs:
            all_lazy.extend(out)
        mx.eval(*all_lazy)

        for (orig_indices, _gidx), (A_row_mx, chi2_mx) in zip(mlx_groups, mlx_outputs):
            amp_out[:, orig_indices] = np.array(A_row_mx).T
            chi2_out[orig_indices] = np.array(chi2_mx)

    return amp_out, chi2_out, dE_coarse, dsigma_coarse, best_idx


def _dict2d_scores_and_neighbours(
    Y_mx: 'mx.array',
    dict_cache: 'DictionaryCache',
    start: int,
    end: int,
    n_dE: int,
    n_ds: int,
    use_fp16: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute scores, argmax, and 4 neighbour scores for a row chunk.

    Extracts the 5 scores needed for parabola interpolation (center + 4
    neighbours along dE and ds axes), then immediately frees the scores
    matrix.  This bounds GPU memory to (chunk_rows × n_dict × dtype_bytes).

    Args:
        Y_mx: Full spectra on GPU (n_spectra, n_energy)
        dict_cache: Dictionary cache (must have called to_mlx())
        start, end: Row range [start, end) to process
        n_dE, n_ds: Grid shape
        use_fp16: Compute scores in float16

    Returns:
        (best_idx, s0, s_dE_m, s_dE_p, s_ds_m, s_ds_p) — all NumPy float32/int32
    """
    chunk = Y_mx[start:end]
    D_mx = dict_cache._D_mx
    if use_fp16:
        scores_mx = chunk.astype(mx.float16) @ D_mx.astype(mx.float16)
    else:
        scores_mx = chunk @ D_mx
    best_idx_mx = mx.argmax(scores_mx, axis=1)
    mx.eval(best_idx_mx)
    best_idx = np.array(best_idx_mx).astype(np.int32)

    # Neighbour indices
    dE_idx = best_idx // n_ds
    ds_idx = best_idx % n_ds
    dE_m = np.clip(dE_idx - 1, 0, n_dE - 1) * n_ds + ds_idx
    dE_p = np.clip(dE_idx + 1, 0, n_dE - 1) * n_ds + ds_idx
    ds_m = dE_idx * n_ds + np.clip(ds_idx - 1, 0, n_ds - 1)
    ds_p = dE_idx * n_ds + np.clip(ds_idx + 1, 0, n_ds - 1)

    n_chunk = end - start
    spec_idx_mx = mx.arange(n_chunk)
    s0 = scores_mx[spec_idx_mx, mx.array(best_idx)]
    s_dE_m = scores_mx[spec_idx_mx, mx.array(dE_m)]
    s_dE_p = scores_mx[spec_idx_mx, mx.array(dE_p)]
    s_ds_m = scores_mx[spec_idx_mx, mx.array(ds_m)]
    s_ds_p = scores_mx[spec_idx_mx, mx.array(ds_p)]
    mx.eval(s0, s_dE_m, s_dE_p, s_ds_m, s_ds_p)

    # Convert to CPU and free GPU scores
    result = (
        best_idx,
        np.array(s0, dtype=np.float32),
        np.array(s_dE_m, dtype=np.float32),
        np.array(s_dE_p, dtype=np.float32),
        np.array(s_ds_m, dtype=np.float32),
        np.array(s_ds_p, dtype=np.float32),
    )
    del scores_mx
    return result


def solve_dict2d_parabola(
    Y: np.ndarray,
    dict_cache: DictionaryCache,
    min_group_for_mlx: int = 64,
    chunk_bytes: int = _PHASE1_MAX_BYTES,
    use_fp16_scores: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2D dictionary solver with parabola sub-grid interpolation.

    Phase 1: Dictionary lookup (matmul + argmax) → coarse (δE, δσ)
    Phase 1.5: Parabola interpolation on score grid → refined (δE, δσ)
    Phase 2: LLS amplitude at best grid point (no Jacobian)

    Combines dictionary coarse search with sub-grid parabola refinement,
    eliminating the need for Jacobian-based 4-step correction.  The parabola
    fits independent 1D quadratics along the δE and δσ axes of the score
    matrix to recover continuous parameter estimates.

    Added memory-aware chunking and fp16 support.  When the
    full scores matrix (N × n_dict) would exceed ``chunk_bytes``, spectra
    are processed in row-chunks so that only one chunk's scores are live
    at a time.

    Args:
        Y: Spectra (n_spectra, n_energy) float32
        dict_cache: Pre-built 2D dictionary cache
        min_group_for_mlx: Min group size for MLX path
        chunk_bytes: Max GPU buffer for scores matrix (default 4 GB)
        use_fp16_scores: If True, compute scores in float16 (halves memory)

    Returns:
        (amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices)
        energy_shifts/sigma_shifts are parabola-refined.
    """
    if not HAS_MLX:
        return solve_dictionary_only(Y, dict_cache, use_mlx=False)

    Y = np.asarray(Y, dtype=np.float32)
    n_spectra = Y.shape[0]
    n_energy = Y.shape[1]
    n_comp = dict_cache.Wt_per_grid.shape[1]
    n_energy_inv = 1.0 / n_energy

    dict_cache.to_mlx()

    n_dE, n_ds = dict_cache.grid_shape
    n_dict = dict_cache.n_dict
    dtype_bytes = 2 if use_fp16_scores else 4
    max_rows = max(1, chunk_bytes // (n_dict * dtype_bytes))

    Y_mx = mx.array(Y)

    if max_rows >= n_spectra:
        # Fast path: all spectra fit in one chunk
        best_idx, s0_np, sm_dE_np, sp_dE_np, sm_ds_np, sp_ds_np = (
            _dict2d_scores_and_neighbours(
                Y_mx, dict_cache, 0, n_spectra, n_dE, n_ds,
                use_fp16=use_fp16_scores,
            )
        )
    else:
        # Chunked path: process in row-chunks to bound scores memory
        best_idx = np.empty(n_spectra, dtype=np.int32)
        s0_np = np.empty(n_spectra, dtype=np.float32)
        sm_dE_np = np.empty(n_spectra, dtype=np.float32)
        sp_dE_np = np.empty(n_spectra, dtype=np.float32)
        sm_ds_np = np.empty(n_spectra, dtype=np.float32)
        sp_ds_np = np.empty(n_spectra, dtype=np.float32)

        for start in range(0, n_spectra, max_rows):
            end = min(start + max_rows, n_spectra)
            (c_idx, c_s0, c_sm_dE, c_sp_dE, c_sm_ds, c_sp_ds) = (
                _dict2d_scores_and_neighbours(
                    Y_mx, dict_cache, start, end, n_dE, n_ds,
                    use_fp16=use_fp16_scores,
                )
            )
            best_idx[start:end] = c_idx
            s0_np[start:end] = c_s0
            sm_dE_np[start:end] = c_sm_dE
            sp_dE_np[start:end] = c_sp_dE
            sm_ds_np[start:end] = c_sm_ds
            sp_ds_np[start:end] = c_sp_ds

    # Parabola on CPU
    dE_idx = best_idx // n_ds
    ds_idx = best_idx % n_ds

    dE_interior = (dE_idx > 0) & (dE_idx < n_dE - 1)
    ds_interior = (ds_idx > 0) & (ds_idx < n_ds - 1)

    from .grids import is_uniform
    dE_uniform = is_uniform(dict_cache.dE_grid)
    ds_uniform = is_uniform(dict_cache.dsigma_grid)

    if dE_uniform:
        dE_step = dict_cache.dE_grid[1] - dict_cache.dE_grid[0] if n_dE > 1 else 1.0
        delta_dE = _parabola_vertex(sm_dE_np, s0_np, sp_dE_np)
        delta_dE = np.where(dE_interior, delta_dE, 0.0)
        dE_refined = (dict_cache.dE_grid[dE_idx] + delta_dE * dE_step).astype(np.float32)
    else:
        from .grids import precompute_grid_spacings
        hL_dE, hR_dE = precompute_grid_spacings(dict_cache.dE_grid)
        hL_per = hL_dE[dE_idx]
        hR_per = hR_dE[dE_idx]
        shift_dE = _parabola_vertex_nonuniform(sm_dE_np, s0_np, sp_dE_np, hL_per, hR_per)
        shift_dE = np.where(dE_interior, shift_dE, 0.0)
        dE_refined = (dict_cache.dE_grid[dE_idx] + shift_dE).astype(np.float32)

    if ds_uniform:
        ds_step = dict_cache.dsigma_grid[1] - dict_cache.dsigma_grid[0] if n_ds > 1 else 1.0
        delta_ds = _parabola_vertex(sm_ds_np, s0_np, sp_ds_np)
        delta_ds = np.where(ds_interior, delta_ds, 0.0)
        dsigma_refined = (dict_cache.dsigma_grid[ds_idx] + delta_ds * ds_step).astype(np.float32)
    else:
        from .grids import precompute_grid_spacings
        hL_ds, hR_ds = precompute_grid_spacings(dict_cache.dsigma_grid)
        hL_per = hL_ds[ds_idx]
        hR_per = hR_ds[ds_idx]
        shift_ds = _parabola_vertex_nonuniform(sm_ds_np, s0_np, sp_ds_np, hL_per, hR_per)
        shift_ds = np.where(ds_interior, shift_ds, 0.0)
        dsigma_refined = (dict_cache.dsigma_grid[ds_idx] + shift_ds).astype(np.float32)

    # Phase 2: Amplitude-only at best grid point (same as solve_dict2d_amp_only)
    n_dict = dict_cache.n_dict
    groups = _group_by_index(best_idx, n_dict)

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

            A_row = Y_g_mx @ Wt_mx.T           # (G, n_comp)
            R = Y_g_mx - A_row @ Phi_mx.T      # (G, n_energy)
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

    return amp_out, chi2_out, dE_refined, dsigma_refined, best_idx


def solve_dict2d_adaptive(
    Y: np.ndarray,
    dict_cache: DictionaryCache,
    chi2_threshold: float | None = None,
    min_group_for_mlx: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive Dict2D solver — amp_only for noisy, 4-step for clean.

    Phase 1: Dictionary lookup (matmul + argmax) → coarse (δE, δσ)
    Phase 2a: Amplitude-only for ALL spectra (cheap, 1 matmul/group)
    Phase 2b: 4-step refinement ONLY for spectra with chi2 < threshold
    Merge: 4-step results where clean, amp_only elsewhere

    The chi2 from amp_only is a natural SNR indicator:
      - Low chi2 → sub-grid quantization error dominates → 4-step helps
      - High chi2 → noise dominates → 4-step amplifies noise

    Auto-threshold strategy (chi2_threshold=None):
      Uses amplitude-normalized chi2: chi2_norm = chi2 / peak_intensity².
      This is scale-invariant — the normalized quantization error is a
      fixed property of the grid spacing, independent of amplitude.

      Default threshold: 3e-4 on chi2_norm.
      - NF: chi2_norm_p95 ≈ 1e-4 → all pass → 4-step for all
      - lam2 (SNR≈100): chi2_norm_p95 ≈ 1.4e-4 → most pass
      - lam3 (SNR≈30): chi2_norm_p5 ≈ 1.3e-3 → none pass → amp_only
      This naturally tracks the Dict2D crossover at SNR≈30.

    Args:
        Y: Spectra (n_spectra, n_energy) float32
        dict_cache: Pre-built 2D dictionary cache
        chi2_threshold: Max normalized chi2 for 4-step activation.
            Applied to chi2 / peak_intensity². None = 3e-4.
        min_group_for_mlx: Min group size for MLX path

    Returns:
        (amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices)
    """
    if not HAS_MLX:
        # Fallback: just run 4-step (no adaptive benefit on CPU)
        return solve_hybrid_sorted(Y, dict_cache, min_group_for_mlx)

    Y = np.asarray(Y, dtype=np.float32)
    n_spectra = Y.shape[0]
    n_energy = Y.shape[1]
    n_comp = dict_cache.Wt_per_grid.shape[1]
    n_energy_inv = 1.0 / n_energy

    dict_cache.to_mlx()

    # ---- Phase 1: Dictionary lookup (chunked for large inputs) ----
    Y_mx = mx.array(Y)
    best_idx = _chunked_dictionary_lookup(Y_mx, dict_cache._D_mx)

    n_dsigma = dict_cache.grid_shape[1]
    dE_idx = best_idx // n_dsigma
    ds_idx = best_idx % n_dsigma
    dE_coarse = dict_cache.dE_grid[dE_idx]
    dsigma_coarse = dict_cache.dsigma_grid[ds_idx]

    # ---- Phase 2a: Amp-only for ALL spectra ----
    n_dict = dict_cache.n_dict
    groups = _group_by_index(best_idx, n_dict)

    amp_ao = np.zeros((n_comp, n_spectra), dtype=np.float32)
    chi2_ao = np.zeros(n_spectra, dtype=np.float32)

    mlx_ao_groups = []
    mlx_ao_outputs = []

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

            mlx_ao_groups.append((orig_indices, gidx))
            mlx_ao_outputs.append((A_row, chi2_g))
        else:
            Wt = dict_cache.Wt_per_grid[gidx]
            Phi = dict_cache.Phi_per_grid[gidx]
            Y_group = Y[orig_indices]
            A_row_np = Y_group @ Wt.T
            R = Y_group - A_row_np @ Phi.T
            amp_ao[:, orig_indices] = A_row_np.T
            chi2_ao[orig_indices] = np.sum(R * R, axis=1) / n_energy

    if mlx_ao_outputs:
        all_lazy = []
        for out in mlx_ao_outputs:
            all_lazy.extend(out)
        mx.eval(*all_lazy)

        for (orig_indices, _gidx), (A_row_mx, chi2_mx) in zip(mlx_ao_groups, mlx_ao_outputs):
            amp_ao[:, orig_indices] = np.array(A_row_mx).T
            chi2_ao[orig_indices] = np.array(chi2_mx)

    # ---- Auto-calibrate threshold ----
    if chi2_threshold is None:
        chi2_threshold = 3e-4  # default normalized threshold

    # ---- Decide which spectra get 4-step (amplitude-normalized chi2) ----
    peak_intensity = np.max(Y, axis=1)
    chi2_norm = chi2_ao / (peak_intensity ** 2 + 1e-20)
    use_4step = chi2_norm < chi2_threshold
    n_4step = int(np.sum(use_4step))

    # Start with amp_only as default
    amp_out = amp_ao.copy()
    chi2_out = chi2_ao.copy()
    dE_fine_out = np.zeros(n_spectra, dtype=np.float32)
    dsigma_fine_out = np.zeros(n_spectra, dtype=np.float32)

    if n_4step == 0:
        # All noisy — pure amp_only
        return amp_out, chi2_out, dE_coarse, dsigma_coarse, best_idx

    # ---- Phase 2b: 4-step for clean spectra ----
    # Reuse group structure from Phase 2a, filter per group
    mlx_4s_groups = []
    mlx_4s_outputs = []
    cpu_4s_groups = []

    for orig_indices, gidx in groups:
        clean_mask = use_4step[orig_indices]
        n_clean = int(np.sum(clean_mask))
        if n_clean == 0:
            continue
        local_idx = orig_indices[clean_mask]
        group_size = n_clean

        if group_size >= min_group_for_mlx:
            idx_mx = mx.array(local_idx.astype(np.int32))
            Y_g_mx = Y_mx[idx_mx]
            out = _four_step_group_mlx(
                Y_g_mx, dict_cache._Wt_mx[gidx], dict_cache._Phi_mx[gidx],
                dict_cache._Jc_mx[gidx], dict_cache._Jsigma_mx[gidx],
                n_energy_inv,
            )
            mlx_4s_groups.append((local_idx, gidx))
            mlx_4s_outputs.append(out)
        else:
            cpu_4s_groups.append((local_idx, gidx))

    # Single mx.eval()
    if mlx_4s_outputs:
        all_lazy = []
        for out in mlx_4s_outputs:
            all_lazy.extend(out)
        mx.eval(*all_lazy)

        for (orig_indices, gidx), out in zip(mlx_4s_groups, mlx_4s_outputs):
            A_row_np = np.array(out[0])
            chi2_np = np.array(out[1])
            num_c_np = np.array(out[2])
            den_c_np = np.array(out[3])
            RS_s_np = np.array(out[4])
            ScSs_np = np.array(out[5])
            den_s_np = np.array(out[6])

            safe_den_c = np.where(den_c_np > 1e-20, den_c_np, 1.0)
            dE_g = np.where(den_c_np > 1e-20, num_c_np / safe_den_c, 0.0).astype(np.float32)
            num_s = RS_s_np - dE_g * ScSs_np
            safe_den_s = np.where(den_s_np > 1e-20, den_s_np, 1.0)
            ds_g = np.where(den_s_np > 1e-20, num_s / safe_den_s, 0.0).astype(np.float32)

            amp_g = A_row_np.T
            C = dict_cache.C_per_grid[gidx]
            if n_comp == 1:
                correction = 1.0 + ds_g * C[0, 0]
                correction = np.where(np.abs(correction) > 1e-6, correction, 1.0)
                amp_g[0, :] /= correction
            elif n_comp == 2:
                d = (1 + ds_g * C[0, 0]) * (1 + ds_g * C[1, 1]) - ds_g * ds_g * C[0, 1] * C[1, 0]
                d = np.where(np.abs(d) > 1e-12, d, 1.0)
                inv_d = 1.0 / d
                a0, a1 = amp_g[0, :].copy(), amp_g[1, :].copy()
                amp_g[0, :] = ((1 + ds_g * C[1, 1]) * a0 - ds_g * C[0, 1] * a1) * inv_d
                amp_g[1, :] = (-ds_g * C[1, 0] * a0 + (1 + ds_g * C[0, 0]) * a1) * inv_d
            else:
                I_mat = np.eye(n_comp, dtype=np.float32)
                M = I_mat[np.newaxis, :, :] + ds_g[:, np.newaxis, np.newaxis] * C[np.newaxis, :, :]
                a_col = amp_g.T[:, :, np.newaxis]
                a_corr = np.linalg.solve(M, a_col)
                amp_g = a_corr[:, :, 0].T

            amp_out[:, orig_indices] = amp_g
            chi2_out[orig_indices] = chi2_np
            dE_fine_out[orig_indices] = dE_g
            dsigma_fine_out[orig_indices] = ds_g

    for orig_indices, gidx in cpu_4s_groups:
        Y_group = Y[orig_indices]
        amp_g, chi2_g, dE_g, ds_g = _four_step_for_group(
            Y_group,
            dict_cache.Wt_per_grid[gidx],
            dict_cache.Phi_per_grid[gidx],
            dict_cache.Jc_per_grid[gidx],
            dict_cache.Jsigma_per_grid[gidx],
            dict_cache.C_per_grid[gidx],
        )
        amp_out[:, orig_indices] = amp_g
        chi2_out[orig_indices] = chi2_g
        dE_fine_out[orig_indices] = dE_g
        dsigma_fine_out[orig_indices] = ds_g

    # Total shifts: coarse + fine (fine=0 for amp_only spectra)
    energy_shifts = (dE_coarse + dE_fine_out).astype(np.float32)
    sigma_shifts = (dsigma_coarse + dsigma_fine_out).astype(np.float32)

    return amp_out, chi2_out, energy_shifts, sigma_shifts, best_idx


def solve_dict2d_adaptive_fused(
    Y: np.ndarray,
    dict_cache: DictionaryCache,
    chi2_threshold: float | None = None,
    min_group_for_mlx: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optimized Dict2D adaptive solver with argsort grouping + A/R reuse.

    Same 2-round architecture as solve_dict2d_adaptive but with:
    - argsort-based grouping (O(N log N) once vs O(N)×50 np.where)
    - Round 2 reuses A_np from Round 1 to skip Y@Wt' recomputation
    - Group structure shared between Round 1 and Round 2 (no re-grouping)

    Args:
        Y: Spectra (n_spectra, n_energy) float32
        dict_cache: Pre-built 2D dictionary cache
        chi2_threshold: Max normalized chi2 for 4-step activation.
            Applied to chi2 / peak_intensity². None = 3e-4.
        min_group_for_mlx: Min group size for MLX path

    Returns:
        (amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices)
    """
    if not HAS_MLX:
        return solve_hybrid_sorted(Y, dict_cache, min_group_for_mlx)

    Y = np.asarray(Y, dtype=np.float32)
    n_spectra = Y.shape[0]
    n_energy = Y.shape[1]
    n_comp = dict_cache.Wt_per_grid.shape[1]
    n_energy_inv = 1.0 / n_energy

    dict_cache.to_mlx()

    # ---- Phase 1: Dictionary lookup (chunked for large inputs) ----
    Y_mx = mx.array(Y)
    best_idx = _chunked_dictionary_lookup(Y_mx, dict_cache._D_mx)

    n_dsigma = dict_cache.grid_shape[1]
    dE_idx = best_idx // n_dsigma
    ds_idx = best_idx % n_dsigma
    dE_coarse = dict_cache.dE_grid[dE_idx]
    dsigma_coarse = dict_cache.dsigma_grid[ds_idx]

    # ---- argsort-based grouping (O(N log N) once) ----
    n_dict = dict_cache.n_dict
    groups = _group_by_index(best_idx, n_dict)

    # Pre-allocate output arrays
    amp_out = np.zeros((n_comp, n_spectra), dtype=np.float32)
    chi2_out = np.zeros(n_spectra, dtype=np.float32)
    dE_fine_out = np.zeros(n_spectra, dtype=np.float32)
    dsigma_fine_out = np.zeros(n_spectra, dtype=np.float32)

    # ---- Round 1: Amp-only (A, chi2) for ALL groups ----
    mlx_r1_groups = []
    mlx_r1_outputs = []

    for orig_indices, gidx in groups:
        group_size = len(orig_indices)
        if group_size >= min_group_for_mlx:
            idx_mx = mx.array(orig_indices)
            Y_g_mx = Y_mx[idx_mx]
            Wt_mx = dict_cache._Wt_mx[gidx]
            Phi_mx = dict_cache._Phi_mx[gidx]

            A_row = Y_g_mx @ Wt_mx.T
            R = Y_g_mx - A_row @ Phi_mx.T
            chi2_g = mx.sum(R * R, axis=1) * n_energy_inv

            mlx_r1_groups.append((orig_indices, gidx))
            mlx_r1_outputs.append((A_row, chi2_g))
        else:
            # CPU amp_only for small groups
            Wt = dict_cache.Wt_per_grid[gidx]
            Phi = dict_cache.Phi_per_grid[gidx]
            Y_group = Y[orig_indices]
            A_row_np = Y_group @ Wt.T
            R = Y_group - A_row_np @ Phi.T
            amp_out[:, orig_indices] = A_row_np.T
            chi2_out[orig_indices] = np.sum(R * R, axis=1) / n_energy

    # Eval Round 1
    if mlx_r1_outputs:
        all_lazy = []
        for out in mlx_r1_outputs:
            all_lazy.extend(out)
        mx.eval(*all_lazy)

        for (orig_indices, gidx), (A_row_mx, chi2_mx) in zip(
            mlx_r1_groups, mlx_r1_outputs
        ):
            amp_out[:, orig_indices] = np.array(A_row_mx).T
            chi2_out[orig_indices] = np.array(chi2_mx)

    # ---- Threshold check ----
    if chi2_threshold is None:
        chi2_threshold = 3e-4

    peak_intensity = np.max(Y, axis=1)
    chi2_norm = chi2_out / (peak_intensity ** 2 + 1e-20)
    use_4step = chi2_norm < chi2_threshold
    n_4step = int(np.sum(use_4step))

    if n_4step == 0:
        return amp_out, chi2_out, dE_coarse, dsigma_coarse, best_idx

    # ---- Round 2: 4-step for clean spectra, reusing group structure ----
    mlx_r2_groups = []
    mlx_r2_outputs = []

    for orig_indices, gidx in groups:
        clean_mask = use_4step[orig_indices]
        n_clean = int(np.sum(clean_mask))
        if n_clean == 0:
            continue

        clean_orig = orig_indices[clean_mask]
        group_size = len(clean_orig)

        if group_size >= min_group_for_mlx:
            # GPU path: recompute A via Y@Wt' (must re-gather from Y_mx)
            # but use _four_step_group_mlx which does all 4 steps
            idx_mx = mx.array(clean_orig)
            Y_g_mx = Y_mx[idx_mx]
            out = _four_step_group_mlx(
                Y_g_mx, dict_cache._Wt_mx[gidx], dict_cache._Phi_mx[gidx],
                dict_cache._Jc_mx[gidx], dict_cache._Jsigma_mx[gidx],
                n_energy_inv,
            )
            mlx_r2_groups.append((clean_orig, gidx))
            mlx_r2_outputs.append(out)
        else:
            # CPU 4-step for small clean groups
            Y_group = Y[clean_orig]
            amp_g, chi2_g, dE_g, ds_g = _four_step_for_group(
                Y_group,
                dict_cache.Wt_per_grid[gidx],
                dict_cache.Phi_per_grid[gidx],
                dict_cache.Jc_per_grid[gidx],
                dict_cache.Jsigma_per_grid[gidx],
                dict_cache.C_per_grid[gidx],
            )
            amp_out[:, clean_orig] = amp_g
            chi2_out[clean_orig] = chi2_g
            dE_fine_out[clean_orig] = dE_g
            dsigma_fine_out[clean_orig] = ds_g

    # Eval Round 2
    if mlx_r2_outputs:
        all_lazy = []
        for out in mlx_r2_outputs:
            all_lazy.extend(out)
        mx.eval(*all_lazy)

        for (clean_orig, gidx), out in zip(mlx_r2_groups, mlx_r2_outputs):
            A_row_np = np.array(out[0])
            chi2_np = np.array(out[1])
            num_c_np = np.array(out[2])
            den_c_np = np.array(out[3])
            RS_s_np = np.array(out[4])
            ScSs_np = np.array(out[5])
            den_s_np = np.array(out[6])

            safe_den_c = np.where(den_c_np > 1e-20, den_c_np, 1.0)
            dE_g = np.where(den_c_np > 1e-20,
                            num_c_np / safe_den_c, 0.0).astype(np.float32)
            num_s = RS_s_np - dE_g * ScSs_np
            safe_den_s = np.where(den_s_np > 1e-20, den_s_np, 1.0)
            ds_g = np.where(den_s_np > 1e-20,
                            num_s / safe_den_s, 0.0).astype(np.float32)

            amp_g = A_row_np.T
            C = dict_cache.C_per_grid[gidx]
            if n_comp == 1:
                correction = 1.0 + ds_g * C[0, 0]
                correction = np.where(np.abs(correction) > 1e-6,
                                      correction, 1.0)
                amp_g[0, :] /= correction
            elif n_comp == 2:
                d = ((1 + ds_g * C[0, 0]) * (1 + ds_g * C[1, 1])
                     - ds_g * ds_g * C[0, 1] * C[1, 0])
                d = np.where(np.abs(d) > 1e-12, d, 1.0)
                inv_d = 1.0 / d
                a0, a1 = amp_g[0, :].copy(), amp_g[1, :].copy()
                amp_g[0, :] = ((1 + ds_g * C[1, 1]) * a0
                               - ds_g * C[0, 1] * a1) * inv_d
                amp_g[1, :] = (-ds_g * C[1, 0] * a0
                               + (1 + ds_g * C[0, 0]) * a1) * inv_d
            else:
                I_mat = np.eye(n_comp, dtype=np.float32)
                M = (I_mat[np.newaxis, :, :]
                     + ds_g[:, np.newaxis, np.newaxis]
                     * C[np.newaxis, :, :])
                a_col = amp_g.T[:, :, np.newaxis]
                a_corr = np.linalg.solve(M, a_col)
                amp_g = a_corr[:, :, 0].T

            amp_out[:, clean_orig] = amp_g
            chi2_out[clean_orig] = chi2_np
            dE_fine_out[clean_orig] = dE_g
            dsigma_fine_out[clean_orig] = ds_g

    # Total shifts: coarse + fine (fine=0 for amp_only spectra)
    energy_shifts = (dE_coarse + dE_fine_out).astype(np.float32)
    sigma_shifts = (dsigma_coarse + dsigma_fine_out).astype(np.float32)

    return amp_out, chi2_out, energy_shifts, sigma_shifts, best_idx
