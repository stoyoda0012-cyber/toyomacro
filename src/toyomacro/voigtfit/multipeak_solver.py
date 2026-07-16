"""
Multi-Peak Alternating Projection Solver.

Production implementation of 2+ component peak fitting:
    - Per-component 2D dictionaries (reuses existing DictionaryCache)
    - Alternating Projection with orthogonal deflation
    - Parabola sub-grid interpolation (δE axis by default)
    - Batch LLS amplitude estimation (2×2 Cramer / general solve)
    - Chunked processing for 8K (33M pixel) data

dev-log 56 Phase 3 → Production.
GPU-native alternating loop (3.6x speedup at 500K).

Architecture:
    Phase 0: Per-component dictionary generation (build_multipeak_dictionaries)
    Phase 1: Alternating Projection — GPU-native (n_iterations × n_comp matmuls)
             Y transferred once, lazy MLX graph, single mx.eval at end.
             CPU↔GPU transfers: 2×(iter×comp) → 2.
    Phase 1.5: Parabola sub-grid refinement on last iteration scores (CPU)
    Phase 2: Joint LLS + chi2 — GPU Cramer's rule for n_comp ≤ 2

Tensor decomposition:
    4D direct-product dictionary (96K × n_E) is NOT materialized.
    Instead, per-component 2D dictionaries D₁(310 × n_E), D₂(310 × n_E)
    are used with Alternating Projection — memory 1/156 of direct product.

Performance (M3 Max, 2-comp, 500K batch):
     0.40 M/s  (CPU projection, per-iter mx.eval)
    + GPU-native loop:    1.46 M/s  (3.6x — lazy MLX graph, 2 sync points)
    + GPU 5-point gather: 3.28 M/s  (8.6x — parabola on GPU, no score transfer)
"""

import time
from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx

    from ._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND a Metal device works
except ImportError:
    HAS_MLX = False

from .dictionary_solver import (
    DictionaryCache,
    build_dictionary,
    parabola_refine_2d,
)
from .gp_compact import (
    GPLMConfig,
    GPLMDiagnostics,
    lm_scale_diag,
    lm_step,
    predicted_reduction,
)
from .grids import is_uniform, precompute_grid_spacings, uniform_grid
from .multipeak_config import ComponentConfig, MultiPeakConfig

if HAS_MLX:
    # Exact per-spectrum Voigt + analytic Jacobian for the exact-Jacobian
    # Newton step (Task A1). Guarded: the numpy fallback never reaches it.
    from .faddeeva_mlx import voigt_exact_jacobian_perspectrum_mlx

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MultiPeakResult:
    """Result from multi-peak alternating projection solver.

    Attributes:
        amplitudes: (batch, n_comp) peak amplitudes.
        delta_E: (batch, n_comp) energy shift per component (eV).
        delta_sigma: (batch, n_comp) width shift per component (eV).
        chi2: (batch,) reduced χ² per spectrum.
        n_iterations: Number of alternating iterations used.
        best_indices: (batch, n_comp) flat dictionary indices.
        timing: Processing time breakdown.
        separation_warning: True if peaks are closer than 3.5σ.
        solvability: CRLB-based solvability assessment (optional).
        cond_number: (batch,) condition number of the post-AP Newton Hessian
            (only set when quality_flags=True). Large => near-degenerate fit,
            e.g. the chi²-flat δσ₁−δσ₂ valley of two overlapping peaks where the
            individual widths are poorly constrained though their sum is not.
        param_std: (batch, 2·n_comp) marginal 1σ of the Newton parameters
            [δE₀, δσ₀, δE₁, δσ₁, …] in eV = sqrt(chi2·diag(H⁻¹)). quality_flags only.
        soft_std: (batch,) 1σ along the least-constrained parameter direction
            = sqrt(chi2 / λ_min(H)); the degeneracy magnitude. quality_flags only.
        background: (batch, bg_degree+1) fitted polynomial background coefficients
            (only set when bg_degree>=0; None otherwise). The columns multiply
            the normalized-energy monomials x**j, x=(E-E.mean())/(ptp(E)/2).
        delta_gamma: (batch, n_comp) per-spectrum Lorentzian-width shift δγ (eV)
            from the post-AP exact Newton step (only set when fit_gamma=True;
            None otherwise). γ_recovered = nominal_gamma + delta_gamma. When the
            chi² gate is active (fit_gamma=True, gamma_chi2_gate=True), this is
            the δγ ACTUALLY APPLIED per spectrum: 0 on spectra where the gate
            rejected the (δE, δσ, δγ) fit in favour of the (δE, δσ) fit.
    """

    amplitudes: np.ndarray
    delta_E: np.ndarray
    delta_sigma: np.ndarray
    chi2: np.ndarray
    n_iterations: int
    best_indices: np.ndarray
    timing: dict | None = None
    separation_warning: bool = False
    solvability: object = None  # Optional[SolvabilityInfo], lazy import
    cond_number: np.ndarray | None = None
    param_std: np.ndarray | None = None
    soft_std: np.ndarray | None = None
    background: np.ndarray | None = None  # (batch, n_bg) BG coeffs when bg_degree>=0
    delta_gamma: np.ndarray | None = None  # (batch, n_comp) δγ when fit_gamma=True
    gp_diagnostics: object = None  # GPLMDiagnostics when newton_jacobian_mode="gp_lm"


@dataclass
class MultiPeak2StageResult:
    """Result from 2-stage γ-calibrated multi-peak solver.

    Stage 1a: Dict2D alternating projection on calibration subset.
    Stage 1b: Per-component γ estimation via deflation + Dict3D.
    Stage 2: Full alternating projection with corrected γ values.

    Attributes:
        amplitudes: (batch, n_comp) peak amplitudes (Stage 2).
        delta_E: (batch, n_comp) energy shift per component (eV).
        delta_sigma: (batch, n_comp) width shift per component (eV).
        chi2: (batch,) reduced χ² per spectrum.
        n_iterations: Number of alternating iterations used.
        best_indices: (batch, n_comp) flat dictionary indices (Stage 2).
        timing: Processing time breakdown.
        separation_warning: True if peaks are closer than 3.5σ.
        corrected_gammas: (n_comp,) per-component corrected γ values.
        global_dgammas: (n_comp,) per-component estimated Δγ.
        dgamma_per_spectrum: list of (n_cal,) arrays, one per component.
        dgamma_stds: (n_comp,) std of per-spectrum δγ per component.
        n_calibration: Number of calibration spectra used in Stage 1.
    """

    amplitudes: np.ndarray
    delta_E: np.ndarray
    delta_sigma: np.ndarray
    chi2: np.ndarray
    n_iterations: int
    best_indices: np.ndarray
    timing: dict | None = None
    separation_warning: bool = False
    corrected_gammas: np.ndarray | None = None
    global_dgammas: np.ndarray | None = None
    dgamma_per_spectrum: list | None = None
    dgamma_stds: np.ndarray | None = None
    n_calibration: int = 0


# ---------------------------------------------------------------------------
# Dictionary building
# ---------------------------------------------------------------------------


def build_multipeak_dictionaries(
    config: MultiPeakConfig,
    grid_type: str = "uniform",
) -> list[DictionaryCache]:
    """Build per-component 2D dictionaries.

    Each component gets an independent DictionaryCache with n_comp=1.
    Reuses existing ``build_dictionary()`` infrastructure.

    Args:
        config: MultiPeakConfig with peak definitions and energy axis.
        grid_type: Grid type — "uniform" (default), "chebyshev", or "sinh".

    Returns:
        List of DictionaryCache, one per component.
    """
    config.check_separation()

    energy = np.asarray(config.energy_axis, dtype=np.float32)
    dicts = []

    for peak in config.peaks:
        # Build grid overrides for custom sizes
        if grid_type == "uniform":
            dE_grid = uniform_grid(peak.n_dE, peak.dE_range).astype(np.float32)
            ds_grid = uniform_grid(peak.n_ds, peak.ds_range).astype(np.float32)
        elif grid_type == "chebyshev":
            from .grids import chebyshev_grid

            dE_grid = chebyshev_grid(peak.n_dE, peak.dE_range).astype(np.float32)
            ds_grid = chebyshev_grid(peak.n_ds, peak.ds_range).astype(np.float32)
        elif grid_type == "sinh":
            from .grids import sinh_grid

            dE_grid = sinh_grid(peak.n_dE, peak.dE_range).astype(np.float32)
            ds_grid = sinh_grid(peak.n_ds, peak.ds_range).astype(np.float32)
        else:
            raise ValueError(f"Unknown grid_type: {grid_type!r}")

        dc = build_dictionary(
            energy=energy,
            centers=np.array([peak.center], dtype=np.float32),
            sigmas=np.array([peak.sigma], dtype=np.float32),
            gamma=peak.gamma,
            dE_range=(-peak.dE_range, peak.dE_range),
            dsigma_range=(-peak.ds_range, peak.ds_range),
            include_hessian=False,
            dE_grid_override=dE_grid,
            dsigma_grid_override=ds_grid,
            so_split=peak.so_split,
            branch_ratio=peak.branch_ratio,
        )
        dicts.append(dc)

    return dicts


# ---------------------------------------------------------------------------
# Batch LLS amplitude estimation
# ---------------------------------------------------------------------------


def _batch_lls_amplitude(
    Y: np.ndarray,
    dicts: list[DictionaryCache],
    best_indices: np.ndarray,
) -> np.ndarray:
    """Batch LLS amplitude estimation.

    For n_comp=2, uses Cramer's rule (fully vectorized, no per-spectrum loop).
    For n_comp>2, uses batched np.linalg.solve on the Gram matrix.

    Args:
        Y: (batch, n_E) spectra.
        dicts: Per-component dictionaries (n_comp=1 each).
        best_indices: (batch, n_comp) flat dictionary indices.

    Returns:
        (batch, n_comp) amplitudes float32.
    """
    n_comp = len(dicts)

    # Gather best-match unnormalized profiles: (batch, n_E) per component
    D_best = []
    for k in range(n_comp):
        phi_k = dicts[k].Phi_per_grid[best_indices[:, k], :, 0]  # (batch, n_E)
        D_best.append(phi_k)

    if n_comp == 2:
        # 2×2 Cramer's rule (fully vectorized)
        d1, d2 = D_best[0], D_best[1]

        g11 = np.sum(d1 * d1, axis=1)
        g12 = np.sum(d1 * d2, axis=1)
        g22 = np.sum(d2 * d2, axis=1)
        r1 = np.sum(d1 * Y, axis=1)
        r2 = np.sum(d2 * Y, axis=1)

        det = g11 * g22 - g12 * g12
        det = np.where(np.abs(det) > 1e-30, det, 1e-30)

        amp_0 = (g22 * r1 - g12 * r2) / det
        amp_1 = (g11 * r2 - g12 * r1) / det
        return np.stack([amp_0, amp_1], axis=1).astype(np.float32)

    elif n_comp == 1:
        d = D_best[0]
        d_sq = np.sum(d * d, axis=1)
        d_sq = np.where(d_sq > 1e-30, d_sq, 1e-30)
        amp = np.sum(d * Y, axis=1) / d_sq
        return amp[:, np.newaxis].astype(np.float32)

    else:
        # General case: batch Gram matrix solve
        batch = Y.shape[0]
        G = np.zeros((batch, n_comp, n_comp), dtype=np.float64)
        rhs = np.zeros((batch, n_comp), dtype=np.float64)

        for i in range(n_comp):
            rhs[:, i] = np.sum(D_best[i].astype(np.float64) * Y.astype(np.float64), axis=1)
            for j in range(i, n_comp):
                gij = np.sum(
                    D_best[i].astype(np.float64) * D_best[j].astype(np.float64),
                    axis=1,
                )
                G[:, i, j] = gij
                if i != j:
                    G[:, j, i] = gij

        # Regularize
        G += 1e-10 * np.eye(n_comp, dtype=np.float64)[np.newaxis, :, :]
        # np.linalg.solve gufunc: (m,m),(m,n)->(m,n) — rhs must be 3D for batched solve
        amp = np.linalg.solve(G, rhs[:, :, np.newaxis])[:, :, 0]
        return amp.astype(np.float32)


# ---------------------------------------------------------------------------
# Orthogonal projection
# ---------------------------------------------------------------------------


def _project_orthogonal(
    Y: np.ndarray,
    dicts: list[DictionaryCache],
    best_np: np.ndarray,
    comp: int,
    n_comp: int,
) -> np.ndarray:
    """Project Y orthogonal to all components except ``comp``.

    For n_comp=2: single-vector projection (fast path).
    For n_comp>2: project ⊥ span of other components via Gram matrix.
    """
    if n_comp == 1:
        return Y

    if n_comp == 2:
        other = 1 - comp
        phi = dicts[other].Phi_per_grid[best_np[:, other], :, 0]  # (batch, n_E)
        phi_sq = np.sum(phi * phi, axis=1, keepdims=True) + 1e-10
        phi_Y = np.sum(phi * Y, axis=1, keepdims=True)
        return Y - (phi_Y / phi_sq) * phi

    # General case: project ⊥ span of other components
    others = [
        dicts[k].Phi_per_grid[best_np[:, k], :, 0]
        for k in range(n_comp)
        if k != comp
    ]
    Phi_other = np.stack(others, axis=2)  # (batch, n_E, K-1)
    K = Phi_other.shape[2]

    G = np.einsum("bek,bel->bkl", Phi_other, Phi_other)
    G += 1e-10 * np.eye(K, dtype=np.float32)[np.newaxis, :, :]
    rhs = np.einsum("bek,be->bk", Phi_other, Y)
    # np.linalg.solve gufunc: (m,m),(m,n)->(m,n) — rhs must be 3D for batched solve
    coeff = np.linalg.solve(G, rhs[:, :, np.newaxis])[:, :, 0]
    proj = np.einsum("bek,bk->be", Phi_other, coeff)
    return Y - proj


# ---------------------------------------------------------------------------
# Parabola refinement helpers
# ---------------------------------------------------------------------------


def _refine_parabola_from_scores(
    scores: np.ndarray,
    best_idx: np.ndarray,
    dc: DictionaryCache,
    apply_dE: bool,
    apply_ds: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract refined δE, δσ from score matrix using parabola interpolation.

    Uses existing ``parabola_refine_2d`` when both axes are requested,
    otherwise does selective refinement.

    Args:
        scores: (batch, n_dict) similarity scores. None to skip parabola.
        best_idx: (batch,) flat dictionary indices.
        dc: DictionaryCache for this component.
        apply_dE: Apply parabola to δE axis.
        apply_ds: Apply parabola to δσ axis.

    Returns:
        (dE, ds) each (batch,) float32.
    """
    n_dE, n_ds = dc.grid_shape
    dE_idx = best_idx // n_ds
    ds_idx = best_idx % n_ds

    if scores is not None and (apply_dE or apply_ds):
        dE_ref, ds_ref = parabola_refine_2d(
            scores, best_idx, dc.grid_shape, dc.dE_grid, dc.dsigma_grid
        )
        dE_out = dE_ref if apply_dE else dc.dE_grid[dE_idx].astype(np.float32)
        ds_out = ds_ref if apply_ds else dc.dsigma_grid[ds_idx].astype(np.float32)
    else:
        dE_out = dc.dE_grid[dE_idx].astype(np.float32)
        ds_out = dc.dsigma_grid[ds_idx].astype(np.float32)

    return dE_out, ds_out


# ---------------------------------------------------------------------------
# GPU parabola refinement
# ---------------------------------------------------------------------------


def _parabola_vertex_mx(s_m, s_0, s_p):
    """GPU parabola vertex in grid units (uniform spacing).

    Same formula as ``_parabola_vertex`` but on MLX arrays.
    Returns fractional grid shift in [-0.5, +0.5].
    """
    den = s_m - 2.0 * s_0 + s_p
    safe = mx.abs(den) > 1e-30
    safe_den = mx.where(safe, den, 1.0)
    delta = 0.5 * (s_m - s_p) / safe_den
    delta = mx.where(safe, delta, 0.0)
    return mx.clip(delta, -0.5, 0.5)


def _parabola_vertex_nonuniform_mx(s_m, s_0, s_p, h_L, h_R):
    """GPU parabola vertex for non-uniform spacing.

    Same formula as ``_parabola_vertex_nonuniform`` but on MLX arrays.
    Returns shift in real coordinate units, clamped to [-h_L, +h_R].
    """
    h_L2 = h_L * h_L
    h_R2 = h_R * h_R
    h_sum = h_L + h_R
    num = -(h_L2 * s_p - (h_L2 - h_R2) * s_0 - h_R2 * s_m)
    den = 2.0 * (h_L * s_p - h_sum * s_0 + h_R * s_m)
    safe = mx.abs(den) > 1e-30
    safe_den = mx.where(safe, den, 1.0)
    shift = mx.where(safe, num / safe_den, 0.0)
    return mx.maximum(mx.minimum(shift, h_R), -h_L)


def _parabola_refine_gpu(scores_mx, best_mx, dc, apply_dE, apply_ds):
    """GPU parabola refinement via 5-point gather.

    Gathers only 5 neighbour scores per spectrum on GPU, computes
    parabola vertex on GPU.  Full scores matrix (batch, n_dict)
    never leaves GPU — transfer reduced from batch×n_dict to batch×2.

    ``solve_dict2d_parabola`` 5-point gather
    (8.1 GB → 20 MB for single-peak) applied to multi-peak.

    Args:
        scores_mx: (batch, n_dict) MLX scores.
        best_mx: (batch,) MLX int32 best indices.
        dc: DictionaryCache for this component.
        apply_dE: Apply parabola to δE axis.
        apply_ds: Apply parabola to δσ axis.

    Returns:
        (dE_refined_mx, ds_refined_mx): lazy MLX arrays (batch,) each.
    """
    n_dE, n_ds = dc.grid_shape
    batch = scores_mx.shape[0]

    dE_idx = best_mx // n_ds
    ds_idx = best_mx % n_ds

    spec_idx = mx.arange(batch)
    s0 = scores_mx[spec_idx, best_mx]

    dE_grid_mx = mx.array(dc.dE_grid.astype(np.float32))
    ds_grid_mx = mx.array(dc.dsigma_grid.astype(np.float32))

    # Default: grid values (no parabola)
    dE_refined = dE_grid_mx[dE_idx]
    ds_refined = ds_grid_mx[ds_idx]

    if apply_dE and n_dE > 2:
        dE_m_flat = mx.clip(dE_idx - 1, 0, n_dE - 1) * n_ds + ds_idx
        dE_p_flat = mx.clip(dE_idx + 1, 0, n_dE - 1) * n_ds + ds_idx
        s_dE_m = scores_mx[spec_idx, dE_m_flat]
        s_dE_p = scores_mx[spec_idx, dE_p_flat]
        dE_interior = (dE_idx > 0) & (dE_idx < n_dE - 1)

        if is_uniform(dc.dE_grid):
            dE_step = float(dc.dE_grid[1] - dc.dE_grid[0])
            delta = _parabola_vertex_mx(s_dE_m, s0, s_dE_p)
            delta = mx.where(dE_interior, delta, 0.0)
            dE_refined = dE_grid_mx[dE_idx] + delta * dE_step
        else:
            hL, hR = precompute_grid_spacings(dc.dE_grid)
            hL_mx = mx.array(hL.astype(np.float32))
            hR_mx = mx.array(hR.astype(np.float32))
            shift = _parabola_vertex_nonuniform_mx(
                s_dE_m, s0, s_dE_p, hL_mx[dE_idx], hR_mx[dE_idx]
            )
            shift = mx.where(dE_interior, shift, 0.0)
            dE_refined = dE_grid_mx[dE_idx] + shift

    if apply_ds and n_ds > 2:
        ds_m_flat = dE_idx * n_ds + mx.clip(ds_idx - 1, 0, n_ds - 1)
        ds_p_flat = dE_idx * n_ds + mx.clip(ds_idx + 1, 0, n_ds - 1)
        s_ds_m = scores_mx[spec_idx, ds_m_flat]
        s_ds_p = scores_mx[spec_idx, ds_p_flat]
        ds_interior = (ds_idx > 0) & (ds_idx < n_ds - 1)

        if is_uniform(dc.dsigma_grid):
            ds_step = float(dc.dsigma_grid[1] - dc.dsigma_grid[0])
            delta = _parabola_vertex_mx(s_ds_m, s0, s_ds_p)
            delta = mx.where(ds_interior, delta, 0.0)
            ds_refined = ds_grid_mx[ds_idx] + delta * ds_step
        else:
            hL, hR = precompute_grid_spacings(dc.dsigma_grid)
            hL_mx = mx.array(hL.astype(np.float32))
            hR_mx = mx.array(hR.astype(np.float32))
            shift = _parabola_vertex_nonuniform_mx(
                s_ds_m, s0, s_ds_p, hL_mx[ds_idx], hR_mx[ds_idx]
            )
            shift = mx.where(ds_interior, shift, 0.0)
            ds_refined = ds_grid_mx[ds_idx] + shift

    return dE_refined, ds_refined


# ---------------------------------------------------------------------------
# B3: polynomial background design
# ---------------------------------------------------------------------------


def build_bg_design(energy: np.ndarray, bg_degree: int) -> np.ndarray:
    """Polynomial background design matrix on a normalized energy axis.

    Columns are monomials of x = (E - E.mean()) / (ptp(E)/2) ∈ [-1, 1]:
    column j = x**j for j in 0..bg_degree. The normalization keeps the
    Gram matrix well-conditioned for low degrees (monomials on [-1, 1]).

    Args:
        energy: (n_E,) energy axis.
        bg_degree: polynomial degree (0=constant, 1=+linear, 2=+quadratic, …).

    Returns:
        (n_E, bg_degree+1) float32 design matrix.
    """
    E = np.asarray(energy, dtype=np.float64).ravel()
    half = np.ptp(E) / 2.0
    if half <= 0:
        half = 1.0
    x = (E - E.mean()) / half
    n_bg = bg_degree + 1
    B = np.empty((E.shape[0], n_bg), dtype=np.float64)
    B[:, 0] = 1.0
    for j in range(1, n_bg):
        B[:, j] = x ** j
    return B.astype(np.float32)


# Largest system size routed through the unrolled GPU Cholesky; bigger
# systems fall back to the CPU-stream LAPACK solve (graph size grows ~m³/6).
# m=24 covers the 10-comp doublet Newton system (p=20); measured faster than
# the dense CPU round trip up to there.
_SPD_GPU_MAX_M = 24


def _spd_cholesky_solve_entries(H_entries, g_entries, m):
    """Unrolled batched Cholesky solve on (batch,) entry dicts.

    ``mx.linalg.solve`` only runs on the CPU stream, which inserts a
    GPU→CPU→GPU synchronization barrier per call (~19 ms per 20K batch for a
    4×4 system). The small SPD systems in this module (joint [peaks | BG]
    Gram, n_comp ≥ 3 amplitude Gram, the (δE, δσ[, δγ])×n_comp Newton
    system) are cheap to factorize with elementwise MLX ops unrolled at
    trace time, so everything stays on the GPU and lazy until the caller's
    ``mx.eval``.

    Args:
        H_entries: {(i, j): (batch,) MLX, i ≤ j} upper-triangular entries of
            an SPD matrix (already regularized).
        g_entries: list of m (batch,) MLX right-hand-side entries.
        m: system size (Python int ≤ ``_SPD_GPU_MAX_M``).

    Returns:
        list of m (batch,) MLX solution entries (lazy).
    """
    a = {}
    for j in range(m):
        for i in range(j, m):
            a[(i, j)] = H_entries[(j, i)]
    inv_diag = []
    for j in range(m):
        s = a[(j, j)]
        for k in range(j):
            s = s - a[(j, k)] * a[(j, k)]
        L_jj = mx.sqrt(mx.maximum(s, mx.array(1e-30)))
        inv_d = 1.0 / L_jj
        a[(j, j)] = L_jj
        inv_diag.append(inv_d)
        for i in range(j + 1, m):
            s = a[(i, j)]
            for k in range(j):
                s = s - a[(i, k)] * a[(j, k)]
            a[(i, j)] = s * inv_d
    # Forward substitution L z = g
    z = []
    for i in range(m):
        s = g_entries[i]
        for k in range(i):
            s = s - a[(i, k)] * z[k]
        z.append(s * inv_diag[i])
    # Back substitution Lᵀ x = z
    x = [None] * m
    for i in reversed(range(m)):
        s = z[i]
        for k in range(i + 1, m):
            s = s - a[(k, i)] * x[k]
        x[i] = s * inv_diag[i]
    return x


def _spd_solve_gpu(G, rhs, m):
    """Batched SPD solve on GPU — dense (batch, m, m) wrapper.

    See :func:`_spd_cholesky_solve_entries`. Returns (batch, m) MLX (lazy).
    """
    H_entries = {(i, j): G[:, i, j] for i in range(m) for j in range(i, m)}
    g_entries = [rhs[:, i] for i in range(m)]
    return mx.stack(_spd_cholesky_solve_entries(H_entries, g_entries, m), axis=1)


def _gram_solve_mx(G, rhs, m):
    """Batched Gram solve: GPU Cholesky for small m, CPU LAPACK beyond."""
    if m <= _SPD_GPU_MAX_M:
        return _spd_solve_gpu(G, rhs, m)
    return mx.linalg.solve(G, rhs[:, :, None], stream=mx.cpu)[:, :, 0]


def _lls_chi2_bg_gpu(Y_mx, phis_mx, B_mx, n_comp):
    """Joint [peaks | background] LLS + chi2 on GPU (general n_comp + n_bg).

    Solves the augmented normal equations for [a_1..a_ncomp | bg_0..bg_(n_bg-1)]
    via a batched Gram solve (MLX Gram, CPU LAPACK solve), reconstructs the
    full model (peaks + background) and returns peak amplitudes, background
    coefficients and chi².

    Args:
        Y_mx: (batch, n_E) MLX spectra.
        phis_mx: list of (batch, n_E) MLX per-peak profiles at the AP pick.
        B_mx: (n_E, n_bg) MLX background design matrix.
        n_comp: number of peaks.

    Returns:
        (amplitudes, background, chi2) numpy arrays:
        (batch, n_comp), (batch, n_bg), (batch,) float32.
    """
    n_bg = B_mx.shape[1]
    batch = Y_mx.shape[0]
    # Assemble per-spectrum design columns: peaks (batch, n_E) + bg (broadcast).
    cols = list(phis_mx)
    B_b = mx.broadcast_to(B_mx[None, :, :], (batch, B_mx.shape[0], n_bg))
    for j in range(n_bg):
        cols.append(B_b[:, :, j])
    D = mx.stack(cols, axis=2)  # (batch, n_E, n_comp+n_bg)
    n_tot = n_comp + n_bg
    D_T = mx.transpose(D, (0, 2, 1))  # (batch, n_tot, n_E)
    G_mx = D_T @ D + 1e-8 * mx.eye(n_tot)
    rhs_mx = (D_T @ Y_mx[:, :, None])[:, :, 0]  # (batch, n_tot)
    coef_mx = _gram_solve_mx(G_mx, rhs_mx, n_tot)
    recon_mx = (D @ coef_mx[:, :, None])[:, :, 0]
    chi2_mx = mx.mean((Y_mx - recon_mx) ** 2, axis=1)
    mx.eval(coef_mx, chi2_mx)
    coef = np.array(coef_mx).astype(np.float32)
    chi2 = np.array(chi2_mx).astype(np.float32)
    return coef[:, :n_comp], coef[:, n_comp:], chi2


def _lls_chi2_bg_numpy(Y, phis, B, n_comp):
    """Joint [peaks | background] LLS + chi2 on CPU (numpy fallback).

    See :func:`_lls_chi2_bg_gpu`. ``phis`` is a list of (batch, n_E) arrays;
    ``B`` is (n_E, n_bg).
    """
    n_bg = B.shape[1]
    batch = Y.shape[0]
    cols = [p.astype(np.float64) for p in phis]
    for j in range(n_bg):
        cols.append(np.broadcast_to(B[:, j].astype(np.float64), (batch, B.shape[0])))
    D = np.stack(cols, axis=2)  # (batch, n_E, n_tot)
    n_tot = n_comp + n_bg
    G = np.einsum("bec,bef->bcf", D, D) + 1e-8 * np.eye(n_tot)
    rhs = np.einsum("bec,be->bc", D, Y.astype(np.float64))
    coef = np.linalg.solve(G, rhs[:, :, None])[:, :, 0]
    recon = np.einsum("bec,bc->be", D, coef)
    chi2 = np.mean((Y.astype(np.float64) - recon) ** 2, axis=1).astype(np.float32)
    coef = coef.astype(np.float32)
    return coef[:, :n_comp], coef[:, n_comp:], chi2


# ---------------------------------------------------------------------------
# GPU-native LLS + chi2
# ---------------------------------------------------------------------------


def _lls_chi2_gpu(Y_mx, Phi_mx, best_mx, n_comp):
    """GPU-native joint LLS amplitude + chi2 for n_comp <= 2.

    Avoids CPU gather by indexing directly into GPU Phi matrices.
    Single mx.eval call for both amplitudes and chi2.

    Args:
        Y_mx: (batch, n_E) MLX array — spectra on GPU.
        Phi_mx: list of (n_dict, n_E) MLX arrays (one per component).
        best_mx: list of (batch,) MLX int32 arrays (best indices).
        n_comp: Number of components (1 or 2).

    Returns:
        (amplitudes, chi2): numpy arrays, shapes (batch, n_comp) and (batch,).
    """
    if n_comp == 1:
        phi = Phi_mx[0][best_mx[0]]  # (batch, n_E)
        d_sq = mx.sum(phi * phi, axis=1) + 1e-10
        amp = mx.sum(phi * Y_mx, axis=1) / d_sq
        recon = amp[:, None] * phi
        chi2_mx = mx.mean((Y_mx - recon) ** 2, axis=1)
        mx.eval(amp, chi2_mx)
        return (
            np.array(amp)[:, np.newaxis].astype(np.float32),
            np.array(chi2_mx).astype(np.float32),
        )

    # n_comp == 2: Cramer's rule (fully vectorized on GPU)
    phi1 = Phi_mx[0][best_mx[0]]  # (batch, n_E)
    phi2 = Phi_mx[1][best_mx[1]]

    g11 = mx.sum(phi1 * phi1, axis=1)
    g12 = mx.sum(phi1 * phi2, axis=1)
    g22 = mx.sum(phi2 * phi2, axis=1)
    r1 = mx.sum(phi1 * Y_mx, axis=1)
    r2 = mx.sum(phi2 * Y_mx, axis=1)

    det = g11 * g22 - g12 * g12
    safe_det = mx.where(mx.abs(det) > 1e-30, det, mx.array(1e-30))

    amp_0 = (g22 * r1 - g12 * r2) / safe_det
    amp_1 = (g11 * r2 - g12 * r1) / safe_det

    recon = amp_0[:, None] * phi1 + amp_1[:, None] * phi2
    chi2_mx = mx.mean((Y_mx - recon) ** 2, axis=1)

    mx.eval(amp_0, amp_1, chi2_mx)
    amplitudes = np.stack(
        [np.array(amp_0), np.array(amp_1)], axis=1
    ).astype(np.float32)
    return amplitudes, np.array(chi2_mx).astype(np.float32)


# ---------------------------------------------------------------------------
# Multipeak Hessian cross-talk cache
# ---------------------------------------------------------------------------


@dataclass
class MultipeakHessianCache:
    """Precomputed per-grid-cell Hessian entries for the Newton step.

    Each entry is the energy-axis integral of two Jacobians evaluated AT the
    grid cells; it does not depend on the AP-refined offsets or the spectrum
    amplitudes, so it is fully cacheable.

    Within-peak (peak k):
        A_cc[k][i_k] = Σ_E J_c_k[i_k, E] · J_c_k[i_k, E]
        A_cs[k][i_k] = Σ_E J_c_k[i_k, E] · J_σ_k[i_k, E]
        A_ss[k][i_k] = Σ_E J_σ_k[i_k, E] · J_σ_k[i_k, E]

    Between-peak (only present for n_comp == 2):
        B_cc[i_1, i_2] = Σ_E J_c_1[i_1, E] · J_c_2[i_2, E]
        B_cs[i_1, i_2] = Σ_E J_c_1[i_1, E] · J_σ_2[i_2, E]
        B_sc[i_1, i_2] = Σ_E J_σ_1[i_1, E] · J_c_2[i_2, E]
        B_ss[i_1, i_2] = Σ_E J_σ_1[i_1, E] · J_σ_2[i_2, E]

    Newton uses these to assemble the per-spectrum 4×4 Hessian via a single
    gather + scalar multiplication by the amplitude product (a_k · a_l)
    instead of 10 reductions over the energy axis.

    Memory: 3·Σ_k n_dict_k + 4·n_dict_1·n_dict_2 floats. For dev-log 77 setup
    (n_dict_k = 465, 2 peaks) this is ~8.6 MB.
    """
    A_cc: list                     # list of MLX (n_dict_k,) per peak
    A_cs: list
    A_ss: list
    B_cc: object | None = None     # (n_dict_1, n_dict_2) MLX or None
    B_cs: object | None = None
    B_sc: object | None = None
    B_ss: object | None = None
    n_comp: int = 0


def build_multipeak_hessian_cache(
    dicts: list[DictionaryCache],
) -> MultipeakHessianCache:
    """Precompute Hessian cross-talk dictionary for multipeak Newton.

    Builds the 3 within-peak vectors (per peak) and the 4 between-peak
    matrices (only for n_comp == 2). All tensors live on GPU via MLX.

    Args:
        dicts: per-peak DictionaryCache list (each with n_comp_internal=1).

    Returns:
        MultipeakHessianCache ready to be consumed by the cached Newton path.
    """
    if not HAS_MLX:
        raise RuntimeError("build_multipeak_hessian_cache requires MLX.")

    n_comp = len(dicts)
    A_cc: list = []
    A_cs: list = []
    A_ss: list = []
    for k in range(n_comp):
        dc = dicts[k]
        dc.to_mlx()
        Jc_k = dc._Jc_mx[:, :, 0]        # (n_dict_k, n_E)
        Js_k = dc._Jsigma_mx[:, :, 0]
        A_cc.append(mx.sum(Jc_k * Jc_k, axis=1))   # (n_dict_k,)
        A_cs.append(mx.sum(Jc_k * Js_k, axis=1))
        A_ss.append(mx.sum(Js_k * Js_k, axis=1))

    B_cc = B_cs = B_sc = B_ss = None
    if n_comp == 2:
        Jc_1 = dicts[0]._Jc_mx[:, :, 0]
        Js_1 = dicts[0]._Jsigma_mx[:, :, 0]
        Jc_2 = dicts[1]._Jc_mx[:, :, 0]
        Js_2 = dicts[1]._Jsigma_mx[:, :, 0]
        # B[i,j] = Σ_E J_*1[i, E] · J_*2[j, E] = J_*1 @ J_*2.T
        B_cc = Jc_1 @ mx.transpose(Jc_2, (1, 0))
        B_cs = Jc_1 @ mx.transpose(Js_2, (1, 0))
        B_sc = Js_1 @ mx.transpose(Jc_2, (1, 0))
        B_ss = Js_1 @ mx.transpose(Js_2, (1, 0))

    mx.eval(*A_cc, *A_cs, *A_ss,
            *(b for b in (B_cc, B_cs, B_sc, B_ss) if b is not None))

    return MultipeakHessianCache(
        A_cc=A_cc, A_cs=A_cs, A_ss=A_ss,
        B_cc=B_cc, B_cs=B_cs, B_sc=B_sc, B_ss=B_ss,
        n_comp=n_comp,
    )


# ---------------------------------------------------------------------------
# B4: per-spectrum quality diagnostics from the Newton Hessian
# ---------------------------------------------------------------------------


def _newton_diagnostics(H_dense, chi2, eps_rel=1e-12):
    """Per-spectrum uncertainty diagnostics from the Newton Hessian.

    The post-AP Gauss-Newton Hessian H = Σ_E (a·J)(a·J)ᵀ is (up to the noise
    variance) the Fisher information of the (δE, δσ) parameters. Treating the
    residual variance estimate ``chi2`` (mean squared residual) as the noise
    variance σ², the parameter covariance is Cov ≈ σ²·H⁻¹. From it:

      * cond_number = λ_max(H) / λ_min(H) — flags near-degenerate fits, e.g.
        two overlapping peaks whose δσ₁−δσ₂ direction is chi²-flat (the
        dev-log 135 doublet valley).
      * param_std[i] = sqrt(chi2 · (H⁻¹)_ii) — marginal 1σ of each parameter.
      * soft_std = sqrt(chi2 / λ_min(H)) — 1σ along the least-constrained
        eigen-direction (the degeneracy magnitude; >> param_std when degenerate).

    Args:
        H_dense: (batch, p, p) float32 raw Gauss-Newton Hessian (no Tikhonov).
        chi2: (batch,) mean squared residual (noise-variance estimate).
        eps_rel: floor on λ_min as a fraction of λ_max (numerical guard).

    Returns:
        dict with 'cond_number' (batch,), 'param_std' (batch, p),
        'soft_std' (batch,) — all numpy float32.
    """
    H64 = H_dense.astype(np.float64)
    p = H64.shape[1]
    eigvals = np.linalg.eigvalsh(H64)                 # ascending, (batch, p)
    lam_max = eigvals[:, -1]
    floor = np.maximum(lam_max * eps_rel, 1e-30)
    lam_min = np.maximum(eigvals[:, 0], floor)
    cond = (lam_max / lam_min).astype(np.float32)

    H_reg = H64 + (floor[:, None, None] * np.eye(p)[None])
    cov_diag = np.linalg.inv(H_reg)[:, np.arange(p), np.arange(p)]
    cov_diag = np.maximum(cov_diag, 0.0)
    chi2_64 = chi2.astype(np.float64)[:, None]
    param_std = np.sqrt(chi2_64 * cov_diag).astype(np.float32)
    soft_std = np.sqrt(chi2.astype(np.float64) / lam_min).astype(np.float32)
    return {"cond_number": cond, "param_std": param_std, "soft_std": soft_std}


# ---------------------------------------------------------------------------
# Post-AP Newton refinement
# ---------------------------------------------------------------------------


def _solve_small_system_gpu(H_entries, g_entries, n_comp, batch):
    """Closed-form batched solve of the (2n_comp)×(2n_comp) Newton system on GPU.

    For n_comp ≤ 2 the system is 2×2 or 4×4 SPD; we apply the same
    ``1e-6·trace(H)`` Tikhonov diagonal load as the CPU path and solve
    analytically — n_comp=1 via the 2×2 inverse, n_comp=2 via a 2×2 block
    (Schur-complement) inverse. All tensors stay on GPU; only ``delta``
    (batch, 2n_comp) is pulled to numpy at the end.

    Args:
        H_entries: dict {(i, j): (batch,) MLX} of upper-triangular Hessian
            entries (i ≤ j); symmetric counterparts are inferred.
        g_entries: list of (batch,) MLX gradient entries, length 2n_comp.
        n_comp: number of peaks (must be 1 or 2).
        batch: batch size (unused; kept for signature symmetry).

    Returns:
        delta — numpy (batch, 2n_comp) float32.
    """
    n_params = 2 * n_comp
    # Symmetric trace = Σ diagonal entries
    trace = H_entries[(0, 0)]
    for d in range(1, n_params):
        trace = trace + H_entries[(d, d)]
    reg = 1e-6 * trace

    if n_comp == 1:
        # H = [[a, b], [b, c]] + reg·I ; g = [g0, g1]
        a = H_entries[(0, 0)] + reg
        b = H_entries[(0, 1)]
        c = H_entries[(1, 1)] + reg
        g0, g1 = g_entries
        det = a * c - b * b
        inv_det = 1.0 / mx.where(mx.abs(det) > 1e-30, det, mx.array(1e-30))
        d0 = (c * g0 - b * g1) * inv_det
        d1 = (a * g1 - b * g0) * inv_det
        mx.eval(d0, d1)
        return np.stack([np.asarray(d0), np.asarray(d1)], axis=1).astype(np.float32)

    # n_comp == 2 → 4×4 SPD via 2×2 block inverse:
    #   H = [[A, B], [B^T, C]], delta = H^{-1} g.
    # Block A is the peak-0 (δE0, δσ0) 2×2; C is peak-1; B is the 2×2 cross-talk.
    a00 = H_entries[(0, 0)] + reg
    a01 = H_entries[(0, 1)]
    a11 = H_entries[(1, 1)] + reg
    c00 = H_entries[(2, 2)] + reg
    c01 = H_entries[(2, 3)]
    c11 = H_entries[(3, 3)] + reg
    # B (rows = peak0 params, cols = peak1 params)
    b00 = H_entries[(0, 2)]
    b01 = H_entries[(0, 3)]
    b10 = H_entries[(1, 2)]
    b11 = H_entries[(1, 3)]

    # A^{-1}
    detA = a00 * a11 - a01 * a01
    invA = 1.0 / mx.where(mx.abs(detA) > 1e-30, detA, mx.array(1e-30))
    iA00 = a11 * invA
    iA01 = -a01 * invA
    iA11 = a00 * invA

    # M = A^{-1} B  (2×2)
    m00 = iA00 * b00 + iA01 * b10
    m01 = iA00 * b01 + iA01 * b11
    m10 = iA01 * b00 + iA11 * b10
    m11 = iA01 * b01 + iA11 * b11

    # Schur complement S = C - B^T A^{-1} B = C - B^T M
    s00 = c00 - (b00 * m00 + b10 * m10)
    s01 = c01 - (b00 * m01 + b10 * m11)
    s11 = c11 - (b01 * m01 + b11 * m11)

    detS = s00 * s11 - s01 * s01
    invS = 1.0 / mx.where(mx.abs(detS) > 1e-30, detS, mx.array(1e-30))
    iS00 = s11 * invS
    iS01 = -s01 * invS
    iS11 = s00 * invS

    g0, g1, g2, g3 = g_entries
    # Solve via block formulas:
    #   y2 = S^{-1} (g_C - B^T A^{-1} g_A)
    #   y1 = A^{-1} (g_A - B y2)
    # g_A = [g0, g1], g_C = [g2, g3]
    # B^T A^{-1} g_A = M^T g_A  (since M = A^{-1} B → B^T A^{-1} = M^T)
    bt_gA_0 = m00 * g0 + m10 * g1
    bt_gA_1 = m01 * g0 + m11 * g1
    rc0 = g2 - bt_gA_0
    rc1 = g3 - bt_gA_1
    y2_0 = iS00 * rc0 + iS01 * rc1
    y2_1 = iS01 * rc0 + iS11 * rc1
    # g_A - B y2
    bA0 = g0 - (b00 * y2_0 + b01 * y2_1)
    bA1 = g1 - (b10 * y2_0 + b11 * y2_1)
    y1_0 = iA00 * bA0 + iA01 * bA1
    y1_1 = iA01 * bA0 + iA11 * bA1

    mx.eval(y1_0, y1_1, y2_0, y2_1)
    return np.stack(
        [np.asarray(y1_0), np.asarray(y1_1), np.asarray(y2_0), np.asarray(y2_1)],
        axis=1,
    ).astype(np.float32)


def _post_ap_newton_refine_mlx(
    Y_mx,
    dicts: list[DictionaryCache],
    best_mx: list,
    delta_E: np.ndarray,
    delta_sigma: np.ndarray,
    amplitudes: np.ndarray,
    clip_step_ratio: float = 0.5,
    hessian_cache: MultipeakHessianCache | None = None,
    n_iter: int = 1,
    exact_jacobian: bool = False,
    compute_diagnostics: bool = False,
    bg_design_mx=None,
    fit_gamma: bool = False,
    lambda_gamma: float = 1e-3,
    gamma_prior_std: float | None = 0.005,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict | None,
    np.ndarray | None, np.ndarray | None,
]:
    """GPU-native (MLX) post-AP Newton step (1+ iterations).

    Equivalent to :func:`_post_ap_newton_refine_chunk` but with the heavy
    element-wise + reduction ops on Apple Silicon GPU. For n_comp ≤ 2 the
    (2n_comp)×(2n_comp) system is solved analytically on GPU (Schur complement,
    see :func:`_solve_small_system_gpu`); for n_comp ≥ 3 it falls back to numpy
    on CPU. Y stays on the GPU the entire time (no extra CPU↔GPU roundtrip on Y).

    With `n_iter > 1`, the body is repeated, each iteration refining
    (δE, δσ, a) from the previous step's output.

    Two linearization modes:

    * **Frozen surrogate** (``exact_jacobian=False``, default): the per-peak
      basis and Jacobians are gathered once at the AP-picked grid cell and held
      constant; each iteration builds a first-order Taylor surrogate
      ``Φ̂_k = Φ_g_k + (δE-δE_grid)·J_c + (δσ-δσ_grid)·J_σ``. The Hessian
      cross-talk cache (``hessian_cache``) accelerates H assembly because the
      grid-cell Jacobians never change. The fixed linearization error caps the
      attainable δσ recovery (≈46 dB on the 2-peak dev-log 77 setup).

    * **Exact Jacobian** (``exact_jacobian=True``): each iteration re-evaluates
      the *exact* per-peak basis Φ_k and analytic Jacobians J_c, J_σ at the
      current absolute parameters ``center_k = nominal_center + δE_cum`` and
      ``sigma_k = nominal_sigma + δσ_cum`` (gamma fixed), via the batched
      analytic Voigt routine. Because the Jacobians change every iteration the
      grid-cell ``hessian_cache`` is invalid here, so H is assembled by
      reductions from the freshly evaluated Jacobians. This removes the frozen
      Taylor ceiling and pushes δσ past it at noise=0.

    Args:
        Y_mx: (batch, n_E) MLX float32 — spectra already on GPU.
        dicts: per-peak DictionaryCache (MLX-converted via .to_mlx()).
        best_mx: list of (batch,) MLX int32 arrays from the AP loop.
        delta_E, delta_sigma: (batch, n_comp) float32 — current refined offsets.
        amplitudes: (batch, n_comp) float32 — Phase-2 LLS amplitudes.
        clip_step_ratio: max step as fraction of grid spacing.
        hessian_cache: optional precomputed Hessian cross-talk cache (ignored
            when exact_jacobian=True).
        n_iter: number of Newton iterations (default 1).
        exact_jacobian: re-evaluate the exact basis/Jacobian each iteration
            (default False; see above).
        compute_diagnostics: if True, also return per-spectrum uncertainty
            diagnostics (cond_number / param_std / soft_std) derived from the
            last-iteration Newton Hessian (default False).
        bg_design_mx: optional (n_E, n_bg) MLX polynomial background design. When
            non-None, each iteration re-solves the JOINT [peaks | background]
            amplitude LLS so the residual driving the (δE, δσ) gradient — and
            the final chi² — are background-subtracted; otherwise the unmodeled
            background leaks back into the width estimate.
        fit_gamma: if True, extend the per-peak Newton parameters to
            (δE, δσ, δγ) — ``n_params = 3·n_comp`` — refining the Lorentzian
            width too. Requires ``exact_jacobian=True`` (γ has no frozen-grid
            basis; the analytic ∂V/∂γ is re-evaluated each iteration). Routes
            the dense CPU solve regardless of n_comp and adds a Levenberg load
            on the γ rows/cols (γ is Fisher-ill-conditioned vs σ).
        lambda_gamma: Levenberg conditioning load on the γ rows/cols when
            fit_gamma (added as ``lambda_gamma · mean_diag``; default 1e-3).
            Stabilises the step direction against the δσ/δγ degeneracy.
        gamma_prior_std: Gaussian ridge-prior width (eV) on the cumulative δγ
            when fit_gamma (default 0.005; None/0 disables). The data already
            resolves δγ only to ≈ sqrt(R_var/H_γγ); a prior at or below that
            resolution suppresses the noise-driven spurious δγ — which is
            anti-correlated ≈ −1 with the induced δσ error via the σ↔γ
            degeneracy — so the nominal-γ δσ stays small, while real γ shifts
            of order this std are still recovered. Implemented in
            noise-normalized units (ρ = R_var/σ_prior²) so it is amplitude- and
            point-count-invariant. Set ~ the data's γ resolving power.

    Returns:
        amplitudes_new, delta_E_new, delta_sigma_new, chi2_new — numpy float32 —
        ``diag``: a dict of per-spectrum diagnostics (or None when
        compute_diagnostics is False; see :func:`_newton_diagnostics`),
        ``bg_cum``: fitted background coeffs (or None), and
        ``dg_cum``: (batch, n_comp) cumulative δγ (or None when not fit_gamma).
    """
    if not HAS_MLX:
        raise RuntimeError("MLX not available; use _post_ap_newton_refine_numpy.")

    if fit_gamma and not exact_jacobian:
        raise ValueError(
            "fit_gamma=True requires exact_jacobian=True: the analytic ∂V/∂γ "
            "is re-evaluated each Newton iteration, and the frozen grid-cell "
            "surrogate has no γ basis."
        )

    batch = int(delta_E.shape[0])
    n_comp = len(dicts)
    pp = 3 if fit_gamma else 2  # params per peak: (δE, δσ[, δγ])
    n_params = pp * n_comp

    use_bg = bg_design_mx is not None
    n_bg = int(bg_design_mx.shape[1]) if use_bg else 0
    bg_b_mx = (
        mx.broadcast_to(bg_design_mx[None, :, :], (batch, bg_design_mx.shape[0], n_bg))
        if use_bg else None
    )

    def _solve_amp_bg(phis):
        """Joint [peaks | background] amplitude LLS on the given per-peak phis.

        phis: list of n_comp (batch, n_E) MLX profiles. Returns
        (amp_mx_peaks (batch, n_comp), bg_mx (batch, n_bg), recon_mx (batch, n_E)).
        """
        cols = list(phis) + [bg_b_mx[:, :, j] for j in range(n_bg)]
        D = mx.stack(cols, axis=2)  # (batch, n_E, n_comp+n_bg)
        n_tot = n_comp + n_bg
        D_T = mx.transpose(D, (0, 2, 1))
        G = D_T @ D + 1e-8 * mx.eye(n_tot)
        rhs = (D_T @ Y_mx[:, :, None])[:, :, 0]
        coef = _gram_solve_mx(G, rhs, n_tot)
        recon = (D @ coef[:, :, None])[:, :, 0]
        return coef[:, :n_comp], coef[:, n_comp:], recon

    # Ensure dicts are on the GPU
    for dc in dicts:
        dc.to_mlx()

    # When exact_jacobian is requested but a dict lacks the nominal model
    # parameters (older cache), fall back to the surrogate path for safety.
    if exact_jacobian:
        for dc in dicts:
            if dc.nominal_centers is None or dc.energy is None:
                exact_jacobian = False
                break

    energy_mx = mx.array(dicts[0].energy.astype(np.float32)) if exact_jacobian else None

    # ------------------------------------------------------------------
    # Setup (constant across Newton iterations):
    #   gather Phi_g, Jc_g, Js_g for each peak's AP-picked grid cell.
    #   (Used by the surrogate path; also the fallback Jacobian source.)
    # ------------------------------------------------------------------
    Phi_g_mx: list = []
    Jc_per_mx: list = []
    Js_per_mx: list = []
    dE_grid_val_mx: list = []
    ds_grid_val_mx: list = []
    dE_step_per = np.empty(n_comp, dtype=np.float32)
    ds_step_per = np.empty(n_comp, dtype=np.float32)
    for k in range(n_comp):
        dc = dicts[k]
        n_ds_k = dc.grid_shape[1]
        dE_grid_mx = mx.array(dc.dE_grid.astype(np.float32))
        ds_grid_mx = mx.array(dc.dsigma_grid.astype(np.float32))
        dE_idx = best_mx[k] // n_ds_k
        ds_idx = best_mx[k] % n_ds_k
        dE_grid_val_mx.append(dE_grid_mx[dE_idx])
        ds_grid_val_mx.append(ds_grid_mx[ds_idx])
        Phi_g_mx.append(dc._Phi_mx[best_mx[k], :, 0])
        Jc_per_mx.append(dc._Jc_mx[best_mx[k], :, 0])
        Js_per_mx.append(dc._Jsigma_mx[best_mx[k], :, 0])
        dE_step_per[k] = (float(dc.dE_grid[1] - dc.dE_grid[0])
                          if len(dc.dE_grid) > 1 else 1.0)
        ds_step_per[k] = (float(dc.dsigma_grid[1] - dc.dsigma_grid[0])
                          if len(dc.dsigma_grid) > 1 else 1.0)

    # Mutable state across iterations
    dE_cum = delta_E.astype(np.float32, copy=True)
    ds_cum = delta_sigma.astype(np.float32, copy=True)
    dg_cum = (np.zeros((batch, n_comp), dtype=np.float32) if fit_gamma else None)
    amp_cum = amplitudes.astype(np.float32, copy=True)
    bg_cum = None  # (batch, n_bg) background coefficients when use_bg
    H_diag_dense = None  # raw last-iter Hessian, captured for diagnostics

    # Exact-mode nominal parameter vectors for the stacked Faddeeva call.
    if exact_jacobian:
        nom_c = np.array([float(dc.nominal_centers[0]) for dc in dicts],
                         dtype=np.float32)
        nom_s = np.array([float(dc.nominal_sigmas[0]) for dc in dicts],
                         dtype=np.float32)
        nom_g = np.array([float(dc.gamma) for dc in dicts], dtype=np.float32)
        so_partner = [k for k, dc in enumerate(dicts) if dc.so_split != 0.0]
        so_split_v = [float(dicts[k].so_split) for k in so_partner]
        so_r = [1.0 / float(dicts[k].branch_ratio) for k in so_partner]
        gamma_uniform = bool(np.all(nom_g == nom_g[0]))

    # Exact-mode basis/Jacobian state at the CURRENT cumulative params.
    # The post-step re-evaluation and the next iteration's loop-top basis are
    # the SAME parameter point, so the basis is evaluated once per step and
    # carried over — re-evaluating at the loop top would double the dominant
    # Faddeeva-table cost (n_iter+1 evals per peak instead of 2·n_iter). The
    # Jacobians stay lazy until the H/g reductions pull them in, so the final
    # step's Jacobians are never computed.
    def _eval_exact_all():
        """Exact Φ/J for ALL peaks in one stacked Faddeeva call.

        Peaks are concatenated along the batch axis ((n_comp·batch,) inputs)
        so the dominant table-interpolation kernels launch once per step
        instead of once per peak; SO-doublet partners (peak k shifted by
        so_split, scaled by 1/branch_ratio) go in a second stacked call.
        Per-spectrum absolute params: nominal + cumulative (δE, δσ[, δγ]).
        Returns per-peak lists of (batch, n_E) lazy MLX arrays.
        """
        c_cat = np.concatenate([nom_c[k] + dE_cum[:, k] for k in range(n_comp)])
        s_cat = np.concatenate([nom_s[k] + ds_cum[:, k] for k in range(n_comp)])
        if fit_gamma:
            g_arg = mx.array(np.concatenate(
                [nom_g[k] + dg_cum[:, k] for k in range(n_comp)]))
        elif gamma_uniform:
            g_arg = float(nom_g[0])
        else:
            g_arg = mx.array(np.repeat(nom_g, batch))
        Phi, Jc, Js, Jg = voigt_exact_jacobian_perspectrum_mlx(
            energy_mx, mx.array(c_cat), mx.array(s_cat), g_arg,
        )
        Phi_l = [Phi[k * batch:(k + 1) * batch] for k in range(n_comp)]
        Jc_l = [Jc[k * batch:(k + 1) * batch] for k in range(n_comp)]
        Js_l = [Js[k * batch:(k + 1) * batch] for k in range(n_comp)]
        Jg_l = [Jg[k * batch:(k + 1) * batch] for k in range(n_comp)]
        if so_partner:
            cp_cat = np.concatenate(
                [c_cat[k * batch:(k + 1) * batch] + np.float32(so_split_v[j])
                 for j, k in enumerate(so_partner)])
            sp_cat = np.concatenate(
                [s_cat[k * batch:(k + 1) * batch] for k in so_partner])
            if fit_gamma:
                gp_arg = mx.array(np.concatenate(
                    [nom_g[k] + dg_cum[:, k] for k in so_partner]))
            elif gamma_uniform:
                gp_arg = float(nom_g[0])
            else:
                gp_arg = mx.array(np.repeat(nom_g[so_partner], batch))
            Phi_p, Jc_p, Js_p, Jg_p = voigt_exact_jacobian_perspectrum_mlx(
                energy_mx, mx.array(cp_cat), mx.array(sp_cat), gp_arg,
            )
            for j, k in enumerate(so_partner):
                sl = slice(j * batch, (j + 1) * batch)
                r = so_r[j]
                Phi_l[k] = Phi_l[k] + r * Phi_p[sl]
                Jc_l[k] = Jc_l[k] + r * Jc_p[sl]
                Js_l[k] = Js_l[k] + r * Js_p[sl]
                Jg_l[k] = Jg_l[k] + r * Jg_p[sl]
        return Phi_l, Jc_l, Js_l, Jg_l

    exact_basis = _eval_exact_all() if exact_jacobian else None

    # ------------------------------------------------------------------
    # Newton iteration body
    # ------------------------------------------------------------------
    for iter_idx in range(n_iter):
        is_last = (iter_idx == n_iter - 1)

        # Per-iteration basis Φ̂ and Jacobians J_c, J_σ.
        if exact_jacobian:
            # Exact basis + analytic Jacobians at the current absolute
            # parameters (carried over from the previous post-step eval).
            Phi_hat_mx, Jc_iter, Js_iter, Jg_iter = exact_basis
        else:
            # Frozen first-order Taylor surrogate from the grid cell.
            dE_off_mx_list = [
                mx.array(dE_cum[:, k]) - dE_grid_val_mx[k] for k in range(n_comp)
            ]
            ds_off_mx_list = [
                mx.array(ds_cum[:, k]) - ds_grid_val_mx[k] for k in range(n_comp)
            ]
            Phi_hat_mx = []
            for k in range(n_comp):
                Phi_hat_mx.append(
                    Phi_g_mx[k]
                    + dE_off_mx_list[k][:, None] * Jc_per_mx[k]
                    + ds_off_mx_list[k][:, None] * Js_per_mx[k]
                )
            Jc_iter = Jc_per_mx
            Js_iter = Js_per_mx

        if use_bg:
            # Re-solve amp + background jointly on the current surrogate basis so
            # the residual driving the gradient has the background removed.
            amp_mx, bg_mx, recon_mx = _solve_amp_bg(Phi_hat_mx)
            R_mx = Y_mx - recon_mx
        else:
            amp_mx = mx.array(amp_cum)
            Y_pred_mx = amp_mx[:, 0:1] * Phi_hat_mx[0]
            for k in range(1, n_comp):
                Y_pred_mx = Y_pred_mx + amp_mx[:, k:k + 1] * Phi_hat_mx[k]
            R_mx = Y_mx - Y_pred_mx

        # Per-spectrum residual variance — the prior on δγ is set relative to
        # this noise level so that ρ = R_var/σ_prior² blends the data and prior
        # in noise-normalized units (see the solve block below). Kept lazy:
        # the GPU Cholesky path folds it into its graph, only the dense CPU
        # fallback evaluates it.
        R_var_mx = None
        if fit_gamma and gamma_prior_std is not None and gamma_prior_std > 0:
            R_var_mx = mx.mean(R_mx * R_mx, axis=1)

        if fit_gamma:
            def jac(p, amp=amp_mx, Jc=Jc_iter, Js=Js_iter, Jg=Jg_iter):
                k, axis = divmod(p, 3)
                col = Jc[k] if axis == 0 else (Js[k] if axis == 1 else Jg[k])
                return amp[:, k:k + 1] * col
        else:
            def jac(p, amp=amp_mx, Jc=Jc_iter, Js=Js_iter):
                k, axis = divmod(p, 2)
                return amp[:, k:k + 1] * (Jc[k] if axis == 0 else Js[k])

        # Gradient + Hessian. The grid-cell cache is only valid for the
        # FROZEN surrogate Jacobians at n_comp ≤ 2 (the B cross-talk tables
        # are 2-comp only). For larger systems the O(p²) per-entry reduction
        # kernels are replaced by a single stacked JᵀJ / JᵀR batched matmul.
        H_entries: dict[tuple[int, int], object] = {}
        use_cache = (not exact_jacobian and not fit_gamma
                     and hessian_cache is not None
                     and hessian_cache.n_comp == n_comp
                     and n_comp <= 2)
        if n_params > 4 and not use_cache:
            Jt = mx.stack([jac(i) for i in range(n_params)], axis=1)  # (b,p,nE)
            Hm = Jt @ mx.transpose(Jt, (0, 2, 1))                     # (b,p,p)
            gm = (Jt @ R_mx[:, :, None])[:, :, 0]                     # (b,p)
            g_entries = [gm[:, i] for i in range(n_params)]
            for i in range(n_params):
                for j in range(i, n_params):
                    H_entries[(i, j)] = Hm[:, i, j]
        else:
            g_entries = [mx.sum(jac(i) * R_mx, axis=1) for i in range(n_params)]
            if use_cache:
                a_kk = [amp_mx[:, k] * amp_mx[:, k] for k in range(n_comp)]
                a_kl = (amp_mx[:, 0] * amp_mx[:, 1]) if n_comp == 2 else None
                for k in range(n_comp):
                    base = 2 * k
                    idx_k = best_mx[k]
                    H_entries[(base, base)] = a_kk[k] * hessian_cache.A_cc[k][idx_k]
                    H_entries[(base, base + 1)] = a_kk[k] * hessian_cache.A_cs[k][idx_k]
                    H_entries[(base + 1, base + 1)] = a_kk[k] * hessian_cache.A_ss[k][idx_k]
                if n_comp == 2 and hessian_cache.B_cc is not None:
                    idx_1 = best_mx[0]
                    idx_2 = best_mx[1]
                    H_entries[(0, 2)] = a_kl * hessian_cache.B_cc[idx_1, idx_2]
                    H_entries[(0, 3)] = a_kl * hessian_cache.B_cs[idx_1, idx_2]
                    H_entries[(1, 2)] = a_kl * hessian_cache.B_sc[idx_1, idx_2]
                    H_entries[(1, 3)] = a_kl * hessian_cache.B_ss[idx_1, idx_2]
            else:
                for i in range(n_params):
                    Si = jac(i)
                    H_entries[(i, i)] = mx.sum(Si * Si, axis=1)
                    for j in range(i + 1, n_params):
                        Sj = jac(j)
                        H_entries[(i, j)] = mx.sum(Si * Sj, axis=1)

        # Solve the small system. Dispatch: hand-rolled Schur closed form for
        # the 2/4-param cases, unrolled GPU Cholesky for larger systems
        # (including fit_gamma) up to n_params ≤ _SPD_GPU_MAX_M, dense CPU
        # LAPACK beyond. When diagnostics are requested, capture the RAW
        # dense Hessian (no Tikhonov) on the last iteration for the
        # covariance estimate.
        capture_H = compute_diagnostics and is_last
        schur_path = n_comp <= 2 and not fit_gamma
        chol_path = not schur_path and n_params <= _SPD_GPU_MAX_M
        dense_path = not schur_path and not chol_path
        H_raw = None
        if dense_path or capture_H:
            flat = list(H_entries.values()) + g_entries
            mx.eval(*flat)
            H_raw = np.empty((batch, n_params, n_params), dtype=np.float32)
            for (i, j), v in H_entries.items():
                arr = np.asarray(v).astype(np.float32, copy=False)
                H_raw[:, i, j] = arr
                if i != j:
                    H_raw[:, j, i] = arr

        if schur_path:
            delta = _solve_small_system_gpu(H_entries, g_entries, n_comp, batch)
        elif chol_path:
            trace_mx = H_entries[(0, 0)]
            for d in range(1, n_params):
                trace_mx = trace_mx + H_entries[(d, d)]
            reg_mx = 1e-6 * trace_mx
            Hd = dict(H_entries)
            for d in range(n_params):
                Hd[(d, d)] = Hd[(d, d)] + reg_mx
            g_list = list(g_entries)
            if fit_gamma:
                # Same two γ-row terms as the dense CPU path below (Levenberg
                # conditioning load + Gaussian ridge prior); see that comment
                # block for the derivation.
                mean_diag_mx = trace_mx * (1.0 / n_params)
                rho_mx = (R_var_mx * (1.0 / gamma_prior_std ** 2)
                          if R_var_mx is not None else None)
                for k in range(n_comp):
                    gi = 3 * k + 2
                    Hd[(gi, gi)] = Hd[(gi, gi)] + lambda_gamma * mean_diag_mx
                    if rho_mx is not None:
                        Hd[(gi, gi)] = Hd[(gi, gi)] + rho_mx
                        g_list[gi] = g_list[gi] - rho_mx * mx.array(dg_cum[:, k])
            delta_mx = mx.stack(
                _spd_cholesky_solve_entries(Hd, g_list, n_params), axis=1,
            )
            mx.eval(delta_mx)
            delta = np.asarray(delta_mx).astype(np.float32)
        else:
            g = np.stack([np.asarray(v).astype(np.float32, copy=False)
                          for v in g_entries], axis=1)
            H = H_raw.copy()
            trace = np.trace(H, axis1=1, axis2=2)
            diag_idx = np.arange(n_params)
            H[:, diag_idx, diag_idx] += (1e-6 * trace)[:, None]
            if fit_gamma:
                g_idx = np.arange(2, n_params, 3)
                # γ is Fisher-ill-conditioned vs σ. Two terms on the γ rows:
                #  (1) a Levenberg conditioning load (∝ mean diagonal) so the
                #      δσ/δγ degeneracy cannot blow up the step direction;
                #  (2) a Gaussian ridge PRIOR pulling the cumulative δγ toward
                #      zero. Minimizing ½‖R−JΔ‖² + ½ρ‖dg_cum+Δγ‖² adds ρ to the
                #      γ-Hessian diagonal and −ρ·dg_cum to the γ-gradient. ρ is
                #      set in noise-normalized units (ρ = R_var/σ_prior², below)
                #      so it is amplitude- and point-count-invariant. It
                #      suppresses the noise-driven spurious δγ — anti-correlated
                #      ≈ −1 with the induced δσ error — when γ is actually
                #      correct, while still admitting real shifts of order
                #      σ_prior. Without it the chi²-flat σ↔γ valley lets noise
                #      leak into δσ (the nominal-γ case degrades by ~30 dB).
                mean_diag = (trace / n_params).astype(np.float32)
                H[:, g_idx, g_idx] += (lambda_gamma * mean_diag)[:, None]
                if R_var_mx is not None:
                    # Gaussian ridge prior N(0, σ_prior²) on the cumulative δγ.
                    # The data γ-Hessian H_ee (= Σ(a·Jg)²) is in residual² units;
                    # the un-normalized parameter posterior variance is
                    # R_var/H_ee. To blend with a σ_prior prior the load on the
                    # γ-diagonal must be ρ = R_var/σ_prior² (same units as H_ee),
                    # and the γ-gradient picks up −ρ·dg_cum. This suppresses the
                    # noise-driven spurious δγ (posterior std ≈ sqrt(R_var/H_ee))
                    # when γ is correct while still admitting real shifts ~σ_prior.
                    mx.eval(R_var_mx)
                    R_var_np = np.asarray(R_var_mx).astype(np.float32)
                    rho = (R_var_np / (gamma_prior_std ** 2))[:, None]  # (batch,1)
                    H[:, g_idx, g_idx] += rho
                    # gradient g is +Jᵀr (the step solves H·Δ = g), so the prior
                    # subtracts ρ·dg_cum from the γ gradient entries.
                    g[:, g_idx] -= rho * dg_cum
            delta = np.linalg.solve(H, g[:, :, None])[:, :, 0]

        if capture_H:
            H_diag_dense = H_raw

        for k in range(n_comp):
            dE_max = clip_step_ratio * dE_step_per[k]
            ds_max = clip_step_ratio * ds_step_per[k]
            np.clip(delta[:, pp * k], -dE_max, dE_max, out=delta[:, pp * k])
            np.clip(delta[:, pp * k + 1], -ds_max, ds_max, out=delta[:, pp * k + 1])
            if fit_gamma:
                # Clip δγ on the same scale as δσ (γ and σ share an eV scale;
                # ds_step_per is a reasonable proxy for the γ step magnitude).
                dg_max = clip_step_ratio * ds_step_per[k]
                np.clip(delta[:, pp * k + 2], -dg_max, dg_max,
                        out=delta[:, pp * k + 2])

        # Update cumulative state
        dE_cum = dE_cum + delta[:, 0::pp]
        ds_cum = ds_cum + delta[:, 1::pp]
        if fit_gamma:
            dg_cum = dg_cum + delta[:, 2::pp]

        # Newton-updated basis for the amplitude resolve. Surrogate: take the
        # extra Taylor step Φ̂ + Δ·J. Exact: re-evaluate the exact basis at the
        # post-step (center, sigma) so the amplitude LLS and chi² are consistent
        # with the refined parameters (no leftover linearization error). The
        # evaluation doubles as the next iteration's loop-top basis.
        if exact_jacobian:
            exact_basis = _eval_exact_all()
            Phi_new_mx = exact_basis[0]
        else:
            delta_mx = mx.array(delta.astype(np.float32))
            Phi_new_mx = []
            for k in range(n_comp):
                d_dE = delta_mx[:, 2 * k:2 * k + 1]
                d_ds = delta_mx[:, 2 * k + 1:2 * k + 2]
                Phi_new_mx.append(Phi_hat_mx[k]
                                  + d_dE * Jc_per_mx[k]
                                  + d_ds * Js_per_mx[k])

        # Joint LLS for amplitudes (Cramer for n_comp ≤ 2, else cpu solve).
        # When a background basis is present, solve [peaks | BG] jointly so the
        # amplitudes and chi² stay background-corrected.
        if use_bg:
            amp_solve_mx, bg_solve_mx, recon = _solve_amp_bg(Phi_new_mx)
            if is_last:
                chi2 = mx.mean((Y_mx - recon) ** 2, axis=1)
                mx.eval(amp_solve_mx, bg_solve_mx, chi2)
                amp_cum = np.asarray(amp_solve_mx).astype(np.float32)
                bg_cum = np.asarray(bg_solve_mx).astype(np.float32)
                chi2_np = np.asarray(chi2).astype(np.float32)
            else:
                mx.eval(amp_solve_mx, bg_solve_mx)
                amp_cum = np.asarray(amp_solve_mx).astype(np.float32)
                bg_cum = np.asarray(bg_solve_mx).astype(np.float32)
        elif n_comp == 1:
            p1 = Phi_new_mx[0]
            d_sq = mx.sum(p1 * p1, axis=1) + 1e-10
            a0 = mx.sum(p1 * Y_mx, axis=1) / d_sq
            if is_last:
                recon = a0[:, None] * p1
                chi2 = mx.mean((Y_mx - recon) ** 2, axis=1)
                mx.eval(a0, chi2)
                amp_cum = np.asarray(a0).astype(np.float32)[:, None]
                chi2_np = np.asarray(chi2).astype(np.float32)
            else:
                mx.eval(a0)
                amp_cum = np.asarray(a0).astype(np.float32)[:, None]
        elif n_comp == 2:
            p1, p2 = Phi_new_mx
            g11 = mx.sum(p1 * p1, axis=1)
            g12 = mx.sum(p1 * p2, axis=1)
            g22 = mx.sum(p2 * p2, axis=1)
            r1 = mx.sum(p1 * Y_mx, axis=1)
            r2 = mx.sum(p2 * Y_mx, axis=1)
            det = g11 * g22 - g12 * g12
            safe_det = mx.where(mx.abs(det) > 1e-30, det, mx.array(1e-30))
            a0 = (g22 * r1 - g12 * r2) / safe_det
            a1 = (g11 * r2 - g12 * r1) / safe_det
            if is_last:
                recon = a0[:, None] * p1 + a1[:, None] * p2
                chi2 = mx.mean((Y_mx - recon) ** 2, axis=1)
                mx.eval(a0, a1, chi2)
                amp_cum = np.stack([np.asarray(a0), np.asarray(a1)],
                                    axis=1).astype(np.float32)
                chi2_np = np.asarray(chi2).astype(np.float32)
            else:
                mx.eval(a0, a1)
                amp_cum = np.stack([np.asarray(a0), np.asarray(a1)],
                                    axis=1).astype(np.float32)
        else:
            Phi_all = mx.stack(Phi_new_mx, axis=2)
            Phi_T = mx.transpose(Phi_all, (0, 2, 1))
            G_mx = Phi_T @ Phi_all + 1e-10 * mx.eye(n_comp)
            rhs_mx = (Phi_T @ Y_mx[:, :, None])[:, :, 0]
            amp_mx_solve = _gram_solve_mx(G_mx, rhs_mx, n_comp)
            if is_last:
                recon = (Phi_all @ amp_mx_solve[:, :, None])[:, :, 0]
                chi2 = mx.mean((Y_mx - recon) ** 2, axis=1)
                mx.eval(amp_mx_solve, chi2)
                amp_cum = np.asarray(amp_mx_solve).astype(np.float32)
                chi2_np = np.asarray(chi2).astype(np.float32)
            else:
                mx.eval(amp_mx_solve)
                amp_cum = np.asarray(amp_mx_solve).astype(np.float32)

    diag = None
    if compute_diagnostics and H_diag_dense is not None:
        diag = _newton_diagnostics(H_diag_dense, chi2_np)

    dg_out = dg_cum.astype(np.float32) if fit_gamma else None
    return (amp_cum, dE_cum.astype(np.float32),
            ds_cum.astype(np.float32), chi2_np, diag, bg_cum, dg_out)


def _post_ap_newton_refine_chunk(
    Y: np.ndarray,
    dicts: list[DictionaryCache],
    best_idx: np.ndarray,
    delta_E: np.ndarray,
    delta_sigma: np.ndarray,
    amplitudes: np.ndarray,
    clip_step_ratio: float,
    jacobian_mode: str = "raw",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single-chunk Newton step — float32, no (batch, n_E, n_params) intermediates.

    The Hessian/gradient are assembled entry-by-entry as scalar reductions
    over the energy axis, never materialising the 4-column Jacobian tensor.
    For 2 peaks this means 10 H entries + 4 g entries, each a single
    np.sum(A * B, axis=1). Memory peak ≈ batch · n_E · float32 per array.

    jacobian_mode (EXPERIMENTAL, default "raw" = existing behavior):
        "raw":           model-side columns S_p = a_k·J (amplitude variation
                         ignored) — the production step, byte-identical.
                         NOTE: "raw" names the JACOBIAN approximation (the
                         a*(θ) dependence is ignored), not the un-regularized
                         Hessian that the MLX diagnostics path captures as
                         ``H_raw`` (= H before the Tikhonov load) — the two
                         "raw"s are unrelated. Since Φ̂ᵀR = 0 at the LLS
                         optimum, all three modes share the same gradient
                         (columns)ᵀR; they differ ONLY in the Gauss-Newton
                         curvature H.
        "kaufman":       variable-projection columns (I−P_Φ)S_p, where P_Φ is
                         the projector onto the span of the current surrogate
                         bases Φ̂. Amplitudes are RE-SOLVED on Φ̂ first so that
                         Φ̂ᵀR = 0 (the VarPro identities require a = a*(θ);
                         the incoming ``amplitudes`` are ignored).
        "golub_pereyra": full reduced-residual Jacobian, adding the second GP
                         term (Φ̂⁺)ᵀ(∂_pΦ̂)ᵀR to the Kaufman columns.
    All modes share the sign system H·Δ = g with g = (columns)ᵀR and are
    evaluated within the frozen grid-cell Taylor surrogate (Φ̂ linear in θ,
    J frozen), so kaufman/golub_pereyra here verify the projection term only
    — NOT the exact-basis curvature (see gp_reference for the exact-model
    float64 comparison). The extra cost is one batched (n×n) Gram solve with
    2·n_params (+n_params for GP) right-hand sides plus 2 (GP: 3) batched
    (n_E×n) matmuls per column.
    """
    batch, n_E = Y.shape
    n_comp = len(dicts)
    n_params = 2 * n_comp

    # Gather per-peak Phi/Jc/Js at the AP-chosen grid cell, plus the Taylor
    # offsets in (δE, δσ). Float32 to halve memory vs float64.
    Phi_hat = np.empty((batch, n_E, n_comp), dtype=np.float32)
    Jc_per = np.empty((batch, n_E, n_comp), dtype=np.float32)
    Js_per = np.empty((batch, n_E, n_comp), dtype=np.float32)

    dE_step_per = np.empty(n_comp, dtype=np.float32)
    ds_step_per = np.empty(n_comp, dtype=np.float32)

    for k in range(n_comp):
        dc = dicts[k]
        n_ds_k = dc.grid_shape[1]
        idx_k = best_idx[:, k]
        dE_grid_idx = idx_k // n_ds_k
        ds_grid_idx = idx_k % n_ds_k
        dE_grid_val = dc.dE_grid[dE_grid_idx].astype(np.float32)
        ds_grid_val = dc.dsigma_grid[ds_grid_idx].astype(np.float32)

        dE_offset = delta_E[:, k] - dE_grid_val  # float32
        ds_offset = delta_sigma[:, k] - ds_grid_val

        Phi_g = dc.Phi_per_grid[idx_k, :, 0]  # already float32
        Jc_g = dc.Jc_per_grid[idx_k, :, 0]
        Js_g = dc.Jsigma_per_grid[idx_k, :, 0]

        np.add(Phi_g, dE_offset[:, None] * Jc_g, out=Phi_hat[:, :, k])
        Phi_hat[:, :, k] += ds_offset[:, None] * Js_g
        Jc_per[:, :, k] = Jc_g
        Js_per[:, :, k] = Js_g

        dE_step_per[k] = (float(dc.dE_grid[1] - dc.dE_grid[0])
                          if len(dc.dE_grid) > 1 else 1.0)
        ds_step_per[k] = (float(dc.dsigma_grid[1] - dc.dsigma_grid[0])
                          if len(dc.dsigma_grid) > 1 else 1.0)

    if jacobian_mode == "raw":
        # Predicted spectrum + residual (memory-tight: no broadcast tensor)
        amp32 = amplitudes.astype(np.float32, copy=False)
        Y_pred = amp32[:, 0:1] * Phi_hat[:, :, 0]
        for k in range(1, n_comp):
            Y_pred = Y_pred + amp32[:, k:k + 1] * Phi_hat[:, :, k]
        R = Y - Y_pred  # (batch, n_E) float32

        # Jacobian columns evaluated lazily: S_(2k) = a_k * Jc_k,
        # S_(2k+1) = a_k * Js_k. We assemble H[i,j] = sum_e S_i * S_j and
        # g[i] = sum_e S_i * R entry by entry, avoiding the materialisation
        # of the (batch, n_E, 4) S tensor.
        def jac_col(p: int):
            k, axis = divmod(p, 2)
            a = amp32[:, k:k + 1]
            J = Jc_per[:, :, k] if axis == 0 else Js_per[:, :, k]
            return a * J
    elif jacobian_mode in ("kaufman", "golub_pereyra"):
        # VarPro identities need a = a*(θ) on the CURRENT surrogate basis:
        # re-solve the joint LLS so Φ̂ᵀR = 0, then build the projected
        # columns without ever forming the (n_E × n_E) projector —
        # (I−P)S = S − Φ̂ G⁻¹(Φ̂ᵀS), (Φ̂⁺)ᵀq = Φ̂ G⁻¹q.
        PhT = Phi_hat.transpose(0, 2, 1)      # (batch, n_comp, n_E) view
        G = PhT @ Phi_hat                     # (batch, n_comp, n_comp)
        G[:, np.arange(n_comp), np.arange(n_comp)] += np.float32(1e-10)
        rhs_a = (PhT @ Y[:, :, None])[:, :, 0]
        amp32 = np.linalg.solve(
            G, rhs_a[:, :, None].astype(G.dtype),
        )[:, :, 0].astype(np.float32)
        R = Y - (Phi_hat @ amp32[:, :, None])[:, :, 0]

        S_cols = []
        for p in range(n_params):
            k, axis = divmod(p, 2)
            J = Jc_per[:, :, k] if axis == 0 else Js_per[:, :, k]
            S_cols.append(amp32[:, k:k + 1] * J)

        # One batched Gram solve for all projection coefficients:
        # columns [Φ̂ᵀS_p | q_p], q_p = e_k · Σ_e(J_p · R).
        U = np.stack(
            [(PhT @ S[:, :, None])[:, :, 0] for S in S_cols], axis=2,
        )
        if jacobian_mode == "golub_pereyra":
            Qm = np.zeros((batch, n_comp, n_params), dtype=U.dtype)
            for p in range(n_params):
                k, axis = divmod(p, 2)
                J = Jc_per[:, :, k] if axis == 0 else Js_per[:, :, k]
                Qm[:, k, p] = np.sum(J * R, axis=1)
            coef = np.linalg.solve(G, np.concatenate([U, Qm], axis=2))
        else:
            coef = np.linalg.solve(G, U)

        M_cols = []
        for p in range(n_params):
            col = S_cols[p] - (Phi_hat @ coef[:, :, p:p + 1])[:, :, 0]
            if jacobian_mode == "golub_pereyra":
                col = col + (
                    Phi_hat @ coef[:, :, n_params + p:n_params + p + 1]
                )[:, :, 0]
            M_cols.append(col.astype(np.float32))
        del S_cols, U, coef

        def jac_col(p: int, _M=M_cols):
            return _M[p]
    else:
        raise ValueError(
            f"jacobian_mode must be 'raw', 'kaufman' or 'golub_pereyra', "
            f"got {jacobian_mode!r}"
        )

    H = np.empty((batch, n_params, n_params), dtype=np.float32)
    g = np.empty((batch, n_params), dtype=np.float32)
    for i in range(n_params):
        Si = jac_col(i)
        g[:, i] = np.sum(Si * R, axis=1)
        H[:, i, i] = np.sum(Si * Si, axis=1)
        for j in range(i + 1, n_params):
            Sj = jac_col(j)
            v = np.sum(Si * Sj, axis=1)
            H[:, i, j] = v
            H[:, j, i] = v

    # Tikhonov regularisation in float32
    trace = np.trace(H, axis1=1, axis2=2)
    diag_reg = (1e-6 * trace).astype(np.float32)
    H[:, np.arange(n_params), np.arange(n_params)] += diag_reg[:, None]

    delta = np.linalg.solve(H, g[:, :, None])[:, :, 0]  # (batch, n_params)

    # Clip per (δE, δσ) using each peak's grid spacing
    for k in range(n_comp):
        dE_max = clip_step_ratio * dE_step_per[k]
        ds_max = clip_step_ratio * ds_step_per[k]
        np.clip(delta[:, 2 * k], -dE_max, dE_max, out=delta[:, 2 * k])
        np.clip(delta[:, 2 * k + 1], -ds_max, ds_max, out=delta[:, 2 * k + 1])

    delta_E_new = delta_E.copy()
    delta_sigma_new = delta_sigma.copy()
    for k in range(n_comp):
        delta_E_new[:, k] = delta_E[:, k] + delta[:, 2 * k]
        delta_sigma_new[:, k] = delta_sigma[:, k] + delta[:, 2 * k + 1]

    # Newton-updated basis (in-place to avoid duplicating Phi_hat)
    for k in range(n_comp):
        Phi_hat[:, :, k] += (delta[:, 2 * k:2 * k + 1] * Jc_per[:, :, k]
                              + delta[:, 2 * k + 1:2 * k + 2] * Js_per[:, :, k])

    # Joint LLS for amplitudes on the updated basis (small n_comp×n_comp system)
    G_amp = np.empty((batch, n_comp, n_comp), dtype=np.float32)
    rhs_amp = np.empty((batch, n_comp), dtype=np.float32)
    for k in range(n_comp):
        rhs_amp[:, k] = np.sum(Phi_hat[:, :, k] * Y, axis=1)
        for l in range(k, n_comp):
            v = np.sum(Phi_hat[:, :, k] * Phi_hat[:, :, l], axis=1)
            G_amp[:, k, l] = v
            G_amp[:, l, k] = v
    G_amp[:, np.arange(n_comp), np.arange(n_comp)] += np.float32(1e-10)
    a_new = np.linalg.solve(G_amp, rhs_amp[:, :, None])[:, :, 0]

    # Chi² at the updated model
    Y_pred_new = a_new[:, 0:1] * Phi_hat[:, :, 0]
    for k in range(1, n_comp):
        Y_pred_new = Y_pred_new + a_new[:, k:k + 1] * Phi_hat[:, :, k]
    R_new = Y - Y_pred_new
    chi2_new = np.mean(R_new * R_new, axis=1).astype(np.float32)

    return (
        a_new.astype(np.float32),
        delta_E_new.astype(np.float32),
        delta_sigma_new.astype(np.float32),
        chi2_new,
    )


def _post_ap_newton_refine_numpy(
    Y: np.ndarray,
    dicts: list[DictionaryCache],
    best_idx: np.ndarray,
    delta_E: np.ndarray,
    delta_sigma: np.ndarray,
    amplitudes: np.ndarray,
    clip_step_ratio: float = 0.5,
    chunk_size: int = 200_000,
    jacobian_mode: str = "raw",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One Gauss-Newton step refining (δE_k, δσ_k) jointly across all peaks.

    Closes the AP argmax bias diagnosed in dev-log 134: dict grid refine and
    per-axis parabola each give ≤1 dB on multipeak, while encoder bypass
    showed AP itself loses ~3.5 dB to systematic argmax bias from inter-peak
    cross-talk. Joint Newton on the residual lets all (δE_k, δσ_k) move
    together off the grid.

    For each spectrum, the basis at the current (δE_k, δσ_k) is reconstructed
    by Taylor extrapolation from the grid pick:

        Φ_k_hat = Φ_g_k + (δE_k - δE_grid) · J_c_k + (δσ_k - δσ_grid) · J_σ_k

    The residual is

        R = Y - Σ_k a_k · Φ_k_hat

    and the 2n_comp-parameter Jacobian columns are

        S_(2k)     = a_k · J_c_k
        S_(2k + 1) = a_k · J_σ_k

    A regularised normal-equation solve gives the joint step Δ; each pair is
    clipped to ±clip_step_ratio of the corresponding grid step for stability.
    After the update, amplitudes are re-solved by joint LLS at the new bases
    and chi² is recomputed.

    Args:
        Y: (batch, n_E) original spectra (float32).
        dicts: per-peak DictionaryCache (each n_comp_internal=1).
        best_idx: (batch, n_comp) grid pick per peak.
        delta_E, delta_sigma: (batch, n_comp) current refined offsets.
        amplitudes: (batch, n_comp) current amplitudes (joint LLS output).
        clip_step_ratio: maximum |Δ| as fraction of grid step (default 0.5).

    Args (additional):
        chunk_size: Process at most this many spectra at once. Caps peak
            memory at ≈ chunk_size · n_E · float32 · 5 arrays (Phi_hat,
            Jc, Js, Y_pred, R). 200K × 125 × 4 B × 5 ≈ 500 MB.
        jacobian_mode: EXPERIMENTAL — "raw" (default, existing behavior),
            "kaufman" or "golub_pereyra". See
            :func:`_post_ap_newton_refine_chunk`. Non-raw modes materialise
            2·n_comp extra (chunk, n_E) column arrays; reduce chunk_size
            accordingly for large n_comp.

    Returns:
        amplitudes_new, delta_E_new, delta_sigma_new, chi2_new — all float32.
    """
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    batch = Y.shape[0]
    n_comp = len(dicts)

    amp_out = np.empty((batch, n_comp), dtype=np.float32)
    dE_out = np.empty((batch, n_comp), dtype=np.float32)
    ds_out = np.empty((batch, n_comp), dtype=np.float32)
    chi2_out = np.empty(batch, dtype=np.float32)

    for s in range(0, batch, chunk_size):
        e = min(s + chunk_size, batch)
        amp_c, dE_c, ds_c, chi2_c = _post_ap_newton_refine_chunk(
            Y[s:e],
            dicts,
            best_idx[s:e],
            delta_E[s:e],
            delta_sigma[s:e],
            amplitudes[s:e],
            clip_step_ratio,
            jacobian_mode=jacobian_mode,
        )
        amp_out[s:e] = amp_c
        dE_out[s:e] = dE_c
        ds_out[s:e] = ds_c
        chi2_out[s:e] = chi2_c

    return amp_out, dE_out, ds_out, chi2_out


# ---------------------------------------------------------------------------
# GP-LM refine: compact
# Golub-Pereyra Hessian + scaled-diagonal Levenberg-Marquardt with
# accept/reject on the frozen grid-cell surrogate. CPU/numpy layer.
# ---------------------------------------------------------------------------


def _gp_lm_chunk(
    Y: np.ndarray,
    dicts: list[DictionaryCache],
    best_idx: np.ndarray,
    delta_E: np.ndarray,
    delta_sigma: np.ndarray,
    amplitudes: np.ndarray,
    chi2_in: np.ndarray,
    config: GPLMConfig,
    bg_design: np.ndarray | None = None,
    n_iter: int = 1,
    debug_out: dict | None = None,
):
    """Single-chunk GP-LM refine on the frozen surrogate.

    Compact system per spectrum (no projected columns, no projector):
        A = [Φ̂ | B],  G = AᵀA,  C = AᵀS,  K = SᵀS,  Q sparse (peak rows),
        H = K − CᵀG⁻¹C + QᵀG⁻¹Q,  g = SᵀR
    then bounded scaled-diagonal LM with a FULL trial re-evaluation
    (basis → joint LLS → residual) and per-spectrum accept/reject.
    Rejected spectra keep their incoming (amplitudes, δE, δσ, chi2) seed —
    the AP/parabola state — untouched. Trials are computed full-batch and
    committed under masks (a deliberate simplicity trade over
    rejected-subset recompute).

    Returns (amp, dE, ds, chi2, bg, diag) — bg is None without bg_design.
    """
    batch, n_E = Y.shape
    n_comp = len(dicts)
    n_params = 2 * n_comp
    n_bg = 0 if bg_design is None else bg_design.shape[1]
    n_lin = n_comp + n_bg

    # Non-finite input spectra would poison the BATCHED LAPACK calls
    # (np.linalg.solve raises for the whole batch, eigvalsh may not
    # converge). Zero them for the computation, never accept a step on
    # them, and report fallback_reason=3 (nonfinite); their outputs stay
    # the untouched seed.
    bad_input = ~np.isfinite(Y).all(axis=1)
    if bad_input.any():
        Y = Y.copy()
        Y[bad_input] = 0.0

    # ---- gather frozen grid-cell bases/Jacobians ----
    Phi_g = np.empty((batch, n_E, n_comp), dtype=np.float32)
    Jc = np.empty_like(Phi_g)
    Js = np.empty_like(Phi_g)
    dE_grid_val = np.empty((batch, n_comp), dtype=np.float32)
    ds_grid_val = np.empty((batch, n_comp), dtype=np.float32)
    clip_box = np.empty(n_params, dtype=np.float64)
    for k, dc in enumerate(dicts):
        n_ds_k = dc.grid_shape[1]
        idx = best_idx[:, k]
        dE_grid_val[:, k] = dc.dE_grid[idx // n_ds_k]
        ds_grid_val[:, k] = dc.dsigma_grid[idx % n_ds_k]
        Phi_g[:, :, k] = dc.Phi_per_grid[idx, :, 0]
        Jc[:, :, k] = dc.Jc_per_grid[idx, :, 0]
        Js[:, :, k] = dc.Jsigma_per_grid[idx, :, 0]
        dE_step = (float(dc.dE_grid[1] - dc.dE_grid[0])
                   if len(dc.dE_grid) > 1 else 1.0)
        ds_step = (float(dc.dsigma_grid[1] - dc.dsigma_grid[0])
                   if len(dc.dsigma_grid) > 1 else 1.0)
        clip_box[2 * k] = config.clip_step_ratio * dE_step
        clip_box[2 * k + 1] = config.clip_step_ratio * ds_step

    B_b = None
    if bg_design is not None:
        B_b = np.broadcast_to(
            bg_design.astype(np.float32)[None], (batch, n_E, n_bg),
        )

    def basis_at(dE, ds):
        Phi_hat = (Phi_g
                   + (dE - dE_grid_val)[:, None, :] * Jc
                   + (ds - ds_grid_val)[:, None, :] * Js)
        if B_b is None:
            return Phi_hat
        return np.concatenate([Phi_hat, B_b], axis=2)

    lin_idx = np.arange(n_lin)

    def joint_lls(A):
        AT = A.transpose(0, 2, 1)
        G = AT @ A
        G[:, lin_idx, lin_idx] += np.float32(config.gram_jitter)
        x = np.linalg.solve(G, AT @ Y[:, :, None])[:, :, 0]
        R = Y - (A @ x[:, :, None])[:, :, 0]
        return G, x, R

    # ---- mutable state (seed preserved until an accept commits) ----
    dE_cur = delta_E.astype(np.float32).copy()
    ds_cur = delta_sigma.astype(np.float32).copy()
    amp_out = amplitudes.astype(np.float32).copy()
    chi2_out = chi2_in.astype(np.float32).copy()
    bg_out = np.zeros((batch, n_bg), dtype=np.float32) if n_bg else None
    lam = np.full(batch, config.lambda0, dtype=np.float64)
    accepted_ever = np.zeros(batch, dtype=bool)
    retry_count = np.zeros(batch, dtype=np.int32)
    step_clipped = np.zeros(batch, dtype=bool)
    fallback = np.zeros(batch, dtype=np.int8)
    last_pred = np.full(batch, np.nan, dtype=np.float64)
    last_ared = np.full(batch, np.nan, dtype=np.float64)
    last_rho = np.full(batch, np.nan, dtype=np.float64)
    cond_gram_out = np.zeros(batch, dtype=np.float32)
    cond_hess_out = np.zeros(batch, dtype=np.float32)
    n_basis_evals = 0

    for _ in range(max(1, n_iter)):
        A = basis_at(dE_cur, ds_cur)
        G, x, R = joint_lls(A)
        n_basis_evals += 1
        if bg_out is not None and not accepted_ever.any():
            bg_out[:] = x[:, n_comp:]  # seed-basis bg for never-accepted
        F0 = 0.5 * np.sum(
            R.astype(np.float64) * R.astype(np.float64), axis=1,
        )

        # ---- compact GP system ----
        S = np.empty((batch, n_E, n_params), dtype=np.float32)
        Q = np.zeros((batch, n_lin, n_params), dtype=np.float32)
        for p in range(n_params):
            k, axis = divmod(p, 2)
            Jcol = Jc[:, :, k] if axis == 0 else Js[:, :, k]
            S[:, :, p] = x[:, k:k + 1] * Jcol
            Q[:, k, p] = np.sum(Jcol * R, axis=1)
        AT = A.transpose(0, 2, 1)
        ST = S.transpose(0, 2, 1)
        C = AT @ S
        K = ST @ S
        g = (ST @ R[:, :, None])[:, :, 0]
        sol = np.linalg.solve(G, np.concatenate([C, Q], axis=2))
        H = (K - C.transpose(0, 2, 1) @ sol[:, :, :n_params]
             + Q.transpose(0, 2, 1) @ sol[:, :, n_params:])
        H = 0.5 * (H + H.transpose(0, 2, 1))

        H64 = H.astype(np.float64)
        g64 = g.astype(np.float64)
        if debug_out is not None and "H" not in debug_out:
            debug_out["H"] = H64.copy()
            debug_out["g"] = g64.copy()
            debug_out["x"] = x.copy()
            debug_out["R"] = R.copy()

        # ---- condition gate (cheap: n_lin, n_params are tiny) ----
        evG = np.linalg.eigvalsh(G.astype(np.float64))
        cond_gram = evG[:, -1] / np.maximum(evG[:, 0], 1e-300)
        evH = np.linalg.eigvalsh(H64)
        cond_hess = evH[:, -1] / np.maximum(evH[:, 0], 1e-300)
        cond_gram_out = np.minimum(cond_gram, 3e38).astype(np.float32)
        cond_hess_out = np.minimum(cond_hess, 3e38).astype(np.float32)
        gate = (np.isfinite(cond_gram) & (cond_gram < config.cond_gate)
                & ~bad_input)
        fallback[bad_input] = 3
        fallback[~gate & ~bad_input & ~accepted_ever] = 1

        d = lm_scale_diag(H64, config.diag_eps)
        pending = gate.copy()

        for _retry in range(config.max_retries + 1):
            if not pending.any():
                break
            delta = lm_step(H64, g64, lam, d)
            clipped = np.any(np.abs(delta) > clip_box, axis=1)
            delta = np.clip(delta, -clip_box, clip_box)
            pred = predicted_reduction(g64, H64, delta)

            dE_try = dE_cur + delta[:, 0::2].astype(np.float32)
            ds_try = ds_cur + delta[:, 1::2].astype(np.float32)
            A_try = basis_at(dE_try, ds_try)
            _, x_try, R_try = joint_lls(A_try)
            n_basis_evals += 1
            F1 = 0.5 * np.sum(
                R_try.astype(np.float64) * R_try.astype(np.float64), axis=1,
            )
            ared = F0 - F1
            rho = ared / np.where(pred > 0, pred, 1.0)
            finite = (np.isfinite(F1) & np.all(np.isfinite(delta), axis=1)
                      & np.all(np.isfinite(x_try), axis=1))
            fallback[pending & ~finite & ~accepted_ever] = 3
            ok = (pending & finite & (pred > 0) & (ared > 0)
                  & (rho > config.eta_accept))

            last_pred[pending] = pred[pending]
            last_ared[pending] = ared[pending]
            last_rho[pending] = rho[pending]

            if ok.any():
                dE_cur[ok] = dE_try[ok]
                ds_cur[ok] = ds_try[ok]
                amp_out[ok] = x_try[ok, :n_comp]
                if bg_out is not None:
                    bg_out[ok] = x_try[ok, n_comp:]
                chi2_out[ok] = (2.0 * F1[ok] / n_E).astype(np.float32)
                F0[ok] = F1[ok]
                accepted_ever |= ok
                step_clipped |= ok & clipped

            rejected = pending & ~ok
            lam = np.where(ok & (rho > config.rho_good),
                           lam * config.lambda_down, lam)
            lam = np.where(rejected | (ok & (rho < config.rho_bad)),
                           lam * config.lambda_up, lam)
            retry_count[rejected] += 1
            pending = rejected & (lam <= config.lambda_max)
            newly_dead = rejected & (lam > config.lambda_max) & ~accepted_ever
            fallback[newly_dead & (fallback == 0)] = 2

    # never-accepted spectra whose retries simply ran out
    exhausted = ~accepted_ever & (fallback == 0)
    fallback[exhausted] = 2

    diag = GPLMDiagnostics(
        cond_gram=cond_gram_out,
        cond_hessian=cond_hess_out,
        lm_lambda=lam.astype(np.float32),
        predicted_reduction=last_pred.astype(np.float32),
        actual_reduction=last_ared.astype(np.float32),
        reduction_ratio=last_rho.astype(np.float32),
        step_accepted=accepted_ever.copy(),
        retry_count=retry_count,
        step_clipped=step_clipped,
        min_amplitude=amp_out.min(axis=1),
        negative_amplitude_fraction=float(np.mean(np.any(amp_out < 0, axis=1))),
        fallback_reason=fallback,
        n_basis_evals=n_basis_evals,
    )
    return amp_out, dE_cur, ds_cur, chi2_out, bg_out, diag


def _gp_lm_chunk_mlx(
    Y: np.ndarray,
    dicts: list[DictionaryCache],
    best_idx: np.ndarray,
    delta_E: np.ndarray,
    delta_sigma: np.ndarray,
    amplitudes: np.ndarray,
    chi2_in: np.ndarray,
    config: GPLMConfig,
    bg_design: np.ndarray | None = None,
    n_iter: int = 1,
):
    """MLX-hybrid GP-LM chunk.

    Division of labour: the O(batch·n_E) tensor work — surrogate basis
    evaluation, Gram/rhs, the compact reductions S/C/K/Q/g and the trial
    residual sums — runs on the GPU as batched matmuls; the small (n_lin)
    and (n_params) linear algebra plus the LM accept/reject bookkeeping
    stays on CPU float64 (identical logic to :func:`_gp_lm_chunk`), so only
    (batch × small-matrix) buffers cross the device boundary. No projected
    Jacobian columns and no explicit projector are ever materialised.

    Semantics match `_gp_lm_chunk`; float32 GPU reductions may flip
    accept decisions on razor-edge spectra (documented tolerance in the
    parity test).
    """
    batch, n_E = Y.shape
    n_comp = len(dicts)
    n_params = 2 * n_comp
    n_bg = 0 if bg_design is None else bg_design.shape[1]
    n_lin = n_comp + n_bg

    bad_input = ~np.isfinite(Y).all(axis=1)
    if bad_input.any():
        Y = Y.copy()
        Y[bad_input] = 0.0

    for dc in dicts:
        dc.to_mlx()

    Y_mx = mx.array(np.ascontiguousarray(Y, dtype=np.float32))
    Phi_g_l, Jc_l, Js_l = [], [], []
    dE_grid_val = np.empty((batch, n_comp), dtype=np.float32)
    ds_grid_val = np.empty((batch, n_comp), dtype=np.float32)
    clip_box = np.empty(n_params, dtype=np.float64)
    for k, dc in enumerate(dicts):
        n_ds_k = dc.grid_shape[1]
        idx = best_idx[:, k]
        dE_grid_val[:, k] = dc.dE_grid[idx // n_ds_k]
        ds_grid_val[:, k] = dc.dsigma_grid[idx % n_ds_k]
        idx_mx = mx.array(idx.astype(np.int32))
        Phi_g_l.append(dc._Phi_mx[idx_mx, :, 0])
        Jc_l.append(dc._Jc_mx[idx_mx, :, 0])
        Js_l.append(dc._Jsigma_mx[idx_mx, :, 0])
        dE_step = (float(dc.dE_grid[1] - dc.dE_grid[0])
                   if len(dc.dE_grid) > 1 else 1.0)
        ds_step = (float(dc.dsigma_grid[1] - dc.dsigma_grid[0])
                   if len(dc.dsigma_grid) > 1 else 1.0)
        clip_box[2 * k] = config.clip_step_ratio * dE_step
        clip_box[2 * k + 1] = config.clip_step_ratio * ds_step

    B_mx = (mx.broadcast_to(
        mx.array(bg_design.astype(np.float32))[None],
        (batch, n_E, n_bg)) if bg_design is not None else None)
    eye_jit = np.float32(config.gram_jitter) * mx.eye(n_lin)

    def basis_at(dE_np, ds_np):
        cols = []
        dE_off = mx.array(dE_np - dE_grid_val)
        ds_off = mx.array(ds_np - ds_grid_val)
        for k in range(n_comp):
            cols.append(Phi_g_l[k]
                        + dE_off[:, k:k + 1] * Jc_l[k]
                        + ds_off[:, k:k + 1] * Js_l[k])
        A = mx.stack(cols, axis=2)
        if B_mx is not None:
            A = mx.concatenate([A, B_mx], axis=2)
        return A

    def gram_rhs(A):
        AT = mx.transpose(A, (0, 2, 1))
        G = AT @ A + eye_jit
        rhs = (AT @ Y_mx[:, :, None])[:, :, 0]
        return G, rhs

    def residual_F(A, x_np):
        x_mx = mx.array(x_np.astype(np.float32))
        R = Y_mx - (A @ x_mx[:, :, None])[:, :, 0]
        F = 0.5 * mx.sum(R * R, axis=1)
        return R, F

    # ---- state (identical bookkeeping to the numpy chunk) ----
    dE_cur = delta_E.astype(np.float32).copy()
    ds_cur = delta_sigma.astype(np.float32).copy()
    amp_out = amplitudes.astype(np.float32).copy()
    chi2_out = chi2_in.astype(np.float32).copy()
    bg_out = np.zeros((batch, n_bg), dtype=np.float32) if n_bg else None
    lam = np.full(batch, config.lambda0, dtype=np.float64)
    accepted_ever = np.zeros(batch, dtype=bool)
    retry_count = np.zeros(batch, dtype=np.int32)
    step_clipped = np.zeros(batch, dtype=bool)
    fallback = np.zeros(batch, dtype=np.int8)
    last_pred = np.full(batch, np.nan)
    last_ared = np.full(batch, np.nan)
    last_rho = np.full(batch, np.nan)
    cond_gram_out = np.zeros(batch, dtype=np.float32)
    cond_hess_out = np.zeros(batch, dtype=np.float32)
    n_basis_evals = 0

    for _ in range(max(1, n_iter)):
        A = basis_at(dE_cur, ds_cur)
        G_mx, rhs_mx = gram_rhs(A)
        mx.eval(G_mx, rhs_mx)
        G_np = np.asarray(G_mx).astype(np.float64)
        x = np.linalg.solve(G_np, np.asarray(rhs_mx)[..., None])[..., 0]
        R_mx, F_mx = residual_F(A, x)
        n_basis_evals += 1

        # compact reductions on GPU
        S_cols = []
        q_cols = []
        for p in range(n_params):
            k, axis = divmod(p, 2)
            Jcol = Jc_l[k] if axis == 0 else Js_l[k]
            S_cols.append(mx.array(x[:, k].astype(np.float32))[:, None] * Jcol)
            q_cols.append(mx.sum(Jcol * R_mx, axis=1))
        S = mx.stack(S_cols, axis=2)                     # (b, n_E, p)
        ST = mx.transpose(S, (0, 2, 1))
        C_mx = mx.transpose(A, (0, 2, 1)) @ S            # (b, n_lin, p)
        K_mx = ST @ S                                    # (b, p, p)
        g_mx = (ST @ R_mx[:, :, None])[:, :, 0]          # (b, p)
        qmat_mx = mx.stack(q_cols, axis=1)               # (b, p)
        mx.eval(C_mx, K_mx, g_mx, qmat_mx, F_mx)

        C = np.asarray(C_mx).astype(np.float64)
        K = np.asarray(K_mx).astype(np.float64)
        g64 = np.asarray(g_mx).astype(np.float64)
        F0 = np.asarray(F_mx).astype(np.float64)
        Q = np.zeros((batch, n_lin, n_params))
        qv = np.asarray(qmat_mx).astype(np.float64)
        for p in range(n_params):
            Q[:, p // 2, p] = qv[:, p]
        if bg_out is not None and not accepted_ever.any():
            bg_out[:] = x[:, n_comp:].astype(np.float32)

        sol = np.linalg.solve(G_np, np.concatenate([C, Q], axis=2))
        H64 = (K - C.transpose(0, 2, 1) @ sol[:, :, :n_params]
               + Q.transpose(0, 2, 1) @ sol[:, :, n_params:])
        H64 = 0.5 * (H64 + H64.transpose(0, 2, 1))

        evG = np.linalg.eigvalsh(G_np)
        cond_gram = evG[:, -1] / np.maximum(evG[:, 0], 1e-300)
        evH = np.linalg.eigvalsh(H64)
        cond_hess = evH[:, -1] / np.maximum(evH[:, 0], 1e-300)
        cond_gram_out = np.minimum(cond_gram, 3e38).astype(np.float32)
        cond_hess_out = np.minimum(cond_hess, 3e38).astype(np.float32)
        gate = (np.isfinite(cond_gram) & (cond_gram < config.cond_gate)
                & ~bad_input)
        fallback[bad_input] = 3
        fallback[~gate & ~bad_input & ~accepted_ever] = 1

        d = lm_scale_diag(H64, config.diag_eps)
        pending = gate.copy()

        for _retry in range(config.max_retries + 1):
            if not pending.any():
                break
            delta = lm_step(H64, g64, lam, d)
            clipped = np.any(np.abs(delta) > clip_box, axis=1)
            delta = np.clip(delta, -clip_box, clip_box)
            pred = predicted_reduction(g64, H64, delta)

            dE_try = dE_cur + delta[:, 0::2].astype(np.float32)
            ds_try = ds_cur + delta[:, 1::2].astype(np.float32)
            A_try = basis_at(dE_try, ds_try)
            G_t_mx, rhs_t_mx = gram_rhs(A_try)
            mx.eval(G_t_mx, rhs_t_mx)
            G_t = np.asarray(G_t_mx).astype(np.float64)
            x_try = np.linalg.solve(
                G_t, np.asarray(rhs_t_mx)[..., None])[..., 0]
            _R_t, F_t_mx = residual_F(A_try, x_try)
            mx.eval(F_t_mx)
            F1 = np.asarray(F_t_mx).astype(np.float64)
            n_basis_evals += 1

            ared = F0 - F1
            rho = ared / np.where(pred > 0, pred, 1.0)
            finite = (np.isfinite(F1) & np.all(np.isfinite(delta), axis=1)
                      & np.all(np.isfinite(x_try), axis=1))
            fallback[pending & ~finite & ~accepted_ever] = 3
            ok = (pending & finite & (pred > 0) & (ared > 0)
                  & (rho > config.eta_accept))

            last_pred[pending] = pred[pending]
            last_ared[pending] = ared[pending]
            last_rho[pending] = rho[pending]

            if ok.any():
                dE_cur[ok] = dE_try[ok]
                ds_cur[ok] = ds_try[ok]
                amp_out[ok] = x_try[ok, :n_comp].astype(np.float32)
                if bg_out is not None:
                    bg_out[ok] = x_try[ok, n_comp:].astype(np.float32)
                chi2_out[ok] = (2.0 * F1[ok] / n_E).astype(np.float32)
                F0[ok] = F1[ok]
                accepted_ever |= ok
                step_clipped |= ok & clipped

            rejected = pending & ~ok
            lam = np.where(ok & (rho > config.rho_good),
                           lam * config.lambda_down, lam)
            lam = np.where(rejected | (ok & (rho < config.rho_bad)),
                           lam * config.lambda_up, lam)
            retry_count[rejected] += 1
            pending = rejected & (lam <= config.lambda_max)
            newly_dead = rejected & (lam > config.lambda_max) & ~accepted_ever
            fallback[newly_dead & (fallback == 0)] = 2

    exhausted = ~accepted_ever & (fallback == 0)
    fallback[exhausted] = 2

    diag = GPLMDiagnostics(
        cond_gram=cond_gram_out,
        cond_hessian=cond_hess_out,
        lm_lambda=lam.astype(np.float32),
        predicted_reduction=last_pred.astype(np.float32),
        actual_reduction=last_ared.astype(np.float32),
        reduction_ratio=last_rho.astype(np.float32),
        step_accepted=accepted_ever.copy(),
        retry_count=retry_count,
        step_clipped=step_clipped,
        min_amplitude=amp_out.min(axis=1),
        negative_amplitude_fraction=float(np.mean(np.any(amp_out < 0, axis=1))),
        fallback_reason=fallback,
        n_basis_evals=n_basis_evals,
    )
    return amp_out, dE_cur, ds_cur, chi2_out, bg_out, diag


def _post_ap_gp_lm_refine_numpy(
    Y: np.ndarray,
    dicts: list[DictionaryCache],
    best_idx: np.ndarray,
    delta_E: np.ndarray,
    delta_sigma: np.ndarray,
    amplitudes: np.ndarray,
    chi2_in: np.ndarray,
    n_iter: int = 1,
    config: GPLMConfig | None = None,
    bg_design: np.ndarray | None = None,
    chunk_size: int = 50_000,
    use_mlx: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray | None, GPLMDiagnostics]:
    """Chunked GP-LM post-AP refine (frozen surrogate).

    See :func:`_gp_lm_chunk`. chunk_size defaults lower than the raw path
    because the compact assembly materialises S (chunk, n_E, 2·n_comp).
    use_mlx=True routes the tensor work through :func:`_gp_lm_chunk_mlx`
    (requires MLX; falls back to CPU silently when unavailable).
    """
    config = config or GPLMConfig()
    chunk_fn = _gp_lm_chunk_mlx if (use_mlx and HAS_MLX) else _gp_lm_chunk
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    batch = Y.shape[0]
    n_comp = len(dicts)
    n_bg = 0 if bg_design is None else bg_design.shape[1]

    amp_out = np.empty((batch, n_comp), dtype=np.float32)
    dE_out = np.empty((batch, n_comp), dtype=np.float32)
    ds_out = np.empty((batch, n_comp), dtype=np.float32)
    chi2_out = np.empty(batch, dtype=np.float32)
    bg_out = np.empty((batch, n_bg), dtype=np.float32) if n_bg else None
    diags = []

    for s in range(0, batch, chunk_size):
        e = min(s + chunk_size, batch)
        amp_c, dE_c, ds_c, chi2_c, bg_c, diag_c = chunk_fn(
            Y[s:e], dicts, best_idx[s:e], delta_E[s:e], delta_sigma[s:e],
            amplitudes[s:e], chi2_in[s:e], config,
            bg_design=bg_design, n_iter=n_iter,
        )
        amp_out[s:e] = amp_c
        dE_out[s:e] = dE_c
        ds_out[s:e] = ds_c
        chi2_out[s:e] = chi2_c
        if bg_out is not None:
            bg_out[s:e] = bg_c
        diags.append(diag_c)

    if len(diags) == 1:
        diag = diags[0]
    else:
        n_tot = sum(d.step_accepted.shape[0] for d in diags)
        diag = GPLMDiagnostics(
            **{f: np.concatenate([getattr(d, f) for d in diags])
               for f in ("cond_gram", "cond_hessian", "lm_lambda",
                          "predicted_reduction", "actual_reduction",
                          "reduction_ratio", "step_accepted", "retry_count",
                          "step_clipped", "min_amplitude", "fallback_reason")},
            negative_amplitude_fraction=float(sum(
                d.negative_amplitude_fraction * d.step_accepted.shape[0]
                for d in diags) / n_tot),
            n_basis_evals=sum(d.n_basis_evals for d in diags),
        )
    return amp_out, dE_out, ds_out, chi2_out, bg_out, diag


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def _solve_alternating_projection_numpy(
    Y: np.ndarray,
    dicts: list[DictionaryCache],
    n_iterations: int = 3,
    parabola_dE: bool = True,
    parabola_ds: bool = False,
    apply_newton: bool = False,
    bg_degree: int = -1,
    newton_jacobian_mode: str = "raw",
) -> MultiPeakResult:
    """CPU/numpy implementation of the alternating-projection multipeak
    solver. Used when MLX is unavailable (Linux x86_64, Windows, Intel
    Mac); the GPU path below is the preferred Apple Silicon route.

    Algorithmically equivalent to :func:`solve_alternating_projection`:
    iterate over components, deflate Y against the other components'
    current dictionary picks, score Y_proj against component k's
    dictionary, take argmax, then on the final iteration apply
    parabola refinement and joint LLS. Reuses the existing numpy
    helpers ``_project_orthogonal``, ``_refine_parabola_from_scores``,
    and ``_batch_lls_amplitude``.

    Throughput is typically 5-20× slower than the MLX path; quality
    matches to within float32 noise.
    """
    Y = np.asarray(Y, dtype=np.float32)
    if not Y.flags["C_CONTIGUOUS"]:
        Y = np.ascontiguousarray(Y)

    batch, n_E = Y.shape
    n_comp = len(dicts)
    timing: dict[str, float] = {}

    # ---- Phase 1: Alternating Projection (numpy) ----
    # Mirrors the MLX inline algorithm: 1-D deflation for n_comp <= 2 (a
    # true projection there), block deflation (joint LLS residual +
    # per-component add-back) for n_comp >= 3. Sequential 1-D deflation is
    # NOT a true orthogonal projection against the other components' span
    # for n_comp >= 3 and leaves a deterministic δE bias (≈ -0.07 eV at 3σ
    # spacing, measured 2026-06-10).
    t0 = time.perf_counter()
    best_np = np.stack(
        [np.full(batch, dc.nominal_index, dtype=np.int32) for dc in dicts],
        axis=1,
    )
    last_scores: list[np.ndarray | None] = [None] * n_comp

    for iteration in range(n_iterations):
        is_last = iteration == n_iterations - 1
        if n_comp >= 3:
            # Block deflation: Y_proj_comp = (Y - Phi_all @ amp) + amp_comp·phi_comp
            phis = [dicts[k].Phi_per_grid[best_np[:, k], :, 0]
                    for k in range(n_comp)]
            Phi_cur = np.stack(phis, axis=2)  # (batch, n_E, n_comp)
            G = (np.einsum("bec,bef->bcf", Phi_cur, Phi_cur)
                 + 1e-10 * np.eye(n_comp, dtype=np.float32))
            rhs = np.einsum("bec,be->bc", Phi_cur, Y)
            amp = np.linalg.solve(G, rhs[:, :, None])[:, :, 0]
            Y_res_full = Y - np.einsum("bec,bc->be", Phi_cur, amp)
            for comp in range(n_comp):
                Y_proj = Y_res_full + amp[:, comp:comp + 1] * phis[comp]
                scores = Y_proj @ dicts[comp].D
                best_np[:, comp] = np.argmax(scores, axis=1).astype(np.int32)
                if is_last and (parabola_dE or parabola_ds):
                    last_scores[comp] = scores
            continue
        for comp in range(n_comp):
            if n_comp == 1:
                Y_proj = Y
            else:  # n_comp == 2 — 1-D deflation IS the true projection here
                other = 1 - comp
                phi = dicts[other].Phi_per_grid[best_np[:, other], :, 0]
                phi_sq = np.sum(phi * phi, axis=1, keepdims=True) + 1e-10
                phi_Y = np.sum(phi * Y, axis=1, keepdims=True)
                Y_proj = Y - (phi_Y / phi_sq) * phi

            # scores = Y_proj @ D : (batch, n_dict). D has shape (n_E, n_dict).
            scores = Y_proj @ dicts[comp].D
            best_np[:, comp] = np.argmax(scores, axis=1).astype(np.int32)
            if is_last and (parabola_dE or parabola_ds):
                last_scores[comp] = scores

    timing["alternating"] = time.perf_counter() - t0

    # ---- Phase 1.5: Parabola refinement (or grid pick) ----
    t0 = time.perf_counter()
    delta_E = np.zeros((batch, n_comp), dtype=np.float32)
    delta_sigma = np.zeros((batch, n_comp), dtype=np.float32)
    for k in range(n_comp):
        dE_k, ds_k = _refine_parabola_from_scores(
            last_scores[k], best_np[:, k], dicts[k], parabola_dE, parabola_ds,
        )
        delta_E[:, k] = dE_k
        delta_sigma[:, k] = ds_k
    timing["parabola"] = time.perf_counter() - t0

    # ---- Phase 2: Joint LLS amplitude + chi2 ----
    t0 = time.perf_counter()
    background = None
    if bg_degree >= 0:
        energy = getattr(dicts[0], "energy", None)
        if energy is None:
            raise ValueError(
                "bg_degree>=0 requires the dictionaries to carry their energy "
                "axis (dicts[0].energy is None). Rebuild the dictionaries with a "
                "current cache that stores the energy axis."
            )
        B = build_bg_design(energy, bg_degree)
        phis = [dicts[k].Phi_per_grid[best_np[:, k], :, 0] for k in range(n_comp)]
        amplitudes, background, chi2 = _lls_chi2_bg_numpy(Y, phis, B, n_comp)
    else:
        amplitudes = _batch_lls_amplitude(Y, dicts, best_np)
        # chi2 from reconstruction
        recon = np.zeros_like(Y)
        for k in range(n_comp):
            phi_k = dicts[k].Phi_per_grid[best_np[:, k], :, 0]  # (batch, n_E)
            recon += amplitudes[:, k:k + 1] * phi_k
        chi2 = np.mean((Y - recon) ** 2, axis=1).astype(np.float32)
    timing["lls"] = time.perf_counter() - t0

    # ---- Phase 3 (optional): post-AP Newton refinement ----
    gp_diag = None
    if apply_newton:
        if bg_degree >= 0 and newton_jacobian_mode != "gp_lm":
            raise NotImplementedError(
                "apply_newton with bg_degree>=0 is only implemented on the MLX "
                "path (or newton_jacobian_mode='gp_lm'); the numpy fallback "
                "supports background only in the Phase-2 joint LLS (set "
                "apply_newton=False)."
            )
        t0 = time.perf_counter()
        if newton_jacobian_mode == "gp_lm":
            bg_design = (build_bg_design(dicts[0].energy, bg_degree)
                         if bg_degree >= 0 else None)
            (amplitudes, delta_E, delta_sigma, chi2, bg_new,
             gp_diag) = _post_ap_gp_lm_refine_numpy(
                Y, dicts, best_np, delta_E, delta_sigma, amplitudes, chi2,
                bg_design=bg_design,
            )
            if bg_new is not None and background is not None:
                acc = gp_diag.step_accepted
                background = np.where(acc[:, None], bg_new,
                                      background).astype(np.float32)
        else:
            amplitudes, delta_E, delta_sigma, chi2 = (
                _post_ap_newton_refine_numpy(
                    Y, dicts, best_np, delta_E, delta_sigma, amplitudes,
                    jacobian_mode=newton_jacobian_mode,
                ))
        timing["newton"] = time.perf_counter() - t0

    timing["total"] = sum(v for v in timing.values() if isinstance(v, (int, float)))
    timing["rate"] = batch / timing["total"] if timing["total"] > 0 else float("inf")

    return MultiPeakResult(
        amplitudes=amplitudes,
        delta_E=delta_E,
        delta_sigma=delta_sigma,
        chi2=chi2,
        n_iterations=n_iterations,
        best_indices=best_np,
        timing=timing,
        background=background,
        gp_diagnostics=gp_diag,
    )


def solve_alternating_projection(
    Y: np.ndarray,
    dicts: list[DictionaryCache],
    n_iterations: int = 3,
    parabola_dE: bool = True,
    parabola_ds: bool = False,
    apply_newton: bool = False,
    newton_iter: int = 1,
    newton_exact: bool = False,
    quality_flags: bool = False,
    bg_degree: int = -1,
    fit_gamma: bool = False,
    gamma_chi2_gate: bool = True,
    gamma_chi2_margin: float = 0.01,
    newton_jacobian_mode: str = "raw",
) -> MultiPeakResult:
    """Multi-peak solver using Alternating Projection.

    On Apple Silicon, runs entirely on GPU via MLX (Y transferred once,
    lazy graph, single ``mx.eval`` at end). On hosts without MLX,
    falls back to a pure-numpy implementation that produces equivalent
    results at lower throughput.

    At each iteration, for each component k:
        1. Project Y ⊥ all other components (orthogonal deflation)
        2. Correlate projected Y with component k's dictionary
        3. Update best_idx[k] via argmax

    After iterations: parabola refinement + joint LLS amplitude solve.

    If apply_newton=True, a final post-AP Gauss-Newton step jointly refines
    (δE_k, δσ_k) across all peaks on CPU/numpy. Closes the AP argmax bias
    diagnosed in dev-log 134 (≈ +4 dB on δσ for 2-peak dev-log 77 setup).

    Args:
        Y: Spectra (batch, n_E) float32.
        dicts: Per-component DictionaryCache (n_comp=1 each).
        n_iterations: Number of alternating iterations (default 3).
        parabola_dE: Apply parabola to δE axis (default True).
        parabola_ds: Apply parabola to δσ axis (default False).
        apply_newton: If True, add post-AP joint Gauss-Newton step (default
            False; CPU/numpy, adds ~1 ms per 10K spectra for 2 peaks).
        newton_iter: number of post-AP Newton iterations (default 1).
        newton_exact: If True, the Newton step re-evaluates the exact basis +
            analytic Jacobian at the refined (center, sigma) each iteration
            instead of a frozen grid-cell Taylor surrogate (default False).
            Breaks the surrogate δσ ceiling at the cost of per-iteration Voigt
            re-evaluation. Requires MLX (raises NotImplementedError on the
            numpy fallback).
        quality_flags: If True (with apply_newton), also populate the result's
            per-spectrum cond_number / param_std / soft_std from the Newton
            Hessian (MLX path only). Flags near-degenerate fits such as the
            δσ₁−δσ₂ valley of overlapping peaks. Negligible extra cost.
        bg_degree: Polynomial background coupling (default −1 = OFF, fully
            backward compatible: keeps the fast Cramer amplitude path). When
            ≥ 0, a polynomial background of that degree (0=constant, 1=+linear,
            2=+quadratic, 3=+cubic) on the normalized energy axis is fit jointly
            with the peak amplitudes in the Phase-2 LLS (and re-solved in the
            post-AP Newton step), removing the unmodeled-background bias that
            otherwise leaks into δσ. The fitted coefficients are
            returned in ``result.background`` (None when bg_degree=−1).
        fit_gamma: If True (with apply_newton), the post-AP Newton step also
            refines the per-peak Lorentzian width δγ — params per peak go
            2→3 (δE, δσ, δγ). Requires ``newton_exact=True`` (γ has no frozen
            grid basis); raises ValueError otherwise. Recovers per-spectrum γ
            and prevents a γ-mismatch leaking into δσ (analogous to the BG
            coupling). Default False. The recovered δγ is returned in
            ``result.delta_gamma`` (None when fit_gamma=False).
        gamma_chi2_gate: Per-spectrum χ² gate on the δγ refinement (only
            meaningful when fit_gamma=True; default True). The extra δγ degree
            of freedom helps when γ is genuinely off (+~6.6 dB δσ on the 2-peak
            dev-log 77 setup) but HURTS when γ is already correct, because noise
            leaks into σ through the chi²-flat σ↔γ valley (noiseless δσ degrades
            ~27 dB). The gate runs BOTH the (δE, δσ) Newton fit and the
            (δE, δσ, δγ) fit and keeps the δγ result PER SPECTRUM only where it
            lowers χ² by more than ``gamma_chi2_margin`` (relative); elsewhere it
            keeps the (δE, δσ) result with δγ=0. This removes the damage on
            γ-correct spectra (≈ OFF) while retaining the gain on γ-off spectra.
            Adds one extra Newton solve when active. When False, behaves as the
            ungated fit_gamma (single δγ fit applied to every spectrum). Ignored
            when fit_gamma=False.
        gamma_chi2_margin: Relative χ² improvement required for the gate to
            accept the δγ fit (default 0.01 = 1%). Per spectrum, δγ is kept iff
            ``chi2_gamma < chi2_nogamma · (1 − gamma_chi2_margin)``. A small
            positive margin prevents noise-only χ² wiggles from flipping the
            gate on γ-correct spectra. Ignored unless gamma_chi2_gate is active.
        newton_jacobian_mode: EXPERIMENTAL (default "raw" = existing
            behavior, byte-identical fast path). "kaufman" and
            "golub_pereyra" replace the post-AP Newton Jacobian with the
            variable-projection / full Golub–Pereyra reduced-residual
            Jacobian, evaluated on the frozen grid-cell surrogate basis.
            "gp_lm" runs the compact-Hessian Golub–Pereyra step under a
            scaled-diagonal Levenberg–Marquardt safeguard with per-spectrum
            accept/reject (rejected spectra keep the AP seed untouched;
            condition-gated); its per-spectrum diagnostics are returned in
            ``result.gp_diagnostics`` (GPLMDiagnostics).
            "raw" here names the Jacobian approximation (S = a·J with the
            amplitude-vs-θ coupling ignored — the current production step);
            it is unrelated to the pre-Tikhonov ``H_raw`` Hessian in the
            quality_flags diagnostics. All modes share the same gradient
            and differ only in the Gauss-Newton curvature.
            Non-raw modes run on CPU/numpy (the MLX Newton kernel is
            bypassed; ``newton_iter`` repeated single steps) and are
            incompatible with ``newton_exact``, ``fit_gamma``,
            ``quality_flags`` diagnostics and ``bg_degree>=0`` (ValueError).
            See gp_reference.py and benchmarks/gp_newton_comparison.py for
            the verification study behind this flag.

    Returns:
        MultiPeakResult with amplitudes, delta_E, delta_sigma, chi2 (and, when
        quality_flags=True, cond_number / param_std / soft_std; when
        bg_degree≥0, background; when fit_gamma, delta_gamma — the δγ actually
        applied per spectrum, 0 where the χ² gate rejected it).
    """
    if fit_gamma and not (apply_newton and newton_exact):
        # Surface the requirement regardless of backend (the MLX Newton helper
        # also enforces it, but checking here gives a clear error before any
        # GPU work and on the numpy path too).
        raise ValueError(
            "fit_gamma=True requires apply_newton=True and newton_exact=True."
        )
    if newton_jacobian_mode not in ("raw", "kaufman", "golub_pereyra",
                                    "gp_lm"):
        raise ValueError(
            "newton_jacobian_mode must be 'raw', 'kaufman', "
            f"'golub_pereyra' or 'gp_lm', got {newton_jacobian_mode!r}"
        )
    if newton_jacobian_mode != "raw" and apply_newton:
        if newton_exact or fit_gamma or quality_flags:
            raise ValueError(
                "newton_jacobian_mode != 'raw' is an experimental path and "
                "does not support newton_exact, fit_gamma or quality_flags."
            )
        if bg_degree >= 0 and newton_jacobian_mode != "gp_lm":
            raise ValueError(
                "bg_degree>=0 with an experimental newton_jacobian_mode is "
                "only supported by 'gp_lm' (fixed polynomial background in "
                "the joint projection); 'kaufman'/'golub_pereyra' do not "
                "support it."
            )
    if not HAS_MLX:
        if fit_gamma:
            raise NotImplementedError(
                "fit_gamma=True requires MLX; the numpy fallback does not "
                "implement the δγ Newton refinement."
            )
        if newton_exact and apply_newton:
            raise NotImplementedError(
                "newton_exact=True requires MLX; the numpy fallback only "
                "supports the frozen-surrogate Newton step."
            )
        return _solve_alternating_projection_numpy(
            Y, dicts, n_iterations, parabola_dE, parabola_ds, apply_newton,
            bg_degree=bg_degree, newton_jacobian_mode=newton_jacobian_mode,
        )

    Y = np.asarray(Y, dtype=np.float32)
    if not Y.flags["C_CONTIGUOUS"]:
        Y = np.ascontiguousarray(Y)

    batch, n_E = Y.shape
    n_comp = len(dicts)
    timing = {}

    # Ensure MLX caches
    for dc in dicts:
        dc.to_mlx()

    # ---- B3: polynomial background design (opt-in) ----
    B_mx = None
    if bg_degree >= 0:
        energy = getattr(dicts[0], "energy", None)
        if energy is None:
            raise ValueError(
                "bg_degree>=0 requires the dictionaries to carry their energy "
                "axis (dicts[0].energy is None). Rebuild the dictionaries with a "
                "current cache that stores the energy axis."
            )
        B_mx = mx.array(build_bg_design(energy, bg_degree))

    # ---- Phase 1: Alternating Projection (GPU-native) ----
    t0 = time.perf_counter()

    Y_mx = mx.array(Y)  # Single CPU→GPU transfer
    best_mx = [mx.full((batch,), dc.nominal_index, dtype=mx.int32) for dc in dicts]

    # Pre-squeeze Phi: (n_dict, n_E) per component (each dict has n_comp=1)
    Phi_mx = [dc._Phi_mx[:, :, 0] for dc in dicts]

    # GPU parabola results (lazy MLX arrays, computed on last iteration)
    para_dE_mx = [None] * n_comp
    para_ds_mx = [None] * n_comp

    # Threshold: use block deflation for n_comp >= this.
    # Sequential 1-D deflation is NOT a true orthogonal projection against
    # the other components' span for n_comp >= 3: at 3σ peak spacing it
    # leaves a deterministic, noise-independent δE bias of ≈ -0.07 eV on the
    # interior/lower peaks (measured 2026-06-10; vanishes entirely under
    # block deflation). Block deflation was originally gated to n_comp >= 6
    # because its per-iteration CPU Gram solve could not be
    # amortized over few components; the unrolled GPU Cholesky
    # (_gram_solve_mx) removed that sync, so block deflation now wins for
    # ALL n_comp >= 3 on both accuracy (no bias) and speed.
    BLOCK_DEFLATION_THRESHOLD = 3

    for iteration in range(n_iterations):
        is_last = iteration == n_iterations - 1

        if n_comp >= BLOCK_DEFLATION_THRESHOLD:
            # Block deflation: single joint solve per iteration
            # Replaces O(n_comp²) sequential deflation with O(n_comp) residual updates.
            #
            # For each comp: Y_proj_comp = Y - Phi_all @ amp + amp[comp] * phi[comp]
            #                            = Y_res_full + amp[comp] * phi[comp]
            # Compute Y_res_full ONCE per iteration via batch LLS, then add
            # each comp's contribution back individually.
            phis_mx = [Phi_mx[k][best_mx[k]] for k in range(n_comp)]
            Phi_current = mx.stack(phis_mx, axis=2)  # (batch, n_E, n_comp)
            Phi_T = mx.transpose(Phi_current, (0, 2, 1))  # (batch, n_comp, n_E)

            G_mx = Phi_T @ Phi_current + 1e-10 * mx.eye(n_comp)
            rhs_mx = (Phi_T @ Y_mx[:, :, None])[:, :, 0]  # (batch, n_comp)

            # Lazy GPU Cholesky — no per-iteration GPU→CPU sync (the CPU
            # solve this replaces was why block deflation used to lose for
            # small n_comp).
            amp_mx = _gram_solve_mx(G_mx, rhs_mx, n_comp)

            # Y_res_full = Y - Phi_current @ amp (contribution from ALL comps)
            Y_res_full = Y_mx - (Phi_current @ amp_mx[:, :, None])[:, :, 0]

            for comp in range(n_comp):
                phi_comp = phis_mx[comp]
                amp_comp = amp_mx[:, comp:comp + 1]
                Y_proj_mx = Y_res_full + amp_comp * phi_comp

                scores_mx = Y_proj_mx @ dicts[comp]._D_mx  # (batch, n_dict)
                best_mx[comp] = mx.argmax(scores_mx, axis=1)

                if is_last and (parabola_dE or parabola_ds):
                    para_dE_mx[comp], para_ds_mx[comp] = _parabola_refine_gpu(
                        scores_mx, best_mx[comp], dicts[comp],
                        parabola_dE, parabola_ds,
                    )
        else:
            # n_comp < BLOCK_DEFLATION_THRESHOLD: sequential deflation
            # (O(n_comp²) GPU ops, no CPU solve — faster for small n_comp)
            for comp in range(n_comp):
                if n_comp == 1:
                    Y_proj_mx = Y_mx
                elif n_comp == 2:
                    other = 1 - comp
                    phi = Phi_mx[other][best_mx[other]]
                    phi_sq = mx.sum(phi * phi, axis=1, keepdims=True) + 1e-10
                    phi_Y = mx.sum(phi * Y_mx, axis=1, keepdims=True)
                    Y_proj_mx = Y_mx - (phi_Y / phi_sq) * phi
                else:
                    # n_comp >= 3: sequential deflation
                    Y_proj_mx = Y_mx
                    for k in range(n_comp):
                        if k == comp:
                            continue
                        phi = Phi_mx[k][best_mx[k]]
                        phi_sq = mx.sum(phi * phi, axis=1, keepdims=True) + 1e-10
                        phi_Y = mx.sum(phi * Y_proj_mx, axis=1, keepdims=True)
                        Y_proj_mx = Y_proj_mx - (phi_Y / phi_sq) * phi

                scores_mx = Y_proj_mx @ dicts[comp]._D_mx  # (batch, n_dict)
                best_mx[comp] = mx.argmax(scores_mx, axis=1)

                if is_last and (parabola_dE or parabola_ds):
                    para_dE_mx[comp], para_ds_mx[comp] = _parabola_refine_gpu(
                        scores_mx, best_mx[comp], dicts[comp],
                        parabola_dE, parabola_ds,
                    )

    # Single batch eval: indices + parabola results (GPU→CPU)
    eval_list = list(best_mx)
    for k in range(n_comp):
        if para_dE_mx[k] is not None:
            eval_list.extend([para_dE_mx[k], para_ds_mx[k]])
    mx.eval(*eval_list)

    best_np = np.stack([np.array(b).astype(np.int32) for b in best_mx], axis=1)

    timing["alternating"] = time.perf_counter() - t0

    # ---- Phase 1.5: Extract parabola results ----
    t0 = time.perf_counter()
    delta_E = np.zeros((batch, n_comp), dtype=np.float32)
    delta_sigma = np.zeros((batch, n_comp), dtype=np.float32)

    for k in range(n_comp):
        if para_dE_mx[k] is not None:
            delta_E[:, k] = np.array(para_dE_mx[k]).astype(np.float32)
            delta_sigma[:, k] = np.array(para_ds_mx[k]).astype(np.float32)
        else:
            # No parabola: grid values from best indices
            n_ds = dicts[k].grid_shape[1]
            dE_idx = best_np[:, k] // n_ds
            ds_idx = best_np[:, k] % n_ds
            delta_E[:, k] = dicts[k].dE_grid[dE_idx]
            delta_sigma[:, k] = dicts[k].dsigma_grid[ds_idx]

    timing["parabola"] = time.perf_counter() - t0

    # ---- Phase 2: Joint LLS amplitude + chi2 ----
    t0 = time.perf_counter()
    background = None
    if bg_degree >= 0:
        # Joint [peaks | background] LLS — general (n_comp + n_bg) solve.
        phis_mx = [Phi_mx[k][best_mx[k]] for k in range(n_comp)]
        amplitudes, background, chi2 = _lls_chi2_bg_gpu(
            Y_mx, phis_mx, B_mx, n_comp,
        )
    elif n_comp <= 2:
        # GPU-native: single gather + Cramer + residual
        amplitudes, chi2 = _lls_chi2_gpu(Y_mx, Phi_mx, best_mx, n_comp)
    else:
        # GPU gather → GPU Gram/rhs → CPU solve → GPU chi2
        phis_mx = [Phi_mx[k][best_mx[k]] for k in range(n_comp)]
        Phi_all_mx = mx.stack(phis_mx, axis=2)  # (batch, n_E, n_comp)
        Phi_T_mx = mx.transpose(Phi_all_mx, (0, 2, 1))  # (batch, n_comp, n_E)
        G_mx = Phi_T_mx @ Phi_all_mx + 1e-10 * mx.eye(n_comp)
        rhs_mx = (Phi_T_mx @ Y_mx[:, :, None])[:, :, 0]  # (batch, n_comp)
        # CPU solve (mx.linalg.solve is CPU-only in MLX)
        amp_mx = mx.linalg.solve(
            G_mx, rhs_mx[:, :, None], stream=mx.cpu,
        )[:, :, 0]  # (batch, n_comp)
        # GPU reconstruction + chi2
        recon_mx = (Phi_all_mx @ amp_mx[:, :, None])[:, :, 0]
        chi2_mx = mx.mean((Y_mx - recon_mx) ** 2, axis=1)
        mx.eval(amp_mx, chi2_mx)
        amplitudes = np.array(amp_mx).astype(np.float32)
        chi2 = np.array(chi2_mx).astype(np.float32)
    timing["lls"] = time.perf_counter() - t0

    # ---- Phase 3 (optional): post-AP Newton refinement ----
    # GPU-native (MLX) path keeps Y on the GPU; only the 4×4 system solve
    # falls back to CPU because LAPACK on small batched systems is faster
    # than MLX's GPU triangular solve at our batch sizes.
    #
    # If `newton_hessian_cache` is non-None, the 10 H-entries are produced
    # by gather + scalar multiply instead of 10 energy-axis reductions —
    # ≈ 1.5-3× speedup. Cache is built once per dict list and stashed on
    # the first dict for re-use across chunks.
    diag = None
    delta_gamma = None
    gp_diag = None
    if apply_newton and newton_jacobian_mode == "gp_lm":
        # EXPERIMENTAL: compact-Hessian GP with LM accept/reject.
        # Rejected spectra keep the AP/parabola seed untouched (including
        # their Phase-2 background coefficients when bg_degree >= 0).
        # Background scope: FIXED polynomial design only — data-dependent
        # backgrounds (Shirley/Tougaard) are NOT part of this projection.
        t0 = time.perf_counter()
        bg_design = (build_bg_design(dicts[0].energy, bg_degree)
                     if bg_degree >= 0 else None)
        (amplitudes, delta_E, delta_sigma, chi2, bg_new,
         gp_diag) = _post_ap_gp_lm_refine_numpy(
            Y, dicts, best_np, delta_E, delta_sigma, amplitudes, chi2,
            n_iter=newton_iter, bg_design=bg_design, use_mlx=True,
        )
        if bg_new is not None and background is not None:
            acc = gp_diag.step_accepted
            background = np.where(acc[:, None], bg_new,
                                  background).astype(np.float32)
        timing["newton"] = time.perf_counter() - t0
    elif apply_newton and newton_jacobian_mode != "raw":
        # EXPERIMENTAL: projected-Jacobian (Kaufman / Golub–Pereyra) Newton.
        # Runs the frozen-surrogate single step on CPU newton_iter times;
        # the MLX Newton kernel and its Hessian cache are bypassed. The
        # parameter guards at the top of this function keep this path away
        # from bg/γ/exact/diagnostics interactions.
        t0 = time.perf_counter()
        for _ in range(max(1, newton_iter)):
            amplitudes, delta_E, delta_sigma, chi2 = (
                _post_ap_newton_refine_numpy(
                    Y, dicts, best_np, delta_E, delta_sigma, amplitudes,
                    jacobian_mode=newton_jacobian_mode,
                ))
        timing["newton"] = time.perf_counter() - t0
    elif apply_newton:
        t0 = time.perf_counter()
        # The grid-cell Hessian cache is only valid for the frozen surrogate;
        # exact_jacobian re-derives H from the freshly evaluated Jacobians.
        cache = None
        if not newton_exact:
            cache = getattr(dicts[0], "_hessian_cache", None)
            if cache is None or getattr(cache, "n_comp", -1) != n_comp:
                cache = build_multipeak_hessian_cache(dicts)
                dicts[0]._hessian_cache = cache

        # The AP seed (delta_E, delta_sigma, amplitudes) is shared between the
        # no-γ and γ Newton calls below. _post_ap_newton_refine_mlx copies these
        # internally (dE_cum = delta_E.astype(..., copy=True)), so passing the
        # same arrays to both calls is safe — neither call mutates the seed.
        newton_kw = dict(
            hessian_cache=cache, n_iter=newton_iter,
            exact_jacobian=newton_exact, bg_design_mx=B_mx,
        )
        if fit_gamma and gamma_chi2_gate:
            # χ² gate: run BOTH the (δE, δσ, δγ) fit and the (δE, δσ)-only fit
            # from the SAME AP seed, then keep the γ result PER SPECTRUM only
            # where it lowers χ² by more than the relative margin. This removes
            # the δσ damage on γ-correct spectra (where the extra δγ DoF only
            # fits noise via the σ↔γ valley) while retaining the gain where γ is
            # genuinely off. quality_flags reports the γ-side diagnostics.
            (amp_g, dE_g, ds_g, chi2_g, diag,
             bg_g, dg_g) = _post_ap_newton_refine_mlx(
                Y_mx, dicts, best_mx, delta_E, delta_sigma, amplitudes,
                compute_diagnostics=quality_flags, fit_gamma=True, **newton_kw,
            )
            (amp_n, dE_n, ds_n, chi2_n, _diag_n,
             bg_n, _dg_n) = _post_ap_newton_refine_mlx(
                Y_mx, dicts, best_mx, delta_E, delta_sigma, amplitudes,
                compute_diagnostics=False, fit_gamma=False, **newton_kw,
            )
            take_g = chi2_g < (chi2_n * (1.0 - gamma_chi2_margin))
            tg2 = take_g[:, None]
            amplitudes = np.where(tg2, amp_g, amp_n)
            delta_E = np.where(tg2, dE_g, dE_n)
            delta_sigma = np.where(tg2, ds_g, ds_n)
            chi2 = np.where(take_g, chi2_g, chi2_n)
            dg_newton = np.where(tg2, dg_g, 0.0).astype(np.float32)
            if bg_g is not None and bg_n is not None:
                bg_newton = np.where(tg2, bg_g, bg_n)
            else:
                bg_newton = bg_g if bg_g is not None else bg_n
        else:
            (amplitudes, delta_E, delta_sigma, chi2, diag,
             bg_newton, dg_newton) = _post_ap_newton_refine_mlx(
                Y_mx, dicts, best_mx, delta_E, delta_sigma, amplitudes,
                compute_diagnostics=quality_flags, fit_gamma=fit_gamma,
                **newton_kw,
            )
        if bg_newton is not None:
            background = bg_newton
        if dg_newton is not None:
            delta_gamma = dg_newton
        timing["newton"] = time.perf_counter() - t0

    timing["total"] = sum(v for v in timing.values() if isinstance(v, (int, float)))
    timing["rate"] = batch / timing["total"] if timing["total"] > 0 else float("inf")

    return MultiPeakResult(
        amplitudes=amplitudes,
        delta_E=delta_E,
        delta_sigma=delta_sigma,
        chi2=chi2,
        n_iterations=n_iterations,
        best_indices=best_np,
        timing=timing,
        cond_number=diag["cond_number"] if diag else None,
        param_std=diag["param_std"] if diag else None,
        soft_std=diag["soft_std"] if diag else None,
        background=background,
        delta_gamma=delta_gamma,
        gp_diagnostics=gp_diag,
    )


# ---------------------------------------------------------------------------
# Chunked processing
# ---------------------------------------------------------------------------


def solve_multipeak_chunked(
    Y: np.ndarray,
    dicts: list[DictionaryCache],
    chunk_size: int | None = None,
    n_iterations: int = 3,
    parabola_dE: bool = True,
    parabola_ds: bool = False,
    bg_degree: int = -1,
    apply_newton: bool = False,
    newton_iter: int = 1,
    newton_exact: bool = False,
    fit_gamma: bool = False,
    gamma_chi2_gate: bool = True,
    gamma_chi2_margin: float = 0.01,
) -> MultiPeakResult:
    """Chunked multi-peak solver for large datasets.

    Splits Y into chunks to stay within Metal GPU memory.
    Each chunk is processed by ``solve_alternating_projection``.

    Args:
        Y: Spectra (n_pixels, n_E).
        dicts: Per-component DictionaryCache list.
        chunk_size: Spectra per chunk (auto-determined if None).
        n_iterations: Alternating iterations per chunk.
        parabola_dE: Apply parabola to δE axis.
        parabola_ds: Apply parabola to δσ axis.
        bg_degree: Polynomial background degree (−1 = OFF, default). See
            :func:`solve_alternating_projection`.
        apply_newton: Add the post-AP Newton step. See
            :func:`solve_alternating_projection`.
        newton_iter: Number of Newton iterations.
        newton_exact: Use the exact per-iteration Jacobian.
        fit_gamma: Also refine δγ in the Newton step (requires
            ``apply_newton`` and ``newton_exact``). See
            :func:`solve_alternating_projection`.
        gamma_chi2_gate: Per-spectrum χ² gate on the δγ refinement (default
            True; only meaningful when fit_gamma=True). See
            :func:`solve_alternating_projection`.
        gamma_chi2_margin: Relative χ² improvement the gate requires to accept
            δγ (default 0.01). See :func:`solve_alternating_projection`.

    Returns:
        MultiPeakResult concatenated from all chunks.
    """
    Y = np.asarray(Y, dtype=np.float32)
    n_pixels, n_E = Y.shape

    if chunk_size is None:
        # Auto: scores matrix = chunk × max(n_dict) × 4 bytes
        # Target ~4 GB for scores (Metal 80 GB budget)
        max_n_dict = max(dc.n_dict for dc in dicts)
        n_comp = len(dicts)
        # Block deflation (n_comp>=6) allocates Phi_current (batch, n_E, n_comp)
        # which can be large for many components.
        block_extra = n_E * n_comp * 4 if n_comp >= 6 else 0
        bytes_per_row = max_n_dict * 4 + n_E * 4 * 3 + block_extra
        chunk_size = int(4e9 / bytes_per_row)
        chunk_size = max(1024, min(chunk_size, n_pixels))

    if n_pixels <= chunk_size:
        return solve_alternating_projection(
            Y, dicts, n_iterations, parabola_dE, parabola_ds,
            apply_newton=apply_newton, newton_iter=newton_iter,
            newton_exact=newton_exact, bg_degree=bg_degree, fit_gamma=fit_gamma,
            gamma_chi2_gate=gamma_chi2_gate, gamma_chi2_margin=gamma_chi2_margin,
        )

    n_comp = len(dicts)
    all_amp = []
    all_dE = []
    all_ds = []
    all_chi2 = []
    all_best = []
    all_bg = []
    all_dg = []
    total_timing = {}

    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)
        result = solve_alternating_projection(
            Y[start:end], dicts, n_iterations, parabola_dE, parabola_ds,
            apply_newton=apply_newton, newton_iter=newton_iter,
            newton_exact=newton_exact, bg_degree=bg_degree, fit_gamma=fit_gamma,
            gamma_chi2_gate=gamma_chi2_gate, gamma_chi2_margin=gamma_chi2_margin,
        )
        all_amp.append(result.amplitudes)
        all_dE.append(result.delta_E)
        all_ds.append(result.delta_sigma)
        all_chi2.append(result.chi2)
        all_best.append(result.best_indices)
        if result.background is not None:
            all_bg.append(result.background)
        if result.delta_gamma is not None:
            all_dg.append(result.delta_gamma)

        if result.timing:
            for key, val in result.timing.items():
                if isinstance(val, (int, float)):
                    total_timing[key] = total_timing.get(key, 0.0) + val

    total_timing["rate"] = (
        n_pixels / total_timing["total"] if total_timing.get("total", 0) > 0 else 0
    )

    return MultiPeakResult(
        amplitudes=np.concatenate(all_amp, axis=0),
        delta_E=np.concatenate(all_dE, axis=0),
        delta_sigma=np.concatenate(all_ds, axis=0),
        chi2=np.concatenate(all_chi2, axis=0),
        n_iterations=n_iterations,
        best_indices=np.concatenate(all_best, axis=0),
        timing=total_timing,
        background=np.concatenate(all_bg, axis=0) if all_bg else None,
        delta_gamma=np.concatenate(all_dg, axis=0) if all_dg else None,
    )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def process_multipeak(
    Y: np.ndarray,
    config: MultiPeakConfig,
    grid_type: str = "uniform",
    n_iterations: int = 3,
    chunk_size: int | None = None,
    parabola_dE: bool = True,
    parabola_ds: bool = False,
    auto_constrain: bool = True,
    dicts: list["DictionaryCache"] | None = None,
    bg_degree: int = -1,
    apply_newton: bool = False,
    newton_iter: int = 1,
    newton_exact: bool = False,
    fit_gamma: bool = False,
    gamma_chi2_gate: bool = True,
    gamma_chi2_margin: float = 0.01,
) -> MultiPeakResult:
    """Multi-peak pipeline entry point.

    Builds per-component dictionaries and runs the alternating projection
    solver with optional chunking for large datasets.

    Usage::

        config = MultiPeakConfig(
            peaks=[
                ComponentConfig(center=284.8, sigma=0.425, gamma=0.125),
                ComponentConfig(center=286.5, sigma=0.425, gamma=0.125),
            ],
            energy_axis=np.linspace(280, 290, 101),
        )
        result = process_multipeak(Y, config)

    Args:
        Y: Spectra (n_pixels, n_E).
        config: MultiPeakConfig.
        grid_type: Grid type for dictionary axes.
        n_iterations: Alternating projection iterations.
        chunk_size: Spectra per chunk (auto if None).
        parabola_dE: Apply parabola to δE axis.
        parabola_ds: Apply parabola to δσ axis.
        auto_constrain: Automatically limit each component's dE_range
            to prevent cross-assignment when peaks are closer than
            2 × dE_range. Default True.
        dicts: Pre-built per-component dictionaries (skip build if provided).
        bg_degree: Polynomial background degree (−1 = OFF, default). See
            :func:`solve_alternating_projection`.
        apply_newton: Add the post-AP Newton step. See
            :func:`solve_alternating_projection`.
        newton_iter: Number of Newton iterations.
        newton_exact: Use the exact per-iteration Jacobian.
        fit_gamma: Also refine δγ in the Newton step (requires
            ``apply_newton`` and ``newton_exact``). See
            :func:`solve_alternating_projection`.
        gamma_chi2_gate: Per-spectrum χ² gate on the δγ refinement (default
            True; only meaningful when fit_gamma=True). See
            :func:`solve_alternating_projection`.
        gamma_chi2_margin: Relative χ² improvement the gate requires to accept
            δγ (default 0.01). See :func:`solve_alternating_projection`.

    Returns:
        MultiPeakResult.
    """
    if auto_constrain:
        config.constrain_dE_ranges()

    timing = {}
    t0 = time.perf_counter()
    if dicts is None:
        dicts = build_multipeak_dictionaries(config, grid_type)
    timing["dict_build"] = time.perf_counter() - t0

    result = solve_multipeak_chunked(
        Y, dicts, chunk_size, n_iterations, parabola_dE, parabola_ds,
        bg_degree=bg_degree, apply_newton=apply_newton, newton_iter=newton_iter,
        newton_exact=newton_exact, fit_gamma=fit_gamma,
        gamma_chi2_gate=gamma_chi2_gate, gamma_chi2_margin=gamma_chi2_margin,
    )

    if result.timing:
        result.timing["dict_build"] = timing["dict_build"]
        result.timing["total"] += timing["dict_build"]
        if result.timing["total"] > 0:
            result.timing["rate"] = Y.shape[0] / result.timing["total"]

    result.separation_warning = config.separation_sigma() < 3.5

    # Compute CRLB solvability (lightweight, ~1ms)
    try:
        result.solvability = config.solvability()
    except Exception:
        pass  # Non-critical: don't break pipeline if CRLB fails

    return result


# ---------------------------------------------------------------------------
# 2-Stage γ-calibrated multi-peak solver
# ---------------------------------------------------------------------------


def _deflate_component(
    Y: np.ndarray,
    config: MultiPeakConfig,
    amplitudes: np.ndarray,
    delta_E: np.ndarray,
    delta_sigma: np.ndarray,
    target_comp: int,
) -> np.ndarray:
    """Remove all components except *target_comp* from Y.

    Reconstructs each non-target component's profile at the parabola-refined
    parameters (center + δE, sigma + δσ, gamma) for accurate deflation.
    This avoids grid discretization artifacts that would bias Dict3D γ
    estimation.

    Args:
        Y: (batch, n_E) spectra.
        config: MultiPeakConfig with peak definitions and energy axis.
        amplitudes: (batch, n_comp) amplitudes from Stage 1a.
        delta_E: (batch, n_comp) parabola-refined δE from Stage 1a.
        delta_sigma: (batch, n_comp) parabola-refined δσ from Stage 1a.
        target_comp: Component index to keep (all others subtracted).

    Returns:
        Y_deflated: (batch, n_E) residual for target component.
    """
    from scipy.special import wofz

    energy = np.asarray(config.energy_axis, dtype=np.float64)
    Y_deflated = Y.copy()

    for j in range(config.n_comp):
        if j == target_comp:
            continue
        peak = config.peaks[j]
        centers_j = peak.center + delta_E[:, j].astype(np.float64)
        sigmas_j = np.maximum(peak.sigma + delta_sigma[:, j].astype(np.float64), 0.01)
        gamma_j = peak.gamma

        # Vectorized Voigt profile at refined parameters
        # z shape: (batch, n_E)
        z = (energy[np.newaxis, :] - centers_j[:, np.newaxis]
             + 1j * gamma_j) / (sigmas_j[:, np.newaxis] * np.sqrt(2.0))
        phi_j = np.real(wofz(z)) / (sigmas_j[:, np.newaxis] * np.sqrt(2.0 * np.pi))

        # SO doublet: add partner peak contribution
        if peak.is_doublet:
            partner_centers = centers_j + peak.so_split
            z_p = (energy[np.newaxis, :] - partner_centers[:, np.newaxis]
                   + 1j * gamma_j) / (sigmas_j[:, np.newaxis] * np.sqrt(2.0))
            phi_p = np.real(wofz(z_p)) / (sigmas_j[:, np.newaxis] * np.sqrt(2.0 * np.pi))
            phi_j = phi_j + phi_p / peak.branch_ratio

        phi_j = phi_j.astype(np.float32)

        Y_deflated -= amplitudes[:, j : j + 1] * phi_j

    return Y_deflated


def solve_multipeak_2stage(
    Y: np.ndarray,
    config: MultiPeakConfig,
    grid_type: str = "uniform",
    n_iterations: int = 3,
    chunk_size: int | None = None,
    parabola_dE: bool = True,
    parabola_ds: bool = False,
    auto_constrain: bool = True,
    n_calibration: int | None = None,
    dgamma_range: tuple[float, float] = (-0.05, 0.05),
    dgamma_step: float | None = None,
    aggregation: str = "mean",
) -> MultiPeak2StageResult:
    """2-stage γ-calibrated multi-peak solver.

    Stage 1a: Standard Dict2D alternating projection on calibration subset
              to separate components (amplitudes, shifts, best_indices).
    Stage 1b: Per-component γ estimation — deflate other components, then
              run Dict3D on the isolated residual. Aggregate per-spectrum
              δγ to a per-component global Δγ via √N averaging.
    Stage 2:  Rebuild Dict2D with corrected γ per component, run full
              alternating projection on all spectra.

    Note: σ variation in the data is required for γ estimation to work
    (Fisher σ-γ coupling). Use with data that has genuine σ variation,
    or ensure ds_range is non-zero in test data.

    Args:
        Y: Spectra (n_spectra, n_energy) float32.
        config: MultiPeakConfig with peak definitions and energy axis.
        grid_type: Grid type for dictionary axes.
        n_iterations: Alternating projection iterations.
        chunk_size: Spectra per chunk for Stage 2 (auto if None).
        parabola_dE: Apply parabola to δE axis.
        parabola_ds: Apply parabola to δσ axis.
        auto_constrain: Constrain dE_ranges for cross-assignment prevention.
        n_calibration: Number of spectra for Stage 1 (None = min(2000, N)).
        dgamma_range: δγ search range for Dict3D.
        dgamma_step: δγ grid step (default 0.01 = fine resolution).
        aggregation: "mean" or "median" for global Δγ from per-spectrum.

    Returns:
        MultiPeak2StageResult with per-component γ correction diagnostics.
    """
    Y = np.asarray(Y, dtype=np.float32)
    n_spectra = Y.shape[0]
    n_comp = config.n_comp
    timing = {}

    if auto_constrain:
        config.constrain_dE_ranges()

    # --- Calibration subset selection ---
    if n_calibration is None:
        n_cal = min(2000, n_spectra)
    else:
        n_cal = min(n_calibration, n_spectra)

    if n_cal < n_spectra:
        cal_idx = np.linspace(0, n_spectra - 1, n_cal, dtype=int)
        Y_cal = Y[cal_idx]
    else:
        cal_idx = None
        Y_cal = Y
        n_cal = n_spectra

    # --- Stage 1a: Dict2D alternating projection on calibration subset ---
    # Use finer Dict2D grids for Stage 1a to improve deflation accuracy.
    # Only the calibration subset is processed, so the extra grid cost is modest.
    cal_peaks = [
        ComponentConfig(
            center=p.center, sigma=p.sigma, gamma=p.gamma,
            dE_range=p.dE_range, ds_range=p.ds_range,
            n_dE=max(p.n_dE, 41),   # ≥41 points for sub-0.1 eV step
            n_ds=max(p.n_ds, 31),    # keep ds fine
        )
        for p in config.peaks
    ]
    cal_config = MultiPeakConfig(peaks=cal_peaks, energy_axis=config.energy_axis)
    if auto_constrain:
        cal_config.constrain_dE_ranges()

    t0 = time.perf_counter()
    dicts = build_multipeak_dictionaries(cal_config, grid_type)
    timing["stage1a_dict_build"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    result_cal = solve_alternating_projection(
        Y_cal, dicts, n_iterations, parabola_dE, parabola_ds,
    )
    timing["stage1a_alternating"] = time.perf_counter() - t0

    # --- Stage 1b: Per-component γ estimation via deflation + Dict3D ---
    # Deflate other components from each component's residual, then run
    # single-peak Dict3D to estimate per-component γ. The deflation uses
    # parabola-refined parameters from Stage 1a for accuracy.
    #
    # Known limitation: Deflation artifact introduces a systematic negative
    # γ bias of ~0.01 eV due to σ-γ coupling in the overlap region
    #. For σ/γ ≈ 3.4 (Gaussian-dominated regime),
    # this is near the Fisher information limit for per-component γ
    # estimation in overlapping peaks. The bias is consistent and can be
    # characterized for a given peak configuration.
    t0 = time.perf_counter()
    global_dgammas = np.zeros(n_comp, dtype=np.float32)
    dgamma_stds = np.zeros(n_comp, dtype=np.float32)
    dgamma_per_spectrum = []

    from .dictionary_solver_3d import get_or_build_dictionary_3d, solve_dict3d_parabola

    for k in range(n_comp):
        # Deflate: remove all other components using parabola-refined params
        Y_k = _deflate_component(
            Y_cal, config, result_cal.amplitudes,
            result_cal.delta_E, result_cal.delta_sigma, k,
        )

        # Build/cache Dict3D for this component (single-peak, n_comp=1)
        peak = config.peaks[k]
        dict3d_k = get_or_build_dictionary_3d(
            energy=np.asarray(config.energy_axis, dtype=np.float32),
            centers=np.array([peak.center], dtype=np.float32),
            sigmas=np.array([peak.sigma], dtype=np.float32),
            gamma=peak.gamma,
            dE_range=(-peak.dE_range, peak.dE_range),
            dsigma_range=(-peak.ds_range, peak.ds_range),
            dsigma_step=0.04,  # Fine grid for Fisher σ-γ separation
            dgamma_range=dgamma_range,
            dgamma_step=dgamma_step or 0.01,
        )

        # Solve Dict3D on deflated residual
        _, _, _, _, dg_k, _ = solve_dict3d_parabola(Y_k, dict3d_k)

        # Aggregate to per-component global Δγ
        if aggregation == "median":
            global_dgammas[k] = float(np.median(dg_k))
        else:
            global_dgammas[k] = float(np.mean(dg_k))
        dgamma_stds[k] = float(np.std(dg_k))
        dgamma_per_spectrum.append(dg_k.astype(np.float32))

    timing["stage1b_gamma_estimation"] = time.perf_counter() - t0

    # --- Stage 2: Rebuild with corrected γ, full solve ---
    corrected_gammas = np.array(
        [peak.gamma + global_dgammas[k] for k, peak in enumerate(config.peaks)],
        dtype=np.float32,
    )

    corrected_peaks = [
        ComponentConfig(
            center=peak.center,
            sigma=peak.sigma,
            gamma=float(corrected_gammas[k]),
            dE_range=peak.dE_range,
            ds_range=peak.ds_range,
            n_dE=peak.n_dE,
            n_ds=peak.n_ds,
        )
        for k, peak in enumerate(config.peaks)
    ]
    corrected_config = MultiPeakConfig(
        peaks=corrected_peaks,
        energy_axis=config.energy_axis,
    )
    if auto_constrain:
        corrected_config.constrain_dE_ranges()

    t0 = time.perf_counter()
    dicts_corrected = build_multipeak_dictionaries(corrected_config, grid_type)
    timing["stage2_dict_build"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    result_full = solve_multipeak_chunked(
        Y, dicts_corrected, chunk_size, n_iterations, parabola_dE, parabola_ds,
    )
    timing["stage2_solve"] = time.perf_counter() - t0

    timing["total"] = sum(v for v in timing.values() if isinstance(v, float))
    timing["rate"] = n_spectra / timing["total"] if timing["total"] > 0 else float("inf")

    return MultiPeak2StageResult(
        amplitudes=result_full.amplitudes,
        delta_E=result_full.delta_E,
        delta_sigma=result_full.delta_sigma,
        chi2=result_full.chi2,
        n_iterations=n_iterations,
        best_indices=result_full.best_indices,
        timing=timing,
        separation_warning=config.separation_sigma() < 3.5,
        corrected_gammas=corrected_gammas,
        global_dgammas=global_dgammas,
        dgamma_per_spectrum=dgamma_per_spectrum,
        dgamma_stds=dgamma_stds,
        n_calibration=n_cal,
    )
