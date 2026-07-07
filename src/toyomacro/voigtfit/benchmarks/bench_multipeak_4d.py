"""
4D MultiPeak Strategy Benchmark
============================================================

Compare 3 search strategies for 2-component (δE₁, δσ₁, δE₂, δσ₂):
  (A) Iterative residual: fit comp1 → subtract → fit comp2
  (B) Joint brute-force: full 4D search (small grid)
  (C) OMP (Orthogonal Matching Pursuit): greedy sparse selection

Also: Experiment 3.3 (4D parabola accuracy) and 3.4 (peak separation).

Usage:
    uv run python -m toyomacro.voigtfit.benchmarks.bench_multipeak_4d
"""

import time
from dataclasses import dataclass

import numpy as np

from toyomacro.voigtfit.dictionary_solver import (
    DictionaryCache,
    _parabola_vertex,
    build_dictionary,
)
from toyomacro.voigtfit.grids import uniform_grid

# ============================================================================
# Constants (C 1s two-component: sp3 + sp2)
# ============================================================================

FWHM_TO_SIGMA = 1.0 / (2 * np.sqrt(2 * np.log(2)))
SIGMA_NOM = 1.0 * FWHM_TO_SIGMA    # 0.4247 eV
GAMMA = 0.125                       # eV
CENTER_1 = 284.8                    # sp3
CENTER_2 = 284.3                    # sp2
ENERGY = np.linspace(280.0, 290.0, 101, dtype=np.float32)
N_ENERGY = len(ENERGY)


# ============================================================================
# 2-component spectra generation
# ============================================================================

def voigt_profile(energy: np.ndarray, center: float,
                  sigma: float, gamma: float) -> np.ndarray:
    """Single Voigt profile via Faddeeva function."""
    from scipy.special import wofz
    z = ((energy - center) + 1j * gamma) / (sigma * np.sqrt(2))
    profile = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    return profile


def generate_2comp_spectra(
    n_spectra: int,
    dE1_range: tuple[float, float] = (-0.5, 0.5),
    dE2_range: tuple[float, float] = (-0.5, 0.5),
    ds1_range: tuple[float, float] = (-0.1, 0.1),
    ds2_range: tuple[float, float] = (-0.1, 0.1),
    amp_ratios: np.ndarray | None = None,
    noise_sigma: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate 2-component Voigt spectra with known parameters.

    Returns:
        Y: (n_spectra, n_energy) spectra
        gt: dict with ground truth arrays
    """
    rng = np.random.default_rng(seed)

    dE1 = rng.uniform(*dE1_range, n_spectra).astype(np.float32)
    dE2 = rng.uniform(*dE2_range, n_spectra).astype(np.float32)
    ds1 = rng.uniform(*ds1_range, n_spectra).astype(np.float32)
    ds2 = rng.uniform(*ds2_range, n_spectra).astype(np.float32)

    if amp_ratios is not None:
        # Cycle through amp_ratios
        ratio_idx = np.arange(n_spectra) % len(amp_ratios)
        a1 = np.ones(n_spectra, dtype=np.float32)
        a2 = amp_ratios[ratio_idx].astype(np.float32)
    else:
        a1 = np.ones(n_spectra, dtype=np.float32)
        a2 = np.ones(n_spectra, dtype=np.float32)

    Y = np.zeros((n_spectra, N_ENERGY), dtype=np.float32)
    for i in range(n_spectra):
        c1 = CENTER_1 + dE1[i]
        s1 = SIGMA_NOM + ds1[i]
        c2 = CENTER_2 + dE2[i]
        s2 = SIGMA_NOM + ds2[i]
        p1 = voigt_profile(ENERGY, c1, max(s1, 0.01), GAMMA)
        p2 = voigt_profile(ENERGY, c2, max(s2, 0.01), GAMMA)
        p1 /= (p1.max() + 1e-30)
        p2 /= (p2.max() + 1e-30)
        Y[i] = a1[i] * p1 + a2[i] * p2

    if noise_sigma > 0:
        Y += rng.normal(0, noise_sigma, Y.shape).astype(np.float32)

    gt = dict(dE1=dE1, dE2=dE2, ds1=ds1, ds2=ds2, a1=a1, a2=a2)
    return Y, gt


# ============================================================================
# Strategy A: Iterative Residual
# ============================================================================

def solve_iterative(
    Y: np.ndarray,
    dc1: DictionaryCache,
    dc2: DictionaryCache,
) -> dict:
    """Fit component 1 → subtract → fit component 2.

    Uses full score matrix for comp1, then residual for comp2.
    Each component uses 2D parabola refinement.
    """
    import mlx.core as mx

    n_spectra = Y.shape[0]

    # --- Component 1 ---
    dc1.to_mlx()
    Y_mx = mx.array(Y)
    scores1 = Y_mx @ dc1._D_mx           # (n, n_dict1)
    best1 = mx.argmax(scores1, axis=1)
    mx.eval(best1)
    best1_np = np.array(best1).astype(np.int32)

    # Parabola refine comp1
    n_dE1, n_ds1 = dc1.grid_shape
    dE1_idx = best1_np // n_ds1
    ds1_idx = best1_np % n_ds1

    # Gather 5 neighbours
    spec_idx = mx.arange(n_spectra)
    dE1_m = np.clip(dE1_idx - 1, 0, n_dE1 - 1) * n_ds1 + ds1_idx
    dE1_p = np.clip(dE1_idx + 1, 0, n_dE1 - 1) * n_ds1 + ds1_idx
    ds1_m = dE1_idx * n_ds1 + np.clip(ds1_idx - 1, 0, n_ds1 - 1)
    ds1_p = dE1_idx * n_ds1 + np.clip(ds1_idx + 1, 0, n_ds1 - 1)

    s0 = np.array(scores1[spec_idx, mx.array(best1_np)])
    sm_dE = np.array(scores1[spec_idx, mx.array(dE1_m)])
    sp_dE = np.array(scores1[spec_idx, mx.array(dE1_p)])
    sm_ds = np.array(scores1[spec_idx, mx.array(ds1_m)])
    sp_ds = np.array(scores1[spec_idx, mx.array(ds1_p)])
    mx.eval(scores1)  # ensure computed

    dE1_step = dc1.dE_grid[1] - dc1.dE_grid[0] if n_dE1 > 1 else 1.0
    ds1_step = dc1.dsigma_grid[1] - dc1.dsigma_grid[0] if n_ds1 > 1 else 1.0
    dE1_int = (dE1_idx > 0) & (dE1_idx < n_dE1 - 1)
    ds1_int = (ds1_idx > 0) & (ds1_idx < n_ds1 - 1)

    delta_dE1 = _parabola_vertex(sm_dE, s0, sp_dE)
    delta_dE1 = np.where(dE1_int, delta_dE1, 0.0)
    dE1_out = (dc1.dE_grid[dE1_idx] + delta_dE1 * dE1_step).astype(np.float32)

    delta_ds1 = _parabola_vertex(sm_ds, s0, sp_ds)
    delta_ds1 = np.where(ds1_int, delta_ds1, 0.0)
    ds1_out = (dc1.dsigma_grid[ds1_idx] + delta_ds1 * ds1_step).astype(np.float32)

    # Amplitude for comp1 at best grid point
    Wt1 = dc1.Wt_per_grid[best1_np]       # (n, 1, n_E)
    a1_est = np.einsum('nce,ne->nc', Wt1, Y)[:, 0]  # (n,)
    # Reconstruct comp1
    Phi1 = dc1.Phi_per_grid[best1_np]      # (n, n_E, 1)
    recon1 = a1_est[:, None] * Phi1[:, :, 0]  # (n, n_E)

    # --- Component 2 on residual ---
    residual = Y - recon1
    dc2.to_mlx()
    res_mx = mx.array(residual)
    scores2 = res_mx @ dc2._D_mx
    best2 = mx.argmax(scores2, axis=1)
    mx.eval(best2)
    best2_np = np.array(best2).astype(np.int32)

    n_dE2, n_ds2 = dc2.grid_shape
    dE2_idx = best2_np // n_ds2
    ds2_idx = best2_np % n_ds2

    dE2_m = np.clip(dE2_idx - 1, 0, n_dE2 - 1) * n_ds2 + ds2_idx
    dE2_p = np.clip(dE2_idx + 1, 0, n_dE2 - 1) * n_ds2 + ds2_idx
    ds2_m = dE2_idx * n_ds2 + np.clip(ds2_idx - 1, 0, n_ds2 - 1)
    ds2_p = dE2_idx * n_ds2 + np.clip(ds2_idx + 1, 0, n_ds2 - 1)

    s0_2 = np.array(scores2[spec_idx, mx.array(best2_np)])
    sm_dE2 = np.array(scores2[spec_idx, mx.array(dE2_m)])
    sp_dE2 = np.array(scores2[spec_idx, mx.array(dE2_p)])
    sm_ds2 = np.array(scores2[spec_idx, mx.array(ds2_m)])
    sp_ds2 = np.array(scores2[spec_idx, mx.array(ds2_p)])

    dE2_step = dc2.dE_grid[1] - dc2.dE_grid[0] if n_dE2 > 1 else 1.0
    ds2_step = dc2.dsigma_grid[1] - dc2.dsigma_grid[0] if n_ds2 > 1 else 1.0
    dE2_int = (dE2_idx > 0) & (dE2_idx < n_dE2 - 1)
    ds2_int = (ds2_idx > 0) & (ds2_idx < n_ds2 - 1)

    delta_dE2 = _parabola_vertex(sm_dE2, s0_2, sp_dE2)
    delta_dE2 = np.where(dE2_int, delta_dE2, 0.0)
    dE2_out = (dc2.dE_grid[dE2_idx] + delta_dE2 * dE2_step).astype(np.float32)

    delta_ds2 = _parabola_vertex(sm_ds2, s0_2, sp_ds2)
    delta_ds2 = np.where(ds2_int, delta_ds2, 0.0)
    ds2_out = (dc2.dsigma_grid[ds2_idx] + delta_ds2 * ds2_step).astype(np.float32)

    # Amplitude for comp2
    Wt2 = dc2.Wt_per_grid[best2_np]
    a2_est = np.einsum('nce,ne->nc', Wt2, residual)[:, 0]

    del scores1, scores2

    return dict(dE1=dE1_out, ds1=ds1_out, a1=a1_est,
                dE2=dE2_out, ds2=ds2_out, a2=a2_est)


# ============================================================================
# Strategy B: Joint Brute-Force (small 4D grid)
# ============================================================================

def solve_joint_bruteforce(
    Y: np.ndarray,
    dc1: DictionaryCache,
    dc2: DictionaryCache,
    top_k: int = 5,
) -> dict:
    """Brute-force joint optimization over top-K candidates per component.

    For each spectrum:
    1. Compute scores for both components independently
    2. Take top-K grid points per component
    3. Try all K² combinations, pick the one with min residual
    4. Parabola refine each axis independently

    This is exact joint optimization over a small candidate set.
    """
    import mlx.core as mx

    n_spectra = Y.shape[0]

    dc1.to_mlx()
    dc2.to_mlx()
    Y_mx = mx.array(Y)

    # Scores for each component dictionary
    scores1 = np.array(Y_mx @ dc1._D_mx)   # (n, n_dict1)
    scores2 = np.array(Y_mx @ dc2._D_mx)   # (n, n_dict2)

    # Top-K indices per component
    topk1 = np.argsort(scores1, axis=1)[:, -top_k:]  # (n, K)
    topk2 = np.argsort(scores2, axis=1)[:, -top_k:]  # (n, K)

    # For each spectrum, try all K×K combinations
    best_i1 = np.zeros(n_spectra, dtype=np.int32)
    best_i2 = np.zeros(n_spectra, dtype=np.int32)
    best_a1 = np.zeros(n_spectra, dtype=np.float32)
    best_a2 = np.zeros(n_spectra, dtype=np.float32)

    for n in range(n_spectra):
        y = Y[n]  # (n_E,)
        min_chi2 = np.inf

        for ki in range(top_k):
            idx1 = topk1[n, ki]
            Phi1 = dc1.Phi_per_grid[idx1]  # (n_E, 1)

            for kj in range(top_k):
                idx2 = topk2[n, kj]
                Phi2 = dc2.Phi_per_grid[idx2]  # (n_E, 1)

                # Joint LLS: y ≈ a1 * phi1 + a2 * phi2
                Phi_joint = np.column_stack([Phi1, Phi2])  # (n_E, 2)
                amps, _, _, _ = np.linalg.lstsq(Phi_joint, y, rcond=None)
                residual = y - Phi_joint @ amps
                chi2 = np.dot(residual, residual)

                if chi2 < min_chi2:
                    min_chi2 = chi2
                    best_i1[n] = idx1
                    best_i2[n] = idx2
                    best_a1[n] = amps[0]
                    best_a2[n] = amps[1]

    # Parabola refine from best indices
    def _refine(best_idx, dc):
        n_dE, n_ds = dc.grid_shape
        dE_idx = best_idx // n_ds
        ds_idx = best_idx % n_ds
        dE_step = dc.dE_grid[1] - dc.dE_grid[0] if n_dE > 1 else 1.0
        ds_step = dc.dsigma_grid[1] - dc.dsigma_grid[0] if n_ds > 1 else 1.0

        # Score at neighbours (from original scores)
        spec_range = np.arange(n_spectra)

        dE_m = np.clip(dE_idx - 1, 0, n_dE - 1) * n_ds + ds_idx
        dE_p = np.clip(dE_idx + 1, 0, n_dE - 1) * n_ds + ds_idx
        ds_m = dE_idx * n_ds + np.clip(ds_idx - 1, 0, n_ds - 1)
        ds_p = dE_idx * n_ds + np.clip(ds_idx + 1, 0, n_ds - 1)

        if dc is dc1:
            s = scores1
        else:
            s = scores2

        s0 = s[spec_range, best_idx]
        delta_dE = _parabola_vertex(
            s[spec_range, dE_m], s0, s[spec_range, dE_p])
        dE_int = (dE_idx > 0) & (dE_idx < n_dE - 1)
        delta_dE = np.where(dE_int, delta_dE, 0.0)
        dE_out = (dc.dE_grid[dE_idx] + delta_dE * dE_step).astype(np.float32)

        delta_ds = _parabola_vertex(
            s[spec_range, ds_m], s0, s[spec_range, ds_p])
        ds_int = (ds_idx > 0) & (ds_idx < n_ds - 1)
        delta_ds = np.where(ds_int, delta_ds, 0.0)
        ds_out = (dc.dsigma_grid[ds_idx] + delta_ds * ds_step).astype(np.float32)

        return dE_out, ds_out

    dE1_out, ds1_out = _refine(best_i1, dc1)
    dE2_out, ds2_out = _refine(best_i2, dc2)

    return dict(dE1=dE1_out, ds1=ds1_out, a1=best_a1,
                dE2=dE2_out, ds2=ds2_out, a2=best_a2)


# ============================================================================
# Strategy C: OMP (Orthogonal Matching Pursuit)
# ============================================================================

def solve_omp(
    Y: np.ndarray,
    dc_combined: DictionaryCache,
    n_comp: int = 2,
) -> dict:
    """OMP: greedily select 2 dictionary atoms that best explain Y.

    Each atom represents a single-component Voigt at some (δE, δσ).
    OMP selects 2 atoms (one per component) and refines with LLS.

    Note: This uses a combined dictionary of BOTH components.
    """
    import mlx.core as mx

    n_spectra = Y.shape[0]
    n_dict = dc_combined.n_dict
    half = n_dict // 2  # First half = comp1, second half = comp2

    dc_combined.to_mlx()
    Y_mx = mx.array(Y)

    # Iteration 1: Find best single atom
    scores = np.array(Y_mx @ dc_combined._D_mx)  # (n, n_dict_total)

    best1 = np.argmax(np.abs(scores), axis=1).astype(np.int32)

    # Compute amplitude and residual for atom 1
    a1_est = np.zeros(n_spectra, dtype=np.float32)
    residual = np.copy(Y)
    for i in range(n_spectra):
        phi = dc_combined.Phi_per_grid[best1[i]]  # (n_E, 1)
        a1_est[i] = float(np.dot(Y[i], phi[:, 0]) / (np.dot(phi[:, 0], phi[:, 0]) + 1e-10))
        residual[i] = Y[i] - a1_est[i] * phi[:, 0]

    # Iteration 2: Find best atom for residual
    res_mx = mx.array(residual)
    scores2 = np.array(res_mx @ dc_combined._D_mx)
    best2 = np.argmax(np.abs(scores2), axis=1).astype(np.int32)

    # Joint LLS refinement
    a1_final = np.zeros(n_spectra, dtype=np.float32)
    a2_final = np.zeros(n_spectra, dtype=np.float32)
    for i in range(n_spectra):
        phi1 = dc_combined.Phi_per_grid[best1[i]][:, 0]
        phi2 = dc_combined.Phi_per_grid[best2[i]][:, 0]
        Phi = np.column_stack([phi1, phi2])
        amps, _, _, _ = np.linalg.lstsq(Phi, Y[i], rcond=None)
        a1_final[i] = amps[0]
        a2_final[i] = amps[1]

    # Separate comp1/comp2 based on which half of dictionary
    # This is a simplification — in practice need assignment logic
    # For now, assume atom from first half = comp1, second half = comp2

    n_dE, n_ds = dc_combined.grid_shape
    dE_step = dc_combined.dE_grid[1] - dc_combined.dE_grid[0] if n_dE > 1 else 1.0
    ds_step = dc_combined.dsigma_grid[1] - dc_combined.dsigma_grid[0] if n_ds > 1 else 1.0

    def _idx_to_params(idx):
        dE_idx = idx // n_ds
        ds_idx = idx % n_ds
        dE = dc_combined.dE_grid[np.clip(dE_idx, 0, n_dE - 1)]
        ds = dc_combined.dsigma_grid[np.clip(ds_idx, 0, n_ds - 1)]
        return dE.astype(np.float32), ds.astype(np.float32)

    dE1_out, ds1_out = _idx_to_params(best1)
    dE2_out, ds2_out = _idx_to_params(best2)

    return dict(dE1=dE1_out, ds1=ds1_out, a1=a1_final,
                dE2=dE2_out, ds2=ds2_out, a2=a2_final)


# ============================================================================
# Strategy D: Alternating Projection (deflation)
# ============================================================================

def solve_alternating(
    Y: np.ndarray,
    dc1: DictionaryCache,
    dc2: DictionaryCache,
    n_iter: int = 3,
) -> dict:
    """Alternating projection: fix one comp, search the other in deflated space.

    At each iteration:
    1. Given current comp2 estimate, project Y ⊥ φ₂, search comp1 in 2D
    2. Given current comp1 estimate, project Y ⊥ φ₁, search comp2 in 2D
    3. Joint LLS for amplitudes

    This properly handles cross-contamination by projecting out the
    other component before correlating with the dictionary.
    """
    import mlx.core as mx

    n_spectra = Y.shape[0]
    n_E = Y.shape[1]

    dc1.to_mlx()
    dc2.to_mlx()

    D1 = dc1.D                 # (n_E, n_dict1)
    D2 = dc2.D                 # (n_E, n_dict2)

    # Initialize comp2 at nominal
    best2_np = np.full(n_spectra, dc2.nominal_index, dtype=np.int32)

    for iteration in range(n_iter):
        # --- Step 1: Fix comp2, search comp1 ---
        # For each spectrum, project Y ⊥ current φ₂, then correlate with D₁
        Phi2_current = dc2.Phi_per_grid[best2_np]  # (n, n_E, 1)
        phi2 = Phi2_current[:, :, 0]                # (n, n_E)

        # Project out φ₂: Y_proj = Y - φ₂·(φ₂'·Y)/(φ₂'·φ₂)
        phi2_sq = np.sum(phi2 * phi2, axis=1, keepdims=True) + 1e-10  # (n, 1)
        phi2_Y = np.sum(phi2 * Y, axis=1, keepdims=True)              # (n, 1)
        Y_proj1 = Y - (phi2_Y / phi2_sq) * phi2                        # (n, n_E)

        # Also project dictionary columns (approximation: use mean φ₂)
        # For exact projection, would need per-spectrum projected dict,
        # but that's O(n × n_dict × n_E). Instead, correlate Y_proj with D₁.
        # This works because <Y_proj, D₁> ≈ <Y - a₂φ₂, D₁> which removes
        # the a₂ cross-contamination term.
        Y_proj1_mx = mx.array(Y_proj1.astype(np.float32))
        scores1 = Y_proj1_mx @ dc1._D_mx
        best1_mx = mx.argmax(scores1, axis=1)
        mx.eval(best1_mx)
        best1_np = np.array(best1_mx).astype(np.int32)

        # --- Step 2: Fix comp1, search comp2 ---
        Phi1_current = dc1.Phi_per_grid[best1_np]
        phi1 = Phi1_current[:, :, 0]

        phi1_sq = np.sum(phi1 * phi1, axis=1, keepdims=True) + 1e-10
        phi1_Y = np.sum(phi1 * Y, axis=1, keepdims=True)
        Y_proj2 = Y - (phi1_Y / phi1_sq) * phi1

        Y_proj2_mx = mx.array(Y_proj2.astype(np.float32))
        scores2 = Y_proj2_mx @ dc2._D_mx
        best2_mx = mx.argmax(scores2, axis=1)
        mx.eval(best2_mx)
        best2_np = np.array(best2_mx).astype(np.int32)

    # Final parabola refinement
    # Scores from last iteration (on projected Y)
    scores1_np = np.array(scores1)
    scores2_np = np.array(scores2)

    def _refine_from_scores(best_idx, scores, dc):
        n_dE, n_ds = dc.grid_shape
        dE_idx = best_idx // n_ds
        ds_idx = best_idx % n_ds
        dE_step = dc.dE_grid[1] - dc.dE_grid[0] if n_dE > 1 else 1.0
        ds_step = dc.dsigma_grid[1] - dc.dsigma_grid[0] if n_ds > 1 else 1.0
        spec_range = np.arange(n_spectra)

        dE_m = np.clip(dE_idx - 1, 0, n_dE - 1) * n_ds + ds_idx
        dE_p = np.clip(dE_idx + 1, 0, n_dE - 1) * n_ds + ds_idx
        ds_m = dE_idx * n_ds + np.clip(ds_idx - 1, 0, n_ds - 1)
        ds_p = dE_idx * n_ds + np.clip(ds_idx + 1, 0, n_ds - 1)

        s0 = scores[spec_range, best_idx]
        dE_int = (dE_idx > 0) & (dE_idx < n_dE - 1)
        ds_int = (ds_idx > 0) & (ds_idx < n_ds - 1)

        delta_dE = _parabola_vertex(
            scores[spec_range, dE_m], s0, scores[spec_range, dE_p])
        delta_dE = np.where(dE_int, delta_dE, 0.0)
        dE_out = (dc.dE_grid[dE_idx] + delta_dE * dE_step).astype(np.float32)

        delta_ds = _parabola_vertex(
            scores[spec_range, ds_m], s0, scores[spec_range, ds_p])
        delta_ds = np.where(ds_int, delta_ds, 0.0)
        ds_out = (dc.dsigma_grid[ds_idx] + delta_ds * ds_step).astype(np.float32)

        return dE_out, ds_out

    dE1_out, ds1_out = _refine_from_scores(best1_np, scores1_np, dc1)
    dE2_out, ds2_out = _refine_from_scores(best2_np, scores2_np, dc2)

    # Joint LLS for amplitudes
    a1_out = np.zeros(n_spectra, dtype=np.float32)
    a2_out = np.zeros(n_spectra, dtype=np.float32)
    for i in range(n_spectra):
        phi1 = dc1.Phi_per_grid[best1_np[i]][:, 0]
        phi2 = dc2.Phi_per_grid[best2_np[i]][:, 0]
        Phi = np.column_stack([phi1, phi2])
        amps, _, _, _ = np.linalg.lstsq(Phi, Y[i], rcond=None)
        a1_out[i] = amps[0]
        a2_out[i] = amps[1]

    return dict(dE1=dE1_out, ds1=ds1_out, a1=a1_out,
                dE2=dE2_out, ds2=ds2_out, a2=a2_out)


# ============================================================================
# Metrics
# ============================================================================

@dataclass
class MultiPeakResult:
    name: str
    dE1_rmse: float
    dE2_rmse: float
    ds1_rmse: float
    ds2_rmse: float
    a1_rmse: float
    a2_rmse: float
    time_s: float
    n_spectra: int

    @property
    def throughput(self) -> float:
        return self.n_spectra / self.time_s if self.time_s > 0 else 0

    def summary(self) -> str:
        return (f"{self.name:<25}: "
                f"δE₁={self.dE1_rmse:.5f} δE₂={self.dE2_rmse:.5f} "
                f"δσ₁={self.ds1_rmse:.5f} δσ₂={self.ds2_rmse:.5f} "
                f"a₁={self.a1_rmse:.4f} a₂={self.a2_rmse:.4f} "
                f"  {self.time_s:.3f}s ({self.throughput/1e3:.1f}K/s)")


def evaluate(result: dict, gt: dict, name: str,
             elapsed: float, n_spectra: int) -> MultiPeakResult:
    """Compute RMSE metrics comparing result to ground truth."""
    return MultiPeakResult(
        name=name,
        dE1_rmse=float(np.sqrt(np.mean((result['dE1'] - gt['dE1'])**2))),
        dE2_rmse=float(np.sqrt(np.mean((result['dE2'] - gt['dE2'])**2))),
        ds1_rmse=float(np.sqrt(np.mean((result['ds1'] - gt['ds1'])**2))),
        ds2_rmse=float(np.sqrt(np.mean((result['ds2'] - gt['ds2'])**2))),
        a1_rmse=float(np.sqrt(np.mean((result['a1'] - gt['a1'])**2))),
        a2_rmse=float(np.sqrt(np.mean((result['a2'] - gt['a2'])**2))),
        time_s=elapsed,
        n_spectra=n_spectra,
    )


# ============================================================================
# Experiments
# ============================================================================

def experiment_3_1_3_2():
    """Experiment 3.1+3.2: 2-component test data + strategy comparison."""
    print("=" * 80)
    print("Experiment 3.1+3.2: 2-Component Strategy Comparison")
    print("=" * 80)

    n_spectra = 500  # Small for joint brute-force (O(n * K²))
    n_dE, n_ds = 15, 31  # Grids per component (Part 2 design: δE coarse, δσ dense)

    # Generate test data
    amp_ratios = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    Y, gt = generate_2comp_spectra(
        n_spectra, amp_ratios=amp_ratios,
        dE1_range=(-0.5, 0.5), dE2_range=(-0.5, 0.5),
        ds1_range=(-0.1, 0.1), ds2_range=(-0.1, 0.1),
        noise_sigma=0.0, seed=42,
    )
    print(f"Test: {n_spectra} spectra, 2-comp (sp3 @ {CENTER_1}, sp2 @ {CENTER_2})")
    print("Ranges: δE ∈ [-0.5, 0.5], δσ ∈ [-0.1, 0.1]")
    print(f"Amp ratios: {amp_ratios}")

    # Build per-component dictionaries
    dE_grid = uniform_grid(n_dE, 1.0).astype(np.float32)
    ds_grid = uniform_grid(n_ds, 0.3).astype(np.float32)

    centers1 = np.array([CENTER_1], dtype=np.float32)
    centers2 = np.array([CENTER_2], dtype=np.float32)
    sigmas = np.array([SIGMA_NOM], dtype=np.float32)

    print(f"Dictionary: {n_dE}×{n_ds}={n_dE*n_ds} entries per component")

    dc1 = build_dictionary(ENERGY, centers1, sigmas, GAMMA,
                           dE_grid_override=dE_grid,
                           dsigma_grid_override=ds_grid)
    dc2 = build_dictionary(ENERGY, centers2, sigmas, GAMMA,
                           dE_grid_override=dE_grid,
                           dsigma_grid_override=ds_grid)

    results: list[MultiPeakResult] = []

    # Strategy A: Iterative
    print("\n  [A] Iterative residual...")
    t0 = time.perf_counter()
    res_a = solve_iterative(Y, dc1, dc2)
    t_a = time.perf_counter() - t0
    mr_a = evaluate(res_a, gt, "Iterative", t_a, n_spectra)
    results.append(mr_a)
    print(f"    {mr_a.summary()}")

    # Strategy B: Joint brute-force (top-5)
    for top_k in [3, 5, 10]:
        print(f"\n  [B] Joint brute-force (top-{top_k})...")
        t0 = time.perf_counter()
        res_b = solve_joint_bruteforce(Y, dc1, dc2, top_k=top_k)
        t_b = time.perf_counter() - t0
        mr_b = evaluate(res_b, gt, f"Joint top-{top_k}", t_b, n_spectra)
        results.append(mr_b)
        print(f"    {mr_b.summary()}")

    # Strategy C: OMP (simplified — uses dc1 for both)
    print("\n  [C] OMP (2-iteration)...")
    t0 = time.perf_counter()
    res_c = solve_omp(Y, dc1, n_comp=2)
    t_c = time.perf_counter() - t0
    mr_c = evaluate(res_c, gt, "OMP", t_c, n_spectra)
    results.append(mr_c)
    print(f"    {mr_c.summary()}")

    # Strategy D: Alternating projection
    for n_it in [1, 3, 5]:
        print(f"\n  [D] Alternating projection ({n_it} iter)...")
        t0 = time.perf_counter()
        res_d = solve_alternating(Y, dc1, dc2, n_iter=n_it)
        t_d = time.perf_counter() - t0
        mr_d = evaluate(res_d, gt, f"Alternating-{n_it}", t_d, n_spectra)
        results.append(mr_d)
        print(f"    {mr_d.summary()}")

    # Summary table
    print("\n" + "-" * 80)
    print(f"{'Strategy':<25} {'δE₁':>8} {'δE₂':>8} {'δσ₁':>8} {'δσ₂':>8} "
          f"{'time':>8} {'K/s':>8}")
    print("-" * 80)
    for mr in results:
        print(f"{mr.name:<25} {mr.dE1_rmse:8.5f} {mr.dE2_rmse:8.5f} "
              f"{mr.ds1_rmse:8.5f} {mr.ds2_rmse:8.5f} "
              f"{mr.time_s:8.3f} {mr.throughput/1e3:8.1f}")

    return results


def experiment_3_4():
    """Experiment 3.4: Peak separation vs accuracy — Iterative vs Alternating."""
    print("\n" + "=" * 80)
    print("Experiment 3.4: Peak Separation vs Accuracy (Iterative vs Alternating)")
    print("=" * 80)

    n_spectra = 200
    n_dE, n_ds = 15, 31

    dE_grid = uniform_grid(n_dE, 1.0).astype(np.float32)
    ds_grid = uniform_grid(n_ds, 0.3).astype(np.float32)
    sigmas = np.array([SIGMA_NOM], dtype=np.float32)

    separations = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    print(f"σ = {SIGMA_NOM:.4f} eV, so separations in σ units:")
    for sep in separations:
        print(f"  Δc = {sep:.1f} eV = {sep/SIGMA_NOM:.2f}σ")

    results_iter = []
    results_alt = []

    for sep in separations:
        c1 = 284.5 + sep / 2
        c2 = 284.5 - sep / 2
        centers1 = np.array([c1], dtype=np.float32)
        centers2 = np.array([c2], dtype=np.float32)

        dc1 = build_dictionary(ENERGY, centers1, sigmas, GAMMA,
                               dE_grid_override=dE_grid,
                               dsigma_grid_override=ds_grid)
        dc2 = build_dictionary(ENERGY, centers2, sigmas, GAMMA,
                               dE_grid_override=dE_grid,
                               dsigma_grid_override=ds_grid)

        # Generate with equal amplitudes, small jitter
        rng = np.random.default_rng(42)
        dE1 = rng.uniform(-0.3, 0.3, n_spectra).astype(np.float32)
        dE2 = rng.uniform(-0.3, 0.3, n_spectra).astype(np.float32)
        ds1 = rng.uniform(-0.08, 0.08, n_spectra).astype(np.float32)
        ds2 = rng.uniform(-0.08, 0.08, n_spectra).astype(np.float32)
        a1 = np.ones(n_spectra, dtype=np.float32)
        a2 = np.ones(n_spectra, dtype=np.float32)

        Y = np.zeros((n_spectra, N_ENERGY), dtype=np.float32)
        for i in range(n_spectra):
            p1 = voigt_profile(ENERGY, c1 + dE1[i], SIGMA_NOM + ds1[i], GAMMA)
            p2 = voigt_profile(ENERGY, c2 + dE2[i], SIGMA_NOM + ds2[i], GAMMA)
            p1 /= (p1.max() + 1e-30)
            p2 /= (p2.max() + 1e-30)
            Y[i] = p1 + p2

        gt_local = dict(dE1=dE1, dE2=dE2, ds1=ds1, ds2=ds2, a1=a1, a2=a2)

        # Iterative
        t0 = time.perf_counter()
        res_i = solve_iterative(Y, dc1, dc2)
        t_i = time.perf_counter() - t0
        mr_i = evaluate(res_i, gt_local, f"Iter Δc={sep:.1f}", t_i, n_spectra)
        results_iter.append((sep, mr_i))

        # Alternating (3 iter)
        t0 = time.perf_counter()
        res_a = solve_alternating(Y, dc1, dc2, n_iter=3)
        t_a = time.perf_counter() - t0
        mr_a = evaluate(res_a, gt_local, f"Alt  Δc={sep:.1f}", t_a, n_spectra)
        results_alt.append((sep, mr_a))

    # Summary: Iterative
    print("\n--- Iterative ---")
    print(f"{'Δc (eV)':<10} {'Δc/σ':<8} {'δE₁':>8} {'δE₂':>8} "
          f"{'δσ₁':>8} {'δσ₂':>8} {'a₁':>8} {'a₂':>8}")
    print("-" * 80)
    for sep, mr in results_iter:
        print(f"{sep:<10.1f} {sep/SIGMA_NOM:<8.2f} "
              f"{mr.dE1_rmse:8.5f} {mr.dE2_rmse:8.5f} "
              f"{mr.ds1_rmse:8.5f} {mr.ds2_rmse:8.5f} "
              f"{mr.a1_rmse:8.4f} {mr.a2_rmse:8.4f}")

    # Summary: Alternating
    print("\n--- Alternating (3 iter) ---")
    print(f"{'Δc (eV)':<10} {'Δc/σ':<8} {'δE₁':>8} {'δE₂':>8} "
          f"{'δσ₁':>8} {'δσ₂':>8} {'a₁':>8} {'a₂':>8}")
    print("-" * 80)
    for sep, mr in results_alt:
        print(f"{sep:<10.1f} {sep/SIGMA_NOM:<8.2f} "
              f"{mr.dE1_rmse:8.5f} {mr.dE2_rmse:8.5f} "
              f"{mr.ds1_rmse:8.5f} {mr.ds2_rmse:8.5f} "
              f"{mr.a1_rmse:8.4f} {mr.a2_rmse:8.4f}")

    # Comparison: δE mean RMSE
    print("\n--- δE Mean RMSE Comparison ---")
    print(f"{'Δc (eV)':<10} {'Δc/σ':<8} {'Iter δE':>10} {'Alt δE':>10} {'Winner':>10}")
    print("-" * 50)
    for (sep, mr_i), (_, mr_a) in zip(results_iter, results_alt):
        mean_i = (mr_i.dE1_rmse + mr_i.dE2_rmse) / 2
        mean_a = (mr_a.dE1_rmse + mr_a.dE2_rmse) / 2
        winner = "Alt" if mean_a < mean_i else "Iter"
        print(f"{sep:<10.1f} {sep/SIGMA_NOM:<8.2f} {mean_i:10.5f} {mean_a:10.5f} {winner:>10}")

    return results_iter, results_alt


def solve_alternating_no_parabola(
    Y: np.ndarray,
    dc1: DictionaryCache,
    dc2: DictionaryCache,
    n_iter: int = 3,
) -> dict:
    """Alternating projection WITHOUT parabola refinement (argmax only)."""
    import mlx.core as mx

    n_spectra = Y.shape[0]
    dc1.to_mlx()
    dc2.to_mlx()

    best2_np = np.full(n_spectra, dc2.nominal_index, dtype=np.int32)

    for iteration in range(n_iter):
        Phi2_current = dc2.Phi_per_grid[best2_np]
        phi2 = Phi2_current[:, :, 0]
        phi2_sq = np.sum(phi2 * phi2, axis=1, keepdims=True) + 1e-10
        phi2_Y = np.sum(phi2 * Y, axis=1, keepdims=True)
        Y_proj1 = Y - (phi2_Y / phi2_sq) * phi2

        Y_proj1_mx = mx.array(Y_proj1.astype(np.float32))
        scores1 = Y_proj1_mx @ dc1._D_mx
        best1_mx = mx.argmax(scores1, axis=1)
        mx.eval(best1_mx)
        best1_np = np.array(best1_mx).astype(np.int32)

        Phi1_current = dc1.Phi_per_grid[best1_np]
        phi1 = Phi1_current[:, :, 0]
        phi1_sq = np.sum(phi1 * phi1, axis=1, keepdims=True) + 1e-10
        phi1_Y = np.sum(phi1 * Y, axis=1, keepdims=True)
        Y_proj2 = Y - (phi1_Y / phi1_sq) * phi1

        Y_proj2_mx = mx.array(Y_proj2.astype(np.float32))
        scores2 = Y_proj2_mx @ dc2._D_mx
        best2_mx = mx.argmax(scores2, axis=1)
        mx.eval(best2_mx)
        best2_np = np.array(best2_mx).astype(np.int32)

    # Grid-point values only (no parabola)
    n_dE1, n_ds1 = dc1.grid_shape
    n_dE2, n_ds2 = dc2.grid_shape
    dE1_idx = best1_np // n_ds1
    ds1_idx = best1_np % n_ds1
    dE2_idx = best2_np // n_ds2
    ds2_idx = best2_np % n_ds2

    dE1_out = dc1.dE_grid[dE1_idx].astype(np.float32)
    ds1_out = dc1.dsigma_grid[ds1_idx].astype(np.float32)
    dE2_out = dc2.dE_grid[dE2_idx].astype(np.float32)
    ds2_out = dc2.dsigma_grid[ds2_idx].astype(np.float32)

    # Joint LLS
    a1_out = np.zeros(n_spectra, dtype=np.float32)
    a2_out = np.zeros(n_spectra, dtype=np.float32)
    for i in range(n_spectra):
        phi1 = dc1.Phi_per_grid[best1_np[i]][:, 0]
        phi2 = dc2.Phi_per_grid[best2_np[i]][:, 0]
        Phi = np.column_stack([phi1, phi2])
        amps, _, _, _ = np.linalg.lstsq(Phi, Y[i], rcond=None)
        a1_out[i] = amps[0]
        a2_out[i] = amps[1]

    return dict(dE1=dE1_out, ds1=ds1_out, a1=a1_out,
                dE2=dE2_out, ds2=ds2_out, a2=a2_out)


def experiment_3_3():
    """Experiment 3.3: Parabola refinement value in alternating context."""
    print("\n" + "=" * 80)
    print("Experiment 3.3: Parabola Refinement Value (Alternating Projection)")
    print("=" * 80)

    n_spectra = 300
    n_dE, n_ds = 15, 31
    dE_grid = uniform_grid(n_dE, 1.0).astype(np.float32)
    ds_grid = uniform_grid(n_ds, 0.3).astype(np.float32)
    sigmas = np.array([SIGMA_NOM], dtype=np.float32)

    separations = [1.0, 2.0, 3.0, 5.0]

    print(f"Grid: {n_dE}×{n_ds}={n_dE*n_ds} entries per component")
    print(f"δE step = {2.0/(n_dE-1):.4f} eV, δσ step = {0.6/(n_ds-1):.4f} eV")

    for sep in separations:
        c1 = 284.5 + sep / 2
        c2 = 284.5 - sep / 2
        centers1 = np.array([c1], dtype=np.float32)
        centers2 = np.array([c2], dtype=np.float32)

        dc1 = build_dictionary(ENERGY, centers1, sigmas, GAMMA,
                               dE_grid_override=dE_grid,
                               dsigma_grid_override=ds_grid)
        dc2 = build_dictionary(ENERGY, centers2, sigmas, GAMMA,
                               dE_grid_override=dE_grid,
                               dsigma_grid_override=ds_grid)

        rng = np.random.default_rng(42)
        dE1 = rng.uniform(-0.3, 0.3, n_spectra).astype(np.float32)
        dE2 = rng.uniform(-0.3, 0.3, n_spectra).astype(np.float32)
        ds1 = rng.uniform(-0.08, 0.08, n_spectra).astype(np.float32)
        ds2 = rng.uniform(-0.08, 0.08, n_spectra).astype(np.float32)

        Y = np.zeros((n_spectra, N_ENERGY), dtype=np.float32)
        for i in range(n_spectra):
            p1 = voigt_profile(ENERGY, c1 + dE1[i], SIGMA_NOM + ds1[i], GAMMA)
            p2 = voigt_profile(ENERGY, c2 + dE2[i], SIGMA_NOM + ds2[i], GAMMA)
            p1 /= (p1.max() + 1e-30)
            p2 /= (p2.max() + 1e-30)
            Y[i] = p1 + p2

        gt_local = dict(dE1=dE1, dE2=dE2, ds1=ds1, ds2=ds2,
                        a1=np.ones(n_spectra, np.float32),
                        a2=np.ones(n_spectra, np.float32))

        # No parabola
        res_np = solve_alternating_no_parabola(Y, dc1, dc2, n_iter=3)
        mr_np = evaluate(res_np, gt_local, "No parabola", 0.0, n_spectra)

        # With parabola
        res_p = solve_alternating(Y, dc1, dc2, n_iter=3)
        mr_p = evaluate(res_p, gt_local, "With parabola", 0.0, n_spectra)

        ratio_dE = (mr_np.dE1_rmse + mr_np.dE2_rmse) / (mr_p.dE1_rmse + mr_p.dE2_rmse + 1e-30)
        ratio_ds = (mr_np.ds1_rmse + mr_np.ds2_rmse) / (mr_p.ds1_rmse + mr_p.ds2_rmse + 1e-30)

        print(f"\n  Δc = {sep:.1f} eV ({sep/SIGMA_NOM:.1f}σ):")
        print(f"    No parabola : δE₁={mr_np.dE1_rmse:.5f} δE₂={mr_np.dE2_rmse:.5f} "
              f"δσ₁={mr_np.ds1_rmse:.5f} δσ₂={mr_np.ds2_rmse:.5f}")
        print(f"    With parabola: δE₁={mr_p.dE1_rmse:.5f} δE₂={mr_p.dE2_rmse:.5f} "
              f"δσ₁={mr_p.ds1_rmse:.5f} δσ₂={mr_p.ds2_rmse:.5f}")
        print(f"    Parabola gain: δE {ratio_dE:.1f}x, δσ {ratio_ds:.1f}x")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("4D MultiPeak Strategy Benchmark")
    print(f"sp3={CENTER_1}, sp2={CENTER_2}, σ={SIGMA_NOM:.4f}, γ={GAMMA}")
    print()

    r12 = experiment_3_1_3_2()
    r4 = experiment_3_4()
    experiment_3_3()

    print("\n" + "=" * 80)
    print("All experiments complete.")
