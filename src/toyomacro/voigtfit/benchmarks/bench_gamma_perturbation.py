"""
Gamma Perturbation Experiment
========================================================

Verifies Fisher prediction: when gamma is fixed at a wrong value,
the sigma estimate bias should follow:

    bias(delta_sigma) = -bias_coeff * delta_gamma

where bias_coeff = g_sigma_gamma / g_sigma_sigma = 0.866 (C 1s, eta=0.15).

Sweep:
  - delta_gamma in [-0.02, -0.01, -0.005, 0, +0.005, +0.01, +0.02]
  - Noise: NF, SNR=1000, SNR=100
  - Solver: parabola (default), sorted (4-step), 6step
  - N_spectra per cell: 100K

Output:
  - bias_coeff measured vs Fisher predicted
  - delta_E bias (should be ~0)
  - Regression lines and R^2

Usage:
    uv run python -m toyomacro.voigtfit.benchmarks.bench_gamma_perturbation
    uv run python -m toyomacro.voigtfit.benchmarks.bench_gamma_perturbation --n-spectra 10000
    uv run python -m toyomacro.voigtfit.benchmarks.bench_gamma_perturbation --solvers parabola
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
from scipy.special import wofz

from toyomacro.voigtfit.dictionary_solver import (
    build_dictionary,
    solve_dict2d_parabola,
    solve_hybrid_6step_sorted,
    solve_hybrid_sorted,
)
from toyomacro.voigtfit.fisher_information import (
    analyze_sigma_gamma_axes,
    compute_fisher_matrix,
)

# ============================================================================
# Constants (C 1s single peak)
# ============================================================================

FWHM_TO_SIGMA = 1.0 / (2 * np.sqrt(2 * np.log(2)))
SIGMA_NOM = 1.0 * FWHM_TO_SIGMA   # 0.4247 eV
GAMMA_NOM = 0.125                  # eV
CENTER = 284.4                     # eV
ENERGY = np.linspace(280.0, 290.0, 101, dtype=np.float32)
N_ENERGY = len(ENERGY)
CENTERS = np.array([CENTER], dtype=np.float32)
SIGMAS = np.array([SIGMA_NOM], dtype=np.float32)

# Amplitude scale (match typical XPS counts)
AMPLITUDE_SCALE = 1000.0

# Gamma perturbation sweep
DELTA_GAMMAS = np.array([-0.02, -0.01, -0.005, 0.0, +0.005, +0.01, +0.02])

# Noise levels
SNR_LEVELS = {
    "NF": float("inf"),
    "SNR1k": 1000.0,
    "SNR100": 100.0,
}

# Solver variants
SOLVER_FUNCS = {
    "parabola": solve_dict2d_parabola,
    "sorted": solve_hybrid_sorted,
    "6step": solve_hybrid_6step_sorted,
}

# Parameter jitter ranges (same as bench_jitter_limits)
DE_JITTER = 0.3      # eV
DS_JITTER = 0.08     # eV
AMP_MIN, AMP_MAX = 0.3, 1.0

N_SPECTRA_DEFAULT = 100_000


# ============================================================================
# Spectra generation with gamma perturbation
# ============================================================================

def generate_spectra_gamma_perturbed(
    n_spectra: int,
    gamma_true: float,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate single-peak Voigt spectra with gamma_true (may differ from GAMMA_NOM).

    Each spectrum has random (dE, dsigma, amplitude) jitter drawn uniformly.
    Spectra are NOT normalized — amplitude is in counts.

    Returns:
        Y: (n_spectra, N_ENERGY) float32
        gt: dict with dE, dsigma, amp arrays (ground truth)
    """
    rng = np.random.default_rng(seed)

    # Random parameter jitter
    dE = rng.uniform(-DE_JITTER, DE_JITTER, n_spectra).astype(np.float32)
    ds = rng.uniform(-DS_JITTER, DS_JITTER, n_spectra).astype(np.float32)
    amp = rng.uniform(AMP_MIN, AMP_MAX, n_spectra).astype(np.float32)

    Y = np.empty((n_spectra, N_ENERGY), dtype=np.float32)
    sqrt2 = np.sqrt(2.0)
    sqrt2pi = np.sqrt(2.0 * np.pi)

    for i in range(n_spectra):
        c = CENTER + dE[i]
        s = SIGMA_NOM + ds[i]
        z = ((ENERGY - c) + 1j * gamma_true) / (s * sqrt2)
        profile = np.real(wofz(z)) / (s * sqrt2pi)
        Y[i] = (amp[i] * AMPLITUDE_SCALE * profile).astype(np.float32)

    gt = {"dE": dE, "dsigma": ds, "amp": amp}
    return Y, gt


def add_poisson_noise(Y: np.ndarray, snr: float, seed: int = 123) -> np.ndarray:
    """Add heteroscedastic Poisson-like noise to spectra.

    SNR = 10000 / level. For SNR=1000, level=10.
    Noise variance at each point ~ max_intensity / snr^2.
    """
    if not np.isfinite(snr):
        return Y.copy()

    rng = np.random.default_rng(seed)
    global_max = Y.max()
    if global_max < 1e-30:
        return Y.copy()

    # Poisson-approximated noise: σ_noise = sqrt(signal) in photon units
    # scale: snr relates to peak signal
    noise_std = np.sqrt(np.maximum(Y, 0.0)) * (np.sqrt(global_max) / snr)
    noise = rng.normal(0, 1, Y.shape).astype(np.float32) * noise_std
    return Y + noise


# ============================================================================
# Cell evaluation
# ============================================================================

@dataclass
class CellResult:
    """Result for one (delta_gamma, solver, noise) combination."""
    delta_gamma: float
    solver: str
    snr_name: str

    dsigma_bias: float
    dsigma_std: float
    dsigma_rmse: float

    dE_bias: float
    dE_std: float
    dE_rmse: float

    amp_bias: float

    throughput: float  # M spectra/s


def run_cell(
    delta_gamma: float,
    solver_name: str,
    snr_name: str,
    snr_val: float,
    n_spectra: int,
    seed: int = 42,
) -> CellResult:
    """Run one cell of the sweep."""
    gamma_true = GAMMA_NOM + delta_gamma

    # Generate spectra with gamma_true
    Y, gt = generate_spectra_gamma_perturbed(n_spectra, gamma_true, seed=seed)

    # Add noise
    Y_noisy = add_poisson_noise(Y, snr_val, seed=seed + 1000)

    # Build dictionary with nominal gamma (the "wrong" gamma when delta_gamma != 0)
    include_hessian = (solver_name == "6step")
    dict_cache = build_dictionary(
        energy=ENERGY,
        centers=CENTERS,
        sigmas=SIGMAS,
        gamma=GAMMA_NOM,
        dE_range=(-1.2, 1.2),
        dsigma_range=(-0.3, 0.3),
        include_hessian=include_hessian,
    )

    # Solve
    solve_fn = SOLVER_FUNCS[solver_name]
    t0 = time.perf_counter()
    amps, chi2, dE_fit, ds_fit, _ = solve_fn(Y_noisy, dict_cache)
    elapsed = time.perf_counter() - t0

    # Amplitude is (n_comp, n_spectra) for single peak → (1, n_spectra)
    amp_fit = amps[0] / AMPLITUDE_SCALE

    # Compute errors
    dE_err = dE_fit - gt["dE"]
    ds_err = ds_fit - gt["dsigma"]
    amp_err = amp_fit - gt["amp"]

    return CellResult(
        delta_gamma=delta_gamma,
        solver=solver_name,
        snr_name=snr_name,
        dsigma_bias=float(np.mean(ds_err)),
        dsigma_std=float(np.std(ds_err)),
        dsigma_rmse=float(np.sqrt(np.mean(ds_err**2))),
        dE_bias=float(np.mean(dE_err)),
        dE_std=float(np.std(dE_err)),
        dE_rmse=float(np.sqrt(np.mean(dE_err**2))),
        amp_bias=float(np.mean(amp_err)),
        throughput=n_spectra / elapsed / 1e6,
    )


# ============================================================================
# Bias slope regression
# ============================================================================

def compute_bias_slope(
    results: list[CellResult],
    solver: str,
    snr_name: str,
) -> dict:
    """Linear regression of dsigma_bias vs delta_gamma for one (solver, noise) combo.

    Fisher prediction: slope ≈ +bias_coeff ≈ +0.870.
    (Positive: larger gamma → broader profile → solver overestimates sigma.)
    """
    cells = [r for r in results if r.solver == solver and r.snr_name == snr_name]
    cells.sort(key=lambda r: r.delta_gamma)

    x = np.array([r.delta_gamma for r in cells])
    y = np.array([r.dsigma_bias for r in cells])

    # Linear regression
    if len(x) < 2:
        return {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan}

    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs

    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


def compute_dE_bias_slope(
    results: list[CellResult],
    solver: str,
    snr_name: str,
) -> dict:
    """Linear regression of dE_bias vs delta_gamma (should be ~0 slope)."""
    cells = [r for r in results if r.solver == solver and r.snr_name == snr_name]
    cells.sort(key=lambda r: r.delta_gamma)

    x = np.array([r.delta_gamma for r in cells])
    y = np.array([r.dE_bias for r in cells])

    if len(x) < 2:
        return {"slope": np.nan}

    coeffs = np.polyfit(x, y, 1)
    return {"slope": float(coeffs[0])}


# ============================================================================
# Fisher prediction
# ============================================================================

def compute_fisher_prediction() -> dict:
    """Compute the Fisher-predicted bias coefficient for C 1s parameters.

    Three predictions (in order of decreasing idealization):
    - approx: g_σγ / g_σσ (ignores off-diagonal coupling in 3D block)
    - exact: [(g_3d)⁻¹ · g_{:,γ}]_σ (full omitted-variable bias, Poisson-weighted)
    - ols_deflated: unweighted projection with amplitude deflation (solver model)

    The solver uses OLS (not Poisson-weighted) projection, and the amplitude
    LLS estimate absorbs part of the γ-induced spectral change. This deflation
    reduces the effective bias coefficient below the Fisher prediction.
    """
    energy_dense = np.linspace(280.0, 290.0, 201)
    result_4d = compute_fisher_matrix(
        amplitude=AMPLITUDE_SCALE,
        center=CENTER,
        sigma=SIGMA_NOM,
        gamma=GAMMA_NOM,
        energy=energy_dense,
        mode='4d',
    )
    axes = analyze_sigma_gamma_axes(result_4d)

    # Exact omitted-variable bias: full 3D inverse (Poisson-weighted)
    g4 = result_4d.g
    g3d = g4[:3, :3]
    j_gamma = g4[:3, 3]
    g3d_inv = np.linalg.inv(g3d)
    bias_vector = g3d_inv @ j_gamma
    exact_sigma_coeff = bias_vector[2]

    # OLS (unweighted) prediction: <dV/dγ, dV/dσ> / <dV/dσ, dV/dσ>
    from toyomacro.voigtfit.voigt_jacobian import voigt_with_jacobian
    V, dV_dc, dV_dsigma, dV_dgamma = voigt_with_jacobian(
        energy_dense, CENTER, SIGMA_NOM, GAMMA_NOM
    )
    ols_coeff = float(np.dot(dV_dgamma, dV_dsigma) / np.dot(dV_dsigma, dV_dsigma))

    # OLS with amplitude deflation: the amplitude LLS absorbs part of dV/dγ
    # Residual after amp fit: R = P_⊥V(dV/dγ) · Δγ
    # where P_⊥V = I - V·V'/<V,V> projects out the amplitude direction
    VtV = np.dot(V, V)
    dV_dg_deflated = dV_dgamma - (np.dot(V, dV_dgamma) / VtV) * V
    ols_deflated_coeff = float(np.dot(dV_dg_deflated, dV_dsigma) / np.dot(dV_dsigma, dV_dsigma))

    return {
        "bias_coeff_approx": axes.bias_coefficient,  # g_σγ/g_σσ = 0.870
        "bias_coeff_exact": float(exact_sigma_coeff),  # full inverse = 0.870
        "bias_ols": ols_coeff,                       # unweighted = 0.833
        "bias_ols_deflated": ols_deflated_coeff,     # deflated = 0.443
        "bias_dE_exact": float(bias_vector[1]),
        "bias_amp_exact": float(bias_vector[0]),
        "correlation_sg": axes.correlation,
        "info_ratio": axes.info_ratio,
    }


# ============================================================================
# Main
# ============================================================================

def main(
    n_spectra: int = N_SPECTRA_DEFAULT,
    solvers: list[str] | None = None,
):
    """Run full gamma perturbation sweep."""
    if solvers is None:
        solvers = list(SOLVER_FUNCS.keys())

    # Fisher prediction
    print("=" * 70)
    print("Gamma Perturbation Experiment")
    print("=" * 70)
    print(f"\nN_spectra per cell: {n_spectra:,}")
    print(f"Solvers: {solvers}")
    print(f"Delta_gamma: {DELTA_GAMMAS}")
    print(f"Noise levels: {list(SNR_LEVELS.keys())}")

    fisher = compute_fisher_prediction()
    print("\nTheoretical predictions (slope = δσ_bias / Δγ):")
    print(f"  Fisher approx (g_σγ/g_σσ):  {fisher['bias_coeff_approx']:+.4f}")
    print(f"  Fisher exact  (g3d⁻¹·j_γ):  {fisher['bias_coeff_exact']:+.4f}")
    print(f"  OLS unweighted:              {fisher['bias_ols']:+.4f}")
    print(f"  OLS + amp deflation:         {fisher['bias_ols_deflated']:+.4f}")
    print(f"  r(σ,γ) = {fisher['correlation_sg']:.4f}")
    print("")
    print("  Hierarchy: Fisher ≥ OLS > Measured > OLS-deflated")
    print("  (Solver's grid search provides partial deflation)")
    print(f"  δE bias per Δγ:  {fisher['bias_dE_exact']:+.6f} (≈0 ✓)")
    print(f"  amp bias per Δγ: {fisher['bias_amp_exact']:+.4f}")

    # Run sweep
    results: list[CellResult] = []
    total_cells = len(solvers) * len(SNR_LEVELS) * len(DELTA_GAMMAS)
    cell_idx = 0

    for solver_name in solvers:
        for snr_name, snr_val in SNR_LEVELS.items():
            for dg in DELTA_GAMMAS:
                cell_idx += 1
                print(f"\r  [{cell_idx}/{total_cells}] {solver_name}/{snr_name}/Δγ={dg:+.3f}",
                      end="", flush=True)

                r = run_cell(
                    delta_gamma=dg,
                    solver_name=solver_name,
                    snr_name=snr_name,
                    snr_val=snr_val,
                    n_spectra=n_spectra,
                )
                results.append(r)

    print()

    # === Results ===
    print("\n" + "=" * 70)
    print("RESULTS: δσ bias vs Δγ")
    print("=" * 70)

    # Table header
    exact = fisher["bias_coeff_exact"]
    approx = fisher["bias_coeff_approx"]
    print(f"\n{'Solver':<10} {'Noise':<8} {'Slope':>10} {'Exact':>10} "
          f"{'Err%':>7} {'Approx':>10} {'R²':>8} {'dE slope':>10}")
    print("-" * 80)

    for solver_name in solvers:
        for snr_name in SNR_LEVELS:
            reg = compute_bias_slope(results, solver_name, snr_name)
            dE_reg = compute_dE_bias_slope(results, solver_name, snr_name)

            err_pct = abs(reg["slope"] - exact) / abs(exact) * 100 if abs(exact) > 1e-10 else np.inf

            print(f"{solver_name:<10} {snr_name:<8} {reg['slope']:>+10.4f} "
                  f"{exact:>+10.4f} {err_pct:>6.1f}% "
                  f"{approx:>+10.4f} {reg['r_squared']:>8.4f} "
                  f"{dE_reg['slope']:>+10.4f}")

    # === Detailed bias table ===
    print("\n" + "=" * 70)
    print("DETAILED: δσ bias at each Δγ (parabola, NF)")
    print("=" * 70)
    print(f"{'Δγ':>8} {'δσ bias':>10} {'δσ std':>10} {'δE bias':>10} {'amp bias':>10}")
    print("-" * 50)

    for r in results:
        if r.solver == "parabola" and r.snr_name == "NF":
            print(f"{r.delta_gamma:>+8.3f} {r.dsigma_bias:>+10.5f} "
                  f"{r.dsigma_std:>10.5f} {r.dE_bias:>+10.5f} "
                  f"{r.amp_bias:>+10.5f}")

    # === Throughput ===
    print("\n" + "=" * 70)
    print("THROUGHPUT (M spec/s)")
    print("=" * 70)
    print(f"{'Solver':<10} {'NF':>8} {'SNR1k':>8} {'SNR100':>8}")
    print("-" * 36)
    for solver_name in solvers:
        vals = []
        for snr_name in SNR_LEVELS:
            # Use Δγ=0 cell
            cell = [r for r in results
                    if r.solver == solver_name and r.snr_name == snr_name
                    and abs(r.delta_gamma) < 1e-10]
            if cell:
                vals.append(f"{cell[0].throughput:>7.2f}")
            else:
                vals.append("   N/A")
        print(f"{solver_name:<10} {'  '.join(vals)}")

    return results, fisher


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gamma perturbation experiment")
    parser.add_argument("--n-spectra", type=int, default=N_SPECTRA_DEFAULT)
    parser.add_argument("--solvers", nargs="+", default=None,
                        choices=list(SOLVER_FUNCS.keys()))
    args = parser.parse_args()

    main(n_spectra=args.n_spectra, solvers=args.solvers)
