"""
Dict3D vs Dict2D PSNR Comparison
==============================================

Compares 3D dictionary solver (δE × δσ × δγ) against 2D baseline (δE × δσ)
across gamma perturbation sweep and noise levels.

Key question: Does Dict3D eliminate the γ-bias that limits δσ accuracy?

dev-log 63 baseline:
  Dict2D bias slope: +0.635 (vs Fisher prediction +0.870)
  Dict3D target: slope ≈ 0 (γ estimated jointly)

Usage:
    uv run python -m toyomacro.voigtfit.benchmarks.bench_dict3d_psnr
    uv run python -m toyomacro.voigtfit.benchmarks.bench_dict3d_psnr --n-spectra 10000
    uv run python -m toyomacro.voigtfit.benchmarks.bench_dict3d_psnr --fast
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np

from toyomacro.voigtfit.benchmarks.bench_gamma_perturbation import (
    AMPLITUDE_SCALE,
    CENTERS,
    DELTA_GAMMAS,
    ENERGY,
    GAMMA_NOM,
    SIGMAS,
    SNR_LEVELS,
    add_poisson_noise,
    generate_spectra_gamma_perturbed,
)
from toyomacro.voigtfit.dictionary_solver import (
    build_dictionary,
    solve_dict2d_parabola,
)
from toyomacro.voigtfit.dictionary_solver_3d import (
    build_dictionary_3d,
    solve_dict3d_parabola,
)

# ============================================================================
# Cell result
# ============================================================================

@dataclass
class PsnrCellResult:
    """Result for one (delta_gamma, solver, noise) combination."""
    delta_gamma: float
    solver: str
    snr_name: str

    dE_psnr: float
    ds_psnr: float
    dg_psnr: float
    amp_psnr: float

    ds_bias: float
    dg_bias: float

    throughput: float  # M spectra/s


def _psnr(true: np.ndarray, est: np.ndarray) -> float:
    """Signal-to-noise ratio in dB: 10·log10(var(true) / MSE)."""
    var = float(np.var(true))
    mse = float(np.mean((est - true) ** 2))
    if mse < 1e-30:
        return 99.9
    if var < 1e-30:
        return 0.0
    return 10.0 * np.log10(var / mse)


# ============================================================================
# Cell execution
# ============================================================================

def run_cell(
    delta_gamma: float,
    solver_name: str,
    snr_name: str,
    snr_val: float,
    n_spectra: int,
    dict_cache_2d=None,
    dict_cache_3d=None,
    seed: int = 42,
) -> PsnrCellResult:
    """Run one cell of the sweep."""
    gamma_true = GAMMA_NOM + delta_gamma

    # Generate spectra with gamma_true
    Y, gt = generate_spectra_gamma_perturbed(n_spectra, gamma_true, seed=seed)
    Y_noisy = add_poisson_noise(Y, snr_val, seed=seed + 1000)

    t0 = time.perf_counter()

    if solver_name == "dict2d":
        amps, chi2, dE_fit, ds_fit, _ = solve_dict2d_parabola(Y_noisy, dict_cache_2d)
        dg_fit = np.zeros(n_spectra, dtype=np.float32)
    elif solver_name == "dict3d":
        amps, chi2, dE_fit, ds_fit, dg_fit, _ = solve_dict3d_parabola(Y_noisy, dict_cache_3d)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    elapsed = time.perf_counter() - t0
    amp_fit = amps[0] / AMPLITUDE_SCALE

    # Ground truth for δγ: delta_gamma is constant for all spectra
    dg_true = np.full(n_spectra, delta_gamma, dtype=np.float32)

    return PsnrCellResult(
        delta_gamma=delta_gamma,
        solver=solver_name,
        snr_name=snr_name,
        dE_psnr=_psnr(gt["dE"], dE_fit),
        ds_psnr=_psnr(gt["dsigma"], ds_fit),
        dg_psnr=_psnr(dg_true, dg_fit) if solver_name == "dict3d" else float("nan"),
        amp_psnr=_psnr(gt["amp"], amp_fit),
        ds_bias=float(np.mean(ds_fit - gt["dsigma"])),
        dg_bias=float(np.mean(dg_fit - dg_true)),
        throughput=n_spectra / elapsed / 1e6,
    )


# ============================================================================
# Bias slope regression
# ============================================================================

def compute_bias_slope(results: list[PsnrCellResult], solver: str, snr_name: str) -> dict:
    """Linear regression of ds_bias vs delta_gamma."""
    cells = [r for r in results if r.solver == solver and r.snr_name == snr_name]
    cells.sort(key=lambda r: r.delta_gamma)

    x = np.array([r.delta_gamma for r in cells])
    y = np.array([r.ds_bias for r in cells])

    if len(x) < 2:
        return {"slope": np.nan, "r_squared": np.nan}

    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

    return {"slope": float(slope), "r_squared": float(r_squared)}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dict3D vs Dict2D PSNR comparison")
    parser.add_argument("--n-spectra", type=int, default=100_000)
    parser.add_argument("--fast", action="store_true", help="Quick run: 10K spectra, NF only")
    args = parser.parse_args()

    n_spectra = args.n_spectra
    if args.fast:
        n_spectra = 10_000
        snr_levels = {"NF": float("inf")}
        delta_gammas = np.array([-0.02, -0.01, 0.0, +0.01, +0.02])
    else:
        snr_levels = SNR_LEVELS
        delta_gammas = DELTA_GAMMAS

    print("=" * 80)
    print("Dict3D vs Dict2D PSNR Comparison")
    print(f"N_spectra per cell: {n_spectra:,}")
    print(f"Δγ sweep: {list(delta_gammas)}")
    print(f"Noise levels: {list(snr_levels.keys())}")
    print("=" * 80)

    # Build dictionaries (shared across all cells)
    print("\nBuilding 2D dictionary...", end=" ", flush=True)
    t0 = time.perf_counter()
    dict_cache_2d = build_dictionary(
        energy=ENERGY, centers=CENTERS, sigmas=SIGMAS, gamma=GAMMA_NOM,
        dE_range=(-1.2, 1.2), dsigma_range=(-0.3, 0.3),
    )
    print(f"{time.perf_counter() - t0:.2f}s  ({dict_cache_2d.n_dict} entries)")

    print("Building 3D dictionary...", end=" ", flush=True)
    t0 = time.perf_counter()
    # Use matched grid density: dE_step=0.1σ (same as 2D), coarser γ
    dict_cache_3d = build_dictionary_3d(
        energy=ENERGY, centers=CENTERS, sigmas=SIGMAS, gamma=GAMMA_NOM,
        dE_range=(-1.2, 1.2),
        dsigma_range=(-0.3, 0.3),
        dgamma_range=(-0.05, 0.05), dgamma_step=0.0125,
    )
    print(f"{time.perf_counter() - t0:.2f}s  ({dict_cache_3d.n_dict} entries)")

    # Run sweep
    results: list[PsnrCellResult] = []
    solvers = ["dict2d", "dict3d"]

    for snr_name, snr_val in snr_levels.items():
        print(f"\n{'─' * 80}")
        print(f"Noise: {snr_name}")
        print(f"{'─' * 80}")
        print(f"{'Solver':<8} {'Δγ':>8} {'dE PSNR':>10} {'dσ PSNR':>10} "
              f"{'dγ PSNR':>10} {'dσ bias':>10} {'dγ bias':>10} {'Mspec/s':>8}")

        for solver_name in solvers:
            for dg in delta_gammas:
                r = run_cell(
                    delta_gamma=float(dg),
                    solver_name=solver_name,
                    snr_name=snr_name,
                    snr_val=snr_val,
                    n_spectra=n_spectra,
                    dict_cache_2d=dict_cache_2d,
                    dict_cache_3d=dict_cache_3d,
                )
                results.append(r)

                dg_psnr_str = f"{r.dg_psnr:8.1f}" if np.isfinite(r.dg_psnr) else "     N/A"
                print(f"{solver_name:<8} {dg:>8.3f} {r.dE_psnr:>10.1f} {r.ds_psnr:>10.1f} "
                      f"{dg_psnr_str:>10} {r.ds_bias:>+10.5f} {r.dg_bias:>+10.5f} "
                      f"{r.throughput:>8.2f}")

    # Bias slope summary
    print(f"\n{'=' * 80}")
    print("Bias Slope Regression: dσ_bias = slope × Δγ")
    print(f"{'=' * 80}")
    print(f"{'Solver':<8} {'Noise':<8} {'Slope':>8} {'R²':>8}   Note")

    for snr_name in snr_levels:
        for solver_name in solvers:
            reg = compute_bias_slope(results, solver_name, snr_name)
            note = ""
            if solver_name == "dict2d":
                note = "(dev-log 63 baseline ≈ +0.635)"
            elif solver_name == "dict3d":
                if abs(reg["slope"]) < 0.1:
                    note = "★ γ-bias eliminated!"
                else:
                    note = "(residual bias)"
            print(f"{solver_name:<8} {snr_name:<8} {reg['slope']:>+8.3f} "
                  f"{reg['r_squared']:>8.4f}   {note}")

    # Performance summary
    print(f"\n{'=' * 80}")
    print("Throughput Summary")
    print(f"{'=' * 80}")
    for solver_name in solvers:
        rates = [r.throughput for r in results if r.solver == solver_name]
        print(f"{solver_name}: {np.mean(rates):.2f} M/s (mean), "
              f"{np.min(rates):.2f} M/s (min)")


if __name__ == "__main__":
    main()
