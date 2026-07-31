"""
Dict2D Grid Resolution Sweep — Find optimal N_ds

Sweeps N_ds from 1 to 50, measuring FWHM PSNR and throughput for both
4-step and amp_only Phase 2 modes.

Usage:
    python -m toyomacro.voigtfit.benchmarks.bench_dict2d_sweep
"""
import time

import numpy as np

from toyomacro.voigtfit.dictionary_solver import (
    build_dictionary_2d,
    solve_dict2d_amp_only,
    solve_hybrid_sorted,
)
from toyomacro.voigtfit.voigt_jacobian import voigt_profile

# ============================================================================
# Config
# ============================================================================

N_SPECTRA = 100_000
ENERGY = np.arange(280.0, 290.0, 0.05, dtype=np.float32)
CENTER = np.array([284.8], dtype=np.float32)
SIGMA = np.array([0.6], dtype=np.float32)
GAMMA = 0.2
DE_RANGE = (-1.0, 1.0)
DS_RANGE = (-0.15, 0.15)

# N_ds values to sweep
N_DS_VALUES = [1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]


def generate_batch(n):
    np.random.seed(42)
    amp = np.random.uniform(0.3, 1.0, n).astype(np.float32)
    dE = np.random.uniform(*DE_RANGE, n).astype(np.float32)
    ds = np.random.uniform(*DS_RANGE, n).astype(np.float32)

    Y = np.zeros((n, len(ENERGY)), dtype=np.float32)
    for i in range(n):
        prof = voigt_profile(ENERGY, CENTER[0] + dE[i],
                              SIGMA[0] + ds[i], GAMMA)
        Y[i] = amp[i] * prof.astype(np.float32)
    return Y, amp, dE, ds


def psnr(x_true, x_pred, peak):
    mse = np.mean((x_true - x_pred) ** 2)
    if mse < 1e-30:
        return 99.9
    return float(10 * np.log10(peak ** 2 / mse))


def main():
    print("Dict2D Grid Resolution Sweep")
    print(f"  N_spectra={N_SPECTRA:,}, n_E={len(ENERGY)}")
    print(f"  N_dE=41 (fixed), N_ds swept: {N_DS_VALUES}")
    print()

    Y, gt_amp, gt_dE, gt_ds = generate_batch(N_SPECTRA)
    gt_fwhm = 2.3548 * (SIGMA[0] + gt_ds)
    fwhm_peak = gt_fwhm.max() - gt_fwhm.min()

    print(f"{'N_ds':>5s} {'n_dict':>6s} | {'FWHM(4s)':>9s} {'FWHM(ao)':>9s} "
          f"| {'δσ_rmse(4s)':>12s} {'δσ_rmse(ao)':>12s} "
          f"| {'M/s(4s)':>8s} {'M/s(ao)':>8s} {'build':>6s}")
    print("-" * 100)

    for N_ds in N_DS_VALUES:
        t0 = time.perf_counter()
        if N_ds == 1:
            # True shift-only: dsigma_range=(0,0)
            dc = build_dictionary_2d(
                ENERGY, CENTER, SIGMA, GAMMA,
                dE_range=(-1.2, 1.2), N_dE=41,
                dsigma_range=(0.0, 0.0), dsigma_step=1.0,
            )
        else:
            dc = build_dictionary_2d(
                ENERGY, CENTER, SIGMA, GAMMA,
                dE_range=(-1.2, 1.2), N_dE=41,
                dsigma_range=(DS_RANGE[0] - 0.02, DS_RANGE[1] + 0.02),
                N_ds=N_ds,
            )
        build_t = time.perf_counter() - t0

        # 4-step
        solve_hybrid_sorted(Y[:1000], dc)
        t0 = time.perf_counter()
        a4, _, dE4, ds4, _ = solve_hybrid_sorted(Y, dc)
        t4 = time.perf_counter() - t0
        fwhm4 = 2.3548 * (SIGMA[0] + ds4)
        fwhm_psnr_4 = psnr(gt_fwhm, fwhm4, fwhm_peak)
        ds_rmse_4 = np.sqrt(np.mean((gt_ds - ds4) ** 2))
        rate_4 = N_SPECTRA / t4 / 1e6

        # amp_only
        solve_dict2d_amp_only(Y[:1000], dc)
        t0 = time.perf_counter()
        aa, _, dEa, dsa, _ = solve_dict2d_amp_only(Y, dc)
        ta = time.perf_counter() - t0
        fwhma = 2.3548 * (SIGMA[0] + dsa)
        fwhm_psnr_a = psnr(gt_fwhm, fwhma, fwhm_peak)
        ds_rmse_a = np.sqrt(np.mean((gt_ds - dsa) ** 2))
        rate_a = N_SPECTRA / ta / 1e6

        print(f"{N_ds:5d} {dc.n_dict:6d} | {fwhm_psnr_4:8.1f}dB {fwhm_psnr_a:8.1f}dB "
              f"| {ds_rmse_4:11.5f}  {ds_rmse_a:11.5f}  "
              f"| {rate_4:7.2f}  {rate_a:7.2f}  {build_t:5.1f}s")


if __name__ == "__main__":
    main()
