"""
Dict2D Adaptive Solver Benchmark — chi2-based Phase 2 switching

Compares Dict2D+4step, Dict2D+amp_only, and Dict2D+adaptive across noise levels.
Key question: Does adaptive achieve NF precision AND noise robustness?

Usage:
    python -m toyomacro.voigtfit.benchmarks.bench_dict2d_adaptive
"""
import time

import numpy as np

from toyomacro.voigtfit.dictionary_solver import (
    build_dictionary_2d,
    solve_dict2d_adaptive,
    solve_dict2d_amp_only,
    solve_hybrid_sorted,
)
from toyomacro.voigtfit.spectra_generator import add_poisson_noise_mlx_fused
from toyomacro.voigtfit.voigt_jacobian import voigt_profile

try:
    import mlx.core as mx

    from toyomacro.voigtfit._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND the default device can execute work
except ImportError:
    HAS_MLX = False


# ============================================================================
# Config
# ============================================================================

N_SPECTRA = 50_000
ENERGY = np.arange(280.0, 290.0, 0.05, dtype=np.float32)
CENTER = np.array([284.8], dtype=np.float32)
SIGMA = np.array([0.6], dtype=np.float32)
GAMMA = 0.2
AMP_RANGE = (0.3, 1.0)
DE_RANGE = (-1.0, 1.0)
DS_RANGE = (-0.15, 0.15)
AMPLITUDE_SCALE = 1000.0
BG_FRACTION = 0.001

NOISE_LEVELS = [0, 1, 10, 100, 1000, 3162, 10000, 100000]
NOISE_LABELS = ['NF', 'lam0', 'lam1', 'lam2', 'lam3', 'lam3.5', 'lam4', 'lam5']


def generate_exact_voigt(energy, centers, sigmas, gamma, amp, dE, ds):
    n = len(amp)
    n_E = len(energy)
    Y = np.zeros((n, n_E), dtype=np.float32)
    for i in range(n):
        prof = voigt_profile(energy, centers[0] + dE[i],
                              sigmas[0] + ds[i], gamma)
        scaled_amp = amp[i] * AMPLITUDE_SCALE
        Y[i] = scaled_amp * prof.astype(np.float32) + scaled_amp * BG_FRACTION
    return Y


def add_noise(spectra, noise_level):
    if noise_level <= 0:
        return spectra
    global_max = spectra.max()
    if HAS_MLX:
        s_mx = mx.array(spectra)
        noisy_mx = add_poisson_noise_mlx_fused(s_mx, noise_level, global_max=global_max)
        mx.eval(noisy_mx)
        return np.array(noisy_mx, dtype=np.float32)
    else:
        from toyomacro.voigtfit.spectra_generator import add_poisson_noise
        return add_poisson_noise(spectra, noise_level, global_max=global_max)


def psnr(x_true, x_pred, peak=None):
    mse = np.mean((x_true - x_pred) ** 2)
    if mse < 1e-30:
        return 99.9
    if peak is None:
        peak = np.max(x_true) - np.min(x_true)
    if peak < 1e-30:
        return 0.0
    return float(10 * np.log10(peak ** 2 / mse))


def main():
    print("Dict2D Adaptive Solver Benchmark")
    print(f"  N_spectra={N_SPECTRA:,}, n_E={len(ENERGY)}")
    print("  Strategy: chi2-based Phase 2 switching (amp_only → 4-step when clean)")
    print()

    # Generate ground truth
    np.random.seed(42)
    gt_amp = np.random.uniform(*AMP_RANGE, N_SPECTRA).astype(np.float32)
    gt_dE = np.random.uniform(*DE_RANGE, N_SPECTRA).astype(np.float32)
    gt_ds = np.random.uniform(*DS_RANGE, N_SPECTRA).astype(np.float32)

    gt_fwhm = 2.3548 * (SIGMA[0] + gt_ds)
    fwhm_peak = gt_fwhm.max() - gt_fwhm.min()

    print("Generating exact Voigt spectra...", end=" ", flush=True)
    Y_clean = generate_exact_voigt(ENERGY, CENTER, SIGMA, GAMMA, gt_amp, gt_dE, gt_ds)
    print("done")

    # Build dictionary
    dc = build_dictionary_2d(
        ENERGY, CENTER, SIGMA, GAMMA,
        dE_range=(-1.2, 1.2), dE_step=0.1 * SIGMA[0],
        dsigma_range=(DS_RANGE[0] - 0.02, DS_RANGE[1] + 0.02),
        dsigma_step=0.02 * SIGMA[0],
    )
    print(f"Dictionary: {dc.n_dE}×{dc.n_dsigma}={dc.n_dict} entries\n")

    # Header
    solver_names = ['Dict2D+4s', 'Dict2D+ao', 'Dict2D+adap']
    print(f"{'':>8s}", end="")
    for name in solver_names:
        print(f"  | {name:>12s}", end="")
    print(f"  | {'% 4step':>8s} {'chi2_norm':>10s}")
    print(f"{'noise':>8s}", end="")
    for _ in solver_names:
        print(f"  | {'FWHM(dB)':>12s}", end="")
    print(f"  | {'':>8s} {'median':>10s}")
    print("-" * 95)

    all_results = {}

    for noise_level, noise_label in zip(NOISE_LEVELS, NOISE_LABELS):
        Y = add_noise(Y_clean.copy(), noise_level) if noise_level > 0 else Y_clean.copy()

        row = {}

        # Dict2D + 4-step
        t0 = time.perf_counter()
        a, _, dE, ds, _ = solve_hybrid_sorted(Y, dc)
        t_4s = time.perf_counter() - t0
        fwhm_out = 2.3548 * (SIGMA[0] + ds)
        row['4s'] = psnr(gt_fwhm, fwhm_out, fwhm_peak)

        # Dict2D + amp_only
        t0 = time.perf_counter()
        a, _, dE, ds, _ = solve_dict2d_amp_only(Y, dc)
        t_ao = time.perf_counter() - t0
        fwhm_out = 2.3548 * (SIGMA[0] + ds)
        row['ao'] = psnr(gt_fwhm, fwhm_out, fwhm_peak)

        # Dict2D + adaptive (auto threshold)
        t0 = time.perf_counter()
        a, chi2_ad, dE, ds, idx = solve_dict2d_adaptive(Y, dc)
        t_ad = time.perf_counter() - t0
        fwhm_out = 2.3548 * (SIGMA[0] + ds)
        row['ad'] = psnr(gt_fwhm, fwhm_out, fwhm_peak)

        # Count how many got 4-step treatment (amplitude-normalized chi2)
        _, chi2_ao, _, _, _ = solve_dict2d_amp_only(Y, dc)
        peak_int = np.max(Y, axis=1)
        chi2_norm = chi2_ao / (peak_int ** 2 + 1e-20)
        chi2_thr_norm = 3e-4  # default threshold
        n_4step = int(np.sum(chi2_norm < chi2_thr_norm))
        pct_4step = 100.0 * n_4step / N_SPECTRA
        chi2_norm_med = float(np.median(chi2_norm))

        print(f"{noise_label:>8s}", end="")
        print(f"  | {row['4s']:11.1f}dB", end="")
        print(f"  | {row['ao']:11.1f}dB", end="")
        print(f"  | {row['ad']:11.1f}dB", end="")
        print(f"  | {pct_4step:7.1f}% {chi2_norm_med:10.2e}")

        all_results[noise_label] = row

    # Summary
    print("\n--- Best Solver per Noise Level ---")
    for nl, row in all_results.items():
        best_name = max(row, key=row.get)
        best_val = row[best_name]
        ad_vs_best = row['ad'] - best_val
        print(f"  {nl:>6s}: best={best_name:5s}({best_val:+.1f}dB)  "
              f"adaptive={row['ad']:+.1f}dB  gap={ad_vs_best:+.1f}dB")

    print("\n--- δσ RMSE Table ---")
    print(f"{'noise':>8s}", end="")
    for name in solver_names:
        print(f"  | {name:>12s}", end="")
    print()
    print("-" * 55)

    for noise_level, noise_label in zip(NOISE_LEVELS, NOISE_LABELS):
        Y = add_noise(Y_clean.copy(), noise_level) if noise_level > 0 else Y_clean.copy()

        _, _, _, ds_4s, _ = solve_hybrid_sorted(Y, dc)
        _, _, _, ds_ao, _ = solve_dict2d_amp_only(Y, dc)
        _, _, _, ds_ad, _ = solve_dict2d_adaptive(Y, dc)

        print(f"{noise_label:>8s}", end="")
        print(f"  |     {np.sqrt(np.mean((gt_ds - ds_4s)**2)):.5f}", end="")
        print(f"  |     {np.sqrt(np.mean((gt_ds - ds_ao)**2)):.5f}", end="")
        print(f"  |     {np.sqrt(np.mean((gt_ds - ds_ad)**2)):.5f}")

    # δE PSNR table
    print("\n--- δE PSNR Table ---")
    print(f"{'noise':>8s}", end="")
    for name in solver_names:
        print(f"  | {name:>12s}", end="")
    print()
    print("-" * 55)

    dE_peak = DE_RANGE[1] - DE_RANGE[0]
    for noise_level, noise_label in zip(NOISE_LEVELS, NOISE_LABELS):
        Y = add_noise(Y_clean.copy(), noise_level) if noise_level > 0 else Y_clean.copy()

        _, _, dE_4s, _, _ = solve_hybrid_sorted(Y, dc)
        _, _, dE_ao, _, _ = solve_dict2d_amp_only(Y, dc)
        _, _, dE_ad, _, _ = solve_dict2d_adaptive(Y, dc)

        print(f"{noise_label:>8s}", end="")
        print(f"  | {psnr(gt_dE, dE_4s, dE_peak):11.1f}dB", end="")
        print(f"  | {psnr(gt_dE, dE_ao, dE_peak):11.1f}dB", end="")
        print(f"  | {psnr(gt_dE, dE_ad, dE_peak):11.1f}dB")


if __name__ == "__main__":
    main()
