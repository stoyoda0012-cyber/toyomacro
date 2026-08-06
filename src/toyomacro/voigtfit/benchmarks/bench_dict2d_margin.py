"""
Dict2D Margin Analysis — Score margin vs noise level and accuracy

Investigates:
1. How does the margin (top1 - top2 score) distribute at each noise level?
2. Does per-spectrum margin predict whether 4-step or amp_only is better?
3. What is the optimal threshold for adaptive switching?

Usage:
    python -m toyomacro.voigtfit.benchmarks.bench_dict2d_margin
"""

import numpy as np

from toyomacro.voigtfit.dictionary_solver import (
    build_dictionary_2d,
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
# Config (same as bench_dict2d_noise.py)
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

NOISE_LEVELS = [0, 100, 1000, 3162, 10000, 100000]
NOISE_LABELS = ['NF', 'lam2', 'lam3', 'lam3.5', 'lam4', 'lam5']


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


def compute_margins(Y, dc):
    """Compute Phase 1 scores and margins."""
    dc.to_mlx()
    Y_mx = mx.array(Y)
    scores = Y_mx @ dc._D_mx  # (N, n_dict)
    mx.eval(scores)
    scores_np = np.array(scores)

    # Top-2 scores
    # Use partition for efficiency
    top2_idx = np.argpartition(-scores_np, 2, axis=1)[:, :2]
    top2_scores = np.take_along_axis(scores_np, top2_idx, axis=1)
    top2_sorted = np.sort(top2_scores, axis=1)[:, ::-1]  # descending

    best_score = top2_sorted[:, 0]
    second_score = top2_sorted[:, 1]
    margin = best_score - second_score

    # Normalize margin by best score for scale-invariance
    margin_rel = margin / np.maximum(best_score, 1e-10)

    best_idx = np.argmax(scores_np, axis=1).astype(np.int32)

    return best_idx, margin, margin_rel, best_score


def per_spectrum_fwhm_error(ds_out, gt_ds):
    """Per-spectrum absolute FWHM error."""
    return np.abs(2.3548 * (ds_out - gt_ds))


def main():
    print("Dict2D Margin Analysis")
    print(f"  N_spectra={N_SPECTRA:,}, n_E={len(ENERGY)}")
    print()

    np.random.seed(42)
    gt_amp = np.random.uniform(*AMP_RANGE, N_SPECTRA).astype(np.float32)
    gt_dE = np.random.uniform(*DE_RANGE, N_SPECTRA).astype(np.float32)
    gt_ds = np.random.uniform(*DS_RANGE, N_SPECTRA).astype(np.float32)

    print("Generating exact Voigt spectra...", end=" ", flush=True)
    Y_clean = generate_exact_voigt(ENERGY, CENTER, SIGMA, GAMMA, gt_amp, gt_dE, gt_ds)
    print("done")

    dc = build_dictionary_2d(
        ENERGY, CENTER, SIGMA, GAMMA,
        dE_range=(-1.2, 1.2), dE_step=0.1 * SIGMA[0],
        dsigma_range=(DS_RANGE[0] - 0.02, DS_RANGE[1] + 0.02),
        dsigma_step=0.02 * SIGMA[0],
    )
    print(f"Dictionary: {dc.n_dE}×{dc.n_dsigma}={dc.n_dict} entries\n")

    # Part 1: Margin statistics at each noise level
    print("=" * 80)
    print("Part 1: Margin Statistics")
    print("=" * 80)
    print(f"{'noise':>8s} | {'margin_abs':>12s} {'margin_rel':>12s} "
          f"| {'p10':>8s} {'p50':>8s} {'p90':>8s} "
          f"| {'correct%':>9s}")
    print("-" * 80)

    # Get noise-free best indices as "correct" reference
    _, best_idx_nf, _, _ = None, None, None, None

    all_margins = {}
    all_Y = {}

    for noise_level, noise_label in zip(NOISE_LEVELS, NOISE_LABELS):
        Y = add_noise(Y_clean.copy(), noise_level) if noise_level > 0 else Y_clean.copy()
        all_Y[noise_label] = Y

        best_idx, margin, margin_rel, best_score = compute_margins(Y, dc)

        if noise_label == 'NF':
            best_idx_nf = best_idx.copy()

        # How many match noise-free result?
        correct = np.mean(best_idx == best_idx_nf) * 100

        all_margins[noise_label] = (margin, margin_rel, best_idx)

        print(f"{noise_label:>8s} | {np.mean(margin):11.4f}  {np.mean(margin_rel):11.6f} "
              f"| {np.percentile(margin_rel, 10):.6f} {np.percentile(margin_rel, 50):.6f} "
              f"{np.percentile(margin_rel, 90):.6f} "
              f"| {correct:8.1f}%")

    # Part 2: Per-spectrum margin vs FWHM error for lam3 (crossover point)
    print("\n" + "=" * 80)
    print("Part 2: Margin predicts FWHM error? (per-spectrum at lam3)")
    print("=" * 80)

    for noise_label in ['lam3', 'lam3.5', 'lam4']:
        Y = all_Y[noise_label]

        # Run both solvers
        amp_4s, _, dE_4s, ds_4s, _ = solve_hybrid_sorted(Y, dc)
        amp_ao, _, dE_ao, ds_ao, _ = solve_dict2d_amp_only(Y, dc)

        fwhm_err_4s = per_spectrum_fwhm_error(ds_4s, gt_ds)
        fwhm_err_ao = per_spectrum_fwhm_error(ds_ao, gt_ds)

        margin, margin_rel, _ = all_margins[noise_label]

        # Is 4-step better per-spectrum?
        prefer_4step = fwhm_err_4s < fwhm_err_ao

        # Margin statistics for 4step-preferred vs amp_only-preferred spectra
        m_prefer_4s = margin_rel[prefer_4step]
        m_prefer_ao = margin_rel[~prefer_4step]

        print(f"\n  {noise_label}: {np.sum(prefer_4step):,} prefer 4step, "
              f"{np.sum(~prefer_4step):,} prefer amp_only")
        print(f"    Margin (4step-preferred):  mean={np.mean(m_prefer_4s):.6f}  "
              f"median={np.median(m_prefer_4s):.6f}")
        print(f"    Margin (ao-preferred):     mean={np.mean(m_prefer_ao):.6f}  "
              f"median={np.median(m_prefer_ao):.6f}")

        # Sweep threshold and find optimal
        thresholds = np.linspace(0, np.percentile(margin_rel, 99), 50)
        best_thresh = 0
        best_rmse = float('inf')

        for thresh in thresholds:
            use_4s = margin_rel > thresh
            ds_adaptive = np.where(use_4s, ds_4s, ds_ao)
            fwhm_err = per_spectrum_fwhm_error(ds_adaptive, gt_ds)
            rmse = np.sqrt(np.mean(fwhm_err ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_thresh = thresh

        # Compare to pure solvers
        rmse_4s = np.sqrt(np.mean(fwhm_err_4s ** 2))
        rmse_ao = np.sqrt(np.mean(fwhm_err_ao ** 2))
        n_4s = np.sum(margin_rel > best_thresh)

        print(f"    Optimal threshold: {best_thresh:.6f}")
        print(f"    FWHM RMSE: 4step={rmse_4s:.5f}  amp_only={rmse_ao:.5f}  "
              f"adaptive={best_rmse:.5f}")
        print(f"    Adaptive uses 4step for {n_4s:,}/{N_SPECTRA:,} "
              f"({100*n_4s/N_SPECTRA:.1f}%)")
        psnr_4s = _psnr(fwhm_err_4s, gt_ds)
        psnr_ao = _psnr(fwhm_err_ao, gt_ds)
        psnr_ad = _psnr_from_ds(np.where(margin_rel > best_thresh, ds_4s, ds_ao), gt_ds)
        print(f"    FWHM PSNR: 4step={psnr_4s:.1f}dB  amp_only={psnr_ao:.1f}dB  "
              f"adaptive={psnr_ad:.1f}dB")

    # Part 3: Check if a single threshold works across noise levels
    print("\n" + "=" * 80)
    print("Part 3: Universal threshold sweep")
    print("=" * 80)

    thresholds_to_try = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    print(f"{'thresh':>8s}", end="")
    for nl in NOISE_LABELS:
        print(f" | {nl:>8s}", end="")
    print()
    print("-" * (10 + 11 * len(NOISE_LABELS)))

    for thresh in thresholds_to_try:
        print(f"{thresh:8.4f}", end="")
        for noise_label in NOISE_LABELS:
            Y = all_Y[noise_label]
            margin_rel = all_margins[noise_label][1]

            amp_4s, _, dE_4s, ds_4s, _ = solve_hybrid_sorted(Y, dc)
            amp_ao, _, dE_ao, ds_ao, _ = solve_dict2d_amp_only(Y, dc)

            use_4s = margin_rel > thresh
            ds_adaptive = np.where(use_4s, ds_4s, ds_ao)
            psnr_val = _psnr_from_ds(ds_adaptive, gt_ds)
            print(f" | {psnr_val:7.1f}dB", end="")
        print()


def _psnr(fwhm_err, gt_ds):
    """FWHM PSNR from absolute errors."""
    gt_fwhm = 2.3548 * (SIGMA[0] + gt_ds)
    peak = gt_fwhm.max() - gt_fwhm.min()
    if peak < 1e-10:
        return 0.0
    mse = np.mean(fwhm_err ** 2)
    if mse < 1e-30:
        return 99.9
    return float(10 * np.log10(peak ** 2 / mse))


def _psnr_from_ds(ds_out, gt_ds):
    """FWHM PSNR from δσ values."""
    gt_fwhm = 2.3548 * (SIGMA[0] + gt_ds)
    out_fwhm = 2.3548 * (SIGMA[0] + ds_out)
    peak = gt_fwhm.max() - gt_fwhm.min()
    if peak < 1e-10:
        return 0.0
    mse = np.mean((gt_fwhm - out_fwhm) ** 2)
    if mse < 1e-30:
        return 99.9
    return float(10 * np.log10(peak ** 2 / mse))


if __name__ == "__main__":
    main()
