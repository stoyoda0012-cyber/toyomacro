"""
Dict2D Noise Sweep — FWHM PSNR vs SNR for 5 solvers

Compares: 4-step Taylor, 6-step Hessian, Dict1D+4step, Dict2D+4step, Dict2D+amp_only
across noise levels from noise-free to Intense (10^5).

Key question: Does Dict2D's dense grid become fragile at high noise?
dev-log 44 showed Dict1D vs 6-step crossover at SNR≈3. Where does Dict2D cross?

Usage:
    PYTHONPATH=python python -m voigtfit.benchmarks.bench_dict2d_noise
"""
import time
from dataclasses import dataclass

import numpy as np

from toyomacro.voigtfit.dictionary_solver import (
    build_dictionary_2d,
    build_dictionary_shift_only,
    solve_dict2d_amp_only,
    solve_hybrid_sorted,
)
from toyomacro.voigtfit.spectra_generator import add_poisson_noise_mlx_fused
from toyomacro.voigtfit.voigt_jacobian import voigt_profile

try:
    import mlx.core as mx
    HAS_MLX = True
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

# Noise levels: None, lam0..lam5 (SNR ~ inf, ~1000, ~300, ~100, ~30, ~10, ~3)
NOISE_LEVELS = [0, 1, 10, 100, 1000, 3162, 10000, 100000]
NOISE_LABELS = ['NF', 'lam0', 'lam1', 'lam2', 'lam3', 'lam3.5', 'lam4', 'lam5']


def generate_exact_voigt(energy, centers, sigmas, gamma,
                          amp, dE, ds) -> np.ndarray:
    """Generate exact Voigt spectra with amplitude scaling + background."""
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
    """Add heteroscedastic Poisson noise."""
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


def run_taylor_4step(Y, gt_amp, gt_dE, gt_ds):
    """Run 4-step Taylor solver (no dictionary)."""
    from toyomacro.voigtfit.pipeline import HybridPipeline
    from toyomacro.voigtfit.weight_cache import WeightMatrixCache

    cache = WeightMatrixCache()
    pipeline = HybridPipeline(cache=cache, enable_stage2=False, use_mlx=True)

    peak_config = {
        'centers': CENTER,
        'sigmas': SIGMA,
        'gamma': GAMMA,
    }

    result = pipeline.process_rowmajor_extended_3param(
        Y, 'C', '1s', ENERGY, peak_config,
        amplitudes_only=True, n_steps=4, amp_correction='full',
    )
    amps = result.amplitudes[0] / AMPLITUDE_SCALE
    dE = result.energy_shifts
    ds = result.sigma_shifts

    return amps, dE, ds


def run_taylor_6step(Y, gt_amp, gt_dE, gt_ds):
    """Run 6-step Hessian Taylor solver (no dictionary)."""
    from toyomacro.voigtfit.pipeline import HybridPipeline
    from toyomacro.voigtfit.weight_cache import WeightMatrixCache

    cache = WeightMatrixCache()
    pipeline = HybridPipeline(cache=cache, enable_stage2=False, use_mlx=True)

    peak_config = {
        'centers': CENTER,
        'sigmas': SIGMA,
        'gamma': GAMMA,
    }

    result = pipeline.process_rowmajor_extended_3param(
        Y, 'C', '1s', ENERGY, peak_config,
        amplitudes_only=True, n_steps=6, amp_correction='full',
    )
    amps = result.amplitudes[0] / AMPLITUDE_SCALE
    dE = result.energy_shifts
    ds = result.sigma_shifts

    return amps, dE, ds


@dataclass
class SolverResult:
    name: str
    fwhm_psnr: float
    dE_psnr: float
    amp_psnr: float
    ds_rmse: float
    ds_corr: float
    dE_rmse: float
    throughput: float  # M/s


def evaluate(name, amp_out, dE_out, ds_out, gt_amp, gt_dE, gt_ds, elapsed):
    fwhm_true = 2.3548 * (SIGMA[0] + gt_ds)
    fwhm_out = 2.3548 * (SIGMA[0] + ds_out)
    fwhm_peak = max(fwhm_true.max() - fwhm_true.min(), 1e-10)

    return SolverResult(
        name=name,
        fwhm_psnr=psnr(fwhm_true, fwhm_out, fwhm_peak),
        dE_psnr=psnr(gt_dE, dE_out, DE_RANGE[1] - DE_RANGE[0]),
        amp_psnr=psnr(gt_amp, amp_out, AMP_RANGE[1] - AMP_RANGE[0]),
        ds_rmse=float(np.sqrt(np.mean((gt_ds - ds_out) ** 2))),
        ds_corr=float(np.corrcoef(gt_ds, ds_out)[0, 1]) if np.std(ds_out) > 1e-10 else 0.0,
        dE_rmse=float(np.sqrt(np.mean((gt_dE - dE_out) ** 2))),
        throughput=N_SPECTRA / max(elapsed, 1e-10) / 1e6,
    )


def main():
    print("Dict2D Noise Sweep Benchmark")
    print(f"  N_spectra={N_SPECTRA:,}, n_E={len(ENERGY)}")
    print(f"  δE range: {DE_RANGE}, δσ range: {DS_RANGE}")
    print(f"  Noise levels: {NOISE_LABELS}")
    print()

    # Generate ground truth
    np.random.seed(42)
    gt_amp = np.random.uniform(*AMP_RANGE, N_SPECTRA).astype(np.float32)
    gt_dE = np.random.uniform(*DE_RANGE, N_SPECTRA).astype(np.float32)
    gt_ds = np.random.uniform(*DS_RANGE, N_SPECTRA).astype(np.float32)

    print("Generating exact Voigt spectra...", end=" ", flush=True)
    t0 = time.perf_counter()
    Y_clean = generate_exact_voigt(ENERGY, CENTER, SIGMA, GAMMA, gt_amp, gt_dE, gt_ds)
    print(f"done ({time.perf_counter()-t0:.1f}s)")

    # Pre-build dictionaries
    print("\nBuilding dictionaries...")

    t0 = time.perf_counter()
    dc_1d = build_dictionary_shift_only(
        ENERGY, CENTER, SIGMA, GAMMA,
        dE_range=(-1.2, 1.2), dE_step=0.1 * SIGMA[0],
    )
    print(f"  Dict1D: {dc_1d.n_dE}×{dc_1d.n_dsigma}={dc_1d.n_dict} ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    dc_2d = build_dictionary_2d(
        ENERGY, CENTER, SIGMA, GAMMA,
        dE_range=(-1.2, 1.2), dE_step=0.1 * SIGMA[0],
        dsigma_range=(DS_RANGE[0] - 0.02, DS_RANGE[1] + 0.02),
        dsigma_step=0.02 * SIGMA[0],
    )
    print(f"  Dict2D: {dc_2d.n_dE}×{dc_2d.n_dsigma}={dc_2d.n_dict} ({time.perf_counter()-t0:.1f}s)")

    # Solver definitions
    solvers = [
        ('4-step', lambda Y: run_taylor_4step_dict(Y)),
        ('6-step', lambda Y: run_taylor_6step_dict(Y)),
        ('Dict1D+4s', lambda Y: run_dict(Y, dc_1d, solve_hybrid_sorted)),
        ('Dict2D+4s', lambda Y: run_dict(Y, dc_2d, solve_hybrid_sorted)),
        ('Dict2D+ao', lambda Y: run_dict(Y, dc_2d, solve_dict2d_amp_only)),
    ]

    # Results table: rows = noise, columns = solvers
    print(f"\n{'':>8s}", end="")
    for name in ['4-step', '6-step', 'Dict1D+4s', 'Dict2D+4s', 'Dict2D+ao']:
        print(f"  | {name:>12s}", end="")
    print()
    print(f"{'noise':>8s}", end="")
    for _ in range(5):
        print(f"  | {'FWHM(dB)':>12s}", end="")
    print()
    print("-" * 85)

    all_results = {}

    for noise_level, noise_label in zip(NOISE_LEVELS, NOISE_LABELS):
        # Add noise
        if noise_level > 0:
            Y = add_noise(Y_clean.copy(), noise_level)
        else:
            Y = Y_clean.copy()

        row_results = []

        # 4-step Taylor
        t0 = time.perf_counter()
        a, dE, ds = run_taylor_4step(Y, gt_amp, gt_dE, gt_ds)
        elapsed = time.perf_counter() - t0
        r = evaluate('4-step', a, dE, ds, gt_amp, gt_dE, gt_ds, elapsed)
        row_results.append(r)

        # 6-step Hessian
        t0 = time.perf_counter()
        a, dE, ds = run_taylor_6step(Y, gt_amp, gt_dE, gt_ds)
        elapsed = time.perf_counter() - t0
        r = evaluate('6-step', a, dE, ds, gt_amp, gt_dE, gt_ds, elapsed)
        row_results.append(r)

        # Dict1D + 4-step
        t0 = time.perf_counter()
        amps, _, dE, ds, _ = solve_hybrid_sorted(Y, dc_1d)
        elapsed = time.perf_counter() - t0
        r = evaluate('Dict1D+4s', amps[0] / AMPLITUDE_SCALE, dE, ds,
                     gt_amp, gt_dE, gt_ds, elapsed)
        row_results.append(r)

        # Dict2D + 4-step
        t0 = time.perf_counter()
        amps, _, dE, ds, _ = solve_hybrid_sorted(Y, dc_2d)
        elapsed = time.perf_counter() - t0
        r = evaluate('Dict2D+4s', amps[0] / AMPLITUDE_SCALE, dE, ds,
                     gt_amp, gt_dE, gt_ds, elapsed)
        row_results.append(r)

        # Dict2D + amp_only
        t0 = time.perf_counter()
        amps, _, dE, ds, _ = solve_dict2d_amp_only(Y, dc_2d)
        elapsed = time.perf_counter() - t0
        r = evaluate('Dict2D+ao', amps[0] / AMPLITUDE_SCALE, dE, ds,
                     gt_amp, gt_dE, gt_ds, elapsed)
        row_results.append(r)

        # Print FWHM row
        print(f"{noise_label:>8s}", end="")
        for r in row_results:
            print(f"  | {r.fwhm_psnr:11.1f}dB", end="")
        print()

        all_results[noise_label] = row_results

    # Detailed table: δσ RMSE
    print(f"\n{'':>8s}", end="")
    for name in ['4-step', '6-step', 'Dict1D+4s', 'Dict2D+4s', 'Dict2D+ao']:
        print(f"  | {name:>12s}", end="")
    print()
    print(f"{'noise':>8s}", end="")
    for _ in range(5):
        print(f"  | {'δσ RMSE':>12s}", end="")
    print()
    print("-" * 85)

    for noise_label, row_results in all_results.items():
        print(f"{noise_label:>8s}", end="")
        for r in row_results:
            print(f"  |     {r.ds_rmse:.5f}", end="")
        print()

    # δE PSNR table
    print(f"\n{'':>8s}", end="")
    for name in ['4-step', '6-step', 'Dict1D+4s', 'Dict2D+4s', 'Dict2D+ao']:
        print(f"  | {name:>12s}", end="")
    print()
    print(f"{'noise':>8s}", end="")
    for _ in range(5):
        print(f"  | {'δE (dB)':>12s}", end="")
    print()
    print("-" * 85)

    for noise_label, row_results in all_results.items():
        print(f"{noise_label:>8s}", end="")
        for r in row_results:
            print(f"  | {r.dE_psnr:11.1f}dB", end="")
        print()

    # Find crossovers
    print("\n--- Crossover Analysis ---")
    for noise_label, row_results in all_results.items():
        d1 = row_results[2].fwhm_psnr  # Dict1D
        d2 = row_results[3].fwhm_psnr  # Dict2D
        s6 = row_results[1].fwhm_psnr  # 6-step
        best = max(row_results, key=lambda r: r.fwhm_psnr)
        print(f"  {noise_label:>6s}: best={best.name:12s}  "
              f"Dict2D-Dict1D={d2-d1:+.1f}dB  Dict2D-6step={d2-s6:+.1f}dB")


if __name__ == "__main__":
    main()
