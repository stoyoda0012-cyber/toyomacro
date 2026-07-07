"""
Dict1D vs Dict2D vs Dict2D_amp_only — Quick Accuracy + Throughput Benchmark

Generates exact Voigt spectra with known (amp, δE, δσ), recovers with
three solver configurations, and reports per-parameter PSNR/RMSE/correlation.

Usage:
    PYTHONPATH=python python -m voigtfit.benchmarks.bench_dict2d
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
from toyomacro.voigtfit.voigt_jacobian import voigt_profile

# ============================================================================
# Config
# ============================================================================

N_SPECTRA = 200_000
ENERGY = np.arange(280.0, 290.0, 0.05, dtype=np.float32)  # C 1s-like
CENTER = np.array([284.8], dtype=np.float32)
SIGMA = np.array([0.6], dtype=np.float32)
GAMMA = 0.2
AMP_RANGE = (0.3, 1.0)
DE_RANGE = (-1.0, 1.0)
DS_RANGE = (-0.15, 0.15)


def generate_exact_voigt(energy, centers, sigmas, gamma,
                          amp, dE, ds) -> np.ndarray:
    """Generate exact Voigt spectra (no Taylor)."""
    n = len(amp)
    n_E = len(energy)
    Y = np.zeros((n, n_E), dtype=np.float32)
    for i in range(n):
        prof = voigt_profile(energy, centers[0] + dE[i],
                              sigmas[0] + ds[i], gamma)
        Y[i] = amp[i] * prof.astype(np.float32)
    return Y


def psnr(x_true, x_pred, peak=None):
    """Per-channel PSNR (dB)."""
    mse = np.mean((x_true - x_pred) ** 2)
    if mse < 1e-30:
        return 99.9
    if peak is None:
        peak = np.max(x_true) - np.min(x_true)
    return float(10 * np.log10(peak ** 2 / mse))


@dataclass
class BenchResult:
    name: str
    n_dict: int
    grid_shape: tuple[int, int]

    amp_psnr: float
    dE_psnr: float
    fwhm_psnr: float

    amp_rmse: float
    dE_rmse: float
    ds_rmse: float

    amp_corr: float
    dE_corr: float
    ds_corr: float

    throughput: float  # M spec/s
    build_time: float  # seconds

    def __str__(self):
        return (
            f"{self.name:25s}  grid={self.grid_shape[0]:3d}×{self.grid_shape[1]:3d}"
            f"  n_dict={self.n_dict:5d}"
            f"  | amp={self.amp_psnr:5.1f}dB  δE={self.dE_psnr:5.1f}dB"
            f"  FWHM={self.fwhm_psnr:5.1f}dB"
            f"  | δσ_rmse={self.ds_rmse:.5f}  δσ_corr={self.ds_corr:.4f}"
            f"  | {self.throughput:.2f}M/s  build={self.build_time:.1f}s"
        )


def run_solver(name, dc, Y, gt_amp, gt_dE, gt_ds, solver_func, build_time):
    """Run a solver and compute metrics."""
    # Warmup
    solver_func(Y[:1000], dc)

    n_iter = 3
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        amps, chi2, dE_out, ds_out, _ = solver_func(Y, dc)
        times.append(time.perf_counter() - t0)

    t_median = np.median(times)
    throughput = len(Y) / t_median / 1e6

    amp_out = amps[0]
    fwhm_true = 2.3548 * (SIGMA[0] + gt_ds)
    fwhm_out = 2.3548 * (SIGMA[0] + ds_out)

    return BenchResult(
        name=name,
        n_dict=dc.n_dict,
        grid_shape=dc.grid_shape,
        amp_psnr=psnr(gt_amp, amp_out, peak=AMP_RANGE[1] - AMP_RANGE[0]),
        dE_psnr=psnr(gt_dE, dE_out, peak=DE_RANGE[1] - DE_RANGE[0]),
        fwhm_psnr=psnr(fwhm_true, fwhm_out, peak=fwhm_true.max() - fwhm_true.min()),
        amp_rmse=float(np.sqrt(np.mean((gt_amp - amp_out) ** 2))),
        dE_rmse=float(np.sqrt(np.mean((gt_dE - dE_out) ** 2))),
        ds_rmse=float(np.sqrt(np.mean((gt_ds - ds_out) ** 2))),
        amp_corr=float(np.corrcoef(gt_amp, amp_out)[0, 1]),
        dE_corr=float(np.corrcoef(gt_dE, dE_out)[0, 1]),
        ds_corr=float(np.corrcoef(gt_ds, ds_out)[0, 1]),
        throughput=throughput,
        build_time=build_time,
    )


def main():
    print("Dict1D vs Dict2D vs Dict2D_amp_only Benchmark")
    print(f"  N_spectra={N_SPECTRA:,}, n_E={len(ENERGY)}")
    print(f"  δE range: {DE_RANGE}, δσ range: {DS_RANGE}")
    print()

    # Generate ground truth
    np.random.seed(42)
    gt_amp = np.random.uniform(*AMP_RANGE, N_SPECTRA).astype(np.float32)
    gt_dE = np.random.uniform(*DE_RANGE, N_SPECTRA).astype(np.float32)
    gt_ds = np.random.uniform(*DS_RANGE, N_SPECTRA).astype(np.float32)

    print("Generating exact Voigt spectra...", end=" ", flush=True)
    t0 = time.perf_counter()
    Y = generate_exact_voigt(ENERGY, CENTER, SIGMA, GAMMA, gt_amp, gt_dE, gt_ds)
    print(f"done ({time.perf_counter()-t0:.1f}s)")

    results = []

    # --- Config 1: Dict1D (shift-only, δσ=0 in dict) + 4-step ---
    print("\nBuilding Dict1D (shift-only)...", end=" ", flush=True)
    t0 = time.perf_counter()
    dc1d = build_dictionary_shift_only(
        ENERGY, CENTER, SIGMA, GAMMA,
        dE_range=(-1.2, 1.2), dE_step=0.1 * SIGMA[0],
    )
    bt1 = time.perf_counter() - t0
    print(f"{dc1d.n_dE}×{dc1d.n_dsigma}={dc1d.n_dict} ({bt1:.1f}s)")
    results.append(run_solver("Dict1D+4step", dc1d, Y, gt_amp, gt_dE, gt_ds,
                               solve_hybrid_sorted, bt1))

    # --- Config 2: Dict2D (dense δσ, step=0.02σ) + 4-step ---
    print("\nBuilding Dict2D (dense, step=0.02σ)...", end=" ", flush=True)
    t0 = time.perf_counter()
    dc2d = build_dictionary_2d(
        ENERGY, CENTER, SIGMA, GAMMA,
        dE_range=(-1.2, 1.2), dE_step=0.1 * SIGMA[0],
        dsigma_range=(DS_RANGE[0] - 0.02, DS_RANGE[1] + 0.02),
        dsigma_step=0.02 * SIGMA[0],
    )
    bt2 = time.perf_counter() - t0
    print(f"{dc2d.n_dE}×{dc2d.n_dsigma}={dc2d.n_dict} ({bt2:.1f}s)")
    results.append(run_solver("Dict2D+4step", dc2d, Y, gt_amp, gt_dE, gt_ds,
                               solve_hybrid_sorted, bt2))

    # --- Config 3: Dict2D + amp_only ---
    print("\nDict2D+amp_only (same dict, no 4-step)...")
    results.append(run_solver("Dict2D+amp_only", dc2d, Y, gt_amp, gt_dE, gt_ds,
                               solve_dict2d_amp_only, bt2))

    # --- Config 4: Dict2D ultra-dense (step=0.01σ) + amp_only ---
    print("\nBuilding Dict2D ultra-dense (step=0.01σ)...", end=" ", flush=True)
    t0 = time.perf_counter()
    dc2d_ultra = build_dictionary_2d(
        ENERGY, CENTER, SIGMA, GAMMA,
        dE_range=(-1.2, 1.2), dE_step=0.05 * SIGMA[0],
        dsigma_range=(DS_RANGE[0] - 0.02, DS_RANGE[1] + 0.02),
        dsigma_step=0.01 * SIGMA[0],
    )
    bt3 = time.perf_counter() - t0
    print(f"{dc2d_ultra.n_dE}×{dc2d_ultra.n_dsigma}={dc2d_ultra.n_dict} ({bt3:.1f}s)")
    results.append(run_solver("Dict2D_ultra+amp_only", dc2d_ultra, Y, gt_amp, gt_dE, gt_ds,
                               solve_dict2d_amp_only, bt3))

    # --- Print summary ---
    print("\n" + "=" * 130)
    print("SUMMARY")
    print("=" * 130)
    for r in results:
        print(r)

    # Highlight improvements
    print("\n--- Improvement vs Dict1D+4step ---")
    base = results[0]
    for r in results[1:]:
        fwhm_gain = r.fwhm_psnr - base.fwhm_psnr
        ds_ratio = base.ds_rmse / max(r.ds_rmse, 1e-10)
        speed_ratio = r.throughput / max(base.throughput, 1e-10)
        print(f"  {r.name:25s}: FWHM {fwhm_gain:+.1f}dB, δσ_rmse {ds_ratio:.1f}×, "
              f"speed {speed_ratio:.2f}×")


if __name__ == "__main__":
    main()
