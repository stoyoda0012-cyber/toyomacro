"""
Fused adaptive Dict2D solver vs baseline — timing comparison.

Tests both the argsort grouping speedup and the fused 1-pass improvement.

Usage:
    python -m toyomacro.voigtfit.benchmarks.bench_dict2d_adaptive_fused
"""
import time

import numpy as np

from toyomacro.voigtfit.dictionary_solver import (
    build_dictionary_2d,
    solve_dict2d_adaptive,
    solve_dict2d_adaptive_fused,
    solve_dict2d_amp_only,
    solve_hybrid_sorted,
)
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

ENERGY = np.arange(280.0, 290.0, 0.05, dtype=np.float32)
CENTER = np.array([284.8], dtype=np.float32)
SIGMA = np.array([0.6], dtype=np.float32)
GAMMA = 0.2
AMP_RANGE = (0.3, 1.0)
DE_RANGE = (-1.0, 1.0)
DS_RANGE = (-0.15, 0.15)
AMPLITUDE_SCALE = 1000.0


def generate_exact_voigt(energy, centers, sigmas, gamma, amp, dE, ds):
    n = len(amp)
    n_E = len(energy)
    Y = np.zeros((n, n_E), dtype=np.float32)
    for i in range(n):
        prof = voigt_profile(energy, centers[0] + dE[i],
                              sigmas[0] + ds[i], gamma)
        Y[i] = amp[i] * prof.astype(np.float32)
    return Y


def add_noise(Y, lam):
    if lam == 0:
        return Y
    scale = AMPLITUDE_SCALE
    Y_scaled = Y * scale
    Y_noisy = Y_scaled + np.random.normal(0, np.sqrt(lam), Y_scaled.shape)
    return (Y_noisy / scale).astype(np.float32)


def bench_solver(name, solver_fn, Y, dc, n_warmup=1, n_iter=3, **kwargs):
    """Benchmark a solver, return avg time and throughput."""
    n_spectra = Y.shape[0]

    # Warmup
    for _ in range(n_warmup):
        solver_fn(Y, dc, **kwargs)
    if HAS_MLX:
        mx.eval(mx.zeros(1))

    # Timed runs
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = solver_fn(Y, dc, **kwargs)
        if HAS_MLX:
            mx.eval(mx.zeros(1))
        dt = time.perf_counter() - t0
        times.append(dt)

    avg_t = np.mean(times)
    throughput = n_spectra / avg_t
    return avg_t, throughput, result


def psnr(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    if mse < 1e-20:
        return 99.0
    peak = np.max(np.abs(gt))
    return 10 * np.log10(peak ** 2 / mse)


def main():
    print("=" * 72)
    print("Dict2D Adaptive V2 vs V1 Benchmark")
    print("=" * 72)

    # Build dictionary
    dc = build_dictionary_2d(
        ENERGY, CENTER, SIGMA, GAMMA,
        dE_range=(-1.2, 1.2), N_dE=41,
        dsigma_range=(DS_RANGE[0] - 0.02, DS_RANGE[1] + 0.02),
        N_ds=25,
    )
    print(f"Dictionary: {dc.grid_shape} = {dc.n_dict} entries")

    noise_configs = [
        ("NF", 0),
        ("Small (lam1)", 10),
        ("Moderate (lam3)", 3162),
        ("Strong (lam4)", 10000),
    ]

    for n_spectra in [200_000, 1_000_000, 5_000_000]:
        print(f"\n{'=' * 72}")
        print(f"N = {n_spectra:,}")
        print(f"{'=' * 72}")

        np.random.seed(42)
        gt_dE = np.random.uniform(*DE_RANGE, n_spectra).astype(np.float32)
        gt_ds = np.random.uniform(*DS_RANGE, n_spectra).astype(np.float32)
        gt_amp = np.random.uniform(*AMP_RANGE, n_spectra).astype(np.float32)
        Y_clean = generate_exact_voigt(ENERGY, CENTER, SIGMA, GAMMA,
                                        gt_amp, gt_dE, gt_ds)

        for noise_name, lam in noise_configs:
            np.random.seed(99)
            Y = add_noise(Y_clean, lam)

            print(f"\n--- {noise_name} (λ={lam}) ---")
            header = f"{'Solver':<25s} {'Time (s)':>10s} {'M spec/s':>10s} {'δE PSNR':>10s} {'δσ PSNR':>10s}"
            print(header)
            print("-" * len(header))

            solvers = [
                ("4-step (sorted)", solve_hybrid_sorted, {}),
                ("amp_only", solve_dict2d_amp_only, {}),
                ("adaptive V1", solve_dict2d_adaptive, {}),
                ("adaptive V2 (fused)", solve_dict2d_adaptive_fused, {}),
            ]

            for sname, sfn, skw in solvers:
                avg_t, throughput, result = bench_solver(
                    sname, sfn, Y, dc, **skw)
                _, _, dE_out, ds_out, _ = result
                dE_psnr = psnr(gt_dE, dE_out)
                ds_psnr = psnr(gt_ds, ds_out)
                print(f"{sname:<25s} {avg_t:10.3f} {throughput/1e6:10.2f} "
                      f"{dE_psnr:10.1f} {ds_psnr:10.1f}")

    print(f"\n{'=' * 72}")
    print("Done!")


if __name__ == "__main__":
    main()
