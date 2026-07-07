"""
Non-Uniform Grid Benchmark
=======================================================

Experiment 1.1: Compare 4 grid types at same entry count (~900)
Experiment 1.2: sinh α parameter sweep
Experiment 1.3: Large δE validation (±2.0 eV)

Usage:
    uv run python -m toyomacro.voigtfit.benchmarks.bench_nonuniform_grid
"""

import time
from dataclasses import dataclass

import numpy as np

from toyomacro.voigtfit.dictionary_solver import (
    build_dictionary,
    solve_dict2d_parabola,
)
from toyomacro.voigtfit.grids import (
    chebyshev_grid,
    sinh_grid,
    uniform_grid,
)

# ============================================================================
# Constants (C 1s single peak)
# ============================================================================

FWHM_TO_SIGMA = 1.0 / (2 * np.sqrt(2 * np.log(2)))

SIGMA_NOM = 1.0 * FWHM_TO_SIGMA    # 0.4247 eV
GAMMA = 0.125                       # eV
CENTER = 284.4                      # eV
ENERGY = np.linspace(280.0, 290.0, 101, dtype=np.float32)
N_ENERGY = len(ENERGY)


# ============================================================================
# Test data generation
# ============================================================================

def generate_exact_voigt_batch(
    dE_values: np.ndarray,
    dsigma_values: np.ndarray,
    amp_values: np.ndarray,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate exact Voigt spectra with known (δE, δσ, amp) parameters.

    Args:
        dE_values: (n_spectra,) energy shifts
        dsigma_values: (n_spectra,) sigma shifts
        amp_values: (n_spectra,) amplitudes
        noise_sigma: Gaussian noise σ (0 = noise-free)
        rng: Random generator for noise

    Returns:
        Y: (n_spectra, n_energy) float32 spectra
    """
    from scipy.special import wofz

    n = len(dE_values)
    Y = np.empty((n, N_ENERGY), dtype=np.float32)

    for i in range(n):
        c = CENTER + dE_values[i]
        s = SIGMA_NOM + dsigma_values[i]
        z = ((ENERGY - c) + 1j * GAMMA) / (s * np.sqrt(2))
        profile = np.real(wofz(z)) / (s * np.sqrt(2 * np.pi))
        # Normalize peak to 1.0
        profile = profile / (profile.max() + 1e-30)
        Y[i] = (amp_values[i] * profile).astype(np.float32)

    if noise_sigma > 0 and rng is not None:
        Y += rng.normal(0, noise_sigma, Y.shape).astype(np.float32)

    return Y


def generate_test_grid(
    dE_range: tuple[float, float] = (-1.0, 1.0),
    ds_range: tuple[float, float] = (-0.12, 0.12),
    n_dE: int = 50,
    n_ds: int = 25,
    amp: float = 1.0,
    noise_sigma: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a controlled grid of test spectra.

    Returns:
        Y: (n_dE * n_ds, n_energy) spectra
        dE_true: (n_dE * n_ds,) true δE values
        ds_true: (n_dE * n_ds,) true δσ values
        amp_true: (n_dE * n_ds,) true amplitudes
    """
    rng = np.random.default_rng(seed)
    dE_vals = np.linspace(dE_range[0], dE_range[1], n_dE, dtype=np.float32)
    ds_vals = np.linspace(ds_range[0], ds_range[1], n_ds, dtype=np.float32)
    dE_2d, ds_2d = np.meshgrid(dE_vals, ds_vals, indexing='ij')
    dE_true = dE_2d.ravel().astype(np.float32)
    ds_true = ds_2d.ravel().astype(np.float32)
    amp_true = np.full_like(dE_true, amp)

    Y = generate_exact_voigt_batch(dE_true, ds_true, amp_true,
                                    noise_sigma=noise_sigma, rng=rng)
    return Y, dE_true, ds_true, amp_true


# ============================================================================
# Grid configurations
# ============================================================================

@dataclass
class GridConfig:
    name: str
    dE_grid: np.ndarray
    ds_grid: np.ndarray

    @property
    def n_entries(self) -> int:
        return len(self.dE_grid) * len(self.ds_grid)


def make_grid_configs(n_dE: int = 29, n_ds: int = 31,
                      dE_range_wide: float = 2.0,
                      ds_range: float = 0.3) -> list[GridConfig]:
    """Create the 4 grid configurations for Experiment 1.1."""
    configs = []

    # (a) sinh(α=2.0) ±2.0 eV
    configs.append(GridConfig(
        name="sinh(α=2.0) ±2.0",
        dE_grid=sinh_grid(n_dE, dE_range_wide, alpha=2.0).astype(np.float32),
        ds_grid=uniform_grid(n_ds, ds_range).astype(np.float32),
    ))

    # (b) Chebyshev ±2.0 eV
    configs.append(GridConfig(
        name="Chebyshev ±2.0",
        dE_grid=chebyshev_grid(n_dE, dE_range_wide).astype(np.float32),
        ds_grid=uniform_grid(n_ds, ds_range).astype(np.float32),
    ))

    # (c) Uniform ±2.0 eV
    configs.append(GridConfig(
        name="Uniform ±2.0",
        dE_grid=uniform_grid(n_dE, dE_range_wide).astype(np.float32),
        ds_grid=uniform_grid(n_ds, ds_range).astype(np.float32),
    ))

    # (d) Uniform ±1.0 eV
    configs.append(GridConfig(
        name="Uniform ±1.0 (baseline)",
        dE_grid=uniform_grid(n_dE, 1.0).astype(np.float32),
        ds_grid=uniform_grid(n_ds, ds_range).astype(np.float32),
    ))

    return configs


# ============================================================================
# Solver runner
# ============================================================================

@dataclass
class SolverResult:
    dE_rmse: float
    ds_rmse: float
    dE_max_err: float
    ds_max_err: float
    amp_rmse: float
    time_build: float
    time_solve: float
    n_entries: int
    throughput: float   # spectra/s


def run_solver(
    Y: np.ndarray,
    dE_true: np.ndarray,
    ds_true: np.ndarray,
    amp_true: np.ndarray,
    grid_config: GridConfig,
) -> SolverResult:
    """Build dictionary with given grid and run parabola solver."""

    centers = np.array([CENTER], dtype=np.float32)
    sigmas = np.array([SIGMA_NOM], dtype=np.float32)

    # Clip δσ grid to keep σ > 0
    min_sigma = float(np.min(sigmas))
    ds_grid = grid_config.ds_grid[grid_config.ds_grid > -0.9 * min_sigma]

    t0 = time.perf_counter()
    dc = build_dictionary(
        energy=ENERGY,
        centers=centers,
        sigmas=sigmas,
        gamma=GAMMA,
        dE_grid_override=grid_config.dE_grid,
        dsigma_grid_override=ds_grid,
    )
    time_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    amps, chi2, dE_out, ds_out, idx = solve_dict2d_parabola(Y, dc)
    time_solve = time.perf_counter() - t0

    n_spectra = Y.shape[0]
    dE_err = dE_out - dE_true
    ds_err = ds_out - ds_true
    amp_err = amps[0, :] - amp_true

    return SolverResult(
        dE_rmse=float(np.sqrt(np.mean(dE_err**2))),
        ds_rmse=float(np.sqrt(np.mean(ds_err**2))),
        dE_max_err=float(np.max(np.abs(dE_err))),
        ds_max_err=float(np.max(np.abs(ds_err))),
        amp_rmse=float(np.sqrt(np.mean(amp_err**2))),
        time_build=time_build,
        time_solve=time_solve,
        n_entries=dc.n_dict,
        throughput=n_spectra / time_solve if time_solve > 0 else 0,
    )


def rmse_to_psnr(rmse: float, peak: float) -> float:
    """Convert RMSE to PSNR (dB)."""
    if rmse < 1e-30:
        return float('inf')
    return 20 * np.log10(peak / rmse)


# ============================================================================
# Experiment 1.1: 4-grid comparison
# ============================================================================

def experiment_1_1():
    """Compare 4 grid types at same entry count on standard range."""
    print("=" * 72)
    print("Experiment 1.1: 4-Grid Comparison (δE ∈ [-1.0, +1.0])")
    print("=" * 72)

    Y, dE_true, ds_true, amp_true = generate_test_grid(
        dE_range=(-1.0, 1.0), ds_range=(-0.12, 0.12),
        n_dE=50, n_ds=25, noise_sigma=0.0,
    )
    print(f"Test data: {Y.shape[0]} spectra, δE ∈ [{dE_true.min():.2f}, {dE_true.max():.2f}], "
          f"δσ ∈ [{ds_true.min():.3f}, {ds_true.max():.3f}]")

    configs = make_grid_configs()
    results: dict[str, SolverResult] = {}

    for cfg in configs:
        print(f"\n  {cfg.name}: {len(cfg.dE_grid)}×{len(cfg.ds_grid)}={cfg.n_entries} entries")
        r = run_solver(Y, dE_true, ds_true, amp_true, cfg)
        results[cfg.name] = r
        dE_psnr = rmse_to_psnr(r.dE_rmse, 2.0)
        ds_psnr = rmse_to_psnr(r.ds_rmse, 0.3)
        print(f"    δE RMSE: {r.dE_rmse:.6f} eV  ({dE_psnr:.1f} dB)  max: {r.dE_max_err:.6f}")
        print(f"    δσ RMSE: {r.ds_rmse:.6f} eV  ({ds_psnr:.1f} dB)  max: {r.ds_max_err:.6f}")
        print(f"    amp RMSE: {r.amp_rmse:.6f}")
        print(f"    Build: {r.time_build:.3f}s  Solve: {r.time_solve:.3f}s  "
              f"({r.throughput/1e6:.2f} M/s)")

    # Summary table
    print("\n" + "-" * 72)
    print(f"{'Grid':<25} {'δE RMSE':>10} {'δE dB':>8} {'δσ RMSE':>10} {'δσ dB':>8} {'Entries':>8}")
    print("-" * 72)
    for name, r in results.items():
        dE_psnr = rmse_to_psnr(r.dE_rmse, 2.0)
        ds_psnr = rmse_to_psnr(r.ds_rmse, 0.3)
        print(f"{name:<25} {r.dE_rmse:10.6f} {dE_psnr:8.1f} {r.ds_rmse:10.6f} {ds_psnr:8.1f} {r.n_entries:8d}")

    return results


# ============================================================================
# Experiment 1.2: sinh α sweep
# ============================================================================

def experiment_1_2():
    """Sweep sinh α parameter to find optimal density concentration."""
    print("\n" + "=" * 72)
    print("Experiment 1.2: sinh α Parameter Sweep (δE ∈ [-1.0, +1.0])")
    print("=" * 72)

    Y, dE_true, ds_true, amp_true = generate_test_grid(
        dE_range=(-1.0, 1.0), ds_range=(-0.12, 0.12),
        n_dE=50, n_ds=25, noise_sigma=0.0,
    )

    alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    n_dE, n_ds = 29, 31
    results: dict[float, SolverResult] = {}

    for alpha in alphas:
        dE_g = sinh_grid(n_dE, 2.0, alpha=alpha).astype(np.float32)
        ds_g = uniform_grid(n_ds, 0.3).astype(np.float32)
        cfg = GridConfig(f"sinh(α={alpha})", dE_g, ds_g)

        r = run_solver(Y, dE_true, ds_true, amp_true, cfg)
        results[alpha] = r
        dE_psnr = rmse_to_psnr(r.dE_rmse, 2.0)
        ds_psnr = rmse_to_psnr(r.ds_rmse, 0.3)
        # Center spacing
        diffs = np.diff(dE_g)
        center_sp = diffs[len(diffs)//2]
        edge_sp = diffs[0]
        print(f"  α={alpha:.1f}: δE RMSE={r.dE_rmse:.6f} ({dE_psnr:.1f}dB), "
              f"δσ RMSE={r.ds_rmse:.6f} ({ds_psnr:.1f}dB), "
              f"center Δ={center_sp:.4f}, edge Δ={edge_sp:.4f}")

    # Summary
    print("\n" + "-" * 60)
    print(f"{'α':>5} {'δE RMSE':>10} {'δE dB':>8} {'δσ RMSE':>10} {'δσ dB':>8}")
    print("-" * 60)
    for alpha, r in results.items():
        dE_psnr = rmse_to_psnr(r.dE_rmse, 2.0)
        ds_psnr = rmse_to_psnr(r.ds_rmse, 0.3)
        print(f"{alpha:5.1f} {r.dE_rmse:10.6f} {dE_psnr:8.1f} {r.ds_rmse:10.6f} {ds_psnr:8.1f}")

    return results


# ============================================================================
# Experiment 1.3: Large δE validation
# ============================================================================

def experiment_1_3():
    """Validate extended range grids with δE up to ±2.0 eV."""
    print("\n" + "=" * 72)
    print("Experiment 1.3: Large δE Validation (δE ∈ [-2.0, +2.0])")
    print("=" * 72)

    Y, dE_true, ds_true, amp_true = generate_test_grid(
        dE_range=(-2.0, 2.0), ds_range=(-0.12, 0.12),
        n_dE=80, n_ds=25, noise_sigma=0.0,
    )
    print(f"Test data: {Y.shape[0]} spectra, δE ∈ [{dE_true.min():.2f}, {dE_true.max():.2f}]")

    configs = make_grid_configs()
    results: dict[str, SolverResult] = {}

    for cfg in configs:
        r = run_solver(Y, dE_true, ds_true, amp_true, cfg)
        results[cfg.name] = r
        dE_psnr = rmse_to_psnr(r.dE_rmse, 4.0)
        ds_psnr = rmse_to_psnr(r.ds_rmse, 0.3)
        print(f"  {cfg.name:<25}: δE RMSE={r.dE_rmse:.4f} ({dE_psnr:.1f}dB), "
              f"δσ RMSE={r.ds_rmse:.6f} ({ds_psnr:.1f}dB), max|δE err|={r.dE_max_err:.4f}")

    # Also show breakdown by δE magnitude
    print("\n  Breakdown by |δE| range:")
    bins = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]

    # Re-run with detailed analysis for each config
    for cfg in configs:
        centers = np.array([CENTER], dtype=np.float32)
        sigmas = np.array([SIGMA_NOM], dtype=np.float32)
        ds_grid = cfg.ds_grid[cfg.ds_grid > -0.9 * float(np.min(sigmas))]
        dc = build_dictionary(ENERGY, centers, sigmas, GAMMA,
                              dE_grid_override=cfg.dE_grid,
                              dsigma_grid_override=ds_grid)
        _, _, dE_out, ds_out, _ = solve_dict2d_parabola(Y, dc)
        dE_err = np.abs(dE_out - dE_true)

        parts = []
        for lo, hi in bins:
            mask = (np.abs(dE_true) >= lo) & (np.abs(dE_true) < hi)
            if mask.sum() > 0:
                rmse = float(np.sqrt(np.mean((dE_out[mask] - dE_true[mask])**2)))
                parts.append(f"|δE|∈[{lo},{hi}): {rmse:.5f}")
            else:
                parts.append(f"|δE|∈[{lo},{hi}): N/A")
        print(f"    {cfg.name:<25}: {', '.join(parts)}")

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Non-Uniform Grid Benchmark")
    print(f"σ_nom={SIGMA_NOM:.4f}, γ={GAMMA}, center={CENTER}, n_E={N_ENERGY}")
    print()

    r1 = experiment_1_1()
    r2 = experiment_1_2()
    r3 = experiment_1_3()

    print("\n" + "=" * 72)
    print("All experiments complete.")
