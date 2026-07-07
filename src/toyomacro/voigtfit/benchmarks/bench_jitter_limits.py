"""
Jitter Limits Benchmark
====================================================

2D precision mapping across (δE, δσ) parameter space to determine
where parabola sub-grid interpolation is accurate and where it breaks down.

Experiment 2.1: Noise-free precision map (4 solver configs)
Experiment 2.2: Bias-variance decomposition (noisy, n_trials=10)
Experiment 2.3: Parabola breakdown boundary (contour extraction)
Experiment 2.4: Jitter limit quantification table

Usage:
    uv run python -m toyomacro.voigtfit.benchmarks.bench_jitter_limits
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
    uniform_grid,
)

# ============================================================================
# Constants (C 1s single peak — same as bench_nonuniform_grid)
# ============================================================================

FWHM_TO_SIGMA = 1.0 / (2 * np.sqrt(2 * np.log(2)))
SIGMA_NOM = 1.0 * FWHM_TO_SIGMA    # 0.4247 eV
GAMMA = 0.125                       # eV
CENTER = 284.4                      # eV
ENERGY = np.linspace(280.0, 290.0, 101, dtype=np.float32)
N_ENERGY = len(ENERGY)
CENTERS = np.array([CENTER], dtype=np.float32)
SIGMAS = np.array([SIGMA_NOM], dtype=np.float32)


# ============================================================================
# Spectra generation (exact Voigt via Faddeeva)
# ============================================================================

def generate_voigt_spectra(
    dE_values: np.ndarray,
    dsigma_values: np.ndarray,
    amp: float = 1.0,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate exact Voigt spectra for given (δE, δσ) arrays."""
    from scipy.special import wofz

    n = len(dE_values)
    Y = np.empty((n, N_ENERGY), dtype=np.float32)

    for i in range(n):
        c = CENTER + dE_values[i]
        s = SIGMA_NOM + dsigma_values[i]
        if s < 0.01:
            s = 0.01  # Guard against non-physical σ
        z = ((ENERGY - c) + 1j * GAMMA) / (s * np.sqrt(2))
        profile = np.real(wofz(z)) / (s * np.sqrt(2 * np.pi))
        profile = profile / (profile.max() + 1e-30)
        Y[i] = (amp * profile).astype(np.float32)

    if noise_sigma > 0 and rng is not None:
        Y += rng.normal(0, noise_sigma, Y.shape).astype(np.float32)

    return Y


# ============================================================================
# Precision map data structure
# ============================================================================

@dataclass
class PrecisionMap:
    """2D precision map over (δE, δσ) space."""
    dE_values: np.ndarray       # (n_dE,)
    ds_values: np.ndarray       # (n_ds,)
    # Bias maps (n_dE, n_ds)
    dE_bias: np.ndarray
    ds_bias: np.ndarray
    amp_bias: np.ndarray
    # Std maps (n_dE, n_ds) — zero for single-trial
    dE_std: np.ndarray
    ds_std: np.ndarray
    amp_std: np.ndarray
    # RMSE = sqrt(bias² + std²)
    solver_name: str = ""
    n_trials: int = 1

    @property
    def dE_rmse(self) -> np.ndarray:
        return np.sqrt(self.dE_bias**2 + self.dE_std**2)

    @property
    def ds_rmse(self) -> np.ndarray:
        return np.sqrt(self.ds_bias**2 + self.ds_std**2)

    @property
    def amp_rmse(self) -> np.ndarray:
        return np.sqrt(self.amp_bias**2 + self.amp_std**2)


# ============================================================================
# Solver configurations
# ============================================================================

@dataclass
class SolverConfig:
    name: str
    dE_grid: np.ndarray
    ds_grid: np.ndarray
    use_parabola: bool = True

    @property
    def n_entries(self) -> int:
        return len(self.dE_grid) * len(self.ds_grid)


def make_solver_configs(n_dE: int = 29, n_ds: int = 31) -> list[SolverConfig]:
    """4 solver configurations for comparison."""
    return [
        SolverConfig(
            name="Uniform ±1.0 + parabola",
            dE_grid=uniform_grid(n_dE, 1.0).astype(np.float32),
            ds_grid=uniform_grid(n_ds, 0.3).astype(np.float32),
        ),
        SolverConfig(
            name="Uniform ±2.0 + parabola",
            dE_grid=uniform_grid(n_dE, 2.0).astype(np.float32),
            ds_grid=uniform_grid(n_ds, 0.3).astype(np.float32),
        ),
        SolverConfig(
            name="Chebyshev ±2.0 + parabola",
            dE_grid=chebyshev_grid(n_dE, 2.0).astype(np.float32),
            ds_grid=uniform_grid(n_ds, 0.3).astype(np.float32),
        ),
        SolverConfig(
            name="Uniform ±1.0 + argmax",
            dE_grid=uniform_grid(n_dE, 1.0).astype(np.float32),
            ds_grid=uniform_grid(n_ds, 0.3).astype(np.float32),
            use_parabola=False,
        ),
    ]


# ============================================================================
# Core computation
# ============================================================================

def compute_precision_map(
    solver_cfg: SolverConfig,
    dE_test_range: tuple[float, float] = (-2.0, 2.0),
    ds_test_range: tuple[float, float] = (-0.3, 0.3),
    n_dE_test: int = 80,
    n_ds_test: int = 40,
    n_trials: int = 1,
    noise_sigma: float = 0.0,
    seed: int = 42,
) -> PrecisionMap:
    """Compute 2D precision map for a solver configuration.

    For each (δE, δσ) test point, generates exact Voigt spectrum(s),
    runs the solver, and records the estimation error.

    Args:
        solver_cfg: Solver configuration (grid + parabola flag)
        dE_test_range: δE test range
        ds_test_range: δσ test range
        n_dE_test: Number of δE test points
        n_ds_test: Number of δσ test points
        n_trials: Number of noise realizations per point (1 for NF)
        noise_sigma: Noise standard deviation (0 = noise-free)
        seed: Random seed

    Returns:
        PrecisionMap with bias/std over the 2D grid
    """
    rng = np.random.default_rng(seed)

    dE_test = np.linspace(dE_test_range[0], dE_test_range[1], n_dE_test,
                          dtype=np.float32)
    ds_test = np.linspace(ds_test_range[0], ds_test_range[1], n_ds_test,
                          dtype=np.float32)

    # Build dictionary
    ds_grid_clipped = solver_cfg.ds_grid[
        solver_cfg.ds_grid > -0.9 * float(np.min(SIGMAS))
    ]
    dc = build_dictionary(
        ENERGY, CENTERS, SIGMAS, GAMMA,
        dE_grid_override=solver_cfg.dE_grid,
        dsigma_grid_override=ds_grid_clipped,
    )

    # Allocate accumulators
    n_pts = n_dE_test * n_ds_test
    dE_errors = np.zeros((n_trials, n_pts), dtype=np.float64)
    ds_errors = np.zeros((n_trials, n_pts), dtype=np.float64)
    amp_errors = np.zeros((n_trials, n_pts), dtype=np.float64)

    # Generate test grid
    dE_2d, ds_2d = np.meshgrid(dE_test, ds_test, indexing='ij')
    dE_true = dE_2d.ravel().astype(np.float32)
    ds_true = ds_2d.ravel().astype(np.float32)

    for trial in range(n_trials):
        Y = generate_voigt_spectra(
            dE_true, ds_true, amp=1.0,
            noise_sigma=noise_sigma,
            rng=rng if noise_sigma > 0 else None,
        )

        if solver_cfg.use_parabola:
            amps, chi2, dE_out, ds_out, idx = solve_dict2d_parabola(Y, dc)
        else:
            # argmax only (no parabola) — use dict lookup + amplitude
            from toyomacro.voigtfit.dictionary_solver import solve_dict2d_amp_only
            amps, chi2, dE_out, ds_out, idx = solve_dict2d_amp_only(Y, dc)

        dE_errors[trial] = dE_out - dE_true
        ds_errors[trial] = ds_out - ds_true
        amp_errors[trial] = amps[0, :] - 1.0  # true amp = 1.0

    # Compute bias and std
    dE_bias_flat = np.mean(dE_errors, axis=0)
    ds_bias_flat = np.mean(ds_errors, axis=0)
    amp_bias_flat = np.mean(amp_errors, axis=0)

    if n_trials > 1:
        dE_std_flat = np.std(dE_errors, axis=0)
        ds_std_flat = np.std(ds_errors, axis=0)
        amp_std_flat = np.std(amp_errors, axis=0)
    else:
        dE_std_flat = np.zeros(n_pts)
        ds_std_flat = np.zeros(n_pts)
        amp_std_flat = np.zeros(n_pts)

    # Reshape to 2D
    shape = (n_dE_test, n_ds_test)
    return PrecisionMap(
        dE_values=dE_test,
        ds_values=ds_test,
        dE_bias=dE_bias_flat.reshape(shape),
        ds_bias=ds_bias_flat.reshape(shape),
        amp_bias=amp_bias_flat.reshape(shape),
        dE_std=dE_std_flat.reshape(shape),
        ds_std=ds_std_flat.reshape(shape),
        amp_std=amp_std_flat.reshape(shape),
        solver_name=solver_cfg.name,
        n_trials=n_trials,
    )


# ============================================================================
# Analysis utilities
# ============================================================================

def rmse_to_db(rmse: float, peak: float) -> float:
    if rmse < 1e-30:
        return 999.9
    return 20 * np.log10(peak / rmse)


def extract_contour_level(pmap: PrecisionMap, param: str,
                          threshold: float) -> tuple[float, float]:
    """Find max |δE| and |δσ| where RMSE < threshold.

    Returns (dE_limit, ds_limit) — the safe region extents.
    """
    if param == "dE":
        rmse_map = pmap.dE_rmse
    else:
        rmse_map = pmap.ds_rmse

    # For each δE value, find max |δσ| where RMSE < threshold
    # For each δσ value, find max |δE| where RMSE < threshold
    safe = rmse_map < threshold

    # δE limit: max |δE| where ALL δσ columns are safe
    dE_all_safe = np.all(safe, axis=1)  # (n_dE,) — row safe if ALL columns safe
    if np.any(dE_all_safe):
        safe_dE_indices = np.where(dE_all_safe)[0]
        dE_limit = max(abs(pmap.dE_values[safe_dE_indices[0]]),
                       abs(pmap.dE_values[safe_dE_indices[-1]]))
    else:
        dE_limit = 0.0

    # δσ limit: max |δσ| where ALL δE rows are safe
    ds_all_safe = np.all(safe, axis=0)  # (n_ds,)
    if np.any(ds_all_safe):
        safe_ds_indices = np.where(ds_all_safe)[0]
        ds_limit = max(abs(pmap.ds_values[safe_ds_indices[0]]),
                       abs(pmap.ds_values[safe_ds_indices[-1]]))
    else:
        ds_limit = 0.0

    return dE_limit, ds_limit


def compute_jitter_table(pmap: PrecisionMap,
                         db_levels: list[float] = [45, 40, 35, 30]) -> list[dict]:
    """Compute jitter limit table at various dB levels."""
    rows = []
    for db in db_levels:
        # Convert dB to RMSE thresholds
        dE_threshold = pmap.dE_values.ptp() / (10 ** (db / 20))  # range as peak
        ds_threshold = pmap.ds_values.ptp() / (10 ** (db / 20))

        # Find safe region for δE RMSE
        dE_rmse = pmap.dE_rmse
        ds_rmse = pmap.ds_rmse

        # For each δσ slice at center (δσ≈0), find max safe |δE|
        center_ds_idx = len(pmap.ds_values) // 2
        dE_profile = dE_rmse[:, center_ds_idx]  # RMSE along δE at δσ=0
        safe_dE = np.abs(dE_profile) < dE_threshold
        if np.any(safe_dE):
            safe_indices = np.where(safe_dE)[0]
            dE_limit = max(abs(pmap.dE_values[safe_indices[0]]),
                           abs(pmap.dE_values[safe_indices[-1]]))
        else:
            dE_limit = 0.0

        # For each δE slice at center (δE≈0), find max safe |δσ|
        center_dE_idx = len(pmap.dE_values) // 2
        ds_profile = ds_rmse[center_dE_idx, :]
        safe_ds = np.abs(ds_profile) < ds_threshold
        if np.any(safe_ds):
            safe_indices = np.where(safe_ds)[0]
            ds_limit = max(abs(pmap.ds_values[safe_indices[0]]),
                           abs(pmap.ds_values[safe_indices[-1]]))
        else:
            ds_limit = 0.0

        rows.append({
            'db': db,
            'dE_limit': dE_limit,
            'ds_limit': ds_limit,
            'dE_over_sigma': dE_limit / SIGMA_NOM,
            'ds_over_sigma': ds_limit / SIGMA_NOM,
        })
    return rows


def print_precision_map_stats(pmap: PrecisionMap):
    """Print summary statistics for a precision map."""
    dE_rmse = pmap.dE_rmse
    ds_rmse = pmap.ds_rmse

    print(f"  {pmap.solver_name} (n_trials={pmap.n_trials})")
    print(f"    δE RMSE: mean={np.mean(dE_rmse):.6f}, "
          f"max={np.max(dE_rmse):.6f}, "
          f"median={np.median(dE_rmse):.6f}")
    print(f"    δσ RMSE: mean={np.mean(ds_rmse):.6f}, "
          f"max={np.max(ds_rmse):.6f}, "
          f"median={np.median(ds_rmse):.6f}")

    # Center region stats (|δE| < 0.5, |δσ| < 0.1)
    dE_center = np.abs(pmap.dE_values) < 0.5
    ds_center = np.abs(pmap.ds_values) < 0.1
    center_mask = np.outer(dE_center, ds_center)
    if center_mask.sum() > 0:
        print(f"    Center (|δE|<0.5, |δσ|<0.1): "
              f"δE RMSE={np.mean(dE_rmse[center_mask]):.6f}, "
              f"δσ RMSE={np.mean(ds_rmse[center_mask]):.6f}")

    # Edge region stats (|δE| > 1.5)
    dE_edge = np.abs(pmap.dE_values) > 1.5
    edge_mask = np.outer(dE_edge, np.ones(len(pmap.ds_values), dtype=bool))
    if edge_mask.sum() > 0:
        print(f"    Edge   (|δE|>1.5):          "
              f"δE RMSE={np.mean(dE_rmse[edge_mask]):.6f}, "
              f"δσ RMSE={np.mean(ds_rmse[edge_mask]):.6f}")


def print_breakdown_boundary(pmap: PrecisionMap, threshold: float = 0.01):
    """Print the breakdown boundary as ASCII art."""
    dE_rmse = pmap.dE_rmse
    ds_rmse = pmap.ds_rmse
    combined = np.maximum(dE_rmse, ds_rmse)

    n_dE = len(pmap.dE_values)
    n_ds = len(pmap.ds_values)

    # Downsample for display
    step_dE = max(1, n_dE // 20)
    step_ds = max(1, n_ds // 20)

    print(f"\n    Breakdown map (threshold={threshold:.4f}, "
          f"● = safe, × = breakdown):")
    print(f"    {'δσ →':>8}", end="")
    for j in range(0, n_ds, step_ds):
        print(f"{pmap.ds_values[j]:6.2f}", end="")
    print()

    for i in range(n_dE - 1, -1, -step_dE):
        print(f"    {pmap.dE_values[i]:6.2f}  ", end="")
        for j in range(0, n_ds, step_ds):
            if combined[i, j] < threshold:
                print("  ●   ", end="")
            else:
                print("  ×   ", end="")
        print()


# ============================================================================
# Experiments
# ============================================================================

def experiment_2_1():
    """Noise-free precision map for 4 solver configurations."""
    print("=" * 76)
    print("Experiment 2.1: Noise-Free Precision Map")
    print("=" * 76)

    configs = make_solver_configs()
    maps: dict[str, PrecisionMap] = {}

    for cfg in configs:
        print(f"\n  Computing: {cfg.name} ({cfg.n_entries} entries)...")
        t0 = time.perf_counter()
        pmap = compute_precision_map(
            cfg,
            dE_test_range=(-2.0, 2.0),
            ds_test_range=(-0.3, 0.3),
            n_dE_test=80,
            n_ds_test=40,
            n_trials=1,
            noise_sigma=0.0,
        )
        elapsed = time.perf_counter() - t0
        maps[cfg.name] = pmap
        print_precision_map_stats(pmap)
        print(f"    Time: {elapsed:.1f}s")

    # Summary comparison
    print("\n" + "-" * 76)
    print(f"{'Solver':<30} {'δE mean':>10} {'δE max':>10} "
          f"{'δσ mean':>10} {'δσ max':>10}")
    print("-" * 76)
    for name, pmap in maps.items():
        de_r = pmap.dE_rmse
        ds_r = pmap.ds_rmse
        print(f"{name:<30} {np.mean(de_r):10.6f} {np.max(de_r):10.6f} "
              f"{np.mean(ds_r):10.6f} {np.max(ds_r):10.6f}")

    # Show breakdown map for key configs
    for name in ["Chebyshev ±2.0 + parabola", "Uniform ±1.0 + parabola"]:
        if name in maps:
            print(f"\n  --- {name} ---")
            print_breakdown_boundary(maps[name], threshold=0.01)

    return maps


def experiment_2_2():
    """Bias-Variance decomposition with moderate noise."""
    print("\n" + "=" * 76)
    print("Experiment 2.2: Bias-Variance Decomposition (Moderate Noise)")
    print("=" * 76)

    # Use Chebyshev ±2.0 (best extended range) and Uniform ±1.0 (baseline)
    configs = [
        SolverConfig(
            name="Chebyshev ±2.0",
            dE_grid=chebyshev_grid(29, 2.0).astype(np.float32),
            ds_grid=uniform_grid(31, 0.3).astype(np.float32),
        ),
        SolverConfig(
            name="Uniform ±1.0",
            dE_grid=uniform_grid(29, 1.0).astype(np.float32),
            ds_grid=uniform_grid(31, 0.3).astype(np.float32),
        ),
    ]

    # Moderate noise level: ~1% of peak height
    noise_sigma = 0.01

    maps: dict[str, PrecisionMap] = {}
    for cfg in configs:
        print(f"\n  Computing: {cfg.name} (10 trials, noise σ={noise_sigma})...")
        t0 = time.perf_counter()
        pmap = compute_precision_map(
            cfg,
            dE_test_range=(-2.0, 2.0),
            ds_test_range=(-0.3, 0.3),
            n_dE_test=50,
            n_ds_test=25,
            n_trials=10,
            noise_sigma=noise_sigma,
            seed=123,
        )
        elapsed = time.perf_counter() - t0
        maps[cfg.name] = pmap
        print_precision_map_stats(pmap)

        # Bias vs variance dominance
        dE_bias2 = pmap.dE_bias**2
        dE_var = pmap.dE_std**2
        bias_frac = np.mean(dE_bias2) / (np.mean(dE_bias2) + np.mean(dE_var) + 1e-30)
        print(f"    δE: bias²/(bias²+var) = {bias_frac:.1%} "
              f"({'bias-dominated' if bias_frac > 0.5 else 'variance-dominated'})")

        ds_bias2 = pmap.ds_bias**2
        ds_var = pmap.ds_std**2
        ds_bias_frac = np.mean(ds_bias2) / (np.mean(ds_bias2) + np.mean(ds_var) + 1e-30)
        print(f"    δσ: bias²/(bias²+var) = {ds_bias_frac:.1%} "
              f"({'bias-dominated' if ds_bias_frac > 0.5 else 'variance-dominated'})")

        # Regional analysis
        n_dE = len(pmap.dE_values)
        n_ds = len(pmap.ds_values)
        center_dE = np.abs(pmap.dE_values) < 0.5
        edge_dE = np.abs(pmap.dE_values) > 1.5

        for region_name, mask_dE in [("Center |δE|<0.5", center_dE),
                                      ("Edge |δE|>1.5", edge_dE)]:
            if not np.any(mask_dE):
                continue
            mask_2d = np.outer(mask_dE, np.ones(n_ds, dtype=bool))
            b2 = np.mean(dE_bias2[mask_2d])
            v = np.mean(dE_var[mask_2d])
            frac = b2 / (b2 + v + 1e-30)
            print(f"      {region_name}: δE bias frac = {frac:.1%}, "
                  f"bias={np.sqrt(b2):.6f}, std={np.sqrt(v):.6f}")

        print(f"    Time: {elapsed:.1f}s")

    return maps


def experiment_2_3(nf_maps: dict[str, PrecisionMap]):
    """Identify parabola breakdown boundaries from NF precision maps."""
    print("\n" + "=" * 76)
    print("Experiment 2.3: Parabola Breakdown Boundaries")
    print("=" * 76)

    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]

    for name, pmap in nf_maps.items():
        print(f"\n  {name}:")
        combined_rmse = np.maximum(pmap.dE_rmse, pmap.ds_rmse)

        for thresh in thresholds:
            safe = combined_rmse < thresh
            safe_frac = safe.mean()

            # Find safe δE range at δσ=0 center slice
            center_ds = len(pmap.ds_values) // 2
            dE_profile = combined_rmse[:, center_ds]
            safe_dE = dE_profile < thresh
            if np.any(safe_dE):
                safe_indices = np.where(safe_dE)[0]
                dE_lo = pmap.dE_values[safe_indices[0]]
                dE_hi = pmap.dE_values[safe_indices[-1]]
                dE_span = f"[{dE_lo:+.2f}, {dE_hi:+.2f}]"
            else:
                dE_span = "NONE"

            # Find safe δσ range at δE=0 center slice
            center_dE = len(pmap.dE_values) // 2
            ds_profile = combined_rmse[center_dE, :]
            safe_ds = ds_profile < thresh
            if np.any(safe_ds):
                safe_indices = np.where(safe_ds)[0]
                ds_lo = pmap.ds_values[safe_indices[0]]
                ds_hi = pmap.ds_values[safe_indices[-1]]
                ds_span = f"[{ds_lo:+.3f}, {ds_hi:+.3f}]"
            else:
                ds_span = "NONE"

            print(f"    thresh={thresh:.3f}: "
                  f"safe={safe_frac:.0%}, "
                  f"δE range={dE_span}, δσ range={ds_span}")


def experiment_2_4(nf_maps: dict[str, PrecisionMap]):
    """Quantify jitter limits at various precision levels."""
    print("\n" + "=" * 76)
    print("Experiment 2.4: Jitter Limit Quantification")
    print("=" * 76)

    # Use absolute RMSE thresholds (not dB-based, simpler and more interpretable)
    rmse_thresholds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

    for name, pmap in nf_maps.items():
        print(f"\n  {name}:")
        print(f"    {'RMSE thresh':>12} {'δE limit':>10} {'δE/σ':>8} "
              f"{'δσ limit':>10} {'δσ/σ':>8} {'Safe area':>10}")
        print("    " + "-" * 62)

        for thresh in rmse_thresholds:
            # δE limit: max |δE| where δE RMSE < thresh (at δσ=0)
            center_ds = len(pmap.ds_values) // 2
            dE_rmse_slice = pmap.dE_rmse[:, center_ds]
            safe_dE = dE_rmse_slice < thresh
            if np.any(safe_dE):
                si = np.where(safe_dE)[0]
                dE_lim = max(abs(pmap.dE_values[si[0]]),
                             abs(pmap.dE_values[si[-1]]))
            else:
                dE_lim = 0.0

            # δσ limit: max |δσ| where δσ RMSE < thresh (at δE=0)
            center_dE = len(pmap.dE_values) // 2
            ds_rmse_slice = pmap.ds_rmse[center_dE, :]
            safe_ds = ds_rmse_slice < thresh
            if np.any(safe_ds):
                si = np.where(safe_ds)[0]
                ds_lim = max(abs(pmap.ds_values[si[0]]),
                             abs(pmap.ds_values[si[-1]]))
            else:
                ds_lim = 0.0

            # Combined safe area fraction
            combined = np.maximum(pmap.dE_rmse, pmap.ds_rmse)
            safe_frac = (combined < thresh).mean()

            print(f"    {thresh:12.3f} {dE_lim:10.3f} {dE_lim/SIGMA_NOM:8.2f} "
                  f"{ds_lim:10.3f} {ds_lim/SIGMA_NOM:8.2f} {safe_frac:10.1%}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Jitter Limits Benchmark (Phase 3 Part 2)")
    print(f"σ_nom={SIGMA_NOM:.4f}, γ={GAMMA}, center={CENTER}")
    print()

    # Experiment 2.1: NF precision maps
    nf_maps = experiment_2_1()

    # Experiment 2.2: Bias-Variance
    bv_maps = experiment_2_2()

    # Experiment 2.3: Breakdown boundaries (from NF maps)
    experiment_2_3(nf_maps)

    # Experiment 2.4: Jitter limit table (from NF maps)
    experiment_2_4(nf_maps)

    print("\n" + "=" * 76)
    print("All experiments complete.")
