"""
N-component Scaling Benchmark
===========================================

Measures solver throughput (spectra/s) vs n_comp for three solvers:
  - amp_only:  HybridPipeline.process_rowmajor_fast
  - 4-step:    HybridPipeline.process_rowmajor_extended_3param
  - dict2d:    solve_alternating_projection (parabola)

Usage:
    uv run python -m toyomacro.voigtfit.benchmarks.bench_ncomp_scaling
    uv run python -m toyomacro.voigtfit.benchmarks.bench_ncomp_scaling --n-spectra 50000
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np

from toyomacro.voigtfit.multipeak_config import ComponentConfig, MultiPeakConfig
from toyomacro.voigtfit.multipeak_solver import (
    build_multipeak_dictionaries,
    solve_alternating_projection,
)
from toyomacro.voigtfit.pipeline import HybridPipeline
from toyomacro.voigtfit.synthetic import (
    ELEMENT_PROFILES,
    ElementProfile,
    make_synthetic_configs,
    make_synthetic_spectra,
)
from toyomacro.voigtfit.weight_cache import WeightMatrixCache

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

N_COMP_LIST = [1, 2, 3, 5, 10]
N_SPECTRA = 100_000
N_WARMUP = 1_000
SNR_DB = 40.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _singlet_profile() -> ElementProfile:
    """Si2p profile forced to singlet (no SO splitting)."""
    p = ELEMENT_PROFILES["Si2p"]
    return ElementProfile(
        name=p.name,
        sigma=p.sigma,
        gamma=p.gamma,
        so_split=None,
        branch_ratio=None,
        typical_dE=p.typical_dE,
        n_comp_range=p.n_comp_range,
    )


def _make_energy_axis(configs: list[ComponentConfig], sigma: float) -> np.ndarray:
    """Auto energy axis covering all component centres + 5 sigma margin."""
    positions = [c.center for c in configs]
    margin = 5 * sigma
    E_min = min(positions) - margin
    E_max = max(positions) + margin
    n_channels = max(128, int((E_max - E_min) / 0.05))
    return np.linspace(E_min, E_max, n_channels, dtype=np.float64)


def _peak_config(configs: list[ComponentConfig]) -> dict:
    """Build peak_config dict for HybridPipeline."""
    return {
        "centers": np.array([c.center for c in configs], dtype=np.float64),
        "sigmas": np.array([c.sigma for c in configs], dtype=np.float64),
        "gamma": float(configs[0].gamma),
    }


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    solver: str
    n_comp: int
    rate: float  # spectra / s
    elapsed: float  # s


def bench_amp_only(
    spectra: np.ndarray,
    energy: np.ndarray,
    peak_config: dict,
    n_warmup: int = N_WARMUP,
) -> BenchResult:
    """Benchmark amp-only solver."""
    cache = WeightMatrixCache()
    pipeline = HybridPipeline(cache, enable_stage2=False)
    n_comp = len(peak_config["centers"])

    # Warmup
    pipeline.process_rowmajor_fast(
        spectra[:n_warmup], "bench", "1s", energy, peak_config,
    )

    t0 = time.perf_counter()
    pipeline.process_rowmajor_fast(spectra, "bench", "1s", energy, peak_config)
    elapsed = time.perf_counter() - t0

    return BenchResult("amp_only", n_comp, spectra.shape[0] / elapsed, elapsed)


def bench_4step(
    spectra: np.ndarray,
    energy: np.ndarray,
    peak_config: dict,
    n_warmup: int = N_WARMUP,
) -> BenchResult:
    """Benchmark 4-step solver."""
    cache = WeightMatrixCache()
    pipeline = HybridPipeline(cache, enable_stage2=False)
    n_comp = len(peak_config["centers"])

    # Warmup
    pipeline.process_rowmajor_extended_3param(
        spectra[:n_warmup], "bench", "1s", energy, peak_config, n_steps=4,
    )

    t0 = time.perf_counter()
    pipeline.process_rowmajor_extended_3param(
        spectra, "bench", "1s", energy, peak_config, n_steps=4,
    )
    elapsed = time.perf_counter() - t0

    return BenchResult("4-step", n_comp, spectra.shape[0] / elapsed, elapsed)


def bench_dict2d(
    spectra: np.ndarray,
    energy: np.ndarray,
    configs: list[ComponentConfig],
    n_warmup: int = N_WARMUP,
) -> BenchResult:
    """Benchmark dict2d via alternating projection."""
    n_comp = len(configs)
    mp_config = MultiPeakConfig(peaks=configs, energy_axis=energy.astype(np.float32))
    mp_config.constrain_dE_ranges()
    dicts = build_multipeak_dictionaries(mp_config)

    # Warmup
    solve_alternating_projection(
        spectra[:n_warmup], dicts, n_iterations=3, parabola_dE=True,
    )

    t0 = time.perf_counter()
    solve_alternating_projection(
        spectra, dicts, n_iterations=3, parabola_dE=True,
    )
    elapsed = time.perf_counter() - t0

    return BenchResult("dict2d", n_comp, spectra.shape[0] / elapsed, elapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_scaling_benchmark(
    n_comp_list: list[int] | None = None,
    n_spectra: int = N_SPECTRA,
) -> list[BenchResult]:
    """Run the full scaling benchmark, return results."""
    if n_comp_list is None:
        n_comp_list = N_COMP_LIST

    profile = _singlet_profile()
    results: list[BenchResult] = []

    for n_comp in n_comp_list:
        print(f"\n--- n_comp = {n_comp} ---")

        # Generate data
        configs, _ = make_synthetic_configs(profile, n_comp=n_comp)
        energy = _make_energy_axis(configs, profile.sigma)
        spectra, _ = make_synthetic_spectra(
            configs, profile, n_spectra=n_spectra,
            E_range=(energy[0], energy[-1]),
            n_channels=len(energy),
            snr_db=SNR_DB,
        )
        pc = _peak_config(configs)

        print(f"  spectra: {spectra.shape}, energy: {len(energy)} ch")

        # amp_only
        r = bench_amp_only(spectra, energy, pc)
        results.append(r)
        print(f"  amp_only:  {r.rate:>12,.0f} spec/s  ({r.elapsed:.3f}s)")

        # 4-step
        r = bench_4step(spectra, energy, pc)
        results.append(r)
        print(f"  4-step:    {r.rate:>12,.0f} spec/s  ({r.elapsed:.3f}s)")

        # dict2d (alternating projection)
        r = bench_dict2d(spectra, energy, configs)
        results.append(r)
        print(f"  dict2d:    {r.rate:>12,.0f} spec/s  ({r.elapsed:.3f}s)")

    return results


def print_table(results: list[BenchResult]) -> None:
    """Print a summary table."""
    print(f"\n{'='*60}")
    print(f"{'Solver':<12} {'n_comp':>6} {'Rate (spec/s)':>15} {'Elapsed (s)':>12}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r.solver:<12} {r.n_comp:>6} {r.rate:>15,.0f} {r.elapsed:>12.3f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="N-comp scaling benchmark")
    parser.add_argument(
        "--n-spectra", type=int, default=N_SPECTRA,
        help=f"Number of spectra (default {N_SPECTRA:,})",
    )
    parser.add_argument(
        "--n-comp", type=int, nargs="+", default=None,
        help=f"Component counts (default {N_COMP_LIST})",
    )
    args = parser.parse_args()

    results = run_scaling_benchmark(
        n_comp_list=args.n_comp,
        n_spectra=args.n_spectra,
    )
    print_table(results)


if __name__ == "__main__":
    main()
