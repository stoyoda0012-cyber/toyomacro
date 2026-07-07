"""
MultiPeak Precision Sweep
=======================================

2-component PSNR × separation distance × noise → scorecard.

Evaluates production ``process_multipeak()`` across:
  - Peak separation: 2σ … 10σ
  - Gaussian noise: NF, SNR=1000, 100, 10
  - Amplitude ratio: 1:1, 1:3

Metrics per cell:
  - δE PSNR (dB) — average of comp 1+2
  - δσ PSNR (dB)
  - δE RMSE (meV), δσ RMSE (meV), amplitude RMSE

Component swap correction: per-spectrum Hungarian-lite (swap if it
reduces total δE error) to handle ambiguous assignment at close separation.

Usage:
    uv run python -m toyomacro.voigtfit.benchmarks.bench_multipeak_sweep
    uv run python -m toyomacro.voigtfit.benchmarks.bench_multipeak_sweep --n-spectra 10000
"""

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toyomacro.voigtfit.multipeak_config import ComponentConfig, MultiPeakConfig
from toyomacro.voigtfit.multipeak_solver import process_multipeak

# ============================================================================
# Constants
# ============================================================================

FWHM_TO_SIGMA = 1.0 / (2 * np.sqrt(2 * np.log(2)))
SIGMA_NOM = 1.0 * FWHM_TO_SIGMA  # 0.4247 eV
GAMMA = 0.125  # eV
CENTER_MID = 285.0  # eV (midpoint between two peaks)
ENERGY = np.linspace(278.0, 292.0, 201, dtype=np.float32)

# Sweep grid
SEPARATIONS_SIGMA = [2.0, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]
SNR_LEVELS = {
    "NF": float("inf"),
    "SNR1k": 1000.0,
    "SNR100": 100.0,
    "SNR10": 10.0,
}
AMP_RATIOS = {
    "1:1": (1.0, 1.0),
    "1:3": (1.0, 3.0),
}

N_SPECTRA = 100_000


# ============================================================================
# Spectra generation
# ============================================================================


def generate_2comp_spectra_exact(
    n_spectra: int,
    center1: float,
    center2: float,
    amp1: float,
    amp2: float,
    snr: float,
    seed: int,
    dE_range: float = 0.3,
    ds_range: float = 0.08,
) -> tuple[np.ndarray, dict]:
    """Generate 2-component Voigt spectra with exact (Faddeeva) profiles.

    Each profile is peak-normalized to 1 before scaling by amplitude,
    matching bench_multipeak_4d.py convention.

    Returns:
        Y: (n_spectra, n_E) float32
        gt: ground truth dict with dE1, dE2, ds1, ds2, a1, a2
    """
    from scipy.special import wofz

    rng = np.random.default_rng(seed)

    dE1 = rng.uniform(-dE_range, dE_range, n_spectra).astype(np.float32)
    dE2 = rng.uniform(-dE_range, dE_range, n_spectra).astype(np.float32)
    ds1 = rng.uniform(-ds_range, ds_range, n_spectra).astype(np.float32)
    ds2 = rng.uniform(-ds_range, ds_range, n_spectra).astype(np.float32)

    # Vectorized exact Voigt
    energy = ENERGY.astype(np.float64)
    SQRT2 = np.sqrt(2.0)
    SQRT2PI = np.sqrt(2.0 * np.pi)

    c1 = (center1 + dE1).astype(np.float64)[:, None]
    s1 = np.maximum(SIGMA_NOM + ds1, 0.01).astype(np.float64)[:, None]
    z1 = ((energy[None, :] - c1) + 1j * GAMMA) / (s1 * SQRT2)
    P1 = np.real(wofz(z1)) / (s1 * SQRT2PI)

    c2 = (center2 + dE2).astype(np.float64)[:, None]
    s2 = np.maximum(SIGMA_NOM + ds2, 0.01).astype(np.float64)[:, None]
    z2 = ((energy[None, :] - c2) + 1j * GAMMA) / (s2 * SQRT2)
    P2 = np.real(wofz(z2)) / (s2 * SQRT2PI)

    # Peak-normalize each profile
    P1 /= P1.max(axis=1, keepdims=True) + 1e-30
    P2 /= P2.max(axis=1, keepdims=True) + 1e-30

    Y = (amp1 * P1 + amp2 * P2).astype(np.float32)

    # Gaussian noise (signal-relative)
    if np.isfinite(snr) and snr > 0:
        signal_max = float(Y.max())
        noise_std = signal_max / snr
        Y += rng.normal(0, noise_std, Y.shape).astype(np.float32)

    gt = dict(
        dE1=dE1,
        dE2=dE2,
        ds1=ds1,
        ds2=ds2,
        a1=np.full(n_spectra, amp1, np.float32),
        a2=np.full(n_spectra, amp2, np.float32),
    )
    return Y, gt


# ============================================================================
# Component swap correction
# ============================================================================


def _correct_swaps(
    delta_E: np.ndarray,
    delta_sigma: np.ndarray,
    amplitudes: np.ndarray,
    gt: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-spectrum swap correction for ambiguous component assignment.

    At close separations, the solver may assign comp1↔comp2 inconsistently.
    For each spectrum, check if swapping reduces the total δE squared error.

    Returns:
        (dE_corr, ds_corr, amp_corr, n_swapped)
    """
    err_orig = (delta_E[:, 0] - gt["dE1"]) ** 2 + (delta_E[:, 1] - gt["dE2"]) ** 2
    err_swap = (delta_E[:, 1] - gt["dE1"]) ** 2 + (delta_E[:, 0] - gt["dE2"]) ** 2

    swap = err_swap < err_orig

    dE_c = delta_E.copy()
    ds_c = delta_sigma.copy()
    amp_c = amplitudes.copy()

    dE_c[swap, 0] = delta_E[swap, 1]
    dE_c[swap, 1] = delta_E[swap, 0]
    ds_c[swap, 0] = delta_sigma[swap, 1]
    ds_c[swap, 1] = delta_sigma[swap, 0]
    amp_c[swap, 0] = amplitudes[swap, 1]
    amp_c[swap, 1] = amplitudes[swap, 0]

    return dE_c, ds_c, amp_c, int(swap.sum())


# ============================================================================
# Metrics
# ============================================================================


@dataclass
class SweepCell:
    """Result for one (separation, noise, ratio) cell."""

    sep_sigma: float
    snr_name: str
    ratio_name: str
    n_spectra: int

    # Per-component RMSE
    dE1_rmse: float
    dE2_rmse: float
    ds1_rmse: float
    ds2_rmse: float
    a1_rmse: float
    a2_rmse: float

    # PSNR (average of comp1+comp2)
    dE_psnr: float
    ds_psnr: float
    amp_psnr: float

    # Swap statistics
    n_swapped: int

    # Performance
    time_s: float
    throughput: float


def _psnr(gt: np.ndarray, rec: np.ndarray, signal_range: float) -> float:
    """PSNR = 10 log10(range² / MSE)."""
    mse = float(np.mean((gt - rec) ** 2))
    if mse < 1e-30:
        return 999.0
    return 10 * np.log10(signal_range**2 / mse)


# ============================================================================
# Single cell evaluation
# ============================================================================


def evaluate_cell(
    sep_sigma: float,
    snr_name: str,
    snr_val: float,
    ratio_name: str,
    amp1: float,
    amp2: float,
    n_spectra: int,
    seed: int,
    auto_constrain: bool = True,
) -> SweepCell:
    """Run one cell of the sweep."""
    sep_eV = sep_sigma * SIGMA_NOM
    c1 = CENTER_MID + sep_eV / 2
    c2 = CENTER_MID - sep_eV / 2

    # Generate
    Y, gt = generate_2comp_spectra_exact(
        n_spectra, c1, c2, amp1, amp2, snr_val, seed
    )

    # Solve (suppress separation warnings for the sweep)
    config = MultiPeakConfig(
        peaks=[
            ComponentConfig(
                center=c1,
                sigma=SIGMA_NOM,
                gamma=GAMMA,
                dE_range=2.0,
                ds_range=0.3,
                n_dE=10,
                n_ds=31,
            ),
            ComponentConfig(
                center=c2,
                sigma=SIGMA_NOM,
                gamma=GAMMA,
                dE_range=2.0,
                ds_range=0.3,
                n_dE=10,
                n_ds=31,
            ),
        ],
        energy_axis=ENERGY,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        result = process_multipeak(
            Y, config, parabola_dE=True, parabola_ds=False,
            auto_constrain=auto_constrain,
        )
        elapsed = time.perf_counter() - t0

    # Swap correction
    dE, ds, amp, n_swapped = _correct_swaps(
        result.delta_E, result.delta_sigma, result.amplitudes, gt
    )

    # RMSE
    dE1_rmse = float(np.sqrt(np.mean((dE[:, 0] - gt["dE1"]) ** 2)))
    dE2_rmse = float(np.sqrt(np.mean((dE[:, 1] - gt["dE2"]) ** 2)))
    ds1_rmse = float(np.sqrt(np.mean((ds[:, 0] - gt["ds1"]) ** 2)))
    ds2_rmse = float(np.sqrt(np.mean((ds[:, 1] - gt["ds2"]) ** 2)))
    a1_rmse = float(np.sqrt(np.mean((amp[:, 0] - gt["a1"]) ** 2)))
    a2_rmse = float(np.sqrt(np.mean((amp[:, 1] - gt["a2"]) ** 2)))

    # PSNR
    dE_jitter_range = 0.6  # 2 × 0.3
    ds_jitter_range = 0.16  # 2 × 0.08
    amp_max = max(amp1, amp2)

    dE_psnr = (
        _psnr(gt["dE1"], dE[:, 0], dE_jitter_range)
        + _psnr(gt["dE2"], dE[:, 1], dE_jitter_range)
    ) / 2
    ds_psnr = (
        _psnr(gt["ds1"], ds[:, 0], ds_jitter_range)
        + _psnr(gt["ds2"], ds[:, 1], ds_jitter_range)
    ) / 2
    amp_psnr = (
        _psnr(gt["a1"] / amp_max, amp[:, 0] / amp_max, 1.0)
        + _psnr(gt["a2"] / amp_max, amp[:, 1] / amp_max, 1.0)
    ) / 2

    throughput = n_spectra / elapsed if elapsed > 0 else float("inf")

    return SweepCell(
        sep_sigma=sep_sigma,
        snr_name=snr_name,
        ratio_name=ratio_name,
        n_spectra=n_spectra,
        dE1_rmse=dE1_rmse,
        dE2_rmse=dE2_rmse,
        ds1_rmse=ds1_rmse,
        ds2_rmse=ds2_rmse,
        a1_rmse=a1_rmse,
        a2_rmse=a2_rmse,
        dE_psnr=dE_psnr,
        ds_psnr=ds_psnr,
        amp_psnr=amp_psnr,
        n_swapped=n_swapped,
        time_s=elapsed,
        throughput=throughput,
    )


# ============================================================================
# Sweep runner
# ============================================================================


def run_sweep(
    separations: list[float] | None = None,
    snr_levels: dict[str, float] | None = None,
    amp_ratios: dict[str, tuple[float, float]] | None = None,
    n_spectra: int = N_SPECTRA,
    auto_constrain: bool = True,
) -> list[SweepCell]:
    """Run full 3D sweep: separation × noise × amplitude ratio."""
    if separations is None:
        separations = SEPARATIONS_SIGMA
    if snr_levels is None:
        snr_levels = SNR_LEVELS
    if amp_ratios is None:
        amp_ratios = AMP_RATIOS

    total_cells = len(separations) * len(snr_levels) * len(amp_ratios)

    print(f"{'=' * 80}")
    print("MultiPeak Precision Sweep: Alternating Projection")
    print(f"{'=' * 80}")
    print(f"σ = {SIGMA_NOM:.4f} eV, γ = {GAMMA} eV, n = {n_spectra // 1000}K per cell")
    constrain_label = "ON" if auto_constrain else "OFF"
    print(
        f"Grid: 10×31 = 310 entries/comp, 3 iterations, "
        f"parabola_dE=True, auto_constrain={constrain_label}"
    )
    print(
        f"Cells: {len(separations)} sep × {len(snr_levels)} noise "
        f"× {len(amp_ratios)} ratio = {total_cells}"
    )
    print()

    results = []
    cell_idx = 0
    t_total = time.perf_counter()

    for ratio_name, (a1, a2) in amp_ratios.items():
        print(f"--- Ratio {ratio_name} (a1={a1}, a2={a2}) ---")

        for sep in separations:
            for snr_name, snr_val in snr_levels.items():
                cell_idx += 1
                seed = cell_idx * 1000 + 42

                cell = evaluate_cell(
                    sep, snr_name, snr_val, ratio_name, a1, a2,
                    n_spectra, seed, auto_constrain,
                )
                results.append(cell)

                swap_pct = cell.n_swapped / n_spectra * 100
                print(
                    f"  [{cell_idx:2d}/{total_cells}] "
                    f"Δc={sep:4.1f}σ {snr_name:<6s}: "
                    f"δE={cell.dE_psnr:5.1f}dB  δσ={cell.ds_psnr:5.1f}dB  "
                    f"amp={cell.amp_psnr:5.1f}dB  "
                    f"swap={swap_pct:4.1f}%  "
                    f"{cell.throughput / 1e6:.2f}M/s"
                )

    elapsed = time.perf_counter() - t_total
    total_spectra = total_cells * n_spectra
    print(
        f"\nTotal: {elapsed:.1f}s "
        f"({total_spectra / 1e6:.1f}M spectra, "
        f"{total_spectra / elapsed / 1e6:.1f}M/s effective)"
    )

    return results


# ============================================================================
# Scorecard
# ============================================================================


def print_scorecard(results: list[SweepCell]) -> None:
    """Print formatted scorecard tables."""
    for ratio_name in dict.fromkeys(r.ratio_name for r in results):
        subset = [r for r in results if r.ratio_name == ratio_name]
        separations = sorted(set(r.sep_sigma for r in subset))
        snr_names = list(dict.fromkeys(r.snr_name for r in subset))

        lookup: dict[tuple[float, str], SweepCell] = {}
        for r in subset:
            lookup[(r.sep_sigma, r.snr_name)] = r

        print(f"\n{'=' * 70}")
        print(f"SCORECARD — Ratio {ratio_name}")
        print(f"{'=' * 70}")

        # δE PSNR table
        print("\n  δE PSNR (dB):")
        print(f"  {'Δc/σ':<8}" + "".join(f"{s:<10}" for s in snr_names))
        print(f"  {'-' * (8 + 10 * len(snr_names))}")
        for sep in separations:
            row = f"  {sep:<8.1f}"
            for sn in snr_names:
                cell = lookup.get((sep, sn))
                row += f"{cell.dE_psnr:<10.1f}" if cell else f"{'—':<10}"
            print(row)

        # δσ PSNR table
        print("\n  δσ PSNR (dB):")
        print(f"  {'Δc/σ':<8}" + "".join(f"{s:<10}" for s in snr_names))
        print(f"  {'-' * (8 + 10 * len(snr_names))}")
        for sep in separations:
            row = f"  {sep:<8.1f}"
            for sn in snr_names:
                cell = lookup.get((sep, sn))
                row += f"{cell.ds_psnr:<10.1f}" if cell else f"{'—':<10}"
            print(row)

        # Amplitude PSNR table
        print("\n  Amplitude PSNR (dB):")
        print(f"  {'Δc/σ':<8}" + "".join(f"{s:<10}" for s in snr_names))
        print(f"  {'-' * (8 + 10 * len(snr_names))}")
        for sep in separations:
            row = f"  {sep:<8.1f}"
            for sn in snr_names:
                cell = lookup.get((sep, sn))
                row += f"{cell.amp_psnr:<10.1f}" if cell else f"{'—':<10}"
            print(row)

        # δE RMSE table (meV)
        print("\n  δE RMSE (meV, avg comp1+2):")
        print(f"  {'Δc/σ':<8}" + "".join(f"{s:<10}" for s in snr_names))
        print(f"  {'-' * (8 + 10 * len(snr_names))}")
        for sep in separations:
            row = f"  {sep:<8.1f}"
            for sn in snr_names:
                cell = lookup.get((sep, sn))
                if cell:
                    rmse_meV = (cell.dE1_rmse + cell.dE2_rmse) / 2 * 1000
                    row += f"{rmse_meV:<10.1f}"
                else:
                    row += f"{'—':<10}"
            print(row)

        # Swap % table
        print("\n  Component swap % (per-spectrum correction):")
        print(f"  {'Δc/σ':<8}" + "".join(f"{s:<10}" for s in snr_names))
        print(f"  {'-' * (8 + 10 * len(snr_names))}")
        for sep in separations:
            row = f"  {sep:<8.1f}"
            for sn in snr_names:
                cell = lookup.get((sep, sn))
                if cell:
                    pct = cell.n_swapped / cell.n_spectra * 100
                    row += f"{pct:<10.1f}"
                else:
                    row += f"{'—':<10}"
            print(row)


# ============================================================================
# Save / Load
# ============================================================================


def save_results(results: list[SweepCell], path: str) -> None:
    """Save results to NPZ."""
    data = {
        "sep_sigma": np.array([r.sep_sigma for r in results]),
        "snr_name": np.array([r.snr_name for r in results]),
        "ratio_name": np.array([r.ratio_name for r in results]),
        "dE_psnr": np.array([r.dE_psnr for r in results]),
        "ds_psnr": np.array([r.ds_psnr for r in results]),
        "amp_psnr": np.array([r.amp_psnr for r in results]),
        "dE1_rmse": np.array([r.dE1_rmse for r in results]),
        "dE2_rmse": np.array([r.dE2_rmse for r in results]),
        "ds1_rmse": np.array([r.ds1_rmse for r in results]),
        "ds2_rmse": np.array([r.ds2_rmse for r in results]),
        "a1_rmse": np.array([r.a1_rmse for r in results]),
        "a2_rmse": np.array([r.a2_rmse for r in results]),
        "n_swapped": np.array([r.n_swapped for r in results]),
        "throughput": np.array([r.throughput for r in results]),
        "n_spectra": np.array([r.n_spectra for r in results]),
    }
    np.savez_compressed(path, **data)
    print(f"Saved: {path}")


# ============================================================================
# CLI
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MultiPeak Precision Sweep",
    )
    parser.add_argument(
        "--n-spectra",
        type=int,
        default=N_SPECTRA,
        help="Spectra per cell (default: 100K)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--no-constrain",
        action="store_true",
        help="Disable auto dE_range constraint",
    )
    args = parser.parse_args()

    results = run_sweep(
        n_spectra=args.n_spectra, auto_constrain=not args.no_constrain
    )
    print_scorecard(results)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = (
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / "outputs"
            / "multipeak_sweep"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_dir / "multipeak_sweep.npz"))


if __name__ == "__main__":
    main()
