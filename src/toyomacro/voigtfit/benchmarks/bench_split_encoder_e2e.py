"""
Split Encoder E2E Solver Test
==========================================

Validates that the Split Fisher-Hilbert encoder eliminates
the 10 dB δσ gap discovered in dev-log 72.

dev-log 72 (Euclidean MultiPeakEncoder, order=4):
    Encoder: δσ = 34.3 dB
    E2E:     δσ = 22.4 dB  → gap = -11.6 dB

dev-log 76 (SplitFisherHilbertEncoder, order=8):
    Encoder: δσ ≈ 58.9 dB
    E2E:     δσ = ???       → gap ≈ 0 (hypothesis)

The gap was caused by correlated encoder quantization error: Euclidean
encoder's σ step ≈ 0.013 eV deforms the Voigt profile shape, creating
systematic solver bias. Split encoder's Fisher-optimal σ quantization
(step ≈ 0.0008 eV at ±0.1 range) should make this negligible.

Usage:
    uv run python -m toyomacro.voigtfit.benchmarks.bench_split_encoder_e2e
    uv run python -m toyomacro.voigtfit.benchmarks.bench_split_encoder_e2e --part 1
"""

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toyomacro.voigtfit.fisher_hilbert_encoder import (
    SplitFisherHilbertConfig,
    SplitFisherHilbertEncoder,
)
from toyomacro.voigtfit.multipeak_config import ComponentConfig, MultiPeakConfig
from toyomacro.voigtfit.multipeak_solver import process_multipeak
from toyomacro.voigtfit.param_encoder import MultiPeakEncoder

from .bench_gvrt_multipeak_roundtrip import (
    correct_swaps,
    generate_2comp_spectra,
    make_preset,
    psnr,
    sample_params,
    theory_min_psnr,
)

# ============================================================================
# Split encoder round trip helper
# ============================================================================


def split_encoder_roundtrip(
    dEs: np.ndarray,
    dSs: np.ndarray,
    encoder: SplitFisherHilbertEncoder,
    dgamma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode → RGB → Decode each peak independently via Split encoder.

    The Split encoder encodes (δE, δσ, δγ) per peak. Amplitude is NOT
    encoded (Fisher null space) — it's solved analytically by LLS.

    Args:
        dEs: (n, n_peaks) energy shifts.
        dSs: (n, n_peaks) width shifts.
        encoder: SplitFisherHilbertEncoder instance.
        dgamma: Fixed δγ value for all spectra (default 0).

    Returns:
        (dEs_dec, dSs_dec) same shape as inputs.
    """
    n, n_peaks = dEs.shape
    dEs_dec = np.zeros_like(dEs)
    dSs_dec = np.zeros_like(dSs)
    dg = np.full(n, dgamma, dtype=np.float64)

    for k in range(n_peaks):
        rec = encoder.encode_decode_roundtrip(
            dEs[:, k].astype(np.float64),
            dSs[:, k].astype(np.float64),
            dg,
        )
        dEs_dec[:, k] = rec['dE']
        dSs_dec[:, k] = rec['dsigma']

    return dEs_dec, dSs_dec


def euclidean_encoder_roundtrip(
    amps: np.ndarray,
    dEs: np.ndarray,
    dSs: np.ndarray,
    encoder: MultiPeakEncoder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode → RGB → Decode via Euclidean MultiPeakEncoder.

    Returns:
        (amps_dec, dEs_dec, dSs_dec) same shape as inputs.
    """
    n = amps.shape[0]
    side = int(np.ceil(np.sqrt(n)))
    n_padded = side * side
    image_shape = (side, side)

    amps_pad = np.zeros((n_padded, 2), dtype=np.float32)
    dEs_pad = np.zeros((n_padded, 2), dtype=np.float32)
    dSs_pad = np.zeros((n_padded, 2), dtype=np.float32)
    amps_pad[:n] = amps
    dEs_pad[:n] = dEs
    dSs_pad[:n] = dSs

    rgb = encoder.decode(amps_pad, dEs_pad, dSs_pad, image_shape)
    amps_dec, dEs_dec, dSs_dec = encoder.encode(rgb)

    return amps_dec[:n], dEs_dec[:n], dSs_dec[:n]


# ============================================================================
# Solver step (shared by both encoder pipelines)
# ============================================================================


def make_solver_config(preset) -> MultiPeakConfig:
    """Create solver config from preset."""
    return MultiPeakConfig(
        peaks=[
            ComponentConfig(
                center=elem.binding_energy,
                sigma=elem.sigma,
                gamma=elem.gamma,
                dE_range=preset.shift_range.max_val,
                ds_range=preset.sigma_range.max_val,
                n_dE=15,
                n_ds=31,
            )
            for elem in preset.elements
        ],
        energy_axis=preset.energy,
    )


def solve_spectra(Y: np.ndarray, config: MultiPeakConfig):
    """Solve spectra with multipeak solver. Returns MultiPeakResult."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return process_multipeak(
            Y, config, parabola_dE=True, parabola_ds=False, auto_constrain=True
        )


# ============================================================================
# PSNR computation
# ============================================================================


def compute_psnr_channels(
    gt_amps, gt_dEs, gt_dSs,
    rec_amps, rec_dEs, rec_dSs,
    preset,
) -> dict[str, float]:
    """Compute per-channel PSNR (averaged over components).

    Returns dict with amp_psnr, dE_psnr, dS_psnr.
    """
    amp_range = preset.amplitude_range.max_val - preset.amplitude_range.min_val
    dE_range = preset.shift_range.max_val - preset.shift_range.min_val
    dS_range = preset.sigma_range.max_val - preset.sigma_range.min_val

    n_peaks = gt_amps.shape[1]
    amp_psnrs = [psnr(gt_amps[:, k], rec_amps[:, k], amp_range) for k in range(n_peaks)]
    dE_psnrs = [psnr(gt_dEs[:, k], rec_dEs[:, k], dE_range) for k in range(n_peaks)]
    dS_psnrs = [psnr(gt_dSs[:, k], rec_dSs[:, k], dS_range) for k in range(n_peaks)]

    return {
        "amp_psnr": float(np.mean(amp_psnrs)),
        "dE_psnr": float(np.mean(dE_psnrs)),
        "dS_psnr": float(np.mean(dS_psnrs)),
    }


# ============================================================================
# Result container
# ============================================================================


@dataclass
class SplitE2EResult:
    """Result from one encoder's E2E evaluation."""

    encoder_name: str
    n_spectra: int
    separation_sigma: float
    noise_level: float | None

    # Encoder-only PSNR
    enc_amp_psnr: float
    enc_dE_psnr: float
    enc_dS_psnr: float

    # Solver-only PSNR (from GT params, no encoder)
    sol_amp_psnr: float
    sol_dE_psnr: float
    sol_dS_psnr: float

    # End-to-end PSNR
    e2e_amp_psnr: float
    e2e_dE_psnr: float
    e2e_dS_psnr: float

    # Theory minimum (independent error model)
    theory_dS_psnr: float

    # Gap: E2E - Theory (negative = correlated error)
    gap_dS: float

    n_swapped: int
    time_s: float


# ============================================================================
# Core evaluation: Split encoder pipeline
# ============================================================================


def evaluate_split(
    n_spectra: int,
    separation_sigma: float,
    noise_level: float | None = None,
    seed: int = 42,
    dE_range: float = 0.5,
    ds_range: float = 0.1,
    dgamma_range: float = 0.05,
) -> SplitE2EResult:
    """Run E2E round trip with SplitFisherHilbertEncoder.

    Pipeline:
        GT params → Split encode (δE, δσ, δγ=0) → decode → spectra gen → solve → PSNR
        Amplitude is NOT encoded (passed directly from GT).
    """
    rng = np.random.default_rng(seed)
    preset = make_preset(separation_sigma, hilbert_order=4,
                         dE_range=dE_range, ds_range=ds_range)

    split_config = SplitFisherHilbertConfig(
        hilbert_order=8,
        dE_range=(-dE_range, dE_range),
        dsigma_range=(-ds_range, ds_range),
        dgamma_range=(-dgamma_range, dgamma_range),
    )
    encoder = SplitFisherHilbertEncoder(config=split_config)
    solver_config = make_solver_config(preset)

    # 1. Sample GT params
    amps_gt, dEs_gt, dSs_gt = sample_params(n_spectra, preset, rng)

    # 2. Encoder round trip (amplitude NOT encoded)
    dEs_enc, dSs_enc = split_encoder_roundtrip(dEs_gt, dSs_gt, encoder)
    amps_enc = amps_gt.copy()  # amplitude passed through perfectly

    enc_psnr = compute_psnr_channels(
        amps_gt, dEs_gt, dSs_gt, amps_enc, dEs_enc, dSs_enc, preset
    )

    # 3. Generate spectra from encoded params + solve
    t0 = time.perf_counter()
    Y = generate_2comp_spectra(amps_enc, dEs_enc, dSs_enc, preset, noise_level)
    result = solve_spectra(Y, solver_config)
    elapsed = time.perf_counter() - t0

    dE_corr, dS_corr, amp_corr, n_swapped = correct_swaps(
        result.delta_E, result.delta_sigma, result.amplitudes,
        dEs_gt, dSs_gt, amps_gt,
    )

    e2e_psnr = compute_psnr_channels(
        amps_gt, dEs_gt, dSs_gt, amp_corr, dE_corr, dS_corr, preset
    )

    # 4. Solver-only PSNR (from GT, no encoder)
    Y_gt = generate_2comp_spectra(amps_gt, dEs_gt, dSs_gt, preset, noise_level)
    result_sol = solve_spectra(Y_gt, solver_config)
    dE_sol, dS_sol, amp_sol, _ = correct_swaps(
        result_sol.delta_E, result_sol.delta_sigma, result_sol.amplitudes,
        dEs_gt, dSs_gt, amps_gt,
    )
    sol_psnr = compute_psnr_channels(
        amps_gt, dEs_gt, dSs_gt, amp_sol, dE_sol, dS_sol, preset
    )

    # 5. Theory and gap
    theory_dS = theory_min_psnr(enc_psnr["dS_psnr"], sol_psnr["dS_psnr"])
    gap_dS = e2e_psnr["dS_psnr"] - theory_dS

    return SplitE2EResult(
        encoder_name="Split Fisher-Hilbert",
        n_spectra=n_spectra,
        separation_sigma=separation_sigma,
        noise_level=noise_level,
        enc_amp_psnr=enc_psnr["amp_psnr"],
        enc_dE_psnr=enc_psnr["dE_psnr"],
        enc_dS_psnr=enc_psnr["dS_psnr"],
        sol_amp_psnr=sol_psnr["amp_psnr"],
        sol_dE_psnr=sol_psnr["dE_psnr"],
        sol_dS_psnr=sol_psnr["dS_psnr"],
        e2e_amp_psnr=e2e_psnr["amp_psnr"],
        e2e_dE_psnr=e2e_psnr["dE_psnr"],
        e2e_dS_psnr=e2e_psnr["dS_psnr"],
        theory_dS_psnr=theory_dS,
        gap_dS=gap_dS,
        n_swapped=n_swapped,
        time_s=elapsed,
    )


# ============================================================================
# Core evaluation: Euclidean encoder pipeline
# ============================================================================


def evaluate_euclidean(
    n_spectra: int,
    separation_sigma: float,
    noise_level: float | None = None,
    seed: int = 42,
    dE_range: float = 0.5,
    ds_range: float = 0.1,
) -> SplitE2EResult:
    """Run E2E round trip with Euclidean MultiPeakEncoder.

    Pipeline:
        GT params → Euclidean encode (amp, δE, δσ) → decode → spectra gen → solve → PSNR
    """
    rng = np.random.default_rng(seed)
    preset = make_preset(separation_sigma, hilbert_order=4,
                         dE_range=dE_range, ds_range=ds_range)
    encoder = MultiPeakEncoder(preset)
    solver_config = make_solver_config(preset)

    # 1. Sample GT params
    amps_gt, dEs_gt, dSs_gt = sample_params(n_spectra, preset, rng)

    # 2. Encoder round trip (amplitude IS encoded)
    amps_enc, dEs_enc, dSs_enc = euclidean_encoder_roundtrip(
        amps_gt, dEs_gt, dSs_gt, encoder
    )

    enc_psnr = compute_psnr_channels(
        amps_gt, dEs_gt, dSs_gt, amps_enc, dEs_enc, dSs_enc, preset
    )

    # 3. Generate spectra from encoded params + solve
    t0 = time.perf_counter()
    Y = generate_2comp_spectra(amps_enc, dEs_enc, dSs_enc, preset, noise_level)
    result = solve_spectra(Y, solver_config)
    elapsed = time.perf_counter() - t0

    dE_corr, dS_corr, amp_corr, n_swapped = correct_swaps(
        result.delta_E, result.delta_sigma, result.amplitudes,
        dEs_gt, dSs_gt, amps_gt,
    )

    e2e_psnr = compute_psnr_channels(
        amps_gt, dEs_gt, dSs_gt, amp_corr, dE_corr, dS_corr, preset
    )

    # 4. Solver-only PSNR (from GT, no encoder)
    Y_gt = generate_2comp_spectra(amps_gt, dEs_gt, dSs_gt, preset, noise_level)
    result_sol = solve_spectra(Y_gt, solver_config)
    dE_sol, dS_sol, amp_sol, _ = correct_swaps(
        result_sol.delta_E, result_sol.delta_sigma, result_sol.amplitudes,
        dEs_gt, dSs_gt, amps_gt,
    )
    sol_psnr = compute_psnr_channels(
        amps_gt, dEs_gt, dSs_gt, amp_sol, dE_sol, dS_sol, preset
    )

    # 5. Theory and gap
    theory_dS = theory_min_psnr(enc_psnr["dS_psnr"], sol_psnr["dS_psnr"])
    gap_dS = e2e_psnr["dS_psnr"] - theory_dS

    return SplitE2EResult(
        encoder_name="Euclidean",
        n_spectra=n_spectra,
        separation_sigma=separation_sigma,
        noise_level=noise_level,
        enc_amp_psnr=enc_psnr["amp_psnr"],
        enc_dE_psnr=enc_psnr["dE_psnr"],
        enc_dS_psnr=enc_psnr["dS_psnr"],
        sol_amp_psnr=sol_psnr["amp_psnr"],
        sol_dE_psnr=sol_psnr["dE_psnr"],
        sol_dS_psnr=sol_psnr["dS_psnr"],
        e2e_amp_psnr=e2e_psnr["amp_psnr"],
        e2e_dE_psnr=e2e_psnr["dE_psnr"],
        e2e_dS_psnr=e2e_psnr["dS_psnr"],
        theory_dS_psnr=theory_dS,
        gap_dS=gap_dS,
        n_swapped=n_swapped,
        time_s=elapsed,
    )


# ============================================================================
# Part 1: Error separation comparison
# ============================================================================


def print_result_table(r: SplitE2EResult) -> None:
    """Print a single result's error decomposition."""
    print(f"\n  {r.encoder_name}:")
    print(f"  {'':16s} {'amp':>8s}  {'δE':>8s}  {'δσ':>8s}")
    print(f"  {'Encoder':16s} {r.enc_amp_psnr:8.1f}  {r.enc_dE_psnr:8.1f}  {r.enc_dS_psnr:8.1f} dB")
    print(f"  {'Solver':16s} {r.sol_amp_psnr:8.1f}  {r.sol_dE_psnr:8.1f}  {r.sol_dS_psnr:8.1f} dB")
    print(f"  {'Theory':16s} {'—':>8s}  {'—':>8s}  {r.theory_dS_psnr:8.1f} dB")
    print(f"  {'End-to-end':16s} {r.e2e_amp_psnr:8.1f}  {r.e2e_dE_psnr:8.1f}  {r.e2e_dS_psnr:8.1f} dB")
    print(f"  {'Gap (E2E-Thy)':16s} {'':>8s}  {'':>8s}  {r.gap_dS:+8.1f} dB")


def part1_error_separation(n_spectra: int = 100_000) -> tuple[SplitE2EResult, SplitE2EResult]:
    """Part 1: Compare error decomposition for both encoders."""
    print("=" * 70)
    print(f"Part 1: Error Separation (Δc=7σ, NF, n={n_spectra:,})")
    print("=" * 70)

    r_euc = evaluate_euclidean(n_spectra, separation_sigma=7.0, seed=42)
    print_result_table(r_euc)

    r_split = evaluate_split(n_spectra, separation_sigma=7.0, seed=42)
    print_result_table(r_split)

    print("\n  Summary:")
    print(f"    δσ E2E improvement: {r_split.e2e_dS_psnr - r_euc.e2e_dS_psnr:+.1f} dB")
    print(f"    δσ gap reduction:   {r_euc.gap_dS:.1f} → {r_split.gap_dS:.1f} dB")

    return r_euc, r_split


# ============================================================================
# Part 2: Separation sweep
# ============================================================================


def part2_separation_sweep(
    n_spectra: int = 50_000,
    separations: list[float] | None = None,
) -> tuple[list[SplitE2EResult], list[SplitE2EResult]]:
    """Part 2: Sweep separation for both encoders."""
    if separations is None:
        separations = [2.0, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]

    print("=" * 70)
    print(f"Part 2: Separation Sweep (n={n_spectra:,} per cell)")
    print("=" * 70)

    euc_results = []
    split_results = []

    print(f"\n  {'Δc/σ':>5s}  {'Euc δσ E2E':>12s}  {'Split δσ E2E':>14s}  {'Euc gap':>8s}  {'Split gap':>10s}")
    for i, sep in enumerate(separations):
        r_euc = evaluate_euclidean(n_spectra, sep, seed=100 + i)
        r_split = evaluate_split(n_spectra, sep, seed=100 + i)
        euc_results.append(r_euc)
        split_results.append(r_split)
        print(
            f"  {sep:5.1f}  {r_euc.e2e_dS_psnr:12.1f}  {r_split.e2e_dS_psnr:14.1f}  "
            f"{r_euc.gap_dS:+8.1f}  {r_split.gap_dS:+10.1f}"
        )

    return euc_results, split_results


# ============================================================================
# Part 3: Noise sweep
# ============================================================================


def part3_noise_sweep(
    n_spectra: int = 50_000,
    noise_levels: list[float | None] | None = None,
) -> tuple[list[SplitE2EResult], list[SplitE2EResult]]:
    """Part 3: Sweep noise level for both encoders."""
    if noise_levels is None:
        noise_levels = [None, 300, 1000, 3000]

    print("=" * 70)
    print(f"Part 3: Noise Sweep (Δc=7σ, n={n_spectra:,} per cell)")
    print("=" * 70)

    euc_results = []
    split_results = []

    print(f"\n  {'Noise':>6s}  {'Euc δσ E2E':>12s}  {'Split δσ E2E':>14s}  {'Euc gap':>8s}  {'Split gap':>10s}")
    for i, level in enumerate(noise_levels):
        r_euc = evaluate_euclidean(n_spectra, 7.0, noise_level=level, seed=300 + i)
        r_split = evaluate_split(n_spectra, 7.0, noise_level=level, seed=300 + i)
        euc_results.append(r_euc)
        split_results.append(r_split)
        label = "NF" if level is None else f"{level:.0f}"
        print(
            f"  {label:>6s}  {r_euc.e2e_dS_psnr:12.1f}  {r_split.e2e_dS_psnr:14.1f}  "
            f"{r_euc.gap_dS:+8.1f}  {r_split.gap_dS:+10.1f}"
        )

    return euc_results, split_results


# ============================================================================
# Visualization (4-panel)
# ============================================================================


def plot_session76(
    r_euc: SplitE2EResult,
    r_split: SplitE2EResult,
    sep_euc: list[SplitE2EResult],
    sep_split: list[SplitE2EResult],
    noise_euc: list[SplitE2EResult],
    noise_split: list[SplitE2EResult],
    output_path: str | None = None,
) -> None:
    """Generate 4-panel dev-log 76 figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Error decomposition comparison
    ax = axes[0, 0]
    params = ["amp", "δE", "δσ"]
    x = np.arange(len(params))
    w = 0.15
    for i, (r, label, color) in enumerate([
        (r_euc, "Euc Enc", "C0"),
        (r_euc, "Euc E2E", "C1"),
        (r_split, "Split Enc", "C4"),
        (r_split, "Split E2E", "C2"),
    ]):
        if "Enc" in label:
            vals = [r.enc_amp_psnr, r.enc_dE_psnr, r.enc_dS_psnr]
        else:
            vals = [r.e2e_amp_psnr, r.e2e_dE_psnr, r.e2e_dS_psnr]
        ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("A: Error Decomposition (Δc=7σ)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: PSNR vs separation
    ax = axes[0, 1]
    seps = [r.separation_sigma for r in sep_euc]
    ax.plot(seps, [r.e2e_dS_psnr for r in sep_euc], "o--", color="gray",
            alpha=0.6, label="Euc δσ E2E (dev-log 72)")
    ax.plot(seps, [r.e2e_dS_psnr for r in sep_split], "s-", color="C2",
            label="Split δσ E2E")
    ax.plot(seps, [r.sol_dS_psnr for r in sep_split], "^:", color="C1",
            alpha=0.7, label="Solver δσ (no encoder)")
    ax.axvline(3.5, color="gray", ls=":", alpha=0.5, label="3.5σ boundary")
    ax.set_xlabel("Δc / σ")
    ax.set_ylabel("δσ PSNR (dB)")
    ax.set_title("B: δσ PSNR vs Separation")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel C: Gap comparison
    ax = axes[1, 0]
    noise_labels = ["NF" if r.noise_level is None else str(int(r.noise_level))
                     for r in noise_euc]
    x_noise = np.arange(len(noise_labels))
    ax.bar(x_noise - 0.2, [r.gap_dS for r in noise_euc], 0.35,
           label="Euclidean gap", color="gray", alpha=0.6)
    ax.bar(x_noise + 0.2, [r.gap_dS for r in noise_split], 0.35,
           label="Split gap", color="C2")
    ax.axhline(0, color="black", ls="-", lw=0.5)
    ax.axhline(-3, color="red", ls="--", alpha=0.5, label="Target: > -3 dB")
    ax.set_xticks(x_noise)
    ax.set_xticklabels(noise_labels)
    ax.set_xlabel("Noise level")
    ax.set_ylabel("δσ Gap (E2E - Theory) dB")
    ax.set_title("C: Gap Elimination")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Panel D: Cumulative improvement
    ax = axes[1, 1]
    sessions = ["S55\nDict2D", "S72\nMultiPeak", "S73\nFisher", "S75\nSplit", "S76\nE2E"]
    # Historical δσ values (from memory notes)
    enc_psnrs = [47.5, 34.3, 58.9, 58.9, r_split.enc_dS_psnr]
    e2e_psnrs = [None, 22.4, None, None, r_split.e2e_dS_psnr]
    ax.bar(range(len(sessions)), enc_psnrs, 0.35, label="Encoder δσ", color="C0", alpha=0.7)
    e2e_valid = [(i, v) for i, v in enumerate(e2e_psnrs) if v is not None]
    if e2e_valid:
        ax.bar([x[0] + 0.35 for x in e2e_valid], [x[1] for x in e2e_valid],
               0.35, label="E2E δσ", color="C2")
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels(sessions, fontsize=8)
    ax.set_ylabel("δσ PSNR (dB)")
    ax.set_title("D: Session-over-Session δσ Improvement")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Split Encoder E2E Gap Verification", fontsize=14, y=1.01)
    plt.tight_layout()

    if output_path is None:
        output_dir = (
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / "output"
            / "gvrt_multipeak"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / "session76_4panel.png")

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved: {output_path}")
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Split Encoder E2E Solver Test",
    )
    parser.add_argument(
        "--part", type=int, default=0,
        help="Run specific part (1-3). 0 = all parts.",
    )
    parser.add_argument(
        "--n-spectra", type=int, default=None,
        help="Override spectra count per cell.",
    )
    args = parser.parse_args()

    n1 = args.n_spectra or 100_000
    n2 = args.n_spectra or 50_000

    r_euc = r_split = None
    sep_euc = sep_split = None
    noise_euc = noise_split = None

    if args.part in (0, 1):
        r_euc, r_split = part1_error_separation(n1)

    if args.part in (0, 2):
        sep_euc, sep_split = part2_separation_sweep(n2)

    if args.part in (0, 3):
        noise_euc, noise_split = part3_noise_sweep(n2)

    # Visualization (only if all parts ran)
    if args.part == 0 and r_euc is not None and sep_euc is not None and noise_euc is not None:
        plot_session76(r_euc, r_split, sep_euc, sep_split, noise_euc, noise_split)


if __name__ == "__main__":
    main()
