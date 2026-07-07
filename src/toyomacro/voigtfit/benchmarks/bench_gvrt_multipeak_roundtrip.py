"""
GVRT MultiPeak Round Trip
=======================================

End-to-end round trip verification of MultiPeakEncoder → spectra generation
→ multipeak solver → PSNR evaluation.

Pipeline:
    GT params → Hilbert encode → RGB image → decode → spectra gen (2-comp Voigt)
    → solve_multipeak (Alternating Projection) → recovered params → PSNR

Parts:
    1. Basic Round Trip (equal amp, Δc=7σ)
    2. Separation sweep (2-10σ)
    3. Amplitude ratio sweep (1:1 → 1:5)
    4. Noise robustness sweep
    5. Error decomposition (encoder vs solver)

Usage:
    uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_multipeak_roundtrip
    uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_multipeak_roundtrip --part 1
    uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_multipeak_roundtrip --part 2 --n-spectra 10000
"""

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toyomacro.voigtfit.multipeak_config import ComponentConfig, MultiPeakConfig
from toyomacro.voigtfit.multipeak_solver import process_multipeak
from toyomacro.voigtfit.param_encoder import (
    MultiPeakEncoder,
    MultiPeakPreset,
    ParamRange,
)
from toyomacro.voigtfit.spectra_generator import (
    ElementSpec,
    add_poisson_noise,
)

# ============================================================================
# Constants
# ============================================================================

FWHM_TO_SIGMA = 1.0 / (2 * np.sqrt(2 * np.log(2)))
SIGMA_NOM = 1.0 * FWHM_TO_SIGMA  # 0.4247 eV
GAMMA = 0.125  # eV (Lorentzian HWHM)
CENTER_MID = 285.0  # eV (midpoint)


# ============================================================================
# Preset factory
# ============================================================================


def make_preset(
    separation_sigma: float,
    hilbert_order: int = 4,
    dE_range: float = 0.5,
    ds_range: float = 0.1,
) -> MultiPeakPreset:
    """Create a MultiPeakPreset with given peak separation.

    Args:
        separation_sigma: Peak separation in σ units.
        hilbert_order: Hilbert curve order (default 4 → 16 levels/dim).
        dE_range: Half-range for δE (eV).
        ds_range: Half-range for δσ (eV).

    Returns:
        MultiPeakPreset with 2 peaks separated by `separation_sigma × σ`.
    """
    sep_eV = separation_sigma * SIGMA_NOM
    c1 = CENTER_MID - sep_eV / 2
    c2 = CENTER_MID + sep_eV / 2

    # Energy axis: cover both peaks with 6 eV margin
    e_min = c1 - 6.0
    e_max = c2 + 6.0
    n_energy = int(round((e_max - e_min) / 0.1)) + 1

    return MultiPeakPreset(
        name=f"C1s_dual_{separation_sigma:.0f}sigma",
        elements=[
            ElementSpec("C", "1s", c1, gaussian_fwhm=1.0, lorentzian_fwhm=0.25),
            ElementSpec("C", "1s", c2, gaussian_fwhm=1.0, lorentzian_fwhm=0.25),
        ],
        energy_range=(e_min, e_max),
        n_energy=n_energy,
        hilbert_order=hilbert_order,
        amplitude_range=ParamRange(0.0, 1.0, "amplitude"),
        shift_range=ParamRange(-dE_range, dE_range, "delta_E"),
        sigma_range=ParamRange(-ds_range, ds_range, "delta_sigma"),
    )


# ============================================================================
# Parameter sampling
# ============================================================================


def sample_params(
    n: int,
    preset: MultiPeakPreset,
    rng: np.random.Generator,
    amp_ratio: float = 1.0,
    amp_min: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample random parameters within preset ranges.

    Peak 1 amplitude is uniform in [amp_min, 1.0].
    Peak 2 amplitude = peak1 * amp_ratio, clipped to [0, 1].

    Returns:
        (amplitudes, delta_Es, delta_sigmas) each (n, 2) float32.
    """
    n_peaks = preset.n_peaks
    amps = np.zeros((n, n_peaks), dtype=np.float32)
    dEs = np.zeros((n, n_peaks), dtype=np.float32)
    dSs = np.zeros((n, n_peaks), dtype=np.float32)

    # Peak 1
    amps[:, 0] = rng.uniform(amp_min, 1.0, n).astype(np.float32)
    # Peak 2: scaled by ratio
    amps[:, 1] = np.clip(amps[:, 0] * amp_ratio, 0.0, 1.0).astype(np.float32)

    for k in range(n_peaks):
        dEs[:, k] = rng.uniform(
            preset.shift_range.min_val, preset.shift_range.max_val, n
        ).astype(np.float32)
        dSs[:, k] = rng.uniform(
            preset.sigma_range.min_val, preset.sigma_range.max_val, n
        ).astype(np.float32)

    return amps, dEs, dSs


# ============================================================================
# Spectra generation (exact Voigt)
# ============================================================================


def generate_2comp_spectra(
    amps: np.ndarray,
    dEs: np.ndarray,
    dSs: np.ndarray,
    preset: MultiPeakPreset,
    noise_level: float | None = None,
) -> np.ndarray:
    """Generate 2-component Voigt spectra from parameters.

    Uses UN-normalized Voigt profiles (raw wofz), matching the dictionary
    solver's convention. This ensures amplitude PSNR is meaningful:
    the solver's LLS amplitudes directly correspond to the GT amplitudes.

    Args:
        amps: (n, 2) amplitudes.
        dEs: (n, 2) energy shifts (eV).
        dSs: (n, 2) width shifts (eV).
        preset: MultiPeakPreset with element specs.
        noise_level: Poisson noise level (None = noise-free).

    Returns:
        Y: (n, n_energy) float32.
    """
    from scipy.special import wofz

    n = amps.shape[0]
    energy = preset.energy.astype(np.float64)
    n_E = len(energy)
    SQRT2 = np.sqrt(2.0)
    SQRT2PI = np.sqrt(2.0 * np.pi)

    Y = np.zeros((n, n_E), dtype=np.float64)

    for k in range(preset.n_peaks):
        elem = preset.elements[k]
        centers = (elem.binding_energy + dEs[:, k]).astype(np.float64)[:, None]
        sigmas = np.maximum(
            elem.sigma + dSs[:, k], 0.01
        ).astype(np.float64)[:, None]
        gamma = elem.gamma

        # Raw Voigt profile (un-normalized), matching dictionary convention
        z = (energy[None, :] - centers + 1j * gamma) / (sigmas * SQRT2)
        P = np.real(wofz(z)) / (sigmas * SQRT2PI)

        Y += amps[:, k : k + 1].astype(np.float64) * P

    Y = Y.astype(np.float32)

    if noise_level is not None and noise_level > 0:
        Y = add_poisson_noise(Y, noise_level)

    return Y


# ============================================================================
# Encoder round trip helper
# ============================================================================


def encoder_roundtrip(
    amps: np.ndarray,
    dEs: np.ndarray,
    dSs: np.ndarray,
    encoder: MultiPeakEncoder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode → RGB → Decode. Pad to square image internally.

    Returns:
        (amps_decoded, dEs_decoded, dSs_decoded) same shape as inputs.
    """
    n = amps.shape[0]
    side = int(np.ceil(np.sqrt(n)))
    n_padded = side * side
    image_shape = (side, side)

    # Pad
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
# Component swap correction (reused from bench_multipeak_sweep)
# ============================================================================


def correct_swaps(
    delta_E: np.ndarray,
    delta_sigma: np.ndarray,
    amplitudes: np.ndarray,
    gt_dE: np.ndarray,
    gt_dS: np.ndarray,
    gt_amp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-spectrum swap correction for ambiguous component assignment.

    Returns:
        (dE_corr, ds_corr, amp_corr, n_swapped)
    """
    err_orig = (
        (delta_E[:, 0] - gt_dE[:, 0]) ** 2
        + (delta_E[:, 1] - gt_dE[:, 1]) ** 2
    )
    err_swap = (
        (delta_E[:, 1] - gt_dE[:, 0]) ** 2
        + (delta_E[:, 0] - gt_dE[:, 1]) ** 2
    )

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
# PSNR computation
# ============================================================================


def psnr(gt: np.ndarray, rec: np.ndarray, signal_range: float) -> float:
    """PSNR = 10 log10(range^2 / MSE)."""
    mse = float(np.mean((gt - rec) ** 2))
    if mse < 1e-30:
        return 999.0
    return 10 * np.log10(signal_range**2 / mse)


def compute_psnr_all(
    gt_amps: np.ndarray,
    gt_dEs: np.ndarray,
    gt_dSs: np.ndarray,
    rec_amps: np.ndarray,
    rec_dEs: np.ndarray,
    rec_dSs: np.ndarray,
    preset: MultiPeakPreset,
) -> dict[str, float]:
    """Compute per-channel and average PSNR.

    Returns dict with amp_psnr, dE_psnr, dS_psnr (averaged over components).
    """
    amp_range = preset.amplitude_range.max_val - preset.amplitude_range.min_val
    dE_range = preset.shift_range.max_val - preset.shift_range.min_val
    dS_range = preset.sigma_range.max_val - preset.sigma_range.min_val

    n_peaks = gt_amps.shape[1]

    amp_psnrs = [psnr(gt_amps[:, k], rec_amps[:, k], amp_range) for k in range(n_peaks)]
    dE_psnrs = [psnr(gt_dEs[:, k], rec_dEs[:, k], dE_range) for k in range(n_peaks)]
    dS_psnrs = [psnr(gt_dSs[:, k], rec_dSs[:, k], dS_range) for k in range(n_peaks)]

    return {
        "amp_psnr": np.mean(amp_psnrs),
        "dE_psnr": np.mean(dE_psnrs),
        "dS_psnr": np.mean(dS_psnrs),
    }


# ============================================================================
# Result container
# ============================================================================


@dataclass
class RoundTripResult:
    """Result from one round trip evaluation."""

    n_spectra: int
    separation_sigma: float
    amp_ratio: float
    noise_level: float | None

    # End-to-end PSNR
    amp_psnr: float
    dE_psnr: float
    dS_psnr: float

    # Encoder-only PSNR
    enc_amp_psnr: float
    enc_dE_psnr: float
    enc_dS_psnr: float

    # Solver-only PSNR
    sol_amp_psnr: float
    sol_dE_psnr: float
    sol_dS_psnr: float

    # Theory minimum
    theory_amp_psnr: float
    theory_dE_psnr: float
    theory_dS_psnr: float

    n_swapped: int
    time_s: float
    throughput: float


def theory_min_psnr(enc: float, sol: float) -> float:
    """Theoretical minimum from independent encoder + solver noise."""
    if enc > 900 or sol > 900:
        return min(enc, sol)
    return float(-10 * np.log10(10 ** (-enc / 10) + 10 ** (-sol / 10)))


# ============================================================================
# Core round trip evaluation
# ============================================================================


def evaluate_roundtrip(
    n_spectra: int,
    separation_sigma: float,
    amp_ratio: float = 1.0,
    noise_level: float | None = None,
    seed: int = 42,
    hilbert_order: int = 4,
) -> RoundTripResult:
    """Run full round trip and return all metrics.

    Pipeline:
        GT params → encode → RGB → decode → spectra gen → solve → compare

    Also computes encoder-only and solver-only PSNR for error decomposition.
    """
    rng = np.random.default_rng(seed)
    preset = make_preset(separation_sigma, hilbert_order=hilbert_order)
    encoder = MultiPeakEncoder(preset)

    # 1. Sample GT params
    amps_gt, dEs_gt, dSs_gt = sample_params(
        n_spectra, preset, rng, amp_ratio=amp_ratio
    )

    # 2. Encoder round trip only
    amps_enc, dEs_enc, dSs_enc = encoder_roundtrip(amps_gt, dEs_gt, dSs_gt, encoder)

    enc_psnr = compute_psnr_all(
        amps_gt, dEs_gt, dSs_gt, amps_enc, dEs_enc, dSs_enc, preset
    )

    # 3. Generate spectra from encoded (quantized) params
    t0 = time.perf_counter()
    Y = generate_2comp_spectra(amps_enc, dEs_enc, dSs_enc, preset, noise_level)

    # 4. Solve
    config = MultiPeakConfig(
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = process_multipeak(
            Y, config, parabola_dE=True, parabola_ds=False, auto_constrain=True
        )

    elapsed = time.perf_counter() - t0

    # 5. Swap correction (using GT dE for assignment)
    dE_corr, dS_corr, amp_corr, n_swapped = correct_swaps(
        result.delta_E, result.delta_sigma, result.amplitudes,
        dEs_gt, dSs_gt, amps_gt,
    )

    # 6. End-to-end PSNR (GT vs solver output)
    e2e_psnr = compute_psnr_all(
        amps_gt, dEs_gt, dSs_gt, amp_corr, dE_corr, dS_corr, preset
    )

    # 7. Solver-only PSNR (GT → spectra from GT → solve → compare with GT)
    Y_from_gt = generate_2comp_spectra(amps_gt, dEs_gt, dSs_gt, preset, noise_level)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_sol = process_multipeak(
            Y_from_gt, config, parabola_dE=True, parabola_ds=False, auto_constrain=True
        )
    dE_sol, dS_sol, amp_sol, _ = correct_swaps(
        result_sol.delta_E, result_sol.delta_sigma, result_sol.amplitudes,
        dEs_gt, dSs_gt, amps_gt,
    )
    sol_psnr = compute_psnr_all(
        amps_gt, dEs_gt, dSs_gt, amp_sol, dE_sol, dS_sol, preset
    )

    # 8. Theory minimum
    theory_amp = theory_min_psnr(enc_psnr["amp_psnr"], sol_psnr["amp_psnr"])
    theory_dE = theory_min_psnr(enc_psnr["dE_psnr"], sol_psnr["dE_psnr"])
    theory_dS = theory_min_psnr(enc_psnr["dS_psnr"], sol_psnr["dS_psnr"])

    return RoundTripResult(
        n_spectra=n_spectra,
        separation_sigma=separation_sigma,
        amp_ratio=amp_ratio,
        noise_level=noise_level,
        amp_psnr=e2e_psnr["amp_psnr"],
        dE_psnr=e2e_psnr["dE_psnr"],
        dS_psnr=e2e_psnr["dS_psnr"],
        enc_amp_psnr=enc_psnr["amp_psnr"],
        enc_dE_psnr=enc_psnr["dE_psnr"],
        enc_dS_psnr=enc_psnr["dS_psnr"],
        sol_amp_psnr=sol_psnr["amp_psnr"],
        sol_dE_psnr=sol_psnr["dE_psnr"],
        sol_dS_psnr=sol_psnr["dS_psnr"],
        theory_amp_psnr=theory_amp,
        theory_dE_psnr=theory_dE,
        theory_dS_psnr=theory_dS,
        n_swapped=n_swapped,
        time_s=elapsed,
        throughput=n_spectra / elapsed if elapsed > 0 else float("inf"),
    )


# ============================================================================
# Part 1: Basic Round Trip
# ============================================================================


def part1_basic(n_spectra: int = 100_000) -> RoundTripResult:
    """Part 1: Basic round trip with equal amplitude, well-separated peaks."""
    print("=" * 70)
    print(f"Part 1: Basic Round Trip (Δc=7σ, amp_ratio=1.0, n={n_spectra:,})")
    print("=" * 70)

    result = evaluate_roundtrip(
        n_spectra=n_spectra,
        separation_sigma=7.0,
        amp_ratio=1.0,
        noise_level=None,
        seed=42,
    )

    print("\n  Error Decomposition:")
    print(f"  {'':16s} {'amp':>8s}  {'δE':>8s}  {'δσ':>8s}")
    print(f"  {'Encoder':16s} {result.enc_amp_psnr:8.1f}  {result.enc_dE_psnr:8.1f}  {result.enc_dS_psnr:8.1f} dB")
    print(f"  {'Solver':16s} {result.sol_amp_psnr:8.1f}  {result.sol_dE_psnr:8.1f}  {result.sol_dS_psnr:8.1f} dB")
    print(f"  {'End-to-end':16s} {result.amp_psnr:8.1f}  {result.dE_psnr:8.1f}  {result.dS_psnr:8.1f} dB")
    print(f"  {'Theory min':16s} {result.theory_amp_psnr:8.1f}  {result.theory_dE_psnr:8.1f}  {result.theory_dS_psnr:8.1f} dB")
    print(f"\n  Swap rate: {result.n_swapped / n_spectra * 100:.2f}%")
    print(f"  Throughput: {result.throughput / 1e6:.2f} M/s")
    print(f"  Elapsed: {result.time_s:.1f}s")

    return result


# ============================================================================
# Part 2: Separation sweep
# ============================================================================


def part2_separation_sweep(
    n_spectra: int = 50_000,
    separations: list[float] | None = None,
) -> list[RoundTripResult]:
    """Part 2: Sweep peak separation to verify 3-regime structure."""
    if separations is None:
        separations = [2.0, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]

    print("=" * 70)
    print(f"Part 2: Separation Sweep (n={n_spectra:,} per cell)")
    print("=" * 70)

    results = []
    for i, sep in enumerate(separations):
        r = evaluate_roundtrip(
            n_spectra=n_spectra,
            separation_sigma=sep,
            amp_ratio=1.0,
            noise_level=None,
            seed=100 + i,
        )
        results.append(r)
        swap_pct = r.n_swapped / n_spectra * 100
        print(
            f"  Δc={sep:5.1f}σ: "
            f"amp={r.amp_psnr:5.1f}  δE={r.dE_psnr:5.1f}  δσ={r.dS_psnr:5.1f} dB  "
            f"(enc δE={r.enc_dE_psnr:5.1f}  sol δE={r.sol_dE_psnr:5.1f})  "
            f"swap={swap_pct:4.1f}%"
        )

    return results


# ============================================================================
# Part 3: Amplitude ratio sweep
# ============================================================================


def part3_amplitude_sweep(
    n_spectra: int = 50_000,
    ratios: list[float] | None = None,
) -> list[RoundTripResult]:
    """Part 3: Sweep amplitude ratio at fixed separation."""
    if ratios is None:
        ratios = [1.0, 1.5, 2.0, 3.0, 5.0]

    print("=" * 70)
    print(f"Part 3: Amplitude Ratio Sweep (Δc=7σ, n={n_spectra:,} per cell)")
    print("=" * 70)

    results = []
    for i, ratio in enumerate(ratios):
        r = evaluate_roundtrip(
            n_spectra=n_spectra,
            separation_sigma=7.0,
            amp_ratio=ratio,
            noise_level=None,
            seed=200 + i,
        )
        results.append(r)
        print(
            f"  ratio=1:{ratio:.1f}: "
            f"amp={r.amp_psnr:5.1f}  δE={r.dE_psnr:5.1f}  δσ={r.dS_psnr:5.1f} dB  "
            f"swap={r.n_swapped / n_spectra * 100:4.1f}%"
        )

    return results


# ============================================================================
# Part 4: Noise sweep
# ============================================================================


def part4_noise_sweep(
    n_spectra: int = 50_000,
    noise_levels: list[float | None] | None = None,
) -> list[RoundTripResult]:
    """Part 4: Sweep noise level at fixed separation and ratio."""
    if noise_levels is None:
        noise_levels = [None, 300, 1000, 3000]

    print("=" * 70)
    print(f"Part 4: Noise Sweep (Δc=7σ, amp=1:1, n={n_spectra:,} per cell)")
    print("=" * 70)

    results = []
    for i, level in enumerate(noise_levels):
        r = evaluate_roundtrip(
            n_spectra=n_spectra,
            separation_sigma=7.0,
            amp_ratio=1.0,
            noise_level=level,
            seed=300 + i,
        )
        results.append(r)
        label = "NF" if level is None else f"{level:.0f}"
        print(
            f"  noise={label:>6s}: "
            f"amp={r.amp_psnr:5.1f}  δE={r.dE_psnr:5.1f}  δσ={r.dS_psnr:5.1f} dB  "
            f"(sol amp={r.sol_amp_psnr:5.1f}  sol δE={r.sol_dE_psnr:5.1f})"
        )

    return results


# ============================================================================
# Part 5: Error decomposition summary
# ============================================================================


def part5_error_decomposition(result: RoundTripResult) -> None:
    """Part 5: Print error decomposition analysis."""
    print("=" * 70)
    print("Part 5: Error Decomposition Analysis")
    print("=" * 70)

    for param, enc, sol, e2e, theory in [
        ("amplitude", result.enc_amp_psnr, result.sol_amp_psnr,
         result.amp_psnr, result.theory_amp_psnr),
        ("delta_E", result.enc_dE_psnr, result.sol_dE_psnr,
         result.dE_psnr, result.theory_dE_psnr),
        ("delta_sigma", result.enc_dS_psnr, result.sol_dS_psnr,
         result.dS_psnr, result.theory_dS_psnr),
    ]:
        bottleneck = "encoder" if enc < sol else "solver"
        gap = e2e - theory
        print(f"\n  {param}:")
        print(f"    Encoder:     {enc:6.1f} dB")
        print(f"    Solver:      {sol:6.1f} dB")
        print(f"    End-to-end:  {e2e:6.1f} dB")
        print(f"    Theory min:  {theory:6.1f} dB  (gap: {gap:+.1f} dB)")
        print(f"    Bottleneck:  {bottleneck}")


# ============================================================================
# Visualization
# ============================================================================


def plot_results(
    sep_results: list[RoundTripResult],
    ratio_results: list[RoundTripResult],
    noise_results: list[RoundTripResult],
    basic_result: RoundTripResult,
    output_path: str | None = None,
) -> None:
    """Generate 4-panel figure.

    Panel A: Error decomposition bar chart
    Panel B: PSNR vs separation
    Panel C: PSNR vs amplitude ratio
    Panel D: PSNR vs noise level
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Error decomposition
    ax = axes[0, 0]
    params = ["amp", "δE", "δσ"]
    enc_vals = [basic_result.enc_amp_psnr, basic_result.enc_dE_psnr, basic_result.enc_dS_psnr]
    sol_vals = [basic_result.sol_amp_psnr, basic_result.sol_dE_psnr, basic_result.sol_dS_psnr]
    e2e_vals = [basic_result.amp_psnr, basic_result.dE_psnr, basic_result.dS_psnr]
    x = np.arange(len(params))
    w = 0.25
    ax.bar(x - w, enc_vals, w, label="Encoder", color="C0")
    ax.bar(x, sol_vals, w, label="Solver", color="C1")
    ax.bar(x + w, e2e_vals, w, label="End-to-end", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("A: Error Decomposition (Δc=7σ)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: PSNR vs separation
    ax = axes[0, 1]
    seps = [r.separation_sigma for r in sep_results]
    ax.plot(seps, [r.amp_psnr for r in sep_results], "o-", label="amp (e2e)")
    ax.plot(seps, [r.dE_psnr for r in sep_results], "s-", label="δE (e2e)")
    ax.plot(seps, [r.dS_psnr for r in sep_results], "^-", label="δσ (e2e)")
    ax.plot(seps, [r.sol_dE_psnr for r in sep_results], "s--", alpha=0.5, label="δE (solver)")
    ax.axvline(3.5, color="gray", ls=":", alpha=0.5, label="3.5σ boundary")
    ax.axvline(7.0, color="gray", ls="--", alpha=0.3, label="7σ boundary")
    ax.set_xlabel("Δc / σ")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("B: PSNR vs Separation")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel C: PSNR vs amplitude ratio
    ax = axes[1, 0]
    ratios = [r.amp_ratio for r in ratio_results]
    ax.plot(ratios, [r.amp_psnr for r in ratio_results], "o-", label="amp")
    ax.plot(ratios, [r.dE_psnr for r in ratio_results], "s-", label="δE")
    ax.plot(ratios, [r.dS_psnr for r in ratio_results], "^-", label="δσ")
    ax.set_xlabel("Amplitude ratio (peak2/peak1)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("C: PSNR vs Amplitude Ratio (Δc=7σ)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel D: PSNR vs noise level
    ax = axes[1, 1]
    noise_labels = ["NF" if r.noise_level is None else str(int(r.noise_level)) for r in noise_results]
    x_noise = np.arange(len(noise_labels))
    ax.plot(x_noise, [r.amp_psnr for r in noise_results], "o-", label="amp (e2e)")
    ax.plot(x_noise, [r.dE_psnr for r in noise_results], "s-", label="δE (e2e)")
    ax.plot(x_noise, [r.dS_psnr for r in noise_results], "^-", label="δσ (e2e)")
    ax.plot(x_noise, [r.sol_dE_psnr for r in noise_results], "s--", alpha=0.5, label="δE (solver)")
    ax.set_xticks(x_noise)
    ax.set_xticklabels(noise_labels)
    ax.set_xlabel("Noise level")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("D: PSNR vs Noise (Δc=7σ)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("GVRT MultiPeak Round Trip", fontsize=14, y=1.01)
    plt.tight_layout()

    if output_path is None:
        output_dir = (
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / "output"
            / "gvrt_multipeak"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / "session72_4panel.png")

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved: {output_path}")
    plt.close(fig)


# ============================================================================
# Save results
# ============================================================================


def save_results(
    results: dict[str, list[RoundTripResult]],
    output_path: str,
) -> None:
    """Save all results to NPZ."""
    data = {}
    for key, rlist in results.items():
        for i, r in enumerate(rlist):
            prefix = f"{key}_{i}"
            data[f"{prefix}_sep"] = r.separation_sigma
            data[f"{prefix}_ratio"] = r.amp_ratio
            data[f"{prefix}_noise"] = r.noise_level if r.noise_level is not None else -1
            data[f"{prefix}_amp_psnr"] = r.amp_psnr
            data[f"{prefix}_dE_psnr"] = r.dE_psnr
            data[f"{prefix}_dS_psnr"] = r.dS_psnr
            data[f"{prefix}_enc_amp_psnr"] = r.enc_amp_psnr
            data[f"{prefix}_enc_dE_psnr"] = r.enc_dE_psnr
            data[f"{prefix}_enc_dS_psnr"] = r.enc_dS_psnr
            data[f"{prefix}_sol_amp_psnr"] = r.sol_amp_psnr
            data[f"{prefix}_sol_dE_psnr"] = r.sol_dE_psnr
            data[f"{prefix}_sol_dS_psnr"] = r.sol_dS_psnr
    np.savez_compressed(output_path, **data)
    print(f"Results saved: {output_path}")


# ============================================================================
# CLI
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GVRT MultiPeak Round Trip",
    )
    parser.add_argument(
        "--part", type=int, default=0,
        help="Run specific part (1-5). 0 = all parts.",
    )
    parser.add_argument(
        "--n-spectra", type=int, default=None,
        help="Override spectra count per cell.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory.",
    )
    args = parser.parse_args()

    n1 = args.n_spectra or 100_000
    n2 = args.n_spectra or 50_000

    all_results = {}

    if args.part in (0, 1):
        basic = part1_basic(n1)
        all_results["basic"] = [basic]

    if args.part in (0, 2):
        sep_results = part2_separation_sweep(n2)
        all_results["separation"] = sep_results

    if args.part in (0, 3):
        ratio_results = part3_amplitude_sweep(n2)
        all_results["ratio"] = ratio_results

    if args.part in (0, 4):
        noise_results = part4_noise_sweep(n2)
        all_results["noise"] = noise_results

    if args.part in (0, 5):
        # Use basic result for decomposition; run it if not done yet
        if "basic" not in all_results:
            basic = part1_basic(n1)
            all_results["basic"] = [basic]
        part5_error_decomposition(all_results["basic"][0])

    # Visualization (only if all parts ran)
    if args.part == 0 and all(k in all_results for k in ("basic", "separation", "ratio", "noise")):
        plot_results(
            all_results["separation"],
            all_results["ratio"],
            all_results["noise"],
            all_results["basic"][0],
            output_path=args.output,
        )

        # Save
        output_dir = Path(args.output) if args.output else (
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / "output"
            / "gvrt_multipeak"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        save_results(all_results, str(output_dir / "session72_results.npz"))


if __name__ == "__main__":
    main()
