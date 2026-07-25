"""
Multi-Image GVRT Generalization Benchmark
=============================================================

Runs SinglePeakEncoder round-trip (encode → exact Voigt → solver → decode)
across multiple images to verify PSNR consistency.

Key question: is solver quality image-independent?
Expected: yes, because each pixel is processed independently.

Images: whatever subdirectories exist under the roundtrip image dir
Solver: 4-step (default)

Usage:
    uv run python -m toyomacro.voigtfit.benchmarks.bench_multi_image
    uv run python -m toyomacro.voigtfit.benchmarks.bench_multi_image --max-pixels 500000
    uv run python -m toyomacro.voigtfit.benchmarks.bench_multi_image --images <name> [<name> ...]
"""

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import special as sps

from toyomacro.voigtfit.param_encoder import (
    C1S_SINGLE_PRESET,
    SinglePeakEncoder,
    SinglePeakPreset,
)
from toyomacro.voigtfit.pipeline import HybridPipeline
from toyomacro.voigtfit.weight_cache import WeightMatrixCache

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
from ._data_paths import roundtrip_image_dir

IMAGE_DIR = roundtrip_image_dir()

# name → image path, discovered from the subdirectories of IMAGE_DIR at
# import time. Nothing is bundled and no file name is assumed: drop an
# image into <IMAGE_DIR>/<name>/ and it becomes selectable as <name>.
from ._data_paths import find_default_image


def _discover_images() -> dict[str, Path]:
    catalog: dict[str, Path] = {}
    if IMAGE_DIR.is_dir():
        for sub in sorted(p for p in IMAGE_DIR.iterdir() if p.is_dir()):
            img = find_default_image(sub)
            if img is not None:
                catalog[sub.name] = img
    return catalog


IMAGE_CATALOG = _discover_images()

AMPLITUDE_SCALE = 1000.0
BG_FRACTION = 0.001
BATCH_SIZE = 500_000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ChannelPSNR:
    """Per-channel PSNR results."""

    amplitude: float  # R channel PSNR (dB)
    shift: float  # G channel PSNR (dB)
    fwhm: float  # B channel PSNR (dB)
    mean: float  # Mean of three channels


@dataclass
class ImageRoundtripResult:
    """Result from one image's round-trip."""

    name: str
    image_shape: tuple[int, int]
    n_pixels: int
    psnr: ChannelPSNR
    gen_time: float  # seconds
    fit_time: float  # seconds
    throughput: float  # spectra / s


# ---------------------------------------------------------------------------
# PSNR computation
# ---------------------------------------------------------------------------


def _channel_psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    """PSNR between two uint8 channel images. MAX_I = 255."""
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse < 1e-20:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def compute_channel_psnr(
    original: np.ndarray, reconstructed: np.ndarray
) -> ChannelPSNR:
    """Compute per-channel PSNR for RGB images (H,W,3) uint8."""
    r = _channel_psnr(original[:, :, 0], reconstructed[:, :, 0])
    g = _channel_psnr(original[:, :, 1], reconstructed[:, :, 1])
    b = _channel_psnr(original[:, :, 2], reconstructed[:, :, 2])
    return ChannelPSNR(amplitude=r, shift=g, fwhm=b, mean=(r + g + b) / 3)


# ---------------------------------------------------------------------------
# Image loading & downscaling
# ---------------------------------------------------------------------------


def load_and_downscale(
    path: Path, max_pixels: int | None = None
) -> np.ndarray:
    """Load image, convert to RGB uint8, optionally downscale."""
    from toyomacro.voigtfit.image_utils import load_image

    image = load_image(path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    if max_pixels is not None:
        H, W = image.shape[:2]
        n_pixels = H * W
        if n_pixels > max_pixels:
            from PIL import Image as PILImage

            scale = np.sqrt(max_pixels / n_pixels)
            new_W = max(1, int(W * scale))
            new_H = max(1, int(H * scale))
            pil_img = PILImage.fromarray(image)
            pil_img = pil_img.resize((new_W, new_H), PILImage.LANCZOS)
            image = np.array(pil_img)

    return image


# ---------------------------------------------------------------------------
# Core round-trip function
# ---------------------------------------------------------------------------


def roundtrip_single_image(
    image: np.ndarray,
    preset: SinglePeakPreset = C1S_SINGLE_PRESET,
    amplitude_scale: float = AMPLITUDE_SCALE,
    bg_fraction: float = BG_FRACTION,
    batch_size: int = BATCH_SIZE,
    solver: str = "4step",
) -> tuple[ChannelPSNR, float, float]:
    """Run noiseless SinglePeakEncoder round-trip on one image.

    Pipeline: RGB → encode (amp, δE, FWHM) → exact Voigt spectra
              → solver → decode → RGB → PSNR

    Parameters
    ----------
    image : ndarray (H, W, 3) uint8
    preset : SinglePeakPreset
    amplitude_scale : float
    bg_fraction : float
    batch_size : int
    solver : str
        ``"4step"`` (default) uses Taylor-based 4-step solver.
        ``"parabola"`` uses dict2d + parabola sub-grid interpolation
        (Jacobian-free, higher δσ accuracy).

    Returns
    -------
    psnr : ChannelPSNR
    gen_time : float (seconds)
    fit_time : float (seconds)
    """
    H, W = image.shape[:2]
    n_pixels = H * W

    # --- Encode ---
    encoder = SinglePeakEncoder(preset)
    gt_amp, gt_dE, gt_fwhm = encoder.encode(image)
    gt_dsigma = encoder.delta_sigma(gt_fwhm)

    # --- Setup pipeline ---
    energy = preset.energy
    center = preset.element.binding_energy
    sigma_nom = preset.element.sigma
    gamma = preset.element.gamma

    peak_config = {
        "centers": np.array([center], dtype=np.float32),
        "sigmas": np.array([sigma_nom], dtype=np.float32),
        "gamma": gamma,
    }

    cache = WeightMatrixCache()
    pipeline = HybridPipeline(
        cache=cache,
        enable_shift_correction=True,
        enable_stage2=False,
        use_mlx=True,
    )

    # --- Process in batches ---
    all_amps = []
    all_dE = []
    all_dsigma = []
    gen_time = 0.0
    fit_time = 0.0

    SQRT2 = np.sqrt(2.0)
    SQRT2PI = np.sqrt(2.0 * np.pi)

    for start in range(0, n_pixels, batch_size):
        end = min(start + batch_size, n_pixels)
        amp_b = gt_amp[start:end]
        dE_b = gt_dE[start:end]
        ds_b = gt_dsigma[start:end]

        # Generate exact Voigt spectra (not Taylor)
        t0 = time.perf_counter()
        energy_f64 = energy.astype(np.float64)
        centers_b = (center + dE_b.astype(np.float64))[:, np.newaxis]
        sigmas_b = (sigma_nom + ds_b.astype(np.float64))[:, np.newaxis]
        z = (
            (energy_f64[np.newaxis, :] - centers_b) + 1j * gamma
        ) / (sigmas_b * SQRT2)
        profiles = np.real(sps.wofz(z)) / (sigmas_b * SQRT2PI)

        scaled_amp = (amp_b * amplitude_scale).astype(np.float64)
        spectra = scaled_amp[:, np.newaxis] * profiles
        spectra += (scaled_amp * bg_fraction)[:, np.newaxis]
        spectra = spectra.astype(np.float32)
        gen_time += time.perf_counter() - t0

        # Fit
        t0 = time.perf_counter()
        if solver == "parabola":
            result = pipeline.process_rowmajor_dictionary_hybrid(
                spectra,
                preset.element.symbol,
                preset.element.orbital,
                energy,
                peak_config,
                amplitudes_only=True,
                solver="parabola",
            )
        else:
            result = pipeline.process_rowmajor_extended_3param(
                spectra,
                preset.element.symbol,
                preset.element.orbital,
                energy,
                peak_config,
                amplitudes_only=True,
                n_steps=4,
            )
        all_amps.append(result.amplitudes[0, :])
        all_dE.append(result.energy_shifts)
        all_dsigma.append(result.sigma_shifts)
        fit_time += time.perf_counter() - t0

        del spectra, result
        gc.collect()

    # --- Decode ---
    rec_amp = np.concatenate(all_amps)
    rec_dE = np.concatenate(all_dE)
    rec_dsigma = np.concatenate(all_dsigma)
    rec_fwhm = encoder.fwhm_from_sigma(sigma_nom + rec_dsigma)

    rec_image = encoder.decode(
        rec_amp / amplitude_scale, rec_dE, rec_fwhm, (H, W)
    )

    # --- PSNR ---
    psnr = compute_channel_psnr(image, rec_image)

    return psnr, gen_time, fit_time


# ---------------------------------------------------------------------------
# Multi-image benchmark runner
# ---------------------------------------------------------------------------


def run_multi_image_benchmark(
    image_names: list[str] | None = None,
    max_pixels: int | None = None,
    verbose: bool = True,
    solver: str = "4step",
) -> list[ImageRoundtripResult]:
    """Run round-trip on multiple images, return comparative results.

    Parameters
    ----------
    image_names : list of str or None
        Image names from IMAGE_CATALOG. None = all 4 images.
    max_pixels : int or None
        Downscale images to at most this many pixels.
    verbose : bool
    solver : str
        ``"4step"`` or ``"parabola"``.

    Returns
    -------
    results : list of ImageRoundtripResult
    """
    if image_names is None:
        image_names = list(IMAGE_CATALOG.keys())

    results: list[ImageRoundtripResult] = []

    for name in image_names:
        subdir, filename = IMAGE_CATALOG[name]
        path = IMAGE_DIR / subdir / filename

        if not path.exists():
            if verbose:
                print(f"  SKIP {name}: {path} not found")
            continue

        if verbose:
            print(f"\n--- {name} ---")

        image = load_and_downscale(path, max_pixels)
        H, W = image.shape[:2]
        n_pixels = H * W

        if verbose:
            print(f"  Image: {W}×{H} ({n_pixels:,} px)")

        psnr, gen_time, fit_time = roundtrip_single_image(image, solver=solver)
        throughput = n_pixels / fit_time if fit_time > 0 else float("inf")

        r = ImageRoundtripResult(
            name=name,
            image_shape=(H, W),
            n_pixels=n_pixels,
            psnr=psnr,
            gen_time=gen_time,
            fit_time=fit_time,
            throughput=throughput,
        )
        results.append(r)

        if verbose:
            print(
                f"  PSNR: R(amp)={psnr.amplitude:.1f}  "
                f"G(δE)={psnr.shift:.1f}  "
                f"B(FWHM)={psnr.fwhm:.1f}  "
                f"avg={psnr.mean:.1f} dB"
            )
            print(
                f"  Gen: {gen_time:.2f}s  Fit: {fit_time:.2f}s  "
                f"({throughput / 1e6:.1f}M spec/s)"
            )

        del image
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# Summary & printing
# ---------------------------------------------------------------------------


def print_summary(results: list[ImageRoundtripResult]) -> None:
    """Print comparison table and consistency analysis."""
    print(f"\n{'=' * 80}")
    print("MULTI-IMAGE GVRT GENERALIZATION BENCHMARK")
    print(f"{'=' * 80}")
    print(
        f"{'Image':<10} {'Shape':>12} {'Pixels':>10} "
        f"{'R(amp)':>8} {'G(δE)':>8} {'B(FWHM)':>8} {'avg':>8} "
        f"{'fit M/s':>8}"
    )
    print(f"{'-' * 80}")

    psnr_means = []
    for r in results:
        H, W = r.image_shape
        print(
            f"{r.name:<10} {W}×{H:>5} {r.n_pixels:>10,} "
            f"{r.psnr.amplitude:>8.1f} {r.psnr.shift:>8.1f} "
            f"{r.psnr.fwhm:>8.1f} {r.psnr.mean:>8.1f} "
            f"{r.throughput / 1e6:>8.1f}"
        )
        psnr_means.append(r.psnr.mean)

    print(f"{'-' * 80}")

    if len(psnr_means) >= 2:
        arr = np.array(psnr_means)
        mean_psnr = arr.mean()
        std_psnr = arr.std()
        spread = arr.max() - arr.min()
        print(f"\nConsistency analysis (n={len(results)} images):")
        print(f"  Mean avg PSNR: {mean_psnr:.1f} dB")
        print(f"  Std:           {std_psnr:.2f} dB")
        print(f"  Spread (max-min): {spread:.2f} dB")

        if spread < 3.0:
            print("  ✓ Image-independent — spread < 3 dB")
        else:
            print(f"  ✗ Spread {spread:.1f} dB ≥ 3 dB — investigate")

    print(f"{'=' * 80}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Image GVRT Generalization Benchmark"
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        choices=list(IMAGE_CATALOG.keys()),
        help="Image names (default: all 4)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Downscale images to at most N pixels (e.g. 500000)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="4step",
        choices=["4step", "parabola"],
        help="Solver: 4step (Taylor) or parabola (dict2d sub-grid)",
    )
    args = parser.parse_args()

    results = run_multi_image_benchmark(
        image_names=args.images,
        max_pixels=args.max_pixels,
        solver=args.solver,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
