"""
GVRT Noise Robustness Visualization
====================================

Noise comparison figure: NF / Moderate (λ=10³) / Strong (λ=10⁴) side-by-side.
Shows that Dict2D+Parabola tracks parameters through noise with graceful
degradation.

Panels:
  (A) Source image with ROI rectangle
  (B) Encoding scheme + noise level legend
  (C) 3×3 grid: noise level (rows) × parameter channel (cols)
  (D) 1 ROI × 3 noise levels
  (E) PSNR comparison table with color coding
"""

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ..param_encoder import (
    C1S_SINGLE_PRESET,
    SinglePeakEncoder,
    SinglePeakPreset,
)
from ..voigt_jacobian import voigt_profile
from .common import (
    NOISE_BADGE_COLORS,
    apply_style,
    psnr_cell_color,
)
from .gvrt_tracking import (
    AMP_SCALE,
    BG_FRACTION,
    SWEEP_DEFS,
    _generate_exact_voigt_spectra,
    _make_gradient,
    _param_psnr,
    select_rois,
)

# ── Default noise levels ────────────────────────────────────────

DEFAULT_NOISE_LEVELS = [
    {'label': 'NF',       'level': None, 'snr_label': '\u221e'},
    {'label': 'Moderate', 'level': 300,  'snr_label': '~33'},
    {'label': 'Strong',   'level': 3000, 'snr_label': '~3'},
]

FINE_NOISE_LEVELS = [
    {'label': 'NF',     'level': None,  'snr_label': '\u221e'},
    {'label': 'L=30',   'level': 30,    'snr_label': '~333'},
    {'label': 'L=100',  'level': 100,   'snr_label': '~100'},
    {'label': 'L=300',  'level': 300,   'snr_label': '~33'},
    {'label': 'L=1K',   'level': 1000,  'snr_label': '~10'},
    {'label': 'L=3K',   'level': 3000,  'snr_label': '~3'},
    {'label': 'L=10K',  'level': 10000, 'snr_label': '~1'},
    {'label': 'L=30K',  'level': 30000, 'snr_label': '~0.3'},
]


# ── Data classes ────────────────────────────────────────────────

@dataclass
class NoisySweepData:
    """Sweep data at one noise level for one channel."""
    channel: str                    # 'R', 'G', 'B'
    noise_label: str                # 'NF', 'Moderate', 'Strong'
    clean_spectra: np.ndarray       # (n_steps, n_E)
    noisy_spectra: np.ndarray | None  # (n_steps, n_E) or None for NF
    fit_spectra: np.ndarray         # (n_steps, n_E)
    nominal_spectrum: np.ndarray    # (n_E,)
    colors: list


@dataclass
class NoisyROIData:
    """ROI data at one noise level."""
    noise_label: str
    name: str
    n_pixels: int
    clean_spectra: np.ndarray       # (N, n_E)
    noisy_spectra: np.ndarray | None  # (N, n_E) or None for NF
    fit_spectra: np.ndarray         # (N, n_E)
    pixel_colors: np.ndarray        # (N, 3) [0,1]
    mean_clean: np.ndarray          # (n_E,)
    mean_noisy: np.ndarray | None  # (n_E,) or None for NF
    mean_fit: np.ndarray            # (n_E,)
    nominal_spectrum: np.ndarray    # (n_E,)


@dataclass
class NoiseLevelResult:
    """Results for one noise level."""
    label: str
    level: float | None
    snr_label: str
    sweeps: dict[str, NoisySweepData]  # {'R', 'G', 'B'}
    roi: NoisyROIData
    amp_psnr: float
    dE_psnr: float
    dsigma_psnr: float
    reconstructed_image: np.ndarray | None = None  # (H, W, 3) uint8


@dataclass
class NoiseComparisonData:
    """Top-level container for noise comparison figure."""
    preset: SinglePeakPreset
    image: np.ndarray
    energy: np.ndarray
    noise_results: list[NoiseLevelResult]
    solver_name: str
    throughput: float
    total_time: float = 0.0


# ── Noise helper ────────────────────────────────────────────────

def _add_noise(spectra: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Add Poisson noise using the spectra_generator infrastructure."""
    from ..spectra_generator import add_poisson_noise
    # Seed with rng for reproducibility
    old_state = np.random.get_state()
    np.random.seed(rng.integers(0, 2**31))
    noisy = add_poisson_noise(
        spectra, level=level,
        global_max=float(spectra.max()),
    )
    np.random.set_state(old_state)
    return noisy


# ── Build pipeline ──────────────────────────────────────────────

def _build_noisy_sweeps(
    energy: np.ndarray,
    center: float,
    sigma_nom: float,
    gamma: float,
    dict_cache,
    noise_label: str,
    level: float | None,
    rng: np.random.Generator,
    bg_fraction: float = BG_FRACTION,
) -> dict[str, NoisySweepData]:
    """Build sweep data for 3 channels at one noise level."""
    from ..dictionary_solver import solve_dict2d_parabola

    sweeps = {}
    for ch in ['R', 'G', 'B']:
        sdef = SWEEP_DEFS[ch]
        amp = sdef['amp']
        dE = sdef['dE']
        dsigma = sdef['dsigma']

        clean = _generate_exact_voigt_spectra(
            energy, center, sigma_nom, gamma, amp, dE, dsigma,
            bg_fraction=bg_fraction,
        )

        if level is not None:
            noisy = _add_noise(clean, level, rng)
            input_spectra = noisy
        else:
            noisy = None
            input_spectra = clean

        fit_amps, _, fit_dE, fit_dsigma, _ = solve_dict2d_parabola(
            input_spectra, dict_cache,
        )
        fit_amp_arr = fit_amps[0, :] / AMP_SCALE

        fit_spectra = _generate_exact_voigt_spectra(
            energy, center, sigma_nom, gamma,
            fit_amp_arr, fit_dE, fit_dsigma,
            bg_fraction=bg_fraction,
        )

        nom = 0.5 * AMP_SCALE * voigt_profile(energy, center, sigma_nom, gamma)
        nom = nom + 0.5 * AMP_SCALE * bg_fraction

        sweeps[ch] = NoisySweepData(
            channel=ch,
            noise_label=noise_label,
            clean_spectra=clean,
            noisy_spectra=noisy,
            fit_spectra=fit_spectra,
            nominal_spectrum=nom.astype(np.float32),
            colors=_make_gradient(ch, len(amp)),
        )

    return sweeps


def _build_noisy_roi(
    image: np.ndarray,
    energy: np.ndarray,
    center: float,
    sigma_nom: float,
    gamma: float,
    gt_amp: np.ndarray,
    gt_dE: np.ndarray,
    gt_dsigma: np.ndarray,
    dict_cache,
    name: str,
    color: str,
    rect: tuple[int, int, int, int],
    flat_indices: np.ndarray,
    noise_label: str,
    level: float | None,
    rng: np.random.Generator,
    bg_fraction: float = BG_FRACTION,
) -> NoisyROIData:
    """Build ROI data at one noise level."""
    from ..dictionary_solver import solve_dict2d_parabola

    H, W = image.shape[:2]
    rows = flat_indices // W
    cols = flat_indices % W
    pixel_colors = image[rows, cols].astype(np.float32) / 255.0

    roi_amp = gt_amp[flat_indices]
    roi_dE = gt_dE[flat_indices]
    roi_dsigma = gt_dsigma[flat_indices]

    clean = _generate_exact_voigt_spectra(
        energy, center, sigma_nom, gamma,
        roi_amp, roi_dE, roi_dsigma,
        bg_fraction=bg_fraction,
    )

    if level is not None:
        noisy = _add_noise(clean, level, rng)
        input_spectra = noisy
    else:
        noisy = None
        input_spectra = clean

    fit_amps, _, fit_dE, fit_dsigma, _ = solve_dict2d_parabola(
        input_spectra, dict_cache,
    )
    fit_amp_arr = fit_amps[0, :] / AMP_SCALE

    fit_spectra = _generate_exact_voigt_spectra(
        energy, center, sigma_nom, gamma,
        fit_amp_arr, fit_dE, fit_dsigma,
        bg_fraction=bg_fraction,
    )

    mean_clean = clean.mean(axis=0)
    mean_noisy = noisy.mean(axis=0) if noisy is not None else None
    mean_fit = fit_spectra.mean(axis=0)

    med_amp = float(np.median(roi_amp))
    nom = med_amp * AMP_SCALE * voigt_profile(energy, center, sigma_nom, gamma)
    nom = nom + med_amp * AMP_SCALE * bg_fraction

    return NoisyROIData(
        noise_label=noise_label,
        name=name,
        n_pixels=len(flat_indices),
        clean_spectra=clean,
        noisy_spectra=noisy,
        fit_spectra=fit_spectra,
        pixel_colors=pixel_colors,
        mean_clean=mean_clean.astype(np.float32),
        mean_noisy=mean_noisy.astype(np.float32) if mean_noisy is not None else None,
        mean_fit=mean_fit.astype(np.float32),
        nominal_spectrum=nom.astype(np.float32),
    )


def build_noise_comparison_data(
    image_path: str,
    preset: SinglePeakPreset = C1S_SINGLE_PRESET,
    noise_levels: list[dict] | None = None,
    max_height: int = 540,
    roi_preset: str = 'demo',
    roi_name: str = 'sky',
    n_roi_samples: int = 500,
    seed: int = 42,
    verbose: bool = True,
    skip_details: bool = False,
    bg_fraction: float = BG_FRACTION,
) -> NoiseComparisonData:
    """Build all data for the noise comparison figure.

    Args:
        image_path: Path to source RGB image
        preset: SinglePeakPreset (default C1s)
        noise_levels: List of dicts with 'label', 'level', 'snr_label'
        max_height: Downsample to this height
        roi_preset: ROI preset name
        roi_name: Which ROI to use (default 'sky')
        n_roi_samples: Max pixels per ROI
        seed: Random seed for reproducibility
        verbose: Print progress
        skip_details: If True, skip sweeps/ROI (only compute full-image fit
                      + reconstructed images). Faster for image-only figures.

    Returns:
        NoiseComparisonData
    """
    from ..dictionary_solver import build_dictionary, solve_dict2d_parabola
    from ..image_utils import load_image

    if noise_levels is None:
        noise_levels = DEFAULT_NOISE_LEVELS

    rng = np.random.default_rng(seed)

    # 1. Load image
    image = load_image(image_path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    H, W = image.shape[:2]
    if H > max_height:
        from PIL import Image as PILImage
        scale = max_height / H
        new_w = int(W * scale)
        image = np.array(
            PILImage.fromarray(image).resize(
                (new_w, max_height), PILImage.LANCZOS)
        )
        H, W = image.shape[:2]
    N = H * W

    if verbose:
        print(f"Image: {Path(image_path).name} ({W}\u00d7{H}, {N:,} px)")

    # 2. Encode
    encoder = SinglePeakEncoder(preset)
    gt_amp, gt_dE, gt_fwhm_g = encoder.encode(image)
    gt_dsigma = encoder.delta_sigma(gt_fwhm_g)

    energy = preset.energy
    center = preset.element.binding_energy
    sigma_nom = preset.element.sigma
    gamma = preset.element.gamma

    # 3. Generate clean full-image spectra
    if verbose:
        print("Generating clean spectra (full image)...")
    t0 = time.perf_counter()
    clean_spectra = _generate_exact_voigt_spectra(
        energy, center, sigma_nom, gamma, gt_amp, gt_dE, gt_dsigma,
        bg_fraction=bg_fraction,
    )
    global_max = float(clean_spectra.max())

    # 4. Build dictionary (shared)
    dE_margin = 0.2
    ds_margin = 0.02
    dE_lo = min(preset.shift_range.min_val - dE_margin, -0.9)
    dE_hi = max(preset.shift_range.max_val + dE_margin, 0.9)
    ds_lo = min(float(gt_dsigma.min()) - ds_margin, -0.16)
    ds_hi = max(float(gt_dsigma.max()) + ds_margin, 0.16)

    if verbose:
        print(f"Building dictionary: dE=[{dE_lo:.2f},{dE_hi:.2f}], "
              f"ds=[{ds_lo:.3f},{ds_hi:.3f}]")

    dict_cache = build_dictionary(
        energy=energy,
        centers=np.array([center], dtype=np.float32),
        sigmas=np.array([sigma_nom], dtype=np.float32),
        gamma=gamma,
        dE_range=(dE_lo, dE_hi),
        dE_step=0.1 * sigma_nom,
        dsigma_range=(ds_lo, ds_hi),
        dsigma_step=0.02 * sigma_nom,
    )

    if verbose:
        print(f"  Dict: {dict_cache.grid_shape[0]}\u00d7"
              f"{dict_cache.grid_shape[1]} = "
              f"{dict_cache.D.shape[1]} entries")

    # 5. Select ROI (just the requested one) — skip if skip_details
    roi_def = None
    if not skip_details:
        roi_defs = select_rois(image, preset_name=roi_preset,
                               n_sample=n_roi_samples)
        for rname, rcolor, rrect, rflat in roi_defs:
            if rname == roi_name:
                roi_def = (rname, rcolor, rrect, rflat)
                break
        if roi_def is None:
            roi_def = roi_defs[0]  # fallback to first

    # 6. Process each noise level
    throughput = 0.0
    results = []

    for nl in noise_levels:
        label = nl['label']
        level = nl['level']
        snr_label = nl['snr_label']

        if verbose:
            print(f"\nNoise level: {label} (level={level})")

        # Full-image PSNR
        if level is not None:
            noisy_full = _add_noise(clean_spectra, level, rng)
        else:
            noisy_full = clean_spectra

        t_fit0 = time.perf_counter()
        fit_amps, _, fit_dE, fit_dsigma, _ = solve_dict2d_parabola(
            noisy_full, dict_cache,
        )
        fit_time = time.perf_counter() - t_fit0
        if throughput == 0.0:
            throughput = N / fit_time / 1e6
        fit_amp_arr = fit_amps[0, :] / AMP_SCALE

        amp_psnr = _param_psnr(gt_amp, fit_amp_arr)
        dE_psnr = _param_psnr(gt_dE, fit_dE)
        dsigma_psnr = _param_psnr(gt_dsigma, fit_dsigma)

        if verbose:
            print(f"  Fit: {fit_time:.2f}s ({N / fit_time / 1e6:.1f}M/s)")
            print(f"  PSNR: amp={amp_psnr:.1f}dB  "
                  f"\u03b4E={dE_psnr:.1f}dB  "
                  f"\u03b4\u03c3={dsigma_psnr:.1f}dB")

        # Reconstructed image: fit params → RGB
        fit_sigma = sigma_nom + fit_dsigma
        fit_fwhm_g = encoder.fwhm_from_sigma(fit_sigma)
        reconstructed = encoder.decode(fit_amp_arr, fit_dE, fit_fwhm_g, (H, W))

        # Sweeps (always computed — trivially fast: 21 spectra per level)
        if verbose:
            print("  Building sweeps...")
        sweeps = _build_noisy_sweeps(
            energy, center, sigma_nom, gamma, dict_cache,
            label, level, rng, bg_fraction=bg_fraction,
        )

        # ROI (skip if skip_details)
        roi_data = None
        if not skip_details:
            rname, rcolor, rrect, rflat = roi_def
            if verbose:
                print(f"  Building ROI '{rname}' ({len(rflat)} px)...")
            roi_data = _build_noisy_roi(
                image, energy, center, sigma_nom, gamma,
                gt_amp, gt_dE, gt_dsigma, dict_cache,
                rname, rcolor, rrect, rflat,
                label, level, rng, bg_fraction=bg_fraction,
            )

        results.append(NoiseLevelResult(
            label=label, level=level, snr_label=snr_label,
            sweeps=sweeps, roi=roi_data,
            amp_psnr=amp_psnr, dE_psnr=dE_psnr,
            dsigma_psnr=dsigma_psnr,
            reconstructed_image=reconstructed,
        ))

        # Release large per-iteration buffers to avoid MLX GPU cache buildup
        # (critical at 8K where each noisy_full is ~17 GB)
        del noisy_full, fit_amps, fit_dE, fit_dsigma, fit_amp_arr, reconstructed
        try:
            import mlx.core as _mx
            _mx.metal.clear_cache()
        except Exception:
            pass
        import gc as _gc
        _gc.collect()

    total_time = time.perf_counter() - t0
    if verbose:
        print(f"\nTotal: {total_time:.1f}s")

    return NoiseComparisonData(
        preset=preset, image=image, energy=energy,
        noise_results=results,
        solver_name='Dict2D+Parabola',
        throughput=throughput,
        total_time=total_time,
    )


# ── Panel drawing ───────────────────────────────────────────────

def _draw_panel_a(ax: plt.Axes, data: NoiseComparisonData) -> None:
    """Panel A: Source image with ROI rectangle."""
    ax.imshow(data.image)
    roi = data.noise_results[0].roi
    r0, c0, r1, c1 = _get_roi_rect(data)
    rect = mpatches.Rectangle(
        (c0, r0), c1 - c0, r1 - r0,
        linewidth=2, edgecolor='#1E88E5', facecolor='none',
        linestyle='-', zorder=5,
    )
    ax.add_patch(rect)
    ax.text(
        c0 + 3, r0 - 4, roi.name,
        fontsize=8, fontweight='bold', color='#1E88E5',
        path_effects=[
            patheffects.withStroke(linewidth=2.5, foreground='white'),
        ],
        zorder=6,
    )
    H, W = data.image.shape[:2]
    ax.set_title(f'(A) Source Image ({W}\u00d7{H})', fontsize=10, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])


def _get_roi_rect(data: NoiseComparisonData) -> tuple[int, int, int, int]:
    """Extract ROI rect from roi_def stored in the first noise result."""
    roi = data.noise_results[0].roi
    # Reconstruct rect from pixel coords
    H, W = data.image.shape[:2]
    rows = np.arange(H * W)  # unused
    # We don't store the rect directly, so re-select
    roi_defs = select_rois(data.image, preset_name='demo')
    for rname, rcolor, rrect, rflat in roi_defs:
        if rname == roi.name:
            return rrect
    return (0, 0, 10, 10)


def _draw_panel_b(ax: plt.Axes, data: NoiseComparisonData) -> None:
    """Panel B: Encoding scheme + noise conditions."""
    p = data.preset
    sep = "\u2500" * 28
    lines = [
        "RGB \u2192 Parameter Encoding",
        sep,
        f"R \u2192 amplitude [{p.amplitude_range.min_val:.1f}, "
        f"{p.amplitude_range.max_val:.1f}]",
        f"G \u2192 \u03b4E shift  [{p.shift_range.min_val:+.1f}, "
        f"{p.shift_range.max_val:+.1f}] eV",
        f"B \u2192 FWHM_G    [{p.fwhm_range.min_val:.1f}, "
        f"{p.fwhm_range.max_val:.1f}] eV",
        "",
        f"Center: {data.preset.element.binding_energy:.1f} eV (C 1s)",
        f"Solver: {data.solver_name}",
        f"{data.throughput:.1f} M spectra/s",
        "",
        sep,
        "Noise Conditions",
        sep,
    ]
    for nr in data.noise_results:
        lines.append(f"  {nr.label:10s}  SNR \u2248 {nr.snr_label}")
    text = "\n".join(lines)

    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes, va='top', ha='left',
        fontsize=8, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                  edgecolor='#CCCCCC', alpha=0.9),
    )
    ax.axis('off')
    ax.set_title('(B) Encoding & Noise', fontsize=10, pad=4)


def _draw_noisy_sweep_subpanel(
    ax: plt.Axes,
    sweep: NoisySweepData,
    energy: np.ndarray,
    show_ylabel: bool = False,
    show_title: bool = True,
) -> None:
    """Draw one sweep subpanel with optional clean silhouette."""
    n = len(sweep.colors)
    is_noisy = sweep.noisy_spectra is not None

    # Nominal reference (gray dashed)
    ax.plot(energy, sweep.nominal_spectrum, color='#888888',
            linestyle='--', linewidth=1.0, zorder=1)

    if is_noisy:
        # Clean silhouette (thin gray, alpha=0.3)
        for i in range(n):
            ax.plot(energy, sweep.clean_spectra[i],
                    color='#888888', linewidth=0.8, alpha=0.3, zorder=1.5)

        # Noisy GT (colored gradient)
        for i in range(n):
            ax.plot(energy, sweep.noisy_spectra[i], color=sweep.colors[i],
                    linewidth=1.2, alpha=0.8, zorder=2)
    else:
        # NF: clean GT (colored gradient)
        for i in range(n):
            ax.plot(energy, sweep.clean_spectra[i], color=sweep.colors[i],
                    linewidth=1.5, zorder=2)

    # Fit spectra (black dotted)
    for i in range(n):
        ax.plot(energy, sweep.fit_spectra[i], color='black',
                linestyle=':', linewidth=0.8, alpha=0.7, zorder=3)

    ax.invert_xaxis()
    if show_title:
        ch_labels = {'R': 'R: amplitude', 'G': 'G: \u03b4E shift', 'B': 'B: FWHM'}
        ax.set_title(ch_labels[sweep.channel], fontsize=8, fontweight='bold', pad=3)
    ax.tick_params(labelsize=5.5)
    if show_ylabel:
        ax.set_ylabel('Intensity', fontsize=7)


def _draw_noisy_roi_subpanel(
    ax: plt.Axes,
    roi: NoisyROIData,
    energy: np.ndarray,
    show_legend: bool = False,
) -> None:
    """Draw ROI band at one noise level."""
    is_noisy = roi.noisy_spectra is not None

    if is_noisy:
        # Clean band (thin gray background)
        for i in range(roi.n_pixels):
            ax.plot(energy, roi.clean_spectra[i],
                    color='#AAAAAA', linewidth=0.3, alpha=0.02, zorder=0.5)

        # Noisy pixel spectra (colored by pixel RGB)
        for i in range(roi.n_pixels):
            rgba = (*roi.pixel_colors[i], 0.04)
            ax.plot(energy, roi.noisy_spectra[i], color=rgba,
                    linewidth=0.3, zorder=1)

        # Mean clean (gray solid)
        ax.plot(energy, roi.mean_clean, color='#888888',
                linewidth=1.2, label='Mean clean', zorder=2)

        # Mean noisy (blue thick)
        ax.plot(energy, roi.mean_noisy, color='#1E88E5',
                linewidth=2.0, label='Mean noisy', zorder=3)
    else:
        # NF: clean pixel spectra
        for i in range(roi.n_pixels):
            rgba = (*roi.pixel_colors[i], 0.05)
            ax.plot(energy, roi.clean_spectra[i], color=rgba,
                    linewidth=0.4, zorder=1)

        # Mean GT (blue)
        ax.plot(energy, roi.mean_clean, color='#1E88E5',
                linewidth=2.0, label='Mean GT', zorder=3)

    # Mean fit (black dashed)
    ax.plot(energy, roi.mean_fit, color='black',
            linestyle='--', linewidth=1.5, label='Mean fit', zorder=4)

    # Nominal
    ax.plot(energy, roi.nominal_spectrum, color='#888888',
            linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)

    ax.invert_xaxis()
    ax.set_title(f'{roi.name} \u2014 {roi.noise_label}',
                 fontsize=8, fontweight='bold', pad=3)
    ax.tick_params(labelsize=5.5)

    if show_legend:
        ax.legend(loc='upper left', fontsize=6, framealpha=0.85,
                  handlelength=1.2, borderpad=0.3)


def _draw_noise_badges(ax: plt.Axes, noise_results: list[NoiseLevelResult]) -> None:
    """Draw noise level badges vertically."""
    ax.axis('off')
    n = len(noise_results)
    for i, nr in enumerate(noise_results):
        y = 1.0 - (i + 0.5) / n
        color = NOISE_BADGE_COLORS.get(nr.label, '#999999')

        # Badge box
        ax.text(
            0.5, y, f"{nr.label}\nSNR \u2248 {nr.snr_label}",
            transform=ax.transAxes, va='center', ha='center',
            fontsize=7.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=color,
                      edgecolor='none', alpha=0.25),
            color=color,
        )


def _draw_psnr_table(ax: plt.Axes, data: NoiseComparisonData) -> None:
    """Panel E: PSNR comparison table with color coding."""
    ax.axis('off')

    cols = ['Noise Level', 'amp (dB)', '\u03b4E (dB)', '\u03b4\u03c3 (dB)']
    rows = []
    cell_colors = []

    for nr in data.noise_results:
        rows.append([
            f"{nr.label} (SNR\u2248{nr.snr_label})",
            f"{nr.amp_psnr:.1f}",
            f"{nr.dE_psnr:.1f}",
            f"{nr.dsigma_psnr:.1f}",
        ])
        row_colors = [
            '#F5F5F5',
            psnr_cell_color(nr.amp_psnr) + '30',
            psnr_cell_color(nr.dE_psnr) + '30',
            psnr_cell_color(nr.dsigma_psnr) + '30',
        ]
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        cellColours=cell_colors,
        colColours=['#E0E0E0'] * 4,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)

    # Header row bold
    for j in range(len(cols)):
        table[0, j].set_text_props(fontweight='bold')

    # Footer line
    H, W = data.image.shape[:2]
    footer = (
        f"{data.solver_name}   {data.throughput:.1f} M/s   "
        f"{W}\u00d7{H} ({W * H:,} spectra)   "
        f"Rate: noise-independent"
    )
    ax.text(
        0.5, -0.05, footer,
        transform=ax.transAxes, va='top', ha='center',
        fontsize=7.5, color='#666666', fontfamily='monospace',
    )
    ax.set_title('(E) Accuracy Summary \u2014 Noise Comparison', fontsize=10, pad=4)


# ── Main figure assembly ────────────────────────────────────────

def plot_noise_comparison(
    data: NoiseComparisonData,
    figsize: tuple[float, float] = (20, 18),
    output_path: str | None = None,
    show: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    """Generate the GVRT Noise Robustness Comparison figure.

    Layout (6 rows x 4 cols):
      Row 0: (A) image + (B) encoding
      Row 1: C sweep NF     (R, G, B) + badge
      Row 2: C sweep Mod    (R, G, B) + badge
      Row 3: C sweep Strong (R, G, B) + badge
      Row 4: D ROI (NF, Mod, Strong)  + legend
      Row 5: E PSNR table

    Args:
        data: NoiseComparisonData from build_noise_comparison_data()
        figsize: Figure size in inches
        output_path: Save path (PNG/PDF)
        show: Show interactively
        dpi: Output DPI

    Returns:
        matplotlib Figure
    """
    apply_style()
    fig = plt.figure(figsize=figsize, facecolor='white')

    gs = GridSpec(
        nrows=6, ncols=4, figure=fig,
        height_ratios=[3, 2.5, 2.5, 2.5, 2.5, 1.2],
        width_ratios=[1, 1, 1, 0.5],
        hspace=0.35, wspace=0.25,
        left=0.05, right=0.95, top=0.96, bottom=0.03,
    )

    # Row 0: (A) Image + (B) Encoding
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])

    _draw_panel_a(ax_a, data)
    _draw_panel_b(ax_b, data)

    # Rows 1-3: Panel C (noise × channel)
    n_noise = len(data.noise_results)
    for i, nr in enumerate(data.noise_results):
        for j, ch in enumerate(['R', 'G', 'B']):
            ax = fig.add_subplot(gs[1 + i, j])
            _draw_noisy_sweep_subpanel(
                ax, nr.sweeps[ch], data.energy,
                show_ylabel=(j == 0),
                show_title=(i == 0),
            )
            # Row label on leftmost column
            if j == 0:
                ax.set_ylabel(
                    f'{nr.label}\n(SNR\u2248{nr.snr_label})',
                    fontsize=7.5, fontweight='bold',
                )

    # Panel C section title
    ax_c0 = fig.axes[2]  # first sweep axis
    ax_c0.text(
        -0.02, 1.25,
        '(C) Synthetic Demo \u2014 Noise Robustness',
        transform=ax_c0.transAxes, fontsize=10, fontweight='bold',
        va='bottom', ha='left',
    )

    # Noise badges (right column, rows 1-3)
    ax_badges = fig.add_subplot(gs[1:1 + n_noise, 3])
    _draw_noise_badges(ax_badges, data.noise_results)

    # Row 4: Panel D (ROI × noise)
    for i, nr in enumerate(data.noise_results):
        ax = fig.add_subplot(gs[4, i])
        _draw_noisy_roi_subpanel(
            ax, nr.roi, data.energy,
            show_legend=(i == 0),
        )
        if i == 0:
            ax.set_ylabel('Intensity', fontsize=7)

    # Panel D section title
    ax_d0 = fig.add_subplot(gs[4, 3])
    ax_d0.axis('off')
    ax_d0.text(
        0.1, 0.95,
        '(D) ROI Band\n\u2014 Noise\nComparison',
        transform=ax_d0.transAxes, fontsize=8, fontweight='bold',
        va='top', ha='left',
    )
    # Legend entries
    legend_items = [
        ('Nominal', '#888888', '--', 0.8),
        ('Mean GT/clean', '#888888', '-', 1.2),
        ('Mean noisy', '#1E88E5', '-', 2.0),
        ('Mean fit', 'black', '--', 1.5),
    ]
    for idx, (lbl, col, ls, lw) in enumerate(legend_items):
        y = 0.65 - idx * 0.15
        ax_d0.plot([0.1, 0.35], [y, y], color=col, linestyle=ls,
                   linewidth=lw, transform=ax_d0.transAxes, clip_on=False)
        ax_d0.text(0.40, y, lbl, transform=ax_d0.transAxes,
                   fontsize=6.5, va='center')

    # Row 5: Panel E (PSNR table)
    ax_e = fig.add_subplot(gs[5, :])
    _draw_psnr_table(ax_e, data)

    # Save / show
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi,
                    facecolor='white', edgecolor='none')

    if show:
        matplotlib.use('macosx')
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_reconstructed_images(
    data: NoiseComparisonData,
    figsize: tuple[float, float] = (18, 5),
    output_path: str | None = None,
    show: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    """Plot original + reconstructed images at each noise level.

    Layout: 1 row of (1 + N) panels: Original | NF recon | Moderate recon | Strong recon

    Args:
        data: NoiseComparisonData (must have reconstructed_image populated)
        figsize: Figure size
        output_path: Save path
        show: Show interactively
        dpi: Output DPI

    Returns:
        matplotlib Figure
    """
    apply_style()
    n = len(data.noise_results)
    fig, axes = plt.subplots(1, n + 1, figsize=figsize, facecolor='white')

    # Original
    axes[0].imshow(data.image)
    axes[0].set_title('Original', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    for i, nr in enumerate(data.noise_results):
        ax = axes[i + 1]
        if nr.reconstructed_image is not None:
            ax.imshow(nr.reconstructed_image)
        else:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
        psnr_text = (
            f"amp={nr.amp_psnr:.1f}  "
            f"\u03b4E={nr.dE_psnr:.1f}  "
            f"\u03b4\u03c3={nr.dsigma_psnr:.1f} dB"
        )
        ax.set_title(f'{nr.label} (SNR\u2248{nr.snr_label})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel(psnr_text, fontsize=7.5, color='#555555')
        ax.axis('off')

    fig.suptitle(
        f'GVRT Reconstructed Images \u2014 {data.solver_name}  '
        f'{data.throughput:.1f} M/s',
        fontsize=12, fontweight='bold', y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi,
                    facecolor='white', edgecolor='none')

    if show:
        matplotlib.use('macosx')
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_channel_noise_grid(
    data: NoiseComparisonData,
    figsize: tuple[float, float] | None = None,
    output_path: str | None = None,
    show: bool = False,
    dpi: int = 150,
    target_width_px: int | None = None,
    target_height_px: int | None = None,
) -> plt.Figure:
    """Plot channel noise grid with spectral interleaves.

    Layout: 7 content rows \u00d7 (1 + N) cols + PSNR footer:
      Row 0: composite RGB images
      Row 1: R sweep spectra (amplitude variation)
      Row 2: R channel images
      Row 3: G sweep spectra (\u03b4E variation)
      Row 4: G channel images
      Row 5: B sweep spectra (FWHM variation)
      Row 6: B channel images
      Row 7: PSNR info

    Column 0 = Original, Columns 1..N = noise levels.
    Each spectral subplot shows clean (gray) + noisy (colored) + fit (black).

    Args:
        data: NoiseComparisonData (needs sweep data in noise_results)
        figsize: Figure size (auto-scaled if None)
        output_path: Save path (PNG/PDF)
        show: Show interactively
        dpi: Output DPI

    Returns:
        matplotlib Figure
    """
    apply_style()
    n_noise = len(data.noise_results)
    n_cols = 1 + n_noise

    # Check if sweep data is available
    has_sweeps = bool(data.noise_results[0].sweeps)

    if has_sweeps:
        # 7 content rows: RGB + (spec_R + img_R) + (spec_G + img_G) + (spec_B + img_B)
        #                  + 1 PSNR row = 8 total
        height_ratios = [1.0, 0.55, 1.0, 0.55, 1.0, 0.55, 1.0, 0.12]
        n_grid_rows = 8
        # Map from channel index -> (spectra_row, image_row)
        ch_layout = [(1, 2), (3, 4), (5, 6)]
    else:
        # 4 content rows: RGB + R + G + B + 1 PSNR row = 5 total
        height_ratios = [1.0, 1.0, 1.0, 1.0, 0.12]
        n_grid_rows = 5
        ch_layout = [(None, 1), (None, 2), (None, 3)]

    if figsize is None:
        fig_h = sum(r for r in height_ratios[:-1]) * 2.5 + 0.5
        figsize = (2.5 * n_cols, fig_h)

    if target_width_px is not None or target_height_px is not None:
        if target_width_px is not None and target_height_px is not None:
            figsize = (target_width_px / dpi, target_height_px / dpi)
        elif target_width_px is not None:
            dpi = int(round(target_width_px / figsize[0]))
        else:
            dpi = int(round(target_height_px / figsize[1]))

    fig = plt.figure(figsize=figsize, facecolor='white')

    gs = GridSpec(
        nrows=n_grid_rows, ncols=n_cols, figure=fig,
        height_ratios=height_ratios,
        hspace=0.18, wspace=0.05,
        left=0.04, right=0.97, top=0.94, bottom=0.02,
    )

    H, W = data.image.shape[:2]
    energy = data.energy
    channel_cmaps = ['Reds', 'Greens', 'Blues']
    channel_keys = ['R', 'G', 'B']
    channel_labels = ['R: amplitude', 'G: \u03b4E shift', 'B: FWHM']
    orig = data.image

    # ── Row 0: RGB composite ──────────────────────────────────────
    for col in range(n_cols):
        ax = fig.add_subplot(gs[0, col])
        if col == 0:
            ax.imshow(orig)
            ax.set_title('Original', fontsize=8, fontweight='bold', pad=3)
        else:
            nr = data.noise_results[col - 1]
            img = nr.reconstructed_image if nr.reconstructed_image is not None else orig
            ax.imshow(img)
            ax.set_title(
                f'{nr.label}\nSNR\u2248{nr.snr_label}',
                fontsize=7, fontweight='bold', pad=3,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel('RGB', fontsize=8, fontweight='bold',
                           rotation=90, labelpad=5)

    # ── Per-channel rows ──────────────────────────────────────────
    for ch_idx, (spec_row, img_row) in enumerate(ch_layout):
        ch_key = channel_keys[ch_idx]
        cmap = channel_cmaps[ch_idx]
        label = channel_labels[ch_idx]

        # Spectra row (if available)
        if spec_row is not None and has_sweeps:
            for col in range(n_cols):
                ax = fig.add_subplot(gs[spec_row, col])
                if col == 0:
                    # Original column: show NF clean sweep as reference
                    nf_sweep = data.noise_results[0].sweeps.get(ch_key)
                    if nf_sweep is not None:
                        _draw_noisy_sweep_subpanel(
                            ax, nf_sweep, energy,
                            show_ylabel=True, show_title=False,
                        )
                    else:
                        ax.axis('off')
                    ax.set_ylabel(label, fontsize=7, fontweight='bold',
                                  rotation=90, labelpad=3)
                else:
                    nr = data.noise_results[col - 1]
                    sweep = nr.sweeps.get(ch_key)
                    if sweep is not None:
                        _draw_noisy_sweep_subpanel(
                            ax, sweep, energy,
                            show_ylabel=False, show_title=False,
                        )
                    else:
                        ax.axis('off')

        # Image row
        for col in range(n_cols):
            ax = fig.add_subplot(gs[img_row, col])
            if col == 0:
                img = orig
            else:
                nr = data.noise_results[col - 1]
                img = nr.reconstructed_image if nr.reconstructed_image is not None else orig
            channel_data = img[:, :, ch_idx]
            ax.imshow(channel_data, cmap=cmap, vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                # Only show label if no spectra row
                if spec_row is None:
                    ax.set_ylabel(label, fontsize=8, fontweight='bold',
                                  rotation=90, labelpad=5)

    # ── PSNR info row ─────────────────────────────────────────────
    psnr_row = n_grid_rows - 1
    for col in range(1, n_cols):
        ax_info = fig.add_subplot(gs[psnr_row, col])
        ax_info.axis('off')
        nr = data.noise_results[col - 1]
        psnr_parts = [
            f"amp {nr.amp_psnr:.0f}",
            f"\u03b4E {nr.dE_psnr:.0f}",
            f"\u03b4\u03c3 {nr.dsigma_psnr:.0f}",
        ]
        ax_info.text(
            0.5, 0.5, ' | '.join([f"{p} dB" for p in psnr_parts]),
            transform=ax_info.transAxes, ha='center', va='center',
            fontsize=5.5, fontfamily='monospace', color='#555555',
        )

    ax_info0 = fig.add_subplot(gs[psnr_row, 0])
    ax_info0.axis('off')
    ax_info0.text(
        0.5, 0.5, 'PSNR:',
        transform=ax_info0.transAxes, ha='center', va='center',
        fontsize=6, fontweight='bold', color='#555555',
    )

    n_total = W * H * n_noise
    fig.suptitle(
        f'GVRT Channel Noise Grid \u2014 {data.solver_name}  '
        f'{data.throughput:.1f} M/s  \u00b7  '
        f'{W}\u00d7{H} \u00d7 {n_noise} levels = {n_total:,} spectra  \u00b7  '
        f'{data.total_time:.1f}s',
        fontsize=10, fontweight='bold', y=0.97,
    )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi,
                    facecolor='white', edgecolor='none')

    if show:
        matplotlib.use('macosx')
        plt.show()
    else:
        plt.close(fig)

    return fig
