"""
GVRT Parameter Tracking Visualization v4
=========================================

5-panel figure showing the GVRT single-peak encoding roundtrip:
  RGB pixel → (amp, δE, δσ) → Voigt spectrum → Dict2D+Parabola fit

Panels:
  (A) Source image with 3 ROI rectangles
  (B) Encoding scheme: R=amplitude, G=δE, B=FWHM
  (C) Synthetic Demo — isolated channel effects (R/G/B sweep)
  (D) Real Image ROI Variants — spectral bands (~500 px each)
  (E) Compact accuracy summary

Panel C shows "what each channel does" in isolation:
  R sweep → height varies (amplitude only)
  G sweep → position varies (δE only)
  B sweep → width varies (δσ only)

Panel D shows real image ROI spectral density bands,
where the band spread direction reveals the dominant parameter.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from ..param_encoder import (
    C1S_SINGLE_PRESET,
    SinglePeakEncoder,
    SinglePeakPreset,
)
from ..voigt_jacobian import voigt_profile
from .common import apply_style

# ── Constants ────────────────────────────────────────────────────────

AMP_SCALE = 1000.0
BG_FRACTION = 0.001
N_SWEEP_STEPS = 7
N_ROI_SAMPLES = 500

# Sweep definitions: (channel, param_name, fixed_amp, dE_values, dsigma_values)
SWEEP_DEFS = {
    'R': {
        'param_name': 'amplitude',
        'amp': np.linspace(0.2, 1.1, N_SWEEP_STEPS).astype(np.float32),
        'dE': np.zeros(N_SWEEP_STEPS, dtype=np.float32),
        'dsigma': np.zeros(N_SWEEP_STEPS, dtype=np.float32),
        'label': 'R: amplitude (height)',
    },
    'G': {
        'param_name': 'delta_E',
        'amp': np.ones(N_SWEEP_STEPS, dtype=np.float32),
        'dE': np.linspace(-0.8, 0.8, N_SWEEP_STEPS).astype(np.float32),
        'dsigma': np.zeros(N_SWEEP_STEPS, dtype=np.float32),
        'label': 'G: \u03b4E shift (position)',
    },
    'B': {
        'param_name': 'delta_sigma',
        'amp': np.ones(N_SWEEP_STEPS, dtype=np.float32),
        'dE': np.zeros(N_SWEEP_STEPS, dtype=np.float32),
        'dsigma': np.linspace(-0.15, 0.15, N_SWEEP_STEPS).astype(np.float32),
        'label': 'B: FWHM (width)',
    },
}

# Color gradients: dark → bright for each channel
SWEEP_COLORS = {
    'R': [(0.45, 0.05, 0.05), (0.95, 0.20, 0.20)],
    'G': [(0.05, 0.30, 0.05), (0.25, 0.85, 0.25)],
    'B': [(0.05, 0.05, 0.45), (0.15, 0.55, 0.95)],
}

# ROI presets: (name, display_color, (frac_r0, frac_c0, frac_r1, frac_c1))
ROI_PRESETS = {
    'fuji': [
        ('sky',   '#1E88E5', (0.02, 0.10, 0.22, 0.35)),
        ('trees', '#43A047', (0.74, 0.05, 0.93, 0.30)),
        ('roof',  '#E53935', (0.37, 0.35, 0.60, 0.55)),
    ],
}


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class SweepData:
    """Data for one synthetic parameter sweep (Panel C subpanel)."""
    channel: str                # 'R', 'G', 'B'
    param_name: str             # 'amplitude', 'delta_E', 'delta_sigma'
    label: str                  # display label
    sweep_values: np.ndarray    # (n_steps,) swept parameter values
    gt_spectra: np.ndarray      # (n_steps, n_E)
    fit_spectra: np.ndarray     # (n_steps, n_E)
    nominal_spectrum: np.ndarray  # (n_E,)
    colors: list                # n_steps RGBA tuples


@dataclass
class ROIData:
    """Data for one image ROI (Panel D subpanel)."""
    name: str                   # 'sky', 'trees', 'roof'
    color: str                  # display color for rectangle
    rect: tuple[int, int, int, int]  # (r0, c0, r1, c1) pixel coords
    n_pixels: int
    pixel_colors: np.ndarray    # (N, 3) float32 [0,1] actual RGB
    gt_spectra: np.ndarray      # (N, n_E)
    mean_gt_spectrum: np.ndarray   # (n_E,)
    mean_fit_spectrum: np.ndarray  # (n_E,)
    nominal_spectrum: np.ndarray   # (n_E,)


@dataclass
class TrackingDataV4:
    """All data needed to render the v4 figure."""
    preset: SinglePeakPreset
    image: np.ndarray
    energy: np.ndarray
    nominal_center: float
    nominal_sigma: float
    nominal_gamma: float
    sweeps: list[SweepData]     # 3 sweeps: R, G, B
    rois: list[ROIData]         # 3 ROIs
    amp_psnr: float
    dE_psnr: float
    dsigma_psnr: float
    solver_name: str
    throughput: float           # M spectra/s


# Backward-compatible aliases
TrackingData = TrackingDataV4


# ── Helpers ──────────────────────────────────────────────────────────

def _make_gradient(channel: str, n: int) -> list:
    """Generate n colors from dark to bright for a channel."""
    lo, hi = SWEEP_COLORS[channel]
    return [
        tuple(lo[c] + (hi[c] - lo[c]) * i / max(n - 1, 1) for c in range(3))
        for i in range(n)
    ]


def _generate_exact_voigt_spectra(
    energy: np.ndarray,
    center: float,
    sigma_nom: float,
    gamma: float,
    amp: np.ndarray,
    dE: np.ndarray,
    dsigma: np.ndarray,
    bg_fraction: float = BG_FRACTION,
) -> np.ndarray:
    """Generate exact Voigt spectra via scipy.special.wofz.

    Adds a constant background = amp * bg_fraction per spectrum. Pass
    ``bg_fraction=0.0`` for pure Voigt (use for solver-only diagnostics;
    the default ``solve_dict2d_parabola`` does not fit this background
    term, so non-zero values leak into amp/δσ bias — see dev-log 108).

    Returns: (N, n_E) float32
    """
    from scipy import special as sps

    energy64 = energy.astype(np.float64)
    SQRT2 = np.sqrt(2.0)
    SQRT2PI = np.sqrt(2.0 * np.pi)

    centers = (center + dE.astype(np.float64))[:, np.newaxis]
    sigmas = (sigma_nom + dsigma.astype(np.float64))[:, np.newaxis]

    z = (energy64[np.newaxis, :] - centers + 1j * gamma) / (sigmas * SQRT2)
    profiles = np.real(sps.wofz(z)) / (sigmas * SQRT2PI)

    scaled_amp = (amp * AMP_SCALE).astype(np.float64)
    spectra = scaled_amp[:, np.newaxis] * profiles
    if bg_fraction != 0.0:
        spectra += (scaled_amp * bg_fraction)[:, np.newaxis]

    return spectra.astype(np.float32)


def _param_psnr(gt: np.ndarray, fit: np.ndarray) -> float:
    """PSNR in parameter space: 10*log10(max_range^2 / MSE)."""
    mse = np.mean((gt.astype(np.float64) - fit.astype(np.float64)) ** 2)
    if mse < 1e-30:
        return float('inf')
    max_range = max(np.ptp(gt), 1e-10)
    return float(10 * np.log10(max_range ** 2 / mse))


# ── Sweep generation ─────────────────────────────────────────────────

def _build_sweep_data(
    energy: np.ndarray,
    center: float,
    sigma_nom: float,
    gamma: float,
    dict_cache,
    channel: str,
) -> SweepData:
    """Generate synthetic parameter sweep data for one channel.

    Sweeps one parameter while keeping the other two fixed at nominal.
    Runs through the full Dict2D+Parabola pipeline.
    """
    from ..dictionary_solver import solve_dict2d_parabola

    sdef = SWEEP_DEFS[channel]
    amp = sdef['amp']
    dE = sdef['dE']
    dsigma = sdef['dsigma']
    n = len(amp)

    # GT spectra
    gt_spectra = _generate_exact_voigt_spectra(
        energy, center, sigma_nom, gamma, amp, dE, dsigma,
    )

    # Fit through solver
    fit_amps, _, fit_dE, fit_dsigma, _ = solve_dict2d_parabola(
        gt_spectra, dict_cache,
    )
    fit_amp_arr = fit_amps[0, :] / AMP_SCALE

    # Reconstruct fit spectra from recovered params
    fit_spectra = _generate_exact_voigt_spectra(
        energy, center, sigma_nom, gamma,
        fit_amp_arr, fit_dE, fit_dsigma,
    )

    # Nominal reference (mid-range amplitude, no shift, no width change)
    nom = 0.5 * AMP_SCALE * voigt_profile(energy, center, sigma_nom, gamma)
    nom = nom + 0.5 * AMP_SCALE * BG_FRACTION

    # Sweep values for display
    if channel == 'R':
        sweep_values = amp
    elif channel == 'G':
        sweep_values = dE
    else:
        sweep_values = dsigma

    return SweepData(
        channel=channel,
        param_name=sdef['param_name'],
        label=sdef['label'],
        sweep_values=sweep_values,
        gt_spectra=gt_spectra,
        fit_spectra=fit_spectra,
        nominal_spectrum=nom.astype(np.float32),
        colors=_make_gradient(channel, n),
    )


# ── ROI selection ────────────────────────────────────────────────────

def select_rois(
    image: np.ndarray,
    preset_name: str = 'fuji',
    n_sample: int = N_ROI_SAMPLES,
    min_size: int = 20,
) -> list[tuple[str, str, tuple[int, int, int, int], np.ndarray]]:
    """Select 3 ROI rectangles from the image.

    Args:
        image: (H, W, 3) uint8
        preset_name: ROI preset name (or 'auto')
        n_sample: max pixels to sample per ROI
        min_size: minimum ROI side length

    Returns:
        List of (name, color, (r0,c0,r1,c1), flat_indices)
    """
    H, W = image.shape[:2]

    if preset_name in ROI_PRESETS:
        rois_def = ROI_PRESETS[preset_name]
    else:
        rois_def = _auto_select_rois(image)

    result = []
    for name, color, (fr0, fc0, fr1, fc1) in rois_def:
        r0 = max(0, int(fr0 * H))
        c0 = max(0, int(fc0 * W))
        r1 = min(H, int(fr1 * H))
        c1 = min(W, int(fc1 * W))

        # Ensure minimum size
        if r1 - r0 < min_size:
            r1 = min(H, r0 + min_size)
        if c1 - c0 < min_size:
            c1 = min(W, c0 + min_size)

        # All pixel indices in the ROI
        rr, cc = np.mgrid[r0:r1, c0:c1]
        all_flat = (rr.ravel() * W + cc.ravel())

        # Subsample if needed
        if len(all_flat) > n_sample:
            rng = np.random.default_rng(hash(name) & 0x7FFFFFFF)
            idx = rng.choice(len(all_flat), n_sample, replace=False)
            flat = all_flat[idx]
        else:
            flat = all_flat

        result.append((name, color, (r0, c0, r1, c1), flat))

    return result


def _auto_select_rois(image: np.ndarray) -> list:
    """Auto-detect 3 ROIs by channel dominance for non-preset images."""
    H, W = image.shape[:2]
    block = max(min(H, W) // 5, 20)
    step = max(block // 2, 1)

    R = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    B = image[:, :, 2].astype(np.float32)
    total = R + G + B + 1e-6

    best = {}
    for ch_name, ch_frac, color in [
        ('sky',   B / total, '#1E88E5'),
        ('trees', G / total, '#43A047'),
        ('roof',  R / total, '#E53935'),
    ]:
        best_score = -np.inf
        best_rect = (0, 0, block / H, block / W)
        for r in range(0, H - block, step):
            for c in range(0, W - block, step):
                score = float(ch_frac[r:r + block, c:c + block].mean())
                if score > best_score:
                    best_score = score
                    best_rect = (r / H, c / W, (r + block) / H, (c + block) / W)
        best[ch_name] = (ch_name, color, best_rect)

    return [best['sky'], best['trees'], best['roof']]


# ── ROI data building ────────────────────────────────────────────────

def _build_roi_data(
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
) -> ROIData:
    """Build ROI data: extract pixels, generate spectra, fit."""
    from ..dictionary_solver import solve_dict2d_parabola

    H, W = image.shape[:2]

    # Pixel colors (actual RGB, float [0,1])
    rows = flat_indices // W
    cols = flat_indices % W
    pixel_colors = image[rows, cols].astype(np.float32) / 255.0

    # GT params for these pixels
    roi_amp = gt_amp[flat_indices]
    roi_dE = gt_dE[flat_indices]
    roi_dsigma = gt_dsigma[flat_indices]

    # Generate GT spectra
    gt_spectra = _generate_exact_voigt_spectra(
        energy, center, sigma_nom, gamma,
        roi_amp, roi_dE, roi_dsigma,
    )

    # Fit through solver
    fit_amps, _, fit_dE, fit_dsigma, _ = solve_dict2d_parabola(
        gt_spectra, dict_cache,
    )
    fit_amp_arr = fit_amps[0, :] / AMP_SCALE

    # Reconstruct fit spectra
    fit_spectra = _generate_exact_voigt_spectra(
        energy, center, sigma_nom, gamma,
        fit_amp_arr, fit_dE, fit_dsigma,
    )

    # Mean spectra
    mean_gt = gt_spectra.mean(axis=0)
    mean_fit = fit_spectra.mean(axis=0)

    # Nominal
    med_amp = float(np.median(roi_amp))
    nom = med_amp * AMP_SCALE * voigt_profile(energy, center, sigma_nom, gamma)
    nom = nom + med_amp * AMP_SCALE * BG_FRACTION

    return ROIData(
        name=name, color=color, rect=rect,
        n_pixels=len(flat_indices),
        pixel_colors=pixel_colors,
        gt_spectra=gt_spectra,
        mean_gt_spectrum=mean_gt.astype(np.float32),
        mean_fit_spectrum=mean_fit.astype(np.float32),
        nominal_spectrum=nom.astype(np.float32),
    )


# ── Main data pipeline ───────────────────────────────────────────────

def build_tracking_data(
    image_path: str,
    preset: SinglePeakPreset = C1S_SINGLE_PRESET,
    solver: str = 'dict2d_parabola',
    max_height: int = 540,
    roi_preset: str = 'fuji',
    n_roi_samples: int = N_ROI_SAMPLES,
    verbose: bool = True,
) -> TrackingDataV4:
    """Build all data for the v4 GVRT tracking figure.

    Full pipeline:
    1. Load + resize image
    2. Encode full image → (amp, dE, dsigma)
    3. Generate all spectra + solver fit → global PSNR
    4. Build dictionary (shared for sweeps + ROIs)
    5. Generate 3 synthetic sweeps (Panel C)
    6. Build 3 ROI datasets (Panel D)
    """
    from ..dictionary_solver import build_dictionary, solve_dict2d_parabola
    from ..image_utils import load_image

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

    # 3. Full-image spectra + fit → global PSNR
    if verbose:
        print("Generating spectra (full image)...")
    t0 = time.perf_counter()
    spectra = _generate_exact_voigt_spectra(
        energy, center, sigma_nom, gamma, gt_amp, gt_dE, gt_dsigma,
    )

    # 4. Build dictionary (shared)
    dE_margin = 0.2
    ds_margin = 0.02
    dE_lo = preset.shift_range.min_val - dE_margin
    dE_hi = preset.shift_range.max_val + dE_margin
    ds_lo = float(gt_dsigma.min()) - ds_margin
    ds_hi = float(gt_dsigma.max()) + ds_margin

    # Extend for sweep ranges
    dE_lo = min(dE_lo, -0.9)
    dE_hi = max(dE_hi, 0.9)
    ds_lo = min(ds_lo, -0.16)
    ds_hi = max(ds_hi, 0.16)

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
        print("Fitting (full image)...")

    t_fit0 = time.perf_counter()
    fit_amps, _, fit_dE, fit_dsigma, _ = solve_dict2d_parabola(
        spectra, dict_cache,
    )
    fit_time = time.perf_counter() - t_fit0
    throughput = N / fit_time / 1e6
    fit_amp_arr = fit_amps[0, :] / AMP_SCALE

    amp_psnr = _param_psnr(gt_amp, fit_amp_arr)
    dE_psnr = _param_psnr(gt_dE, fit_dE)
    dsigma_psnr = _param_psnr(gt_dsigma, fit_dsigma)

    if verbose:
        print(f"  Fit: {fit_time:.2f}s ({throughput:.1f}M/s)")
        print(f"  PSNR: amp={amp_psnr:.1f}dB  "
              f"\u03b4E={dE_psnr:.1f}dB  "
              f"\u03b4\u03c3={dsigma_psnr:.1f}dB")

    # 5. Synthetic sweeps (Panel C)
    if verbose:
        print("Building synthetic sweeps...")
    sweeps = []
    for ch in ['R', 'G', 'B']:
        sweeps.append(_build_sweep_data(
            energy, center, sigma_nom, gamma, dict_cache, ch,
        ))

    # 6. ROI datasets (Panel D)
    if verbose:
        print("Building ROI bands...")
    roi_defs = select_rois(image, preset_name=roi_preset,
                           n_sample=n_roi_samples)
    rois = []
    for name, color, rect, flat_idx in roi_defs:
        if verbose:
            print(f"  ROI '{name}': {len(flat_idx)} px")
        rois.append(_build_roi_data(
            image, energy, center, sigma_nom, gamma,
            gt_amp, gt_dE, gt_dsigma, dict_cache,
            name, color, rect, flat_idx,
        ))

    total_time = time.perf_counter() - t0
    if verbose:
        print(f"Total: {total_time:.1f}s")

    return TrackingDataV4(
        preset=preset, image=image, energy=energy,
        nominal_center=center, nominal_sigma=sigma_nom,
        nominal_gamma=gamma,
        sweeps=sweeps, rois=rois,
        amp_psnr=amp_psnr, dE_psnr=dE_psnr,
        dsigma_psnr=dsigma_psnr,
        solver_name='Dict2D+Parabola', throughput=throughput,
    )


# Backward-compatible alias
build_tracking_data_v4 = build_tracking_data


# ── Panel drawing ────────────────────────────────────────────────────

def _draw_panel_a(ax: plt.Axes, data: TrackingDataV4) -> None:
    """Panel A: Source image with ROI rectangles."""
    ax.imshow(data.image)
    for roi in data.rois:
        r0, c0, r1, c1 = roi.rect
        rect = mpatches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0,
            linewidth=2, edgecolor=roi.color, facecolor='none',
            linestyle='-', zorder=5,
        )
        ax.add_patch(rect)
        ax.text(
            c0 + 3, r0 - 4, roi.name,
            fontsize=8, fontweight='bold', color=roi.color,
            path_effects=[
                matplotlib.patheffects.withStroke(
                    linewidth=2.5, foreground='white'),
            ],
            zorder=6,
        )
    H, W = data.image.shape[:2]
    ax.set_title(f'(A) Source Image ({W}\u00d7{H})', fontsize=10, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_panel_b(ax: plt.Axes, data: TrackingDataV4) -> None:
    """Panel B: Encoding explanation text."""
    p = data.preset
    sep = "\u2500" * 24
    text = "\n".join([
        "RGB \u2192 Parameter Encoding",
        sep,
        f"R \u2192 amplitude [{p.amplitude_range.min_val:.1f}, "
        f"{p.amplitude_range.max_val:.1f}]",
        f"G \u2192 \u03b4E shift  [{p.shift_range.min_val:+.1f}, "
        f"{p.shift_range.max_val:+.1f}] eV",
        f"B \u2192 FWHM_G    [{p.fwhm_range.min_val:.1f}, "
        f"{p.fwhm_range.max_val:.1f}] eV",
        "",
        f"Center: {data.nominal_center:.1f} eV (C 1s)",
        f"\u03c3={data.nominal_sigma:.4f}  "
        f"\u03b3={data.nominal_gamma:.3f} eV",
        f"Solver: {data.solver_name}",
        f"{data.throughput:.1f} M spectra/s",
    ])
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes, va='top', ha='left',
        fontsize=8.5, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                  edgecolor='#CCCCCC', alpha=0.9),
    )
    ax.axis('off')
    ax.set_title('(B) Encoding Scheme', fontsize=10, pad=4)


def _draw_sweep_subpanel(
    ax: plt.Axes,
    sweep: SweepData,
    energy: np.ndarray,
    show_legend: bool = False,
) -> None:
    """Draw one synthetic parameter sweep: gradient GT + fit + nominal."""
    n = len(sweep.colors)

    # Nominal reference (gray dashed)
    ax.plot(energy, sweep.nominal_spectrum, color='#888888',
            linestyle='--', linewidth=1.2, label='Nominal', zorder=1)

    # GT spectra (colored gradient: dark → bright)
    for i in range(n):
        lbl = 'GT' if i == n - 1 else None
        ax.plot(energy, sweep.gt_spectra[i], color=sweep.colors[i],
                linewidth=1.5, label=lbl, zorder=2)

    # Fit spectra (black dotted)
    for i in range(n):
        lbl = 'Fit' if i == n - 1 else None
        ax.plot(energy, sweep.fit_spectra[i], color='black',
                linestyle=':', linewidth=0.8, alpha=0.7,
                label=lbl, zorder=3)

    ax.invert_xaxis()
    ax.set_title(sweep.label, fontsize=9, fontweight='bold', pad=3)
    ax.set_xlabel('BE (eV)', fontsize=7)
    ax.tick_params(labelsize=6)

    if show_legend:
        ax.legend(loc='upper left', fontsize=6.5, framealpha=0.85,
                  handlelength=1.5, borderpad=0.3)


def _draw_panel_c(
    axes: list[plt.Axes],
    data: TrackingDataV4,
) -> None:
    """Panel C: Synthetic parameter sweeps (R, G, B)."""
    for i, ax in enumerate(axes):
        _draw_sweep_subpanel(
            ax, data.sweeps[i], data.energy,
            show_legend=(i == 0),
        )
    axes[0].set_ylabel('Intensity (a.u.)', fontsize=7)
    axes[0].text(
        -0.02, 1.18,
        '(C) Synthetic Demo \u2014 Isolated Channel Effects',
        transform=axes[0].transAxes, fontsize=9, fontweight='bold',
        va='bottom', ha='left',
    )


def _draw_roi_subpanel(
    ax: plt.Axes,
    roi: ROIData,
    energy: np.ndarray,
    show_legend: bool = False,
) -> None:
    """Draw one ROI spectral band: alpha=0.05 × N spectra."""
    # Semi-transparent GT spectra colored by pixel RGB
    for i in range(roi.n_pixels):
        rgba = (*roi.pixel_colors[i], 0.05)
        ax.plot(energy, roi.gt_spectra[i], color=rgba,
                linewidth=0.4, zorder=1)

    # Nominal reference
    ax.plot(energy, roi.nominal_spectrum, color='#888888',
            linestyle='--', linewidth=1.2, label='Nominal', zorder=2)

    # Mean GT (thick, ROI color)
    ax.plot(energy, roi.mean_gt_spectrum, color=roi.color,
            linewidth=2.0, label='Mean GT', zorder=3)

    # Mean fit (black dashed)
    ax.plot(energy, roi.mean_fit_spectrum, color='black',
            linestyle='--', linewidth=1.5, label='Mean Fit', zorder=4)

    ax.invert_xaxis()
    ax.set_title(f'{roi.name} ({roi.n_pixels} px)', fontsize=9,
                 fontweight='bold', color=roi.color, pad=3)
    ax.set_xlabel('BE (eV)', fontsize=7)
    ax.tick_params(labelsize=6)

    if show_legend:
        ax.legend(loc='upper left', fontsize=6.5, framealpha=0.85,
                  handlelength=1.5, borderpad=0.3)


def _draw_panel_d(
    axes: list[plt.Axes],
    data: TrackingDataV4,
) -> None:
    """Panel D: Real image ROI spectral bands."""
    for i, ax in enumerate(axes):
        _draw_roi_subpanel(
            ax, data.rois[i], data.energy,
            show_legend=(i == 0),
        )
    axes[0].set_ylabel('Intensity (a.u.)', fontsize=7)
    axes[0].text(
        -0.02, 1.18,
        '(D) Real Image ROI \u2014 Spectral Bands',
        transform=axes[0].transAxes, fontsize=9, fontweight='bold',
        va='bottom', ha='left',
    )


def _draw_panel_e(ax: plt.Axes, data: TrackingDataV4) -> None:
    """Panel E: Compact accuracy summary."""
    ax.axis('off')
    H, W = data.image.shape[:2]
    text = (
        f"PSNR: amp={data.amp_psnr:.1f} dB   "
        f"\u03b4E={data.dE_psnr:.1f} dB   "
        f"\u03b4\u03c3={data.dsigma_psnr:.1f} dB   \u2502   "
        f"{data.solver_name}   {data.throughput:.1f} M/s   \u2502   "
        f"{W}\u00d7{H} ({W * H:,} spectra)   \u2502   Noise-Free"
    )
    ax.text(
        0.5, 0.6, text,
        transform=ax.transAxes, va='center', ha='center',
        fontsize=9, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                  edgecolor='#CCCCCC', alpha=0.95),
    )
    ax.set_title('(E) Accuracy Summary', fontsize=10, pad=2)


# ── Main figure assembly ─────────────────────────────────────────────

def plot_gvrt_tracking(
    data: TrackingDataV4,
    figsize: tuple[float, float] = (18, 14),
    output_path: str | None = None,
    show: bool = False,
    dpi: int = 200,
) -> plt.Figure:
    """Generate the 5-panel GVRT Parameter Tracking figure (v4).

    Args:
        data: TrackingDataV4 from build_tracking_data()
        figsize: Figure size in inches
        output_path: Save path (PNG/PDF)
        show: Show interactively
        dpi: DPI for saved figure

    Returns:
        matplotlib Figure
    """
    import matplotlib.patheffects  # noqa: F811

    apply_style()
    fig = plt.figure(figsize=figsize, facecolor='white')

    outer = GridSpec(
        4, 1, figure=fig,
        height_ratios=[2.2, 2.5, 2.5, 0.5],
        hspace=0.35,
        left=0.05, right=0.95, top=0.96, bottom=0.03,
    )

    # Row 0: Panel A (image) + Panel B (text)
    row0 = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[1.5, 1.0], wspace=0.08,
    )
    ax_a = fig.add_subplot(row0[0])
    ax_b = fig.add_subplot(row0[1])

    # Row 1: Panel C (3 sweep subpanels)
    row1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.25)
    ax_c = [fig.add_subplot(row1[i]) for i in range(3)]

    # Row 2: Panel D (3 ROI subpanels)
    row2 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.25)
    ax_d = [fig.add_subplot(row2[i]) for i in range(3)]

    # Row 3: Panel E (compact accuracy)
    ax_e = fig.add_subplot(outer[3])

    # Draw all panels
    _draw_panel_a(ax_a, data)
    _draw_panel_b(ax_b, data)
    _draw_panel_c(ax_c, data)
    _draw_panel_d(ax_d, data)
    _draw_panel_e(ax_e, data)

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


# Backward-compatible alias
plot_gvrt_tracking_v4 = plot_gvrt_tracking
