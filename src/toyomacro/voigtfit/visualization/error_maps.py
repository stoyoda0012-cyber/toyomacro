"""Error Map composite figure: spatial visualization of solver estimation errors.

Main entry point: plot_error_maps() generates a multi-panel figure with:
  - Panel A: 5×3 grid of error maps (solvers × noise levels)
  - Panel B: Bias-Variance scatter (side panel)
  - Panel C: Error histograms (bottom strip)
  - Panel D: Stats table (bottom right)
"""

import matplotlib
import numpy as np

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from ..evaluation.benchmark_result import BenchmarkResult
from ..evaluation.error_stats import ErrorStats, compute_error_stats
from .bias_variance import plot_bias_variance_scatter
from .common import (
    SOLVER_COLORS,
    SOLVER_LABELS,
    SOLVER_MARKERS,
    apply_style,
    get_noise_label,
)
from .error_histograms import plot_error_histograms

# ── Param display names ─────────────────────────────────────────────

PARAM_LABELS = {
    'dsigma': 'δσ',
    'dE': 'δE (eV)',
    'amp': 'Amplitude',
}

PARAM_UNITS = {
    'dsigma': '',
    'dE': 'eV',
    'amp': '',
}


def _get_param_arrays(
    br: BenchmarkResult,
    param: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (estimated, true) arrays for the specified parameter."""
    if param == 'dsigma':
        return br.dsigma_est, br.dsigma_true
    elif param == 'dE':
        return br.dE_est, br.dE_true
    elif param == 'amp':
        return br.amp_est, br.amp_true
    else:
        raise ValueError(f"Unknown param: {param!r}. Use 'dsigma', 'dE', or 'amp'.")


def _render_error_map(
    ax: plt.Axes,
    errors: np.ndarray,
    image_shape: tuple[int, int],
    clim: tuple[float, float],
    cmap: str = 'RdBu_r',
    show_badge: bool = True,
    badge_text: str | None = None,
) -> matplotlib.image.AxesImage:
    """Render a single error map on the given axes.

    Args:
        ax: Matplotlib axes to draw on
        errors: (n_pixels,) signed error values
        image_shape: (H, W) for reshaping
        clim: (vmin, vmax) symmetric color limits
        cmap: Diverging colormap name
        show_badge: Show RMSE badge in corner
        badge_text: Custom badge text (default: auto RMSE)

    Returns:
        AxesImage object
    """
    H, W = image_shape
    error_map = errors.reshape(H, W)

    # Use TwoSlopeNorm for symmetric diverging colormap centered at 0
    norm = TwoSlopeNorm(vmin=clim[0], vcenter=0.0, vmax=clim[1])
    im = ax.imshow(error_map, cmap=cmap, norm=norm, aspect='equal',
                   interpolation='nearest')

    ax.set_xticks([])
    ax.set_yticks([])

    if show_badge:
        if badge_text is None:
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            badge_text = f'RMSE={rmse:.4f}'
        ax.text(
            0.97, 0.03, badge_text,
            transform=ax.transAxes,
            fontsize=6, fontweight='bold',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray',
                      alpha=0.85, lw=0.5),
        )

    return im


def _compute_clim_per_row(
    all_errors: dict[str, dict[str, np.ndarray]],
    noise_levels: list[str],
    solver_names: list[str],
    percentile: float = 99.0,
) -> dict[str, tuple[float, float]]:
    """Compute symmetric color limits per noise-level row.

    Uses the given percentile of absolute errors across all solvers
    within each row for a fair comparison.
    """
    clims = {}
    for nl in noise_levels:
        max_val = 0.0
        for sn in solver_names:
            if nl in all_errors and sn in all_errors[nl]:
                errors = all_errors[nl][sn]
                p = float(np.percentile(np.abs(errors), percentile))
                max_val = max(max_val, p)
        if max_val < 1e-10:
            max_val = 0.01
        clims[nl] = (-max_val, max_val)
    return clims


def _compute_clim_shared(
    all_errors: dict[str, dict[str, np.ndarray]],
    percentile: float = 99.0,
) -> tuple[float, float]:
    """Compute a single symmetric color limit across all panels."""
    max_val = 0.0
    for nl_dict in all_errors.values():
        for errors in nl_dict.values():
            p = float(np.percentile(np.abs(errors), percentile))
            max_val = max(max_val, p)
    if max_val < 1e-10:
        max_val = 0.01
    return (-max_val, max_val)


def _draw_stats_table(
    ax: plt.Axes,
    stats: dict[str, dict[str, ErrorStats]],
    noise_levels: list[str],
    solver_names: list[str],
) -> None:
    """Draw a compact stats table on the given axes."""
    ax.axis('off')

    # Build table data
    col_labels = ['Solver', 'Noise', 'bias', 'std', 'RMSE', 'MAE', 'P95']
    rows = []
    for nl in noise_levels:
        for sn in solver_names:
            if nl not in stats or sn not in stats[nl]:
                continue
            s = stats[nl][sn]
            rows.append([
                SOLVER_LABELS.get(sn, sn),
                get_noise_label(nl),
                f'{s.bias:+.4f}',
                f'{s.std:.4f}',
                f'{s.rmse:.4f}',
                f'{s.mae:.4f}',
                f'{s.p95:.4f}',
            ])

    if not rows:
        return

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.2)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#E5E7EB')
        cell.set_text_props(fontweight='bold')

    # Color-code solver column
    for i, row in enumerate(rows, start=1):
        sn_label = row[0]
        for sn_key, label in SOLVER_LABELS.items():
            if label == sn_label:
                color = SOLVER_COLORS.get(sn_key, '#333333')
                table[i, 0].set_text_props(color=color, fontweight='bold')
                break


def plot_error_maps(
    results: dict[str, dict[str, BenchmarkResult]],
    param: str = 'dsigma',
    clim_mode: str = 'per_row',
    clim_range: float | None = None,
    cmap: str = 'RdBu_r',
    figsize: tuple[float, float] = (22, 14),
    output_path: str | None = None,
    solver_order: list[str] | None = None,
    noise_order: list[str] | None = None,
    hist_style: str = 'violin',
    show: bool = False,
) -> plt.Figure:
    """Generate the full Error Map composite figure.

    Args:
        results: Nested dict results[noise_level][solver_name] = BenchmarkResult
        param: Parameter to visualize ('dsigma', 'dE', or 'amp')
        clim_mode: Color scale mode ('shared', 'per_row', 'per_panel')
        clim_range: Manual symmetric color range (overrides auto)
        cmap: Diverging colormap name
        figsize: Figure size in inches
        output_path: Save path (PNG/PDF). None = don't save.
        solver_order: Custom solver column order
        noise_order: Custom noise row order
        hist_style: 'violin', 'histogram', or 'box'
        show: Show figure interactively

    Returns:
        matplotlib Figure
    """
    apply_style()

    # Determine order
    if noise_order is None:
        noise_order = list(results.keys())
    if solver_order is None:
        # Collect all solver names across noise levels
        all_solvers = set()
        for nl_dict in results.values():
            all_solvers.update(nl_dict.keys())
        # Default order
        default_order = ['4step', '6step', 'dict1d', 'dict2d', 'adaptive']
        solver_order = [s for s in default_order if s in all_solvers]
        # Append any extras
        for s in sorted(all_solvers):
            if s not in solver_order:
                solver_order.append(s)

    n_noise = len(noise_order)
    n_solvers = len(solver_order)

    # Get image_shape from first result
    image_shape = None
    for nl_dict in results.values():
        for br in nl_dict.values():
            image_shape = br.image_shape
            break
        if image_shape is not None:
            break

    # ── Compute all errors and stats ────────────────────────────────
    all_errors = {}  # [noise][solver] -> np.ndarray
    all_stats = {}   # [noise][solver] -> ErrorStats

    for nl in noise_order:
        all_errors[nl] = {}
        all_stats[nl] = {}
        if nl not in results:
            continue
        for sn in solver_order:
            if sn not in results[nl]:
                continue
            br = results[nl][sn]
            est, true = _get_param_arrays(br, param)
            stats = compute_error_stats(est, true)
            all_errors[nl][sn] = stats.errors
            all_stats[nl][sn] = stats

    # ── Compute color limits ────────────────────────────────────────
    if clim_range is not None:
        clim_per_row = {nl: (-clim_range, clim_range) for nl in noise_order}
        clim_shared = (-clim_range, clim_range)
    elif clim_mode == 'shared':
        clim_shared = _compute_clim_shared(all_errors)
        clim_per_row = {nl: clim_shared for nl in noise_order}
    elif clim_mode == 'per_row':
        clim_per_row = _compute_clim_per_row(
            all_errors, noise_order, solver_order)
    else:  # per_panel — handled individually
        clim_per_row = None

    # ── Figure layout ───────────────────────────────────────────────
    # Main grid: error maps (left) + side panels (right)
    # Bottom: histograms (left) + stats table (right)
    fig = plt.figure(figsize=figsize)

    # Outer grid: 2 rows (maps+side | bottom)
    # hspace=0.18 to avoid B-V xlabel overlapping with bottom panels
    outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[3.5, 1],
        hspace=0.18,
    )

    # Top section: error maps (left) + right panel (right)
    # wspace=0.15 to avoid overlap between colorbar and B-V axis labels
    top = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[4, 1.3],
        wspace=0.15,
    )

    # Error map grid (n_noise rows × n_solvers cols + colorbar column)
    map_grid = GridSpecFromSubplotSpec(
        n_noise, n_solvers + 1, subplot_spec=top[0],
        width_ratios=[1] * n_solvers + [0.05],
        wspace=0.04, hspace=0.08,
    )

    # Right panel: bias-var scatter (top 85%) + padding (bottom 15%)
    # The padding prevents xlabel from overlapping the bottom panels
    right_panel = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=top[1],
        height_ratios=[5, 1],
        hspace=0.05,
    )
    ax_bv = fig.add_subplot(right_panel[0])
    # bottom padding cell — invisible
    ax_pad = fig.add_subplot(right_panel[1])
    ax_pad.axis('off')

    # Bottom section: histograms (left) + stats table (right)
    bottom = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1],
        width_ratios=[3, 1.5],
        wspace=0.12,
    )
    ax_hist = fig.add_subplot(bottom[0])
    ax_table = fig.add_subplot(bottom[1])

    # ── Panel A: Error Maps ─────────────────────────────────────────
    last_im_per_row = {}  # for colorbars

    for i, nl in enumerate(noise_order):
        for j, sn in enumerate(solver_order):
            ax = fig.add_subplot(map_grid[i, j])

            if nl in all_errors and sn in all_errors[nl]:
                errors = all_errors[nl][sn]
                stats = all_stats[nl][sn]

                if clim_mode == 'per_panel':
                    p99 = float(np.percentile(np.abs(errors), 99))
                    clim = (-max(p99, 1e-6), max(p99, 1e-6))
                else:
                    clim = clim_per_row[nl]

                im = _render_error_map(
                    ax, errors, image_shape, clim, cmap=cmap,
                    badge_text=f'{stats.rmse:.4f}',
                )
                last_im_per_row[i] = (im, clim)
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])

            # Column headers (top row)
            if i == 0:
                label = SOLVER_LABELS.get(sn, sn)
                color = SOLVER_COLORS.get(sn, '#333333')
                ax.set_title(label, fontsize=9, fontweight='bold', color=color)

            # Row labels (left column)
            if j == 0:
                ax.set_ylabel(get_noise_label(nl), fontsize=8, fontweight='bold')

    # Add colorbars (one per row)
    for i, nl in enumerate(noise_order):
        ax_cb = fig.add_subplot(map_grid[i, n_solvers])
        if i in last_im_per_row:
            im, clim = last_im_per_row[i]
            cb = plt.colorbar(im, cax=ax_cb)
            cb.ax.tick_params(labelsize=6)
            unit = PARAM_UNITS.get(param, '')
            if unit:
                cb.set_label(f'Error ({unit})', fontsize=6)
        else:
            ax_cb.axis('off')

    # ── Panel B: Bias-Variance Scatter ──────────────────────────────
    # Rearrange stats: stats[solver][noise] for bias-var plot
    stats_by_solver = {}
    for nl in noise_order:
        for sn in solver_order:
            if nl in all_stats and sn in all_stats[nl]:
                if sn not in stats_by_solver:
                    stats_by_solver[sn] = {}
                stats_by_solver[sn][nl] = all_stats[nl][sn]

    plot_bias_variance_scatter(
        ax_bv, stats_by_solver, SOLVER_COLORS, SOLVER_MARKERS,
        show_rmse_contours=True,
        noise_order=noise_order,
    )

    # ── Panel C: Error Histograms ───────────────────────────────────
    # Rearrange for histogram function
    stats_for_hist = {}
    for sn in solver_order:
        stats_for_hist[sn] = {}
        for nl in noise_order:
            if nl in all_stats and sn in all_stats[nl]:
                stats_for_hist[sn][nl] = all_stats[nl][sn]

    plot_error_histograms(
        ax_hist, stats_for_hist, SOLVER_COLORS,
        style=hist_style, noise_order=noise_order,
    )

    # ── Panel D: Stats Table ────────────────────────────────────────
    _draw_stats_table(ax_table, all_stats, noise_order, solver_order)

    # ── Title ───────────────────────────────────────────────────────
    param_label = PARAM_LABELS.get(param, param)
    fig.suptitle(
        f'Error Maps: {param_label} estimation error (est − true)',
        fontsize=13, fontweight='bold', y=0.98,
    )

    # ── Save / Show ─────────────────────────────────────────────────
    if output_path is not None:
        fig.savefig(str(output_path), dpi=200, bbox_inches='tight',
                    facecolor='white')
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


# ======================================================================
# RGB Error Composite: R=|amp_err|, G=|δE_err|, B=|δσ_err|
# ======================================================================

def plot_rgb_error_maps(
    results: dict[str, dict[str, BenchmarkResult]],
    scale_mode: str = 'per_row',
    figsize: tuple[float, float] = (20, 12),
    output_path: str | None = None,
    solver_order: list[str] | None = None,
    noise_order: list[str] | None = None,
    gamma: float = 0.5,
    show: bool = False,
) -> plt.Figure:
    """RGB error composite: each channel encodes a different parameter error.

    R = |amplitude error| / scale
    G = |δE error| / scale
    B = |δσ error| / scale
    Black = accurate, color = error type.

    Color reading:
      Green  = position error dominant (δE wrong)
      Blue   = width error dominant (δσ wrong)
      Cyan   = both δE and δσ wrong
      Red    = amplitude error dominant
      White  = all three wrong

    Args:
        results: results[noise_level][solver_name] = BenchmarkResult
        scale_mode: 'per_row' (normalize per noise level) or 'shared'
        figsize: Figure size
        output_path: Save path
        solver_order: Column order
        noise_order: Row order
        gamma: Gamma correction (< 1 brightens dark regions). Default 0.5.
        show: Show interactively

    Returns:
        matplotlib Figure
    """
    apply_style()

    # Determine order
    if noise_order is None:
        noise_order = list(results.keys())
    if solver_order is None:
        all_solvers = set()
        for nl_dict in results.values():
            all_solvers.update(nl_dict.keys())
        default_order = ['4step', '6step', 'dict1d', 'dict2d', 'adaptive']
        solver_order = [s for s in default_order if s in all_solvers]
        for s in sorted(all_solvers):
            if s not in solver_order:
                solver_order.append(s)

    n_noise = len(noise_order)
    n_solvers = len(solver_order)

    # Get image_shape
    image_shape = None
    for nl_dict in results.values():
        for br in nl_dict.values():
            image_shape = br.image_shape
            break
        if image_shape is not None:
            break

    # ── Compute per-pixel absolute errors ───────────────────────────
    # errors_rgb[nl][sn] = (|amp_err|, |dE_err|, |dsigma_err|) each (n_px,)
    errors_rgb = {}
    for nl in noise_order:
        errors_rgb[nl] = {}
        if nl not in results:
            continue
        for sn in solver_order:
            if sn not in results[nl]:
                continue
            br = results[nl][sn]
            amp_err = np.abs(br.amp_est - br.amp_true).astype(np.float64)
            dE_err = np.abs(br.dE_est - br.dE_true).astype(np.float64)
            ds_err = np.abs(br.dsigma_est - br.dsigma_true).astype(np.float64)
            errors_rgb[nl][sn] = (amp_err, dE_err, ds_err)

    # ── Compute normalization scales ────────────────────────────────
    # p99 across all solvers within each row (or globally)
    PERCENTILE = 99.0
    if scale_mode == 'shared':
        # Global scale
        all_amp, all_dE, all_ds = [], [], []
        for nl_dict in errors_rgb.values():
            for (ae, de, dse) in nl_dict.values():
                all_amp.append(np.percentile(ae, PERCENTILE))
                all_dE.append(np.percentile(de, PERCENTILE))
                all_ds.append(np.percentile(dse, PERCENTILE))
        s_amp = max(max(all_amp), 1e-10)
        s_dE = max(max(all_dE), 1e-10)
        s_ds = max(max(all_ds), 1e-10)
        scales = {nl: (s_amp, s_dE, s_ds) for nl in noise_order}
    else:
        # Per-row scale
        scales = {}
        for nl in noise_order:
            amps, dEs, dss = [], [], []
            if nl in errors_rgb:
                for (ae, de, dse) in errors_rgb[nl].values():
                    amps.append(np.percentile(ae, PERCENTILE))
                    dEs.append(np.percentile(de, PERCENTILE))
                    dss.append(np.percentile(dse, PERCENTILE))
            s_amp = max(max(amps), 1e-10) if amps else 1.0
            s_dE = max(max(dEs), 1e-10) if dEs else 1.0
            s_ds = max(max(dss), 1e-10) if dss else 1.0
            scales[nl] = (s_amp, s_dE, s_ds)

    # ── Figure layout ───────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, facecolor='white')

    gs = GridSpec(
        n_noise, n_solvers, figure=fig,
        wspace=0.03, hspace=0.08,
        left=0.06, right=0.94, top=0.92, bottom=0.08,
    )

    H, W = image_shape

    for i, nl in enumerate(noise_order):
        s_amp, s_dE, s_ds = scales[nl]

        for j, sn in enumerate(solver_order):
            ax = fig.add_subplot(gs[i, j])
            ax.set_facecolor('#F0F0F0')  # light gray for empty areas

            if nl in errors_rgb and sn in errors_rgb[nl]:
                amp_err, dE_err, ds_err = errors_rgb[nl][sn]

                # Normalize to [0, 1] and clip
                r = np.clip(amp_err / s_amp, 0, 1)
                g = np.clip(dE_err / s_dE, 0, 1)
                b = np.clip(ds_err / s_ds, 0, 1)

                # Gamma correction (brighten dark pixels)
                if gamma != 1.0:
                    r = r ** gamma
                    g = g ** gamma
                    b = b ** gamma

                # Build RGB image
                rgb = np.stack([r, g, b], axis=-1).reshape(H, W, 3)
                rgb_u8 = (rgb * 255).astype(np.uint8)

                ax.imshow(rgb_u8, aspect='equal', interpolation='nearest')

                # RMSE badge
                rmse_amp = float(np.sqrt(np.mean(amp_err ** 2)))
                rmse_dE = float(np.sqrt(np.mean(dE_err ** 2)))
                rmse_ds = float(np.sqrt(np.mean(ds_err ** 2)))
                badge = f'δE={rmse_dE:.3f}  δσ={rmse_ds:.4f}'
                ax.text(
                    0.97, 0.03, badge,
                    transform=ax.transAxes,
                    fontsize=5.5, fontweight='bold', color='#333333',
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='gray', alpha=0.85, lw=0.5),
                )
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='gray')

            ax.set_xticks([])
            ax.set_yticks([])

            # Column headers
            if i == 0:
                slabel = SOLVER_LABELS.get(sn, sn)
                scolor = SOLVER_COLORS.get(sn, '#333333')
                ax.set_title(slabel, fontsize=9, fontweight='bold',
                             color=scolor, pad=4)

            # Row labels
            if j == 0:
                ax.set_ylabel(get_noise_label(nl), fontsize=8,
                              fontweight='bold', color='#333333')

    # ── Legend ───────────────────────────────────────────────────────
    fig.text(
        0.5, 0.02,
        'R = |amp error|    G = |δE error|    B = |δσ error|'
        '        Dark = accurate    Bright = large error',
        ha='center', fontsize=9, color='#555555',
        fontfamily='monospace',
    )

    fig.suptitle(
        'RGB Error Composite',
        fontsize=14, fontweight='bold', color='#222222', y=0.97,
    )

    # Channel scale info (per-row normalization values)
    for i, nl in enumerate(noise_order):
        s_amp, s_dE, s_ds = scales[nl]
        fig.text(
            0.96, 0.92 - i * (0.84 / n_noise),
            f'R:{s_amp:.1f} G:{s_dE:.3f} B:{s_ds:.4f}',
            fontsize=5, color='#888888', ha='right', va='top',
            fontfamily='monospace',
        )

    # ── Save / Show ─────────────────────────────────────────────────
    if output_path is not None:
        fig.savefig(str(output_path), dpi=200, bbox_inches='tight',
                    facecolor='white')
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig
