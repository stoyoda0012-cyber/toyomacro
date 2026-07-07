"""Bias-Variance scatter plot for solver comparison.

Plots |bias| vs std for each solver, with noise-level trajectory lines.
RMSE = sqrt(bias² + std²) iso-contours in background.

Uses log-log scale to accommodate the large dynamic range
(e.g. Dict2D NF |bias|≈0.001 vs 4-step |bias|≈0.45).
"""


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from ..evaluation.error_stats import ErrorStats
from .common import SOLVER_LABELS, get_noise_label


def plot_bias_variance_scatter(
    ax: plt.Axes,
    stats: dict[str, dict[str, ErrorStats]],
    solver_colors: dict[str, str],
    solver_markers: dict[str, str],
    show_rmse_contours: bool = True,
    noise_order: list[str] | None = None,
) -> None:
    """Draw Bias-Variance scatter with noise-level trajectories (log-log).

    Args:
        ax: Matplotlib axes to draw on
        stats: stats[solver_name][noise_level] = ErrorStats
        solver_colors: Color per solver
        solver_markers: Marker per solver
        show_rmse_contours: Draw RMSE iso-lines
        noise_order: Ordered noise levels for trajectory drawing
    """
    # ── Collect data range ──────────────────────────────────────────
    all_bias = []
    all_std = []
    for sn_dict in stats.values():
        for es in sn_dict.values():
            b = abs(es.bias)
            s = es.std
            if b > 0:
                all_bias.append(b)
            if s > 0:
                all_std.append(s)

    if not all_bias or not all_std:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center')
        return

    min_val = min(min(all_bias), min(all_std)) * 0.5
    max_val = max(max(all_bias), max(all_std)) * 2.0

    # ── RMSE iso-contours (quarter-circles in log space) ────────────
    if show_rmse_contours:
        theta = np.linspace(0, np.pi / 2, 200)

        # Log-spaced RMSE levels
        log_min = np.floor(np.log10(min_val))
        log_max = np.ceil(np.log10(max_val))
        decades = np.arange(log_min, log_max + 1)
        levels = []
        for d in decades:
            levels.extend([10**d, 3 * 10**d])
        levels = [l for l in levels if min_val * 0.5 < l < max_val * 2]

        for rmse_val in levels:
            x = rmse_val * np.cos(theta)
            y = rmse_val * np.sin(theta)
            # Only plot where both x, y > min_val
            mask = (x > min_val * 0.3) & (y > min_val * 0.3)
            if np.sum(mask) > 2:
                ax.plot(x[mask], y[mask], '-', color='#E5E7EB',
                        linewidth=0.5, zorder=0)
                # Label at 45°
                lx = rmse_val * 0.707
                ly = rmse_val * 0.707
                if min_val < lx < max_val and min_val < ly < max_val:
                    if rmse_val >= 0.1:
                        txt = f'{rmse_val:.1f}'
                    elif rmse_val >= 0.01:
                        txt = f'{rmse_val:.2f}'
                    else:
                        txt = f'{rmse_val:.0e}'
                    ax.text(lx, ly, txt, fontsize=5, color='#B0B0B0',
                            rotation=-45, ha='center', va='center', zorder=0)

    # ── Diagonal reference line (bias = std) ────────────────────────
    diag = np.array([min_val * 0.5, max_val * 2])
    ax.plot(diag, diag, '--', color='#D1D5DB', linewidth=0.7, zorder=0)

    # ── Noise order ─────────────────────────────────────────────────
    noise_levels_available = set()
    for sn_dict in stats.values():
        noise_levels_available.update(sn_dict.keys())

    if noise_order is None:
        noise_order = sorted(noise_levels_available)

    n_levels = len(noise_order)
    sizes = np.linspace(90, 35, max(n_levels, 1))

    # ── Annotation offsets per solver (cycle to reduce overlap) ─────
    _OFFSETS = [
        (6, 4), (-6, 6), (6, -8), (-8, -6), (8, 0),
    ]

    # ── Solver trajectories ─────────────────────────────────────────
    for si, (sn, sn_dict) in enumerate(stats.items()):
        color = solver_colors.get(sn, '#333333')
        marker = solver_markers.get(sn, 'o')
        label = SOLVER_LABELS.get(sn, sn)
        offset = _OFFSETS[si % len(_OFFSETS)]

        biases = []
        stds = []
        nl_labels = []
        for nl in noise_order:
            if nl in sn_dict:
                b = abs(sn_dict[nl].bias)
                s = sn_dict[nl].std
                # Floor at tiny value for log scale
                biases.append(max(b, min_val * 0.1))
                stds.append(max(s, min_val * 0.1))
                nl_labels.append(nl)

        if not biases:
            continue

        biases = np.array(biases)
        stds = np.array(stds)
        n_pts = len(biases)

        # Trajectory line
        if n_pts > 1:
            ax.plot(biases, stds, '-', color=color, linewidth=1.5,
                    alpha=0.4, zorder=1)

        # Points + selective annotation (first & last only)
        for k, (b, s, nl) in enumerate(zip(biases, stds, nl_labels)):
            idx = noise_order.index(nl) if nl in noise_order else k
            sz = sizes[min(idx, len(sizes) - 1)]
            ax.scatter(
                [b], [s], c=color, marker=marker, s=sz,
                edgecolors='white', linewidth=0.8, zorder=3,
                label=label if k == 0 else None,
            )

            # Annotate only endpoints (first and last noise level)
            if k == 0 or k == n_pts - 1:
                ax.annotate(
                    get_noise_label(nl), (b, s),
                    textcoords='offset points',
                    xytext=offset,
                    fontsize=5, color=color, alpha=0.8,
                    fontweight='bold',
                )

    # ── Axis formatting ─────────────────────────────────────────────
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.set_xlabel('|Bias| (systematic)', fontsize=8, labelpad=2)
    ax.set_ylabel('Std (random)', fontsize=8, labelpad=2)
    ax.set_title('Bias–Variance\nDecomposition', fontsize=9,
                 fontweight='bold', pad=4)

    # Cleaner tick labels
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(mticker.FuncFormatter(
            lambda val, pos: f'{val:.0e}' if val < 0.01
            else f'{val:.2f}' if val < 1
            else f'{val:.1f}'
        ))
        axis.set_minor_formatter(mticker.NullFormatter())

    ax.legend(loc='upper left', fontsize=6.5, framealpha=0.85,
              borderpad=0.4, handletextpad=0.4)
    ax.grid(True, which='major', alpha=0.15, linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.07, linewidth=0.3)
