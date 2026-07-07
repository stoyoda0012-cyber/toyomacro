"""Error distribution histograms / violin plots for solver comparison.

Shows the distribution of (est - true) per solver, grouped by noise level.
Reveals bias (center shift) and variance (width) at a glance.
"""


import matplotlib.pyplot as plt
import numpy as np

from ..evaluation.error_stats import ErrorStats
from .common import SOLVER_LABELS, get_noise_label


def plot_error_histograms(
    ax: plt.Axes,
    stats: dict[str, dict[str, ErrorStats]],
    solver_colors: dict[str, str],
    style: str = 'violin',
    noise_order: list[str] | None = None,
    show_outliers: bool = True,
) -> None:
    """Draw error distribution plots (violin, histogram, or box).

    Args:
        ax: Matplotlib axes
        stats: stats[solver_name][noise_level] = ErrorStats
        solver_colors: Color per solver
        style: 'violin', 'histogram', or 'box'
        noise_order: Ordered noise levels
        show_outliers: Show outlier points (box style only)
    """
    # Collect all noise levels
    all_noise = set()
    for sn_dict in stats.values():
        all_noise.update(sn_dict.keys())
    if noise_order is None:
        noise_order = sorted(all_noise)

    solver_names = list(stats.keys())
    n_solvers = len(solver_names)
    n_noise = len(noise_order)

    if style == 'violin':
        _plot_violin(ax, stats, solver_colors, solver_names,
                     noise_order, n_solvers, n_noise)
    elif style == 'histogram':
        _plot_overlaid_histogram(ax, stats, solver_colors, solver_names,
                                noise_order, n_solvers, n_noise)
    elif style == 'box':
        _plot_box(ax, stats, solver_colors, solver_names,
                  noise_order, n_solvers, n_noise, show_outliers)
    else:
        raise ValueError(f"Unknown style: {style!r}")

    ax.axhline(0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)
    ax.set_ylabel('Error (est − true)', fontsize=8)
    ax.set_title('Error Distributions', fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.2, linewidth=0.5)


def _plot_violin(ax, stats, solver_colors, solver_names,
                 noise_order, n_solvers, n_noise):
    """Grouped violin plots: groups = noise levels, violins = solvers."""
    positions = []
    colors = []
    data = []
    tick_positions = []
    tick_labels = []

    group_width = n_solvers + 1  # spacing between groups
    for gi, nl in enumerate(noise_order):
        group_center = gi * group_width + n_solvers / 2
        tick_positions.append(group_center)
        tick_labels.append(get_noise_label(nl))

        for si, sn in enumerate(solver_names):
            pos = gi * group_width + si
            positions.append(pos)
            colors.append(solver_colors.get(sn, '#333333'))

            if nl in stats[sn]:
                errors = stats[sn][nl].errors
                # Subsample for performance if very large
                if len(errors) > 10000:
                    idx = np.random.RandomState(42).choice(
                        len(errors), 10000, replace=False)
                    errors = errors[idx]
                data.append(errors)
            else:
                data.append(np.array([0.0]))

    # Draw violins
    parts = ax.violinplot(
        data, positions=positions,
        showmeans=True, showmedians=False, showextrema=False,
        widths=0.8,
    )

    # Color each violin
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(0.6)
        body.set_edgecolor(colors[i])
        body.set_linewidth(0.8)

    parts['cmeans'].set_color('#333333')
    parts['cmeans'].set_linewidth(1.0)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)

    # Legend
    for sn in solver_names:
        label = SOLVER_LABELS.get(sn, sn)
        color = solver_colors.get(sn, '#333333')
        ax.plot([], [], 's', color=color, label=label, markersize=6)
    ax.legend(loc='upper right', fontsize=6, ncol=min(n_solvers, 5),
              framealpha=0.8)


def _plot_overlaid_histogram(ax, stats, solver_colors, solver_names,
                             noise_order, n_solvers, n_noise):
    """Overlaid histograms per noise level, each solver as a different color."""
    # For each noise level, plot all solvers overlaid
    # Use alpha for overlap visibility
    for nl in noise_order:
        for sn in solver_names:
            if nl not in stats[sn]:
                continue
            errors = stats[sn][nl].errors
            color = solver_colors.get(sn, '#333333')
            label = f'{SOLVER_LABELS.get(sn, sn)} ({get_noise_label(nl)})'

            # Auto-determine bins
            n_bins = min(100, max(20, len(errors) // 100))
            ax.hist(
                errors, bins=n_bins, alpha=0.3, color=color,
                density=True, label=label, histtype='stepfilled',
            )
            ax.hist(
                errors, bins=n_bins, alpha=0.8, color=color,
                density=True, histtype='step', linewidth=1.0,
            )

    ax.legend(loc='upper right', fontsize=5, ncol=2, framealpha=0.8)
    ax.set_xlabel('Error (est − true)', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)


def _plot_box(ax, stats, solver_colors, solver_names,
              noise_order, n_solvers, n_noise, show_outliers):
    """Grouped box plots."""
    positions = []
    data = []
    colors_list = []
    tick_positions = []
    tick_labels = []

    group_width = n_solvers + 1
    for gi, nl in enumerate(noise_order):
        group_center = gi * group_width + n_solvers / 2
        tick_positions.append(group_center)
        tick_labels.append(get_noise_label(nl))

        for si, sn in enumerate(solver_names):
            pos = gi * group_width + si
            positions.append(pos)
            colors_list.append(solver_colors.get(sn, '#333333'))

            if nl in stats[sn]:
                errors = stats[sn][nl].errors
                if len(errors) > 10000:
                    idx = np.random.RandomState(42).choice(
                        len(errors), 10000, replace=False)
                    errors = errors[idx]
                data.append(errors)
            else:
                data.append(np.array([0.0]))

    bp = ax.boxplot(
        data, positions=positions,
        widths=0.7, patch_artist=True,
        showfliers=show_outliers,
        flierprops=dict(marker='.', markersize=1, alpha=0.3),
    )

    for i, (patch, color) in enumerate(zip(bp['boxes'], colors_list)):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)

    for sn in solver_names:
        label = SOLVER_LABELS.get(sn, sn)
        color = solver_colors.get(sn, '#333333')
        ax.plot([], [], 's', color=color, label=label, markersize=6)
    ax.legend(loc='upper right', fontsize=6, ncol=min(n_solvers, 5),
              framealpha=0.8)
