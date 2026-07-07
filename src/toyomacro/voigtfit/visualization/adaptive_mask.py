"""Adaptive solver Phase-2 selection mask visualization.

Shows which pixels used 4-step refinement (clean) vs amp-only (noisy)
in the adaptive Dict2D solver.
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from ..evaluation.benchmark_result import BenchmarkResult
from .common import apply_style, get_noise_label

# Two-color map: blue = amp_only, orange = 4-step
_ADAPTIVE_CMAP = ListedColormap(['#93C5FD', '#F97316'])  # light-blue, orange


def plot_adaptive_mask(
    ax: plt.Axes,
    phase2_mask: np.ndarray,
    image_shape: tuple[int, int],
    chi2_values: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    """Draw the adaptive Phase-2 selection mask.

    Args:
        ax: Axes to draw on
        phase2_mask: (n_pixels,) bool — True where 4-step was applied
        image_shape: (H, W)
        chi2_values: (n_pixels,) optional chi2_norm values for continuous map
        title: Optional title
    """
    H, W = image_shape

    if chi2_values is not None:
        # Continuous chi2 map with threshold overlay
        chi2_map = chi2_values.reshape(H, W)
        im = ax.imshow(
            np.log10(chi2_map + 1e-20), cmap='viridis',
            aspect='equal', interpolation='nearest',
        )
        # Overlay mask boundary
        mask_map = phase2_mask.reshape(H, W).astype(float)
        ax.contour(mask_map, levels=[0.5], colors='white', linewidths=0.8)
        plt.colorbar(im, ax=ax, shrink=0.8, label='log₁₀(χ²/peak²)')
    else:
        # Binary mask
        mask_map = phase2_mask.astype(np.int32).reshape(H, W)
        ax.imshow(mask_map, cmap=_ADAPTIVE_CMAP, vmin=0, vmax=1,
                  aspect='equal', interpolation='nearest')

    n_4step = int(np.sum(phase2_mask))
    n_total = len(phase2_mask)
    pct = 100.0 * n_4step / max(n_total, 1)

    ax.set_xticks([])
    ax.set_yticks([])

    # Badge with percentage
    ax.text(
        0.97, 0.03,
        f'4-step: {pct:.1f}%',
        transform=ax.transAxes,
        fontsize=7, fontweight='bold',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray',
                  alpha=0.85, lw=0.5),
    )

    if title:
        ax.set_title(title, fontsize=9)


def plot_adaptive_mask_grid(
    results: dict[str, BenchmarkResult],
    figsize: tuple[float, float] = (16, 4),
    output_path: str | None = None,
    show: bool = False,
) -> plt.Figure | None:
    """Plot adaptive masks for multiple noise levels in a row.

    Args:
        results: results[noise_level] = BenchmarkResult (adaptive solver)
        figsize: Figure size
        output_path: Save path
        show: Show interactively

    Returns:
        Figure, or None if no adaptive masks found
    """
    apply_style()

    # Filter to results with phase2_mask
    valid = {nl: br for nl, br in results.items()
             if br.phase2_mask is not None}
    if not valid:
        return None

    noise_levels = list(valid.keys())
    n = len(noise_levels)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, nl in zip(axes, noise_levels):
        br = valid[nl]
        plot_adaptive_mask(
            ax, br.phase2_mask, br.image_shape,
            title=f'Adaptive: {get_noise_label(nl)}',
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#93C5FD', label='Amp-only'),
        Patch(facecolor='#F97316', label='4-Step'),
    ]
    fig.legend(
        handles=legend_elements, loc='lower center',
        ncol=2, fontsize=8, framealpha=0.8,
    )

    fig.suptitle('Adaptive Solver: Phase-2 Selection Map',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    if output_path:
        fig.savefig(str(output_path), dpi=200, bbox_inches='tight',
                    facecolor='white')
        print(f"Saved: {output_path}")
    if show:
        plt.show()

    return fig
