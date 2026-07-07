"""Visualization module: error maps, bias-variance plots, and diagnostics."""

from .adaptive_mask import plot_adaptive_mask
from .bias_variance import plot_bias_variance_scatter
from .common import (
    NOISE_BADGE_COLORS,
    NOISE_DISPLAY,
    PSNR_THRESHOLDS,
    SOLVER_COLORS,
    SOLVER_LABELS,
    SOLVER_MARKERS,
    psnr_cell_color,
)
from .error_histograms import plot_error_histograms
from .error_maps import plot_error_maps
from .gvrt_noise_comparison import (
    FINE_NOISE_LEVELS,
    build_noise_comparison_data,
    plot_channel_noise_grid,
    plot_noise_comparison,
    plot_reconstructed_images,
)
from .gvrt_tracking import (
    build_tracking_data,
    build_tracking_data_v4,
    plot_gvrt_tracking,
    plot_gvrt_tracking_v4,
)

__all__ = [
    'plot_error_maps',
    'plot_bias_variance_scatter',
    'plot_error_histograms',
    'plot_adaptive_mask',
    'plot_gvrt_tracking',
    'build_tracking_data',
    'plot_gvrt_tracking_v4',
    'build_tracking_data_v4',
    'plot_noise_comparison',
    'build_noise_comparison_data',
    'plot_reconstructed_images',
    'plot_channel_noise_grid',
    'FINE_NOISE_LEVELS',
    'SOLVER_COLORS', 'SOLVER_MARKERS', 'SOLVER_LABELS', 'NOISE_DISPLAY',
    'NOISE_BADGE_COLORS', 'PSNR_THRESHOLDS', 'psnr_cell_color',
]
