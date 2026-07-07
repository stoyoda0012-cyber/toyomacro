"""Shared constants and style definitions for error map visualization."""

import matplotlib

# ── Solver visual identity ──────────────────────────────────────────
# Consistent with JSAP figure palette, extended for new solvers

SOLVER_COLORS = {
    '4step':     '#2563EB',  # blue
    '6step':     '#059669',  # emerald green
    'dict1d':    '#D97706',  # amber
    'dict2d':    '#7C3AED',  # purple
    'adaptive':  '#DC2626',  # red
}

SOLVER_MARKERS = {
    '4step':     'o',
    '6step':     's',
    'dict1d':    'D',
    'dict2d':    '^',
    'adaptive':  'P',  # plus (filled)
}

SOLVER_LABELS = {
    '4step':     '4-Step',
    '6step':     '6-Step Hessian',
    'dict1d':    'Dict1D+4Step',
    'dict2d':    'Dict2D+4Step',
    'adaptive':  'Adaptive',
}

# ── Noise level display names ───────────────────────────────────────

NOISE_DISPLAY = {
    'NF':       'Noise-Free',
    'None':     'Noise-Free',
    'Moderate': 'Moderate (λ=10³)',
    'Strong':   'Strong (λ=10¹)',
    'lam1':     'λ=10¹',
    'lam2':     'λ=10²',
    'lam3':     'Moderate (λ=10³)',
    'lam3.5':   'λ=10³·⁵',
    'lam4':     'λ=10⁴',
}

# ── Noise badge colors (for noise comparison figure) ─────────────

NOISE_BADGE_COLORS = {
    'NF':       '#22C55E',  # green
    'Moderate': '#EAB308',  # yellow
    'Strong':   '#F97316',  # orange
}

# ── PSNR color thresholds (dB → color) ──────────────────────────
# Sorted descending: first match wins

PSNR_THRESHOLDS = [
    (45, '#22C55E'),  # excellent — green
    (30, '#EAB308'),  # good — yellow
    (20, '#F97316'),  # degraded — orange
    (0,  '#EF4444'),  # poor — red
]


def psnr_cell_color(val: float) -> str:
    """Return a color hex string for a PSNR value (dB)."""
    for thresh, color in PSNR_THRESHOLDS:
        if val >= thresh:
            return color
    return '#EF4444'


# ── Matplotlib RC defaults ──────────────────────────────────────────

RCPARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7.5,
    'figure.dpi': 150,
}


def apply_style():
    """Apply shared RC params."""
    matplotlib.rcParams.update(RCPARAMS)


def get_noise_label(noise_level: str) -> str:
    """Human-readable noise level label."""
    return NOISE_DISPLAY.get(noise_level, noise_level)
