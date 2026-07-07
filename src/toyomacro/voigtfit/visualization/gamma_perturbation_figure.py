"""
Gamma Perturbation Visualization
==============================================

3-panel figure:
  Left:   δσ_bias vs Δγ regression (per solver + Fisher prediction)
  Center: δE_bias vs Δγ (should be ~0)
  Right:  Bias coefficient comparison (bar chart)

Usage:
    uv run python -m toyomacro.voigtfit.visualization.gamma_perturbation_figure
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from toyomacro.voigtfit.benchmarks.bench_gamma_perturbation import (
    SNR_LEVELS,
    SOLVER_FUNCS,
    CellResult,
    compute_bias_slope,
)
from toyomacro.voigtfit.benchmarks.bench_gamma_perturbation import (
    main as run_benchmark,
)

# Colors for solvers
SOLVER_COLORS = {
    "parabola": "#1f77b4",
    "sorted": "#ff7f0e",
    "6step": "#2ca02c",
}

SOLVER_LABELS = {
    "parabola": "Parabola",
    "sorted": "4-step",
    "6step": "6-step",
}


def plot_gamma_perturbation(
    results: list[CellResult],
    fisher: dict,
    snr_filter: str = "NF",
    output_path: str | None = None,
):
    """Generate 3-panel gamma perturbation figure.

    Args:
        results: List of CellResult from benchmark
        fisher: Fisher prediction dict
        snr_filter: Which noise level to show in left/center panels
        output_path: Save path (default: outputs/session63/gamma_perturbation.png)
    """
    if output_path is None:
        output_path = "outputs/session63/gamma_perturbation.png"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw={"wspace": 0.35})

    solvers = [s for s in SOLVER_FUNCS if any(r.solver == s for r in results)]
    exact_coeff = fisher["bias_coeff_exact"]

    # === Panel 1: δσ bias vs Δγ ===
    ax = axes[0]
    dg_line = np.linspace(-0.025, 0.025, 100)

    for solver in solvers:
        cells = [r for r in results
                 if r.solver == solver and r.snr_name == snr_filter]
        cells.sort(key=lambda r: r.delta_gamma)

        x = np.array([r.delta_gamma for r in cells])
        y = np.array([r.dsigma_bias for r in cells])

        reg = compute_bias_slope(results, solver, snr_filter)

        ax.scatter(x, y, s=30, color=SOLVER_COLORS.get(solver, "gray"),
                   zorder=3)
        ax.plot(dg_line, reg["slope"] * dg_line + reg["intercept"],
                color=SOLVER_COLORS.get(solver, "gray"), linewidth=1.5,
                label=f'{SOLVER_LABELS.get(solver, solver)} ({reg["slope"]:+.3f})')

    # Fisher prediction line
    ax.plot(dg_line, exact_coeff * dg_line,
            color="red", linewidth=1.5, linestyle="--",
            label=f'Fisher ({exact_coeff:+.3f})')

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("$\\Delta\\gamma$ (eV)")
    ax.set_ylabel("$\\delta\\sigma$ bias (eV)")
    ax.set_title(f"$\\delta\\sigma$ bias vs $\\Delta\\gamma$ ({snr_filter})")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # === Panel 2: δE bias vs Δγ ===
    ax = axes[1]

    for solver in solvers:
        cells = [r for r in results
                 if r.solver == solver and r.snr_name == snr_filter]
        cells.sort(key=lambda r: r.delta_gamma)

        x = np.array([r.delta_gamma for r in cells])
        y = np.array([r.dE_bias for r in cells])
        y_scale = 1e3  # Show in meV for visibility

        ax.scatter(x, y * y_scale, s=30,
                   color=SOLVER_COLORS.get(solver, "gray"),
                   label=SOLVER_LABELS.get(solver, solver), zorder=3)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("$\\Delta\\gamma$ (eV)")
    ax.set_ylabel("$\\delta E$ bias (meV)")
    ax.set_title(f"$\\delta E$ bias vs $\\Delta\\gamma$ ({snr_filter})")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # === Panel 3: Bias coefficient comparison (bar chart) ===
    ax = axes[2]

    bar_data = []
    bar_labels = []
    bar_colors = []

    for solver in solvers:
        for snr_name in SNR_LEVELS:
            reg = compute_bias_slope(results, solver, snr_name)
            bar_data.append(reg["slope"])
            bar_labels.append(f"{SOLVER_LABELS.get(solver, solver)}\n{snr_name}")
            bar_colors.append(SOLVER_COLORS.get(solver, "gray"))

    x_pos = np.arange(len(bar_data))
    bars = ax.bar(x_pos, bar_data, color=bar_colors, alpha=0.7, width=0.7)

    # Fisher line
    ax.axhline(exact_coeff, color="red", linewidth=1.5, linestyle="--",
               label=f"Fisher = {exact_coeff:.3f}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Bias coefficient (slope)")
    ax.set_title("Measured vs Fisher prediction")
    ax.legend(fontsize=8)

    plt.tight_layout()

    # Ensure output directory
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {output_path}")
    plt.close(fig)


def main():
    """Run benchmark and generate figure."""
    print("Running benchmark (10K spectra for figure)...")
    results, fisher = run_benchmark(n_spectra=10_000)

    plot_gamma_perturbation(results, fisher, snr_filter="NF")

    # Also generate noise comparison
    plot_gamma_perturbation(
        results, fisher, snr_filter="SNR100",
        output_path="outputs/session63/gamma_perturbation_snr100.png",
    )


if __name__ == "__main__":
    main()
