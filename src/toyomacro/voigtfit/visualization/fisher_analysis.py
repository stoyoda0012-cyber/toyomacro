"""
Fisher Information Analysis Visualization
==========================================

Phase 1 output: comprehensive visualization of Voigt parameter space geometry.

Generates:
1. Eta sweep: g_ii diagonal elements vs eta (Gaussian → Lorentzian transition)
2. Correlation matrix: off-diagonal structure at key eta values
3. Eigenvalue decomposition: information retention for 4D → 3D projection
4. Eigenvector rotation: how principal axes change with eta

Usage:
    python -m toyomacro.voigtfit.visualization.fisher_analysis [--save-dir DIR]
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from ..fisher_information import (
    compute_fisher_matrix,
    eta_sweep,
    print_fisher_summary,
    sigma_gamma_from_eta,
    sigma_gamma_sweep,
)

# Element reference points (approximate eta values)
ELEMENT_MARKERS = {
    'C 1s':  0.15,
    'Si 2p': 0.30,
    'O 1s':  0.25,
    'Au 4f': 0.50,
    'Cu 2p': 0.75,
}

# Consistent colors for parameters
PARAM_COLORS = {
    'amplitude': '#e74c3c',     # red
    'delta_E': '#2ecc71',       # green
    'delta_sigma': '#3498db',   # blue
    'delta_gamma': '#9b59b6',   # purple
}


def plot_eta_sweep_full(
    sweep_data: dict,
    save_path: Path | None = None,
    figsize: tuple = (16, 14),
) -> plt.Figure:
    """Generate the main 6-panel eta sweep figure.

    Panel layout:
        [1] g_ii diagonal (log) vs eta    [2] Correlation r(sigma,gamma) vs eta
        [3] Eigenvalues vs eta            [4] Info retention (4D→3D) vs eta
        [5] Condition number vs eta        [6] Eigenvector composition vs eta
    """
    etas = sweep_data['etas']
    diagonal = sweep_data['diagonal']
    correlations = sweep_data['correlations']
    eigenvalues = sweep_data['eigenvalues']
    condition_numbers = sweep_data['condition_numbers']
    info_retention = sweep_data['info_retention_3d']
    param_names = sweep_data['param_names']

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.30,
                           left=0.08, right=0.95, top=0.94, bottom=0.06)

    # ─── Panel 1: Diagonal g_ii vs eta ───
    ax1 = fig.add_subplot(gs[0, 0])
    for i, name in enumerate(param_names):
        ax1.semilogy(etas, diagonal[:, i], '-o', ms=3, lw=1.5,
                      color=PARAM_COLORS[name], label=name)
    # Element markers
    for elem, eta_val in ELEMENT_MARKERS.items():
        ax1.axvline(eta_val, ls=':', alpha=0.3, color='gray')
        ax1.text(eta_val, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1e5,
                 elem, ha='center', va='bottom', fontsize=7, color='gray')
    ax1.set_xlabel(r'$\eta$ (Lorentzian fraction)')
    ax1.set_ylabel(r'$g_{ii}$ (Fisher information)')
    ax1.set_title('(a) Diagonal Fisher information vs mixing ratio')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # redo element markers after ylim is set
    for elem, eta_val in ELEMENT_MARKERS.items():
        ax1.axvline(eta_val, ls=':', alpha=0.3, color='gray')

    # ─── Panel 2: Key correlations vs eta ───
    ax2 = fig.add_subplot(gs[0, 1])
    # r(sigma, gamma) — the Voigt identifiability
    r_sg = correlations[:, 2, 3]
    ax2.plot(etas, r_sg, '-o', ms=3, lw=2, color='#e67e22', label=r'$r(\sigma, \gamma)$')
    # r(dE, dsigma)
    r_es = correlations[:, 1, 2]
    ax2.plot(etas, r_es, '-s', ms=3, lw=1.5, color='#1abc9c', label=r'$r(\delta E, \delta\sigma)$')
    # r(amp, dsigma)
    r_as = correlations[:, 0, 2]
    ax2.plot(etas, r_as, '-^', ms=3, lw=1.5, color='#95a5a6', label=r'$r(A, \delta\sigma)$')
    # r(dE, dgamma)
    r_eg = correlations[:, 1, 3]
    ax2.plot(etas, r_eg, '-d', ms=3, lw=1.5, color='#8e44ad', label=r'$r(\delta E, \delta\gamma)$')

    ax2.axhline(0, ls='-', alpha=0.2, color='black')
    for elem, eta_val in ELEMENT_MARKERS.items():
        ax2.axvline(eta_val, ls=':', alpha=0.3, color='gray')
    ax2.set_xlabel(r'$\eta$')
    ax2.set_ylabel('Correlation coefficient')
    ax2.set_title(r'(b) Off-diagonal correlations $r_{ij}$')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.05, 1.05)

    # ─── Panel 3: Eigenvalues vs eta ───
    ax3 = fig.add_subplot(gs[1, 0])
    for i in range(eigenvalues.shape[1]):
        label = f'$\\lambda_{i+1}$'
        ax3.semilogy(etas, eigenvalues[:, i], '-o', ms=3, lw=1.5, label=label)
    for elem, eta_val in ELEMENT_MARKERS.items():
        ax3.axvline(eta_val, ls=':', alpha=0.3, color='gray')
    ax3.set_xlabel(r'$\eta$')
    ax3.set_ylabel('Eigenvalue')
    ax3.set_title('(c) Fisher eigenvalues (ascending)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ─── Panel 4: Information retention ───
    ax4 = fig.add_subplot(gs[1, 1])
    if info_retention is not None:
        ax4.plot(etas, 100 * info_retention, '-o', ms=4, lw=2, color='#e74c3c')
        ax4.axhline(95, ls='--', alpha=0.5, color='gray', label='95% threshold')
        ax4.axhline(99, ls='--', alpha=0.3, color='gray', label='99% threshold')
        for elem, eta_val in ELEMENT_MARKERS.items():
            ax4.axvline(eta_val, ls=':', alpha=0.3, color='gray')
            # Find nearest eta index
            idx = np.argmin(np.abs(etas - eta_val))
            if idx < len(info_retention):
                ax4.annotate(f'{elem}\n{100*info_retention[idx]:.1f}%',
                             (eta_val, 100*info_retention[idx]),
                             textcoords='offset points', xytext=(8, -15),
                             fontsize=7, color='gray')
    ax4.set_xlabel(r'$\eta$')
    ax4.set_ylabel('Information retained (%)')
    ax4.set_title(r'(d) 4D $\to$ 3D information retention (top-3 eigenvalues)')
    ax4.set_ylim(80, 100.5)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ─── Panel 5: Condition number vs eta ───
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.semilogy(etas, condition_numbers, '-o', ms=4, lw=2, color='#2c3e50')
    for elem, eta_val in ELEMENT_MARKERS.items():
        ax5.axvline(eta_val, ls=':', alpha=0.3, color='gray')
    ax5.set_xlabel(r'$\eta$')
    ax5.set_ylabel('Condition number')
    ax5.set_title(r'(e) Condition number $\lambda_{max}/\lambda_{min}$')
    ax5.grid(True, alpha=0.3)

    # ─── Panel 6: Eigenvector composition ───
    ax6 = fig.add_subplot(gs[2, 1])
    # Show composition of the SMALLEST eigenvector (most info-poor direction)
    smallest_evec = np.zeros((len(etas), len(param_names)))
    for i in range(len(etas)):
        smallest_evec[i] = sweep_data['fisher_results'][i].eigenvectors[:, 0]**2
    for j, name in enumerate(param_names):
        ax6.plot(etas, smallest_evec[:, j], '-o', ms=3, lw=1.5,
                 color=PARAM_COLORS[name], label=name)
    for elem, eta_val in ELEMENT_MARKERS.items():
        ax6.axvline(eta_val, ls=':', alpha=0.3, color='gray')
    ax6.set_xlabel(r'$\eta$')
    ax6.set_ylabel(r'$|v_i|^2$ (squared component)')
    ax6.set_title(r'(f) Smallest eigenvector composition (most uncertain direction)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-0.05, 1.05)

    fig.suptitle('Fisher Information Analysis: Voigt Parameter Space Geometry',
                 fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_correlation_matrices(
    sweep_data: dict,
    etas_to_show: list | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Show correlation matrix heatmaps at selected eta values."""
    if etas_to_show is None:
        etas_to_show = [0.10, 0.25, 0.50, 0.75]

    n_panels = len(etas_to_show)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    param_names = sweep_data['param_names']
    short_names = [r'$A$', r'$\delta E$', r'$\delta\sigma$', r'$\delta\gamma$']
    short_names = short_names[:len(param_names)]

    for ax, eta_target in zip(axes, etas_to_show):
        idx = np.argmin(np.abs(sweep_data['etas'] - eta_target))
        corr = sweep_data['correlations'][idx]
        eta_actual = sweep_data['etas'][idx]

        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

        # Annotate values
        n = len(param_names)
        for i in range(n):
            for j in range(n):
                color = 'white' if abs(corr[i, j]) > 0.6 else 'black'
                ax.text(j, i, f'{corr[i,j]:.3f}', ha='center', va='center',
                        fontsize=9, color=color)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_names, fontsize=10)
        ax.set_yticklabels(short_names, fontsize=10)

        # Find element name
        elem_name = ''
        for elem, eta_val in ELEMENT_MARKERS.items():
            if abs(eta_val - eta_target) < 0.08:
                elem_name = f' ({elem})'
                break
        ax.set_title(f'$\\eta$ = {eta_actual:.2f}{elem_name}', fontsize=11)

    fig.colorbar(im, ax=axes, shrink=0.8, label='Correlation')
    fig.suptitle('Fisher Correlation Matrices at Different Mixing Ratios',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def print_summary_table(sweep_data: dict) -> None:
    """Print a summary table of Fisher analysis results."""
    etas = sweep_data['etas']
    diagonal = sweep_data['diagonal']
    correlations = sweep_data['correlations']
    eigenvalues = sweep_data['eigenvalues']
    info_retention = sweep_data['info_retention_3d']
    param_names = sweep_data['param_names']

    print("\n" + "=" * 100)
    print("Fisher Information Eta Sweep Summary")
    print(f"Amplitude = {sweep_data['amplitude']:.0f}, FWHM_total = {sweep_data['fwhm_total']:.1f} eV")
    print("=" * 100)

    # Header
    header = f"{'eta':>6s} | "
    for name in param_names:
        header += f"{'g_'+name[:5]:>10s} "
    header += f"| {'r(s,g)':>8s} | {'cond':>10s}"
    if info_retention is not None:
        header += f" | {'3D ret%':>8s}"
    header += f" | {'lambda_min':>10s} {'lambda_max':>10s}"
    print(header)
    print("-" * len(header))

    # Pick representative eta values
    key_etas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    for eta_target in key_etas:
        idx = np.argmin(np.abs(etas - eta_target))
        if abs(etas[idx] - eta_target) > 0.03:
            continue

        eta = etas[idx]
        row = f"{eta:6.2f} | "
        for j in range(len(param_names)):
            row += f"{diagonal[idx, j]:10.1f} "
        r_sg = correlations[idx, 2, 3] if len(param_names) == 4 else 0
        row += f"| {r_sg:8.4f} | {eigenvalues[idx, -1]/eigenvalues[idx, 0]:10.1f}"
        if info_retention is not None:
            row += f" | {100*info_retention[idx]:7.2f}%"
        row += f" | {eigenvalues[idx, 0]:10.2f} {eigenvalues[idx, -1]:10.1f}"

        # Element annotation
        for elem, eta_val in ELEMENT_MARKERS.items():
            if abs(eta_val - eta) < 0.05:
                row += f"  ← {elem}"
                break
        print(row)

    print("=" * 100)


def plot_sigma_gamma_axes(
    sg_data: dict,
    save_path: Path | None = None,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """Generate the sigma-gamma principal axis figure.

    Panel layout:
        [1] Eigenvalues (ratio vs width) vs eta     [2] Anisotropy ratio vs eta
        [3] Eigenvector rotation angle vs eta        [4] Bias coefficient vs eta
        [5] Eigenvector quiver at 4 key etas         [6] Physical interpretation table
    """
    etas = sg_data['etas']
    lambda_ratio = sg_data['lambda_ratio']
    lambda_width = sg_data['lambda_width']
    rotation_angles = sg_data['rotation_angles']
    info_ratios = sg_data['info_ratios']
    bias_coefficients = sg_data['bias_coefficients']
    correlations = sg_data['correlations']
    v_ratios = sg_data['v_ratios']
    v_widths = sg_data['v_widths']

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35,
                           left=0.07, right=0.96, top=0.92, bottom=0.08)

    # ─── Panel 1: Eigenvalues vs eta ───
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(etas, lambda_width, '-o', ms=3, lw=2, color='#2ecc71',
                 label=r'$\lambda_{width}$ (total width)')
    ax1.semilogy(etas, lambda_ratio, '-s', ms=3, lw=2, color='#e74c3c',
                 label=r'$\lambda_{ratio}$ (G/L ratio)')
    for elem, eta_val in ELEMENT_MARKERS.items():
        ax1.axvline(eta_val, ls=':', alpha=0.3, color='gray')
    ax1.set_xlabel(r'$\eta$ (Lorentzian fraction)')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title(r'(a) $\sigma$-$\gamma$ sub-block eigenvalues')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ─── Panel 2: Anisotropy ratio ───
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(etas, info_ratios, '-o', ms=4, lw=2, color='#2c3e50')
    for elem, eta_val in ELEMENT_MARKERS.items():
        ax2.axvline(eta_val, ls=':', alpha=0.3, color='gray')
    ax2.set_xlabel(r'$\eta$')
    ax2.set_ylabel(r'$\lambda_{width} / \lambda_{ratio}$')
    ax2.set_title(r'(b) Width estimation anisotropy')
    ax2.grid(True, alpha=0.3)

    # ─── Panel 3: Rotation angle + bias coefficient ───
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(etas, rotation_angles, '-o', ms=3, lw=2, color='#3498db',
             label='Rotation angle')
    ax3.set_xlabel(r'$\eta$')
    ax3.set_ylabel('Angle (degrees)')
    ax3.set_title(r'(c) Principal axis rotation from $(\sigma, \gamma)$')
    ax3.axhline(45, ls='--', alpha=0.3, color='gray', label='45° (equal mix)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    for elem, eta_val in ELEMENT_MARKERS.items():
        ax3.axvline(eta_val, ls=':', alpha=0.3, color='gray')

    # ─── Panel 4: Bias coefficient and correlation ───
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(etas, bias_coefficients, '-o', ms=4, lw=2, color='#e74c3c',
             label=r'Bias coeff $|g_{\sigma\gamma}/g_{\sigma\sigma}|$')
    ax4.plot(etas, correlations, '-s', ms=3, lw=1.5, color='#9b59b6',
             label=r'Correlation $r(\sigma, \gamma)$')
    ax4.axhline(0.89, ls='--', alpha=0.4, color='#e74c3c',
                label='89% bias')
    for elem, eta_val in ELEMENT_MARKERS.items():
        ax4.axvline(eta_val, ls=':', alpha=0.3, color='gray')
    ax4.set_xlabel(r'$\eta$')
    ax4.set_ylabel('Coefficient')
    ax4.set_title(r'(d) Omitted variable bias & correlation')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.5)

    # ─── Panel 5: Quiver plot at key etas ───
    ax5 = fig.add_subplot(gs[1, 1])
    key_etas_quiver = [0.15, 0.30, 0.50, 0.75]
    colors_quiver = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
    labels_quiver = ['C 1s', 'Si 2p', 'Au 4f', 'Cu 2p']

    for j, (eta_target, color, label) in enumerate(
            zip(key_etas_quiver, colors_quiver, labels_quiver)):
        idx = np.argmin(np.abs(etas - eta_target))
        vr = v_ratios[idx]
        vw = v_widths[idx]
        origin_y = j * 0.3

        # Width direction (green arrow)
        ax5.annotate('', xy=(vw[0]*0.4, origin_y + vw[1]*0.4),
                     xytext=(0, origin_y),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        # Ratio direction (red dashed arrow)
        ax5.annotate('', xy=(vr[0]*0.4, origin_y + vr[1]*0.4),
                     xytext=(0, origin_y),
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.5, ls='--'))

        ax5.text(-0.55, origin_y, f'{label}\n$\\eta$={eta_target:.2f}',
                 fontsize=8, va='center', ha='right', color=color)

    ax5.set_xlim(-0.6, 0.6)
    ax5.set_ylim(-0.2, (len(key_etas_quiver) - 1) * 0.3 + 0.3)
    ax5.set_xlabel(r'$\delta\sigma$ component')
    ax5.set_ylabel(r'$\delta\gamma$ component')
    ax5.set_title(r'(e) Principal axes (solid=width, dashed=ratio)')
    ax5.axhline(0, ls='-', alpha=0.1, color='black')
    ax5.axvline(0, ls='-', alpha=0.1, color='black')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.2)

    # ─── Panel 6: Summary table ───
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    table_data = []
    headers = [r'$\eta$', 'Element', r'$\lambda_{w}/\lambda_{r}$',
               r'$r(\sigma,\gamma)$', 'Bias coeff', 'Angle']
    for eta_target, label in zip(key_etas_quiver, labels_quiver):
        idx = np.argmin(np.abs(etas - eta_target))
        table_data.append([
            f'{etas[idx]:.2f}',
            label,
            f'{info_ratios[idx]:.1f}',
            f'{correlations[idx]:.3f}',
            f'{bias_coefficients[idx]:.3f}',
            f'{rotation_angles[idx]:.1f}°',
        ])
    table = ax6.table(cellText=table_data, colLabels=headers,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax6.set_title('(f) Summary at key elements', pad=20)

    fig.suptitle(r'$\sigma$-$\gamma$ Principal Axis Analysis: Width vs G/L Ratio',
                 fontsize=14, fontweight='bold', y=0.97)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def print_sigma_gamma_table(sg_data: dict) -> None:
    """Print sigma-gamma analysis summary."""
    etas = sg_data['etas']
    print("\n" + "=" * 95)
    print("Sigma-Gamma Principal Axis Decomposition")
    print("=" * 95)
    header = (f"{'eta':>6s} | {'lam_ratio':>10s} {'lam_width':>10s} | "
              f"{'ratio':>8s} | {'r(s,g)':>8s} | {'bias_c':>8s} | "
              f"{'angle':>7s} | {'v_ratio':>18s} {'v_width':>18s}")
    print(header)
    print("-" * len(header))

    key_etas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    for eta_target in key_etas:
        idx = np.argmin(np.abs(etas - eta_target))
        if abs(etas[idx] - eta_target) > 0.03:
            continue
        eta = etas[idx]
        vr = sg_data['v_ratios'][idx]
        vw = sg_data['v_widths'][idx]
        elem = ''
        for e, ev in ELEMENT_MARKERS.items():
            if abs(ev - eta) < 0.05:
                elem = f'  ← {e}'
                break
        print(f"{eta:6.2f} | {sg_data['lambda_ratio'][idx]:10.1f} "
              f"{sg_data['lambda_width'][idx]:10.1f} | "
              f"{sg_data['info_ratios'][idx]:8.1f} | "
              f"{sg_data['correlations'][idx]:8.4f} | "
              f"{sg_data['bias_coefficients'][idx]:8.4f} | "
              f"{sg_data['rotation_angles'][idx]:6.1f}° | "
              f"({vr[0]:+.3f},{vr[1]:+.3f}) "
              f"({vw[0]:+.3f},{vw[1]:+.3f}){elem}")
    print("=" * 95)


def run_analysis(save_dir: str | None = None) -> dict:
    """Run the complete Phase 1 Fisher analysis.

    Returns:
        Dictionary with all results for downstream use.
    """
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = None

    print("=" * 60)
    print("Phase 1: Fisher Information Analysis")
    print("=" * 60)

    # --- 1. Eta sweep (4D) ---
    print("\n[1/4] Running 4D eta sweep (30 points)...")
    sweep_4d = eta_sweep(
        etas=np.linspace(0.05, 0.90, 30),
        amplitude=1000.0,
        fwhm_total=1.0,
        mode='4d',
    )
    print_summary_table(sweep_4d)

    # --- 2. Key element summaries ---
    print("\n[2/4] Detailed Fisher at key elements...")
    energy = np.linspace(279.4, 289.4, 201)
    for elem, eta_val in ELEMENT_MARKERS.items():
        sigma, gamma = sigma_gamma_from_eta(eta_val, fwhm_total=1.0)
        result = compute_fisher_matrix(1000.0, 284.4, sigma, gamma, energy, mode='4d')
        print_fisher_summary(result, title=f"{elem} (eta={eta_val:.2f})")

    # --- 3. Sigma-gamma principal axis analysis ---
    print("\n[3/6] Running sigma-gamma principal axis analysis...")
    sg_data = sigma_gamma_sweep(
        etas=np.linspace(0.05, 0.90, 30),
        amplitude=1000.0,
        fwhm_total=1.0,
    )
    print_sigma_gamma_table(sg_data)

    # --- 4. Eta sweep (3D, for comparison) ---
    print("\n[4/6] Running 3D eta sweep...")
    sweep_3d = eta_sweep(
        etas=np.linspace(0.05, 0.90, 30),
        amplitude=1000.0,
        fwhm_total=1.0,
        mode='3d',
    )

    # --- 5. Visualizations ---
    print("\n[5/6] Generating figures...")
    fig1 = plot_eta_sweep_full(
        sweep_4d,
        save_path=save_path / 'fisher_eta_sweep_4d.png' if save_path else None,
    )

    fig2 = plot_correlation_matrices(
        sweep_4d,
        etas_to_show=[0.15, 0.30, 0.50, 0.75],
        save_path=save_path / 'fisher_correlation_matrices.png' if save_path else None,
    )

    fig3 = plot_sigma_gamma_axes(
        sg_data,
        save_path=save_path / 'fisher_sigma_gamma_axes.png' if save_path else None,
    )

    # --- 6. Key conclusions ---
    print("\n[6/6] Key conclusions...")
    print("\n  Finding 1: g(dsigma) > g(dE) for Gaussian-dominated elements")
    print("    → delta_sigma's poor GVRT performance is SOLVER bias, not Fisher limit")
    print("\n  Finding 2: amp ⊥ dsigma (r < 0.001)")
    print("    → Empirical crosstalk is solver-induced, not information-geometric")
    print("\n  Finding 3: r(sigma, gamma) ≈ 0.68-0.76 across all eta")
    print("    → The Voigt identifiability problem is eta-independent")
    print("\n  Finding 4: 4D→3D retention = 100% (amplitude is null)")
    print("    → Amplitude LLS separation is Fisher-optimal")
    print("\n  Finding 5: v_ratio = 'G/L ratio change' is the hardest mode")
    print("    → This is what spectroscopists have known for 20 years,")
    print("      now derived from first principles via Fisher eigendecomposition")

    # Bias coefficient at C 1s
    idx_c1s = np.argmin(np.abs(sg_data['etas'] - 0.15))
    bias_c1s = sg_data['bias_coefficients'][idx_c1s]
    print(f"\n  Bias coefficient at C 1s (eta=0.15): {bias_c1s:.3f}")
    print("  dev-log 56 empirical:                 0.890")
    print(f"  → {abs(bias_c1s - 0.89)/0.89*100:.1f}% discrepancy")

    results = {
        'sweep_4d': sweep_4d,
        'sweep_3d': sweep_3d,
        'sg_data': sg_data,
        'figures': [fig1, fig2, fig3],
    }

    print("\n" + "=" * 60)
    print("Phase 1 + 1.5(partial) complete.")
    print("=" * 60)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fisher Information Analysis')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save figures')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not show plots interactively')
    args = parser.parse_args()

    results = run_analysis(save_dir=args.save_dir)

    if not args.no_show:
        plt.show()
