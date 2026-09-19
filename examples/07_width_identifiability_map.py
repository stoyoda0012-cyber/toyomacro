"""
Example 07: how well can a spectrum tell the Gaussian from the Lorentzian width?
===============================================================================

Maps the identifiability of the two Voigt widths over measurement
conditions, with ``toyomacro.voigtfit.identifiability`` (experimental).

A Voigt profile is a Lorentzian convolved with a Gaussian, and the two
widths are hard to separate for several different reasons that are
usually blamed on each other. This example pulls them apart:

- panel (c): as the Gaussian component shrinks, the bound on *sigma*
  diverges while the bound on the *variance* sigma**2 does not move. That
  is the coordinate, not the data. What does grow is the uncertainty of
  the variance relative to itself: a small Gaussian component cannot be
  detected however the width is parameterised.
- panels (a), (b), (d): narrowing the window removes the tails that carry
  the Lorentzian width, with or without a background. In a narrow window
  a background that has to be estimated from the same spectrum costs far
  more than its shot noise: at +-0.6 FWHM the shot noise raises the
  bound on gamma by 9 % and estimating the level by another factor 19.
  The two cross near +-3 FWHM, and at +-15 FWHM it is the other way
  round: the shot noise doubles the bound and estimating the level adds
  a tenth. The dashed line is what a diagnostic that treats amplitude,
  position and background as known would report.

Everything here is computed from a model (Poisson Fisher information at
stated parameters). It is not a measurement and not a statement about any
fit; see the module docstring for what the labels do and do not mean.

Output: examples/output/07_width_identifiability.{png,npz} (not tracked).
Runs in a few seconds, NumPy only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from toyomacro.voigtfit.identifiability import ScanGrid, scan_identifiability

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)
STEM = "07_width_identifiability"

# one sequential hue for magnitude, fixed categorical order for identity
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

HALF_WIDTHS = tuple(np.round(np.geomspace(0.3, 15.0, 12), 3))
RATIOS = (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)


def run_scans():
    common = dict(half_widths=HALF_WIDTHS, ratios=RATIOS, area=1.0e4)
    exposure = scan_identifiability(ScanGrid(levels=(1.0,), **common))
    counts = scan_identifiability(
        ScanGrid(levels=(1.0e6,), normalization="total_counts", **common)
    )
    return exposure, counts


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c9c8c2")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def heatmap(ax, scan, background, title):
    b = list(scan["backgrounds"]).index(background)
    # ratios[0] is sigma = 0: it has no place on a log axis; drawn as the bottom row
    values = scan["worst_relative_sd"][b, 0, :, :, 0]
    rows = np.arange(len(RATIOS) + 1)
    mesh = ax.pcolormesh(
        np.arange(len(HALF_WIDTHS) + 1), rows, values,
        cmap=SEQUENTIAL, norm=LogNorm(1e-3, 10.0), edgecolors=SURFACE, linewidth=0.8,
    )
    weak = scan["status"][b, 0, :, :, 0] >= 1
    for (r, h) in zip(*np.nonzero(weak)):
        ax.plot(h + 0.5, r + 0.5, marker="x", color=SURFACE, markersize=4, mew=1.2)
    ax.set_xticks(np.arange(len(HALF_WIDTHS))[::2] + 0.5, [f"{v:.2g}" for v in HALF_WIDTHS[::2]])
    ax.set_yticks(rows[:-1] + 0.5, ["0"] + [f"{v:g}" for v in RATIOS[1:]])
    ax.set_xlabel("window half-width / FWHM", fontsize=8)
    ax.set_ylabel("sigma / gamma", fontsize=8)
    style(ax, title)
    return mesh


def main():
    exposure, counts = run_scans()
    np.savez(OUT_DIR / f"{STEM}.npz", **{f"exposure__{k}": v for k, v in exposure.items()},
             **{f"total_counts__{k}": v for k, v in counts.items()})

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4), facecolor=SURFACE, layout="constrained")
    (ax_a, ax_b), (ax_c, ax_d) = axes

    mesh = heatmap(ax_a, exposure, "estimated", "(a) flat background, estimated")
    heatmap(ax_b, exposure, "none", "(b) no background")
    bar = fig.colorbar(mesh, ax=[ax_a, ax_b], shrink=0.9, pad=0.01)
    bar.set_label("bound on the least determined width direction / its scale\n"
                  "(x: above 0.1, 'weakly identified')", fontsize=8, color=MUTED)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    bar.outline.set_visible(False)

    est = list(exposure["backgrounds"]).index("estimated")
    wide = len(HALF_WIDTHS) - 3
    ratios = np.array(RATIOS[1:])
    series = (
        ("sd_sigma", "sigma, over FWHM/2.35", ORANGE),
        ("sd_variance", "variance, over FWHM$^2$/5.55", BLUE),
        ("variance_relative_sd", "variance, over itself", AQUA),
    )
    for key, label, color in series:
        y = exposure[key][est, 0, 1:, wide, 0]
        ax_c.plot(ratios, y, color=color, lw=2, marker="o", ms=4, label=label)
        ax_c.annotate(label, (ratios[0], y[0]), xytext=(6, 4), textcoords="offset points",
                      fontsize=8, color=INK)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel(
        f"sigma / gamma   (fixed total FWHM; window +-{HALF_WIDTHS[wide]:.2g} FWHM, background as in a)",
        fontsize=8,
    )
    ax_c.set_ylabel("relative bound on the Gaussian width", fontsize=8)
    ax_c.grid(True, which="major", color="#e6e5e0", lw=0.6)
    ax_c.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    style(ax_c, "(c) the coordinate, or the information?")

    half = np.array(HALF_WIDTHS)
    one = RATIOS.index(1.0)
    for name, color in (("none", BLUE), ("known", ORANGE), ("estimated", AQUA)):
        b = list(exposure["backgrounds"]).index(name)
        ax_d.plot(half, exposure["sd_gamma"][b, 0, one, :, 0], color=color, lw=2, marker="o",
                  ms=4, label=f"background {name}")
    ax_d.plot(half, exposure["sd_gamma_conditional"][est, 0, one, :, 0], color=AQUA, lw=2,
              ls=(0, (4, 2)), label="estimated, everything else taken as known")
    ax_d.axhline(0.1, color=MUTED, lw=0.8)
    ax_d.annotate("0.1 of the FWHM", (half[-1], 0.1), xytext=(0, 3), textcoords="offset points",
                  ha="right", fontsize=8, color=MUTED)
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel(
        "window half-width / FWHM   (sigma/gamma = 1; fixed step and exposure per point)", fontsize=8
    )
    ax_d.set_ylabel("bound on gamma / FWHM", fontsize=8)
    ax_d.grid(True, which="major", color="#e6e5e0", lw=0.6)
    ax_d.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    style(ax_d, "(d) cost of a narrow window and of a background")

    fig.suptitle("Identifiability of the Voigt widths: model bounds, not measurements",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT_DIR / f"{STEM}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # the same table the panels are drawn from, for the terminal
    print("sigma/gamma = 1, exposure per point fixed: bound on gamma / FWHM")
    print("  half-width   none      known     estimated  (est., rest known)   same counts, est.")
    ct = list(counts["backgrounds"]).index("estimated")
    for h, value in enumerate(HALF_WIDTHS):
        cols = [exposure["sd_gamma"][list(exposure["backgrounds"]).index(k), 0, one, h, 0]
                for k in ("none", "known", "estimated")]
        print(f"  {value:9.3g}  {cols[0]:9.2e} {cols[1]:9.2e} {cols[2]:9.2e}"
              f"      {exposure['sd_gamma_conditional'][est, 0, one, h, 0]:9.2e}"
              f"          {counts['sd_gamma'][ct, 0, one, h, 0]:9.2e}")
    print(f"saved -> {OUT_DIR / (STEM + '.png')}, {OUT_DIR / (STEM + '.npz')}")


if __name__ == "__main__":
    main()
