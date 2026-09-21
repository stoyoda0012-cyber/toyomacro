"""
Example 09: can this Fermi edge tell the resolution from the temperature?
=========================================================================

Four questions a Fermi-edge measurement has to answer before its fitted
resolution means anything, with
``toyomacro.fitting.fermi_edge_identifiability`` (experimental).

- panel (a): the edge width is one number, ``kappa_2 = v + (pi^2/3) tau``,
  and every count in the spectrum measures it. Splitting it into an
  instrumental half and a thermal half is a different question, and only
  the fourth cumulant answers it. The share is determined where the curve
  is below the dashed line and undetermined above it.
- panel (b): what fitting the temperature costs. Freeing it multiplies
  the uncertainty of the resolution by between 1.3 and 78 over these
  conditions, and the shorter the thermal tail the worse -- at
  kT/sigma = 0.1 there is nothing in the spectrum to measure a
  temperature with, so the fit spends the resolution's precision on it
  instead.
- panel (c): which error bar belongs to which estimator. The package's
  own fitter minimises a weighted sum of squares with unit weights, so
  its uncertainty is a sandwich covariance, not the Cramer-Rao bound.
  Quoting the bound for it understates the error by a quarter to a factor
  of 2.5.
- panel (d): the density of states is an assumption. The same spectrum
  fitted with the DOS kinked at E_F and continued smoothly through it
  gives two resolutions; the gap between them is a systematic that the
  statistical error bar does not contain, and it is *not* something to
  resolve by keeping the better fit. On this one spectrum the gap is
  larger than the error bars; whether it is depends on the counts and
  on kT/sigma, and the panel is one draw, not a general claim.

Panels (a)-(c) are computed from a model (Poisson Fisher information at
stated parameters): bounds, not measurements, and not statements about
any particular fit. Panel (d) fits one simulated Poisson spectrum from a
fixed seed.
See the module docstring and docs/design/fermi-edge-identifiability.md.

Output: examples/output/09_fermi_edge_identifiability.{png,npz} (not tracked).
Runs in a few seconds, NumPy + matplotlib only.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from toyomacro.fitting.fermi_edge import compare_dos_forms, fermi_edge
from toyomacro.fitting.fermi_edge_identifiability import (
    EdgeScanGrid,
    FermiEdge,
    assess_edge_identifiability,
    constant_background,
    scan_edge_identifiability,
)

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)
STEM = "09_fermi_edge_identifiability"

# fixed categorical order, never cycled; one sequential hue is not needed here
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e3e2dd"
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

RATIOS = (0.1, 0.2, 0.5, 1.0, 2.0, 3.0)
MODES = ("fixed_temperature", "temperature_prior", "free")
MODE_LABEL = {"fixed_temperature": "T fixed", "temperature_prior": "T ± 10 %",
              "free": "T fitted"}
MODE_COLOUR = {"fixed_temperature": BLUE, "temperature_prior": AQUA, "free": ORANGE}

# panel (c): a 2 eV window in 10 meV steps, 2000 counts per channel on the plateau
CONDITIONS = ((0.30, 300.0), (0.10, 300.0), (0.050, 300.0), (0.030, 30.0))


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c9c8c2")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)


def run_scan():
    """The module's own sweep, at one window and an estimated background."""
    grid = EdgeScanGrid(ratios=RATIOS, half_widths=(6.0,), slopes=(0.3,),
                        backgrounds=("estimated",), temperature_modes=MODES,
                        dos_forms=("occupied",))
    return grid, scan_edge_identifiability(grid)


def panel_split(ax, grid, scan):
    """(a) the total width is easy, the share is not.

    Only two curves, not three. With the temperature fixed the module
    reports the width information of the *free* matrix on purpose: the
    split is then an assumption, labelled ``assumed``, and the number
    describes what the data could have said rather than what the fit
    claimed. So `fixed_temperature` and `free` are the same curve.
    """
    share = scan["sd_share"][:, 0, 0, 0, 0, :, 0]
    total = scan["sd_kappa2"][:, 0, 0, 0, 0, :, 0]
    free = list(grid.temperature_modes).index("free")
    prior = list(grid.temperature_modes).index("temperature_prior")
    ax.plot(RATIOS, total[:, free], marker="s", ms=5, lw=2, ls=":", color=MUTED,
            label="the total width κ₂")
    ax.plot(RATIOS, share[:, free], marker="o", ms=5, lw=2, color=ORANGE,
            label="its instrumental share, from this spectrum alone")
    ax.plot(RATIOS, share[:, prior], marker="o", ms=5, lw=2, color=AQUA,
            label="its share, given T to ± 10 %")
    ax.axhline(grid.thresholds.separation_sd, color=INK, lw=1, ls="--")
    ax.text(RATIOS[-1], grid.thresholds.separation_sd * 1.25, "not separable above",
            fontsize=7.5, color=INK, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(3e-3, 8.0)
    style(ax, "(a) one width is measured, its split is not",
          "kT / σ", "relative standard deviation")
    ax.legend(fontsize=7, frameon=False, labelcolor=MUTED, loc="lower left")
    ax.text(0.98, 0.955, "with T held fixed the split is assumed,\nand the orange curve is "
            "what the data\ncould have said instead",
            transform=ax.transAxes, fontsize=7, color=MUTED, ha="right", va="top")


def panel_inflation(ax, grid, scan):
    """(b) what fitting the temperature costs."""
    sd = scan["sd_sigma"][:, 0, 0, 0, 0, :, 0]
    fixed = sd[:, list(grid.temperature_modes).index("fixed_temperature")]
    for mode in ("temperature_prior", "free"):
        j = list(grid.temperature_modes).index(mode)
        ax.plot(RATIOS, sd[:, j] / fixed, marker="o", ms=5, lw=2, color=MODE_COLOUR[mode],
                label=MODE_LABEL[mode])
    ax.axhline(1.0, color=INK, lw=1, ls="--")
    ax.text(RATIOS[-1], 1.08, "no cost", fontsize=7.5, color=INK, ha="right")
    worst = float(np.max(sd[:, list(grid.temperature_modes).index("free")] / fixed))
    ax.annotate(f"×{worst:.0f}", (RATIOS[0], worst), fontsize=9, color=ORANGE,
                xytext=(6, -2), textcoords="offset points", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "(b) sd(σ) paid for fitting the temperature", "kT / σ",
          "× the uncertainty with T fixed")
    ax.legend(fontsize=7, frameon=False, labelcolor=MUTED, loc="upper right")


def run_estimators():
    """(c) the bound and the shipped fitter's sandwich, side by side."""
    energy = np.arange(-1.0, 1.0 + 1e-9, 0.01)
    background = constant_background(energy, 50.0)
    rows = []
    for fwhm, temperature in CONDITIONS:
        sigma = fwhm / FWHM_PER_SIGMA
        edge = FermiEdge(ef=0.0, amplitude=2000.0, sigma=sigma, temperature=temperature,
                         dos_c1=0.3)
        report = assess_edge_identifiability(
            energy, edge, background, temperature_mode="fixed_temperature",
            estimator_weights="unit")
        resolution = report.resolution
        rows.append((fwhm, temperature, resolution.sd_sigma_bound,
                     resolution.sd_estimator / (2.0 * sigma),
                     resolution.temperature_sensitivity))
    return rows


def panel_estimators(ax, rows):
    x = np.arange(len(rows))
    bound = np.array([r[2] for r in rows]) * 1e3
    sandwich = np.array([r[3] for r in rows]) * 1e3
    ax.bar(x - 0.19, bound, 0.36, color=BLUE, label="Cramér–Rao bound", zorder=3)
    ax.bar(x + 0.19, sandwich, 0.36, color=ORANGE, zorder=3,
           label="the shipped fitter (unit weights)")
    for i, (lo, hi) in enumerate(zip(bound, sandwich)):
        ax.text(i + 0.19, hi * 1.03, f"×{hi / lo:.2f}", ha="center", fontsize=7.5,
                color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{1e3 * r[0]:.0f} meV\n{r[1]:.0f} K" for r in rows], fontsize=7.5)
    ax.set_ylim(0.0, 1.35 * max(sandwich.max(), bound.max()))
    style(ax, "(c) the bound is not this fitter's error bar", None, "sd(σ), meV")
    ax.legend(fontsize=7, frameon=False, labelcolor=MUTED, loc="upper center", ncol=2)


def run_dos_forms():
    """(d) one simulated spectrum, two assumptions about the DOS at E_F.

    Poisson counts from a fixed seed, so the statistical error bar the
    systematic is compared against is a real one and not zero.
    """
    energy = np.arange(-0.6, 0.6 + 1e-9, 0.01)
    sigma, temperature = 0.04828, 560.3          # kT/sigma = 1 at kappa_2 = (0.1 eV)^2
    mean = fermi_edge(energy, ef=0.0, amplitude=2000.0, fwhm_g=FWHM_PER_SIGMA * sigma,
                      temperature=temperature, dos_c1=1.5, bg_const=50.0)
    spectrum = np.random.default_rng(9).poisson(mean).astype(float)
    comparison = compare_dos_forms(
        energy, spectrum, convention="BE", dos="linear", background="constant",
        temperature=temperature, resolution=FWHM_PER_SIGMA * sigma, ef_init=0.0)
    return energy, spectrum, sigma, comparison


def panel_dos_forms(ax, sigma, comparison):
    forms = comparison.forms
    values = np.array([comparison.values["resolution"][f] for f in forms]) / FWHM_PER_SIGMA
    errors = np.array([comparison.fits[f].resolution_err for f in forms]) / FWHM_PER_SIGMA
    x = np.arange(len(forms))
    ax.errorbar(x, 1e3 * values, yerr=1e3 * errors, fmt="o", ms=7, capsize=5, lw=2,
                color=BLUE, zorder=3, label="fitted σ ± its statistical error")
    ax.set_ylim(min(1e3 * (values - errors).min(), 1e3 * sigma) - 3.0,
                1e3 * (values + errors).max() + 4.0)
    ax.axhline(1e3 * sigma, color=MUTED, lw=1, ls="--")
    ax.text(-0.42, 1e3 * sigma, "truth", fontsize=7.5, color=MUTED, va="bottom")
    lo, hi = 1e3 * values.min(), 1e3 * values.max()
    ax.annotate("", xy=(0.5, lo), xytext=(0.5, hi),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.6))
    ax.text(0.56, 0.5 * (lo + hi), f"systematic\n{hi - lo:.1f} meV", fontsize=8,
            color=ORANGE, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(["DOS kinked at E_F\n(the default)", "DOS smooth through E_F"],
                       fontsize=7.5)
    ax.set_xlim(-0.5, len(forms) - 0.1)
    style(ax, "(d) here the DOS assumption moves σ further than the noise does",
          None, "fitted σ, meV")
    ax.legend(fontsize=7, frameon=False, labelcolor=MUTED, loc="lower left")


def main():
    grid, scan = run_scan()
    rows = run_estimators()
    energy, spectrum, sigma, comparison = run_dos_forms()

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), facecolor=SURFACE)
    panel_split(axes[0, 0], grid, scan)
    panel_inflation(axes[0, 1], grid, scan)
    panel_estimators(axes[1, 0], rows)
    panel_dos_forms(axes[1, 1], sigma, comparison)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.suptitle("Fermi-edge resolution: model bounds and one fitted spectrum, "
                 "not measurements", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT_DIR / f"{STEM}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    np.savez(OUT_DIR / f"{STEM}.npz", ratios=np.asarray(RATIOS),
             modes=np.asarray(grid.temperature_modes), sd_share=scan["sd_share"],
             sd_kappa2=scan["sd_kappa2"], sd_sigma=scan["sd_sigma"],
             estimators=np.asarray(rows), energy=energy, spectrum=spectrum)

    print("(a) relative sd, window ±6 edge widths, background estimated")
    print("  kT/sigma   total width      share of it   verdict (T fitted)")
    share = scan["sd_share"][:, 0, 0, 0, 0, list(grid.temperature_modes).index("free"), 0]
    total = scan["sd_kappa2"][:, 0, 0, 0, 0, list(grid.temperature_modes).index("free"), 0]
    for i, ratio in enumerate(RATIOS):
        verdict = "separable" if share[i] <= grid.thresholds.separation_sd else "NOT separable"
        print(f"  {ratio:8.2f}   {total[i]:11.4f}   {share[i]:14.4f}   {verdict}")

    print("\n(b) sd(sigma) with the temperature fitted, over sd(sigma) with it fixed")
    sd = scan["sd_sigma"][:, 0, 0, 0, 0, :, 0]
    fixed = sd[:, list(grid.temperature_modes).index("fixed_temperature")]
    for i, ratio in enumerate(RATIOS):
        free = sd[:, list(grid.temperature_modes).index("free")][i]
        print(f"  kT/sigma = {ratio:4.1f}:  {fixed[i]:8.4f} -> {free:8.4f}   "
              f"x{free / fixed[i]:6.1f}")

    print("\n(c) which estimator the standard deviation belongs to")
    print("  FWHM      T     sd(sigma) bound   unit-weight fit   ratio   dsigma/dT")
    for fwhm, temperature, bound, sandwich, sensitivity in rows:
        print(f"  {1e3 * fwhm:5.0f} meV {temperature:5.0f} K   {1e3 * bound:11.4f} meV   "
              f"{1e3 * sandwich:11.4f} meV   {sandwich / bound:5.2f}   "
              f"{1e6 * sensitivity:8.1f} ueV/K")

    print("\n(d) the same spectrum, two assumptions about the DOS at E_F")
    for form in comparison.forms:
        fit = comparison.fits[form]
        print(f"  {form:11s}  sigma = {1e3 * fit.resolution / FWHM_PER_SIGMA:7.3f} "
              f"+- {1e3 * fit.resolution_err / FWHM_PER_SIGMA:.3f} meV (statistical)   "
              f"E_F = {1e3 * fit.ef:+7.3f} meV")
    print(f"  spread between the forms: {1e3 * comparison.spread['resolution'] / FWHM_PER_SIGMA:.3f}"
          f" meV on sigma, {1e3 * comparison.spread['ef']:.3f} meV on E_F -- a systematic,")
    print("  to be reported beside the statistical error and never added to it in quadrature.")
    print(f"\nsaved -> {OUT_DIR / (STEM + '.png')}, {OUT_DIR / (STEM + '.npz')}")


if __name__ == "__main__":
    main()
