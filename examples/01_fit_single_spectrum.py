"""Fit a single XPS spectrum with AutoFitter and plot the decomposition.

Generates a synthetic Si 2p spectrum (metallic Si + SiO2, each a
spin-orbit doublet) with a linear background and Gaussian noise, fits it
with :class:`~toyomacro.fitting.AutoFitter` (which auto-selects the
``Si2p_oxide`` template from ``spectrum.element``), and saves a peak
decomposition plot.

Run from the repository root::

    python examples/01_fit_single_spectrum.py

Output: ``examples/output/01_si2p_decomposition.png``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toyomacro.core import Spectrum
from toyomacro.fitting import AutoFitter
from toyomacro.lineshape import create_lineshape

OUT_DIR = Path(__file__).parent / "output"

# Si 2p physical constants
SO_SPLIT = 0.608  # eV, 2p spin-orbit splitting
BRANCH_RATIO = 2.0  # I(2p3/2) / I(2p1/2)


def doublet(energy, center, area, fwhm_g, fwhm_l, voigt):
    """Voigt spin-orbit doublet: main peak + partner / branch_ratio."""
    main = voigt.evaluate(energy, center=center, amplitude=area, fwhm_g=fwhm_g, fwhm_l=fwhm_l)
    partner = voigt.evaluate(
        energy, center=center + SO_SPLIT, amplitude=area / BRANCH_RATIO,
        fwhm_g=fwhm_g, fwhm_l=fwhm_l,
    )
    return main + partner


def make_synthetic_si2p(seed: int = 42):
    """Synthetic Si 2p: Si0 (99.3 eV) + Si4+ (103.4 eV) doublets."""
    rng = np.random.default_rng(seed)
    energy = np.linspace(96.0, 108.0, 401)
    voigt = create_lineshape("voigt")

    signal = doublet(energy, 99.3, 6000.0, 0.45 * 2.355, 0.10 * 2.0, voigt)
    signal += doublet(energy, 103.4, 4000.0, 0.60 * 2.355, 0.10 * 2.0, voigt)

    background = 300.0 + 25.0 * (energy - energy[0])  # linear slope
    noise = rng.normal(0.0, 15.0, energy.shape)
    return energy, signal + background + noise


def component_curve(comp, energy, voigt):
    """Evaluate one fitted PeakComponent as a single Voigt peak.

    In FittingResult the spin-orbit partner is stored as its own
    component (the main peak carries ``so_split``/``branch_ratio`` as
    metadata), so each component is rendered as a singlet.

    Note: PeakComponent follows the MATLAB ``fitpara`` column layout —
    ``comp.area`` holds the peak height (col 0) and ``comp.height``
    holds the integrated area (col 8).
    """
    return voigt.evaluate(
        energy, center=comp.center, amplitude=comp.height,
        fwhm_g=comp.fwhm_g, fwhm_l=comp.fwhm_l,
    )


def group_chemical_states(components):
    """Group components into chemical states (main + SO partner).

    Components are ordered main, partner, main, partner, ... — a main
    peak has ``so_split > 0`` and is immediately followed by its partner.
    Returns a list of component lists, one per chemical state.
    """
    states, i = [], 0
    while i < len(components):
        comp = components[i]
        if comp.so_split > 0 and i + 1 < len(components):
            states.append([comp, components[i + 1]])
            i += 2
        else:
            states.append([comp])
            i += 1
    return states


def main():
    energy, intensity = make_synthetic_si2p()
    spectrum = Spectrum(
        energy=energy, intensity=intensity, energy_type="BE", element="Si2p",
    )

    # element="Si2p" matches the built-in Si2p_oxide template (5 oxidation
    # states); AutoFitter routes to the template path automatically.
    result = AutoFitter().fit(spectrum)
    fit = result.fitting_result

    states = group_chemical_states(fit.components)

    print(f"success={result.success}  R^2={result.r_squared:.5f}  "
          f"n_components={fit.n_components}  n_states={len(states)}")
    print(f"{'state @ eV':>12} {'total area':>10} {'FWHM_G':>7} {'FWHM_L':>7}")
    for comps in states:
        main = comps[0]
        total_area = sum(c.height for c in comps)  # height = integrated area
        print(f"{main.center:12.2f} {total_area:10.0f} "
              f"{main.fwhm_g:7.3f} {main.fwhm_l:7.3f}")

    # --- Plot ---
    voigt = create_lineshape("voigt")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(energy, intensity, ".", ms=3, color="0.4", label="measured")
    ax.plot(fit.energy, fit.background_curve, "--", color="tab:orange",
            label="background")
    for i, comps in enumerate(states):
        curve = sum(component_curve(c, energy, voigt) for c in comps)
        if curve.max() < 0.005 * intensity.max():
            continue  # skip states the fit pruned to ~zero
        ax.fill_between(energy, fit.background_curve,
                        curve + fit.background_curve, alpha=0.35,
                        label=f"state @ {comps[0].center:.1f} eV")
    ax.plot(fit.energy, fit.fitted_intensity, "-", color="tab:red", lw=1.2,
            label="envelope")
    ax.set_xlabel("Binding energy (eV)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title(f"Si 2p AutoFitter decomposition (R$^2$={result.r_squared:.4f})")
    ax.invert_xaxis()  # XPS convention: BE decreases to the right
    ax.legend(fontsize=8)
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "01_si2p_decomposition.png"
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
