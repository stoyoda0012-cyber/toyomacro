"""
Example 11: which combinations of peak parameters does a spectrum not pin down?
==============================================================================

``identifiability_modes`` (experimental) reports the least determined
*directions* of the peak parameters, with the background estimated from
the same spectrum, and what a move along each looks like in the
spectrum. Two configurations of one Voigt peak on a linear background:

- a wide window (+-10 gamma) at sigma = gamma, where every mode is
  determined to under 1 % of the peak's own scales, and
- a window of +-gamma at sigma/gamma = 1/3, where the softest mode is
  known worse than the peak's own scales: the window holds too little of
  the tails to tell a larger peak, with more of its width Lorentzian, on
  a lower flat background from the original (the background level is
  what follows that mode).

The lower panels show each mode's signature: the change of the expected
counts for a one-SD move along it, background re-estimated, divided by
the counting noise sqrt(mu) in each channel. Every curve has the same
total size, sum of squares 1 -- one unit of noise spread over the window,
which is why the data cannot tell it apart. What differs between modes is
the shape, and the weights printed below say which parameters move
together to produce it.

Everything here is computed from a model (Poisson Fisher information at
stated parameters). It is not a measurement, not a confidence interval
and not a statement about any fit; the directions depend on the scales
the parameters are measured in, which is a convention -- see the
docstring of ``identifiability_modes``.

Output: examples/output/11_identifiability_modes.png (not tracked).
Runs in about a second, NumPy only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from toyomacro.voigtfit.identifiability import (
    VoigtPeak,
    identifiability_modes,
    linear_background,
)

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)
GAMMA = 0.3

CASES = {
    "wide window, sigma = gamma": (10.0, GAMMA),
    "+-gamma window, sigma = gamma/3": (1.0, GAMMA / 3.0),
}


def describe(name, report):
    print(f"\n{name}")
    print("  mode  rel. SD   status             largest weights          background follows")
    for mode in report.modes:
        top = sorted(mode.weights.items(), key=lambda kv: -abs(kv[1]))[:3]
        weights = ", ".join(f"{n} {w:+.2f}" for n, w in top)
        shift = ", ".join(f"{n} {v:+.3g}" for n, v in mode.background_shift.items())
        print(f"  {mode.index + 1:4d}  {mode.relative_sd:8.4f}  {mode.status:17s}  "
              f"{weights:30s}  {shift}")


fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex="col",
                         gridspec_kw={"height_ratios": [1.2, 1]})
for column, (name, (half_width, sigma)) in enumerate(CASES.items()):
    energy = np.arange(-half_width * GAMMA, half_width * GAMMA + 1e-9, 0.02)
    peak = VoigtPeak(2.0e4, 0.0, sigma, GAMMA)
    report = identifiability_modes(energy, [peak], linear_background(energy, 300.0, 10.0))
    describe(name, report)

    mu = report.fisher.expected_counts
    axes[0, column].plot(energy, mu, color="0.2")
    axes[0, column].set_title(name, fontsize=10)
    axes[0, column].set_ylabel("expected counts")
    for mode in report.modes:
        axes[1, column].plot(
            energy, mode.signature / np.sqrt(mu),
            label=f"mode {mode.index + 1}: rel. SD {mode.relative_sd:.3g}",
        )
    axes[1, column].axhline(0.0, color="0.7", lw=0.8)
    axes[1, column].set_xlabel("energy (eV)")
    axes[1, column].set_ylabel("signature / sqrt(mu)")
    axes[1, column].legend(fontsize=8)

fig.tight_layout()
path = OUT_DIR / "11_identifiability_modes.png"
fig.savefig(path, dpi=130)
print(f"\nwrote {path.name}")
