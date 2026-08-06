"""XPS quantification: from peak areas to composition.

Demonstrates the bundled reference data in ``toyomacro.data``:

- ``BindingEnergy``  — core-level binding energies
- ``CrossSection``   — photoionization cross-sections (package default
                       table: Yeh & Lindau 1985)
- ``IMFP``           — TPP-2M inelastic mean free path

A SiO2 sample (true O/Si = 2.0) is forward-modelled: the "measured"
Si 2p and O 1s peak areas are proportional to n * sigma(hv) * lambda(KE).
The composition is then recovered at increasing correction levels,

- Level 0:   raw area ratio
- Level 1:   / sigma           (cross-section)
- Level 1.5: / (sigma*lambda)  (+ IMFP)

for both a lab source (Al Ka) and a HAXPES source (Ga Ka).

sigma*lambda is an *intrinsic* sensitivity, **not** a complete AMRSF and
not an instrument-specific sensitivity factor. It omits analyzer
transmission, detector response, elastic-scattering / EAL corrections,
the photoelectron angular distribution, x-ray polarization and the
source/analyzer geometry — hence "Level 1.5" rather than a level number
implying the chain is finished. An analyzer transmission adapter exists
(``toyomacro.data.transmission``) but the vendor curves it reads are
instrument-specific and are not bundled; it is deliberately not used
here. If you do use it, apply T either to the spectrum or to the
sensitivity, never to both.

Note on the IMFP. Tanuma, Powell & Penn fit TPP-2M over 50-2000 eV and
state it should not be used above 2000 eV, so the ~8-9 keV Ga Ka values
here are extrapolations, and in the non-relativistic form at that (see
``docs/API.md``). This matters more than it may look: sigma cancels
identically once both lines are divided by it, so **Level 1 is nothing
but 2 x lambda_O/lambda_Si** — the IMFP is its sole driver, not a minor
correction. Level 1.5 then returns 2.0 because lambda cancels too, up to
the 1% per-line noise injected below (which does not cancel: the printed
values are 1.973 and 2.004, not 2.000); that is arithmetic, not evidence
that the IMFPs are right. The
figure shows how the correction *chain* behaves, not that HAXPES IMFPs
from TPP-2M are validated.

The forward model is deliberately circular — areas are built from
n * sigma * lambda and then divided by the same sigma * lambda — so
Level 1.5 recovering the input stoichiometry demonstrates that the
bookkeeping is self-consistent, and nothing about physical accuracy.

Run from the repository root::

    python examples/03_quantification.py

Output: ``examples/output/03_quantification.png``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toyomacro.data import IMFP, BindingEnergy, CrossSection

OUT_DIR = Path(__file__).parent / "output"

SOURCES = {"Al Ka": 1486.6, "Ga Ka": 9251.7}
TRUE_O_SI = 2.0  # SiO2 stoichiometry
COMPOUND = "SiO2"  # IMFP matrix


def sensitivity(element: str, orbital: str, hv: float):
    """Per-line quantities: BE, KE, sigma, lambda."""
    be = BindingEnergy.lookup(element, orbital)
    ke = hv - be
    sigma = CrossSection.lookup(element, orbital, hv)
    lam = IMFP.tpp2m(kinetic_energy=ke, compound=COMPOUND)
    return {"be": be, "ke": ke, "sigma": sigma, "lambda": lam}


def main():
    rng = np.random.default_rng(42)
    results = {}  # (source, level) -> O/Si ratio

    for source, hv in SOURCES.items():
        si = sensitivity("Si", "2p", hv)
        o = sensitivity("O", "1s", hv)

        print(f"\n--- {source} (hv = {hv:.1f} eV) ---")
        for name, s in [("Si 2p", si), ("O 1s", o)]:
            print(f"{name}: BE={s['be']:7.1f} eV  KE={s['ke']:7.1f} eV  "
                  f"sigma={s['sigma']:.4g}  lambda={s['lambda']:.3f} nm")

        # Forward model: area = n * sigma * lambda (+ 1% measurement noise)
        area_si = 1.0 * si["sigma"] * si["lambda"] * rng.normal(1.0, 0.01)
        area_o = TRUE_O_SI * o["sigma"] * o["lambda"] * rng.normal(1.0, 0.01)

        # Recover O/Si at each correction level
        results[(source, "Level 0\n(raw area)")] = area_o / area_si
        results[(source, "Level 1\n(/ sigma)")] = (
            (area_o / o["sigma"]) / (area_si / si["sigma"]))
        results[(source, "Level 1.5\n(/ sigma*lambda)")] = (
            (area_o / (o["sigma"] * o["lambda"]))
            / (area_si / (si["sigma"] * si["lambda"])))

    levels = ["Level 0\n(raw area)", "Level 1\n(/ sigma)",
              "Level 1.5\n(/ sigma*lambda)"]
    print(f"\nO/Si atomic ratio (true = {TRUE_O_SI}):")
    print(f"{'':14}" + "".join(f"{lv.splitlines()[0]:>12}" for lv in levels))
    for source in SOURCES:
        row = [results[(source, lv)] for lv in levels]
        print(f"{source:14}" + "".join(f"{v:12.3f}" for v in row))

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(levels))
    width = 0.35
    for i, source in enumerate(SOURCES):
        vals = [results[(source, lv)] for lv in levels]
        ax.bar(x + (i - 0.5) * width, vals, width, label=source)
    ax.axhline(TRUE_O_SI, color="k", ls="--", lw=1, label="true O/Si = 2.0")
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel("recovered O/Si atomic ratio")
    ax.set_title(
        "SiO2 quantification vs. correction level\n"
        "synthetic: areas built from the same sigma and lambda used to "
        "correct them",
        fontsize=10,
    )
    ax.text(
        0.5, -0.16,
        "Level 1 is 2 x lambda_O/lambda_Si (sigma cancels); Level 1.5 "
        "returns 2.0 by construction, up to the 1% injected noise.\n"
        "Ga Ka lambda values are TPP-2M extrapolated beyond its "
        "50-2000 eV fitted range.",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
        color="0.35",
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "03_quantification.png"
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
