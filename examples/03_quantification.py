"""XPS quantification: from peak areas to composition.

Demonstrates the bundled reference data in ``toyomacro.data``:

- ``BindingEnergy``  — core-level binding energies
- ``CrossSection``   — Scofield photoionization cross-sections
- ``IMFP``           — TPP-2M inelastic mean free path

A SiO2 sample (true O/Si = 2.0) is forward-modelled: the "measured"
Si 2p and O 1s peak areas are proportional to n * sigma(hv) * lambda(KE).
The composition is then recovered at increasing correction levels,

- Level 0:   raw area ratio
- Level 1:   / sigma           (cross-section)
- Level 1.5: / (sigma*lambda)  (+ IMFP; AMRSF-equivalent)

for both a lab source (Al Ka) and a HAXPES source (Ga Ka). The
analyzer transmission correction T(KE/Ep) is also available
(``toyomacro.data.transmission``) but requires vendor data files that
are not bundled.

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
    ax.set_title("SiO2 quantification vs. correction level")
    ax.legend(fontsize=9)
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "03_quantification.png"
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
