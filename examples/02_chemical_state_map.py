"""Batch-fit an imaging-XPS map and plot chemical-state images.

Simulates a 64x64 pixel Si 2p map of a patterned oxide film (SiO2
islands on metallic Si), fits all 4,096 spectra in one call with
:class:`~toyomacro.fitting.FastVoigtFitter`, and renders ground-truth
vs. recovered chemical-state maps.

The high-level batch API fits each chemical state as a single Voigt
component; spin-orbit doublet models are available through the
low-level ``toyomacro.voigtfit`` multipeak API. Runs on the pure-numpy
backend; the ``mlx`` extra moves the fit onto the Apple Silicon GPU,
which matters for larger maps rather than for this 4,096-spectrum one.

Run from the repository root::

    python examples/02_chemical_state_map.py

Output: ``examples/output/02_chemical_state_map.png``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wofz

from toyomacro.fitting import FastFitConfig, FastVoigtFitter

OUT_DIR = Path(__file__).parent / "output"

# Two Si 2p chemical states (singlet approximation)
STATES = {
    "Si0 (metal)": {"center": 99.3, "sigma": 0.45},
    "Si4+ (SiO2)": {"center": 103.4, "sigma": 0.60},
}
GAMMA = 0.10  # Lorentzian width shared by both states
ENERGY = np.linspace(96.0, 107.0, 111)
MAP_SIZE = 64
NOISE_SIGMA = 0.02  # counts; peak heights are ~0.3-0.75 for unit area


def voigt_profile(energy, center, sigma, gamma):
    """Unit-area Voigt profile.

    FastVoigtFitter returns amplitudes as coefficients of unit-area
    Voigt profiles (i.e. integrated peak areas — the quantity that is
    proportional to concentration in XPS), so the ground truth is
    generated in the same convention.
    """
    z = ((energy - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def make_oxide_pattern(n: int, seed: int = 42):
    """Ground-truth oxide fraction map: two SiO2 islands on Si."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    blob1 = np.exp(-(((x - 0.35) ** 2 + (y - 0.40) ** 2) / 0.018))
    blob2 = np.exp(-(((x - 0.72) ** 2 + (y - 0.68) ** 2) / 0.045))
    frac = np.clip(blob1 + blob2, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    frac += rng.normal(0.0, 0.01, frac.shape)  # slight film roughness
    return np.clip(frac, 0.0, 1.0)


def main():
    # --- Ground truth: per-pixel amplitudes for both states ---
    oxide_frac = make_oxide_pattern(MAP_SIZE)
    amp_si0_gt = (1.0 - oxide_frac).ravel()
    amp_si4_gt = oxide_frac.ravel()
    n_px = amp_si0_gt.size

    # --- Generate one spectrum per pixel ---
    rng = np.random.default_rng(0)
    basis = np.stack([
        voigt_profile(ENERGY, p["center"], p["sigma"], GAMMA)
        for p in STATES.values()
    ])  # (n_states, n_energy)
    spectra = (amp_si0_gt[:, None] * basis[0]
               + amp_si4_gt[:, None] * basis[1]).astype(np.float32)
    spectra += rng.normal(0.0, NOISE_SIGMA, spectra.shape).astype(np.float32)

    # --- Batch fit: all pixels in one call ---
    fitter = FastVoigtFitter(FastFitConfig(mode="fast"))
    result = fitter.fit_batch_rowmajor(
        spectra,
        ENERGY,
        centers=np.array([p["center"] for p in STATES.values()]),
        sigmas=np.array([p["sigma"] for p in STATES.values()]),
        gamma=GAMMA,
        element="Si",
        orbital="2p",
    )
    print(f"fitted {result.n_spectra} spectra x {result.n_components} states "
          f"in {result.elapsed_ms:.0f} ms "
          f"({result.spectra_per_second / 1e3:.0f}K spectra/s)")

    # result.amplitudes: (n_states, n_spectra) -> maps
    amp_si0 = result.amplitudes[0].reshape(MAP_SIZE, MAP_SIZE)
    amp_si4 = result.amplitudes[1].reshape(MAP_SIZE, MAP_SIZE)
    frac_fit = amp_si4 / np.clip(amp_si0 + amp_si4, 1e-6, None)

    amp_rmse = float(np.sqrt(np.mean(
        (result.amplitudes.T - np.stack([amp_si0_gt, amp_si4_gt], axis=1)) ** 2
    )))
    rmse = float(np.sqrt(np.mean((frac_fit - oxide_frac) ** 2)))
    corr = float(np.corrcoef(frac_fit.ravel(), oxide_frac.ravel())[0, 1])
    print(f"area recovery RMSE={amp_rmse:.4f}  "
          f"oxide fraction map: RMSE={rmse:.4f}  corr={corr:.4f}")

    # --- Plot: GT vs fitted maps + one example pixel spectrum ---
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, img, title in [
        (axes[0], oxide_frac, "oxide fraction (ground truth)"),
        (axes[1], frac_fit, "oxide fraction (fitted)"),
        (axes[2], amp_si0, "Si0 amplitude (fitted)"),
    ]:
        im = ax.imshow(img, origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    px = (MAP_SIZE // 2) * MAP_SIZE + int(0.35 * MAP_SIZE)  # inside island 1
    axes[3].plot(ENERGY, spectra[px], ".", ms=3, color="0.4", label="pixel data")
    model = (result.amplitudes[0][px] * basis[0]
             + result.amplitudes[1][px] * basis[1])
    axes[3].plot(ENERGY, model, "-", color="tab:red", label="fit")
    axes[3].set_title(f"example pixel (oxide frac {frac_fit.ravel()[px]:.2f})",
                      fontsize=9)
    axes[3].set_xlabel("Binding energy (eV)")
    axes[3].invert_xaxis()
    axes[3].legend(fontsize=8)
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "02_chemical_state_map.png"
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
