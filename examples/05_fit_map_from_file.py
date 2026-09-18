"""Fit a chemical-state map read from a file.

Example 02 builds its map in memory. This one starts from a file on
disk, which is the step between the examples and your own measurement:

1. read the spectra — an HDF5 ``specdata`` map (the batch pipeline's
   format, see ``examples/data/README.md``) or any single file the
   readers accept (Scienta ``.pxt`` / SES ``.txt``, VAMAS ``.vms``,
   ``.npl``), whose columns — angles, slices — become the pixels;
2. put the energy axis in ascending order and remove a linear
   background drawn between the averaged ends of each spectrum;
3. batch-fit every spectrum with one Voigt component per chemical
   state (:class:`~toyomacro.fitting.FastVoigtFitter`);
4. save per-state area maps and the fraction of each state.

With no argument it fits the synthetic 8x8 Si 2p map written by
``examples/data/make_example_data.py`` (generated here if missing) and
compares the result with that file's known ground truth.

Two simplifications to keep in mind before trusting numbers from your
own data. Each chemical state is fitted as a *single* Voigt, so a
spin-orbit doublet (Si 2p: 0.61 eV splitting) is approximated by one
broadened line; areas are biased by several percent — the synthetic run
prints by how much — while area *fractions* are much less affected.
And the linear background is the simplest choice, not a quantitative
one: for composition, the background model (Shirley, Tougaard) matters
and belongs to your analysis, not to this example.

Run from the repository root::

    python examples/05_fit_map_from_file.py
    python examples/05_fit_map_from_file.py my_map.h5 --shape 64 64 \\
        --state Si0:99.5:0.45 --state Si4+:103.6:0.75

``--state NAME:CENTER:FWHM_G`` gives each chemical state its nominal
center and Gaussian FWHM (eV, on the file's energy scale); the fitter
refines both per pixel. ``--shape NY NX`` arranges the spectra into an
image in row-major order; without it the result is plotted as a series.

Output: ``examples/output/05_map_from_file.png``
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toyomacro.fitting import FastFitConfig, FastVoigtFitter
from toyomacro.voigtfit import voigt_profile

HERE = Path(__file__).parent
OUT_DIR = HERE / "output"
DEFAULT_MAP = HERE / "data" / "si2p_map_8x8.h5"

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

# Nominal states for the default Si 2p map: the area-weighted centroid of
# each 2p3/2 + 2p1/2 doublet (2p3/2 + 0.61 eV x 1/3), and its 2p3/2 FWHM.
DEFAULT_STATES = ("Si0:99.50:0.42", "Si4+:103.60:0.72")


def parse_state(text: str) -> tuple[str, float, float]:
    """``"NAME:CENTER:FWHM_G"`` -> (name, center_eV, fwhm_g_eV)."""
    name, center, fwhm = text.rsplit(":", 2)
    return name, float(center), float(fwhm)


def load_spectra(
    path: Path, region: int = 0, photon_energy: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Read ``(energy, spectra, energy_scale)`` from a map or reader file.

    ``spectra`` is ``(n_spectra, n_energy)``. HDF5 files are read as the
    ``specdata`` layout and their energy axis is used as stored. Other
    files go through the format readers (``region`` picks one region of a
    multi-region file); every column of the returned array (angle, slice)
    becomes one spectrum and sweeps are summed. A kinetic-energy axis is
    converted to binding energy with ``photon_energy``, or with the
    photon energy the file states; files that record none (or 0) are left
    in kinetic energy.
    """
    if path.suffix.lower() in (".h5", ".hdf5"):
        from toyomacro.voigtfit import read_spectra

        data = read_spectra(path)
        return (np.asarray(data.energy, dtype=np.float64),
                np.asarray(data.spectra, dtype=np.float32), "as stored")

    from toyomacro.io.readers import create_reader

    raw = create_reader(path).read(region_index=region)
    block = np.asarray(raw.specdata, dtype=np.float64)
    if block.ndim == 3:
        block = block.sum(axis=2)  # repeated sweeps of the same spectra
    spectra = block.reshape(block.shape[0], -1).T  # (n_columns, n_energy)
    energy = np.asarray(raw.energy, dtype=np.float64)

    meta = raw.metadata
    if meta.energy_scale.lower().startswith("kinetic"):
        hv = photon_energy or meta.excitation_energy
        if hv is None or hv <= 0:  # some files store 0 for "not recorded"
            return energy, spectra.astype(np.float32), "KE (no photon energy)"
        return hv - energy, spectra.astype(np.float32), f"BE = {hv:g} eV - KE"
    return energy, spectra.astype(np.float32), meta.energy_scale or "as stored"


def subtract_linear_background(energy, spectra, n_end=5):
    """Sort to ascending energy and remove a straight line joining the ends.

    Each end level is the mean of ``n_end`` channels, so the line is not
    pinned to a single noisy point, and is placed at the mean energy of
    those channels, so a sloped background is removed without an offset.
    """
    order = np.argsort(energy)
    energy = energy[order]
    spectra = spectra[:, order].astype(np.float64)
    left = spectra[:, :n_end].mean(axis=1, keepdims=True)
    right = spectra[:, -n_end:].mean(axis=1, keepdims=True)
    e_left, e_right = energy[:n_end].mean(), energy[-n_end:].mean()
    t = (energy - e_left) / (e_right - e_left)
    background = left + (right - left) * t
    return energy, np.ascontiguousarray(spectra - background, dtype=np.float32), background


def synthetic_truth(n_pixels: int) -> np.ndarray | None:
    """Per-state areas of the default map, ``(n_states, n_pixels)``."""
    gen = runpy.run_path(str(HERE / "data" / "make_example_data.py"))
    n = int(round(np.sqrt(n_pixels)))
    if n * n != n_pixels:
        return None
    frac = np.tile(np.arange(n) / (n - 1), n)  # oxide ramp along x
    scale = {"Si0": 1.0 - 0.9 * frac, "Si4+": 0.1 + 0.9 * frac}
    doublet = 1.0 + gen["BRANCH"]
    return np.stack([area * scale[name] * doublet
                     for name, _c, area, _w in gen["STATES"]])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", nargs="?", type=Path, default=None,
                   help="HDF5 specdata map or a reader-supported file "
                        "(default: the synthetic Si 2p 8x8 map)")
    p.add_argument("--state", action="append", default=None,
                   metavar="NAME:CENTER:FWHM_G",
                   help="one chemical state; repeat for each (default: Si 2p "
                        "metal and oxide)")
    p.add_argument("--gamma", type=float, default=0.05,
                   help="Lorentzian half width, eV, shared by all states")
    p.add_argument("--shape", type=int, nargs=2, metavar=("NY", "NX"),
                   help="arrange spectra as an NY x NX image (row-major)")
    p.add_argument("--end-channels", type=int, default=5,
                   help="channels averaged at each end for the background")
    p.add_argument("--region", type=int, default=0,
                   help="region of a multi-region reader file (default 0)")
    p.add_argument("--photon-energy", type=float, default=None,
                   help="photon energy, eV, to convert a kinetic-energy axis "
                        "(overrides the file's value)")
    p.add_argument("--min-area-frac", type=float, default=0.05,
                   help="leave fractions blank where the total fitted area is "
                        "below this share of the largest (default 0.05)")
    args = p.parse_args(argv)

    path = args.path
    if path is None:
        path = DEFAULT_MAP
        if not path.exists():
            runpy.run_path(str(HERE / "data" / "make_example_data.py"),
                           run_name="__main__")
        if args.shape is None:
            args.shape = (8, 8)
    states = [parse_state(s) for s in (args.state or DEFAULT_STATES)]

    energy, spectra, scale = load_spectra(path, args.region, args.photon_energy)
    energy, signal, _bg = subtract_linear_background(
        energy, spectra, args.end_channels)
    n_px = signal.shape[0]
    print(f"read {n_px} spectra x {energy.size} channels from {path.name} "
          f"({energy[0]:.2f}-{energy[-1]:.2f} eV, {scale})")

    names = [s[0] for s in states]
    centers = np.array([s[1] for s in states])
    outside = [n for n, c in zip(names, centers) if not energy[0] < c < energy[-1]]
    if outside:
        raise SystemExit(
            f"state center(s) {', '.join(outside)} outside the energy window "
            f"{energy[0]:.2f}-{energy[-1]:.2f} eV ({scale}); check --state, "
            "--region and --photon-energy.")
    sigmas = np.array([s[2] for s in states]) / FWHM_PER_SIGMA
    fitter = FastVoigtFitter(FastFitConfig(mode="fast"))
    result = fitter.fit_batch_rowmajor(
        signal, energy, centers, sigmas, args.gamma)
    print(f"fitted {n_px} spectra x {len(states)} states in "
          f"{result.elapsed_ms:.0f} ms")

    areas = np.clip(result.amplitudes, 0.0, None)  # (n_states, n_px)
    total = areas.sum(axis=0)
    weak = total < args.min_area_frac * total.max()
    fractions = areas / np.where(weak, np.nan, total)
    if weak.any():
        print(f"{weak.sum()} of {n_px} spectra below {args.min_area_frac:.0%} of "
              "the largest total area: fractions left blank")
    for i, name in enumerate(names):
        print(f"  {name:>6}: center {np.median(result.centers[i]):7.2f} eV "
              f"(nominal {centers[i]:.2f}), mean fraction "
              f"{np.nanmean(fractions[i]):.3f}")

    if args.path is None:
        truth = synthetic_truth(n_px)
        if truth is not None:
            ratio = np.median(areas / truth, axis=1)
            true_frac = truth / truth.sum(axis=0)
            rmse = np.sqrt(np.nanmean((fractions - true_frac) ** 2))
            print("vs ground truth: area recovered " + ", ".join(
                f"{n} {r:.1%}" for n, r in zip(names, ratio))
                + f"; fraction RMSE {rmse:.4f}")

    # --- Plot: one fraction panel per state + one example spectrum ---
    n_states = len(states)
    fig, axes = plt.subplots(1, n_states + 1, figsize=(4 * (n_states + 1), 3.6))
    for i, name in enumerate(names):
        ax = axes[i]
        if args.shape is not None and np.prod(args.shape) == n_px:
            im = ax.imshow(fractions[i].reshape(args.shape), origin="lower",
                           cmap="viridis", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.plot(fractions[i], ".", ms=2)
            ax.set_ylim(0, 1)
            ax.set_xlabel("spectrum index")
        ax.set_title(f"{name} fraction", fontsize=9)

    px = int(np.nanargmax(total))
    model = sum(areas[i, px] * voigt_profile(
        energy, result.centers[i, px], result.sigmas[i, px], args.gamma)
        for i in range(n_states))
    ax = axes[-1]
    ax.plot(energy, signal[px], ".", ms=3, color="0.4", label="data - background")
    ax.plot(energy, model, "-", color="tab:red", label="fit")
    ax.set_title(f"strongest spectrum (#{px})", fontsize=9)
    ax.set_xlabel("Energy (eV)")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "05_map_from_file.png"
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
