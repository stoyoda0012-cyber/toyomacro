"""Calibrate a binding-energy axis on a metal Fermi edge, then read Au 4f.

The usual reference measurement: a gold (or other metal) sample, one
region over the Fermi edge and one over a core level. This example

1. fits the Fermi edge with :func:`toyomacro.fitting.fermi_edge.fit_fermi_edge`
   for three DOS models ('flat', 'linear', 'quadratic') and reports how
   far E_F moves between the models whose reduced chi-square is within
   2x the best (a working cut, not a formal test). On real data this *sensitivity* is
   often far larger than the statistical ``ef_err``; it is a check on
   the model choice, not an uncertainty;
2. shifts the axis so that E_F = 0 (:func:`to_binding_energy`);
3. fits the Au 4f doublet on that calibrated axis with two free Voigt
   peaks and a linear background, and prints the 4f7/2 position, which
   you compare with the reference value you calibrate against.

With no argument it runs on synthetic spectra whose axis is off by a
known amount (-0.45 eV), and checks that the offset is recovered. On
your own data::

    python examples/06_fermi_edge_calibration.py data.vms \\
        --vb-region 2 --au4f-region 1 --photon-energy 1253.6

Regions are read with the format readers (VAMAS ``.vms``, Scienta
``.pxt`` / SES ``.txt``, ``.npl``); a kinetic-energy axis is converted
with ``--photon-energy``. The E_F window defaults to 1.2 eV either side
of the first clear rise seen from the empty (low binding energy) side,
which assumes the first tenth of the scan, plus about 0.5 eV, lies on
the empty side clear of the edge (so long scans need a larger margin
above E_F); keep the
window clear of the next band (for Au the 5d bands, about 2 eV below
E_F). Otherwise pass ``--window``.

Simplifications: the Au 4f peaks are symmetric Voigts, although the
metal lines are slightly asymmetric, which moves a fitted centre by a
small amount; the Au 4f background is a straight line where a step
(Shirley-type) background is expected; and the temperature is held at
``--temperature``.

Output: ``examples/output/06_fermi_edge_calibration.png``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from toyomacro.fitting.fermi_edge import fermi_edge, fit_fermi_edge, to_binding_energy
from toyomacro.voigtfit import voigt_profile

OUT_DIR = Path(__file__).parent / "output"
DOS_MODELS = ("flat", "linear", "quadratic")
SYNTHETIC_OFFSET = -0.45  # eV: where E_F sits on the uncalibrated axis


def synthetic_spectra(seed: int = 0):
    """Uncalibrated BE axes: a Au Fermi edge and a Au 4f doublet, Poisson counts."""
    rng = np.random.default_rng(seed)
    vb = np.arange(-3.0, 3.0 + 1e-9, 0.05)
    vb_clean = fermi_edge(vb, ef=SYNTHETIC_OFFSET, amplitude=3000.0, fwhm_g=0.40,
                          dos_c1=0.3, bg_const=150.0, bg_ref=0.0)
    core = np.arange(80.0, 92.0 + 1e-9, 0.05)
    core_clean = 400.0 + 8.0 * (core - 80.0)
    for center, area in ((84.0, 4.0), (87.67, 3.0)):  # synthetic doublet
        core_clean += 6000.0 * area * voigt_profile(core, center + SYNTHETIC_OFFSET, 0.30, 0.15)
    return (vb, rng.poisson(vb_clean).astype(float),
            core, rng.poisson(core_clean).astype(float))


def read_region(path: Path, region: int, photon_energy: float | None):
    """(binding-energy axis, counts) of one region, sweeps and columns summed."""
    from toyomacro.io.readers import create_reader

    reader = create_reader(path)
    if not 0 <= region < reader.n_regions:
        raise SystemExit(f"{path.name} has regions {list(enumerate(reader.region_names))}; "
                         f"region {region} does not exist")
    raw = reader.read(region_index=region)
    energy = np.asarray(raw.energy, dtype=float)
    counts = np.asarray(raw.specdata, dtype=float).reshape(energy.size, -1).sum(axis=1)
    if raw.metadata.energy_scale.lower().startswith("kinetic"):
        hv = photon_energy or raw.metadata.excitation_energy
        if hv is None or hv <= 0:
            raise SystemExit(f"region {region} is on a kinetic axis: pass --photon-energy")
        energy = hv - energy
    return energy, counts


def fit_ef_models(be, counts, window, temperature):
    """Fermi-edge fits for each DOS model on the same window."""
    weights = 1.0 / np.maximum(counts, 1.0)
    return {dos: fit_fermi_edge(be, counts, convention="BE", dos=dos, window=window,
                                temperature=temperature, weights=weights)
            for dos in DOS_MODELS}


def default_window(be, counts, half_width=1.2, smooth_ev=0.15):
    """``half_width`` eV either side of the Fermi edge, found from the empty side.

    The steepest rise of a valence band is often not the Fermi edge (for
    Au the 5d bands rise more steeply ~2 eV deeper). So walk in from the
    unoccupied (low binding energy) end, take the first point where the
    smoothed intensity clearly leaves the baseline, and refine to the
    steepest point within 0.5 eV of it. The baseline is the first tenth
    of the scan, so that tenth plus about 0.5 eV must lie on the empty
    side clear of the edge.
    """
    order = np.argsort(be)
    e, y = be[order], counts[order]
    n = max(int(round(smooth_ev / float(np.median(np.diff(e))))), 3)
    ys = np.convolve(y, np.ones(n) / n, mode="same")
    base_pts = slice(n, n + max(len(e) // 10, 5))
    base = float(np.mean(ys[base_pts]))
    noise = float(np.std(y[base_pts])) / np.sqrt(n) + 1e-12
    rising = np.nonzero(ys[n:-n] > base + 10.0 * noise)[0]
    onset = e[n + rising[0]] if rising.size else e[len(e) // 2]
    near = (e > onset - 0.2) & (e < onset + 0.5)
    grad = np.gradient(ys, e)
    ef0 = e[near][int(np.argmax(grad[near]))] if near.any() else onset
    return (ef0 - half_width, ef0 + half_width)


def fit_doublet(be, counts, window=(80.0, 92.0)):
    """Two free Voigt peaks + linear background. Returns (params, model, mask)."""
    mask = (be >= window[0]) & (be <= window[1])
    e, y = be[mask], counts[mask]
    order = np.argsort(e)
    e, y = e[order], y[order]
    top = e[np.argsort(y)[::-1]]
    c1 = float(top[0])
    others = [x for x in top if abs(x - c1) > 1.5]
    if not others:
        raise SystemExit("no second peak more than 1.5 eV from the first in the Au 4f window")
    c2 = float(others[0])
    lo, hi = sorted((c1, c2))
    span = y.max() - y.min()

    def model(p, x):
        a1, m1, s1, a2, m2, s2, g, b0, b1 = p
        return (a1 * voigt_profile(x, m1, s1, g) + a2 * voigt_profile(x, m2, s2, g)
                + b0 + b1 * (x - x.mean()))

    p0 = [span, lo, 0.3, span, hi, 0.3, 0.1, y.min(), 0.0]
    bounds = ([0, lo - 1, 0.01, 0, hi - 1, 0.01, 0.0, -np.inf, -np.inf],
              [np.inf, lo + 1, 2.0, np.inf, hi + 1, 2.0, 1.0, np.inf, np.inf])
    w = 1.0 / np.sqrt(np.maximum(y, 1.0))
    res = least_squares(lambda p: (model(p, e) - y) * w, p0, bounds=bounds)
    return res.x, (lambda x: model(res.x, x)), (e, y)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", nargs="?", type=Path, default=None,
                   help="data file with a Fermi-edge and an Au 4f region "
                        "(default: synthetic spectra)")
    p.add_argument("--vb-region", type=int, default=0, help="region with the Fermi edge")
    p.add_argument("--au4f-region", type=int, default=1, help="region with Au 4f")
    p.add_argument("--photon-energy", type=float, default=None,
                   help="photon energy, eV, for kinetic-energy regions")
    p.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"),
                   help="Fermi-edge fit window on the uncalibrated BE axis, eV")
    p.add_argument("--temperature", type=float, default=300.0,
                   help="sample temperature, K, held fixed (default 300)")
    p.add_argument("--dos", choices=DOS_MODELS, default="linear",
                   help="DOS model used for the calibration (default linear)")
    args = p.parse_args(argv)

    if args.path is None:
        vb, vb_counts, core, core_counts = synthetic_spectra()
        source = f"synthetic (true E_F at {SYNTHETIC_OFFSET:+.2f} eV)"
    else:
        vb, vb_counts = read_region(args.path, args.vb_region, args.photon_energy)
        core, core_counts = read_region(args.path, args.au4f_region, args.photon_energy)
        source = args.path.name

    window = tuple(args.window) if args.window else default_window(vb, vb_counts)
    fits = fit_ef_models(vb, vb_counts, window, args.temperature)
    print(f"Fermi edge, {source}, window {window[0]:.2f} to {window[1]:.2f} eV:")
    for dos, r in fits.items():
        flag = "" if r.success else f"   <-- {r.message.split(';')[0]}"
        print(f"  {dos:9s} E_F {r.ef:+.4f} ± {r.ef_err:.4f} eV   "
              f"FWHM {r.resolution:.3f} eV   chi2_red {r.reduced_chi2:.2f}{flag}")
    converged = {d: r for d, r in fits.items() if r.success}
    best_chi2 = min((r.reduced_chi2 for r in converged.values()), default=np.nan)
    kept = [d for d, r in converged.items() if r.reduced_chi2 <= 2.0 * best_chi2]
    efs = [fits[d].ef for d in kept]
    spread = (max(efs) - min(efs)) if len(efs) > 1 else 0.0
    chosen = fits[args.dos]
    print(f"  -> E_F = {chosen.ef:+.4f} eV ({args.dos}); statistical ± "
          f"{chosen.ef_err:.4f} eV; moves by {spread:.3f} eV across the DOS models "
          f"within 2x the best reduced chi-square ({', '.join(kept)})")
    if args.dos not in kept:
        print(f"  warning: the chosen '{args.dos}' model fits much worse than the best")
    if not chosen.success:
        raise SystemExit(f"the {args.dos} fit did not converge cleanly: {chosen.message}")

    core_cal = to_binding_energy(core, chosen.ef, "BE")
    params, model, (e_fit, y_fit) = fit_doublet(core_cal, core_counts)
    c_72, c_52 = sorted((params[1], params[4]))
    print(f"Au 4f on the calibrated axis: 4f7/2 {c_72:.3f} eV, 4f5/2 {c_52:.3f} eV "
          f"(splitting {c_52 - c_72:.3f} eV)")
    if args.path is None:
        print(f"  synthetic check: generated 4f7/2 at 84.000 eV, recovered {c_72:.3f} eV")

    # --- Plot: edge fit (chosen model) and calibrated Au 4f ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(to_binding_energy(chosen.energy, chosen.ef), chosen.fit_curve + chosen.residual,
             ".", ms=3, color="0.4", label="data")
    for dos, r in fits.items():
        if r.success:
            ax1.plot(to_binding_energy(r.energy, chosen.ef), r.fit_curve, lw=1.2,
                     label=f"{dos} (E_F {r.ef - chosen.ef:+.3f})")
    ax1.axvline(0.0, color="k", lw=0.6, ls=":")
    ax1.set_xlabel("Binding energy rel. to E_F (eV)")
    ax1.set_title(f"Fermi edge, FWHM {chosen.resolution:.3f} eV", fontsize=9)
    ax1.invert_xaxis()
    ax1.legend(fontsize=7)
    ax2.plot(e_fit, y_fit, ".", ms=3, color="0.4", label="data")
    ax2.plot(e_fit, model(e_fit), "-", color="tab:red", label="2 Voigt + linear")
    ax2.axvline(c_72, color="k", lw=0.6, ls=":")
    ax2.set_xlabel("Binding energy (eV, E_F = 0)")
    ax2.set_title(f"Au 4f7/2 at {c_72:.3f} eV", fontsize=9)
    ax2.invert_xaxis()
    ax2.legend(fontsize=7)
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "06_fermi_edge_calibration.png"
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
