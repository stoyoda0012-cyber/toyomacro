"""PTFE composition from public data: the estimate, what moves it, and what is not evaluated.

The answer to "what is the error bar?" in this version, on one example:

- an **estimate** of the homogeneous-equivalent F:C composition of PTFE,
- its **condition dependence** — how the estimate moves when the
  cross-section table, the background, or an assumption about the stored
  data (what the intensity is, whether the transmission was applied) is
  changed (differences between conditions, not an uncertainty),
- the **assumptions** it rests on, listed beside it,
- and what is **not evaluated**: here the statistical uncertainty, because
  the file does not state that the intensity is raw counts — and in this
  version no scope is validated for it in any case
  (docs/design/composition-uncertainty.md §7).

Data. The first F 1s / C 1s pair (source blocks 0 and 1) of
``Kratos Axis Ultra/PTFE.vms`` in the polymer-degradation dataset of
Zenodo 10.5281/zenodo.7074887 (CC BY 4.0), the data for the Surface and
Interface Analysis paper "Revisiting Degradation in the XPS Analysis of
Polymers", DOI 10.1002/sia.7151. It is not bundled:

    python examples/10_composition_ptfe.py --data "Degradation Polymers.zip"
    python examples/10_composition_ptfe.py --download     # fetch it once

``--data`` also reads ``TOYOMACRO_VAMAS_POLYMER_ZIP``. With neither, the
example runs on **synthetic** spectra shaped like that pair and says so.

What it is not. The first stored pair is not an undamaged surface (the
series follows X-ray degradation), the ideal F:C = 2:1 is a stoichiometric
expectation and not a certified value, and this is not the paper's own
procedure (which used the instrument's RSF library and its own settings;
the paper states a 40 eV pass energy, the file records 20 eV, and the file
is used). The stored intensity's meaning and whether the transmission
column was applied are unknown in the file; the assumptions the example
makes about them are printed, not hidden.

Output: ``examples/output/10_composition_ptfe.png``
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import tempfile
import urllib.request
import zipfile
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toyomacro.composition import Conditions, Line, Matrix, composition, condition_dependence
from toyomacro.io.readers.vamas_reader import VAMASReader

OUT_DIR = Path(__file__).parent / "output"
MEMBER = "Kratos Axis Ultra/PTFE.vms"
URL = "https://zenodo.org/records/7074887/files/Degradation%20Polymers.zip?download=1"
SHA256 = "99fea65bca138543dbac2f377ef4c4c0ce2317e5b78dcd9cfa71d108d8e81453"
CACHE = Path.home() / ".cache" / "toyomacro" / "zenodo-7074887" / "Degradation Polymers.zip"

#: Binding-energy windows (eV) on the stored, uncorrected axis.
WINDOWS_BE = {"F": (675.0, 697.0), "C": (279.0, 297.0)}

#: PTFE, (C2F4)n. Mw and Nv follow from the formula (Nv = 2x4 + 4x7 valence
#: electrons per unit); the density is a nominal value and the band gap an
#: assumption — the example shows how far the band gap moves the answer.
PTFE = Matrix(name="PTFE (C2F4)n",
              source="Mw, Nv from the formula; density 2.2 g/cm3 nominal; Eg 7.7 eV assumed",
              Nv=36.0, density=2.2, Mw=100.02, Eg=7.7)


def fetch(dest: Path = CACHE) -> Path:
    """Download the dataset once, and refuse it unless its SHA-256 matches."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.rename(dest)
    if hashlib.sha256(dest.read_bytes()).hexdigest() != SHA256:
        dest.unlink()  # do not keep a wrong file to fail on next time
        raise ValueError(f"{dest} was not the archive this example was written for; removed")
    return dest


def load_pair(archive: Path):
    """The first F 1s / C 1s pair of the Kratos PTFE file, as read."""
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "PTFE.vms"
        path.write_bytes(z.read(MEMBER))
        reader = VAMASReader(path)
        f1s, c1s = reader.read(0), reader.read(1)
    if (f1s.metadata.region, c1s.metadata.region) != ("F 1s/2", "C 1s/3"):
        raise ValueError("the first two blocks are not the F 1s / C 1s pair expected")
    return {"F": f1s, "C": c1s}


def synthetic_pair(seed: int = 7):
    """Spectra shaped like the real pair (positions, counts, transmission column).

    A stand-in so the example runs offline; its numbers say nothing about PTFE.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for el, be0, height, base, t, ke0 in (("F", 685.7, 6.0e4, 600.0, 0.668, 786.69),
                                          ("C", 288.6, 1.1e4, 250.0, 0.578, 1186.69)):
        n = 301 if el == "F" else 251
        e = ke0 + 0.1 * np.arange(n)
        be = 1486.6 - e
        mu = base + height * np.exp(-0.5 * ((be - be0) / 0.7) ** 2)
        mu += 0.6 * base * (be > be0)  # a step behind the peak
        counts = rng.poisson(mu).astype(float)
        out[el] = _Synthetic(e, counts, np.full(n, t))
    return out


class _Synthetic:
    """Just enough of RawSpectrumData for build_lines."""

    def __init__(self, energy, counts, transmission):
        from toyomacro.io import CorrespondingVariable, SpectrumMetadata

        self.energy = energy
        self.specdata = counts.reshape(-1, 1)
        self.corresponding_variables = (
            CorrespondingVariable("Intensity", "d", counts),
            CorrespondingVariable("Transmission", "d", transmission))
        self.metadata = SpectrumMetadata(
            excitation_energy=1486.6, energy_scale="Kinetic", n_sweeps=4,
            signal_collection_time=0.2 if transmission[0] > 0.6 else 0.24,
            field_origins={"n_sweeps": "file", "signal_collection_time": "file"})


def build_lines(pair) -> list[Line]:
    """Lines for composition(): windows, exposure and transmission from the file."""
    lines = []
    for el, orb in (("F", "1s"), ("C", "1s")):
        d = pair[el]
        md = d.metadata
        hv = md.excitation_energy
        counts = d.specdata[:, 0]
        lo, hi = WINDOWS_BE[el]
        peak_be = hv - d.energy[int(np.argmax(counts))]
        transmission = next(v.values for v in d.corresponding_variables
                            if v.label.lower() == "transmission")
        lines.append(Line(
            element=el, orbital=orb, energy=d.energy, intensity=counts,
            window=(hv - hi, hv - lo), energy_scale="kinetic",
            # The apparent peak position: it includes any charging, and is
            # used only for the IMFP's kinetic energy and the window check.
            binding_energy=float(peak_be),
            exposure_s=md.signal_collection_time * md.n_sweeps,
            exposure_basis=(f"file: collection time {md.signal_collection_time} s x "
                            f"{md.n_sweeps} scans, assumed per scan and point"),
            transmission=transmission,
            source=f"Kratos PTFE.vms, block {getattr(md, 'source_region_index', '?')}"))
    return lines


def conditions(**kw) -> Conditions:
    base = dict(photon_energy=1486.6, table="scofield", background="shirley", matrix=PTFE,
                intensity_semantics="unknown", assume_intensity="integrated_counts",
                transmission_applied="unknown", assume_transmission="divide_by_curve",
                declared_elements=("F", "C"))
    base.update(kw)
    return Conditions(**base)


def analyse(lines: list[Line]) -> dict:
    """Everything the example prints, as data (tested)."""
    base = composition(lines, conditions())
    grid = condition_dependence(lines, conditions(), tables=("scofield", "yeh_lindau"),
                                backgrounds=("shirley", "linear"))
    no_t = composition(lines, conditions(assume_transmission="equal_across_lines"))
    rate = composition(lines, conditions(assume_intensity="count_rate"))
    eg = {g: composition(lines, conditions(matrix=replace(PTFE, Eg=g))).fractions["F"]
          for g in (6.7, 8.7)}
    return {"base": base, "grid": grid, "transmission_equal": no_t, "count_rate": rate,
            "eg": eg}


def report(res: dict, synthetic: bool) -> None:
    b = res["base"]
    tag = "SYNTHETIC stand-in (pass --data or --download for the real pair)" if synthetic \
        else "Zenodo 7074887, Kratos PTFE, first F 1s / C 1s pair"
    print(f"Data: {tag}")
    print(f"Quantity: {b.quantity}")
    print(f"Status: {b.status}")
    print("Estimate (atomic fraction):",
          ", ".join(f"{el} {100 * f:.1f} %" for el, f in b.fractions.items()),
          "   [stoichiometric expectation F 66.7 %, C 33.3 % — a reference, not a certificate]")
    print("Condition dependence (change in F, percentage points; not an uncertainty):")
    for ch in res["grid"].changes:
        print(f"  table {ch.table:15s} background {ch.background:8s} "
              f"{100 * ch.change['F']:+6.2f}")
    print(f"  2x2 interaction (F): {100 * res['grid'].interaction['F']:+.2f}")
    print(f"  intensity read as a count rate instead of integrated counts: "
          f"{100 * (res['count_rate'].fractions['F'] - b.fractions['F']):+.2f}")
    print(f"  transmission assumed equal instead of divided out: "
          f"{100 * (res['transmission_equal'].fractions['F'] - b.fractions['F']):+.2f}")
    print(f"  band gap 6.7 / 8.7 eV instead of 7.7: "
          f"{100 * (res['eg'][6.7] - b.fractions['F']):+.2f} / "
          f"{100 * (res['eg'][8.7] - b.fractions['F']):+.2f}")
    print("Assumptions:")
    for a in b.assumptions:
        print(f"  - {a}")
    for ln, (lo, hi) in zip(b.lines, (WINDOWS_BE["F"], WINDOWS_BE["C"])):
        print(f"  - {ln.element} {ln.orbital}: window {lo:g}-{hi:g} eV on the stored axis; "
              f"exposure {ln.exposure_s:g} s ({ln.exposure_basis})")
    print(f"  - matrix {PTFE.name}: {PTFE.source}")
    print("Not evaluated:")
    for n in b.not_evaluated:
        print(f"  - {n}")


def plot(lines: list[Line], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, ln in zip(axes, lines):
        hv = 1486.6
        ax.plot(hv - ln.energy, ln.intensity, lw=0.8)
        ax.axvspan(hv - ln.window[1], hv - ln.window[0], alpha=0.12)
        ax.set_xlabel("binding energy, stored axis (eV)")
        ax.set_title(f"{ln.element} {ln.orbital}")
        ax.invert_xaxis()
    axes[0].set_ylabel("stored intensity")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=os.environ.get("TOYOMACRO_VAMAS_POLYMER_ZIP"),
                   help="the Zenodo 7074887 archive")
    p.add_argument("--download", action="store_true",
                   help=f"fetch the archive once into {CACHE.parent}")
    args = p.parse_args(argv)
    archive = fetch() if args.download else (Path(args.data) if args.data else None)
    if archive is not None and hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f"{archive} is not the archive this example was written for")
    pair = load_pair(archive) if archive else synthetic_pair()
    lines = build_lines(pair)
    res = analyse(lines)
    report(res, synthetic=archive is None)
    plot(lines, OUT_DIR / "10_composition_ptfe.png")
    return res


if __name__ == "__main__":
    main()
