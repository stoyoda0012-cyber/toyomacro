"""MCP server exposing the toyomacro XPS analysis engine.

Makes the public engine callable from any MCP client (Claude Code,
Claude Desktop, custom agents) — reference-data lookups, spectrum
fitting, and GVRT roundtrip experiments.

Requires the ``mcp`` extra::

    pip install toyomacro[mcp]

Run::

    python -m toyomacro.mcp_server

Every tool returns a JSON string so results are directly machine-readable.
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("toyomacro", instructions=(
    "XPS (X-ray Photoelectron Spectroscopy) analysis engine. "
    "Look up binding energies, compute simplified intrinsic "
    "sensitivities (cross-section x IMFP, no instrument response), list "
    "fitting templates, fit spectrum files, and run GVRT image-roundtrip "
    "experiments to characterize solver accuracy vs. noise."
))


@mcp.tool()
def lookup_binding_energy(element: str, orbital: str) -> str:
    """Look up the reference binding energy for a core level.

    Args:
        element: Element symbol (e.g. 'Si', 'O', 'C', 'Ta', 'Au').
        orbital: Orbital name (e.g. '2p', '1s', '4f', '3d').
    """
    from toyomacro.data import BindingEnergy

    be = BindingEnergy.lookup(element, orbital)
    if be is None:
        return json.dumps({"error": f"No binding energy for {element} {orbital}"})
    return json.dumps({
        "element": element,
        "orbital": orbital,
        "binding_energy_eV": be,
        "spin_orbit_split_eV": BindingEnergy.get_spin_orbit_split(element, orbital),
    })


@mcp.tool()
def calculate_sensitivity(
    element: str,
    orbital: str,
    photon_energy: float = 9251.7,
    compound: str = "SiO2",
) -> str:
    """Calculate a simplified intrinsic sensitivity, sigma x lambda.

    sigma is the photoionization cross-section from whichever table is
    the process default (reported as `cross_section_table` in the result;
    Yeh & Lindau 1985 unless changed) and lambda the TPP-2M inelastic
    mean free path in the given matrix compound. Divide a measured peak
    area by the sensitivity to get a quantity proportional to atomic
    concentration.

    The result is NOT a complete AMRSF and NOT an instrument-specific
    sensitivity factor. It omits analyzer transmission, detector
    response, elastic-scattering / EAL corrections, the photoelectron
    angular distribution, x-ray polarization, and the source/analyzer
    geometry. Absolute composition from this number alone is not
    traceable; use it for relative comparisons, or supply the instrument
    factors yourself.

    Analyzer transmission T(KE/Ep) is a separate, instrument- and
    configuration-specific factor that this tool never applies. Do not
    apply it twice: if the peak areas were taken from a
    transmission-corrected spectrum, leave this sensitivity as it is.

    Both factors extrapolate at HAXPES energies, and the default
    photon energy is a HAXPES line, so the *default* call extrapolates
    both:

    - lambda excludes elastic scattering, so it is an IMFP and not an
      effective attenuation length. TPP-2M is fitted over 50-2000 eV
      kinetic energy; outside that the returned lambda is an
      extrapolation, flagged as `imfp_extrapolated`.
    - sigma is interpolated from a finite table grid, reported as
      `cross_section_table_range_eV`. Above the top grid energy the
      value is a power law fitted to the last few points, flagged as
      `cross_section_extrapolated`. The default Yeh & Lindau table
      stops at 8047.8 eV, below the 9251.7 eV default.

    `cross_section_extrapolated` only covers the energy axis, and only
    one-sidedly. False does NOT mean the value came from the table:

    - within the range, sigma is a polynomial fit in log-log space over
      the whole grid, not a local interpolation, and a sparsely
      tabulated orbital can still be extrapolated;
    - if the table lacks the requested orbital entirely, `lookup` falls
      back to a cross-element fit in log(Z) and returns a number with
      this flag still False. On the default table, asking for a
      j-resolved level such as '2p3/2' takes that path and lands near
      the *whole* 2p doublet. Prefer the bare orbital there, and see
      `docs/API.md` section 5 for the per-table behavior.

    Args:
        element: Element symbol (e.g. 'Si', 'O').
        orbital: Orbital name (e.g. '2p', '1s').
        photon_energy: X-ray photon energy in eV.
            Default 9251.7 (Ga Ka); Al Ka is 1486.6.
        compound: Matrix compound for the IMFP (e.g. 'SiO2', 'Si').
    """
    from toyomacro.data import IMFP, BindingEnergy, CrossSection
    from toyomacro.data.imfp import TPP2M_FITTED_RANGE_EV

    be = BindingEnergy.lookup(element, orbital)
    if be is None:
        return json.dumps({"error": f"No binding energy for {element} {orbital}"})
    ke = photon_energy - be
    sigma = CrossSection.lookup(element, orbital, photon_energy)
    if sigma is None:
        return json.dumps({
            "error": f"No cross-section for {element} {orbital} at {photon_energy} eV"
        })
    lam = IMFP.tpp2m(kinetic_energy=ke, compound=compound)
    lo, hi = TPP2M_FITTED_RANGE_EV
    # Report sigma's grid limits alongside lambda's. Flagging only the
    # IMFP would imply the cross-section is on firmer ground at the same
    # energy, and at the HAXPES default it is not.
    grid = CrossSection.get_available_photon_energies()
    cs_lo, cs_hi = (min(grid), max(grid)) if grid else (None, None)
    return json.dumps({
        "element": element,
        "orbital": orbital,
        "photon_energy_eV": photon_energy,
        "binding_energy_eV": be,
        "kinetic_energy_eV": ke,
        "cross_section": sigma,
        "cross_section_table": CrossSection.get_default_table(),
        "cross_section_table_range_eV": [cs_lo, cs_hi],
        "cross_section_extrapolated": (
            cs_lo is None or not (cs_lo <= photon_energy <= cs_hi)
        ),
        "imfp_nm": lam,
        "imfp_model": "TPP-2M (Tanuma, Powell & Penn 1994); inelastic only",
        "imfp_fitted_range_eV": [lo, hi],
        "imfp_extrapolated": not (lo <= ke <= hi),
        "sensitivity": sigma * lam,
        "sensitivity_model": (
            "simplified intrinsic sensitivity = cross-section x IMFP; "
            "not a complete AMRSF. Excludes analyzer transmission, "
            "detector response, elastic scattering/EAL, angular "
            "distribution, polarization and geometry"
        ),
        "compound": compound,
    })


@mcp.tool()
def list_fitting_templates() -> str:
    """List the built-in XPS fitting templates (chemical-state models)."""
    from toyomacro.fitting.templates import get_template, list_templates

    out = []
    for name in list_templates():
        t = get_template(name)
        out.append({
            "name": t.name,
            "element": t.element,
            "description": t.description,
            "peaks": [p.name for p in t.peaks],
        })
    return json.dumps(out)


@mcp.tool()
def fit_spectrum_file(
    path: str,
    element: str = "",
    region: int = 0,
) -> str:
    """Fit an XPS spectrum file with AutoFitter and return the decomposition.

    Supported formats: VAMAS (.vms), Igor Pro (.pxt), NPL (.npl),
    SES text (.txt). Angle-resolved data is summed over angles.

    Args:
        path: Path to the spectrum file.
        element: XPS label for template-based fitting (e.g. 'Si2p',
            'Ta4f', 'C1s'). Empty = automatic peak detection.
        region: Region index for multi-region files.
    """
    import numpy as np

    from toyomacro.core import Spectrum
    from toyomacro.fitting import AutoFitter
    from toyomacro.io.readers import create_reader

    raw = create_reader(path).read(region)
    intensity = np.asarray(raw.specdata, dtype=np.float64)
    while intensity.ndim > 1:
        intensity = intensity.sum(axis=-1)
    energy_type = "KE" if raw.metadata.energy_scale.lower().startswith("k") else "BE"
    spectrum = Spectrum(
        energy=np.asarray(raw.energy, dtype=np.float64),
        intensity=intensity,
        energy_type=energy_type,
        element=element or None,
    )

    result = AutoFitter().fit(spectrum)
    fit = result.fitting_result
    components = [
        {
            "center_eV": c.center,
            # PeakComponent follows the MATLAB fitpara layout:
            # `height` holds the integrated area, `area` the peak height.
            "integrated_area": c.height,
            "peak_height": c.area,
            "fwhm_gauss_eV": c.fwhm_g,
            "fwhm_lorentz_eV": c.fwhm_l,
        }
        for c in fit.components
    ]
    return json.dumps({
        "success": result.success,
        "message": result.message,
        "r_squared": result.r_squared,
        "n_components": fit.n_components,
        "background": fit.background.type,
        "components": components,
    })


@mcp.tool()
def gvrt_run(
    image_path: str = "",
    size: int = 256,
    solver: str = "taylor",
    noise: str = "Moderate",
    exact: bool = False,
    output_png: str = "",
) -> str:
    """Run a GVRT roundtrip: image -> Voigt spectra -> noise -> fit -> image.

    Encodes an RGB image as per-pixel Voigt parameters (R=amplitude,
    G=energy shift, B=FWHM), generates one spectrum per pixel, adds
    Poisson noise, fits every spectrum, and reconstructs the image.
    Per-channel PSNR measures how well each parameter survives the
    roundtrip — a direct, visual accuracy metric for the solver.

    Args:
        image_path: Input image file. Empty = built-in demo image.
        size: Demo image edge length in pixels (ignored with image_path).
        solver: 'taylor' (fast), 'taylor6', or 'dict2d_parabola'
            (best quality).
        noise: 'None', 'Subtle', 'Weak', 'Small', 'Moderate', 'Strong',
            'Intense', or 'lamX' notation.
        exact: Use exact Voigt generation (slower; avoids Taylor
            linearization artifacts at large shifts).
        output_png: If set, save an original/reconstructed/difference
            figure to this path.
    """
    result = _run_gvrt(image_path, size, solver, noise, exact)

    if output_png:
        _save_gvrt_figure(result, output_png)

    return json.dumps(_gvrt_result_dict(result))


@mcp.tool()
def gvrt_sweep(
    noise_levels: str = "None,Weak,Moderate,Strong",
    size: int = 192,
    solver: str = "taylor",
    exact: bool = False,
) -> str:
    """Run GVRT roundtrips across noise levels (accuracy-vs-noise ladder).

    Args:
        noise_levels: Comma-separated noise level names or lamX values.
        size: Demo image edge length in pixels.
        solver: 'taylor', 'taylor6', or 'dict2d_parabola'.
        exact: Use exact Voigt generation.
    """
    out = []
    for level in [s.strip() for s in noise_levels.split(",") if s.strip()]:
        result = _run_gvrt("", size, solver, level, exact)
        out.append(_gvrt_result_dict(result))
    return json.dumps(out)


# ---------------------------------------------------------------------------
# GVRT helpers
# ---------------------------------------------------------------------------


def _run_gvrt(image_path: str, size: int, solver: str, noise: str, exact: bool):
    from toyomacro.voigtfit.gvrt_service import (
        GVRTConfig,
        GVRTService,
        make_demo_image,
        prepare_image,
    )

    image = prepare_image(image_path) if image_path else make_demo_image(size)
    config = GVRTConfig(solver=solver, noise_level=noise, exact_voigt=exact)
    return GVRTService().run(image, config)


def _gvrt_result_dict(result) -> dict:
    return {
        "solver": result.solver,
        "noise_level": result.noise_level,
        "n_spectra": result.n_spectra,
        "psnr_dB": {
            "amplitude": result.psnr.amplitude,
            "shift": result.psnr.shift,
            "fwhm": result.psnr.fwhm,
            "average": result.psnr.total,
        },
        "amp_rmse": result.amp_rmse,
        "shift_rmse_eV": result.shift_rmse,
        "fwhm_rmse_eV": result.fwhm_rmse,
        "throughput_spectra_per_s": result.throughput,
        "gen_time_s": result.gen_time,
        "fit_time_s": result.fit_time,
    }


def _save_gvrt_figure(result, output_png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    diff = np.abs(
        result.original.astype(np.int16) - result.reconstructed.astype(np.int16)
    ).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, img, title in [
        (axes[0], result.original, "original"),
        (axes[1], result.reconstructed, "reconstructed"),
        (axes[2], diff, "|difference|"),
    ]:
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"avg PSNR {result.psnr.total:.1f} dB", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    mcp.run()
