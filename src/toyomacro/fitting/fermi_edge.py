"""Fermi-edge fitting: Fermi level position and instrumental resolution.

Fits the Fermi edge of a metal reference (Au, Ag, Pt, a metal clip in
contact with the sample) with

    I(E) = A * DOS(u) * f(E; E_F, T)  (x)  G(FWHM_G)  +  bg(E)

    u        distance from E_F into the occupied side, u = max(s*(E - E_F), 0),
             with s = +1 on a binding-energy axis and -1 on a kinetic one
    DOS(u)   1 + c1*u + c2*u^2   -- the shape just past the edge
             'flat'      c1 = c2 = 0
             'linear'    c1 free (default)
             'quadratic' c1, c2 free (e.g. the Au 5d shoulder in a wide window)
    f        Fermi-Dirac occupation at temperature T (K)
    G        Gaussian instrumental broadening, FWHM_G (eV)
    bg(E)    'none' | 'constant' | 'linear', fitted in the window

Uses:

- **Energy-axis calibration.** :func:`to_binding_energy` puts a spectrum
  on a binding-energy axis with the fitted E_F at 0 eV.
- **Instrumental resolution.** With the temperature fixed at the known
  sample temperature, the fitted Gaussian FWHM is the analyser plus
  source resolution.
- **Rigid edge shifts** between two conditions (charging, bias):
  :func:`differential_ef`.

Validity and limits -- read before trusting a number:

- **Temperature and resolution are nearly degenerate.** Both broaden the
  edge; the thermal 10-90% width is 2*ln(9)*kB*T (about 0.11 eV at
  300 K). Fit one with the other fixed. Fitting both is only meaningful
  when the Gaussian FWHM is not much larger than that thermal width.
- **The model is a Gaussian-broadened step on a low-order DOS.** It
  assumes the DOS varies slowly across the edge (true for Au/Ag/Pt near
  E_F over a window of a few eV), a Gaussian instrument function, and no
  lifetime broadening. A DOS model that is too simple for the data
  biases E_F: a 'flat' DOS on an edge whose DOS rises is the usual case.
- **``ef_err`` and the other uncertainties are 1-sigma from the
  linearised covariance, scaled by the reduced chi-square.** They are
  statistical only and assume the model is correct and the weights are
  inverse variances (Poisson: ``weights = 1/max(I, 1)``); they do not
  include the model or window choice. Checked on simulated Poisson
  edges with about 2000 counts per channel at the edge: over 200
  realisations the E_F and FWHM pulls have unit width and zero mean. At
  low counts (~100 per channel) the pulls widen to about 1.25, partly
  because the weights are taken from the data.
- **E_F is where the fitted Fermi-Dirac function is 1/2 before
  broadening**; it is not the inflection point of the measured edge,
  which a sloped DOS moves.
- **The window must stop before the next band.** For Au the 5d bands
  set in about 2 eV below E_F; a window reaching into them cannot be
  described by a low-order DOS and the fit fails (typically with the
  resolution running to its upper bound). ``success`` is False, with
  the reason in ``message``, when a parameter ends on a bound, an
  uncertainty is not finite, E_F falls outside the window, or the fitted
  resolution is below half a channel. When the resolution is below
  about a channel, or much smaller than the thermal width, it cannot be
  determined: hold it fixed (``fit_resolution=False``) and E_F is still
  fitted normally.
- **Compare DOS models on real data.** On two measured Au reference
  edges (0.36-0.5 eV resolution; the measurements are not distributed),
  'flat', 'linear' and 'quadratic' moved E_F by 0.05-0.14 eV, an order
  of magnitude beyond ``ef_err``; 'flat' had a reduced chi-square
  several times the best on both, and between the other two E_F moved
  by 0.01-0.05 eV. Treat the spread among the models that fit about as
  well as the best as a *sensitivity check*: it is neither a bound nor
  an uncertainty -- it can exceed the real error when one model is
  clearly right, and understate it because all three share the
  Gaussian, no-lifetime assumptions.

Origin: a generalisation of ``toyomacro.lineshape.FermiDirac`` (same
model on a binding-energy axis, :func:`fermi_edge` adds kinetic axes and
a background) and of a two-stage valence-band analysis that fixed the
temperature to find the DOS shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.optimize import least_squares

__all__ = [
    "KB_EV",
    "FermiEdgeResult",
    "detect_convention",
    "differential_ef",
    "fermi_edge",
    "fit_fermi_edge",
    "to_binding_energy",
]

#: Boltzmann constant, eV/K (CODATA 2018: 8.617333262e-5)
KB_EV = 8.617333262e-5

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _sign_of(convention: str) -> int:
    """+1 for a binding-energy axis, -1 for kinetic: s*(E - E_F) > 0 is occupied."""
    c = convention.upper()
    if c in ("BE", "BINDING"):
        return +1
    if c in ("KE", "KINETIC"):
        return -1
    raise ValueError(f"unknown convention {convention!r} (use 'BE', 'KE' or 'auto')")


def detect_convention(energy: NDArray, intensity: NDArray) -> str:
    """Guess the axis direction from which side of the edge is bright.

    Only the occupied side carries intensity, so on an ascending axis a
    brighter high-energy end means binding energy. Pass the convention
    explicitly when the file states it.
    """
    e = np.asarray(energy, dtype=float)
    y = np.asarray(intensity, dtype=float)[np.argsort(e)]
    n = max(len(y) // 5, 3)
    return "BE" if y[-n:].mean() > y[:n].mean() else "KE"


def _broaden(grid: NDArray, profile: NDArray, fwhm_g: float) -> NDArray:
    """Gaussian convolution on a uniform ascending grid (area-preserving)."""
    if fwhm_g <= 1e-9 or len(grid) < 3:
        return profile
    sigma_pts = fwhm_g * _FWHM_TO_SIGMA / float(grid[1] - grid[0])
    if sigma_pts < 0.05:  # much narrower than the grid: no-op
        return profile
    return gaussian_filter1d(profile, sigma_pts, mode="nearest")


def _padded_grid(e_sorted: NDArray, pad: float) -> tuple[NDArray, slice | None]:
    """Extend an ascending axis by ``pad`` eV on both sides.

    A uniform axis (to 1e-4 of its step, so float32 axes qualify) is
    extended with its own step and the returned slice selects the
    original points exactly. A non-uniform one is replaced by a uniform
    grid at its smallest step, but no finer than 1/20000 of the span
    (slice None: interpolate back).
    """
    if len(e_sorted) < 3:
        return e_sorted, slice(0, len(e_sorted))
    d = np.diff(e_sorted)
    if np.any(d <= 0):
        raise ValueError("energy axis has repeated values")
    dx = float(np.mean(d))
    if np.max(np.abs(d - dx)) < 1e-4 * dx:
        n = int(np.ceil(max(pad, 0.0) / dx))
        left = e_sorted[0] - dx * np.arange(n, 0, -1)
        right = e_sorted[-1] + dx * np.arange(1, n + 1)
        return np.concatenate([left, e_sorted, right]), slice(n, n + len(e_sorted))
    dx_u = max(float(np.min(d)), float(e_sorted[-1] - e_sorted[0]) / 20000.0)
    pad = max(pad, 0.0)
    return np.arange(e_sorted[0] - pad, e_sorted[-1] + pad + 0.5 * dx_u, dx_u), None


def fermi_edge(
    energy: NDArray,
    ef: float,
    amplitude: float = 1.0,
    fwhm_g: float = 0.2,
    temperature: float = 300.0,
    dos_c1: float = 0.0,
    dos_c2: float = 0.0,
    bg_const: float = 0.0,
    bg_slope: float = 0.0,
    convention: str = "BE",
    bg_ref: float | None = None,
) -> NDArray[np.float64]:
    """Evaluate the Fermi-edge model, background included.

    Args:
        energy: Energy axis, eV, ascending or descending.
        ef: Fermi level position E_F, eV, on the same axis.
        amplitude: Height of the occupied-side step at E_F (before the DOS
            slope), in intensity units.
        fwhm_g: Gaussian instrumental FWHM, eV.
        temperature: Temperature, K.
        dos_c1, dos_c2: DOS coefficients, eV^-1 and eV^-2.
        bg_const, bg_slope: Background level and slope (per eV).
        convention: ``'BE'`` or ``'KE'``.
        bg_ref: Energy the background slope pivots on; the axis mean if None.

    Returns:
        Model intensity on ``energy``, in its original order.

    The Gaussian convolution is evaluated on a grid padded by 4 sigma
    beyond both ends of ``energy``, so channels near the ends see the
    edge profile that continues past them instead of a clamped copy of
    the end value. A spectrum cut to a fit window is therefore modelled
    as the window of a full spectrum.
    """
    e = np.asarray(energy, dtype=float)
    s = _sign_of(convention)
    t_k = max(float(temperature), 1.0)

    order = np.argsort(e)
    e_sorted = e[order]
    grid, inner = _padded_grid(e_sorted, 4.0 * float(fwhm_g) * _FWHM_TO_SIGMA)
    u_signed = s * (grid - ef)
    u = np.maximum(u_signed, 0.0)
    dos = np.maximum(1.0 + dos_c1 * u + dos_c2 * u * u, 0.0)
    arg = np.clip(-u_signed / (KB_EV * t_k), -700.0, 700.0)
    occupation = 1.0 / (1.0 + np.exp(arg))

    profile = _broaden(grid, amplitude * dos * occupation, float(fwhm_g))
    profile = profile[inner] if inner is not None else np.interp(e_sorted, grid, profile)
    profile = profile[np.argsort(order)]
    ref = float(np.mean(e)) if bg_ref is None else float(bg_ref)
    return profile + bg_const + bg_slope * (e - ref)


@dataclass
class FermiEdgeResult:
    """Result of :func:`fit_fermi_edge`.

    Uncertainties (``*_err``) are 1-sigma, statistical only; ``nan`` for a
    parameter that was held fixed. See the module docstring for what they
    assume.
    """

    ef: float
    ef_err: float
    resolution: float  #: Gaussian FWHM, eV
    resolution_err: float
    temperature: float  #: K
    temperature_err: float
    amplitude: float
    amplitude_err: float
    dos_c1: float
    dos_c2: float
    bg_const: float
    bg_slope: float
    width_1090: float  #: 10-90% width of the fitted edge on a flat DOS, eV
    energy: NDArray  #: energies inside the fit window
    fit_curve: NDArray  #: model incl. background, on ``energy``
    background: NDArray
    residual: NDArray  #: data minus model
    reduced_chi2: float
    r_squared: float
    success: bool
    convention: str
    dos: str
    message: str = ""

    def summary(self) -> str:
        """One-line summary for logs."""
        t_txt = (f"{self.temperature:.0f} K (fixed)" if np.isnan(self.temperature_err)
                 else f"{self.temperature:.0f}±{self.temperature_err:.0f} K")
        res_txt = (f"{self.resolution:.3f} eV (fixed)" if np.isnan(self.resolution_err)
                   else f"{self.resolution:.3f}±{self.resolution_err:.3f} eV")
        return (f"E_F = {self.ef:.4f} ± {self.ef_err:.4f} eV [{self.convention}] | "
                f"FWHM_G = {res_txt} | T = {t_txt} | 10-90% = {self.width_1090:.3f} eV | "
                f"DOS {self.dos} | chi2_red = {self.reduced_chi2:.3g}")


def _initial_ef(energy: NDArray, intensity: NDArray, s: int, smooth_ev: float = 0.15) -> float:
    """E_F starting value: the steepest point of the edge smoothed over ``smooth_ev``.

    The smoothing is a width in eV, not a point count, so finely stepped
    spectra are smoothed as much as coarse ones.
    """
    step = float(np.median(np.diff(energy)))
    size = max(int(round(smooth_ev / step)), 3) if step > 0 else 3
    y = uniform_filter1d(intensity, size=size)
    return float(energy[int(np.argmax(s * np.gradient(y, energy)))])


def _width_1090(ef, fwhm_g, temperature, convention) -> float:
    """10-90% width of the fitted edge rebuilt on a flat DOS, eV.

    Evaluated on its own grid around E_F, wide enough for both plateaus,
    so it does not depend on the fit window. Removes the DOS slope and
    background, so the sharpness can be compared between conditions.
    """
    s = _sign_of(convention)
    half = 4.0 * float(fwhm_g) + 20.0 * KB_EV * max(float(temperature), 1.0)
    grid = np.linspace(ef - half, ef + half, 4001)
    edge = fermi_edge(grid, ef=ef, amplitude=1.0, fwhm_g=fwhm_g,
                      temperature=temperature, convention=convention)
    order = np.argsort(s * grid)
    u, f = (s * grid)[order], edge[order]  # rises from ~0 to ~1 along u
    return float(np.interp(0.9, f, u) - np.interp(0.1, f, u))


def fit_fermi_edge(
    energy: NDArray,
    intensity: NDArray,
    *,
    convention: str = "auto",
    dos: str = "linear",
    background: str = "linear",
    window: tuple[float, float] | None = None,
    ef_init: float | None = None,
    ef_bounds: tuple[float, float] | None = None,
    resolution: float = 0.2,
    fit_resolution: bool = True,
    temperature: float = 300.0,
    fit_temperature: bool = False,
    weights: NDArray | None = None,
    max_nfev: int = 4000,
) -> FermiEdgeResult:
    """Fit a Fermi edge; return E_F, the Gaussian resolution and 1-sigma errors.

    Args:
        energy, intensity: One spectrum (angle-integrated if need be); the
            axis may be ascending or descending.
        convention: ``'BE'``, ``'KE'`` or ``'auto'`` (guessed from which side
            of the edge is bright; pass it explicitly when the file says).
        dos: ``'flat'``, ``'linear'`` (default) or ``'quadratic'`` DOS past
            the edge.
        background: ``'none'``, ``'constant'`` or ``'linear'`` (default),
            fitted inside the window. Remove a wide-range background first.
        window: ``(lo, hi)`` fit window, eV. Leave roughly 1 eV of plateau
            on each side of the edge, and use the same window for spectra
            you compare.
        ef_init: Starting E_F; the steepest point of the edge if None.
        ef_bounds: E_F search range; ``ef_init ± 1 eV`` if None.
        resolution: Gaussian FWHM, eV: the starting value, or the fixed
            value when ``fit_resolution`` is False.
        fit_resolution: Fit the Gaussian FWHM (default True).
        temperature: Sample temperature, K: fixed value, or the starting
            value when ``fit_temperature`` is True.
        fit_temperature: Fit the temperature too. It is strongly
            correlated with the resolution; see the module docstring.
        weights: Per-point weights, proportional to inverse variance
            (Poisson: ``1/max(I, 1)``). Unit weights if None.
        max_nfev: Maximum function evaluations for the least-squares solver.

    Returns:
        FermiEdgeResult.
    """
    e_all = np.asarray(energy, dtype=float).ravel()
    y_all = np.asarray(intensity, dtype=float).ravel()
    if e_all.size != y_all.size:
        raise ValueError("energy and intensity differ in length")
    if convention == "auto":
        convention = detect_convention(e_all, y_all)
    convention = "BE" if _sign_of(convention) > 0 else "KE"
    s = _sign_of(convention)
    if dos not in ("flat", "linear", "quadratic"):
        raise ValueError(f"unknown dos {dos!r} (use 'flat', 'linear' or 'quadratic')")
    if background not in ("none", "constant", "linear"):
        raise ValueError(f"unknown background {background!r} "
                         "(use 'none', 'constant' or 'linear')")

    if window is not None:
        lo, hi = min(window), max(window)
        mask = (e_all >= lo) & (e_all <= hi)
    else:
        mask = np.ones_like(e_all, dtype=bool)
    E, Y = e_all[mask], y_all[mask]
    if E.size < 8:
        raise ValueError("fewer than 8 points in the fit window")
    W = (np.asarray(weights, dtype=float).ravel()[mask]
         if weights is not None else np.ones_like(Y))
    sqrtW = np.sqrt(np.clip(W, 0.0, None))
    span = float(np.ptp(E))
    e_ref = float(np.mean(E))

    # --- starting values ---
    y_ord = Y[np.argsort(s * E)]  # unoccupied side first
    n_end = max(len(Y) // 10, 3)
    plateau_unocc = float(np.mean(y_ord[:n_end]))
    plateau_occ = float(np.mean(y_ord[-n_end:]))
    amp0 = max(plateau_occ - plateau_unocc, 1e-12 + 0.1 * abs(plateau_occ))
    if ef_init is None:
        asc = np.argsort(E)
        ef_init = _initial_ef(E[asc], Y[asc], s)
    if ef_bounds is None:
        ef_bounds = (ef_init - 1.0, ef_init + 1.0)

    # --- parameter vector: ef, amp, [fwhm_g], [T], [c1], [c2], [bg_const], [bg_slope] ---
    p0 = [float(ef_init), float(amp0)]
    lb = [float(min(ef_bounds)), 0.0]
    ub = [float(max(ef_bounds)), np.inf]
    names = ["ef", "amp"]

    def add(name, start, lower, upper):
        p0.append(start)
        lb.append(lower)
        ub.append(upper)
        names.append(name)

    if fit_resolution:
        add("fwhm_g", max(float(resolution), 1e-3), 1e-3, span)
    if fit_temperature:
        add("temperature", max(float(temperature), 1.0), 1.0, 1.0e4)
    if dos in ("linear", "quadratic"):
        add("dos_c1", 0.0, -10.0, 10.0)
    if dos == "quadratic":
        add("dos_c2", 0.0, -10.0, 10.0)
    if background in ("constant", "linear"):
        add("bg_const", plateau_unocc, -np.inf, np.inf)
    if background == "linear":
        add("bg_slope", 0.0, -np.inf, np.inf)

    def unpack(p):
        v = dict(zip(names, p))
        return {
            "ef": v["ef"],
            "amplitude": v["amp"],
            "fwhm_g": v.get("fwhm_g", resolution),
            "temperature": v.get("temperature", temperature),
            "dos_c1": v.get("dos_c1", 0.0),
            "dos_c2": v.get("dos_c2", 0.0),
            "bg_const": v.get("bg_const", 0.0),
            "bg_slope": v.get("bg_slope", 0.0),
        }

    def model(p, e):
        return fermi_edge(e, convention=convention, bg_ref=e_ref, **unpack(p))

    def residual(p):
        return (model(p, E) - Y) * sqrtW

    result = least_squares(residual, np.asarray(p0), bounds=(np.asarray(lb), np.asarray(ub)),
                           method="trf", max_nfev=max_nfev)

    dof = max(int(np.count_nonzero(W > 0)) - len(p0), 1)
    red_chi2 = float(np.sum(result.fun ** 2)) / dof
    try:
        J = result.jac
        cov = np.linalg.inv(J.T @ J) * red_chi2
        perr = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    except np.linalg.LinAlgError:
        perr = np.full_like(result.x, np.nan)
    err = dict(zip(names, perr))

    # The trust-region solver stays strictly inside the bounds, so "on a
    # bound" means within 0.1% of the smaller of the allowed range and the
    # value's own magnitude (so a wide range does not flag a good fit that
    # merely sits near its lower end, e.g. T = 10 K in [1, 1e4] K).
    at_bound = []
    for n, x, lo_, hi_ in zip(names, result.x, lb, ub):
        tol = (1e-3 * min(hi_ - lo_, max(1.0, abs(x))) if np.isfinite(hi_ - lo_)
               else 1e-6 * max(1.0, abs(x)))
        if (np.isfinite(lo_) and x - lo_ < tol) or (np.isfinite(hi_) and hi_ - x < tol):
            at_bound.append(n)
    # Sanity gates: a converged solver can still return a degenerate fit
    # (at low counts the edge can collapse between two channels, leaving a
    # singular covariance and E_F outside the window).
    problems = []
    if at_bound:
        problems.append(f"parameter(s) at a bound: {', '.join(at_bound)}")
    if not np.all(np.isfinite(perr)):
        problems.append("non-finite uncertainty (singular covariance)")
    if not float(np.min(E)) <= result.x[0] <= float(np.max(E)):
        problems.append("E_F outside the fit window")
    if fit_resolution:
        step = float(np.median(np.diff(np.sort(E))))
        if dict(zip(names, result.x))["fwhm_g"] < 0.5 * step:
            problems.append("resolution below half a channel")
    message = "; ".join(problems + [str(result.message)])

    best = unpack(result.x)
    fit_curve = model(result.x, E)
    ss_res = float(np.sum((Y - fit_curve) ** 2))
    ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))

    return FermiEdgeResult(
        ef=float(best["ef"]),
        ef_err=float(err.get("ef", np.nan)),
        resolution=float(best["fwhm_g"]),
        resolution_err=float(err.get("fwhm_g", np.nan)),
        temperature=float(best["temperature"]),
        temperature_err=float(err.get("temperature", np.nan)),
        amplitude=float(best["amplitude"]),
        amplitude_err=float(err.get("amp", np.nan)),
        dos_c1=float(best["dos_c1"]),
        dos_c2=float(best["dos_c2"]),
        bg_const=float(best["bg_const"]),
        bg_slope=float(best["bg_slope"]),
        width_1090=_width_1090(best["ef"], best["fwhm_g"], best["temperature"], convention),
        energy=E,
        fit_curve=fit_curve,
        background=best["bg_const"] + best["bg_slope"] * (E - e_ref),
        residual=Y - fit_curve,
        reduced_chi2=red_chi2,
        r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        success=bool(result.success) and not problems,
        convention=convention,
        dos=dos,
        message=message,
    )


def differential_ef(energy_a, intensity_a, energy_b, intensity_b, **kwargs) -> dict:
    """Fit two spectra with the same model and window; return E_F(B) - E_F(A).

    For rigid edge shifts (charging, bias, a second reference). The error
    is the quadrature sum of the two 1-sigma errors, which assumes the two
    fits are independent; systematic errors common to both cancel in the
    difference only if the same ``window`` and model are used, which is
    why ``kwargs`` are shared.

    Returns:
        dict with ``delta_ef``, ``delta_ef_err``, ``ef_a``, ``ef_a_err``,
        ``ef_b``, ``ef_b_err``, ``fit_a``, ``fit_b``.
    """
    res_a = fit_fermi_edge(energy_a, intensity_a, **kwargs)
    res_b = fit_fermi_edge(energy_b, intensity_b, **kwargs)
    return {
        "delta_ef": float(res_b.ef - res_a.ef),
        "delta_ef_err": float(np.hypot(res_a.ef_err, res_b.ef_err)),
        "ef_a": res_a.ef, "ef_a_err": res_a.ef_err,
        "ef_b": res_b.ef, "ef_b_err": res_b.ef_err,
        "fit_a": res_a, "fit_b": res_b,
    }


def to_binding_energy(energy: NDArray, ef: float, convention: str = "BE") -> NDArray:
    """Binding-energy axis with the fitted E_F at 0 eV.

    ``BE = E - E_F`` on a binding-energy axis and ``E_F - E`` on a kinetic
    one. Applies to other spectra only if they share the reference's
    energy scale (same analyser settings and photon energy, no drift).
    """
    e = np.asarray(energy, dtype=float)
    return (e - ef) if _sign_of(convention) > 0 else (ef - e)
