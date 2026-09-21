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
  because the weights are taken from the data. With the **default unit
  weights** on counts that span the background and the plateau,
  ``ef_err`` runs 9 to 20% below the actual scatter of E_F; when
  ``intensity`` is raw counts, ``poisson_err`` carries the sandwich
  covariance of the same estimator, which does not make that
  assumption.
- **The DOS is flat below E_F and rises above it**, so its slope changes
  at E_F. That kink is an assumption about the sample, and it is not
  free: on simulated data whose DOS instead runs smoothly through E_F,
  the fitted resolution comes out low by 0.19 to 1.55 of its own error
  bar once kT is comparable with the resolution, and at kT/sigma = 0.1
  not at all. Past that it stops being merely biased: it collapses onto
  its lower bound and the fit reports itself a failure, naming the
  parameter. A DOS with curvature through E_F that ``dos='linear'``
  cannot hold moved E_F by up to 49 meV -- an energy-axis calibration
  error, not a resolution one -- which ``dos='quadratic'`` removes.
  Refit with both and treat the spread as a systematic:
  :func:`compare_dos_forms` does that for the two ``dos_form`` values,
  and the same reasoning applies to ``dos`` (see "Compare DOS models on
  real data" below).
- **E_F is where the fitted Fermi-Dirac function is 1/2 before
  broadening**; it is not the inflection point of the measured edge,
  which a sloped DOS moves.
- **The window must stop before the next band.** For Au the 5d bands
  set in about 2 eV below E_F; a window reaching into them cannot be
  described by a low-order DOS and the fit fails (typically with the
  resolution running to its upper bound). ``success`` is False, with
  the reason in ``message``, when a parameter ends on a bound, an
  uncertainty is not finite, E_F falls outside the window or its 1-sigma
  error is wider than the window or than the fitted edge's own 10-90%
  width (the data do not place the edge within its own width), or the
  fitted resolution is below half a channel. When the resolution is below
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
  Gaussian, no-lifetime assumptions. On simulated data where the true
  DOS is known, the spread between the two ``dos_form`` values recovered
  54 to 103% of the actual bias of the resolution, worst where the
  misspecification is strongest. That is the evidence for using a
  spread this way, and it says to read it as a *lower bound* on the
  systematic; it is also evidence from one model family over one range
  of conditions, not a general result.

Origin: a generalisation of ``toyomacro.lineshape.FermiDirac`` (same
model on a binding-energy axis, :func:`fermi_edge` adds kinetic axes and
a background) and of a two-stage valence-band analysis that fixed the
temperature to find the DOS shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

__all__ = [
    "DOS_FORMS",
    "KB_EV",
    "DosFormComparison",
    "FermiEdgeResult",
    "compare_dos_forms",
    "detect_convention",
    "differential_ef",
    "fermi_edge",
    "fit_fermi_edge",
    "to_binding_energy",
]

#: Boltzmann constant, eV/K (CODATA 2018: 8.617333262e-5)
KB_EV = 8.617333262e-5

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
_MAX_REFINEMENT = 1000

#: Where the DOS polynomial applies. ``'occupied'`` is the default and
#: the historical behaviour: flat below E_F, so the slope changes there.
#: ``'both_sides'`` continues the same polynomial through E_F.
DOS_FORMS = ("occupied", "both_sides")


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


def _broaden_fft(grid: NDArray, profile: NDArray, fwhm_g: float) -> NDArray:
    """The kernel ``_broaden`` uses (sampled, cut at 4 sigma, unit sum), applied by FFT.

    Zero padding instead of 'nearest' at the ends: callers keep every
    point they read at least the kernel radius inside the grid, where the
    two agree to rounding. For the long kernels of a refined grid.
    """
    sigma_pts = fwhm_g * _FWHM_TO_SIGMA / float(grid[1] - grid[0])
    radius = int(4.0 * sigma_pts + 0.5)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_pts) ** 2)
    return fftconvolve(profile, kernel / kernel.sum(), mode="same")


def _refinement(grid: NDArray, fwhm_g: float, temperature: float) -> int:
    """Integer refinement of a uniform grid so its step is at most kT and sigma.

    The occupation is sampled before it is convolved. On a step coarser
    than kT the samples cannot follow the edge: the model stops depending
    on E_F inside a channel, and on T, and a fitted E_F is pulled toward
    the nearest grid point. Capped at 1000; 1 where the grid is already
    fine enough, which leaves the model as it was.
    """
    step = float(grid[1] - grid[0])
    scale = KB_EV * temperature
    if fwhm_g > 1e-9:
        scale = min(scale, fwhm_g * _FWHM_TO_SIGMA)
    return int(min(max(np.ceil(step / scale), 1.0), _MAX_REFINEMENT))


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
    dos_form: str = "occupied",
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
        dos_form: Where the DOS polynomial applies. ``'occupied'``
            (default) puts it on the occupied side only, so its slope
            changes at E_F; ``'both_sides'`` continues the same
            polynomial smoothly through E_F. The two differ only where
            the occupation is neither 0 nor 1, i.e. within a few kT of
            E_F, so they are indistinguishable when kT is far below the
            resolution and not otherwise.

    Returns:
        Model intensity on ``energy``, in its original order.

    The Gaussian convolution is evaluated on a grid padded by 4 sigma
    beyond both ends of ``energy``, so channels near the ends see the
    edge profile that continues past them instead of a clamped copy of
    the end value. A spectrum cut to a fit window is therefore modelled
    as the window of a full spectrum.

    Where the step of that grid is coarser than kT or the Gaussian
    sigma, the occupation is evaluated on a grid refined by an integer
    factor (at most 1000) and read back at the original points, so the
    model follows E_F inside a channel and follows T when kT is below
    a channel. Where the grid is fine enough the model is unchanged.
    """
    e = np.asarray(energy, dtype=float)
    s = _sign_of(convention)
    t_k = max(float(temperature), 1.0)
    if dos_form not in DOS_FORMS:
        raise ValueError(f"unknown dos_form {dos_form!r} (use one of {DOS_FORMS})")

    order = np.argsort(e)
    e_sorted = e[order]
    grid, inner = _padded_grid(e_sorted, 4.0 * float(fwhm_g) * _FWHM_TO_SIGMA)
    m = _refinement(grid, float(fwhm_g), t_k) if grid.size >= 2 else 1
    fine = grid if m == 1 else grid[0] + (float(grid[1] - grid[0]) / m) * np.arange(
        (grid.size - 1) * m + 1)
    u_signed = s * (fine - ef)
    u = np.maximum(u_signed, 0.0) if dos_form == "occupied" else u_signed
    dos = np.maximum(1.0 + dos_c1 * u + dos_c2 * u * u, 0.0)
    arg = np.clip(-u_signed / (KB_EV * t_k), -700.0, 700.0)
    occupation = 1.0 / (1.0 + np.exp(arg))

    profile = amplitude * dos * occupation
    if m == 1:
        profile = _broaden(fine, profile, float(fwhm_g))
    elif fwhm_g > 1e-9:
        profile = _broaden_fft(fine, profile, float(fwhm_g))[::m]
    else:
        profile = profile[::m]
    profile = profile[inner] if inner is not None else np.interp(e_sorted, grid, profile)
    profile = profile[np.argsort(order)]
    ref = float(np.mean(e)) if bg_ref is None else float(bg_ref)
    return profile + bg_const + bg_slope * (e - ref)


#: internal parameter name -> the result field it belongs to
_PUBLIC_NAME = {"amp": "amplitude", "fwhm_g": "resolution"}


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
    dos_form: str = "occupied"
    message: str = ""
    poisson_err: dict[str, float] = field(default_factory=dict)
    """1-sigma from the sandwich covariance, **valid only if ``intensity``
    was raw counts**, keyed by the names above (``'ef'``, ``'resolution'``,
    ``'temperature'``, ``'amplitude'``, ``'dos_c1'``, ``'dos_c2'``,
    ``'bg_const'``, ``'bg_slope'``), for the parameters that were fitted.
    **Empty when ``intensity`` is not non-negative integers**, since
    nothing else here can tell raw counts from counts per second or from
    a spectrum that has had a background subtracted, and the sandwich is
    wrong by sqrt(dwell) on the wrong scale. Also empty if the
    covariance was singular.

    The ``*_err`` fields are what a least-squares solver reports: the
    covariance scaled by the reduced chi-squared, which is the right
    answer only if every channel had the same variance. Poisson channels
    do not -- on a Fermi edge the mean runs from the background to the
    plateau. Over a 53-fold range of it, nine runs of 400 simulated
    realisations (three kT/sigma x three seeds) with the temperature
    fixed, ``ef_err`` came out 9 to 20% smaller than the actual scatter
    of E_F while this field was within 10% of it. For the resolution the
    two agree with each other and are within 25% of the scatter; the
    worst of that is at kT/sigma = 3, where a sixth of the fits are
    rejected and the survivors are a selected subset.

    With the temperature *fitted* neither describes the scatter: the
    estimator is then pinned by its bounds and the linearisation both
    rest on does not apply. See
    ``toyomacro.fitting.fermi_edge_identifiability`` for whether the
    temperature and the resolution can be separated at all in a given
    measurement.

    This is *not* a Cramer-Rao bound: this function computes a weighted
    least-squares estimator, not a maximum-likelihood one, and with the
    default unit weights its spread is about a third above the bound. The
    ``*_err`` fields are unchanged; which of the two to quote is the
    caller's decision."""

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
    dos_form: str = "occupied",
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
        dos_form: ``'occupied'`` (default, and what earlier versions did)
            or ``'both_sides'``; see :func:`fermi_edge`. Do not choose
            between them from the data -- fit both and use
            :func:`compare_dos_forms`, whose spread is a guide to the
            systematic the choice carries.
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
    if dos_form not in DOS_FORMS:
        raise ValueError(f"unknown dos_form {dos_form!r} (use one of {DOS_FORMS})")
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
    # (internal names; `poisson_err` is keyed by the result's own field names)
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
        return fermi_edge(e, convention=convention, bg_ref=e_ref, dos_form=dos_form,
                          **unpack(p))

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
    at_bound = [n for n, a in zip(names, result.active_mask) if a != 0]
    for n, x, lo_, hi_ in zip(names, result.x, lb, ub):
        if n in at_bound:
            continue
        tol = (1e-3 * min(hi_ - lo_, max(1.0, abs(x))) if np.isfinite(hi_ - lo_)
               else 1e-6 * max(1.0, abs(x)))
        if (np.isfinite(lo_) and x - lo_ < tol) or (np.isfinite(hi_) and hi_ - x < tol):
            at_bound.append(n)
    # Sanity gates: a converged solver can still return a degenerate fit
    # (at low counts the edge can collapse between two channels, leaving a
    # singular covariance and E_F outside the window).
    best = unpack(result.x)
    width_1090 = _width_1090(best["ef"], best["fwhm_g"], best["temperature"], convention)
    problems = []
    if at_bound:
        problems.append(f"parameter(s) at a bound: {', '.join(at_bound)}")
    if not np.all(np.isfinite(perr)):
        problems.append("non-finite uncertainty (singular covariance)")
    if not float(np.min(E)) <= result.x[0] <= float(np.max(E)):
        problems.append("E_F outside the fit window")
    elif perr[0] > span:
        problems.append("E_F undetermined (1-sigma wider than the window)")
    elif perr[0] > width_1090:
        problems.append("E_F undetermined (1-sigma wider than the edge's own 10-90% width)")
    if fit_resolution:
        step = float(np.median(np.diff(np.sort(E))))
        if dict(zip(names, result.x))["fwhm_g"] < 0.5 * step:
            problems.append("resolution below half a channel")
    message = "; ".join(problems + [str(result.message)])

    fit_curve = model(result.x, E)
    # A second set of 1-sigma errors, for intensities that are raw counts.
    # `perr` above is the least-squares covariance scaled by the reduced
    # chi-squared, which is exact only if every channel had the same
    # variance; Poisson channels do not, and the difference is not small
    # (see the module docstring). This is the sandwich covariance of the
    # same estimator, (G'WG)^-1 G'W diag(mu) WG (G'WG)^-1, with mu the
    # fitted model. result.jac is G scaled by sqrt(W), so G'WG = J'J and
    # the middle factor is J' diag(W mu) J.
    # Only for raw counts. Nothing else here can tell counts from
    # counts-per-second or from a background-subtracted spectrum, and a
    # sandwich built on the wrong scale is wrong by sqrt(dwell) -- a larger
    # error than the one this field exists to remove. Non-integer
    # intensities leave it empty rather than quietly wrong.
    poisson_err: dict[str, float] = {}
    counts_like = bool(np.all(Y >= 0.0) and np.all(Y == np.rint(Y)))
    try:
        if not counts_like:
            raise np.linalg.LinAlgError
        a_inv = np.linalg.inv(result.jac.T @ result.jac)
        middle = result.jac.T @ (result.jac * (W * np.maximum(fit_curve, 0.0))[:, np.newaxis])
        sandwich = np.sqrt(np.clip(np.diag(a_inv @ middle @ a_inv), 0.0, np.inf))
        poisson_err = {_PUBLIC_NAME.get(n, n): float(v) for n, v in zip(names, sandwich)}
    except np.linalg.LinAlgError:
        pass

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
        width_1090=width_1090,
        energy=E,
        fit_curve=fit_curve,
        background=best["bg_const"] + best["bg_slope"] * (E - e_ref),
        residual=Y - fit_curve,
        reduced_chi2=red_chi2,
        r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        success=bool(result.success) and not problems,
        convention=convention,
        dos=dos,
        dos_form=dos_form,
        message=message,
        poisson_err=poisson_err,
    )


#: Scalar fit results :func:`compare_dos_forms` reports across DOS forms.
_COMPARED = ("ef", "resolution", "temperature", "amplitude", "dos_c1", "dos_c2",
             "bg_const", "bg_slope", "width_1090")


@dataclass
class DosFormComparison:
    """One spectrum fitted with several DOS forms, and how far they disagree.

    ``spread`` is **a guide to the systematic carried by the DOS-form
    assumption**, not a statistical error. Keep it separate from
    ``*_err`` and ``poisson_err``: do not add the two in quadrature, and
    do not report one in place of the other.

    Nothing here averages the fits, picks one of them, or ranks them by
    goodness of fit. Two forms that describe the data about equally well
    can disagree by more than either one's error bar, and that
    disagreement is the point.

    Attributes:
        forms: The DOS forms fitted, in the order given
        fits: form -> its :class:`FermiEdgeResult`
        values: parameter -> form -> fitted value
        spread: parameter -> largest minus smallest across the forms
        success: True only if every fit succeeded; when it is False the
            spread mixes a systematic with a failure and means nothing

    How far the spread can be trusted as the systematic was measured on
    simulated data, and only there. On a window of +-0.6 eV in steps of
    0.01 eV, 2000 counts per channel on the plateau over a background of
    50, with v + (pi^2/3)(kT)^2 held at (0.1 eV)^2, swept over kT/sigma
    from 0.1 to 3 and DOS changes across the window from 0.02 to 0.9:
    over the 66 cells where both fits succeed and the bias exceeds
    0.1 meV, **the spread recovers 54 to 103% of the true bias**. It
    works because the bias is antisymmetric in which form is wrong, and
    it works less well the further that antisymmetry goes -- the
    recovery falls smoothly with both kT/sigma and the DOS slope, and at
    its worst corner, kT/sigma around 2 to 2.5 with a steep DOS, it
    understates the systematic by a factor of 1.9. **Read the spread as
    a lower bound on the systematic, not an estimate of it.**

    Two limits on that. It is one model family over one range of
    conditions, not a general result. And the range that could be tested
    at all is capped: a linear DOS through E_F reaches zero at the
    window edge once its change across the window reaches 1, which holds
    the slope to half of what the identifiability scan uses.
    """

    forms: tuple[str, ...]
    fits: dict[str, FermiEdgeResult]
    values: dict[str, dict[str, float]]
    spread: dict[str, float]
    success: bool

    def summary(self) -> str:
        """One line: the spread of E_F and the resolution across the forms."""
        state = "" if self.success else " [a fit failed: the spread is meaningless]"
        return (f"DOS forms {', '.join(self.forms)} | E_F spread "
                f"{self.spread['ef']:.4f} eV | FWHM_G spread "
                f"{self.spread['resolution']:.4f} eV{state}")


def compare_dos_forms(
    energy: NDArray,
    intensity: NDArray,
    *,
    forms: tuple[str, ...] = DOS_FORMS,
    **kwargs,
) -> DosFormComparison:
    """Fit one spectrum with each DOS form and report how far they disagree.

    The DOS form is an assumption about the sample that the data near
    E_F constrain weakly, so choosing between the forms by fit quality
    substitutes a guess for a measurement. Fitting both and reporting
    the spread does not.

    Args:
        energy, intensity: One spectrum, as for :func:`fit_fermi_edge`
        forms: DOS forms to fit; any of :data:`DOS_FORMS`
        kwargs: Passed unchanged to :func:`fit_fermi_edge` (``dos_form``
            is not accepted -- ``forms`` sets it)

    Returns:
        DosFormComparison. Read ``spread`` beside the statistical error,
        never instead of it or combined with it.
    """
    if "dos_form" in kwargs:
        raise TypeError("compare_dos_forms sets dos_form itself; pass `forms` instead")
    forms = tuple(forms)
    if len(forms) < 2 or len(set(forms)) != len(forms) or set(forms) - set(DOS_FORMS):
        raise ValueError(f"forms must be two or more distinct entries of {DOS_FORMS}, "
                         f"got {forms}")
    fits = {form: fit_fermi_edge(energy, intensity, dos_form=form, **kwargs)
            for form in forms}
    values = {name: {form: float(getattr(fits[form], name)) for form in forms}
              for name in _COMPARED}
    spread = {name: max(v.values()) - min(v.values()) for name, v in values.items()}
    return DosFormComparison(forms=forms, fits=fits, values=values, spread=spread,
                             success=all(f.success for f in fits.values()))



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
