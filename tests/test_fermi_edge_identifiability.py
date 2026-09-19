"""toyomacro.fitting.fermi_edge_identifiability: the edge and its derivatives.

Implementation fidelity: both evaluation routes against an adaptive
quadrature (scipy.integrate.quad) of the same integrals, written
independently here; against finite differences of the value; against
``fermi_edge``, the model the fitter uses. The known structure of the
problem: the cumulants of the thermal kernel, the tau -> 0 and v -> 0
limits and the rates at which they are reached.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate
from scipy.special import expit

from toyomacro.fitting import fermi_edge_identifiability as fi
from toyomacro.fitting.fermi_edge import KB_EV, fermi_edge

SIGMA = 0.05
E = np.arange(-0.6, 0.6 + 1e-9, 0.005)
COLUMNS = ("value", "d_ef", "d_variance", "d_dos_c1", "d_dos_c2", "d_tau")


def _stack(d):
    return np.stack([getattr(d, c) for c in COLUMNS], axis=1)


def _derivs(energy=E, ef=0.0031, variance=SIGMA**2, tau=0.0, c1=0.35, c2=0.0,
            form="occupied", route="auto", convention="BE"):
    return fi.edge_derivatives(energy, ef, 1.0, variance, tau, dos_c1=c1, dos_c2=c2,
                               dos_form=form, route=route, convention=convention)


def _relative(a, b):
    """Worst column error, each column judged against its own largest magnitude."""
    return float((np.abs(a - b).max(0) / np.maximum(np.abs(b).max(0), 1e-300)).max())


def _reference(u, variance, tau, c1, c2, form):
    """Columns at one point by adaptive quadrature, unit amplitude, BE (d_ef = -d/du)."""
    kt = math.sqrt(tau)

    def gauss(y):
        return math.exp(-0.5 * (u - y) ** 2 / variance) / math.sqrt(2.0 * math.pi * variance)

    def pieces(y):
        f = float(expit(y / kt))
        f1 = f * (1.0 - f) / kt
        f2 = f1 * (1.0 - 2.0 * f) / kt
        if form == "both_sides":
            dc, d, d1, d2 = y, 1.0 + c1 * y + c2 * y * y, c1 + 2.0 * c2 * y, 2.0 * c2
        else:
            occupied = y > 0.0
            dc = y if occupied else 0.0
            d = 1.0 + c1 * dc + c2 * dc * dc
            d1 = c1 + 2.0 * c2 * y if occupied else 0.0
            d2 = 2.0 * c2 if occupied else 0.0
        return (d * f, d1 * f + d * f1, d2 * f + 2.0 * d1 * f1 + d * f2, dc * f, dc * dc * f,
                d * (-(y / (2.0 * tau)) * f1))

    lo, hi = u - 12.0 * math.sqrt(variance), u + 12.0 * math.sqrt(variance)
    points = sorted({p for p in (-3.0 * kt, -kt, 0.0, kt, 3.0 * kt, u) if lo < p < hi})
    out = [integrate.quad(lambda y, k=k: gauss(y) * pieces(y)[k], lo, hi, points=points,
                          epsabs=0.0, epsrel=1e-13, limit=2000)[0] for k in range(6)]
    kink = 0.5 * c1 * gauss(0.0) if form == "occupied" else 0.0
    return np.array([out[0], -out[1], 0.5 * (out[2] + kink), out[3], out[4], out[5]])


# --- thermal kernel -----------------------------------------------------------


def test_cumulants_of_the_thermal_kernel():
    """-df/dy is a logistic density of scale kT: kappa_2 = (pi^2/3) (kT)^2,
    kappa_4 = (2 pi^4/15) (kT)^4 (excess kurtosis 6/5), by quadrature."""
    kt = KB_EV * 300.0
    y = np.linspace(-80.0 * kt, 80.0 * kt, 400001)
    f = expit(y / kt)
    density = f * (1.0 - f) / kt
    m2 = np.trapezoid(y**2 * density, y)
    m4 = np.trapezoid(y**4 * density, y)
    assert np.trapezoid(density, y) == pytest.approx(1.0, rel=1e-10)
    assert m2 == pytest.approx(math.pi**2 / 3.0 * kt**2, rel=1e-9)
    assert m4 - 3.0 * m2**2 == pytest.approx(2.0 * math.pi**4 / 15.0 * kt**4, rel=1e-8)


def test_temperature_conversions():
    assert fi.tau_from_temperature(300.0) == pytest.approx((KB_EV * 300.0) ** 2, rel=1e-15)
    assert fi.temperature_from_tau(fi.tau_from_temperature(12.5)) == pytest.approx(12.5)


# --- both routes against an independent quadrature ------------------------------


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("ratio", [0.02, 0.07, 0.1, 0.5, 1.0, 3.0])
@pytest.mark.parametrize("form,c1,c2", [("occupied", 0.0, 0.0), ("occupied", 0.35, 0.0),
                                        ("occupied", -0.3, 0.2), ("both_sides", 0.35, 0.0),
                                        ("both_sides", -0.3, 0.2)])
def test_routes_match_adaptive_quadrature(ratio, form, c1, c2):
    """kT/sigma from 0.02 to 3, 17 energies across +-0.4 eV. Measured: at
    most 2.6e-13 for the quadrature route (at kT/sigma = 0.02, where the
    d/dtau integrand cancels, in both it and the reference) and 6e-13 for
    the series (at 0.1, above its switch). The linear DOS continued below
    E_F turns negative inside the thermal tail at kT/sigma = 3; that case
    is refused, not evaluated."""
    variance, tau = SIGMA**2, (ratio * SIGMA) ** 2
    u = np.linspace(-0.4, 0.4, 17) + 0.0031
    if form == "both_sides" and c1 > 0 and c2 == 0 and ratio >= 3.0:
        # 1 + 0.35 y is negative below y = -2.9 eV, inside 37 kT = 5.6 eV
        with pytest.raises(ValueError, match="not positive"):
            _derivs(u, 0.0, variance, tau, c1, c2, form)
        return
    ref = np.array([_reference(x, variance, tau, c1, c2, form) for x in u])
    routes = ("quadrature", "series") if ratio <= 0.1 else ("quadrature",)
    for route in routes:
        got = _stack(_derivs(u, 0.0, variance, tau, c1, c2, form, route))
        assert _relative(got, ref) < 1e-12, route


def test_the_routes_agree_around_the_switch():
    """kT/sigma 0.03-0.1: both at their floor (measured <= 8.5e-13)."""
    for ratio in (0.03, 0.05, 0.069, 0.071, 0.1):
        tau = (ratio * SIGMA) ** 2
        q = _stack(_derivs(tau=tau, c2=0.1, route="quadrature"))
        s = _stack(_derivs(tau=tau, c2=0.1, route="series"))
        assert _relative(q, s) < 2e-12, ratio


def test_auto_takes_the_series_where_the_quadrature_cancels():
    """Below the switch the d/dtau integrand is odd about E_F with a
    magnitude of 1/kT, so the quadrature loses about eps * sigma/kT:
    measured against the series 9e-12 at kT/sigma = 1e-3 and 9e-10 at
    1e-5. 'auto' uses the series there and the quadrature above."""
    for ratio in (1e-5, 1e-3, 0.069):
        tau = (ratio * SIGMA) ** 2
        assert np.array_equal(_stack(_derivs(tau=tau)), _stack(_derivs(tau=tau, route="series")))
    tau = (0.071 * SIGMA) ** 2
    assert np.array_equal(_stack(_derivs(tau=tau)), _stack(_derivs(tau=tau, route="quadrature")))
    q = _stack(_derivs(tau=(1e-5 * SIGMA) ** 2, route="quadrature"))
    s = _stack(_derivs(tau=(1e-5 * SIGMA) ** 2, route="series"))
    assert 1e-11 < _relative(q, s) < 1e-8


def test_series_is_asymptotic_and_its_term_count_stays_below_the_limit():
    """Term n carries He_n(t) phi(t) (kT/sigma)**(n+1); He_n phi grows like
    sqrt(n!), so terms shrink only while n < (sigma/kT)**2. The term count
    must stay below that at the switch, and the series fails where it
    does not: at kT/sigma = 0.3, (sigma/kT)**2 = 11 < 30 terms and the
    series is off by order one, while the quadrature is exact."""
    assert fi._SERIES_TERMS < (1.0 / fi._SERIES_SWITCH) ** 2
    tau = (0.3 * SIGMA) ** 2
    q = _stack(_derivs(tau=tau, c2=0.1, route="quadrature"))
    s = _stack(_derivs(tau=tau, c2=0.1, route="series"))
    assert _relative(s, q) > 0.1


# --- limits -----------------------------------------------------------------------


@pytest.mark.parametrize("form,c1,c2,per_decade", [("occupied", 0.0, 0.0, 100.0),
                                                    ("both_sides", 0.35, 0.1, 100.0),
                                                    ("occupied", 0.35, 0.0, 10.0)])
def test_d_tau_reaches_its_zero_temperature_limit(form, c1, c2, per_decade):
    """d mu/dtau at tau = 0 is closed form, A [(pi^2/6) G' - (pi^2/12) c1 G]
    for the occupied-side DOS ((pi^2/6) c1 G on both sides). Small tau
    approaches it at rate tau for a smooth DOS -- the Sommerfeld expansion
    has even powers of kT only -- but at rate sqrt(tau) for the occupied-
    side DOS, whose kink at E_F adds a tau**(3/2) term to the mean.
    Measured gap ratios per decade of kT: 100, 100 and 9-11."""
    zero = _derivs(tau=0.0, c1=c1, c2=c2, form=form).d_tau
    gaps = [np.abs(_derivs(tau=(r * SIGMA) ** 2, c1=c1, c2=c2, form=form).d_tau - zero).max()
            for r in (1e-2, 1e-3, 1e-4)]
    for a, b in zip(gaps[:-1], gaps[1:]):
        assert 0.75 * per_decade < a / b < 1.4 * per_decade


def test_small_variance_reaches_the_pointwise_limit():
    """At v = 0 the columns are the integrands at the energy point. Away
    from E_F (here by 20 sigma or more) small v approaches that at rate v."""
    tau = fi.tau_from_temperature(300.0)
    far = E[np.abs(E - 0.0031) > 0.01]
    zero = _stack(_derivs(far, variance=0.0, tau=tau))
    gaps = [_relative(_stack(_derivs(far, variance=s * s, tau=tau)), zero) for s in (1e-4, 1e-5)]
    assert gaps[0] < 1e-4 and 50.0 < gaps[0] / gaps[1] < 200.0


def test_the_dos_forms_differ_only_through_the_thermal_tail():
    """Below E_F the two forms differ, but the step kills y < 0: at tau = 0
    D times the step is (1 + c1 y) for y > 0 in both, so every column but
    d/dtau is the same, and d/dtau differs by (pi^2/12) c1 G_v(u), the
    missing tail. At small tau the values differ at rate tau (the tail's
    weight). The slope change at E_F that D times the step has in both
    forms is also why d/dv carries (c1/2) G_v(u) as tau goes to 0, in both."""
    occ, both = _derivs(tau=0.0), _derivs(tau=0.0, form="both_sides")
    for name in COLUMNS[:-1]:
        np.testing.assert_array_equal(getattr(occ, name), getattr(both, name))
    gau = np.exp(-0.5 * (E - 0.0031) ** 2 / SIGMA**2) / math.sqrt(2 * math.pi * SIGMA**2)
    np.testing.assert_allclose(occ.d_tau - both.d_tau, math.pi**2 / 12 * 0.35 * gau,
                               atol=1e-12 * gau.max())
    gaps = [np.abs(_derivs(tau=(r * SIGMA) ** 2).value
                   - _derivs(tau=(r * SIGMA) ** 2, form="both_sides").value).max()
            for r in (1e-2, 1e-3)]
    assert 80.0 < gaps[0] / gaps[1] < 120.0


# --- fidelity to the value and to the fitter ---------------------------------------


def test_columns_are_derivatives_of_the_value():
    """Central differences, relative step 1e-5; measured 1e-10 to 2e-9."""
    base = dict(ef=0.0031, c1=0.35, c2=0.1, variance=SIGMA**2, tau=fi.tau_from_temperature(100))
    d = _derivs(ef=base["ef"], variance=base["variance"], tau=base["tau"], c1=0.35, c2=0.1)

    def value(**kw):
        p = dict(base, **kw)
        return _derivs(ef=p["ef"], variance=p["variance"], tau=p["tau"], c1=p["c1"],
                       c2=p["c2"]).value

    for key, column in (("ef", d.d_ef), ("c1", d.d_dos_c1), ("c2", d.d_dos_c2),
                        ("variance", d.d_variance), ("tau", d.d_tau)):
        h = 1e-5 * max(abs(base[key]), 1e-3 if key == "ef" else 1e-6)
        fd = (value(**{key: base[key] + h}) - value(**{key: base[key] - h})) / (2 * h)
        assert np.abs(fd - column).max() < 1e-8 * np.abs(column).max(), key


@pytest.mark.parametrize("step,fwhm,temperature", [(0.02, 0.3, 300.0), (0.005, 0.05, 30.0),
                                                   (0.02, 0.05, 10.0)])
@pytest.mark.parametrize("c1", [0.0, 0.35])
def test_value_is_the_fitters_model(step, fwhm, temperature, c1):
    """Same model as fermi_edge up to the fitter's sampled Gaussian kernel,
    which is good to about 3e-5 of the step (measured 2.4e-5 to 3.2e-5)."""
    e = np.arange(-1.0, 1.0 + 1e-9, step)
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    ours = fi.edge_derivatives(e, 0.37 * step, 1.0, sigma**2, fi.tau_from_temperature(temperature),
                               dos_c1=c1).value
    theirs = fermi_edge(e, 0.37 * step, fwhm_g=fwhm, temperature=temperature, dos_c1=c1)
    assert np.abs(ours - theirs).max() < 1e-4


def test_kinetic_axis_is_the_mirror():
    tau = fi.tau_from_temperature(80.0)
    be = _derivs(E, ef=0.0031, tau=tau)
    ke = _derivs(-E, ef=-0.0031, tau=tau, convention="KE")
    for name in COLUMNS:
        sign = -1.0 if name == "d_ef" else 1.0
        np.testing.assert_allclose(getattr(ke, name), sign * getattr(be, name), rtol=1e-12,
                                   atol=1e-12 * np.abs(getattr(be, name)).max())


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="bare step"):
        _derivs(variance=0.0, tau=0.0)
    with pytest.raises(ValueError):
        _derivs(variance=-1e-6, tau=1e-4)
    with pytest.raises(ValueError, match="not positive"):
        _derivs(tau=1e-4, c1=-5.0)
    with pytest.raises(ValueError, match="delta"):
        fi.edge_derivatives(np.array([-0.1, 0.0, 0.1]), 0.0, 1.0, 0.0, 1e-4, dos_c1=0.35)
    with pytest.raises(ValueError, match="dos_form"):
        _derivs(tau=1e-4, form="kinked")
    with pytest.raises(ValueError, match="route"):
        _derivs(tau=1e-4, route="fast")
