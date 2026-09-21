"""toyomacro.fitting.fermi_edge_identifiability: the edge and its derivatives.

Implementation fidelity: both evaluation routes against an adaptive
quadrature (scipy.integrate.quad) of the same integrals, written
independently here; against finite differences of the value; against
``fermi_edge``, the model the fitter uses. The known structure of the
problem: the cumulants of the thermal kernel, the tau -> 0 and v -> 0
limits and the rates at which they are reached.
"""

from __future__ import annotations

import dataclasses
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
    the series (at 0.1, above its switch)."""
    variance, tau = SIGMA**2, (ratio * SIGMA) ** 2
    u = np.linspace(-0.4, 0.4, 17) + 0.0031
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
        _derivs(tau=1e-4, c1=-5.0)       # falls through zero on the occupied side
    with pytest.raises(ValueError, match="not positive"):
        _derivs(tau=1e-4, c1=200.0, form="both_sides")  # and below E_F, within 3 kT
    with pytest.raises(ValueError, match="delta"):
        fi.edge_derivatives(np.array([-0.1, 0.0, 0.1]), 0.0, 1.0, 0.0, 1e-4, dos_c1=0.35)
    with pytest.raises(ValueError, match="dos_form"):
        _derivs(tau=1e-4, form="kinked")
    with pytest.raises(ValueError, match="route"):
        _derivs(tau=1e-4, route="fast")


# --- Poisson Fisher matrix -------------------------------------------------------------

from toyomacro._identifiability import effective_information  # noqa: E402


def _edge(**kw):
    base = dict(ef=0.0031, amplitude=1000.0, sigma=0.05, temperature=300.0, dos_c1=0.35)
    base.update(kw)
    return fi.FermiEdge(**base)


def _unit_free(m):
    d = np.sqrt(np.diag(m))
    return m / np.outer(d, d)


def test_axis_rescaling_moves_no_hidden_constant():
    """Energies times c: the value is unchanged and each column scales as its
    coordinate does (E_F by 1/c, v and tau by 1/c**2). Nothing in the columns
    is tied to eV."""
    c, tau = 1000.0, fi.tau_from_temperature(80.0)
    a = fi.edge_derivatives(E, 0.0031, 1.0, SIGMA**2, tau, dos_c1=0.35)
    b = fi.edge_derivatives(c * E, c * 0.0031, 1.0, c * c * SIGMA**2, c * c * tau,
                            dos_c1=0.35 / c)
    np.testing.assert_allclose(b.value, a.value, rtol=1e-12, atol=1e-15)
    for name, power in (("d_ef", 1), ("d_variance", 2), ("d_tau", 2), ("d_dos_c1", -1)):
        scale = np.abs(getattr(a, name)).max()
        np.testing.assert_allclose(getattr(b, name) * c**power, getattr(a, name),
                                   rtol=1e-11, atol=1e-11 * scale)


def test_fisher_is_linear_in_exposure():
    bg = fi.constant_background(E, 50.0)
    one = fi.edge_fisher(E, _edge(), bg)
    three = fi.edge_fisher(E, _edge(), bg, exposure=3.0)
    np.testing.assert_allclose(three.fisher, 3.0 * one.fisher, rtol=1e-13)
    np.testing.assert_allclose(three.expected_counts, 3.0 * one.expected_counts, rtol=1e-15)


@pytest.mark.parametrize("temperature,sigma", [(300.0, 0.05), (30.0, 0.05), (300.0, 0.01)])
@pytest.mark.parametrize("parameterization", ["var_tau", "sigma_T"])
def test_fisher_is_the_hessian_of_the_expected_deviance(temperature, sigma, parameterization):
    """Linear DOS, estimated linear background. Second differences of the
    expected Poisson negative log-likelihood at the truth, written as
    sum(mu - mu0 - mu0 log1p((mu - mu0)/mu0)) so no large constant cancels,
    steps of 1e-3 of each parameter's bound (at most 1e-3 of its value).
    Measured, relative to sqrt(I_ii I_jj): 1e-7 to 2.7e-6."""
    edge = _edge(sigma=sigma, temperature=temperature)
    bg = fi.linear_background(E, 50.0, 3.0)
    fisher = fi.edge_fisher(E, edge, bg, parameterization=parameterization)
    theta0, mu0 = fisher.param_values.copy(), fisher.expected_counts
    mid = 0.5 * (E[0] + E[-1])

    def mean(theta):
        p = dict(zip(fisher.param_names, theta))
        if parameterization == "var_tau":
            v, tau = p["variance"], p["tau"]
        else:
            v, tau = p["sigma"] ** 2, fi.tau_from_temperature(p["temperature"])
        d = fi.edge_derivatives(E, p["ef"], p["amplitude"], v, tau, dos_c1=p["dos_c1"])
        return d.value + p["bg_level"] + p["bg_slope"] * (E - mid)

    def nll(theta):
        dm = mean(theta) - mu0
        return float(np.sum(dm - mu0 * np.log1p(dm / mu0)))

    sd = np.sqrt(np.diag(np.linalg.inv(fisher.fisher)))
    h = np.minimum(1e-3 * sd, 1e-3 * np.maximum(np.abs(theta0), 1e-4))
    n = theta0.size
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = h[i], h[j]
            hess[i, j] = hess[j, i] = (nll(theta0 + ei + ej) - nll(theta0 + ei - ej)
                                       - nll(theta0 - ei + ej) + nll(theta0 - ei - ej)) / (
                4.0 * h[i] * h[j])
    d = np.sqrt(np.diag(fisher.fisher))
    assert np.abs((hess - fisher.fisher) / np.outer(d, d)).max() < 1e-5


def test_the_two_coordinates_are_one_model():
    """I_sigma_T = T^T I_var_tau T with dv/dsigma = 2 sigma, dtau/dT = 2 k^2 T."""
    bg = fi.constant_background(E, 50.0)
    vt = fi.edge_fisher(E, _edge(), bg)
    st = fi.edge_fisher(E, _edge(), bg, parameterization="sigma_T")
    jac = np.eye(vt.fisher.shape[0])
    iv, it = vt.width_index
    jac[iv, iv] = 2.0 * 0.05
    jac[it, it] = 2.0 * KB_EV**2 * 300.0
    np.testing.assert_allclose(st.fisher, jac.T @ vt.fisher @ jac, rtol=1e-12)
    assert st.param_names[iv] == "sigma" and st.param_names[it] == "temperature"


def test_temperature_coordinate_degenerates_at_zero_and_tau_does_not():
    """Flat DOS, constant background, sigma 0.05 eV, T from 20 K down to
    1.25 K: I_TT / T^2 settles (4.965e-7 at 20 K, 5.001e-7 at 1.25 K) --
    the coordinate's own zero -- while I_tau_tau stays finite (2.251e9 to
    2.267e9)."""
    bg = fi.constant_background(E, 50.0)
    ratios, info = [], []
    for t in (20.0, 10.0, 5.0, 2.5, 1.25):
        edge = _edge(temperature=t, dos_c1=0.0, dos="flat")
        st = fi.edge_fisher(E, edge, bg, parameterization="sigma_T")
        vt = fi.edge_fisher(E, edge, bg)
        ratios.append(st.fisher[st.width_index[1], st.width_index[1]] / t**2)
        info.append(vt.fisher[vt.width_index[1], vt.width_index[1]])
    assert abs(ratios[-1] / ratios[-2] - 1.0) < 1e-3 and abs(ratios[0] / ratios[-1] - 1.0) < 0.01
    assert abs(info[-1] / info[-2] - 1.0) < 1e-3 and abs(info[0] / info[-1] - 1.0) < 0.01


def test_the_routes_agree_at_the_fisher_level_around_the_switch():
    """The switch is set where the full Fisher matrix, not a profile, agrees
    between routes. Measured: correlation matrices within 5e-14; the stiff
    eigenvalue of the effective (v, tau) block within 5e-15; the soft one --
    the information that separates v from tau, 3e-7 to 3e-5 of the stiff --
    within 2.3e-10 at kT/sigma = 0.03 and 1.9e-11 at the switch."""
    bg = fi.constant_background(E, 50.0)
    scale = np.diag([1.0, 3.0 / math.pi**2])  # tau -> (pi^2/3) tau
    for ratio, soft_tol in ((0.03, 1e-9), (0.05, 3e-10), (0.069, 1e-10), (0.1, 1e-10)):
        edge = _edge(temperature=ratio * 0.05 / KB_EV)
        q = fi.edge_fisher(E, edge, bg, route="quadrature")
        s = fi.edge_fisher(E, edge, bg, route="series")
        assert np.abs(_unit_free(q.fisher) - _unit_free(s.fisher)).max() < 1e-12
        lq = np.linalg.eigvalsh(scale @ effective_information(q.fisher, q.width_index).effective
                                @ scale)
        ls = np.linalg.eigvalsh(scale @ effective_information(s.fisher, s.width_index).effective
                                @ scale)
        assert abs(lq[1] - ls[1]) < 1e-13 * lq[1]
        assert abs(lq[0] - ls[0]) < soft_tol * lq[0], ratio


def test_dos_terms_and_background_terms_become_parameters():
    bg = fi.linear_background(E, 50.0, 3.0) + fi.constant_background(E, 5.0, estimated=False)
    for dos, names in (("flat", ()), ("linear", ("dos_c1",)), ("quadratic", ("dos_c1", "dos_c2"))):
        edge = _edge(dos=dos, dos_c1=0.0 if dos == "flat" else 0.35,
                     dos_c2=0.1 if dos == "quadratic" else 0.0)
        f = fi.edge_fisher(E, edge, bg)
        assert f.param_names == ("ef", "amplitude", *names, "variance", "tau", "bg_level", "bg_slope")
        assert f.param_roles[-2:] == ("background", "background")
        known = fi.edge_derivatives(E, edge.ef, edge.amplitude, edge.variance, edge.tau,
                                    dos_c1=edge.dos_c1, dos_c2=edge.dos_c2).value
        np.testing.assert_allclose(f.expected_counts, known + bg.counts(), rtol=1e-14)


def test_edge_and_fisher_inputs_are_checked():
    with pytest.raises(ValueError, match="positive"):
        fi.FermiEdge(ef=0.0, amplitude=0.0, sigma=0.05, temperature=300.0)
    with pytest.raises(ValueError, match="bare step"):
        fi.FermiEdge(ef=0.0, amplitude=1.0, sigma=0.0, temperature=0.0)
    with pytest.raises(ValueError, match="omitted"):
        fi.FermiEdge(ef=0.0, amplitude=1.0, sigma=0.05, temperature=300.0, dos_c1=0.3, dos="flat")
    with pytest.raises(ValueError, match="expected counts"):
        # 2.5 eV below E_F at 1 K is 50 sigma: the mean underflows to 0 with no background
        fi.edge_fisher(np.linspace(-2.5, 0.3, 200), _edge(temperature=1.0), None)
    with pytest.raises(ValueError, match="parameterization"):
        fi.edge_fisher(E, _edge(), fi.constant_background(E, 5.0), parameterization="fwhm")
    with pytest.raises(ValueError, match="exposure"):
        fi.edge_fisher(E, _edge(), fi.constant_background(E, 5.0), exposure=0.0)


# --- the temperature: free, fixed, or with a prior ----------------------------------


@pytest.mark.parametrize("temperature,sigma", [(300.0, 0.05), (30.0, 0.02)])
@pytest.mark.parametrize("parameterization", ["var_tau", "sigma_T"])
def test_temperature_prior_tends_to_fixed_and_to_free(temperature, sigma, parameterization):
    """Bounds on every parameter (square roots of the inverse's diagonal).
    Measured: at sd_T = 1e-6 K within 1e-12 of 'fixed_temperature', at
    1e9 K within 5e-13 of 'free', each approached as sd**2 and 1/sd**2.
    Holding T fixed rather than free shrinks the bounds by up to 6.3x
    (300 K, sigma 0.05 eV) and 56x (30 K, 0.02 eV) here."""
    edge = _edge(temperature=temperature, sigma=sigma)
    bg = fi.constant_background(E, 50.0)

    def bounds(**kw):
        f = fi.edge_fisher(E, edge, bg, parameterization=parameterization, **kw)
        return np.sqrt(np.diag(np.linalg.inv(f.fisher))), f

    free, f_free = bounds()
    fixed, _ = bounds(temperature_mode="fixed_temperature")
    keep = [i for i in range(free.size) if i != f_free.width_index[1]]

    def gap_fixed(sd):
        b, _ = bounds(temperature_mode="temperature_prior", temperature_sd=sd)
        return np.abs(b[keep] / fixed - 1.0).max()

    def gap_free(sd):
        b, _ = bounds(temperature_mode="temperature_prior", temperature_sd=sd)
        return np.abs(b / free - 1.0).max()

    assert gap_fixed(1e-6) < 1e-10 and gap_free(1e9) < 1e-10
    # the rates, from gaps well above the rounding floor (~1e-12)
    assert 50.0 < gap_fixed(1e-2) / gap_fixed(1e-3) < 200.0     # ~ sd**2
    assert 50.0 < gap_free(1e5) / gap_free(1e6) < 200.0         # ~ 1/sd**2
    assert np.max(free[keep] / fixed) > 5.0


def test_prior_is_the_same_prior_in_both_coordinates():
    """A prior of sd_T in K becomes 1/sd_T**2 on T in 'sigma_T' and
    1/(2 k^2 T sd_T)**2 on tau in 'var_tau'; the two matrices are then
    related by I_phi = T^T I T as without a prior (measured 3e-16)."""
    bg = fi.constant_background(E, 50.0)
    kw = dict(temperature_mode="temperature_prior", temperature_sd=5.0)
    vt = fi.edge_fisher(E, _edge(), bg, **kw)
    st = fi.edge_fisher(E, _edge(), bg, parameterization="sigma_T", **kw)
    jac = np.eye(vt.fisher.shape[0])
    iv, it = vt.width_index
    jac[iv, iv], jac[it, it] = 2.0 * 0.05, 2.0 * KB_EV**2 * 300.0
    np.testing.assert_allclose(st.fisher, jac.T @ vt.fisher @ jac, rtol=1e-12)
    assert st.prior[it, it] == pytest.approx(1.0 / 25.0)
    assert np.count_nonzero(vt.prior) == 1


def test_fixed_temperature_is_the_free_matrix_without_its_temperature():
    bg = fi.linear_background(E, 50.0, 3.0)
    free = fi.edge_fisher(E, _edge(), bg)
    fixed = fi.edge_fisher(E, _edge(), bg, temperature_mode="fixed_temperature")
    keep = [i for i in range(len(free.param_names)) if free.param_names[i] != "tau"]
    np.testing.assert_allclose(fixed.fisher, free.fisher[np.ix_(keep, keep)], rtol=1e-14)
    assert "tau" not in fixed.param_names and fixed.width_index == (keep.index(free.width_index[0]),)


def test_the_prior_does_not_scale_with_exposure():
    bg = fi.constant_background(E, 50.0)
    kw = dict(temperature_mode="temperature_prior", temperature_sd=5.0)
    one = fi.edge_fisher(E, _edge(), bg, **kw)
    ten = fi.edge_fisher(E, _edge(), bg, exposure=10.0, **kw)
    np.testing.assert_allclose(ten.fisher - ten.prior, 10.0 * (one.fisher - one.prior), rtol=1e-13)
    np.testing.assert_array_equal(ten.prior, one.prior)


def test_temperature_mode_inputs_are_checked():
    bg = fi.constant_background(E, 50.0)
    with pytest.raises(ValueError, match="temperature_sd"):
        fi.edge_fisher(E, _edge(), bg, temperature_mode="temperature_prior")
    with pytest.raises(ValueError, match="applies to"):
        fi.edge_fisher(E, _edge(), bg, temperature_sd=5.0)
    with pytest.raises(ValueError, match="temperature_mode"):
        fi.edge_fisher(E, _edge(), bg, temperature_mode="known")
    with pytest.raises(ValueError, match="no width in tau"):
        fi.edge_fisher(E, _edge(temperature=0.0), bg, temperature_mode="temperature_prior",
                       temperature_sd=1.0)


# --- effective information on (v, tau), in the edge's own scale ----------------------


def _width_info(temperature=300.0, sigma=0.05, form="occupied", dos="linear", c1=0.35, **kw):
    edge = _edge(temperature=temperature, sigma=sigma, dos=dos, dos_c1=c1)
    return fi.edge_width_information(
        fi.edge_fisher(E, edge, fi.constant_background(E, 50.0), dos_form=form, **kw))


def test_effective_is_the_inverse_of_the_full_inverse_and_below_conditional():
    """inv(effective) is the (v, tau) block of the full inverse Fisher
    matrix, in the same coordinates; conditional - effective is positive
    semi-definite (measured sd ratios effective/conditional 1.18 and 1.17
    for x_v and x_tau at 300 K, sigma 0.05 eV)."""
    edge = _edge()
    f = fi.edge_fisher(E, edge, fi.constant_background(E, 50.0))
    w = fi.edge_width_information(f)
    iv, it = f.width_index
    block = np.linalg.inv(f.fisher)[np.ix_([iv, it], [iv, it])]
    scale_inv = np.diag([1.0 / w.kappa2, (math.pi**2 / 3.0) / w.kappa2])
    np.testing.assert_allclose(np.linalg.inv(w.effective), scale_inv @ block @ scale_inv,
                               rtol=1e-8)
    gap = np.linalg.eigvalsh(w.conditional - w.effective)
    assert gap.min() > -1e-12 * np.abs(w.conditional).max()
    assert math.sqrt(np.linalg.inv(w.effective)[0, 0] / np.linalg.inv(w.conditional)[0, 0]) > 1.1


def test_reported_quantities_do_not_depend_on_the_energy_unit():
    """The same model with energies in meV: each Fisher entry picks up the
    unit factors of its two parameters (E_F 1/c, v and tau 1/c**2, the
    DOS slope c) and the parameter values scale the other way. Every
    reported quantity is unit-free and must not move."""
    f = fi.edge_fisher(E, _edge(), fi.linear_background(E, 50.0, 3.0))
    c = 1000.0
    unit = {"ef": 1 / c, "amplitude": 1.0, "dos_c1": c, "variance": 1 / c**2, "tau": 1 / c**2,
            "bg_level": 1.0, "bg_slope": c}
    d = np.array([unit[n] for n in f.param_names])
    import dataclasses
    g = dataclasses.replace(f, fisher=f.fisher * np.outer(d, d), param_values=f.param_values / d)
    a, b = fi.edge_width_information(f), fi.edge_width_information(g)
    assert b.kappa2 == pytest.approx(c * c * a.kappa2, rel=1e-14)
    np.testing.assert_allclose(b.effective, a.effective, rtol=1e-10)
    for name in ("alignment", "sd_kappa2", "sd_share"):
        assert getattr(b, name) == pytest.approx(getattr(a, name), rel=1e-10)
    np.testing.assert_allclose(b.stiff_direction, a.stiff_direction, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("form,dos,c1", [("occupied", "linear", 0.35),
                                         ("both_sides", "linear", 0.35),
                                         ("occupied", "flat", 0.0)])
def test_the_data_fix_the_total_width_and_not_its_split(form, dos, c1):
    """Window +-0.6 eV, 5 meV, amplitude 1000, estimated constant
    background 50, sigma 0.05 eV. The stiff direction is v + (pi^2/3) tau:
    |cos| >= 0.99974 for kT/sigma <= 0.3 in all three DOS forms (0.99974
    to 1.000000 measured), and >= 0.9961 at kT/sigma = 1. kappa_2 itself is
    bounded at 3.5-4.2% relative throughout, while the instrument's share
    of it is bounded at 12 (kT/sigma 0.05), 0.49 (0.3) and 0.14 (1.0)."""
    for ratio, floor in ((0.1, 0.9999), (0.3, 0.9997), (1.0, 0.995)):
        w = _width_info(temperature=ratio * 0.05 / KB_EV, form=form, dos=dos, c1=c1)
        assert w.alignment > floor, ratio
        assert 0.03 < w.sd_kappa2 < 0.05
    shares = [_width_info(temperature=r * 0.05 / KB_EV, form=form, dos=dos, c1=c1).sd_share
              for r in (0.05, 0.3, 1.0)]
    assert shares[0] > 5.0 and 0.3 < shares[1] < 0.7 and 0.1 < shares[2] < 0.2


@pytest.mark.parametrize("form,dos,c1", [("occupied", "linear", 0.35),
                                         ("both_sides", "linear", 0.35),
                                         ("occupied", "flat", 0.0)])
def test_the_separating_information_vanishes_as_tau_squared(form, dos, c1):
    """soft/stiff eigenvalue ratio from 40 K down to 2.5 K, sigma 0.05 eV.
    Its slope against log tau tends to 2 in every DOS form (1.998-2.000
    between 5 K and 2.5 K): the fourth cumulant, not the first, carries
    the separation. The kink of the occupied-side DOS adds a tau**(3/2)
    term to the mean, but its column lies in the E_F and DOS directions
    and does not change the rate."""
    temps = (5.0, 2.5)
    ratio = [(lambda w: w.eigenvalues[0] / w.eigenvalues[1])(
        _width_info(temperature=t, form=form, dos=dos, c1=c1)) for t in temps]
    slope = math.log(ratio[0] / ratio[1]) / math.log(fi.tau_from_temperature(temps[0])
                                                     / fi.tau_from_temperature(temps[1]))
    assert abs(slope - 2.0) < 0.01


def test_a_temperature_prior_sharpens_the_split():
    """300 K, sigma 0.05: the instrument's share bounded at 0.25 with T
    free, 0.19 / 0.086 / 0.034 / 0.016 with a prior of sd 100 / 30 / 10 /
    1 K; kappa_2 moves much less (0.036 to 0.033)."""
    free = _width_info().sd_share
    shares = [_width_info(temperature_mode="temperature_prior", temperature_sd=s).sd_share
              for s in (100.0, 30.0, 10.0, 1.0)]
    assert free > shares[0] > shares[1] > shares[2] > shares[3]
    assert shares[3] < 0.1 * free


def test_width_information_needs_both_width_coordinates():
    bg = fi.constant_background(E, 50.0)
    with pytest.raises(ValueError, match="var_tau"):
        fi.edge_width_information(fi.edge_fisher(E, _edge(), bg, parameterization="sigma_T"))
    with pytest.raises(ValueError, match="var_tau"):
        fi.edge_width_information(fi.edge_fisher(E, _edge(), bg,
                                                 temperature_mode="fixed_temperature"))


# --- judgement, and the resolution handed to the Voigt side --------------------------


def _report(temperature=300.0, sigma=0.05, energy=E, **kw):
    edge = _edge(temperature=temperature, sigma=sigma)
    return fi.assess_edge_identifiability(energy, edge, fi.constant_background(energy, 50.0), **kw)


def test_poisson_weights_reach_the_bound_and_unit_weights_do_not():
    """The sandwich covariance of a weighted least-squares fit: with
    weights 1/mu it is the inverse Fisher matrix exactly; with the
    fitter's default unit weights the resolution's sd is 1.7 times the
    bound with T free and 1.25 times with T fixed (this model, 1000
    counts on a background of 50)."""
    edge = _edge()
    bg = fi.constant_background(E, 50.0)
    fisher = fi.edge_fisher(E, edge, bg)
    cov, names, label = fi.edge_wls_covariance(E, edge, bg, weights="poisson")
    np.testing.assert_allclose(cov, np.linalg.inv(fisher.fisher), rtol=1e-9)
    assert "poisson" in label
    unit, _, unit_label = fi.edge_wls_covariance(E, edge, bg, weights="unit")
    i = names.index("variance")
    ratio = math.sqrt(unit[i, i] / cov[i, i])
    assert 1.5 < ratio < 2.0 and "unit" in unit_label
    fixed = fi.edge_wls_covariance(E, edge, bg, weights="unit",
                                   temperature_mode="fixed_temperature")[0]
    fixed_bound = np.linalg.inv(fi.edge_fisher(E, edge, bg,
                                               temperature_mode="fixed_temperature").fisher)
    assert 1.1 < math.sqrt(fixed[i, i] / fixed_bound[i, i]) < 1.5


def test_the_labels_follow_the_temperature_mode():
    """300 K, sigma 0.05 eV (kT/sigma 0.52), 1000 counts on 50: with T
    free the split is not separable and sd(tau) is withheld; a 10 K prior
    makes it separable; with T fixed it is 'assumed' and there is no tau
    parameter at all."""
    free = _report()
    assert free.separation == "not_separable" and free.sd_tau is None
    assert free.near_boundary_tau is None
    assert next(p.status for p in free.parameters if p.name == "variance") == "weakly_identified"

    prior = _report(temperature_mode="temperature_prior", temperature_sd=10.0)
    assert prior.separation == "separable" and prior.sd_tau is not None
    assert prior.near_boundary_tau is False
    assert next(p.status for p in prior.parameters if p.name == "variance") == "identified"

    fixed = _report(temperature_mode="fixed_temperature")
    assert fixed.separation == "assumed" and fixed.sd_tau is None
    assert "tau" not in [p.name for p in fixed.parameters]
    assert fixed.sd_variance < free.sd_variance / 5.0


def test_reference_scales_are_the_edges_own_width():
    """sqrt(kappa_2) for E_F, |A| for the amplitude, kappa_2 for v and
    tau, 1/sqrt(kappa_2) for the DOS slope; a background term is not
    judged."""
    report = _report()
    kappa2 = report.width.kappa2
    expected = {"ef": math.sqrt(kappa2), "amplitude": 1000.0, "dos_c1": 1.0 / math.sqrt(kappa2),
                "variance": kappa2, "tau": kappa2, "bg_level": None}
    for p in report.parameters:
        assert p.reference_scale == pytest.approx(expected[p.name]) if expected[p.name] else (
            p.reference_scale is None)
        if p.reference_scale is not None:
            assert p.relative_sd == pytest.approx(p.sd / p.reference_scale)
        else:
            assert p.status == "not_assessed"


def test_an_edge_thinner_than_two_channels_is_not_judged():
    coarse = np.arange(-1.2, 1.2 + 1e-9, 0.1)
    report = _report(temperature=10.0, sigma=0.01, energy=coarse)
    assert report.undersampled and report.separation == "undersampled"
    assert all(p.status == "undersampled" for p in report.parameters if p.role == "width")
    assert not _report().undersampled


def test_a_vanishing_gaussian_is_flagged_near_the_boundary():
    assert _report(sigma=0.002, temperature_mode="fixed_temperature").near_boundary_v
    assert not _report(sigma=0.05, temperature_mode="fixed_temperature").near_boundary_v


@pytest.mark.parametrize("temperature,sigma,expected", [(300.0, 0.05, -0.1351e-3),
                                                        (30.0, 0.02, -0.0363e-3)])
def test_temperature_sensitivity_matches_a_refit_with_a_wrong_temperature(temperature, sigma,
                                                                          expected):
    """d sigma / dT is what a fit actually does when the temperature it
    holds fixed is wrong. Refitting the noiseless model with tau fixed 5 K
    either side (Nelder-Mead on the expected deviance, everything else
    free) moves sigma by -0.1351 meV/K at 300 K and -0.0364 meV/K at 30 K;
    the reported numbers are -0.1351 and -0.0363. The kappa_2-conserving
    guess, -(pi^2/3) dtau/dT / (2 sigma), gives -0.1466 and -0.0366."""
    from scipy.optimize import minimize

    edge = _edge(temperature=temperature, sigma=sigma)
    bg = fi.constant_background(E, 50.0)
    report = fi.assess_edge_identifiability(E, edge, bg, temperature_mode="fixed_temperature")
    assert report.resolution.temperature_sensitivity == pytest.approx(expected, rel=0.02)

    coarse = E[::2]  # 121 channels: the refit below is the slow part of this file
    truth = fi.edge_derivatives(coarse, edge.ef, edge.amplitude, edge.variance, edge.tau,
                                dos_c1=edge.dos_c1).value + 50.0

    def deviance(p, tau_fixed):
        ef, amplitude, c1, variance, level = p
        mu = fi.edge_derivatives(coarse, ef, amplitude, max(variance, 1e-12), tau_fixed,
                                 dos_c1=c1).value + level
        return float(np.sum(mu - truth - truth * np.log(mu / truth)))

    sigmas = []
    for shift in (-5.0, 5.0):
        best = minimize(deviance, [edge.ef, edge.amplitude, edge.dos_c1, edge.variance, 50.0],
                        args=(fi.tau_from_temperature(temperature + shift),), method="Nelder-Mead",
                        options=dict(xatol=1e-11, fatol=1e-13, maxiter=4000, maxfev=4000))
        sigmas.append(math.sqrt(best.x[3]))
    refit = (sigmas[1] - sigmas[0]) / 10.0
    assert refit == pytest.approx(report.resolution.temperature_sensitivity, rel=0.02)


def test_the_resolution_hands_the_voigt_side_a_floor_and_its_sd():
    """What the two sides exchange: variance and one of the two sds. The
    Voigt peak's excess Gaussian variance then carries the edge's
    uncertainty too."""
    from toyomacro.voigtfit.identifiability import (
        VoigtPeak,
        assess_identifiability,
        constant_background,
    )

    report = _report(temperature_mode="fixed_temperature", estimator_weights="unit")
    resolution = report.resolution
    assert resolution.estimator == "least squares (unit weights)"
    assert resolution.sd_estimator > resolution.sd_bound
    assert resolution.fwhm == pytest.approx(2.0 * math.sqrt(2.0 * math.log(2.0)) * 0.05)

    energy = np.linspace(-6.0, 6.0, 241)
    peaks = [VoigtPeak(amplitude=2.0e4, center=0.0, sigma=0.09, gamma=0.3)]
    bg = constant_background(energy, 50.0)
    without = assess_identifiability(energy, peaks, bg, variance_floor=resolution.variance)
    with_sd = assess_identifiability(energy, peaks, bg, variance_floor=resolution.variance,
                                     variance_floor_sd=resolution.sd_estimator)
    assert with_sd.widths[0].sd_variance == pytest.approx(
        math.hypot(without.widths[0].sd_variance, resolution.sd_estimator))
    assert with_sd.widths[0].variance_relative_sd > without.widths[0].variance_relative_sd


def test_assessment_inputs_are_checked():
    bg = fi.constant_background(E, 50.0)
    with pytest.raises(ValueError, match="thresholds"):
        fi.assess_edge_identifiability(E, _edge(), bg, thresholds=fi.EdgeThresholds(0.1, 3.0, 0.0))
    with pytest.raises(ValueError, match="weights"):
        fi.edge_wls_covariance(E, _edge(), bg, weights="inverse")
    with pytest.raises(ValueError, match="weights"):
        fi.edge_wls_covariance(E, _edge(), bg, weights=np.ones(3))


# --- scan over measurement conditions -------------------------------------------------


@pytest.fixture(scope="module")
def scan():
    return fi.EdgeScanGrid(), fi.scan_edge_identifiability(fi.EdgeScanGrid())


def test_scan_has_an_axis_per_condition_and_saves(scan, tmp_path):
    grid, out = scan
    shape = (len(grid.ratios), len(grid.half_widths), len(grid.levels), len(grid.slopes),
             len(grid.backgrounds), len(grid.temperature_modes), len(grid.dos_forms))
    for name in ("sd_kappa2", "sd_share", "sd_sigma", "alignment", "soft_over_stiff",
                 "temperature_sensitivity", "separation", "variance_status", "near_boundary_v",
                 "condition_number", "null_space_dim", "n_energy", "total_counts"):
        assert out[name].shape == shape, name
    np.savez(tmp_path / "scan.npz", **out)
    back = np.load(tmp_path / "scan.npz")
    np.testing.assert_array_equal(back["sd_share"], out["sd_share"])
    assert list(back["temperature_modes"]) == list(np.asarray(grid.temperature_modes))


def test_nothing_in_the_scan_depends_on_the_energy_unit(scan):
    """``width`` is sqrt(kappa_2) in eV and sets the unit of the axis;
    every reported number is dimensionless."""
    _, out = scan
    other = fi.scan_edge_identifiability(fi.EdgeScanGrid(width=0.4))
    for name in ("sd_kappa2", "sd_share", "sd_sigma", "alignment", "soft_over_stiff",
                 "temperature_sensitivity", "separation", "near_boundary_v", "n_energy"):
        np.testing.assert_allclose(other[name], out[name], rtol=1e-9, equal_nan=True, err_msg=name)


def test_the_scan_spans_the_three_regimes_and_labels_them(scan):
    """Window 6 widths, estimated constant background at 5% of the step,
    slope 0.3 per width, occupied-side DOS: with the temperature free the
    instrument's share is bounded at 3.35 (kT/sigma 0.1), 0.29 (0.5), 0.18
    (1.0) and 0.089 (3.0), so only the last counts as separable at the
    default threshold of 0.1."""
    grid, out = scan
    free = list(grid.temperature_modes).index("free")
    cell = (slice(None), 1, 0, 1, list(grid.backgrounds).index("estimated"), free, 0)
    share = out["sd_share"][cell]
    assert share[0] > 3.0 and share[-1] < 0.1
    assert np.all(np.diff(share) < 0.0)
    codes = out["separation"][cell]
    assert codes[0] == 1 and codes[-1] == 0  # not_separable -> separable
    assert np.all(out["separation"][:, 1, 0, 1, 2,
                                    list(grid.temperature_modes).index("fixed_temperature"),
                                    0] == 2)


def test_a_temperature_prior_and_a_known_background_both_sharpen_the_split(scan):
    grid, out = scan
    modes = list(grid.temperature_modes)
    backgrounds = list(grid.backgrounds)
    free = out["sd_share"][:, 1, 0, 1, backgrounds.index("estimated"), modes.index("free"), 0]
    prior = out["sd_share"][:, 1, 0, 1, backgrounds.index("estimated"),
                            modes.index("temperature_prior"), 0]
    assert np.all(prior <= free + 1e-12)
    estimated = out["sd_sigma"][:, 1, 0, 1, backgrounds.index("estimated"), modes.index("free"), 0]
    known = out["sd_sigma"][:, 1, 0, 1, backgrounds.index("known"), modes.index("free"), 0]
    none = out["sd_sigma"][:, 1, 0, 1, backgrounds.index("none"), modes.index("free"), 0]
    assert np.all(known <= estimated + 1e-12) and np.all(none <= known + 1e-12)


def test_the_kink_in_the_density_of_states_is_worth_a_factor_in_the_split(scan):
    """The fitter's DOS is flat below E_F and rises above it, so its slope
    changes exactly where the thermal tail is. Continuing the same
    polynomial smoothly instead widens the bound on the instrument's share
    by 1.6 to 3.0 over these conditions (measured 3.35 vs 5.22 at
    kT/sigma 0.1, 0.089 vs 0.263 at 3.0). The split is that much a
    property of the DOS model, not only of the fourth cumulant."""
    grid, out = scan
    occupied = out["sd_share"][:, 1, 0, 1, 2, 0, list(grid.dos_forms).index("occupied")]
    smooth = out["sd_share"][:, 1, 0, 1, 2, 0, list(grid.dos_forms).index("both_sides")]
    ratio = smooth / occupied
    assert np.all(ratio > 1.5) and np.all(ratio < 3.5)
    flat = out["sd_share"][:, 1, 0, 0, 2, 0, :]  # slope 0: the forms coincide
    np.testing.assert_allclose(flat[:, 0], flat[:, 1], rtol=1e-12)


def test_bounds_scale_with_the_exposure_and_with_the_total_counts():
    """Every bound is a square root of an inverse Fisher matrix, and the
    matrix is linear in exposure."""
    base = fi.EdgeScanGrid(ratios=(1.0,), half_widths=(6.0,), levels=(1.0, 4.0), slopes=(0.3,),
                           backgrounds=("estimated",), temperature_modes=("free",),
                           dos_forms=("occupied",))
    out = fi.scan_edge_identifiability(base)
    np.testing.assert_allclose(out["sd_sigma"][0, 0, 1], out["sd_sigma"][0, 0, 0] / 2.0, rtol=1e-9)
    counts = fi.scan_edge_identifiability(
        dataclasses.replace(base, normalization="total_counts", levels=(1e5, 4e5)))
    np.testing.assert_allclose(counts["total_counts"][0, 0, 0], 1e5, rtol=1e-9)
    np.testing.assert_allclose(counts["sd_sigma"][0, 0, 1], counts["sd_sigma"][0, 0, 0] / 2.0,
                               rtol=1e-9)


def test_refused_cells_stay_nan():
    """A DOS that is negative where it carries weight, and a window with no
    background where the mean underflows, are both refused; the scan
    leaves those cells NaN rather than guessing."""
    steep = fi.scan_edge_identifiability(fi.EdgeScanGrid(
        ratios=(3.0,), half_widths=(6.0,), slopes=(2.0,), backgrounds=("estimated",),
        temperature_modes=("free",), dos_forms=("both_sides",)))
    assert np.isnan(steep["sd_share"]).all()
    # 100 edge widths below E_F at kT/sigma = 0.1 the mean is below 1e-308
    empty = fi.scan_edge_identifiability(fi.EdgeScanGrid(
        ratios=(0.1,), half_widths=(100.0,), slopes=(0.0,), backgrounds=("none",),
        temperature_modes=("free",), dos_forms=("occupied",)))
    assert np.isnan(empty["sd_share"]).all()


def test_scan_grid_inputs_are_checked():
    for kw in (dict(ratios=()), dict(ratios=(-1.0,)), dict(half_widths=(0.0,)),
               dict(levels=(0.0,)), dict(step=0.0), dict(width=-1.0), dict(amplitude=0.0),
               dict(backgrounds=("shirley",)), dict(temperature_modes=("guessed",)),
               dict(dos_forms=("kinked",)), dict(normalization="counts"),
               dict(background_kind="shirley"), dict(prior_fraction=0.0)):
        with pytest.raises(ValueError):
            fi.EdgeScanGrid(**kw)


def test_freeing_the_temperature_inflates_the_resolution_across_the_scan(scan):
    """The module's most practical output, pinned as a population rather
    than at one cell. Over every (kT/sigma, window, background) of the
    default scan at a DOS rising 30% per edge width, freeing the
    temperature multiplies sd(sigma)/sigma by 1.3 to 78; with the
    temperature fixed that quantity itself runs 1.6% to 107%. The
    largest inflation is at kT/sigma = 0.1, where the thermal tail is
    too short to measure a temperature with and fitting one anyway
    spends the resolution's precision on it."""
    grid, out = scan
    modes = list(grid.temperature_modes)
    islope = list(grid.slopes).index(0.3)
    iform = list(grid.dos_forms).index("occupied")
    free = out["sd_sigma"][:, :, 0, islope, :, modes.index("free"), iform]
    fixed = out["sd_sigma"][:, :, 0, islope, :, modes.index("fixed_temperature"), iform]
    usable = np.isfinite(free) & np.isfinite(fixed) & (fixed > 0.0)
    assert usable.sum() >= 30
    inflation = free[usable] / fixed[usable]
    assert inflation.min() == pytest.approx(1.3, abs=0.2)
    assert inflation.max() == pytest.approx(78.0, rel=0.15)
    assert np.all(inflation > 1.0)                      # freeing never helps
    assert fixed[usable].min() == pytest.approx(0.016, abs=0.004)
    assert fixed[usable].max() == pytest.approx(1.07, rel=0.15)
    # the worst inflation is the shortest thermal tail
    worst = np.unravel_index(np.argmax(np.where(usable, free / fixed, -1.0)), free.shape)
    assert grid.ratios[worst[0]] == min(grid.ratios)


@pytest.mark.parametrize(
    "fwhm, temperature, ratio",
    [(0.30, 300.0, 1.37), (0.10, 300.0, 1.30), (0.050, 300.0, 1.25),
     (0.030, 30.0, 1.32), (0.010, 10.0, 2.55)])
def test_the_unit_weight_estimator_stays_above_the_bound(fwhm, temperature, ratio):
    """What §6 of the design record quotes. The package's own fitter uses
    unit weights by default, so its sandwich standard deviation is 1.25
    to 2.55 times the Cramer-Rao bound over these conditions -- a 2 eV
    window in 10 meV steps, 2000 counts per channel over a background of
    50, temperature fixed. Pairing that estimate with the bound would
    understate its error by a quarter to a factor of 2.5."""
    energy = np.arange(-1.0, 1.0 + 1e-9, 0.01)
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    edge = fi.FermiEdge(ef=0.0, amplitude=2000.0, sigma=sigma, temperature=temperature,
                        dos_c1=0.3)
    report = fi.assess_edge_identifiability(
        energy, edge, fi.constant_background(energy, 50.0),
        temperature_mode="fixed_temperature", estimator_weights="unit")
    resolution = report.resolution
    sandwich = resolution.sd_estimator / (2.0 * sigma)
    assert sandwich / resolution.sd_sigma_bound == pytest.approx(ratio, abs=0.05)
    assert "unit" in resolution.estimator
