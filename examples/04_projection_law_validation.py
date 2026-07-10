"""Projection-law validation of VarProFitter's misspecification bias.

Voigt peak fitting is a *separable* inverse problem: given the peak center t,
the model I = A(t) C is linear in C = [peak amplitude(s); polynomial-background
coefficients], and the amplitudes are eliminated exactly as in variable
projection (VarPro). For such estimators, a first-order projection law governs
how an unmodeled systematic error delta (e.g. an asymmetric tail or a weak
satellite) biases the fitted center:

    dt_pred = < s_t , (I - P_a) delta > / g''_tt ,   g''_tt = < s_t, (I - P_a) s_t >

where s_t = a0 * dv/dt is the center-sensitivity vector, P_a the (regularized,
possibly constrained) hat matrix of the linear subproblem, and g''_tt the
profiled Fisher information of the center. The law and its derivation are given
in the companion paper [REF: IP companion, in preparation]; this example
validates it numerically on voigtfit's own Voigt basis, at three levels:

  level 1  analytic ridge inner solve, bounds inactive        -> slope ~ 1
  level 2  amplitude bound a>=0 active (pinned neighbor):
           tangent-space projector stays exact; the naive
           unconstrained projector fails by ~150x
  level 3  the production VarProFitter itself                 -> slope ~ 1

plus two corollaries: (b) error components inside range(P_a) are absorbed by
the amplitudes while null-complement components leak to the center, and
(c) the data-supplied g''_tt collapses as two peaks merge -- the center becomes
non-identifiable below the Voigt FWHM (an identifiability certificate, not a
bias estimate).

Level 3 caveat, stated precisely: VarProFitter's estimator class (ridge-free
variable projection with sigma co-fit) matches the law's assumptions, which is
why it reproduces the law at production precision; production solvers with
different regularizers (e.g. an L1 penalty) are only directionally described.

Deterministic (seed=42). Output: examples/output/04_projection_law_* and a
tracked copy of the figure in docs/figures/.
"""
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear, minimize_scalar
from scipy.special import wofz

from toyomacro.voigtfit import VarProFitter
from toyomacro.voigtfit import voigt_profile as tm_voigt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'output'
OUT_DIR.mkdir(exist_ok=True)
DOCS_FIGS = HERE.parent / 'docs' / 'figures'
DOCS_FIGS.mkdir(parents=True, exist_ok=True)
STEM = '04_projection_law_validation'

SEED = 42
ALPHA = 1e-2                      # ridge strength (depth convention)
M = 161
E = np.linspace(-8.0, 8.0, M)     # energy grid (eV, relative)
T0 = 0.0                          # true peak center
SIG, GAM = 0.5, 0.3               # Voigt widths
A0 = 1.0                          # true amplitude
BG0 = np.array([0.05, 0.01, -0.002])   # poly background coeffs (const, lin, quad)
ES = E / 8.0                      # scaled energy for bg columns
HT = 1e-4                         # center finite-diff step for s_t
EPS_SWEEP = np.geomspace(3e-4, 3e-3, 8)
NEIGHBOR_OFF = 1.2                # level-2 pinned neighbor offset (eV)


def voigt(x, center, sigma, gamma):
    """Voigt profile via the Faddeeva function (area-normalized shape)."""
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def design(t, with_neighbor=False):
    """A(t): [voigt(center=t), (voigt neighbor,) 1, E, E^2]."""
    cols = [voigt(E, t, SIG, GAM)]
    if with_neighbor:
        cols.append(voigt(E, t + NEIGHBOR_OFF, SIG, GAM))
    cols += [np.ones(M), ES, ES ** 2]
    return np.stack(cols, axis=1)


def ridge_hat(A, alpha=ALPHA):
    n = A.shape[1]
    return A @ np.linalg.solve(A.T @ A + alpha * np.eye(n), A.T)


def inner_solve(A, y, alpha=ALPHA, lb=None, ub=None):
    """Constrained ridge inner solve via the augmented least-squares form."""
    n = A.shape[1]
    A_aug = np.vstack([A, np.sqrt(alpha) * np.eye(n)])
    y_aug = np.concatenate([y, np.zeros(n)])
    if lb is None:
        return np.linalg.solve(A.T @ A + alpha * np.eye(n), A.T @ y)
    res = lsq_linear(A_aug, y_aug, bounds=(lb, ub), method='bvls')
    return res.x


def profile_g(t, y, with_neighbor=False, lb=None, ub=None):
    A = design(t, with_neighbor)
    C = inner_solve(A, y, lb=lb, ub=ub)
    r = y - A @ C
    return 0.5 * r @ r + 0.5 * ALPHA * (C @ C)


def t_hat(y, with_neighbor=False, lb=None, ub=None, span=1.5):
    res = minimize_scalar(lambda t: profile_g(t, y, with_neighbor, lb, ub),
                          bounds=(T0 - span, T0 + span), method='bounded',
                          options={'xatol': 1e-10})
    return res.x


def s_t_vec(with_neighbor=False):
    """s_t = A'(t0) C0 = a0 * dv/dt (bg and pinned-neighbor columns constant)."""
    vp = (voigt(E, T0 + HT, SIG, GAM) - voigt(E, T0 - HT, SIG, GAM)) / (2 * HT)
    return A0 * vp


def delta_asym_tail():
    """Systematic unrepresentable error: one-sided (Doniach-Sunjic-like)
    decaying tail on the high-energy side of the peak, smoothed; unit norm."""
    tail = np.where(E > T0, np.exp(-(E - T0) / 2.0), 0.0)
    k = np.exp(-0.5 * (np.arange(-8, 9) * (E[1] - E[0]) / 0.25) ** 2)
    tail = np.convolve(tail, k / k.sum(), mode='same')
    return tail / np.linalg.norm(tail)


def delta_neg_satellite():
    """Level-2 error: negative satellite at the pinned-neighbor position.
    Unconstrained fitting absorbs it with a2<0; the a2>=0 bound forbids that,
    so the constrained estimator must let it leak -- Ptil predicts this, the
    naive P does not."""
    sat = -voigt(E, T0 + NEIGHBOR_OFF, SIG, GAM)
    return sat / np.linalg.norm(sat)


def origin_fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope = float(x @ y / (x @ x))
    ss = float(np.sum((y - slope * x) ** 2))
    tot = float(np.sum(y ** 2))
    return slope, (1 - ss / tot if tot > 0 else float('nan'))


def _panel_label(ax, lab, x=-0.14):
    ax.text(x, 1.04, lab, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='bottom', ha='left', clip_on=False)


# ======================================================================
def main():
    rng = np.random.default_rng(SEED)
    out = {'seed': SEED, 'alpha': ALPHA, 'grid': dict(m=M, e_min=-8.0, e_max=8.0),
           'peak': dict(t0=T0, sigma=SIG, gamma=GAM, a0=A0)}

    # ---- the design column IS the library's own basis (independent wofz check) ----
    rel = float(np.max(np.abs(voigt(E, T0, SIG, GAM) - tm_voigt(E, T0, SIG, GAM)))
                / np.max(voigt(E, T0, SIG, GAM)))
    out['voigt_crosscheck_relerr'] = rel
    assert rel < 1e-12, 'inline Voigt must match toyomacro.voigtfit.voigt_profile'
    print(f'voigt column vs toyomacro.voigtfit.voigt_profile: max rel err = {rel:.2e}')

    # =========== LEVEL 1: single peak, ridge, box inactive ===========
    A1 = design(T0)
    C0 = np.concatenate([[A0], BG0])
    I0 = A1 @ C0
    t_clean = t_hat(I0)
    P1 = ridge_hat(A1)
    ImP1 = np.eye(M) - P1
    st = s_t_vec()
    g_schur = float(st @ (ImP1 @ st))
    hh = 1e-3
    g_prof = (profile_g(T0 + hh, I0) - 2 * profile_g(T0, I0)
              + profile_g(T0 - hh, I0)) / hh ** 2
    d1 = delta_asym_tail()
    pred1, num1 = [], []
    for eps in EPS_SWEEP:
        pred1.append(eps * float(st @ (ImP1 @ d1)) / g_schur)
        num1.append(t_hat(I0 + eps * d1) - t_clean)
    sl1, r21 = origin_fit(pred1, num1)
    out['level1'] = dict(
        clean_recovery_err=abs(t_clean - T0), g_schur=g_schur,
        g_profile_2nd_diff=float(g_prof), g_ratio=float(g_prof / g_schur),
        delta='one-sided smoothed exponential tail (DS-like asymmetry), unit norm',
        leak_frac=float(np.linalg.norm(ImP1 @ d1)),
        eps=EPS_SWEEP.tolist(), pred=pred1, num=num1, slope=sl1, r2=r21)
    print(f"L1: clean err {abs(t_clean-T0):.2e}  g''schur={g_schur:.4f} "
          f"profile2nd={g_prof:.4f} (ratio {g_prof/g_schur:.3f})")
    print(f"L1: slope={sl1:.4f}  R2={r21:.6f}")

    # =========== LEVEL 2: neighbor amplitude pinned at a2 = 0 ===========
    # Strict complementarity requires the bound to be genuinely active at the
    # anchor. The clean unconstrained ridge gives a2 = +1.1e-3 (inactive), so a
    # base dose of the negative satellite is baked into the anchor data first;
    # the sweep then measures the INCREMENTAL response at fixed active set
    # (fixed-active-set discipline of the companion paper).
    A2 = design(T0, with_neighbor=True)
    n2 = A2.shape[1]
    C0_2 = np.concatenate([[A0], [0.0], BG0])       # neighbor at its bound
    I0_2 = A2 @ C0_2                                 # == I0
    lb = np.array([0.0, 0.0, -np.inf, -np.inf, -np.inf])
    ub = np.full(n2, np.inf)
    d2 = delta_neg_satellite()
    # base dose: cross the a2=0 boundary with margin for the whole sweep
    a2_resp = np.linalg.solve(A2.T @ A2 + ALPHA * np.eye(n2), A2.T @ d2)[1]
    a2_clean = np.linalg.solve(A2.T @ A2 + ALPHA * np.eye(n2), A2.T @ I0_2)[1]
    eps_base = float(-a2_clean / a2_resp) * 1.5 + float(EPS_SWEEP.max())
    y_base = I0_2 + eps_base * d2
    t_anchor = t_hat(y_base, with_neighbor=True, lb=lb, ub=ub)
    A2a = design(t_anchor, with_neighbor=True)
    C_anchor = inner_solve(A2a, y_base, lb=lb, ub=ub)
    assert C_anchor[1] == 0.0, 'neighbor bound must be active at the anchor'
    # unconstrained ridge at the anchor wants a2 < 0 -> strictly active
    a2_unc_base = np.linalg.solve(A2a.T @ A2a + ALPHA * np.eye(n2), A2a.T @ y_base)[1]
    # s_t at the anchor (fitted amplitude; only the fitted-peak column moves with t)
    vpa = (voigt(E, t_anchor + HT, SIG, GAM) - voigt(E, t_anchor - HT, SIG, GAM)) / (2 * HT)
    st2 = C_anchor[0] * vpa
    # constrained hat: Z drops the active neighbor column (tangent-space construction of the companion paper)
    keep = [0, 2, 3, 4]
    Zt = np.zeros((n2, 4))
    Zt[keep, range(4)] = 1.0
    At = A2a @ Zt
    Pt = At @ np.linalg.solve(At.T @ At + ALPHA * np.eye(4), At.T)
    ImPt = np.eye(M) - Pt
    Pn = ridge_hat(A2a)                              # naive: neighbor column kept
    ImPn = np.eye(M) - Pn
    g2 = float(st2 @ (ImPt @ st2))
    pred2c, pred2n, num2 = [], [], []
    for eps in EPS_SWEEP:
        pred2c.append(eps * float(st2 @ (ImPt @ d2)) / g2)
        pred2n.append(eps * float(st2 @ (ImPn @ d2)) / float(st2 @ (ImPn @ st2)))
        num2.append(t_hat(y_base + eps * d2, with_neighbor=True, lb=lb, ub=ub) - t_anchor)
    sl2c, r22c = origin_fit(pred2c, num2)
    sl2n, r22n = origin_fit(pred2n, num2)
    out['level2'] = dict(
        eps_base=eps_base, t_anchor=float(t_anchor),
        a2_unconstrained_at_anchor=float(a2_unc_base),
        strictly_active=bool(a2_unc_base < 0), g_schur=g2,
        delta='negative satellite at the pinned-neighbor position, unit norm',
        eps=EPS_SWEEP.tolist(), pred_constrained=pred2c, pred_naive=pred2n,
        num=num2, slope_constrained=sl2c, r2_constrained=r22c,
        slope_naive=sl2n, r2_naive=r22n)
    print(f"L2: anchor t={t_anchor:.4f} (a2_unc={a2_unc_base:.2e} -> strictly active)")
    print(f"L2: constrained slope={sl2c:.4f} R2={r22c:.6f} | "
          f"naive slope={sl2n:.4f} R2={r22n:.6f}")

    # =========== PANEL B: absorbed vs leaked (level-1 operator) ===========
    amp = 1e-3
    absorbed, leaked = [], []
    for _ in range(20):
        v = rng.standard_normal(M)
        for res_list, Mx in ((absorbed, P1), (leaked, ImP1)):
            d = Mx @ v
            d = d / np.linalg.norm(d) * amp
            res_list.append(abs(t_hat(I0 + d) - t_clean))
    med_a, med_l = float(np.median(absorbed)), float(np.median(leaked))
    out['panelB'] = dict(amp=amp, absorbed_median=med_a, leaked_median=med_l,
                         contrast=med_l / max(med_a, 1e-15),
                         absorbed=absorbed, leaked=leaked)
    print(f"B: absorbed {med_a:.2e} A vs leaked {med_l:.2e} -> "
          f"contrast {med_l/max(med_a,1e-15):.0f}x")

    # =========== PANEL C: g'' collapse as two peaks merge ===========
    seps = np.linspace(3.0, 0.05, 25)
    g_ridge, g_data = [], []
    for dd in seps:
        cols = [voigt(E, T0, SIG, GAM), voigt(E, T0 + dd, SIG, GAM),
                np.ones(M), ES, ES ** 2]
        Ax = np.stack(cols, axis=1)
        g_ridge.append(float(st @ ((np.eye(M) - ridge_hat(Ax)) @ st)))
        # alpha -> 0: orthogonal projector onto span(A) (SVD, rcond)
        U, sv, _ = np.linalg.svd(Ax, full_matrices=False)
        Ur = U[:, sv > sv.max() * 1e-8]
        g_data.append(float(st @ (st - Ur @ (Ur.T @ st))))
    out['panelC'] = dict(
        separation=seps.tolist(), g_ridge=g_ridge, g_data=g_data,
        fwhm_voigt=float(1.0692 * GAM + np.sqrt(0.8664 * GAM ** 2
                                                + 8 * np.log(2) * SIG ** 2)),
        note=('g_data (alpha->0, orthogonal projection) is the data-supplied Fisher '
              'information: it collapses as the neighbor column can mimic dv/dt. '
              'g_ridge stays on an alpha-supplied floor -- regularization-supplied, '
              'not data-supplied, curvature (the peak-fit analog of the prior term '
              'in the depth estimator).'))
    print(f"C: data-info g'' {max(g_data):.3f} -> {min(g_data):.2e}; "
          f"ridge floor {min(g_ridge):.3f} (alpha-supplied)")

    # =========== LEVEL 3: the production VarProFitter itself ===========
    if True:
        fitter = VarProFitter()
        I0_nb = A0 * voigt(E, T0, SIG, GAM)          # no bg (VarPro model)
        A_nb = voigt(E, T0, SIG, GAM)[:, None]
        ImP_nb = np.eye(M) - ridge_hat(A_nb)
        g_nb = float(st @ (ImP_nb @ st))
        def _center(rr):
            for k in ('centers', 'center', 'c'):
                if k in rr:
                    return float(np.ravel(rr[k])[0])
            raise KeyError(f'center key not found in {list(rr.keys())}')
        r0 = fitter.fit_single(I0_nb, E, np.array([T0]), np.array([SIG]), GAM)
        t3_clean = _center(r0)
        pred3, num3 = [], []
        for eps in EPS_SWEEP * 10:                   # solver noise floor needs larger eps
            r = fitter.fit_single(I0_nb + eps * d1, E, np.array([T0]),
                                  np.array([SIG]), GAM)
            num3.append(_center(r) - t3_clean)
            pred3.append(eps * float(st @ (ImP_nb @ d1)) / g_nb)
        sl3, r23 = origin_fit(pred3, num3)
        out['level3'] = dict(solver='toyomacro.voigtfit.VarProFitter (production)',
                             clean_center=t3_clean, eps=(EPS_SWEEP * 10).tolist(),
                             pred=pred3, num=num3, slope=sl3, r2=r23,
                             note='different estimator (no ridge, sigma co-fit, no bg '
                                  'columns): directionally correct, scale not calibrated')
        print(f"L3 (production VarPro): slope={sl3:.3f} R2={r23:.4f}")
    # sanity gates (loose, cross-platform BLAS tolerant) -- CI smoke doubles
    # as a numerical regression guard on the law itself
    assert 0.99 < sl1 < 1.01 and r21 > 0.9999, f'level 1 off: {sl1}, {r21}'
    assert 0.97 < sl2c < 1.02 and r22c > 0.999, f'level 2 off: {sl2c}, {r22c}'
    assert sl2n > 50, f'naive projector should fail badly, got slope {sl2n}'
    assert med_l / max(med_a, 1e-15) > 100, 'subspace-immunity contrast lost'
    assert 0.98 < sl3 < 1.02 and r23 > 0.999, f'level 3 off: {sl3}, {r23}'

    (OUT_DIR / f'{STEM}.json').write_text(json.dumps(out, indent=1))
    plot(out)


def plot(out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) delta-response
    a = ax[0]
    L1, L2 = out['level1'], out['level2']
    a.plot(L1['pred'], L1['num'], 'o', ms=6, color='#2e8b57',
           label=f"level 1 (ridge): slope={L1['slope']:.3f}, $R^2$={L1['r2']:.4f}")
    a.plot(L2['pred_constrained'], L2['num'], '^', ms=6, color='#7a4fa5',
           label=f"level 2, $\\tilde P_\\alpha$: slope={L2['slope_constrained']:.3f}, "
                 f"$R^2$={L2['r2_constrained']:.4f}")
    a.plot(L2['pred_naive'], L2['num'], 'x', ms=7, color='#d1495b',
           label=f"level 2, naive $P_\\alpha$: slope={L2['slope_naive']:.2f}")
    lim = np.array([0, max(max(L1['pred']), max(L2['pred_constrained'])) * 1.1])
    a.plot(lim, lim, 'k--', lw=1, label='y=x')
    a.set(xlabel=r'formula $\Delta\hat t$ (eV)', ylabel=r'numerical $\Delta\hat t$ (eV)')
    a.xaxis.set_major_locator(plt.MaxNLocator(4))
    a.ticklabel_format(style='sci', axis='both', scilimits=(-2, 2))
    _panel_label(a, '(a)')
    a.grid(alpha=0.3)
    a.legend(fontsize=7.5, loc='upper left')

    # (b) absorbed vs leaked
    a = ax[1]
    B = out['panelB']
    data = [B['absorbed'], B['leaked']]
    a.boxplot(data, tick_labels=[r'$\delta\in\mathrm{range}(\tilde P_\alpha)$'
                                 + '\n(absorbed)',
                                 r'$\delta\in\mathrm{range}(I-\tilde P_\alpha)$'
                                 + '\n(leaks)'], showfliers=False)
    rng = np.random.default_rng(1)
    for i, dd in enumerate(data, 1):
        a.scatter(np.full(len(dd), i) + rng.normal(0, 0.04, len(dd)),
                  np.maximum(dd, 1e-12), s=12, alpha=0.4)
    a.set_yscale('log')
    a.set(ylabel=r'$|\Delta\hat t|$ (eV)',
          title=f"contrast {B['contrast']:.0f}$\\times$")
    _panel_label(a, '(b)')
    a.grid(alpha=0.3, axis='y', which='both')

    # (c) identifiability collapse
    a = ax[2]
    C = out['panelC']
    a.semilogy(C['separation'], C['g_data'], '-o', ms=4, color='#7a4fa5',
               label=r'data information ($\alpha\to0$)')
    a.semilogy(C['separation'], C['g_ridge'], 's--', ms=4, color='#999',
               label=f'ridge ($\\alpha$={ALPHA:g}) floor')
    a.legend(fontsize=7.5, loc='lower left')
    a.axvline(C['fwhm_voigt'], color='#888', ls='--', lw=1.2)
    # note: the x-axis is inverted, so data-x = fwhm + gap sits visually LEFT of
    # the dashed line; ha='right' then extends the text further left, away from it
    a.text(C['fwhm_voigt'] + 0.10, 2.2, 'Voigt FWHM', fontsize=8,
           color='#666', ha='right', va='top')
    a.invert_xaxis()
    a.set(xlabel='two-peak center separation (eV)',
          ylabel=r"profiled Fisher information  $g''_{tt}$")
    _panel_label(a, '(c)')
    a.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(OUT_DIR / f'{STEM}.png', dpi=220)
    fig.savefig(OUT_DIR / f'{STEM}.pdf')
    plt.close(fig)
    shutil.copy(OUT_DIR / f'{STEM}.png', DOCS_FIGS / f'{STEM}.png')
    print(f"saved -> {OUT_DIR / (STEM + '.png')} (+ docs/figures copy)")


if __name__ == '__main__':
    main()
