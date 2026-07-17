"""Exact-K candidate-fit orchestration (Phase 3).

The discrete inverse problem for the model order K, structured as::

    outer:  K = 1 .. K_max          (finite comparison, never a gradient)
    inner:  min_theta || (I - P_[Phi(theta)|B]) W^(1/2) y ||^2

For each K the inner problem refits EXACTLY K structured components:
centers (and a shared Gaussian width) are optimized nonlinearly while
amplitudes and background coefficients are eliminated by residual
projection (VARPRO, via lstsq). The outer layer then compares AICc/BIC,
rank diagnostics, and residuals across K — the result is a supported
RANGE plus non-identifiability evidence, never a bare argmin-IC K.

Frozen conventions honored here (README §2.6, §6 + Gate 2/3 reviews):

- every candidate carries exactly K components; there is no
  ``max_peaks`` notion anywhere in this module;
- ONE shared ``ParameterScales`` is built from the data and passed to
  every candidate's rank diagnostics (cross-K comparability);
- fitted centers are sorted ascending before reporting (no label
  switching between candidates or starts);
- several deterministic starts per K; their RSS spread is recorded so a
  local minimum is never silently mistaken for evidence about K;
- the background family is fixed across the whole comparison;
- v1 is UNCONSTRAINED linear LS: negative amplitudes are flagged and
  collected as evidence for a future NNLS variant, never clipped.

Gaussian noise only in v1 (matches the Phase 4 SNR grid); the scoring
layer already supports Poisson for a later path. SciPy/NumPy only.
Not exported from ``toyomacro.voigtfit.__init__``; no CLI/GUI/Web wiring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares

from .model_selection import (
    CandidateFit,
    CandidateModel,
    CandidateScore,
    PeakCountSelection,
    score_candidate,
    select_peak_count,
)
from .multipeak_config import ComponentConfig
from .multipeak_solver import build_bg_design
from .rank_diagnostics import (
    ParameterScales,
    RankDiagnostics,
    RankThresholds,
    compute_rank_diagnostics,
    structured_basis,
    voigt_fwhm_approx,
    whitener,
)

__all__ = ["ExactKConfig", "CandidateResult", "ExactKReport", "run_exact_k"]

_NEG_AMP_REL_TOL = 1e-8


@dataclass(frozen=True)
class ExactKConfig:
    """Configuration for one exact-K comparison run.

    ``sigma_init`` / ``gamma`` encode prior lineshape knowledge and are
    required. ``bg_degree`` and the SO structure are FIXED across every
    candidate (frozen design: same noise model, background family, and
    constraint conventions for all K). ``delta_ic_threshold`` is the
    "small IC difference" implementation parameter — echoed into the
    report, never a hidden hard-coded conclusion.
    """

    sigma_init: float
    gamma: float
    k_max: int = 5
    bg_degree: int | None = None
    so_split: float = 0.0
    branch_ratio: float = 1.0
    sigma_bounds: tuple[float, float] = (0.02, 5.0)
    n_starts: int = 3
    delta_ic_threshold: float = 2.0
    multistart_spread_rel_tol: float = 0.01
    # Phase 3b — plug-in background (Shirley/Tougaard). The curve is
    # estimated from the OBSERVED spectrum ONCE before the K loop, then
    # held fixed for every candidate; it is neither a linear column nor
    # a free parameter. Shirley runs with fixed endpoints (auto_range
    # deliberately off); Tougaard uses the universal C, no B/C joint
    # estimation. Joint/iterative peak-background estimation is a
    # separate research phase (the background would depend on theta and
    # the inner problem would no longer be a plain VARPRO projection).
    plugin_background: Literal["shirley", "tougaard"] | None = None
    shirley_max_iter: int = 50
    shirley_tol: float = 1e-5
    tougaard_c: float = 1643.0

    def __post_init__(self) -> None:
        if self.plugin_background is not None and self.bg_degree is not None:
            raise ValueError(
                "plugin_background and bg_degree are mutually exclusive: "
                "the background family is fixed to ONE mechanism per "
                "comparison (Phase 3b contract)"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sigma_init": self.sigma_init,
            "gamma": self.gamma,
            "k_max": self.k_max,
            "bg_degree": self.bg_degree,
            "so_split": self.so_split,
            "branch_ratio": self.branch_ratio,
            "sigma_bounds": list(self.sigma_bounds),
            "n_starts": self.n_starts,
            "delta_ic_threshold": self.delta_ic_threshold,
            "multistart_spread_rel_tol": self.multistart_spread_rel_tol,
            "plugin_background": self.plugin_background,
            "shirley_max_iter": self.shirley_max_iter,
            "shirley_tol": self.shirley_tol,
            "tougaard_c": self.tougaard_c,
        }


@dataclass(frozen=True)
class CandidateResult:
    """Everything recorded for one exact-K candidate."""

    k_structured: int
    model: CandidateModel
    score: CandidateScore
    diagnostics: RankDiagnostics | None
    centers: tuple[float, ...]
    sigma: float
    amplitudes: tuple[float, ...]
    bg_coefficients: tuple[float, ...]
    rss_per_start: tuple[float | None, ...]
    start_converged: tuple[bool, ...]
    chosen_start: int | None
    boundary_flags: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_structured": self.k_structured,
            "model": self.model.to_dict(),
            "score": self.score.to_dict(),
            "diagnostics": None if self.diagnostics is None else self.diagnostics.to_dict(),
            "centers": list(self.centers),
            "sigma": self.sigma,
            "amplitudes": list(self.amplitudes),
            "bg_coefficients": list(self.bg_coefficients),
            "rss_per_start": list(self.rss_per_start),
            "start_converged": list(self.start_converged),
            "chosen_start": self.chosen_start,
            "boundary_flags": list(self.boundary_flags),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ExactKReport:
    """Outcome of the discrete inverse problem over K.

    ``supported_k`` is the IC-supported RANGE; ``verdict`` is
    ``supported`` / ``ambiguous`` / ``unsupported`` per the frozen §6
    wording, with ``verdict_reasons`` carrying the evidence. The single
    best-IC K is available via ``selection.best_by`` but is deliberately
    not the headline result.
    """

    candidates: tuple[CandidateResult, ...]
    selection: PeakCountSelection
    supported_k: tuple[int, ...]
    verdict: str
    verdict_reasons: tuple[str, ...]
    evidence: tuple[dict[str, Any], ...]
    negative_amplitude_k: tuple[int, ...]
    parameter_scales: dict[str, Any]
    config: dict[str, Any]
    background: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "selection": self.selection.to_dict(),
            "supported_k": list(self.supported_k),
            "verdict": self.verdict,
            "verdict_reasons": list(self.verdict_reasons),
            "evidence": list(self.evidence),
            "negative_amplitude_k": list(self.negative_amplitude_k),
            "parameter_scales": self.parameter_scales,
            "config": self.config,
            "background": self.background,
            "warnings": list(self.warnings),
        }


def _plugin_background_curve(
    energy: np.ndarray, y: np.ndarray, cfg: ExactKConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate the plug-in background ONCE from the observed spectrum.

    Imports are lazy so ``voigtfit`` stays importable standalone when no
    plug-in background is requested.
    """
    if cfg.plugin_background == "shirley":
        from toyomacro.background import Shirley

        curve = Shirley().calculate(
            energy, y, max_iter=cfg.shirley_max_iter, tol=cfg.shirley_tol,
            auto_range=False,
        )
        record: dict[str, Any] = {
            "mode": "plugin_shirley",
            "settings": {
                "max_iter": cfg.shirley_max_iter,
                "tol": cfg.shirley_tol,
                "auto_range": False,
            },
            "convergence": "not exposed by Shirley.calculate; settings recorded",
        }
    elif cfg.plugin_background == "tougaard":
        from toyomacro.background import Tougaard

        curve = Tougaard().calculate(energy, y, C=cfg.tougaard_c)
        record = {
            "mode": "plugin_tougaard",
            # "universal" reflects the ACTUAL C: a user-fixed non-universal
            # C is still fixed-during-comparison (what 3b requires is no
            # joint B/C estimation), but must never be mislabelled.
            "settings": {
                "C": cfg.tougaard_c,
                "universal": bool(cfg.tougaard_c == 1643.0),
            },
            "convergence": "non-iterative",
        }
    else:
        raise ValueError(
            f"unknown plugin_background {cfg.plugin_background!r} "
            "(expected 'shirley' or 'tougaard')"
        )
    curve = np.asarray(curve, dtype=np.float64)
    if curve.shape != y.shape:
        raise ValueError(
            f"plug-in background returned shape {curve.shape}, "
            f"expected {y.shape}"
        )
    if not np.all(np.isfinite(curve)):
        raise ValueError(
            "plug-in background contains non-finite values — refusing to "
            "subtract it silently"
        )
    record["curve_summary"] = {
        "min": float(np.min(curve)),
        "max": float(np.max(curve)),
        "first": float(curve[0]),
        "last": float(curve[-1]),
    }
    return curve, record


def _components(centers: np.ndarray, sigma: float, cfg: ExactKConfig) -> list[ComponentConfig]:
    return [
        ComponentConfig(
            center=float(c), sigma=float(sigma), gamma=cfg.gamma,
            so_split=cfg.so_split, branch_ratio=cfg.branch_ratio,
        )
        for c in centers
    ]


def _varpro_solve(
    theta: np.ndarray,
    k: int,
    energy: np.ndarray,
    yw: np.ndarray,
    w_sqrt: np.ndarray,
    bg: np.ndarray,
    cfg: ExactKConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(whitened residual, amplitudes, bg coefficients) at theta.

    Amplitudes and background are eliminated by unconstrained linear LS
    on the whitened design — the VARPRO inner projection.
    """
    phi = structured_basis(energy, _components(theta[:k], theta[k], cfg))
    a_mat = w_sqrt[:, None] * np.hstack([phi, bg])
    coef, *_ = np.linalg.lstsq(a_mat, yw, rcond=None)
    residual = yw - a_mat @ coef
    return residual, coef[:k], coef[k:]


def _initial_center_sets(
    energy: np.ndarray, y: np.ndarray, k: int, cfg: ExactKConfig
) -> list[np.ndarray]:
    """Deterministic multi-start center initializations for one K.

    Start 0: greedy peak picking on the offset-removed, lightly smoothed
    data (returned in pick order — typically NOT energy-ordered, which
    exercises the canonical-ordering guarantee downstream). Start 1:
    equally spaced interior points. Starts 2+: seeded jitters of start 0.
    """
    fwhm = voigt_fwhm_approx(cfg.sigma_init, cfg.gamma)
    de = float(np.median(np.diff(energy)))
    yy = y - np.percentile(y, 10)
    win = max(3, int(round(fwhm / (4.0 * de))) | 1)
    smooth = np.convolve(yy, np.ones(win) / win, mode="same")

    picks: list[float] = []
    mask = smooth.copy()
    for _ in range(k):
        i = int(np.argmax(mask))
        picks.append(float(energy[i]))
        mask[np.abs(energy - energy[i]) < fwhm] = -np.inf
    pick_arr = np.array(picks)

    lo, hi = float(energy[0]), float(energy[-1])
    equal = lo + (hi - lo) * (np.arange(1, k + 1) / (k + 1))

    starts = [pick_arr, equal]
    for j in range(2, cfg.n_starts):
        rng = np.random.default_rng(j)
        jitter = pick_arr + rng.uniform(-0.5, 0.5, size=k) * fwhm
        starts.append(np.clip(jitter, lo, hi))
    return starts[: cfg.n_starts]


def _fit_one_candidate(
    k: int,
    energy: np.ndarray,
    y: np.ndarray,
    yw: np.ndarray,
    w_sqrt: np.ndarray,
    bg: np.ndarray,
    cfg: ExactKConfig,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, tuple, tuple, int | None, bool, str]:
    """Inner VARPRO problem for exactly K components, multi-start.

    The winner is the minimum-RSS start AMONG CONVERGED STARTS only
    (Gate 3 review): a non-converged start with a lower RSS must not
    fail the candidate when a converged solution exists. The candidate
    fails only when no start converges.

    Returns (sorted centers, sigma, amplitudes, bg coefficients,
    rss_per_start, start_converged, chosen_start, success,
    termination_reason).
    """
    lo, hi = float(energy[0]), float(energy[-1])
    lb = np.array([lo] * k + [cfg.sigma_bounds[0]])
    ub = np.array([hi] * k + [cfg.sigma_bounds[1]])

    def residual_fn(theta: np.ndarray) -> np.ndarray:
        return _varpro_solve(theta, k, energy, yw, w_sqrt, bg, cfg)[0]

    rss_per_start: list[float | None] = []
    start_converged: list[bool] = []
    solutions: list[tuple[float, np.ndarray, str]] = []
    for centers0 in _initial_center_sets(energy, y, k, cfg):
        theta0 = np.clip(
            np.append(centers0, cfg.sigma_init), lb + 1e-12, ub - 1e-12
        )
        try:
            res = least_squares(residual_fn, theta0, bounds=(lb, ub))
            rss = float(2.0 * res.cost)
            ok = bool(res.status > 0) and math.isfinite(rss)
            rss_per_start.append(rss)
            start_converged.append(ok)
            solutions.append((rss, res.x, res.message))
        except Exception as exc:  # noqa: BLE001 — record, never hide, a failed start
            rss_per_start.append(None)
            start_converged.append(False)
            solutions.append((math.inf, theta0, f"start raised: {exc}"))

    converged_idx = [i for i, ok in enumerate(start_converged) if ok]
    if not converged_idx:
        message = "; ".join(
            f"start {i}: {solutions[i][2]}" for i in range(len(solutions))
        )
        return (
            np.array([]), cfg.sigma_init, np.array([]), np.array([]),
            tuple(rss_per_start), tuple(start_converged), None, False,
            f"no start converged ({message})",
        )

    best_idx = min(converged_idx, key=lambda i: solutions[i][0])
    rss_best, theta_best, message = solutions[best_idx]
    del rss_best

    order = np.argsort(theta_best[:k])
    centers_sorted = theta_best[:k][order]
    theta_sorted = np.append(centers_sorted, theta_best[k])
    _, amplitudes, bg_coef = _varpro_solve(
        theta_sorted, k, energy, yw, w_sqrt, bg, cfg
    )
    return (
        centers_sorted, float(theta_best[k]), amplitudes, bg_coef,
        tuple(rss_per_start), tuple(start_converged), best_idx, True, message,
    )


def _boundary_flags(
    centers: np.ndarray, sigma: float, amplitudes: np.ndarray,
    energy: np.ndarray, cfg: ExactKConfig,
) -> list[str]:
    flags: list[str] = []
    lo, hi = float(energy[0]), float(energy[-1])
    edge = 1e-3 * (hi - lo)
    for i, c in enumerate(centers):
        if c - lo < edge or hi - c < edge:
            flags.append(f"center_{i}_at_energy_bound")
    for bound in cfg.sigma_bounds:
        if abs(sigma - bound) < 1e-6 * max(1.0, bound):
            flags.append("sigma_at_bound")
    amp_ref = float(np.max(np.abs(amplitudes))) if amplitudes.size else 0.0
    for i, a in enumerate(amplitudes):
        if a < -_NEG_AMP_REL_TOL * max(amp_ref, 1.0):
            flags.append(f"negative_amplitude_{i}")
    return flags


def _verdict(
    selection: PeakCountSelection,
    evidence: list[dict[str, Any]],
    supported: list[int],
) -> tuple[str, list[str]]:
    """Frozen §6 wording: the verdict carries reasons, never a bare K.

    ``supported`` additionally requires (Gate 3 review): AICc and BIC
    must name the SAME best K — a criterion disagreement is `ambiguous`
    even when the both-deltas-within-threshold range is a single K —
    and the selected candidate must be free of negative amplitudes.
    """
    reasons: list[str] = []
    best_aicc = selection.best_by["aicc"]
    best_bic = selection.best_by["bic"]
    if best_aicc is None and best_bic is None:
        reasons.append("no successful candidate selectable by any criterion")
        return "unsupported", reasons
    if best_aicc != best_bic:
        # Checked before the support-set size: when the criteria agree,
        # their common best K has both deltas 0 and the support set
        # cannot be empty — so an empty set IS a disagreement symptom
        # (e.g. AICc chasing noise with an extra component while BIC
        # rejects it), and the frozen rule labels that ambiguous.
        reasons.append(
            f"information criteria disagree (AICc best K={best_aicc}, "
            f"BIC best K={best_bic}) — frozen rule: IC disagreement is "
            "ambiguous regardless of the support-set size"
        )
        return "ambiguous", reasons
    if not supported:
        reasons.append("no candidate inside the IC support threshold")
        return "unsupported", reasons
    if len(supported) > 1:
        reasons.append(
            f"IC differences among K={supported} are below the threshold — "
            "the data do not single out one K"
        )
        return "ambiguous", reasons
    k_star = supported[0]
    ev = next(e for e in evidence if e["k"] == k_star)
    if ev["negative_amplitudes"]:
        reasons.append(
            f"the supported candidate K={k_star} carries negative "
            f"amplitudes ({ev['negative_amplitudes']}) under unconstrained "
            "LS — not claiming 'supported' on an unphysical solution"
        )
        return "ambiguous", reasons
    if ev["rank_supported"]:
        reasons.append(
            f"ICs agree on K={k_star} and the local rank supports the "
            "nominal degrees of freedom"
        )
        return "supported", reasons
    reasons.append(
        f"ICs prefer K={k_star} but its local rank is deficient "
        f"(linear deficit {ev['linear_rank_deficit']}, profiled nonlinear "
        f"deficit {ev['nonlinear_rank_deficit']}) — IC and rank disagree"
    )
    return "ambiguous", reasons


def run_exact_k(
    energy: np.ndarray,
    y: np.ndarray,
    cfg: ExactKConfig,
    *,
    sigma_std: float | np.ndarray,
    thresholds: RankThresholds | None = None,
) -> ExactKReport:
    """Solve the discrete inverse problem for K on one spectrum.

    Gaussian noise (``sigma_std`` = standard deviation, matching the
    rank/CRLB layers). Every candidate K = 1..k_max is refit with
    exactly K structured components under the same whitening, background
    family, and constraint conventions, then scored and rank-diagnosed
    with ONE shared ``ParameterScales``.
    """
    energy = np.asarray(energy, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if energy.shape != y.shape:
        raise ValueError("energy and y must have the same shape")
    # XPS binding-energy axes are often descending: normalize to ascending
    # internally (Gate 3 review); non-monotonic axes are rejected outright.
    d_energy = np.diff(energy)
    if np.all(d_energy < 0):
        energy = energy[::-1].copy()
        y = y[::-1].copy()
        if np.ndim(sigma_std) > 0:
            sigma_std = np.asarray(sigma_std)[::-1].copy()
    elif not np.all(d_energy > 0):
        raise ValueError("energy axis must be strictly monotonic")
    thresholds = thresholds or RankThresholds()
    warnings: list[str] = []

    # Phase 3b: the plug-in background is estimated once from the
    # observed spectrum, subtracted, and held fixed for EVERY candidate.
    # It is not a linear column and contributes nothing to the degrees
    # of freedom.
    if cfg.plugin_background is not None:
        bg_curve, background_record = _plugin_background_curve(energy, y, cfg)
        y = y - bg_curve
        warnings.append(
            f"{background_record['mode']}: information criteria are "
            "heuristics conditioned on a background estimated from the "
            "data; the background estimation is NOT counted in the "
            "degrees of freedom, and ICs must not be compared across "
            "background families"
        )
    else:
        background_record = {
            "mode": "polynomial" if cfg.bg_degree is not None else "none",
            "bg_degree": cfg.bg_degree,
        }

    n = energy.shape[0]
    w_sqrt, _ = whitener(n, sigma_std=sigma_std)
    yw = w_sqrt * y
    if cfg.bg_degree is not None:
        bg = build_bg_design(energy, cfg.bg_degree).astype(np.float64)
    else:
        bg = np.empty((n, 0), dtype=np.float64)

    # ONE ParameterScales for the whole candidate family (Gate 1 handoff):
    # representative amplitude = typical height x FWHM (unit-area basis).
    fwhm = voigt_fwhm_approx(cfg.sigma_init, cfg.gamma)
    rep_height = float(np.percentile(y, 99) - np.percentile(y, 5))
    if rep_height <= 0:
        rep_height = float(np.max(np.abs(y))) or 1.0
        warnings.append("flat data: representative height fell back to max|y|")
    shared_scales = ParameterScales.from_representative(rep_height * fwhm, fwhm)

    candidates: list[CandidateResult] = []
    scores: list[CandidateScore] = []
    for k in range(1, cfg.k_max + 1):
        centers, sigma, amps, bg_coef, rss_starts, starts_ok, chosen, ok, reason = (
            _fit_one_candidate(k, energy, y, yw, w_sqrt, bg, cfg)
        )
        cand_warnings: list[str] = []

        conv_rss = [
            r for r, c in zip(rss_starts, starts_ok) if c and r is not None
        ]
        if ok and len(conv_rss) > 1:
            spread = (max(conv_rss) - min(conv_rss)) / max(min(conv_rss), 1e-300)
            if spread > cfg.multistart_spread_rel_tol:
                cand_warnings.append(
                    f"multi-start RSS spread {spread:.3g} exceeds "
                    f"{cfg.multistart_spread_rel_tol}: local minima present — "
                    "this K's evidence is initialization-dependent"
                )
        if ok and not all(starts_ok):
            cand_warnings.append(
                f"{starts_ok.count(False)}/{len(starts_ok)} starts did not "
                "converge; winner chosen among converged starts only"
            )

        if ok:
            flags = _boundary_flags(centers, sigma, amps, energy, cfg)
            model = CandidateModel(
                components=tuple(_components(centers, sigma, cfg)),
                bg_degree=cfg.bg_degree,
                amplitudes_free=True, centers_free=True,
                sigmas_free=True, shared_sigma=True,
                gammas_free=False,
                variance_estimated=True,
            )
            rss = float(min(conv_rss))
            fit = CandidateFit(
                n_points=n, rss=rss, success=True,
                termination_reason=reason, boundary_flags=tuple(flags),
            )
            # shared_sigma=True: diagnose the K+1 nonlinear directions the
            # fit actually optimizes (K centers + one shared width), not
            # 2K independent ones (Gate 3 review).
            diag = compute_rank_diagnostics(
                energy, list(model.components), amps,
                bg_degree=cfg.bg_degree, sigma_std=sigma_std,
                scales=shared_scales, thresholds=thresholds,
                include_gamma=False, shared_sigma=True,
            )
            neg = [f for f in flags if f.startswith("negative_amplitude")]
            if neg:
                cand_warnings.append(
                    f"unconstrained LS produced negative amplitudes ({neg}) — "
                    "recorded as evidence for a future NNLS variant"
                )
        else:
            flags = []
            model = CandidateModel(
                components=tuple(
                    _components(np.full(k, float(np.mean(energy))), cfg.sigma_init, cfg)
                ),
                bg_degree=cfg.bg_degree,
                variance_estimated=True,
            )
            fit = CandidateFit(
                n_points=n, rss=math.inf, success=False,
                termination_reason=reason,
            )
            diag = None
            centers, sigma = np.array([]), cfg.sigma_init
            amps, bg_coef = np.array([]), np.array([])

        score = score_candidate(model, fit, "gaussian")
        scores.append(score)
        candidates.append(
            CandidateResult(
                k_structured=k, model=model, score=score, diagnostics=diag,
                centers=tuple(float(c) for c in centers), sigma=float(sigma),
                amplitudes=tuple(float(a) for a in amps),
                bg_coefficients=tuple(float(b) for b in bg_coef),
                rss_per_start=rss_starts, start_converged=starts_ok,
                chosen_start=chosen,
                boundary_flags=tuple(flags), warnings=tuple(cand_warnings),
            )
        )

    selection = select_peak_count(scores)

    evidence: list[dict[str, Any]] = []
    supported: list[int] = []
    for i, cand in enumerate(candidates):
        d_aicc = selection.delta_aicc[i]
        d_bic = selection.delta_bic[i]
        ic_ok = (
            d_aicc is not None and d_bic is not None
            and d_aicc <= cfg.delta_ic_threshold
            and d_bic <= cfg.delta_ic_threshold
        )
        if cand.diagnostics is not None:
            n_lin = len(cand.diagnostics.column_names["linear"])
            n_nl = len(cand.diagnostics.column_names["nonlinear"])
            lin_def = n_lin - cand.diagnostics.linear.rank_primary
            nl_def = n_nl - cand.diagnostics.profiled_nonlinear.rank_primary
        else:
            lin_def = nl_def = None
        rank_ok = lin_def == 0 and nl_def == 0
        if ic_ok:
            supported.append(cand.k_structured)
        evidence.append(
            {
                "k": cand.k_structured,
                "delta_aicc": d_aicc,
                "delta_bic": d_bic,
                "ic_supported": ic_ok,
                "linear_rank_deficit": lin_def,
                "nonlinear_rank_deficit": nl_def,
                "rank_supported": rank_ok,
                "negative_amplitudes": [
                    f for f in cand.boundary_flags if f.startswith("negative_amplitude")
                ],
                "multistart_warning": any(
                    "local minima" in w for w in cand.warnings
                ),
            }
        )

    verdict, reasons = _verdict(selection, evidence, supported)
    neg_k = tuple(
        c.k_structured for c in candidates
        if any(f.startswith("negative_amplitude") for f in c.boundary_flags)
    )
    if neg_k:
        warnings.append(
            f"negative amplitudes at K={list(neg_k)}: unconstrained-LS "
            "artifact frequency is being recorded as NNLS-need evidence"
        )
    warnings.extend(selection.warnings)

    return ExactKReport(
        candidates=tuple(candidates),
        selection=selection,
        supported_k=tuple(supported),
        verdict=verdict,
        verdict_reasons=tuple(reasons),
        evidence=tuple(evidence),
        negative_amplitude_k=neg_k,
        parameter_scales=shared_scales.to_dict(),
        config=cfg.to_dict(),
        background=background_record,
        warnings=tuple(warnings),
    )
