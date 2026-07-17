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
from typing import Any

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
            "warnings": list(self.warnings),
        }


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
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, tuple, int | None, bool, str]:
    """Inner VARPRO problem for exactly K components, multi-start.

    Returns (sorted centers, sigma, amplitudes, bg coefficients,
    rss_per_start, chosen_start, success, termination_reason).
    """
    lo, hi = float(energy[0]), float(energy[-1])
    lb = np.array([lo] * k + [cfg.sigma_bounds[0]])
    ub = np.array([hi] * k + [cfg.sigma_bounds[1]])

    def residual_fn(theta: np.ndarray) -> np.ndarray:
        return _varpro_solve(theta, k, energy, yw, w_sqrt, bg, cfg)[0]

    rss_per_start: list[float | None] = []
    solutions: list[tuple[float, np.ndarray, str, bool]] = []
    for centers0 in _initial_center_sets(energy, y, k, cfg):
        theta0 = np.clip(
            np.append(centers0, cfg.sigma_init), lb + 1e-12, ub - 1e-12
        )
        try:
            res = least_squares(residual_fn, theta0, bounds=(lb, ub))
            rss = float(2.0 * res.cost)
            rss_per_start.append(rss)
            solutions.append((rss, res.x, res.message, res.status > 0))
        except Exception as exc:  # noqa: BLE001 — record, never hide, a failed start
            rss_per_start.append(None)
            solutions.append((math.inf, theta0, f"start raised: {exc}", False))

    best_idx = int(np.argmin([s[0] for s in solutions]))
    rss_best, theta_best, message, converged = solutions[best_idx]
    if not math.isfinite(rss_best):
        return (
            np.array([]), cfg.sigma_init, np.array([]), np.array([]),
            tuple(rss_per_start), None, False, message,
        )

    order = np.argsort(theta_best[:k])
    centers_sorted = theta_best[:k][order]
    theta_sorted = np.append(centers_sorted, theta_best[k])
    _, amplitudes, bg_coef = _varpro_solve(
        theta_sorted, k, energy, yw, w_sqrt, bg, cfg
    )
    return (
        centers_sorted, float(theta_best[k]), amplitudes, bg_coef,
        tuple(rss_per_start), best_idx, converged, message,
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
    """Frozen §6 wording: the verdict carries reasons, never a bare K."""
    reasons: list[str] = []
    if selection.best_by["aicc"] is None and selection.best_by["bic"] is None:
        reasons.append("no successful candidate selectable by any criterion")
        return "unsupported", reasons
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
    thresholds = thresholds or RankThresholds()
    warnings: list[str] = []

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
        centers, sigma, amps, bg_coef, rss_starts, chosen, ok, reason = (
            _fit_one_candidate(k, energy, y, yw, w_sqrt, bg, cfg)
        )
        cand_warnings: list[str] = []

        finite_rss = [r for r in rss_starts if r is not None]
        if ok and len(finite_rss) > 1:
            spread = (max(finite_rss) - min(finite_rss)) / max(min(finite_rss), 1e-300)
            if spread > cfg.multistart_spread_rel_tol:
                cand_warnings.append(
                    f"multi-start RSS spread {spread:.3g} exceeds "
                    f"{cfg.multistart_spread_rel_tol}: local minima present — "
                    "this K's evidence is initialization-dependent"
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
            rss = float(min(finite_rss))
            fit = CandidateFit(
                n_points=n, rss=rss, success=True,
                termination_reason=reason, boundary_flags=tuple(flags),
            )
            diag = compute_rank_diagnostics(
                energy, list(model.components), amps,
                bg_degree=cfg.bg_degree, sigma_std=sigma_std,
                scales=shared_scales, thresholds=thresholds,
                include_gamma=False,
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
                rss_per_start=rss_starts, chosen_start=chosen,
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
        warnings=tuple(warnings),
    )
