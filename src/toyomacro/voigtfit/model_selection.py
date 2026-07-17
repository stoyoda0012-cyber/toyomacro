"""Information-criterion scoring for already-fitted candidate models.

Pure functions (Phase 2 of the rank/model-selection plan): everything
here scores fits that some OTHER layer produced — no optimizer, no data
generation, no rank computation. The exact-K orchestration that produces
the fits arrives in Phase 3; the verdict rules that combine these scores
with rank diagnostics also live there.

Frozen conventions (benchmarks/README_rank_model_selection.md §2.5):

- Gaussian, unknown common variance::

      AIC  = n log(RSS/n) + 2k
      AICc = AIC + 2k(k+1)/(n-k-1)     (inf + warning when n <= k+1)
      BIC  = n log(RSS/n) + k log(n)

  This is the profile-likelihood form: the variance MLE ``RSS/n`` is
  substituted in, so the variance IS an estimated parameter. Gaussian
  RSS scoring therefore REQUIRES ``variance_estimated=True`` on the
  model (Gate 2 review); a known-variance Gaussian path would use a
  different log-likelihood and is deliberately not implemented yet.

- Poisson ICs use the exact log-likelihood
  ``sum(y log mu - mu - lgamma(y+1))``; the deviance is carried as a
  diagnostic (its saturated-model constant cancels in delta-IC but not
  in absolute IC). Non-positive model predictions raise — never a
  silent clip.
- ``k`` counts actual free parameters: fixed/shared parameters and
  fixed SO splitting / branch ratio contribute nothing; free background
  coefficients count; an estimated Gaussian variance is counted
  explicitly via ``variance_estimated``.
- A structured SO doublet is ONE amplitude parameter (and two visible
  peaks).
- Failed candidates keep their scores for the report but are excluded
  from delta baselines and best-by-criterion selection (Gate 2, E6).

Not exported from ``toyomacro.voigtfit.__init__``; no CLI/GUI/Web wiring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .multipeak_config import ComponentConfig

__all__ = [
    "CandidateModel",
    "CandidateFit",
    "CandidateScore",
    "PeakCountSelection",
    "gaussian_ic",
    "poisson_ic",
    "poisson_loglik",
    "poisson_deviance",
    "score_candidate",
    "select_peak_count",
]

Criterion = Literal["aic", "aicc", "bic"]


@dataclass(frozen=True)
class CandidateModel:
    """A candidate structure with an explicit free-parameter declaration.

    ``components`` defines exactly K structured components — this type
    has no notion of "up to K" (frozen design §2.6). Freedom flags
    declare what the fit actually optimized; nothing is inferred.
    """

    components: tuple[ComponentConfig, ...]
    bg_degree: int | None = None
    amplitudes_free: bool = True
    centers_free: bool = True
    sigmas_free: bool = True
    gammas_free: bool = False
    shared_sigma: bool = False
    shared_gamma: bool = False
    variance_estimated: bool = False

    @property
    def k_structured(self) -> int:
        """Number of structured components (a doublet counts once)."""
        return len(self.components)

    @property
    def n_visible_peaks(self) -> int:
        """Number of visible lines (a doublet contributes two)."""
        return sum(2 if c.is_doublet else 1 for c in self.components)

    @property
    def n_free_parameters(self) -> int:
        """Actual free-parameter count under the declared constraints.

        Per structured component: one amplitude (doublet = one), one
        center, one sigma / gamma unless shared (then one total). Fixed
        SO splitting and branch ratio contribute nothing. Background
        adds ``bg_degree + 1``; an estimated Gaussian variance adds one.
        """
        n = self.k_structured
        k = 0
        if self.amplitudes_free:
            k += n
        if self.centers_free:
            k += n
        if self.sigmas_free:
            k += 1 if self.shared_sigma else n
        if self.gammas_free:
            k += 1 if self.shared_gamma else n
        if self.bg_degree is not None:
            k += self.bg_degree + 1
        if self.variance_estimated:
            k += 1
        return k

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_structured": self.k_structured,
            "n_visible_peaks": self.n_visible_peaks,
            "n_free_parameters": self.n_free_parameters,
            "bg_degree": self.bg_degree,
            "flags": {
                "amplitudes_free": self.amplitudes_free,
                "centers_free": self.centers_free,
                "sigmas_free": self.sigmas_free,
                "gammas_free": self.gammas_free,
                "shared_sigma": self.shared_sigma,
                "shared_gamma": self.shared_gamma,
                "variance_estimated": self.variance_estimated,
            },
        }


@dataclass(frozen=True)
class CandidateFit:
    """Outcome of one candidate fit, as reported by the fitting layer.

    Gaussian scoring reads ``rss``; Poisson scoring reads ``loglik``
    (compute it with :func:`poisson_loglik`) and optionally carries the
    ``deviance`` diagnostic. ``success=False`` keeps the candidate in
    the report but out of every selection decision.
    """

    n_points: int
    rss: float | None = None
    loglik: float | None = None
    deviance: float | None = None
    success: bool = True
    termination_reason: str = ""
    boundary_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_points": self.n_points,
            "rss": self.rss,
            "loglik": self.loglik,
            "deviance": self.deviance,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "boundary_flags": list(self.boundary_flags),
        }


@dataclass(frozen=True)
class CandidateScore:
    """Information criteria for one candidate (pure result)."""

    k_structured: int
    n_visible_peaks: int
    n_free_parameters: int
    n_points: int
    noise_model: str
    aic: float
    aicc: float
    bic: float
    rss: float | None
    loglik: float | None
    deviance: float | None
    success: bool
    termination_reason: str
    boundary_flags: tuple[str, ...]
    warnings: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_structured": self.k_structured,
            "n_visible_peaks": self.n_visible_peaks,
            "n_free_parameters": self.n_free_parameters,
            "n_points": self.n_points,
            "noise_model": self.noise_model,
            "aic": self.aic,
            "aicc": self.aicc,
            "bic": self.bic,
            "rss": self.rss,
            "loglik": self.loglik,
            "deviance": self.deviance,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "boundary_flags": list(self.boundary_flags),
            "warnings": list(self.warnings),
        }


def _aicc_correction(n: int, k: int) -> tuple[float, list[str]]:
    if n <= k + 1:
        return math.inf, [
            f"AICc undefined for n={n} <= k+1={k + 1}: reported as inf"
        ]
    return 2.0 * k * (k + 1) / (n - k - 1), []


def gaussian_ic(rss: float, n: int, k: int) -> tuple[float, float, float, list[str]]:
    """(AIC, AICc, BIC, warnings) for Gaussian unknown-common-variance.

    Reference formulas of the frozen design §2.5 — RSS-based, so the
    Gaussian constant terms are dropped consistently across candidates.
    """
    if rss < 0:
        raise ValueError(f"rss must be non-negative, got {rss}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    warnings: list[str] = []
    if rss == 0.0:
        warnings.append("RSS == 0: information criteria are -inf (exact fit)")
        base = -math.inf
    else:
        base = n * math.log(rss / n)
    aic = base + 2.0 * k
    corr, w = _aicc_correction(n, k)
    warnings += w
    aicc = math.inf if math.isinf(corr) else aic + corr
    bic = base + k * math.log(n)
    return aic, aicc, bic, warnings


def poisson_loglik(y: np.ndarray, mu: np.ndarray) -> float:
    """Exact Poisson log-likelihood ``sum(y log mu - mu - lgamma(y+1))``.

    Raises on non-positive predictions — the frozen design forbids
    silent clipping (§2.5). ``mu == 0`` is accepted only where ``y == 0``
    (limit contribution 0).
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if y.shape != mu.shape:
        raise ValueError("y and mu must have the same shape")
    if np.any(y < 0):
        raise ValueError("Poisson data y must be non-negative")
    if np.any(mu < 0) or np.any((mu == 0) & (y > 0)):
        raise ValueError(
            "non-positive Poisson prediction mu where y > 0: refusing to clip "
            "silently (frozen design §2.5); fix the model or mask the points"
        )
    lgamma = np.array([math.lgamma(v + 1.0) for v in y])
    terms = np.where(y > 0, y * np.log(np.where(mu > 0, mu, 1.0)), 0.0)
    return float(np.sum(terms - mu - lgamma))


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance ``2 sum(y log(y/mu) - (y - mu))`` (diagnostic).

    Input validation mirrors :func:`poisson_loglik` (Gate 2 review).
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if y.shape != mu.shape:
        raise ValueError("y and mu must have the same shape")
    if np.any(y < 0):
        raise ValueError("Poisson data y must be non-negative")
    if np.any(mu < 0) or np.any((mu == 0) & (y > 0)):
        raise ValueError("non-positive Poisson prediction mu where y > 0")
    ylogy = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / np.where(mu > 0, mu, 1.0)), 0.0)
    return float(2.0 * np.sum(ylogy - (y - mu)))


def poisson_ic(loglik: float, n: int, k: int) -> tuple[float, float, float, list[str]]:
    """(AIC, AICc, BIC, warnings) from an exact Poisson log-likelihood."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    aic = -2.0 * loglik + 2.0 * k
    corr, warnings = _aicc_correction(n, k)
    aicc = math.inf if math.isinf(corr) else aic + corr
    bic = -2.0 * loglik + k * math.log(n)
    return aic, aicc, bic, warnings


def score_candidate(
    model: CandidateModel,
    fit: CandidateFit,
    noise_model: Literal["gaussian", "poisson"] = "gaussian",
) -> CandidateScore:
    """Score one already-fitted candidate. Pure; never raises on failure.

    A failed fit is still scored when its statistics allow (for the
    report) but carries ``success=False`` so :func:`select_peak_count`
    excludes it from every decision.
    """
    k = model.n_free_parameters
    warnings: list[str] = []
    if noise_model == "gaussian":
        if not model.variance_estimated:
            raise ValueError(
                "gaussian RSS scoring uses the profile likelihood with the "
                "variance MLE substituted in, so the variance is an estimated "
                "parameter: set variance_estimated=True (a known-variance "
                "Gaussian path is a separate, unimplemented log-likelihood)"
            )
        if fit.rss is None:
            raise ValueError("gaussian scoring requires fit.rss")
        aic, aicc, bic, w = gaussian_ic(fit.rss, fit.n_points, k)
    elif noise_model == "poisson":
        if fit.loglik is None:
            raise ValueError(
                "poisson scoring requires fit.loglik (use poisson_loglik)"
            )
        aic, aicc, bic, w = poisson_ic(fit.loglik, fit.n_points, k)
    else:
        raise ValueError(f"unknown noise_model {noise_model!r}")
    warnings += w
    if not fit.success:
        warnings.append(
            f"fit failed ({fit.termination_reason or 'no reason given'}): "
            "excluded from selection"
        )
    if fit.boundary_flags:
        warnings.append(
            "boundary reached: " + ", ".join(fit.boundary_flags)
            + " — regularity conditions for AIC/BIC may not hold"
        )
    return CandidateScore(
        k_structured=model.k_structured,
        n_visible_peaks=model.n_visible_peaks,
        n_free_parameters=k,
        n_points=fit.n_points,
        noise_model=noise_model,
        aic=aic,
        aicc=aicc,
        bic=bic,
        rss=fit.rss,
        loglik=fit.loglik,
        deviance=fit.deviance,
        success=fit.success,
        termination_reason=fit.termination_reason,
        boundary_flags=fit.boundary_flags,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class PeakCountSelection:
    """IC comparison across candidates (no rank/verdict layer here).

    ``delta_aicc`` / ``delta_bic`` are relative to the best SUCCESSFUL
    candidate; failed candidates carry ``None`` deltas. ``best_by``
    maps each criterion to the ``k_structured`` of the best successful
    candidate (``None`` when nothing succeeded).
    """

    scores: tuple[CandidateScore, ...]
    delta_aicc: tuple[float | None, ...]
    delta_bic: tuple[float | None, ...]
    best_by: dict[str, int | None]
    excluded_k: tuple[int, ...]
    warnings: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": [s.to_dict() for s in self.scores],
            "delta_aicc": list(self.delta_aicc),
            "delta_bic": list(self.delta_bic),
            "best_by": dict(self.best_by),
            "excluded_k": list(self.excluded_k),
            "warnings": list(self.warnings),
        }


def _criterion_selection(
    scores: list[CandidateScore], ok: list[CandidateScore], attr: str
) -> tuple[int | None, tuple[float | None, ...], list[str]]:
    """Best K and per-candidate deltas for one criterion, NaN-free.

    Non-finite handling (Gate 2 review):
    - all successful values +inf  -> best None, deltas None, warning
    - two or more -inf (exact fits) -> tie: best None, deltas None, warning
    - exactly one -inf             -> it wins; its delta 0.0, others +inf
    - otherwise                    -> min over finite values; +inf
      candidates keep delta +inf (never NaN)
    """
    none_deltas = tuple(None for _ in scores)
    if not ok:
        return None, none_deltas, []
    vals = {id(s): getattr(s, attr) for s in ok}
    neg = [s for s in ok if vals[id(s)] == -math.inf]
    finite = [vals[id(s)] for s in ok if math.isfinite(vals[id(s)])]

    if len(neg) >= 2:
        return None, none_deltas, [
            f"{attr}: {len(neg)} exact fits (IC = -inf) tie — selection "
            "undefined; deltas suppressed"
        ]
    if len(neg) == 1:
        winner = neg[0]
        deltas = tuple(
            (0.0 if s is winner else math.inf) if s.success else None
            for s in scores
        )
        return winner.k_structured, deltas, [
            f"{attr}: exact fit (IC = -inf) at K={winner.k_structured} "
            "dominates — deltas to it are +inf"
        ]
    if not finite:
        return None, none_deltas, [
            f"{attr}: all successful candidates are non-finite (+inf) — "
            "no selection possible for this criterion"
        ]
    base = min(finite)
    best = min(
        (s for s in ok if math.isfinite(vals[id(s)])),
        key=lambda s: vals[id(s)],
    ).k_structured
    deltas = tuple(
        (getattr(s, attr) - base) if s.success else None for s in scores
    )
    return best, deltas, []


def select_peak_count(scores: list[CandidateScore]) -> PeakCountSelection:
    """Compare scored candidates; failed fits never win (Gate 2, E6).

    Purely an IC comparison table — the supported/ambiguous/unsupported
    verdict that folds in rank diagnostics is Phase 3. Non-finite IC
    values never produce NaN deltas or a spurious winner (see
    :func:`_criterion_selection`).
    """
    if not scores:
        raise ValueError("no candidates to select from")
    warnings: list[str] = []
    ok = [s for s in scores if s.success]
    excluded = tuple(s.k_structured for s in scores if not s.success)
    if excluded:
        warnings.append(f"excluded failed candidates: K={sorted(excluded)}")
    if not ok:
        warnings.append("all candidate fits failed: no selection possible")

    best_aic, _, w_aic = _criterion_selection(scores, ok, "aic")
    best_aicc, delta_aicc, w_aicc = _criterion_selection(scores, ok, "aicc")
    best_bic, delta_bic, w_bic = _criterion_selection(scores, ok, "bic")
    warnings += w_aic + w_aicc + w_bic

    return PeakCountSelection(
        scores=tuple(scores),
        delta_aicc=delta_aicc,
        delta_bic=delta_bic,
        best_by={"aic": best_aic, "aicc": best_aicc, "bic": best_bic},
        excluded_k=excluded,
        warnings=tuple(warnings),
    )
