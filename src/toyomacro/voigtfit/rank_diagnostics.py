"""Rank diagnostics for structured Voigt models (SciPy/NumPy reference).

Research/diagnostic layer answering "how many independent directions does
the data support?" for a given peak placement, background family, and
spin-orbit constraints. Reports four ranks side by side (data matrix,
linear basis, local full-parameter, VARPRO-profiled nonlinear) plus the
full singular-value spectra, entropy effective rank, and condition
numbers — never a single "true" rank.

Frozen evaluation design: ``benchmarks/README_rank_model_selection.md``
(Gate 0 artifact). Conventions inherited from the existing engine:

- Peak columns are unit-area Voigt profiles (``voigt_jacobian.voigt_profile``).
- A fixed SO doublet is ONE structured amplitude column
  ``V(c) + V(c + so_split) / branch_ratio`` (``multipeak_solver`` line
  convention, ``branch_ratio = I_main / I_partner``).
- Gaussian whitening input is the noise STANDARD DEVIATION (sigma), not
  the variance — matching ``crlb.compute_multipeak_fisher(noise_std=...)``.
  A variance entry point exists and is converted explicitly.

Not exported from ``toyomacro.voigtfit.__init__``; not wired to any CLI,
GUI, or Web surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .multipeak_config import ComponentConfig
from .multipeak_solver import build_bg_design
from .voigt_jacobian import voigt_profile, voigt_with_jacobian

__all__ = [
    "RankThresholds",
    "ParameterScales",
    "SVDReport",
    "RankDiagnostics",
    "structured_basis",
    "structured_jacobian",
    "whitener",
    "svd_report",
    "compute_rank_diagnostics",
]

_GAUSS_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))  # sigma -> Gaussian FWHM


def voigt_fwhm_approx(sigma: float, gamma: float) -> float:
    """Olivero–Longbothum Voigt FWHM approximation.

    Nominal accuracy about 0.02 %; the largest error measured against the
    exact half-maximum width is 2.37e-4 (at f_L/f_G = 0.29), so "< 0.02 %"
    is not a bound. See ``identifiability.voigt_fwhm``.
    """
    f_g = _GAUSS_FWHM * sigma
    f_l = 2.0 * gamma
    return 0.5346 * f_l + math.sqrt(0.2166 * f_l * f_l + f_g * f_g)


@dataclass(frozen=True)
class RankThresholds:
    """Relative singular-value thresholds for effective-rank reporting.

    ``rel_primary`` is the headline threshold, ``rel_reference`` a softer
    reference value reported alongside. Neither is treated as ground
    truth (frozen design §2.3).
    """

    rel_primary: float = 1e-2
    rel_reference: float = 1e-3

    def to_dict(self) -> dict[str, float]:
        return {
            "rel_primary": self.rel_primary,
            "rel_reference": self.rel_reference,
        }


@dataclass(frozen=True)
class ParameterScales:
    """Absolute physical scales (model units) for Jacobian columns.

    Each Jacobian column ``d(model)/d(p)`` is multiplied by ``scale_p``
    before any SVD so that ranks are unit-independent. Frozen defaults
    (design §2.4) are 10 % of representative amplitude / FWHM, built via
    :meth:`from_representative`.
    """

    amplitude: float
    center: float
    sigma: float
    gamma: float
    background: float
    source: str = "explicit"

    @classmethod
    def from_representative(
        cls,
        representative_amplitude: float,
        representative_fwhm: float,
        fraction: float = 0.1,
        amplitude_floor: float = 1e-12,
    ) -> ParameterScales:
        """Frozen defaults: ``fraction`` × representative amplitude / FWHM.

        ``amplitude_floor`` guards the near-zero-amplitude case. The
        background coefficient scale is NOT the amplitude scale: peak
        amplitudes multiply unit-AREA profiles (units counts·energy)
        while background coefficients multiply the dimensionless
        normalized monomials of ``build_bg_design`` (units counts), so
        the default is the typical-height proxy
        ``amplitude_scale / representative_fwhm`` — unit-consistent and
        invariant under energy-unit changes (asserted by test M6).
        """
        amp_scale = max(fraction * abs(representative_amplitude), amplitude_floor)
        width_scale = fraction * representative_fwhm
        if width_scale <= 0:
            raise ValueError("representative_fwhm must be positive")
        return cls(
            amplitude=amp_scale,
            center=width_scale,
            sigma=width_scale,
            gamma=width_scale,
            background=amp_scale / representative_fwhm,
            source=(
                f"from_representative(amplitude={representative_amplitude!r}, "
                f"fwhm={representative_fwhm!r}, fraction={fraction!r})"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "amplitude": self.amplitude,
            "center": self.center,
            "sigma": self.sigma,
            "gamma": self.gamma,
            "background": self.background,
            "source": self.source,
        }


@dataclass(frozen=True)
class SVDReport:
    """Singular-value spectrum of one whitened (scaled) matrix."""

    singular_values: tuple[float, ...]
    rank_primary: int
    rank_reference: int
    entropy_rank: float
    condition_number: float
    n_rows: int
    n_cols: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "singular_values": list(self.singular_values),
            "rank_primary": self.rank_primary,
            "rank_reference": self.rank_reference,
            "entropy_rank": self.entropy_rank,
            "condition_number": self.condition_number,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
        }


@dataclass(frozen=True)
class RankDiagnostics:
    """JSON-serializable bundle of all rank layers for one problem."""

    linear: SVDReport
    local_full: SVDReport
    profiled_nonlinear: SVDReport
    data_matrix: SVDReport | None
    whitening: dict[str, Any]
    parameter_scales: dict[str, Any]
    thresholds: dict[str, float]
    column_names: dict[str, list[str]]
    warnings: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "linear": self.linear.to_dict(),
            "local_full": self.local_full.to_dict(),
            "profiled_nonlinear": self.profiled_nonlinear.to_dict(),
            "data_matrix": None if self.data_matrix is None else self.data_matrix.to_dict(),
            "whitening": self.whitening,
            "parameter_scales": self.parameter_scales,
            "thresholds": self.thresholds,
            "column_names": self.column_names,
            "warnings": list(self.warnings),
        }


def structured_basis(
    energy: np.ndarray, components: list[ComponentConfig]
) -> np.ndarray:
    """(n_energy, K) basis; one structured column per logical component.

    A fixed SO doublet contributes a single column
    ``V(c) + V(c + so_split) / branch_ratio`` — it is never counted as
    two peaks (frozen design §2.1).
    """
    energy = np.asarray(energy, dtype=np.float64)
    cols = []
    for comp in components:
        col = voigt_profile(energy, comp.center, comp.sigma, comp.gamma)
        if comp.is_doublet:
            col = col + (
                voigt_profile(
                    energy, comp.center + comp.so_split, comp.sigma, comp.gamma
                )
                / comp.branch_ratio
            )
        cols.append(col)
    return np.column_stack(cols)


def structured_jacobian(
    energy: np.ndarray,
    components: list[ComponentConfig],
    amplitudes: np.ndarray,
    include_gamma: bool = True,
    shared_sigma: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Nonlinear Jacobian ``S = d(Phi(theta) a)/d(theta)`` and column names.

    Per component ``k`` the columns are ``center_k``, ``sigma_k`` and
    optionally ``gamma_k``. Doublet partners share sigma/gamma and shift
    rigidly with the main line, so their derivatives are summed into the
    same column (weighted ``1 / branch_ratio``).

    ``shared_sigma=True`` diagnoses a model with ONE Gaussian width for
    all components (Gate 3 review): the per-component sigma columns are
    summed into a single ``sigma_shared`` column,
    ``df/d(sigma_shared) = sum_k df/d(sigma_k)``, giving K + 1 nonlinear
    directions instead of 2K (for ``include_gamma=False``). The
    independent-sigma diagnosis remains the default.
    """
    energy = np.asarray(energy, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if len(amplitudes) != len(components):
        raise ValueError("amplitudes and components must have equal length")

    cols: list[np.ndarray] = []
    names: list[str] = []
    sigma_shared_col = np.zeros_like(energy)
    for k, comp in enumerate(components):
        _, d_dc, d_ds, d_dg = voigt_with_jacobian(
            energy, comp.center, comp.sigma, comp.gamma
        )
        if comp.is_doublet:
            _, p_dc, p_ds, p_dg = voigt_with_jacobian(
                energy, comp.center + comp.so_split, comp.sigma, comp.gamma
            )
            r = 1.0 / comp.branch_ratio
            d_dc = d_dc + r * p_dc
            d_ds = d_ds + r * p_ds
            d_dg = d_dg + r * p_dg
        a_k = amplitudes[k]
        cols.append(a_k * d_dc)
        names.append(f"center_{k}")
        if shared_sigma:
            sigma_shared_col = sigma_shared_col + a_k * d_ds
        else:
            cols.append(a_k * d_ds)
            names.append(f"sigma_{k}")
        if include_gamma:
            cols.append(a_k * d_dg)
            names.append(f"gamma_{k}")
    if shared_sigma:
        cols.append(sigma_shared_col)
        names.append("sigma_shared")
    return np.column_stack(cols), names


def whitener(
    n_energy: int,
    *,
    sigma_std: float | np.ndarray | None = None,
    variance: float | np.ndarray | None = None,
    expected_counts: np.ndarray | None = None,
    poisson_floor: float = 1e-30,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(w_sqrt, description)`` with ``w_sqrt`` the W^(1/2) diagonal.

    Exactly one noise specification is accepted. ``sigma_std`` is the
    Gaussian noise STANDARD DEVIATION (matching ``crlb`` ``noise_std``);
    ``variance`` is sigma squared and is converted via an explicit sqrt;
    ``expected_counts`` selects Poisson whitening ``1/sqrt(max(f, floor))``.
    The returned description records which input was used so results can
    never be ambiguous about sigma-vs-sigma².
    """
    given = [
        name
        for name, val in (
            ("sigma_std", sigma_std),
            ("variance", variance),
            ("expected_counts", expected_counts),
        )
        if val is not None
    ]
    if len(given) != 1:
        raise ValueError(
            f"exactly one of sigma_std / variance / expected_counts required, got {given}"
        )

    if sigma_std is not None:
        s = np.broadcast_to(np.asarray(sigma_std, dtype=np.float64), (n_energy,))
        if np.any(s <= 0):
            raise ValueError("sigma_std must be positive")
        desc: dict[str, Any] = {"model": "gaussian", "input": "sigma_std"}
        w = 1.0 / s
    elif variance is not None:
        v = np.broadcast_to(np.asarray(variance, dtype=np.float64), (n_energy,))
        if np.any(v <= 0):
            raise ValueError("variance must be positive")
        desc = {"model": "gaussian", "input": "variance", "note": "sqrt applied"}
        w = 1.0 / np.sqrt(v)
    else:
        f = np.asarray(expected_counts, dtype=np.float64)
        if f.shape != (n_energy,):
            raise ValueError("expected_counts must have shape (n_energy,)")
        if np.any(f < 0):
            raise ValueError("expected_counts must be non-negative")
        desc = {"model": "poisson", "input": "expected_counts", "floor": poisson_floor}
        w = 1.0 / np.sqrt(np.maximum(f, poisson_floor))
    return np.ascontiguousarray(w), desc


def svd_report(matrix: np.ndarray, thresholds: RankThresholds) -> SVDReport:
    """Full singular-value spectrum + threshold/entropy ranks + condition.

    Entropy effective rank uses Fisher-eigenvalue weights (Gate 1 review):
    ``erank = exp(-sum p_i ln p_i)`` with ``p_i = s_i^2 / sum_j s_j^2`` —
    consistent with the M5 identity ``lambda_i(Fisher) = s_i^2``.

    Condition number is ``inf`` whenever ``n_cols > n_rows``: a wide
    matrix necessarily has a null space in parameter directions, so as an
    identifiability diagnostic it is singular regardless of the stored
    ``min(m, n)`` singular values.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    s = np.linalg.svd(matrix, compute_uv=False)
    n_rows, n_cols = matrix.shape
    if s.size == 0 or s[0] == 0.0:
        return SVDReport(
            singular_values=tuple(float(x) for x in s),
            rank_primary=0,
            rank_reference=0,
            entropy_rank=0.0,
            condition_number=math.inf,
            n_rows=n_rows,
            n_cols=n_cols,
        )
    rel = s / s[0]
    rank_primary = int(np.sum(rel > thresholds.rel_primary))
    rank_reference = int(np.sum(rel > thresholds.rel_reference))
    p = s * s / np.sum(s * s)
    entropy = -float(np.sum(p * np.log(p, where=p > 0, out=np.zeros_like(p))))
    entropy_rank = float(np.exp(entropy))
    if n_cols > n_rows or s[-1] == 0.0:
        condition = math.inf
    else:
        condition = float(s[0] / s[-1])
    return SVDReport(
        singular_values=tuple(float(x) for x in s),
        rank_primary=rank_primary,
        rank_reference=rank_reference,
        entropy_rank=entropy_rank,
        condition_number=condition,
        n_rows=n_rows,
        n_cols=n_cols,
    )


def _range_projector(a_whitened: np.ndarray) -> tuple[np.ndarray, int]:
    """Orthonormal basis U_r of range(A) via SVD (numerical tolerance).

    Unpivoted QR over-projects when A is rank-deficient, so the projector
    uses SVD with the standard machine tolerance ``s > s_1 · max(m,n) · eps``
    — deliberately NOT the diagnostic reporting thresholds.
    """
    u, s, _ = np.linalg.svd(a_whitened, full_matrices=False)
    if s.size == 0 or s[0] == 0.0:
        return u[:, :0], 0
    tol = s[0] * max(a_whitened.shape) * np.finfo(np.float64).eps
    r = int(np.sum(s > tol))
    return u[:, :r], r


def compute_rank_diagnostics(
    energy: np.ndarray,
    components: list[ComponentConfig],
    amplitudes: np.ndarray,
    *,
    bg_degree: int | None = None,
    sigma_std: float | np.ndarray | None = None,
    variance: float | np.ndarray | None = None,
    expected_counts: np.ndarray | None = None,
    scales: ParameterScales | None = None,
    thresholds: RankThresholds | None = None,
    include_gamma: bool = True,
    shared_sigma: bool = False,
    data_matrix: np.ndarray | None = None,
) -> RankDiagnostics:
    """All rank layers for one structured-Voigt problem (frozen design §2.2).

    ``scales=None`` derives the frozen defaults from this problem
    (representative amplitude = max |a|, representative FWHM = median
    Voigt FWHM) and records the derivation in the result.
    ``data_matrix`` is an optional (n_energy, n_spectra) stack Y for the
    data-matrix rank layer.

    Candidate-model comparison (Phase 3 handoff): when ranking candidate
    models K = 1..Kmax against each other, build ONE ``ParameterScales``
    from the shared problem and pass it to every candidate explicitly —
    per-candidate auto-derived representative scales would shift the
    rank layers and make them incomparable across K.
    """
    energy = np.asarray(energy, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    thresholds = thresholds or RankThresholds()
    warnings: list[str] = []

    n_energy = energy.shape[0]
    w_sqrt, whitening_desc = whitener(
        n_energy,
        sigma_std=sigma_std,
        variance=variance,
        expected_counts=expected_counts,
    )

    if scales is None:
        rep_amp = float(np.max(np.abs(amplitudes))) if amplitudes.size else 0.0
        if rep_amp == 0.0:
            warnings.append("all amplitudes zero: amplitude scale fell back to floor")
        rep_fwhm = float(
            np.median([voigt_fwhm_approx(c.sigma, c.gamma) for c in components])
        )
        scales = ParameterScales.from_representative(rep_amp, rep_fwhm)

    phi = structured_basis(energy, components)
    names_amp = [f"amplitude_{k}" for k in range(len(components))]
    if bg_degree is not None:
        bg = build_bg_design(energy, bg_degree).astype(np.float64)
        names_bg = [f"bg_{j}" for j in range(bg.shape[1])]
    else:
        bg = np.empty((n_energy, 0), dtype=np.float64)
        names_bg = []

    a_mat = np.hstack([phi, bg])
    n_linear = a_mat.shape[1]
    if n_energy < n_linear:
        warnings.append(
            f"n_energy={n_energy} < n_linear_columns={n_linear}: underdetermined"
        )

    wa = w_sqrt[:, None] * a_mat
    d_linear = np.array(
        [scales.amplitude] * phi.shape[1] + [scales.background] * bg.shape[1]
    )
    # Frozen design §2.4: every Jacobian column is scaled before any SVD.
    # Amplitude columns carry 1/energy units (unit-area profiles) while
    # background columns are dimensionless, so the unscaled mix would be
    # unit-dependent (caught by test M6).
    linear = svd_report(wa * d_linear[None, :], thresholds)

    s_mat, names_nl = structured_jacobian(
        energy, components, amplitudes,
        include_gamma=include_gamma, shared_sigma=shared_sigma,
    )
    ws = w_sqrt[:, None] * s_mat
    d_nl = np.array(
        [
            {"center": scales.center, "sigma": scales.sigma, "gamma": scales.gamma}[
                name.rsplit("_", 1)[0]
            ]
            for name in names_nl
        ]
    )
    local_full_mat = np.hstack([wa * d_linear[None, :], ws * d_nl[None, :]])
    local_full = svd_report(local_full_mat, thresholds)

    u_r, proj_rank = _range_projector(wa)
    if proj_rank < n_linear:
        warnings.append(
            f"linear basis numerically rank-deficient: projector rank "
            f"{proj_rank} < {n_linear} columns"
        )
    wsd = ws * d_nl[None, :]
    profiled_mat = wsd - u_r @ (u_r.T @ wsd)
    profiled = svd_report(profiled_mat, thresholds)

    data_report = None
    if data_matrix is not None:
        y = np.asarray(data_matrix, dtype=np.float64)
        if y.ndim != 2 or y.shape[0] != n_energy:
            raise ValueError("data_matrix must have shape (n_energy, n_spectra)")
        data_report = svd_report(w_sqrt[:, None] * y, thresholds)

    if n_energy < n_linear + len(names_nl):
        warnings.append(
            f"n_energy={n_energy} < n_total_parameters={n_linear + len(names_nl)}"
        )

    return RankDiagnostics(
        linear=linear,
        local_full=local_full,
        profiled_nonlinear=profiled,
        data_matrix=data_report,
        whitening=whitening_desc,
        parameter_scales=scales.to_dict(),
        thresholds=thresholds.to_dict(),
        column_names={
            "linear": names_amp + names_bg,
            "nonlinear": names_nl,
        },
        warnings=tuple(warnings),
    )
