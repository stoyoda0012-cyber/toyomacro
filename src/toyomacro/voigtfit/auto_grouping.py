"""Fisher-justified automatic component grouping for hierarchical fitting.

dev-log 81 established that hierarchical deflation only works for
truly isolated components. The key metric is the absolute separation
d_leaf = |ΔE| / σ_avg to the nearest neighbor — NOT the congestion ratio.

Width ordering creates asymmetry: Si⁰ (σ=0.150) has d=5.3σ to Si¹⁺,
while Si⁴⁺ (σ=0.398) has d=2.8σ to Si³⁺. Narrow peaks are more
isolated because their tails decay faster.

Algorithm:
    1. Sort components by energy (chain graph)
    2. Compute pairwise separation d_ij = |ΔE| / σ_avg
    3. Check chain endpoints (leaves): d_leaf > threshold → deflate
    4. Repeat inward until no more leaves qualify
    5. Remaining components → simultaneous group

Threshold ≈ 4σ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .multipeak_config import ComponentConfig, MultiPeakConfig


class GroupingResult(NamedTuple):
    """Result of automatic component grouping.

    Attributes:
        groups: Ordered list of index groups. Earlier groups are
            deflated first, last group is fit simultaneously.
            Example: [[0], [1, 2, 3, 4]] for Si 2p.
        rationale: Human-readable explanation for each stage.
        separations: Adjacent-pair separation table {(i,j): d_σ}.
    """

    groups: list[list[int]]
    rationale: list[str]
    separations: dict[tuple[int, int], float]


def separation_matrix(configs: list[ComponentConfig]) -> np.ndarray:
    """Compute pairwise separation d_ij = |ΔE_ij| / σ_avg_ij.

    Args:
        configs: List of ComponentConfig.

    Returns:
        d: (n, n) symmetric matrix, diagonal = inf.
    """
    n = len(configs)
    d = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            sigma_avg = (configs[i].sigma + configs[j].sigma) / 2
            dij = abs(configs[i].center - configs[j].center) / sigma_avg
            d[i, j] = d[j, i] = dij
    return d


def auto_group(
    configs: list[ComponentConfig],
    threshold: float = 4.0,
) -> GroupingResult:
    """Fisher-justified automatic grouping.

    Identifies components that can be cleanly deflated (d_leaf > threshold)
    by peeling leaves from the energy-ordered chain.

    The threshold is calibrated from dev-log 81:
        - 5.3σ: clean deflation (+6.7 dB gain)
        - 2.8σ: failed deflation (-8.4 dB loss)
        - Default 4.0σ is conservative midpoint

    Args:
        configs: List of ComponentConfig (any order).
        threshold: Minimum d_leaf (σ units) for deflation.

    Returns:
        GroupingResult with ordered groups and rationale.
    """
    n = len(configs)
    if n <= 1:
        return GroupingResult([[0]] if n == 1 else [], ["single"], {})

    d = separation_matrix(configs)

    # Energy-ordered chain
    chain = sorted(range(n), key=lambda i: configs[i].center)

    # Adjacent-pair separations (for reporting)
    sep_dict: dict[tuple[int, int], float] = {}
    for k in range(len(chain) - 1):
        i, j = chain[k], chain[k + 1]
        sep_dict[(i, j)] = float(d[i, j])

    # Peel leaves from chain ends
    remaining = list(chain)
    groups: list[list[int]] = []
    rationale: list[str] = []

    while len(remaining) > 1:
        left = remaining[0]
        right = remaining[-1]

        d_left = d[left, remaining[1]]
        d_right = d[right, remaining[-2]]

        # Check if either leaf qualifies
        left_ok = d_left >= threshold
        right_ok = d_right >= threshold

        if not left_ok and not right_ok:
            break  # No more leaves can be peeled

        if left_ok and right_ok:
            # Both qualify — pick the more isolated one
            if d_left >= d_right:
                best, d_best, side = left, d_left, "left"
            else:
                best, d_best, side = right, d_right, "right"
        elif left_ok:
            best, d_best, side = left, d_left, "left"
        else:
            best, d_best, side = right, d_right, "right"

        groups.append([best])
        rationale.append(
            f"deflate [{best}] d_leaf={d_best:.2f}σ ≥ {threshold:.1f}σ ({side})"
        )
        remaining.remove(best)

    # Remaining → simultaneous group
    if remaining:
        groups.append(remaining)
        rationale.append(f"simultaneous {remaining}")

    return GroupingResult(groups, rationale, sep_dict)


# ---------------------------------------------------------------------------
# Generic grouped solver
# ---------------------------------------------------------------------------


@dataclass
class GroupedResult:
    """Result from auto-grouped hierarchical solver.

    Attributes:
        amplitudes: (N, n_comp) combined amplitudes in original config order.
        delta_E: (N, n_comp) energy shifts.
        delta_sigma: (N, n_comp) width shifts.
        grouping: The GroupingResult that determined the execution plan.
        stage_results: Per-stage MultiPeakResult objects.
        stage_residuals: Per-stage residual spectra (after deflation).
    """

    amplitudes: np.ndarray
    delta_E: np.ndarray
    delta_sigma: np.ndarray
    grouping: GroupingResult
    stage_results: list = field(default_factory=list)
    stage_residuals: list = field(default_factory=list)


def _deflate_fitted(
    spectra: np.ndarray,
    config: MultiPeakConfig,
    amplitudes: np.ndarray,
    delta_E: np.ndarray,
    delta_sigma: np.ndarray,
) -> np.ndarray:
    """Subtract fitted Voigt profiles from spectra.

    Generic deflation: reconstructs each component at refined parameters
    and subtracts from spectra.

    Args:
        spectra: (N, n_E) input spectra.
        config: MultiPeakConfig for the components being subtracted.
        amplitudes: (N, n_comp) fitted amplitudes.
        delta_E: (N, n_comp) energy shifts.
        delta_sigma: (N, n_comp) width shifts.

    Returns:
        residual: (N, n_E) spectra with fitted components removed.
    """
    from scipy.special import wofz

    energy = np.asarray(config.energy_axis, dtype=np.float64)
    residual = spectra.astype(np.float64).copy()

    for j in range(config.n_comp):
        peak = config.peaks[j]
        centers_j = peak.center + delta_E[:, j].astype(np.float64)
        sigmas_j = np.maximum(
            peak.sigma + delta_sigma[:, j].astype(np.float64), 0.01
        )
        gamma_j = peak.gamma

        z = (energy[np.newaxis, :] - centers_j[:, np.newaxis]
             + 1j * gamma_j) / (sigmas_j[:, np.newaxis] * np.sqrt(2.0))
        phi_j = np.real(wofz(z)) / (sigmas_j[:, np.newaxis] * np.sqrt(2.0 * np.pi))

        if peak.is_doublet:
            partner_centers = centers_j + peak.so_split
            z_p = (energy[np.newaxis, :] - partner_centers[:, np.newaxis]
                   + 1j * gamma_j) / (sigmas_j[:, np.newaxis] * np.sqrt(2.0))
            phi_p = np.real(wofz(z_p)) / (sigmas_j[:, np.newaxis] * np.sqrt(2.0 * np.pi))
            phi_j = phi_j + phi_p / peak.branch_ratio

        residual -= amplitudes[:, j : j + 1].astype(np.float64) * phi_j

    return residual.astype(np.float32)


def solve_grouped(
    spectra: np.ndarray,
    configs: list[ComponentConfig],
    energy: np.ndarray,
    threshold: float = 4.0,
    n_iterations: int = 5,
    n_outer: int = 2,
    min_dE_range: float = 0.15,
    parabola_dE: bool = True,
) -> GroupedResult:
    """Generic hierarchical solver driven by auto_group.

    Automatically determines which components to deflate first based on
    Fisher-justified separation analysis, then executes staged fitting.

    Pipeline:
        1. auto_group(configs, threshold) → [[isolated], ..., [rest]]
        2. For each stage: build config → fit → deflate from spectra
        3. Outer refinement loop: re-fit earlier stages on cleaned spectra

    Works for any element system — Si 2p, Au 4f, C 1s, etc.

    Args:
        spectra: (N, n_E) input spectra.
        configs: List of ComponentConfig (any element).
        energy: Energy axis (eV).
        threshold: Separation threshold for auto_group (σ units).
        n_iterations: Alternating projection iterations per stage.
        n_outer: Outer refinement loops.
        min_dE_range: Minimum dE range after constraint.
        parabola_dE: Apply parabola sub-grid refinement.

    Returns:
        GroupedResult with combined amplitudes in original config order.
    """
    from .multipeak_solver import process_multipeak

    grouping = auto_group(configs, threshold=threshold)
    n_comp = len(configs)
    N = spectra.shape[0]
    n_stages = len(grouping.groups)

    # Build per-stage MultiPeakConfigs
    stage_configs: list[MultiPeakConfig] = []
    for group in grouping.groups:
        peaks = [configs[i] for i in group]
        cfg = MultiPeakConfig(peaks=peaks, energy_axis=energy)
        if len(peaks) > 1:
            cfg.constrain_dE_ranges(min_dE_range=min_dE_range)
        stage_configs.append(cfg)

    # Storage for per-stage results
    stage_results: list[object | None] = [None] * n_stages
    stage_residuals: list[np.ndarray | None] = [None] * n_stages

    for outer in range(n_outer):
        current_spectra = spectra.copy()

        for s_idx in range(n_stages):
            group = grouping.groups[s_idx]

            # Subtract all OTHER already-fitted stages from spectra
            if outer > 0 or s_idx > 0:
                current_spectra = spectra.astype(np.float64).copy()
                for other_idx in range(n_stages):
                    if other_idx == s_idx or stage_results[other_idx] is None:
                        continue
                    other_result = stage_results[other_idx]
                    current_spectra = _deflate_fitted(
                        current_spectra.astype(np.float32),
                        stage_configs[other_idx],
                        other_result.amplitudes,
                        other_result.delta_E,
                        other_result.delta_sigma,
                    ).astype(np.float64)
                current_spectra = np.maximum(current_spectra, 0.0).astype(np.float32)

            # Fit this stage
            result = process_multipeak(
                current_spectra if isinstance(current_spectra, np.ndarray)
                    and current_spectra.dtype == np.float32
                    else current_spectra.astype(np.float32),
                stage_configs[s_idx],
                n_iterations=n_iterations,
                parabola_dE=parabola_dE,
                auto_constrain=False,
            )

            stage_results[s_idx] = result
            stage_residuals[s_idx] = current_spectra

    # Combine into unified result (original config order)
    amplitudes = np.zeros((N, n_comp), dtype=np.float32)
    delta_E = np.zeros((N, n_comp), dtype=np.float32)
    delta_sigma = np.zeros((N, n_comp), dtype=np.float32)

    for s_idx, group in enumerate(grouping.groups):
        result = stage_results[s_idx]
        for local_i, global_i in enumerate(group):
            amplitudes[:, global_i] = result.amplitudes[:, local_i]
            delta_E[:, global_i] = result.delta_E[:, local_i]
            delta_sigma[:, global_i] = result.delta_sigma[:, local_i]

    return GroupedResult(
        amplitudes=amplitudes,
        delta_E=delta_E,
        delta_sigma=delta_sigma,
        grouping=grouping,
        stage_results=list(stage_results),
        stage_residuals=list(stage_residuals),
    )
