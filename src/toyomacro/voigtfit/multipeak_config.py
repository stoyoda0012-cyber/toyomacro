"""
Multi-Peak Configuration for Alternating Projection Solver.

dev-log 56 Phase 3 → Production implementation.

Design confirmed by Phase 3 experiments:
    - δE: 10pt, Uniform ±2.0, parabola refinement (essential)
    - δσ: 31pt, Uniform ±0.3, parabola optional (limited effect)
    - Alternating Projection: 3 iterations sufficient
    - Separation limit: Δc ≥ 3.5σ for reliable decomposition
"""

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class ComponentConfig:
    """Configuration for a single peak component (or SO doublet).

    Defines the nominal peak parameters and grid search ranges
    for one component in a multi-peak fit.

    When ``so_split > 0``, the component is treated as an SO doublet:
    a main peak at ``center`` plus a partner at ``center + so_split``
    with intensity ratio ``branch_ratio`` (main/partner).  The doublet
    shares σ and γ and shifts rigidly as one logical component.

    Attributes:
        center: Nominal peak position (eV) of the main line.
        sigma: Nominal Gaussian width (eV).
        gamma: Lorentzian width (eV).
        dE_range: δE search half-range (eV). Grid spans [-dE_range, +dE_range].
        ds_range: δσ search half-range (eV). Grid spans [-ds_range, +ds_range].
        n_dE: Number of δE grid points.
        n_ds: Number of δσ grid points.
        so_split: Spin-orbit splitting (eV). 0 = singlet (no partner).
            Sign convention: partner at ``center + so_split``.
        branch_ratio: Intensity ratio I_main / I_partner (e.g. 4/3 for f-shell).
    """

    center: float
    sigma: float
    gamma: float
    dE_range: float = 2.0
    ds_range: float = 0.3
    n_dE: int = 10
    n_ds: int = 31
    so_split: float = 0.0
    branch_ratio: float = 1.0

    @property
    def is_doublet(self) -> bool:
        """True if this component is an SO doublet."""
        return self.so_split != 0.0


@dataclass
class MultiPeakConfig:
    """Configuration for multi-component peak fitting.

    Wraps a list of ComponentConfig with the shared energy axis.

    Usage::

        config = MultiPeakConfig(
            peaks=[
                ComponentConfig(center=284.8, sigma=0.425, gamma=0.125),
                ComponentConfig(center=286.5, sigma=0.425, gamma=0.125),
            ],
            energy_axis=np.linspace(280, 290, 101),
        )
    """

    peaks: list[ComponentConfig]
    energy_axis: np.ndarray

    @property
    def n_comp(self) -> int:
        return len(self.peaks)

    def separation_sigma(self) -> float:
        """Minimum peak separation in units of average σ."""
        if self.n_comp < 2:
            return float("inf")
        distances = []
        for i in range(self.n_comp):
            for j in range(i + 1, self.n_comp):
                dc = abs(self.peaks[i].center - self.peaks[j].center)
                sigma_avg = (self.peaks[i].sigma + self.peaks[j].sigma) / 2
                distances.append(dc / sigma_avg)
        return min(distances)

    def check_separation(self, threshold: float = 3.5) -> None:
        """Warn if peak separation is below threshold."""
        sep = self.separation_sigma()
        if sep < threshold:
            warnings.warn(
                f"Peak separation {sep:.1f}σ < {threshold}σ: "
                f"reliable separation may not be achievable.",
                stacklevel=2,
            )

    def solvability(
        self,
        snr: float = 100.0,
        amplitudes: np.ndarray | None = None,
        noise_model: str = 'gaussian',
    ) -> "SolvabilityInfo":
        """Assess solvability of this multi-peak configuration.

        Computes the Cramer-Rao Lower Bound (CRLB) for the current peak
        arrangement and classifies difficulty as EASY / HARD / SHOULDER /
        IMPOSSIBLE based on the theoretical minimum variance for center
        shift estimation.

        Args:
            snr: Signal-to-noise ratio (default 100)
            amplitudes: Peak amplitudes, shape (n_comp,). Default: all 1.0
            noise_model: 'gaussian' or 'poisson'

        Returns:
            SolvabilityInfo with level, CRLB bounds (meV), condition number
        """
        from .crlb import classify_solvability

        centers = np.array([p.center for p in self.peaks])
        sigmas = np.array([p.sigma for p in self.peaks])
        gammas = np.array([p.gamma for p in self.peaks])

        return classify_solvability(
            centers=centers,
            sigmas=sigmas,
            gammas=gammas,
            amplitudes=amplitudes,
            energy=self.energy_axis.astype(np.float64),
            snr=snr,
            noise_model=noise_model,
        )

    def constrain_dE_ranges(self, min_dE_range: float = 0.1) -> "MultiPeakConfig":
        """Constrain each component's dE_range to prevent cross-assignment.

        For each component, limits dE_range so the dictionary cannot reach
        any neighbouring peak's nominal center.  The constraint is::

            dE_range_k = min(original, nearest_dist / 2 - σ_avg)

        where ``nearest_dist`` is the distance to the closest other peak
        and ``σ_avg`` is the average Gaussian width of the pair.

        This prevents the matmul-argmax from preferring a strong neighbour
        over a weak component's own peak.

        Args:
            min_dE_range: Floor value for dE_range (eV). Ensures at least
                a minimal search range even at very close separations.

        Returns:
            Self (modified in-place for convenience).
        """
        if self.n_comp < 2:
            return self

        for i, peak_i in enumerate(self.peaks):
            nearest = float("inf")
            for j, peak_j in enumerate(self.peaks):
                if i == j:
                    continue
                dist = abs(peak_i.center - peak_j.center)
                sigma_avg = (peak_i.sigma + peak_j.sigma) / 2
                nearest = min(nearest, dist / 2 - sigma_avg)

            if nearest < peak_i.dE_range:
                new_range = max(nearest, min_dE_range)
                peak_i.dE_range = new_range

        return self
