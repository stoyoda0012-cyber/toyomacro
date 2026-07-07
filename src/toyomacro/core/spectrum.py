"""Spectrum data class for XPS data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class Spectrum:
    """
    XPS spectrum data container.

    Attributes:
        energy: Energy axis (Binding Energy or Kinetic Energy) in eV
        intensity: Intensity values (counts or counts/s)
        energy_type: Type of energy axis ("BE" for Binding Energy, "KE" for Kinetic Energy)
        element: Element and orbital (e.g., "Si2p", "C1s")
        metadata: Additional metadata dictionary
    """

    energy: NDArray[np.float64]
    intensity: NDArray[np.float64]
    energy_type: Literal["BE", "KE"] = "BE"
    element: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate data after initialization."""
        self.energy = np.asarray(self.energy, dtype=np.float64)
        self.intensity = np.asarray(self.intensity, dtype=np.float64)

        if self.energy.shape != self.intensity.shape:
            raise ValueError(
                f"Energy and intensity must have same shape: "
                f"{self.energy.shape} vs {self.intensity.shape}"
            )

        if self.energy.ndim != 1:
            raise ValueError(f"Expected 1D arrays, got {self.energy.ndim}D")

    @property
    def n_points(self) -> int:
        """Number of data points."""
        return len(self.energy)

    @property
    def energy_range(self) -> tuple[float, float]:
        """Energy range (min, max)."""
        return float(self.energy.min()), float(self.energy.max())

    @property
    def energy_step(self) -> float:
        """Average energy step size."""
        return float(np.abs(np.mean(np.diff(self.energy))))

    @property
    def is_ascending(self) -> bool:
        """Check if energy is in ascending order."""
        return bool(self.energy[0] < self.energy[-1])

    def to_ascending(self) -> Spectrum:
        """Return spectrum with energy in ascending order."""
        if self.is_ascending:
            return self
        return Spectrum(
            energy=self.energy[::-1].copy(),
            intensity=self.intensity[::-1].copy(),
            energy_type=self.energy_type,
            element=self.element,
            metadata=self.metadata.copy(),
        )

    def slice(self, e_min: float, e_max: float) -> Spectrum:
        """Extract a slice of the spectrum within energy range."""
        mask = (self.energy >= e_min) & (self.energy <= e_max)
        return Spectrum(
            energy=self.energy[mask].copy(),
            intensity=self.intensity[mask].copy(),
            energy_type=self.energy_type,
            element=self.element,
            metadata=self.metadata.copy(),
        )

    def to_energy_type(
        self, target: Literal["BE", "KE"], photon_energy: float
    ) -> Spectrum:
        """Convert energy axis between BE and KE.

        Uses the relation KE = hν - BE (and vice versa).

        Args:
            target: Target energy type ("BE" or "KE").
            photon_energy: Photon energy hν in eV.

        Returns:
            New Spectrum with converted energy axis, or self if already
            in the target type.
        """
        if self.energy_type == target:
            return self
        return Spectrum(
            energy=(photon_energy - self.energy),
            intensity=self.intensity.copy(),
            energy_type=target,
            element=self.element,
            metadata={**self.metadata, "photon_energy": photon_energy},
        )

    def normalize(self, method: Literal["max", "area", "minmax"] = "max") -> Spectrum:
        """
        Return normalized spectrum.

        Args:
            method: Normalization method
                - "max": Divide by maximum intensity
                - "area": Divide by integrated area
                - "minmax": Scale to [0, 1] range
        """
        if method == "max":
            factor = self.intensity.max()
        elif method == "area":
            factor = np.trapezoid(self.intensity, self.energy)
        elif method == "minmax":
            i_min, i_max = self.intensity.min(), self.intensity.max()
            return Spectrum(
                energy=self.energy.copy(),
                intensity=(self.intensity - i_min) / (i_max - i_min),
                energy_type=self.energy_type,
                element=self.element,
                metadata={**self.metadata, "normalization": method},
            )
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return Spectrum(
            energy=self.energy.copy(),
            intensity=self.intensity / factor,
            energy_type=self.energy_type,
            element=self.element,
            metadata={**self.metadata, "normalization": method},
        )

    def copy(self) -> Spectrum:
        """Create a deep copy of the spectrum."""
        return Spectrum(
            energy=self.energy.copy(),
            intensity=self.intensity.copy(),
            energy_type=self.energy_type,
            element=self.element,
            metadata=self.metadata.copy(),
        )

    def __repr__(self) -> str:
        element_str = f", element={self.element!r}" if self.element else ""
        return (
            f"Spectrum(n_points={self.n_points}, "
            f"energy_range={self.energy_range}{element_str})"
        )
