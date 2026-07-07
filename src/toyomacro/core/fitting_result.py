"""Fitting result data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


class LineshapeType(IntEnum):
    """Lineshape function types (matching MATLAB TypeID convention).

    MATLAB source: +toyomacro/+lineshape/{Voigt,PseudoVoigt,...}.m TypeID property.
    """

    VOIGT = 1
    PSEUDO_VOIGT = 2
    GAUSSIAN = 3
    LORENTZIAN = 4
    DONIACH_SUNJIC = 5
    FERMI_DIRAC = 6


@dataclass
class PeakComponent:
    """
    Single peak component parameters.

    Field mapping to MATLAB fitpara columns:
        area  → col 0: Peak height (intensity at center, Int0)
        center → col 1: Peak center position (energy)
        fwhm_g → col 2: Gaussian FWHM
        fwhm_l → col 3: Lorentzian FWHM
        asymmetry → col 4: Asymmetry parameter (for Doniach-Sunjic)
        branch_ratio → col 5: Spin-orbit branch ratio
        so_split → col 6: Spin-orbit splitting (eV)
        lineshape → col 7: Lineshape type
        height → col 8: Integrated area (trapz)

    Note: ``area`` stores peak height and ``height`` stores integrated area.
    This matches the MATLAB fitpara convention where col 0 = Int0 (peak height)
    and col 8 = integrated area.  For Voigt amplitude (integrated area), use
    ``height``; for display peak intensity, use ``area``.
    """

    area: float
    center: float
    fwhm_g: float = 0.5
    fwhm_l: float = 0.5
    asymmetry: float = 0.0
    branch_ratio: float = 0.0
    so_split: float = 0.0
    lineshape: LineshapeType = LineshapeType.VOIGT
    height: float | None = None

    def to_array(self) -> NDArray[np.float64]:
        """Convert to 9-element array (MATLAB fitpara format)."""
        return np.array([
            self.area,
            self.center,
            self.fwhm_g,
            self.fwhm_l,
            self.asymmetry,
            self.branch_ratio,
            self.so_split,
            float(self.lineshape),
            self.height if self.height is not None else 0.0,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> PeakComponent:
        """Create from 9-element array (MATLAB fitpara format)."""
        return cls(
            area=float(arr[0]),
            center=float(arr[1]),
            fwhm_g=float(arr[2]),
            fwhm_l=float(arr[3]),
            asymmetry=float(arr[4]),
            branch_ratio=float(arr[5]),
            so_split=float(arr[6]),
            lineshape=LineshapeType(int(arr[7])),
            height=float(arr[8]) if arr[8] != 0 else None,
        )


@dataclass
class BackgroundParams:
    """Background parameters."""

    type: str  # "shirley", "tougaard", "linear", "constant"
    param_a: float = 0.0
    param_b: float = 0.0
    param_c: float = 0.0


@dataclass
class FittingResult:
    """
    Complete fitting result for a spectrum.

    Attributes:
        components: List of peak components
        background: Background parameters
        energy: Energy axis used for fitting
        fitted_intensity: Total fitted curve
        background_curve: Background curve
        residuals: Fitting residuals
        chi2: Chi-squared value
        r_squared: R² goodness of fit
        metadata: Additional metadata
    """

    components: list[PeakComponent]
    background: BackgroundParams
    energy: NDArray[np.float64]
    fitted_intensity: NDArray[np.float64]
    background_curve: NDArray[np.float64]
    residuals: NDArray[np.float64]
    chi2: float = 0.0
    r_squared: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        """Number of peak components."""
        return len(self.components)

    @property
    def total_area(self) -> float:
        """Total integrated area of all peaks.

        Uses ``height`` field (= integrated area, col 8) when available,
        falls back to ``area`` field for backward compatibility.
        """
        return sum(
            (c.height if c.height is not None else c.area)
            for c in self.components
        )

    def get_component_curve(
        self, index: int, energy: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """
        Calculate curve for a single component.

        This is a placeholder - actual implementation requires lineshape module.
        """
        raise NotImplementedError("Requires lineshape module")

    def to_fitpara(self) -> NDArray[np.float64]:
        """
        Convert to MATLAB-compatible fitpara matrix (9 x n_components).
        """
        if not self.components:
            return np.zeros((9, 0), dtype=np.float64)
        return np.column_stack([c.to_array() for c in self.components])

    def to_otherpara(self) -> NDArray[np.float64]:
        """
        Convert to MATLAB-compatible otherpara vector (1 x 10).
        """
        e_min, e_max = self.energy.min(), self.energy.max()
        bg_type_map = {"shirley": 1, "tougaard": 2, "linear": 3, "constant": 4}
        bg_type = bg_type_map.get(self.background.type, 0)

        max_height = max((c.height or 0 for c in self.components), default=0)

        return np.array([
            self.n_components,
            e_min,
            e_max,
            bg_type,
            self.background.param_a,
            self.background.param_b,
            self.background.param_c,
            max_height,
            self.total_area,
            self.chi2,
        ], dtype=np.float64)

    @classmethod
    def from_matlab(
        cls,
        fitpara: NDArray[np.float64],
        otherpara: NDArray[np.float64],
        energy: NDArray[np.float64],
    ) -> FittingResult:
        """
        Create from MATLAB format arrays.

        Args:
            fitpara: 9 x n_components parameter matrix
            otherpara: 1 x 10 other parameters vector
            energy: Energy axis
        """
        # Parse components
        n_components = int(otherpara[0])
        components = []
        for i in range(n_components):
            if i < fitpara.shape[1]:
                components.append(PeakComponent.from_array(fitpara[:, i]))

        # Parse background
        bg_type_map = {1: "shirley", 2: "tougaard", 3: "linear", 4: "constant"}
        bg_type = bg_type_map.get(int(otherpara[3]), "linear")
        background = BackgroundParams(
            type=bg_type,
            param_a=otherpara[4],
            param_b=otherpara[5],
            param_c=otherpara[6],
        )

        # Placeholder arrays (need actual calculation)
        n_points = len(energy)
        return cls(
            components=components,
            background=background,
            energy=energy,
            fitted_intensity=np.zeros(n_points),
            background_curve=np.zeros(n_points),
            residuals=np.zeros(n_points),
            chi2=otherpara[9],
        )

    def __repr__(self) -> str:
        return (
            f"FittingResult(n_components={self.n_components}, "
            f"r_squared={self.r_squared:.4f}, chi2={self.chi2:.2e})"
        )
