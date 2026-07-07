"""Base class for background algorithms."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseBackground(ABC):
    """Abstract base class for background subtraction algorithms."""

    name: str = "base"

    @abstractmethod
    def calculate(
        self,
        energy: NDArray[np.float64],
        intensity: NDArray[np.float64],
        **params,
    ) -> NDArray[np.float64]:
        """
        Calculate background curve.

        Args:
            energy: Energy axis
            intensity: Intensity values
            **params: Algorithm-specific parameters

        Returns:
            Background curve
        """
        pass

    def subtract(
        self,
        energy: NDArray[np.float64],
        intensity: NDArray[np.float64],
        **params,
    ) -> NDArray[np.float64]:
        """
        Subtract background from intensity.

        Args:
            energy: Energy axis
            intensity: Intensity values
            **params: Algorithm-specific parameters

        Returns:
            Background-subtracted intensity
        """
        background = self.calculate(energy, intensity, **params)
        return intensity - background

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
