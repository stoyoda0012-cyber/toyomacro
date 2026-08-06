"""Base class for lineshape profiles."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseLineshape(ABC):
    """
    Abstract base class for lineshape profiles.

    All lineshape classes must implement the `evaluate` method.
    """

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        x: NDArray[np.float64],
        center: float,
        amplitude: float,
        **params,
    ) -> NDArray[np.float64]:
        """
        Evaluate the lineshape profile.

        Args:
            x: Energy axis
            center: Peak center position
            amplitude: Peak amplitude (area or height depending on normalization)
            **params: Additional lineshape-specific parameters

        Returns:
            Profile values at each x position
        """
        pass

    @abstractmethod
    def fwhm(self, **params) -> float:
        """
        Calculate the Full Width at Half Maximum.

        Args:
            **params: Lineshape-specific parameters

        Returns:
            FWHM value
        """
        pass

    def area_to_height(self, area: float, **params) -> float:
        """
        Convert area to peak height.

        Default implementation assumes area-normalized profile.
        Subclasses may override for different normalizations.
        """
        raise NotImplementedError("Subclass must implement area_to_height")

    def height_to_area(self, height: float, **params) -> float:
        """
        Convert peak height to area.

        Default implementation assumes area-normalized profile.
        """
        raise NotImplementedError("Subclass must implement height_to_area")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
