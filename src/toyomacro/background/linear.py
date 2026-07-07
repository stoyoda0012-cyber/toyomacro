"""Linear background algorithm."""

import numpy as np
from numpy.typing import NDArray

from toyomacro.background.base import BaseBackground


class Linear(BaseBackground):
    """
    Simple linear background between spectrum endpoints.

    B(E) = I_left + (I_right - I_left) * (E - E_left) / (E_right - E_left)
    """

    name = "linear"

    def calculate(
        self,
        energy: NDArray[np.float64],
        intensity: NDArray[np.float64],
        **params,
    ) -> NDArray[np.float64]:
        """
        Calculate linear background.

        Args:
            energy: Energy axis
            intensity: Intensity values

        Returns:
            Linear background curve
        """
        e_left, e_right = energy[0], energy[-1]
        i_left, i_right = intensity[0], intensity[-1]

        # Linear interpolation
        slope = (i_right - i_left) / (e_right - e_left)
        background = i_left + slope * (energy - e_left)

        return background


class Constant(BaseBackground):
    """
    Constant background at minimum intensity level.
    """

    name = "constant"

    def calculate(
        self,
        energy: NDArray[np.float64],
        intensity: NDArray[np.float64],
        level: float | None = None,
        **params,
    ) -> NDArray[np.float64]:
        """
        Calculate constant background.

        Args:
            energy: Energy axis
            intensity: Intensity values
            level: Background level (default: minimum of endpoints)

        Returns:
            Constant background curve
        """
        if level is None:
            level = min(intensity[0], intensity[-1])

        return np.full_like(intensity, level)
