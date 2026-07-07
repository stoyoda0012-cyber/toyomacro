"""Shirley background algorithm."""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from toyomacro.background.base import BaseBackground


def find_bg_endpoints(
    energy: NDArray[np.float64],
    intensity: NDArray[np.float64],
    search_frac: float = 0.3,
    window: int = 7,
    edge_preference: float = 3.0,
) -> tuple[float, float, int, int]:
    """Find flat baseline regions at both ends of a spectrum.

    Searches the outer ``search_frac`` of each side for the window of
    length ``window`` with the smallest average |gradient|, with a
    preference for regions closer to the spectrum edge.

    The scoring function is::

        score = mean_gradient * (1 + edge_preference * distance_from_edge)

    where *distance_from_edge* is normalised to [0, 1] within the
    search region.  This ensures that, among equally flat regions, the
    one closest to the edge is preferred.

    Args:
        energy: Energy axis (ascending or descending).
        intensity: Intensity values.
        search_frac: Fraction of spectrum to search from each end.
        window: Number of points in the sliding window.
        edge_preference: Weight for preferring edge-proximal regions.
            0 = no preference (pure flatness), higher = stronger pull
            toward the edge.

    Returns:
        (i_left, i_right, left_idx, right_idx) where
        *i_left* / *i_right* are the averaged endpoint intensities and
        *left_idx* / *right_idx* are the inner-boundary indices of the
        detected flat regions (the Shirley BG is computed between these).
    """
    n = len(intensity)
    search_n = max(window + 2, int(n * search_frac))

    # Savitzky-Golay smoothing for robust gradient
    sw = min(window, n // 5)
    if sw % 2 == 0:
        sw += 1
    sw = max(sw, 5)
    smoothed = savgol_filter(intensity, sw, min(3, sw - 1))

    grad = np.abs(np.gradient(smoothed, energy))

    def _flattest_window_left(
        grad_region: NDArray, win: int
    ) -> tuple[int, int]:
        """Find flattest window, preferring positions closer to start."""
        n_search = len(grad_region)
        best_score = np.inf
        best_start = 0
        for start in range(n_search - win + 1):
            flatness = float(np.mean(grad_region[start : start + win]))
            # distance: 0 at start (edge), 1 at end of search region
            dist = (start + win / 2) / n_search
            score = flatness * (1.0 + edge_preference * dist)
            if score < best_score:
                best_score = score
                best_start = start
        return best_start, best_start + win

    def _flattest_window_right(
        grad_region: NDArray, win: int
    ) -> tuple[int, int]:
        """Find flattest window, preferring positions closer to end."""
        n_search = len(grad_region)
        best_score = np.inf
        best_start = 0
        for start in range(n_search - win + 1):
            flatness = float(np.mean(grad_region[start : start + win]))
            # distance: 0 at end (edge), 1 at start of search region
            dist = 1.0 - (start + win / 2) / n_search
            score = flatness * (1.0 + edge_preference * dist)
            if score < best_score:
                best_score = score
                best_start = start
        return best_start, best_start + win

    # Left side
    l_start, l_end = _flattest_window_left(grad[:search_n], window)
    i_left = float(np.mean(intensity[l_start:l_end]))
    left_idx = l_end - 1  # rightmost point of left flat region

    # Right side
    offset = n - search_n
    r_start, r_end = _flattest_window_right(grad[offset:], window)
    r_start += offset
    r_end += offset
    i_right = float(np.mean(intensity[r_start:r_end]))
    right_idx = r_start  # leftmost point of right flat region

    return i_left, i_right, left_idx, right_idx


class Shirley(BaseBackground):
    """
    Shirley (Proctor-Sherwood) iterative background.

    The Shirley background assumes that the background at any point is
    proportional to the integrated intensity of all peaks at higher
    kinetic energy (lower binding energy in XPS convention).

    Algorithm:
        B(E) = I_right + (I_left - I_right) * S(E) / S_total

    where:
        - I_left, I_right are intensities at the spectrum endpoints
        - S(E) is the cumulative sum of (intensity - background)
        - S_total is the total integrated intensity

    Reference:
        D.A. Shirley, Phys. Rev. B 5, 4709 (1972)
    """

    name = "shirley"

    def calculate(
        self,
        energy: NDArray[np.float64],
        intensity: NDArray[np.float64],
        max_iter: int = 50,
        tol: float = 1e-5,
        auto_range: bool = False,
        **params,
    ) -> NDArray[np.float64]:
        """
        Calculate Shirley background iteratively.

        Args:
            energy: Energy axis (can be ascending or descending)
            intensity: Intensity values
            max_iter: Maximum iterations (default 50)
            tol: Convergence tolerance (default 1e-5)
            auto_range: If True, automatically detect flat baseline
                regions and compute Shirley only within that range,
                extrapolating flat values outside.

        Returns:
            Shirley background curve
        """
        if auto_range:
            return self._calculate_auto_range(
                energy, intensity, max_iter=max_iter, tol=tol
            )

        return self._calculate_fixed(energy, intensity, max_iter=max_iter, tol=tol)

    def _calculate_fixed(
        self,
        energy: NDArray[np.float64],
        intensity: NDArray[np.float64],
        max_iter: int = 50,
        tol: float = 1e-5,
    ) -> NDArray[np.float64]:
        """Original fixed-endpoint Shirley calculation."""
        # Ensure we work with ascending energy for consistent integration
        if energy[0] > energy[-1]:
            energy = energy[::-1]
            intensity = intensity[::-1]
            reversed_input = True
        else:
            reversed_input = False

        n = len(intensity)
        background = np.zeros(n, dtype=np.float64)

        i_left = intensity[0]
        i_right = intensity[-1]

        for iteration in range(max_iter):
            signal = intensity - background
            cumsum = np.cumsum(signal)
            total = cumsum[-1]

            if abs(total) < 1e-10:
                break

            background_new = i_left + (i_right - i_left) * cumsum / total

            max_change = np.max(np.abs(background_new - background))
            if max_change < tol:
                background = background_new
                break

            background = background_new

        if reversed_input:
            background = background[::-1]

        return background

    def _calculate_auto_range(
        self,
        energy: NDArray[np.float64],
        intensity: NDArray[np.float64],
        max_iter: int = 50,
        tol: float = 1e-5,
    ) -> NDArray[np.float64]:
        """Shirley with automatic flat-region endpoint detection.

        Detects the flattest regions at each end of the spectrum, uses
        their mean intensities as the Shirley endpoints, and computes
        the iterative background only within the detected range.
        Outside that range the background is extrapolated as a constant.
        """
        n = len(intensity)
        i_left, i_right, left_idx, right_idx = find_bg_endpoints(energy, intensity)

        # Ensure left_idx < right_idx (valid sub-range)
        if left_idx >= right_idx:
            # Fallback to fixed-endpoint
            return self._calculate_fixed(energy, intensity, max_iter=max_iter, tol=tol)

        # Extract sub-range
        e_sub = energy[left_idx : right_idx + 1]
        i_sub = intensity[left_idx : right_idx + 1]

        # Ensure ascending energy
        if e_sub[0] > e_sub[-1]:
            e_sub = e_sub[::-1]
            i_sub = i_sub[::-1]
            sub_i_left, sub_i_right = i_right, i_left
            reversed_sub = True
        else:
            sub_i_left, sub_i_right = i_left, i_right
            reversed_sub = False

        n_sub = len(i_sub)
        bg_sub = np.zeros(n_sub, dtype=np.float64)

        for _ in range(max_iter):
            signal = i_sub - bg_sub
            cumsum = np.cumsum(signal)
            total = cumsum[-1]

            if abs(total) < 1e-10:
                break

            bg_new = sub_i_left + (sub_i_right - sub_i_left) * cumsum / total

            if np.max(np.abs(bg_new - bg_sub)) < tol:
                bg_sub = bg_new
                break

            bg_sub = bg_new

        if reversed_sub:
            bg_sub = bg_sub[::-1]

        # Assemble full background with flat extrapolation
        background = np.empty(n, dtype=np.float64)
        background[:left_idx] = i_left
        background[left_idx : right_idx + 1] = bg_sub
        background[right_idx + 1 :] = i_right

        return background


class ShirleyWithOffset(Shirley):
    """
    Shirley background with adjustable endpoint offsets.

    Useful when the spectrum doesn't reach a true baseline at the endpoints.
    """

    name = "shirley_offset"

    def calculate(
        self,
        energy: NDArray[np.float64],
        intensity: NDArray[np.float64],
        left_offset: float = 0.0,
        right_offset: float = 0.0,
        **params,
    ) -> NDArray[np.float64]:
        """
        Calculate Shirley background with endpoint offsets.

        Args:
            energy: Energy axis
            intensity: Intensity values
            left_offset: Offset to add to left endpoint
            right_offset: Offset to add to right endpoint
            **params: Additional parameters passed to parent

        Returns:
            Shirley background curve
        """
        # Adjust endpoints
        intensity_adjusted = intensity.copy()
        intensity_adjusted[0] += left_offset
        intensity_adjusted[-1] += right_offset

        return super().calculate(energy, intensity_adjusted, **params)
