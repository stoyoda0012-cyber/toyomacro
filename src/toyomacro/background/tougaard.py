"""Tougaard background algorithm.

Implements the universal 3-parameter Tougaard inelastic background using
the loss cross-section:

    K(T) = B * T / (C + T^2)^2

where T is the energy loss and B, C are constants.  The *universal*
cross-section (C = 1643 eV^2) is used by default.

References:
    S. Tougaard, Surf. Interface Anal. 11, 453 (1988)
    S. Tougaard, J. Vac. Sci. Technol. A 14, 1415 (1996)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from toyomacro.background.base import BaseBackground

# Universal loss cross-section constant (eV^2)
_UNIVERSAL_C: float = 1643.0


class Tougaard(BaseBackground):
    """Tougaard inelastic background.

    The background at energy E is the integral of the loss function
    weighted by the signal above E (for ascending energy / KE) or
    below E (for descending energy / BE).

    The scaling coefficient *B_tg* is determined automatically so that
    the background matches the spectrum level at the high-loss end.

    Parameters can be supplied explicitly or are auto-estimated from
    the spectrum endpoints (same convention as the MATLAB Toyomacro
    Tougaard implementation).
    """

    name = "tougaard"

    def calculate(
        self,
        energy: NDArray[np.float64],
        intensity: NDArray[np.float64],
        C: float = _UNIVERSAL_C,
        **params,
    ) -> NDArray[np.float64]:
        """Calculate Tougaard background.

        Args:
            energy: Energy axis (ascending or descending).
            intensity: Spectrum intensity.
            C: Loss function denominator constant (default 1643 eV^2).

        Returns:
            Background curve with the same shape as *intensity*.
        """
        n = len(intensity)
        if n < 3:
            return np.full(n, np.mean(intensity))

        energy = np.asarray(energy, dtype=np.float64)
        intensity = np.asarray(intensity, dtype=np.float64)

        # --- auto-estimate endpoint parameters --------------------------
        m = max(1, n // 10)  # 10% of spectrum
        i_left = float(np.mean(intensity[:m]))
        i_right = float(np.mean(intensity[-m:]))

        # a = higher endpoint, b = lower endpoint
        a = max(i_left, i_right)
        b = min(i_left, i_right)

        # If flat spectrum → constant background
        if a - b < 1e-10 * max(abs(a), 1.0):
            return np.full(n, b)

        # Direction: c == 1 → ascending (higher values at end → integrate forward)
        #            c == 2 → descending (higher values at start → integrate backward)
        ascending = i_right >= i_left
        energy_step = abs(energy[-1] - energy[0]) / (n - 1)

        # --- build energy loss matrix -----------------------------------
        idx = np.arange(n)
        # Eij[i, j] = (i - j) in energy units — positive means energy loss
        Eij = (idx[:, None] - idx[None, :]).astype(np.float64) * energy_step

        if ascending:
            # Forward integration: keep Eij >= 0 only
            Eij = np.maximum(Eij, 0.0)
        else:
            # Backward integration: flip sign, keep >= 0
            Eij = np.maximum(-Eij, 0.0)

        # --- loss function: T / (C + T^2)^2 -----------------------------
        denom = (C + Eij**2) ** 2
        # Avoid division by zero at T=0 (loss function is 0 there anyway)
        with np.errstate(divide="ignore", invalid="ignore"):
            loss = np.where(denom > 0, Eij / denom, 0.0)

        # --- weighted integral: (spectrum - b) * loss --------------------
        signal = intensity - b
        # dbg[i, j] = signal[j] * loss[i, j]
        dbg = signal[None, :] * loss  # broadcast (n, n)
        integral = dbg.sum(axis=1)    # sum over j → (n,)

        # --- determine scaling coefficient B_tg -------------------------
        if ascending:
            # Match at end (high-loss) region
            sig_end = np.mean(signal[-m:])
            int_end = np.mean(integral[-m:])
        else:
            # Match at start (high-loss) region
            sig_end = np.mean(signal[:m])
            int_end = np.mean(integral[:m])

        if abs(int_end) < 1e-30:
            return np.full(n, b)

        B_tg = sig_end / int_end

        # --- final background -------------------------------------------
        bg = B_tg * integral + b

        return bg
