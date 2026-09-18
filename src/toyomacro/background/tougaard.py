"""Tougaard background algorithm.

Implements the **two-parameter** Universal Tougaard inelastic background
using the loss cross-section

    K(T) = B * T / (C + T^2)^2

where T is the energy loss.  The source gives it for most metals, their
oxides and alloys, with C = 1643 eV^2 and B ~ 3000 eV^2.  Here C
defaults to 1643 eV^2 (it can be overridden per call); B is not taken
from the literature but determined from the spectrum, by matching the
background to the signal level at the high-loss end (see `Tougaard`).
The loss integral is a plain sum over channels, so that scale is not in
eV^2 and is not comparable with the literature B.

This is not the *three-parameter* Universal cross-section,
K(T) = B*T / [(C - T^2)^2 + D*T^2], whose B, C, D are tabulated per
class of materials (e.g. polymers, semiconductors, free-electron-like
solids).  Validity limits of the two-parameter form, as summarised in
Tougaard (1998) from the 1997 critical review:

- quite accurate when the loss cross-section FWHM is >~ 20 eV;
- at a FWHM of 10-15 eV, still fairly good far from the peak
  (>~ 30 eV loss) but less accurate near it (<~ 10 eV loss);
- at a FWHM <~ 5 eV (narrow plasmon structure), the three-parameter
  form is always more accurate.

The source does not place individual elements in these bands.  Si
belongs to a class it lists for the three-parameter form, and has a
narrow bulk plasmon, so a Si analysis with this background class should
state these limits; that is an inference, not a value from the source.

References:
    Universal cross-section, C = 1643 eV^2:
        S. Tougaard, Solid State Commun. 61 (1987) 547,
        doi:10.1016/0038-1098(87)90166-9
    The background algorithm:
        S. Tougaard, Surf. Interface Anal. 11 (1988) 453,
        doi:10.1002/sia.740110902
    Critical review, two- and three-parameter forms:
        S. Tougaard, Surf. Interface Anal. 25 (1997) 137
    Validity limits as quoted above (Eqns 8-9 and the text after them):
        S. Tougaard, Surf. Interface Anal. 26 (1998) 249
    Related:
        S. Tougaard, J. Vac. Sci. Technol. A 14 (1996) 1415
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
