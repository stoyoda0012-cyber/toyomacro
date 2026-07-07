"""Scienta analyzer transmission function T(Ek/Ep).

Loads transmission function data from Scienta XOP TSV files and provides
interpolation for XPS quantitative analysis.

Correction formula (from Scienta Transmission.ipf):
    I_normalized = I_measured / T(Ek/Ep) × 1000  →  Counts/Sr

For quantification:
    sensitivity = σ(hν) × λ(KE) × T(KE/Ep)
    at% ∝ Area / sensitivity

Supported analyzers:
    - EW4000: Transmission, Angular45, Angular56
    - Hipp3: T_Swift
    - R3000: Transmission
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

# Default Scienta XOP transmission-function data directory.
# Set ``TOYOMACRO_SCIENTA_DATA_DIR`` to point at your own
# ``ScientaXOP/Transmission/data`` folder. The bundled scanner
# expects the analyzer/mode subdirectory layout defined below.
_SCIENTA_DATA_ROOT = Path(
    os.environ.get(
        "TOYOMACRO_SCIENTA_DATA_DIR",
        str(Path.home() / "scienta_xop" / "Transmission" / "data"),
    )
).expanduser()


# Analyzer → mode → subdirectory mapping
_ANALYZER_MODES: dict[str, dict[str, str]] = {
    "EW4000": {
        "Transmission": "EW4000/Transmission",
        "Angular45": "EW4000/Angular45",
        "Angular56": "EW4000/Angular56",
    },
    "Hipp3": {
        "T_Swift": "Hipp3/T_Swift",
    },
    "R3000": {
        "Transmission": "R3000/Transmission",
    },
}


def _parse_filename(name: str) -> tuple[float, str] | None:
    """Parse slit and spot from filename like 'Slit0p5_spot1x0p3.txt'.

    Returns (slit_eV, spot_label) or None if parsing fails.
    """
    m = re.match(
        r"Slit(\d+p\d+)_spot(.+)\.txt$", name, re.IGNORECASE
    )
    if not m:
        return None
    slit_str = m.group(1).replace("p", ".")
    spot_label = m.group(2).replace("p", ".")
    return float(slit_str), spot_label


class TransmissionFunction:
    """Scienta analyzer transmission function with interpolation.

    Parameters
    ----------
    ek_ep : np.ndarray
        Energy ratio Ek/Ep (kinetic energy / pass energy).
    transmission : np.ndarray
        Transmission values in mSr.
    analyzer : str
        Analyzer name (e.g., "EW4000").
    mode : str
        Measurement mode (e.g., "Transmission").
    slit : float
        Slit width in eV.
    spot : str
        Spot size label (e.g., "0.1x0.1").
    """

    def __init__(
        self,
        ek_ep: np.ndarray,
        transmission: np.ndarray,
        analyzer: str = "EW4000",
        mode: str = "Transmission",
        slit: float = 0.5,
        spot: str = "0.1x0.1",
    ):
        self.ek_ep = ek_ep
        self.transmission = transmission
        self.analyzer = analyzer
        self.mode = mode
        self.slit = slit
        self.spot = spot

        # Log-log interpolation for smooth power-law behavior
        self._interp = interp1d(
            np.log(ek_ep),
            np.log(transmission),
            kind="linear",
            bounds_error=False,
            fill_value=(np.log(transmission[0]), np.log(transmission[-1])),
        )

        # Fit power law T = A × (Ek/Ep)^alpha for reference
        log_x = np.log(ek_ep)
        log_y = np.log(transmission)
        self.alpha, log_a = np.polyfit(log_x, log_y, 1)
        self._A = np.exp(log_a)

    def __call__(self, ek_ep: float | np.ndarray) -> float | np.ndarray:
        """Evaluate T(Ek/Ep) using log-log interpolation.

        Parameters
        ----------
        ek_ep : float or array
            Energy ratio KE / Ep.

        Returns
        -------
        float or array
            Transmission in mSr.
        """
        scalar = np.isscalar(ek_ep)
        x = np.atleast_1d(np.asarray(ek_ep, dtype=np.float64))
        x = np.clip(x, self.ek_ep[0], self.ek_ep[-1])
        result = np.exp(self._interp(np.log(x)))
        return float(result[0]) if scalar else result

    @property
    def label(self) -> str:
        """Human-readable label for display."""
        return f"{self.analyzer} {self.mode} Slit{self.slit}"

    @property
    def ek_ep_range(self) -> tuple[float, float]:
        """(min, max) of the Ek/Ep data."""
        return float(self.ek_ep[0]), float(self.ek_ep[-1])

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        analyzer: str = "EW4000",
        mode: str = "Transmission",
        slit: float = 0.5,
        spot: str = "0.1x0.1",
        data_root: Path | str | None = None,
    ) -> TransmissionFunction:
        """Load transmission function from Scienta TSV file.

        Parameters
        ----------
        analyzer : str
            Analyzer name: "EW4000", "Hipp3", "R3000".
        mode : str
            Mode: "Transmission", "Angular45", "Angular56", "T_Swift".
        slit : float
            Slit width in eV (e.g., 0.5, 0.8, 1.0, 1.5, 2.5, 4.0).
        spot : str
            Spot size like "0.1x0.1", "1x0.05", "2x0.3".
        data_root : Path, optional
            Override default Scienta data directory.

        Returns
        -------
        TransmissionFunction

        Raises
        ------
        FileNotFoundError
            If the data file is not found.
        """
        root = Path(data_root) if data_root else _SCIENTA_DATA_ROOT

        modes = _ANALYZER_MODES.get(analyzer, {})
        subdir = modes.get(mode)
        if subdir is None:
            raise ValueError(
                f"Unknown analyzer/mode: {analyzer}/{mode}. "
                f"Available: {list(_ANALYZER_MODES.keys())}"
            )

        # Build filename: Slit0p5_spot0p1x0p1.txt
        slit_str = f"{slit:.1f}".replace(".", "p").rstrip("0").rstrip("p")
        # Ensure at least one decimal: 0p5, 1p0, 2p5, 4p0
        if "p" not in slit_str:
            slit_str += "p0"
        spot_str = spot.replace(".", "p")
        filename = f"Slit{slit_str}_spot{spot_str}.txt"

        filepath = root / subdir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Transmission file not found: {filepath}\n"
                f"Available files: {list((root / subdir).glob('*.txt'))}"
            )

        data = np.loadtxt(filepath, delimiter="\t")
        # Remove any rows with NaN or zero
        mask = (data[:, 0] > 0) & (data[:, 1] > 0) & np.isfinite(data).all(axis=1)
        data = data[mask]

        return cls(
            ek_ep=data[:, 0],
            transmission=data[:, 1],
            analyzer=analyzer,
            mode=mode,
            slit=slit,
            spot=spot,
        )

    @classmethod
    def from_power_law(
        cls,
        alpha: float = -0.35,
        A: float = 45.0,
        ek_ep_range: tuple[float, float] = (1.0, 200.0),
        n_points: int = 50,
    ) -> TransmissionFunction:
        """Create a synthetic transmission function from power law.

        T(Ek/Ep) = A × (Ek/Ep)^alpha

        Useful when data files are not available.
        """
        ek_ep = np.geomspace(ek_ep_range[0], ek_ep_range[1], n_points)
        transmission = A * ek_ep ** alpha
        return cls(
            ek_ep=ek_ep,
            transmission=transmission,
            analyzer="Synthetic",
            mode=f"power_law(α={alpha:.2f})",
            slit=0.0,
            spot="",
        )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def available(
        data_root: Path | str | None = None,
    ) -> bool:
        """Check if Scienta transmission data is available."""
        root = Path(data_root) if data_root else _SCIENTA_DATA_ROOT
        return root.exists() and any(root.rglob("*.txt"))

    @staticmethod
    def list_configs(
        analyzer: str = "EW4000",
        mode: str = "Transmission",
        data_root: Path | str | None = None,
    ) -> list[tuple[float, str]]:
        """List available (slit, spot) configurations.

        Returns list of (slit_eV, spot_label) tuples.
        """
        root = Path(data_root) if data_root else _SCIENTA_DATA_ROOT
        modes = _ANALYZER_MODES.get(analyzer, {})
        subdir = modes.get(mode)
        if subdir is None:
            return []

        dirpath = root / subdir
        if not dirpath.exists():
            return []

        configs = []
        for f in sorted(dirpath.glob("*.txt")):
            parsed = _parse_filename(f.name)
            if parsed:
                configs.append(parsed)
        return configs

    @staticmethod
    def list_analyzers(
        data_root: Path | str | None = None,
    ) -> list[str]:
        """List available analyzer names."""
        root = Path(data_root) if data_root else _SCIENTA_DATA_ROOT
        if not root.exists():
            return []
        return [
            name for name in _ANALYZER_MODES
            if (root / list(_ANALYZER_MODES[name].values())[0]).exists()
        ]
