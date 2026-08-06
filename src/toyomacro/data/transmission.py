"""Scienta analyzer transmission function T(Ek/Ep).

Loads transmission function data from Scienta XOP TSV files and provides
interpolation for XPS quantitative analysis.

Analyzer transmission is **instrument- and configuration-specific data
and is not bundled with toyomacro** — this module is an adapter only.
Users must supply curves they are authorized to use (see
``TOYOMACRO_SCIENTA_DATA_DIR`` below). Transmission is a separate
instrument-response factor, evaluated independently of the intrinsic
cross-section and IMFP terms; nothing here folds it into them for you.

The tabulated abscissa is the dimensionless ratio Ek/Ep, so pass energy
is *not* a table dimension: a curve is selected by (analyzer, lens mode,
slit, spot) and then evaluated at KE/Ep with the Ep you acquired at.
Slit and spot are physical apertures in **mm**, not energy widths.
Ratios outside the tabulated range are clamped to the endpoint values,
so a curve can silently run flat at the ends — check ``ek_ep_range``
against your KE/Ep span first. HAXPES conditions push KE/Ep well beyond
the range of curves measured for soft-x-ray work, where the clamp is
extrapolation in all but name.

Correction formula (from Scienta Transmission.ipf):
    I_normalized = I_measured / T(Ek/Ep) × 1000  →  Counts/Sr

The factor 1000 converts the tabulated mSr into Sr. It applies only
because these curves are stored in mSr; it is not a universal constant
and must not be carried over to transmission data in other units.

Apply transmission on **one** side of the quantification, never both.
Either correct the spectrum:

    I_corrected = I_measured / T(KE/Ep) × 1000
    quantity ∝ I_corrected / S_intrinsic          # S_intrinsic = σ × λ

or correct the sensitivity and leave the spectrum raw:

    S_instrument = S_intrinsic × T(KE/Ep) / 1000
    quantity ∝ I_measured / S_instrument

The two are algebraically the same. Dividing a transmission-corrected
spectrum by a transmission-corrected sensitivity applies T twice and is
a silent, energy-dependent error.

Note that σ × λ is an *intrinsic* sensitivity, not a complete AMRSF:
even with T applied it still omits elastic-scattering/EAL corrections,
the photoelectron angular distribution, x-ray polarization and the
source/analyzer geometry.

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
_DEFAULT_DATA_ROOT = Path.home() / "scienta_xop" / "Transmission" / "data"


def _data_root(data_root: Path | str | None = None) -> Path:
    """Resolve the data directory, honouring the environment each call.

    Read at call time, not import time: a session that sets
    ``TOYOMACRO_SCIENTA_DATA_DIR`` after importing toyomacro would
    otherwise be silently ignored and every lookup would report "no
    data".
    """
    if data_root is not None:
        return Path(data_root).expanduser()
    env = os.environ.get("TOYOMACRO_SCIENTA_DATA_DIR")
    return (Path(env).expanduser() if env else _DEFAULT_DATA_ROOT)


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

    Returns (slit_mm, spot_label) or None if parsing fails.
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
        Slit width in mm (the analyzer's physical aperture, not an
        energy).
    spot : str
        Spot size label in mm (e.g., "0.1x0.1").
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
        ek_ep = np.asarray(ek_ep, dtype=np.float64)
        transmission = np.asarray(transmission, dtype=np.float64)

        # Both axes are log-interpolated, and the endpoint clamp in
        # __call__ assumes ascending x — so validate and sort here rather
        # than silently returning interpolated nonsense.
        if ek_ep.shape != transmission.shape or ek_ep.ndim != 1:
            raise ValueError(
                f"ek_ep and transmission must be 1-D arrays of equal "
                f"length, got {ek_ep.shape} and {transmission.shape}")
        if ek_ep.size < 2:
            raise ValueError(
                f"need at least 2 points to interpolate, got {ek_ep.size}")
        if not (np.all(ek_ep > 0) and np.all(transmission > 0)):
            raise ValueError(
                "ek_ep and transmission must be strictly positive "
                "(both are log-interpolated)")
        if not np.isfinite(ek_ep).all() or not np.isfinite(transmission).all():
            raise ValueError("ek_ep and transmission must be finite")

        order = np.argsort(ek_ep)
        ek_ep, transmission = ek_ep[order], transmission[order]
        if np.any(np.diff(ek_ep) == 0):
            raise ValueError("ek_ep contains duplicate values")

        self.ek_ep = ek_ep
        self.transmission = transmission
        self.analyzer = analyzer
        self.mode = mode
        self.slit = slit
        self.spot = spot

        # Log-log interpolation for smooth power-law behavior
        log_x = np.log(ek_ep)
        log_y = np.log(transmission)
        self._interp = interp1d(
            log_x,
            log_y,
            kind="linear",
            bounds_error=False,
            fill_value=(log_y[0], log_y[-1]),
        )

        # Fit power law T = A × (Ek/Ep)^alpha for reference
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
        base = f"{self.analyzer} {self.mode}"
        # Synthetic curves (from_power_law) have no physical slit.
        return f"{base} Slit{self.slit}mm" if self.slit else base

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
            Slit width in mm (e.g., 0.5, 0.8, 1.0, 1.5, 2.5, 4.0 for
            EW4000). A physical aperture, not an energy width.
        spot : str
            Spot size in mm, like "0.1x0.1", "1x0.05", "2x0.3".
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
        root = _data_root(data_root)

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
        # The filename carries one decimal, so an off-grid request like
        # slit=0.54 would round to the 0.5 mm file and quietly apply the
        # wrong calibration. Refuse instead of guessing.
        if abs(float(slit_str.replace("p", ".")) - slit) > 1e-9:
            raise ValueError(
                f"slit={slit} is not an available slit width (filenames "
                f"carry one decimal). Use list_configs() to see the "
                f"widths present for {analyzer}/{mode}.")
        spot_str = spot.replace(".", "p")
        filename = f"Slit{slit_str}_spot{spot_str}.txt"

        filepath = root / subdir / filename
        if not filepath.exists():
            # Names only: the full paths made this message thousands of
            # characters long and leaked the data directory into logs.
            names = sorted(p.name for p in (root / subdir).glob("*.txt"))
            shown = ", ".join(names[:12])
            more = f", ... (+{len(names) - 12} more)" if len(names) > 12 else ""
            raise FileNotFoundError(
                f"Transmission file {filename!r} not found in "
                f"{analyzer}/{mode}. Available: {shown}{more}"
                if names else
                f"Transmission file {filename!r} not found and "
                f"{analyzer}/{mode} contains no .txt data. Is "
                f"TOYOMACRO_SCIENTA_DATA_DIR set correctly?"
            )

        data = np.loadtxt(filepath, delimiter="\t")
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(
                f"expected two tab-separated columns (Ek/Ep, T) in "
                f"{filename}, got array of shape {data.shape}")
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
        root = _data_root(data_root)
        return root.exists() and any(root.rglob("*.txt"))

    @staticmethod
    def list_configs(
        analyzer: str = "EW4000",
        mode: str = "Transmission",
        data_root: Path | str | None = None,
    ) -> list[tuple[float, str]]:
        """List available (slit, spot) configurations.

        Returns list of (slit_mm, spot_label) tuples.
        """
        root = _data_root(data_root)
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
        """List analyzer names that have data in at least one mode.

        An analyzer counts as present if *any* of its mode directories
        holds data — checking only the first one hid installs that carry,
        say, EW4000 angular modes but not the standard Transmission mode.
        """
        root = _data_root(data_root)
        if not root.exists():
            return []
        return [
            name for name, modes in _ANALYZER_MODES.items()
            if any(any((root / sub).glob("*.txt")) for sub in modes.values())
        ]
