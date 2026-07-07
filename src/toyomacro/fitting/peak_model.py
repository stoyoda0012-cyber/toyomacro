"""Peak models for fitting using lmfit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from lmfit import Model, Parameters
from lmfit.models import GaussianModel, LorentzianModel, PseudoVoigtModel, VoigtModel
from numpy.typing import NDArray

if TYPE_CHECKING:
    from lmfit.model import ModelResult


def _doniach_sunjic_func(
    x: NDArray[np.float64],
    center: float = 0.0,
    amplitude: float = 1.0,
    sigma: float = 0.3,
    gamma: float = 0.25,
    asymmetry: float = 0.1,
) -> NDArray[np.float64]:
    """Doniach-Sunjic model function for lmfit.

    Uses the toyomacro DoniachSunjic class internally,
    but maps lmfit parameter names (sigma, gamma) to DS names (fwhm_g, fwhm_l).

    Args:
        x: Energy axis
        center: Peak center
        amplitude: Integrated area
        sigma: Gaussian sigma (fwhm_g = sigma * 2.3548)
        gamma: Lorentzian half-width (fwhm_l = gamma * 2)
        asymmetry: Singularity index alpha
    """
    from toyomacro.lineshape.doniach_sunjic import DoniachSunjic

    ds = DoniachSunjic()
    fwhm_g = sigma * 2.3548  # sigma -> FWHM
    fwhm_l = gamma * 2.0     # half-width -> FWHM
    return ds.evaluate(x, center, amplitude, fwhm_g=fwhm_g, fwhm_l=fwhm_l, asymmetry=asymmetry)


class PeakModel:
    """
    Single peak model wrapper for lmfit.

    Supports Voigt, Gaussian, Lorentzian, and Doniach-Sunjic profiles.
    """

    def __init__(
        self,
        prefix: str = "",
        lineshape: str = "voigt",
    ):
        """
        Initialize peak model.

        Args:
            prefix: Parameter prefix (e.g., "p1_" for peak 1)
            lineshape: Lineshape type ("voigt", "gaussian", "lorentzian", "doniach_sunjic")
        """
        self.prefix = prefix
        self.lineshape = lineshape.lower()

        # Create appropriate lmfit model
        if self.lineshape == "voigt":
            self.model = VoigtModel(prefix=prefix)
        elif self.lineshape == "gaussian":
            self.model = GaussianModel(prefix=prefix)
        elif self.lineshape == "lorentzian":
            self.model = LorentzianModel(prefix=prefix)
        elif self.lineshape == "pseudovoigt":
            self.model = PseudoVoigtModel(prefix=prefix)
        elif self.lineshape == "doniach_sunjic":
            self.model = Model(_doniach_sunjic_func, prefix=prefix)
        else:
            raise ValueError(f"Unknown lineshape: {lineshape}")

    def make_params(
        self,
        center: float,
        amplitude: float | None = None,
        sigma: float = 0.3,
        gamma: float | None = None,
        asymmetry: float | None = None,
        **kwargs,
    ) -> Parameters:
        """
        Create initial parameters for this peak.

        Args:
            center: Peak center position
            amplitude: Peak amplitude (area)
            sigma: Gaussian width
            gamma: Lorentzian width (for Voigt / Doniach-Sunjic)
            asymmetry: Singularity index (for Doniach-Sunjic)

        Returns:
            lmfit Parameters object
        """
        params = self.model.make_params()

        # Set center
        params[f"{self.prefix}center"].set(value=center, min=center - 2, max=center + 2)

        # Set amplitude
        if amplitude is not None:
            params[f"{self.prefix}amplitude"].set(value=amplitude, min=0)

        # Set sigma (Gaussian width)
        params[f"{self.prefix}sigma"].set(value=sigma, min=0.1, max=2.0)

        # Set gamma for Voigt / Doniach-Sunjic
        # Note: lmfit VoigtModel sets gamma.vary=False by default (tied to sigma).
        # We must explicitly set vary=True to allow independent Lorentzian optimization.
        if self.lineshape in ("voigt", "doniach_sunjic") and gamma is not None:
            params[f"{self.prefix}gamma"].set(value=gamma, min=0.05, max=1.0, vary=True)

        # PseudoVoigt: fraction parameter (0 = pure Gaussian, 1 = pure Lorentzian)
        # When asymmetry is provided, use it as initial fraction (PV stores fraction in asymmetry)
        if self.lineshape == "pseudovoigt":
            frac_key = f"{self.prefix}fraction"
            if frac_key in params:
                frac_init = asymmetry if asymmetry is not None and 0 <= asymmetry <= 1 else 0.5
                params[frac_key].set(value=frac_init, min=0.0, max=1.0, vary=True)

        # Set asymmetry for Doniach-Sunjic
        if self.lineshape == "doniach_sunjic":
            asym_val = asymmetry if asymmetry is not None else 0.1
            params[f"{self.prefix}asymmetry"].set(
                value=asym_val, min=0.0, max=0.5, vary=True
            )

        return params


class MultiPeakModel:
    """
    Multi-peak model for fitting multiple peaks simultaneously.
    """

    def __init__(self, lineshape: str = "voigt"):
        """
        Initialize multi-peak model.

        Args:
            lineshape: Default lineshape for all peaks
        """
        self.lineshape = lineshape
        self.peaks: list[PeakModel] = []
        self.composite_model: Model | None = None

    def add_peak(
        self,
        center: float,
        amplitude: float | None = None,
        sigma: float = 0.3,
        gamma: float | None = None,
        asymmetry: float | None = None,
        prefix: str | None = None,
        lineshape: str | None = None,
    ) -> PeakModel:
        """
        Add a peak to the model.

        Args:
            center: Initial peak center
            amplitude: Initial amplitude
            sigma: Initial Gaussian width
            gamma: Initial Lorentzian width
            asymmetry: Singularity index (for Doniach-Sunjic)
            prefix: Parameter prefix (auto-generated if None)
            lineshape: Override default lineshape

        Returns:
            The added PeakModel
        """
        if prefix is None:
            prefix = f"p{len(self.peaks) + 1}_"

        peak = PeakModel(
            prefix=prefix,
            lineshape=lineshape or self.lineshape,
        )
        self.peaks.append(peak)

        # Store initial values
        peak._init_center = center
        peak._init_amplitude = amplitude
        peak._init_sigma = sigma
        peak._init_gamma = gamma
        peak._init_asymmetry = asymmetry

        # Rebuild composite model
        self._build_composite()

        return peak

    def _build_composite(self):
        """Build the composite model from all peaks."""
        if not self.peaks:
            self.composite_model = None
            return

        self.composite_model = self.peaks[0].model
        for peak in self.peaks[1:]:
            self.composite_model = self.composite_model + peak.model

    def make_params(self) -> Parameters:
        """
        Create combined parameters for all peaks.

        Returns:
            lmfit Parameters object
        """
        if not self.peaks:
            return Parameters()

        params = Parameters()
        for peak in self.peaks:
            peak_params = peak.make_params(
                center=peak._init_center,
                amplitude=peak._init_amplitude,
                sigma=peak._init_sigma,
                gamma=peak._init_gamma,
                asymmetry=getattr(peak, "_init_asymmetry", None),
            )
            params.update(peak_params)

        return params

    def fit(
        self,
        energy: NDArray[np.float64],
        intensity: NDArray[np.float64],
        params: Parameters | None = None,
        **kwargs,
    ) -> ModelResult:
        """
        Fit the model to data.

        Args:
            energy: Energy axis
            intensity: Intensity values
            params: Initial parameters (auto-generated if None)
            **kwargs: Additional arguments to model.fit()

        Returns:
            lmfit ModelResult
        """
        if self.composite_model is None:
            raise ValueError("No peaks added to model")

        if params is None:
            params = self.make_params()

        # lmfit's Model.fit() deep-copies params, which defeats any
        # pre-stripping we do.  Instead we deep-copy once ourselves,
        # strip the expensive view-only expressions (fwhm/height that
        # contain wofz() calls evaluated via asteval every iteration),
        # then build ModelResult directly — bypassing the redundant copy.
        from copy import deepcopy

        from lmfit.model import ModelResult

        params = deepcopy(params)

        _saved_exprs: dict[str, str] = {}
        for name, par in params.items():
            if par.expr and ("fwhm" in name or "height" in name):
                _saved_exprs[name] = par.expr
                par.expr = None
                par.vary = False

        # Build ModelResult manually (mirrors lmfit Model.fit internals)
        result = ModelResult(
            self.composite_model, params, method="leastsq",
            fcn_kws={"x": energy}, **kwargs,
        )
        result.fit(data=intensity)
        result.components = self.composite_model.components

        # Restore fwhm/height expressions for reporting
        for name, expr in _saved_exprs.items():
            result.params[name].expr = expr
        if _saved_exprs:
            result.params.update_constraints()

        return result

    def evaluate(
        self,
        energy: NDArray[np.float64],
        params: Parameters,
    ) -> NDArray[np.float64]:
        """
        Evaluate the model with given parameters.

        Args:
            energy: Energy axis
            params: Parameters to use

        Returns:
            Model values
        """
        if self.composite_model is None:
            return np.zeros_like(energy)

        return self.composite_model.eval(params, x=energy)

    @property
    def n_peaks(self) -> int:
        """Number of peaks in the model."""
        return len(self.peaks)

    def clear(self):
        """Remove all peaks."""
        self.peaks.clear()
        self.composite_model = None
