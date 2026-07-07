"""Automatic peak fitting for XPS spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, savgol_filter

from toyomacro.background.shirley import Shirley
from toyomacro.core.fitting_result import (
    BackgroundParams,
    FittingResult,
    LineshapeType,
    PeakComponent,
)
from toyomacro.core.xps_database import SOInfo, get_so_info
from toyomacro.fitting.peak_model import MultiPeakModel
from toyomacro.fitting.templates import FittingTemplate, TemplatePeak, get_template

if TYPE_CHECKING:
    from toyomacro.core.spectrum import Spectrum
    from toyomacro.voigtfit.auto_grouping import GroupedResult


@dataclass
class AutoFitConfig:
    """Configuration for automatic fitting."""

    # Peak detection
    min_peak_height: float = 0.05  # Relative to max intensity
    min_peak_distance: int = 10  # Minimum points between peaks
    min_peak_prominence: float = 0.03  # Relative prominence (to max signal)
    min_relative_prominence: float = 0.04  # Min prominence relative to dominant peak

    # Adaptive noise thresholds for peak detection
    noise_prominence_sigma: float = 5.0  # Prominence must exceed this × noise σ
    noise_height_sigma: float = 3.0  # Height must exceed this × noise σ

    # Fitting
    lineshape: str = "voigt"
    max_peaks: int = 10
    initial_sigma: float = 0.3
    initial_gamma: float = 0.2

    # Background
    background_type: str = "shirley"
    background_auto_range: bool = True  # Auto-detect flat regions for Shirley endpoints

    # Smoothing for peak detection
    smooth_window: int = 11
    smooth_order: int = 3


@dataclass
class AutoFitResult:
    """Result from automatic fitting."""

    fitting_result: FittingResult
    detected_peaks: list[float]  # Peak positions detected
    r_squared: float
    success: bool
    message: str = ""


class AutoFitter:
    """
    Automatic peak fitter for XPS spectra.

    Workflow:
    1. Estimate and subtract background
    2. Detect peaks in background-subtracted spectrum
    3. Initialize peak parameters
    4. Fit multi-peak model
    5. Return results
    """

    def __init__(self, config: AutoFitConfig | None = None):
        """
        Initialize AutoFitter.

        Args:
            config: Fitting configuration
        """
        self.config = config or AutoFitConfig()

    def fit(
        self, spectrum: Spectrum, *, use_template: bool = True
    ) -> AutoFitResult:
        """
        Automatically fit a spectrum.

        If *use_template* is True (default) and ``spectrum.element`` matches
        a registered :class:`FittingTemplate`, the template-based fitting
        path is used automatically.

        Args:
            spectrum: Spectrum to fit
            use_template: Try template-based fitting when a matching
                template exists.  Set to False to force peak-detection
                based fitting.

        Returns:
            AutoFitResult with fitting results
        """
        # Auto-template routing
        if use_template and spectrum.element:
            from toyomacro.fitting.templates import get_templates_for_element

            templates = get_templates_for_element(spectrum.element)
            if templates:
                return self.fit_with_template(spectrum, templates[0].name)

        energy = spectrum.energy
        intensity = spectrum.intensity

        # Step 1: Calculate background
        background, bg_params = self._calculate_background(energy, intensity)

        # Step 2: Detect peaks
        signal = intensity - background
        peak_positions = self._detect_peaks(energy, signal)

        if len(peak_positions) == 0:
            return AutoFitResult(
                fitting_result=self._empty_result(energy, intensity, background, bg_params),
                detected_peaks=[],
                r_squared=0.0,
                success=False,
                message="No peaks detected",
            )

        # Step 3: Build and fit model (with SO doublet support)
        model = MultiPeakModel(lineshape=self.config.lineshape)

        so_info = get_so_info(spectrum.element or "") if spectrum.element else None

        # Identify which detected peaks are already SO partners of each other
        main_positions = list(peak_positions)
        if so_info is not None:
            main_positions = self._resolve_so_pairs(
                peak_positions, so_info, energy_type=spectrum.energy_type,
            )

        e_min, e_max = float(np.min(energy)), float(np.max(energy))
        # List of (main_peak_number, partner_peak_number) for SO pairs
        # Peak numbers are 1-based as used by lmfit prefixes (p1_, p2_, ...)
        so_pairs: list[tuple[int, int]] = []

        for i, pos in enumerate(main_positions):
            # Estimate amplitude from signal at peak position
            idx = np.argmin(np.abs(energy - pos))
            amp_estimate = signal[idx] * self.config.initial_sigma * np.sqrt(2 * np.pi)

            model.add_peak(
                center=pos,
                amplitude=max(amp_estimate, 10),
                sigma=self.config.initial_sigma,
                gamma=self.config.initial_gamma,
            )
            main_peak_num = len(model.peaks)  # 1-based peak number

            # Add SO partner peak if applicable and within energy range
            if so_info is not None:
                is_be = spectrum.energy_type == "BE"
                if is_be:
                    partner_center = pos + so_info.so_split
                else:
                    partner_center = pos - so_info.so_split
                if e_min <= partner_center <= e_max:
                    partner_amp = max(amp_estimate * so_info.branch_ratio, 1)
                    model.add_peak(
                        center=partner_center,
                        amplitude=max(partner_amp, 1),
                        sigma=self.config.initial_sigma,
                        gamma=self.config.initial_gamma,
                    )
                    partner_peak_num = len(model.peaks)
                    so_pairs.append((main_peak_num, partner_peak_num))

        # Step 4: Fit — apply SO constraints before fitting
        try:
            params = model.make_params()

            if so_info is not None and so_pairs:
                params = self._apply_so_constraints(
                    params, model, so_info, so_pairs, energy,
                    energy_type=spectrum.energy_type,
                )

            result = model.fit(energy, signal, params=params)
            fitted_curve = result.best_fit + background

            # Calculate R²
            ss_res = np.sum((intensity - fitted_curve) ** 2)
            ss_tot = np.sum((intensity - np.mean(intensity)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Extract peak parameters and prune negligible components
            components = self._extract_components(result, model)

            # Stamp SO metadata on main peaks (no-template lmfit path)
            if so_info is not None and so_pairs:
                br = 1.0 / so_info.branch_ratio if so_info.branch_ratio > 0 else 0.0
                for main_num, _partner_num in so_pairs:
                    idx = main_num - 1  # peak numbers are 1-based
                    if 0 <= idx < len(components):
                        components[idx].so_split = so_info.so_split
                        components[idx].branch_ratio = br

            components = self._prune_negligible_peaks(
                components, signal, so_info,
            )

            fitting_result = FittingResult(
                components=components,
                background=bg_params,
                energy=energy,
                fitted_intensity=fitted_curve,
                background_curve=background,
                residuals=intensity - fitted_curve,
                chi2=result.chisqr,
                r_squared=r_squared,
                metadata={"lmfit_result": result},
            )

            return AutoFitResult(
                fitting_result=fitting_result,
                detected_peaks=peak_positions,
                r_squared=r_squared,
                success=True,
                message=f"Fitted {len(components)} peaks, R² = {r_squared:.4f}",
            )

        except Exception as e:
            return AutoFitResult(
                fitting_result=self._empty_result(energy, intensity, background, bg_params),
                detected_peaks=peak_positions,
                r_squared=0.0,
                success=False,
                message=f"Fitting failed: {str(e)}",
            )

    def _calculate_background(
        self,
        energy: NDArray[np.float64],
        intensity: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], BackgroundParams]:
        """Calculate background curve."""
        if self.config.background_type == "shirley":
            bg_algo = Shirley()
            background = bg_algo.calculate(
                energy, intensity, auto_range=self.config.background_auto_range
            )
        elif self.config.background_type == "tougaard":
            from toyomacro.background import Tougaard
            background = Tougaard().calculate(energy, intensity)
        elif self.config.background_type == "linear":
            from toyomacro.background import Linear
            background = Linear().calculate(energy, intensity)
        else:
            # Fallback: linear ramp
            background = np.linspace(intensity[0], intensity[-1], len(intensity))

        # MATLAB BackgroundEngine convention:
        # param_a = max(mean first 10%, mean last 10%)  — higher endpoint
        # param_b = min(mean first 10%, mean last 10%)  — lower endpoint
        # param_c = 1 if first >= last else 2            — direction
        n = len(intensity)
        m = max(round(n / 10), 1)
        data_ini = float(np.mean(intensity[:m]))
        data_fin = float(np.mean(intensity[-m:]))
        params = BackgroundParams(
            type=self.config.background_type,
            param_a=max(data_ini, data_fin),
            param_b=min(data_ini, data_fin),
            param_c=1.0 if data_ini >= data_fin else 2.0,
        )

        return background, params

    def _detect_peaks(
        self,
        energy: NDArray[np.float64],
        signal: NDArray[np.float64],
    ) -> list[float]:
        """Detect peaks in background-subtracted signal.

        Uses S/N-adaptive prominence thresholding.  When S/N is high the
        relative prominence threshold (``min_peak_prominence`` × max signal)
        dominates, allowing minor spectral features to be detected.  When
        S/N is low the noise-based threshold (``noise_prominence_sigma`` ×
        noise σ) takes over to suppress false peaks.

        The noise sigma multiplier itself is scaled by S/N so that it
        transitions smoothly:

        * S/N ≤ 10  → full ``noise_prominence_sigma`` (default 5.0)
        * S/N ≥ 50  → reduced to ``noise_prominence_sigma * 0.4``
        * Between   → linear interpolation

        For high S/N (> 20), the relative prominence threshold is the
        primary discriminator for real vs. spurious peaks.  The noise
        threshold only dominates when S/N is genuinely low.
        """
        # Smooth signal for peak detection
        if len(signal) > self.config.smooth_window:
            smoothed = savgol_filter(
                signal,
                self.config.smooth_window,
                self.config.smooth_order,
            )
        else:
            smoothed = signal

        max_intensity = np.max(smoothed)

        # --- Adaptive prominence threshold ---
        # (a) Relative threshold (original)
        relative_prom = max_intensity * self.config.min_peak_prominence

        # (b) Noise-based threshold: estimate noise as std of (signal - smoothed)
        noise_std = np.std(signal - smoothed)

        # S/N-adaptive sigma scaling
        snr = max_intensity / noise_std if noise_std > 1e-10 else 1e6
        # Linear scale: snr=10 → factor=1.0, snr=50 → factor=0.4
        snr_factor = np.clip(1.0 - 0.6 * (snr - 10) / 40, 0.4, 1.0)
        effective_prom_sigma = self.config.noise_prominence_sigma * snr_factor
        effective_height_sigma = self.config.noise_height_sigma * snr_factor

        noise_prom = effective_prom_sigma * noise_std

        prominence_threshold = max(relative_prom, noise_prom)
        height_threshold = max(
            max_intensity * self.config.min_peak_height,
            effective_height_sigma * noise_std,
        )

        peaks, properties = find_peaks(
            smoothed,
            height=height_threshold,
            distance=self.config.min_peak_distance,
            prominence=prominence_threshold,
        )

        if len(peaks) == 0:
            return []

        prominences = properties.get("prominences", np.ones(len(peaks)))

        # Secondary filter: remove peaks whose prominence is less than
        # a fraction of the maximum prominence.  This filters out noise
        # features that pass the absolute threshold but are clearly minor
        # compared to the dominant peak.
        max_prom = np.max(prominences)
        min_relative = max_prom * self.config.min_relative_prominence
        keep_mask = prominences >= min_relative
        peaks = peaks[keep_mask]
        prominences = prominences[keep_mask]

        # Limit number of peaks
        if len(peaks) > self.config.max_peaks:
            # Keep peaks with highest prominence
            sorted_idx = np.argsort(prominences)[::-1]
            peaks = peaks[sorted_idx[: self.config.max_peaks]]
            peaks = np.sort(peaks)

        # Convert indices to energy positions
        peak_positions = [float(energy[p]) for p in peaks]

        return peak_positions

    def _extract_components(
        self,
        result,
        model: MultiPeakModel,
    ) -> list[PeakComponent]:
        """Extract peak components from fit result."""
        components = []

        for i, peak in enumerate(model.peaks):
            prefix = peak.prefix
            params = result.params

            # Get fitted parameters
            center = params[f"{prefix}center"].value
            amplitude = params[f"{prefix}amplitude"].value
            sigma = params[f"{prefix}sigma"].value

            # Get gamma for Voigt / Doniach-Sunjic
            gamma_key = f"{prefix}gamma"
            gamma = params[gamma_key].value if gamma_key in params else 0

            # Get asymmetry for Doniach-Sunjic
            asym_key = f"{prefix}asymmetry"
            asymmetry = params[asym_key].value if asym_key in params else 0.0

            # Convert lmfit sigma/gamma to fwhm_g/fwhm_l per lineshape
            if peak.lineshape == "lorentzian":
                # LorentzianModel: sigma = HWHM → fwhm_l = sigma * 2
                fwhm_g = 0.0
                fwhm_l = sigma * 2.0
            elif peak.lineshape == "gaussian":
                # GaussianModel: sigma = Gaussian σ → fwhm_g = sigma * 2.3548
                fwhm_g = sigma * 2.3548
                fwhm_l = 0.0
            elif peak.lineshape == "pseudovoigt":
                # PseudoVoigtModel: sigma shared → G FWHM = σ*2.3548, L FWHM = σ*2.0
                frac_key = f"{prefix}fraction"
                fraction = params[frac_key].value if frac_key in params else 0.5
                fwhm_g = sigma * 2.3548  # actual Gaussian FWHM
                fwhm_l = sigma * 2.0     # actual Lorentzian FWHM
                asymmetry = fraction     # store fraction in asymmetry field
            else:
                # Voigt / DS: sigma = Gaussian σ, gamma = Lorentzian HWHM
                fwhm_g = sigma * 2.3548
                fwhm_l = gamma * 2 if gamma > 0 else 0

            # Compute peak height (intensity at center) from area
            # MATLAB fitpara: col 0 = peak height, col 8 = integrated area
            from toyomacro.lineshape import create_lineshape as _create_ls

            if peak.lineshape == "doniach_sunjic" and asymmetry > 0:
                ls_obj = _create_ls("doniach_sunjic")
                peak_height = ls_obj.area_to_height(
                    amplitude, fwhm_g=fwhm_g, fwhm_l=fwhm_l, asymmetry=asymmetry
                )
            elif peak.lineshape == "lorentzian":
                ls_obj = _create_ls("lorentzian")
                peak_height = ls_obj.area_to_height(amplitude, fwhm=fwhm_l)
            elif peak.lineshape == "gaussian":
                peak_height = amplitude / (sigma * np.sqrt(2 * np.pi))
            elif peak.lineshape == "pseudovoigt":
                ls_obj = _create_ls("pseudovoigt")
                peak_height = ls_obj.area_to_height(
                    amplitude, fwhm_g=fwhm_g, fwhm_l=fwhm_l, eta=fraction,
                )
            elif fwhm_g > 0 and fwhm_l > 0:
                ls_obj = _create_ls("voigt")
                peak_height = ls_obj.area_to_height(amplitude, fwhm_g=fwhm_g, fwhm_l=fwhm_l)
            else:
                peak_height = amplitude

            lineshape_type = {
                "voigt": LineshapeType.VOIGT,
                "pseudovoigt": LineshapeType.PSEUDO_VOIGT,
                "gaussian": LineshapeType.GAUSSIAN,
                "lorentzian": LineshapeType.LORENTZIAN,
                "doniach_sunjic": LineshapeType.DONIACH_SUNJIC,
            }.get(peak.lineshape, LineshapeType.VOIGT)

            components.append(
                PeakComponent(
                    area=peak_height,       # col 0: peak height
                    center=center,
                    fwhm_g=fwhm_g,
                    fwhm_l=fwhm_l,
                    asymmetry=asymmetry,
                    lineshape=lineshape_type,
                    height=amplitude,       # col 8: integrated area
                )
            )

        return components

    @staticmethod
    def _prune_negligible_peaks(
        components: list[PeakComponent],
        signal: NDArray[np.float64],
        so_info: SOInfo | None,
        area_fraction: float = 0.01,
    ) -> list[PeakComponent]:
        """Remove peaks whose fitted area is negligible.

        A peak is considered negligible if its area is less than
        ``area_fraction`` of the total fitted area across all components.

        For SO-coupled spectra, peaks are identified as pairs by checking
        if consecutive peaks have center spacing close to the SO split.
        Pairs are pruned together to maintain physical consistency.

        Args:
            components: Fitted peak components.
            signal: Background-subtracted signal (for reference).
            so_info: Spin-orbit info, or None.
            area_fraction: Minimum area as a fraction of total.

        Returns:
            Pruned list of components.
        """
        if not components:
            return components

        total_area = sum(abs(c.area) for c in components)
        if total_area < 1e-10:
            return components

        threshold = total_area * area_fraction

        if so_info is not None:
            # Group into SO pairs: check consecutive peaks for SO split
            pruned = []
            i = 0
            while i < len(components):
                main = components[i]
                # Check if next peak is an SO partner
                if i + 1 < len(components):
                    partner = components[i + 1]
                    spacing = abs(main.center - partner.center)
                    if abs(spacing - so_info.so_split) < 0.5:
                        # It's an SO pair
                        pair_area = abs(main.area) + abs(partner.area)
                        if pair_area >= threshold:
                            pruned.append(main)
                            pruned.append(partner)
                        i += 2
                        continue
                # Singlet peak
                if abs(main.area) >= threshold:
                    pruned.append(main)
                i += 1
            return pruned
        else:
            return [c for c in components if abs(c.area) >= threshold]

    @staticmethod
    def _resolve_so_pairs(
        peak_positions: list[float],
        so_info: SOInfo,
        tolerance: float = 0.3,
        energy_type: str = "KE",
    ) -> list[float]:
        """Identify detected peaks that are SO partners of each other.

        If two detected peaks are separated by approximately ``so_info.so_split``
        (within ``tolerance`` eV), they are treated as a single doublet.
        Only the main peak (j+ component) is kept; the partner will be
        generated automatically.

        On a KE axis the j+ component is at higher energy; on a BE axis it
        is at lower energy.

        Args:
            peak_positions: Detected peak positions.
            so_info: Spin-orbit info.
            tolerance: Maximum deviation from expected SO split (eV).
            energy_type: ``"BE"`` or ``"KE"``.

        Returns:
            List of main peak positions (j+ components only).
        """
        if len(peak_positions) <= 1:
            return list(peak_positions)

        is_be = energy_type == "BE"
        used = set()
        main_peaks = []
        # For KE: highest first (j+ is higher KE).
        # For BE: lowest first (j+ is lower BE).
        positions = sorted(peak_positions, reverse=not is_be)

        for i, pos in enumerate(positions):
            if i in used:
                continue
            # Look for a partner: KE → pos - split, BE → pos + split
            if is_be:
                expected_partner = pos + so_info.so_split
            else:
                expected_partner = pos - so_info.so_split
            best_j = None
            best_dist = tolerance
            for j, other in enumerate(positions):
                if j in used or j == i:
                    continue
                dist = abs(other - expected_partner)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            main_peaks.append(pos)
            if best_j is not None:
                used.add(best_j)  # mark partner as consumed

        return sorted(main_peaks)

    def _apply_so_constraints(
        self,
        params,
        model: MultiPeakModel,
        so_info: SOInfo,
        so_pairs: list[tuple[int, int]],
        energy: NDArray[np.float64],
        energy_type: str = "KE",
    ):
        """Apply spin-orbit constraints to lmfit Parameters.

        For each SO pair (main j+ component and its j- partner), constrain:
          - center offset: BE → j- = j+ + so_split, KE → j- = j+ - so_split
          - area ratio: j- amplitude = j+ amplitude × branch_ratio
          - width: j- sigma = j+ sigma, j- gamma = j+ gamma

        Args:
            params: lmfit Parameters object.
            model: MultiPeakModel.
            so_info: Spin-orbit coupling info.
            so_pairs: List of (main_peak_num, partner_peak_num) tuples,
                where peak numbers are 1-based.
            energy: Energy axis.
            energy_type: "BE" or "KE".
        """
        energy_range = abs(energy[-1] - energy[0])
        so_sign = "+" if energy_type == "BE" else "-"

        for main_num, partner_num in so_pairs:
            main_prefix = f"p{main_num}_"
            partner_prefix = f"p{partner_num}_"

            # Check both prefixes exist
            if f"{partner_prefix}center" not in params:
                continue

            # Center constraint: BE → partner = main + so_split
            #                    KE → partner = main - so_split
            params[f"{partner_prefix}center"].set(
                expr=f"{main_prefix}center {so_sign} {so_info.so_split}",
            )

            # Amplitude constraint: partner = main × branch_ratio
            params[f"{partner_prefix}amplitude"].set(
                expr=f"{main_prefix}amplitude * {so_info.branch_ratio}",
            )

            # Width constraint: same sigma and gamma
            params[f"{partner_prefix}sigma"].set(
                expr=f"{main_prefix}sigma",
            )
            gamma_key = f"{partner_prefix}gamma"
            if gamma_key in params:
                params[gamma_key].set(
                    expr=f"{main_prefix}gamma",
                )

            # Widen center search range for main peak
            main_center = params[f"{main_prefix}center"].value
            params[f"{main_prefix}center"].set(
                min=main_center - energy_range * 0.15,
                max=main_center + energy_range * 0.15,
            )

        return params

    def fit_with_initial(
        self,
        spectrum: Spectrum,
        initial_params: list[dict],
    ) -> AutoFitResult:
        """
        Fit spectrum using given initial parameters.

        Args:
            spectrum: Spectrum to fit
            initial_params: List of dicts with keys: center, area, fwhm_g, fwhm_l

        Returns:
            AutoFitResult with fitting results
        """
        energy = spectrum.energy
        intensity = spectrum.intensity

        # Step 1: Calculate background
        background, bg_params = self._calculate_background(energy, intensity)
        signal = intensity - background

        if len(initial_params) == 0:
            return AutoFitResult(
                fitting_result=self._empty_result(energy, intensity, background, bg_params),
                detected_peaks=[],
                r_squared=0.0,
                success=False,
                message="No initial parameters provided",
            )

        # Step 2: Build model with initial parameters
        model = MultiPeakModel(lineshape=self.config.lineshape)

        peak_positions = []
        for params in initial_params:
            pos = params['center']
            peak_positions.append(pos)

            fwhm_g = params.get('fwhm_g', 0.5)
            fwhm_l = params.get('fwhm_l', 0.3)

            # Per-peak lineshape override
            ls_name = params.get('lineshape', None) or self.config.lineshape

            # Map fwhm_g/fwhm_l to lmfit sigma/gamma per lineshape
            if ls_name == "lorentzian":
                # LorentzianModel: sigma = HWHM = fwhm_l / 2
                sigma = max(fwhm_l / 2.0, 0.05)
                gamma = None
            elif ls_name == "gaussian":
                # GaussianModel: sigma = fwhm_g / 2.3548
                sigma = max(fwhm_g / 2.3548, 0.05)
                gamma = None
            elif ls_name == "pseudovoigt":
                # PseudoVoigtModel: shared sigma, G FWHM = σ*2.3548, L FWHM = σ*2.0
                # fwhm_g IS the actual Gaussian FWHM → direct recovery
                sigma = max(fwhm_g / 2.3548, 0.05) if fwhm_g > 0 else max(fwhm_l / 2.0, 0.05)
                gamma = None  # fraction handled in PeakModel
            else:
                # Voigt / DS: sigma = fwhm_g / 2.3548, gamma = fwhm_l / 2
                sigma = max(fwhm_g / 2.3548, 0.1)
                gamma = max(fwhm_l / 2.0, 0.05)

            model.add_peak(
                center=pos,
                amplitude=params.get('area', 100),
                sigma=sigma,
                gamma=gamma,
                asymmetry=params.get('asymmetry', None),
                lineshape=ls_name,
            )

        # Step 3: Fit
        try:
            result = model.fit(energy, signal)
            fitted_curve = result.best_fit + background

            # Calculate R²
            ss_res = np.sum((intensity - fitted_curve) ** 2)
            ss_tot = np.sum((intensity - np.mean(intensity)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Extract peak parameters
            components = self._extract_components(result, model)

            fitting_result = FittingResult(
                components=components,
                background=bg_params,
                energy=energy,
                fitted_intensity=fitted_curve,
                background_curve=background,
                residuals=intensity - fitted_curve,
                chi2=result.chisqr,
                r_squared=r_squared,
                metadata={"lmfit_result": result},
            )

            return AutoFitResult(
                fitting_result=fitting_result,
                detected_peaks=peak_positions,
                r_squared=r_squared,
                success=True,
                message=f"Re-fitted {len(peak_positions)} peaks, R² = {r_squared:.4f}",
            )

        except Exception as e:
            return AutoFitResult(
                fitting_result=self._empty_result(energy, intensity, background, bg_params),
                detected_peaks=peak_positions,
                r_squared=0.0,
                success=False,
                message=f"Re-fitting failed: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Template-based fitting
    # ------------------------------------------------------------------

    def fit_with_template(
        self,
        spectrum: Spectrum,
        template_name: str,
        reference_center: float | None = None,
    ) -> AutoFitResult:
        """Fit a spectrum using a predefined template.

        Templates encode domain knowledge for well-known chemical systems
        (e.g. Si 2p oxide interface, Ta 4f oxide, C 1s organic).  They
        provide initial peak positions via chemical shifts, SO doublet
        constraints, and width ordering.

        Args:
            spectrum: Spectrum to fit.
            template_name: Name of a registered template
                (e.g. ``"Si2p_oxide"``).
            reference_center: Position of the reference peak on the
                spectrum's energy axis.  If ``None``, the reference
                position is auto-detected from the data.

        Returns:
            AutoFitResult with fitting results.
        """
        template = get_template(template_name)
        if template is None:
            # Unknown template — fall back to normal auto-fit
            return self.fit(spectrum)

        energy = spectrum.energy
        intensity = spectrum.intensity
        is_be = spectrum.energy_type == "BE"

        # Step 1: Background
        background, bg_params = self._calculate_background(energy, intensity)
        signal = intensity - background

        # Step 2: Find reference peak position
        if reference_center is not None:
            ref_position = reference_center
        else:
            ref_position = self._find_reference_position(
                energy, signal, template, spectrum.energy_type,
                spectrum.metadata,
            )

        # Step 3: Convert template ΔBE to spectrum axis positions
        def _to_axis(delta_be: float) -> float:
            if is_be:
                return ref_position + delta_be
            return ref_position - delta_be

        # Step 4: Build model
        so_info = get_so_info(template.element) if template.apply_so else None

        _LINESHAPE_MAP = {
            LineshapeType.VOIGT: "voigt",
            LineshapeType.PSEUDO_VOIGT: "pseudovoigt",
            LineshapeType.GAUSSIAN: "gaussian",
            LineshapeType.LORENTZIAN: "lorentzian",
            LineshapeType.DONIACH_SUNJIC: "doniach_sunjic",
        }

        model = MultiPeakModel(lineshape=self.config.lineshape)
        so_pairs: list[tuple[int, int]] = []
        # Metadata per template peak for width-ordering constraints
        peak_metadata: list[dict] = []

        e_min, e_max = float(np.min(energy)), float(np.max(energy))

        for tpeak in template.peaks:
            pos = _to_axis(tpeak.delta_be)

            # Skip peaks outside spectrum range
            if pos < e_min or pos > e_max:
                continue

            # Estimate amplitude from signal at peak position
            idx = int(np.argmin(np.abs(energy - pos)))
            signal_at_peak = max(float(signal[idx]), 1.0)
            sigma_init = tpeak.fwhm_g / 2.3548
            amp_estimate = signal_at_peak * sigma_init * np.sqrt(2 * np.pi)
            if tpeak.relative_area is not None:
                amp_estimate *= tpeak.relative_area

            gamma_init = tpeak.fwhm_l / 2.0
            lineshape_str = _LINESHAPE_MAP.get(tpeak.lineshape, "voigt")

            model.add_peak(
                center=pos,
                amplitude=max(amp_estimate, 10),
                sigma=sigma_init,
                gamma=gamma_init,
                asymmetry=tpeak.asymmetry if tpeak.asymmetry > 0 else None,
                lineshape=lineshape_str,
            )
            main_num = len(model.peaks)
            peak_metadata.append({
                "template_peak": tpeak,
                "peak_num": main_num,
            })

            # Add SO partner if applicable
            if so_info is not None:
                if is_be:
                    partner_center = pos + so_info.so_split
                else:
                    partner_center = pos - so_info.so_split

                if e_min <= partner_center <= e_max:
                    model.add_peak(
                        center=partner_center,
                        amplitude=max(amp_estimate * so_info.branch_ratio, 1),
                        sigma=sigma_init,
                        gamma=gamma_init,
                        asymmetry=(
                            tpeak.asymmetry if tpeak.asymmetry > 0 else None
                        ),
                        lineshape=lineshape_str,
                    )
                    partner_num = len(model.peaks)
                    so_pairs.append((main_num, partner_num))

        if model.n_peaks == 0:
            return AutoFitResult(
                fitting_result=self._empty_result(
                    energy, intensity, background, bg_params,
                ),
                detected_peaks=[],
                r_squared=0.0,
                success=False,
                message="No template peaks fall within spectrum range",
            )

        # Step 5–6: Fit
        try:
            # Fast path: all-Voigt + SO + width ordering → direct scipy
            all_voigt = all(
                (tp.lineshape in (LineshapeType.VOIGT, None) and tp.asymmetry <= 0)
                for tp in template.peaks
            )
            if all_voigt and so_info is not None and template.width_ordering:
                return self._fast_template_fit(
                    energy, intensity, signal, background, bg_params,
                    template, peak_metadata, so_info, is_be,
                    ref_position, _to_axis,
                )

            # Slow path: general lmfit (needed for DS, mixed lineshapes, etc.)
            params = model.make_params()

            if so_info is not None and so_pairs:
                params = self._apply_so_constraints(
                    params, model, so_info, so_pairs, energy,
                    energy_type=spectrum.energy_type,
                )

            if template.width_ordering:
                params = self._apply_width_ordering(
                    params, template, peak_metadata,
                )

            params = self._apply_shift_bounds(
                params, template, peak_metadata, ref_position, is_be,
            )

            result = model.fit(energy, signal, params=params)
            fitted_curve = result.best_fit + background

            ss_res = np.sum((intensity - fitted_curve) ** 2)
            ss_tot = np.sum((intensity - np.mean(intensity)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            components = self._extract_components(result, model)

            # Stamp SO metadata on main peaks (lmfit path)
            if so_info is not None and so_pairs:
                br = 1.0 / so_info.branch_ratio if so_info.branch_ratio > 0 else 0.0
                for main_num, _partner_num in so_pairs:
                    idx = main_num - 1  # peak numbers are 1-based
                    if 0 <= idx < len(components):
                        components[idx].so_split = so_info.so_split
                        components[idx].branch_ratio = br

            peak_positions = [
                _to_axis(tp.delta_be) for tp in template.peaks
            ]

            fitting_result = FittingResult(
                components=components,
                background=bg_params,
                energy=energy,
                fitted_intensity=fitted_curve,
                background_curve=background,
                residuals=intensity - fitted_curve,
                chi2=result.chisqr,
                r_squared=r_squared,
                metadata={
                    "lmfit_result": result,
                    "template": template.name,
                },
            )

            return AutoFitResult(
                fitting_result=fitting_result,
                detected_peaks=peak_positions,
                r_squared=r_squared,
                success=True,
                message=(
                    f"Template '{template.name}': "
                    f"{len(components)} components, "
                    f"R² = {r_squared:.4f}"
                ),
            )

        except Exception as e:
            return AutoFitResult(
                fitting_result=self._empty_result(
                    energy, intensity, background, bg_params,
                ),
                detected_peaks=[],
                r_squared=0.0,
                success=False,
                message=f"Template fitting failed: {e!s}",
            )

    # ------------------------------------------------------------------
    # Hierarchical grouped solver
    # ------------------------------------------------------------------

    def fit_grouped(
        self,
        spectrum: Spectrum,
        template_name: str,
        reference_center: float | None = None,
        threshold: float = 4.0,
    ) -> AutoFitResult:
        """Fit a spectrum using hierarchical solve_grouped pipeline.

        Uses ``template_to_configs()`` to convert a FittingTemplate
        into ComponentConfig list, then runs the auto_group +
        solve_grouped hierarchical solver.

        This path is faster than lmfit for high-throughput scenarios
        (batch fitting, BrowserTab live-fit) and handles SO doublets
        natively via ComponentConfig.

        Falls back to ``fit_with_template()`` if the template contains
        Doniach-Sunjic peaks (not supported by solve_grouped).

        Args:
            spectrum: Spectrum to fit.
            template_name: Name of a registered template.
            reference_center: Position of the reference peak.  If None,
                auto-detected from the data.
            threshold: Separation threshold for auto_group (σ units).

        Returns:
            AutoFitResult with fitting results.
        """
        from toyomacro.fitting.template_bridge import template_to_configs

        template = get_template(template_name)
        if template is None:
            return self.fit(spectrum)

        energy = spectrum.energy
        intensity = spectrum.intensity
        is_be = spectrum.energy_type == "BE"

        # Step 1: Background
        background, bg_params = self._calculate_background(energy, intensity)
        signal = intensity - background

        # Step 2: Find reference position
        if reference_center is not None:
            ref_position = reference_center
        else:
            ref_position = self._find_reference_position(
                energy, signal, template, spectrum.energy_type,
                spectrum.metadata,
            )

        # Step 3: Template → ComponentConfig
        configs = template_to_configs(
            template, ref_position, energy_type=spectrum.energy_type,
        )

        # Step 4: DS check — fallback to lmfit if DS peaks present
        has_ds = any(
            p.lineshape == LineshapeType.DONIACH_SUNJIC
            for p in template.peaks
        )
        if has_ds:
            return self.fit_with_template(
                spectrum, template_name, reference_center=reference_center,
            )

        # Step 5: solve_grouped
        try:
            from toyomacro.voigtfit.auto_grouping import solve_grouped

            # solve_grouped expects (N, n_E) — single spectrum → (1, n_E)
            spectra_2d = np.maximum(signal, 0.0)[np.newaxis, :].astype(
                np.float32
            )

            grouped_result = solve_grouped(
                spectra_2d,
                configs,
                energy,
                threshold=threshold,
            )

            # Step 6: GroupedResult → AutoFitResult
            return self._grouped_to_autofit(
                grouped_result, configs, template,
                energy, intensity, signal, background, bg_params, is_be,
            )

        except Exception as e:
            return AutoFitResult(
                fitting_result=self._empty_result(
                    energy, intensity, background, bg_params,
                ),
                detected_peaks=[],
                r_squared=0.0,
                success=False,
                message=f"Grouped fitting failed: {e!s}",
            )

    def _grouped_to_autofit(
        self,
        grouped: GroupedResult,
        configs: list,
        template: FittingTemplate,
        energy: NDArray[np.float64],
        intensity: NDArray[np.float64],
        signal: NDArray[np.float64],
        background: NDArray[np.float64],
        bg_params: BackgroundParams,
        is_be: bool,
    ) -> AutoFitResult:
        """Convert GroupedResult (single spectrum) to AutoFitResult.

        Expands SO doublets: each ComponentConfig with so_split>0
        becomes two PeakComponent (main + partner).

        Args:
            grouped: Result from solve_grouped (N=1).
            configs: List of ComponentConfig used for fitting.
            template: Original FittingTemplate (for metadata).
            energy: Energy axis.
            intensity: Original spectrum intensity.
            signal: Background-subtracted signal.
            background: Background curve.
            bg_params: Background parameters.
            is_be: True if binding energy axis.
        """
        from toyomacro.lineshape import Voigt

        voigt = Voigt()
        components: list[PeakComponent] = []
        fitted_model = np.zeros_like(energy, dtype=np.float64)

        for k, config in enumerate(configs):
            amp_k = float(grouped.amplitudes[0, k])
            dE_k = float(grouped.delta_E[0, k])
            ds_k = float(grouped.delta_sigma[0, k])

            center_k = config.center + dE_k
            sigma_k = max(config.sigma + ds_k, 0.01)
            gamma_k = config.gamma

            fwhm_g_k = sigma_k * 2.3548
            fwhm_l_k = gamma_k * 2.0
            # Peak height at center (proper Voigt, not Gaussian-only)
            peak_height_k = voigt.area_to_height(amp_k, fwhm_g=fwhm_g_k, fwhm_l=fwhm_l_k) if sigma_k > 0 else amp_k

            # Main peak
            main_curve = voigt.evaluate(
                energy, center_k, amp_k,
                fwhm_g=fwhm_g_k, fwhm_l=fwhm_l_k,
            )
            fitted_model += main_curve

            components.append(
                PeakComponent(
                    area=peak_height_k,       # col 0: peak height
                    center=center_k,
                    fwhm_g=fwhm_g_k,
                    fwhm_l=fwhm_l_k,
                    asymmetry=0.0,
                    lineshape=LineshapeType.VOIGT,
                    height=amp_k,             # col 8: integrated area
                    so_split=config.so_split,
                    branch_ratio=(
                        1.0 / config.branch_ratio
                        if config.branch_ratio > 0
                        else 0.0
                    ),
                )
            )

            # SO partner
            if config.is_doublet:
                partner_center = center_k + config.so_split
                partner_amp = amp_k / config.branch_ratio
                partner_peak_height = voigt.area_to_height(partner_amp, fwhm_g=fwhm_g_k, fwhm_l=fwhm_l_k) if sigma_k > 0 else partner_amp
                partner_curve = voigt.evaluate(
                    energy, partner_center, partner_amp,
                    fwhm_g=fwhm_g_k, fwhm_l=fwhm_l_k,
                )
                fitted_model += partner_curve

                components.append(
                    PeakComponent(
                        area=partner_peak_height,   # col 0: peak height
                        center=partner_center,
                        fwhm_g=fwhm_g_k,
                        fwhm_l=fwhm_l_k,
                        asymmetry=0.0,
                        lineshape=LineshapeType.VOIGT,
                        height=partner_amp,         # col 8: integrated area
                    )
                )

        fitted_curve = fitted_model + background

        ss_res = float(np.sum((intensity - fitted_curve) ** 2))
        ss_tot = float(np.sum((intensity - np.mean(intensity)) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        chi2 = float(np.sum((intensity - fitted_curve) ** 2))

        peak_positions = [c.center for c in configs]

        fitting_result = FittingResult(
            components=components,
            background=bg_params,
            energy=energy,
            fitted_intensity=fitted_curve,
            background_curve=background,
            residuals=intensity - fitted_curve,
            chi2=chi2,
            r_squared=r_squared,
            metadata={
                "template": template.name,
                "solver": "hierarchical",
                "grouping": str(grouped.grouping.groups),
            },
        )

        return AutoFitResult(
            fitting_result=fitting_result,
            detected_peaks=peak_positions,
            r_squared=r_squared,
            success=True,
            message=(
                f"Template '{template.name}' (hierarchical): "
                f"{len(components)} components, "
                f"R² = {r_squared:.4f}"
            ),
        )

    # ------------------------------------------------------------------
    # Fast scipy-direct template fitting (bypasses lmfit asteval overhead)
    # ------------------------------------------------------------------

    def _fast_template_fit(
        self,
        energy: NDArray[np.float64],
        intensity: NDArray[np.float64],
        signal: NDArray[np.float64],
        background: NDArray[np.float64],
        bg_params: BackgroundParams,
        template: FittingTemplate,
        peak_metadata: list[dict],
        so_info: SOInfo,
        is_be: bool,
        ref_position: float,
        _to_axis,
    ) -> AutoFitResult:
        """Fast Voigt-only template fit via scipy.optimize.least_squares.

        Encodes SO doublet and width-ordering constraints directly in
        Python (no asteval), giving ~10x speedup over the lmfit path
        for 10-component models.
        """
        from scipy.optimize import least_squares

        from toyomacro.lineshape import Voigt

        voigt = Voigt()
        so_split = so_info.so_split
        so_br = so_info.branch_ratio
        so_sign = +1.0 if is_be else -1.0  # partner offset direction

        # Build per-peak data: initial values, bounds, width fractions
        n_states = len(peak_metadata)
        # Free params per state: amplitude, center
        # Plus 4 global: ref_sigma, ref_gamma, max_sigma, max_gamma
        # x = [amp0, ctr0, amp1, ctr1, ..., ampN-1, ctrN-1,
        #      ref_sigma, ref_gamma, max_sigma, max_gamma]

        x0 = []
        lo = []
        hi = []
        fracs = []  # width_fraction per state

        for meta in peak_metadata:
            tpeak: TemplatePeak = meta["template_peak"]
            pos = _to_axis(tpeak.delta_be)
            idx = int(np.argmin(np.abs(energy - pos)))
            sig_at = max(float(signal[idx]), 1.0)
            sigma_init = tpeak.fwhm_g / 2.3548
            amp_est = sig_at * sigma_init * np.sqrt(2 * np.pi)
            if tpeak.relative_area is not None:
                amp_est *= tpeak.relative_area

            x0.extend([max(amp_est, 10), pos])
            lo.extend([0, pos - 0.75 if tpeak.is_reference else pos - 0.5])
            hi.extend([1e8, pos + 0.75 if tpeak.is_reference else pos + 0.5])
            fracs.append(tpeak.width_fraction if tpeak.width_fraction is not None else 0.0)

        # Global width params: ref (frac=0) and max (frac=1)
        ref_peak = template.reference_peak
        max_peak = max(template.peaks, key=lambda p: p.width_fraction or 0.0)
        x0.extend([
            ref_peak.fwhm_g / 2.3548, ref_peak.fwhm_l / 2.0,
            max_peak.fwhm_g / 2.3548, max_peak.fwhm_l / 2.0,
        ])
        lo.extend([0.02, 0.02, 0.02, 0.02])
        hi.extend([1.5, 1.0, 3.0, 1.0])

        x0 = np.asarray(x0, dtype=np.float64)
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)

        # Clip initial values to bounds
        x0 = np.clip(x0, lo, hi)

        fracs_arr = np.asarray(fracs, dtype=np.float64)

        def _residual(x):
            model_y = np.zeros_like(energy)
            ref_sig = x[-4]
            ref_gam = x[-3]
            max_sig = x[-2]
            max_gam = x[-1]

            for i in range(n_states):
                amp = x[2 * i]
                ctr = x[2 * i + 1]
                f = fracs_arr[i]
                sig = ref_sig + f * (max_sig - ref_sig)
                gam = ref_gam + f * (max_gam - ref_gam)
                fwhm_g = sig * 2.3548
                fwhm_l = gam * 2.0

                # Main peak
                model_y += voigt.evaluate(
                    energy, ctr, amp, fwhm_g=fwhm_g, fwhm_l=fwhm_l,
                )
                # SO partner
                model_y += voigt.evaluate(
                    energy, ctr + so_sign * so_split,
                    amp * so_br, fwhm_g=fwhm_g, fwhm_l=fwhm_l,
                )
            return model_y - signal

        sol = least_squares(_residual, x0, bounds=(lo, hi), method="trf")

        # Build model curve & R²
        best_fit = signal + sol.fun  # model = signal + residual
        fitted_curve = best_fit + background
        ss_res = float(np.sum((intensity - fitted_curve) ** 2))
        ss_tot = float(np.sum((intensity - np.mean(intensity)) ** 2))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        chi2 = float(np.sum(sol.fun ** 2))

        # Build PeakComponent list (main + SO partner for each state)
        components: list[PeakComponent] = []
        xopt = sol.x
        ref_sig = xopt[-4]
        ref_gam = xopt[-3]
        max_sig = xopt[-2]
        max_gam = xopt[-1]

        from toyomacro.lineshape import Voigt
        _voigt = Voigt()

        for i in range(n_states):
            amp = float(xopt[2 * i])
            ctr = float(xopt[2 * i + 1])
            f = fracs_arr[i]
            sig = ref_sig + f * (max_sig - ref_sig)
            gam = ref_gam + f * (max_gam - ref_gam)
            fwhm_g = float(sig * 2.3548)
            fwhm_l = float(gam * 2.0)
            # Peak height at center (proper Voigt formula)
            peak_height = _voigt.area_to_height(amp, fwhm_g=fwhm_g, fwhm_l=fwhm_l) if sig > 0 else amp

            partner_amp = amp * so_br
            partner_peak_height = _voigt.area_to_height(partner_amp, fwhm_g=fwhm_g, fwhm_l=fwhm_l) if sig > 0 else partner_amp

            # Main peak — carries SO metadata for doublet detection
            components.append(PeakComponent(
                area=peak_height, center=ctr,       # col 0: peak height
                fwhm_g=fwhm_g, fwhm_l=fwhm_l,
                asymmetry=0.0, lineshape=LineshapeType.VOIGT,
                height=amp,                          # col 8: integrated area
                so_split=so_split,
                branch_ratio=1.0 / so_br if so_br > 0 else 0.0,
            ))
            # SO partner
            components.append(PeakComponent(
                area=partner_peak_height,            # col 0: peak height
                center=ctr + so_sign * so_split,
                fwhm_g=fwhm_g, fwhm_l=fwhm_l,
                asymmetry=0.0, lineshape=LineshapeType.VOIGT,
                height=partner_amp,                  # col 8: integrated area
            ))

        peak_positions = [_to_axis(tp.delta_be) for tp in template.peaks]

        fitting_result = FittingResult(
            components=components,
            background=bg_params,
            energy=energy,
            fitted_intensity=fitted_curve,
            background_curve=background,
            residuals=intensity - fitted_curve,
            chi2=chi2,
            r_squared=r_squared,
            metadata={"template": template.name},
        )

        return AutoFitResult(
            fitting_result=fitting_result,
            detected_peaks=peak_positions,
            r_squared=r_squared,
            success=True,
            message=(
                f"Template '{template.name}': "
                f"{len(components)} components, "
                f"R² = {r_squared:.4f}"
            ),
        )

    def _find_reference_position(
        self,
        energy: NDArray[np.float64],
        signal: NDArray[np.float64],
        template: FittingTemplate,
        energy_type: str,
        metadata: dict,
    ) -> float:
        """Auto-detect the reference peak position in the spectrum.

        Strategy:

        1. Run peak detection on the background-subtracted signal.
        2. If ``template.reference_be`` is set, convert it to the
           spectrum's energy axis and find the closest detected peak.
        3. Fallback: use the strongest detected peak.

        Args:
            energy: Energy axis.
            signal: Background-subtracted signal.
            template: Fitting template.
            energy_type: ``"BE"`` or ``"KE"``.
            metadata: Spectrum metadata (may contain ``photon_energy``).

        Returns:
            Reference peak position on the spectrum's energy axis.
        """
        detected = self._detect_peaks(energy, signal)

        if template.reference_be is not None:
            if energy_type == "BE":
                target = template.reference_be
            else:
                hv = metadata.get("photon_energy", 1486.7)
                target = hv - template.reference_be
        else:
            target = None

        if not detected:
            if target is not None:
                return target
            return float(np.mean(energy))

        if target is not None:
            distances = [abs(p - target) for p in detected]
            return detected[int(np.argmin(distances))]

        # Fallback: strongest peak
        peak_intensities = [
            float(signal[np.argmin(np.abs(energy - p))]) for p in detected
        ]
        return detected[int(np.argmax(peak_intensities))]

    @staticmethod
    def _apply_width_ordering(
        params,
        template: FittingTemplate,
        peak_metadata: list[dict],
    ):
        """Apply monotonic width ordering via interpolation constraints.

        For peaks with ``width_fraction`` defined in the template:

        * The reference peak (fraction = 0.0) has free sigma.
        * The broadest peak  (fraction = 1.0) has free sigma.
        * Intermediate peaks get:
          ``sigma_i = sigma_ref + frac_i × (sigma_max − sigma_ref)``

        This guarantees monotonically increasing widths by construction.

        The same interpolation is applied to gamma (Lorentzian width).
        """
        ordered = [
            (pm["peak_num"], pm["template_peak"])
            for pm in peak_metadata
            if pm["template_peak"].width_fraction is not None
        ]

        if len(ordered) < 2:
            return params

        ref_num = None
        max_num = None
        for num, tpeak in ordered:
            if tpeak.width_fraction == 0.0:
                ref_num = num
            if tpeak.width_fraction == 1.0:
                max_num = num

        if ref_num is None or max_num is None:
            return params

        ref_prefix = f"p{ref_num}_"
        max_prefix = f"p{max_num}_"

        # Ensure sigma bounds allow room for ordering
        params[f"{ref_prefix}sigma"].set(min=0.05, max=1.5)
        params[f"{max_prefix}sigma"].set(min=0.05, max=3.0)

        # Constrain intermediate peaks
        for num, tpeak in ordered:
            frac = tpeak.width_fraction
            if frac == 0.0 or frac == 1.0:
                continue

            prefix = f"p{num}_"
            params[f"{prefix}sigma"].set(
                expr=(
                    f"{ref_prefix}sigma + {frac} * "
                    f"({max_prefix}sigma - {ref_prefix}sigma)"
                ),
            )

            gamma_key = f"{prefix}gamma"
            if gamma_key in params:
                params[gamma_key].set(
                    expr=(
                        f"{ref_prefix}gamma + {frac} * "
                        f"({max_prefix}gamma - {ref_prefix}gamma)"
                    ),
                )

        return params

    @staticmethod
    def _apply_shift_bounds(
        params,
        template: FittingTemplate,
        peak_metadata: list[dict],
        ref_position: float,
        is_be: bool,
        tolerance: float = 0.5,
    ):
        """Set center bounds based on expected chemical shifts.

        Each peak's center is bounded to ``expected ± tolerance``.
        The reference peak gets wider tolerance (×1.5) since its
        position was estimated from peak detection.

        Args:
            params: lmfit Parameters.
            template: Fitting template.
            peak_metadata: Per-peak metadata from model building.
            ref_position: Reference peak position on spectrum axis.
            is_be: Whether the spectrum is on a BE axis.
            tolerance: Allowed drift from template position (eV).
        """
        for pm in peak_metadata:
            tpeak: TemplatePeak = pm["template_peak"]
            num: int = pm["peak_num"]
            prefix = f"p{num}_"

            if is_be:
                expected = ref_position + tpeak.delta_be
            else:
                expected = ref_position - tpeak.delta_be

            tol = tolerance * 1.5 if tpeak.is_reference else tolerance
            params[f"{prefix}center"].set(
                min=expected - tol, max=expected + tol,
            )

        return params

    def _empty_result(
        self,
        energy: NDArray[np.float64],
        intensity: NDArray[np.float64],
        background: NDArray[np.float64],
        bg_params: BackgroundParams,
    ) -> FittingResult:
        """Create empty fitting result."""
        return FittingResult(
            components=[],
            background=bg_params,
            energy=energy,
            fitted_intensity=background,
            background_curve=background,
            residuals=intensity - background,
            chi2=0.0,
            r_squared=0.0,
        )
