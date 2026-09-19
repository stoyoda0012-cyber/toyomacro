"""
Fast Voigt Fitting Engine - voigtfit integration
================================================

Integrates the voigtfit high-performance engine via HybridPipeline:
- preview: amp-only Stage1 (400M spec/s, amplitude only)
- fast: 4-step residual projection (1.5M/s, δE+δσ recovery)
- precise: adaptive dict+4-step hybrid (0.4M/s, best quality)
- gamma_calibrated: 2-stage Dict3D→Dict2D with γ correction (2+ M/s cached)

Usage:
    from toyomacro.fitting import FastVoigtFitter, FastFitConfig

    fitter = FastVoigtFitter(FastFitConfig(mode="fast"))
    result = fitter.fit_batch_rowmajor(spectra, energy, centers, sigmas, gamma)

    # γ-calibrated mode: corrects systematic γ mismatch (+19 dB δσ improvement)
    fitter = FastVoigtFitter(FastFitConfig(mode="gamma_calibrated"))
    result = fitter.fit_batch_rowmajor(spectra, energy, centers, sigmas, gamma)
    print(result.gamma)  # corrected γ value
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    from toyomacro.voigtfit.pipeline import FitResult as VoigtFitResult
    from toyomacro.voigtfit.pipeline import HybridPipeline
    from toyomacro.voigtfit.weight_cache import WeightMatrixCache
    VOIGTFIT_AVAILABLE = True
except ImportError:
    VOIGTFIT_AVAILABLE = False
    HybridPipeline = None
    VoigtFitResult = None
    WeightMatrixCache = None


@dataclass
class FastFitConfig:
    """Configuration for fast Voigt fitting."""

    # Solver selection
    mode: Literal["preview", "fast", "precise", "gamma_calibrated", "balanced"] = "fast"
    # preview: amp-only Stage1 (400M spec/s)
    # fast: 4-step residual projection (1.5M/s, δE+δσ recovery)
    # precise: adaptive dict+4-step hybrid (0.4M/s, best quality)
    # gamma_calibrated: 2-stage Dict3D→Dict2D with γ correction (2+ M/s cached)
    # balanced: dict2d + parabola sub-grid (5M/s, best δσ without γ calibration)

    # MLX (Apple Silicon GPU) settings
    use_mlx: bool = True

    # Anomaly threshold (MAD units)
    chi2_threshold: float = 3.0

    # Batch size for processing
    batch_size: int = 100_000

    # Memory management
    memory_mode: Literal["auto", "high", "low"] = "auto"
    # auto: select based on available memory (>=32GB → high, <32GB → low)
    # high: full scores matrix, fastest (default on 128GB)
    # low:  chunked scores + optional fp16 (for 16GB environments)
    chunk_bytes: int | None = None  # None = auto from memory_mode
    use_fp16_scores: bool = False  # halves scores memory, negligible PSNR impact

    @property
    def solver_name(self) -> str:
        """Map mode to voigtfit solver name."""
        return {
            "preview": "amp_only",
            "fast": "4step",
            "precise": "adaptive",
            "gamma_calibrated": "2stage",
            "balanced": "parabola",
        }[self.mode]


@dataclass
class FastFitResult:
    """Result from fast Voigt fitting."""

    amplitudes: NDArray[np.float64]  # (n_components, n_spectra)
    centers: NDArray[np.float64]     # (n_components,) or (n_components, n_spectra)
    sigmas: NDArray[np.float64]      # (n_components,) or (n_components, n_spectra)
    gamma: float                      # Lorentzian width
    chi2: NDArray[np.float64]        # (n_spectra,)
    anomaly_mask: NDArray[np.bool_]  # (n_spectra,)

    # Timing info
    elapsed_ms: float
    spectra_per_second: float

    # Internal voigtfit result (for debugging timing)
    _voigtfit_result: object | None = None

    @property
    def n_spectra(self) -> int:
        return self.chi2.shape[0]

    @property
    def n_components(self) -> int:
        return self.amplitudes.shape[0]

    def to_fitpara(
        self,
        max_components: int = 6,
        energy: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float32]:
        """
        Convert to fitpara format (n_spectra, max_components, 9).

        Compatible with voigtfit/MATLAB fitpara convention:
            [0] peakHeight  - peak height (intensity at center = amplitude × V(center))
            [1] center      - peak center energy (eV)
            [2] fwhm_g      - Gaussian FWHM (eV)
            [3] fwhm_l      - Lorentzian FWHM (eV)
            [4] asymmetry   - not used (0)
            [5] branchRatio  - not used (0)
            [6] soSplit      - not used (0)
            [7] functionType - lineshape (1=Voigt)
            [8] integratedArea - trapz integrated area (eV·counts)

        Args:
            max_components: Maximum number of components (pads with NaN)
            energy: Energy axis for area calculation. If None, col 8 = 0.

        Returns:
            (n_spectra, max_components, 9) float32 array
        """
        n_spec = self.n_spectra
        n_comp = self.n_components
        mc = max(max_components, n_comp)

        fitpara = np.full((n_spec, mc, 9), np.nan, dtype=np.float32)

        # col 0: peak height = amplitude × V(center)
        # V(center) = Re(wofz(iγ/(σ√2))) / (σ√(2π))
        from scipy.special import wofz as _wofz
        sqrt2 = np.sqrt(2)
        sqrt2pi = np.sqrt(2 * np.pi)
        for i in range(n_comp):
            if self.sigmas.ndim == 1:
                # Fixed sigma: scalar v_center
                si = float(self.sigmas[i])
                z_center = 1j * self.gamma / (si * sqrt2)
                v_center = float(np.real(_wofz(z_center))) / (si * sqrt2pi)
                fitpara[:, i, 0] = self.amplitudes[i, :].T.astype(np.float32) * v_center
            else:
                # Per-spectrum sigma: vectorized v_center
                si = self.sigmas[i, :].astype(np.float64)  # (n_spectra,)
                z_center = 1j * self.gamma / (si * sqrt2)
                v_center = np.real(_wofz(z_center)) / (si * sqrt2pi)  # (n_spectra,)
                fitpara[:, i, 0] = (self.amplitudes[i, :] * v_center).astype(np.float32)

        # centers: (n_components,) or (n_components, n_spectra)
        if self.centers.ndim == 1:
            fitpara[:, :n_comp, 1] = self.centers[:n_comp]
        else:
            fitpara[:, :n_comp, 1] = self.centers[:n_comp, :].T

        # sigmas → FWHM_Gaussian: σ * 2√(2ln2) ≈ σ * 2.3548
        if self.sigmas.ndim == 1:
            fitpara[:, :n_comp, 2] = self.sigmas[:n_comp] * 2.3548
        else:
            fitpara[:, :n_comp, 2] = (self.sigmas[:n_comp, :] * 2.3548).T

        # gamma → FWHM_Lorentzian: γ * 2
        fitpara[:, :n_comp, 3] = self.gamma * 2

        # cols 4-6: asymmetry, branch_ratio, so_split (0 for active Voigt components)
        fitpara[:, :n_comp, 4] = 0  # asymmetry / singularity
        fitpara[:, :n_comp, 5] = 0  # branch_ratio
        fitpara[:, :n_comp, 6] = 0  # so_split

        # function type = 1 (Voigt)
        fitpara[:, :n_comp, 7] = 1

        # integratedArea: amplitude × trapz(unit_voigt, energy) per component
        if energy is not None:
            from toyomacro.lineshape import Voigt
            voigt = Voigt()
            energy_f64 = np.asarray(energy, dtype=np.float64)
            for i in range(n_comp):
                if self.sigmas.ndim == 1:
                    si = float(self.sigmas[i])
                    ci = float(self.centers[i])
                else:
                    # Use mean sigma/center for basis integral (δσ~1% → area error ~0.1%)
                    si = float(np.mean(self.sigmas[i, :]))
                    ci = float(np.mean(self.centers[i, :]) if self.centers.ndim > 1
                               else self.centers[i])
                unit_profile = voigt.evaluate(
                    energy_f64,
                    center=ci,
                    amplitude=1.0,
                    fwhm_g=si * 2.3548,
                    fwhm_l=self.gamma * 2,
                )
                basis_integral = float(np.trapezoid(unit_profile, energy_f64))
                # area = amplitude × basis_integral
                fitpara[:, i, 8] = self.amplitudes[i, :].astype(np.float32) * basis_integral

        return fitpara


class FastVoigtFitter:
    """
    High-performance Voigt fitter using voigtfit HybridPipeline.

    Provides massive speedups over lmfit for batch processing:
    - Preview mode: 400M spec/s (amp-only, memory-bandwidth bound)
    - Fast mode: 1.5M spec/s (4-step, δE+δσ recovery)
    - Precise mode: 0.4M spec/s (adaptive dict+4-step, best quality)
    """

    def __init__(self, config: FastFitConfig | None = None):
        """
        Initialize the fast fitter.

        Args:
            config: Fitting configuration. Default uses 'fast' mode.
        """
        if not VOIGTFIT_AVAILABLE:
            raise ImportError(
                "voigtfit not available: toyomacro.voigtfit failed to "
                "import. It is part of the base install, not an extra."
            )

        self.config = config or FastFitConfig()
        self._cache = WeightMatrixCache()
        self._pipeline: HybridPipeline | None = None
        # Dictionary cache for dict-based solvers (element -> DictionaryCache)
        # Avoids rebuilding the dictionary on every call (~250ms savings)
        self._dict_caches: dict[str, object] = {}
        # Multipeak dictionary cache (element -> list[DictionaryCache])
        # Avoids rebuilding per-component dictionaries on every call (~30-80ms savings)
        self._mp_dict_caches: dict[str, list] = {}
        # Profile counter for detailed timing logs (first 10 calls only)
        self._profile_count: int = 0

    def reset_profile_count(self):
        """Reset profile counter (call after warmup to start fresh logging)."""
        self._profile_count = 0

    def _get_pipeline(self) -> HybridPipeline:
        """Get or create the HybridPipeline instance."""
        if self._pipeline is None:
            self._pipeline = HybridPipeline(
                cache=self._cache,
                use_mlx=self.config.use_mlx,
                # Stage 2 is a legacy compatibility path; this fitter uses
                # the 4-step / adaptive dictionary solvers instead.
                enable_stage2=False,
                chi2_threshold=self.config.chi2_threshold,
            )
        return self._pipeline

    def _effective_chunk_bytes(self) -> int | None:
        """Resolve chunk_bytes from config, respecting memory_mode.

        Returns None to use solver default, or explicit byte limit.
        """
        if self.config.chunk_bytes is not None:
            return self.config.chunk_bytes
        if self.config.memory_mode == "high":
            return None  # solver default (4 GB)
        if self.config.memory_mode == "low":
            # Conservative: 256 MB scores budget
            return 256 * 1024**2
        # auto mode
        try:
            from toyomacro.voigtfit.memory import get_available_memory_gb
            avail = get_available_memory_gb()
            if avail < 32.0:
                return 256 * 1024**2  # low-memory
        except ImportError:
            pass
        return None  # default

    def fit_batch(
        self,
        spectra: NDArray[np.float64],
        energy: NDArray[np.float64],
        centers: NDArray[np.float64],
        sigmas: NDArray[np.float64],
        gamma: float,
        element: str = "unknown",
        orbital: str = "1s",
    ) -> FastFitResult:
        """
        Fit a batch of spectra with pre-defined peak positions.

        Accepts both (n_energy, n_spectra) and (n_spectra, n_energy) layouts.
        Internally converts to row-major and delegates to fit_batch_rowmajor.

        Args:
            spectra: (n_energy, n_spectra) or (n_spectra, n_energy) spectral data
            energy: (n_energy,) energy axis in eV
            centers: (n_components,) peak center positions in eV
            sigmas: (n_components,) Gaussian widths in eV
            gamma: Lorentzian width in eV
            element: Element name (for caching)
            orbital: Orbital name (for caching)

        Returns:
            FastFitResult with amplitudes, chi2, etc.
        """
        # Normalize to row-major (n_spectra, n_energy)
        if spectra.ndim == 1:
            spectra = spectra[np.newaxis, :]
        elif spectra.shape[1] == len(energy) and spectra.shape[0] != len(energy):
            pass  # Already (n_spectra, n_energy)
        elif spectra.shape[0] == len(energy):
            spectra = spectra.T  # (n_energy, n_spectra) → (n_spectra, n_energy)

        spectra_f32 = np.ascontiguousarray(spectra, dtype=np.float32)

        return self.fit_batch_rowmajor(
            spectra=spectra_f32,
            energy=energy,
            centers=centers,
            sigmas=sigmas,
            gamma=gamma,
            element=element,
            orbital=orbital,
        )

    def fit_batch_rowmajor(
        self,
        spectra: NDArray[np.float32],
        energy: NDArray[np.float64],
        centers: NDArray[np.float64],
        sigmas: NDArray[np.float64],
        gamma: float,
        element: str = "unknown",
        orbital: str = "1s",
    ) -> FastFitResult:
        """
        Fit a batch of spectra in row-major format (optimized path).

        Routes to the appropriate voigtfit solver based on config.mode
        and number of components:

        Single-component (n_comp == 1):
        - preview → amp-only (400M spec/s)
        - fast → 4-step residual projection (1.5M/s, δE+δσ)
        - precise → adaptive dict+4-step (0.4M/s, best quality)
        - balanced → dict2d + parabola (5M/s, best δσ)

        Multi-component (n_comp >= 2):
        - All modes → multipeak alternating projection (3.3M/s for 2-comp)
          with per-component δE/δσ recovery

        Args:
            spectra: (n_spectra, n_energy) C-contiguous spectral data
            energy: (n_energy,) energy axis in eV
            centers: (n_components,) peak center positions in eV
            sigmas: (n_components,) Gaussian widths in eV
            gamma: Lorentzian width in eV
            element: Element name (for caching)
            orbital: Orbital name (for caching)

        Returns:
            FastFitResult with amplitudes, chi2, etc.
        """
        import time
        t0 = time.perf_counter()

        self._profile_count += 1

        centers_arr = np.asarray(centers, dtype=np.float64)
        sigmas_arr = np.asarray(sigmas, dtype=np.float64)
        energy_arr = np.asarray(energy, dtype=np.float64)
        n_comp = len(centers_arr)

        # Multi-component: use multipeak alternating projection solver
        # which provides per-component δE/δσ (vs global shift in single-peak)
        if n_comp >= 2:
            return self._fit_batch_multipeak(
                spectra, energy_arr, centers_arr, sigmas_arr,
                gamma, element, orbital, t0,
            )

        peak_config = {
            "centers": centers_arr,
            "sigmas": sigmas_arr,
            "gamma": float(gamma),
        }

        pipeline = self._get_pipeline()
        solver = self.config.solver_name

        if solver == "amp_only":
            result = pipeline.process_rowmajor_fast(
                Y=spectra, element=element, orbital=orbital,
                energy=energy_arr,
                peak_config=peak_config, skip_stage2=True,
            )
        elif solver == "4step":
            result = pipeline.process_rowmajor_extended_3param(
                Y=spectra, element=element, orbital=orbital,
                energy=energy_arr,
                peak_config=peak_config, n_steps=4,
            )
        elif solver in ("adaptive", "2stage", "parabola"):
            # Use cached dictionary to avoid rebuilding (~250ms) on each call
            cache_key = f"{element}_{orbital}"
            cached_dict = self._dict_caches.get(cache_key)
            if cached_dict is None:
                from toyomacro.voigtfit.dictionary_solver import build_dictionary
                cached_dict = build_dictionary(
                    energy=energy_arr,
                    centers=peak_config["centers"],
                    sigmas=peak_config["sigmas"],
                    gamma=peak_config["gamma"],
                    include_hessian=False,
                )
                self._dict_caches[cache_key] = cached_dict
            result = pipeline.process_rowmajor_dictionary_hybrid(
                Y=spectra, element=element, orbital=orbital,
                energy=energy_arr,
                peak_config=peak_config, solver=solver,
                dict_cache=cached_dict,
                chunk_bytes=self._effective_chunk_bytes(),
                use_fp16_scores=self.config.use_fp16_scores,
            )
        else:
            raise ValueError(f"Unknown solver: {solver}")

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        n_spectra = spectra.shape[0]

        # For 2-stage, use corrected gamma from timing info
        result_gamma = gamma
        if solver == "2stage" and result.timing and "corrected_gamma" in result.timing:
            result_gamma = result.timing["corrected_gamma"]

        return FastFitResult(
            amplitudes=result.amplitudes,
            centers=result.centers if result.centers is not None else centers_arr,
            sigmas=result.sigmas if result.sigmas is not None else sigmas_arr,
            gamma=result_gamma,
            chi2=result.chi2,
            anomaly_mask=result.anomaly_mask,
            elapsed_ms=elapsed_ms,
            spectra_per_second=n_spectra / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
            _voigtfit_result=result,
        )

    def _fit_batch_multipeak(
        self,
        spectra: NDArray[np.float32],
        energy: NDArray[np.float64],
        centers: NDArray[np.float64],
        sigmas: NDArray[np.float64],
        gamma: float,
        element: str,
        orbital: str,
        t0: float,
    ) -> FastFitResult:
        """Multi-component batch fit via alternating projection.

        Uses per-component δE/δσ recovery (vs global shift in single-peak).
        Routes to process_multipeak (3.3M/s for 2-comp on M3 Max).

        Args:
            spectra: (n_spectra, n_energy) spectra
            energy: (n_energy,) energy axis
            centers: (n_components,) peak centers
            sigmas: (n_components,) Gaussian widths
            gamma: Lorentzian width
            element: Element name
            orbital: Orbital name
            t0: Start time from caller
        """
        import time

        from toyomacro.voigtfit.multipeak_config import (
            ComponentConfig,
            MultiPeakConfig,
        )

        n_comp = len(centers)
        solver = self.config.solver_name

        # Build MultiPeakConfig from peak parameters
        peaks = []
        for i in range(n_comp):
            peaks.append(ComponentConfig(
                center=float(centers[i]),
                sigma=float(sigmas[i]),
                gamma=float(gamma),
                dE_range=2.0,
                ds_range=0.3,
                n_dE=10,
                n_ds=31,
            ))
        mp_config = MultiPeakConfig(peaks=peaks, energy_axis=energy)
        mp_config.constrain_dE_ranges()

        pipeline = self._get_pipeline()

        # Use parabola refinement on δE for all modes except amp_only
        use_parabola_dE = solver != "amp_only"
        # Use parabola on δσ for balanced/precise modes
        use_parabola_ds = solver in ("parabola", "adaptive")

        # Use cached multipeak dictionaries (avoids ~30-80ms rebuild per call)
        cache_key = f"{element}_{orbital}"
        cached_mp_dicts = self._mp_dict_caches.get(cache_key)
        if cached_mp_dicts is None:
            from toyomacro.voigtfit.multipeak_solver import build_multipeak_dictionaries
            cached_mp_dicts = build_multipeak_dictionaries(mp_config, "uniform")
            self._mp_dict_caches[cache_key] = cached_mp_dicts

        mp_result = pipeline.process_multipeak(
            Y=spectra,
            config=mp_config,
            n_iterations=3,
            parabola_dE=use_parabola_dE,
            parabola_ds=use_parabola_ds,
            dicts=cached_mp_dicts,
        )

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        n_spectra = spectra.shape[0]

        # Convert MultiPeakResult → FastFitResult
        # MultiPeakResult: amplitudes (batch, n_comp), delta_E/delta_sigma (batch, n_comp)
        # FastFitResult: amplitudes (n_comp, n_spectra), centers/sigmas (n_comp, n_spectra)
        amps_t = mp_result.amplitudes.T  # (n_comp, n_spectra)

        # Absolute centers/sigmas = nominal + delta (per-spectrum, per-component)
        centers_2d = centers[:, np.newaxis] + mp_result.delta_E.T  # (n_comp, n_spectra)
        sigmas_2d = sigmas[:, np.newaxis] + mp_result.delta_sigma.T  # (n_comp, n_spectra)

        # Anomaly detection from chi2
        chi2 = mp_result.chi2
        if len(chi2) > 10:
            median_chi2 = np.median(chi2)
            mad = np.median(np.abs(chi2 - median_chi2))
            threshold = median_chi2 + self.config.chi2_threshold * 1.4826 * mad
            anomaly_mask = chi2 > threshold
        else:
            anomaly_mask = np.zeros(n_spectra, dtype=bool)

        return FastFitResult(
            amplitudes=amps_t,
            centers=centers_2d,
            sigmas=sigmas_2d,
            gamma=gamma,
            chi2=chi2,
            anomaly_mask=anomaly_mask,
            elapsed_ms=elapsed_ms,
            spectra_per_second=n_spectra / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
            _voigtfit_result=mp_result,
        )

    def fit_batch_mlx(
        self,
        spectra_mlx,  # MLX array (n_spectra, n_energy)
        energy: NDArray[np.float64],
        centers: NDArray[np.float64],
        sigmas: NDArray[np.float64],
        gamma: float,
        element: str = "unknown",
        orbital: str = "1s",
    ) -> FastFitResult:
        """
        Fit a batch of spectra already in MLX format (zero conversion overhead).

        For amp_only mode, passes MLX arrays directly to the pipeline.
        For 4step/adaptive, converts to NumPy (these solvers require it).

        Args:
            spectra_mlx: MLX array (n_spectra, n_energy) - row-major
            energy: (n_energy,) energy axis in eV
            centers: (n_components,) peak center positions in eV
            sigmas: (n_components,) Gaussian widths in eV
            gamma: Lorentzian width in eV
            element: Element name (for caching)
            orbital: Orbital name (for caching)

        Returns:
            FastFitResult with amplitudes, chi2, etc.
        """
        solver = self.config.solver_name

        if solver != "amp_only":
            # 4-step and adaptive solvers require NumPy input
            spectra_np = np.array(spectra_mlx, dtype=np.float32)
            return self.fit_batch_rowmajor(
                spectra=spectra_np, energy=energy, centers=centers,
                sigmas=sigmas, gamma=gamma, element=element, orbital=orbital,
            )

        # amp_only: pass MLX array directly for zero-copy path
        import time
        t0 = time.perf_counter()

        peak_config = {
            "centers": np.asarray(centers, dtype=np.float64),
            "sigmas": np.asarray(sigmas, dtype=np.float64),
            "gamma": float(gamma),
        }

        pipeline = self._get_pipeline()
        result = pipeline.process_rowmajor_fast(
            Y=spectra_mlx, element=element, orbital=orbital,
            energy=np.asarray(energy, dtype=np.float64),
            peak_config=peak_config, skip_stage2=True,
        )

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        n_spectra = spectra_mlx.shape[0]

        return FastFitResult(
            amplitudes=result.amplitudes,
            centers=result.centers if result.centers is not None else centers,
            sigmas=result.sigmas if result.sigmas is not None else sigmas,
            gamma=gamma,
            chi2=result.chi2,
            anomaly_mask=result.anomaly_mask,
            elapsed_ms=elapsed_ms,
            spectra_per_second=n_spectra / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
            _voigtfit_result=result,
        )

    def fit_single(
        self,
        intensity: NDArray[np.float64],
        energy: NDArray[np.float64],
        centers: NDArray[np.float64],
        sigmas: NDArray[np.float64],
        gamma: float,
    ) -> FastFitResult:
        """
        Fit a single spectrum.

        For single spectra, this wraps fit_batch with n_spectra=1.

        Args:
            intensity: (n_energy,) spectral intensity
            energy: (n_energy,) energy axis
            centers: (n_components,) peak centers
            sigmas: (n_components,) Gaussian widths
            gamma: Lorentzian width

        Returns:
            FastFitResult
        """
        return self.fit_batch(
            spectra=intensity[:, np.newaxis],
            energy=energy,
            centers=centers,
            sigmas=sigmas,
            gamma=gamma,
        )


# Numba-optimized Shirley background
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True)
    def shirley_calculate_numba(
        energy: np.ndarray,
        intensity: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-5,
    ) -> np.ndarray:
        """
        Numba-optimized Shirley background calculation.

        ~10x faster than pure Python iteration.

        Args:
            energy: Energy axis (must be monotonic)
            intensity: Spectral intensity
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Background array
        """
        n = len(intensity)
        background = np.empty(n, dtype=np.float64)

        # Determine integration direction based on energy ordering
        if energy[0] > energy[-1]:
            # Binding energy (decreasing)
            i_left = n - 1
            i_right = 0
        else:
            # Kinetic energy (increasing)
            i_left = 0
            i_right = n - 1

        y_left = intensity[i_left]
        y_right = intensity[i_right]

        # Initialize background
        for i in range(n):
            background[i] = y_right

        # Iterate
        for iteration in range(max_iter):
            max_change = 0.0

            # Calculate integrals
            if energy[0] > energy[-1]:
                # Binding energy - integrate from right
                integral_total = 0.0
                for i in range(n - 1, -1, -1):
                    signal = intensity[i] - background[i]
                    if signal > 0:
                        integral_total += signal

                # Update background
                if integral_total > 0:
                    running_integral = 0.0
                    for i in range(n - 1, -1, -1):
                        signal = intensity[i] - background[i]
                        if signal > 0:
                            running_integral += signal

                        new_bg = y_right + (y_left - y_right) * running_integral / integral_total
                        change = abs(new_bg - background[i])
                        if change > max_change:
                            max_change = change
                        background[i] = new_bg
            else:
                # Kinetic energy - integrate from left
                integral_total = 0.0
                for i in range(n):
                    signal = intensity[i] - background[i]
                    if signal > 0:
                        integral_total += signal

                if integral_total > 0:
                    running_integral = 0.0
                    for i in range(n):
                        signal = intensity[i] - background[i]
                        if signal > 0:
                            running_integral += signal

                        new_bg = y_left + (y_right - y_left) * running_integral / integral_total
                        change = abs(new_bg - background[i])
                        if change > max_change:
                            max_change = change
                        background[i] = new_bg

            # Check convergence
            if max_change < tol:
                break

        return background


    @njit(parallel=True, fastmath=True, cache=True)
    def shirley_batch_numba(
        energy: np.ndarray,
        spectra: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-5,
    ) -> np.ndarray:
        """
        Batch Shirley calculation with parallel processing.

        Args:
            energy: (n_energy,) energy axis
            spectra: (n_energy, n_spectra) spectral data
            max_iter: Maximum iterations per spectrum
            tol: Convergence tolerance

        Returns:
            (n_energy, n_spectra) background arrays
        """
        n_energy, n_spectra = spectra.shape
        backgrounds = np.empty((n_energy, n_spectra), dtype=np.float64)

        # Determine integration direction (binding energy = decreasing)
        is_binding = energy[0] > energy[-1]

        for spec_idx in prange(n_spectra):
            intensity = spectra[:, spec_idx]
            background = np.empty(n_energy, dtype=np.float64)

            # Endpoints
            if is_binding:
                y_left = intensity[n_energy - 1]
                y_right = intensity[0]
            else:
                y_left = intensity[0]
                y_right = intensity[n_energy - 1]

            # Initialize
            for i in range(n_energy):
                background[i] = y_right

            # Iterate
            for iteration in range(max_iter):
                max_change = 0.0

                if is_binding:
                    # Calculate total integral
                    integral_total = 0.0
                    for i in range(n_energy - 1, -1, -1):
                        signal = intensity[i] - background[i]
                        if signal > 0:
                            integral_total += signal

                    if integral_total > 0:
                        running_integral = 0.0
                        for i in range(n_energy - 1, -1, -1):
                            signal = intensity[i] - background[i]
                            if signal > 0:
                                running_integral += signal

                            new_bg = y_right + (y_left - y_right) * running_integral / integral_total
                            change = abs(new_bg - background[i])
                            if change > max_change:
                                max_change = change
                            background[i] = new_bg
                else:
                    integral_total = 0.0
                    for i in range(n_energy):
                        signal = intensity[i] - background[i]
                        if signal > 0:
                            integral_total += signal

                    if integral_total > 0:
                        running_integral = 0.0
                        for i in range(n_energy):
                            signal = intensity[i] - background[i]
                            if signal > 0:
                                running_integral += signal

                            new_bg = y_left + (y_right - y_left) * running_integral / integral_total
                            change = abs(new_bg - background[i])
                            if change > max_change:
                                max_change = change
                            background[i] = new_bg

                if max_change < tol:
                    break

            backgrounds[:, spec_idx] = background

        return backgrounds

else:
    # Fallback: no Numba
    def shirley_calculate_numba(energy, intensity, max_iter=50, tol=1e-5):
        """Fallback Shirley (no Numba)."""
        from toyomacro.background import Shirley
        shirley = Shirley(max_iter=max_iter, tol=tol)
        return shirley.calculate(energy, intensity)

    def shirley_batch_numba(energy, spectra, max_iter=50, tol=1e-5):
        """Fallback batch Shirley (no Numba)."""
        n_spectra = spectra.shape[1]
        backgrounds = np.empty_like(spectra)
        for i in range(n_spectra):
            backgrounds[:, i] = shirley_calculate_numba(
                energy, spectra[:, i], max_iter, tol
            )
        return backgrounds


# Export availability flags
__all__ = [
    "VOIGTFIT_AVAILABLE",
    "NUMBA_AVAILABLE",
    "FastVoigtFitter",
    "FastFitConfig",
    "FastFitResult",
    "shirley_calculate_numba",
    "shirley_batch_numba",
]
