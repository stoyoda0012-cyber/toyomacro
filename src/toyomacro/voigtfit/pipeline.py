"""
2-Stage Hybrid Processing Pipeline
===================================

Stage 1: Fast Screening (MLX)
    - Pre-computed weight matrix: A = Wt @ Y
    - Performance: 500-600M spectra/s
    - Output: amplitudes + residual χ²

Stage 2: Precision Refinement (Gauss-Newton)
    - Only for anomalous pixels (typically <1%)
    - Three modes:
        - LIGHT: center only (charging correction)
        - MEDIUM: center + σ (instrument variation)
        - FULL: center + σ + γ (unknown peaks)
    - Uses analytical Jacobian for fast convergence
    - Performance:
        - CPU: ~4K spectra/s
        - MLX (GPU): ~50K spectra/s

Expected performance (8K image = 33M spectra, 1% anomaly = 330K):
    Stage 1: 33M @ 500M/s = 0.07s
    Stage 2 (CPU): 330K @ 4K/s = 82s
    Stage 2 (MLX): 330K @ 50K/s = 6.6s
    Total (MLX): ~7s
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from .stage2 import Stage2Refiner
    from .stage2_mlx import Stage2MLXRefiner
    from .weight_cache import WeightMatrixCache

try:
    import mlx.core as mx

    from ._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND a Metal device works
except ImportError:
    HAS_MLX = False


def _shift_kernel_mlx(Y_mx, W, Phi_T, J_c_T, n_energy_inv):
    """Pure MLX kernel for 2-step residual projection (compilable).

    All inputs and outputs are mx.array — no NumPy mixing.
    mx.compile() fuses element-wise ops between matmuls.

    Args:
        Y_mx: Spectra (n_spectra, n_energy)
        W: Weight matrix (n_energy, n_comp)
        Phi_T: Basis transposed (n_comp, n_energy)
        J_c_T: Jacobian transposed (n_comp, n_energy)
        n_energy_inv: 1/n_energy scalar

    Returns:
        (A_row, chi2, numerator, denominator)
    """
    # Step 1: Standard linear fit
    A_row = Y_mx @ W  # (n_spectra, n_comp)

    # Step 2: Residual + shift projection
    R = Y_mx - A_row @ Phi_T  # (n_spectra, n_energy)
    S = A_row @ J_c_T          # (n_spectra, n_energy)

    # Element-wise ops → fusible by mx.compile
    numerator = mx.sum(R * S, axis=1)
    denominator = mx.sum(S * S, axis=1)
    chi2 = mx.sum(R * R, axis=1) * n_energy_inv

    return A_row, chi2, numerator, denominator


if HAS_MLX:
    from ._mlx_support import LazyCompiled
    _shift_kernel_compiled = LazyCompiled(_shift_kernel_mlx)
else:
    _shift_kernel_compiled = None


def _shift_width_kernel_mlx(Y_mx, W, Phi_T, J_c_T, J_s_T, n_energy_inv):
    """Pure MLX kernel for 3-step residual projection (compilable).

    Recovers amplitude, center shift (δE), and sigma shift (δσ)
    in a single fused GPU pass.

    Returns all numerics needed for CPU post-processing:
        δE = num_c / den_c
        δσ = (RS_s - δE * ScSs) / den_s
    """
    A_row = Y_mx @ W                          # (N, n_comp)
    R = Y_mx - A_row @ Phi_T                  # (N, n_energy)
    S_c = A_row @ J_c_T                       # (N, n_energy)
    S_s = A_row @ J_s_T                       # (N, n_energy)

    num_c = mx.sum(R * S_c, axis=1)           # shift numerator
    den_c = mx.sum(S_c * S_c, axis=1)         # shift denominator
    RS_s = mx.sum(R * S_s, axis=1)            # raw sigma numerator
    ScSs = mx.sum(S_c * S_s, axis=1)          # cross-talk term
    den_s = mx.sum(S_s * S_s, axis=1)         # sigma denominator
    chi2 = mx.sum(R * R, axis=1) * n_energy_inv

    return A_row, chi2, num_c, den_c, RS_s, ScSs, den_s


if HAS_MLX:
    _shift_width_kernel_compiled = LazyCompiled(_shift_width_kernel_mlx)
else:
    _shift_width_kernel_compiled = None


def _six_step_kernel_mlx(Y_mx, W, Phi_T, J_c_T, J_s_T, H_cc_T, n_energy_inv):
    """Pure MLX kernel for 6-step residual projection (compilable).

    Extends the 3-step kernel with Hessian d²V/dc² correction for δσ.
    The ½δE²·H_cc contamination in the σ-direction is computed on GPU,
    then used in CPU post-processing to correct the δσ estimate.

    Returns:
        (A_row, chi2, num_c, den_c, RS_s, ScSs, den_s, hcc_ss)
        where hcc_ss = Σ(H·S_s) is the Hessian contamination term.
    """
    A_row = Y_mx @ W                            # matmul 1
    R = Y_mx - A_row @ Phi_T                    # matmul 2
    S_c = A_row @ J_c_T                          # matmul 3
    S_s = A_row @ J_s_T                          # matmul 4
    H = A_row @ H_cc_T                           # matmul 5 (Hessian projection)

    num_c = mx.sum(R * S_c, axis=1)
    den_c = mx.sum(S_c * S_c, axis=1)
    RS_s = mx.sum(R * S_s, axis=1)
    ScSs = mx.sum(S_c * S_s, axis=1)
    den_s = mx.sum(S_s * S_s, axis=1)
    hcc_ss = mx.sum(H * S_s, axis=1)            # Hessian contamination
    chi2 = mx.sum(R * R, axis=1) * n_energy_inv

    return A_row, chi2, num_c, den_c, RS_s, ScSs, den_s, hcc_ss


if HAS_MLX:
    _six_step_kernel_compiled = LazyCompiled(_six_step_kernel_mlx)
else:
    _six_step_kernel_compiled = None


@dataclass
class FitResult:
    """
    Fitting result container.

    Attributes:
        amplitudes: Peak amplitudes (n_components, n_spectra)
        chi2: Reduced χ² per spectrum (n_spectra,)
        anomaly_mask: Boolean mask for anomalous spectra (n_spectra,)
        refined_params: Stage 2 refinement results (Stage2Result or None)
        timing: Processing time breakdown (dict)
        centers: Peak centers (n_components,) or (n_components, n_spectra) if refined
        sigmas: Gaussian widths (n_components,) or (n_components, n_spectra) if refined
        gammas: Lorentzian widths (n_components,) or (n_components, n_spectra) if refined
    """

    amplitudes: np.ndarray
    chi2: np.ndarray
    anomaly_mask: np.ndarray
    refined_params: Any | None = None
    timing: dict | None = None
    centers: np.ndarray | None = None
    sigmas: np.ndarray | None = None
    gammas: np.ndarray | None = None
    energy_shifts: np.ndarray | None = None  # (n_spectra,) δE in eV
    sigma_shifts: np.ndarray | None = None   # (n_spectra,) δσ in eV
    gamma_shifts: np.ndarray | None = None   # (n_spectra,) δγ in eV


class HybridPipeline:
    """
    2-Stage Hybrid Processing Pipeline.

    Usage:
        from voigtfit import WeightMatrixCache, HybridPipeline

        cache = WeightMatrixCache()
        pipeline = HybridPipeline(cache)

        result = pipeline.process(
            Y=spectra,              # (n_energy, n_spectra)
            element="Fe",
            orbital="2p3/2",
            energy=energy_axis,
            peak_config={
                "centers": np.array([707.0, 709.5, 711.0]),
                "sigmas": np.array([0.8, 0.9, 0.8]),
                "gamma": 0.2,
            }
        )
    """

    def __init__(
        self,
        cache: "WeightMatrixCache",
        chi2_threshold: float = 3.0,
        use_mlx: bool = True,
        enable_stage2: bool = True,
        stage2_mode: Literal["light", "medium", "full"] = "medium",
        use_mlx_stage2: bool = True,
        enable_shift_correction: bool = False,
    ):
        """
        Initialize hybrid pipeline.

        Args:
            cache: Weight matrix cache instance
            chi2_threshold: Anomaly detection threshold (in MAD units)
            use_mlx: Prefer MLX GPU acceleration for Stage 1.  This is
                opportunistic: when MLX is not installed or no Metal
                device can execute work (see ``_mlx_support.mlx_usable``),
                the pipeline silently falls back to the NumPy backend,
                which produces numerically identical results.  To assert
                that the GPU path is actually in use, call
                ``toyomacro.voigtfit.require_mlx()`` first — it raises an
                actionable error otherwise.
            enable_stage2: Enable Stage 2 refinement
            stage2_mode: Stage 2 refinement mode
                - 'light': center only (fastest, for charging correction)
                - 'medium': center + σ (default, for instrument variation)
                - 'full': center + σ + γ (slowest, for unknown peaks)
            use_mlx_stage2: Use MLX for Stage 2 (GPU acceleration, ~13x faster)
            enable_shift_correction: Enable extended SVD shift correction.
                Adds a global energy shift column to the basis matrix,
                recovering per-spectrum δE in a single linear solve.
                When enabled, Stage 2 is not used.
        """
        self.cache = cache
        self.chi2_threshold = chi2_threshold
        self.use_mlx = use_mlx and HAS_MLX
        self.enable_stage2 = enable_stage2
        self.stage2_mode = stage2_mode
        self.use_mlx_stage2 = use_mlx_stage2 and HAS_MLX
        self.enable_shift_correction = enable_shift_correction
        self.use_compiled_shift = use_mlx and HAS_MLX  # mx.compile for shift kernel
        self._stage2_refiner: Stage2Refiner | None = None
        self._stage2_mlx_refiner: Stage2MLXRefiner | None = None

    def stage1_screening(
        self,
        Y: Any,  # np.ndarray or mx.array
        Wt: Any,
        Phi: Any,
        chi2_sample_ratio: float = 0.0001,  # 0.01% = optimal speed
        return_mlx: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Stage 1: Fast linear fitting via pre-computed weight matrix.

        Performance: ~290M spectra/s on M1 Max with MLX (0.01% chi2 sampling)

        Uses sampled χ² calculation for speed:
        1. Compute amplitudes for all spectra
        2. Compute χ² for sampled spectra only
        3. Interpolate/estimate χ² for non-sampled

        Args:
            Y: Spectra matrix (n_energy, n_spectra) - NumPy or MLX array
            Wt: Weight matrix (n_components, n_energy)
            Phi: Basis matrix (n_energy, n_components)
            chi2_sample_ratio: Fraction of spectra for χ² calculation
            return_mlx: If True, return MLX arrays (avoids conversion overhead)

        Returns:
            (amplitudes, chi2): Fitted amplitudes and reduced χ²
        """
        n_spectra = Y.shape[1]
        n_energy = Y.shape[0]

        if self.use_mlx:
            # Check if Y is already MLX array (avoid expensive conversion)
            if HAS_MLX and isinstance(Y, mx.array):
                Y_mx = Y
            else:
                # Only convert if necessary - this is the expensive operation
                Y_mx = mx.array(Y if isinstance(Y, np.ndarray) else np.asarray(Y, dtype=np.float32))

            # Core computation: A = Wt @ Y (fast: 500M+ spec/s)
            A = Wt @ Y_mx

            # Sampled χ² calculation for speed (fused - single eval)
            sample_step = max(1, int(1.0 / chi2_sample_ratio))
            if sample_step > 1 and n_spectra > 10000:
                # Sample mode: compute χ² for subset
                Y_sample = Y_mx[:, ::sample_step]
                A_sample = A[:, ::sample_step]
                Y_fit_sample = Phi @ A_sample
                chi2_sample = mx.sum((Y_sample - Y_fit_sample)**2, axis=0) / n_energy
                # Single fused eval for both A and chi2
                mx.eval(A, chi2_sample)

                # Expand to full size (nearest neighbor) - done after eval
                chi2_sample_np = np.array(chi2_sample)
                chi2 = np.repeat(chi2_sample_np, sample_step)[:n_spectra]
                return np.array(A), chi2
            else:
                # Full χ² calculation for small datasets
                Y_fit = Phi @ A
                chi2 = mx.sum((Y_mx - Y_fit)**2, axis=0) / n_energy
                # Single fused eval
                mx.eval(A, chi2)
                return np.array(A), np.array(chi2)
        else:
            # NumPy fallback
            A = Wt @ Y
            Y_fit = Phi @ A
            residuals = Y - Y_fit
            chi2 = np.sum(residuals**2, axis=0) / Y.shape[0]
            return A, chi2

    def stage1_screening_rowmajor(
        self,
        Y: Any,  # np.ndarray or mx.array, shape (n_spectra, n_energy)
        W: Any,   # Weight matrix (n_energy, n_components) - note: not transposed
        Phi: Any,  # Basis matrix (n_energy, n_components)
        chi2_sample_ratio: float = 0.0001,  # 0.01% = optimal speed
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Stage 1: Fast linear fitting for row-major (C-contiguous) data.

        This version avoids the expensive transpose operation by computing:
            A = Y @ W  instead of  A = Wt @ Y.T

        Performance: ~290M spectra/s on M1 Max (0.01% chi2 sampling)

        Args:
            Y: Spectra matrix (n_spectra, n_energy) - C-contiguous!
            W: Weight matrix (n_energy, n_components) - NOT transposed
            Phi: Basis matrix (n_energy, n_components)
            chi2_sample_ratio: Fraction of spectra for χ² calculation

        Returns:
            (amplitudes, chi2): amplitudes shape (n_components, n_spectra), chi2 shape (n_spectra,)
        """
        n_spectra = Y.shape[0]
        n_energy = Y.shape[1]

        if self.use_mlx:
            # Check if Y is already MLX array
            if HAS_MLX and isinstance(Y, mx.array):
                Y_mx = Y
            else:
                # Convert - should be fast if Y is C-contiguous
                Y_mx = mx.array(Y if isinstance(Y, np.ndarray) else np.asarray(Y, dtype=np.float32))

            # Core computation: A = Y @ W, result is (n_spectra, n_components)
            A_row = Y_mx @ W

            # Sampled χ² calculation for speed (5x faster than full)
            # For large datasets, sample 1% of spectra for chi2
            sample_step = max(1, int(1.0 / chi2_sample_ratio))
            if sample_step > 1 and n_spectra > 10000:
                # Sample mode: compute chi2 for 1% of spectra
                Y_sample = Y_mx[::sample_step, :]
                A_sample = A_row[::sample_step, :]
                Y_fit_sample = A_sample @ Phi.T
                chi2_sample = mx.sum((Y_sample - Y_fit_sample)**2, axis=1) / n_energy
                mx.eval(A_row, chi2_sample)

                # Expand to full size (nearest neighbor interpolation)
                chi2_sample_np = np.array(chi2_sample)
                chi2 = np.repeat(chi2_sample_np, sample_step)[:n_spectra]
                return np.array(A_row).T, chi2
            else:
                # Full χ² calculation for small datasets
                Y_fit = A_row @ Phi.T
                chi2 = mx.sum((Y_mx - Y_fit)**2, axis=1) / n_energy
                mx.eval(A_row, chi2)
                return np.array(A_row).T, np.array(chi2)
        else:
            # NumPy fallback
            A_row = Y @ W  # (n_spectra, n_components)
            Y_fit = A_row @ Phi.T  # (n_spectra, n_energy)
            chi2 = np.sum((Y - Y_fit)**2, axis=1) / n_energy
            return A_row.T, chi2

    def detect_anomalies(self, chi2: np.ndarray, use_sampling: bool = True) -> np.ndarray:
        """
        Detect anomalous spectra using adaptive MAD-based threshold.

        Uses Median Absolute Deviation (MAD) for robust outlier detection:
            threshold = median(χ²) + k × 1.4826 × MAD(χ²)

        The factor 1.4826 makes MAD consistent with standard deviation
        for normal distributions.

        Optimization: For large datasets (>10K), uses sampled median/MAD
        calculation which is 2x faster with negligible accuracy loss.

        Args:
            chi2: Reduced χ² values (n_spectra,) - NumPy or MLX array
            use_sampling: Use sampled median for large datasets (default True)

        Returns:
            anomaly_mask: Boolean mask (True = anomalous)
        """
        n_spectra = chi2.shape[0]

        # For large datasets, use sampled median/MAD (2x faster)
        if use_sampling and n_spectra > 10000:
            sample_size = min(10000, n_spectra)
            sample_idx = np.random.choice(n_spectra, sample_size, replace=False)
            chi2_sample = chi2[sample_idx] if isinstance(chi2, np.ndarray) else np.array(chi2)[sample_idx]
            median_chi2 = np.median(chi2_sample)
            mad = np.median(np.abs(chi2_sample - median_chi2))
        else:
            # Full calculation for small datasets
            chi2_np = chi2 if isinstance(chi2, np.ndarray) else np.array(chi2)
            median_chi2 = np.median(chi2_np)
            mad = np.median(np.abs(chi2_np - median_chi2))

        # Avoid division by zero for perfect fits
        if mad < 1e-10:
            return np.zeros(n_spectra, dtype=bool)

        threshold = median_chi2 + self.chi2_threshold * 1.4826 * mad

        # Return NumPy array
        if isinstance(chi2, np.ndarray):
            return chi2 > threshold
        else:
            return np.array(chi2) > threshold

    def detect_anomalies_mlx(self, chi2_mx: "mx.array", threshold: float) -> "mx.array":
        """
        MLX-native anomaly detection (fused version).

        For maximum performance when chi2 is already in MLX.
        Threshold should be pre-computed from sampled data.

        Args:
            chi2_mx: χ² values as MLX array
            threshold: Pre-computed threshold value

        Returns:
            anomaly_mask: Boolean MLX array
        """
        return chi2_mx > threshold

    def compute_threshold_from_sample(self, chi2: np.ndarray, sample_size: int = 10000) -> float:
        """
        Compute anomaly threshold from sampled chi2 values.

        Args:
            chi2: Full chi2 array (NumPy)
            sample_size: Number of samples for median/MAD calculation

        Returns:
            threshold: Computed threshold value
        """
        n_spectra = len(chi2)
        if n_spectra <= sample_size:
            chi2_sample = chi2
        else:
            sample_idx = np.random.choice(n_spectra, sample_size, replace=False)
            chi2_sample = chi2[sample_idx]

        median_chi2 = np.median(chi2_sample)
        mad = np.median(np.abs(chi2_sample - median_chi2))

        if mad < 1e-10:
            return float('inf')  # No anomalies if MAD is zero

        return median_chi2 + self.chi2_threshold * 1.4826 * mad

    def stage2_refinement(
        self,
        Y: np.ndarray,
        energy: np.ndarray,
        anomaly_indices: np.ndarray,
        initial_amplitudes: np.ndarray,
        peak_config: dict,
    ):
        """
        Stage 2: Precision Gauss-Newton refinement for anomalous spectra.

        Uses analytical Jacobian to optimize non-linear parameters
        (center, sigma, gamma) while analytically solving for amplitudes.

        Modes:
        - LIGHT: center only (charging correction)
        - MEDIUM: center + σ (instrument variation)
        - FULL: center + σ + γ (unknown peaks)

        Backend selection:
        - MLX (default if available): ~50K spectra/s (13x faster)
        - CPU fallback: ~4K spectra/s

        Args:
            Y: Full spectra matrix (n_energy, n_spectra)
            energy: Energy axis (n_energy,)
            anomaly_indices: Indices of anomalous spectra
            initial_amplitudes: Stage 1 amplitudes for initialization
            peak_config: Peak configuration dict

        Returns:
            Stage2Result or Stage2MLXResult with refined parameters
        """
        # Extract anomalous spectra
        Y_anomaly = Y[:, anomaly_indices]

        # Prepare gamma array
        gamma = peak_config.get("gamma", 0.2)
        n_comp = len(peak_config["centers"])
        if np.isscalar(gamma):
            gammas = np.full(n_comp, gamma)
        else:
            gammas = np.asarray(gamma)

        # Choose backend: MLX (GPU) or CPU
        if self.use_mlx_stage2:
            return self._stage2_refinement_mlx(
                Y_anomaly, energy, peak_config, gammas, anomaly_indices
            )
        else:
            return self._stage2_refinement_cpu(
                Y_anomaly, energy, peak_config, gammas, anomaly_indices
            )

    def _stage2_refinement_mlx(
        self,
        Y_anomaly: np.ndarray,
        energy: np.ndarray,
        peak_config: dict,
        gammas: np.ndarray,
        anomaly_indices: np.ndarray,
    ):
        """MLX-accelerated Stage 2 refinement (~50K spectra/s)."""
        if self._stage2_mlx_refiner is None:
            from .stage2_mlx import Stage2MLXConfig, Stage2MLXRefiner

            config = Stage2MLXConfig(mode=self.stage2_mode)
            self._stage2_mlx_refiner = Stage2MLXRefiner(config)

        result = self._stage2_mlx_refiner.refine_batch(
            Y_anomaly=Y_anomaly,
            energy=energy,
            centers0=peak_config["centers"],
            sigmas0=peak_config["sigmas"],
            gammas0=gammas,
            anomaly_indices=anomaly_indices,
        )

        return result

    def _stage2_refinement_cpu(
        self,
        Y_anomaly: np.ndarray,
        energy: np.ndarray,
        peak_config: dict,
        gammas: np.ndarray,
        anomaly_indices: np.ndarray,
    ):
        """CPU-based Stage 2 refinement (~4K spectra/s)."""
        if self._stage2_refiner is None:
            from .stage2 import Stage2Config, Stage2Refiner

            config = Stage2Config(mode=self.stage2_mode)
            self._stage2_refiner = Stage2Refiner(config)

        result = self._stage2_refiner.refine_batch(
            Y_anomaly=Y_anomaly,
            energy=energy,
            centers0=peak_config["centers"],
            sigmas0=peak_config["sigmas"],
            gammas0=gammas,
            anomaly_indices=anomaly_indices,
        )

        return result

    def process(
        self,
        Y: Any,  # np.ndarray or mx.array
        element: str,
        orbital: str,
        energy: np.ndarray,
        peak_config: dict,
        Y_is_mlx: bool = False,
    ) -> FitResult:
        """
        Run full 2-stage hybrid pipeline.

        Args:
            Y: Spectra matrix (n_energy, n_spectra) - NumPy or MLX array
            element: Element symbol (e.g., "Fe")
            orbital: Orbital name (e.g., "2p3/2")
            energy: Energy axis (n_energy,)
            peak_config: Peak configuration dict with keys:
                - centers: Peak positions (n_components,)
                - sigmas: Gaussian widths (n_components,)
                - gamma: Lorentzian width (scalar)

            Y_is_mlx: If True, Y is already an MLX array (skips conversion)

        Returns:
            FitResult with amplitudes, χ², anomaly mask, and timing
        """
        timing = {}
        n_spectra = Y.shape[1]

        # Convert to MLX once at entry point (avoid repeated conversions)
        if self.use_mlx and HAS_MLX:
            if not isinstance(Y, mx.array):
                t0 = time.perf_counter()
                # Avoid unnecessary copy if already float32
                if isinstance(Y, np.ndarray) and Y.dtype == np.float32:
                    Y_np = Y
                else:
                    Y_np = np.asarray(Y, dtype=np.float32)
                Y = mx.array(Y_np)
                timing["mlx_convert"] = time.perf_counter() - t0
            Y_np_for_stage2 = None  # Lazy: only convert back if needed for Stage 2
        else:
            Y = np.asarray(Y, dtype=np.float32)
            Y_np_for_stage2 = Y

        energy = np.asarray(energy, dtype=np.float32)

        # Get cached weight matrix
        t0 = time.perf_counter()
        Wt, Phi = self.cache.get_or_create(
            element=element,
            orbital=orbital,
            energy=energy,
            centers=peak_config["centers"],
            sigmas=peak_config["sigmas"],
            gamma=peak_config.get("gamma", 0.2),
            use_mlx=self.use_mlx,
            alphas=peak_config.get("alphas"),
            lineshape_types=peak_config.get("lineshape_types"),
        )
        timing["cache_lookup"] = time.perf_counter() - t0

        # Stage 1: Fast screening (Y is already MLX array if use_mlx)
        t0 = time.perf_counter()
        amplitudes, chi2 = self.stage1_screening(Y, Wt, Phi)
        timing["stage1"] = time.perf_counter() - t0
        timing["stage1_rate"] = n_spectra / timing["stage1"] if timing["stage1"] > 0 else float("inf")

        # Anomaly detection
        t0 = time.perf_counter()
        anomaly_mask = self.detect_anomalies(chi2)
        n_anomaly = np.sum(anomaly_mask)
        timing["anomaly_detection"] = time.perf_counter() - t0
        timing["n_anomaly"] = int(n_anomaly)
        timing["anomaly_ratio"] = n_anomaly / n_spectra

        # Initialize output parameter arrays (Stage 1 values)
        n_comp = len(peak_config["centers"])
        centers_out = np.tile(peak_config["centers"][:, np.newaxis], (1, n_spectra))
        sigmas_out = np.tile(peak_config["sigmas"][:, np.newaxis], (1, n_spectra))
        gamma = peak_config.get("gamma", 0.2)
        if np.isscalar(gamma):
            gammas_out = np.full((n_comp, n_spectra), gamma)
        else:
            gammas_out = np.tile(np.asarray(gamma)[:, np.newaxis], (1, n_spectra))

        # Stage 2: Precision refinement (if enabled and needed)
        refined = None
        if self.enable_stage2 and n_anomaly > 0:
            t0 = time.perf_counter()
            anomaly_indices = np.where(anomaly_mask)[0]

            # Convert Y back to NumPy if needed for Stage 2
            if Y_np_for_stage2 is None and HAS_MLX and isinstance(Y, mx.array):
                Y_np_for_stage2 = np.array(Y)

            refined = self.stage2_refinement(
                Y=Y_np_for_stage2,
                energy=energy,
                anomaly_indices=anomaly_indices,
                initial_amplitudes=amplitudes,
                peak_config=peak_config,
            )

            # Update amplitudes and parameters with refined values
            if refined is not None:
                amplitudes[:, anomaly_indices] = refined.amplitudes
                centers_out[:, anomaly_indices] = refined.centers
                sigmas_out[:, anomaly_indices] = refined.sigmas
                gammas_out[:, anomaly_indices] = refined.gammas
                # Update chi2 for refined spectra
                chi2[anomaly_indices] = refined.chi2

            timing["stage2"] = time.perf_counter() - t0
            timing["stage2_rate"] = n_anomaly / timing["stage2"] if timing["stage2"] > 0 else float("inf")
            timing["stage2_converged"] = int(refined.converged.sum()) if refined is not None else 0
            timing["stage2_backend"] = "MLX" if self.use_mlx_stage2 else "CPU"

        timing["total"] = sum(
            v for k, v in timing.items()
            if k not in ["stage1_rate", "stage2_rate", "n_anomaly", "anomaly_ratio", "stage2_converged", "stage2_backend"]
            and isinstance(v, (int, float))
        )

        return FitResult(
            amplitudes=amplitudes,
            chi2=chi2,
            anomaly_mask=anomaly_mask,
            refined_params=refined,
            timing=timing,
            centers=centers_out,
            sigmas=sigmas_out,
            gammas=gammas_out,
        )

    def process_batch(
        self,
        Y_list: list,
        element: str,
        orbital: str,
        energy: np.ndarray,
        peak_config: dict,
    ) -> list:
        """
        Process multiple spectral datasets.

        Useful for processing multiple spatial tiles or time points.

        Args:
            Y_list: List of spectra matrices
            element, orbital, energy, peak_config: Same as process()

        Returns:
            List of FitResult objects
        """
        results = []
        for Y in Y_list:
            result = self.process(Y, element, orbital, energy, peak_config)
            results.append(result)
        return results

    def process_rowmajor(
        self,
        Y: Any,  # np.ndarray or mx.array, shape (n_spectra, n_energy)
        element: str,
        orbital: str,
        energy: np.ndarray,
        peak_config: dict,
        amplitudes_only: bool = False,
    ) -> FitResult:
        """
        Run 2-stage pipeline for row-major (C-contiguous) data.

        This is the optimized version for HDF5 data which is stored as
        (n_spectra, n_energy). Avoids expensive transpose operations.

        Performance: ~10M spectra/s (5x faster than column-major with transpose)

        When amplitudes_only=True, skips anomaly detection, Stage 2, and
        output parameter arrays (centers/sigmas/gammas). Returns only
        amplitudes and chi2. ~40% faster for large datasets where Stage 2
        is disabled or not needed (e.g., roundtrip benchmarks).

        Args:
            Y: Spectra matrix (n_spectra, n_energy) - C-contiguous!
            element: Element symbol (e.g., "Fe")
            orbital: Orbital name (e.g., "2p3/2")
            energy: Energy axis (n_energy,)
            peak_config: Peak configuration dict
            amplitudes_only: If True, skip anomaly detection and output
                parameter construction for maximum throughput.

        Returns:
            FitResult with amplitudes (n_components, n_spectra), χ², etc.
        """
        timing = {}
        n_spectra = Y.shape[0]
        n_energy = Y.shape[1]

        # Ensure float32 and C-contiguous
        if isinstance(Y, np.ndarray):
            if Y.dtype != np.float32:
                Y = Y.astype(np.float32)
            # Check C-contiguous - if not, make a copy
            if not Y.flags['C_CONTIGUOUS']:
                t0 = time.perf_counter()
                Y = np.ascontiguousarray(Y)
                timing["contiguous_copy"] = time.perf_counter() - t0

        energy = np.asarray(energy, dtype=np.float32)

        # Get cached weight matrix (get W instead of Wt for row-major)
        t0 = time.perf_counter()
        Wt, Phi = self.cache.get_or_create(
            element=element,
            orbital=orbital,
            energy=energy,
            centers=peak_config["centers"],
            sigmas=peak_config["sigmas"],
            gamma=peak_config.get("gamma", 0.2),
            use_mlx=self.use_mlx,
            alphas=peak_config.get("alphas"),
            lineshape_types=peak_config.get("lineshape_types"),
        )
        # W = Wt.T for row-major computation
        if self.use_mlx and HAS_MLX:
            W = Wt.T  # MLX transpose is a view, very fast
        else:
            W = Wt.T
        timing["cache_lookup"] = time.perf_counter() - t0

        # Convert Y to MLX if needed (should be fast for C-contiguous)
        if self.use_mlx and HAS_MLX and not isinstance(Y, mx.array):
            t0 = time.perf_counter()
            Y = mx.array(Y)
            timing["mlx_convert"] = time.perf_counter() - t0

        # Stage 1: Fast screening (row-major version)
        t0 = time.perf_counter()
        amplitudes, chi2 = self.stage1_screening_rowmajor(Y, W, Phi)
        timing["stage1"] = time.perf_counter() - t0
        timing["stage1_rate"] = n_spectra / timing["stage1"] if timing["stage1"] > 0 else float("inf")

        # Fast path: skip anomaly detection and output construction
        if amplitudes_only:
            timing["total"] = sum(
                v for k, v in timing.items()
                if k not in ["stage1_rate"] and isinstance(v, (int, float))
            )
            return FitResult(
                amplitudes=amplitudes,
                chi2=chi2,
                anomaly_mask=np.empty(0, dtype=bool),
                timing=timing,
            )

        # Anomaly detection
        t0 = time.perf_counter()
        anomaly_mask = self.detect_anomalies(chi2)
        n_anomaly = np.sum(anomaly_mask)
        timing["anomaly_detection"] = time.perf_counter() - t0
        timing["n_anomaly"] = int(n_anomaly)
        timing["anomaly_ratio"] = n_anomaly / n_spectra

        # Initialize output parameter arrays
        n_comp = len(peak_config["centers"])
        centers_out = np.tile(peak_config["centers"][:, np.newaxis], (1, n_spectra))
        sigmas_out = np.tile(peak_config["sigmas"][:, np.newaxis], (1, n_spectra))
        gamma = peak_config.get("gamma", 0.2)
        if np.isscalar(gamma):
            gammas_out = np.full((n_comp, n_spectra), gamma)
        else:
            gammas_out = np.tile(np.asarray(gamma)[:, np.newaxis], (1, n_spectra))

        # Stage 2 (if enabled) - needs column-major Y
        refined = None
        if self.enable_stage2 and n_anomaly > 0:
            t0 = time.perf_counter()
            anomaly_indices = np.where(anomaly_mask)[0]

            # Convert Y to column-major for Stage 2
            if HAS_MLX and isinstance(Y, mx.array):
                Y_colmajor = np.array(Y).T  # (n_energy, n_spectra)
            else:
                Y_colmajor = Y.T

            refined = self.stage2_refinement(
                Y=Y_colmajor,
                energy=energy,
                anomaly_indices=anomaly_indices,
                initial_amplitudes=amplitudes,
                peak_config=peak_config,
            )

            if refined is not None:
                amplitudes[:, anomaly_indices] = refined.amplitudes
                centers_out[:, anomaly_indices] = refined.centers
                sigmas_out[:, anomaly_indices] = refined.sigmas
                gammas_out[:, anomaly_indices] = refined.gammas
                chi2[anomaly_indices] = refined.chi2

            timing["stage2"] = time.perf_counter() - t0
            timing["stage2_rate"] = n_anomaly / timing["stage2"] if timing["stage2"] > 0 else float("inf")
            timing["stage2_converged"] = int(refined.converged.sum()) if refined is not None else 0
            timing["stage2_backend"] = "MLX" if self.use_mlx_stage2 else "CPU"

        timing["total"] = sum(
            v for k, v in timing.items()
            if k not in ["stage1_rate", "stage2_rate", "n_anomaly", "anomaly_ratio", "stage2_converged", "stage2_backend"]
            and isinstance(v, (int, float))
        )

        return FitResult(
            amplitudes=amplitudes,
            chi2=chi2,
            anomaly_mask=anomaly_mask,
            refined_params=refined,
            timing=timing,
            centers=centers_out,
            sigmas=sigmas_out,
            gammas=gammas_out,
        )

    def stage1_shift_rowmajor(
        self,
        Y: Any,  # np.ndarray or mx.array, shape (n_spectra, n_energy)
        W: Any,   # Weight matrix (n_energy, n_comp)
        Phi: Any,  # Basis matrix (n_energy, n_comp)
        J_c: Any,  # Center Jacobian (n_energy, n_comp)
        chi2_sample_ratio: float = 0.0001,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2-step residual projection for global energy shift estimation.

        Step 1: Standard linear fit  A = Y @ W
        Step 2: Residual projection  δE = -Σ(r·s)/Σ(s²)
                where r = Y - A@Φ', s = A@J_c' (amplitude-weighted shift direction)

        The physical model: Y(E) ≈ (Φ - δE·J_c) @ a
        Residual: r ≈ -δE · J_c @ a
        Solve: δE = -r·(J_c@a) / |J_c@a|²

        Sign convention: positive δE means peaks shifted to higher energy.
        Physical model: y(E) ≈ Φ(c+δE) @ a ≈ (Φ + δE·J_c) @ a
        Residual: r = y - Φ@a ≈ δE · (J_c @ a)
        Solution: δE = r·s / |s|²  where s = J_c @ a

        Args:
            Y: Spectra (n_spectra, n_energy) C-contiguous
            W: Weight matrix (n_energy, n_comp) — Wt.T
            Phi: Basis matrix (n_energy, n_comp)
            J_c: Center Jacobian (n_energy, n_comp)
            chi2_sample_ratio: Fraction for χ² sampling

        Returns:
            (amplitudes, chi2, energy_shifts):
                amplitudes: (n_comp, n_spectra)
                chi2: (n_spectra,) — after shift correction
                energy_shifts: (n_spectra,) δE in eV
        """
        n_spectra = Y.shape[0]
        n_energy = Y.shape[1]

        if self.use_mlx:
            if HAS_MLX and isinstance(Y, mx.array):
                Y_mx = Y
            else:
                Y_mx = mx.array(Y if isinstance(Y, np.ndarray) else np.asarray(Y, dtype=np.float32))

            # Pre-transpose for compiled kernel (avoids transpose inside compiled fn)
            Phi_T = Phi.T if not hasattr(Phi, '_shift_transposed') else Phi._Phi_T
            J_c_T = J_c.T if not hasattr(J_c, '_shift_transposed') else J_c._J_c_T
            n_energy_inv = mx.array(1.0 / n_energy)

            if self.use_compiled_shift and _shift_kernel_compiled is not None:
                A_row, chi2, numerator, denominator = _shift_kernel_compiled(
                    Y_mx, W, Phi_T, J_c_T, n_energy_inv
                )
            else:
                A_row, chi2, numerator, denominator = _shift_kernel_mlx(
                    Y_mx, W, Phi_T, J_c_T, n_energy_inv
                )

            mx.eval(A_row, chi2, numerator, denominator)

            A_np = np.array(A_row)
            chi2_np = np.array(chi2)
            num_np = np.array(numerator)
            den_np = np.array(denominator)

            amplitudes = A_np.T  # (n_comp, n_spectra)

        else:
            A_row = Y @ np.asarray(W)  # (n_spectra, n_comp)
            Y_fit = A_row @ np.asarray(Phi).T
            R = Y - Y_fit
            S = A_row @ np.asarray(J_c).T
            num_np = np.sum(R * S, axis=1)
            den_np = np.sum(S * S, axis=1)
            chi2_np = np.sum(R * R, axis=1) / n_energy
            amplitudes = A_row.T

        # Physical shift: δE = num/den
        # Guard against zero denominator (flat spectrum)
        safe_den = np.where(den_np > 1e-20, den_np, 1.0)
        energy_shifts = np.where(den_np > 1e-20, num_np / safe_den, 0.0)

        return amplitudes, chi2_np, energy_shifts.astype(np.float32)

    def stage1_shift_width_rowmajor(
        self,
        Y: Any,
        W: Any,
        Phi: Any,
        J_c: Any,
        J_sigma: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """3-step residual projection: amplitude + shift + width.

        Step 1: A = Y @ W                    (standard fit → amplitude)
        Step 2: δE = Σ(R·S_c) / Σ(S_c²)     (shift recovery)
        Step 3: δσ = Σ(R₂·S_σ) / Σ(S_σ²)    (width recovery, cross-talk corrected)

        Returns:
            (amplitudes, chi2, energy_shifts, sigma_shifts)
        """
        n_spectra = Y.shape[0]
        n_energy = Y.shape[1]

        if self.use_mlx:
            if HAS_MLX and isinstance(Y, mx.array):
                Y_mx = Y
            else:
                Y_mx = mx.array(Y if isinstance(Y, np.ndarray)
                                else np.asarray(Y, dtype=np.float32))

            Phi_T = Phi.T
            J_c_T = J_c.T
            J_s_T = J_sigma.T
            n_energy_inv = mx.array(1.0 / n_energy)

            if self.use_compiled_shift and _shift_width_kernel_compiled is not None:
                out = _shift_width_kernel_compiled(
                    Y_mx, W, Phi_T, J_c_T, J_s_T, n_energy_inv
                )
            else:
                out = _shift_width_kernel_mlx(
                    Y_mx, W, Phi_T, J_c_T, J_s_T, n_energy_inv
                )

            A_row, chi2, num_c, den_c, RS_s, ScSs, den_s = out
            mx.eval(A_row, chi2, num_c, den_c, RS_s, ScSs, den_s)

            A_np = np.array(A_row)
            chi2_np = np.array(chi2)
            num_c_np = np.array(num_c)
            den_c_np = np.array(den_c)
            RS_s_np = np.array(RS_s)
            ScSs_np = np.array(ScSs)
            den_s_np = np.array(den_s)
            amplitudes = A_np.T

        else:
            W_np = np.asarray(W)
            Phi_np = np.asarray(Phi)
            J_c_np = np.asarray(J_c)
            J_s_np = np.asarray(J_sigma)

            A_row = Y @ W_np
            R = Y - A_row @ Phi_np.T
            S_c = A_row @ J_c_np.T
            S_s = A_row @ J_s_np.T

            num_c_np = np.sum(R * S_c, axis=1)
            den_c_np = np.sum(S_c * S_c, axis=1)
            RS_s_np = np.sum(R * S_s, axis=1)
            ScSs_np = np.sum(S_c * S_s, axis=1)
            den_s_np = np.sum(S_s * S_s, axis=1)
            chi2_np = np.sum(R * R, axis=1) / n_energy
            amplitudes = A_row.T

        # CPU post-processing: sequential δE → δσ
        safe_den_c = np.where(den_c_np > 1e-20, den_c_np, 1.0)
        energy_shifts = np.where(den_c_np > 1e-20, num_c_np / safe_den_c, 0.0)

        # Cross-talk corrected sigma: δσ = (RS_s - δE·ScSs) / den_s
        num_s_corrected = RS_s_np - energy_shifts * ScSs_np
        safe_den_s = np.where(den_s_np > 1e-20, den_s_np, 1.0)
        sigma_shifts = np.where(den_s_np > 1e-20, num_s_corrected / safe_den_s, 0.0)

        return (amplitudes, chi2_np,
                energy_shifts.astype(np.float32),
                sigma_shifts.astype(np.float32))

    def stage1_shift_width_rowmajor_6step(
        self,
        Y: Any,
        W: Any,
        Phi: Any,
        J_c: Any,
        J_sigma: Any,
        H_cc: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """6-step residual projection: amplitude + shift + Hessian-corrected width.

        Step 1: A = Y @ W                    (standard fit → amplitude)
        Step 2: δE = Σ(R·S_c) / Σ(S_c²)     (shift recovery)
        Step 2.5: Compute ½δE²·Σ(H_cc·S_σ)  (Hessian contamination)
        Step 3: δσ = (Σ(R·S_σ) - δE·ScSs - ½δE²·hcc_ss) / Σ(S_σ²)

        Returns:
            (amplitudes, chi2, energy_shifts, sigma_shifts)
        """
        n_spectra = Y.shape[0]
        n_energy = Y.shape[1]

        if self.use_mlx:
            if HAS_MLX and isinstance(Y, mx.array):
                Y_mx = Y
            else:
                Y_mx = mx.array(Y if isinstance(Y, np.ndarray)
                                else np.asarray(Y, dtype=np.float32))

            Phi_T = Phi.T
            J_c_T = J_c.T
            J_s_T = J_sigma.T
            H_cc_T = H_cc.T
            n_energy_inv = mx.array(1.0 / n_energy)

            if self.use_compiled_shift and _six_step_kernel_compiled is not None:
                out = _six_step_kernel_compiled(
                    Y_mx, W, Phi_T, J_c_T, J_s_T, H_cc_T, n_energy_inv
                )
            else:
                out = _six_step_kernel_mlx(
                    Y_mx, W, Phi_T, J_c_T, J_s_T, H_cc_T, n_energy_inv
                )

            A_row, chi2, num_c, den_c, RS_s, ScSs, den_s, hcc_ss = out
            mx.eval(A_row, chi2, num_c, den_c, RS_s, ScSs, den_s, hcc_ss)

            A_np = np.array(A_row)
            chi2_np = np.array(chi2)
            num_c_np = np.array(num_c)
            den_c_np = np.array(den_c)
            RS_s_np = np.array(RS_s)
            ScSs_np = np.array(ScSs)
            den_s_np = np.array(den_s)
            hcc_ss_np = np.array(hcc_ss)
            amplitudes = A_np.T

        else:
            W_np = np.asarray(W)
            Phi_np = np.asarray(Phi)
            J_c_np = np.asarray(J_c)
            J_s_np = np.asarray(J_sigma)
            H_cc_np = np.asarray(H_cc)

            A_row = Y @ W_np
            R = Y - A_row @ Phi_np.T
            S_c = A_row @ J_c_np.T
            S_s = A_row @ J_s_np.T
            H = A_row @ H_cc_np.T

            num_c_np = np.sum(R * S_c, axis=1)
            den_c_np = np.sum(S_c * S_c, axis=1)
            RS_s_np = np.sum(R * S_s, axis=1)
            ScSs_np = np.sum(S_c * S_s, axis=1)
            den_s_np = np.sum(S_s * S_s, axis=1)
            hcc_ss_np = np.sum(H * S_s, axis=1)
            chi2_np = np.sum(R * R, axis=1) / n_energy
            amplitudes = A_row.T

        # CPU post-processing: sequential δE → δσ with Hessian correction
        safe_den_c = np.where(den_c_np > 1e-20, den_c_np, 1.0)
        energy_shifts = np.where(den_c_np > 1e-20, num_c_np / safe_den_c, 0.0)

        # Step 2.5 + Step 3: Hessian-corrected sigma
        # δσ = (RS_s - δE·ScSs - ½δE²·hcc_ss) / den_s
        correction = 0.5 * energy_shifts * energy_shifts * hcc_ss_np
        num_s_corrected = RS_s_np - energy_shifts * ScSs_np - correction
        safe_den_s = np.where(den_s_np > 1e-20, den_s_np, 1.0)
        sigma_shifts = np.where(den_s_np > 1e-20, num_s_corrected / safe_den_s, 0.0)

        return (amplitudes, chi2_np,
                energy_shifts.astype(np.float32),
                sigma_shifts.astype(np.float32))

    def process_rowmajor_extended_3param(
        self,
        Y: Any,
        element: str,
        orbital: str,
        energy: np.ndarray,
        peak_config: dict,
        amplitudes_only: bool = False,
        n_steps: int = 4,
        amp_correction: str = 'full',
    ) -> FitResult:
        """Run 4-step or 6-step shift+width corrected pipeline for row-major data.

        Steps 1-3: amplitude, δE, δσ recovery (GPU kernel)
        Step 2.5 (6-step only): Hessian ½δE²·H_cc decontamination of δσ
        Step 3.5 (6-step, n_comp≥2): Augmented correction M = I + δσ·C_σ + δE·C_c
        Step 4: Amplitude correction for σ-coupling bias (CPU)

        Args:
            n_steps: 4 for standard pipeline, 6 for Hessian-corrected pipeline.
            amp_correction: 'full' (default M correction), 'sigma_only' (M uses C_σ only,
                ignoring C_c even for 6-step), 'skip' (no amplitude correction),
                'refit' (re-estimate amp using corrected basis Φ+δE·J_c+δσ·J_σ).

        Returns:
            FitResult with energy_shifts and sigma_shifts populated
        """
        timing = {}
        n_spectra = Y.shape[0]
        n_comp = len(peak_config["centers"])

        if isinstance(Y, np.ndarray):
            if Y.dtype != np.float32:
                Y = Y.astype(np.float32)
            if not Y.flags['C_CONTIGUOUS']:
                t0 = time.perf_counter()
                Y = np.ascontiguousarray(Y)
                timing["contiguous_copy"] = time.perf_counter() - t0

        energy = np.asarray(energy, dtype=np.float32)

        t0 = time.perf_counter()
        # DS Jacobian/Hessian not yet implemented
        ls_types = peak_config.get("lineshape_types")
        if ls_types is not None and "ds" in ls_types:
            raise ValueError(
                "DS components are not supported with extended SVD solvers "
                "(Jacobian/Hessian not implemented for DS). "
                "Use amp-only mode (process/process_rowmajor) instead."
            )
        if n_steps == 6:
            Wt, Phi, J_c, J_sigma, H_cc = self.cache.get_or_create_with_hessian(
                element=element, orbital=orbital, energy=energy,
                centers=peak_config["centers"], sigmas=peak_config["sigmas"],
                gamma=peak_config.get("gamma", 0.2), use_mlx=self.use_mlx,
            )
        else:
            Wt, Phi, J_c, J_sigma = self.cache.get_or_create_with_full_jacobian(
                element=element, orbital=orbital, energy=energy,
                centers=peak_config["centers"], sigmas=peak_config["sigmas"],
                gamma=peak_config.get("gamma", 0.2), use_mlx=self.use_mlx,
            )
            H_cc = None
        W = Wt.T
        timing["cache_lookup"] = time.perf_counter() - t0

        if self.use_mlx and HAS_MLX and not isinstance(Y, mx.array):
            t0 = time.perf_counter()
            Y = mx.array(Y)
            timing["mlx_convert"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if n_steps == 6:
            amplitudes, chi2, energy_shifts, sigma_shifts = \
                self.stage1_shift_width_rowmajor_6step(Y, W, Phi, J_c, J_sigma, H_cc)
        else:
            amplitudes, chi2, energy_shifts, sigma_shifts = \
                self.stage1_shift_width_rowmajor(Y, W, Phi, J_c, J_sigma)
        timing["stage1"] = time.perf_counter() - t0
        timing["stage1_rate"] = n_spectra / timing["stage1"] if timing["stage1"] > 0 else float("inf")

        # Step 3.5 + Step 4: Amplitude correction.
        # amp_correction='full':  M = I + δσ·C_σ + δE·C_c  (6-step, n_comp≥2)
        # amp_correction='sigma_only': M = I + δσ·C_σ  (ignore C_c even for 6-step)
        # amp_correction='skip':  No amplitude correction at all
        # amp_correction='refit': Re-estimate amp using corrected basis Φ+δE·J_c+δσ·J_σ
        if amp_correction == 'refit':
            # Refit: construct per-spectrum corrected basis and re-estimate amplitudes
            # Φ'(E) = Φ(E) + δE·J_c(E) + δσ·J_σ(E)  per spectrum
            # Then â = argmin ||Y - â·Φ'||² → least squares
            t0_refit = time.perf_counter()
            Phi_np = np.asarray(Phi) if not isinstance(Phi, np.ndarray) else Phi
            J_c_np = np.asarray(J_c) if not isinstance(J_c, np.ndarray) else J_c
            J_s_np = np.asarray(J_sigma) if not isinstance(J_sigma, np.ndarray) else J_sigma
            Y_np = np.asarray(Y) if not isinstance(Y, np.ndarray) else Y

            if n_comp == 1:
                # Vectorized 1-comp refit: â = Σ(Y·Φ') / Σ(Φ'·Φ')
                # Phi shape is (n_energy, 1), need (1, n_energy) for broadcasting
                phi_row = Phi_np[:, 0:1].T     # (1, n_energy)
                jc_row = J_c_np[:, 0:1].T      # (1, n_energy)
                js_row = J_s_np[:, 0:1].T      # (1, n_energy)
                # Φ'(E) = Phi + dE*J_c + ds*J_s, broadcast (n_spectra, n_energy)
                Phi_corr = (phi_row
                            + energy_shifts[:, np.newaxis] * jc_row
                            + sigma_shifts[:, np.newaxis] * js_row)
                num = np.sum(Y_np * Phi_corr, axis=1)
                den = np.sum(Phi_corr * Phi_corr, axis=1)
                den = np.where(den > 1e-20, den, 1.0)
                amplitudes = (num / den)[np.newaxis, :]
            else:
                # n_comp ≥ 2: per-spectrum Gram matrix solve
                # Phi shape: (n_energy, n_comp), transpose to (n_comp, n_energy)
                phi_t = Phi_np.T    # (n_comp, n_energy)
                jc_t = J_c_np.T    # (n_comp, n_energy)
                js_t = J_s_np.T    # (n_comp, n_energy)
                # Φ'_j(E) = phi_t[j,:] + dE*jc_t[j,:] + ds*js_t[j,:]
                # Broadcast: (n_spectra, n_comp, n_energy)
                Phi_corr = (phi_t[np.newaxis, :, :]
                            + energy_shifts[:, np.newaxis, np.newaxis] * jc_t[np.newaxis, :, :]
                            + sigma_shifts[:, np.newaxis, np.newaxis] * js_t[np.newaxis, :, :])
                # Gram: (n_spectra, n_comp, n_comp)
                Gram = np.einsum('sce,sde->scd', Phi_corr, Phi_corr)
                rhs = np.einsum('se,sce->sc', Y_np, Phi_corr)
                amp_refit = np.linalg.solve(Gram, rhs)  # (n_spectra, n_comp)
                amplitudes = amp_refit.T  # (n_comp, n_spectra)
            timing["refit"] = time.perf_counter() - t0_refit
        elif amp_correction != 'skip':
            Phi_np = np.asarray(Phi) if not isinstance(Phi, np.ndarray) else Phi
            J_s_np = np.asarray(J_sigma) if not isinstance(J_sigma, np.ndarray) else J_sigma
            AtA = Phi_np.T @ Phi_np
            C_sigma = np.linalg.solve(AtA, Phi_np.T @ J_s_np)  # (n_comp, n_comp)

            # Step 3.5: compute C_c for augmented correction (6-step, n_comp≥2)
            C_c = None
            if amp_correction == 'full' and n_steps == 6 and n_comp >= 2:
                J_c_np = np.asarray(J_c) if not isinstance(J_c, np.ndarray) else J_c
                C_c = np.linalg.solve(AtA, Phi_np.T @ J_c_np)  # (n_comp, n_comp)

            if n_comp == 1:
                correction = 1.0 + sigma_shifts * C_sigma[0, 0]
                correction = np.where(np.abs(correction) > 1e-6, correction, 1.0)
                amplitudes = amplitudes / correction[np.newaxis, :]
            elif n_comp == 2:
                ds = sigma_shifts
                # Build M = I + δσ·C_σ + δE·C_c
                M00 = 1 + ds * C_sigma[0, 0]
                M01 = ds * C_sigma[0, 1]
                M10 = ds * C_sigma[1, 0]
                M11 = 1 + ds * C_sigma[1, 1]
                if C_c is not None:
                    de = energy_shifts
                    M00 = M00 + de * C_c[0, 0]
                    M01 = M01 + de * C_c[0, 1]
                    M10 = M10 + de * C_c[1, 0]
                    M11 = M11 + de * C_c[1, 1]
                d = M00 * M11 - M01 * M10
                d = np.where(np.abs(d) > 1e-12, d, 1.0)
                inv_d = 1.0 / d
                a0 = amplitudes[0, :].copy()
                a1 = amplitudes[1, :].copy()
                amplitudes[0, :] = (M11 * a0 - M01 * a1) * inv_d
                amplitudes[1, :] = (-M10 * a0 + M00 * a1) * inv_d
            else:
                I_mat = np.eye(n_comp, dtype=np.float32)
                M = (I_mat[np.newaxis, :, :]
                     + sigma_shifts[:, np.newaxis, np.newaxis] * C_sigma[np.newaxis, :, :])
                if C_c is not None:
                    M = M + energy_shifts[:, np.newaxis, np.newaxis] * C_c[np.newaxis, :, :]
                a_col = amplitudes.T[:, :, np.newaxis]
                a_corr = np.linalg.solve(M, a_col)
                amplitudes = a_corr[:, :, 0].T

        if amplitudes_only:
            timing["total"] = sum(
                v for k, v in timing.items()
                if k not in ["stage1_rate"] and isinstance(v, (int, float))
            )
            return FitResult(
                amplitudes=amplitudes, chi2=chi2,
                anomaly_mask=np.empty(0, dtype=bool),
                timing=timing,
                energy_shifts=energy_shifts,
                sigma_shifts=sigma_shifts,
            )

        t0 = time.perf_counter()
        anomaly_mask = self.detect_anomalies(chi2)
        timing["anomaly_detection"] = time.perf_counter() - t0

        centers_out = np.tile(peak_config["centers"][:, np.newaxis], (1, n_spectra))
        centers_out += energy_shifts[np.newaxis, :]

        sigmas_arr = np.asarray(peak_config["sigmas"], dtype=np.float32)
        sigmas_out = np.tile(sigmas_arr[:, np.newaxis], (1, n_spectra))
        sigmas_out += sigma_shifts[np.newaxis, :]

        gamma = peak_config.get("gamma", 0.2)
        if np.isscalar(gamma):
            gammas_out = np.full((n_comp, n_spectra), gamma)
        else:
            gammas_out = np.tile(np.asarray(gamma)[:, np.newaxis], (1, n_spectra))

        timing["total"] = sum(
            v for k, v in timing.items()
            if k not in ["stage1_rate"] and isinstance(v, (int, float))
        )

        return FitResult(
            amplitudes=amplitudes, chi2=chi2, anomaly_mask=anomaly_mask,
            timing=timing, centers=centers_out, sigmas=sigmas_out,
            gammas=gammas_out,
            energy_shifts=energy_shifts, sigma_shifts=sigma_shifts,
        )

    def process_rowmajor_extended(
        self,
        Y: Any,
        element: str,
        orbital: str,
        energy: np.ndarray,
        peak_config: dict,
        amplitudes_only: bool = False,
    ) -> FitResult:
        """
        Run shift-corrected pipeline for row-major data.

        Uses 2-step residual projection: standard linear fit followed by
        per-spectrum global energy shift estimation via Jacobian projection.
        No nonlinear optimization (Stage 2) needed.

        Performance: ~same as standard Stage 1 (3 matmuls + element-wise ops).

        Args:
            Y: Spectra (n_spectra, n_energy) C-contiguous
            element: Element symbol
            orbital: Orbital name
            energy: Energy axis (n_energy,)
            peak_config: Peak configuration dict
            amplitudes_only: Skip anomaly detection, output params

        Returns:
            FitResult with energy_shifts field populated
        """
        timing = {}
        n_spectra = Y.shape[0]
        n_comp = len(peak_config["centers"])

        if isinstance(Y, np.ndarray):
            if Y.dtype != np.float32:
                Y = Y.astype(np.float32)
            if not Y.flags['C_CONTIGUOUS']:
                t0 = time.perf_counter()
                Y = np.ascontiguousarray(Y)
                timing["contiguous_copy"] = time.perf_counter() - t0

        energy = np.asarray(energy, dtype=np.float32)

        # Get cached weight matrix + Jacobian
        # DS Jacobian not yet implemented
        ls_types = peak_config.get("lineshape_types")
        if ls_types is not None and "ds" in ls_types:
            raise ValueError(
                "DS components are not supported with shift correction "
                "(Jacobian not implemented for DS). "
                "Use amp-only mode (process/process_rowmajor) instead."
            )
        t0 = time.perf_counter()
        Wt, Phi, J_c = self.cache.get_or_create_with_jacobian(
            element=element,
            orbital=orbital,
            energy=energy,
            centers=peak_config["centers"],
            sigmas=peak_config["sigmas"],
            gamma=peak_config.get("gamma", 0.2),
            use_mlx=self.use_mlx,
        )
        if self.use_mlx and HAS_MLX:
            W = Wt.T
        else:
            W = Wt.T
        timing["cache_lookup"] = time.perf_counter() - t0

        # Convert Y to MLX
        if self.use_mlx and HAS_MLX and not isinstance(Y, mx.array):
            t0 = time.perf_counter()
            Y = mx.array(Y)
            timing["mlx_convert"] = time.perf_counter() - t0

        # 2-step shift estimation
        t0 = time.perf_counter()
        amplitudes, chi2, energy_shifts = self.stage1_shift_rowmajor(
            Y, W, Phi, J_c
        )
        timing["stage1"] = time.perf_counter() - t0
        timing["stage1_rate"] = n_spectra / timing["stage1"] if timing["stage1"] > 0 else float("inf")

        if amplitudes_only:
            timing["total"] = sum(
                v for k, v in timing.items()
                if k not in ["stage1_rate"] and isinstance(v, (int, float))
            )
            return FitResult(
                amplitudes=amplitudes,
                chi2=chi2,
                anomaly_mask=np.empty(0, dtype=bool),
                timing=timing,
                energy_shifts=energy_shifts,
            )

        # Anomaly detection (diagnostics only)
        t0 = time.perf_counter()
        anomaly_mask = self.detect_anomalies(chi2)
        n_anomaly = np.sum(anomaly_mask)
        timing["anomaly_detection"] = time.perf_counter() - t0
        timing["n_anomaly"] = int(n_anomaly)
        timing["anomaly_ratio"] = n_anomaly / n_spectra

        # Output parameter arrays: centers adjusted by δE
        centers_out = np.tile(peak_config["centers"][:, np.newaxis], (1, n_spectra))
        centers_out += energy_shifts[np.newaxis, :]

        sigmas_out = np.tile(peak_config["sigmas"][:, np.newaxis], (1, n_spectra))
        gamma = peak_config.get("gamma", 0.2)
        if np.isscalar(gamma):
            gammas_out = np.full((n_comp, n_spectra), gamma)
        else:
            gammas_out = np.tile(np.asarray(gamma)[:, np.newaxis], (1, n_spectra))

        timing["total"] = sum(
            v for k, v in timing.items()
            if k not in ["stage1_rate", "n_anomaly", "anomaly_ratio"]
            and isinstance(v, (int, float))
        )

        return FitResult(
            amplitudes=amplitudes,
            chi2=chi2,
            anomaly_mask=anomaly_mask,
            timing=timing,
            centers=centers_out,
            sigmas=sigmas_out,
            gammas=gammas_out,
            energy_shifts=energy_shifts,
        )

    def process_rowmajor_fast(
        self,
        Y: Any,  # np.ndarray or mx.array, shape (n_spectra, n_energy)
        element: str,
        orbital: str,
        energy: np.ndarray,
        peak_config: dict,
        skip_stage2: bool = True,
    ) -> FitResult:
        """
        Optimized row-major pipeline with minimal NumPy<->MLX conversions.

        Key optimizations:
        1. MLX conversion done once at entry
        2. Chi2 stays in MLX until threshold comparison
        3. Sampled median for anomaly detection
        4. Fused threshold comparison in MLX

        Performance target: 25M+ spec/s (2x faster than process_rowmajor)

        Args:
            Y: Spectra matrix (n_spectra, n_energy) - C-contiguous!
            element: Element symbol
            orbital: Orbital name
            energy: Energy axis
            peak_config: Peak configuration dict
            skip_stage2: Skip Stage 2 refinement (default True for speed)

        Returns:
            FitResult
        """
        if not HAS_MLX or not self.use_mlx:
            # Fall back to regular method if MLX not available
            return self.process_rowmajor(Y, element, orbital, energy, peak_config)

        timing = {}
        n_spectra = Y.shape[0]
        n_energy = Y.shape[1]

        # === Single MLX conversion at entry ===
        t0 = time.perf_counter()
        if isinstance(Y, np.ndarray):
            if Y.dtype != np.float32:
                Y = Y.astype(np.float32)
            if not Y.flags['C_CONTIGUOUS']:
                Y = np.ascontiguousarray(Y)
            Y_mx = mx.array(Y)
        elif isinstance(Y, mx.array):
            Y_mx = Y
        else:
            Y_mx = mx.array(np.asarray(Y, dtype=np.float32))
        timing["mlx_convert"] = time.perf_counter() - t0

        energy = np.asarray(energy, dtype=np.float32)

        # === Get cached weight matrix ===
        t0 = time.perf_counter()
        Wt, Phi = self.cache.get_or_create(
            element=element,
            orbital=orbital,
            energy=energy,
            centers=peak_config["centers"],
            sigmas=peak_config["sigmas"],
            gamma=peak_config.get("gamma", 0.2),
            use_mlx=True,
            alphas=peak_config.get("alphas"),
            lineshape_types=peak_config.get("lineshape_types"),
        )
        W = Wt.T  # MLX transpose is a view
        timing["cache_lookup"] = time.perf_counter() - t0

        # === Stage 1: Core computation (stays in MLX) ===
        t0 = time.perf_counter()

        # A = Y @ W (all in MLX)
        A_row = Y_mx @ W  # (n_spectra, n_components)

        # Full chi2 calculation in MLX
        Y_fit = A_row @ Phi.T  # (n_spectra, n_energy)
        chi2_mx = mx.sum((Y_mx - Y_fit)**2, axis=1) / n_energy

        # Single eval for both A and chi2
        mx.eval(A_row, chi2_mx)
        timing["stage1_compute"] = time.perf_counter() - t0

        # === Anomaly detection with sampled threshold ===
        t0 = time.perf_counter()

        # Sample chi2 for threshold calculation (small NumPy operation)
        sample_size = min(10000, n_spectra)
        if n_spectra > sample_size:
            # Sample indices - use mx.take for MLX array indexing
            sample_idx = np.random.choice(n_spectra, sample_size, replace=False)
            sample_idx_mx = mx.array(sample_idx.astype(np.int32))
            chi2_sample = np.array(mx.take(chi2_mx, sample_idx_mx))
        else:
            chi2_sample = np.array(chi2_mx)

        # Compute threshold from sample
        median_chi2 = np.median(chi2_sample)
        mad = np.median(np.abs(chi2_sample - median_chi2))
        if mad < 1e-10:
            threshold = float('inf')
        else:
            threshold = median_chi2 + self.chi2_threshold * 1.4826 * mad

        # Fused comparison in MLX
        anomaly_mask_mx = chi2_mx > threshold
        mx.eval(anomaly_mask_mx)

        timing["anomaly_detection"] = time.perf_counter() - t0

        # === Convert results to NumPy (single conversion) ===
        t0 = time.perf_counter()
        amplitudes = np.array(A_row).T  # (n_components, n_spectra)
        chi2 = np.array(chi2_mx)
        anomaly_mask = np.array(anomaly_mask_mx)
        timing["result_convert"] = time.perf_counter() - t0

        n_anomaly = np.sum(anomaly_mask)
        timing["n_anomaly"] = int(n_anomaly)
        timing["anomaly_ratio"] = n_anomaly / n_spectra

        # === Initialize output arrays ===
        n_comp = len(peak_config["centers"])
        centers_out = np.tile(peak_config["centers"][:, np.newaxis], (1, n_spectra))
        sigmas_out = np.tile(peak_config["sigmas"][:, np.newaxis], (1, n_spectra))
        gamma = peak_config.get("gamma", 0.2)
        if np.isscalar(gamma):
            gammas_out = np.full((n_comp, n_spectra), gamma)
        else:
            gammas_out = np.tile(np.asarray(gamma)[:, np.newaxis], (1, n_spectra))

        # === Stage 2 (optional) ===
        refined = None
        if not skip_stage2 and self.enable_stage2 and n_anomaly > 0:
            t0 = time.perf_counter()
            anomaly_indices = np.where(anomaly_mask)[0]

            # Convert Y to column-major for Stage 2
            Y_colmajor = np.array(Y_mx).T  # (n_energy, n_spectra)

            refined = self.stage2_refinement(
                Y=Y_colmajor,
                energy=energy,
                anomaly_indices=anomaly_indices,
                initial_amplitudes=amplitudes,
                peak_config=peak_config,
            )

            if refined is not None:
                amplitudes[:, anomaly_indices] = refined.amplitudes
                centers_out[:, anomaly_indices] = refined.centers
                sigmas_out[:, anomaly_indices] = refined.sigmas
                gammas_out[:, anomaly_indices] = refined.gammas
                chi2[anomaly_indices] = refined.chi2

            timing["stage2"] = time.perf_counter() - t0
            timing["stage2_rate"] = n_anomaly / timing["stage2"] if timing["stage2"] > 0 else float("inf")

        # === Timing summary ===
        timing["total"] = sum(
            v for k, v in timing.items()
            if k not in ["stage1_rate", "stage2_rate", "n_anomaly", "anomaly_ratio", "stage2_converged", "stage2_backend"]
            and isinstance(v, (int, float))
        )
        timing["stage1_rate"] = n_spectra / timing["total"] if timing["total"] > 0 else float("inf")

        return FitResult(
            amplitudes=amplitudes,
            chi2=chi2,
            anomaly_mask=anomaly_mask,
            refined_params=refined,
            timing=timing,
            centers=centers_out,
            sigmas=sigmas_out,
            gammas=gammas_out,
        )

    def process_rowmajor_dictionary_hybrid(
        self,
        Y: Any,
        element: str,
        orbital: str,
        energy: np.ndarray,
        peak_config: dict,
        amplitudes_only: bool = False,
        dict_cache: Any | None = None,
        dE_range: tuple = (-2.0, 2.0),
        dsigma_range: tuple = (-0.3, 0.3),
        dgamma_range: tuple = (-0.05, 0.05),
        solver: str = "hybrid",
        chunk_bytes: int | None = None,
        use_fp16_scores: bool = False,
    ) -> FitResult:
        """Dictionary hybrid pipeline for wide-range parameter recovery.

        Phase 1: Dictionary lookup (matmul + argmax) for coarse δE, δσ.
        Phase 2: Residual projection from nearest grid point (4-step or 6-step).

        This eliminates the Taylor expansion limit (δ/σ < 0.35) of the
        pure 4-step pipeline by having the dictionary absorb nonlinearity
        on a discrete grid.

        Args:
            Y: Spectra (n_spectra, n_energy) float32
            element: Element symbol
            orbital: Orbital name
            energy: Energy axis
            peak_config: dict with 'centers', 'sigmas', 'gamma'
            amplitudes_only: Skip anomaly detection if True
            dict_cache: Pre-built DictionaryCache (built on first call if None)
            dE_range: δE search range for dictionary
            dsigma_range: δσ search range for dictionary
            dgamma_range: δγ search range for dictionary (only used by solver="3d")
            solver: Solver variant — "hybrid" (CPU sort), "sorted" (MLX sort-and-batch),
                    "6step" (Hessian-corrected, better δσ),
                    "adaptive" (Taylor fast-path + hybrid for outliers),
                    "3d" (3D dictionary over δE×δσ×δγ with parabola),
                    "2stage" (Dict3D global γ calibration → Dict2D precision fit)

        Returns:
            FitResult with energy_shifts, sigma_shifts populated
        """
        from .dictionary_solver import (
            adaptive_solve,
            build_dictionary,
            solve_dict2d_parabola,
            solve_hybrid,
            solve_hybrid_6step_sorted,
            solve_hybrid_sorted,
        )

        timing = {}
        n_spectra = Y.shape[0]

        if isinstance(Y, np.ndarray):
            if Y.dtype != np.float32:
                Y = Y.astype(np.float32)
            if not Y.flags['C_CONTIGUOUS']:
                Y = np.ascontiguousarray(Y)

        energy = np.asarray(energy, dtype=np.float32)
        centers = np.asarray(peak_config["centers"], dtype=np.float32)
        sigmas_arr = np.asarray(peak_config["sigmas"], dtype=np.float32)
        gamma = peak_config.get("gamma", 0.2)

        # Ensure NumPy input for hybrid solver
        if HAS_MLX and isinstance(Y, mx.array):
            Y_np = np.array(Y)
        else:
            Y_np = Y

        gamma_shifts_out = None

        # 2-stage hybrid handles its own dict build + solve internally
        if solver == "2stage":
            from .dictionary_solver_3d import solve_2stage_hybrid
            t0 = time.perf_counter()
            result_2s = solve_2stage_hybrid(
                Y=Y_np, energy=energy, centers=centers, sigmas=sigmas_arr,
                gamma=gamma,
                dE_range=dE_range, dsigma_range=dsigma_range,
                dgamma_range=dgamma_range,
            )
            timing.update(result_2s.timing)
            timing["solver"] = "2stage"
            timing["global_dgamma"] = result_2s.global_dgamma
            timing["corrected_gamma"] = result_2s.corrected_gamma

            n_comp = len(centers)
            centers_out = np.tile(centers[:, np.newaxis], (1, n_spectra))
            centers_out += result_2s.energy_shifts[np.newaxis, :]
            sigmas_out = np.tile(sigmas_arr[:, np.newaxis], (1, n_spectra))
            sigmas_out += result_2s.sigma_shifts[np.newaxis, :]
            gammas_out = np.full((n_comp, n_spectra), result_2s.corrected_gamma)

            return FitResult(
                amplitudes=result_2s.amplitudes,
                chi2=result_2s.chi2,
                anomaly_mask=self.detect_anomalies(result_2s.chi2) if not amplitudes_only else np.empty(0, dtype=bool),
                timing=timing,
                centers=centers_out if not amplitudes_only else None,
                sigmas=sigmas_out if not amplitudes_only else None,
                gammas=gammas_out if not amplitudes_only else None,
                energy_shifts=result_2s.energy_shifts,
                sigma_shifts=result_2s.sigma_shifts,
                gamma_shifts=result_2s.dgamma_per_spectrum,
            )

        # 3D solver has its own dictionary; other solvers share 2D dict
        if solver == "3d":
            from .dictionary_solver_3d import get_or_build_dictionary_3d, solve_dict3d_parabola
            if dict_cache is None:
                t0 = time.perf_counter()
                dict_cache = get_or_build_dictionary_3d(
                    energy=energy,
                    centers=centers,
                    sigmas=sigmas_arr,
                    gamma=gamma,
                    dE_range=dE_range,
                    dsigma_range=dsigma_range,
                    dgamma_range=dgamma_range,
                )
                timing["dict_build"] = time.perf_counter() - t0
        else:
            # Build 2D dictionary if not provided
            if dict_cache is None:
                t0 = time.perf_counter()
                dict_cache = build_dictionary(
                    energy=energy,
                    centers=centers,
                    sigmas=sigmas_arr,
                    gamma=gamma,
                    dE_range=dE_range,
                    dsigma_range=dsigma_range,
                    include_hessian=(solver == "6step"),
                )
                timing["dict_build"] = time.perf_counter() - t0

        # Resolve default chunk_bytes from memory detection
        from .dictionary_solver import _PHASE1_MAX_BYTES
        _eff_chunk_bytes = chunk_bytes if chunk_bytes is not None else _PHASE1_MAX_BYTES

        t0 = time.perf_counter()
        if solver == "3d":
            amplitudes, chi2, energy_shifts, sigma_shifts, gamma_shifts_out, dict_indices = \
                solve_dict3d_parabola(
                    Y_np, dict_cache,
                    chunk_bytes=_eff_chunk_bytes,
                    use_fp16_scores=use_fp16_scores,
                )
        elif solver == "6step":
            amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices = \
                solve_hybrid_6step_sorted(Y_np, dict_cache)
        elif solver == "parabola":
            amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices = \
                solve_dict2d_parabola(
                    Y_np, dict_cache,
                    chunk_bytes=_eff_chunk_bytes,
                    use_fp16_scores=use_fp16_scores,
                )
        elif solver == "sorted":
            amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices = \
                solve_hybrid_sorted(Y_np, dict_cache)
        elif solver == "adaptive":
            amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices = \
                adaptive_solve(Y_np, dict_cache)
        else:
            amplitudes, chi2, energy_shifts, sigma_shifts, dict_indices = \
                solve_hybrid(Y_np, dict_cache, use_mlx=self.use_mlx)
        timing["hybrid_solve"] = time.perf_counter() - t0
        timing["hybrid_rate"] = n_spectra / timing["hybrid_solve"] if timing["hybrid_solve"] > 0 else float("inf")
        timing["solver"] = solver

        if amplitudes_only:
            timing["total"] = sum(
                v for k, v in timing.items()
                if k not in ["hybrid_rate", "solver"] and isinstance(v, (int, float))
            )
            return FitResult(
                amplitudes=amplitudes, chi2=chi2,
                anomaly_mask=np.empty(0, dtype=bool),
                timing=timing,
                energy_shifts=energy_shifts,
                sigma_shifts=sigma_shifts,
                gamma_shifts=gamma_shifts_out,
            )

        t0 = time.perf_counter()
        anomaly_mask = self.detect_anomalies(chi2)
        timing["anomaly_detection"] = time.perf_counter() - t0

        n_comp = len(centers)
        centers_out = np.tile(centers[:, np.newaxis], (1, n_spectra))
        centers_out += energy_shifts[np.newaxis, :]

        sigmas_out = np.tile(sigmas_arr[:, np.newaxis], (1, n_spectra))
        sigmas_out += sigma_shifts[np.newaxis, :]

        if np.isscalar(gamma):
            gammas_out = np.full((n_comp, n_spectra), gamma)
        else:
            gammas_out = np.tile(np.asarray(gamma)[:, np.newaxis], (1, n_spectra))

        if gamma_shifts_out is not None:
            gammas_out += gamma_shifts_out[np.newaxis, :]

        timing["total"] = sum(
            v for k, v in timing.items()
            if k not in ["hybrid_rate", "solver"] and isinstance(v, (int, float))
        )

        return FitResult(
            amplitudes=amplitudes, chi2=chi2, anomaly_mask=anomaly_mask,
            timing=timing,
            centers=centers_out, sigmas=sigmas_out, gammas=gammas_out,
            energy_shifts=energy_shifts, sigma_shifts=sigma_shifts,
            gamma_shifts=gamma_shifts_out,
        )

    def process_multipeak(
        self,
        Y: Any,
        config: Any,
        grid_type: str = "uniform",
        n_iterations: int = 3,
        chunk_size: int | None = None,
        parabola_dE: bool = True,
        parabola_ds: bool = False,
        dicts: Any = None,
    ) -> Any:
        """Multi-peak pipeline via Alternating Projection.

        Builds per-component 2D dictionaries and solves using
        alternating projection with orthogonal deflation.

        Args:
            Y: Spectra (n_spectra, n_energy) float32.
            config: MultiPeakConfig with peak definitions and energy axis.
            grid_type: Grid type for dictionary axes.
            n_iterations: Alternating projection iterations.
            chunk_size: Spectra per chunk (auto if None).
            parabola_dE: Apply parabola to δE axis.
            parabola_ds: Apply parabola to δσ axis.
            dicts: Pre-built per-component dictionaries (skip build if provided).

        Returns:
            MultiPeakResult with per-component amplitudes, shifts, chi2.
        """
        from .multipeak_solver import process_multipeak

        if isinstance(Y, np.ndarray):
            if Y.dtype != np.float32:
                Y = Y.astype(np.float32)
            if not Y.flags["C_CONTIGUOUS"]:
                Y = np.ascontiguousarray(Y)
        elif HAS_MLX and isinstance(Y, mx.array):
            Y = np.array(Y).astype(np.float32)

        return process_multipeak(
            Y, config, grid_type, n_iterations, chunk_size,
            parabola_dE, parabola_ds, dicts=dicts,
        )

    def process_multipeak_2stage(
        self,
        Y: Any,
        config: Any,
        grid_type: str = "uniform",
        n_iterations: int = 3,
        chunk_size: int | None = None,
        parabola_dE: bool = True,
        parabola_ds: bool = False,
        dgamma_range: tuple = (-0.05, 0.05),
        n_calibration: int | None = None,
        aggregation: str = "mean",
    ) -> Any:
        """Multi-peak 2-stage pipeline: Dict2D alternating + Dict3D γ cal.

        Stage 1a: Standard alternating projection on calibration subset.
        Stage 1b: Per-component γ estimation via deflation + Dict3D.
        Stage 2: Full solve with corrected γ values.

        Args:
            Y: Spectra (n_spectra, n_energy) float32.
            config: MultiPeakConfig with peak definitions and energy axis.
            grid_type: Grid type for dictionary axes.
            n_iterations: Alternating projection iterations.
            chunk_size: Spectra per chunk for Stage 2 (auto if None).
            parabola_dE: Apply parabola to δE axis.
            parabola_ds: Apply parabola to δσ axis.
            dgamma_range: δγ search range for Dict3D.
            n_calibration: Spectra for Stage 1 (None = min(2000, N)).
            aggregation: "mean" or "median" for global Δγ.

        Returns:
            MultiPeak2StageResult with per-component γ correction.
        """
        from .multipeak_solver import solve_multipeak_2stage

        if isinstance(Y, np.ndarray):
            if Y.dtype != np.float32:
                Y = Y.astype(np.float32)
            if not Y.flags["C_CONTIGUOUS"]:
                Y = np.ascontiguousarray(Y)
        elif HAS_MLX and isinstance(Y, mx.array):
            Y = np.array(Y).astype(np.float32)

        return solve_multipeak_2stage(
            Y, config, grid_type, n_iterations, chunk_size,
            parabola_dE, parabola_ds,
            dgamma_range=dgamma_range,
            n_calibration=n_calibration,
            aggregation=aggregation,
        )
