"""
GVRT 1-Billion Spectra Round-Trip Benchmark
=========================================================

Verifies that Dict2D+Parabola can process 10^9 spectra in practical
time and memory on M3 Max (128GB unified memory).

Pipeline per chunk:
  1. Generate random GT parameters (amp, δE, δσ)
  2. Build Taylor spectra: amp × (Φ + δE·J_c + δσ·J_σ) + background
  3. Optionally add Poisson noise (MLX GPU)
  4. solve_dict2d_parabola → recovered (amp, δE, δσ)
  5. Write to numpy memmap

Usage:
  # Part 1: Scale-up benchmark
  uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_1b --mode scale

  # Part 2: Full 1B run
  uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_1b --mode full

  # Part 3: PSNR verification (after full run)
  uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_1b --mode psnr

  # Custom N and chunk
  uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_1b --mode full -N 100_000_000 --chunk 5_000_000
"""

import argparse
import gc
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from toyomacro.voigtfit.dictionary_solver import build_dictionary, solve_dict2d_parabola
from toyomacro.voigtfit.param_encoder import (
    C1S_SINGLE_PRESET,
    FWHM_TO_SIGMA,
    SinglePeakEncoder,
)
from toyomacro.voigtfit.weight_cache import WeightMatrixCache

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("outputs/gvrt_1b")
GT_PATH = OUTPUT_DIR / "gt_params.npy"      # (N, 3) float32 memmap
REC_PATH = OUTPUT_DIR / "rec_params.npy"     # (N, 3) float32 memmap

PRESET = C1S_SINGLE_PRESET
AMPLITUDE_SCALE = 1000.0
BG_FRACTION = 0.001

# Default noise: moderate Poisson
DEFAULT_NOISE_LEVEL = 1000.0  # SNR ≈ 10


# ============================================================================
# Helpers
# ============================================================================

def _get_rss_gb() -> float:
    """Current RSS in GB (macOS/Linux)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / (1024 ** 3)  # macOS: bytes


def _psnr(gt: np.ndarray, rec: np.ndarray) -> float:
    """PSNR in parameter space.  MAX = dynamic range of gt."""
    mse = np.mean((gt.astype(np.float64) - rec.astype(np.float64)) ** 2)
    if mse < 1e-30:
        return float('inf')
    max_val = float(gt.max() - gt.min())
    if max_val < 1e-30:
        return float('inf')
    return 10.0 * np.log10(max_val ** 2 / mse)


@dataclass
class ChunkResult:
    """Stats from a single chunk."""
    chunk_id: int
    n_spectra: int
    gen_time: float
    fit_time: float
    total_time: float
    throughput: float  # fit only, spec/s
    rss_gb: float


@dataclass
class BottleneckResult:
    """Fine-grained timing for a single chunk."""
    n_spectra: int
    t_gt_gen: float       # numpy random GT params
    t_mlx_upload: float   # numpy → mx.array transfer
    t_voigt_compute: float  # Humlicek w4 element-wise ops (before eval)
    t_mlx_eval: float     # mx.eval() sync
    t_np_download: float  # mx → numpy conversion
    t_noise: float        # Poisson noise
    t_dict_solve: float   # solve_dict2d_parabola
    t_reconstruct: float  # column_stack + memmap write
    rss_gb: float

    @property
    def total(self) -> float:
        return (self.t_gt_gen + self.t_mlx_upload + self.t_voigt_compute
                + self.t_mlx_eval + self.t_np_download + self.t_noise
                + self.t_dict_solve + self.t_reconstruct)

    def as_dict(self) -> dict:
        return {
            'gt_gen': self.t_gt_gen,
            'mlx_upload': self.t_mlx_upload,
            'voigt_compute': self.t_voigt_compute,
            'mlx_eval': self.t_mlx_eval,
            'np_download': self.t_np_download,
            'noise': self.t_noise,
            'dict_solve': self.t_dict_solve,
            'reconstruct': self.t_reconstruct,
        }


# ============================================================================
# Core benchmark
# ============================================================================

class GVRTBillionBenchmark:
    """Giga Voigt Round Trip: 10^9 spectra end-to-end."""

    def __init__(
        self,
        n_total: int = 1_000_000_000,
        chunk_size: int = 10_000_000,
        noise_level: float = DEFAULT_NOISE_LEVEL,
        output_dir: Path = OUTPUT_DIR,
        seed: int = 42,
        solver_chunk_bytes: int | None = None,
        use_fp16_scores: bool = False,
    ):
        self.n_total = n_total
        self.chunk_size = chunk_size
        self.noise_level = noise_level
        self.output_dir = output_dir
        self.seed = seed
        self.solver_chunk_bytes = solver_chunk_bytes
        self.use_fp16_scores = use_fp16_scores

        self.preset = PRESET
        self.encoder = SinglePeakEncoder(self.preset)
        self.energy = self.preset.energy  # (n_energy,)
        self.n_energy = len(self.energy)

        center = self.preset.element.binding_energy
        sigma = self.preset.element.sigma
        gamma = self.preset.element.gamma

        self.peak_config = {
            'centers': np.array([center], dtype=np.float32),
            'sigmas': np.array([sigma], dtype=np.float32),
            'gamma': gamma,
        }

        # Build Phi, J_c, J_sigma for Taylor spectra generation
        self.cache = WeightMatrixCache()
        self.Phi, self.J_c, self.J_sigma = self.cache.build_basis_voigt_with_full_jacobian(
            self.energy,
            self.peak_config['centers'],
            self.peak_config['sigmas'],
            gamma,
        )

        # Pre-compute Taylor basis for CPU generation
        # Phi (nE, n_comp), J_c (nE, n_comp), J_sigma (nE, n_comp)
        # → _taylor_phi (nE,), _taylor_jacobian (2, nE) for BLAS matmul
        self._taylor_phi = self.Phi[:, 0].astype(np.float32)          # (nE,)
        self._taylor_jacobian = np.vstack([
            self.J_c[:, 0], self.J_sigma[:, 0],
        ]).astype(np.float32)  # (2, nE)

        # Build dictionary (once, reused across all chunks)
        self._dict_cache = None

    def _build_dict(self):
        """Build or return cached dictionary."""
        if self._dict_cache is not None:
            return self._dict_cache

        sigma = float(self.peak_config['sigmas'][0])
        # δE range covers encoder range + margin
        dE_lo = self.preset.shift_range.min_val - 0.2
        dE_hi = self.preset.shift_range.max_val + 0.2

        # δσ range from FWHM range
        sigma_nom = self.preset.element.sigma
        ds_min = self.preset.fwhm_range.min_val * FWHM_TO_SIGMA - sigma_nom
        ds_max = self.preset.fwhm_range.max_val * FWHM_TO_SIGMA - sigma_nom
        ds_lo = ds_min - 0.02
        ds_hi = ds_max + 0.02

        print(f"Building Dict2D: δE=[{dE_lo:.2f},{dE_hi:.2f}], "
              f"δσ=[{ds_lo:.3f},{ds_hi:.3f}]")

        t0 = time.perf_counter()
        self._dict_cache = build_dictionary(
            energy=self.energy,
            centers=self.peak_config['centers'],
            sigmas=self.peak_config['sigmas'],
            gamma=self.peak_config['gamma'],
            dE_range=(dE_lo, dE_hi),
            dE_step=0.1 * sigma,
            dsigma_range=(ds_lo, ds_hi),
            dsigma_step=0.02 * sigma,
        )
        dt = time.perf_counter() - t0
        dc = self._dict_cache
        mem_mb = dc.D.nbytes / 1e6
        print(f"  Dict: {dc.n_dE}×{dc.n_dsigma} = {dc.n_dict} entries "
              f"({mem_mb:.1f} MB, {dt:.2f}s)")

        # Pre-build profile matrix for CPU dict-based generation
        # Phi_per_grid[i] is (n_energy, n_comp) — actual Voigt profile at grid point
        # For single peak (n_comp=1): extract column 0 to get (n_energy,)
        self._profile_matrix = np.array(
            [dc.Phi_per_grid[i][:, 0] for i in range(dc.n_dict)],
            dtype=np.float32,
        )  # (n_dict, n_energy)
        print(f"  Profile matrix: {self._profile_matrix.shape}, "
              f"{self._profile_matrix.nbytes / 1e6:.1f} MB")

        return self._dict_cache

    def _generate_gt_params(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate ground-truth (amp, dE, dsigma) for n spectra.

        Returns: (n, 3) float32
        """
        amp = rng.uniform(
            self.preset.amplitude_range.min_val,
            self.preset.amplitude_range.max_val,
            size=n,
        ).astype(np.float32)

        dE = rng.uniform(
            self.preset.shift_range.min_val,
            self.preset.shift_range.max_val,
            size=n,
        ).astype(np.float32)

        fwhm_g = rng.uniform(
            self.preset.fwhm_range.min_val,
            self.preset.fwhm_range.max_val,
            size=n,
        ).astype(np.float32)
        dsigma = (fwhm_g * FWHM_TO_SIGMA - self.preset.element.sigma).astype(np.float32)

        return np.column_stack([amp, dE, dsigma])

    def _generate_spectra(self, gt: np.ndarray) -> np.ndarray:
        """Exact Voigt spectra from GT params.

        MLX: Humlicek w4 rational approximation (real arithmetic only).
        CPU fallback: scipy.special.wofz.
        Sub-chunks to limit GPU memory usage.

        Returns: (n, n_energy) float32
        """
        amp = gt[:, 0]
        dE = gt[:, 1]
        dsigma = gt[:, 2]

        center = float(self.preset.element.binding_energy)
        sigma_nom = float(self.preset.element.sigma)
        gamma = float(self.preset.element.gamma)

        scaled_amp = amp * AMPLITUDE_SCALE
        n = len(amp)

        if HAS_MLX:
            # Sub-chunk for GPU memory: (n, n_energy) float32 × ~10 arrays ≈ 40n×n_E bytes
            # Target 8 GB → n_max = 8e9 / (40 * n_energy)
            n_max = int(8e9 / (40 * self.n_energy))
            n_max = max(n_max, 10_000)
        else:
            # scipy wofz: complex128 (n, n_energy) ≈ 16n×n_E bytes
            n_max = int(4e9 / (16 * self.n_energy))
            n_max = max(n_max, 1000)

        if n <= n_max:
            return self._generate_spectra_one(
                scaled_amp, dE, dsigma, center, sigma_nom, gamma,
            )

        parts = []
        for s in range(0, n, n_max):
            e = min(s + n_max, n)
            part = self._generate_spectra_one(
                scaled_amp[s:e], dE[s:e], dsigma[s:e],
                center, sigma_nom, gamma,
            )
            parts.append(part)
        return np.concatenate(parts, axis=0)

    def _generate_spectra_one(
        self, scaled_amp, dE, dsigma, center, sigma_nom, gamma,
    ) -> np.ndarray:
        """Generate exact Voigt spectra for one batch."""
        if HAS_MLX:
            return self._gen_voigt_mlx(scaled_amp, dE, dsigma, center, sigma_nom, gamma)
        return self._gen_voigt_cpu(scaled_amp, dE, dsigma, center, sigma_nom, gamma)

    def _gen_voigt_cpu(
        self, scaled_amp, dE, dsigma, center, sigma_nom, gamma,
    ) -> np.ndarray:
        """Exact Voigt via scipy wofz (CPU)."""
        from scipy import special as sps

        energy = self.energy.astype(np.float64)
        SQRT2 = np.sqrt(2.0)
        SQRT2PI = np.sqrt(2.0 * np.pi)

        centers = (center + dE.astype(np.float64))[:, np.newaxis]
        sigmas = (sigma_nom + dsigma.astype(np.float64))[:, np.newaxis]

        z = ((energy[np.newaxis, :] - centers) + 1j * gamma) / (sigmas * SQRT2)
        profiles = np.real(sps.wofz(z)) / (sigmas * SQRT2PI)

        spectra = scaled_amp[:, np.newaxis].astype(np.float64) * profiles
        bg = scaled_amp.astype(np.float64) * BG_FRACTION
        spectra += bg[:, np.newaxis]

        return spectra.astype(np.float32)

    def _gen_voigt_mlx(
        self, scaled_amp, dE, dsigma, center, sigma_nom, gamma,
    ) -> np.ndarray:
        """Exact Voigt via Humlicek w4 on MLX (real arithmetic, no complex).

        The Humlicek (1982) Region 3 rational approximation gives
        Re[w(x+iy)] to ~1e-4 accuracy for y > 0.  For XPS Voigt with
        γ/σ ≈ 0.3 and |x| < 15, this covers all practical cases.

        All computation uses real-valued MLX arrays (no complex type needed).
        """
        energy_mx = mx.array(self.energy)    # (nE,) float32
        amp_mx = mx.array(scaled_amp)        # (n,) float32
        dE_mx = mx.array(dE)                 # (n,)
        ds_mx = mx.array(dsigma)             # (n,)

        SQRT2 = float(np.sqrt(2.0))
        SQRT2PI = float(np.sqrt(2.0 * np.pi))

        sigmas_mx = sigma_nom + ds_mx        # (n,)
        inv_s2 = 1.0 / (sigmas_mx * SQRT2)  # (n,)

        # x, y for Faddeeva: z = x + iy
        # x = (energy - center - dE) / (sigma * sqrt(2))
        # y = gamma / (sigma * sqrt(2))
        x = (energy_mx[None, :] - center - dE_mx[:, None]) * inv_s2[:, None]  # (n, nE)
        y = gamma * inv_s2                                                      # (n,)
        y_2d = y[:, None]  # (n, 1) broadcast

        # Humlicek w4 Region 3 using real arithmetic:
        # t = y - ix  → t_re = y, t_im = -x
        # Numerator: P(t) = a0 + t*(a1 + t*(a2 + t*(a3 + t*a4)))
        # Denominator: Q(t) = b0 + t*(b1 + t*(b2 + t*(b3 + t*(b4 + t))))
        # We need Re[P(t)/Q(t)] = (P_re*Q_re + P_im*Q_im) / (Q_re² + Q_im²)

        # Coefficients
        a0, a1, a2, a3, a4 = 16.4955, 20.20933, 11.96482, 3.778987, 0.5642236
        b0, b1, b2, b3, b4 = 16.4955, 38.82363, 39.27121, 21.69274, 6.699398

        # Horner evaluation for P(t) = a0 + t*(a1 + t*(a2 + t*(a3 + t*a4)))
        # With t = (y, -x), complex multiply: (a,b)*(c,d) = (ac-bd, ad+bc)
        # Start from innermost: h = a4 (real)
        h_re = mx.full(x.shape, a4, dtype=x.dtype)   # (n, nE)
        h_im = mx.zeros(x.shape, dtype=x.dtype)

        # h = a3 + t*h = a3 + (y*h_re - (-x)*h_im, y*h_im + (-x)*h_re)
        #              = a3 + (y*h_re + x*h_im, y*h_im - x*h_re)
        new_re = a3 + y_2d * h_re + x * h_im
        new_im = y_2d * h_im - x * h_re
        h_re, h_im = new_re, new_im

        new_re = a2 + y_2d * h_re + x * h_im
        new_im = y_2d * h_im - x * h_re
        h_re, h_im = new_re, new_im

        new_re = a1 + y_2d * h_re + x * h_im
        new_im = y_2d * h_im - x * h_re
        h_re, h_im = new_re, new_im

        P_re = a0 + y_2d * h_re + x * h_im
        P_im = y_2d * h_im - x * h_re

        # Horner for Q(t) = b0 + t*(b1 + t*(b2 + t*(b3 + t*(b4 + t))))
        # innermost: h = b4 + t = (b4 + y, -x)
        h_re = b4 + y_2d
        h_im = -x

        new_re = b3 + y_2d * h_re + x * h_im
        new_im = y_2d * h_im - x * h_re
        h_re, h_im = new_re, new_im

        new_re = b2 + y_2d * h_re + x * h_im
        new_im = y_2d * h_im - x * h_re
        h_re, h_im = new_re, new_im

        new_re = b1 + y_2d * h_re + x * h_im
        new_im = y_2d * h_im - x * h_re
        h_re, h_im = new_re, new_im

        Q_re = b0 + y_2d * h_re + x * h_im
        Q_im = y_2d * h_im - x * h_re

        # Re[P/Q] = (P_re*Q_re + P_im*Q_im) / (Q_re² + Q_im²)
        denom = Q_re * Q_re + Q_im * Q_im
        voigt = (P_re * Q_re + P_im * Q_im) / denom  # (n, nE)

        # V = voigt / (sigma * sqrt(2*pi))
        profiles = voigt / (sigmas_mx[:, None] * SQRT2PI)

        spectra = amp_mx[:, None] * profiles + (amp_mx * BG_FRACTION)[:, None]

        mx.eval(spectra)
        return np.array(spectra, dtype=np.float32)

    def _gen_voigt_mlx_lazy(
        self, scaled_amp, dE, dsigma, center, sigma_nom, gamma,
    ):
        """Humlicek w4 on MLX — returns lazy mx.array (NOT evaluated).

        Same as _gen_voigt_mlx but skips mx.eval() and np conversion.
        Returns: mx.array (n, nE) — lazy graph, call mx.eval() separately.
        """
        energy_mx = mx.array(self.energy)
        amp_mx = mx.array(np.asarray(scaled_amp, dtype=np.float32))
        dE_mx = mx.array(np.asarray(dE, dtype=np.float32))
        ds_mx = mx.array(np.asarray(dsigma, dtype=np.float32))

        SQRT2 = float(np.sqrt(2.0))
        SQRT2PI = float(np.sqrt(2.0 * np.pi))

        sigmas_mx = sigma_nom + ds_mx
        inv_s2 = 1.0 / (sigmas_mx * SQRT2)

        x = (energy_mx[None, :] - center - dE_mx[:, None]) * inv_s2[:, None]
        y = gamma * inv_s2
        y_2d = y[:, None]

        a0, a1, a2, a3, a4 = 16.4955, 20.20933, 11.96482, 3.778987, 0.5642236
        b0, b1, b2, b3, b4 = 16.4955, 38.82363, 39.27121, 21.69274, 6.699398

        h_re = mx.full(x.shape, a4, dtype=x.dtype)
        h_im = mx.zeros(x.shape, dtype=x.dtype)
        new_re = a3 + y_2d * h_re + x * h_im
        new_im = y_2d * h_im - x * h_re
        h_re, h_im = new_re, new_im
        new_re = a2 + y_2d * h_re + x * h_im
        new_im = y_2d * h_im - x * h_re
        h_re, h_im = new_re, new_im
        new_re = a1 + y_2d * h_re + x * h_im
        new_im = y_2d * h_im - x * h_re
        h_re, h_im = new_re, new_im
        P_re = a0 + y_2d * h_re + x * h_im
        P_im = y_2d * h_im - x * h_re

        h_re = b4 + y_2d
        h_im = -x
        new_re = b3 + y_2d * h_re + x * h_im
        new_im = y_2d * h_im - x * h_re
        h_re, h_im = new_re, new_im
        new_re = b2 + y_2d * h_re + x * h_im
        new_im = y_2d * h_im - x * h_re
        h_re, h_im = new_re, new_im
        new_re = b1 + y_2d * h_re + x * h_im
        new_im = y_2d * h_im - x * h_re
        h_re, h_im = new_re, new_im
        Q_re = b0 + y_2d * h_re + x * h_im
        Q_im = y_2d * h_im - x * h_re

        denom = Q_re * Q_re + Q_im * Q_im
        voigt = (P_re * Q_re + P_im * Q_im) / denom
        profiles = voigt / (sigmas_mx[:, None] * SQRT2PI)
        spectra = amp_mx[:, None] * profiles + (amp_mx * BG_FRACTION)[:, None]
        return spectra  # lazy — do NOT eval here

    def _process_chunk_bottleneck(
        self,
        gt: np.ndarray,
        dict_cache,
        global_max: float,
    ) -> BottleneckResult:
        """Fine-grained timing of each pipeline stage."""
        n = gt.shape[0]
        center = float(self.preset.element.binding_energy)
        sigma_nom = float(self.preset.element.sigma)
        gamma = float(self.preset.element.gamma)

        # 1) GT gen (already done outside, but measure column prep)
        t0 = time.perf_counter()
        amp = gt[:, 0]
        dE = gt[:, 1]
        dsigma = gt[:, 2]
        scaled_amp = amp * AMPLITUDE_SCALE
        t_gt = time.perf_counter() - t0

        # 2) MLX upload (numpy → mx.array)
        t0 = time.perf_counter()
        energy_mx = mx.array(self.energy)
        amp_mx = mx.array(scaled_amp.astype(np.float32))
        dE_mx = mx.array(dE.astype(np.float32))
        ds_mx = mx.array(dsigma.astype(np.float32))
        mx.eval(energy_mx, amp_mx, dE_mx, ds_mx)
        t_upload = time.perf_counter() - t0

        # 3) Voigt compute (lazy graph construction + GPU dispatch)
        t0 = time.perf_counter()
        spectra_mx = self._gen_voigt_mlx_lazy(
            scaled_amp, dE, dsigma, center, sigma_nom, gamma,
        )
        t_compute = time.perf_counter() - t0

        # 4) mx.eval() — GPU synchronization
        t0 = time.perf_counter()
        mx.eval(spectra_mx)
        t_eval = time.perf_counter() - t0

        # 5) np download (mx → numpy)
        t0 = time.perf_counter()
        spectra = np.array(spectra_mx, dtype=np.float32)
        del spectra_mx
        t_download = time.perf_counter() - t0

        # 6) Noise
        t0 = time.perf_counter()
        spectra = self._add_noise(spectra, global_max)
        t_noise = time.perf_counter() - t0

        # 7) Dict solve
        t0 = time.perf_counter()
        amps, chi2, dE_rec, ds_rec = self._solve_subchuked(spectra, dict_cache)
        if HAS_MLX:
            mx.eval()
        t_solve = time.perf_counter() - t0

        # 8) Reconstruct + cleanup
        t0 = time.perf_counter()
        rec = np.column_stack([
            amps[0, :] / AMPLITUDE_SCALE,
            dE_rec,
            ds_rec,
        ]).astype(np.float32)
        del spectra, amps, chi2, rec
        gc.collect()
        t_recon = time.perf_counter() - t0

        return BottleneckResult(
            n_spectra=n,
            t_gt_gen=t_gt,
            t_mlx_upload=t_upload,
            t_voigt_compute=t_compute,
            t_mlx_eval=t_eval,
            t_np_download=t_download,
            t_noise=t_noise,
            t_dict_solve=t_solve,
            t_reconstruct=t_recon,
            rss_gb=_get_rss_gb(),
        )

    def _add_noise(self, spectra: np.ndarray, global_max: float) -> np.ndarray:
        """Add Poisson noise on GPU (MLX)."""
        if self.noise_level <= 0:
            return spectra
        if HAS_MLX:
            from toyomacro.voigtfit.spectra_generator import add_poisson_noise_mlx_fused
            spectra_mx = mx.array(spectra)
            noisy_mx = add_poisson_noise_mlx_fused(
                spectra_mx, self.noise_level, global_max=global_max,
            )
            mx.eval(noisy_mx)
            return np.array(noisy_mx, dtype=np.float32)
        else:
            from toyomacro.voigtfit.spectra_generator import add_poisson_noise
            return add_poisson_noise(spectra, self.noise_level, global_max=global_max)

    def _generate_and_noise_fused(
        self, gt: np.ndarray, global_max: float,
    ) -> np.ndarray:
        """Generate spectra + add noise in a single GPU pass.

        Fuses Voigt generation (lazy) with Poisson noise (compiled) into
        one mx.eval() call, eliminating the intermediate GPU→CPU→GPU
        round-trip between generation and noise addition.
        """
        amp = gt[:, 0]
        dE = gt[:, 1]
        dsigma = gt[:, 2]
        scaled_amp = amp * AMPLITUDE_SCALE
        center = float(self.preset.element.binding_energy)
        sigma_nom = float(self.preset.element.sigma)
        gamma = float(self.preset.element.gamma)

        n = len(amp)
        # Sub-chunk for GPU memory
        n_max = int(8e9 / (40 * self.n_energy)) if HAS_MLX else int(4e9 / (16 * self.n_energy))
        n_max = max(n_max, 10_000)

        if n <= n_max:
            return self._gen_and_noise_one(
                scaled_amp, dE, dsigma, center, sigma_nom, gamma, global_max,
            )

        parts = []
        for s in range(0, n, n_max):
            e = min(s + n_max, n)
            part = self._gen_and_noise_one(
                scaled_amp[s:e], dE[s:e], dsigma[s:e],
                center, sigma_nom, gamma, global_max,
            )
            parts.append(part)
        return np.concatenate(parts, axis=0)

    def _gen_and_noise_one(
        self, scaled_amp, dE, dsigma, center, sigma_nom, gamma, global_max,
    ) -> np.ndarray:
        """Single batch: Voigt gen (lazy) → noise (compiled) → one eval."""
        if not HAS_MLX:
            spectra = self._gen_voigt_cpu(scaled_amp, dE, dsigma, center, sigma_nom, gamma)
            from toyomacro.voigtfit.spectra_generator import add_poisson_noise
            return add_poisson_noise(spectra, self.noise_level, global_max=global_max)

        # Build lazy Voigt graph
        spectra_mx = self._gen_voigt_mlx_lazy(
            scaled_amp, dE, dsigma, center, sigma_nom, gamma,
        )
        # Chain noise (also lazy via mx.compile)
        if self.noise_level > 0:
            from toyomacro.voigtfit.spectra_generator import add_poisson_noise_mlx_fused
            spectra_mx = add_poisson_noise_mlx_fused(
                spectra_mx, self.noise_level, global_max=global_max,
            )
        # Single GPU sync
        mx.eval(spectra_mx)
        result = np.array(spectra_mx, dtype=np.float32)
        del spectra_mx
        return result

    def _solve_subchuked(
        self,
        spectra: np.ndarray,
        dict_cache,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run solve_dict2d_parabola with sub-chunking to avoid Metal OOM.

        The score matrix (n_spectra, n_dict) can exceed Metal's ~80GB buffer
        limit for large n_spectra. Sub-chunk to stay within GPU memory.

        Returns: (amps, chi2, dE, dsigma) concatenated.
        """
        n = spectra.shape[0]
        n_dict = dict_cache.D.shape[1]

        # Metal max buffer ~80 GB; score matrix = n × n_dict × 4 bytes
        # Leave headroom: target 50 GB for scores
        max_sub = int(50e9 / (n_dict * 4))
        max_sub = max(max_sub, 10_000)  # floor

        solve_kwargs = {}
        if self.solver_chunk_bytes is not None:
            solve_kwargs['chunk_bytes'] = self.solver_chunk_bytes
        if self.use_fp16_scores:
            solve_kwargs['use_fp16_scores'] = True

        if n <= max_sub:
            amps, chi2, dE, ds, _ = solve_dict2d_parabola(
                spectra, dict_cache, **solve_kwargs,
            )
            return amps, chi2, dE, ds

        # Sub-chunk
        all_amps, all_chi2, all_dE, all_ds = [], [], [], []
        for s in range(0, n, max_sub):
            e = min(s + max_sub, n)
            a, c, de, ds_, _ = solve_dict2d_parabola(
                spectra[s:e], dict_cache, **solve_kwargs,
            )
            all_amps.append(a)
            all_chi2.append(c)
            all_dE.append(de)
            all_ds.append(ds_)
        return (
            np.concatenate(all_amps, axis=1),
            np.concatenate(all_chi2),
            np.concatenate(all_dE),
            np.concatenate(all_ds),
        )

    def _process_chunk_fused(
        self,
        chunk_id: int,
        gt: np.ndarray,
        dict_cache,
        global_max: float,
    ) -> tuple[np.ndarray, ChunkResult]:
        """Process one chunk with fused gen+noise (single mx.eval).

        Same as _process_chunk but eliminates the intermediate GPU→CPU→GPU
        round-trip between Voigt generation and noise addition.
        """
        n = gt.shape[0]

        t_gen0 = time.perf_counter()
        spectra = self._generate_and_noise_fused(gt, global_max)
        t_gen = time.perf_counter() - t_gen0

        t_fit0 = time.perf_counter()
        amps, chi2, dE_rec, ds_rec = self._solve_subchuked(spectra, dict_cache)
        if HAS_MLX:
            mx.eval()
        t_fit = time.perf_counter() - t_fit0

        rec = np.column_stack([
            amps[0, :] / AMPLITUDE_SCALE,
            dE_rec,
            ds_rec,
        ]).astype(np.float32)

        del spectra, amps, chi2
        gc.collect()

        stats = ChunkResult(
            chunk_id=chunk_id,
            n_spectra=n,
            gen_time=t_gen,
            fit_time=t_fit,
            total_time=t_gen + t_fit,
            throughput=n / t_fit if t_fit > 0 else float('inf'),
            rss_gb=_get_rss_gb(),
        )
        return rec, stats

    def _process_chunk(
        self,
        chunk_id: int,
        gt: np.ndarray,
        dict_cache,
        global_max: float,
    ) -> tuple[np.ndarray, ChunkResult]:
        """Process one chunk: gen → noise → solve → return recovered params."""
        n = gt.shape[0]

        # Generate spectra
        t_gen0 = time.perf_counter()
        spectra = self._generate_spectra(gt)
        spectra = self._add_noise(spectra, global_max)
        t_gen = time.perf_counter() - t_gen0

        # Solve (with sub-chunking for Metal OOM avoidance)
        t_fit0 = time.perf_counter()
        amps, chi2, dE_rec, ds_rec = self._solve_subchuked(spectra, dict_cache)
        if HAS_MLX:
            mx.eval()  # ensure GPU work completes
        t_fit = time.perf_counter() - t_fit0

        # Reconstruct params: (n, 3) = (amp, dE, dsigma)
        rec = np.column_stack([
            amps[0, :] / AMPLITUDE_SCALE,  # back to [0,1] range
            dE_rec,
            ds_rec,
        ]).astype(np.float32)

        del spectra, amps, chi2
        gc.collect()

        stats = ChunkResult(
            chunk_id=chunk_id,
            n_spectra=n,
            gen_time=t_gen,
            fit_time=t_fit,
            total_time=t_gen + t_fit,
            throughput=n / t_fit if t_fit > 0 else float('inf'),
            rss_gb=_get_rss_gb(),
        )
        return rec, stats

    # ------------------------------------------------------------------
    # Part 1: Scale-up benchmark
    # ------------------------------------------------------------------

    def run_scale_benchmark(self):
        """Measure throughput at increasing scales."""
        scales = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
        dict_cache = self._build_dict()
        rng = np.random.default_rng(self.seed)

        global_max = AMPLITUDE_SCALE * (1.0 + BG_FRACTION)

        print()
        print("=" * 80)
        print("GVRT Scale-Up Benchmark (Dict2D + Parabola)")
        print("=" * 80)
        print(f"{'N':>12} {'Gen (s)':>10} {'Fit (s)':>10} {'Total (s)':>10} "
              f"{'Fit M/s':>10} {'RSS GB':>8}")
        print("-" * 80)

        for N in scales:
            gt = self._generate_gt_params(N, rng)
            rec, stats = self._process_chunk(0, gt, dict_cache, global_max)

            # Quick PSNR check
            psnr_amp = _psnr(gt[:, 0], rec[:, 0])
            psnr_dE = _psnr(gt[:, 1], rec[:, 1])
            psnr_ds = _psnr(gt[:, 2], rec[:, 2])

            print(f"{N:>12,} {stats.gen_time:>10.3f} {stats.fit_time:>10.3f} "
                  f"{stats.total_time:>10.3f} {stats.throughput/1e6:>10.2f} "
                  f"{stats.rss_gb:>8.2f}  "
                  f"PSNR: amp={psnr_amp:.1f} δE={psnr_dE:.1f} δσ={psnr_ds:.1f}")

            del gt, rec
            gc.collect()

        print("=" * 80)

    # ------------------------------------------------------------------
    # Part 2: Full billion-spectra run
    # ------------------------------------------------------------------

    def run_full(self):
        """Process N_total spectra in chunks, writing to memmap."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dict_cache = self._build_dict()
        rng = np.random.default_rng(self.seed)

        n_chunks = (self.n_total + self.chunk_size - 1) // self.chunk_size
        global_max = AMPLITUDE_SCALE * (1.0 + BG_FRACTION)

        # Create memmap files
        gt_mm = np.memmap(
            str(GT_PATH), dtype='float32', mode='w+',
            shape=(self.n_total, 3),
        )
        rec_mm = np.memmap(
            str(REC_PATH), dtype='float32', mode='w+',
            shape=(self.n_total, 3),
        )

        print(flush=True)
        print("=" * 80, flush=True)
        print(f"GVRT Full Run: {self.n_total:,} spectra, "
              f"{n_chunks} chunks × {self.chunk_size:,}", flush=True)
        print(f"GT  → {GT_PATH}  ({self.n_total * 3 * 4 / 1e9:.1f} GB)", flush=True)
        print(f"REC → {REC_PATH} ({self.n_total * 3 * 4 / 1e9:.1f} GB)", flush=True)
        print("=" * 80, flush=True)
        print(f"{'Chunk':>6} {'N':>12} {'Gen':>8} {'Fit':>8} {'M/s':>8} "
              f"{'RSS GB':>8} {'Cumul %':>8} {'ETA':>8}", flush=True)
        print("-" * 80, flush=True)

        total_gen = 0.0
        total_fit = 0.0
        t_start = time.perf_counter()

        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, self.n_total)
            n = end - start

            # Generate GT params
            gt = self._generate_gt_params(n, rng)
            gt_mm[start:end] = gt

            # Process
            rec, stats = self._process_chunk(i, gt, dict_cache, global_max)
            rec_mm[start:end] = rec

            total_gen += stats.gen_time
            total_fit += stats.fit_time

            # ETA
            elapsed = time.perf_counter() - t_start
            frac = end / self.n_total
            eta = elapsed / frac * (1.0 - frac) if frac > 0 else 0

            print(f"{i+1:>6}/{n_chunks} {n:>12,} {stats.gen_time:>8.2f} "
                  f"{stats.fit_time:>8.2f} {stats.throughput/1e6:>8.2f} "
                  f"{stats.rss_gb:>8.2f} {frac*100:>7.1f}% "
                  f"{eta:>7.0f}s", flush=True)

            del gt, rec
            gc.collect()

        # Flush memmap
        gt_mm.flush()
        rec_mm.flush()
        del gt_mm, rec_mm

        total_time = time.perf_counter() - t_start
        effective_throughput = self.n_total / total_fit

        print("=" * 80)
        print(f"DONE: {self.n_total:,} spectra in {total_time:.1f}s")
        print(f"  Gen: {total_gen:.1f}s | Fit: {total_fit:.1f}s | Overhead: {total_time - total_gen - total_fit:.1f}s")
        print(f"  Effective fit throughput: {effective_throughput/1e6:.2f} M/s")
        print(f"  End-to-end throughput: {self.n_total/total_time/1e6:.2f} M/s")
        print(f"  Peak RSS: {_get_rss_gb():.2f} GB")
        print("=" * 80)

    def run_full_fused(self):
        """Process N_total spectra with fused gen+noise.

        Same pipeline as run_full but uses _process_chunk_fused which
        eliminates the GPU→CPU→GPU round-trip between Voigt gen and noise.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dict_cache = self._build_dict()
        rng = np.random.default_rng(self.seed)

        n_chunks = (self.n_total + self.chunk_size - 1) // self.chunk_size
        global_max = AMPLITUDE_SCALE * (1.0 + BG_FRACTION)

        gt_mm = np.memmap(str(GT_PATH), dtype='float32', mode='w+',
                          shape=(self.n_total, 3))
        rec_mm = np.memmap(str(REC_PATH), dtype='float32', mode='w+',
                           shape=(self.n_total, 3))

        print(flush=True)
        print("=" * 80, flush=True)
        print(f"GVRT Fused Run: {self.n_total:,} spectra, "
              f"{n_chunks} chunks × {self.chunk_size:,}", flush=True)
        print("Mode: fused gen+noise (single mx.eval), sequential", flush=True)
        print("=" * 80, flush=True)
        print(f"{'Chunk':>6} {'N':>12} {'Gen':>8} {'Fit':>8} {'M/s':>8} "
              f"{'RSS GB':>8} {'Cumul %':>8} {'ETA':>8}", flush=True)
        print("-" * 80, flush=True)

        total_gen = 0.0
        total_fit = 0.0
        t_start = time.perf_counter()

        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, self.n_total)
            n = end - start

            gt = self._generate_gt_params(n, rng)
            gt_mm[start:end] = gt

            rec, stats = self._process_chunk_fused(i, gt, dict_cache, global_max)
            rec_mm[start:end] = rec

            total_gen += stats.gen_time
            total_fit += stats.fit_time

            elapsed = time.perf_counter() - t_start
            frac = end / self.n_total
            eta = elapsed / frac * (1.0 - frac) if frac > 0 else 0

            print(f"{i+1:>6}/{n_chunks} {n:>12,} {stats.gen_time:>8.2f} "
                  f"{stats.fit_time:>8.2f} {stats.throughput/1e6:>8.2f} "
                  f"{stats.rss_gb:>8.2f} {frac*100:>7.1f}% "
                  f"{eta:>7.0f}s", flush=True)

            del gt, rec
            gc.collect()

        gt_mm.flush()
        rec_mm.flush()
        del gt_mm, rec_mm

        total_time = time.perf_counter() - t_start
        effective_throughput = self.n_total / total_fit

        print("=" * 80)
        print(f"DONE (fused): {self.n_total:,} spectra in {total_time:.1f}s")
        print(f"  Gen: {total_gen:.1f}s | Fit: {total_fit:.1f}s")
        print(f"  Effective fit throughput: {effective_throughput/1e6:.2f} M/s")
        print(f"  End-to-end throughput: {self.n_total/total_time/1e6:.2f} M/s")
        print(f"  Peak RSS: {_get_rss_gb():.2f} GB")
        print("=" * 80)

    # ------------------------------------------------------------------
    # Part 2a-v2: Pipelined full run
    # ------------------------------------------------------------------

    def _generate_spectra_dict_cpu(
        self, gt: np.ndarray, dict_cache,
    ) -> np.ndarray:
        """Dictionary-based spectra generation on CPU (pure numpy, GPU-free).

        For each spectrum, looks up the closest (δE, δσ) grid point in the
        dictionary and multiplies by amplitude. Discretization error is
        bounded by the grid step (~0.04 eV for δE, ~0.008 eV for δσ).

        This avoids the Taylor approximation breakdown at |δE/σ| > 2 while
        running entirely on CPU for pipeline overlap with GPU fitting.
        """
        amp = gt[:, 0] * AMPLITUDE_SCALE  # (n,)
        dE = gt[:, 1]                      # (n,)
        dsigma = gt[:, 2]                  # (n,)

        # Map continuous (dE, ds) to nearest grid indices
        dE_idx = np.searchsorted(dict_cache.dE_grid, dE)
        dE_idx = np.clip(dE_idx, 0, dict_cache.n_dE - 1)
        ds_idx = np.searchsorted(dict_cache.dsigma_grid, dsigma)
        ds_idx = np.clip(ds_idx, 0, dict_cache.n_dsigma - 1)

        # Snap to nearest (not just lower bound)
        # For dE
        mask = dE_idx > 0
        lower = dict_cache.dE_grid[np.maximum(dE_idx - 1, 0)]
        upper = dict_cache.dE_grid[dE_idx]
        closer_to_lower = np.abs(dE - lower) < np.abs(dE - upper)
        dE_idx = np.where(mask & closer_to_lower, dE_idx - 1, dE_idx)
        # For ds
        mask = ds_idx > 0
        lower = dict_cache.dsigma_grid[np.maximum(ds_idx - 1, 0)]
        upper = dict_cache.dsigma_grid[ds_idx]
        closer_to_lower = np.abs(dsigma - lower) < np.abs(dsigma - upper)
        ds_idx = np.where(mask & closer_to_lower, ds_idx - 1, ds_idx)

        flat_idx = dE_idx * dict_cache.n_dsigma + ds_idx

        # Gather profiles via pre-built matrix: (n_dict, n_energy)
        # Single fancy index + broadcast multiply — no Python loop
        profiles = self._profile_matrix[flat_idx]  # (n, nE)
        spectra = amp[:, np.newaxis] * profiles
        spectra += (amp * BG_FRACTION)[:, np.newaxis]
        return spectra

    def _add_noise_cpu(self, spectra: np.ndarray, global_max: float,
                       rng: np.random.Generator) -> np.ndarray:
        """CPU-only Poisson noise with template-based random generation.

        Uses K=1000 unique noise row templates sampled via fancy indexing
        (75x faster than full rng.standard_normal for 2M×101 arrays).
        In-place operations minimize memory allocation overhead.
        """
        if self.noise_level <= 0:
            return spectra

        n = spectra.shape[0]
        snr_target = 10000.0 / self.noise_level
        scale_factor = snr_target * snr_target / global_max

        # Template-based noise: K unique rows, sampled by index
        K = 1000
        templates = rng.standard_normal((K, self.n_energy), dtype=np.float32)
        idx = rng.integers(0, K, size=n)
        noise = templates[idx]  # (n, nE) via fancy indexing, fast

        # In-place noise application
        np.maximum(spectra, 0, out=spectra)
        spectra *= scale_factor                     # → scaled
        std = np.sqrt(spectra)                      # sqrt(scaled)
        noise *= std                                # noise * sqrt(scaled)
        spectra += noise                            # scaled + noise_term
        np.maximum(spectra, 0, out=spectra)         # clip negatives
        spectra *= (1.0 / scale_factor)             # scale back
        return spectra

    def run_full_pipelined(self):
        """Process N_total spectra with gen-fit pipeline overlap.

        Pipeline design (CPU-GPU split):
          Gen thread (CPU only):  Taylor spectra + numpy noise
          Main thread (GPU):      solve_dict2d_parabola (MLX)

        Taylor gen is ~30x faster than exact Voigt and runs on CPU,
        enabling true overlap with GPU fitting. Since the dict solver
        uses the same Taylor basis, this is a matched-filter test.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dict_cache = self._build_dict()
        rng = np.random.default_rng(self.seed)

        n_chunks = (self.n_total + self.chunk_size - 1) // self.chunk_size
        global_max = AMPLITUDE_SCALE * (1.0 + BG_FRACTION)

        # Create memmap files
        gt_mm = np.memmap(
            str(GT_PATH), dtype='float32', mode='w+',
            shape=(self.n_total, 3),
        )
        rec_mm = np.memmap(
            str(REC_PATH), dtype='float32', mode='w+',
            shape=(self.n_total, 3),
        )

        print(flush=True)
        print("=" * 80, flush=True)
        print(f"GVRT Pipelined Run: {self.n_total:,} spectra, "
              f"{n_chunks} chunks × {self.chunk_size:,}", flush=True)
        print("Mode: CPU Taylor gen + GPU fit pipeline overlap", flush=True)
        print(f"GT  → {GT_PATH}  ({self.n_total * 3 * 4 / 1e9:.1f} GB)", flush=True)
        print(f"REC → {REC_PATH} ({self.n_total * 3 * 4 / 1e9:.1f} GB)", flush=True)
        print("=" * 80, flush=True)
        print(f"{'Chunk':>6} {'N':>12} {'Wall':>8} {'Gen':>8} {'Fit':>8} "
              f"{'Fit M/s':>8} {'RSS GB':>8} {'Cumul %':>8} {'ETA':>8}", flush=True)
        print("-" * 80, flush=True)

        total_gen = 0.0
        total_fit = 0.0
        t_start = time.perf_counter()

        def gen_chunk_cpu(gt_chunk, seed_val):
            """CPU-only: dict-based gen + numpy noise (runs in thread)."""
            chunk_rng = np.random.default_rng(seed_val)
            spectra = self._generate_spectra_dict_cpu(gt_chunk, dict_cache)
            spectra = self._add_noise_cpu(spectra, global_max, chunk_rng)
            return spectra

        # Pipeline: gen[i+1] on CPU thread overlaps with fit[i] on GPU
        executor = ThreadPoolExecutor(max_workers=1)
        pending_future = None  # type: Optional[Future]
        pending_chunk_id = -1
        pending_gt = None
        pending_start = 0
        pending_n = 0

        for i in range(n_chunks):
            start_idx = i * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.n_total)
            n = end_idx - start_idx

            # Generate GT params (CPU, ~1ms)
            gt = self._generate_gt_params(n, rng)
            gt_mm[start_idx:end_idx] = gt

            # Generate a unique seed for this chunk's noise RNG
            noise_seed = self.seed + i + 1

            if pending_future is None:
                # First chunk: submit gen and wait
                t_gen0 = time.perf_counter()
                pending_future = executor.submit(gen_chunk_cpu, gt, noise_seed)
                pending_chunk_id = i
                pending_gt = gt
                pending_start = start_idx
                pending_n = n
                continue

            # --- Steady state: fit previous chunk, gen current chunk ---
            t_iter0 = time.perf_counter()

            # Get previous chunk's spectra (should be ready or nearly ready)
            prev_spectra = pending_future.result()
            t_gen_done = time.perf_counter()
            t_gen_wait = t_gen_done - t_iter0

            # Submit current gen in background (CPU thread)
            pending_future = executor.submit(gen_chunk_cpu, gt, noise_seed)

            # Fit previous chunk on GPU (overlaps with CPU gen)
            t_fit0 = time.perf_counter()
            amps, chi2, dE_rec, ds_rec = self._solve_subchuked(
                prev_spectra, dict_cache,
            )
            if HAS_MLX:
                mx.eval()
            t_fit = time.perf_counter() - t_fit0

            # Write results
            rec = np.column_stack([
                amps[0, :] / AMPLITUDE_SCALE,
                dE_rec,
                ds_rec,
            ]).astype(np.float32)
            rec_mm[pending_start:pending_start + pending_n] = rec

            total_gen += t_gen_wait
            total_fit += t_fit

            # Progress
            t_wall = time.perf_counter() - t_iter0
            done_count = pending_start + pending_n
            elapsed = time.perf_counter() - t_start
            frac = done_count / self.n_total
            eta = elapsed / frac * (1.0 - frac) if frac > 0 else 0
            throughput = pending_n / t_fit if t_fit > 0 else float('inf')

            print(f"{pending_chunk_id + 1:>6}/{n_chunks} {pending_n:>12,} "
                  f"{t_wall:>8.2f} {t_gen_wait:>8.2f} {t_fit:>8.2f} "
                  f"{throughput/1e6:>8.2f} {_get_rss_gb():>8.2f} "
                  f"{frac*100:>7.1f}% {eta:>7.0f}s", flush=True)

            # Update pending state
            pending_chunk_id = i
            pending_gt = gt
            pending_start = start_idx
            pending_n = n

            del prev_spectra, amps, chi2, rec

        # Process the last chunk
        if pending_future is not None:
            t_iter0 = time.perf_counter()
            last_spectra = pending_future.result()
            t_gen_wait = time.perf_counter() - t_iter0

            t_fit0 = time.perf_counter()
            amps, chi2, dE_rec, ds_rec = self._solve_subchuked(
                last_spectra, dict_cache,
            )
            if HAS_MLX:
                mx.eval()
            t_fit = time.perf_counter() - t_fit0

            rec = np.column_stack([
                amps[0, :] / AMPLITUDE_SCALE,
                dE_rec,
                ds_rec,
            ]).astype(np.float32)
            rec_mm[pending_start:pending_start + pending_n] = rec

            total_gen += t_gen_wait
            total_fit += t_fit

            t_wall = time.perf_counter() - t_iter0
            done_count = pending_start + pending_n
            elapsed = time.perf_counter() - t_start
            frac = done_count / self.n_total
            throughput = pending_n / t_fit if t_fit > 0 else float('inf')

            print(f"{pending_chunk_id + 1:>6}/{n_chunks} {pending_n:>12,} "
                  f"{t_wall:>8.2f} {t_gen_wait:>8.2f} {t_fit:>8.2f} "
                  f"{throughput/1e6:>8.2f} {_get_rss_gb():>8.2f} "
                  f"{frac*100:>7.1f}%      0s", flush=True)

            del last_spectra, amps, chi2, rec

        executor.shutdown(wait=False)
        gc.collect()

        # Flush memmap
        gt_mm.flush()
        rec_mm.flush()
        del gt_mm, rec_mm

        total_time = time.perf_counter() - t_start
        effective_throughput = self.n_total / total_fit

        print("=" * 80)
        print(f"DONE (pipelined): {self.n_total:,} spectra in {total_time:.1f}s")
        print(f"  Gen wait: {total_gen:.1f}s | Fit: {total_fit:.1f}s")
        print(f"  Effective fit throughput: {effective_throughput/1e6:.2f} M/s")
        print(f"  End-to-end throughput: {self.n_total/total_time/1e6:.2f} M/s")
        print(f"  Peak RSS: {_get_rss_gb():.2f} GB")
        print("=" * 80)

    # ------------------------------------------------------------------
    # Part 2b: Bottleneck analysis
    # ------------------------------------------------------------------

    def run_bottleneck(self, sizes=None, n_warmup: int = 1, n_repeat: int = 3):
        """Detailed per-stage timing at multiple chunk sizes.

        Runs each size n_warmup + n_repeat times, reports median of repeats.
        """
        if sizes is None:
            sizes = [100_000, 500_000, 1_000_000, 2_000_000]

        dict_cache = self._build_dict()
        rng = np.random.default_rng(self.seed)
        global_max = AMPLITUDE_SCALE * (1.0 + BG_FRACTION)

        stages = [
            'gt_gen', 'mlx_upload', 'voigt_compute', 'mlx_eval',
            'np_download', 'noise', 'dict_solve', 'reconstruct',
        ]
        stage_labels = {
            'gt_gen': 'GT gen',
            'mlx_upload': 'MLX upload',
            'voigt_compute': 'Voigt build',
            'mlx_eval': 'mx.eval()',
            'np_download': 'NP download',
            'noise': 'Noise',
            'dict_solve': 'Dict solve',
            'reconstruct': 'Reconstruct',
        }

        print(flush=True)
        print("=" * 90, flush=True)
        print("GVRT Bottleneck Analysis", flush=True)
        print(f"Warmup: {n_warmup}, Repeats: {n_repeat} (median)", flush=True)
        print("=" * 90, flush=True)

        all_results = {}  # {size: [BottleneckResult, ...]}

        for N in sizes:
            print(f"\n--- N = {N:,} ---", flush=True)
            results = []

            for trial in range(n_warmup + n_repeat):
                gt = self._generate_gt_params(N, rng)
                br = self._process_chunk_bottleneck(gt, dict_cache, global_max)
                if trial >= n_warmup:
                    results.append(br)
                label = "warmup" if trial < n_warmup else f"trial {trial - n_warmup + 1}"
                print(f"  {label}: total={br.total:.3f}s  "
                      f"voigt={br.t_voigt_compute + br.t_mlx_eval:.3f}s  "
                      f"solve={br.t_dict_solve:.3f}s", flush=True)
                del gt
                gc.collect()

            all_results[N] = results

        # Summary table
        print(flush=True)
        print("=" * 90, flush=True)
        print("Bottleneck Summary (median of repeats)", flush=True)
        print("=" * 90, flush=True)

        # Header
        header = f"{'N':>10} │ {'Total':>7}"
        for s in stages:
            header += f" │ {stage_labels[s]:>12}"
        print(header, flush=True)
        print("─" * len(header), flush=True)

        for N in sizes:
            results = all_results[N]
            # Median per stage
            medians = {}
            for s in stages:
                vals = [r.as_dict()[s] for r in results]
                medians[s] = float(np.median(vals))
            total = sum(medians.values())

            row = f"{N:>10,} │ {total:>7.3f}"
            for s in stages:
                row += f" │ {medians[s]:>12.4f}"
            print(row, flush=True)

        # Percentage breakdown
        print(flush=True)
        print("Percentage breakdown:", flush=True)
        header2 = f"{'N':>10} │ {'Total':>7}"
        for s in stages:
            header2 += f" │ {stage_labels[s]:>12}"
        print(header2, flush=True)
        print("─" * len(header2), flush=True)

        for N in sizes:
            results = all_results[N]
            medians = {}
            for s in stages:
                vals = [r.as_dict()[s] for r in results]
                medians[s] = float(np.median(vals))
            total = sum(medians.values())

            row = f"{N:>10,} │ {total:>6.3f}s"
            for s in stages:
                pct = medians[s] / total * 100 if total > 0 else 0
                row += f" │ {pct:>11.1f}%"
            print(row, flush=True)

        # Throughput (E2E and fit-only)
        print(flush=True)
        print("Throughput:", flush=True)
        print(f"{'N':>10} │ {'E2E M/s':>10} │ {'Fit M/s':>10} │ {'Gen M/s':>10} │ "
              f"{'Voigt M/s':>10} │ {'Solve M/s':>10}", flush=True)
        print("─" * 75, flush=True)

        for N in sizes:
            results = all_results[N]
            medians = {}
            for s in stages:
                vals = [r.as_dict()[s] for r in results]
                medians[s] = float(np.median(vals))
            total = sum(medians.values())
            t_voigt = medians['voigt_compute'] + medians['mlx_eval']
            t_gen = (medians['gt_gen'] + medians['mlx_upload']
                     + t_voigt + medians['np_download'])
            t_solve = medians['dict_solve']

            print(f"{N:>10,} │ {N/total/1e6:>10.2f} │ {N/t_solve/1e6:>10.2f} │ "
                  f"{N/t_gen/1e6:>10.2f} │ {N/t_voigt/1e6:>10.2f} │ "
                  f"{N/t_solve/1e6:>10.2f}", flush=True)

        print("=" * 90, flush=True)

    # ------------------------------------------------------------------
    # Part 2c: Memory-constrained simulation
    # ------------------------------------------------------------------

    def run_memory_constrained(self, simulate_gb: float = 12.0):
        """Simulate 16GB environment on 128GB machine.

        Overrides chunk_bytes via optimal_chunk_size(available_gb=simulate_gb)
        and runs a full round-trip with constrained memory parameters.

        This verifies that the memory-aware chunking from dev-log 70 works
        correctly without requiring an actual 16GB machine.

        Args:
            simulate_gb: Simulated available memory in GB (default 12.0 for
                16GB machine with ~4GB for OS/other processes).
        """
        from toyomacro.voigtfit.memory import (
            MemoryProfiler,
            get_available_memory_gb,
            optimal_chunk_size,
            optimal_dict3d_cache_size,
        )

        dict_cache = self._build_dict()
        n_dict = dict_cache.n_dict

        # Compute constrained parameters
        actual_available = get_available_memory_gb()
        dtype_bytes = 2 if self.use_fp16_scores else 4
        constrained_chunk = optimal_chunk_size(
            n_dict=n_dict, dtype_bytes=dtype_bytes,
            available_gb=simulate_gb,
        )
        constrained_cache_size = optimal_dict3d_cache_size(
            available_gb=simulate_gb,
        )

        # Override solver_chunk_bytes to match constrained chunk
        # scores matrix = chunk × n_dict × dtype_bytes
        self.solver_chunk_bytes = constrained_chunk * n_dict * dtype_bytes

        rng = np.random.default_rng(self.seed)
        n_chunks = (self.n_total + self.chunk_size - 1) // self.chunk_size
        global_max = AMPLITUDE_SCALE * (1.0 + BG_FRACTION)

        print(flush=True)
        print("=" * 80, flush=True)
        print("GVRT Memory-Constrained Simulation", flush=True)
        print(f"  Simulated available: {simulate_gb:.1f} GB "
              f"(actual: {actual_available:.1f} GB)", flush=True)
        print(f"  Solver chunk: {constrained_chunk:,} spectra "
              f"(scores: {constrained_chunk * n_dict * dtype_bytes / 1e9:.2f} GB)", flush=True)
        print(f"  Outer chunk: {self.chunk_size:,} spectra", flush=True)
        print(f"  Dict: {n_dict} entries, fp16={self.use_fp16_scores}", flush=True)
        print(f"  Dict3D cache: max_size={constrained_cache_size}", flush=True)
        print(f"  N total: {self.n_total:,} spectra, {n_chunks} chunks", flush=True)
        print("=" * 80, flush=True)
        print(f"{'Chunk':>6} {'N':>12} {'Gen':>8} {'Fit':>8} {'M/s':>8} "
              f"{'RSS GB':>8} {'Cumul %':>8} {'ETA':>8}", flush=True)
        print("-" * 80, flush=True)

        total_gen = 0.0
        total_fit = 0.0
        all_gt = []
        all_rec = []
        t_start = time.perf_counter()

        with MemoryProfiler() as prof:
            for i in range(n_chunks):
                start = i * self.chunk_size
                end = min(start + self.chunk_size, self.n_total)
                n = end - start

                gt = self._generate_gt_params(n, rng)
                rec, stats = self._process_chunk(i, gt, dict_cache, global_max)

                all_gt.append(gt)
                all_rec.append(rec)

                total_gen += stats.gen_time
                total_fit += stats.fit_time

                elapsed = time.perf_counter() - t_start
                frac = end / self.n_total
                eta = elapsed / frac * (1.0 - frac) if frac > 0 else 0

                print(f"{i+1:>6}/{n_chunks} {n:>12,} {stats.gen_time:>8.2f} "
                      f"{stats.fit_time:>8.2f} {stats.throughput/1e6:>8.2f} "
                      f"{stats.rss_gb:>8.2f} {frac*100:>7.1f}% "
                      f"{eta:>7.0f}s", flush=True)

                prof.snapshot(f"chunk_{i}")
                del gt, rec
                gc.collect()

        total_time = time.perf_counter() - t_start

        # Concatenate and compute PSNR
        gt_all = np.concatenate(all_gt, axis=0)
        rec_all = np.concatenate(all_rec, axis=0)
        del all_gt, all_rec

        psnr_amp = _psnr(gt_all[:, 0], rec_all[:, 0])
        psnr_dE = _psnr(gt_all[:, 1], rec_all[:, 1])
        psnr_ds = _psnr(gt_all[:, 2], rec_all[:, 2])

        rmse_amp = float(np.sqrt(np.mean((gt_all[:, 0] - rec_all[:, 0]) ** 2)))
        rmse_dE = float(np.sqrt(np.mean((gt_all[:, 1] - rec_all[:, 1]) ** 2)))
        rmse_ds = float(np.sqrt(np.mean((gt_all[:, 2] - rec_all[:, 2]) ** 2)))

        mem_report = prof.report()

        print("=" * 80)
        print(f"DONE (memory-constrained): {self.n_total:,} spectra in "
              f"{total_time:.1f}s")
        print(f"  Gen: {total_gen:.1f}s | Fit: {total_fit:.1f}s")
        print(f"  Effective fit throughput: "
              f"{self.n_total / total_fit / 1e6:.2f} M/s")
        print(f"  End-to-end throughput: "
              f"{self.n_total / total_time / 1e6:.2f} M/s")
        print()
        print(f"Memory ({simulate_gb:.0f}GB simulation):")
        print(f"  {mem_report}")
        print()
        print("PSNR (quality):")
        print(f"  {'Channel':<12} {'PSNR (dB)':>12} {'RMSE':>12}")
        print(f"  {'-' * 40}")
        print(f"  {'amplitude':<12} {psnr_amp:>12.1f} {rmse_amp:>12.6f}")
        print(f"  {'δE (eV)':<12} {psnr_dE:>12.1f} {rmse_dE:>12.6f}")
        print(f"  {'δσ (eV)':<12} {psnr_ds:>12.1f} {rmse_ds:>12.6f}")
        print()
        print("Reference:")
        print("  amp=53.3 dB, δE=106.5 dB, δσ=47.3 dB, 2.11 M/s")
        print("=" * 80)

        del gt_all, rec_all
        gc.collect()

    # ------------------------------------------------------------------
    # Part 3: PSNR verification
    # ------------------------------------------------------------------

    def run_psnr(self, n_sample: int = 1_000_000):
        """Sample from memmap and compute PSNR vs GT."""
        if not GT_PATH.exists() or not REC_PATH.exists():
            print(f"ERROR: Run --mode full first. Missing {GT_PATH} or {REC_PATH}")
            return

        # Detect actual file size
        file_bytes = GT_PATH.stat().st_size
        n_total = file_bytes // (3 * 4)  # float32, 3 columns
        print(f"Loading memmap: {n_total:,} spectra")

        gt_mm = np.memmap(str(GT_PATH), dtype='float32', mode='r', shape=(n_total, 3))
        rec_mm = np.memmap(str(REC_PATH), dtype='float32', mode='r', shape=(n_total, 3))

        # Sample
        rng = np.random.default_rng(123)
        n_sample = min(n_sample, n_total)
        idx = rng.choice(n_total, n_sample, replace=False)
        idx.sort()  # sequential access for memmap performance

        gt_sample = gt_mm[idx]
        rec_sample = rec_mm[idx]

        # PSNR per channel
        psnr_amp = _psnr(gt_sample[:, 0], rec_sample[:, 0])
        psnr_dE = _psnr(gt_sample[:, 1], rec_sample[:, 1])
        psnr_ds = _psnr(gt_sample[:, 2], rec_sample[:, 2])

        # RMSE per channel
        rmse_amp = float(np.sqrt(np.mean((gt_sample[:, 0] - rec_sample[:, 0]) ** 2)))
        rmse_dE = float(np.sqrt(np.mean((gt_sample[:, 1] - rec_sample[:, 1]) ** 2)))
        rmse_ds = float(np.sqrt(np.mean((gt_sample[:, 2] - rec_sample[:, 2]) ** 2)))

        # Correlation
        corr_amp = float(np.corrcoef(gt_sample[:, 0], rec_sample[:, 0])[0, 1])
        corr_dE = float(np.corrcoef(gt_sample[:, 1], rec_sample[:, 1])[0, 1])
        corr_ds = float(np.corrcoef(gt_sample[:, 2], rec_sample[:, 2])[0, 1])

        print()
        print("=" * 70)
        print(f"GVRT PSNR Verification ({n_sample:,} / {n_total:,} sampled)")
        print("=" * 70)
        print(f"{'Channel':<12} {'PSNR (dB)':>12} {'RMSE':>12} {'Corr':>10}")
        print("-" * 70)
        print(f"{'amplitude':<12} {psnr_amp:>12.1f} {rmse_amp:>12.6f} {corr_amp:>10.6f}")
        print(f"{'δE (eV)':<12} {psnr_dE:>12.1f} {rmse_dE:>12.6f} {corr_dE:>10.6f}")
        print(f"{'δσ (eV)':<12} {psnr_ds:>12.1f} {rmse_ds:>12.6f} {corr_ds:>10.6f}")
        print("-" * 70)
        print(f"{'average':<12} {(psnr_amp + psnr_dE + psnr_ds)/3:>12.1f}")
        print("=" * 70)

        # amp≈55.0, δE≈57.7, δσ≈47.1
        print()
        print("Reference:")
        print("  amp=55.0 dB, δE=57.7 dB, δσ=47.1 dB")

        del gt_mm, rec_mm


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GVRT 1-Billion Spectra Benchmark",
    )
    parser.add_argument(
        "--mode",
        choices=["scale", "full", "fused", "pipeline", "psnr", "bottleneck", "low16"],
        default="scale",
        help="Benchmark mode (low16: 16GB memory simulation)",
    )
    parser.add_argument(
        "-N", type=int, default=1_000_000_000,
        help="Total spectra count (default: 1B)",
    )
    parser.add_argument(
        "--chunk", type=int, default=2_000_000,
        help="Chunk size (default: 2M; optimal for M3 Max throughput)",
    )
    parser.add_argument(
        "--noise", type=float, default=DEFAULT_NOISE_LEVEL,
        help="Poisson noise level (0=none, 1000=moderate)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed",
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_DIR),
        help="Output directory",
    )
    parser.add_argument(
        "--simulate-memory", type=float, default=None,
        help="Simulate available memory in GB (e.g. 12.0 for 16GB machine)",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Use fp16 scores (halves memory, ~72%% argmax agreement)",
    )
    args = parser.parse_args()

    # low16 mode defaults
    if args.mode == "low16":
        if args.N == 1_000_000_000:
            args.N = 100_000_000  # 100M default for 16GB sim
        if args.chunk == 2_000_000:
            args.chunk = 500_000  # 500K default for 16GB sim
        if args.simulate_memory is None:
            args.simulate_memory = 12.0  # 16GB - 4GB OS overhead

    bench = GVRTBillionBenchmark(
        n_total=args.N,
        chunk_size=args.chunk,
        noise_level=args.noise,
        output_dir=Path(args.output),
        seed=args.seed,
        use_fp16_scores=args.fp16,
    )

    if args.mode == "scale":
        bench.run_scale_benchmark()
    elif args.mode == "full":
        bench.run_full()
    elif args.mode == "fused":
        bench.run_full_fused()
    elif args.mode == "pipeline":
        bench.run_full_pipelined()
    elif args.mode == "psnr":
        bench.run_psnr()
    elif args.mode == "bottleneck":
        bench.run_bottleneck()
    elif args.mode == "low16":
        bench.run_memory_constrained(simulate_gb=args.simulate_memory)


if __name__ == "__main__":
    main()
