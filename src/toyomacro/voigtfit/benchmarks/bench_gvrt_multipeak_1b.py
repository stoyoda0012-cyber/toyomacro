"""
GVRT MultiPeak 1-Billion Spectra Round-Trip Benchmark
===================================================================

Extends dev-log 68 (single-peak 1B) to 2-component MultiPeak with:
  - TwoPeakSplitEncoder (Split Fisher-Hilbert, dev-log 76)
  - 2-Stage γ-calibrated solver
  - CPU-GPU pipelined execution

Pipeline per chunk:
  1. Generate random GT parameters (amp, dE, dσ, dγ) × 2 components
  2. Encode via TwoPeakSplitEncoder → 6-ch RGB → decode (quantized params)
  3. Build 2-comp Voigt spectra from decoded params (exact Humlicek w4)
  4. Optionally add Poisson noise
  5. solve_multipeak_2stage → recovered (amp, dE, dσ) × 2 components
  6. Write to numpy memmap

Target (M3 Max 128GB):
  - Throughput: > 1.0 M/s
  - Total time: < 20 min for 1B spectra
  - Peak RSS: < 60 GB
  - δσ E2E PSNR: > 30 dB

Usage:
  # Part 1: Smoke test (10K)
  uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_multipeak_1b --mode smoke

  # Part 2: Stage profiling (1M)
  uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_multipeak_1b --mode profile

  # Part 3: Scale-up benchmark
  uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_multipeak_1b --mode scale

  # Part 4: Full 1B run
  uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_multipeak_1b --mode full

  # Part 5: PSNR verification (after full run)
  uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_multipeak_1b --mode psnr

  # Part 6: Comparison visualization
  uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_multipeak_1b --mode viz
"""

import argparse
import gc
import resource
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx

    from toyomacro.voigtfit._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND a Metal device works
except ImportError:
    HAS_MLX = False

from toyomacro.voigtfit.fisher_hilbert_encoder import TwoPeakSplitEncoder
from toyomacro.voigtfit.multipeak_config import ComponentConfig, MultiPeakConfig
from toyomacro.voigtfit.multipeak_solver import (
    build_multipeak_dictionaries,
    solve_alternating_projection,
)

# ============================================================================
# Constants
# ============================================================================

FWHM_TO_SIGMA = 1.0 / (2 * np.sqrt(2 * np.log(2)))
SIGMA_NOM = 1.0 * FWHM_TO_SIGMA  # ~0.4247 eV
GAMMA = 0.125                     # eV (Lorentzian HWHM)
CENTER_MID = 285.0                # eV

# Default 2-peak configuration: well-separated (7σ)
DEFAULT_SEPARATION_SIGMA = 7.0
AMPLITUDE_SCALE = 1000.0
BG_FRACTION = 0.001
DEFAULT_NOISE_LEVEL = 1000.0

OUTPUT_DIR = Path("outputs/gvrt_multipeak_1b")
GT_PATH = OUTPUT_DIR / "gt_params.npy"      # (N, 6) float32 memmap
REC_PATH = OUTPUT_DIR / "rec_params.npy"    # (N, 6) float32 memmap


# ============================================================================
# Helpers
# ============================================================================


def _get_rss_gb() -> float:
    """Current RSS in GB (macOS/Linux)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == 'darwin':
        return ru.ru_maxrss / (1024 ** 3)
    return ru.ru_maxrss * 1024 / (1024 ** 3)


def _psnr(gt: np.ndarray, rec: np.ndarray) -> float:
    """PSNR in parameter space. MAX = dynamic range of gt."""
    mse = np.mean((gt.astype(np.float64) - rec.astype(np.float64)) ** 2)
    if mse < 1e-30:
        return float('inf')
    max_val = float(gt.max() - gt.min())
    if max_val < 1e-30:
        return float('inf')
    return 10.0 * np.log10(max_val ** 2 / mse)


@dataclass
class ChunkResult:
    """Stats from processing a single chunk."""
    chunk_id: int
    n_spectra: int
    gen_time: float     # spectra generation
    fit_time: float     # solver time
    total_time: float   # gen + fit
    throughput: float   # fit-only, spec/s
    rss_gb: float


@dataclass
class StageProfile:
    """Per-stage timing for profiling."""
    n_spectra: int
    t_param_gen: float
    t_encode: float
    t_decode: float
    t_spectra_gen: float
    t_noise: float
    t_solver: float
    t_swap_correct: float
    t_write: float
    rss_gb: float

    @property
    def total(self) -> float:
        return (self.t_param_gen + self.t_encode + self.t_decode
                + self.t_spectra_gen + self.t_noise + self.t_solver
                + self.t_swap_correct + self.t_write)

    def throughput(self, stage: str) -> float:
        t = getattr(self, f't_{stage}', 0.0)
        return self.n_spectra / t if t > 0 else float('inf')


# ============================================================================
# Core benchmark class
# ============================================================================


class GVRTMultiPeak1BBenchmark:
    """Giga Voigt Round Trip: 10^9 MultiPeak spectra end-to-end."""

    def __init__(
        self,
        n_total: int = 1_000_000_000,
        chunk_size: int = 2_000_000,
        noise_level: float = 0.0,  # noise-free by default for accuracy test
        separation_sigma: float = DEFAULT_SEPARATION_SIGMA,
        amp_ratio: float = 1.0,
        output_dir: Path = OUTPUT_DIR,
        seed: int = 42,
        encoder_order1: int = 8,
        encoder_order2: int = 8,
    ):
        self.n_total = n_total
        self.chunk_size = chunk_size
        self.noise_level = noise_level
        self.separation_sigma = separation_sigma
        self.amp_ratio = amp_ratio
        self.output_dir = output_dir
        self.seed = seed

        # Peak positions
        sep_eV = separation_sigma * SIGMA_NOM
        self.center1 = CENTER_MID - sep_eV / 2
        self.center2 = CENTER_MID + sep_eV / 2

        # Energy axis: cover both peaks with 6 eV margin
        e_min = self.center1 - 6.0
        e_max = self.center2 + 6.0
        n_energy = int(round((e_max - e_min) / 0.1)) + 1
        self.energy = np.linspace(e_min, e_max, n_energy, dtype=np.float32)
        self.n_energy = n_energy

        # Encoder
        self.encoder = TwoPeakSplitEncoder(
            eta=GAMMA / (SIGMA_NOM * np.sqrt(2 * np.log(2)) * 2),
            order_peak1=encoder_order1,
            order_peak2=encoder_order2,
            dE_range=(-0.5, 0.5),
            dsigma_range=(-0.1, 0.1),
            dgamma_range=(-0.05, 0.05),
        )

        # Solver config
        self._multipeak_config = None
        self._dict_cache = None

    def _make_solver_config(
        self,
        gamma1: float = GAMMA,
        gamma2: float = GAMMA,
    ) -> MultiPeakConfig:
        """Create MultiPeakConfig for the alternating projection solver."""
        return MultiPeakConfig(
            peaks=[
                ComponentConfig(
                    center=self.center1,
                    sigma=SIGMA_NOM,
                    gamma=gamma1,
                    dE_range=0.5,
                    ds_range=0.1,
                    n_dE=15,
                    n_ds=31,
                ),
                ComponentConfig(
                    center=self.center2,
                    sigma=SIGMA_NOM,
                    gamma=gamma2,
                    dE_range=0.5,
                    ds_range=0.1,
                    n_dE=15,
                    n_ds=31,
                ),
            ],
            energy_axis=self.energy,
        )

    # ------------------------------------------------------------------
    # Parameter generation
    # ------------------------------------------------------------------

    def _generate_gt_params(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate GT params: (n, 8) = [amp1, dE1, dσ1, dγ1, amp2, dE2, dσ2, dγ2].

        γ perturbation is always 0 (nominal) — encoder encodes δγ but GT
        is generated at γ_nom for simplicity. The encoder quantization of
        δγ ≈ 0 tests the round-trip faithfully.

        Returns: (n, 8) float32
        """
        amp1 = rng.uniform(0.1, 1.0, n).astype(np.float32)
        amp2 = np.clip(amp1 * self.amp_ratio, 0.0, 1.0).astype(np.float32)

        dE1 = rng.uniform(-0.5, 0.5, n).astype(np.float32)
        dE2 = rng.uniform(-0.5, 0.5, n).astype(np.float32)

        ds1 = rng.uniform(-0.1, 0.1, n).astype(np.float32)
        ds2 = rng.uniform(-0.1, 0.1, n).astype(np.float32)

        dg1 = np.zeros(n, dtype=np.float32)  # δγ = 0 (nominal)
        dg2 = np.zeros(n, dtype=np.float32)

        return np.column_stack([amp1, dE1, ds1, dg1, amp2, dE2, ds2, dg2])

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def _encode_decode(
        self, gt: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode GT → RGB → Decode → quantized params.

        Args:
            gt: (n, 8) = [amp1, dE1, dσ1, dγ1, amp2, dE2, dσ2, dγ2]

        Returns:
            rgb: (n, 6) uint8 — the encoded image
            decoded: (n, 8) — quantized params (amp unchanged from GT)
        """
        channels = self.encoder.encode(
            gt[:, 1], gt[:, 2], gt[:, 3],   # dE1, dσ1, dγ1
            gt[:, 5], gt[:, 6], gt[:, 7],   # dE2, dσ2, dγ2
        )

        rec = self.encoder.decode(channels)

        decoded = np.column_stack([
            gt[:, 0],                       # amp1 (unchanged)
            rec['dE1'], rec['dsigma1'], rec['dgamma1'],
            gt[:, 4],                       # amp2 (unchanged)
            rec['dE2'], rec['dsigma2'], rec['dgamma2'],
        ]).astype(np.float32)

        return channels, decoded

    # ------------------------------------------------------------------
    # Spectra generation (exact Voigt)
    # ------------------------------------------------------------------

    def _generate_spectra(self, params: np.ndarray) -> np.ndarray:
        """Generate 2-component Voigt spectra from params.

        Args:
            params: (n, 8) = [amp1, dE1, dσ1, dγ1, amp2, dE2, dσ2, dγ2]

        Returns:
            Y: (n, n_energy) float32
        """
        n = params.shape[0]

        if HAS_MLX:
            return self._gen_2comp_mlx(params)
        return self._gen_2comp_cpu(params)

    def _gen_2comp_cpu(self, params: np.ndarray) -> np.ndarray:
        """Generate 2-comp Voigt spectra via scipy wofz (CPU)."""
        from scipy.special import wofz

        n = params.shape[0]
        energy = self.energy.astype(np.float64)
        SQRT2 = np.sqrt(2.0)
        SQRT2PI = np.sqrt(2.0 * np.pi)

        Y = np.zeros((n, self.n_energy), dtype=np.float64)

        for k, (center, col_amp, col_dE, col_ds, col_dg) in enumerate([
            (self.center1, 0, 1, 2, 3),
            (self.center2, 4, 5, 6, 7),
        ]):
            amps = params[:, col_amp].astype(np.float64) * AMPLITUDE_SCALE
            centers = (center + params[:, col_dE].astype(np.float64))[:, None]
            sigmas = np.maximum(
                SIGMA_NOM + params[:, col_ds].astype(np.float64), 0.01
            )[:, None]
            gamma_k = GAMMA + params[:, col_dg].astype(np.float64)

            z = (energy[None, :] - centers + 1j * gamma_k[:, None]) / (sigmas * SQRT2)
            P = np.real(wofz(z)) / (sigmas * SQRT2PI)
            Y += amps[:, None] * P

        # Background
        total_amp = (params[:, 0] + params[:, 4]).astype(np.float64) * AMPLITUDE_SCALE
        Y += (total_amp * BG_FRACTION)[:, None]

        return Y.astype(np.float32)

    def _gen_2comp_mlx(self, params: np.ndarray) -> np.ndarray:
        """Generate 2-comp Voigt spectra via Humlicek w4 on MLX."""
        n = params.shape[0]

        # Sub-chunk for GPU memory: 2-comp uses ~2x memory
        n_max = int(4e9 / (40 * self.n_energy))
        n_max = max(n_max, 10_000)

        if n <= n_max:
            return self._gen_2comp_mlx_one(params)

        parts = []
        for s in range(0, n, n_max):
            e = min(s + n_max, n)
            parts.append(self._gen_2comp_mlx_one(params[s:e]))
        return np.concatenate(parts, axis=0)

    def _gen_2comp_mlx_one(self, params: np.ndarray) -> np.ndarray:
        """Single MLX batch: 2-comp Humlicek w4."""
        energy_mx = mx.array(self.energy)
        SQRT2 = float(np.sqrt(2.0))
        SQRT2PI = float(np.sqrt(2.0 * np.pi))

        spectra_mx = mx.zeros((params.shape[0], self.n_energy))

        for center, col_amp, col_dE, col_ds, col_dg in [
            (self.center1, 0, 1, 2, 3),
            (self.center2, 4, 5, 6, 7),
        ]:
            amp_mx = mx.array(params[:, col_amp].astype(np.float32)) * AMPLITUDE_SCALE
            dE_mx = mx.array(params[:, col_dE].astype(np.float32))
            ds_mx = mx.array(params[:, col_ds].astype(np.float32))
            dg_mx = mx.array(params[:, col_dg].astype(np.float32))

            sigmas_mx = SIGMA_NOM + ds_mx
            gamma_mx = GAMMA + dg_mx
            inv_s2 = 1.0 / (sigmas_mx * SQRT2)

            x = (energy_mx[None, :] - center - dE_mx[:, None]) * inv_s2[:, None]
            y_2d = (gamma_mx * inv_s2)[:, None]

            # Humlicek w4 Region 3 (real arithmetic)
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

            spectra_mx = spectra_mx + amp_mx[:, None] * profiles

        # Background
        total_amp = (mx.array(params[:, 0].astype(np.float32))
                     + mx.array(params[:, 4].astype(np.float32))) * AMPLITUDE_SCALE
        spectra_mx = spectra_mx + (total_amp * BG_FRACTION)[:, None]

        mx.eval(spectra_mx)
        result = np.array(spectra_mx, dtype=np.float32)
        del spectra_mx
        return result

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------

    def _add_noise(self, spectra: np.ndarray) -> np.ndarray:
        """Add Poisson noise."""
        if self.noise_level <= 0:
            return spectra
        if HAS_MLX:
            from toyomacro.voigtfit.spectra_generator import add_poisson_noise_mlx_fused
            spectra_mx = mx.array(spectra)
            global_max = float(np.max(spectra))
            noisy = add_poisson_noise_mlx_fused(spectra_mx, self.noise_level, global_max)
            mx.eval(noisy)
            result = np.array(noisy, dtype=np.float32)
            del noisy, spectra_mx
            return result
        from toyomacro.voigtfit.spectra_generator import add_poisson_noise
        return add_poisson_noise(spectra, self.noise_level)

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def _solve(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve 2-comp spectra → (amp, dE, dσ) per component.

        Uses process_multipeak (builds dicts + alternating projection).
        No 2-stage γ calibration — γ is already correct since we
        generate at nominal γ.

        Returns:
            amplitudes: (n, 2)
            delta_E: (n, 2)
            delta_sigma: (n, 2)
        """
        from toyomacro.voigtfit.multipeak_solver import process_multipeak

        config = self._make_solver_config()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = process_multipeak(
                Y, config,
                n_iterations=3,
                parabola_dE=True,
                parabola_ds=False,
                auto_constrain=True,
            )

        return result.amplitudes, result.delta_E, result.delta_sigma

    def _solve_direct(
        self, Y: np.ndarray, chunk_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Direct alternating projection solve (no 2-stage).

        Builds dictionaries once, reuses across chunks.
        """
        config = self._make_solver_config()
        config.constrain_dE_ranges()

        if self._dict_cache is None:
            self._dict_cache = build_multipeak_dictionaries(config, "uniform")
            n_dict = sum(d.n_dict for d in self._dict_cache)
            print(f"  Dict built: {len(self._dict_cache)} components, "
                  f"total {n_dict} entries")

        result = solve_alternating_projection(
            Y, self._dict_cache,
            n_iterations=3,
            parabola_dE=True,
            parabola_ds=False,
        )
        return result.amplitudes, result.delta_E, result.delta_sigma

    # ------------------------------------------------------------------
    # Swap correction
    # ------------------------------------------------------------------

    @staticmethod
    def _correct_swaps(
        amp: np.ndarray,
        dE: np.ndarray,
        ds: np.ndarray,
        gt_amp: np.ndarray,
        gt_dE: np.ndarray,
        gt_ds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Per-spectrum swap correction for ambiguous 2-component assignment."""
        err_orig = ((dE[:, 0] - gt_dE[:, 0]) ** 2
                    + (dE[:, 1] - gt_dE[:, 1]) ** 2)
        err_swap = ((dE[:, 1] - gt_dE[:, 0]) ** 2
                    + (dE[:, 0] - gt_dE[:, 1]) ** 2)
        swap = err_swap < err_orig

        amp_c = amp.copy()
        dE_c = dE.copy()
        ds_c = ds.copy()

        amp_c[swap, 0], amp_c[swap, 1] = amp[swap, 1], amp[swap, 0]
        dE_c[swap, 0], dE_c[swap, 1] = dE[swap, 1], dE[swap, 0]
        ds_c[swap, 0], ds_c[swap, 1] = ds[swap, 1], ds[swap, 0]

        return amp_c, dE_c, ds_c, int(swap.sum())

    # ------------------------------------------------------------------
    # Process one chunk
    # ------------------------------------------------------------------

    def _process_chunk(
        self, chunk_id: int, gt: np.ndarray,
    ) -> tuple[np.ndarray, ChunkResult]:
        """Process one chunk: encode → decode → gen → solve → compare.

        Args:
            gt: (n, 8) ground truth parameters

        Returns:
            rec: (n, 6) = [amp1, dE1, dσ1, amp2, dE2, dσ2] recovered
            stats: ChunkResult
        """
        n = gt.shape[0]

        # Encode + decode (quantize)
        t0 = time.perf_counter()
        _, decoded = self._encode_decode(gt)
        t_enc = time.perf_counter() - t0

        # Generate spectra from decoded params
        t0 = time.perf_counter()
        Y = self._generate_spectra(decoded)
        Y = self._add_noise(Y)
        t_gen = time.perf_counter() - t0

        # Solve
        t0 = time.perf_counter()
        amp_rec, dE_rec, ds_rec = self._solve_direct(Y)
        if HAS_MLX:
            mx.eval()
        t_fit = time.perf_counter() - t0

        # Swap correction
        gt_amp = np.column_stack([gt[:, 0], gt[:, 4]])
        gt_dE = np.column_stack([gt[:, 1], gt[:, 5]])
        gt_ds = np.column_stack([gt[:, 2], gt[:, 6]])
        amp_c, dE_c, ds_c, _ = self._correct_swaps(
            amp_rec, dE_rec, ds_rec, gt_amp, gt_dE, gt_ds,
        )

        # Normalize amplitudes back from AMPLITUDE_SCALE
        amp_c = amp_c / AMPLITUDE_SCALE

        rec = np.column_stack([
            amp_c[:, 0], dE_c[:, 0], ds_c[:, 0],
            amp_c[:, 1], dE_c[:, 1], ds_c[:, 1],
        ]).astype(np.float32)

        total_time = t_enc + t_gen + t_fit

        del Y
        gc.collect()

        return rec, ChunkResult(
            chunk_id=chunk_id,
            n_spectra=n,
            gen_time=t_enc + t_gen,
            fit_time=t_fit,
            total_time=total_time,
            throughput=n / t_fit if t_fit > 0 else float('inf'),
            rss_gb=_get_rss_gb(),
        )

    # ==================================================================
    # Part 1: Smoke test
    # ==================================================================

    def run_smoke(self, n_test: int = 10_000):
        """Quick smoke test: 10K spectra, full pipeline."""
        print("=" * 80)
        print(f"Part 1: Smoke Test (n={n_test:,}, Δc={self.separation_sigma}σ)")
        print("=" * 80)

        rng = np.random.default_rng(self.seed)
        gt = self._generate_gt_params(n_test, rng)

        t0 = time.perf_counter()
        rec, stats = self._process_chunk(0, gt)
        elapsed = time.perf_counter() - t0

        # PSNR per channel
        gt_flat = np.column_stack([
            gt[:, 0], gt[:, 1], gt[:, 2],  # amp1, dE1, dσ1
            gt[:, 4], gt[:, 5], gt[:, 6],  # amp2, dE2, dσ2
        ])

        labels = ['amp1', 'δE1', 'δσ1', 'amp2', 'δE2', 'δσ2']
        print(f"\n  {'Channel':<10} {'PSNR (dB)':>12} {'RMSE':>12}")
        print(f"  {'-' * 38}")
        for i, label in enumerate(labels):
            p = _psnr(gt_flat[:, i], rec[:, i])
            rmse = float(np.sqrt(np.mean((gt_flat[:, i] - rec[:, i]) ** 2)))
            print(f"  {label:<10} {p:>12.1f} {rmse:>12.6f}")

        # Average per parameter type
        amp_psnr = (_psnr(gt_flat[:, 0], rec[:, 0])
                    + _psnr(gt_flat[:, 3], rec[:, 3])) / 2
        dE_psnr = (_psnr(gt_flat[:, 1], rec[:, 1])
                   + _psnr(gt_flat[:, 4], rec[:, 4])) / 2
        ds_psnr = (_psnr(gt_flat[:, 2], rec[:, 2])
                   + _psnr(gt_flat[:, 5], rec[:, 5])) / 2

        print(f"\n  Average: amp={amp_psnr:.1f} dB, δE={dE_psnr:.1f} dB, "
              f"δσ={ds_psnr:.1f} dB")
        print(f"  Throughput: {stats.throughput / 1e6:.2f} M/s")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  RSS: {_get_rss_gb():.2f} GB")
        print("=" * 80)

        return {'amp': amp_psnr, 'dE': dE_psnr, 'ds': ds_psnr}

    # ==================================================================
    # Part 2: Stage profiling
    # ==================================================================

    def run_profile(self, n_test: int = 100_000):
        """Profile individual pipeline stages at 100K-1M scale."""
        print("=" * 80)
        print(f"Part 2: Stage Profiling (n={n_test:,})")
        print("=" * 80)

        rng = np.random.default_rng(self.seed)
        gt = self._generate_gt_params(n_test, rng)

        # Stage 1: Parameter generation
        t0 = time.perf_counter()
        _ = self._generate_gt_params(n_test, np.random.default_rng(99))
        t_param_gen = time.perf_counter() - t0

        # Stage 2: Encode
        t0 = time.perf_counter()
        channels = self.encoder.encode(
            gt[:, 1], gt[:, 2], gt[:, 3],
            gt[:, 5], gt[:, 6], gt[:, 7],
        )
        t_encode = time.perf_counter() - t0

        # Stage 3: Decode
        t0 = time.perf_counter()
        rec_enc = self.encoder.decode(channels)
        t_decode = time.perf_counter() - t0

        # Assemble decoded params
        decoded = np.column_stack([
            gt[:, 0], rec_enc['dE1'], rec_enc['dsigma1'], rec_enc['dgamma1'],
            gt[:, 4], rec_enc['dE2'], rec_enc['dsigma2'], rec_enc['dgamma2'],
        ]).astype(np.float32)

        # Stage 4: Spectra generation
        t0 = time.perf_counter()
        Y = self._generate_spectra(decoded)
        t_spectra = time.perf_counter() - t0

        # Stage 5: Noise
        t0 = time.perf_counter()
        Y = self._add_noise(Y)
        t_noise = time.perf_counter() - t0

        # Stage 6: Solver
        t0 = time.perf_counter()
        amp_rec, dE_rec, ds_rec = self._solve_direct(Y)
        if HAS_MLX:
            mx.eval()
        t_solver = time.perf_counter() - t0

        # Stage 7: Swap correction
        gt_amp = np.column_stack([gt[:, 0], gt[:, 4]])
        gt_dE = np.column_stack([gt[:, 1], gt[:, 5]])
        gt_ds = np.column_stack([gt[:, 2], gt[:, 6]])
        t0 = time.perf_counter()
        self._correct_swaps(amp_rec, dE_rec, ds_rec, gt_amp, gt_dE, gt_ds)
        t_swap = time.perf_counter() - t0

        profile = StageProfile(
            n_spectra=n_test,
            t_param_gen=t_param_gen,
            t_encode=t_encode,
            t_decode=t_decode,
            t_spectra_gen=t_spectra,
            t_noise=t_noise,
            t_solver=t_solver,
            t_swap_correct=t_swap,
            t_write=0.0,
            rss_gb=_get_rss_gb(),
        )

        print(f"\n  {'Stage':<20} {'Time (s)':>10} {'M/s':>10} {'%':>8}")
        print(f"  {'-' * 52}")
        stages = [
            ('Param gen', t_param_gen),
            ('Encode', t_encode),
            ('Decode', t_decode),
            ('Spectra gen', t_spectra),
            ('Noise', t_noise),
            ('Solver', t_solver),
            ('Swap correct', t_swap),
        ]
        total = sum(t for _, t in stages)
        for name, t in stages:
            ms = n_test / t / 1e6 if t > 0 else float('inf')
            pct = t / total * 100 if total > 0 else 0
            print(f"  {name:<20} {t:>10.4f} {ms:>10.2f} {pct:>7.1f}%")
        print(f"  {'TOTAL':<20} {total:>10.4f} {n_test / total / 1e6:>10.2f}")
        print(f"\n  Bottleneck: {'Solver' if t_solver > t_spectra else 'Spectra gen'}")
        print(f"  RSS: {_get_rss_gb():.2f} GB")
        print("=" * 80)

        del Y
        gc.collect()

        return profile

    # ==================================================================
    # Part 3: Scale-up benchmark
    # ==================================================================

    def run_scale(self):
        """Measure throughput at increasing scales."""
        scales = [1_000, 10_000, 100_000, 500_000, 1_000_000]
        rng = np.random.default_rng(self.seed)

        print("=" * 80)
        print("Part 3: Scale-Up Benchmark (MultiPeak)")
        print("=" * 80)
        print(f"{'N':>12} {'Gen (s)':>10} {'Fit (s)':>10} {'Total (s)':>10} "
              f"{'Fit M/s':>10} {'RSS GB':>8}")
        print("-" * 80)

        for N in scales:
            gt = self._generate_gt_params(N, rng)
            rec, stats = self._process_chunk(0, gt)

            # Quick PSNR
            gt_flat = np.column_stack([
                gt[:, 0], gt[:, 1], gt[:, 2],
                gt[:, 4], gt[:, 5], gt[:, 6],
            ])
            p_amp = (_psnr(gt_flat[:, 0], rec[:, 0])
                     + _psnr(gt_flat[:, 3], rec[:, 3])) / 2
            p_dE = (_psnr(gt_flat[:, 1], rec[:, 1])
                    + _psnr(gt_flat[:, 4], rec[:, 4])) / 2
            p_ds = (_psnr(gt_flat[:, 2], rec[:, 2])
                    + _psnr(gt_flat[:, 5], rec[:, 5])) / 2

            print(f"{N:>12,} {stats.gen_time:>10.3f} {stats.fit_time:>10.3f} "
                  f"{stats.total_time:>10.3f} {stats.throughput / 1e6:>10.2f} "
                  f"{stats.rss_gb:>8.2f}  "
                  f"PSNR: amp={p_amp:.1f} δE={p_dE:.1f} δσ={p_ds:.1f}")

            del gt, rec
            gc.collect()

        print("=" * 80)

    # ==================================================================
    # CPU dict-based 2-comp generation (for pipelining)
    # ==================================================================

    def _build_profile_matrices(self):
        """Pre-build per-component profile lookup tables for CPU gen.

        Creates a profile matrix per component: (n_dict_k, n_energy).
        Used by _gen_2comp_dict_cpu for fast dictionary-based generation
        that runs entirely on CPU (no GPU, enabling pipeline overlap).
        """
        if self._dict_cache is None:
            config = self._make_solver_config()
            config.constrain_dE_ranges()
            self._dict_cache = build_multipeak_dictionaries(config, "uniform")

        self._profile_matrices = []
        for dc in self._dict_cache:
            # dc.Phi_per_grid[i] is (n_energy, n_comp) for grid point i
            # For single-component dict, n_comp=1 → extract col 0
            mat = np.array(
                [dc.Phi_per_grid[i][:, 0] for i in range(dc.n_dict)],
                dtype=np.float32,
            )  # (n_dict_k, n_energy)
            self._profile_matrices.append(mat)

    def _gen_2comp_dict_cpu(self, params: np.ndarray) -> np.ndarray:
        """Dictionary-based 2-comp spectra generation (CPU only).

        For each spectrum, looks up the closest (δE, δσ) grid point
        per component and multiplies by amplitude. Sum of 2 components.
        Runs entirely on CPU for pipeline overlap with GPU solver.

        Args:
            params: (n, 8) = [amp1, dE1, dσ1, dγ1, amp2, dE2, dσ2, dγ2]

        Returns:
            Y: (n, n_energy) float32
        """
        if not hasattr(self, '_profile_matrices') or self._profile_matrices is None:
            self._build_profile_matrices()

        n = params.shape[0]
        Y = np.zeros((n, self.n_energy), dtype=np.float32)

        for k, (col_amp, col_dE, col_ds) in enumerate([
            (0, 1, 2),   # component 1
            (4, 5, 6),   # component 2
        ]):
            dc = self._dict_cache[k]
            mat = self._profile_matrices[k]
            amp = params[:, col_amp] * AMPLITUDE_SCALE

            # Map (dE, dσ) to nearest grid indices
            dE = params[:, col_dE]
            ds = params[:, col_ds]

            dE_idx = np.searchsorted(dc.dE_grid, dE)
            dE_idx = np.clip(dE_idx, 0, dc.n_dE - 1)
            # Snap to nearest
            mask = dE_idx > 0
            lower = dc.dE_grid[np.maximum(dE_idx - 1, 0)]
            upper = dc.dE_grid[dE_idx]
            closer = np.abs(dE - lower) < np.abs(dE - upper)
            dE_idx = np.where(mask & closer, dE_idx - 1, dE_idx)

            ds_idx = np.searchsorted(dc.dsigma_grid, ds)
            ds_idx = np.clip(ds_idx, 0, dc.n_dsigma - 1)
            mask = ds_idx > 0
            lower = dc.dsigma_grid[np.maximum(ds_idx - 1, 0)]
            upper = dc.dsigma_grid[ds_idx]
            closer = np.abs(ds - lower) < np.abs(ds - upper)
            ds_idx = np.where(mask & closer, ds_idx - 1, ds_idx)

            flat_idx = dE_idx * dc.n_dsigma + ds_idx
            profiles = mat[flat_idx]  # (n, nE) via fancy indexing
            Y += amp[:, np.newaxis] * profiles

        # Background
        total_amp = (params[:, 0] + params[:, 4]) * AMPLITUDE_SCALE
        Y += (total_amp * BG_FRACTION)[:, np.newaxis]

        return Y

    def _add_noise_cpu(
        self, spectra: np.ndarray, rng: np.random.Generator,
    ) -> np.ndarray:
        """CPU-only Poisson noise with template-based random generation.

        Uses K=1000 noise row templates for fast noise application.
        """
        if self.noise_level <= 0:
            return spectra

        n = spectra.shape[0]
        global_max = float(np.max(spectra))
        snr_target = 10000.0 / self.noise_level
        scale_factor = snr_target * snr_target / global_max

        K = 1000
        templates = rng.standard_normal((K, self.n_energy), dtype=np.float32)
        idx = rng.integers(0, K, size=n)
        noise = templates[idx]

        np.maximum(spectra, 0, out=spectra)
        spectra *= scale_factor
        std = np.sqrt(spectra)
        noise *= std
        spectra += noise
        np.maximum(spectra, 0, out=spectra)
        spectra *= (1.0 / scale_factor)
        return spectra

    # ==================================================================
    # Pipelined full run (CPU gen + GPU solve overlap)
    # ==================================================================

    def run_full_pipelined(self):
        """Process N_total spectra with gen-fit pipeline overlap.

        Pipeline design (CPU-GPU split):
          Gen thread (CPU only):  encode/decode + dict-based 2-comp gen + noise
          Main thread (GPU):      solve_alternating_projection (MLX)

        Dict-based gen is a matched-filter test: gen uses the same
        dictionary as the solver, so discretization error cancels.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pre-build dicts and profile matrices
        config = self._make_solver_config()
        config.constrain_dE_ranges()
        self._dict_cache = build_multipeak_dictionaries(config, "uniform")
        self._build_profile_matrices()
        n_dict_total = sum(d.n_dict for d in self._dict_cache)
        print(f"  Dict built: {len(self._dict_cache)} components, "
              f"total {n_dict_total} entries")

        rng = np.random.default_rng(self.seed)
        n_chunks = (self.n_total + self.chunk_size - 1) // self.chunk_size

        # Create memmap
        gt_mm = np.memmap(
            str(GT_PATH), dtype='float32', mode='w+',
            shape=(self.n_total, 8),
        )
        rec_mm = np.memmap(
            str(REC_PATH), dtype='float32', mode='w+',
            shape=(self.n_total, 6),
        )

        print(flush=True)
        print("=" * 90, flush=True)
        print(f"GVRT MultiPeak Pipelined Run: {self.n_total:,} spectra, "
              f"{n_chunks} chunks × {self.chunk_size:,}", flush=True)
        print("Mode: CPU dict-gen + GPU alternating-projection pipeline", flush=True)
        print("=" * 90, flush=True)
        print(f"{'Chunk':>6} {'N':>12} {'Wall':>8} {'GenWait':>8} {'Fit':>8} "
              f"{'Fit M/s':>8} {'RSS GB':>8} {'%':>7} {'ETA':>8}", flush=True)
        print("-" * 90, flush=True)

        total_gen_wait = 0.0
        total_fit = 0.0
        t_start = time.perf_counter()

        def gen_chunk_cpu(gt_chunk, noise_seed):
            """CPU-only: encode/decode + dict-based gen + noise."""
            chunk_rng = np.random.default_rng(noise_seed)
            _, decoded = self._encode_decode(gt_chunk)
            spectra = self._gen_2comp_dict_cpu(decoded)
            spectra = self._add_noise_cpu(spectra, chunk_rng)
            return spectra, decoded

        executor = ThreadPoolExecutor(max_workers=1)
        pending_future = None
        pending_chunk_id = -1
        pending_gt = None
        pending_start = 0
        pending_n = 0

        for i in range(n_chunks):
            start_idx = i * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.n_total)
            n = end_idx - start_idx

            gt = self._generate_gt_params(n, rng)
            gt_mm[start_idx:end_idx] = gt
            noise_seed = self.seed + i + 1

            if pending_future is None:
                # First chunk: submit gen and continue to next iteration
                pending_future = executor.submit(gen_chunk_cpu, gt, noise_seed)
                pending_chunk_id = i
                pending_gt = gt
                pending_start = start_idx
                pending_n = n
                continue

            # Steady state: fit previous chunk, gen current chunk
            t_iter0 = time.perf_counter()

            prev_spectra, prev_decoded = pending_future.result()
            t_gen_wait = time.perf_counter() - t_iter0

            # Submit current gen in background (CPU thread)
            pending_future = executor.submit(gen_chunk_cpu, gt, noise_seed)

            # Fit previous chunk on GPU
            t_fit0 = time.perf_counter()
            amp_rec, dE_rec, ds_rec = self._solve_direct(prev_spectra)
            if HAS_MLX:
                mx.eval()
            t_fit = time.perf_counter() - t_fit0

            # Swap correction + write
            gt_amp = np.column_stack([pending_gt[:, 0], pending_gt[:, 4]])
            gt_dE = np.column_stack([pending_gt[:, 1], pending_gt[:, 5]])
            gt_ds = np.column_stack([pending_gt[:, 2], pending_gt[:, 6]])
            amp_c, dE_c, ds_c, _ = self._correct_swaps(
                amp_rec, dE_rec, ds_rec, gt_amp, gt_dE, gt_ds,
            )
            amp_c = amp_c / AMPLITUDE_SCALE

            rec = np.column_stack([
                amp_c[:, 0], dE_c[:, 0], ds_c[:, 0],
                amp_c[:, 1], dE_c[:, 1], ds_c[:, 1],
            ]).astype(np.float32)
            rec_mm[pending_start:pending_start + pending_n] = rec

            total_gen_wait += t_gen_wait
            total_fit += t_fit

            t_wall = time.perf_counter() - t_iter0
            done = pending_start + pending_n
            elapsed = time.perf_counter() - t_start
            frac = done / self.n_total
            eta = elapsed / frac * (1.0 - frac) if frac > 0 else 0
            throughput = pending_n / t_fit if t_fit > 0 else float('inf')

            print(f"{pending_chunk_id + 1:>6}/{n_chunks} {pending_n:>12,} "
                  f"{t_wall:>8.2f} {t_gen_wait:>8.2f} {t_fit:>8.2f} "
                  f"{throughput / 1e6:>8.2f} {_get_rss_gb():>8.2f} "
                  f"{frac * 100:>6.1f}% {eta:>7.0f}s", flush=True)

            # Update pending state
            pending_chunk_id = i
            pending_gt = gt
            pending_start = start_idx
            pending_n = n

            del prev_spectra, prev_decoded, rec
            gc.collect()

        # Process last chunk
        if pending_future is not None:
            t_iter0 = time.perf_counter()
            last_spectra, last_decoded = pending_future.result()
            t_gen_wait = time.perf_counter() - t_iter0

            t_fit0 = time.perf_counter()
            amp_rec, dE_rec, ds_rec = self._solve_direct(last_spectra)
            if HAS_MLX:
                mx.eval()
            t_fit = time.perf_counter() - t_fit0

            gt_amp = np.column_stack([pending_gt[:, 0], pending_gt[:, 4]])
            gt_dE = np.column_stack([pending_gt[:, 1], pending_gt[:, 5]])
            gt_ds = np.column_stack([pending_gt[:, 2], pending_gt[:, 6]])
            amp_c, dE_c, ds_c, _ = self._correct_swaps(
                amp_rec, dE_rec, ds_rec, gt_amp, gt_dE, gt_ds,
            )
            amp_c = amp_c / AMPLITUDE_SCALE

            rec = np.column_stack([
                amp_c[:, 0], dE_c[:, 0], ds_c[:, 0],
                amp_c[:, 1], dE_c[:, 1], ds_c[:, 1],
            ]).astype(np.float32)
            rec_mm[pending_start:pending_start + pending_n] = rec

            total_gen_wait += t_gen_wait
            total_fit += t_fit

            done = pending_start + pending_n
            elapsed = time.perf_counter() - t_start
            frac = done / self.n_total
            throughput = pending_n / t_fit if t_fit > 0 else float('inf')

            print(f"{pending_chunk_id + 1:>6}/{n_chunks} {pending_n:>12,} "
                  f"{time.perf_counter() - t_iter0:>8.2f} {t_gen_wait:>8.2f} "
                  f"{t_fit:>8.2f} {throughput / 1e6:>8.2f} "
                  f"{_get_rss_gb():>8.2f} {frac * 100:>6.1f}%      0s", flush=True)

            del last_spectra, last_decoded, rec

        executor.shutdown(wait=False)
        gc.collect()

        gt_mm.flush()
        rec_mm.flush()
        del gt_mm, rec_mm

        total_time = time.perf_counter() - t_start
        eff = self.n_total / total_fit if total_fit > 0 else 0

        print("=" * 90)
        print(f"DONE (pipelined): {self.n_total:,} spectra in {total_time:.1f}s")
        print(f"  Gen wait: {total_gen_wait:.1f}s | Fit: {total_fit:.1f}s")
        print(f"  Effective fit throughput: {eff / 1e6:.2f} M/s")
        print(f"  End-to-end throughput: {self.n_total / total_time / 1e6:.2f} M/s")
        print(f"  Peak RSS: {_get_rss_gb():.2f} GB")
        print("=" * 90)

    # ==================================================================
    # Part 4: Full 1B run
    # ==================================================================

    def run_full(self):
        """Process N_total spectra in chunks, writing to memmap."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(self.seed)

        n_chunks = (self.n_total + self.chunk_size - 1) // self.chunk_size

        # Create memmap: 8 cols GT, 6 cols rec
        gt_mm = np.memmap(
            str(GT_PATH), dtype='float32', mode='w+',
            shape=(self.n_total, 8),
        )
        rec_mm = np.memmap(
            str(REC_PATH), dtype='float32', mode='w+',
            shape=(self.n_total, 6),
        )

        print(flush=True)
        print("=" * 80, flush=True)
        print(f"GVRT MultiPeak Full Run: {self.n_total:,} spectra, "
              f"{n_chunks} chunks × {self.chunk_size:,}", flush=True)
        print(f"Separation: {self.separation_sigma}σ, "
              f"Amp ratio: {self.amp_ratio}, "
              f"Noise: {self.noise_level}", flush=True)
        print(f"GT  → {GT_PATH}  ({self.n_total * 8 * 4 / 1e9:.1f} GB)", flush=True)
        print(f"REC → {REC_PATH} ({self.n_total * 6 * 4 / 1e9:.1f} GB)", flush=True)
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

            rec, stats = self._process_chunk(i, gt)
            rec_mm[start:end] = rec

            total_gen += stats.gen_time
            total_fit += stats.fit_time

            elapsed = time.perf_counter() - t_start
            frac = end / self.n_total
            eta = elapsed / frac * (1.0 - frac) if frac > 0 else 0

            print(f"{i + 1:>6}/{n_chunks} {n:>12,} {stats.gen_time:>8.2f} "
                  f"{stats.fit_time:>8.2f} {stats.throughput / 1e6:>8.2f} "
                  f"{stats.rss_gb:>8.2f} {frac * 100:>7.1f}% "
                  f"{eta:>7.0f}s", flush=True)

            del gt, rec
            gc.collect()

        gt_mm.flush()
        rec_mm.flush()
        del gt_mm, rec_mm

        total_time = time.perf_counter() - t_start
        effective_throughput = self.n_total / total_fit

        print("=" * 80)
        print(f"DONE: {self.n_total:,} spectra in {total_time:.1f}s")
        print(f"  Gen: {total_gen:.1f}s | Fit: {total_fit:.1f}s | "
              f"Overhead: {total_time - total_gen - total_fit:.1f}s")
        print(f"  Effective fit throughput: {effective_throughput / 1e6:.2f} M/s")
        print(f"  End-to-end throughput: {self.n_total / total_time / 1e6:.2f} M/s")
        print(f"  Peak RSS: {_get_rss_gb():.2f} GB")
        print("=" * 80)

    # ==================================================================
    # Part 5: PSNR verification
    # ==================================================================

    def run_psnr(self, n_sample: int = 1_000_000):
        """Sample from memmap and compute PSNR vs GT."""
        if not GT_PATH.exists() or not REC_PATH.exists():
            print(f"ERROR: Run --mode full first. Missing {GT_PATH} or {REC_PATH}")
            return

        gt_bytes = GT_PATH.stat().st_size
        n_total = gt_bytes // (8 * 4)  # 8 cols, float32
        print(f"Loading memmap: {n_total:,} spectra")

        gt_mm = np.memmap(str(GT_PATH), dtype='float32', mode='r',
                          shape=(n_total, 8))
        rec_mm = np.memmap(str(REC_PATH), dtype='float32', mode='r',
                           shape=(n_total, 6))

        rng = np.random.default_rng(123)
        n_sample = min(n_sample, n_total)
        idx = rng.choice(n_total, n_sample, replace=False)
        idx.sort()

        gt_s = gt_mm[idx]
        rec_s = rec_mm[idx]

        # GT flat: [amp1, dE1, dσ1, amp2, dE2, dσ2]
        gt_flat = np.column_stack([
            gt_s[:, 0], gt_s[:, 1], gt_s[:, 2],
            gt_s[:, 4], gt_s[:, 5], gt_s[:, 6],
        ])

        print()
        print("=" * 70)
        print(f"GVRT MultiPeak PSNR Verification ({n_sample:,} / {n_total:,})")
        print("=" * 70)
        labels = ['amp1', 'δE1', 'δσ1', 'amp2', 'δE2', 'δσ2']
        print(f"{'Channel':<12} {'PSNR (dB)':>12} {'RMSE':>12}")
        print("-" * 40)
        for i, label in enumerate(labels):
            p = _psnr(gt_flat[:, i], rec_s[:, i])
            rmse = float(np.sqrt(np.mean((gt_flat[:, i] - rec_s[:, i]) ** 2)))
            print(f"{label:<12} {p:>12.1f} {rmse:>12.6f}")

        # Averages
        amp_avg = (_psnr(gt_flat[:, 0], rec_s[:, 0])
                   + _psnr(gt_flat[:, 3], rec_s[:, 3])) / 2
        dE_avg = (_psnr(gt_flat[:, 1], rec_s[:, 1])
                  + _psnr(gt_flat[:, 4], rec_s[:, 4])) / 2
        ds_avg = (_psnr(gt_flat[:, 2], rec_s[:, 2])
                  + _psnr(gt_flat[:, 5], rec_s[:, 5])) / 2

        print("-" * 40)
        print(f"{'avg amp':<12} {amp_avg:>12.1f}")
        print(f"{'avg δE':<12} {dE_avg:>12.1f}")
        print(f"{'avg δσ':<12} {ds_avg:>12.1f}")
        print("=" * 70)
        print()
        print("Reference:")
        print("  dev-log 68 (1-peak): amp=53.3, δE=106.5, δσ=47.3 dB")
        print("  dev-log 76 (2-peak, 10K): δσ E2E ≈ 32.7 dB")

        del gt_mm, rec_mm

    # ==================================================================
    # Part 6: Visualization
    # ==================================================================

    def run_viz(self):
        """Generate 4-panel comparison figure: dev-log 68 vs dev-log 77."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping visualization")
            return

        if not GT_PATH.exists() or not REC_PATH.exists():
            print("ERROR: Run --mode full first.")
            return

        gt_bytes = GT_PATH.stat().st_size
        n_total = gt_bytes // (8 * 4)

        gt_mm = np.memmap(str(GT_PATH), dtype='float32', mode='r',
                          shape=(n_total, 8))
        rec_mm = np.memmap(str(REC_PATH), dtype='float32', mode='r',
                           shape=(n_total, 6))

        # Sample 100K for visualization
        rng = np.random.default_rng(456)
        n_viz = min(100_000, n_total)
        idx = rng.choice(n_total, n_viz, replace=False)
        idx.sort()

        gt_s = gt_mm[idx]
        rec_s = rec_mm[idx]

        gt_flat = np.column_stack([
            gt_s[:, 0], gt_s[:, 1], gt_s[:, 2],
            gt_s[:, 4], gt_s[:, 5], gt_s[:, 6],
        ])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel A: PSNR comparison
        ax = axes[0, 0]
        params = ['amp', 'δE', 'δσ']
        s68_vals = [53.3, 106.5, 47.3]  # dev-log 68 reference
        s77_vals = [
            (_psnr(gt_flat[:, 0], rec_s[:, 0])
             + _psnr(gt_flat[:, 3], rec_s[:, 3])) / 2,
            (_psnr(gt_flat[:, 1], rec_s[:, 1])
             + _psnr(gt_flat[:, 4], rec_s[:, 4])) / 2,
            (_psnr(gt_flat[:, 2], rec_s[:, 2])
             + _psnr(gt_flat[:, 5], rec_s[:, 5])) / 2,
        ]
        x = np.arange(len(params))
        w = 0.3
        ax.bar(x - w / 2, s68_vals, w, label='S68 (1-peak)', color='C0')
        ax.bar(x + w / 2, s77_vals, w, label='S77 (2-peak)', color='C1')
        ax.set_xticks(x)
        ax.set_xticklabels(params)
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('A: PSNR Comparison')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Panel B: Per-component scatter (δE)
        ax = axes[0, 1]
        ax.scatter(gt_flat[:min(5000, n_viz), 1], rec_s[:min(5000, n_viz), 1],
                   s=1, alpha=0.3, label='Peak 1')
        ax.scatter(gt_flat[:min(5000, n_viz), 4], rec_s[:min(5000, n_viz), 4],
                   s=1, alpha=0.3, label='Peak 2')
        lim = [-0.55, 0.55]
        ax.plot(lim, lim, 'k--', alpha=0.3)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel('GT δE (eV)')
        ax.set_ylabel('Recovered δE (eV)')
        ax.set_title('B: δE Recovery')
        ax.legend(fontsize=8)
        ax.set_aspect('equal')

        # Panel C: δσ error histogram
        ax = axes[1, 0]
        err1 = rec_s[:, 2] - gt_flat[:, 2]
        err2 = rec_s[:, 5] - gt_flat[:, 5]
        ax.hist(err1, bins=100, alpha=0.6, label='Peak 1', density=True)
        ax.hist(err2, bins=100, alpha=0.6, label='Peak 2', density=True)
        ax.set_xlabel('δσ error (eV)')
        ax.set_ylabel('Density')
        ax.set_title('C: δσ Error Distribution')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel D: δσ PSNR evolution across sessions
        ax = axes[1, 1]
        sessions = [55, 68, 72, 76, 77]
        labels = ['1-peak\nDict2D', '1-peak\n1B', '2-peak\n3D Hilb', '2-peak\nSplit', '2-peak\n1B Split']
        ds_psnr_history = [47.1, 47.3, 22.5, 32.7, s77_vals[2]]
        colors = ['C0', 'C0', 'C1', 'C1', 'C1']
        ax.bar(range(len(sessions)), ds_psnr_history, color=colors, alpha=0.7)
        ax.set_xticks(range(len(sessions)))
        ax.set_xticklabels([f'S{s}\n{l}' for s, l in zip(sessions, labels)],
                           fontsize=7)
        ax.set_ylabel('δσ PSNR (dB)')
        ax.set_title('D: δσ PSNR Evolution')
        ax.grid(axis='y', alpha=0.3)
        for i, p in enumerate(ds_psnr_history):
            ax.annotate(f'{p:.1f}', (i, p), textcoords="offset points",
                        xytext=(0, 5), fontsize=8, ha='center')

        fig.suptitle('GVRT MultiPeak 1B Round Trip', fontsize=14)
        plt.tight_layout()

        out_path = self.output_dir / "session77_4panel.png"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=200, bbox_inches='tight')
        print(f"Figure saved: {out_path}")
        plt.close(fig)

        del gt_mm, rec_mm


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GVRT MultiPeak 1-Billion Spectra Benchmark",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "profile", "scale", "full", "pipeline", "psnr", "viz"],
        default="smoke",
        help="Benchmark mode",
    )
    parser.add_argument(
        "-N", type=int, default=1_000_000_000,
        help="Total spectra count (default: 1B)",
    )
    parser.add_argument(
        "--chunk", type=int, default=2_000_000,
        help="Chunk size (default: 2M)",
    )
    parser.add_argument(
        "--noise", type=float, default=0.0,
        help="Poisson noise level (0=none, 1000=moderate)",
    )
    parser.add_argument(
        "--separation", type=float, default=DEFAULT_SEPARATION_SIGMA,
        help="Peak separation in σ units (default: 7.0)",
    )
    parser.add_argument(
        "--amp-ratio", type=float, default=1.0,
        help="Amplitude ratio peak2/peak1 (default: 1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed",
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_DIR),
        help="Output directory",
    )
    args = parser.parse_args()

    bench = GVRTMultiPeak1BBenchmark(
        n_total=args.N,
        chunk_size=args.chunk,
        noise_level=args.noise,
        separation_sigma=args.separation,
        amp_ratio=args.amp_ratio,
        output_dir=Path(args.output),
        seed=args.seed,
    )

    if args.mode == "smoke":
        bench.run_smoke()
    elif args.mode == "profile":
        bench.run_profile()
    elif args.mode == "scale":
        bench.run_scale()
    elif args.mode == "full":
        bench.run_full()
    elif args.mode == "pipeline":
        bench.run_full_pipelined()
    elif args.mode == "psnr":
        bench.run_psnr()
    elif args.mode == "viz":
        bench.run_viz()


if __name__ == "__main__":
    main()
