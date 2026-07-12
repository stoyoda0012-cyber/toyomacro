"""Interactive GVRT (Giga Voigt Round Trip) service.

Headless, chunked roundtrip engine: RGB image -> (amp, deltaE, FWHM)
parameters -> Voigt spectra + Poisson noise -> solver -> parameters ->
reconstructed image + per-channel PSNR.

Wraps the proven numerics of benchmarks/param_roundtrip_benchmark.py as
an incremental generator so callers (GUI playground tab, MCP server,
in-app agent) can render the reconstructed image as it fills in,
chunk by chunk.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .benchmarks.param_roundtrip_benchmark import (
    ChannelPSNR,
    ParamRoundtripBenchmark,
    compute_per_channel_psnr,
    parse_noise_level,
)
from .param_encoder import (
    C1S_MULTIPEAK_PRESET,
    C1S_SINGLE_PRESET,
    MultiPeakEncoder,
    SinglePeakPreset,
)
from .spectra_generator import NOISE_LEVELS

# Solvers exposed to interactive callers (subset of benchmark solvers)
GVRT_SOLVERS = {
    'taylor': '4-step Taylor (fast)',
    'taylor6': '6-step Taylor (+Hessian)',
    'dict2d_parabola': 'Dict2D + parabola (best quality)',
}

# Noise levels exposed to interactive callers.  Each name maps to a
# dimensionless noise-severity `level` (see spectra_generator.NOISE_LEVELS);
# peak-count SNR = 1e4/level, peak Poisson mean = (1e4/level)^2.
GVRT_NOISE_LEVELS = [
    'None', 'Subtle', 'Weak', 'Small', 'Moderate', 'Strong', 'Intense',
]


@dataclass
class GVRTConfig:
    """Configuration for an interactive GVRT run."""
    solver: str = 'taylor'              # key of GVRT_SOLVERS (1-peak mode only)
    noise_level: str = 'Moderate'       # NOISE_LEVELS name or 'lamX' notation
    chunk_pixels: int = 16_384          # spectra per chunk (= per progress frame)
    max_pixels: int = 1_048_576         # images above this are downscaled
    exact_voigt: bool = False           # force exact Voigt generation
    bg_fraction: float = 0.001
    amplitude_scale: float = 1000.0
    n_peaks: int = 1                    # 1 = SinglePeakEncoder (RGB=amp/dE/FWHM),
                                        # 2 = MultiPeakEncoder (Hilbert) + AP solver


@dataclass
class GVRTResult:
    """Final result of a GVRT run."""
    n_peaks: int
    solver: str
    noise_level: str
    noise_value: float
    image_shape: tuple[int, int]
    n_spectra: int
    gen_time: float
    fit_time: float
    total_time: float
    throughput: float                   # fit spectra/s
    psnr: ChannelPSNR
    amp_corr: float
    shift_corr: float
    fwhm_corr: float
    amp_rmse: float
    shift_rmse: float
    fwhm_rmse: float
    original: np.ndarray                # (H, W, 3) uint8
    reconstructed: np.ndarray           # (H, W, 3) uint8

    @property
    def summary(self) -> str:
        p = self.psnr
        return (
            f"{self.n_spectra:,} spectra  solver={self.solver}  "
            f"noise={self.noise_level}\n"
            f"PSNR: R(amp)={p.amplitude:.1f}  G(dE)={p.shift:.1f}  "
            f"B(FWHM)={p.fwhm:.1f}  avg={p.total:.1f} dB\n"
            f"gen {self.gen_time:.2f}s + fit {self.fit_time:.2f}s  "
            f"({self.throughput / 1e6:.2f} M spec/s)"
        )


@dataclass
class GVRTPixelInspection:
    """Single-pixel spectrum-space view of the roundtrip.

    Produced by GVRTService.inspect_pixel() after a run: the pixel's
    clean ground-truth spectrum, a noisy realization (same noise level
    as the run), the solver's fit of that exact noisy spectrum, and the
    parameters from all three views (ground truth / this fit / the
    batch run's recovered values).

    `peaks` holds one dict per component with keys gt_amp/gt_dE/gt_fwhm,
    fit_amp/fit_dE/fit_fwhm, batch_amp/batch_dE/batch_fwhm. The scalar
    fields mirror peaks[0] for single-peak convenience.
    """
    x: int
    y: int
    energy: np.ndarray              # (n_energy,)
    clean: np.ndarray               # ground-truth noise-free spectrum
    noisy: np.ndarray               # the spectrum that was fitted (fresh noise)
    fit: np.ndarray                 # model curve at the fitted parameters
    peaks: list[dict]               # per-component parameters (see above)
    gt_amp: float                   # scaled amplitude (same units as spectra)
    gt_dE: float
    gt_fwhm: float
    fit_amp: float
    fit_dE: float
    fit_fwhm: float
    batch_amp: float
    batch_dE: float
    batch_fwhm: float
    residual_rms: float             # RMS(noisy - fit)


@dataclass
class GVRTProgress:
    """One progress frame yielded per processed chunk."""
    done: int                           # pixels fitted so far
    total: int
    image: np.ndarray                   # (H, W, 3) uint8, unfitted region dark
    gen_time: float                     # cumulative seconds
    fit_time: float
    throughput: float                   # fit spectra/s so far
    message: str = ''
    result: GVRTResult | None = None    # set on the final frame only


def prepare_image(
    source: str | Path | np.ndarray,
    max_pixels: int = 1_048_576,
) -> np.ndarray:
    """Load and normalize an image to (H, W, 3) uint8, downscaling if needed.

    Accepts a file path or an ndarray (grayscale, RGB, or RGBA).
    Downscaling uses nearest-neighbor striding — adequate for a playground.
    """
    if isinstance(source, (str, Path)):
        from .image_utils import load_image
        img = load_image(source)
    else:
        img = np.asarray(source)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    n = img.shape[0] * img.shape[1]
    if n > max_pixels:
        factor = int(np.ceil(np.sqrt(n / max_pixels)))
        img = img[::factor, ::factor]
    return np.ascontiguousarray(img)


def make_demo_image(size: int = 384) -> np.ndarray:
    """Synthetic demo image: smooth gradients + geometric shapes.

    Designed so all three encoded parameters (R=amp, G=deltaE, B=FWHM)
    have both smooth regions and sharp edges.
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    r = 255 * (0.25 + 0.75 * x)                       # amplitude: keep > 0
    g = 255 * y                                       # deltaE gradient
    cx, cy = 0.5, 0.5
    rad = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    b = 255 * np.clip(1.5 * (1 - rad * 2), 0, 1)      # FWHM radial

    img = np.stack([r, g, b], axis=-1)

    # Shapes: circle (high amp), square (deltaE step), ring (FWHM step)
    circle = (x - 0.3) ** 2 + (y - 0.3) ** 2 < 0.02
    img[circle] = (240, 80, 160)
    square = (np.abs(x - 0.7) < 0.1) & (np.abs(y - 0.65) < 0.1)
    img[square] = (120, 220, 60)
    ring = (np.abs(rad - 0.35) < 0.02)
    img[ring] = (200, 160, 230)

    return np.clip(img, 0, 255).astype(np.uint8)


def _parse_noise(noise_level: str | float) -> tuple[str, float]:
    """Resolve a noise level name/lam-notation to (name, value)."""
    noise_name = parse_noise_level(str(noise_level))
    noise_value = NOISE_LEVELS.get(noise_name, 0)
    if noise_value == 0 and noise_name.startswith('Custom_'):
        noise_value = float(noise_name.split('_', 1)[1])
    return noise_name, noise_value


def _add_noise_arr(
    spectra: np.ndarray, noise_value: float, global_max: float | None,
) -> np.ndarray:
    """Poisson noise injection (MLX fused when available)."""
    if noise_value <= 0:
        return spectra
    if HAS_MLX:
        from .spectra_generator import add_poisson_noise_mlx_fused
        noisy = add_poisson_noise_mlx_fused(
            mx.array(spectra), noise_value, global_max=global_max,
        )
        mx.eval(noisy)
        return np.array(noisy, dtype=np.float32)
    from .spectra_generator import add_poisson_noise
    return add_poisson_noise(spectra, noise_value, global_max=global_max)


def _make_multipeak_gen(preset, amplitude_scale: float, bg_fraction: float):
    """Exact Voigt sum generator for per-spectrum multi-peak parameters.

    Returns gen(amps, dEs, dSs) with each arg (n, n_peaks) -> (n, n_energy)
    float32. MLX Faddeeva table when available, scipy.wofz otherwise.
    """
    centers = np.array([e.binding_energy for e in preset.elements], np.float32)
    sigmas = np.array([e.sigma for e in preset.elements], np.float32)
    gamma = float(preset.elements[0].gamma)
    energy = preset.energy

    if HAS_MLX:
        from .faddeeva_mlx import voigt_exact_jacobian_perspectrum_mlx

        def gen(amps, dEs, dSs):
            scaled = mx.array((amps * amplitude_scale).astype(np.float32))
            total = None
            for j in range(len(centers)):
                Phi, _, _, _ = voigt_exact_jacobian_perspectrum_mlx(
                    energy, centers[j] + dEs[:, j], sigmas[j] + dSs[:, j], gamma,
                )
                contrib = scaled[:, j:j + 1] * Phi
                total = contrib if total is None else total + contrib
            total = total + mx.sum(scaled, axis=1, keepdims=True) * bg_fraction
            mx.eval(total)
            return np.array(total, dtype=np.float32)
    else:
        from scipy import special as sps
        SQRT2 = np.sqrt(2.0)
        SQRT2PI = np.sqrt(2.0 * np.pi)

        def gen(amps, dEs, dSs):
            e = energy.astype(np.float64)
            scaled = (amps * amplitude_scale).astype(np.float64)
            total = np.zeros((amps.shape[0], len(e)))
            for j in range(len(centers)):
                c = (centers[j] + dEs[:, j].astype(np.float64))[:, None]
                s = (sigmas[j] + dSs[:, j].astype(np.float64))[:, None]
                z = ((e[None, :] - c) + 1j * gamma) / (s * SQRT2)
                total += scaled[:, j:j + 1] * np.real(sps.wofz(z)) / (s * SQRT2PI)
            total += scaled.sum(axis=1, keepdims=True) * bg_fraction
            return total.astype(np.float32)

    return gen


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation that returns 1.0 for constant arrays instead of NaN."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 1.0 if np.allclose(a, b, atol=1e-6) else 0.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


try:
    import mlx.core as mx

    from ._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND a Metal device works
except ImportError:
    HAS_MLX = False


_MLX_WARMED = False


def warmup_mlx() -> None:
    """Initialize the Metal device from the calling thread (idempotent).

    When MLX is mixed with a Qt event loop, first-time Metal device
    initialization should happen on the main thread before any worker
    thread runs GPU work. Call this from the GUI thread once, before
    starting a background GVRT run.
    """
    global _MLX_WARMED
    if _MLX_WARMED:
        return
    try:
        import mlx.core as mx
        mx.eval(mx.array([1.0]) * 2)
    except ImportError:
        pass
    _MLX_WARMED = True


def _clear_mlx_cache() -> None:
    """Release MLX GPU buffer cache (known buildup pattern in batch loops)."""
    try:
        import mlx.core as mx
        if hasattr(mx, 'clear_cache'):
            mx.clear_cache()
        elif hasattr(mx, 'metal'):
            mx.metal.clear_cache()
    except Exception:
        pass


class GVRTService:
    """Chunked GVRT roundtrip runner with dictionary-cache reuse.

    Stateful only for caching: dictionary caches are keyed by
    (preset name, solver) and built from the preset's full parameter
    ranges, so they are reusable across images.

    Not thread-safe (MLX): run one roundtrip at a time.
    """

    def __init__(self):
        self._dict_caches: dict[tuple[str, str], object] = {}
        self._mp_caches: dict[str, tuple] = {}
        self._last_run: dict | None = None

    @property
    def has_last_run(self) -> bool:
        return self._last_run is not None

    @property
    def last_run_shape(self) -> tuple[int, int] | None:
        """(H, W) of the last completed run, or None."""
        if self._last_run is None:
            return None
        if self._last_run['mode'] == 'multi':
            return self._last_run['image_shape']
        return self._last_run['bench'].image_shape

    def iter_roundtrip(
        self,
        image: np.ndarray,
        config: GVRTConfig | None = None,
        preset: SinglePeakPreset = C1S_SINGLE_PRESET,
    ) -> Generator[GVRTProgress, None, None]:
        """Run the roundtrip, yielding one GVRTProgress per chunk.

        The final frame has .result set. `image` must be (H, W, 3) uint8
        (use prepare_image() first).
        """
        config = config or GVRTConfig()
        if config.n_peaks == 2:
            yield from self._iter_multipeak(image, config)
            return
        if config.n_peaks != 1:
            raise ValueError(f"n_peaks must be 1 or 2, got {config.n_peaks}")
        if config.solver not in GVRT_SOLVERS:
            raise ValueError(
                f"Unknown solver '{config.solver}' (choose from {list(GVRT_SOLVERS)})"
            )
        self._last_run = None  # invalidate pixel inspection during the run

        bench = ParamRoundtripBenchmark(
            image=image,
            preset=preset,
            solver=config.solver,
            bg_fraction=config.bg_fraction,
            amplitude_scale=config.amplitude_scale,
            verbose=False,
            exact_voigt=config.exact_voigt,
        )
        H, W = bench.image_shape
        n = bench.n_pixels

        noise_name = parse_noise_level(str(config.noise_level))
        noise_value = NOISE_LEVELS.get(noise_name, 0)
        if noise_value == 0 and noise_name.startswith('Custom_'):
            noise_value = float(noise_name.split('_', 1)[1])

        global_max = None
        if noise_value > 0:
            max_amp = bench.gt_amp.max() * bench.amplitude_scale
            global_max = max_amp * (1.0 + bench.bg_fraction)

        # Dark canvas: unfitted pixels stay gray so progress is visible
        partial = np.full((n, 3), 40, dtype=np.uint8)

        use_dict = config.solver == 'dict2d_parabola'
        solve_dict = None
        if use_dict:
            from .dictionary_solver import solve_dict2d_parabola
            solve_dict = solve_dict2d_parabola
            if (preset.name, config.solver) not in self._dict_caches:
                yield GVRTProgress(
                    done=0, total=n, image=partial.reshape(H, W, 3).copy(),
                    gen_time=0.0, fit_time=0.0, throughput=0.0,
                    message='Building dictionary...',
                )
            bench._dict_cache = self._get_dict_cache(bench, preset, config.solver)

        use_exact = use_dict or config.solver == 'taylor6' or config.exact_voigt
        if use_exact:
            # MLX exact Voigt (Faddeeva table, ~20x scipy.wofz) when available
            gen_func = (self._make_exact_gen_mlx(bench) if HAS_MLX
                        else bench._generate_spectra_exact)
        else:
            gen_func = bench._generate_spectra_batch

        rec_amp = np.zeros(n, dtype=np.float32)
        rec_dE = np.zeros(n, dtype=np.float32)
        rec_dsigma = np.zeros(n, dtype=np.float32)
        gen_time = 0.0
        fit_time = 0.0
        sigma_nom = preset.element.sigma

        for start in range(0, n, config.chunk_pixels):
            end = min(start + config.chunk_pixels, n)

            t0 = time.perf_counter()
            spectra = gen_func(
                bench.gt_amp[start:end],
                bench.gt_dE[start:end],
                bench.gt_dsigma[start:end],
            )
            spectra = bench._add_noise(spectra, noise_value, global_max)
            gen_time += time.perf_counter() - t0

            t0 = time.perf_counter()
            a_out, dE_out, ds_out = self._solve_spectra(
                bench, config, spectra, solve_dict,
            )
            fit_time += time.perf_counter() - t0

            rec_amp[start:end] = a_out
            rec_dE[start:end] = dE_out
            rec_dsigma[start:end] = ds_out

            # Decode just this chunk into the running canvas
            fwhm_chunk = bench.encoder.fwhm_from_sigma(sigma_nom + ds_out)
            rgb = bench.encoder.decode(
                a_out / bench.amplitude_scale, dE_out, fwhm_chunk, (1, end - start),
            ).reshape(end - start, 3)
            partial[start:end] = rgb

            is_last = end >= n
            result = None
            if is_last:
                result = self._finalize(
                    bench, partial.reshape(H, W, 3).copy(),
                    rec_amp, rec_dE, rec_dsigma,
                    config.solver, noise_name, noise_value, gen_time, fit_time,
                )
                # Keep run state so inspect_pixel() can replay single pixels
                self._last_run = {
                    'mode': 'single',
                    'bench': bench,
                    'config': config,
                    'gen_func': gen_func,
                    'solve_dict': solve_dict,
                    'noise_value': noise_value,
                    'global_max': global_max,
                    'rec_amp': rec_amp,
                    'rec_dE': rec_dE,
                    'rec_dsigma': rec_dsigma,
                }
                _clear_mlx_cache()

            yield GVRTProgress(
                done=end,
                total=n,
                image=partial.reshape(H, W, 3).copy(),
                gen_time=gen_time,
                fit_time=fit_time,
                throughput=end / fit_time if fit_time > 0 else 0.0,
                result=result,
            )

    def _iter_multipeak(
        self,
        image: np.ndarray,
        config: GVRTConfig,
    ) -> Generator[GVRTProgress, None, None]:
        """2-component roundtrip: Hilbert encoder + alternating projection."""
        self._last_run = None
        preset = C1S_MULTIPEAK_PRESET
        encoder = MultiPeakEncoder(preset)
        H, W = image.shape[:2]
        n = H * W

        gt_amps, gt_dEs, gt_dSs = encoder.encode(image)
        noise_name, noise_value = _parse_noise(config.noise_level)
        global_max = None
        if noise_value > 0:
            max_amp = gt_amps.sum(axis=1).max() * config.amplitude_scale
            global_max = max_amp * (1.0 + config.bg_fraction)

        partial = np.full((n, 3), 40, dtype=np.uint8)
        if (preset.name not in self._mp_caches):
            yield GVRTProgress(
                done=0, total=n, image=partial.reshape(H, W, 3).copy(),
                gen_time=0.0, fit_time=0.0, throughput=0.0,
                message='Building multipeak dictionaries...',
            )
        mp_config, dicts = self._get_mp_dicts(preset)
        gen = _make_multipeak_gen(preset, config.amplitude_scale, config.bg_fraction)

        from .multipeak_solver import process_multipeak

        n_comp = preset.n_peaks
        rec_amp = np.zeros((n, n_comp), dtype=np.float32)
        rec_dE = np.zeros((n, n_comp), dtype=np.float32)
        rec_ds = np.zeros((n, n_comp), dtype=np.float32)
        gen_time = 0.0
        fit_time = 0.0

        for start in range(0, n, config.chunk_pixels):
            end = min(start + config.chunk_pixels, n)

            t0 = time.perf_counter()
            Y = gen(gt_amps[start:end], gt_dEs[start:end], gt_dSs[start:end])
            Y = _add_noise_arr(Y, noise_value, global_max)
            gen_time += time.perf_counter() - t0

            t0 = time.perf_counter()
            r = process_multipeak(
                Y, mp_config, dicts=dicts, n_iterations=3,
                parabola_ds=True, auto_constrain=False,
            )
            fit_time += time.perf_counter() - t0

            rec_amp[start:end] = r.amplitudes
            rec_dE[start:end] = r.delta_E
            rec_ds[start:end] = r.delta_sigma

            rgb = encoder.decode(
                r.amplitudes / config.amplitude_scale, r.delta_E, r.delta_sigma,
                (1, end - start),
            ).reshape(end - start, 3)
            partial[start:end] = rgb

            is_last = end >= n
            result = None
            if is_last:
                rec_image = partial.reshape(H, W, 3).copy()
                psnr = compute_per_channel_psnr(image, rec_image)
                gt_amp_scaled = gt_amps * config.amplitude_scale
                result = GVRTResult(
                    n_peaks=n_comp,
                    solver='multipeak_ap',
                    noise_level=noise_name,
                    noise_value=noise_value,
                    image_shape=(H, W),
                    n_spectra=n,
                    gen_time=gen_time,
                    fit_time=fit_time,
                    total_time=gen_time + fit_time,
                    throughput=n / fit_time if fit_time > 0 else float('inf'),
                    psnr=psnr,
                    amp_corr=_safe_corr(gt_amp_scaled.ravel(), rec_amp.ravel()),
                    shift_corr=_safe_corr(gt_dEs.ravel(), rec_dE.ravel()),
                    fwhm_corr=_safe_corr(gt_dSs.ravel(), rec_ds.ravel()),
                    amp_rmse=float(np.sqrt(np.mean((gt_amp_scaled - rec_amp) ** 2))),
                    shift_rmse=float(np.sqrt(np.mean((gt_dEs - rec_dE) ** 2))),
                    fwhm_rmse=float(np.sqrt(np.mean((gt_dSs - rec_ds) ** 2))),
                    original=image,
                    reconstructed=rec_image,
                )
                self._last_run = {
                    'mode': 'multi',
                    'preset': preset,
                    'encoder': encoder,
                    'config': config,
                    'mp_config': mp_config,
                    'dicts': dicts,
                    'gen_func': gen,
                    'image_shape': (H, W),
                    'gt_amps': gt_amps, 'gt_dEs': gt_dEs, 'gt_dSs': gt_dSs,
                    'noise_value': noise_value,
                    'global_max': global_max,
                    'rec_amp': rec_amp, 'rec_dE': rec_dE, 'rec_ds': rec_ds,
                }
                _clear_mlx_cache()

            yield GVRTProgress(
                done=end,
                total=n,
                image=partial.reshape(H, W, 3).copy(),
                gen_time=gen_time,
                fit_time=fit_time,
                throughput=end / fit_time if fit_time > 0 else 0.0,
                result=result,
            )

    def run(
        self,
        image: np.ndarray,
        config: GVRTConfig | None = None,
        preset: SinglePeakPreset = C1S_SINGLE_PRESET,
    ) -> GVRTResult:
        """Run the full roundtrip without intermediate frames."""
        result = None
        for progress in self.iter_roundtrip(image, config, preset):
            result = progress.result
        assert result is not None
        return result

    def _finalize(
        self, bench, rec_image, rec_amp, rec_dE, rec_dsigma,
        solver, noise_name, noise_value, gen_time, fit_time,
    ) -> GVRTResult:
        psnr = compute_per_channel_psnr(bench.image, rec_image)
        gt_amp_scaled = bench.gt_amp * bench.amplitude_scale
        return GVRTResult(
            n_peaks=1,
            solver=solver,
            noise_level=noise_name,
            noise_value=noise_value,
            image_shape=bench.image_shape,
            n_spectra=bench.n_pixels,
            gen_time=gen_time,
            fit_time=fit_time,
            total_time=gen_time + fit_time,
            throughput=bench.n_pixels / fit_time if fit_time > 0 else float('inf'),
            psnr=psnr,
            amp_corr=_safe_corr(gt_amp_scaled, rec_amp),
            shift_corr=_safe_corr(bench.gt_dE, rec_dE),
            fwhm_corr=_safe_corr(bench.gt_dsigma, rec_dsigma),
            amp_rmse=float(np.sqrt(np.mean((gt_amp_scaled - rec_amp) ** 2))),
            shift_rmse=float(np.sqrt(np.mean((bench.gt_dE - rec_dE) ** 2))),
            fwhm_rmse=float(np.sqrt(np.mean((bench.gt_dsigma - rec_dsigma) ** 2))),
            original=bench.image,
            reconstructed=rec_image,
        )

    @staticmethod
    def _solve_spectra(bench, config, spectra, solve_dict):
        """Run the configured solver on a batch; returns (amp, dE, dsigma)."""
        if solve_dict is not None:
            amps, _, dE_out, ds_out, _ = solve_dict(spectra, bench._dict_cache)
            return amps[0, :], dE_out, ds_out
        r = bench.pipeline.process_rowmajor_extended_3param(
            spectra,
            bench.preset.element.symbol,
            bench.preset.element.orbital,
            bench.energy,
            bench.peak_config,
            amplitudes_only=True,
            n_steps=6 if config.solver == 'taylor6' else 4,
            amp_correction=bench.amp_correction,
        )
        return r.amplitudes[0, :], r.energy_shifts, r.sigma_shifts

    def inspect_pixel(self, x: int, y: int) -> GVRTPixelInspection:
        """Replay the roundtrip for one pixel of the last completed run.

        Generates the pixel's clean spectrum, draws a fresh noise
        realization at the run's noise level, fits that exact spectrum
        with the run's solver, and returns everything needed to plot
        measurement vs fit vs ground truth. Each call re-draws noise, so
        repeated clicks show realization-to-realization variability.
        """
        if self._last_run is None:
            raise RuntimeError("No completed run to inspect — run a roundtrip first")
        if self._last_run['mode'] == 'multi':
            return self._inspect_pixel_multi(x, y)
        return self._inspect_pixel_single(x, y)

    def _inspect_pixel_multi(self, x: int, y: int) -> GVRTPixelInspection:
        from .multipeak_solver import process_multipeak
        from .param_encoder import FWHM_TO_SIGMA

        lr = self._last_run
        preset = lr['preset']
        H, W = lr['image_shape']
        x = int(np.clip(x, 0, W - 1))
        y = int(np.clip(y, 0, H - 1))
        idx = y * W + x
        sl = slice(idx, idx + 1)
        scale = lr['config'].amplitude_scale

        clean = lr['gen_func'](lr['gt_amps'][sl], lr['gt_dEs'][sl], lr['gt_dSs'][sl])
        noisy = _add_noise_arr(clean, lr['noise_value'], lr['global_max'])
        r = process_multipeak(
            noisy, lr['mp_config'], dicts=lr['dicts'],
            n_iterations=3, parabola_ds=True, auto_constrain=False,
        )
        fit = lr['gen_func'](r.amplitudes / scale, r.delta_E, r.delta_sigma)

        peaks = []
        for j, elem in enumerate(preset.elements):
            sigma_j = float(elem.sigma)
            peaks.append({
                'center': float(elem.binding_energy),
                'gt_amp': float(lr['gt_amps'][idx, j] * scale),
                'gt_dE': float(lr['gt_dEs'][idx, j]),
                'gt_fwhm': (sigma_j + float(lr['gt_dSs'][idx, j])) / FWHM_TO_SIGMA,
                'fit_amp': float(r.amplitudes[0, j]),
                'fit_dE': float(r.delta_E[0, j]),
                'fit_fwhm': (sigma_j + float(r.delta_sigma[0, j])) / FWHM_TO_SIGMA,
                'batch_amp': float(lr['rec_amp'][idx, j]),
                'batch_dE': float(lr['rec_dE'][idx, j]),
                'batch_fwhm': (sigma_j + float(lr['rec_ds'][idx, j])) / FWHM_TO_SIGMA,
            })

        p0 = peaks[0]
        return GVRTPixelInspection(
            x=x, y=y,
            energy=np.asarray(preset.energy).copy(),
            clean=clean[0], noisy=noisy[0], fit=fit[0],
            peaks=peaks,
            gt_amp=p0['gt_amp'], gt_dE=p0['gt_dE'], gt_fwhm=p0['gt_fwhm'],
            fit_amp=p0['fit_amp'], fit_dE=p0['fit_dE'], fit_fwhm=p0['fit_fwhm'],
            batch_amp=p0['batch_amp'], batch_dE=p0['batch_dE'],
            batch_fwhm=p0['batch_fwhm'],
            residual_rms=float(np.sqrt(np.mean((noisy[0] - fit[0]) ** 2))),
        )

    def _inspect_pixel_single(self, x: int, y: int) -> GVRTPixelInspection:
        lr = self._last_run
        bench = lr['bench']
        H, W = bench.image_shape
        x = int(np.clip(x, 0, W - 1))
        y = int(np.clip(y, 0, H - 1))
        idx = y * W + x
        sl = slice(idx, idx + 1)

        clean = lr['gen_func'](bench.gt_amp[sl], bench.gt_dE[sl], bench.gt_dsigma[sl])
        noisy = bench._add_noise(clean, lr['noise_value'], lr['global_max'])
        a, dE, ds = self._solve_spectra(bench, lr['config'], noisy, lr['solve_dict'])
        fit_amp, fit_dE, fit_ds = float(a[0]), float(dE[0]), float(ds[0])
        fit = lr['gen_func'](
            np.array([fit_amp / bench.amplitude_scale], dtype=np.float32),
            np.array([fit_dE], dtype=np.float32),
            np.array([fit_ds], dtype=np.float32),
        )

        enc = bench.encoder
        sigma_nom = float(bench.preset.element.sigma)
        peak = {
            'center': float(bench.preset.element.binding_energy),
            'gt_amp': float(bench.gt_amp[idx] * bench.amplitude_scale),
            'gt_dE': float(bench.gt_dE[idx]),
            'gt_fwhm': float(bench.gt_fwhm[idx]),
            'fit_amp': fit_amp,
            'fit_dE': fit_dE,
            'fit_fwhm': float(enc.fwhm_from_sigma(sigma_nom + fit_ds)),
            'batch_amp': float(lr['rec_amp'][idx]),
            'batch_dE': float(lr['rec_dE'][idx]),
            'batch_fwhm': float(enc.fwhm_from_sigma(sigma_nom + lr['rec_dsigma'][idx])),
        }
        return GVRTPixelInspection(
            x=x, y=y,
            energy=np.asarray(bench.energy).copy(),
            clean=clean[0],
            noisy=noisy[0],
            fit=fit[0],
            peaks=[peak],
            gt_amp=peak['gt_amp'],
            gt_dE=peak['gt_dE'],
            gt_fwhm=peak['gt_fwhm'],
            fit_amp=peak['fit_amp'],
            fit_dE=peak['fit_dE'],
            fit_fwhm=peak['fit_fwhm'],
            batch_amp=peak['batch_amp'],
            batch_dE=peak['batch_dE'],
            batch_fwhm=peak['batch_fwhm'],
            residual_rms=float(np.sqrt(np.mean((noisy[0] - fit[0]) ** 2))),
        )

    @staticmethod
    def _make_exact_gen_mlx(bench):
        """Exact Voigt generator on GPU with per-spectrum (center, sigma).

        Same math as ParamRoundtripBenchmark._generate_spectra_exact
        (scipy.wofz) but via the MLX Faddeeva table (~20x faster, V accuracy
        ~1e-9). The unused Jacobian outputs are never evaluated (lazy MLX).
        """
        from .faddeeva_mlx import voigt_exact_jacobian_perspectrum_mlx

        center = float(bench.preset.element.binding_energy)
        sigma_nom = float(bench.preset.element.sigma)
        gamma = float(bench.preset.element.gamma)
        energy = bench.energy

        def gen(amp: np.ndarray, dE: np.ndarray, dsigma: np.ndarray) -> np.ndarray:
            Phi, _, _, _ = voigt_exact_jacobian_perspectrum_mlx(
                energy, center + dE, sigma_nom + dsigma, gamma,
            )
            scaled_amp = mx.array((amp * bench.amplitude_scale).astype(np.float32))
            spectra = scaled_amp[:, None] * Phi
            spectra = spectra + (scaled_amp * bench.bg_fraction)[:, None]
            mx.eval(spectra)
            return np.array(spectra, dtype=np.float32)

        return gen

    def _get_mp_dicts(self, preset) -> tuple:
        """(MultiPeakConfig, dictionaries) for a multipeak preset, cached."""
        cached = self._mp_caches.get(preset.name)
        if cached is not None:
            return cached

        from .multipeak_config import ComponentConfig, MultiPeakConfig
        from .multipeak_solver import build_multipeak_dictionaries

        mp_config = MultiPeakConfig(
            peaks=[
                ComponentConfig(
                    center=float(e.binding_energy),
                    sigma=float(e.sigma),
                    gamma=float(e.gamma),
                    dE_range=preset.shift_range.max_val,
                    ds_range=preset.sigma_range.max_val,
                )
                for e in preset.elements
            ],
            energy_axis=preset.energy,
        )
        mp_config.constrain_dE_ranges()
        dicts = build_multipeak_dictionaries(mp_config)
        self._mp_caches[preset.name] = (mp_config, dicts)
        return mp_config, dicts

    def _get_dict_cache(self, bench, preset: SinglePeakPreset, solver: str):
        """Dictionary cache built from preset-wide ranges (image-independent)."""
        key = (preset.name, solver)
        cached = self._dict_caches.get(key)
        if cached is not None:
            return cached

        from .dictionary_solver import build_dictionary

        sigma = preset.element.sigma
        dE_margin = 0.2
        dE_lo = preset.shift_range.min_val - dE_margin
        dE_hi = preset.shift_range.max_val + dE_margin
        # delta-sigma range from the preset's full FWHM range
        ds_bounds = bench.encoder.delta_sigma(
            np.array([preset.fwhm_range.min_val, preset.fwhm_range.max_val],
                     dtype=np.float32)
        )
        ds_margin = 0.02
        ds_lo = float(ds_bounds.min()) - ds_margin
        ds_hi = float(ds_bounds.max()) + ds_margin

        cache = build_dictionary(
            energy=bench.energy,
            centers=bench.peak_config['centers'],
            sigmas=bench.peak_config['sigmas'],
            gamma=bench.peak_config['gamma'],
            dE_range=(dE_lo, dE_hi),
            dE_step=0.1 * sigma,
            dsigma_range=(ds_lo, ds_hi),
            dsigma_step=0.02 * sigma,
        )
        self._dict_caches[key] = cache
        return cache
