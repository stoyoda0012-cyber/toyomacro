"""Error Map Benchmark: Run multiple solvers × noise levels, save per-pixel data.

Wraps ParamRoundtripBenchmark to capture per-pixel parameter arrays
for spatial error analysis. Does NOT modify existing benchmark code.

Usage:
    PYTHONPATH=python python -m voigtfit.benchmarks.error_map_benchmark \
        --image /path/to/image.jpg \
        --solvers 4step 6step dict1d dict2d adaptive \
        --noise-levels None Moderate Strong \
        --output python/outputs/error_maps/

    # Plot from cached results:
    PYTHONPATH=python python -m voigtfit.benchmarks.error_map_benchmark \
        --cache python/outputs/error_maps/error_map_results.npz \
        --param dsigma
Date: 2026-02-21
"""

import argparse
import gc
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from toyomacro.voigtfit.evaluation.benchmark_result import (
    BenchmarkResult,
    load_multi_results,
    save_multi_results,
)
from toyomacro.voigtfit.evaluation.error_stats import compute_error_stats
from toyomacro.voigtfit.image_utils import load_image
from toyomacro.voigtfit.param_encoder import (
    C1S_SINGLE_PRESET,
    SinglePeakEncoder,
    SinglePeakPreset,
)
from toyomacro.voigtfit.pipeline import HybridPipeline
from toyomacro.voigtfit.spectra_generator import NOISE_LEVELS
from toyomacro.voigtfit.weight_cache import WeightMatrixCache

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


# ── Solver name mapping ─────────────────────────────────────────────
# Map our display names to param_roundtrip_benchmark solver strings
SOLVER_MAP = {
    '4step': 'taylor',
    '6step': 'taylor6',
    'dict1d': 'dictionary',
    'dict2d': 'dict2d',
    'adaptive': 'dict2d',  # adaptive uses dict2d cache + special solve
}


def parse_noise_level(s: str) -> str:
    """Parse noise level string, supporting lam notation."""
    if s.startswith('lam'):
        exp = float(s[3:])
        val = 10 ** exp
        for name, nval in NOISE_LEVELS.items():
            if abs(nval - val) / max(val, 1e-10) < 1e-4:
                return name
        return f'Custom_{val:.0f}'
    return s


class ErrorMapBenchmark:
    """Run benchmarks for multiple solvers, capturing per-pixel parameters.

    Re-uses the same spectra across solvers for fair comparison.
    """

    def __init__(
        self,
        image_path: str,
        preset: SinglePeakPreset = C1S_SINGLE_PRESET,
        batch_size: int = 500_000,
        bg_fraction: float = 0.001,
        amplitude_scale: float = 1000.0,
        verbose: bool = True,
    ):
        self.image_path = Path(image_path)
        self.preset = preset
        self.batch_size = batch_size
        self.bg_fraction = bg_fraction
        self.amplitude_scale = amplitude_scale
        self.verbose = verbose

        # Load image and encode GT
        self.image = load_image(self.image_path)
        if self.image.ndim == 2:
            self.image = np.stack([self.image] * 3, axis=-1)
        self.image_shape = self.image.shape[:2]
        self.n_pixels = self.image_shape[0] * self.image_shape[1]

        self.encoder = SinglePeakEncoder(preset)
        self.gt_amp, self.gt_dE, self.gt_fwhm = self.encoder.encode(self.image)
        self.gt_dsigma = self.encoder.delta_sigma(self.gt_fwhm)

        # Build basis
        self.energy = preset.energy
        self.peak_config = {
            'centers': np.array([preset.element.binding_energy], dtype=np.float32),
            'sigmas': np.array([preset.element.sigma], dtype=np.float32),
            'gamma': preset.element.gamma,
        }

        self.cache = WeightMatrixCache()
        self.pipeline = HybridPipeline(
            cache=self.cache,
            enable_shift_correction=True,
            enable_stage2=False,
            use_mlx=True,
        )
        self.Phi, self.J_c, self.J_sigma = \
            self.cache.build_basis_voigt_with_full_jacobian(
                self.energy,
                self.peak_config['centers'],
                self.peak_config['sigmas'],
                self.peak_config['gamma'],
            )

        # Dict caches (lazy, one per grid density)
        self._dict_cache_1d = None
        self._dict_cache_2d = None

        if self.verbose:
            H, W = self.image_shape
            print(f"Image: {self.image_path.name} ({W}x{H}, {self.n_pixels:,} px)")

    def _get_dict_cache(self, is_dict2d: bool):
        """Build or return cached dictionary."""
        attr = '_dict_cache_2d' if is_dict2d else '_dict_cache_1d'
        if getattr(self, attr) is not None:
            return getattr(self, attr)

        from toyomacro.voigtfit.dictionary_solver import build_dictionary

        sigma = self.preset.element.sigma
        dE_margin = 0.2
        dE_lo = self.preset.shift_range.min_val - dE_margin
        dE_hi = self.preset.shift_range.max_val + dE_margin
        ds_min = float(self.gt_dsigma.min())
        ds_max = float(self.gt_dsigma.max())
        ds_margin = 0.02
        ds_lo, ds_hi = ds_min - ds_margin, ds_max + ds_margin

        dE_step = 0.1 * sigma
        dsigma_step = 0.02 * sigma if is_dict2d else 0.1 * sigma

        if self.verbose:
            grid = '2D' if is_dict2d else '1D'
            print(f"Building {grid} dictionary: δE=[{dE_lo:.2f},{dE_hi:.2f}], "
                  f"δσ=[{ds_lo:.3f},{ds_hi:.3f}], step={dsigma_step:.4f}")

        t0 = time.perf_counter()
        dc = build_dictionary(
            energy=self.energy,
            centers=self.peak_config['centers'],
            sigmas=self.peak_config['sigmas'],
            gamma=self.peak_config['gamma'],
            dE_range=(dE_lo, dE_hi),
            dE_step=dE_step,
            dsigma_range=(ds_lo, ds_hi),
            dsigma_step=dsigma_step,
        )
        dt = time.perf_counter() - t0

        if self.verbose:
            print(f"  {dc.n_dE}x{dc.n_dsigma} = {dc.n_dict} entries ({dt:.1f}s)")

        setattr(self, attr, dc)
        return dc

    def _generate_spectra(
        self,
        amp: np.ndarray,
        dE: np.ndarray,
        dsigma: np.ndarray,
        use_exact: bool = True,
    ) -> np.ndarray:
        """Generate spectra (exact Voigt or Taylor)."""
        if use_exact:
            from scipy import special as sps
            energy = self.energy.astype(np.float64)
            center = float(self.preset.element.binding_energy)
            sigma_nom = float(self.preset.element.sigma)
            gamma = float(self.preset.element.gamma)
            SQRT2 = np.sqrt(2.0)
            SQRT2PI = np.sqrt(2.0 * np.pi)

            centers = (center + dE.astype(np.float64))[:, np.newaxis]
            sigmas = (sigma_nom + dsigma.astype(np.float64))[:, np.newaxis]
            z = ((energy[np.newaxis, :] - centers) + 1j * gamma) / (sigmas * SQRT2)
            profiles = np.real(sps.wofz(z)) / (sigmas * SQRT2PI)

            scaled_amp = (amp * self.amplitude_scale).astype(np.float64)
            spectra = scaled_amp[:, np.newaxis] * profiles
            bg = scaled_amp * self.bg_fraction
            spectra += bg[:, np.newaxis]
            return spectra.astype(np.float32)
        else:
            Phi = self.Phi[:, 0]
            J_c = self.J_c[:, 0]
            J_s = self.J_sigma[:, 0]
            profiles = (Phi[np.newaxis, :]
                        + dE[:, np.newaxis] * J_c[np.newaxis, :]
                        + dsigma[:, np.newaxis] * J_s[np.newaxis, :])
            scaled_amp = amp * self.amplitude_scale
            spectra = scaled_amp[:, np.newaxis] * profiles
            bg = scaled_amp * self.bg_fraction
            spectra += bg[:, np.newaxis]
            return spectra.astype(np.float32)

    def _add_noise(self, spectra, noise_value, global_max=None):
        """Add Poisson noise."""
        if noise_value <= 0:
            return spectra
        if HAS_MLX:
            from toyomacro.voigtfit.spectra_generator import add_poisson_noise_mlx_fused
            noisy = add_poisson_noise_mlx_fused(
                mx.array(spectra), noise_value, global_max=global_max)
            mx.eval(noisy)
            return np.array(noisy, dtype=np.float32)
        else:
            from toyomacro.voigtfit.spectra_generator import add_poisson_noise
            return add_poisson_noise(spectra, noise_value, global_max=global_max)

    def _run_solver_chunk(
        self,
        solver_name: str,
        spectra: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Run a single solver on a spectra chunk (no size limit).

        Returns:
            (amp_est, dE_est, dsigma_est, phase2_mask_or_None)
        """
        from toyomacro.voigtfit.dictionary_solver import (
            solve_dict2d_adaptive,
            solve_hybrid_sorted,
        )

        if solver_name == '4step':
            result = self.pipeline.process_rowmajor_extended_3param(
                spectra,
                self.preset.element.symbol,
                self.preset.element.orbital,
                self.energy,
                self.peak_config,
                amplitudes_only=True,
                n_steps=4,
                amp_correction='full',
            )
            return result.amplitudes[0, :], result.energy_shifts, result.sigma_shifts, None

        elif solver_name == '6step':
            result = self.pipeline.process_rowmajor_extended_3param(
                spectra,
                self.preset.element.symbol,
                self.preset.element.orbital,
                self.energy,
                self.peak_config,
                amplitudes_only=True,
                n_steps=6,
                amp_correction='full',
            )
            return result.amplitudes[0, :], result.energy_shifts, result.sigma_shifts, None

        elif solver_name == 'dict1d':
            dc = self._get_dict_cache(is_dict2d=False)
            amps, _, dE, ds, _ = solve_hybrid_sorted(spectra, dc)
            return amps[0, :], dE, ds, None

        elif solver_name == 'dict2d':
            dc = self._get_dict_cache(is_dict2d=True)
            amps, _, dE, ds, _ = solve_hybrid_sorted(spectra, dc)
            return amps[0, :], dE, ds, None

        elif solver_name == 'adaptive':
            dc = self._get_dict_cache(is_dict2d=True)
            amps, chi2, dE, ds, best_idx = solve_dict2d_adaptive(spectra, dc)
            peak_intensity = np.max(spectra, axis=1)
            chi2_norm = chi2 / (peak_intensity ** 2 + 1e-20)
            phase2_mask = chi2_norm < 3e-4
            return amps[0, :], dE, ds, phase2_mask

        else:
            raise ValueError(f"Unknown solver: {solver_name!r}")

    # Max spectra per chunk to keep GPU memory under control.
    # Tuned per-solver to stay within Metal buffer limits:
    #   4-step/6-step: 2M (5 intermediates × 2M × 101 × 4B ≈ 4 GB)
    #   dict solvers: 1M (scores matrix 1M × n_dict × 4B + intermediates)
    _CHUNK_4STEP = 2_000_000
    _CHUNK_DICT = 1_000_000

    def run_solver(
        self,
        solver_name: str,
        spectra: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Run solver with automatic chunking for large inputs.

        Splits spectra into GPU-friendly chunks to avoid Metal OOM
        and swap thrashing on 8K+ images (33M+ pixels).

        Returns:
            (amp_est, dE_est, dsigma_est, phase2_mask_or_None)
        """
        n_spectra = spectra.shape[0]
        if solver_name in ('4step', '6step'):
            chunk = self._CHUNK_4STEP
        else:
            chunk = self._CHUNK_DICT

        if n_spectra <= chunk:
            return self._run_solver_chunk(solver_name, spectra)

        # Chunked path
        amp_parts, dE_parts, ds_parts, mask_parts = [], [], [], []
        for start in range(0, n_spectra, chunk):
            end = min(start + chunk, n_spectra)
            a, d, s, m = self._run_solver_chunk(solver_name, spectra[start:end])
            amp_parts.append(a)
            dE_parts.append(d)
            ds_parts.append(s)
            if m is not None:
                mask_parts.append(m)

        amp_out = np.concatenate(amp_parts)
        dE_out = np.concatenate(dE_parts)
        ds_out = np.concatenate(ds_parts)
        mask_out = np.concatenate(mask_parts) if mask_parts else None
        return amp_out, dE_out, ds_out, mask_out

    def run_all(
        self,
        solver_names: list[str],
        noise_levels: list[str],
        seed: int = 42,
    ) -> dict[str, dict[str, BenchmarkResult]]:
        """Run all solver × noise combinations.

        Uses the same random seed per noise level across solvers
        for fair comparison.

        Returns:
            results[noise_level][solver_name] = BenchmarkResult
        """
        results = {}
        gt_amp_scaled = self.gt_amp * self.amplitude_scale

        for noise_name in noise_levels:
            noise_value = NOISE_LEVELS.get(noise_name, 0)
            if noise_value == 0 and noise_name.startswith('Custom_'):
                noise_value = float(noise_name.split('_', 1)[1])

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Noise: {noise_name} (level={noise_value})")
                print(f"{'='*60}")

            # Generate spectra ONCE per noise level (exact Voigt)
            global_max = None
            if noise_value > 0:
                max_amp = self.gt_amp.max() * self.amplitude_scale
                global_max = max_amp * (1.0 + self.bg_fraction)

            # Generate in batches, concatenate
            all_spectra = []
            for start in range(0, self.n_pixels, self.batch_size):
                end = min(start + self.batch_size, self.n_pixels)
                amp_b = self.gt_amp[start:end]
                dE_b = self.gt_dE[start:end]
                ds_b = self.gt_dsigma[start:end]

                spec = self._generate_spectra(amp_b, dE_b, ds_b, use_exact=True)

                # Set seed per batch for reproducibility across solvers
                if noise_value > 0:
                    np.random.seed(seed + hash(noise_name) % (2**31) + start)
                    if HAS_MLX:
                        mx.random.seed(seed + hash(noise_name) % (2**31) + start)
                    spec = self._add_noise(spec, noise_value, global_max)

                all_spectra.append(spec)
            spectra = np.concatenate(all_spectra, axis=0)
            del all_spectra
            gc.collect()

            results[noise_name] = {}

            for sn in solver_names:
                if self.verbose:
                    print(f"  Solver: {sn} ...", end=' ', flush=True)

                t0 = time.perf_counter()
                amp_est, dE_est, dsigma_est, phase2_mask = \
                    self.run_solver(sn, spectra)
                dt = time.perf_counter() - t0
                throughput = self.n_pixels / dt if dt > 0 else float('inf')

                br = BenchmarkResult(
                    solver_name=sn,
                    noise_level=noise_name,
                    amp_est=amp_est.astype(np.float32),
                    dE_est=dE_est.astype(np.float32),
                    dsigma_est=dsigma_est.astype(np.float32),
                    amp_true=gt_amp_scaled.astype(np.float32),
                    dE_true=self.gt_dE.astype(np.float32),
                    dsigma_true=self.gt_dsigma.astype(np.float32),
                    image_shape=self.image_shape,
                    throughput=throughput,
                    phase2_mask=phase2_mask,
                )
                results[noise_name][sn] = br

                if self.verbose:
                    ds_stats = compute_error_stats(dsigma_est, self.gt_dsigma)
                    dE_stats = compute_error_stats(dE_est, self.gt_dE)
                    print(f"{dt:.1f}s ({throughput/1e6:.1f}M/s)  "
                          f"δσ RMSE={ds_stats.rmse:.4f}  "
                          f"δE RMSE={dE_stats.rmse:.4f}")

            del spectra
            gc.collect()

        return results


def print_summary(results: dict[str, dict[str, BenchmarkResult]]) -> None:
    """Print a summary table of all results."""
    print(f"\n{'='*80}")
    print("ERROR MAP BENCHMARK SUMMARY")
    print(f"{'='*80}")

    noise_levels = list(results.keys())
    solver_names = sorted(set(
        sn for nl_dict in results.values() for sn in nl_dict
    ))

    header = f"{'Noise':<12} {'Solver':<12} {'δσ RMSE':<10} {'δE RMSE':<10} {'amp RMSE':<10} {'M/s':<8}"
    print(header)
    print("-" * 62)

    for nl in noise_levels:
        for sn in solver_names:
            if sn not in results[nl]:
                continue
            br = results[nl][sn]
            ds_stats = compute_error_stats(br.dsigma_est, br.dsigma_true)
            dE_stats = compute_error_stats(br.dE_est, br.dE_true)
            amp_stats = compute_error_stats(br.amp_est, br.amp_true)
            print(f"{nl:<12} {sn:<12} {ds_stats.rmse:<10.4f} {dE_stats.rmse:<10.4f} "
                  f"{amp_stats.rmse:<10.2f} {br.throughput/1e6:<8.1f}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Error Map Benchmark: multi-solver per-pixel error analysis',
    )
    parser.add_argument('--image', type=str, default=None,
                        help='Input image path')
    parser.add_argument('--solvers', nargs='+',
                        default=['4step', '6step', 'dict2d', 'adaptive'],
                        choices=['4step', '6step', 'dict1d', 'dict2d', 'adaptive'],
                        help='Solvers to benchmark')
    parser.add_argument('--noise-levels', nargs='+',
                        default=['None', 'Moderate', 'Strong'],
                        help='Noise levels (names or lam notation)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--cache', type=str, default=None,
                        help='Load cached results from NPZ')
    parser.add_argument('--param', type=str, default='dsigma',
                        choices=['dsigma', 'dE', 'amp'],
                        help='Parameter to visualize')
    parser.add_argument('--clim-mode', type=str, default='per_row',
                        choices=['shared', 'per_row', 'per_panel'],
                        help='Color scale mode')
    parser.add_argument('--hist-style', type=str, default='violin',
                        choices=['violin', 'histogram', 'box'],
                        help='Histogram style')
    parser.add_argument('--batch-size', type=int, default=500_000)
    parser.add_argument('--plot-only', action='store_true',
                        help='Only plot (requires --cache)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')
    parser.add_argument('--adaptive-mask', action='store_true',
                        help='Also generate adaptive mask figure')
    parser.add_argument('--rgb', action='store_true',
                        help='Generate RGB error composite (R=amp, G=δE, B=δσ)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Gamma correction for RGB composite (default 0.5)')

    args = parser.parse_args()

    output_dir = args.output
    if output_dir is None:
        output_dir = str(
            Path(__file__).resolve().parent.parent.parent / 'outputs' / 'error_maps'
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load or compute ─────────────────────────────────────────────
    if args.cache or args.plot_only:
        cache_path = args.cache or str(output_dir / 'error_map_results.npz')
        print(f"Loading cache: {cache_path}")
        results = load_multi_results(cache_path)
    else:
        if args.image is None:
            parser.error('--image is required when not using --cache')

        noise_levels = [parse_noise_level(s) for s in args.noise_levels]

        bench = ErrorMapBenchmark(
            image_path=args.image,
            batch_size=args.batch_size,
        )
        results = bench.run_all(
            solver_names=args.solvers,
            noise_levels=noise_levels,
        )

        print_summary(results)

        # Save
        npz_path = output_dir / 'error_map_results.npz'
        save_multi_results(results, npz_path)
        print(f"Saved: {npz_path}")

    # ── Plot ────────────────────────────────────────────────────────
    if not args.no_plot:
        from toyomacro.voigtfit.visualization.error_maps import plot_error_maps

        fig_path = output_dir / f'error_maps_{args.param}.png'
        plot_error_maps(
            results=results,
            param=args.param,
            clim_mode=args.clim_mode,
            output_path=str(fig_path),
            hist_style=args.hist_style,
        )

    # RGB error composite (independent of --no-plot)
    if args.rgb:
        from toyomacro.voigtfit.visualization.error_maps import plot_rgb_error_maps
        rgb_path = output_dir / 'error_maps_rgb.png'
        fig_rgb = plot_rgb_error_maps(
            results=results,
            output_path=str(rgb_path),
            gamma=args.gamma,
        )
        plt.close(fig_rgb)

    # Adaptive mask figure (independent of --no-plot)
    if not args.no_plot and args.adaptive_mask:
            from toyomacro.voigtfit.visualization.adaptive_mask import plot_adaptive_mask_grid
            adaptive_results = {}
            for nl, solvers in results.items():
                if 'adaptive' in solvers:
                    adaptive_results[nl] = solvers['adaptive']
            if adaptive_results:
                mask_path = output_dir / 'adaptive_mask.png'
                plot_adaptive_mask_grid(
                    adaptive_results,
                    output_path=str(mask_path),
                )


if __name__ == '__main__':
    main()
