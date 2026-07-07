"""
MLX-Accelerated Spectra Generator
=================================

Full GPU-accelerated pipeline for spectra generation and noise injection.
Targets 10-50x speedup over CPU for 8K+ images.

Key optimizations:
1. Voigt profile computation via Faddeeva lookup table on GPU
2. Batched spectra generation with vectorized operations
3. Gaussian-approximation Poisson noise on GPU
4. Single np→mlx and mlx→np transfer per batch

Benchmark targets (Apple M1/M2/M3):
- 8K image (7680x4320 = 33M pixels): < 10s total (was ~110s on CPU)
- Noise injection: < 2s (was ~107s on CPU)
- Spectra generation: < 3s (was ~20s on CPU)
Date: 2026-01-21
"""

import time
from typing import Optional

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

from .faddeeva_mlx import SQRT2, SQRT2PI, get_faddeeva_table


class SpectraGeneratorMLX:
    """
    MLX-accelerated spectra generator.

    Generates Voigt spectra with Poisson noise entirely on GPU.

    Example:
        gen = SpectraGeneratorMLX()
        spectra = gen.generate_batch(
            amplitudes,      # (n_comp, n_spectra)
            energy,          # (n_energy,)
            centers,         # (n_comp,)
            sigmas,          # (n_comp,)
            gamma,           # float
            noise_level=1e4, # Poisson level (0 = no noise)
        )
    """

    def __init__(self):
        if not HAS_MLX:
            raise ImportError("MLX not available")

        # Initialize Faddeeva table (lazy load)
        self._table = None

    def _ensure_table(self):
        """Ensure Faddeeva table is loaded."""
        if self._table is None:
            self._table = get_faddeeva_table()
            self._table._ensure_mlx()

    def voigt_profile_batch(
        self,
        energy: "mx.array",
        center: float,
        sigma: float,
        gamma: float,
    ) -> "mx.array":
        """
        Compute Voigt profile for all energy points.

        Args:
            energy: Energy axis (n_energy,) MLX array
            center: Peak center (scalar)
            sigma: Gaussian width (scalar)
            gamma: Lorentzian width (scalar)

        Returns:
            profile: Voigt profile (n_energy,) MLX array
        """
        self._ensure_table()
        table = self._table

        # Compute x = (E - E0) / (σ√2)
        x = (energy - center) / (sigma * SQRT2)
        y = gamma / (sigma * SQRT2)

        # Find y index (y is scalar)
        iy = np.searchsorted(table.y_grid, y) - 1
        iy = max(0, min(iy, table.ny - 2))
        iy0, iy1 = int(iy), int(iy + 1)

        log_y = np.log(y)
        log_y0 = np.log(table.y_grid[iy0])
        log_y1 = np.log(table.y_grid[iy1])
        fy = float((log_y - log_y0) / (log_y1 - log_y0 + 1e-10))
        fy = max(0.0, min(1.0, fy))

        # Get table rows
        w_r_y0 = table._w_real_mlx[iy0]
        w_r_y1 = table._w_real_mlx[iy1]

        # Compute x indices
        ix = (x - table.x_min) / table.dx
        ix0 = mx.floor(ix).astype(mx.int32)
        ix0 = mx.clip(ix0, 0, table.nx - 2)
        ix1 = ix0 + 1
        fx = ix - ix0.astype(mx.float32)
        fx = mx.clip(fx, 0.0, 1.0)

        # Gather and interpolate
        w00_r = mx.take(w_r_y0, ix0)
        w01_r = mx.take(w_r_y0, ix1)
        w10_r = mx.take(w_r_y1, ix0)
        w11_r = mx.take(w_r_y1, ix1)

        one_minus_fx = 1.0 - fx
        one_minus_fy = 1.0 - fy

        w_r = (one_minus_fx * one_minus_fy * w00_r +
               fx * one_minus_fy * w01_r +
               one_minus_fx * fy * w10_r +
               fx * fy * w11_r)

        # Voigt profile
        profile = w_r / (sigma * SQRT2PI)

        return profile

    def generate_spectra_batch(
        self,
        amplitudes: "mx.array",
        energy: "mx.array",
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float,
        bg_base: float = 0.001,
        amplitude_scale: float = 1e6,
        profiles_cache: Optional["mx.array"] = None,
    ) -> "mx.array":
        """
        Generate Voigt spectra for a batch of amplitudes.

        All computation stays on GPU until result is returned.

        Args:
            amplitudes: Amplitude values (n_comp, n_spectra) MLX array
            energy: Energy axis (n_energy,) MLX array
            centers: Peak centers (n_comp,) numpy array
            sigmas: Gaussian widths (n_comp,) numpy array
            gamma: Lorentzian width (scalar)
            bg_base: Background base level
            amplitude_scale: Amplitude scaling factor
            profiles_cache: Pre-computed profiles (n_comp, n_energy) for reuse

        Returns:
            spectra: Generated spectra (n_spectra, n_energy) MLX array
        """
        n_comp = len(centers)
        n_spectra = amplitudes.shape[1]
        n_energy = energy.shape[0]

        # Use cached profiles or compute them
        if profiles_cache is not None:
            profiles_stack = profiles_cache
        else:
            # Pre-compute normalized Voigt profiles for each component
            profiles = []
            for j in range(n_comp):
                p = self.voigt_profile_batch(energy, centers[j], sigmas[j], gamma)
                p_max = mx.max(p)
                p = p / (p_max + 1e-10)
                profiles.append(p)
            profiles_stack = mx.stack(profiles, axis=0)

        # Compute spectra: (n_spectra, n_energy) = (n_spectra, n_comp) @ (n_comp, n_energy)
        spectra = mx.matmul(amplitudes.T, profiles_stack) * amplitude_scale

        # Add background: bg = total_amplitude * scale * bg_base
        total_amp = mx.sum(amplitudes, axis=0)  # (n_spectra,)
        bg = total_amp[:, None] * amplitude_scale * bg_base
        spectra = spectra + bg

        return spectra

    def precompute_profiles(
        self,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float,
    ) -> "mx.array":
        """
        Pre-compute Voigt profiles for caching.

        Call this once per element, then pass to generate_spectra_batch.

        Args:
            energy: Energy axis (n_energy,) numpy array
            centers: Peak centers (n_comp,) numpy array
            sigmas: Gaussian widths (n_comp,) numpy array
            gamma: Lorentzian width (scalar)

        Returns:
            profiles: (n_comp, n_energy) MLX array
        """
        energy_mx = mx.array(energy.astype(np.float32))
        profiles = []
        for j in range(len(centers)):
            p = self.voigt_profile_batch(energy_mx, centers[j], sigmas[j], gamma)
            p_max = mx.max(p)
            p = p / (p_max + 1e-10)
            profiles.append(p)
        profiles_stack = mx.stack(profiles, axis=0)
        mx.eval(profiles_stack)
        return profiles_stack

    def add_poisson_noise_batch(
        self,
        spectra: "mx.array",
        level: float,
        use_gaussian_approx: bool = True,
    ) -> "mx.array":
        """
        Add Poisson noise to spectra batch on GPU.

        Uses Cornish-Fisher approximation: Gaussian + skewness correction.
        z_cf = z + (1/√λ)·(z²-1)/6 restores Poisson skewness at ~10% cost.

        Args:
            spectra: Input spectra (n_spectra, n_energy) MLX array
            level: Noise level (1 to 1e6, higher = more noise)
                   SNR = 10000 / level
            use_gaussian_approx: Use Gaussian approximation (default True)

        Returns:
            noisy_spectra: Noisy spectra (n_spectra, n_energy) MLX array
        """
        if level <= 0:
            return spectra

        # Compute scaling
        # SNR = 10000 / level
        # λ = SNR² * normalized_signal
        snr_target = 10000.0 / level
        lambda_scale = snr_target * snr_target

        # Get max for normalization
        spectra_pos = mx.maximum(spectra, 0)
        spectra_max = mx.max(spectra_pos)

        if float(mx.max(spectra_max)) == 0:
            return spectra

        # Normalize to [0, 1] and scale
        normalized = spectra_pos / spectra_max
        scaled = lambda_scale * normalized

        # Clip to avoid overflow
        scaled = mx.clip(scaled, 0, 1e9)

        if use_gaussian_approx:
            # Cornish-Fisher approximation: Gaussian + skewness correction
            std = mx.sqrt(mx.maximum(scaled, 1.0))
            z = mx.random.normal(shape=scaled.shape, dtype=mx.float32)
            gamma1 = 1.0 / std
            z = z + gamma1 * (z * z - 1.0) / 6.0
            noisy = mx.maximum(scaled + std * z, 0)
        else:
            # Fall back to CPU for exact Poisson (rarely needed)
            mx.eval(scaled)
            scaled_np = np.array(scaled)
            noisy_np = np.random.poisson(scaled_np).astype(np.float32)
            noisy = mx.array(noisy_np)

        # Scale back
        result = noisy / lambda_scale * spectra_max

        return result

    def generate_batch(
        self,
        amplitudes: np.ndarray,
        energy: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        gamma: float,
        noise_level: float = 0.0,
        bg_base: float = 0.001,
        amplitude_scale: float = 1e6,
        profiles_cache: Optional["mx.array"] = None,
    ) -> np.ndarray:
        """
        Full pipeline: generate spectra with optional noise.

        Single np→mlx transfer at start, single mlx→np at end.

        Args:
            amplitudes: Amplitude values (n_comp, n_spectra) numpy array
            energy: Energy axis (n_energy,) numpy array
            centers: Peak centers (n_comp,) numpy array
            sigmas: Gaussian widths (n_comp,) numpy array
            gamma: Lorentzian width (scalar)
            noise_level: Poisson noise level (0 = no noise)
            bg_base: Background base level
            amplitude_scale: Amplitude scaling factor
            profiles_cache: Pre-computed profiles (n_comp, n_energy) MLX array

        Returns:
            spectra: Generated spectra (n_spectra, n_energy) numpy array
        """
        # Transfer to GPU (single transfer)
        amp_mx = mx.array(amplitudes.astype(np.float32))
        energy_mx = mx.array(energy.astype(np.float32))

        # Generate spectra on GPU
        spectra_mx = self.generate_spectra_batch(
            amp_mx, energy_mx, centers, sigmas, gamma,
            bg_base=bg_base, amplitude_scale=amplitude_scale,
            profiles_cache=profiles_cache,
        )

        # Add noise on GPU
        if noise_level > 0:
            spectra_mx = self.add_poisson_noise_batch(spectra_mx, noise_level)

        # Single eval and transfer back
        mx.eval(spectra_mx)
        return np.array(spectra_mx, dtype=np.float32)


def generate_element_spectra_mlx(
    amplitudes_dict: dict[str, np.ndarray],
    element_configs: dict[str, dict],
    noise_level: float = 0.0,
    batch_size: int = 500_000,
    verbose: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Generate spectra for multiple elements using MLX.

    Processes in batches to manage GPU memory.

    Args:
        amplitudes_dict: Dict of element_key -> amplitudes (n_comp, n_spectra)
        element_configs: Dict of element_key -> {
            'energy': np.ndarray (n_energy,),
            'centers': np.ndarray (n_comp,),
            'sigmas': np.ndarray (n_comp,),
            'gamma': float,
        }
        noise_level: Poisson noise level (0 = no noise)
        batch_size: Spectra per batch for memory management
        verbose: Print progress

    Returns:
        Dict of element_key -> (spectra, energy)
            spectra: (n_spectra, n_energy)
            energy: (n_energy,)
    """
    if not HAS_MLX:
        raise ImportError("MLX not available")

    gen = SpectraGeneratorMLX()
    results = {}

    for elem_key, config in element_configs.items():
        amplitudes = amplitudes_dict[elem_key]
        energy = config['energy']
        centers = config['centers']
        sigmas = config['sigmas']
        gamma = config['gamma']

        n_comp, n_spectra = amplitudes.shape
        n_batches = (n_spectra + batch_size - 1) // batch_size

        if verbose:
            print(f"  {elem_key}: {n_spectra:,} spectra, {n_batches} batches")

        # Pre-allocate output
        spectra_all = np.zeros((n_spectra, len(energy)), dtype=np.float32)

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_spectra)
            batch_slice = slice(batch_start, batch_end)

            # Generate batch
            amp_batch = amplitudes[:, batch_slice]
            spectra_batch = gen.generate_batch(
                amp_batch, energy, centers, sigmas, gamma,
                noise_level=noise_level,
            )

            spectra_all[batch_slice, :] = spectra_batch

            if verbose and n_batches > 1 and batch_idx % 10 == 0:
                progress = (batch_idx + 1) / n_batches * 100
                print(f"    {progress:.0f}%")

        results[elem_key] = (spectra_all, energy)

    return results


def benchmark_spectra_generator_mlx():
    """Benchmark MLX spectra generator performance."""
    print("=" * 70)
    print("MLX Spectra Generator Benchmark")
    print("=" * 70)

    if not HAS_MLX:
        print("MLX not available")
        return

    from .spectra_generator import add_poisson_noise, voigt_profile

    # Setup
    n_energy = 101
    energy = np.linspace(95, 105, n_energy).astype(np.float32)
    centers = np.array([99.0, 103.0], dtype=np.float32)
    sigmas = np.array([0.8, 0.9], dtype=np.float32)
    gamma = 0.25

    gen = SpectraGeneratorMLX()

    print("\n1. Voigt Profile Generation")
    print("-" * 50)

    # Compare accuracy
    energy_mx = mx.array(energy)

    for c, s in zip(centers, sigmas):
        # NumPy reference
        profile_np = voigt_profile(energy, c, s, gamma)
        profile_np = profile_np / (profile_np.max() + 1e-10)

        # MLX
        profile_mx = gen.voigt_profile_batch(energy_mx, float(c), float(s), gamma)
        mx.eval(profile_mx)
        profile_mlx = np.array(profile_mx)
        profile_mlx = profile_mlx / (profile_mlx.max() + 1e-10)

        max_err = np.max(np.abs(profile_np - profile_mlx))
        print(f"  c={c}, σ={s}: max_err = {max_err:.2e}")

    print("\n2. Full Pipeline Benchmark (with profile caching)")
    print("-" * 50)

    # Pre-compute profiles (one-time cost)
    profiles_cache = gen.precompute_profiles(energy, centers, sigmas, gamma)
    print(f"  Profile cache shape: {profiles_cache.shape}")

    for n_spectra in [10_000, 100_000, 1_000_000]:
        # Random amplitudes
        np.random.seed(42)
        amplitudes = np.random.rand(2, n_spectra).astype(np.float32)

        # Warm up
        _ = gen.generate_batch(
            amplitudes[:, :1000], energy, centers, sigmas, gamma,
            noise_level=1e4, profiles_cache=profiles_cache
        )

        # Benchmark MLX with cache
        t0 = time.perf_counter()
        spectra_mlx = gen.generate_batch(
            amplitudes, energy, centers, sigmas, gamma,
            noise_level=1e4, profiles_cache=profiles_cache
        )
        t_mlx = time.perf_counter() - t0

        throughput = n_spectra / t_mlx
        print(f"  {n_spectra:>10,} spectra: {t_mlx*1000:.1f}ms ({throughput/1e6:.2f}M spec/s)")

    print("\n3. Noise Injection Benchmark")
    print("-" * 50)

    n_spectra = 1_000_000
    spectra_clean = np.random.rand(n_spectra, n_energy).astype(np.float32) * 1e6

    for noise_level in [1, 1e2, 1e4, 1e6]:
        # NumPy reference
        t0 = time.perf_counter()
        noisy_np = add_poisson_noise(spectra_clean.copy(), noise_level)
        t_np = time.perf_counter() - t0

        # MLX
        spectra_mx = mx.array(spectra_clean)
        t0 = time.perf_counter()
        noisy_mx = gen.add_poisson_noise_batch(spectra_mx, noise_level)
        mx.eval(noisy_mx)
        t_mlx = time.perf_counter() - t0

        speedup = t_np / t_mlx
        print(f"  level={noise_level:.0e}: NumPy {t_np*1000:.1f}ms, MLX {t_mlx*1000:.1f}ms ({speedup:.1f}x)")

    print("\n4. 8K Image Simulation (with profile caching)")
    print("-" * 50)

    # Simulate 8K image processing
    n_pixels = 7680 * 4320  # 33M pixels
    n_elements = 5
    n_comp_per_elem = 2
    batch_size = 1_000_000  # Process in 1M batches for memory efficiency

    print(f"  Simulating {n_pixels:,} pixels ({n_elements} elements)")
    print(f"  Batch size: {batch_size:,}")

    # Pre-compute profiles (shared across batches)
    profiles_cache = gen.precompute_profiles(energy, centers[:n_comp_per_elem],
                                             sigmas[:n_comp_per_elem], gamma)

    total_time = 0
    for elem_idx in range(n_elements):
        amplitudes = np.random.rand(n_comp_per_elem, n_pixels).astype(np.float32)
        n_batches = (n_pixels + batch_size - 1) // batch_size

        t0 = time.perf_counter()
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_pixels)
            amp_batch = amplitudes[:, start:end]

            spectra = gen.generate_batch(
                amp_batch, energy, centers[:n_comp_per_elem],
                sigmas[:n_comp_per_elem], gamma,
                noise_level=1e4, profiles_cache=profiles_cache
            )
        t_elem = time.perf_counter() - t0
        total_time += t_elem

        print(f"    Element {elem_idx+1}: {t_elem:.1f}s ({n_pixels/t_elem/1e6:.2f}M spec/s)")

    print(f"\n  Total: {total_time:.1f}s")
    print(f"  Throughput: {n_pixels * n_elements / total_time / 1e6:.2f}M spec/s")

    print("\n✓ Benchmark complete")


if __name__ == "__main__":
    benchmark_spectra_generator_mlx()
