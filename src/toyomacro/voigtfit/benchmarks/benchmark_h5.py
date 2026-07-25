"""
Unified HDF5 format for benchmark data.

Consolidates all noise levels and elements into a single file for
easier management and analysis.

Structure:
    benchmark.h5
    ├── /metadata
    │   ├── image_shape: (H, W)
    │   ├── n_elements: int
    │   ├── element_keys: ['Si1s', 'O1s', ...]
    │   ├── color_mapping: (3, n_components)
    │   ├── component_order: [('Si1s', 0), ...]
    │   └── source_image: str
    │
    ├── /elements
    │   ├── /Si1s
    │   │   ├── energy: (n_energy,)
    │   │   ├── centers: (n_comp,)
    │   │   ├── sigmas: (n_comp,)
    │   │   └── gamma: float
    │   ├── /O1s
    │   │   └── ...
    │   └── ...
    │
    ├── /conditions
    │   ├── /noise_none
    │   │   ├── noise_type: 'none'
    │   │   ├── poisson_level: 0
    │   │   ├── gaussian_std: 0
    │   │   │
    │   │   ├── /spectra
    │   │   │   ├── /Si1s: (n_spectra, n_energy)
    │   │   │   ├── /O1s: (n_spectra, n_energy)
    │   │   │   └── ...
    │   │   │
    │   │   ├── /fitpara
    │   │   │   ├── /Si1s: (n_spectra, n_comp, 9)
    │   │   │   └── ...
    │   │   │
    │   │   └── /results
    │   │       ├── amplitudes: dict of (n_comp, n_spectra)
    │   │       ├── reconstructed_image: (H, W, 3)
    │   │       ├── psnr_total: float
    │   │       ├── psnr_rgb: (3,)
    │   │       └── processing_time: float
    │   │
    │   ├── /poisson_1e4
    │   │   └── ...
    │   └── ...
    │
    └── /summary
        ├── noise_levels: [0, 1e4, 1e6, ...]
        ├── psnr_values: [40.7, 40.6, ...]
        └── snr_estimates: [inf, 100, 10, ...]
Date: 2026-01-22
"""

import gc
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..spectra_generator import (
    HAS_MLX,
    ElementSpec,
    GeneratorConfig,
    NoiseConfig,
    add_gaussian_noise,
    add_poisson_noise,
    add_poisson_noise_mlx,
    decompose_image_to_amplitudes,
    voigt_profile,
)

# MLX-accelerated spectra generator (optional)
try:
    from ..spectra_generator_mlx import SpectraGeneratorMLX
    HAS_MLX_GENERATOR = HAS_MLX
except ImportError:
    HAS_MLX_GENERATOR = False
    SpectraGeneratorMLX = None

from ..image_utils import (
    amplitudes_to_rgb,
    compare_images,
    get_demo_color_mapping,
    load_image,
)
from ..pipeline import HybridPipeline
from ..weight_cache import WeightMatrixCache


def get_available_memory_gb() -> float:
    """
    Get available physical memory in GB.

    Works on macOS, Linux, and Windows.
    Based on MATLAB DepthProfiler.m line 4018-4022.
    """
    system = platform.system()

    if system == 'Darwin':  # macOS
        try:
            # Get total memory from sysctl
            result = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True, text=True
            )
            total_bytes = int(result.stdout.strip())

            # Get free memory from vm_stat
            result = subprocess.run(['vm_stat'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            page_size = 4096  # Default page size

            free_pages = 0
            inactive_pages = 0
            for line in lines:
                if 'page size of' in line:
                    page_size = int(line.split()[-2])
                elif 'Pages free' in line:
                    free_pages = int(line.split()[-1].rstrip('.'))
                elif 'Pages inactive' in line:
                    inactive_pages = int(line.split()[-1].rstrip('.'))

            # Available = free + inactive (can be reclaimed)
            available_bytes = (free_pages + inactive_pages) * page_size
            return available_bytes / (1024**3)
        except Exception:
            # Fallback: assume 8GB available
            return 8.0

    elif system == 'Linux':
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        kb = int(line.split()[1])
                        return kb / (1024**2)
            return 8.0
        except Exception:
            return 8.0

    elif system == 'Windows':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', c_ulonglong),
                    ('ullAvailPhys', c_ulonglong),
                    ('ullTotalPageFile', c_ulonglong),
                    ('ullAvailPageFile', c_ulonglong),
                    ('ullTotalVirtual', c_ulonglong),
                    ('ullAvailVirtual', c_ulonglong),
                    ('ullAvailExtendedVirtual', c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys / (1024**3)
        except Exception:
            return 8.0

    return 8.0  # Default fallback


def calculate_batch_size(
    n_pixels: int,
    n_energy: int,
    n_components: int = 2,
    available_memory_gb: float = None,
    memory_fraction: float = 0.5,
) -> int:
    """
    Calculate optimal batch size based on available memory.

    Based on MATLAB DepthProfiler.m division logic.

    Args:
        n_pixels: Total number of pixels/spectra
        n_energy: Number of energy points per spectrum
        n_components: Max components per element
        available_memory_gb: Available memory (auto-detected if None)
        memory_fraction: Fraction of available memory to use (default 50%)

    Returns:
        Optimal batch size (number of spectra per batch)
    """
    if available_memory_gb is None:
        available_memory_gb = get_available_memory_gb()

    # Memory per spectrum (float32):
    # - spectra: n_energy * 4 bytes
    # - fitpara: n_components * 9 * 4 bytes
    # - working buffers: ~2x overhead
    bytes_per_spectrum = (n_energy * 4 + n_components * 9 * 4) * 2

    # Available bytes for batch
    available_bytes = available_memory_gb * memory_fraction * (1024**3)

    # Calculate batch size
    batch_size = int(available_bytes / bytes_per_spectrum)

    # Clamp to reasonable range
    # Max 1M spectra per batch to avoid memory fragmentation
    max_batch = 1_000_000
    batch_size = max(10_000, min(batch_size, max_batch, n_pixels))

    return batch_size


def calculate_optimal_chunk_size(
    n_pixels: int,
    n_energy: int,
    n_components: int = 2,
    target_chunk_mb: float = 1.0,
    max_chunk_rows: int = 1_000_000,
) -> tuple[tuple[int, int], tuple[int, int, int]]:
    """
    Calculate optimal HDF5 chunk sizes for spectra and fitpara datasets.

    HDF5 chunking considerations:
    1. Chunk size affects compression efficiency and I/O performance
    2. Target ~1MB chunks for good cache utilization (HDF5 default cache = 1MB)
    3. Align chunks with typical access patterns:
       - VoigtFit reads entire rows (1 spectrum at a time)
       - Batch processing writes multiple rows at once
    4. MATLAB DepthProfiler uses max 1M rows per chunk

    Args:
        n_pixels: Total number of pixels/spectra
        n_energy: Number of energy points per spectrum
        n_components: Max components per element
        target_chunk_mb: Target chunk size in MB (default 1.0)
        max_chunk_rows: Maximum rows per chunk (MATLAB uses 1M)

    Returns:
        Tuple of (spectra_chunk_shape, fitpara_chunk_shape)
    """
    target_bytes = target_chunk_mb * 1024 * 1024

    # Spectra: (n_pixels, n_energy), float32
    # Always include full energy axis for row-wise access
    bytes_per_row = n_energy * 4
    spectra_rows = min(
        max(1, int(target_bytes / bytes_per_row)),
        max_chunk_rows,
        n_pixels
    )
    spectra_chunk = (spectra_rows, n_energy)

    # Fitpara: (n_pixels, n_components, 9), float32
    # Keep full component and parameter dimensions
    bytes_per_fitpara_row = n_components * 9 * 4
    fitpara_rows = min(
        max(1, int(target_bytes / bytes_per_fitpara_row)),
        max_chunk_rows,
        n_pixels
    )
    fitpara_chunk = (fitpara_rows, n_components, 9)

    return spectra_chunk, fitpara_chunk

# Type alias for denoise function
# denoise_fn(spectra: np.ndarray, energy: np.ndarray, noise_level: float) -> np.ndarray
DenoiseFunction = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


@dataclass
class BenchmarkH5Config:
    """Configuration for unified benchmark H5 file."""
    # Noise sweep
    poisson_levels: list[float] = field(default_factory=lambda: [0, 1e4, 1e6, 1e8])
    gaussian_std: float = 0.0

    # Processing
    batch_size: int = 100_000
    enable_stage2: bool = False
    use_mlx_noise: bool = True  # Use MLX for noise generation (~30x faster)
    use_mlx_generator: bool = True  # Use full MLX pipeline for spectra generation (~10x faster)

    # Denoise hook (optional)
    # Function signature: denoise_fn(spectra, energy, noise_level) -> denoised_spectra
    # Set to None to disable denoising
    denoise_fn: DenoiseFunction | None = None
    denoise_name: str = 'custom'  # Name for results labeling

    # Storage (compression=None for speed, 'gzip' for smaller files)
    compression: str | None = None  # None=fastest (~6GB/s), 'gzip'=slow (~30MB/s)
    compression_opts: int | None = None
    store_spectra: bool = True  # Can disable to save space
    store_reconstructed: bool = True
    store_denoised: bool = True  # Store denoised spectra if denoise_fn is set


def noise_key(noise_config: NoiseConfig) -> str:
    """Generate consistent key for noise configuration."""
    if noise_config.noise_type == 'none':
        return 'noise_none'
    elif noise_config.noise_type == 'poisson':
        return f'poisson_{noise_config.poisson_level:.0e}'
    elif noise_config.noise_type == 'gaussian':
        return f'gaussian_{noise_config.gaussian_std:.0e}'
    else:
        return f'mixed_p{noise_config.poisson_level:.0e}_g{noise_config.gaussian_std:.0e}'


class BenchmarkH5:
    """
    Unified HDF5 file for benchmark data.

    Example:
        # Create benchmark
        bench = BenchmarkH5.create(
            output_path='benchmark.h5',
            image_path='input.jpg',
            config=BenchmarkH5Config(poisson_levels=[0, 1e4, 1e8]),
        )
        bench.run_all()
        bench.close()

        # Read results
        with BenchmarkH5.open('benchmark.h5') as bench:
            summary = bench.get_summary()
            print(summary['psnr_values'])
    """

    def __init__(self, h5file: h5py.File, mode: str = 'r'):
        self.h5 = h5file
        self.mode = mode

    @classmethod
    def create(
        cls,
        output_path: str | Path,
        image_path: str | Path,
        config: BenchmarkH5Config | None = None,
        elements: list[ElementSpec] | None = None,
    ) -> 'BenchmarkH5':
        """
        Create new benchmark H5 file.

        Args:
            output_path: Path for output H5 file
            image_path: Source image path
            config: Benchmark configuration
            elements: Element specifications (uses the demo preset if None)
        """
        config = config or BenchmarkH5Config()
        output_path = Path(output_path)

        # Load image
        image = load_image(image_path)
        height, width = image.shape[:2]
        n_pixels = height * width

        # Default elements (6-component demo preset)
        if elements is None:
            elements = [
                ElementSpec('Si', '1s', 1839.1, sigma=1.02, gamma=0.25),
                ElementSpec('Si', '1s', 1843.8, sigma=1.09, gamma=0.25),
                ElementSpec('Ti', '1s', 4968.7, sigma=0.87, gamma=0.25),
                ElementSpec('Al', '1s', 1561.8, sigma=1.02, gamma=0.25),
                ElementSpec('C', '1s', 284.4, sigma=1.01, gamma=0.25),
                ElementSpec('O', '1s', 531.0, sigma=1.02, gamma=0.25),
            ]

        # Color mapping
        color_mapping, component_order = get_demo_color_mapping()

        # Create H5 file
        h5 = h5py.File(output_path, 'w')

        # Store metadata
        meta = h5.create_group('metadata')
        meta.attrs['image_shape'] = (height, width)
        meta.attrs['n_pixels'] = n_pixels
        meta.attrs['n_elements'] = len(elements)
        meta.attrs['source_image'] = str(image_path)
        meta.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')

        # Store original image (compressed)
        meta.create_dataset('original_image', data=image,
                           compression=config.compression,
                           compression_opts=config.compression_opts)

        # Color mapping
        meta.create_dataset('color_mapping', data=color_mapping)

        # Component order as string array
        comp_str = [f'{k}:{i}' for k, i in component_order]
        dt = h5py.special_dtype(vlen=str)
        meta.create_dataset('component_order', data=np.array(comp_str, dtype=object), dtype=dt)

        # Element configurations - group by file_key while preserving order
        elem_grp = h5.create_group('elements')
        element_keys_ordered = []  # Maintain order
        file_key_data = {}  # Collect data per file_key

        for elem in elements:
            key = elem.file_key
            if key not in file_key_data:
                element_keys_ordered.append(key)
                file_key_data[key] = {
                    'symbol': elem.symbol,
                    'orbital': elem.orbital,
                    'gamma': elem.gamma,
                    'centers': [],
                    'sigmas': [],
                }
            file_key_data[key]['centers'].append(elem.binding_energy)
            file_key_data[key]['sigmas'].append(elem.sigma)

        # Create groups with collected data
        for key in element_keys_ordered:
            data = file_key_data[key]
            eg = elem_grp.create_group(key)
            eg.attrs['symbol'] = data['symbol']
            eg.attrs['orbital'] = data['orbital']
            eg.attrs['gamma'] = data['gamma']
            eg.create_dataset('centers', data=np.array(data['centers'], dtype='float32'))
            eg.create_dataset('sigmas', data=np.array(data['sigmas'], dtype='float32'))

        # Store element keys (ordered)
        meta.create_dataset('element_keys', data=np.array(element_keys_ordered, dtype=object), dtype=dt)

        # Create conditions group
        h5.create_group('conditions')

        # Create summary group
        h5.create_group('summary')

        # Store instance data
        obj = cls(h5, mode='w')
        obj.config = config
        obj.image = image
        obj.elements = elements
        obj.color_mapping = color_mapping
        obj.component_order = component_order
        obj.output_path = output_path

        return obj

    @classmethod
    def open(cls, path: str | Path, mode: str = 'r') -> 'BenchmarkH5':
        """Open existing benchmark H5 file."""
        h5 = h5py.File(path, mode)
        obj = cls(h5, mode=mode)

        # Load metadata
        meta = h5['metadata']
        obj.image = meta['original_image'][:]
        obj.color_mapping = meta['color_mapping'][:]

        # Parse component order
        comp_strs = meta['component_order'][:]
        obj.component_order = []
        for s in comp_strs:
            if isinstance(s, bytes):
                s = s.decode('utf-8')
            key, idx = s.split(':')
            obj.component_order.append((key, int(idx)))

        return obj

    def close(self):
        """Close H5 file."""
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def add_condition(
        self,
        noise_config: NoiseConfig,
        verbose: bool = True,
    ) -> str:
        """
        Add a noise condition: generate spectra, run fitting, store results.

        Returns:
            Key for the condition group
        """
        key = noise_key(noise_config)

        if verbose:
            print(f'\n{"="*60}')
            print(f'Adding condition: {key}')
            print(f'{"="*60}')

        # Get metadata
        meta = self.h5['metadata']
        height, width = meta.attrs['image_shape']
        n_pixels = height * width

        # Create condition group
        if key in self.h5['conditions']:
            del self.h5['conditions'][key]

        cond = self.h5['conditions'].create_group(key)
        cond.attrs['noise_type'] = noise_config.noise_type
        cond.attrs['poisson_level'] = noise_config.poisson_level
        cond.attrs['gaussian_std'] = noise_config.gaussian_std

        # Decompose image to amplitudes
        if verbose:
            print('Decomposing image...')

        amplitudes_raw = decompose_image_to_amplitudes(
            self.image, self.color_mapping, method='pinv'
        )

        # Generate spectra for each element
        if verbose:
            use_mlx_gen = self.config.use_mlx_generator and HAS_MLX_GENERATOR
            print(f'Generating spectra... (MLX generator: {use_mlx_gen})')

        spectra_grp = cond.create_group('spectra')
        fitpara_grp = cond.create_group('fitpara')

        gen_config = GeneratorConfig()
        gen_start = time.perf_counter()

        elem_grp = self.h5['elements']
        element_keys = [k.decode() if isinstance(k, bytes) else k
                       for k in meta['element_keys'][:]]

        # Build mapping from (file_key, comp_idx) to amplitude row index
        # component_order defines the order of amplitudes_raw rows
        comp_to_amp_idx = {(fk, ci): i for i, (fk, ci) in enumerate(self.component_order)}

        # Initialize MLX generator if enabled
        mlx_gen = None
        if self.config.use_mlx_generator and HAS_MLX_GENERATOR:
            mlx_gen = SpectraGeneratorMLX()

        for file_key in element_keys:
            eg = elem_grp[file_key]
            centers = eg['centers'][:]
            sigmas = eg['sigmas'][:]
            gamma = eg.attrs['gamma']
            n_comp = len(centers)

            # Energy axis
            center_mean = np.mean(centers)
            energy = np.arange(
                center_mean - gen_config.energy_edge,
                center_mean + gen_config.energy_edge + gen_config.energy_step,
                gen_config.energy_step,
                dtype=np.float32
            )
            n_energy = len(energy)

            # Get amplitude indices for this element's components
            comp_amp_indices = []
            for j in range(n_comp):
                amp_idx = comp_to_amp_idx.get((file_key, j))
                if amp_idx is None:
                    raise ValueError(f"Component ({file_key}, {j}) not found in component_order")
                comp_amp_indices.append(amp_idx)

            # Pre-compute Voigt profiles (only needed for CPU path)
            profiles = None
            if mlx_gen is None:
                profiles = []
                for j in range(n_comp):
                    profile = voigt_profile(energy, centers[j], sigmas[j], gamma)
                    profile = profile / (profile.max() + 1e-10)
                    profiles.append(profile)

            # Calculate optimal batch size based on available memory
            batch_size = calculate_batch_size(n_pixels, n_energy, n_comp)
            n_batches = (n_pixels + batch_size - 1) // batch_size

            # Calculate optimal chunk sizes for HDF5 (target ~1MB chunks)
            spectra_chunk, fitpara_chunk = calculate_optimal_chunk_size(
                n_pixels, n_energy, n_comp
            )

            if verbose and n_batches > 1:
                mem_gb = get_available_memory_gb()
                backend = 'MLX' if mlx_gen else 'CPU'
                print(f'  {file_key}: {n_batches} batches (batch={batch_size:,}, '
                      f'chunk={spectra_chunk[0]:,}x{spectra_chunk[1]}, mem={mem_gb:.1f}GB, {backend})')

            # Pre-create HDF5 datasets with optimized chunking
            if self.config.store_spectra:
                spectra_ds = spectra_grp.create_dataset(
                    file_key,
                    shape=(n_pixels, n_energy),
                    dtype='float32',
                    chunks=spectra_chunk,
                    compression=self.config.compression,
                    compression_opts=self.config.compression_opts,
                )
                spectra_ds.attrs['energy'] = energy

            fitpara_ds = fitpara_grp.create_dataset(
                f'{file_key}_truth',
                shape=(n_pixels, n_comp, 9),
                dtype='float32',
                chunks=fitpara_chunk,
                compression=self.config.compression,
                compression_opts=self.config.compression_opts,
            )

            # Determine noise level for this condition
            noise_level = 0.0
            if noise_config.noise_type == 'poisson' or noise_config.noise_type == 'mixed':
                noise_level = noise_config.poisson_level

            # Pre-compute profiles cache for MLX (once per element)
            profiles_cache = None
            if mlx_gen is not None:
                profiles_cache = mlx_gen.precompute_profiles(energy, centers, sigmas, float(gamma))

            # Process in batches (MATLAB DepthProfiler.m style)
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_pixels)
                batch_slice = slice(batch_start, batch_end)
                batch_len = batch_end - batch_start

                # Get amplitude values for this batch: (n_comp, batch_len)
                amp_batch = np.array([amplitudes_raw[i, batch_slice] for i in comp_amp_indices],
                                     dtype=np.float32)

                # Generate spectra using MLX or CPU
                if mlx_gen is not None:
                    # Full MLX pipeline: spectra generation + noise on GPU
                    spectra_batch = mlx_gen.generate_batch(
                        amp_batch, energy, centers, sigmas, float(gamma),
                        noise_level=noise_level,
                        bg_base=gen_config.bg_base,
                        amplitude_scale=gen_config.amplitude_scale,
                        profiles_cache=profiles_cache,
                    )

                    # Add Gaussian noise if mixed (MLX handles Poisson above)
                    if noise_config.noise_type == 'mixed' and noise_config.gaussian_std > 0:
                        spectra_batch = add_gaussian_noise(spectra_batch, noise_config.gaussian_std)
                    elif noise_config.noise_type == 'gaussian':
                        spectra_batch = add_gaussian_noise(spectra_batch, noise_config.gaussian_std)

                else:
                    # CPU path (original implementation)
                    spectra_batch = np.zeros((batch_len, n_energy), dtype=np.float32)

                    for j, (amp_idx, profile) in enumerate(zip(comp_amp_indices, profiles)):
                        amp_values = amplitudes_raw[amp_idx, batch_slice]
                        spectra_batch += amp_values[:, np.newaxis] * gen_config.amplitude_scale * profile

                    # Add background
                    total_amp_batch = np.sum([amplitudes_raw[i, batch_slice] for i in comp_amp_indices], axis=0)
                    spectra_batch += total_amp_batch[:, np.newaxis] * gen_config.amplitude_scale * gen_config.bg_base

                    # Add noise (use MLX if available and enabled)
                    if noise_config.noise_type == 'poisson':
                        if self.config.use_mlx_noise and HAS_MLX:
                            spectra_batch = add_poisson_noise_mlx(spectra_batch, noise_config.poisson_level)
                        else:
                            spectra_batch = add_poisson_noise(spectra_batch, noise_config.poisson_level)
                    elif noise_config.noise_type == 'gaussian':
                        spectra_batch = add_gaussian_noise(spectra_batch, noise_config.gaussian_std)
                    elif noise_config.noise_type == 'mixed':
                        if self.config.use_mlx_noise and HAS_MLX:
                            spectra_batch = add_poisson_noise_mlx(spectra_batch, noise_config.poisson_level)
                        else:
                            spectra_batch = add_poisson_noise(spectra_batch, noise_config.poisson_level)
                        spectra_batch = add_gaussian_noise(spectra_batch, noise_config.gaussian_std)

                # Store fitpara
                fitpara_batch = np.full((batch_len, n_comp, 9), np.nan, dtype=np.float32)
                for j, amp_idx in enumerate(comp_amp_indices):
                    amp_values = amplitudes_raw[amp_idx, batch_slice]
                    fitpara_batch[:, j, 0] = amp_values * gen_config.amplitude_scale
                    fitpara_batch[:, j, 1] = centers[j]
                    fitpara_batch[:, j, 2] = sigmas[j]
                    fitpara_batch[:, j, 3] = gamma
                    fitpara_batch[:, j, 7] = gen_config.func_type

                # Write batch to HDF5 (partial write)
                if self.config.store_spectra:
                    spectra_ds[batch_slice, :] = spectra_batch
                fitpara_ds[batch_slice, :, :] = fitpara_batch

                # Progress (every 10 batches or at end)
                if verbose and n_batches > 1 and (batch_idx % 10 == 0 or batch_idx == n_batches - 1):
                    progress = (batch_idx + 1) / n_batches * 100
                    print(f'    {file_key}: {progress:.0f}% ({batch_idx + 1}/{n_batches})')

        gen_time = time.perf_counter() - gen_start
        cond.attrs['generation_time'] = gen_time

        if verbose:
            print(f'Generation: {gen_time:.1f}s')

        # Apply denoise hook if configured
        denoise_time = 0.0
        if self.config.denoise_fn is not None and noise_config.noise_type != 'none':
            if verbose:
                print(f'Running denoise ({self.config.denoise_name})...')

            denoise_start = time.perf_counter()
            denoised_grp = cond.create_group('denoised')

            for file_key in element_keys:
                # Get spectra dataset reference (not loading all)
                spectra_ds = spectra_grp[file_key]
                energy = spectra_ds.attrs['energy']
                n_spectra = spectra_ds.shape[0]
                n_energy = spectra_ds.shape[1]

                # Calculate batch size for denoising
                denoise_batch_size = calculate_batch_size(n_spectra, n_energy, 2)
                n_denoise_batches = (n_spectra + denoise_batch_size - 1) // denoise_batch_size

                # Pre-create denoised dataset if storing
                if self.config.store_denoised:
                    spectra_chunk, _ = calculate_optimal_chunk_size(n_spectra, n_energy, 2)
                    denoised_ds = denoised_grp.create_dataset(
                        file_key,
                        shape=(n_spectra, n_energy),
                        dtype='float32',
                        chunks=spectra_chunk,
                        compression=self.config.compression,
                        compression_opts=self.config.compression_opts,
                    )
                    denoised_ds.attrs['energy'] = energy

                if verbose and n_denoise_batches > 1:
                    print(f'  {file_key}: {n_denoise_batches} denoise batches')

                # Process in batches
                for batch_idx in range(n_denoise_batches):
                    batch_start = batch_idx * denoise_batch_size
                    batch_end = min(batch_start + denoise_batch_size, n_spectra)
                    batch_slice = slice(batch_start, batch_end)

                    # Read batch
                    noisy_batch = spectra_ds[batch_slice, :]

                    # Apply denoise function
                    denoised_batch = self.config.denoise_fn(
                        noisy_batch, energy, noise_config.poisson_level
                    )

                    # Store denoised batch
                    if self.config.store_denoised:
                        denoised_ds[batch_slice, :] = denoised_batch

                    # Replace spectra for fitting (write back to original dataset)
                    spectra_ds[batch_slice, :] = denoised_batch

                    # Progress
                    if verbose and n_denoise_batches > 1 and (batch_idx % 10 == 0 or batch_idx == n_denoise_batches - 1):
                        progress = (batch_idx + 1) / n_denoise_batches * 100
                        print(f'    {file_key}: {progress:.0f}%')

            denoise_time = time.perf_counter() - denoise_start
            cond.attrs['denoise_time'] = denoise_time
            cond.attrs['denoise_method'] = self.config.denoise_name

            if verbose:
                print(f'Denoise: {denoise_time:.1f}s')

        # Run VoigtFit
        if verbose:
            print('Running VoigtFit...')

        fit_start = time.perf_counter()

        cache = WeightMatrixCache()
        pipeline = HybridPipeline(cache, enable_stage2=self.config.enable_stage2, use_mlx=True)

        fitted_amplitudes = {}

        for file_key in element_keys:
            eg = elem_grp[file_key]
            centers = eg['centers'][:]
            sigmas = eg['sigmas'][:]
            gamma = eg.attrs['gamma']

            # Get spectra dataset (NOT loading all into memory)
            if not self.config.store_spectra:
                raise ValueError("store_spectra=False not supported for fitting")

            spectra_ds = spectra_grp[file_key]  # HDF5 dataset reference
            energy = spectra_ds.attrs['energy']
            n_spectra = spectra_ds.shape[0]
            n_energy = spectra_ds.shape[1]

            # Parse element/orbital
            symbol = eg.attrs['symbol']
            orbital = eg.attrs['orbital']

            peak_config = {
                'centers': centers.astype(np.float32),
                'sigmas': sigmas.astype(np.float32),
                'gamma': float(gamma),
            }

            # Calculate memory-aware batch size for VoigtFit
            fit_batch_size = calculate_batch_size(n_spectra, n_energy, len(centers))
            n_fit_batches = (n_spectra + fit_batch_size - 1) // fit_batch_size

            if verbose and n_fit_batches > 1:
                print(f'  {file_key}: {n_fit_batches} fit batches (batch={fit_batch_size:,})')

            # Pre-allocate output array (avoid list append memory fragmentation)
            n_comp = len(centers)
            fitted_amps = np.zeros((n_comp, n_spectra), dtype=np.float32)

            for batch_idx in range(n_fit_batches):
                batch_start = batch_idx * fit_batch_size
                batch_end = min(batch_start + fit_batch_size, n_spectra)

                # Read batch from HDF5 (not full array)
                Y_batch = spectra_ds[batch_start:batch_end, :]

                result = pipeline.process_rowmajor(
                    Y_batch, symbol, orbital, energy, peak_config
                )

                # Write directly to pre-allocated array
                fitted_amps[:, batch_start:batch_end] = result.amplitudes

                # Clear batch memory
                del Y_batch, result

                # Progress
                if verbose and n_fit_batches > 1 and (batch_idx % 10 == 0 or batch_idx == n_fit_batches - 1):
                    progress = (batch_idx + 1) / n_fit_batches * 100
                    print(f'    {file_key}: {progress:.0f}% ({batch_idx + 1}/{n_fit_batches})')

            fitted_amplitudes[file_key] = fitted_amps
            del fitted_amps

        fit_time = time.perf_counter() - fit_start
        cond.attrs['fitting_time'] = fit_time

        if verbose:
            print(f'Fitting: {fit_time:.1f}s')

        # Reconstruct image
        reconstructed = amplitudes_to_rgb(
            amplitudes=fitted_amplitudes,
            color_mapping=self.color_mapping,
            component_order=self.component_order,
            image_shape=(height, width),
        )

        # Calculate PSNR
        psnr_result = compare_images(self.image, reconstructed)

        # Store results
        results = cond.create_group('results')
        results.attrs['psnr_total'] = psnr_result.psnr_total
        results.attrs['psnr_r'] = psnr_result.psnr_r
        results.attrs['psnr_g'] = psnr_result.psnr_g
        results.attrs['psnr_b'] = psnr_result.psnr_b
        results.attrs['mae'] = psnr_result.mean_abs_error
        results.attrs['processing_time'] = gen_time + fit_time

        if self.config.store_reconstructed:
            results.create_dataset(
                'reconstructed_image', data=reconstructed,
                compression=self.config.compression,
                compression_opts=self.config.compression_opts,
            )

        # Store fitted amplitudes
        amp_grp = results.create_group('amplitudes')
        for elem_key, amp in fitted_amplitudes.items():
            amp_grp.create_dataset(elem_key, data=amp,
                                  compression=self.config.compression,
                                  compression_opts=self.config.compression_opts)

        if verbose:
            print(f'PSNR: {psnr_result.psnr_total:.2f} dB')

        # Clean up large objects
        del fitted_amplitudes, reconstructed, amplitudes_raw
        gc.collect()

        return key  # This is the noise_key, not elem_key

    def run_all(self, verbose: bool = True) -> dict[str, float]:
        """
        Run all noise conditions specified in config.

        Always processes noise_type='none' first as true baseline,
        then calculates PSNR vs both original image and noise-free reconstruction.

        Returns:
            Dict mapping noise key to PSNR vs original
        """
        results = {}
        noisefree_reconstructed = None

        # Step 1: Always run noise-free baseline first
        if verbose:
            print('\n[Baseline] Processing noise-free condition...')

        noisefree_config = NoiseConfig(noise_type='none')
        noisefree_key = self.add_condition(noisefree_config, verbose=verbose)
        noisefree_reconstructed = self.h5['conditions'][noisefree_key]['results']['reconstructed_image'][:]
        self.h5['conditions'][noisefree_key]['results'].attrs['psnr_vs_noisefree'] = float('inf')
        results[noisefree_key] = self.h5['conditions'][noisefree_key]['results'].attrs['psnr_total']

        if verbose:
            print('  (Baseline for noise-free comparison)')

        # Step 2: Process all noise levels, sorted from lowest to highest
        sorted_levels = sorted([l for l in self.config.poisson_levels if l > 0])

        for poisson_level in sorted_levels:
            if self.config.gaussian_std == 0:
                noise_config = NoiseConfig(
                    poisson_level=poisson_level,
                    noise_type='poisson'
                )
            else:
                noise_config = NoiseConfig(
                    poisson_level=poisson_level,
                    gaussian_std=self.config.gaussian_std,
                    noise_type='mixed'
                )

            key = self.add_condition(noise_config, verbose=verbose)

            # Get reconstructed image for this condition
            reconstructed = self.h5['conditions'][key]['results']['reconstructed_image'][:]

            # Calculate PSNR vs noise-free reconstruction
            psnr_vs_nf = compare_images(noisefree_reconstructed, reconstructed)
            self.h5['conditions'][key]['results'].attrs['psnr_vs_noisefree'] = psnr_vs_nf.psnr_total
            self.h5['conditions'][key]['results'].attrs['psnr_vs_nf_r'] = psnr_vs_nf.psnr_r
            self.h5['conditions'][key]['results'].attrs['psnr_vs_nf_g'] = psnr_vs_nf.psnr_g
            self.h5['conditions'][key]['results'].attrs['psnr_vs_nf_b'] = psnr_vs_nf.psnr_b
            if verbose:
                print(f'  PSNR vs Noise-Free: {psnr_vs_nf.psnr_total:.2f} dB')

            # Get PSNR vs original
            psnr = self.h5['conditions'][key]['results'].attrs['psnr_total']
            results[key] = psnr

            # Clean up memory after each condition
            del reconstructed
            gc.collect()

        # Clean up baseline reference
        del noisefree_reconstructed
        gc.collect()

        # Update summary
        self._update_summary()

        return results

    def _update_summary(self):
        """Update summary group with aggregated results."""
        summary = self.h5['summary']

        # Clear existing
        for key in list(summary.keys()):
            del summary[key]

        # Collect results
        noise_levels = []
        psnr_values = []

        for key in sorted(self.h5['conditions'].keys()):
            cond = self.h5['conditions'][key]

            if cond.attrs['noise_type'] == 'none':
                level = 0
            else:
                level = cond.attrs['poisson_level']

            noise_levels.append(level)
            psnr_values.append(cond['results'].attrs['psnr_total'])

        summary.create_dataset('noise_levels', data=noise_levels)
        summary.create_dataset('psnr_values', data=psnr_values)

    def get_summary(self) -> dict[str, np.ndarray]:
        """Get summary data."""
        summary = self.h5['summary']
        return {
            'noise_levels': summary['noise_levels'][:],
            'psnr_values': summary['psnr_values'][:],
        }

    def get_condition(self, key: str) -> dict[str, Any]:
        """Get data for a specific condition."""
        cond = self.h5['conditions'][key]
        results = cond['results']

        data = {
            'noise_type': cond.attrs['noise_type'],
            'poisson_level': cond.attrs['poisson_level'],
            'gaussian_std': cond.attrs['gaussian_std'],
            'psnr_total': results.attrs['psnr_total'],
            'psnr_r': results.attrs['psnr_r'],
            'psnr_g': results.attrs['psnr_g'],
            'psnr_b': results.attrs['psnr_b'],
            'mae': results.attrs['mae'],
            'processing_time': results.attrs['processing_time'],
        }

        if 'reconstructed_image' in results:
            data['reconstructed_image'] = results['reconstructed_image'][:]

        return data

    def print_summary(self):
        """Print formatted summary with both vs Original and vs Noise-Free PSNR."""
        print('\n' + '=' * 100)
        print('BENCHMARK SUMMARY')
        print('=' * 100)

        meta = self.h5['metadata']
        print(f"Image: {meta.attrs['source_image']}")
        print(f"Shape: {tuple(meta.attrs['image_shape'])}")
        print(f"Pixels: {meta.attrs['n_pixels']:,}")

        # Check if any condition has denoise
        has_denoise = any(
            'denoise_method' in self.h5['conditions'][k].attrs
            for k in self.h5['conditions'].keys()
        )

        if has_denoise:
            print(f"\n{'Condition':<18} {'vs Original':<12} {'vs NoiseFree':<12} {'Denoise':<10} {'R':<7} {'G':<7} {'B':<7} {'Time':<7}")
            print('-' * 100)
        else:
            print(f"\n{'Condition':<18} {'vs Original':<14} {'vs NoiseFree':<14} {'R':<8} {'G':<8} {'B':<8} {'Time':<8}")
            print('-' * 100)

        # Sort by poisson level for consistent ordering
        sorted_keys = sorted(self.h5['conditions'].keys(),
                            key=lambda k: self.h5['conditions'][k].attrs['poisson_level'])

        for key in sorted_keys:
            cond = self.h5['conditions'][key]
            r = cond['results'].attrs
            psnr_vs_nf = r.get('psnr_vs_noisefree', float('inf'))
            vs_nf_str = 'baseline' if psnr_vs_nf == float('inf') else f'{psnr_vs_nf:.2f}'

            if has_denoise:
                denoise_method = cond.attrs.get('denoise_method', '-')
                print(f"{key:<18} {r['psnr_total']:<12.2f} {vs_nf_str:<12} {denoise_method:<10} "
                      f"{r['psnr_r']:<7.2f} {r['psnr_g']:<7.2f} {r['psnr_b']:<7.2f} {r['processing_time']:<7.1f}s")
            else:
                print(f"{key:<18} {r['psnr_total']:<14.2f} {vs_nf_str:<14} {r['psnr_r']:<8.2f} "
                      f"{r['psnr_g']:<8.2f} {r['psnr_b']:<8.2f} {r['processing_time']:<8.1f}s")

        print('=' * 100)
        print("\nInterpretation (vs NoiseFree = pure noise effect, excluding inverse transform error):")
        print("  > 40 dB: Excellent | 30-40 dB: Good | 20-30 dB: Fair | < 20 dB: Poor")


def run_unified_benchmark(
    image_path: str,
    output_path: str,
    poisson_levels: list[float] = [0, 1e4, 1e6, 1e8],
    verbose: bool = True,
) -> str:
    """
    Convenience function to run unified benchmark.

    Args:
        image_path: Source image
        output_path: Output H5 file path
        poisson_levels: Noise levels to test
        verbose: Print progress

    Returns:
        Path to output H5 file
    """
    config = BenchmarkH5Config(poisson_levels=poisson_levels)

    bench = BenchmarkH5.create(
        output_path=output_path,
        image_path=image_path,
        config=config,
    )

    bench.run_all(verbose=verbose)
    bench.print_summary()
    bench.close()

    return output_path


if __name__ == '__main__':
    from ._data_paths import find_default_image, psnr_test_dir

    _psnr = psnr_test_dir()
    _img = find_default_image(_psnr) or find_default_image()
    if _img is None:
        raise SystemExit(
            'No benchmark image found — set VOIGTFIT_DATA_ROOT and place '
            'an image under PSNRTest/ or Roundtrip/image/.'
        )
    run_unified_benchmark(
        image_path=str(_img),
        output_path=str(_psnr / 'benchmark_unified.h5'),
        poisson_levels=[0, 1e4, 1e6, 1e8],
    )
