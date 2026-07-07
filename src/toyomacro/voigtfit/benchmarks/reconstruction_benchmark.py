"""
Image Reconstruction Benchmark for VoigtFit.

Evaluates the quality of spectral fitting by comparing reconstructed images
with the original source image using PSNR and other metrics.

Workflow:
    1. Original Image → 2. Spectra Generation → 3. VoigtFit → 4. Image Reconstruction
                        (MATLAB/DepthProfiler)   (This module)   (This module)
                                ↓
                         Noise injection, etc.
Date: 2026-01-22
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from ..image_utils import (
    PSNRResult,
    amplitudes_to_rgb,
    compare_images,
    get_fuji_color_mapping,
    load_image,
    save_image,
)
from ..pipeline import HybridPipeline
from ..weight_cache import WeightMatrixCache


@dataclass
class ElementConfig:
    """Configuration for a single element's spectral data."""
    file_key: str           # e.g., 'C1s', 'O1s'
    element: str            # e.g., 'C', 'O'
    orbital: str            # e.g., '1s', '2p'
    n_components: int       # Number of Voigt components
    centers: np.ndarray     # Peak centers (eV)
    sigmas: np.ndarray      # Gaussian widths
    gamma: float            # Lorentzian width
    h5_path: Path | None = None


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    # Processing info
    total_spectra: int
    processing_time: float
    throughput: float  # spectra/sec

    # Quality metrics
    psnr_vs_original: PSNRResult | None = None
    psnr_vs_matlab: PSNRResult | None = None

    # Reconstructed image
    reconstructed_image: np.ndarray | None = None

    # Per-element results
    element_results: dict[str, Any] = field(default_factory=dict)


class ReconstructionBenchmark:
    """
    Benchmark for evaluating VoigtFit reconstruction quality.

    Example usage:
        benchmark = ReconstructionBenchmark(
            project_dir='/path/to/SpeedTest',
            project_name='250820OCAlTiSi',
            original_image_path='/path/to/original.jpg',
        )

        # Run VoigtFit and compare
        result = benchmark.run()
        print(f"PSNR vs Original: {result.psnr_vs_original.psnr_total:.2f} dB")

        # Save reconstructed image
        benchmark.save_result(result, 'reconstructed.png')
    """

    def __init__(
        self,
        project_dir: str | Path,
        project_name: str,
        original_image_path: str | Path | None = None,
        matlab_image_path: str | Path | None = None,
        image_shape: tuple | None = None,
        batch_size: int = 2_000_000,
        enable_stage2: bool = False,
    ):
        """
        Initialize benchmark.

        Args:
            project_dir: Directory containing H5 files
            project_name: Project name (e.g., '250820OCAlTiSi')
            original_image_path: Path to original source image for PSNR comparison
            matlab_image_path: Path to MATLAB-generated image for comparison
            image_shape: (height, width) - auto-detected if not provided
            batch_size: Spectra per batch for processing
            enable_stage2: Enable Stage 2 refinement
        """
        self.project_dir = Path(project_dir)
        self.project_name = project_name
        self.batch_size = batch_size
        self.enable_stage2 = enable_stage2

        # Load reference images
        self.original_image = None
        if original_image_path:
            self.original_image = load_image(original_image_path)

        self.matlab_image = None
        if matlab_image_path:
            self.matlab_image = load_image(matlab_image_path)

        # Auto-detect image shape
        if image_shape:
            self.image_shape = image_shape
        elif self.original_image is not None:
            self.image_shape = self.original_image.shape[:2]
        elif self.matlab_image is not None:
            self.image_shape = self.matlab_image.shape[:2]
        else:
            self.image_shape = None

        # Initialize pipeline
        self.cache = WeightMatrixCache()
        self.pipeline = HybridPipeline(
            self.cache,
            enable_stage2=enable_stage2,
            use_mlx=True,
        )

        # Element configurations (will be populated)
        self.element_configs: list[ElementConfig] = []

        # Color mapping
        self.color_mapping = None
        self.component_order = None

    def configure_elements_from_fitpara(
        self,
        element_files: list[str],
    ) -> None:
        """
        Configure elements by reading parameters from existing fitpara.

        Args:
            element_files: List of element file prefixes (e.g., ['C1s', 'O1s'])
        """
        self.element_configs = []

        for file_key in element_files:
            h5_path = self.project_dir / f"{file_key}_{self.project_name}.h5"

            with h5py.File(h5_path, 'r') as f:
                # Get number of components from fitpara shape
                fitpara = f['fitpara']
                n_comp = fitpara.shape[1]

                # Read first spectrum's parameters
                fp = fitpara[0, :, :]

                # Count valid components (non-NaN amplitude)
                valid_mask = ~np.isnan(fp[:, 0])
                n_valid = np.sum(valid_mask)

                centers = fp[:n_valid, 1]
                sigmas = fp[:n_valid, 2]
                gamma = fp[0, 3]

                # Parse element/orbital from file key
                # Common patterns: C1s, O1s, Si2p, Ti2p3/2
                if '2p3' in file_key:
                    elem = file_key.split('2p')[0]
                    orbital = '2p3/2'
                elif '2p' in file_key:
                    elem = file_key.split('2p')[0]
                    orbital = '2p'
                elif '1s' in file_key:
                    elem = file_key.split('1s')[0]
                    orbital = '1s'
                else:
                    elem = file_key
                    orbital = '1s'

            config = ElementConfig(
                file_key=file_key,
                element=elem,
                orbital=orbital,
                n_components=n_valid,
                centers=centers.astype(np.float32),
                sigmas=sigmas.astype(np.float32),
                gamma=float(gamma),
                h5_path=h5_path,
            )
            self.element_configs.append(config)

        # Auto-detect image shape if not set
        if self.image_shape is None:
            with h5py.File(self.element_configs[0].h5_path, 'r') as f:
                n_spectra = f['specdata'].shape[0] - 1
                # Try to infer from xytdata
                if 'xytdata' in f:
                    xyt = f['xytdata'][:]
                    width = int(xyt[0].max())
                    height = int(xyt[1].max())
                    self.image_shape = (height, width)
                else:
                    # Assume square
                    side = int(np.sqrt(n_spectra))
                    self.image_shape = (side, side)

    def configure_fuji_8k(self) -> None:
        """
        Configure for Fuji 8K test data with standard settings.
        """
        element_files = ['Si1s', 'Ti1s', 'Al1s', 'C1s', 'O1s']
        self.configure_elements_from_fitpara(element_files)

        # Set color mapping for Fuji data
        self.color_mapping, self.component_order = get_fuji_color_mapping()
        self.image_shape = (4320, 7680)  # 8K

    def run(
        self,
        write_mode: Literal['none', 'new_file', 'backup'] = 'none',
        verbose: bool = True,
    ) -> BenchmarkResult:
        """
        Run VoigtFit processing and generate reconstructed image.

        Args:
            write_mode: How to save fitpara results
            verbose: Print progress

        Returns:
            BenchmarkResult with metrics and reconstructed image
        """
        if not self.element_configs:
            raise ValueError("No elements configured. Call configure_* first.")

        if verbose:
            print("=" * 70)
            print("VoigtFit Reconstruction Benchmark")
            print("=" * 70)

        total_spectra = 0
        total_start = time.perf_counter()
        amplitudes = {}

        # Process each element
        for config in self.element_configs:
            if verbose:
                print(f"\n--- {config.file_key} ({config.n_components} components) ---")

            elem_start = time.perf_counter()
            all_amps = []

            with h5py.File(config.h5_path, 'r') as f:
                specdata = f['specdata']
                energy = specdata[0, :].astype(np.float32)
                n_spectra = specdata.shape[0] - 1

                peak_config = {
                    'centers': config.centers,
                    'sigmas': config.sigmas,
                    'gamma': config.gamma,
                }

                for batch_start in range(0, n_spectra, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, n_spectra)
                    Y_batch = specdata[batch_start + 1:batch_end + 1, :].astype(np.float32)

                    result = self.pipeline.process_rowmajor(
                        Y_batch,
                        config.element,
                        config.orbital,
                        energy,
                        peak_config,
                    )

                    all_amps.append(result.amplitudes)

            # Concatenate all batches
            amplitudes[config.file_key] = np.concatenate(all_amps, axis=1)
            total_spectra += n_spectra

            elem_time = time.perf_counter() - elem_start
            if verbose:
                rate = n_spectra / elem_time / 1e6
                print(f"  Time: {elem_time:.2f}s, Rate: {rate:.1f}M spec/s")

        total_time = time.perf_counter() - total_start
        throughput = total_spectra / total_time

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Total: {total_spectra:,} spectra in {total_time:.1f}s")
            print(f"Throughput: {throughput / 1e6:.1f}M spec/s")

        # Generate reconstructed image
        if self.color_mapping is None or self.component_order is None:
            self.color_mapping, self.component_order = get_fuji_color_mapping()

        reconstructed = amplitudes_to_rgb(
            amplitudes=amplitudes,
            color_mapping=self.color_mapping,
            component_order=self.component_order,
            image_shape=self.image_shape,
        )

        # Calculate PSNR metrics
        psnr_original = None
        psnr_matlab = None

        if self.original_image is not None:
            psnr_original = compare_images(self.original_image, reconstructed)
            if verbose:
                print(f"\nPSNR vs Original: {psnr_original.psnr_total:.2f} dB")

        if self.matlab_image is not None:
            psnr_matlab = compare_images(self.matlab_image, reconstructed)
            if verbose:
                print(f"PSNR vs MATLAB:   {psnr_matlab.psnr_total:.2f} dB")

        return BenchmarkResult(
            total_spectra=total_spectra,
            processing_time=total_time,
            throughput=throughput,
            psnr_vs_original=psnr_original,
            psnr_vs_matlab=psnr_matlab,
            reconstructed_image=reconstructed,
            element_results={'amplitudes': amplitudes},
        )

    def run_with_noise(
        self,
        noise_levels: list[float] = [0.0, 0.01, 0.05, 0.1],
        verbose: bool = True,
    ) -> dict[float, BenchmarkResult]:
        """
        Run benchmark with different noise levels.

        Note: This requires re-generating spectra with noise,
        which is done in MATLAB (SaveAngleProfileSpectra_Callback).
        This method is a placeholder for future integration.

        Args:
            noise_levels: List of noise levels (fraction of signal)
            verbose: Print progress

        Returns:
            Dict mapping noise_level to BenchmarkResult
        """
        # TODO: Integrate with MATLAB noise injection
        # For now, just run the baseline
        results = {}

        if verbose:
            print("Note: Noise injection requires MATLAB integration.")
            print("Running baseline (noise_level=0) only.\n")

        results[0.0] = self.run(verbose=verbose)
        return results

    def save_result(
        self,
        result: BenchmarkResult,
        output_path: str | Path,
    ) -> None:
        """
        Save reconstructed image to file.

        Args:
            result: BenchmarkResult from run()
            output_path: Output image path
        """
        if result.reconstructed_image is not None:
            save_image(result.reconstructed_image, output_path)

    def print_summary(
        self,
        result: BenchmarkResult,
    ) -> None:
        """
        Print detailed benchmark summary.
        """
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        print("\nProcessing:")
        print(f"  Total spectra: {result.total_spectra:,}")
        print(f"  Time: {result.processing_time:.1f}s")
        print(f"  Throughput: {result.throughput / 1e6:.1f}M spec/s")

        print("\nImage Quality:")
        if result.psnr_vs_original:
            r = result.psnr_vs_original
            print(f"  vs Original: {r.psnr_total:.2f} dB (R:{r.psnr_r:.2f}, G:{r.psnr_g:.2f}, B:{r.psnr_b:.2f})")

        if result.psnr_vs_matlab:
            r = result.psnr_vs_matlab
            print(f"  vs MATLAB:   {r.psnr_total:.2f} dB (R:{r.psnr_r:.2f}, G:{r.psnr_g:.2f}, B:{r.psnr_b:.2f})")

        print("\n" + "=" * 70)


def run_fuji_benchmark(
    project_dir: str | None = None,
    project_name: str = '250820OCAlTiSi',
    save_output: bool = True,
) -> BenchmarkResult:
    """
    Convenience function to run Fuji 8K benchmark.

    Args:
        project_dir: Path to SpeedTest directory
        project_name: Project name
        save_output: Save reconstructed image

    Returns:
        BenchmarkResult
    """
    if project_dir is None:
        from ._data_paths import speedtest_dir
        project_dir = speedtest_dir()
    else:
        project_dir = Path(project_dir)

    benchmark = ReconstructionBenchmark(
        project_dir=project_dir,
        project_name=project_name,
        original_image_path=project_dir / 'churei-tower-mount-fuji-in-japan-8k-68-7680x4320.jpg',
        matlab_image_path=project_dir / 'Fuji_8K_Color.png',
    )

    benchmark.configure_fuji_8k()
    result = benchmark.run(verbose=True)

    if save_output:
        output_path = project_dir / 'Fuji_8K_VoigtFit_benchmark.png'
        benchmark.save_result(result, output_path)
        print(f"\nSaved: {output_path}")

    benchmark.print_summary(result)
    return result


if __name__ == '__main__':
    run_fuji_benchmark()
