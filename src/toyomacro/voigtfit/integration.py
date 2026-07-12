"""
DepthProfiler + VoigtFit Integration Module
==========================================

Provides end-to-end workflows for:
1. Synthetic spectrum generation from depth models
2. Batch VoigtFit processing of image data
3. Result export for MATLAB consumption

Usage:
    from toyomacro.voigtfit.integration import DepthProfilerBridge

    bridge = DepthProfilerBridge()

    # From synthetic data (depth model → spectra → fit)
    result = bridge.process_synthetic(
        depth_profile=depth_profile,  # (n_depth, n_elements)
        peak_configs=peak_configs,     # List of peak configurations
        energy_axes=energy_axes,       # List of energy axes per element
    )

    # From image data (map_*.h5 → fit)
    result = bridge.process_from_h5('map_data.h5', element='C', orbital='1s')
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

try:
    import mlx.core as mx

    from ._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND a Metal device works
except ImportError:
    HAS_MLX = False

from .h5io import FitResultSummary, StreamingFitparaWriter, StreamingWriterConfig
from .pipeline import FitResult, HybridPipeline
from .weight_cache import WeightMatrixCache


@dataclass
class PeakConfig:
    """Configuration for a single element's peaks."""
    element: str
    orbital: str
    centers: np.ndarray      # Peak positions [eV]
    sigmas: np.ndarray       # Gaussian widths [eV]
    gamma: float             # Lorentzian width [eV]
    energy: np.ndarray       # Energy axis [eV]
    alphas: np.ndarray | None = None         # DS asymmetry per component (None=all Voigt)
    lineshape_types: list[str] | None = None  # "voigt"/"ds" per component (None=all Voigt)

    def to_dict(self) -> dict:
        d = {
            "centers": self.centers,
            "sigmas": self.sigmas,
            "gamma": self.gamma,
        }
        if self.alphas is not None:
            d["alphas"] = self.alphas
        if self.lineshape_types is not None:
            d["lineshape_types"] = self.lineshape_types
        return d


@dataclass
class IntegrationResult:
    """Result container for integration workflow."""
    amplitudes: dict[str, np.ndarray]   # {element: (n_components, n_spectra)}
    chi2: dict[str, np.ndarray]         # {element: (n_spectra,)}
    anomaly_mask: dict[str, np.ndarray] # {element: (n_spectra,)}
    timing: dict[str, float]            # Processing time breakdown
    metadata: dict[str, Any] = field(default_factory=dict)

    def save_npz(self, path: str):
        """Save results to NPZ format for MATLAB consumption."""
        data = {}
        for elem in self.amplitudes.keys():
            data[f'{elem}_amplitudes'] = self.amplitudes[elem]
            data[f'{elem}_chi2'] = self.chi2[elem]
            data[f'{elem}_anomaly'] = self.anomaly_mask[elem]
        data['timing'] = np.array([self.timing.get('total', 0)])
        np.savez(path, **data)

    def save_h5(self, path: str):
        """Save results to HDF5 format."""
        with h5py.File(path, 'w') as f:
            for elem in self.amplitudes.keys():
                grp = f.create_group(elem)
                grp.create_dataset('amplitudes', data=self.amplitudes[elem])
                grp.create_dataset('chi2', data=self.chi2[elem])
                grp.create_dataset('anomaly_mask', data=self.anomaly_mask[elem])

            # Timing info
            timing_grp = f.create_group('timing')
            for k, v in self.timing.items():
                if isinstance(v, (int, float)):
                    timing_grp.attrs[k] = v


class DepthProfilerBridge:
    """
    Bridge between DepthProfiler and VoigtFit.

    Handles data conversion, batch processing, and result formatting
    for seamless integration between MATLAB and Python workflows.
    """

    def __init__(
        self,
        use_mlx: bool = True,
        enable_stage2: bool = True,
        stage2_mode: str = "medium",
        chi2_threshold: float = 3.0,
        enable_shift_correction: bool = False,
    ):
        """
        Initialize the bridge.

        Args:
            use_mlx: Use MLX for GPU acceleration
            enable_stage2: Enable Stage 2 refinement for anomalies
            stage2_mode: 'light', 'medium', or 'full'
            chi2_threshold: Anomaly detection threshold (MAD units)
            enable_shift_correction: Enable extended SVD shift correction.
                When True, Stage 2 is not used; all spectra get global
                energy shift estimation in a single linear solve.
        """
        self.use_mlx = use_mlx and HAS_MLX
        self.enable_shift_correction = enable_shift_correction
        self.cache = WeightMatrixCache()
        self.pipeline = HybridPipeline(
            cache=self.cache,
            use_mlx=self.use_mlx,
            enable_stage2=enable_stage2 and not enable_shift_correction,
            stage2_mode=stage2_mode,
            chi2_threshold=chi2_threshold,
            enable_shift_correction=enable_shift_correction,
        )

    def process_spectra(
        self,
        spectra: np.ndarray,
        peak_config: PeakConfig,
    ) -> FitResult:
        """
        Process a batch of spectra for a single element.

        Args:
            spectra: (n_energy, n_spectra) spectral data
            peak_config: Peak configuration for this element

        Returns:
            FitResult with amplitudes, chi2, etc.
        """
        return self.pipeline.process(
            Y=spectra,
            element=peak_config.element,
            orbital=peak_config.orbital,
            energy=peak_config.energy,
            peak_config=peak_config.to_dict(),
        )

    def process_spectra_rowmajor(
        self,
        spectra: np.ndarray,
        peak_config: PeakConfig,
        skip_stage2: bool = True,
    ) -> FitResult:
        """
        Process a batch of spectra in row-major format (fastest path).

        This is the optimized version for HDF5 data which is stored as
        (n_spectra, n_energy). Avoids expensive transpose operations.

        When enable_shift_correction is active, uses extended SVD pipeline
        instead of standard Stage 1 + Stage 2.

        Args:
            spectra: (n_spectra, n_energy) C-contiguous spectral data
            peak_config: Peak configuration for this element
            skip_stage2: Skip Stage 2 refinement (default True for speed)

        Returns:
            FitResult with amplitudes, chi2, etc.
            If shift correction enabled, energy_shifts field is populated.
        """
        if self.enable_shift_correction:
            return self.pipeline.process_rowmajor_extended(
                Y=spectra,
                element=peak_config.element,
                orbital=peak_config.orbital,
                energy=peak_config.energy,
                peak_config=peak_config.to_dict(),
            )
        return self.pipeline.process_rowmajor_fast(
            Y=spectra,
            element=peak_config.element,
            orbital=peak_config.orbital,
            energy=peak_config.energy,
            peak_config=peak_config.to_dict(),
            skip_stage2=skip_stage2,
        )

    def process_spectra_mlx(
        self,
        spectra_mlx,  # mx.array, shape (n_spectra, n_energy)
        peak_config: PeakConfig,
        skip_stage2: bool = True,
    ) -> FitResult:
        """
        Process a batch of spectra already in MLX format (zero conversion overhead).

        This is the fastest path when spectra are already MLX arrays.
        Eliminates the NumPy→MLX conversion bottleneck.

        Args:
            spectra_mlx: MLX array (n_spectra, n_energy) - row-major
            peak_config: Peak configuration for this element
            skip_stage2: Skip Stage 2 refinement (default True for speed)

        Returns:
            FitResult with amplitudes, chi2, etc.
        """
        # Pass MLX array directly - pipeline.process_rowmajor_fast handles it
        return self.pipeline.process_rowmajor_fast(
            Y=spectra_mlx,
            element=peak_config.element,
            orbital=peak_config.orbital,
            energy=peak_config.energy,
            peak_config=peak_config.to_dict(),
            skip_stage2=skip_stage2,
        )

    def process_multi_element(
        self,
        spectra_dict: dict[str, np.ndarray],
        peak_configs: dict[str, PeakConfig],
    ) -> IntegrationResult:
        """
        Process multiple elements sequentially.

        Args:
            spectra_dict: {element: (n_energy, n_spectra)}
            peak_configs: {element: PeakConfig}

        Returns:
            IntegrationResult with all elements
        """
        timing = {"total": 0}
        amplitudes = {}
        chi2 = {}
        anomaly_mask = {}

        t_total = time.perf_counter()

        for elem, spectra in spectra_dict.items():
            if elem not in peak_configs:
                print(f"Warning: No peak config for {elem}, skipping")
                continue

            t0 = time.perf_counter()
            result = self.process_spectra(spectra, peak_configs[elem])
            t_elem = time.perf_counter() - t0

            amplitudes[elem] = result.amplitudes
            chi2[elem] = result.chi2
            anomaly_mask[elem] = result.anomaly_mask
            timing[f'{elem}_time'] = t_elem
            timing[f'{elem}_rate'] = spectra.shape[1] / t_elem

        timing['total'] = time.perf_counter() - t_total

        return IntegrationResult(
            amplitudes=amplitudes,
            chi2=chi2,
            anomaly_mask=anomaly_mask,
            timing=timing,
        )

    def generate_synthetic_spectra(
        self,
        depth_profile: np.ndarray,
        transformation_matrix: np.ndarray,
        peak_config: PeakConfig,
        snr: float = 100.0,
        noise_type: str = "poisson",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic spectra from a depth profile.

        Simulates the forward model: depth → angle-resolved intensity → spectrum

        Args:
            depth_profile: (n_depth,) or (n_depth, n_spatial) depth composition
            transformation_matrix: (n_angles, n_depth) transformation matrix
            peak_config: Peak configuration
            snr: Signal-to-noise ratio
            noise_type: 'poisson' or 'gaussian'

        Returns:
            (spectra, clean_spectra): Both (n_energy, n_spectra)
        """
        from .voigt_jacobian import voigt_profile

        # Depth → Angle-resolved intensities
        # I(theta) = T @ C(depth), where C is concentration profile
        if depth_profile.ndim == 1:
            depth_profile = depth_profile[:, np.newaxis]

        intensities = transformation_matrix @ depth_profile  # (n_angles, n_spatial)
        n_angles, n_spatial = intensities.shape
        n_spectra = n_angles * n_spatial

        # Build basis functions
        energy = peak_config.energy
        n_energy = len(energy)
        n_components = len(peak_config.centers)

        # Clean spectra
        clean_spectra = np.zeros((n_energy, n_spectra))

        for i, (c, s) in enumerate(zip(peak_config.centers, peak_config.sigmas)):
            profile = voigt_profile(energy, c, s, peak_config.gamma)
            # Broadcast: profile (n_energy,) * intensities.flatten() (n_spectra,)
            # Each spectrum gets intensity scaled by the corresponding angle/spatial point
            clean_spectra += np.outer(profile, intensities.flatten())

        # Add noise
        if noise_type == "poisson":
            # Scale to get approximate SNR at peak
            peak_val = clean_spectra.max()
            scale = (peak_val / snr) ** 2
            noisy_spectra = np.random.poisson(
                np.maximum(clean_spectra / scale, 0.1)
            ).astype(np.float32) * scale
        else:
            noise_std = clean_spectra.max() / snr
            noisy_spectra = clean_spectra + noise_std * np.random.randn(*clean_spectra.shape)

        return noisy_spectra.astype(np.float32), clean_spectra.astype(np.float32)

    def process_synthetic(
        self,
        depth_profile: np.ndarray,
        transformation_matrix: np.ndarray,
        peak_config: PeakConfig,
        snr: float = 100.0,
        return_truth: bool = True,
    ) -> tuple[FitResult, np.ndarray | None]:
        """
        End-to-end synthetic workflow: depth model → spectra → fit.

        Args:
            depth_profile: (n_depth,) or (n_depth, n_spatial)
            transformation_matrix: (n_angles, n_depth)
            peak_config: Peak configuration
            snr: Signal-to-noise ratio
            return_truth: Return clean spectra for validation

        Returns:
            (FitResult, clean_spectra if return_truth else None)
        """
        spectra, clean = self.generate_synthetic_spectra(
            depth_profile, transformation_matrix, peak_config, snr
        )

        result = self.process_spectra(spectra, peak_config)

        return result, clean if return_truth else None

    def load_h5_data(
        self,
        h5_path: str,
        element_idx: int | None = None,
    ) -> dict[str, Any]:
        """
        Load experimental data from DepthProfiler HDF5 format (map_*.h5).

        Args:
            h5_path: Path to map_*.h5 file
            element_idx: Specific element index (or None for all)

        Returns:
            Dict with keys: 'Ismp', 'BindingEnergy', 'StrData', 'thetaobs', etc.
        """
        data = {}
        with h5py.File(h5_path, 'r') as f:
            # Core data
            data['Ismp'] = f['Ismp'][:]  # (tsmp, elements, angles)
            data['BindingEnergy'] = f['BindingEnergy'][:].flatten()
            data['thetaobs'] = f['thetaobs'][:].flatten()

            # Element info (StrData is object array)
            if 'StrData' in f:
                str_data = f['StrData']
                # Handle object references
                data['StrData'] = []
                for i in range(str_data.shape[0]):
                    ref = str_data[i, 0]
                    if isinstance(ref, h5py.Reference):
                        data['StrData'].append(f[ref][()].tobytes().decode('utf-16'))
                    else:
                        data['StrData'].append(str(ref))

            if 'tsmp' in f:
                data['tsmp'] = f['tsmp'][:].flatten()

        return data

    def load_xps_spectra(
        self,
        h5_path: str,
    ) -> dict[str, Any]:
        """
        Load XPS spectra from Toyomacro-exported HDF5 format.

        This format is used for element-specific XPS data files like:
        C1s_*.h5, O1s_*.h5, etc.

        File structure:
            specdata: (n_energy+1, n_spectra+1) - first row is energy axis
            fitpara: (n_spectra, max_components, n_params) - fit parameters
            xytdata: (3, n_spectra) - x, y, t coordinates
            otherpara: (n_params, n_spectra) - other parameters

        Args:
            h5_path: Path to element HDF5 file (e.g., C1s_*.h5)

        Returns:
            Dict with keys:
                'energy': Energy axis (n_energy,)
                'spectra': Spectra matrix (n_energy, n_spectra)
                'fitpara': Fit parameters (n_spectra, max_comp, n_params)
                'xytdata': Coordinates (3, n_spectra)
                'otherpara': Other parameters
                'n_spectra': Number of spectra
        """
        data = {}
        with h5py.File(h5_path, 'r') as f:
            specdata = f['specdata'][:]

            # specdata layout: row 0 = energy, rows 1: = spectra (transposed)
            data['energy'] = specdata[0, :].astype(np.float32)
            data['spectra'] = specdata[1:, :].T.astype(np.float32)  # (n_energy, n_spectra)
            data['n_spectra'] = data['spectra'].shape[1]

            if 'fitpara' in f:
                data['fitpara'] = f['fitpara'][:]

            if 'xytdata' in f:
                data['xytdata'] = f['xytdata'][:]

            if 'otherpara' in f:
                data['otherpara'] = f['otherpara'][:]

            # Extract misc info
            if 'misc' in f:
                misc = f['misc']
                if 'fermienergy' in misc:
                    data['fermi_energy'] = float(misc['fermienergy'][0])
                if 'maxcomp' in misc:
                    data['max_components'] = int(misc['maxcomp'][0])

        return data

    def load_xps_spectra_lazy(
        self,
        h5_path: str,
        rdcc_nbytes: int = 64 * 1024 * 1024,
    ) -> dict[str, Any]:
        """
        Load XPS spectra metadata without loading all data (for large files).

        Returns file handle and metadata for streaming access.

        Args:
            h5_path: Path to element HDF5 file
            rdcc_nbytes: HDF5 chunk cache size (default 64MB for better I/O)

        Returns:
            Dict with keys:
                'file': h5py.File handle (caller must close!)
                'energy': Energy axis (n_energy,)
                'n_spectra': Number of spectra
                'n_energy': Number of energy points
                'max_components': Max components from misc
                'specdata_dset': Dataset reference for streaming
        """
        # Open with optimized chunk cache for faster batch reads
        f = h5py.File(h5_path, 'r',
                      rdcc_nbytes=rdcc_nbytes,
                      rdcc_nslots=10007,
                      rdcc_w0=0.75)

        specdata = f['specdata']
        energy = specdata[0, :].astype(np.float32)

        # Detect layout: small files have (n_spec+1, n_energy), large have (n_spec+1, n_energy)
        n_spectra = specdata.shape[0] - 1
        n_energy = specdata.shape[1]

        data = {
            'file': f,
            'energy': energy,
            'n_spectra': n_spectra,
            'n_energy': n_energy,
            'specdata_dset': specdata,
        }

        if 'misc' in f and 'maxcomp' in f['misc']:
            data['max_components'] = int(f['misc/maxcomp'][0])

        if 'fitpara' in f:
            data['fitpara_dset'] = f['fitpara']

        return data

    def process_xps_file_streaming(
        self,
        h5_path: str,
        centers: np.ndarray | None = None,
        sigmas: np.ndarray | None = None,
        gamma: float = 0.25,
        batch_size: int = 1_000_000,
        progress_callback: Callable[[int, int], None] | None = None,
        output_path: str | None = None,
        write_mode: Literal['inplace', 'new_file', 'temp_then_move'] = 'new_file',
        compression: Literal['none', 'standard', 'full'] | None = None,
    ) -> tuple[FitResult | FitResultSummary, dict[str, Any]]:
        """
        Process large XPS file with streaming (batch processing).

        Suitable for files with millions of spectra.

        Args:
            h5_path: Path to element HDF5 file
            centers: Peak centers (optional, auto-detected if None)
            sigmas: Gaussian widths (optional, default 0.5 if None)
            gamma: Lorentzian width
            batch_size: Number of spectra per batch
            progress_callback: Optional callback(processed, total)
            output_path: If provided, write results directly to HDF5 (streaming mode).
                        Returns FitResultSummary instead of FitResult.
            write_mode: How to handle output file (only used if output_path is set):
                - 'new_file': Create new file (default, safest)
                - 'inplace': Overwrite input file's fitpara
                - 'temp_then_move': Write to temp file, then rename
            compression: Post-process compression (only used if output_path is set):
                - None or 'none': No compression (default)
                - 'standard': fitpara -> FitparaCodec (int16+col+LZ4, ~6x)
                - 'full': standard + otherpara/xytdata -> LZ4 + specdata -> uint16

        Returns:
            (FitResult or FitResultSummary, metadata):
                - FitResult if output_path is None (in-memory mode)
                - FitResultSummary if output_path is set (streaming mode)
        """
        # Load metadata
        data = self.load_xps_spectra_lazy(h5_path)
        f = data['file']
        energy = data['energy']
        n_spectra = data['n_spectra']
        specdata_dset = data['specdata_dset']

        # Extract element/orbital from filename
        fname = Path(h5_path).stem
        parts = fname.split('_')[0]
        element = ''.join(c for c in parts if c.isalpha())
        orbital = ''.join(c for c in parts if c.isdigit() or c == 's' or c == 'p' or c == 'd')

        # Get peak parameters from first valid spectrum
        if centers is None and 'fitpara_dset' in data:
            fitpara_dset = data['fitpara_dset']
            # Find first spectrum with valid fitpara
            for i in range(min(100, n_spectra)):
                fitpara_i = fitpara_dset[i+1, :, :]
                centers_list = []
                for j in range(fitpara_i.shape[0]):
                    c = fitpara_i[j, 1]
                    if not np.isnan(c) and c > 0:
                        centers_list.append(c)
                if centers_list:
                    centers = np.array(centers_list, dtype=np.float32)
                    break

            if centers is None:
                centers = np.array([energy.mean()], dtype=np.float32)

        if sigmas is None:
            sigmas = np.full(len(centers), 0.5, dtype=np.float32)

        n_components = len(centers)

        # Peak config
        peak_config = {
            'centers': centers,
            'sigmas': sigmas,
            'gamma': gamma,
        }

        n_batches = (n_spectra + batch_size - 1) // batch_size
        n_energy = len(energy)

        # Pre-allocate buffer for batch reading (reduces memory allocation overhead)
        buffer = np.empty((batch_size, n_energy), dtype=np.float32)
        src_dtype = specdata_dset.dtype

        timing_total = {'stage1': 0, 'anomaly_detection': 0, 'io': 0}

        # Determine output path for streaming mode
        if output_path is not None:
            actual_output = h5_path if write_mode == 'inplace' else output_path
            writer_config = StreamingWriterConfig(
                output_path=actual_output,
                write_mode=write_mode,
                n_spectra=n_spectra,
                n_components=n_components,
                copy_metadata=(write_mode != 'inplace'),
            )

            # Streaming mode: write directly to file
            try:
                with StreamingFitparaWriter(writer_config, input_path=h5_path) as writer:
                    for batch_idx in range(n_batches):
                        start = batch_idx * batch_size
                        end = min(start + batch_size, n_spectra)
                        count = end - start

                        # Load batch
                        import time as _time
                        t_io_start = _time.perf_counter()

                        if src_dtype == np.float32:
                            try:
                                specdata_dset.read_direct(
                                    buffer[:count],
                                    source_sel=np.s_[start + 1:end + 1, :]
                                )
                                Y = buffer[:count]
                            except TypeError:
                                Y = specdata_dset[start + 1:end + 1, :].astype(np.float32)
                        else:
                            Y = specdata_dset[start + 1:end + 1, :].astype(np.float32)

                        timing_total['io'] += _time.perf_counter() - t_io_start

                        # Process
                        result = self.pipeline.process_rowmajor(Y, element, orbital, energy, peak_config)

                        # Write batch immediately (no memory accumulation)
                        writer.write_batch(
                            batch_idx=batch_idx,
                            start_idx=start,
                            fit_result=result,
                            peak_config=peak_config,
                            anomaly_mask=result.anomaly_mask,
                        )

                        if result.timing:
                            timing_total['stage1'] += result.timing.get('stage1', 0)
                            timing_total['anomaly_detection'] += result.timing.get('anomaly_detection', 0)

                        if progress_callback:
                            progress_callback(end, n_spectra)

                    # Get summary from writer
                    summary = writer.get_summary()

            finally:
                f.close()

            # Post-process compression
            if compression and compression != 'none':
                from .h5io import compress_h5_file
                compression_stats = compress_h5_file(actual_output, compression=compression, repack=True)
            else:
                compression_stats = None

            timing_total['total'] = timing_total['stage1'] + timing_total['anomaly_detection']
            timing_total['n_spectra'] = n_spectra
            timing_total['rate'] = n_spectra / timing_total['total'] if timing_total['total'] > 0 else 0

            metadata = {
                'n_spectra': n_spectra,
                'n_energy': n_energy,
                'energy': energy,
                'centers': centers,
                'sigmas': sigmas,
                'gamma': gamma,
                'element': element,
                'orbital': orbital,
                'output_path': actual_output,
                'timing': timing_total,
            }
            if compression_stats is not None:
                metadata['compression'] = compression_stats

            return summary, metadata

        else:
            # In-memory mode (original behavior)
            all_amplitudes = []
            all_chi2 = []
            all_anomaly = []

            try:
                for batch_idx in range(n_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, n_spectra)
                    count = end - start

                    # Load batch with optimized I/O
                    import time as _time
                    t_io_start = _time.perf_counter()

                    if src_dtype == np.float32:
                        try:
                            specdata_dset.read_direct(
                                buffer[:count],
                                source_sel=np.s_[start + 1:end + 1, :]
                            )
                            Y = buffer[:count]
                        except TypeError:
                            Y = specdata_dset[start + 1:end + 1, :].astype(np.float32)
                    else:
                        Y = specdata_dset[start + 1:end + 1, :].astype(np.float32)

                    timing_total['io'] += _time.perf_counter() - t_io_start

                    # Process using row-major optimized method
                    result = self.pipeline.process_rowmajor(Y, element, orbital, energy, peak_config)

                    all_amplitudes.append(result.amplitudes)
                    all_chi2.append(result.chi2)
                    all_anomaly.append(result.anomaly_mask)

                    if result.timing:
                        timing_total['stage1'] += result.timing.get('stage1', 0)
                        timing_total['anomaly_detection'] += result.timing.get('anomaly_detection', 0)

                    if progress_callback:
                        progress_callback(end, n_spectra)

            finally:
                f.close()

            # Aggregate results
            amplitudes = np.concatenate(all_amplitudes, axis=1)
            chi2 = np.concatenate(all_chi2)
            anomaly_mask = np.concatenate(all_anomaly)

            timing_total['total'] = timing_total['stage1'] + timing_total['anomaly_detection']
            timing_total['n_spectra'] = n_spectra
            timing_total['rate'] = n_spectra / timing_total['total'] if timing_total['total'] > 0 else 0

            aggregated = FitResult(
                amplitudes=amplitudes,
                chi2=chi2,
                anomaly_mask=anomaly_mask,
                timing=timing_total,
            )

            metadata = {
                'n_spectra': n_spectra,
                'n_energy': n_energy,
                'energy': energy,
                'centers': centers,
                'sigmas': sigmas,
                'gamma': gamma,
                'element': element,
                'orbital': orbital,
            }

            return aggregated, metadata

    def process_xps_file(
        self,
        h5_path: str,
        centers: np.ndarray | None = None,
        sigmas: np.ndarray | None = None,
        gamma: float = 0.25,
        streaming_threshold: int = 1_000_000,
    ) -> tuple[FitResult, dict[str, Any]]:
        """
        Process XPS spectra from Toyomacro HDF5 file.

        Automatically uses streaming for large files (>1M spectra).

        Args:
            h5_path: Path to element HDF5 file
            centers: Peak centers (optional, extracted from fitpara if None)
            sigmas: Gaussian widths (optional, default 0.5 if None)
            gamma: Lorentzian width
            streaming_threshold: Use streaming for files with more spectra

        Returns:
            (FitResult, data_dict): Fit result and loaded data
        """
        # Check file size
        with h5py.File(h5_path, 'r') as f:
            n_spectra = f['specdata'].shape[0] - 1

        if n_spectra > streaming_threshold:
            return self.process_xps_file_streaming(
                h5_path, centers, sigmas, gamma
            )

        # Load data (small file)
        data = self.load_xps_spectra(h5_path)

        # Extract element/orbital from filename
        fname = Path(h5_path).stem
        parts = fname.split('_')[0]  # e.g., "C1s" from "C1s_240419OCAlTiSi"
        element = ''.join(c for c in parts if c.isalpha())
        orbital = ''.join(c for c in parts if c.isdigit() or c == 's' or c == 'p' or c == 'd')

        # Get peak parameters
        if centers is None and 'fitpara' in data:
            # Extract centers from fitpara
            # fitpara[i, j, 1] = center of component j for spectrum i
            fitpara = data['fitpara']
            centers_list = []
            for j in range(fitpara.shape[1]):
                c = fitpara[0, j, 1]  # Use first spectrum as reference
                if not np.isnan(c) and c > 0:
                    centers_list.append(c)
            if centers_list:
                centers = np.array(centers_list, dtype=np.float32)
            else:
                # Fallback: use energy range center
                centers = np.array([data['energy'].mean()], dtype=np.float32)

        if sigmas is None:
            sigmas = np.full(len(centers), 0.5, dtype=np.float32)

        # Create peak config
        peak_config = PeakConfig(
            element=element,
            orbital=orbital,
            centers=centers,
            sigmas=sigmas,
            gamma=gamma,
            energy=data['energy'],
        )

        # Process
        result = self.process_spectra(data['spectra'], peak_config)

        return result, data

    def load_project(
        self,
        prj_path: str,
    ) -> dict[str, Any]:
        """
        Load Toyomacro project file (.prj).

        Args:
            prj_path: Path to .prj file

        Returns:
            Dict with keys:
                'data_txt': List of element file names
                'pathname': List of paths
                'elements': List of (element, orbital) tuples
                'h5_files': List of full HDF5 file paths
                'fittingspeed': Original MATLAB fitting speeds
        """
        import scipy.io as sio

        prj = sio.loadmat(prj_path)

        # Extract file lists
        data_txt_raw = prj['data_txt'].flatten()
        pathname_raw = prj['pathname'].flatten()

        data_txt = [str(d[0]) for d in data_txt_raw]
        pathname = [str(p[0]) for p in pathname_raw]

        # Build full paths
        h5_files = []
        elements = []
        for dt, pn in zip(data_txt, pathname):
            h5_path = Path(pn) / f"{dt}.h5"
            h5_files.append(str(h5_path))

            # Extract element/orbital from filename
            parts = dt.split('_')[0]
            element = ''.join(c for c in parts if c.isalpha())
            orbital = ''.join(c for c in parts if c.isdigit() or c == 's' or c == 'p' or c == 'd')
            elements.append((element, orbital))

        # Extract timing info if available
        fittingspeed = []
        if 'fittingspeed' in prj:
            speed_raw = prj['fittingspeed'].flatten()
            fittingspeed = [float(s[0]) for s in speed_raw]

        return {
            'data_txt': data_txt,
            'pathname': pathname,
            'elements': elements,
            'h5_files': h5_files,
            'fittingspeed': fittingspeed,
            'n_elements': len(data_txt),
        }

    def process_project(
        self,
        prj_path: str,
        batch_size: int = 5_000_000,
        progress_callback: Callable | None = None,
        parallel: bool = False,
    ) -> dict[str, Any]:
        """
        Process all elements in a Toyomacro project.

        Args:
            prj_path: Path to .prj file
            batch_size: Batch size for streaming
            progress_callback: Optional callback(element, processed, total)
            parallel: Process elements in parallel (experimental)

        Returns:
            Dict with keys:
                'results': {element: FitResult}
                'metadata': {element: metadata}
                'timing': Timing breakdown
                'summary': Summary statistics
        """
        # Load project
        project = self.load_project(prj_path)

        results = {}
        metadata = {}
        timing = {
            'elements': {},
            'total_spectra': 0,
            'total_time': 0,
        }

        total_start = time.perf_counter()

        for i, (h5_path, (element, orbital)) in enumerate(zip(project['h5_files'], project['elements'])):
            elem_key = f"{element}{orbital}"

            if not Path(h5_path).exists():
                print(f"Warning: {h5_path} not found, skipping")
                continue

            # Element progress callback
            def elem_progress(done, total):
                if progress_callback:
                    progress_callback(elem_key, done, total, i+1, project['n_elements'])

            elem_start = time.perf_counter()

            try:
                result, meta = self.process_xps_file_streaming(
                    h5_path,
                    batch_size=batch_size,
                    progress_callback=elem_progress,
                )

                results[elem_key] = result
                metadata[elem_key] = meta

                elem_time = time.perf_counter() - elem_start
                n_spectra = meta['n_spectra']

                timing['elements'][elem_key] = {
                    'time': elem_time,
                    'n_spectra': n_spectra,
                    'rate': n_spectra / elem_time if elem_time > 0 else 0,
                }
                timing['total_spectra'] += n_spectra

            except Exception as e:
                print(f"Error processing {elem_key}: {e}")
                continue

        timing['total_time'] = time.perf_counter() - total_start
        timing['overall_rate'] = timing['total_spectra'] / timing['total_time'] if timing['total_time'] > 0 else 0

        # Summary
        summary = {
            'n_elements': len(results),
            'total_spectra': timing['total_spectra'],
            'total_time': timing['total_time'],
            'overall_rate': timing['overall_rate'],
            'anomaly_counts': {k: int(r.anomaly_mask.sum()) for k, r in results.items()},
            'chi2_means': {k: float(r.chi2.mean()) for k, r in results.items()},
        }

        # Compare with original MATLAB timing
        if project['fittingspeed']:
            matlab_rate = np.mean(project['fittingspeed'])
            summary['matlab_rate'] = matlab_rate
            summary['speedup'] = timing['overall_rate'] / matlab_rate if matlab_rate > 0 else 0

        return {
            'results': results,
            'metadata': metadata,
            'timing': timing,
            'summary': summary,
            'project': project,
        }


def demo_synthetic_workflow():
    """Demonstrate synthetic data workflow."""
    print("=" * 60)
    print("VoigtFit + DepthProfiler Integration Demo")
    print("=" * 60)

    # Setup
    bridge = DepthProfilerBridge(
        use_mlx=HAS_MLX,
        enable_stage2=False,  # Disable for speed demo
    )

    # Synthetic parameters
    n_depth = 100
    n_angles = 14
    n_spatial = 1000  # Simulate 1000 spatial points

    # Simple depth profile: exponential decay
    depth = np.linspace(0, 10, n_depth)  # nm
    depth_profile = np.exp(-depth / 3)[:, np.newaxis] * np.ones((1, n_spatial))

    # Simplified transformation matrix
    angles = np.linspace(0, 70, n_angles)  # degrees
    # T_ij = exp(-d_j / (lambda * cos(theta_i)))
    imfp = 2.0  # nm
    cos_theta = np.cos(np.radians(angles))
    transformation_matrix = np.exp(-depth[np.newaxis, :] / (imfp * cos_theta[:, np.newaxis]))

    # Peak configuration (e.g., C 1s)
    peak_config = PeakConfig(
        element="C",
        orbital="1s",
        centers=np.array([284.5, 286.0, 288.5]),
        sigmas=np.array([0.8, 0.9, 0.7]),
        gamma=0.3,
        energy=np.linspace(280, 292, 100),
    )

    # Run workflow
    print(f"\nProcessing {n_angles * n_spatial} spectra...")
    t0 = time.perf_counter()

    result, clean = bridge.process_synthetic(
        depth_profile=depth_profile,
        transformation_matrix=transformation_matrix,
        peak_config=peak_config,
        snr=100.0,
    )

    total_time = time.perf_counter() - t0
    n_spectra = n_angles * n_spatial

    print("\nResults:")
    print(f"  Total spectra: {n_spectra}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Rate: {n_spectra/total_time:.0f} spectra/s")
    print(f"  Anomalies: {result.anomaly_mask.sum()} ({100*result.anomaly_mask.mean():.2f}%)")
    print(f"  Mean χ²: {result.chi2.mean():.4f}")
    print(f"\nAmplitudes shape: {result.amplitudes.shape}")

    return result, clean


if __name__ == "__main__":
    demo_synthetic_workflow()
