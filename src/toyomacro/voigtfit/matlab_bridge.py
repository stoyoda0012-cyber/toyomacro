#!/usr/bin/env python3
"""
MATLAB Bridge - File-based interface for MATLAB-Python communication
====================================================================

This script provides HDF5-based data exchange between MATLAB and VoigtFit.

Usage:
    python matlab_bridge.py input.h5 output.h5

Input HDF5 format:
    /Y         - Spectra matrix (n_energy, n_spectra), float32
    /energy    - Energy axis (n_energy,), float32
    /centers   - Peak centers (n_components,), float32
    /sigmas    - Gaussian widths (n_components,), float32
    attrs:
        gamma          - Lorentzian width, float
        element        - Element name, string
        orbital        - Orbital name, string
        enable_stage2  - Enable Stage 2, bool

Output HDF5 format:
    /amplitudes   - (n_components, n_spectra), float32
    /chi2         - (n_spectra,), float32
    /anomaly_mask - (n_spectra,), bool
    attrs:
        stage1_time    - Stage 1 processing time
        total_time     - Total processing time
        rate           - Processing rate (spectra/s)
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np

try:
    import mlx.core as mx

    from ._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND the default device can execute work
except ImportError:
    HAS_MLX = False


def process_file(input_path: str, output_path: str) -> dict:
    """
    Process VoigtFit from HDF5 file.

    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file

    Returns:
        dict with timing information
    """
    from .pipeline import HybridPipeline
    from .weight_cache import WeightMatrixCache

    # Load input
    with h5py.File(input_path, 'r') as f:
        Y = f['Y'][:].astype(np.float32)
        energy = f['energy'][:].flatten().astype(np.float32)
        centers = f['centers'][:].flatten().astype(np.float32)
        sigmas = f['sigmas'][:].flatten().astype(np.float32)
        gamma = float(f.attrs.get('gamma', 0.3))
        element = f.attrs.get('element', b'Unknown')
        orbital = f.attrs.get('orbital', b'Unknown')
        enable_stage2 = bool(f.attrs.get('enable_stage2', False))

        # Handle bytes vs string
        if isinstance(element, bytes):
            element = element.decode('utf-8')
        if isinstance(orbital, bytes):
            orbital = orbital.decode('utf-8')

    n_spectra = Y.shape[1]

    # Setup pipeline
    cache = WeightMatrixCache()
    pipeline = HybridPipeline(
        cache=cache,
        use_mlx=HAS_MLX,
        enable_stage2=enable_stage2,
        chi2_threshold=3.0,
    )

    peak_config = {
        "centers": centers,
        "sigmas": sigmas,
        "gamma": gamma,
    }

    # Process
    t_total = time.perf_counter()
    result = pipeline.process(
        Y=Y,
        element=element,
        orbital=orbital,
        energy=energy,
        peak_config=peak_config,
    )
    total_time = time.perf_counter() - t_total

    # Write output
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('amplitudes', data=result.amplitudes)
        f.create_dataset('chi2', data=result.chi2)
        f.create_dataset('anomaly_mask', data=result.anomaly_mask)

        # Timing attributes
        f.attrs['total_time'] = total_time
        f.attrs['rate'] = n_spectra / total_time
        if result.timing:
            f.attrs['stage1_time'] = result.timing.get('stage1', 0)
            f.attrs['n_anomaly'] = result.timing.get('n_anomaly', 0)

    return {
        'total_time': total_time,
        'n_spectra': n_spectra,
        'rate': n_spectra / total_time,
    }


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.h5 output.h5")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    try:
        result = process_file(input_path, output_path)
        print(f"Processed {result['n_spectra']} spectra in {result['total_time']:.3f}s "
              f"({result['rate']:.0f} spec/s)")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
