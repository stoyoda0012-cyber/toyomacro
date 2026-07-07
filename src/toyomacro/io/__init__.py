"""
Toyomacro I/O Module.

MATLAB .mat and HDF5 file reading/writing with lazy loading support
for large-scale XPS data (100K+ spectra).

Usage:
    from toyomacro.io import read_mat, read_hdf5, write_hdf5
    from toyomacro.io import LazySpectrum, MATReader, HDF5Writer

    # Simple reading
    spectrum = read_mat('/path/to/data.mat', idx=0)
    spectrum = read_hdf5('/path/to/data.h5', idx=0)

    # Lazy loading for large files
    with LazySpectrum('/path/to/large.h5') as lazy:
        preview = lazy.preview(max_points=1000)
        for batch in lazy.iter_batches():
            process(batch)

    # Writing (MATLAB-compatible HDF5)
    write_hdf5('/path/to/output.h5', spectra, energy, fitpara)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from toyomacro.io.compression import (
    compress_h5_file_streaming,
    convert_folder,
    decode_fitpara_from_h5,
)
from toyomacro.io.hdf5_cache import (
    Coordinates4D,
    DimensionConfig,
    FileMetadata,
    HDF5Cache,
    MultiSpectrumData,
    SpectrumData,
    decode_index,
    encode_index,
)
from toyomacro.io.importer import (
    RAW_EXTENSIONS,
    ImportConfig,
    ImportResult,
    detect_element,
    ensure_h5,
    import_file,
    import_folder,
)
from toyomacro.io.readers import (
    BaseReader,
    DatasetInfo,
    LazySpectrum,
    MATReader,
    NPLReader,
    PXTReader,
    RawSpectrumData,
    SESTxtReader,
    SpectrumMetadata,
    VAMASReader,
    create_reader,
    detect_format,
    detect_mat_version,
)
from toyomacro.io.recovery import (
    clear_incomplete_marker,
    get_file_status,
    recover_partial_fit,
    validate_fitpara_file,
)
from toyomacro.io.schema import ToyomacroSchema
from toyomacro.io.utils import (
    calculate_batch_size,
    estimate_file_size_mb,
    get_available_memory_gb,
    open_h5_optimized,
)
from toyomacro.io.writers import (
    FitResultSummary,
    HDF5CreateOptions,
    HDF5Writer,
    StreamingFitparaWriter,
    StreamingWriterConfig,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from toyomacro.core import Spectrum


def read_mat(filepath: str, idx: int = 0) -> Spectrum:
    """
    Read a spectrum from a MATLAB .mat file.

    Args:
        filepath: Path to .mat file
        idx: Spectrum index (0-based)

    Returns:
        Spectrum object
    """
    with MATReader(filepath) as reader:
        return reader.get_spectrum(idx)


def read_hdf5(filepath: str, idx: int = 0) -> Spectrum:
    """
    Read a spectrum from an HDF5 file.

    Args:
        filepath: Path to .h5 file
        idx: Spectrum index (0-based)

    Returns:
        Spectrum object
    """
    with LazySpectrum(filepath) as reader:
        return reader.get_spectrum(idx)


def write_hdf5(
    filepath: str,
    spectra: NDArray[np.floating],
    energy: NDArray[np.floating],
    fitpara: NDArray[np.floating] | None = None,
    otherpara: NDArray[np.floating] | None = None,
    misc: dict | None = None,
) -> None:
    """
    Write spectra to a voigtfit-compatible HDF5 file.

    Uses contiguous layout (no chunking/compression) for maximum read speed.

    Args:
        filepath: Output file path
        spectra: Spectra array of shape (n_spectra, n_energy)
        energy: Energy axis of shape (n_energy,)
        fitpara: Optional fitpara array of shape (n_spectra, maxcomp, 9)
        otherpara: Optional otherpara array of shape (n_spectra, 10)
        misc: Optional metadata dictionary
    """
    n_spectra = spectra.shape[0]
    n_energy = spectra.shape[1]
    max_components = fitpara.shape[1] if fitpara is not None else 6

    with HDF5Writer(filepath) as writer:
        writer.create(
            n_spectra=n_spectra,
            n_energy=n_energy,
            max_components=max_components,
            energy=energy,
            misc=misc,
        )
        writer.write_spectra_batch(0, spectra)

        if fitpara is not None:
            writer.write_fitpara_batch(0, fitpara)

        if otherpara is not None:
            writer.write_otherpara_batch(0, otherpara)


__all__ = [
    # Schema
    "ToyomacroSchema",
    # HDF5 Cache (fast spectrum access)
    "HDF5Cache",
    "DimensionConfig",
    "FileMetadata",
    "Coordinates4D",
    "SpectrumData",
    "MultiSpectrumData",
    "decode_index",
    "encode_index",
    # Classes
    "LazySpectrum",
    "DatasetInfo",
    "MATReader",
    "HDF5Writer",
    "HDF5CreateOptions",
    # Streaming writer
    "StreamingFitparaWriter",
    "StreamingWriterConfig",
    "FitResultSummary",
    # Convenience functions
    "read_mat",
    "read_hdf5",
    "write_hdf5",
    "detect_mat_version",
    # Raw data readers
    "BaseReader",
    "RawSpectrumData",
    "SpectrumMetadata",
    "PXTReader",
    "SESTxtReader",
    "VAMASReader",
    "NPLReader",
    "create_reader",
    "detect_format",
    # Import pipeline
    "ImportConfig",
    "ImportResult",
    "RAW_EXTENSIONS",
    "detect_element",
    "ensure_h5",
    "import_file",
    "import_folder",
    # Recovery functions
    "validate_fitpara_file",
    "recover_partial_fit",
    "clear_incomplete_marker",
    "get_file_status",
    # Compression
    "compress_h5_file_streaming",
    "convert_folder",
    "decode_fitpara_from_h5",
    # Utilities
    "get_available_memory_gb",
    "calculate_batch_size",
    "estimate_file_size_mb",
    "open_h5_optimized",
]
