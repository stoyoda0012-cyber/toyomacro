"""
HDF5 Compression Module for Toyomacro.

Provides streaming compression of large HDF5 specdata files
using uint16 + LZ4 encoding. Self-contained implementation
based on the compression algorithms from DepthProfiler/voigtfit/h5io.py.

Compression tiers:
    Tier 1: fitpara  -> int16 quantization + column-wise + LZ4 (8-18x)
    Tier 2: otherpara/xytdata -> LZ4 (3-242x)
    Tier 3: specdata -> uint16 (zero precision loss for integer counts)

Usage:
    # CLI
    toyomacro convert /path/to/data --compression full

    # Python
    from toyomacro.io.compression import convert_folder
    convert_folder('/path/to/data')
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# LZ4 compression support
try:
    import lz4.frame

    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False


# =============================================================================
# Tier 1: Fitpara Codec (int16 + column-wise + LZ4)
# =============================================================================

INT16_RANGE = 32767.0


def _encode_fitpara_to_h5(
    h5_out: h5py.File,
    fitpara: np.ndarray,
    dataset_name: str = "fitpara_compressed",
) -> dict[str, Any]:
    """Encode fitpara to compressed format and write to HDF5.

    Pipeline: float32 -> per-column int16 quantization -> column-wise concat -> LZ4

    Args:
        h5_out: Output HDF5 file (writable)
        fitpara: (n_spectra, n_comp, 9) float32 array
        dataset_name: Name for the compressed group

    Returns:
        Dict with original_bytes, compressed_bytes, ratio
    """
    if not HAS_LZ4:
        raise ImportError("lz4 package required for compression. Install with: pip install lz4")

    n_spectra, n_comp, n_cols = fitpara.shape
    original_bytes = fitpara.nbytes

    # Per-column quantization
    scales = np.zeros(n_cols, dtype=np.float64)
    offsets = np.zeros(n_cols, dtype=np.float64)
    cols_int16 = []

    for col_idx in range(n_cols):
        col = fitpara[:, :, col_idx]
        col_min = float(np.nanmin(col))
        col_max = float(np.nanmax(col))

        if col_max == col_min or np.isnan(col_max - col_min):
            scales[col_idx] = 1.0
            offsets[col_idx] = col_min if not np.isnan(col_min) else 0.0
            cols_int16.append(np.zeros((n_spectra, n_comp), dtype=np.int16))
        else:
            scales[col_idx] = INT16_RANGE / (col_max - col_min)
            offsets[col_idx] = col_min
            col_clean = np.where(np.isnan(col), col_min, col)
            cols_int16.append(((col_clean - col_min) * scales[col_idx]).astype(np.int16))

    # Column-wise concatenation + LZ4
    cols_bytes = b"".join(col.tobytes() for col in cols_int16)
    compressed_bytes = lz4.frame.compress(cols_bytes)

    # Write to HDF5
    if dataset_name in h5_out:
        del h5_out[dataset_name]
    grp = h5_out.create_group(dataset_name)
    grp.create_dataset("data", data=np.frombuffer(compressed_bytes, dtype=np.uint8))
    grp.attrs["shape"] = (n_spectra, n_comp, n_cols)
    grp.attrs["scales"] = scales
    grp.attrs["offsets"] = offsets
    grp.attrs["compressed_size"] = len(compressed_bytes)
    grp.attrs["original_size"] = original_bytes
    grp.attrs["compression_ratio"] = original_bytes / len(compressed_bytes)

    return {
        "original_bytes": original_bytes,
        "compressed_bytes": len(compressed_bytes),
        "ratio": original_bytes / len(compressed_bytes),
    }


def _encode_fitpara_chunked_to_h5(
    h5_out: h5py.File,
    compact_amplitudes: np.ndarray,
    compact_constants: np.ndarray,
    compact_basis_integrals: np.ndarray,
    compact_chi2: np.ndarray | None = None,
    chunk_size: int = 5_000_000,
    dataset_name: str = "fitpara_compressed",
) -> dict[str, Any]:
    """Encode fitpara from compact buffers in chunks (low peak memory).

    Instead of reconstructing the full (n, comp, 9) array, processes
    chunk_size spectra at a time. Produces bit-identical output to
    _encode_fitpara_to_h5() because global scales/offsets are pre-computed
    from the same data.

    Args:
        h5_out: Output HDF5 file (writable)
        compact_amplitudes: (n_spectra, n_comp) float32 - peak heights
        compact_constants: (n_comp, 7) float32 - cols 1-7 (constant)
        compact_basis_integrals: (n_comp,) float32 - area/amplitude ratio
        compact_chi2: (n_spectra,) float32 - chi2 values (unused in encoding)
        chunk_size: Spectra per chunk (default 5M = ~540MB peak float32)
        dataset_name: HDF5 group name for output

    Returns:
        Dict with original_bytes, compressed_bytes, ratio
    """
    if not HAS_LZ4:
        raise ImportError("lz4 package required for compression")

    n_spectra, n_comp = compact_amplitudes.shape
    n_cols = 9

    # === Pass 1: compute global per-column min/max ===
    scales = np.zeros(n_cols, dtype=np.float64)
    offsets = np.zeros(n_cols, dtype=np.float64)

    # Col 0: amplitude (varies per spectrum)
    col0_min = float(np.nanmin(compact_amplitudes))
    col0_max = float(np.nanmax(compact_amplitudes))
    if col0_max != col0_min and not np.isnan(col0_max - col0_min):
        scales[0] = INT16_RANGE / (col0_max - col0_min)
        offsets[0] = col0_min
    else:
        scales[0] = 1.0
        offsets[0] = col0_min if not np.isnan(col0_min) else 0.0

    # Cols 1-7: constants (broadcast from first batch)
    for i in range(7):
        col_vals = compact_constants[:, i]
        c_min = float(np.nanmin(col_vals))
        c_max = float(np.nanmax(col_vals))
        if c_max != c_min and not np.isnan(c_max - c_min):
            scales[1 + i] = INT16_RANGE / (c_max - c_min)
            offsets[1 + i] = c_min
        else:
            scales[1 + i] = 1.0
            offsets[1 + i] = c_min if not np.isnan(c_min) else 0.0

    # Col 8: area = amplitude * basis_integral (compute min/max in chunks)
    col8_min = float("inf")
    col8_max = float("-inf")
    basis = compact_basis_integrals[np.newaxis, :]  # (1, n_comp)
    for start in range(0, n_spectra, chunk_size):
        end = min(start + chunk_size, n_spectra)
        areas_chunk = compact_amplitudes[start:end, :] * basis
        col8_min = min(col8_min, float(np.nanmin(areas_chunk)))
        col8_max = max(col8_max, float(np.nanmax(areas_chunk)))
    if col8_max != col8_min and not np.isnan(col8_max - col8_min):
        scales[8] = INT16_RANGE / (col8_max - col8_min)
        offsets[8] = col8_min
    else:
        scales[8] = 1.0
        offsets[8] = col8_min if col8_min != float("inf") else 0.0

    # === Pass 2: chunk-wise reconstruct + int16 quantize ===
    # Pre-allocate per-column int16 accumulation arrays
    col_size = n_spectra * n_comp
    cols_int16 = [np.empty(col_size, dtype=np.int16) for _ in range(n_cols)]

    constants_broadcast = compact_constants[np.newaxis, :, :]  # (1, n_comp, 7)

    for start in range(0, n_spectra, chunk_size):
        end = min(start + chunk_size, n_spectra)
        cs = end - start
        out_start = start * n_comp
        out_end = end * n_comp

        # Reconstruct (cs, n_comp, 9) float32 chunk
        amps = compact_amplitudes[start:end, :]  # (cs, n_comp)
        areas = amps * basis  # (cs, n_comp)

        # Col 0: amplitude
        col = amps
        _quantize_col_into(col, scales[0], offsets[0], cols_int16[0], out_start, out_end)

        # Cols 1-7: constants (broadcast)
        for i in range(7):
            col = np.broadcast_to(
                compact_constants[np.newaxis, :, i], (cs, n_comp)
            )
            _quantize_col_into(col, scales[1 + i], offsets[1 + i], cols_int16[1 + i], out_start, out_end)

        # Col 8: area
        _quantize_col_into(areas, scales[8], offsets[8], cols_int16[8], out_start, out_end)

    # === Final: column-wise concat + single LZ4 compress ===
    cols_bytes = b"".join(col.tobytes() for col in cols_int16)
    compressed_bytes = lz4.frame.compress(cols_bytes)
    original_bytes = n_spectra * n_comp * n_cols * 4  # float32 equivalent

    # Write to HDF5 (same format as _encode_fitpara_to_h5)
    if dataset_name in h5_out:
        del h5_out[dataset_name]
    grp = h5_out.create_group(dataset_name)
    grp.create_dataset("data", data=np.frombuffer(compressed_bytes, dtype=np.uint8))
    grp.attrs["shape"] = (n_spectra, n_comp, n_cols)
    grp.attrs["scales"] = scales
    grp.attrs["offsets"] = offsets
    grp.attrs["compressed_size"] = len(compressed_bytes)
    grp.attrs["original_size"] = original_bytes
    grp.attrs["compression_ratio"] = original_bytes / len(compressed_bytes)

    return {
        "original_bytes": original_bytes,
        "compressed_bytes": len(compressed_bytes),
        "ratio": original_bytes / len(compressed_bytes),
    }


def _encode_fitpara_streaming_to_h5(
    h5_out: h5py.File,
    compact_amplitudes,  # ndarray OR h5py.Dataset (duck-typing)
    compact_constants: np.ndarray,
    compact_basis_integrals: np.ndarray,
    compact_chi2=None,  # unused in encoding, kept for API compat
    chunk_size: int = 5_000_000,
    dataset_name: str = "fitpara_compressed",
) -> dict[str, Any]:
    """Streaming encoder: 2-pass, never holds all int16 in RAM.

    Pass 1: Scan amplitudes once to get global min/max for col 0 and col 8.
    Pass 2: Column-by-column quantize + LZ4 streaming compress (no temp dataset).

    Accepts h5py.Dataset for compact_amplitudes (disk-backed staging file).
    Peak memory: ~60MB float32 chunk + ~30MB int16 chunk.
    On-disk format is identical to _encode_fitpara_chunked_to_h5.
    """
    if not HAS_LZ4:
        raise ImportError("lz4 package required for compression")

    n_spectra = compact_amplitudes.shape[0]
    n_comp = compact_amplitudes.shape[1]
    n_cols = 9
    basis = compact_basis_integrals[np.newaxis, :]  # (1, n_comp)

    # === Pass 1: compute global per-column min/max (single scan) ===
    scales = np.zeros(n_cols, dtype=np.float64)
    offsets = np.zeros(n_cols, dtype=np.float64)

    # Col 0 + Col 8 in one loop (avoids reading staging twice)
    col0_min = float("inf")
    col0_max = float("-inf")
    col8_min = float("inf")
    col8_max = float("-inf")
    for start in range(0, n_spectra, chunk_size):
        end = min(start + chunk_size, n_spectra)
        amps_chunk = np.asarray(compact_amplitudes[start:end, :])
        # Col 0: amplitude
        c0_min = float(np.nanmin(amps_chunk))
        c0_max = float(np.nanmax(amps_chunk))
        if c0_min < col0_min:
            col0_min = c0_min
        if c0_max > col0_max:
            col0_max = c0_max
        # Col 8: area = amplitude * basis
        areas_chunk = amps_chunk * basis
        c8_min = float(np.nanmin(areas_chunk))
        c8_max = float(np.nanmax(areas_chunk))
        if c8_min < col8_min:
            col8_min = c8_min
        if c8_max > col8_max:
            col8_max = c8_max

    if col0_max != col0_min and not np.isnan(col0_max - col0_min):
        scales[0] = INT16_RANGE / (col0_max - col0_min)
        offsets[0] = col0_min
    else:
        scales[0] = 1.0
        offsets[0] = col0_min if not np.isnan(col0_min) else 0.0

    if col8_max != col8_min and not np.isnan(col8_max - col8_min):
        scales[8] = INT16_RANGE / (col8_max - col8_min)
        offsets[8] = col8_min
    else:
        scales[8] = 1.0
        offsets[8] = col8_min if col8_min != float("inf") else 0.0

    # Cols 1-7: constants (small array, no chunking needed)
    for i in range(7):
        col_vals = compact_constants[:, i]
        c_min = float(np.nanmin(col_vals))
        c_max = float(np.nanmax(col_vals))
        if c_max != c_min and not np.isnan(c_max - c_min):
            scales[1 + i] = INT16_RANGE / (c_max - c_min)
            offsets[1 + i] = c_min
        else:
            scales[1 + i] = 1.0
            offsets[1 + i] = c_min if not np.isnan(c_min) else 0.0

    # === Pass 2: column-by-column quantize + LZ4 streaming compress ===
    # Column-wise layout: all of col0, then all of col1, ..., then all of col8
    # Each column is (n_spectra * n_comp) int16 values
    compressor = lz4.frame.LZ4FrameCompressor()
    compressed_parts = [compressor.begin()]

    for col_idx in range(n_cols):
        if 1 <= col_idx <= 7:
            # Constants: broadcast from compact_constants (in RAM, no disk read)
            for start in range(0, n_spectra, chunk_size):
                end = min(start + chunk_size, n_spectra)
                cs = end - start
                col = np.broadcast_to(
                    compact_constants[np.newaxis, :, col_idx - 1], (cs, n_comp)
                )
                chunk_int16 = _quantize_chunk(col, scales[col_idx], offsets[col_idx])
                compressed_parts.append(compressor.compress(chunk_int16.tobytes()))
        else:
            # Col 0 (amplitude) or Col 8 (area): read from staging
            for start in range(0, n_spectra, chunk_size):
                end = min(start + chunk_size, n_spectra)
                amps = np.asarray(compact_amplitudes[start:end, :])
                if col_idx == 0:
                    col = amps
                else:  # col_idx == 8
                    col = amps * basis
                chunk_int16 = _quantize_chunk(col, scales[col_idx], offsets[col_idx])
                compressed_parts.append(compressor.compress(chunk_int16.tobytes()))

    compressed_parts.append(compressor.flush())
    compressed_bytes = b"".join(compressed_parts)

    # Write final compressed blob (same format as existing)
    original_bytes = n_spectra * n_comp * n_cols * 4

    if dataset_name in h5_out:
        del h5_out[dataset_name]
    grp = h5_out.create_group(dataset_name)
    grp.create_dataset("data", data=np.frombuffer(compressed_bytes, dtype=np.uint8))
    grp.attrs["shape"] = (n_spectra, n_comp, n_cols)
    grp.attrs["scales"] = scales
    grp.attrs["offsets"] = offsets
    grp.attrs["compressed_size"] = len(compressed_bytes)
    grp.attrs["original_size"] = original_bytes
    grp.attrs["compression_ratio"] = original_bytes / len(compressed_bytes)

    return {
        "original_bytes": original_bytes,
        "compressed_bytes": len(compressed_bytes),
        "ratio": original_bytes / len(compressed_bytes),
    }


def _encode_fitpara_streaming_from_dataset(
    h5_out: h5py.File,
    fitpara_ds: h5py.Dataset,
    chunk_size: int = 500_000,
    dataset_name: str = "fitpara_compressed",
) -> dict[str, Any]:
    """Streaming encoder for raw fitpara dataset (convert command).

    2-pass approach:
      Pass 1: Scan fitpara in chunks to get per-column global min/max.
      Pass 2: Column-by-column quantize + LZ4 streaming compress.

    Unlike _encode_fitpara_streaming_to_h5 which accepts compact buffers,
    this reads directly from a (n_spectra, n_comp, 9) h5py.Dataset.

    Peak memory: ~chunk_size * n_comp * 4 bytes (~20MB at 500K chunks, 2 comp).
    """
    if not HAS_LZ4:
        raise ImportError("lz4 package required for compression")

    n_spectra, n_comp, n_cols = fitpara_ds.shape
    original_bytes = n_spectra * n_comp * n_cols * 4

    # === Pass 1: per-column min/max scan ===
    col_mins = np.full(n_cols, float("inf"), dtype=np.float64)
    col_maxs = np.full(n_cols, float("-inf"), dtype=np.float64)

    for start in range(0, n_spectra, chunk_size):
        end = min(start + chunk_size, n_spectra)
        chunk = fitpara_ds[start:end, :, :]  # (cs, n_comp, 9)
        for col_idx in range(n_cols):
            col = chunk[:, :, col_idx]
            c_min = float(np.nanmin(col))
            c_max = float(np.nanmax(col))
            if c_min < col_mins[col_idx]:
                col_mins[col_idx] = c_min
            if c_max > col_maxs[col_idx]:
                col_maxs[col_idx] = c_max

    scales = np.zeros(n_cols, dtype=np.float64)
    offsets = np.zeros(n_cols, dtype=np.float64)
    for col_idx in range(n_cols):
        c_min, c_max = col_mins[col_idx], col_maxs[col_idx]
        if c_max != c_min and not np.isnan(c_max - c_min):
            scales[col_idx] = INT16_RANGE / (c_max - c_min)
            offsets[col_idx] = c_min
        else:
            scales[col_idx] = 1.0
            offsets[col_idx] = c_min if not np.isnan(c_min) else 0.0

    # === Pass 2: column-by-column quantize + LZ4 streaming compress ===
    compressor = lz4.frame.LZ4FrameCompressor()
    compressed_parts = [compressor.begin()]

    for col_idx in range(n_cols):
        for start in range(0, n_spectra, chunk_size):
            end = min(start + chunk_size, n_spectra)
            col = fitpara_ds[start:end, :, col_idx]  # (cs, n_comp)
            chunk_int16 = _quantize_chunk(col, scales[col_idx], offsets[col_idx])
            compressed_parts.append(compressor.compress(chunk_int16.tobytes()))

    compressed_parts.append(compressor.flush())
    compressed_bytes = b"".join(compressed_parts)

    # Write to HDF5 (same format as other encoders)
    if dataset_name in h5_out:
        del h5_out[dataset_name]
    grp = h5_out.create_group(dataset_name)
    grp.create_dataset("data", data=np.frombuffer(compressed_bytes, dtype=np.uint8))
    grp.attrs["shape"] = (n_spectra, n_comp, n_cols)
    grp.attrs["scales"] = scales
    grp.attrs["offsets"] = offsets
    grp.attrs["compressed_size"] = len(compressed_bytes)
    grp.attrs["original_size"] = original_bytes
    grp.attrs["compression_ratio"] = original_bytes / len(compressed_bytes)

    return {
        "original_bytes": original_bytes,
        "compressed_bytes": len(compressed_bytes),
        "ratio": original_bytes / len(compressed_bytes),
    }


def _quantize_chunk(
    col: np.ndarray,
    scale: float,
    offset: float,
) -> np.ndarray:
    """Quantize a (cs, n_comp) float32 column to int16 array."""
    if scale == 1.0:
        c_min = float(np.nanmin(col))
        c_max = float(np.nanmax(col))
        if c_min == c_max:
            return np.zeros(col.size, dtype=np.int16)
    col_clean = np.where(np.isnan(col), offset, col)
    return ((col_clean - offset) * scale).astype(np.int16).ravel()


def _quantize_col_into(
    col: np.ndarray,
    scale: float,
    offset: float,
    out: np.ndarray,
    out_start: int,
    out_end: int,
) -> None:
    """Quantize a (cs, n_comp) float32 column to int16 into a pre-allocated array."""
    if scale == 1.0:
        c_min = float(np.nanmin(col))
        c_max = float(np.nanmax(col))
        if c_min == c_max:
            out[out_start:out_end] = 0
            return
    col_clean = np.where(np.isnan(col), offset, col)
    out[out_start:out_end] = ((col_clean - offset) * scale).astype(np.int16).ravel()


def decode_fitpara_from_h5(
    h5_file: h5py.File,
    dataset_name: str = "fitpara_compressed",
) -> np.ndarray:
    """Decode compressed fitpara from HDF5.

    Args:
        h5_file: HDF5 file with compressed fitpara group
        dataset_name: Name of the compressed group

    Returns:
        (n_spectra, n_comp, 9) float32 array
    """
    if not HAS_LZ4:
        raise ImportError("lz4 package required for decompression")

    grp = h5_file[dataset_name]
    compressed_bytes = grp["data"][:].tobytes()
    shape = tuple(grp.attrs["shape"])
    scales = grp.attrs["scales"]
    offsets = grp.attrs["offsets"]

    n_spectra, n_comp, n_cols = shape

    # LZ4 decompress
    decompressed = lz4.frame.decompress(compressed_bytes)
    arr_int16 = np.frombuffer(decompressed, dtype=np.int16).copy()

    # Column-wise decode
    fitpara = np.zeros((n_spectra, n_comp, n_cols), dtype=np.float32)
    col_size = n_spectra * n_comp

    for col_idx in range(n_cols):
        start = col_idx * col_size
        col_int16 = arr_int16[start : start + col_size].reshape(n_spectra, n_comp)
        scale = float(scales[col_idx])
        offset = float(offsets[col_idx])

        if scale != 0:
            fitpara[:, :, col_idx] = col_int16.astype(np.float32) / scale + offset
        else:
            fitpara[:, :, col_idx] = offset

    return fitpara


# =============================================================================
# Tier 2: LZ4 Array Compression (otherpara, xytdata)
# =============================================================================


def _compress_array_to_h5(
    h5_out: h5py.File,
    arr: np.ndarray,
    dataset_name: str,
) -> dict[str, Any]:
    """Compress numpy array with LZ4 and write to HDF5.

    Args:
        h5_out: Output HDF5 file
        arr: numpy array to compress
        dataset_name: Name for the compressed group

    Returns:
        Dict with original_bytes, compressed_bytes, ratio
    """
    if not HAS_LZ4:
        raise ImportError("lz4 package required for compression")

    raw_bytes = arr.tobytes()
    compressed_bytes = lz4.frame.compress(raw_bytes)

    if dataset_name in h5_out:
        del h5_out[dataset_name]
    grp = h5_out.create_group(dataset_name)
    grp.create_dataset("data", data=np.frombuffer(compressed_bytes, dtype=np.uint8))
    grp.attrs["shape"] = arr.shape
    grp.attrs["dtype"] = str(arr.dtype)
    grp.attrs["compressed_size"] = len(compressed_bytes)
    grp.attrs["original_size"] = len(raw_bytes)
    grp.attrs["compression_ratio"] = len(raw_bytes) / len(compressed_bytes)

    return {
        "original_bytes": len(raw_bytes),
        "compressed_bytes": len(compressed_bytes),
        "ratio": len(raw_bytes) / len(compressed_bytes),
    }


# =============================================================================
# Tier 3: Streaming Specdata uint16 Conversion
# =============================================================================


def _convert_specdata_streaming(
    f_in: h5py.File,
    f_out: h5py.File,
    chunk_size: int = 500_000,
    verbose: bool = False,
) -> dict[str, Any]:
    """Convert specdata float32 -> uint16 in streaming chunks.

    Reads input specdata in chunks to avoid loading 14+ GB into memory.

    Args:
        f_in: Input HDF5 file (read-only)
        f_out: Output HDF5 file (writable)
        chunk_size: Rows per chunk (default 500K = ~200 MB)
        verbose: Print progress

    Returns:
        Dict with stats (original_bytes, compressed_bytes, ratio, n_spectra)
    """
    specdata = f_in["specdata"]
    n_rows = specdata.shape[0]
    n_energy = specdata.shape[1]
    n_spectra = n_rows - 1  # Row 0 = energy

    # Read and store energy axis
    energy = specdata[0, :].astype(np.float32)
    if "specdata_uint16_energy" in f_out:
        del f_out["specdata_uint16_energy"]
    f_out.create_dataset("specdata_uint16_energy", data=energy)

    # Create output dataset (chunked for efficient writes)
    out_chunks = (min(chunk_size, n_spectra), n_energy)
    if "specdata_uint16" in f_out:
        del f_out["specdata_uint16"]
    ds_out = f_out.create_dataset(
        "specdata_uint16",
        shape=(n_spectra, n_energy),
        dtype=np.uint16,
        chunks=out_chunks,
    )
    ds_out.attrs["dtype_original"] = "uint16"

    original_bytes = n_spectra * n_energy * 4  # float32
    compressed_bytes = n_spectra * n_energy * 2  # uint16
    max_count = 0
    skipped = False

    # Stream in chunks
    n_chunks = (n_spectra + chunk_size - 1) // chunk_size
    for i, start in enumerate(range(0, n_spectra, chunk_size)):
        end = min(start + chunk_size, n_spectra)
        # +1 to skip energy row in source
        chunk = specdata[start + 1 : end + 1, :]

        # Validate range
        chunk_max = float(chunk.max())
        chunk_min = float(chunk.min())
        if chunk_max > 65535 or chunk_min < 0:
            if verbose:
                print(
                    f"  WARNING: specdata range [{chunk_min:.1f}, {chunk_max:.1f}] "
                    f"outside uint16 at rows {start}-{end}. Skipping specdata conversion."
                )
            # Clean up partial output
            del f_out["specdata_uint16"]
            del f_out["specdata_uint16_energy"]
            skipped = True
            break

        max_count = max(max_count, int(chunk_max))
        ds_out[start:end, :] = chunk.astype(np.uint16)

        if verbose and (i % max(1, n_chunks // 10) == 0 or i == n_chunks - 1):
            pct = (end / n_spectra) * 100
            print(f"  specdata: {pct:.0f}% ({end:,}/{n_spectra:,})")

    if skipped:
        # Copy original specdata as-is
        f_in.copy("specdata", f_out, "specdata")
        return {"skipped": True, "reason": "out_of_range"}

    ds_out.attrs["max_count"] = max_count
    f_out.flush()

    # Get actual compressed size from HDF5 storage (gzip makes it smaller than raw uint16)
    storage_size = ds_out.id.get_storage_size()
    if storage_size > 0:
        compressed_bytes = storage_size

    return {
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "ratio": original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0,
        "n_spectra": n_spectra,
        "max_count": max_count,
    }


# =============================================================================
# Main Conversion Functions
# =============================================================================


def compress_h5_file_streaming(
    input_path: str | Path,
    output_path: str | Path,
    compression: str = "full",
    chunk_size: int = 500_000,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compress an HDF5 file using streaming approach.

    Creates a NEW compressed file at output_path. Input file is never modified.

    Args:
        input_path: Path to input HDF5 file (read-only)
        output_path: Path to output compressed HDF5 file
        compression:
            'standard': fitpara only (int16+LZ4)
            'full': fitpara + otherpara/xytdata (LZ4) + specdata (uint16)
        chunk_size: Rows per specdata streaming chunk
        verbose: Print progress

    Returns:
        Dict with per-dataset compression stats
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    stats: dict[str, Any] = {"compressed_datasets": [], "input": str(input_path)}

    if not HAS_LZ4:
        raise ImportError(
            "lz4 package required for compression.\n"
            "Install with: pip install lz4  (or: uv pip install lz4)"
        )

    t_start = time.time()

    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        # --- Pass 1: Copy misc/other groups ---
        skip_datasets = {"specdata", "fitpara", "otherpara", "xytdata"}
        for name, obj in f_in.items():
            if name in skip_datasets:
                continue
            # Copy groups (misc, etc.) and other datasets as-is
            f_in.copy(obj, f_out, name)

        # Copy root-level attributes
        for key, val in f_in.attrs.items():
            f_out.attrs[key] = val

        # --- Tier 1: fitpara -> int16 + column-wise + LZ4 (streaming) ---
        if "fitpara" in f_in and isinstance(f_in["fitpara"], h5py.Dataset):
            fitpara_ds = f_in["fitpara"]
            orig_mb = fitpara_ds.nbytes / 1e6
            if verbose:
                print(f"  Compressing fitpara {fitpara_ds.shape} ({orig_mb:.0f} MB, streaming)...")
            fp_stats = _encode_fitpara_streaming_from_dataset(
                f_out, fitpara_ds, chunk_size=chunk_size
            )
            comp_mb = fp_stats["compressed_bytes"] / 1e6
            stats["fitpara"] = fp_stats
            stats["compressed_datasets"].append("fitpara")
            if verbose:
                print(f"  fitpara: {orig_mb:.1f} MB -> {comp_mb:.1f} MB ({fp_stats['ratio']:.1f}x)")

        if compression == "full":
            # --- Tier 2: otherpara -> LZ4 ---
            if "otherpara" in f_in and isinstance(f_in["otherpara"], h5py.Dataset):
                if verbose:
                    print("  Compressing otherpara...")
                arr = f_in["otherpara"][:]
                other_stats = _compress_array_to_h5(f_out, arr, "otherpara_compressed")
                stats["otherpara"] = other_stats
                stats["compressed_datasets"].append("otherpara")
                if verbose:
                    orig_mb = other_stats["original_bytes"] / 1e6
                    comp_mb = other_stats["compressed_bytes"] / 1e6
                    print(f"  otherpara: {orig_mb:.1f} MB -> {comp_mb:.1f} MB ({other_stats['ratio']:.0f}x)")
                del arr

            # --- Tier 2: xytdata -> LZ4 ---
            if "xytdata" in f_in and isinstance(f_in["xytdata"], h5py.Dataset):
                if verbose:
                    print("  Compressing xytdata...")
                arr = f_in["xytdata"][:]
                xyt_stats = _compress_array_to_h5(f_out, arr, "xytdata_compressed")
                stats["xytdata"] = xyt_stats
                stats["compressed_datasets"].append("xytdata")
                if verbose:
                    orig_mb = xyt_stats["original_bytes"] / 1e6
                    comp_mb = xyt_stats["compressed_bytes"] / 1e6
                    print(f"  xytdata: {orig_mb:.1f} MB -> {comp_mb:.2f} MB ({xyt_stats['ratio']:.0f}x)")
                del arr

            # --- Tier 3: specdata -> uint16 (streaming) ---
            if "specdata" in f_in and isinstance(f_in["specdata"], h5py.Dataset):
                if verbose:
                    shape = f_in["specdata"].shape
                    orig_gb = f_in["specdata"].nbytes / 1e9
                    print(f"  Converting specdata {shape} ({orig_gb:.1f} GB) -> uint16 (streaming)...")
                spec_stats = _convert_specdata_streaming(f_in, f_out, chunk_size, verbose)
                stats["specdata"] = spec_stats
                if not spec_stats.get("skipped"):
                    stats["compressed_datasets"].append("specdata")
                    if verbose:
                        orig_gb = spec_stats["original_bytes"] / 1e9
                        comp_gb = spec_stats["compressed_bytes"] / 1e9
                        print(f"  specdata: {orig_gb:.1f} GB -> {comp_gb:.1f} GB ({spec_stats['ratio']:.1f}x)")
        else:
            # Standard mode: copy specdata/otherpara/xytdata as-is
            for name in ("specdata", "otherpara", "xytdata"):
                if name in f_in:
                    f_in.copy(name, f_out, name)

    elapsed = time.time() - t_start
    stats["elapsed_seconds"] = round(elapsed, 1)

    # File sizes
    stats["input_size_mb"] = round(input_path.stat().st_size / 1e6, 1)
    stats["output_size_mb"] = round(output_path.stat().st_size / 1e6, 1)
    stats["overall_ratio"] = round(
        stats["input_size_mb"] / stats["output_size_mb"]
        if stats["output_size_mb"] > 0
        else 1.0,
        1,
    )

    return stats


def convert_folder(
    input_folder: str | Path,
    output_folder: str | Path | None = None,
    compression: str = "full",
    chunk_size: int = 500_000,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Convert all HDF5 files in a folder to compressed format.

    Output goes to a separate folder (default: {input_folder}_compressed).
    Original files are never modified.

    Args:
        input_folder: Folder containing original HDF5 files
        output_folder: Output folder (default: {input_folder}_compressed)
        compression: 'standard' or 'full'
        chunk_size: Rows per specdata chunk
        dry_run: Show plan without executing
        verbose: Print progress

    Returns:
        Dict with overall summary
    """
    input_folder = Path(input_folder)
    if not input_folder.exists() or not input_folder.is_dir():
        raise ValueError(f"Input folder does not exist: {input_folder}")

    if output_folder is None:
        output_folder = input_folder.parent / f"{input_folder.name}_compressed"
    else:
        output_folder = Path(output_folder)

    # Find HDF5 files (exclude fitresult_* for now — those are output files)
    h5_files = sorted(input_folder.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {input_folder}")
        return {"files": [], "total_input_mb": 0, "total_output_mb": 0}

    # Summary
    total_input_mb = sum(f.stat().st_size / 1e6 for f in h5_files)
    print(f"Input:  {input_folder}")
    print(f"Output: {output_folder}")
    print(f"Files:  {len(h5_files)}")
    print(f"Total:  {total_input_mb / 1000:.1f} GB")
    print(f"Mode:   {compression}")
    print()

    if dry_run:
        print("=== DRY RUN ===")
        for f in h5_files:
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name:40s}  {size_mb:8.1f} MB")
        print(f"\nWould create: {output_folder}")
        return {"dry_run": True, "files": [f.name for f in h5_files]}

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    results = []
    total_output_mb = 0

    for i, h5_file in enumerate(h5_files, 1):
        output_path = output_folder / h5_file.name
        input_mb = h5_file.stat().st_size / 1e6

        print(f"[{i}/{len(h5_files)}] {h5_file.name} ({input_mb:.0f} MB)")

        try:
            stats = compress_h5_file_streaming(
                h5_file, output_path, compression, chunk_size, verbose
            )
            output_mb = stats["output_size_mb"]
            total_output_mb += output_mb
            ratio = stats["overall_ratio"]
            elapsed = stats["elapsed_seconds"]
            print(f"  -> {output_mb:.0f} MB ({ratio:.1f}x) in {elapsed:.1f}s\n")
            results.append(stats)
        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append({"error": str(e), "input": str(h5_file)})

    # Summary
    print("=" * 60)
    print(f"Total input:  {total_input_mb / 1000:.1f} GB")
    print(f"Total output: {total_output_mb / 1000:.1f} GB")
    if total_output_mb > 0:
        print(f"Overall ratio: {total_input_mb / total_output_mb:.1f}x")
    print(f"Output: {output_folder}")

    return {
        "files": results,
        "total_input_mb": round(total_input_mb, 1),
        "total_output_mb": round(total_output_mb, 1),
        "overall_ratio": round(total_input_mb / total_output_mb, 1) if total_output_mb > 0 else 1.0,
        "output_folder": str(output_folder),
    }
