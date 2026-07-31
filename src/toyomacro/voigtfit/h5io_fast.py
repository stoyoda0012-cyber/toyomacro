"""
High-Performance HDF5 I/O for VoigtFit
======================================

Optimized I/O routines that eliminate:
1. Full array reads ([:]) when only batches are needed
2. Unnecessary astype() copies
3. Non-contiguous memory layouts

Key optimizations:
- Streaming batch reads with pre-allocated buffers
- read_direct() for zero-copy when dtype matches
- Ring buffer for I/O + compute overlap (prefetch)
- rdcc_nbytes tuning for HDF5 chunk cache

Performance targets (M1 Max, SSD):
- Raw I/O: 5-7 GB/s (SSD limit)
- With float64->float32 conversion: 3-4 GB/s
- With prefetch overlap: hide I/O latency behind compute
"""

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from queue import Queue
from typing import Any

import h5py
import numpy as np

# =============================================================================
# Constants and Defaults
# =============================================================================

# HDF5 chunk cache settings
# rdcc_nbytes: cache size in bytes (default 1MB is too small)
# rdcc_nslots: number of hash table slots (should be prime)
# rdcc_w0: preemption policy (0 = fully read, 1 = never preempt)
DEFAULT_RDCC_NBYTES = 64 * 1024 * 1024  # 64MB cache
DEFAULT_RDCC_NSLOTS = 10007  # Prime number for hash
DEFAULT_RDCC_W0 = 0.75  # Favor chunks being read fully

# Optimal batch sizes for different scenarios
BATCH_SIZE_SMALL = 50_000      # < 10M spectra
BATCH_SIZE_MEDIUM = 200_000    # 10M - 50M spectra
BATCH_SIZE_LARGE = 500_000     # > 50M spectra


# =============================================================================
# Data Containers
# =============================================================================

@dataclass
class BatchData:
    """Container for a batch of spectral data."""
    data: np.ndarray          # (batch_size, n_energy) float32, C-contiguous
    batch_idx: int            # Batch index (0-based)
    start_idx: int            # Starting spectrum index in full dataset
    end_idx: int              # Ending spectrum index (exclusive)
    is_last: bool = False     # True if this is the last batch


@dataclass
class DatasetInfo:
    """Metadata about an HDF5 dataset."""
    path: str                 # Dataset path in HDF5 file
    shape: tuple[int, ...]    # Dataset shape
    dtype: np.dtype           # Dataset dtype
    chunks: tuple[int, ...] | None  # Chunk shape
    raw_bytes: int            # Total raw size in bytes
    n_spectra: int            # Number of spectra (rows)
    n_energy: int             # Number of energy points (columns)


# =============================================================================
# Low-Level I/O Functions
# =============================================================================

def get_optimal_batch_size(n_spectra: int, n_energy: int, dtype: np.dtype) -> int:
    """
    Calculate optimal batch size based on data size.

    Considerations:
    - Larger batches = fewer I/O operations = higher throughput
    - But too large = memory pressure and poor cache utilization
    - Sweet spot is usually 1-16 MB per batch

    Args:
        n_spectra: Total number of spectra
        n_energy: Number of energy points
        dtype: Data type

    Returns:
        Optimal batch size in number of spectra
    """
    bytes_per_spectrum = n_energy * np.dtype(np.float32).itemsize

    # Target 4MB per batch (good for L3 cache and HDF5 chunk cache)
    target_bytes = 4 * 1024 * 1024
    batch_size = target_bytes // bytes_per_spectrum

    # Clamp to reasonable range
    batch_size = max(10_000, min(batch_size, 1_000_000, n_spectra))

    return batch_size


def open_h5_optimized(
    path: str,
    rdcc_nbytes: int = DEFAULT_RDCC_NBYTES,
    rdcc_nslots: int = DEFAULT_RDCC_NSLOTS,
    rdcc_w0: float = DEFAULT_RDCC_W0,
) -> h5py.File:
    """
    Open HDF5 file with optimized cache settings.

    Args:
        path: Path to HDF5 file
        rdcc_nbytes: Chunk cache size in bytes
        rdcc_nslots: Number of hash table slots
        rdcc_w0: Preemption policy

    Returns:
        h5py.File handle
    """
    return h5py.File(
        path, 'r',
        rdcc_nbytes=rdcc_nbytes,
        rdcc_nslots=rdcc_nslots,
        rdcc_w0=rdcc_w0,
    )


def get_dataset_info(f: h5py.File, dataset_path: str) -> DatasetInfo:
    """Get metadata about a dataset without reading data."""
    dset = f[dataset_path]

    # Handle specdata format where row 0 is energy axis
    if dataset_path == 'specdata':
        n_spectra = dset.shape[0] - 1  # First row is energy
        n_energy = dset.shape[1]
    else:
        n_spectra = dset.shape[0]
        n_energy = dset.shape[1] if len(dset.shape) > 1 else 1

    return DatasetInfo(
        path=dataset_path,
        shape=dset.shape,
        dtype=dset.dtype,
        chunks=dset.chunks,
        raw_bytes=np.prod(dset.shape) * dset.dtype.itemsize,
        n_spectra=n_spectra,
        n_energy=n_energy,
    )


def read_batch_direct(
    dset: h5py.Dataset,
    out: np.ndarray,
    start: int,
    end: int,
    row_offset: int = 0,
) -> np.ndarray:
    """
    Read batch directly into pre-allocated buffer.

    This avoids memory allocation for each batch.

    Args:
        dset: HDF5 dataset
        out: Pre-allocated output buffer (batch_size, n_energy), float32
        start: Starting row index
        end: Ending row index (exclusive)
        row_offset: Offset for specdata format (1 if row 0 is energy)

    Returns:
        View into out buffer containing the read data
    """
    count = end - start

    # Use read_direct if dtype matches (fastest)
    if dset.dtype == np.float32:
        # Create memory space for the slice
        mspace = h5py.h5s.create_simple((count, dset.shape[1]))
        fspace = dset.id.get_space()
        fspace.select_hyperslab((start + row_offset, 0), (count, dset.shape[1]))

        dset.id.read(mspace, fspace, out[:count], dxpl=None)
        return out[:count]
    else:
        # Fall back to regular slice + astype (creates intermediate copy)
        out[:count] = dset[start + row_offset:end + row_offset, :].astype(np.float32)
        return out[:count]


def read_batch_slice(
    dset: h5py.Dataset,
    start: int,
    end: int,
    row_offset: int = 0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Read batch using standard slicing.

    Simpler than read_direct but creates new array each time.

    Args:
        dset: HDF5 dataset
        start: Starting row index
        end: Ending row index (exclusive)
        row_offset: Offset for specdata format
        dtype: Output dtype

    Returns:
        Batch data as contiguous array
    """
    data = dset[start + row_offset:end + row_offset, :]

    if data.dtype != dtype:
        data = data.astype(dtype, copy=False)

    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)

    return data


# =============================================================================
# Batch Iterators
# =============================================================================

def iter_batches(
    h5_path: str,
    dataset_path: str = 'specdata',
    batch_size: int | None = None,
    dtype: np.dtype = np.float32,
    preallocate: bool = True,
) -> Iterator[BatchData]:
    """
    Iterate over dataset in batches.

    This is the main interface for streaming reads.

    Args:
        h5_path: Path to HDF5 file
        dataset_path: Path to dataset within file
        batch_size: Number of spectra per batch (auto if None)
        dtype: Output dtype
        preallocate: Pre-allocate buffer for reuse

    Yields:
        BatchData objects

    Example:
        for batch in iter_batches('data.h5', batch_size=100000):
            process(batch.data)  # (batch_size, n_energy) float32
    """
    with open_h5_optimized(h5_path) as f:
        info = get_dataset_info(f, dataset_path)
        dset = f[dataset_path]

        # Determine row offset (1 for specdata where row 0 is energy)
        row_offset = 1 if dataset_path == 'specdata' else 0

        # Auto batch size
        if batch_size is None:
            batch_size = get_optimal_batch_size(
                info.n_spectra, info.n_energy, info.dtype
            )

        # Pre-allocate buffer
        buffer = None
        if preallocate:
            buffer = np.empty((batch_size, info.n_energy), dtype=dtype)

        # Iterate
        n_batches = (info.n_spectra + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, info.n_spectra)
            is_last = (batch_idx == n_batches - 1)

            if preallocate:
                data = read_batch_direct(dset, buffer, start, end, row_offset)
                # Must copy for last partial batch if buffer is larger
                if end - start < batch_size:
                    data = data.copy()
            else:
                data = read_batch_slice(dset, start, end, row_offset, dtype)

            yield BatchData(
                data=data,
                batch_idx=batch_idx,
                start_idx=start,
                end_idx=end,
                is_last=is_last,
            )


def iter_batches_with_energy(
    h5_path: str,
    dataset_path: str = 'specdata',
    batch_size: int | None = None,
    dtype: np.dtype = np.float32,
) -> Iterator[tuple[BatchData, np.ndarray]]:
    """
    Iterate over batches, also returning energy axis.

    For specdata format, reads energy from row 0.

    Yields:
        (BatchData, energy) tuples
    """
    with open_h5_optimized(h5_path) as f:
        info = get_dataset_info(f, dataset_path)
        dset = f[dataset_path]

        # Read energy axis (row 0 for specdata)
        if dataset_path == 'specdata':
            energy = dset[0, :].astype(dtype)
        else:
            # Try to find energy in file
            if 'energy' in f:
                energy = f['energy'][:].astype(dtype)
            else:
                energy = np.arange(info.n_energy, dtype=dtype)

        # Use base iterator
        for batch in iter_batches(h5_path, dataset_path, batch_size, dtype):
            yield batch, energy


# =============================================================================
# Prefetch Pipeline (I/O + Compute Overlap)
# =============================================================================

class PrefetchReader:
    """
    Asynchronous HDF5 reader with prefetching.

    Loads next batch in background while current batch is being processed.
    Uses a ring buffer to minimize allocations.

    Performance: Can hide I/O latency behind compute time.
    """

    def __init__(
        self,
        h5_path: str,
        dataset_path: str = 'specdata',
        batch_size: int = 200_000,
        n_buffers: int = 3,
        dtype: np.dtype = np.float32,
    ):
        """
        Initialize prefetch reader.

        Args:
            h5_path: Path to HDF5 file
            dataset_path: Dataset path
            batch_size: Spectra per batch
            n_buffers: Number of ring buffers (2-4 typical)
            dtype: Output dtype
        """
        self.h5_path = h5_path
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.n_buffers = n_buffers
        self.dtype = dtype

        # Get dataset info
        with open_h5_optimized(h5_path) as f:
            self.info = get_dataset_info(f, dataset_path)

        # Ring buffer pool
        self._buffers: list[np.ndarray] = [
            np.empty((batch_size, self.info.n_energy), dtype=dtype)
            for _ in range(n_buffers)
        ]
        self._buffer_idx = 0

        # Threading
        self._queue: Queue[BatchData | None] = Queue(maxsize=n_buffers)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Stats
        self.io_time = 0.0
        self.wait_time = 0.0

    def _get_next_buffer(self) -> np.ndarray:
        """Get next buffer from ring."""
        buf = self._buffers[self._buffer_idx]
        self._buffer_idx = (self._buffer_idx + 1) % self.n_buffers
        return buf

    def _load_worker(self):
        """Background thread: loads batches into queue."""
        try:
            row_offset = 1 if self.dataset_path == 'specdata' else 0

            with open_h5_optimized(self.h5_path) as f:
                dset = f[self.dataset_path]
                n_spectra = self.info.n_spectra
                n_batches = (n_spectra + self.batch_size - 1) // self.batch_size

                for batch_idx in range(n_batches):
                    if self._stop_event.is_set():
                        break

                    start = batch_idx * self.batch_size
                    end = min(start + self.batch_size, n_spectra)
                    is_last = (batch_idx == n_batches - 1)

                    # Get buffer and read data
                    t0 = time.perf_counter()

                    buf = self._get_next_buffer()
                    count = end - start

                    # Read directly into buffer
                    if dset.dtype == self.dtype:
                        dset.read_direct(
                            buf[:count],
                            source_sel=np.s_[start + row_offset:end + row_offset, :]
                        )
                        data = buf[:count]
                    else:
                        data = dset[start + row_offset:end + row_offset, :].astype(self.dtype)

                    self.io_time += time.perf_counter() - t0

                    # Ensure C-contiguous
                    if not data.flags['C_CONTIGUOUS']:
                        data = np.ascontiguousarray(data)

                    batch = BatchData(
                        data=data,
                        batch_idx=batch_idx,
                        start_idx=start,
                        end_idx=end,
                        is_last=is_last,
                    )

                    # Put into queue (blocks if full)
                    self._queue.put(batch)

                # Signal end
                self._queue.put(None)

        except Exception as e:
            print(f"PrefetchReader error: {e}")
            self._queue.put(None)

    def start(self):
        """Start background loading thread."""
        self._stop_event.clear()
        self.io_time = 0.0
        self.wait_time = 0.0
        self._thread = threading.Thread(target=self._load_worker, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __iter__(self) -> Iterator[BatchData]:
        """Iterate over batches with prefetching."""
        self.start()
        try:
            while True:
                t0 = time.perf_counter()
                batch = self._queue.get()
                self.wait_time += time.perf_counter() - t0

                if batch is None:
                    break
                yield batch
        finally:
            self.stop()

    def get_stats(self) -> dict[str, float]:
        """Get timing statistics."""
        return {
            'io_time': self.io_time,
            'wait_time': self.wait_time,
            'io_throughput': self.info.raw_bytes / self.io_time if self.io_time > 0 else 0,
        }


# =============================================================================
# High-Level Processing Functions
# =============================================================================

def process_file_streaming(
    h5_path: str,
    process_fn,
    dataset_path: str = 'specdata',
    batch_size: int | None = None,
    use_prefetch: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Process HDF5 file with streaming I/O.

    Args:
        h5_path: Path to HDF5 file
        process_fn: Function to process each batch: process_fn(batch_data, energy) -> result
        dataset_path: Dataset path
        batch_size: Batch size (auto if None)
        use_prefetch: Use background prefetching
        verbose: Print progress

    Returns:
        Dict with aggregated results and timing
    """
    # Get file info
    with open_h5_optimized(h5_path) as f:
        info = get_dataset_info(f, dataset_path)

        # Read energy
        if dataset_path == 'specdata':
            energy = f[dataset_path][0, :].astype(np.float32)
        elif 'energy' in f:
            energy = f['energy'][:].astype(np.float32)
        else:
            energy = np.arange(info.n_energy, dtype=np.float32)

    # Auto batch size
    if batch_size is None:
        batch_size = get_optimal_batch_size(info.n_spectra, info.n_energy, info.dtype)

    if verbose:
        print(f"Processing {h5_path}")
        print(f"  Spectra: {info.n_spectra:,}")
        print(f"  Energy points: {info.n_energy}")
        print(f"  Batch size: {batch_size:,}")
        print(f"  Prefetch: {use_prefetch}")

    # Setup iterator
    if use_prefetch:
        reader = PrefetchReader(
            h5_path, dataset_path,
            batch_size=batch_size,
            n_buffers=3,
        )
        batch_iter = reader
    else:
        batch_iter = iter_batches(h5_path, dataset_path, batch_size)
        reader = None

    # Process batches
    results = []
    t_total_start = time.perf_counter()
    t_compute = 0.0

    for batch in batch_iter:
        t0 = time.perf_counter()
        result = process_fn(batch.data, energy)
        t_compute += time.perf_counter() - t0
        results.append(result)

        if verbose and batch.batch_idx % 10 == 0:
            progress = 100 * batch.end_idx / info.n_spectra
            print(f"    Progress: {progress:.1f}%")

    t_total = time.perf_counter() - t_total_start

    # Timing stats
    timing = {
        'total_time': t_total,
        'compute_time': t_compute,
        'n_spectra': info.n_spectra,
        'throughput_spectra': info.n_spectra / t_total,
        'throughput_bytes': info.raw_bytes / t_total,
    }

    if reader is not None:
        reader_stats = reader.get_stats()
        timing['io_time'] = reader_stats['io_time']
        timing['wait_time'] = reader_stats['wait_time']
        timing['io_throughput'] = reader_stats['io_throughput']

    if verbose:
        print("\nResults:")
        print(f"  Total time: {t_total:.2f}s")
        print(f"  Compute time: {t_compute:.2f}s")
        print(f"  Throughput: {timing['throughput_spectra']/1e6:.1f}M spectra/s")
        print(f"  I/O throughput: {timing['throughput_bytes']/1e9:.2f} GB/s")

    return {
        'results': results,
        'timing': timing,
        'energy': energy,
    }


# =============================================================================
# Utility Functions
# =============================================================================

def convert_to_float32(
    input_path: str,
    output_path: str,
    dataset_path: str = 'specdata',
    chunk_size: tuple[int, int] | None = None,
    compression: str | None = None,
    verbose: bool = True,
) -> str:
    """
    Convert HDF5 dataset to float32 with optimal layout.

    Creates a new file with:
    - float32 dtype (if not already)
    - Optimal chunking for row-major access
    - Optional compression

    Args:
        input_path: Input HDF5 file
        output_path: Output HDF5 file
        dataset_path: Dataset to convert
        chunk_size: Output chunk size (auto if None)
        compression: 'gzip', 'lzf', or None
        verbose: Print progress

    Returns:
        Path to output file
    """
    with h5py.File(input_path, 'r') as fin:
        info = get_dataset_info(fin, dataset_path)
        dset_in = fin[dataset_path]

        # Auto chunk size: (10000, n_energy) for row-major access
        if chunk_size is None:
            chunk_rows = min(10000, info.n_spectra if dataset_path != 'specdata' else info.n_spectra + 1)
            chunk_size = (chunk_rows, info.n_energy)

        if verbose:
            print(f"Converting {input_path} -> {output_path}")
            print(f"  Shape: {dset_in.shape}")
            print(f"  Input dtype: {dset_in.dtype}")
            print("  Output dtype: float32")
            print(f"  Chunks: {chunk_size}")
            print(f"  Compression: {compression or 'none'}")

        with h5py.File(output_path, 'w') as fout:
            # Create output dataset
            dset_out = fout.create_dataset(
                dataset_path,
                shape=dset_in.shape,
                dtype=np.float32,
                chunks=chunk_size,
                compression=compression,
            )

            # Copy in batches
            batch_size = chunk_size[0]
            n_rows = dset_in.shape[0]

            t0 = time.perf_counter()
            for start in range(0, n_rows, batch_size):
                end = min(start + batch_size, n_rows)
                dset_out[start:end] = dset_in[start:end].astype(np.float32)

                if verbose and (start // batch_size) % 10 == 0:
                    progress = 100 * end / n_rows
                    print(f"    Progress: {progress:.1f}%")

            elapsed = time.perf_counter() - t0

            # Copy other datasets
            for name, obj in fin.items():
                if name != dataset_path:
                    if isinstance(obj, h5py.Dataset):
                        fout.copy(obj, name)
                    elif isinstance(obj, h5py.Group):
                        fout.copy(obj, name)

        if verbose:
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Throughput: {info.raw_bytes / elapsed / 1e9:.2f} GB/s")

    return output_path


def recommend_chunk_size(
    n_spectra: int,
    n_energy: int,
    access_pattern: str = 'row',
    target_chunk_mb: float = 4.0,
) -> tuple[int, int]:
    """
    Recommend optimal chunk size for given data shape and access pattern.

    Args:
        n_spectra: Number of spectra (rows)
        n_energy: Number of energy points (columns)
        access_pattern: 'row' (read full rows) or 'column' (read full columns)
        target_chunk_mb: Target chunk size in MB

    Returns:
        (chunk_rows, chunk_cols)
    """
    target_bytes = target_chunk_mb * 1024 * 1024
    bytes_per_float = 4  # float32

    if access_pattern == 'row':
        # For row access: span full row width
        row_bytes = n_energy * bytes_per_float
        chunk_rows = int(target_bytes / row_bytes)
        chunk_rows = max(1, min(chunk_rows, n_spectra, 100_000))
        return (chunk_rows, n_energy)

    elif access_pattern == 'column':
        # For column access: span full column height
        col_bytes = n_spectra * bytes_per_float
        chunk_cols = int(target_bytes / col_bytes)
        chunk_cols = max(1, min(chunk_cols, n_energy))
        return (n_spectra, chunk_cols)

    else:
        raise ValueError(f"Unknown access pattern: {access_pattern}")


# =============================================================================
# CLI for Testing
# =============================================================================

def main():
    """Command-line interface for testing."""
    import argparse

    parser = argparse.ArgumentParser(description='HDF5 Fast I/O Utilities')
    parser.add_argument('action', choices=['benchmark', 'convert', 'info'],
                       help='Action to perform')
    parser.add_argument('input', help='Input HDF5 file')
    parser.add_argument('--output', '-o', help='Output file (for convert)')
    parser.add_argument('--dataset', '-d', default='specdata',
                       help='Dataset path')
    parser.add_argument('--batch-size', '-b', type=int,
                       help='Batch size')
    parser.add_argument('--prefetch', action='store_true',
                       help='Use prefetching')

    args = parser.parse_args()

    if args.action == 'info':
        with open_h5_optimized(args.input) as f:
            info = get_dataset_info(f, args.dataset)
            print(f"Dataset: {info.path}")
            print(f"Shape: {info.shape}")
            print(f"dtype: {info.dtype}")
            print(f"Chunks: {info.chunks}")
            print(f"Size: {info.raw_bytes / 1e9:.2f} GB")

    elif args.action == 'benchmark':
        # Simple benchmark: just read all data
        def process(data, energy):
            return data.mean()

        result = process_file_streaming(
            args.input,
            process,
            args.dataset,
            args.batch_size,
            args.prefetch,
            verbose=True,
        )

    elif args.action == 'convert':
        if not args.output:
            print("Error: --output required for convert")
            return 1

        convert_to_float32(args.input, args.output, args.dataset, verbose=True)


if __name__ == '__main__':
    main()
