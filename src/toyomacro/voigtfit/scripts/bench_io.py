#!/usr/bin/env python3
"""
HDF5 I/O Benchmark Script
=========================

Comprehensive benchmark comparing:
1. Current implementation (full read + astype)
2. Batch reading (various sizes)
3. Prefetch pipeline (I/O + compute overlap)
4. Different HDF5 cache settings

Reports:
- Raw I/O throughput (GB/s)
- Pipeline throughput (spectra/s)
- Memory usage
- Bottleneck identification

Usage:
    python -m voigtfit.scripts.bench_io data.h5
    python -m voigtfit.scripts.bench_io data.h5 --synthetic 10000000
    python -m voigtfit.scripts.bench_io data.h5 --sweep-batch
Date: 2026-01-23
"""

import argparse
import gc
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Try to import monitoring tools
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import mlx.core as mx
    from .._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND the default device can execute work
except ImportError:
    HAS_MLX = False


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    n_spectra: int
    n_energy: int
    data_bytes: int
    time_sec: float
    throughput_gbs: float
    throughput_spectra_per_sec: float
    extra: dict[str, Any] = None

    def __str__(self):
        return (
            f"{self.name:<30} "
            f"{self.time_sec:>8.3f}s "
            f"{self.throughput_gbs:>6.2f} GB/s "
            f"{self.throughput_spectra_per_sec/1e6:>6.2f}M spec/s"
        )


def format_size(nbytes: int) -> str:
    """Format bytes as human-readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} PB"


def create_synthetic_h5(
    path: str,
    n_spectra: int,
    n_energy: int = 151,
    dtype: np.dtype = np.float32,
    chunks: tuple[int, int] | None = None,
    compression: str | None = None,
) -> str:
    """Create synthetic HDF5 file for benchmarking."""
    print(f"Creating synthetic data: {n_spectra:,} spectra x {n_energy} energy")

    # Default chunks optimized for row-major access
    if chunks is None:
        chunk_rows = min(10000, n_spectra + 1)
        chunks = (chunk_rows, n_energy)

    # Create in chunks to avoid memory issues
    write_batch = 1_000_000

    with h5py.File(path, 'w') as f:
        # specdata format: row 0 = energy, rows 1: = spectra
        dset = f.create_dataset(
            'specdata',
            shape=(n_spectra + 1, n_energy),
            dtype=dtype,
            chunks=chunks,
            compression=compression,
        )

        # Write energy axis (row 0)
        dset[0, :] = np.linspace(95, 110, n_energy).astype(dtype)

        # Write spectra in batches
        for start in range(0, n_spectra, write_batch):
            end = min(start + write_batch, n_spectra)
            count = end - start
            # Generate random spectra with realistic values
            data = np.random.randn(count, n_energy).astype(dtype) * 1000 + 10000
            dset[start + 1:end + 1, :] = data

            progress = 100 * end / n_spectra
            print(f"  Writing: {progress:.0f}%", end='\r')

    print(f"\nCreated: {path}")
    size = os.path.getsize(path)
    print(f"Size: {format_size(size)}")

    return path


def benchmark_full_read(f: h5py.File, dset: h5py.Dataset, n_iterations: int = 3) -> BenchmarkResult:
    """Benchmark full array read with [:]."""
    times = []
    n_spectra = dset.shape[0] - 1
    n_energy = dset.shape[1]
    data_bytes = n_spectra * n_energy * 4  # float32

    for _ in range(n_iterations):
        gc.collect()
        t0 = time.perf_counter()
        data = dset[1:, :].astype(np.float32)  # Skip energy row
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del data

    avg_time = np.mean(times)
    return BenchmarkResult(
        name="Full read [:]",
        n_spectra=n_spectra,
        n_energy=n_energy,
        data_bytes=data_bytes,
        time_sec=avg_time,
        throughput_gbs=data_bytes / avg_time / 1e9,
        throughput_spectra_per_sec=n_spectra / avg_time,
    )


def benchmark_batch_read(
    f: h5py.File,
    dset: h5py.Dataset,
    batch_size: int,
    n_iterations: int = 3
) -> BenchmarkResult:
    """Benchmark batch reading."""
    times = []
    n_spectra = dset.shape[0] - 1
    n_energy = dset.shape[1]
    data_bytes = n_spectra * n_energy * 4

    for _ in range(n_iterations):
        gc.collect()
        t0 = time.perf_counter()

        for start in range(0, n_spectra, batch_size):
            end = min(start + batch_size, n_spectra)
            batch = dset[start + 1:end + 1, :].astype(np.float32)

        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = np.mean(times)
    return BenchmarkResult(
        name=f"Batch read (batch={batch_size:,})",
        n_spectra=n_spectra,
        n_energy=n_energy,
        data_bytes=data_bytes,
        time_sec=avg_time,
        throughput_gbs=data_bytes / avg_time / 1e9,
        throughput_spectra_per_sec=n_spectra / avg_time,
        extra={'batch_size': batch_size}
    )


def benchmark_read_direct(
    f: h5py.File,
    dset: h5py.Dataset,
    batch_size: int,
    n_iterations: int = 3
) -> BenchmarkResult:
    """Benchmark read_direct with pre-allocated buffer."""
    n_spectra = dset.shape[0] - 1
    n_energy = dset.shape[1]
    data_bytes = n_spectra * n_energy * 4

    # Pre-allocate buffer
    buffer = np.empty((batch_size, n_energy), dtype=np.float32)

    times = []
    for _ in range(n_iterations):
        gc.collect()
        t0 = time.perf_counter()

        for start in range(0, n_spectra, batch_size):
            end = min(start + batch_size, n_spectra)
            count = end - start

            if dset.dtype == np.float32:
                # Zero-copy if dtype matches
                dset.read_direct(
                    buffer[:count],
                    source_sel=np.s_[start + 1:end + 1, :]
                )
            else:
                # Must convert
                buffer[:count] = dset[start + 1:end + 1, :].astype(np.float32)

        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = np.mean(times)
    return BenchmarkResult(
        name=f"read_direct (batch={batch_size:,})",
        n_spectra=n_spectra,
        n_energy=n_energy,
        data_bytes=data_bytes,
        time_sec=avg_time,
        throughput_gbs=data_bytes / avg_time / 1e9,
        throughput_spectra_per_sec=n_spectra / avg_time,
        extra={'batch_size': batch_size}
    )


def benchmark_prefetch(
    path: str,
    batch_size: int,
    n_iterations: int = 2
) -> BenchmarkResult:
    """Benchmark prefetch pipeline."""
    from ..h5io_fast import PrefetchReader, get_dataset_info, open_h5_optimized

    with open_h5_optimized(path) as f:
        info = get_dataset_info(f, 'specdata')

    n_spectra = info.n_spectra
    n_energy = info.n_energy
    data_bytes = n_spectra * n_energy * 4

    times = []
    for _ in range(n_iterations):
        gc.collect()

        reader = PrefetchReader(
            path, 'specdata',
            batch_size=batch_size,
            n_buffers=3,
        )

        t0 = time.perf_counter()
        for batch in reader:
            # Simulate minimal compute
            _ = batch.data.mean()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = np.mean(times)
    return BenchmarkResult(
        name=f"Prefetch (batch={batch_size:,})",
        n_spectra=n_spectra,
        n_energy=n_energy,
        data_bytes=data_bytes,
        time_sec=avg_time,
        throughput_gbs=data_bytes / avg_time / 1e9,
        throughput_spectra_per_sec=n_spectra / avg_time,
        extra={'batch_size': batch_size}
    )


def benchmark_pipeline_full(
    path: str,
    batch_size: int,
    n_iterations: int = 2
) -> BenchmarkResult:
    """Benchmark full VoigtFit pipeline with I/O."""
    from ..h5io_fast import PrefetchReader, get_dataset_info, open_h5_optimized
    from ..pipeline import HybridPipeline
    from ..weight_cache import WeightMatrixCache

    with open_h5_optimized(path) as f:
        info = get_dataset_info(f, 'specdata')
        energy = f['specdata'][0, :].astype(np.float32)

    n_spectra = info.n_spectra
    n_energy = info.n_energy
    data_bytes = n_spectra * n_energy * 4

    # Setup pipeline
    cache = WeightMatrixCache()
    pipeline = HybridPipeline(
        cache=cache,
        use_mlx=HAS_MLX,
        enable_stage2=False,  # Disable for raw I/O benchmark
    )

    peak_config = {
        'centers': np.array([99.5, 103.0, 107.0], dtype=np.float32),
        'sigmas': np.array([0.6, 0.8, 0.7], dtype=np.float32),
        'gamma': 0.25,
    }

    times = []
    compute_times = []

    for _ in range(n_iterations):
        gc.collect()

        reader = PrefetchReader(
            path, 'specdata',
            batch_size=batch_size,
            n_buffers=3,
        )

        t_total_start = time.perf_counter()
        t_compute = 0.0

        for batch in reader:
            t0 = time.perf_counter()
            result = pipeline.process_rowmajor(
                batch.data, 'Fe', '2p', energy, peak_config
            )
            t_compute += time.perf_counter() - t0

        t_total = time.perf_counter() - t_total_start
        times.append(t_total)
        compute_times.append(t_compute)

    avg_time = np.mean(times)
    avg_compute = np.mean(compute_times)

    return BenchmarkResult(
        name=f"Full pipeline (batch={batch_size:,})",
        n_spectra=n_spectra,
        n_energy=n_energy,
        data_bytes=data_bytes,
        time_sec=avg_time,
        throughput_gbs=data_bytes / avg_time / 1e9,
        throughput_spectra_per_sec=n_spectra / avg_time,
        extra={
            'batch_size': batch_size,
            'compute_time': avg_compute,
            'io_time': avg_time - avg_compute,
            'compute_fraction': avg_compute / avg_time,
        }
    )


def benchmark_cache_settings(path: str, batch_size: int = 200000) -> list[BenchmarkResult]:
    """Benchmark different HDF5 cache settings."""
    results = []

    cache_sizes = [
        (1024 * 1024, "1MB cache"),
        (16 * 1024 * 1024, "16MB cache"),
        (64 * 1024 * 1024, "64MB cache"),
        (256 * 1024 * 1024, "256MB cache"),
    ]

    for cache_bytes, name in cache_sizes:
        times = []

        for _ in range(3):
            gc.collect()

            with h5py.File(
                path, 'r',
                rdcc_nbytes=cache_bytes,
                rdcc_nslots=10007,
                rdcc_w0=0.75,
            ) as f:
                dset = f['specdata']
                n_spectra = dset.shape[0] - 1
                n_energy = dset.shape[1]

                t0 = time.perf_counter()
                for start in range(0, n_spectra, batch_size):
                    end = min(start + batch_size, n_spectra)
                    batch = dset[start + 1:end + 1, :].astype(np.float32)
                times.append(time.perf_counter() - t0)

        avg_time = np.mean(times)
        data_bytes = n_spectra * n_energy * 4

        results.append(BenchmarkResult(
            name=name,
            n_spectra=n_spectra,
            n_energy=n_energy,
            data_bytes=data_bytes,
            time_sec=avg_time,
            throughput_gbs=data_bytes / avg_time / 1e9,
            throughput_spectra_per_sec=n_spectra / avg_time,
            extra={'cache_bytes': cache_bytes}
        ))

    return results


def sweep_batch_sizes(path: str) -> list[BenchmarkResult]:
    """Sweep batch sizes to find optimal."""
    results = []

    batch_sizes = [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]

    with h5py.File(path, 'r') as f:
        dset = f['specdata']
        n_spectra = dset.shape[0] - 1

        for batch_size in batch_sizes:
            if batch_size > n_spectra:
                continue

            result = benchmark_batch_read(f, dset, batch_size)
            results.append(result)

    return results


def run_full_benchmark(path: str, synthetic_n: int | None = None):
    """Run complete benchmark suite."""
    # Create synthetic data if requested
    if synthetic_n:
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            path = create_synthetic_h5(tmp.name, synthetic_n)
            cleanup = True
    else:
        cleanup = False

    # File info
    with h5py.File(path, 'r') as f:
        dset = f['specdata']
        n_spectra = dset.shape[0] - 1
        n_energy = dset.shape[1]
        dtype = dset.dtype
        chunks = dset.chunks
        compression = dset.compression

    data_bytes = n_spectra * n_energy * 4
    file_size = os.path.getsize(path)

    print("=" * 80)
    print("HDF5 I/O Benchmark")
    print("=" * 80)
    print(f"File: {path}")
    print(f"Spectra: {n_spectra:,}")
    print(f"Energy points: {n_energy}")
    print(f"Data type: {dtype}")
    print(f"Chunks: {chunks}")
    print(f"Compression: {compression or 'none'}")
    print(f"Raw data size: {format_size(data_bytes)}")
    print(f"File size: {format_size(file_size)}")
    print(f"MLX available: {HAS_MLX}")
    print()

    results = []

    # 1. Full read benchmark
    print("-" * 80)
    print("1. Full Read Benchmarks")
    print("-" * 80)

    with h5py.File(path, 'r') as f:
        dset = f['specdata']

        r = benchmark_full_read(f, dset)
        results.append(r)
        print(r)

    # 2. Batch size sweep
    print()
    print("-" * 80)
    print("2. Batch Size Sweep")
    print("-" * 80)

    batch_results = sweep_batch_sizes(path)
    for r in batch_results:
        results.append(r)
        print(r)

    # Find optimal batch size
    best_batch = max(batch_results, key=lambda x: x.throughput_gbs)
    print(f"\nOptimal batch size: {best_batch.extra['batch_size']:,}")

    # 3. read_direct vs slice
    print()
    print("-" * 80)
    print("3. read_direct vs Slice")
    print("-" * 80)

    optimal_batch = best_batch.extra['batch_size']

    with h5py.File(path, 'r') as f:
        dset = f['specdata']
        r = benchmark_read_direct(f, dset, optimal_batch)
        results.append(r)
        print(r)

    # 4. Prefetch pipeline
    print()
    print("-" * 80)
    print("4. Prefetch Pipeline")
    print("-" * 80)

    try:
        r = benchmark_prefetch(path, optimal_batch)
        results.append(r)
        print(r)
    except Exception as e:
        print(f"Prefetch benchmark failed: {e}")

    # 5. HDF5 cache settings
    print()
    print("-" * 80)
    print("5. HDF5 Cache Settings")
    print("-" * 80)

    cache_results = benchmark_cache_settings(path, optimal_batch)
    for r in cache_results:
        results.append(r)
        print(r)

    # 6. Full pipeline benchmark
    print()
    print("-" * 80)
    print("6. Full VoigtFit Pipeline")
    print("-" * 80)

    try:
        r = benchmark_pipeline_full(path, optimal_batch)
        results.append(r)
        print(r)
        if r.extra:
            print(f"   Compute: {r.extra.get('compute_time', 0):.3f}s "
                  f"({100*r.extra.get('compute_fraction', 0):.0f}%)")
            print(f"   I/O overhead: {r.extra.get('io_time', 0):.3f}s")
    except Exception as e:
        print(f"Pipeline benchmark failed: {e}")

    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    # Theoretical max (SSD bandwidth)
    ssd_max_gbs = 7.4  # Typical NVMe SSD
    print(f"Theoretical SSD max: {ssd_max_gbs:.1f} GB/s")

    # Best achieved
    best = max(results, key=lambda x: x.throughput_gbs)
    print(f"Best achieved: {best.throughput_gbs:.2f} GB/s ({best.name})")
    print(f"SSD utilization: {100 * best.throughput_gbs / ssd_max_gbs:.0f}%")

    # Recommendations
    print()
    print("Recommendations:")
    if dtype == np.float64:
        print("  - Convert to float32 to halve I/O")
    if chunks is None or (chunks and chunks[1] < n_energy):
        print(f"  - Use chunks=({min(10000, n_spectra)}, {n_energy}) for row-major access")
    if compression:
        print("  - Disable compression for frequently read data")
    print(f"  - Use batch_size={optimal_batch:,} for optimal throughput")

    # Cleanup
    if cleanup:
        os.unlink(path)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='HDF5 I/O Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('h5_file', nargs='?', help='HDF5 file to benchmark')
    parser.add_argument('--synthetic', '-s', type=int, metavar='N',
                       help='Create synthetic data with N spectra')
    parser.add_argument('--sweep-batch', action='store_true',
                       help='Run batch size sweep only')

    args = parser.parse_args()

    if args.synthetic:
        run_full_benchmark(None, args.synthetic)
    elif args.h5_file:
        if not Path(args.h5_file).exists():
            print(f"Error: File not found: {args.h5_file}")
            return 1
        run_full_benchmark(args.h5_file)
    else:
        # Default: create small synthetic test
        print("No file specified, creating synthetic test data...")
        run_full_benchmark(None, 1_000_000)


if __name__ == '__main__':
    exit(main() or 0)
