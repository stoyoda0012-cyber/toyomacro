#!/usr/bin/env python3
"""
Minimal I/O Benchmark: Before vs After Optimization
====================================================

Compares I/O performance for:
1. BEFORE: h5py.File default cache + full read [:]
2. AFTER:  h5py.File 64MB cache + batch read

Measures:
- I/O throughput (GB/s)
- Spectra/s throughput
- Memory allocation overhead

Usage:
    python -m toyomacro.voigtfit.scripts.bench_io_minimal data.h5
    python -m toyomacro.voigtfit.scripts.bench_io_minimal --synthetic 10000000
    python -m toyomacro.voigtfit.scripts.bench_io_minimal --synthetic 10000000 --dtype float64
"""

import argparse
import gc
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


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
    dtype: str = 'float32',
    chunks: tuple[int, int] = None,
) -> tuple[str, int]:
    """Create synthetic HDF5 file for benchmarking."""
    dtype = np.dtype(dtype)
    print(f"Creating synthetic data: {n_spectra:,} spectra x {n_energy} energy (dtype={dtype})")

    if chunks is None:
        chunks = (min(10000, n_spectra + 1), n_energy)

    with h5py.File(path, 'w') as f:
        # specdata format: row 0 = energy, rows 1: = spectra
        dset = f.create_dataset(
            'specdata',
            shape=(n_spectra + 1, n_energy),
            dtype=dtype,
            chunks=chunks,
        )

        # Write energy axis
        dset[0, :] = np.linspace(95, 110, n_energy).astype(dtype)

        # Write spectra in batches
        write_batch = 500_000
        for start in range(0, n_spectra, write_batch):
            end = min(start + write_batch, n_spectra)
            data = np.random.randn(end - start, n_energy).astype(dtype) * 1000 + 10000
            dset[start + 1:end + 1, :] = data
            print(f"  Writing: {100 * end / n_spectra:.0f}%", end='\r')

    size = os.path.getsize(path)
    print(f"\nCreated: {path} ({format_size(size)})")
    return path, size


def bench_before_full_read(path: str, n_iter: int = 3) -> dict[str, Any]:
    """
    BEFORE: Default cache + full read with [:] + astype.
    This is the baseline (worst case).
    """
    times = []

    with h5py.File(path, 'r') as f:
        dset = f['specdata']
        n_spectra = dset.shape[0] - 1
        n_energy = dset.shape[1]
        raw_bytes = n_spectra * n_energy * 4  # float32 output
        src_dtype = dset.dtype

    for i in range(n_iter):
        gc.collect()

        # Default h5py.File (1MB cache)
        with h5py.File(path, 'r') as f:
            dset = f['specdata']

            t0 = time.perf_counter()
            # Full read + astype (the worst pattern)
            data = dset[1:, :].astype(np.float32)
            t1 = time.perf_counter()

        times.append(t1 - t0)
        del data

    avg_time = np.mean(times)
    return {
        'name': 'BEFORE: Full read [:] + astype',
        'cache_mb': 1,
        'batch_size': 'N/A (full)',
        'time_sec': avg_time,
        'throughput_gbs': raw_bytes / avg_time / 1e9,
        'throughput_spectra_per_sec': n_spectra / avg_time,
        'n_spectra': n_spectra,
        'raw_bytes': raw_bytes,
    }


def bench_after_batch_read(
    path: str,
    batch_size: int = 200_000,
    n_iter: int = 3,
) -> dict[str, Any]:
    """
    AFTER: 64MB cache + batch read.
    This is the optimized pattern.
    """
    times = []

    with h5py.File(path, 'r') as f:
        dset = f['specdata']
        n_spectra = dset.shape[0] - 1
        n_energy = dset.shape[1]
        raw_bytes = n_spectra * n_energy * 4
        src_dtype = dset.dtype

    for i in range(n_iter):
        gc.collect()

        # Optimized h5py.File (64MB cache)
        with h5py.File(path, 'r',
                       rdcc_nbytes=64 * 1024 * 1024,
                       rdcc_nslots=10007,
                       rdcc_w0=0.75) as f:
            dset = f['specdata']

            t0 = time.perf_counter()

            # Batch read (the optimized pattern)
            for start in range(0, n_spectra, batch_size):
                end = min(start + batch_size, n_spectra)
                batch = dset[start + 1:end + 1, :].astype(np.float32)
                # In real usage, we'd process the batch here

            t1 = time.perf_counter()

        times.append(t1 - t0)

    avg_time = np.mean(times)
    return {
        'name': f'AFTER: Batch read (batch={batch_size:,})',
        'cache_mb': 64,
        'batch_size': batch_size,
        'time_sec': avg_time,
        'throughput_gbs': raw_bytes / avg_time / 1e9,
        'throughput_spectra_per_sec': n_spectra / avg_time,
        'n_spectra': n_spectra,
        'raw_bytes': raw_bytes,
    }


def bench_after_read_direct(
    path: str,
    batch_size: int = 200_000,
    n_iter: int = 3,
) -> dict[str, Any]:
    """
    AFTER + read_direct: 64MB cache + batch read + buffer reuse.
    Only effective if source dtype is float32.
    """
    times = []

    with h5py.File(path, 'r') as f:
        dset = f['specdata']
        n_spectra = dset.shape[0] - 1
        n_energy = dset.shape[1]
        raw_bytes = n_spectra * n_energy * 4
        src_dtype = dset.dtype

    # Pre-allocate buffer
    buffer = np.empty((batch_size, n_energy), dtype=np.float32)

    for i in range(n_iter):
        gc.collect()

        with h5py.File(path, 'r',
                       rdcc_nbytes=64 * 1024 * 1024,
                       rdcc_nslots=10007,
                       rdcc_w0=0.75) as f:
            dset = f['specdata']

            t0 = time.perf_counter()

            for start in range(0, n_spectra, batch_size):
                end = min(start + batch_size, n_spectra)
                count = end - start

                if src_dtype == np.float32:
                    # Zero-copy with read_direct
                    try:
                        dset.read_direct(
                            buffer[:count],
                            source_sel=np.s_[start + 1:end + 1, :]
                        )
                    except TypeError:
                        # Fallback
                        buffer[:count] = dset[start + 1:end + 1, :].astype(np.float32)
                else:
                    # dtype conversion required
                    buffer[:count] = dset[start + 1:end + 1, :].astype(np.float32)

            t1 = time.perf_counter()

        times.append(t1 - t0)

    avg_time = np.mean(times)
    direct_note = '+ read_direct' if src_dtype == np.float32 else '(astype fallback)'
    return {
        'name': f'AFTER: Batch {direct_note} (batch={batch_size:,})',
        'cache_mb': 64,
        'batch_size': batch_size,
        'time_sec': avg_time,
        'throughput_gbs': raw_bytes / avg_time / 1e9,
        'throughput_spectra_per_sec': n_spectra / avg_time,
        'n_spectra': n_spectra,
        'raw_bytes': raw_bytes,
    }


def bench_cache_comparison(
    path: str,
    batch_size: int = 200_000,
    n_iter: int = 3,
) -> dict[str, Any]:
    """
    Compare default 1MB cache vs 64MB cache with same batch pattern.
    """
    with h5py.File(path, 'r') as f:
        dset = f['specdata']
        n_spectra = dset.shape[0] - 1
        n_energy = dset.shape[1]
        raw_bytes = n_spectra * n_energy * 4

    times_1mb = []
    times_64mb = []

    for i in range(n_iter):
        gc.collect()

        # 1MB cache (default)
        with h5py.File(path, 'r') as f:
            dset = f['specdata']
            t0 = time.perf_counter()
            for start in range(0, n_spectra, batch_size):
                end = min(start + batch_size, n_spectra)
                batch = dset[start + 1:end + 1, :].astype(np.float32)
            times_1mb.append(time.perf_counter() - t0)

        gc.collect()

        # 64MB cache
        with h5py.File(path, 'r',
                       rdcc_nbytes=64 * 1024 * 1024,
                       rdcc_nslots=10007) as f:
            dset = f['specdata']
            t0 = time.perf_counter()
            for start in range(0, n_spectra, batch_size):
                end = min(start + batch_size, n_spectra)
                batch = dset[start + 1:end + 1, :].astype(np.float32)
            times_64mb.append(time.perf_counter() - t0)

    avg_1mb = np.mean(times_1mb)
    avg_64mb = np.mean(times_64mb)

    return {
        '1mb_time': avg_1mb,
        '64mb_time': avg_64mb,
        'speedup': avg_1mb / avg_64mb,
        '1mb_gbs': raw_bytes / avg_1mb / 1e9,
        '64mb_gbs': raw_bytes / avg_64mb / 1e9,
    }


def run_benchmark(path: str, batch_sizes: list = None):
    """Run complete benchmark suite."""
    if batch_sizes is None:
        batch_sizes = [100_000, 200_000, 500_000]

    # Get file info
    with h5py.File(path, 'r') as f:
        dset = f['specdata']
        n_spectra = dset.shape[0] - 1
        n_energy = dset.shape[1]
        src_dtype = dset.dtype
        chunks = dset.chunks

    raw_bytes = n_spectra * n_energy * 4
    file_size = os.path.getsize(path)

    print("=" * 70)
    print("HDF5 I/O Benchmark: Before vs After Optimization")
    print("=" * 70)
    print(f"File: {path}")
    print(f"Spectra: {n_spectra:,}")
    print(f"Energy points: {n_energy}")
    print(f"Source dtype: {src_dtype}")
    print(f"Chunks: {chunks}")
    print(f"Raw data size (float32 out): {format_size(raw_bytes)}")
    print(f"File size: {format_size(file_size)}")
    print()

    results = []

    # BEFORE: Full read
    print("Running BEFORE benchmark (full read)...")
    r_before = bench_before_full_read(path)
    results.append(r_before)
    print(f"  Time: {r_before['time_sec']:.3f}s, "
          f"Throughput: {r_before['throughput_gbs']:.2f} GB/s, "
          f"{r_before['throughput_spectra_per_sec']/1e6:.2f}M spec/s")

    # Cache comparison
    print("\nComparing cache sizes (same batch pattern)...")
    cache_cmp = bench_cache_comparison(path, batch_sizes[1] if len(batch_sizes) > 1 else batch_sizes[0])
    print(f"  1MB cache: {cache_cmp['1mb_time']:.3f}s ({cache_cmp['1mb_gbs']:.2f} GB/s)")
    print(f"  64MB cache: {cache_cmp['64mb_time']:.3f}s ({cache_cmp['64mb_gbs']:.2f} GB/s)")
    print(f"  Cache speedup: {cache_cmp['speedup']:.2f}x")

    # AFTER: Batch read (various sizes)
    print()
    for batch_size in batch_sizes:
        if batch_size > n_spectra:
            continue

        print(f"Running AFTER benchmark (batch={batch_size:,})...")
        r_after = bench_after_batch_read(path, batch_size)
        results.append(r_after)
        print(f"  Time: {r_after['time_sec']:.3f}s, "
              f"Throughput: {r_after['throughput_gbs']:.2f} GB/s, "
              f"{r_after['throughput_spectra_per_sec']/1e6:.2f}M spec/s")

    # AFTER + read_direct (best case)
    best_batch = batch_sizes[1] if len(batch_sizes) > 1 else batch_sizes[0]
    print(f"\nRunning AFTER + buffer reuse benchmark (batch={best_batch:,})...")
    r_direct = bench_after_read_direct(path, best_batch)
    results.append(r_direct)
    print(f"  Time: {r_direct['time_sec']:.3f}s, "
          f"Throughput: {r_direct['throughput_gbs']:.2f} GB/s, "
          f"{r_direct['throughput_spectra_per_sec']/1e6:.2f}M spec/s")

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Method':<45} {'Time':>8} {'GB/s':>8} {'M spec/s':>10} {'Speedup':>8}")
    print("-" * 70)

    baseline_time = results[0]['time_sec']
    for r in results:
        speedup = baseline_time / r['time_sec']
        print(f"{r['name']:<45} {r['time_sec']:>7.3f}s {r['throughput_gbs']:>7.2f} "
              f"{r['throughput_spectra_per_sec']/1e6:>9.2f} {speedup:>7.2f}x")

    # Find best result
    best = max(results, key=lambda x: x['throughput_gbs'])
    best_speedup = baseline_time / best['time_sec']

    print()
    print(f"Best result: {best['name']}")
    print(f"Speedup: {best_speedup:.2f}x over baseline")

    # Note about dtype
    if src_dtype != np.float32:
        print()
        print(f"Note: Source dtype is {src_dtype}, so astype conversion is required.")
        print("      For maximum speedup, regenerate HDF5 with dtype=float32.")
        print(f"      Current overhead: read {src_dtype} + convert to float32")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='HDF5 I/O Benchmark: Before vs After',
    )
    parser.add_argument('h5_file', nargs='?', help='HDF5 file to benchmark')
    parser.add_argument('--synthetic', '-s', type=int, metavar='N',
                       help='Create synthetic data with N spectra')
    parser.add_argument('--dtype', '-d', default='float32',
                       choices=['float32', 'float64'],
                       help='Source dtype for synthetic data')
    parser.add_argument('--batch-sizes', '-b', type=int, nargs='+',
                       default=[100_000, 200_000, 500_000],
                       help='Batch sizes to test')

    args = parser.parse_args()

    if args.synthetic:
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            path, _ = create_synthetic_h5(tmp.name, args.synthetic, dtype=args.dtype)
            run_benchmark(path, args.batch_sizes)
            os.unlink(path)
    elif args.h5_file:
        if not Path(args.h5_file).exists():
            print(f"Error: File not found: {args.h5_file}")
            return 1
        run_benchmark(args.h5_file, args.batch_sizes)
    else:
        # Default: create 1M spectra test
        print("No file specified, creating synthetic test (1M spectra)...")
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            path, _ = create_synthetic_h5(tmp.name, 1_000_000)
            run_benchmark(path, args.batch_sizes)
            os.unlink(path)


if __name__ == '__main__':
    exit(main() or 0)
