#!/usr/bin/env python3
"""
HDF5 Structure Inspector for I/O Optimization
==============================================

Inspects HDF5 files and reports:
- Dataset shapes, dtypes, chunks, compression
- Memory layout (C vs Fortran order)
- Estimated read patterns and bottlenecks
- Recommendations for optimal chunk sizes

Usage:
    python -m voigtfit.tools.inspect_h5 input.h5
    python -m voigtfit.tools.inspect_h5 input.h5 --dataset Y
    python -m voigtfit.tools.inspect_h5 input.h5 --benchmark
Date: 2026-01-23
"""

import argparse
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def format_size(nbytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} PB"


def format_rate(bytes_per_sec: float) -> str:
    """Format throughput as human-readable string."""
    for unit in ['B/s', 'KB/s', 'MB/s', 'GB/s']:
        if bytes_per_sec < 1024:
            return f"{bytes_per_sec:.2f} {unit}"
        bytes_per_sec /= 1024
    return f"{bytes_per_sec:.2f} TB/s"


def estimate_memory_order(dset: h5py.Dataset) -> str:
    """Estimate memory order from chunk layout."""
    if dset.chunks is None:
        return 'contiguous'

    shape = dset.shape
    chunks = dset.chunks

    if len(shape) == 2:
        # For 2D: if chunks[0] >> chunks[1], likely Fortran order
        # if chunks[1] >> chunks[0], likely C order
        if chunks[0] >= shape[0] and chunks[1] < shape[1]:
            return 'C-order (row-major)'
        elif chunks[1] >= shape[1] and chunks[0] < shape[0]:
            return 'Fortran-order (column-major)'
        else:
            return 'mixed'
    return 'unknown'


def get_compression_info(dset: h5py.Dataset) -> dict[str, Any]:
    """Get compression information."""
    info = {
        'compression': dset.compression,
        'compression_opts': dset.compression_opts,
        'shuffle': dset.shuffle,
        'fletcher32': dset.fletcher32,
    }

    # Estimate compression ratio if possible
    if dset.id.get_storage_size() > 0:
        raw_size = np.prod(dset.shape) * dset.dtype.itemsize
        stored_size = dset.id.get_storage_size()
        info['compression_ratio'] = raw_size / stored_size if stored_size > 0 else 1.0
        info['stored_size'] = stored_size

    return info


def inspect_dataset(dset: h5py.Dataset, name: str) -> dict[str, Any]:
    """Inspect a single dataset and return detailed info."""
    raw_size = np.prod(dset.shape) * dset.dtype.itemsize

    info = {
        'name': name,
        'shape': dset.shape,
        'dtype': str(dset.dtype),
        'dtype_size': dset.dtype.itemsize,
        'chunks': dset.chunks,
        'raw_size': raw_size,
        'raw_size_str': format_size(raw_size),
        'memory_order': estimate_memory_order(dset),
        'fillvalue': dset.fillvalue,
    }

    # Add compression info
    comp_info = get_compression_info(dset)
    info.update(comp_info)

    # Calculate read efficiency metrics
    if dset.chunks:
        chunk_bytes = np.prod(dset.chunks) * dset.dtype.itemsize
        info['chunk_bytes'] = chunk_bytes
        info['chunk_bytes_str'] = format_size(chunk_bytes)

        # For row-major access pattern (read rows)
        if len(dset.shape) == 2:
            row_bytes = dset.shape[1] * dset.dtype.itemsize
            chunks_per_row = np.ceil(dset.shape[1] / dset.chunks[1])
            info['row_bytes'] = row_bytes
            info['chunks_per_row'] = chunks_per_row

            # Efficiency: how much of chunk data is used when reading a row
            if dset.chunks[0] > 1:
                info['row_read_efficiency'] = 1.0 / dset.chunks[0]  # Only 1/N rows used
            else:
                info['row_read_efficiency'] = 1.0

    return info


def inspect_file(h5_path: str) -> dict[str, Any]:
    """Inspect HDF5 file structure."""
    result = {
        'path': h5_path,
        'datasets': {},
        'groups': [],
        'total_size': 0,
    }

    with h5py.File(h5_path, 'r') as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                info = inspect_dataset(obj, name)
                result['datasets'][name] = info
                result['total_size'] += info['raw_size']
            elif isinstance(obj, h5py.Group):
                result['groups'].append(name)

        f.visititems(visitor)

    result['total_size_str'] = format_size(result['total_size'])
    return result


def print_report(info: dict[str, Any], verbose: bool = False):
    """Print formatted inspection report."""
    print("=" * 80)
    print("HDF5 File Inspection Report")
    print("=" * 80)
    print(f"File: {info['path']}")
    print(f"Total raw size: {info['total_size_str']}")
    print(f"Groups: {len(info['groups'])}")
    print(f"Datasets: {len(info['datasets'])}")
    print()

    # Sort datasets by size
    sorted_datasets = sorted(
        info['datasets'].items(),
        key=lambda x: x[1]['raw_size'],
        reverse=True
    )

    print("-" * 80)
    print(f"{'Dataset':<25} {'Shape':<25} {'dtype':<10} {'Chunks':<25} {'Size':<12}")
    print("-" * 80)

    for name, dinfo in sorted_datasets:
        shape_str = str(dinfo['shape'])
        chunks_str = str(dinfo['chunks']) if dinfo['chunks'] else 'contiguous'
        print(f"{name:<25} {shape_str:<25} {dinfo['dtype']:<10} {chunks_str:<25} {dinfo['raw_size_str']:<12}")

    print()

    # Detailed analysis for large datasets
    print("=" * 80)
    print("Detailed Analysis (datasets > 1MB)")
    print("=" * 80)

    for name, dinfo in sorted_datasets:
        if dinfo['raw_size'] < 1024 * 1024:
            continue

        print(f"\n--- {name} ---")
        print(f"  Shape: {dinfo['shape']}")
        print(f"  dtype: {dinfo['dtype']} ({dinfo['dtype_size']} bytes)")
        print(f"  Size: {dinfo['raw_size_str']}")
        print(f"  Memory order: {dinfo['memory_order']}")

        if dinfo['chunks']:
            print(f"  Chunks: {dinfo['chunks']} ({dinfo['chunk_bytes_str']})")
            if 'row_read_efficiency' in dinfo:
                print(f"  Row read efficiency: {dinfo['row_read_efficiency']*100:.1f}%")
                if dinfo['row_read_efficiency'] < 0.5:
                    print("    WARNING: Low efficiency - chunks span multiple rows")
        else:
            print("  Chunks: contiguous (not chunked)")

        if dinfo['compression']:
            ratio = dinfo.get('compression_ratio', 1.0)
            stored = format_size(dinfo.get('stored_size', dinfo['raw_size']))
            print(f"  Compression: {dinfo['compression']} (ratio: {ratio:.2f}x, stored: {stored})")
        else:
            print("  Compression: none")

        # Recommendations
        if len(dinfo['shape']) == 2:
            n_rows, n_cols = dinfo['shape']
            print("\n  Recommendations:")

            # For spectra data (many rows, ~150 cols)
            if n_cols < 1000 and n_rows > 10000:
                optimal_row_chunk = min(10000, n_rows)
                print(f"    - Optimal chunks for row-major read: ({optimal_row_chunk}, {n_cols})")
                print(f"    - This allows reading {optimal_row_chunk} spectra per I/O")

            # For column-major data
            if n_rows < 1000 and n_cols > 10000:
                optimal_col_chunk = min(100000, n_cols)
                print(f"    - Optimal chunks for column-major read: ({n_rows}, {optimal_col_chunk})")

            # dtype recommendation
            if dinfo['dtype'] == 'float64':
                print("    - Consider float32 to halve I/O (if precision allows)")


def benchmark_read(h5_path: str, dataset: str = None, n_iterations: int = 3) -> dict[str, float]:
    """Benchmark read performance for different access patterns."""
    results = {}

    with h5py.File(h5_path, 'r') as f:
        # Auto-detect dataset if not specified
        if dataset is None:
            # Find largest 2D dataset
            largest = None
            largest_size = 0
            for name, obj in f.items():
                if isinstance(obj, h5py.Dataset) and len(obj.shape) == 2:
                    size = np.prod(obj.shape) * obj.dtype.itemsize
                    if size > largest_size:
                        largest = name
                        largest_size = size

            # Also check common names
            for name in ['specdata', 'Y', 'data', 'spectra']:
                if name in f:
                    dataset = name
                    break

            if dataset is None and largest:
                dataset = largest

            if dataset is None:
                print("No suitable dataset found for benchmark")
                return results

        dset = f[dataset]
        shape = dset.shape
        dtype = dset.dtype
        raw_size = np.prod(shape) * dtype.itemsize

        print(f"\nBenchmarking: {dataset}")
        print(f"Shape: {shape}, dtype: {dtype}")
        print(f"Size: {format_size(raw_size)}")
        print()

        # Test 1: Full read with [:]
        times = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            data = dset[:]
            times.append(time.perf_counter() - t0)
            del data

        avg_time = np.mean(times)
        throughput = raw_size / avg_time
        results['full_read'] = {
            'time': avg_time,
            'throughput': throughput,
            'throughput_str': format_rate(throughput),
        }
        print(f"Full read [:]: {avg_time:.3f}s ({format_rate(throughput)})")

        # Test 2: Full read with astype
        if dtype != np.float32:
            times = []
            for _ in range(n_iterations):
                t0 = time.perf_counter()
                data = dset[:].astype(np.float32)
                times.append(time.perf_counter() - t0)
                del data

            avg_time = np.mean(times)
            throughput = raw_size / avg_time
            results['full_read_astype'] = {
                'time': avg_time,
                'throughput': throughput,
                'throughput_str': format_rate(throughput),
            }
            print(f"Full read + astype: {avg_time:.3f}s ({format_rate(throughput)})")

        # Test 3: Batch read (row-major)
        if len(shape) == 2:
            n_rows = shape[0]
            batch_sizes = [10000, 100000, 500000, 1000000]

            for batch_size in batch_sizes:
                if batch_size > n_rows:
                    continue

                times = []
                for _ in range(n_iterations):
                    t0 = time.perf_counter()
                    for start in range(0, n_rows, batch_size):
                        end = min(start + batch_size, n_rows)
                        batch = dset[start:end, :]
                    times.append(time.perf_counter() - t0)

                avg_time = np.mean(times)
                throughput = raw_size / avg_time
                key = f'batch_{batch_size}'
                results[key] = {
                    'time': avg_time,
                    'throughput': throughput,
                    'throughput_str': format_rate(throughput),
                }
                print(f"Batch read (batch={batch_size:,}): {avg_time:.3f}s ({format_rate(throughput)})")

        # Test 4: read_direct (if dtype matches)
        if len(shape) == 2:
            times = []
            out = np.empty(shape, dtype=np.float32)

            for _ in range(n_iterations):
                t0 = time.perf_counter()
                if dtype == np.float32:
                    dset.read_direct(out)
                else:
                    # read_direct requires matching dtype
                    out = dset[:].astype(np.float32)
                times.append(time.perf_counter() - t0)

            avg_time = np.mean(times)
            throughput = raw_size / avg_time
            results['read_direct'] = {
                'time': avg_time,
                'throughput': throughput,
                'throughput_str': format_rate(throughput),
            }
            print(f"read_direct: {avg_time:.3f}s ({format_rate(throughput)})")

        print()

    return results


def print_io_issues(info: dict[str, Any]):
    """Print identified I/O issues and recommendations."""
    print("\n" + "=" * 80)
    print("I/O Optimization Issues and Recommendations")
    print("=" * 80)

    issues = []

    for name, dinfo in info['datasets'].items():
        if dinfo['raw_size'] < 1024 * 1024:
            continue

        # Issue: float64 when float32 would suffice
        if dinfo['dtype'] == 'float64':
            issues.append({
                'severity': 'HIGH',
                'dataset': name,
                'issue': 'Using float64 (doubles I/O bandwidth)',
                'fix': 'Convert to float32 if 7 decimal digits precision is sufficient',
            })

        # Issue: Poor chunk alignment
        if dinfo['chunks'] and len(dinfo['shape']) == 2:
            n_rows, n_cols = dinfo['shape']
            chunk_rows, chunk_cols = dinfo['chunks']

            # For row-major access, chunk should span full columns
            if chunk_cols < n_cols:
                issues.append({
                    'severity': 'HIGH',
                    'dataset': name,
                    'issue': f'Chunk width ({chunk_cols}) < row width ({n_cols})',
                    'fix': f'Set chunks=({min(10000, n_rows)}, {n_cols}) for row-major access',
                })

            # Chunks too small
            chunk_bytes = np.prod(dinfo['chunks']) * dinfo['dtype_size']
            if chunk_bytes < 256 * 1024:  # < 256KB
                issues.append({
                    'severity': 'MEDIUM',
                    'dataset': name,
                    'issue': f'Chunks too small ({format_size(chunk_bytes)})',
                    'fix': 'Increase chunk size to 1-16MB for better I/O efficiency',
                })

        # Issue: No chunking on large dataset
        if dinfo['chunks'] is None and dinfo['raw_size'] > 100 * 1024 * 1024:
            issues.append({
                'severity': 'MEDIUM',
                'dataset': name,
                'issue': 'Large contiguous dataset (no chunking)',
                'fix': 'Enable chunking for partial I/O and compression support',
            })

        # Issue: Compression on frequently read data
        if dinfo['compression'] and dinfo['raw_size'] > 100 * 1024 * 1024:
            issues.append({
                'severity': 'LOW',
                'dataset': name,
                'issue': f'Compression enabled ({dinfo["compression"]})',
                'fix': 'Consider disabling compression for frequently read data (faster I/O)',
            })

    if not issues:
        print("\nNo significant I/O issues detected.")
        return

    # Group by severity
    for severity in ['HIGH', 'MEDIUM', 'LOW']:
        sev_issues = [i for i in issues if i['severity'] == severity]
        if sev_issues:
            print(f"\n[{severity}]")
            for issue in sev_issues:
                print(f"  Dataset: {issue['dataset']}")
                print(f"    Issue: {issue['issue']}")
                print(f"    Fix: {issue['fix']}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect HDF5 files for I/O optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m voigtfit.tools.inspect_h5 data.h5
    python -m voigtfit.tools.inspect_h5 data.h5 --benchmark
    python -m voigtfit.tools.inspect_h5 data.h5 --dataset specdata --benchmark
        """
    )
    parser.add_argument('h5_file', help='Path to HDF5 file')
    parser.add_argument('--dataset', '-d', help='Specific dataset to analyze')
    parser.add_argument('--benchmark', '-b', action='store_true',
                       help='Run read performance benchmark')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    if not Path(args.h5_file).exists():
        print(f"Error: File not found: {args.h5_file}")
        return 1

    # Inspect file
    info = inspect_file(args.h5_file)
    print_report(info, args.verbose)
    print_io_issues(info)

    # Optional benchmark
    if args.benchmark:
        benchmark_read(args.h5_file, args.dataset)

    return 0


if __name__ == '__main__':
    exit(main())
