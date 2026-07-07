#!/usr/bin/env python3
"""
HDF5 Repack Tool for I/O Optimization
=====================================

Converts HDF5 files to optimal layout for VoigtFit:
- float32 dtype (if float64)
- Row-major chunking for spectra reads
- Optional compression (lzf for speed, gzip for size)
- Metadata preservation

Usage:
    python -m voigtfit.tools.repack_h5 input.h5 output.h5
    python -m voigtfit.tools.repack_h5 input.h5 output.h5 --compress lzf
    python -m voigtfit.tools.repack_h5 input.h5 --inplace  # Dangerous!
Date: 2026-01-23
"""

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def format_size(nbytes: int) -> str:
    """Format byte size as human-readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if nbytes < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} PB"


def get_optimal_chunks(
    shape: tuple[int, ...],
    dtype: np.dtype,
    target_mb: float = 4.0,
    access_pattern: str = 'row',
) -> tuple[int, ...]:
    """
    Calculate optimal chunk shape for given access pattern.

    Args:
        shape: Dataset shape
        dtype: Dataset dtype
        target_mb: Target chunk size in MB
        access_pattern: 'row' or 'column'

    Returns:
        Optimal chunk shape
    """
    if len(shape) == 1:
        return (min(shape[0], 100000),)

    target_bytes = target_mb * 1024 * 1024
    bytes_per_elem = np.dtype(dtype).itemsize

    if len(shape) == 2:
        n_rows, n_cols = shape

        if access_pattern == 'row':
            # Full columns, multiple rows
            row_bytes = n_cols * bytes_per_elem
            chunk_rows = int(target_bytes / row_bytes)
            chunk_rows = max(1, min(chunk_rows, n_rows, 100_000))
            return (chunk_rows, n_cols)

        else:  # column
            col_bytes = n_rows * bytes_per_elem
            chunk_cols = int(target_bytes / col_bytes)
            chunk_cols = max(1, min(chunk_cols, n_cols))
            return (n_rows, chunk_cols)

    elif len(shape) == 3:
        # For fitpara: (n_spectra, n_comp, 9)
        # Access pattern is typically all params for a spectrum
        n_spectra, n_comp, n_params = shape
        row_bytes = n_comp * n_params * bytes_per_elem
        chunk_rows = int(target_bytes / row_bytes)
        chunk_rows = max(1, min(chunk_rows, n_spectra, 100_000))
        return (chunk_rows, n_comp, n_params)

    return shape  # Default: single chunk


def analyze_file(path: str) -> dict[str, Any]:
    """Analyze HDF5 file and return optimization report."""
    report = {
        'path': path,
        'file_size': os.path.getsize(path),
        'datasets': {},
        'issues': [],
        'recommendations': [],
    }

    with h5py.File(path, 'r') as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                raw_bytes = np.prod(obj.shape) * obj.dtype.itemsize
                stored_bytes = obj.id.get_storage_size()

                info = {
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'chunks': obj.chunks,
                    'compression': obj.compression,
                    'raw_bytes': raw_bytes,
                    'stored_bytes': stored_bytes,
                }
                report['datasets'][name] = info

                # Check for issues
                if obj.dtype == np.float64 and raw_bytes > 1024 * 1024:
                    report['issues'].append(f"{name}: float64 (should be float32)")
                    report['recommendations'].append(
                        f"Convert {name} to float32 (save {format_size(raw_bytes // 2)})"
                    )

                if len(obj.shape) == 2 and raw_bytes > 10 * 1024 * 1024:
                    n_rows, n_cols = obj.shape
                    if obj.chunks is None:
                        report['issues'].append(f"{name}: not chunked")
                        report['recommendations'].append(
                            f"Add chunks=({min(10000, n_rows)}, {n_cols}) to {name}"
                        )
                    elif obj.chunks[1] < n_cols:
                        report['issues'].append(
                            f"{name}: chunks don't span full columns "
                            f"({obj.chunks[1]} < {n_cols})"
                        )
                        report['recommendations'].append(
                            f"Change {name} chunks to ({obj.chunks[0]}, {n_cols})"
                        )

        f.visititems(visitor)

    return report


def repack_dataset(
    src_dset: h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    target_dtype: np.dtype | None = None,
    target_chunks: tuple | None = None,
    compression: str | None = None,
    compression_opts: int | None = None,
    batch_size: int = 100_000,
    verbose: bool = True,
) -> None:
    """
    Repack a single dataset with optimal settings.

    Args:
        src_dset: Source dataset
        dst_group: Destination group
        name: Dataset name
        target_dtype: Target dtype (None = auto)
        target_chunks: Target chunks (None = auto)
        compression: Compression algorithm
        compression_opts: Compression options
        batch_size: Batch size for copying
        verbose: Print progress
    """
    shape = src_dset.shape
    src_dtype = src_dset.dtype

    # Determine target dtype
    if target_dtype is None:
        if src_dtype == np.float64:
            target_dtype = np.float32
        else:
            target_dtype = src_dtype

    # Determine target chunks
    if target_chunks is None:
        target_chunks = get_optimal_chunks(
            shape, target_dtype, target_mb=4.0, access_pattern='row'
        )

    # Create output dataset
    dst_dset = dst_group.create_dataset(
        name,
        shape=shape,
        dtype=target_dtype,
        chunks=target_chunks,
        compression=compression,
        compression_opts=compression_opts,
    )

    # Copy attributes
    for key, val in src_dset.attrs.items():
        dst_dset.attrs[key] = val

    # Copy data in batches
    if len(shape) == 0:
        # Scalar
        dst_dset[()] = src_dset[()]
    elif len(shape) == 1:
        dst_dset[:] = src_dset[:].astype(target_dtype)
    elif len(shape) == 2:
        n_rows = shape[0]
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            dst_dset[start:end] = src_dset[start:end].astype(target_dtype)
            if verbose and (start // batch_size) % 10 == 0:
                progress = 100 * end / n_rows
                print(f"    {name}: {progress:.0f}%", end='\r')
        if verbose:
            print(f"    {name}: done" + " " * 20)
    elif len(shape) == 3:
        n_rows = shape[0]
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            dst_dset[start:end] = src_dset[start:end].astype(target_dtype)
            if verbose and (start // batch_size) % 10 == 0:
                progress = 100 * end / n_rows
                print(f"    {name}: {progress:.0f}%", end='\r')
        if verbose:
            print(f"    {name}: done" + " " * 20)
    else:
        # Higher dimensions: just copy
        dst_dset[:] = src_dset[:].astype(target_dtype)


def repack_file(
    input_path: str,
    output_path: str,
    force_float32: bool = True,
    compression: str | None = None,
    compression_opts: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Repack HDF5 file with optimal settings.

    Args:
        input_path: Input HDF5 file
        output_path: Output HDF5 file
        force_float32: Convert float64 to float32
        compression: 'gzip', 'lzf', or None
        compression_opts: Compression level (for gzip: 1-9)
        verbose: Print progress

    Returns:
        Report dict with before/after stats
    """
    if verbose:
        print("=" * 70)
        print("HDF5 Repack")
        print("=" * 70)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Compression: {compression or 'none'}")
        print()

    input_size = os.path.getsize(input_path)
    t_start = time.perf_counter()

    with h5py.File(input_path, 'r') as fin:
        with h5py.File(output_path, 'w') as fout:
            # Copy file-level attributes
            for key, val in fin.attrs.items():
                fout.attrs[key] = val

            # Process all items
            def copy_item(name, obj):
                if verbose:
                    print(f"  Processing: {name}")

                if isinstance(obj, h5py.Dataset):
                    # Get parent group
                    parts = name.rsplit('/', 1)
                    if len(parts) == 2:
                        parent_name, dset_name = parts
                        if parent_name not in fout:
                            fout.create_group(parent_name)
                        parent = fout[parent_name]
                    else:
                        dset_name = name
                        parent = fout

                    # Determine target dtype
                    target_dtype = None
                    if force_float32 and obj.dtype == np.float64:
                        target_dtype = np.float32

                    repack_dataset(
                        obj, parent, dset_name,
                        target_dtype=target_dtype,
                        compression=compression,
                        compression_opts=compression_opts,
                        verbose=verbose,
                    )

                elif isinstance(obj, h5py.Group):
                    if name not in fout:
                        fout.create_group(name)
                    # Copy group attributes
                    for key, val in obj.attrs.items():
                        fout[name].attrs[key] = val

            fin.visititems(copy_item)

    elapsed = time.perf_counter() - t_start
    output_size = os.path.getsize(output_path)

    report = {
        'input_path': input_path,
        'output_path': output_path,
        'input_size': input_size,
        'output_size': output_size,
        'compression_ratio': input_size / output_size if output_size > 0 else 1.0,
        'time': elapsed,
        'throughput': input_size / elapsed,
    }

    if verbose:
        print()
        print("-" * 70)
        print(f"Input size:  {format_size(input_size)}")
        print(f"Output size: {format_size(output_size)}")
        print(f"Ratio: {report['compression_ratio']:.2f}x")
        print(f"Time: {elapsed:.1f}s")
        print(f"Throughput: {format_size(int(report['throughput']))}/s")

    return report


def verify_repack(original_path: str, repacked_path: str, tolerance: float = 1e-5) -> bool:
    """
    Verify repacked file matches original (within float32 precision).

    Args:
        original_path: Original file
        repacked_path: Repacked file
        tolerance: Maximum allowed relative difference

    Returns:
        True if files match
    """
    print("\nVerifying repack...")

    with h5py.File(original_path, 'r') as f_orig:
        with h5py.File(repacked_path, 'r') as f_new:
            def check_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    if name not in f_new:
                        print(f"  MISSING: {name}")
                        return False

                    new_dset = f_new[name]

                    # Check shape
                    if obj.shape != new_dset.shape:
                        print(f"  SHAPE MISMATCH: {name} ({obj.shape} vs {new_dset.shape})")
                        return False

                    # Check data (sample for large datasets)
                    if np.prod(obj.shape) > 1_000_000:
                        # Sample check
                        idx = tuple(slice(0, min(1000, s)) for s in obj.shape)
                        orig_data = obj[idx].astype(np.float32)
                        new_data = new_dset[idx].astype(np.float32)
                    else:
                        orig_data = obj[:].astype(np.float32)
                        new_data = new_dset[:].astype(np.float32)

                    # Check values
                    if orig_data.size > 0:
                        max_diff = np.max(np.abs(orig_data - new_data))
                        max_val = np.max(np.abs(orig_data))
                        rel_diff = max_diff / max_val if max_val > 0 else max_diff

                        if rel_diff > tolerance:
                            print(f"  VALUE MISMATCH: {name} (rel_diff={rel_diff:.2e})")
                            return False

                    print(f"  OK: {name}")
                return True

            # Check all datasets
            all_ok = True
            def visitor(name, obj):
                nonlocal all_ok
                if not check_dataset(name, obj):
                    all_ok = False

            f_orig.visititems(visitor)

    if all_ok:
        print("\nVerification PASSED")
    else:
        print("\nVerification FAILED")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description='Repack HDF5 files for optimal I/O',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic repack (float64 -> float32, optimal chunks)
    python -m voigtfit.tools.repack_h5 input.h5 output.h5

    # With lzf compression (fast)
    python -m voigtfit.tools.repack_h5 input.h5 output.h5 --compress lzf

    # With gzip compression (smaller files)
    python -m voigtfit.tools.repack_h5 input.h5 output.h5 --compress gzip --level 6

    # Analyze file only
    python -m voigtfit.tools.repack_h5 input.h5 --analyze

    # In-place repack (creates backup)
    python -m voigtfit.tools.repack_h5 input.h5 --inplace
        """
    )
    parser.add_argument('input', help='Input HDF5 file')
    parser.add_argument('output', nargs='?', help='Output HDF5 file')
    parser.add_argument('--compress', '-c', choices=['gzip', 'lzf'],
                       help='Compression algorithm')
    parser.add_argument('--level', '-l', type=int, default=4,
                       help='Compression level (gzip: 1-9, default 4)')
    parser.add_argument('--analyze', '-a', action='store_true',
                       help='Analyze only, no repack')
    parser.add_argument('--inplace', action='store_true',
                       help='Repack in-place (creates backup)')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification')
    parser.add_argument('--keep-float64', action='store_true',
                       help='Keep float64 dtype')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Analyze mode
    if args.analyze:
        report = analyze_file(args.input)
        print("=" * 70)
        print(f"Analysis: {args.input}")
        print("=" * 70)
        print(f"File size: {format_size(report['file_size'])}")
        print(f"Datasets: {len(report['datasets'])}")

        print("\nDatasets:")
        for name, info in report['datasets'].items():
            print(f"  {name}: {info['shape']} {info['dtype']}")
            print(f"    chunks={info['chunks']}, compression={info['compression']}")
            print(f"    raw={format_size(info['raw_bytes'])}, stored={format_size(info['stored_bytes'])}")

        if report['issues']:
            print("\nIssues:")
            for issue in report['issues']:
                print(f"  - {issue}")

        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")

        return 0

    # Determine output path
    if args.inplace:
        # Create temp file, then replace
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            output_path = tmp.name
        backup_path = args.input + '.backup'
    elif args.output:
        output_path = args.output
    else:
        print("Error: Output path required (use --inplace for in-place)")
        return 1

    # Compression options
    compression_opts = args.level if args.compress == 'gzip' else None

    # Repack
    report = repack_file(
        args.input,
        output_path,
        force_float32=not args.keep_float64,
        compression=args.compress,
        compression_opts=compression_opts,
        verbose=True,
    )

    # Verify
    if not args.no_verify:
        if not verify_repack(args.input, output_path):
            print("Error: Verification failed, keeping original")
            if args.inplace:
                os.unlink(output_path)
            return 1

    # In-place: replace original
    if args.inplace:
        print(f"\nCreating backup: {backup_path}")
        shutil.move(args.input, backup_path)
        shutil.move(output_path, args.input)
        print("Replaced original with repacked file")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())
