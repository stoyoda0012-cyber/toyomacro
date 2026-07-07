"""
VoigtFit Tools Package
======================

Diagnostic and optimization utilities for HDF5 I/O.

Tools:
    inspect_h5 - Analyze HDF5 file structure and identify I/O issues
    repack_h5 - Convert HDF5 files to optimal layout

Usage:
    python -m voigtfit.tools.inspect_h5 data.h5
    python -m voigtfit.tools.repack_h5 input.h5 output.h5
"""

from .inspect_h5 import benchmark_read, inspect_file
from .repack_h5 import analyze_file, repack_file

__all__ = [
    'inspect_file',
    'benchmark_read',
    'repack_file',
    'analyze_file',
]
