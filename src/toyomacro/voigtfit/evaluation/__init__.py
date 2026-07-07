"""Evaluation module: error statistics and benchmark result storage."""

from .benchmark_result import BenchmarkResult
from .error_stats import ErrorStats, compute_error_stats

__all__ = ['ErrorStats', 'compute_error_stats', 'BenchmarkResult']
