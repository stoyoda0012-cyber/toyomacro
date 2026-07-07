"""
Memory Management Utilities
============================

Automatic memory detection and chunk size optimization
for running on 16GB–128GB environments.

Key functions:
    get_available_memory_gb() — current free physical memory
    optimal_chunk_size()      — auto-determine chunk size from memory budget
    MemoryProfiler            — context manager for peak RSS tracking
"""

import os
import resource
import time
from dataclasses import dataclass, field


def get_available_memory_gb() -> float:
    """Return available physical memory in GB.

    Uses psutil if installed, otherwise falls back to os.sysconf (macOS/Linux).
    """
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        pass
    # Fallback for macOS / Linux
    try:
        pages = os.sysconf('SC_AVPHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
        return pages * page_size / 1e9
    except (ValueError, OSError, AttributeError):
        # Last resort: assume 8 GB available (conservative)
        return 8.0


def get_total_memory_gb() -> float:
    """Return total physical memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        pass
    try:
        pages = os.sysconf('SC_PHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
        return pages * page_size / 1e9
    except (ValueError, OSError, AttributeError):
        return 16.0


def get_rss_gb() -> float:
    """Return current RSS (Resident Set Size) in GB."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # macOS: ru_maxrss is in bytes; Linux: in kilobytes
    import sys
    if sys.platform == 'darwin':
        return ru.ru_maxrss / 1e9
    else:
        return ru.ru_maxrss * 1e3 / 1e9


def optimal_chunk_size(
    n_dict: int,
    dtype_bytes: int = 4,
    memory_fraction: float = 0.3,
    available_gb: float | None = None,
    min_chunk: int = 100_000,
    max_chunk: int = 10_000_000,
) -> int:
    """Determine optimal chunk size from available memory.

    The dominant memory consumer during dictionary lookup is the scores
    matrix: (N × n_dict × dtype_bytes).  We size N so scores fits within
    ``available_memory × memory_fraction``.

    Args:
        n_dict: Number of dictionary entries (e.g. 310 for Dict2D, 2079 for Dict3D)
        dtype_bytes: Bytes per element (4 for float32, 2 for float16)
        memory_fraction: Fraction of available memory to use (default 0.3)
        available_gb: Override available memory (for testing / simulation)
        min_chunk: Minimum chunk size (default 100K)
        max_chunk: Maximum chunk size (default 10M)

    Returns:
        Optimal chunk size (number of spectra per chunk)

    Examples:
        16GB available → ~500K (n_dict=310, fp32)
        128GB available → 10M (capped at max_chunk)
    """
    if available_gb is None:
        available_gb = get_available_memory_gb()
    budget_bytes = available_gb * 1e9 * memory_fraction
    n = int(budget_bytes / (n_dict * dtype_bytes))
    return max(min_chunk, min(max_chunk, n))


def optimal_dict3d_cache_size(
    entry_size_gb: float = 0.75,
    memory_fraction: float = 0.2,
    available_gb: float | None = None,
) -> int:
    """Auto-determine Dict3D LRU cache max_size from available memory.

    Args:
        entry_size_gb: Approximate size of one Dict3D cache entry (default 0.75 GB)
        memory_fraction: Fraction of available memory for cache (default 0.2)
        available_gb: Override available memory (for testing)

    Returns:
        max_size: Number of Dict3D entries to cache (minimum 1)

    Examples:
        16GB  → max_size=1
        32GB  → max_size=2
        64GB  → max_size=3
        128GB → max_size=4
    """
    if available_gb is None:
        available_gb = get_available_memory_gb()
    budget_gb = available_gb * memory_fraction
    return max(1, int(budget_gb / entry_size_gb))


@dataclass
class MemorySnapshot:
    """A single memory measurement point."""
    label: str
    timestamp: float
    rss_gb: float


@dataclass
class MemoryReport:
    """Summary of memory usage during profiled section."""
    baseline_gb: float
    peak_gb: float
    delta_gb: float
    snapshots: list = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Peak: {self.peak_gb:.2f} GB, "
            f"Baseline: {self.baseline_gb:.2f} GB, "
            f"Delta: {self.delta_gb:.2f} GB"
        ]
        if self.snapshots:
            lines.append("Snapshots:")
            for s in self.snapshots:
                lines.append(f"  [{s.label}] RSS={s.rss_gb:.2f} GB "
                             f"(+{s.timestamp:.3f}s)")
        return "\n".join(lines)


class MemoryProfiler:
    """Context manager for tracking memory usage during processing.

    Usage::

        with MemoryProfiler() as prof:
            result = solve_batch(...)
            prof.snapshot("after solve")
        print(prof.report())
        # → Peak: 8.3 GB, Baseline: 2.1 GB, Delta: 6.2 GB
    """

    def __init__(self):
        self._start_rss: float = 0.0
        self._peak_rss: float = 0.0
        self._start_time: float = 0.0
        self._snapshots: list = []
        self._report: MemoryReport | None = None

    def __enter__(self):
        self._start_rss = self._current_rss()
        self._peak_rss = self._start_rss
        self._start_time = time.monotonic()
        self._snapshots = []
        return self

    def __exit__(self, *args):
        final_rss = self._current_rss()
        self._peak_rss = max(self._peak_rss, final_rss)
        self._report = MemoryReport(
            baseline_gb=self._start_rss,
            peak_gb=self._peak_rss,
            delta_gb=self._peak_rss - self._start_rss,
            snapshots=list(self._snapshots),
        )

    def snapshot(self, label: str = "") -> float:
        """Record current RSS with a label. Returns current RSS in GB."""
        rss = self._current_rss()
        self._peak_rss = max(self._peak_rss, rss)
        elapsed = time.monotonic() - self._start_time
        self._snapshots.append(MemorySnapshot(
            label=label, timestamp=elapsed, rss_gb=rss
        ))
        return rss

    def report(self) -> MemoryReport:
        """Return the final memory report (available after exiting context)."""
        if self._report is None:
            # Still inside context — generate interim report
            rss = self._current_rss()
            self._peak_rss = max(self._peak_rss, rss)
            return MemoryReport(
                baseline_gb=self._start_rss,
                peak_gb=self._peak_rss,
                delta_gb=self._peak_rss - self._start_rss,
                snapshots=list(self._snapshots),
            )
        return self._report

    @staticmethod
    def _current_rss() -> float:
        """Get current process RSS in GB."""
        ru = resource.getrusage(resource.RUSAGE_SELF)
        import sys
        if sys.platform == 'darwin':
            return ru.ru_maxrss / 1e9
        else:
            return ru.ru_maxrss * 1e3 / 1e9


# ---------------------------------------------------------------------------
# Memory mode configuration
# ---------------------------------------------------------------------------

def select_memory_mode(available_gb: float | None = None) -> str:
    """Auto-select memory mode based on available memory.

    Returns:
        "high" if >= 32 GB available
        "low"  if < 32 GB available
    """
    if available_gb is None:
        available_gb = get_available_memory_gb()
    return "high" if available_gb >= 32.0 else "low"
