"""I/O utility functions."""

from __future__ import annotations

import platform
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import h5py


def get_available_memory_gb() -> float:
    """
    Get available physical memory in GB.

    Works on macOS, Linux, and Windows. Returns a conservative estimate (8GB)
    if detection fails.

    Returns:
        Available memory in gigabytes
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        try:
            result = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, check=True
            )
            lines = result.stdout.split("\n")
            page_size = 4096
            free_pages = 0
            inactive_pages = 0

            for line in lines:
                if "page size of" in line:
                    page_size = int(line.split()[-2])
                elif "Pages free" in line:
                    free_pages = int(line.split()[-1].rstrip("."))
                elif "Pages inactive" in line:
                    inactive_pages = int(line.split()[-1].rstrip("."))

            return (free_pages + inactive_pages) * page_size / (1024**3)
        except Exception:
            return 8.0

    elif system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / (1024**2)
            return 8.0
        except Exception:
            return 8.0

    elif system == "Windows":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulonglong),
                    ("ullAvailPhys", c_ulonglong),
                    ("ullTotalPageFile", c_ulonglong),
                    ("ullAvailPageFile", c_ulonglong),
                    ("ullTotalVirtual", c_ulonglong),
                    ("ullAvailVirtual", c_ulonglong),
                    ("ullAvailExtendedVirtual", c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys / (1024**3)
        except Exception:
            return 8.0

    return 8.0


def calculate_batch_size(
    n_spectra: int,
    n_energy: int,
    n_components: int = 2,
    available_memory_gb: float | None = None,
    memory_fraction: float = 0.5,
    min_batch: int = 10_000,
    max_batch: int = 1_000_000,
) -> int:
    """
    Calculate optimal batch size based on available memory.

    Args:
        n_spectra: Total number of spectra
        n_energy: Number of energy points per spectrum
        n_components: Number of peak components (for fitpara memory)
        available_memory_gb: Available memory in GB (None for auto-detect)
        memory_fraction: Fraction of memory to use
        min_batch: Minimum batch size
        max_batch: Maximum batch size

    Returns:
        Optimal batch size
    """
    if available_memory_gb is None:
        available_memory_gb = get_available_memory_gb()

    # Memory per spectrum: spectra (float32) + fitpara + overhead
    bytes_per_spectrum = (n_energy * 4 + n_components * 9 * 4) * 2

    available_bytes = available_memory_gb * memory_fraction * (1024**3)
    batch_size = int(available_bytes / bytes_per_spectrum)

    return max(min_batch, min(batch_size, max_batch, n_spectra))


def open_h5_optimized(
    filepath: str,
    mode: str = "r",
    swmr: bool = False,
    rdcc_nbytes: int = 64 * 1024 * 1024,
    rdcc_nslots: int = 10007,
    rdcc_w0: float = 0.75,
) -> h5py.File:
    """
    Open HDF5 file with optimized cache settings.

    Args:
        filepath: Path to HDF5 file
        mode: File mode ('r', 'r+', 'w', 'w-', 'a')
        swmr: Enable Single Writer Multiple Reader mode
        rdcc_nbytes: Raw data chunk cache size in bytes (default 64MB)
        rdcc_nslots: Number of chunk cache slots
        rdcc_w0: Preemption policy (0-1, higher = more aggressive eviction)

    Returns:
        h5py.File object with optimized settings
    """
    import h5py

    return h5py.File(
        filepath,
        mode,
        swmr=swmr,
        rdcc_nbytes=rdcc_nbytes,
        rdcc_nslots=rdcc_nslots,
        rdcc_w0=rdcc_w0,
    )


def estimate_file_size_mb(
    n_spectra: int,
    n_energy: int,
    max_components: int = 6,
    compression: bool = True,
) -> float:
    """
    Estimate HDF5 file size in MB.

    Args:
        n_spectra: Number of spectra
        n_energy: Number of energy points
        max_components: Maximum peak components
        compression: Whether compression is enabled

    Returns:
        Estimated file size in megabytes
    """
    # specdata: (n_energy, 1 + n_spectra) * 4 bytes
    specdata_bytes = n_energy * (1 + n_spectra) * 4

    # fitpara: (9, max_components, n_spectra) * 4 bytes
    fitpara_bytes = 9 * max_components * n_spectra * 4

    # otherpara: (n_spectra, 10) * 4 bytes
    otherpara_bytes = n_spectra * 10 * 4

    # xytdata: (n_spectra, 3) * 4 bytes
    xytdata_bytes = n_spectra * 3 * 4

    total_bytes = specdata_bytes + fitpara_bytes + otherpara_bytes + xytdata_bytes

    # Compression typically achieves 2-3x reduction for XPS data
    compression_factor = 0.4 if compression else 1.0

    return total_bytes * compression_factor / (1024 * 1024)
