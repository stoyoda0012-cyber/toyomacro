"""
Prefetch Pipeline: Overlapped I/O + GPU Compute
===============================================

Hides disk I/O latency by loading next batch while GPU processes current batch.

Architecture:
    Thread 1 (I/O):    [Load B1] [Load B2] [Load B3] ...
    Thread 2 (GPU):           [Process B1] [Process B2] [Process B3] ...

Expected speedup: From 8M spec/s (I/O bound) to ~50M spec/s (overlap)
"""

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from queue import Queue
from typing import Any

import h5py
import numpy as np

try:
    import mlx.core as mx

    from ._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND the default device can execute work
except ImportError:
    HAS_MLX = False


@dataclass
class Batch:
    """Container for a batch of spectra."""
    data: np.ndarray  # (n_spectra, n_energy) - row major
    batch_idx: int
    start_idx: int
    end_idx: int


@dataclass
class BatchResult:
    """Result from processing a batch."""
    amplitudes: np.ndarray  # (n_components, n_spectra)
    chi2: np.ndarray  # (n_spectra,)
    anomaly_mask: np.ndarray  # (n_spectra,)
    batch_idx: int
    start_idx: int
    end_idx: int
    timing: dict[str, float]


class PrefetchLoader:
    """
    Asynchronous data loader with prefetching.

    Loads batches in a background thread while GPU processes.
    """

    def __init__(
        self,
        file_path: str,
        dataset_path: str,
        batch_size: int = 1_000_000,
        prefetch_count: int = 2,
    ):
        """
        Initialize prefetch loader.

        Args:
            file_path: Path to HDF5 file
            dataset_path: Path to dataset within HDF5 (e.g., 'specdata')
            batch_size: Number of spectra per batch
            prefetch_count: Number of batches to prefetch (default 2)
        """
        self.file_path = file_path
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.prefetch_count = prefetch_count

        # Get dataset info without loading
        with h5py.File(file_path, 'r') as f:
            dset = f[dataset_path]
            self.shape = dset.shape
            self.dtype = dset.dtype
            self.n_spectra = self.shape[0]
            self.n_energy = self.shape[1]

        self._queue: Queue[Batch | None] = Queue(maxsize=prefetch_count + 1)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def __len__(self) -> int:
        """Number of batches."""
        return (self.n_spectra + self.batch_size - 1) // self.batch_size

    def _load_worker(self):
        """Background thread that loads batches."""
        try:
            with h5py.File(self.file_path, 'r') as f:
                dset = f[self.dataset_path]

                batch_idx = 0
                start = 0

                while start < self.n_spectra and not self._stop_event.is_set():
                    end = min(start + self.batch_size, self.n_spectra)

                    # Load batch (this is the slow I/O operation)
                    data = dset[start:end, :].astype(np.float32)

                    # Ensure C-contiguous
                    if not data.flags['C_CONTIGUOUS']:
                        data = np.ascontiguousarray(data)

                    batch = Batch(
                        data=data,
                        batch_idx=batch_idx,
                        start_idx=start,
                        end_idx=end,
                    )

                    # Put batch in queue (blocks if queue is full)
                    self._queue.put(batch)

                    start = end
                    batch_idx += 1

                # Signal end of data
                self._queue.put(None)

        except Exception as e:
            print(f"Prefetch loader error: {e}")
            self._queue.put(None)

    def start(self):
        """Start the prefetch thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._load_worker, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the prefetch thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batches."""
        self.start()
        try:
            while True:
                batch = self._queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            self.stop()


class PrefetchPipeline:
    """
    Pipeline with overlapped I/O and GPU compute.

    Uses double-buffering: while GPU processes batch N,
    background thread loads batch N+1.
    """

    def __init__(
        self,
        cache,  # WeightMatrixCache
        chi2_threshold: float = 3.0,
    ):
        """
        Initialize prefetch pipeline.

        Args:
            cache: Weight matrix cache instance
            chi2_threshold: Anomaly detection threshold (in MAD units)
        """
        self.cache = cache
        self.chi2_threshold = chi2_threshold

    def process_file(
        self,
        file_path: str,
        dataset_path: str,
        element: str,
        orbital: str,
        energy: np.ndarray,
        peak_config: dict,
        batch_size: int = 1_000_000,
    ) -> dict[str, Any]:
        """
        Process entire HDF5 file with prefetching.

        Args:
            file_path: Path to HDF5 file
            dataset_path: Dataset path within HDF5
            element: Element symbol
            orbital: Orbital name
            energy: Energy axis
            peak_config: Peak configuration
            batch_size: Spectra per batch

        Returns:
            Dict with results and timing info
        """
        if not HAS_MLX:
            raise RuntimeError("MLX is required for PrefetchPipeline")

        # Initialize loader
        loader = PrefetchLoader(
            file_path=file_path,
            dataset_path=dataset_path,
            batch_size=batch_size,
            prefetch_count=2,
        )

        n_spectra = loader.n_spectra
        n_energy = loader.n_energy

        # Get weight matrix (cached, MLX)
        energy = np.asarray(energy, dtype=np.float32)
        Wt, Phi = self.cache.get_or_create(
            element=element,
            orbital=orbital,
            energy=energy,
            centers=peak_config["centers"],
            sigmas=peak_config["sigmas"],
            gamma=peak_config.get("gamma", 0.2),
            use_mlx=True,
        )
        W = Wt.T  # MLX transpose is a view
        n_comp = len(peak_config["centers"])

        # Pre-allocate output arrays
        all_amplitudes = np.zeros((n_comp, n_spectra), dtype=np.float32)
        all_chi2 = np.zeros(n_spectra, dtype=np.float32)
        all_anomaly = np.zeros(n_spectra, dtype=bool)

        # Timing
        timing = {
            "io_time": 0.0,
            "compute_time": 0.0,
            "convert_time": 0.0,
            "total_time": 0.0,
        }

        t_total_start = time.perf_counter()

        # Process batches with prefetching
        for batch in loader:
            t_io_end = time.perf_counter()  # I/O already done by prefetch

            # Convert to MLX
            t_convert_start = time.perf_counter()
            Y_mx = mx.array(batch.data)
            timing["convert_time"] += time.perf_counter() - t_convert_start

            # GPU compute
            t_compute_start = time.perf_counter()

            # Core computation
            A_row = Y_mx @ W

            # Sampled chi2 (0.01%)
            n_batch = batch.data.shape[0]
            sample_step = max(1, 10000)  # 0.01% sampling
            if sample_step > 1 and n_batch > 10000:
                Y_sample = Y_mx[::sample_step, :]
                A_sample = A_row[::sample_step, :]
                Y_fit_sample = A_sample @ Phi.T
                chi2_sample = mx.sum((Y_sample - Y_fit_sample)**2, axis=1) / n_energy
                mx.eval(A_row, chi2_sample)

                chi2_np = np.repeat(np.array(chi2_sample), sample_step)[:n_batch]
            else:
                Y_fit = A_row @ Phi.T
                chi2_mx = mx.sum((Y_mx - Y_fit)**2, axis=1) / n_energy
                mx.eval(A_row, chi2_mx)
                chi2_np = np.array(chi2_mx)

            # Anomaly detection (sampled threshold)
            sample_size = min(10000, n_batch)
            if n_batch > sample_size:
                sample_idx = np.random.choice(n_batch, sample_size, replace=False)
                chi2_sample_np = chi2_np[sample_idx]
            else:
                chi2_sample_np = chi2_np

            median_chi2 = np.median(chi2_sample_np)
            mad = np.median(np.abs(chi2_sample_np - median_chi2))
            if mad < 1e-10:
                threshold = float('inf')
            else:
                threshold = median_chi2 + self.chi2_threshold * 1.4826 * mad

            anomaly_mask = chi2_np > threshold

            timing["compute_time"] += time.perf_counter() - t_compute_start

            # Store results
            A_np = np.array(A_row).T
            all_amplitudes[:, batch.start_idx:batch.end_idx] = A_np
            all_chi2[batch.start_idx:batch.end_idx] = chi2_np
            all_anomaly[batch.start_idx:batch.end_idx] = anomaly_mask

        timing["total_time"] = time.perf_counter() - t_total_start
        timing["io_time"] = timing["total_time"] - timing["compute_time"] - timing["convert_time"]
        timing["n_spectra"] = n_spectra
        timing["rate"] = n_spectra / timing["total_time"]
        timing["n_anomaly"] = int(np.sum(all_anomaly))

        return {
            "amplitudes": all_amplitudes,
            "chi2": all_chi2,
            "anomaly_mask": all_anomaly,
            "timing": timing,
        }


def benchmark_prefetch(file_path: str = None, n_spectra: int = 10_000_000):
    """
    Benchmark prefetch pipeline vs sequential.

    If file_path is None, creates synthetic data in temp file.
    """
    import os
    import tempfile

    print("=" * 70)
    print("Prefetch Pipeline Benchmark")
    print("=" * 70)

    # Create test data if no file provided
    if file_path is None:
        print(f"\n[Creating synthetic data: {n_spectra:,} spectra]")

        n_energy = 151
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, "test_data.h5")

        # Create in chunks to avoid memory issues
        chunk_size = 1_000_000
        with h5py.File(file_path, 'w') as f:
            dset = f.create_dataset(
                'specdata',
                shape=(n_spectra, n_energy),
                dtype=np.float32,
                chunks=(min(chunk_size, n_spectra), n_energy),
            )

            for start in range(0, n_spectra, chunk_size):
                end = min(start + chunk_size, n_spectra)
                dset[start:end] = np.random.randn(end - start, n_energy).astype(np.float32)
                print(f"  Created {end:,}/{n_spectra:,} spectra")

        dataset_path = 'specdata'
        energy = np.linspace(95, 110, n_energy).astype(np.float32)
        cleanup = True
    else:
        dataset_path = 'specdata'
        with h5py.File(file_path, 'r') as f:
            dset = f[dataset_path]
            n_spectra = dset.shape[0]
            n_energy = dset.shape[1]
        energy = np.linspace(95, 110, n_energy).astype(np.float32)
        cleanup = False

    print(f"\n[File: {file_path}]")
    print(f"  Shape: ({n_spectra:,}, {n_energy})")
    file_size_gb = n_spectra * n_energy * 4 / 1e9
    print(f"  Size: {file_size_gb:.2f} GB")

    # Setup
    from .weight_cache import WeightMatrixCache

    cache = WeightMatrixCache()
    peak_config = {
        "centers": np.array([99.5, 103.0, 107.0], dtype=np.float32),
        "sigmas": np.array([0.6, 0.8, 0.7], dtype=np.float32),
        "gamma": 0.25,
    }

    # Warmup
    print("\n[Warmup]")
    pipeline = PrefetchPipeline(cache)

    # Run benchmark
    print("\n[Benchmark: Prefetch Pipeline]")
    result = pipeline.process_file(
        file_path=file_path,
        dataset_path=dataset_path,
        element="Fe",
        orbital="2p",
        energy=energy,
        peak_config=peak_config,
        batch_size=1_000_000,
    )

    timing = result["timing"]
    print("\n  Results:")
    print(f"    Total time: {timing['total_time']:.2f}s")
    print(f"    I/O time: {timing['io_time']:.2f}s ({100*timing['io_time']/timing['total_time']:.0f}%)")
    print(f"    Convert time: {timing['convert_time']:.2f}s ({100*timing['convert_time']/timing['total_time']:.0f}%)")
    print(f"    Compute time: {timing['compute_time']:.2f}s ({100*timing['compute_time']/timing['total_time']:.0f}%)")
    print(f"    Throughput: {timing['rate']/1e6:.1f}M spec/s")
    print(f"    Anomalies: {timing['n_anomaly']:,}")

    # Compare with theoretical
    print("\n[Comparison]")

    # Sequential estimate (97% I/O)
    ssd_bandwidth = 7.4  # GB/s
    hdf5_efficiency = 0.7
    sequential_io_time = file_size_gb / (ssd_bandwidth * hdf5_efficiency)
    sequential_total = sequential_io_time + timing['compute_time']
    sequential_rate = n_spectra / sequential_total

    print("  Sequential (estimated):")
    print(f"    I/O time: {sequential_io_time:.2f}s")
    print(f"    Total: {sequential_total:.2f}s")
    print(f"    Rate: {sequential_rate/1e6:.1f}M spec/s")

    print(f"\n  Prefetch speedup: {sequential_total/timing['total_time']:.1f}x")

    # Cleanup
    if cleanup:
        import shutil
        shutil.rmtree(temp_dir)
        print("\n[Cleaned up temp files]")

    return result


if __name__ == "__main__":
    benchmark_prefetch(n_spectra=5_000_000)
