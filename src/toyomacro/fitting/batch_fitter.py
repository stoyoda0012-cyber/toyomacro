"""
Batch fitting for XPS spectra with parallelization and inheritance.

Features:
- Initial value inheritance from previous fits for faster convergence
- Parallel fitting using ProcessPoolExecutor
- Prefetching for smooth browsing experience
- Progress callbacks for UI integration

Usage:
    from toyomacro.fitting import BatchFitter
    from toyomacro.io import HDF5Cache

    cache = HDF5Cache()
    fitter = BatchFitter(cache, n_workers=8)

    # Single fit with inheritance
    result = fitter.fit_spectrum('/path/to/file.h5', index=100)

    # Batch fit with prefetch
    results = fitter.fit_batch('/path/to/file.h5', range(100, 200))

    # Async prefetch for browsing
    fitter.prefetch_async('/path/to/file.h5', indices=[101, 102, 103])
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from toyomacro.fitting.autofitter import AutoFitResult
    from toyomacro.io.hdf5_cache import HDF5Cache


@dataclass
class BatchFitConfig:
    """Configuration for batch fitting."""

    # Parallelization - 2D grid configuration
    n_workers: int = 8  # Total workers (will be split for 2D mode)
    use_processes: bool = False  # Use threads by default (GIL released in numpy)

    # 2D parallel mode: split workers between index and element axes
    parallel_2d: bool = True
    n_index_workers: int = 4   # Workers for index/prefetch axis
    n_element_workers: int = 4  # Workers for element axis (per index worker)
    # Total parallelism = n_index_workers × n_element_workers = 16 fits

    # Inheritance
    use_inheritance: bool = True
    max_inheritance_distance: int = 10  # Max index gap for inheritance

    # Prefetch
    prefetch_ahead: int = 5
    prefetch_behind: int = 2

    # Fitting
    refit_on_failure: bool = True
    min_r_squared: float = 0.9

    # Multi-element parallel fitting (1D mode fallback)
    element_parallel: bool = True  # Fit multiple elements in parallel


@dataclass
class BatchFitResult:
    """Result from a batch fit operation."""

    results: dict[int, AutoFitResult]  # index -> result
    total_time_ms: float
    spectra_per_second: float
    cache_hits: int
    cache_misses: int
    inheritance_used: int


@dataclass
class FitCacheEntry:
    """Cached fit result with metadata."""

    result: AutoFitResult
    timestamp: float
    inherited_from: int | None = None


class BatchFitter:
    """
    High-performance batch fitter for XPS spectra.

    Uses initial value inheritance and parallelization to achieve
    fast fitting rates (400-800 spectra/sec on 8 cores).
    """

    def __init__(
        self,
        cache: HDF5Cache,
        config: BatchFitConfig | None = None,
    ):
        """
        Initialize BatchFitter.

        Args:
            cache: HDF5Cache instance for spectrum access
            config: Batch fitting configuration
        """
        self.cache = cache
        self.config = config or BatchFitConfig()

        # Fit result cache: path -> {index -> FitCacheEntry}
        self._fit_cache: dict[str, dict[int, FitCacheEntry]] = {}

        # Executors for parallel fitting
        self._executor: ThreadPoolExecutor | ProcessPoolExecutor | None = None

        # 2D mode: separate executors for index and element axes
        self._index_executor: ThreadPoolExecutor | None = None
        self._element_executor: ThreadPoolExecutor | None = None

        # Pending futures for async prefetch
        self._pending: dict[str, dict[int, Future]] = {}

    def _get_executor(self) -> ThreadPoolExecutor | ProcessPoolExecutor:
        """Get or create the main executor (1D mode)."""
        if self._executor is None:
            if self.config.use_processes:
                self._executor = ProcessPoolExecutor(
                    max_workers=self.config.n_workers
                )
            else:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.config.n_workers
                )
        return self._executor

    def _get_2d_executors(self) -> tuple[ThreadPoolExecutor, ThreadPoolExecutor]:
        """Get or create the 2D executors (index + element axes)."""
        if self._index_executor is None:
            self._index_executor = ThreadPoolExecutor(
                max_workers=self.config.n_index_workers
            )
        if self._element_executor is None:
            self._element_executor = ThreadPoolExecutor(
                max_workers=self.config.n_element_workers
            )
        return self._index_executor, self._element_executor

    def fit_spectrum(
        self,
        path: str,
        index: int,
        force_refit: bool = False,
        use_template: bool = True,
    ) -> AutoFitResult:
        """
        Fit a single spectrum with optional inheritance.

        Args:
            path: Path to HDF5 file
            index: Spectrum index
            force_refit: If True, ignore cached result
            use_template: If True (default), use template-based fitting
                when a matching template exists. Set to False to force
                peak-detection based fitting.

        Returns:
            AutoFitResult
        """
        # Check cache first
        if not force_refit:
            cached = self._get_cached(path, index)
            if cached is not None:
                return cached.result

        # Get spectrum data
        spec_data = self.cache.get_spectrum(path, index)
        energy = spec_data.energy
        intensity = spec_data.intensity

        # Create Spectrum object
        from toyomacro.core.spectrum import Spectrum

        spectrum = Spectrum(
            energy=np.asarray(energy),
            intensity=np.asarray(intensity),
            element=spec_data.element,
        )

        # Find initial parameters from nearby cached fit
        initial_params = None
        inherited_from = None

        if self.config.use_inheritance:
            initial_params, inherited_from = self._find_inheritance(path, index)

        # Perform fit
        from toyomacro.fitting.autofitter import AutoFitter

        fitter = AutoFitter()

        if initial_params is not None:
            result = fitter.fit_with_initial(spectrum, initial_params)
        else:
            result = fitter.fit(spectrum, use_template=use_template)

        # Cache result
        self._cache_result(path, index, result, inherited_from)

        return result

    def fit_batch(
        self,
        path: str,
        indices: list[int] | range,
        progress_callback: Callable[[int, int], None] | None = None,
        parallel: bool = True,
    ) -> BatchFitResult:
        """
        Fit a batch of spectra with chunk-parallel strategy.

        Strategy:
        - Divide work into chunks (one per worker)
        - Within each chunk: sequential with inheritance
        - Between chunks: parallel execution

        Args:
            path: Path to HDF5 file
            indices: List or range of spectrum indices
            progress_callback: Optional callback(current, total)
            parallel: If True, use parallel chunk processing

        Returns:
            BatchFitResult with all results and statistics
        """
        t0 = time.perf_counter()
        indices = list(indices)
        n_total = len(indices)

        results: dict[int, AutoFitResult] = {}
        cache_hits = 0
        cache_misses = 0
        inheritance_used = 0

        # First pass: collect cached results and prepare work
        work_indices = []
        for idx in indices:
            cached = self._get_cached(path, idx)
            if cached is not None:
                results[idx] = cached.result
                cache_hits += 1
            else:
                work_indices.append(idx)
                cache_misses += 1

        # Sort work indices for better inheritance
        work_indices.sort()

        if not work_indices:
            # All cached
            t1 = time.perf_counter()
            return BatchFitResult(
                results=results,
                total_time_ms=(t1 - t0) * 1000,
                spectra_per_second=n_total / (t1 - t0) if t1 > t0 else 0,
                cache_hits=cache_hits,
                cache_misses=0,
                inheritance_used=0,
            )

        if parallel and len(work_indices) > self.config.n_workers:
            # Chunk-parallel: divide into chunks, parallel between chunks
            chunk_results, inheritance_used = self._fit_chunks_parallel(
                path, work_indices, progress_callback, cache_hits, n_total
            )
            results.update(chunk_results)
        else:
            # Sequential with inheritance
            for i, idx in enumerate(work_indices):
                result = self.fit_spectrum(path, idx)
                results[idx] = result

                cache_entry = self._get_cached(path, idx)
                if cache_entry and cache_entry.inherited_from is not None:
                    inheritance_used += 1

                if progress_callback:
                    progress_callback(cache_hits + i + 1, n_total)

        t1 = time.perf_counter()
        total_time_ms = (t1 - t0) * 1000
        spectra_per_second = n_total / (t1 - t0) if t1 > t0 else 0

        return BatchFitResult(
            results=results,
            total_time_ms=total_time_ms,
            spectra_per_second=spectra_per_second,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            inheritance_used=inheritance_used,
        )

    def _fit_chunks_parallel(
        self,
        path: str,
        work_indices: list[int],
        progress_callback: Callable[[int, int], None] | None,
        cache_hits: int,
        n_total: int,
    ) -> tuple[dict[int, AutoFitResult], int]:
        """
        Fit spectra using chunk-parallel strategy.

        Divides work into chunks, processes chunks in parallel,
        but within each chunk uses sequential fitting with inheritance.

        Returns:
            (results_dict, inheritance_count)
        """
        from concurrent.futures import as_completed

        n_workers = self.config.n_workers
        n_work = len(work_indices)

        # Divide into chunks
        chunk_size = max(1, n_work // n_workers)
        chunks = []
        for i in range(0, n_work, chunk_size):
            chunk = work_indices[i : i + chunk_size]
            if chunk:
                chunks.append(chunk)

        # Process chunks in parallel
        executor = self._get_executor()
        futures = {}

        for chunk in chunks:
            # Each chunk is processed by _fit_chunk_sequential
            future = executor.submit(self._fit_chunk_sequential, path, chunk)
            futures[future] = chunk

        # Collect results
        results: dict[int, AutoFitResult] = {}
        inheritance_used = 0
        completed = 0

        for future in as_completed(futures):
            chunk_results, chunk_inheritance = future.result()
            results.update(chunk_results)
            inheritance_used += chunk_inheritance
            completed += len(futures[future])

            if progress_callback:
                progress_callback(cache_hits + completed, n_total)

        return results, inheritance_used

    def _fit_chunk_sequential(
        self,
        path: str,
        indices: list[int],
    ) -> tuple[dict[int, AutoFitResult], int]:
        """
        Fit a chunk of spectra sequentially with inheritance.

        Used as worker function for parallel chunk processing.

        Returns:
            (results_dict, inheritance_count)
        """
        results: dict[int, AutoFitResult] = {}
        inheritance_count = 0

        for idx in indices:
            result = self.fit_spectrum(path, idx)
            results[idx] = result

            cache_entry = self._get_cached(path, idx)
            if cache_entry and cache_entry.inherited_from is not None:
                inheritance_count += 1

        return results, inheritance_count

    @staticmethod
    def _dedup_elements(
        paths: list[str], cache: HDF5Cache,
    ) -> dict[str, str]:
        """Build path→unique-element-name mapping with de-duplication.

        Delegates to :func:`~toyomacro.io.hdf5_cache.dedup_element_names`.
        """
        from toyomacro.io.hdf5_cache import dedup_element_names

        names = dedup_element_names(paths, cache)
        return dict(zip(paths, names))

    def fit_multi_element(
        self,
        paths: list[str],
        index: int,
        use_template: bool = True,
    ) -> dict[str, AutoFitResult]:
        """
        Fit multiple elements at the same index in parallel.

        Args:
            paths: List of HDF5 file paths (one per element)
            index: Spectrum index
            use_template: If True (default), use template-based fitting.

        Returns:
            Dict mapping unique element name to AutoFitResult
        """
        from concurrent.futures import as_completed

        path_to_elem = self._dedup_elements(paths, self.cache)

        if not self.config.element_parallel or len(paths) <= 1:
            # Sequential
            results = {}
            for path in paths:
                result = self.fit_spectrum(path, index, use_template=use_template)
                results[path_to_elem[path]] = result
            return results

        # Parallel element fitting
        executor = self._get_executor()
        futures = {}

        for path in paths:
            future = executor.submit(self.fit_spectrum, path, index,
                                     use_template=use_template)
            futures[future] = path

        results = {}
        for future in as_completed(futures):
            path = futures[future]
            result = future.result()
            results[path_to_elem[path]] = result

        return results

    def fit_multi_element_batch(
        self,
        paths: list[str],
        indices: list[int] | range,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[int, dict[str, AutoFitResult]]:
        """
        Fit multiple elements across a range of indices.

        2D parallel strategy:
        - Outer: chunk-parallel across indices (with inheritance)
        - Inner: element-parallel at each index

        Args:
            paths: List of HDF5 file paths
            indices: List or range of spectrum indices
            progress_callback: Optional callback(current, total)

        Returns:
            Dict mapping index to {element: AutoFitResult}
        """
        t0 = time.perf_counter()
        indices = list(indices)
        n_total = len(indices)

        all_results: dict[int, dict[str, AutoFitResult]] = {}

        for i, idx in enumerate(indices):
            # Fit all elements at this index (parallel within)
            elem_results = self.fit_multi_element(paths, idx)
            all_results[idx] = elem_results

            if progress_callback:
                progress_callback(i + 1, n_total)

        t1 = time.perf_counter()
        print(f"Multi-element batch: {n_total} indices × {len(paths)} elements "
              f"in {(t1-t0)*1000:.0f}ms "
              f"({n_total * len(paths) / (t1-t0):.0f} fits/sec)")

        return all_results

    def fit_multi_element_batch_2d(
        self,
        paths: list[str],
        indices: list[int] | range,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[int, dict[str, AutoFitResult]]:
        """
        Fit multiple elements across indices using true 2D parallelism.

        2D Grid Strategy:
        - Index axis: n_index_workers parallel index processing
        - Element axis: n_element_workers parallel element fitting per index
        - Total parallelism: n_index_workers × n_element_workers

        Example with 4×4 = 16 cores:
        ┌─────────────────────────────────────────────────────┐
        │  Index Worker 0    Index Worker 1    ...            │
        │  ┌──────────────┐  ┌──────────────┐                 │
        │  │ Idx 0        │  │ Idx 4        │                 │
        │  │ ┌──┬──┬──┬──┐│  │ ┌──┬──┬──┬──┐│                 │
        │  │ │Si│O │C │N ││  │ │Si│O │C │N ││  (4 elem each)  │
        │  │ └──┴──┴──┴──┘│  │ └──┴──┴──┴──┘│                 │
        │  │ Idx 1        │  │ Idx 5        │                 │
        │  │ ...          │  │ ...          │                 │
        │  └──────────────┘  └──────────────┘                 │
        └─────────────────────────────────────────────────────┘

        Args:
            paths: List of HDF5 file paths (one per element)
            indices: List or range of spectrum indices
            progress_callback: Optional callback(current, total)

        Returns:
            Dict mapping index to {element: AutoFitResult}
        """
        from concurrent.futures import as_completed

        t0 = time.perf_counter()
        indices = list(indices)
        n_total = len(indices)
        n_elements = len(paths)

        if not self.config.parallel_2d or n_total < 2 or n_elements < 2:
            # Fallback to 1D mode
            return self.fit_multi_element_batch(paths, indices, progress_callback)

        # Get 2D executors
        index_executor, element_executor = self._get_2d_executors()

        # Divide indices into chunks for index workers
        n_index_workers = self.config.n_index_workers
        chunk_size = max(1, n_total // n_index_workers)

        chunks = []
        for i in range(0, n_total, chunk_size):
            chunk = indices[i : i + chunk_size]
            if chunk:
                chunks.append(chunk)

        # Submit index chunks to index executor
        # Each chunk will process its indices sequentially (with inheritance)
        # but fit elements in parallel using element_executor
        futures = {}
        for chunk in chunks:
            future = index_executor.submit(
                self._process_index_chunk_2d,
                paths,
                chunk,
                element_executor,
            )
            futures[future] = chunk

        # Collect results
        all_results: dict[int, dict[str, AutoFitResult]] = {}
        completed = 0

        for future in as_completed(futures):
            chunk_results = future.result()
            all_results.update(chunk_results)
            completed += len(futures[future])

            if progress_callback:
                progress_callback(completed, n_total)

        t1 = time.perf_counter()
        total_fits = n_total * n_elements
        fits_per_sec = total_fits / (t1 - t0) if t1 > t0 else 0

        print(f"2D parallel: {n_total} indices × {n_elements} elements = {total_fits} fits "
              f"in {(t1-t0)*1000:.0f}ms ({fits_per_sec:.0f} fits/sec)")
        print(f"  Grid: {n_index_workers} index workers × {self.config.n_element_workers} element workers")

        return all_results

    def _process_index_chunk_2d(
        self,
        paths: list[str],
        indices: list[int],
        element_executor: ThreadPoolExecutor,
    ) -> dict[int, dict[str, AutoFitResult]]:
        """
        Process a chunk of indices with element-parallel fitting.

        Sequential over indices (for inheritance), parallel over elements.
        """
        from concurrent.futures import as_completed

        results: dict[int, dict[str, AutoFitResult]] = {}

        for idx in indices:
            # Fit all elements at this index in parallel
            elem_futures = {}
            for path in paths:
                future = element_executor.submit(self.fit_spectrum, path, idx)
                elem_futures[future] = path

            # Collect element results
            elem_results: dict[str, AutoFitResult] = {}
            for future in as_completed(elem_futures):
                path = elem_futures[future]
                result = future.result()
                meta = self.cache.get_metadata(path)
                elem = meta.element if meta else path
                elem_results[elem] = result

            results[idx] = elem_results

        return results

    def prefetch_async(
        self,
        path: str,
        indices: list[int],
    ) -> None:
        """
        Start async prefetch of fit results.

        Non-blocking: results will be available from cache when needed.

        Args:
            path: Path to HDF5 file
            indices: List of indices to prefetch
        """
        executor = self._get_executor()

        if path not in self._pending:
            self._pending[path] = {}

        for idx in indices:
            # Skip if already cached or pending
            if self._get_cached(path, idx) is not None:
                continue
            if idx in self._pending.get(path, {}):
                continue

            # Submit fit task
            future = executor.submit(self.fit_spectrum, path, idx)
            self._pending[path][idx] = future

    def prefetch_for_browsing(
        self,
        path: str,
        current_index: int,
        direction: int = 1,  # 1 for forward, -1 for backward
        step: int = 1,
    ) -> None:
        """
        Prefetch fit results for browsing.

        Args:
            path: Path to HDF5 file
            current_index: Current spectrum index
            direction: Browsing direction (1=forward, -1=backward)
            step: Step size for browsing
        """
        meta = self.cache.get_metadata(path)
        if meta is None:
            return

        max_idx = meta.n_spectra - 1

        # Calculate prefetch indices
        prefetch_indices = []

        if direction >= 0:
            # Forward prefetch
            for i in range(1, self.config.prefetch_ahead + 1):
                idx = current_index + i * step
                if 0 <= idx <= max_idx:
                    prefetch_indices.append(idx)
        else:
            # Backward prefetch
            for i in range(1, self.config.prefetch_ahead + 1):
                idx = current_index - i * step
                if 0 <= idx <= max_idx:
                    prefetch_indices.append(idx)

        self.prefetch_async(path, prefetch_indices)

    def _find_inheritance(
        self,
        path: str,
        index: int,
    ) -> tuple[list[dict] | None, int | None]:
        """
        Find initial parameters from nearby cached fit.

        Returns:
            (initial_params, inherited_from_index) or (None, None)
        """
        if path not in self._fit_cache:
            return None, None

        cache = self._fit_cache[path]
        max_dist = self.config.max_inheritance_distance

        # Search nearby indices (prefer closer ones)
        for dist in range(1, max_dist + 1):
            for offset in [dist, -dist]:
                nearby_idx = index + offset
                if nearby_idx in cache:
                    entry = cache[nearby_idx]
                    result = entry.result

                    if result.success and result.fitting_result.components:
                        # Extract parameters from components
                        params = []
                        for comp in result.fitting_result.components:
                            params.append({
                                "center": comp.center,
                                "area": comp.area,
                                "fwhm_g": comp.fwhm_g,
                                "fwhm_l": comp.fwhm_l,
                            })
                        return params, nearby_idx

        return None, None

    def _get_cached(self, path: str, index: int) -> FitCacheEntry | None:
        """Get cached fit result if available."""
        if path in self._fit_cache:
            return self._fit_cache[path].get(index)

        # Check pending futures
        if path in self._pending and index in self._pending[path]:
            future = self._pending[path][index]
            if future.done():
                # Move from pending to cache
                del self._pending[path][index]
                return self._get_cached(path, index)

        return None

    def _cache_result(
        self,
        path: str,
        index: int,
        result: AutoFitResult,
        inherited_from: int | None,
    ) -> None:
        """Cache a fit result."""
        if path not in self._fit_cache:
            self._fit_cache[path] = {}

        self._fit_cache[path][index] = FitCacheEntry(
            result=result,
            timestamp=time.time(),
            inherited_from=inherited_from,
        )

    def get_cached_result(self, path: str, index: int) -> AutoFitResult | None:
        """Get cached result if available (public interface)."""
        entry = self._get_cached(path, index)
        return entry.result if entry else None

    def clear_cache(self, path: str | None = None) -> None:
        """
        Clear fit cache.

        Args:
            path: If provided, only clear cache for this file
        """
        if path is None:
            self._fit_cache.clear()
        elif path in self._fit_cache:
            del self._fit_cache[path]

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total_cached = sum(len(c) for c in self._fit_cache.values())
        total_pending = sum(len(p) for p in self._pending.values())

        return {
            "files_cached": len(self._fit_cache),
            "total_cached_results": total_cached,
            "pending_fits": total_pending,
        }

    def shutdown(self) -> None:
        """Shutdown all executors and clear caches."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

        # Shutdown 2D executors
        if self._index_executor is not None:
            self._index_executor.shutdown(wait=False)
            self._index_executor = None

        if self._element_executor is not None:
            self._element_executor.shutdown(wait=False)
            self._element_executor = None

        self._fit_cache.clear()
        self._pending.clear()

    def __enter__(self) -> BatchFitter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    def __del__(self) -> None:
        self.shutdown()


def fit_spectrum_worker(
    energy: NDArray,
    intensity: NDArray,
    element: str,
    initial_params: list[dict] | None,
) -> AutoFitResult:
    """
    Worker function for parallel fitting.

    Designed to be picklable for ProcessPoolExecutor.
    """
    from toyomacro.core.spectrum import Spectrum
    from toyomacro.fitting.autofitter import AutoFitter

    spectrum = Spectrum(
        energy=energy,
        intensity=intensity,
        element=element,
    )

    fitter = AutoFitter()

    if initial_params is not None:
        return fitter.fit_with_initial(spectrum, initial_params)
    else:
        return fitter.fit(spectrum)
