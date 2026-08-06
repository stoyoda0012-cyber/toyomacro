"""
Streaming HDF5 writer for large-scale fitting results.

Provides memory-efficient writing of fitting parameters for 10M+ spectra
by writing batches directly to HDF5 without accumulating in memory.

Features:
- Incremental statistics tracking (chi2 mean/std/min/max)
- Crash recovery via incomplete markers
- Three write modes: new_file, inplace, temp_then_move
- Compatible with voigtfit/DepthProfiler format

Usage:
    config = StreamingWriterConfig(
        output_path='output.h5',
        n_spectra=10_000_000,
        n_components=3,
    )
    with StreamingFitparaWriter(config, input_path='input.h5') as writer:
        for batch_idx, (start_idx, batch) in enumerate(batches):
            writer.write_batch(batch_idx, start_idx, fitpara_batch, chi2_batch)
    summary = writer.get_summary()
"""

from __future__ import annotations

import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import h5py
import numpy as np
from numpy.typing import NDArray

from toyomacro.io.schema import ToyomacroSchema

if TYPE_CHECKING:
    pass


@dataclass
class StreamingWriterConfig:
    """Configuration for streaming fitpara writer."""

    output_path: str
    n_spectra: int
    n_components: int
    write_mode: Literal["inplace", "new_file", "temp_then_move", "memory"] = "new_file"
    copy_metadata: bool = True  # Copy specdata, xytdata, misc from input
    compress_on_close: bool = False  # Compress fitpara to int16+LZ4 on close
    compress_fitpara: bool = True  # Write fitpara_compressed (False → raw /fitpara)
    ring_buffer_segments: int = 0  # 0=classic compact, 3=ring buffer + staging
    segment_size: int = 500_000  # Must match pipeline SEGMENT_SIZE


@dataclass
class FitResultSummary:
    """
    Lightweight summary of fitting results for GUI use.

    Full data remains in HDF5 file and can be loaded on demand.

    Attributes:
        n_spectra: Total number of spectra processed
        n_components: Number of peak components
        chi2_mean: Mean chi-squared value
        chi2_std: Standard deviation of chi-squared
        chi2_min: Minimum chi-squared value
        chi2_max: Maximum chi-squared value
        anomaly_count: Number of anomalous spectra detected
        anomaly_rate: Fraction of spectra that are anomalous
        output_path: Path to the output HDF5 file
        anomaly_indices: Optional array of indices for anomalous spectra
    """

    n_spectra: int
    n_components: int
    chi2_mean: float
    chi2_std: float
    chi2_min: float
    chi2_max: float
    anomaly_count: int
    anomaly_rate: float
    output_path: str
    anomaly_indices: NDArray[np.int64] | None = None

    def _load_fitpara(self, f: h5py.File) -> NDArray[np.float32]:
        """Load full fitpara from file, handling both compressed and uncompressed."""
        if "fitpara_compressed" in f:
            from toyomacro.io.compression import decode_fitpara_from_h5

            return decode_fitpara_from_h5(f)
        return f[ToyomacroSchema.PATH_FITPARA][:].astype(np.float32)

    def load_slice(self, start: int, end: int) -> NDArray[np.float32]:
        """
        Load a range of fitpara from file.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            fitpara array of shape (end-start, n_components, 9)
        """
        with h5py.File(self.output_path, "r") as f:
            if "fitpara_compressed" in f:
                return self._load_fitpara(f)[start:end, :, :]
            return f[ToyomacroSchema.PATH_FITPARA][start:end, :, :].astype(np.float32)

    def load_downsampled(self, step: int = 100) -> NDArray[np.float32]:
        """
        Load downsampled fitpara for preview.

        Args:
            step: Sampling step (e.g., 100 = every 100th spectrum)

        Returns:
            fitpara array of shape (n_spectra//step, n_components, 9)
        """
        with h5py.File(self.output_path, "r") as f:
            if "fitpara_compressed" in f:
                return self._load_fitpara(f)[::step, :, :]
            return f[ToyomacroSchema.PATH_FITPARA][::step, :, :].astype(np.float32)

    def load_anomalies_only(self) -> NDArray[np.float32]:
        """
        Load only anomalous spectra.

        Returns:
            fitpara array of shape (anomaly_count, n_components, 9)
        """
        if self.anomaly_indices is None or len(self.anomaly_indices) == 0:
            return np.empty((0, self.n_components, 9), dtype=np.float32)

        sorted_idx = np.sort(self.anomaly_indices)
        with h5py.File(self.output_path, "r") as f:
            if "fitpara_compressed" in f:
                return self._load_fitpara(f)[sorted_idx, :, :]
            return f[ToyomacroSchema.PATH_FITPARA][sorted_idx, :, :].astype(np.float32)


class StreamingFitparaWriter:
    """
    Context manager for streaming fitpara writes.

    Writes batches directly to HDF5 without accumulating in memory.
    Tracks summary statistics incrementally.

    Usage:
        config = StreamingWriterConfig(
            output_path='output.h5',
            n_spectra=10_000_000,
            n_components=3,
        )
        with StreamingFitparaWriter(config, input_path='input.h5') as writer:
            for batch_idx, (start_idx, batch) in enumerate(batches):
                writer.write_batch(batch_idx, start_idx, fitpara_batch, chi2_batch)
        summary = writer.get_summary()

    Attributes:
        config: Writer configuration
        input_path: Source file for metadata copying
    """

    def __init__(
        self,
        config: StreamingWriterConfig,
        input_path: str | Path | None = None,
    ):
        """
        Initialize streaming writer.

        Args:
            config: Writer configuration
            input_path: Source file for metadata copying (optional)
        """
        self.config = config
        self.input_path = str(input_path) if input_path else None
        self._h5_out: h5py.File | None = None
        self._fitpara_ds: h5py.Dataset | None = None
        self._temp_path: str | None = None
        self._memory_buffer: NDArray[np.float32] | None = None  # For memory mode (legacy)

        # Compact memory mode buffers
        # Varying cols (per-spectrum): amp(0), center(1), fwhm_g(2), area(8)
        self._compact_varying: NDArray[np.float32] | None = None  # (n_spectra, n_comp, 4)
        self._compact_chi2: NDArray[np.float32] | None = None  # (n_spectra,) actual chi2
        # Constant cols (same for all spectra): fwhm_l(3), asymm(4), branch(5), so(6), func(7)
        self._compact_constants: NDArray[np.float32] | None = None  # (n_comp, 5) cols 3-7
        # Per-spectrum BG params: (n_spectra, 3) → otherpara rows 4(a), 5(b), 6(c)
        self._compact_bg_params: NDArray[np.float32] | None = None

        # Ring buffer mode (for large datasets)
        self._ring_buffers: list[NDArray[np.float32]] | None = None
        self._ring_chi2_buffers: list[NDArray[np.float32]] | None = None
        self._ring_write_idx: int = 0
        self._ring_flush_futures: list[Future | None] | None = None  # per-slot
        self._ring_flush_executor: ThreadPoolExecutor | None = None
        self._staged_h5: h5py.File | None = None
        self._staged_path: str | None = None

        # Otherpara metadata (rows 0-2: constant for all spectra)
        # Set via set_otherpara_meta() before finalize
        self._otherpara_meta: dict[str, Any] | None = None

        # Incremental statistics
        self._chi2_sum = 0.0
        self._chi2_sum_sq = 0.0
        self._chi2_min = float("inf")
        self._chi2_max = float("-inf")
        self._anomaly_count = 0
        self._processed_count = 0
        self._anomaly_indices: list[int] = []

    def set_otherpara_meta(
        self,
        n_components: int,
        e_min: float,
        e_max: float,
        bg_type: int = 1,
    ):
        """Set constant otherpara metadata for batch writes.

        These values (rows 0-3) are constant for all spectra in an element.
        Rows 4-6 (bg_params) are per-spectrum, set via write_batch(bg_params=...).
        Rows 7-8 (max_height, total_area) are computed from fitpara during finalize.
        Row 9 (chi2) is already handled by write_batch.

        Args:
            n_components: Number of peak components.
            e_min: Minimum energy (eV).
            e_max: Maximum energy (eV).
            bg_type: Background type (1=Shirley, 2=Tougaard, 3=Linear, 4=Constant).
        """
        self._otherpara_meta = {
            "n_components": n_components,
            "e_min": e_min,
            "e_max": e_max,
            "bg_type": bg_type,
        }

    def __enter__(self) -> StreamingFitparaWriter:
        self._setup_output_file()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalize(success=(exc_type is None))

    def _setup_output_file(self):
        """Prepare output file and create fitpara dataset."""
        output_path = self.config.output_path

        if self.config.write_mode == "memory":
            n_spec = self.config.n_spectra
            n_comp = self.config.n_components

            if self.config.ring_buffer_segments > 0:
                # Ring buffer mode: small ring buffers + disk staging
                # Each slot stores 4 varying cols: amp(0), center(1), fwhm_g(2), area(8)
                n_ring = self.config.ring_buffer_segments
                seg_size = self.config.segment_size

                self._ring_buffers = [
                    np.zeros((seg_size, n_comp, 4), dtype=np.float32)
                    for _ in range(n_ring)
                ]
                self._ring_chi2_buffers = [
                    np.zeros(seg_size, dtype=np.float32)
                    for _ in range(n_ring)
                ]

                self._staged_path = output_path + ".staging.h5"
                self._staged_h5 = h5py.File(self._staged_path, "w")
                self._staged_h5.create_dataset(
                    "varying",
                    shape=(n_spec, n_comp, 4),
                    dtype=np.float32,
                    chunks=(min(seg_size, n_spec), n_comp, 4),
                )
                self._staged_h5.create_dataset(
                    "chi2",
                    shape=(n_spec,),
                    dtype=np.float32,
                    chunks=(min(seg_size, n_spec),),
                )
                self._staged_h5.create_dataset(
                    "bg_params",
                    shape=(n_spec, 3),
                    dtype=np.float32,
                    chunks=(min(seg_size, n_spec), 3),
                )

                # Per-slot ring buffers for bg_params
                self._ring_bg_buffers = [
                    np.zeros((seg_size, 3), dtype=np.float32)
                    for _ in range(n_ring)
                ]

                self._ring_flush_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="flush"
                )
                self._ring_flush_futures = [None] * n_ring  # per-slot futures
                self._compact_constants = None  # Set from first batch

                ring_mb = sum(b.nbytes for b in self._ring_buffers) / 1e6
                ring_mb += sum(b.nbytes for b in self._ring_chi2_buffers) / 1e6
                stage_gb = n_spec * n_comp * 4 * 4 / 1e9
                print(
                    f"[Writer] Ring buffer mode: {n_ring} slots × {seg_size:,} rows = "
                    f"{ring_mb:.0f}MB + staging {stage_gb:.1f}GB at {self._staged_path}",
                    flush=True,
                )
                return

            # Compact memory mode: store 4 varying cols per-spectrum + chi2
            # Varying (per-spectrum): amp(0), center(1), fwhm_g(2), area(8)
            # Constant (same for all): fwhm_l(3), asymm(4), branch(5), so(6), func(7)
            # This reduces memory from (n, comp, 9) to (n, comp, 4) + (n,) ≈ 1/2
            self._compact_varying = np.zeros(
                (n_spec, n_comp, 4), dtype=np.float32
            )
            self._compact_chi2 = np.zeros(n_spec, dtype=np.float32)
            self._compact_bg_params = np.zeros((n_spec, 3), dtype=np.float32)
            self._compact_constants = None  # Set from first batch

            vary_mb = self._compact_varying.nbytes / 1e6
            chi2_mb = self._compact_chi2.nbytes / 1e6
            total_mb = vary_mb + chi2_mb
            print(
                f"[Writer] Compact memory mode: {total_mb:.0f}MB buffer "
                f"for {n_spec:,} spectra × {n_comp} comp "
                f"(was {n_spec * n_comp * 9 * 4 / 1e6:.0f}MB)",
                flush=True,
            )
            return

        if self.config.write_mode == "new_file":
            self._h5_out = h5py.File(output_path, "w")
        elif self.config.write_mode == "inplace":
            self._h5_out = h5py.File(output_path, "r+")
            # Delete existing fitpara if shape doesn't match
            if "fitpara" in self._h5_out:
                existing = self._h5_out["fitpara"]
                if (
                    existing.shape[0] != self.config.n_spectra
                    or existing.shape[1] < self.config.n_components
                ):
                    del self._h5_out["fitpara"]
        elif self.config.write_mode == "temp_then_move":
            self._temp_path = output_path + ".tmp"
            self._h5_out = h5py.File(self._temp_path, "w")
        else:
            raise ValueError(f"Unknown write_mode: {self.config.write_mode}")

        # Copy metadata from input if needed
        if (
            self.config.copy_metadata
            and self.input_path
            and self.config.write_mode != "inplace"
        ):
            self._copy_metadata_from_input()

        # Create fitpara dataset if not exists
        if "fitpara" not in self._h5_out:
            # voigtfit format: contiguous (no chunking) for maximum read speed
            self._fitpara_ds = self._h5_out.create_dataset(
                ToyomacroSchema.PATH_FITPARA,
                shape=(
                    self.config.n_spectra,
                    self.config.n_components,
                    ToyomacroSchema.FITPARA_COLS,
                ),
                dtype=ToyomacroSchema.DTYPE_FITPARA,
                fillvalue=np.nan,
            )
        else:
            self._fitpara_ds = self._h5_out["fitpara"]

        # Mark as incomplete (for crash recovery)
        self._h5_out.attrs["_toyomacro_incomplete"] = True
        self._h5_out.attrs["_toyomacro_last_batch"] = -1
        self._h5_out.flush()

    def _copy_metadata_from_input(self):
        """Copy non-fitpara objects and root attributes from input file."""
        if not self.input_path:
            return

        with h5py.File(self.input_path, "r") as f_in:
            # Inherit root attributes (e.g. toyomacro_schema_version).
            # Crash-recovery markers are per-file state, not metadata —
            # they are re-stamped for this output right after this copy.
            for key, val in f_in.attrs.items():
                if key.startswith(("_toyomacro_", "_voigtfit_")):
                    continue
                self._h5_out.attrs[key] = val

            for name, obj in f_in.items():
                if name == "fitpara":
                    continue  # Skip fitpara
                if isinstance(obj, h5py.Dataset):
                    f_in.copy(obj, self._h5_out, name)
                elif isinstance(obj, h5py.Group):
                    f_in.copy(obj, self._h5_out, name)

    def write_batch(
        self,
        batch_idx: int,
        start_idx: int,
        fitpara: NDArray[np.float32],
        chi2: NDArray[np.float32] | None = None,
        anomaly_mask: NDArray[np.bool_] | None = None,
        bg_params: NDArray[np.float32] | None = None,
    ):
        """
        Write one batch of fitting results.

        Args:
            batch_idx: Batch index (for progress tracking)
            start_idx: Starting spectrum index
            fitpara: Fitpara array of shape (n_batch, n_components, 9)
            chi2: Optional chi2 array of shape (n_batch,) for statistics
            anomaly_mask: Optional anomaly mask for this batch
            bg_params: Optional BG params array (n_batch, 3) for otherpara rows 4-6.
                       Col 0=a (higher endpoint), 1=b (lower endpoint), 2=c (direction).
        """
        n_batch = fitpara.shape[0]
        end_idx = start_idx + n_batch

        # Write to dataset or memory buffer
        if self._ring_buffers is not None:
            # Ring buffer mode: write to ring slot, async flush to staging
            n_ring = len(self._ring_buffers)
            slot = self._ring_write_idx % n_ring
            n_comp = self._ring_buffers[slot].shape[1]

            # Wait only if THIS slot's flush is still in progress
            if self._ring_flush_futures[slot] is not None:
                self._ring_flush_futures[slot].result()
                self._ring_flush_futures[slot] = None

            seg_size = self._ring_buffers[slot].shape[0]
            if n_batch > seg_size:
                raise ValueError(
                    f"Batch size {n_batch} exceeds ring buffer slot size {seg_size}"
                )
            # Store 4 varying cols: amp(0), center(1), fwhm_g(2), area(8)
            self._ring_buffers[slot][:n_batch, :, 0] = fitpara[:, :n_comp, 0]
            self._ring_buffers[slot][:n_batch, :, 1] = fitpara[:, :n_comp, 1]
            self._ring_buffers[slot][:n_batch, :, 2] = fitpara[:, :n_comp, 2]
            self._ring_buffers[slot][:n_batch, :, 3] = fitpara[:, :n_comp, 8]
            if chi2 is not None:
                self._ring_chi2_buffers[slot][:n_batch] = chi2
            if bg_params is not None and hasattr(self, '_ring_bg_buffers'):
                self._ring_bg_buffers[slot][:n_batch] = bg_params

            # Capture constants from first batch: cols 3-7
            if self._compact_constants is None:
                self._compact_constants = fitpara[0, :n_comp, 3:8].copy()

            # Async flush to staging file
            self._ring_flush_futures[slot] = self._ring_flush_executor.submit(
                self._flush_ring_slot, slot, start_idx, n_batch
            )
            self._ring_write_idx += 1

            # Update statistics (same as compact path)
            if chi2 is not None:
                self._chi2_sum += float(np.sum(chi2))
                self._chi2_sum_sq += float(np.sum(chi2**2))
                self._chi2_min = min(self._chi2_min, float(np.min(chi2)))
                self._chi2_max = max(self._chi2_max, float(np.max(chi2)))
            self._processed_count += n_batch
            if anomaly_mask is not None:
                batch_anomaly_indices = np.where(anomaly_mask)[0] + start_idx
                self._anomaly_indices.extend(batch_anomaly_indices.tolist())
                self._anomaly_count += int(np.sum(anomaly_mask))
            return

        if self._compact_varying is not None:
            # Compact memory mode: store 4 varying cols per-spectrum
            n_comp = self._compact_varying.shape[1]
            self._compact_varying[start_idx:end_idx, :, 0] = fitpara[:, :n_comp, 0]  # amp
            self._compact_varying[start_idx:end_idx, :, 1] = fitpara[:, :n_comp, 1]  # center
            self._compact_varying[start_idx:end_idx, :, 2] = fitpara[:, :n_comp, 2]  # fwhm_g
            self._compact_varying[start_idx:end_idx, :, 3] = fitpara[:, :n_comp, 8]  # area
            # Store actual chi2 from parameter
            if chi2 is not None:
                self._compact_chi2[start_idx:end_idx] = chi2
            # Store bg_params per spectrum
            if bg_params is not None and self._compact_bg_params is not None:
                self._compact_bg_params[start_idx:end_idx] = bg_params
            # Capture constants from first batch: cols 3-7
            if self._compact_constants is None:
                self._compact_constants = fitpara[0, :n_comp, 3:8].copy()
        elif self._memory_buffer is not None:
            self._memory_buffer[start_idx:end_idx, :, :] = fitpara
        else:
            self._fitpara_ds[start_idx:end_idx, :, :] = fitpara

        # Update incremental statistics
        if chi2 is not None:
            self._chi2_sum += float(np.sum(chi2))
            self._chi2_sum_sq += float(np.sum(chi2**2))
            self._chi2_min = min(self._chi2_min, float(np.min(chi2)))
            self._chi2_max = max(self._chi2_max, float(np.max(chi2)))
        self._processed_count += n_batch

        # Track anomalies
        if anomaly_mask is not None:
            batch_anomaly_indices = np.where(anomaly_mask)[0] + start_idx
            self._anomaly_indices.extend(batch_anomaly_indices.tolist())
            self._anomaly_count += int(np.sum(anomaly_mask))

        # Update progress marker (flush every 10 batches for performance)
        if self._h5_out is not None:
            self._h5_out.attrs["_toyomacro_last_batch"] = batch_idx
            if batch_idx % 10 == 0:
                self._h5_out.flush()

    def get_summary(self) -> FitResultSummary:
        """Generate summary from accumulated statistics."""
        n = self._processed_count
        if n == 0:
            return FitResultSummary(
                n_spectra=0,
                n_components=self.config.n_components,
                chi2_mean=0.0,
                chi2_std=0.0,
                chi2_min=0.0,
                chi2_max=0.0,
                anomaly_count=0,
                anomaly_rate=0.0,
                output_path=self.config.output_path,
            )

        chi2_mean = self._chi2_sum / n if self._chi2_sum > 0 else 0.0
        chi2_var = (self._chi2_sum_sq / n) - (chi2_mean**2) if self._chi2_sum > 0 else 0.0
        chi2_std = np.sqrt(max(0, chi2_var))

        anomaly_indices = (
            np.array(self._anomaly_indices, dtype=np.int64)
            if self._anomaly_indices
            else None
        )

        return FitResultSummary(
            n_spectra=n,
            n_components=self.config.n_components,
            chi2_mean=chi2_mean,
            chi2_std=chi2_std,
            chi2_min=self._chi2_min if self._chi2_min != float("inf") else 0.0,
            chi2_max=self._chi2_max if self._chi2_max != float("-inf") else 0.0,
            anomaly_count=self._anomaly_count,
            anomaly_rate=self._anomaly_count / n if n > 0 else 0.0,
            output_path=self.config.output_path,
            anomaly_indices=anomaly_indices,
        )

    def _flush_ring_slot(self, slot: int, start_idx: int, count: int):
        """Flush one ring buffer slot to staging HDF5 (background thread).

        No .copy() needed: flush executor is max_workers=1, and write_batch()
        waits for THIS slot's flush to complete before overwriting the slot.
        """
        self._staged_h5["varying"][start_idx : start_idx + count, :, :] = (
            self._ring_buffers[slot][:count, :, :]
        )
        self._staged_h5["chi2"][start_idx : start_idx + count] = (
            self._ring_chi2_buffers[slot][:count]
        )
        if hasattr(self, '_ring_bg_buffers') and self._ring_bg_buffers is not None:
            self._staged_h5["bg_params"][start_idx : start_idx + count] = (
                self._ring_bg_buffers[slot][:count]
            )
        self._staged_h5.flush()  # Ensures data hits disk for later staging read

    def _write_from_staging_compressed(self):
        """Read staged varying cols from disk, reconstruct + compress, write final.

        Reads staged data in chunks, reconstructs fitpara, and compresses.
        For simplicity, reconstructs full array then compresses — this is acceptable
        since ring buffer mode is used for large datasets where the file I/O
        dominates over the memory allocation.
        """
        import time as _time

        staged_varying = self._staged_h5["varying"]
        n = self.config.n_spectra
        n_comp = staged_varying.shape[1]

        try:
            from toyomacro.io.compression import _encode_fitpara_to_h5
        except ImportError:
            print("[Writer] lz4 not available, writing uncompressed fitpara", flush=True)
            self._write_from_staging_uncompressed()
            return

        # Reconstruct full fitpara from staging
        fitpara = np.full((n, n_comp, 9), np.nan, dtype=np.float32)
        chunk = 5_000_000
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            vary = staged_varying[s:e, :, :]
            fitpara[s:e, :, 0] = vary[:, :, 0]  # amplitude
            fitpara[s:e, :, 1] = vary[:, :, 1]  # center
            fitpara[s:e, :, 2] = vary[:, :, 2]  # fwhm_g
            fitpara[s:e, :, 8] = vary[:, :, 3]  # area
            if self._compact_constants is not None:
                fitpara[s:e, :, 3:8] = self._compact_constants[np.newaxis, :, :]

        path = self.config.output_path
        t0 = _time.perf_counter()
        h5_mode = "r+" if Path(path).exists() else "w"

        try:
            with h5py.File(path, h5_mode) as f:
                if "fitpara_compressed" in f:
                    del f["fitpara_compressed"]
                if "fitpara" in f:
                    del f["fitpara"]

                stats = _encode_fitpara_to_h5(
                    f, fitpara, dataset_name="fitpara_compressed"
                )

                self._write_otherpara(f, fitpara)

            t_ms = (_time.perf_counter() - t0) * 1000
            ratio = stats["ratio"]
            orig_mb = stats["original_bytes"] / 1e6
            comp_mb = stats["compressed_bytes"] / 1e6
            print(
                f"[Writer] Compressed fitpara (staged): {orig_mb:.0f}MB -> {comp_mb:.0f}MB "
                f"({ratio:.1f}x) in {t_ms:.0f}ms",
                flush=True,
            )
        except Exception as e:
            print(f"[Writer] Staged compress failed, trying uncompressed: {e}", flush=True)
            self._write_from_staging_uncompressed()

    def _finalize(self, success: bool):
        """Close file and handle temp file if needed."""
        # Memory mode: write compressed directly to file (no intermediate /fitpara)
        if self.config.write_mode == "memory":
            # Ring buffer mode
            if self._ring_buffers is not None:
                # Wait for all pending flushes
                if self._ring_flush_futures is not None:
                    for i, fut in enumerate(self._ring_flush_futures):
                        if fut is not None:
                            fut.result()
                            self._ring_flush_futures[i] = None
                if self._ring_flush_executor is not None:
                    self._ring_flush_executor.shutdown(wait=True)
                    self._ring_flush_executor = None

                if success:
                    if self.config.compress_fitpara:
                        self._write_from_staging_compressed()
                    else:
                        self._write_from_staging_uncompressed()

                # Cleanup staging
                if self._staged_h5 is not None:
                    self._staged_h5.close()
                    self._staged_h5 = None
                if self._staged_path and Path(self._staged_path).exists():
                    Path(self._staged_path).unlink()
                    self._staged_path = None

                self._ring_buffers = None
                self._ring_chi2_buffers = None
                if hasattr(self, '_ring_bg_buffers'):
                    self._ring_bg_buffers = None
                self._ring_flush_futures = None
                self._compact_constants = None
                return

            # Classic compact mode
            if success and (self._compact_varying is not None or self._memory_buffer is not None):
                if self.config.compress_fitpara:
                    self._write_memory_compressed()
                else:
                    self._write_memory_uncompressed()
            self._compact_varying = None
            self._compact_chi2 = None
            self._compact_bg_params = None
            self._compact_constants = None
            self._memory_buffer = None
            return

        if self._h5_out is None:
            return

        if success:
            # Remove incomplete markers
            if "_toyomacro_incomplete" in self._h5_out.attrs:
                del self._h5_out.attrs["_toyomacro_incomplete"]
            if "_toyomacro_last_batch" in self._h5_out.attrs:
                del self._h5_out.attrs["_toyomacro_last_batch"]

        self._h5_out.close()
        self._h5_out = None

        # Handle temp file
        if self.config.write_mode == "temp_then_move" and self._temp_path:
            if success:
                shutil.move(self._temp_path, self.config.output_path)
            # On failure, keep temp file for potential recovery

        # Post-write compression pass
        if success and self.config.compress_on_close:
            self._compress_fitpara_inplace()

    def _compress_fitpara_inplace(self):
        """Compress /fitpara to /fitpara_compressed in-place, then delete /fitpara.

        Reads the full uncompressed /fitpara dataset, compresses it using
        int16 quantization + LZ4, writes as /fitpara_compressed group,
        and removes the temporary /fitpara dataset.

        On failure, keeps uncompressed /fitpara as safe fallback.
        """
        import time as _time

        try:
            from toyomacro.io.compression import _encode_fitpara_to_h5
        except ImportError:
            print(
                "[Writer] lz4 not available, keeping uncompressed fitpara",
                flush=True,
            )
            return

        path = self.config.output_path
        t0 = _time.perf_counter()

        try:
            with h5py.File(path, "r+") as f:
                if "fitpara" not in f:
                    return

                # Read the full fitpara array
                fitpara = f["fitpara"][:].astype(np.float32)

                # Compress and write to fitpara_compressed group
                stats = _encode_fitpara_to_h5(
                    f, fitpara, dataset_name="fitpara_compressed"
                )

                # Delete uncompressed fitpara only after successful compression
                del f["fitpara"]

            t_ms = (_time.perf_counter() - t0) * 1000
            ratio = stats["ratio"]
            orig_mb = stats["original_bytes"] / 1e6
            comp_mb = stats["compressed_bytes"] / 1e6
            print(
                f"[Writer] Compressed fitpara: {orig_mb:.0f}MB -> {comp_mb:.0f}MB "
                f"({ratio:.1f}x) in {t_ms:.0f}ms",
                flush=True,
            )
        except Exception as e:
            print(
                f"[Writer] Compression failed, keeping uncompressed fitpara: {e}",
                flush=True,
            )

    def _reconstruct_fitpara_chunk(
        self,
        start: int,
        end: int,
    ) -> NDArray[np.float32]:
        """Reconstruct fitpara (n, n_comp, 9) from compact buffers for a range.

        Only allocates memory for the chunk, not the full array.
        """
        n_comp = self._compact_varying.shape[1]
        chunk_size = end - start
        fitpara = np.full((chunk_size, n_comp, 9), np.nan, dtype=np.float32)

        # Varying cols (per-spectrum): amp(0), center(1), fwhm_g(2), area(8)
        fitpara[:, :, 0] = self._compact_varying[start:end, :, 0]
        fitpara[:, :, 1] = self._compact_varying[start:end, :, 1]
        fitpara[:, :, 2] = self._compact_varying[start:end, :, 2]
        fitpara[:, :, 8] = self._compact_varying[start:end, :, 3]

        # Constant cols 3-7 (broadcast from first batch)
        if self._compact_constants is not None:
            fitpara[:, :, 3:8] = self._compact_constants[np.newaxis, :, :]

        return fitpara

    def _write_memory_compressed(self):
        """Write memory buffer directly as fitpara_compressed (no intermediate /fitpara).

        For compact mode, uses chunked encoder to avoid allocating
        the full (n_spectra, n_comp, 9) array at once.

        Opens the file briefly in r+ mode, replaces fitpara_compressed with
        the new data from memory, and closes. No dead space is created because
        we never write then delete a /fitpara dataset.
        """
        if self._compact_varying is not None:
            self._write_memory_compressed_chunked()
            return

        if self._memory_buffer is None:
            return

        import time as _time

        fitpara = self._memory_buffer

        try:
            from toyomacro.io.compression import _encode_fitpara_to_h5
        except ImportError:
            print("[Writer] lz4 not available, writing uncompressed fitpara", flush=True)
            self._write_memory_uncompressed_from(fitpara)
            return

        path = self.config.output_path
        t0 = _time.perf_counter()
        h5_mode = "r+" if Path(path).exists() else "w"

        try:
            with h5py.File(path, h5_mode) as f:
                if "fitpara_compressed" in f:
                    del f["fitpara_compressed"]
                if "fitpara" in f:
                    del f["fitpara"]

                stats = _encode_fitpara_to_h5(
                    f, fitpara, dataset_name="fitpara_compressed"
                )

                self._write_otherpara(f, fitpara)

            t_ms = (_time.perf_counter() - t0) * 1000
            ratio = stats["ratio"]
            orig_mb = stats["original_bytes"] / 1e6
            comp_mb = stats["compressed_bytes"] / 1e6
            print(
                f"[Writer] Compressed fitpara: {orig_mb:.0f}MB -> {comp_mb:.0f}MB "
                f"({ratio:.1f}x) in {t_ms:.0f}ms",
                flush=True,
            )
        except Exception as e:
            print(f"[Writer] Memory compress failed, trying uncompressed: {e}", flush=True)
            self._write_memory_uncompressed_from(fitpara)

    def _write_memory_compressed_chunked(self):
        """Reconstruct from compact varying buffers + compress and write.

        Reconstructs the full fitpara from compact buffers, then uses the
        standard compression encoder to write fitpara_compressed.
        """
        import time as _time

        fitpara = self._reconstruct_fitpara_chunk(0, self.config.n_spectra)

        try:
            from toyomacro.io.compression import _encode_fitpara_to_h5
        except ImportError:
            print("[Writer] lz4 not available, writing uncompressed fitpara", flush=True)
            self._write_memory_uncompressed_from(fitpara)
            return

        path = self.config.output_path
        t0 = _time.perf_counter()
        h5_mode = "r+" if Path(path).exists() else "w"

        try:
            with h5py.File(path, h5_mode) as f:
                if "fitpara_compressed" in f:
                    del f["fitpara_compressed"]
                if "fitpara" in f:
                    del f["fitpara"]

                stats = _encode_fitpara_to_h5(
                    f, fitpara, dataset_name="fitpara_compressed"
                )

                self._write_otherpara(f, fitpara)

            t_ms = (_time.perf_counter() - t0) * 1000
            ratio = stats["ratio"]
            orig_mb = stats["original_bytes"] / 1e6
            comp_mb = stats["compressed_bytes"] / 1e6
            print(
                f"[Writer] Compressed fitpara (compact): {orig_mb:.0f}MB -> {comp_mb:.0f}MB "
                f"({ratio:.1f}x) in {t_ms:.0f}ms",
                flush=True,
            )
        except Exception as e:
            print(f"[Writer] Compact compress failed, trying uncompressed: {e}", flush=True)
            self._write_memory_uncompressed_from(fitpara)

    def _write_memory_uncompressed(self):
        """Write memory buffer as raw /fitpara (no compression).

        Handles both compact mode (reconstruct from amplitudes + constants)
        and full-buffer mode.
        """
        import time as _time

        if self._compact_varying is not None:
            # Reconstruct full fitpara from compact buffers
            fitpara = self._reconstruct_fitpara_chunk(0, self.config.n_spectra)
        elif self._memory_buffer is not None:
            fitpara = self._memory_buffer
        else:
            return

        path = self.config.output_path
        t0 = _time.perf_counter()
        h5_mode = "r+" if Path(path).exists() else "w"

        try:
            with h5py.File(path, h5_mode) as f:
                if "fitpara_compressed" in f:
                    del f["fitpara_compressed"]
                if "fitpara" in f:
                    del f["fitpara"]
                f.create_dataset("fitpara", data=fitpara)

                self._write_otherpara(f, fitpara)

            t_ms = (_time.perf_counter() - t0) * 1000
            mb = fitpara.nbytes / 1e6
            print(
                f"[Writer] Wrote fitpara: {mb:.0f}MB in {t_ms:.0f}ms",
                flush=True,
            )
        except Exception as e:
            print(f"[Writer] Uncompressed write failed: {e}", flush=True)

    def _write_from_staging_uncompressed(self):
        """Read staged varying cols from disk, reconstruct and write as raw /fitpara."""
        import time as _time

        staged_varying = self._staged_h5["varying"]
        staged_chi2 = self._staged_h5["chi2"]

        path = self.config.output_path
        t0 = _time.perf_counter()
        h5_mode = "r+" if Path(path).exists() else "w"

        try:
            with h5py.File(path, h5_mode) as f:
                if "fitpara_compressed" in f:
                    del f["fitpara_compressed"]
                if "fitpara" in f:
                    del f["fitpara"]

                n = self.config.n_spectra
                n_comp = staged_varying.shape[1]
                fp_ds = f.create_dataset(
                    "fitpara",
                    shape=(n, n_comp, 9),
                    dtype=np.float32,
                )

                # Write in chunks to avoid large memory allocation
                chunk = 5_000_000
                for s in range(0, n, chunk):
                    e = min(s + chunk, n)
                    chunk_size = e - s
                    row = np.full((chunk_size, n_comp, 9), np.nan, dtype=np.float32)
                    vary = staged_varying[s:e, :, :]
                    row[:, :, 0] = vary[:, :, 0]  # amplitude
                    row[:, :, 1] = vary[:, :, 1]  # center
                    row[:, :, 2] = vary[:, :, 2]  # fwhm_g
                    row[:, :, 8] = vary[:, :, 3]  # area
                    if self._compact_constants is not None:
                        row[:, :, 3:8] = self._compact_constants[np.newaxis, :, :]
                    fp_ds[s:e, :, :] = row

                # Write otherpara from staging data
                if "otherpara" in f:
                    otherpara = f["otherpara"]
                    if otherpara.shape[0] >= 10:
                        # Rows 0-3: constants
                        if self._otherpara_meta is not None:
                            otherpara[0, :n] = self._otherpara_meta["n_components"]
                            otherpara[1, :n] = self._otherpara_meta["e_min"]
                            otherpara[2, :n] = self._otherpara_meta["e_max"]
                            otherpara[3, :n] = self._otherpara_meta.get("bg_type", 1)
                        # Rows 4-6: BG params from staging
                        staged_bg = self._staged_h5.get("bg_params")
                        if staged_bg is not None:
                            for s in range(0, n, chunk):
                                e = min(s + chunk, n)
                                bg = staged_bg[s:e, :]
                                otherpara[4, s:e] = bg[:, 0]  # a (higher)
                                otherpara[5, s:e] = bg[:, 1]  # b (lower)
                                otherpara[6, s:e] = bg[:, 2]  # c (direction)
                        # Rows 7-9 from staging varying/chi2
                        for s in range(0, n, chunk):
                            e = min(s + chunk, n)
                            vary = staged_varying[s:e, :, :]
                            # Row 7: max peak height (col 0 of varying)
                            otherpara[7, s:e] = np.nanmax(vary[:, :, 0], axis=1)
                            # Row 8: total area (col 3 of varying = fitpara col 8)
                            otherpara[8, s:e] = np.nansum(vary[:, :, 3], axis=1)
                            # Row 9: chi2
                            otherpara[9, s:e] = staged_chi2[s:e]

            t_ms = (_time.perf_counter() - t0) * 1000
            mb = n * n_comp * 9 * 4 / 1e6
            print(
                f"[Writer] Wrote fitpara (staged): {mb:.0f}MB in {t_ms:.0f}ms",
                flush=True,
            )
        except Exception as e:
            print(f"[Writer] Staged uncompressed write failed: {e}", flush=True)

    def _write_memory_uncompressed_from(self, fitpara: NDArray[np.float32]):
        """Fallback: write fitpara array as uncompressed /fitpara."""
        path = self.config.output_path
        h5_mode = "r+" if Path(path).exists() else "w"
        try:
            with h5py.File(path, h5_mode) as f:
                if "fitpara" in f:
                    del f["fitpara"]
                f.create_dataset("fitpara", data=fitpara)
            print(f"[Writer] Wrote uncompressed fitpara to {Path(path).name}", flush=True)
        except Exception as e:
            print(f"[Writer] Uncompressed write failed: {e}", flush=True)

    def _write_otherpara(self, f: h5py.File, fitpara: NDArray[np.float32] | None = None):
        """Write otherpara rows 0-9 to an open HDF5 file.

        Rows 0-3: constant metadata from set_otherpara_meta()
        Rows 4-6: per-spectrum BG params from compact/staging buffer
        Rows 7-8: computed from fitpara (max_height, total_area) per spectrum
        Row 9: chi2 (from compact buffer)

        Args:
            f: Open HDF5 file in r+ mode
            fitpara: Full fitpara array if available (for rows 7-8). If None,
                     only writes rows 0-3 and 9.
        """
        if "otherpara" not in f:
            return
        otherpara = f["otherpara"]
        if otherpara.shape[0] < 10:
            return
        n_spec = min(self.config.n_spectra, otherpara.shape[1])

        # Rows 0-3: constant metadata
        if self._otherpara_meta is not None:
            otherpara[0, :n_spec] = self._otherpara_meta["n_components"]
            otherpara[1, :n_spec] = self._otherpara_meta["e_min"]
            otherpara[2, :n_spec] = self._otherpara_meta["e_max"]
            otherpara[3, :n_spec] = self._otherpara_meta.get("bg_type", 1)

        # Rows 4-6: per-spectrum BG params (a, b, c)
        if self._compact_bg_params is not None:
            otherpara[4, :n_spec] = self._compact_bg_params[:n_spec, 0]  # a (higher)
            otherpara[5, :n_spec] = self._compact_bg_params[:n_spec, 1]  # b (lower)
            otherpara[6, :n_spec] = self._compact_bg_params[:n_spec, 2]  # c (direction)

        # Rows 7-8: per-spectrum area stats from fitpara
        if fitpara is not None:
            n_comp = fitpara.shape[1]
            # Row 7: max peak height (col 0) across components
            col0 = fitpara[:, :n_comp, 0]  # (n_spec, n_comp)
            otherpara[7, :n_spec] = np.nanmax(col0, axis=1)
            # Row 8: total integrated area (col 8) across components
            col8 = fitpara[:, :n_comp, 8]  # (n_spec, n_comp)
            otherpara[8, :n_spec] = np.nansum(col8, axis=1)

        # Row 9: chi2
        if self._compact_chi2 is not None:
            otherpara[9, :n_spec] = self._compact_chi2[:n_spec]

    def __repr__(self) -> str:
        status = "open" if self._h5_out is not None else "closed"
        return (
            f"StreamingFitparaWriter({self.config.output_path}, "
            f"n_spectra={self.config.n_spectra}, {status})"
        )
