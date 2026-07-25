"""Tests for chunked fitpara encoding vs monolithic encoder.

Verifies that _encode_fitpara_chunked_to_h5 produces output within ±1 int16 LSB
of _encode_fitpara_to_h5. The ±1 difference arises from float32 rounding in
intermediate (col - offset) * scale computation with different memory layouts
(3D slice view vs 2D contiguous array).
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from toyomacro.io.compression import (
    HAS_LZ4,
    INT16_RANGE,
    _encode_fitpara_chunked_to_h5,
    _encode_fitpara_to_h5,
    decode_fitpara_from_h5,
)

pytestmark = pytest.mark.skipif(not HAS_LZ4, reason="lz4 not installed")


def _make_compact_buffers(
    n_spectra: int = 100_000,
    n_comp: int = 3,
    seed: int = 42,
):
    """Create synthetic compact buffers mimicking real batch fit output."""
    rng = np.random.default_rng(seed)
    amplitudes = rng.exponential(scale=500.0, size=(n_spectra, n_comp)).astype(np.float32)
    constants = np.array([
        [99.5, 0.8, 0.3, 0.0, 0.5, 0.6, 3.0],
        [100.5, 0.9, 0.35, 0.0, 0.5, 0.6, 3.0],
        [102.0, 1.0, 0.4, 0.0, 0.0, 0.0, 3.0],
    ], dtype=np.float32)[:n_comp]
    basis_integrals = np.array([1.23, 1.45, 1.67], dtype=np.float32)[:n_comp]
    chi2 = rng.exponential(scale=0.01, size=n_spectra).astype(np.float32)
    return amplitudes, constants, basis_integrals, chi2


def _reconstruct_fitpara(amplitudes, constants, basis_integrals):
    """Reconstruct full (n, comp, 9) fitpara from compact buffers."""
    n_spectra, n_comp = amplitudes.shape
    fitpara = np.zeros((n_spectra, n_comp, 9), dtype=np.float32)
    fitpara[:, :, 0] = amplitudes
    fitpara[:, :, 1:8] = constants[np.newaxis, :, :]
    fitpara[:, :, 8] = amplitudes * basis_integrals[np.newaxis, :]
    return fitpara


def _assert_within_int16_lsb(decoded_mono, decoded_chunk, max_lsb_diff=1):
    """Assert decoded arrays differ by at most ±max_lsb_diff int16 steps.

    The int16 quantization has resolution = range / 32767 per column.
    ±1 LSB difference is inherent to float32 rounding during quantization.
    """
    assert decoded_mono.shape == decoded_chunk.shape
    diff = np.abs(decoded_mono - decoded_chunk)

    # Compute per-column int16 LSB size
    for col_idx in range(decoded_mono.shape[2]):
        col_mono = decoded_mono[:, :, col_idx]
        col_diff = diff[:, :, col_idx]
        col_range = col_mono.max() - col_mono.min()
        if col_range == 0:
            # Constant column: must be exact
            assert np.all(col_diff == 0), f"Col {col_idx}: constant but has diff"
            continue
        lsb = col_range / INT16_RANGE
        max_diff = col_diff.max()
        assert max_diff <= lsb * max_lsb_diff * 1.01, (  # 1% margin for float32
            f"Col {col_idx}: max_diff={max_diff:.6f} > {max_lsb_diff} LSB ({lsb:.6f})"
        )


class TestChunkedEncoding:
    """Verify chunked encoder produces output within ±1 int16 LSB of monolithic."""

    def test_close_to_monolithic_small(self):
        """100K spectra: decoded output within ±1 int16 LSB."""
        amplitudes, constants, basis_integrals, chi2 = _make_compact_buffers(
            n_spectra=100_000, n_comp=3
        )
        fitpara = _reconstruct_fitpara(amplitudes, constants, basis_integrals)

        with tempfile.TemporaryDirectory() as tmpdir:
            mono_path = Path(tmpdir) / "mono.h5"
            chunk_path = Path(tmpdir) / "chunk.h5"

            with h5py.File(mono_path, "w") as f:
                _encode_fitpara_to_h5(f, fitpara)
            with h5py.File(chunk_path, "w") as f:
                _encode_fitpara_chunked_to_h5(
                    f, amplitudes, constants, basis_integrals, chunk_size=30_000,
                )

            with h5py.File(mono_path, "r") as f:
                decoded_mono = decode_fitpara_from_h5(f)
            with h5py.File(chunk_path, "r") as f:
                decoded_chunk = decode_fitpara_from_h5(f)

            _assert_within_int16_lsb(decoded_mono, decoded_chunk)

    def test_scales_offsets_identical(self):
        """Verify scales/offsets metadata match exactly."""
        amplitudes, constants, basis_integrals, chi2 = _make_compact_buffers(
            n_spectra=50_000, n_comp=3
        )
        fitpara = _reconstruct_fitpara(amplitudes, constants, basis_integrals)

        with tempfile.TemporaryDirectory() as tmpdir:
            mono_path = Path(tmpdir) / "mono.h5"
            chunk_path = Path(tmpdir) / "chunk.h5"

            with h5py.File(mono_path, "w") as f:
                _encode_fitpara_to_h5(f, fitpara)
            with h5py.File(chunk_path, "w") as f:
                _encode_fitpara_chunked_to_h5(
                    f, amplitudes, constants, basis_integrals, chunk_size=20_000,
                )

            with h5py.File(mono_path, "r") as f:
                mono_scales = f["fitpara_compressed"].attrs["scales"]
                mono_offsets = f["fitpara_compressed"].attrs["offsets"]
            with h5py.File(chunk_path, "r") as f:
                chunk_scales = f["fitpara_compressed"].attrs["scales"]
                chunk_offsets = f["fitpara_compressed"].attrs["offsets"]

            np.testing.assert_array_equal(mono_scales, chunk_scales)
            np.testing.assert_array_equal(mono_offsets, chunk_offsets)

    def test_single_component(self):
        """Edge case: single peak component."""
        amplitudes, constants, basis_integrals, chi2 = _make_compact_buffers(
            n_spectra=10_000, n_comp=1
        )
        fitpara = _reconstruct_fitpara(amplitudes, constants, basis_integrals)

        with tempfile.TemporaryDirectory() as tmpdir:
            mono_path = Path(tmpdir) / "mono.h5"
            chunk_path = Path(tmpdir) / "chunk.h5"

            with h5py.File(mono_path, "w") as f:
                _encode_fitpara_to_h5(f, fitpara)
            with h5py.File(chunk_path, "w") as f:
                _encode_fitpara_chunked_to_h5(
                    f, amplitudes, constants, basis_integrals, chunk_size=3_000,
                )

            with h5py.File(mono_path, "r") as f:
                decoded_mono = decode_fitpara_from_h5(f)
            with h5py.File(chunk_path, "r") as f:
                decoded_chunk = decode_fitpara_from_h5(f)

            _assert_within_int16_lsb(decoded_mono, decoded_chunk)

    def test_constant_amplitudes(self):
        """Edge case: all amplitudes identical (scale=1, offset=const)."""
        n_spectra = 5_000
        n_comp = 2
        amplitudes = np.full((n_spectra, n_comp), 42.0, dtype=np.float32)
        constants = np.array([
            [100.0, 0.5, 0.2, 0.0, 0.0, 0.0, 3.0],
            [101.0, 0.6, 0.3, 0.0, 0.0, 0.0, 3.0],
        ], dtype=np.float32)
        basis_integrals = np.array([1.0, 1.5], dtype=np.float32)

        fitpara = _reconstruct_fitpara(amplitudes, constants, basis_integrals)

        with tempfile.TemporaryDirectory() as tmpdir:
            mono_path = Path(tmpdir) / "mono.h5"
            chunk_path = Path(tmpdir) / "chunk.h5"

            with h5py.File(mono_path, "w") as f:
                _encode_fitpara_to_h5(f, fitpara)
            with h5py.File(chunk_path, "w") as f:
                _encode_fitpara_chunked_to_h5(
                    f, amplitudes, constants, basis_integrals, chunk_size=2_000,
                )

            with h5py.File(mono_path, "r") as f:
                decoded_mono = decode_fitpara_from_h5(f)
            with h5py.File(chunk_path, "r") as f:
                decoded_chunk = decode_fitpara_from_h5(f)

            np.testing.assert_array_equal(decoded_mono, decoded_chunk)

    def test_chunk_size_larger_than_data(self):
        """Edge case: chunk_size > n_spectra (single chunk)."""
        amplitudes, constants, basis_integrals, chi2 = _make_compact_buffers(
            n_spectra=1_000, n_comp=3
        )
        fitpara = _reconstruct_fitpara(amplitudes, constants, basis_integrals)

        with tempfile.TemporaryDirectory() as tmpdir:
            mono_path = Path(tmpdir) / "mono.h5"
            chunk_path = Path(tmpdir) / "chunk.h5"

            with h5py.File(mono_path, "w") as f:
                _encode_fitpara_to_h5(f, fitpara)
            with h5py.File(chunk_path, "w") as f:
                _encode_fitpara_chunked_to_h5(
                    f, amplitudes, constants, basis_integrals,
                    chunk_size=10_000_000,
                )

            with h5py.File(mono_path, "r") as f:
                decoded_mono = decode_fitpara_from_h5(f)
            with h5py.File(chunk_path, "r") as f:
                decoded_chunk = decode_fitpara_from_h5(f)

            _assert_within_int16_lsb(decoded_mono, decoded_chunk)

    def test_compression_ratio_comparable(self):
        """Chunked and monolithic should produce similar compression ratios."""
        amplitudes, constants, basis_integrals, chi2 = _make_compact_buffers(
            n_spectra=100_000, n_comp=3
        )
        fitpara = _reconstruct_fitpara(amplitudes, constants, basis_integrals)

        with tempfile.TemporaryDirectory() as tmpdir:
            mono_path = Path(tmpdir) / "mono.h5"
            chunk_path = Path(tmpdir) / "chunk.h5"

            with h5py.File(mono_path, "w") as f:
                mono_stats = _encode_fitpara_to_h5(f, fitpara)
            with h5py.File(chunk_path, "w") as f:
                chunk_stats = _encode_fitpara_chunked_to_h5(
                    f, amplitudes, constants, basis_integrals, chunk_size=30_000,
                )

            # Ratios should be within 5% of each other
            ratio_diff = abs(mono_stats["ratio"] - chunk_stats["ratio"])
            assert ratio_diff < mono_stats["ratio"] * 0.05, (
                f"Ratio diff too large: mono={mono_stats['ratio']:.2f}, "
                f"chunk={chunk_stats['ratio']:.2f}"
            )

    def test_streaming_writer_roundtrip(self):
        """End-to-end: StreamingFitparaWriter compact mode uses chunked encoder."""
        from toyomacro.io.writers import StreamingFitparaWriter, StreamingWriterConfig

        n_spectra = 50_000
        n_comp = 3
        amplitudes, constants, basis_integrals, chi2 = _make_compact_buffers(
            n_spectra=n_spectra, n_comp=n_comp
        )
        fitpara = _reconstruct_fitpara(amplitudes, constants, basis_integrals)

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(tmpdir) / "source.h5"
            with h5py.File(src_path, "w") as f:
                f.create_dataset("specdata", shape=(100, 10), dtype=np.float32)

            config = StreamingWriterConfig(
                output_path=str(src_path),
                n_spectra=n_spectra,
                n_components=n_comp,
                write_mode="memory",
            )
            writer = StreamingFitparaWriter(config, input_path=str(src_path))
            writer.__enter__()

            batch_size = 10_000
            for batch_idx, start in enumerate(range(0, n_spectra, batch_size)):
                end = min(start + batch_size, n_spectra)
                writer.write_batch(
                    batch_idx=batch_idx,
                    start_idx=start,
                    fitpara=fitpara[start:end],
                    chi2=chi2[start:end],
                )

            writer.__exit__(None, None, None)

            with h5py.File(src_path, "r") as f:
                assert "fitpara_compressed" in f
                decoded = decode_fitpara_from_h5(f)

            # Compare with monolithic encode
            mono_path = Path(tmpdir) / "mono.h5"
            with h5py.File(mono_path, "w") as f:
                _encode_fitpara_to_h5(f, fitpara)
            with h5py.File(mono_path, "r") as f:
                decoded_mono = decode_fitpara_from_h5(f)

            _assert_within_int16_lsb(decoded_mono, decoded)
