"""Tests for fitpara compression codec.

Tests the int16 + column-wise + LZ4 compression pipeline.
"""

import time

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from toyomacro.voigtfit.h5io import (
    HAS_LZ4,
    FitparaCodec,
    FitparaCodecConfig,
    compress_array_lz4,
    compress_array_to_h5,
    compress_fitpara,
    decompress_array_from_h5,
    decompress_array_lz4,
    decompress_fitpara,
    read_specdata_uint16,
    write_specdata_uint16,
)

# Skip all tests if LZ4 not available
pytestmark = pytest.mark.skipif(not HAS_LZ4, reason="lz4 not installed")


def _median_time(fn, repeats=5, warmup=3):
    """Median wall time of ``fn`` over ``repeats`` runs, after warm-up."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# Gate decode against a copy of the array it produces, rather than against
# an absolute spec/s. An absolute rate is not calibratable here: the same
# workload measures 16.1-17.0M spec/s inside a full pytest run and
# 22.7-33.3M standalone on this host, so the harness moves it further than
# some hardware does.
#
# The two sides are NOT symmetric, and the bound has to be read with that
# in mind. A copy is bandwidth-bound; decode is not. Decomposed on the
# array codec (40 MB payload, 4-vCPU Xeon @2.80GHz):
#
#     lz4.frame.decompress   27.8 ms   73.5% of decode   1.44 GB/s out
#     frombuffer().copy()     5.4 ms   14.3% of decode
#     bare ndarray.copy()     5.3 ms                    15.0 GB/s (r+w)
#
# Decode moves 80 MB and the copy moves 80 MB, so a bandwidth-only floor
# would be 1.0; the measured ratio is ~7 because the LZ4 token loop runs
# an order of magnitude below memcpy per byte. For the fitpara codec the
# int16 -> float32 dequantise dominates instead. The ratio is therefore
# (scalar, branchy) / (streaming copy), and those track each other only
# loosely across microarchitectures -- the bound is deliberately far above
# anything measured rather than tight.
#
# Measured ratios, five runs each on that host: 10.3-13.2 (fitpara),
# 6.7-7.4 (array). Not established on other hardware: the CI matrix
# (ubuntu + macOS x 3.11/3.12) passes, but records no value.
MAX_DECODE_OVER_COPY = 40


def generate_realistic_fitpara(n_spectra: int, n_comp: int = 3) -> np.ndarray:
    """Generate realistic fitpara data for testing.

    Simulates:
    - Spatially smooth amplitude variations
    - Small center shifts around base positions
    - Mostly constant sigma/gamma
    - Sparse alpha/BR/SO columns
    """
    # Use fixed seed for reproducibility
    np.random.seed(42)

    # 2D map shape (approximate)
    side = int(np.sqrt(n_spectra))
    shape_2d = (side, side)
    actual_n = side * side

    fitpara = np.zeros((actual_n, n_comp, 9), dtype=np.float32)

    # [0] amplitude: spatially smooth positive values
    base_amps = [1000.0, 500.0, 200.0][:n_comp]
    for k in range(n_comp):
        smooth = gaussian_filter(np.random.randn(*shape_2d), sigma=15).astype(np.float32).ravel()
        fitpara[:, k, 0] = base_amps[k] * (0.5 + np.abs(smooth / smooth.std()))

    # [1] center: base ± small shift
    base_centers = [530.0, 531.5, 533.0][:n_comp]
    for k in range(n_comp):
        smooth = gaussian_filter(np.random.randn(*shape_2d), sigma=20).astype(np.float32).ravel()
        fitpara[:, k, 1] = base_centers[k] + 0.3 * smooth / smooth.std()

    # [2] sigma: mostly constant
    base_sigmas = [0.7, 0.8, 0.9][:n_comp]
    for k in range(n_comp):
        fitpara[:, k, 2] = base_sigmas[k]

    # [3] gamma: constant
    fitpara[:, :, 3] = 0.25

    # [4] alpha: zeros
    fitpara[:, :, 4] = 0.0

    # [5] BR: first component = 1, others = 0
    fitpara[:, 0, 5] = 1.0

    # [6] SO: zeros
    fitpara[:, :, 6] = 0.0

    # [7] func_type: constant 1 (Voigt)
    fitpara[:, :, 7] = 1.0

    # [8] chi2: spatially smooth small values
    smooth = gaussian_filter(np.random.randn(*shape_2d), sigma=10).astype(np.float32).ravel()
    fitpara[:, :, 8] = (0.01 + 0.005 * np.abs(smooth / smooth.std()))[:, np.newaxis]

    return fitpara


class TestFitparaCodec:
    """Tests for FitparaCodec class."""

    def test_encode_decode_roundtrip(self):
        """Test that encode->decode preserves data within tolerance."""
        fitpara = generate_realistic_fitpara(10000, n_comp=3)

        codec = FitparaCodec()
        compressed = codec.encode(fitpara)
        restored = codec.decode(compressed)

        # Check shape preserved
        assert restored.shape == fitpara.shape

        # Check precision per column
        col_names = ['amplitude', 'center', 'sigma', 'gamma', 'alpha', 'BR', 'SO', 'func_type', 'chi2']
        tolerances = {
            0: 0.002,   # amplitude: 0.2% relative
            1: 0.001,   # center: 0.001 eV absolute
            2: 0.001,   # sigma: 0.1%
            3: 0.001,   # gamma: 0.1%
            4: 0.0,     # alpha: exact (zeros)
            5: 0.0001,  # BR: 0.01%
            6: 0.0,     # SO: exact (zeros)
            7: 0.0,     # func_type: exact (constant)
            8: 0.0001,  # chi2: 0.01%
        }

        for col_idx, name in enumerate(col_names):
            orig = fitpara[:, :, col_idx]
            rest = restored[:, :, col_idx]

            if orig.max() != orig.min():
                rel_error = np.abs(orig - rest) / (np.abs(orig) + 1e-10)
                max_rel = rel_error.max()
                assert max_rel < tolerances[col_idx], f"{name}: max rel error {max_rel:.6f} > {tolerances[col_idx]}"
            else:
                # Constant column: check exact match
                abs_error = np.abs(orig - rest).max()
                assert abs_error < 1e-6, f"{name}: constant column error {abs_error}"

    def test_compression_ratio(self):
        """Test that compression achieves expected ratio."""
        fitpara = generate_realistic_fitpara(100000, n_comp=3)

        codec = FitparaCodec()
        compressed = codec.encode(fitpara)

        # Expect at least 4x compression (benchmark showed 6.1x)
        assert compressed.compression_ratio > 4.0, f"Compression ratio {compressed.compression_ratio:.2f}x < 4x"

        # Compressed size should be significantly smaller
        assert compressed.compressed_size < compressed.original_size / 3

    def test_decode_speed(self):
        """Decode stays within a fixed multiple of copying its output.

        The claim this guards is that decode does not bottleneck the E2E
        pipeline, which is a statement about cost relative to the data,
        not an absolute rate -- so the assertion is a ratio against a
        plain copy of the array decode produces.

        An absolute target was not calibratable: this workload measures
        16.1-17.0M spec/s inside a full pytest run and 22.7-33.3M
        standalone on one host, and the published figures for it span
        3.4M to 35.8M across environments whose measurement commands
        differ (see the decode_speed history in
        docs/CUDA_BACKEND_POC.md).

        Validity limit: this is not a bandwidth-vs-bandwidth comparison.
        Decode here is dominated by the int16 -> float32 dequantise loop,
        the copy by memcpy, so the ratio is compute over bandwidth and is
        only loosely stable across microarchitectures. See
        MAX_DECODE_OVER_COPY for the decomposition and the measured
        range; the bound is set far above it rather than tight.
        """
        fitpara = generate_realistic_fitpara(1000000, n_comp=3)
        codec = FitparaCodec()
        compressed = codec.encode(fitpara)
        decoded = codec.decode(compressed)

        t_decode = _median_time(lambda: codec.decode(compressed))
        t_copy = _median_time(decoded.copy)

        ratio = t_decode / t_copy
        rate = fitpara.shape[0] / t_decode / 1e6
        assert ratio < MAX_DECODE_OVER_COPY, (
            f"decode costs {ratio:.1f}x a copy of the same array "
            f"({rate:.1f}M spec/s), over the {MAX_DECODE_OVER_COPY}x bound"
        )

    def test_disabled_compression(self):
        """Test codec with compression disabled."""
        fitpara = generate_realistic_fitpara(1000, n_comp=2)

        config = FitparaCodecConfig(enable_compression=False)
        codec = FitparaCodec(config)

        compressed = codec.encode(fitpara)
        restored = codec.decode(compressed)

        # Should be exact match (no quantization)
        np.testing.assert_array_equal(fitpara, restored)

        # No compression
        assert compressed.compression_ratio == 1.0

    def test_handles_nan_values(self):
        """Test that NaN values are handled correctly."""
        fitpara = generate_realistic_fitpara(1000, n_comp=2)

        # Introduce some NaN values
        fitpara[10:20, :, :] = np.nan

        codec = FitparaCodec()
        compressed = codec.encode(fitpara)
        restored = codec.decode(compressed)

        # Non-NaN values should be preserved
        mask = ~np.isnan(fitpara)
        rel_error = np.abs(fitpara[mask] - restored[mask]) / (np.abs(fitpara[mask]) + 1e-10)
        assert rel_error.max() < 0.01

    def test_single_component(self):
        """Test with single component."""
        fitpara = generate_realistic_fitpara(10000, n_comp=1)

        codec = FitparaCodec()
        compressed = codec.encode(fitpara)
        restored = codec.decode(compressed)

        assert restored.shape == fitpara.shape
        # Amplitude precision
        rel_error = np.abs(fitpara[:, :, 0] - restored[:, :, 0]) / (np.abs(fitpara[:, :, 0]) + 1e-10)
        assert rel_error.max() < 0.002


class TestConvenienceFunctions:
    """Tests for compress_fitpara and decompress_fitpara."""

    def test_convenience_roundtrip(self):
        """Test convenience function roundtrip."""
        fitpara = generate_realistic_fitpara(5000, n_comp=3)

        compressed = compress_fitpara(fitpara)
        restored = decompress_fitpara(compressed)

        assert restored.shape == fitpara.shape

        # Check amplitude precision
        rel_error = np.abs(fitpara[:, :, 0] - restored[:, :, 0]) / (np.abs(fitpara[:, :, 0]) + 1e-10)
        assert rel_error.max() < 0.002


class TestH5Integration:
    """Tests for HDF5 integration."""

    def test_h5_roundtrip(self, tmp_path):
        """Test encode/decode via HDF5."""
        import h5py

        fitpara = generate_realistic_fitpara(10000, n_comp=3)
        h5_path = tmp_path / "test_compressed.h5"

        codec = FitparaCodec()

        # Write compressed to HDF5
        with h5py.File(h5_path, 'w') as f:
            codec.encode_to_h5(f, fitpara, 'fitpara_compressed')

        # Read back
        with h5py.File(h5_path, 'r') as f:
            restored = codec.decode_from_h5(f, 'fitpara_compressed')

        assert restored.shape == fitpara.shape

        # Check precision
        rel_error = np.abs(fitpara[:, :, 0] - restored[:, :, 0]) / (np.abs(fitpara[:, :, 0]) + 1e-10)
        assert rel_error.max() < 0.002

    def test_h5_metadata(self, tmp_path):
        """Test that metadata is correctly stored in HDF5."""
        import h5py

        fitpara = generate_realistic_fitpara(5000, n_comp=2)
        h5_path = tmp_path / "test_metadata.h5"

        codec = FitparaCodec()

        with h5py.File(h5_path, 'w') as f:
            codec.encode_to_h5(f, fitpara)

        # Check metadata
        with h5py.File(h5_path, 'r') as f:
            grp = f['fitpara_compressed']
            assert 'data' in grp
            assert 'shape' in grp.attrs
            assert 'scales' in grp.attrs
            assert 'offsets' in grp.attrs
            assert 'compression_ratio' in grp.attrs

            shape = tuple(grp.attrs['shape'])
            assert shape == fitpara.shape


class TestArrayLZ4Compression:
    """Tests for simple LZ4 array compression (otherpara/xytdata)."""

    def test_otherpara_compression(self):
        """Test compression of otherpara (mostly constant values)."""
        n_spectra = 100_000

        # otherpara: (10, n_spectra) with mostly constant values
        otherpara = np.zeros((10, n_spectra), dtype=np.float32)
        otherpara[0] = 1.0
        otherpara[1] = 528.5
        otherpara[2] = 533.5
        otherpara[3:] = np.random.rand(7)[:, np.newaxis]

        compressed = compress_array_lz4(otherpara)
        restored = decompress_array_lz4(compressed)

        # Should be exact match
        np.testing.assert_array_equal(otherpara, restored)

        # Expect very high compression (>100x)
        assert compressed.compression_ratio > 100, f"Ratio {compressed.compression_ratio:.1f}x"

    def test_xytdata_compression(self):
        """Test compression of xytdata (grid coordinates)."""
        n_spectra = 100_000
        side = int(np.sqrt(n_spectra))

        # xytdata: (3, n_spectra) - x, y grid coordinates, t constant
        xytdata = np.zeros((3, side * side), dtype=np.float32)
        xytdata[0] = np.tile(np.arange(side, dtype=np.float32), side)
        xytdata[1] = np.repeat(np.arange(side, dtype=np.float32), side)
        xytdata[2] = 1.0

        compressed = compress_array_lz4(xytdata)
        restored = decompress_array_lz4(compressed)

        # Should be exact match
        np.testing.assert_array_equal(xytdata, restored)

        # Expect very high compression (>100x)
        assert compressed.compression_ratio > 100, f"Ratio {compressed.compression_ratio:.1f}x"

    def test_h5_roundtrip(self, tmp_path):
        """Test HDF5 roundtrip for array compression."""
        import h5py

        n_spectra = 10_000
        otherpara = np.zeros((10, n_spectra), dtype=np.float32)
        otherpara[0] = 1.0
        otherpara[1] = 528.5

        h5_path = tmp_path / "test_array.h5"

        # Write
        with h5py.File(h5_path, 'w') as f:
            compress_array_to_h5(f, otherpara, 'otherpara_compressed')

        # Read
        with h5py.File(h5_path, 'r') as f:
            restored = decompress_array_from_h5(f, 'otherpara_compressed')

        np.testing.assert_array_equal(otherpara, restored)

    def test_decode_speed(self):
        """Decode of a large array stays within a fixed multiple of a copy.

        A ratio for the same reason as `TestFitparaCodec.test_decode_speed`.
        Two things worth knowing about this one:

        Encode is not the baseline because decode is *slower* than encode
        here -- a nearly constant payload compresses almost for free,
        while decompression writes 80 MB (40 MB out of LZ4, then 40 MB
        again through `np.frombuffer(...).copy()`).

        That second copy is literally the operation this test compares
        against, so the ratio has a structural floor near 1, and the
        headroom above it is the LZ4 token loop -- 73.5% of decode at
        1.44 GB/s against 15.0 GB/s for memcpy on the host measured.
        """
        n_spectra = 1_000_000

        # Typical otherpara
        otherpara = np.zeros((10, n_spectra), dtype=np.float32)
        otherpara[0] = 1.0
        otherpara[1] = 528.5

        compressed = compress_array_lz4(otherpara)
        decoded = decompress_array_lz4(compressed)

        t_decode = _median_time(lambda: decompress_array_lz4(compressed))
        t_copy = _median_time(decoded.copy)

        ratio = t_decode / t_copy
        rate = n_spectra / t_decode / 1e6
        assert ratio < MAX_DECODE_OVER_COPY, (
            f"decode costs {ratio:.1f}x a copy of the same array "
            f"({rate:.1f}M spec/s), over the {MAX_DECODE_OVER_COPY}x bound"
        )


class TestSpecdataUint16:
    """Tests for uint16 specdata I/O (MLX direct transfer)."""

    def test_write_read_roundtrip(self, tmp_path):
        """Test write and read uint16 specdata."""
        n_spectra = 10_000
        n_energy = 101

        # Simulate XPS count data (integer values)
        specdata = np.random.poisson(100, size=(n_spectra, n_energy)).astype(np.float32)
        energy = np.linspace(525, 540, n_energy).astype(np.float32)

        h5_path = tmp_path / "test_uint16.h5"

        # Write
        stats = write_specdata_uint16(h5_path, specdata, energy)
        assert stats['n_spectra'] == n_spectra
        assert stats['n_energy'] == n_energy

        # Read
        restored, restored_energy = read_specdata_uint16(h5_path)

        # Check dtype is uint16 (no CPU conversion)
        assert restored.dtype == np.uint16

        # Check values match
        np.testing.assert_array_equal(specdata.astype(np.uint16), restored)
        np.testing.assert_allclose(energy, restored_energy)

    def test_file_size_reduction(self, tmp_path):
        """Test that uint16 reduces file size by ~50%."""
        import h5py

        n_spectra = 100_000
        n_energy = 101

        specdata = np.random.poisson(100, size=(n_spectra, n_energy)).astype(np.float32)
        energy = np.linspace(525, 540, n_energy).astype(np.float32)

        # Write float32
        h5_f32 = tmp_path / "float32.h5"
        with h5py.File(h5_f32, 'w') as f:
            f.create_dataset('specdata', data=specdata)
        size_f32 = h5_f32.stat().st_size

        # Write uint16
        h5_u16 = tmp_path / "uint16.h5"
        write_specdata_uint16(h5_u16, specdata, energy)
        size_u16 = h5_u16.stat().st_size

        # uint16 should be roughly 50% of float32
        ratio = size_u16 / size_f32
        assert ratio < 0.55, f"Size ratio {ratio:.2f} > 0.55"

    def test_partial_read(self, tmp_path):
        """Test reading subset of spectra."""
        n_spectra = 10_000
        n_energy = 101

        specdata = np.random.poisson(100, size=(n_spectra, n_energy)).astype(np.float32)
        energy = np.linspace(525, 540, n_energy).astype(np.float32)

        h5_path = tmp_path / "test_partial.h5"
        write_specdata_uint16(h5_path, specdata, energy)

        # Read subset
        start_idx = 1000
        count = 500
        restored, _ = read_specdata_uint16(h5_path, start_idx=start_idx, count=count)

        assert restored.shape == (count, n_energy)
        np.testing.assert_array_equal(
            specdata[start_idx:start_idx+count].astype(np.uint16),
            restored
        )

    def test_mlx_direct_transfer(self, tmp_path):
        """Test that uint16 data can be directly transferred to MLX."""
        from toyomacro.voigtfit._mlx_support import mlx_usable
        if not mlx_usable():
            pytest.skip("MLX not usable (not installed, or the default device failed the probe)")
        import mlx.core as mx

        n_spectra = 10_000
        n_energy = 101
        n_comp = 3

        specdata = np.random.poisson(100, size=(n_spectra, n_energy)).astype(np.float32)
        energy = np.linspace(525, 540, n_energy).astype(np.float32)

        h5_path = tmp_path / "test_mlx.h5"
        write_specdata_uint16(h5_path, specdata, energy)

        # Read as uint16
        specdata_u16, _ = read_specdata_uint16(h5_path)
        assert specdata_u16.dtype == np.uint16

        # Transfer to MLX
        specdata_mx = mx.array(specdata_u16.T)  # (n_energy, n_spectra)
        assert specdata_mx.dtype == mx.uint16

        # Weight matrix
        W = mx.array(np.random.randn(n_comp, n_energy).astype(np.float32))

        # Matmul should work
        result = W @ specdata_mx
        mx.eval(result)

        # Result should be float32
        assert result.dtype == mx.float32
        assert result.shape == (n_comp, n_spectra)

    def test_mlx_throughput(self, tmp_path):
        """Test MLX throughput with uint16 data."""
        from toyomacro.voigtfit._mlx_support import mlx_usable
        if not mlx_usable():
            pytest.skip("MLX not usable (not installed, or the default device failed the probe)")
        import time

        import mlx.core as mx

        n_spectra = 1_000_000
        n_energy = 101
        n_comp = 3

        # Generate data in memory (skip file I/O for this test)
        specdata_u16 = np.random.randint(0, 500, size=(n_energy, n_spectra), dtype=np.uint16)
        W = mx.array(np.random.randn(n_comp, n_energy).astype(np.float32))

        # Warmup
        specdata_mx = mx.array(specdata_u16)
        for _ in range(3):
            result = W @ specdata_mx
            mx.eval(result)

        # Benchmark
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            specdata_mx = mx.array(specdata_u16)
            result = W @ specdata_mx
            mx.eval(result)
            times.append(time.perf_counter() - t0)

        median_time = np.median(times)
        rate = n_spectra / median_time / 1e6

        # Should exceed 28M target significantly
        assert rate > 50, f"uint16→MLX rate {rate:.1f}M spec/s < 50M"

    def test_reject_out_of_range(self, tmp_path):
        """Test that out-of-range values are rejected."""
        n_spectra = 100
        n_energy = 10

        energy = np.linspace(525, 540, n_energy).astype(np.float32)
        h5_path = tmp_path / "test_range.h5"

        # Test max > 65535
        specdata_too_large = np.full((n_spectra, n_energy), 70000, dtype=np.float32)
        with pytest.raises(ValueError, match="exceeds uint16"):
            write_specdata_uint16(h5_path, specdata_too_large, energy)

        # Test negative values
        specdata_negative = np.full((n_spectra, n_energy), -1, dtype=np.float32)
        with pytest.raises(ValueError, match="negative"):
            write_specdata_uint16(h5_path, specdata_negative, energy)

    def test_gzip_compression(self, tmp_path):
        """Test gzip compression option."""
        n_spectra = 10_000
        n_energy = 101

        specdata = np.random.poisson(100, size=(n_spectra, n_energy)).astype(np.float32)
        energy = np.linspace(525, 540, n_energy).astype(np.float32)

        h5_path = tmp_path / "test_gzip.h5"

        # Write with gzip
        stats = write_specdata_uint16(h5_path, specdata, energy, compression='gzip')
        assert stats['compression'] == 'gzip'

        # Read back
        restored, restored_energy = read_specdata_uint16(h5_path)
        assert restored.dtype == np.uint16
        np.testing.assert_array_equal(specdata.astype(np.uint16), restored)
        np.testing.assert_allclose(energy, restored_energy)

    def test_lz4_compression(self, tmp_path):
        """Test LZ4 compression option."""
        n_spectra = 10_000
        n_energy = 101

        specdata = np.random.poisson(100, size=(n_spectra, n_energy)).astype(np.float32)
        energy = np.linspace(525, 540, n_energy).astype(np.float32)

        h5_path = tmp_path / "test_lz4.h5"

        # Write with lz4
        stats = write_specdata_uint16(h5_path, specdata, energy, compression='lz4')
        assert stats['compression'] == 'lz4'

        # Read back
        restored, restored_energy = read_specdata_uint16(h5_path)
        assert restored.dtype == np.uint16
        np.testing.assert_array_equal(specdata.astype(np.uint16), restored)
        np.testing.assert_allclose(energy, restored_energy)

    def test_lz4_partial_read(self, tmp_path):
        """Test partial read from LZ4 compressed data."""
        n_spectra = 10_000
        n_energy = 101

        specdata = np.random.poisson(100, size=(n_spectra, n_energy)).astype(np.float32)
        energy = np.linspace(525, 540, n_energy).astype(np.float32)

        h5_path = tmp_path / "test_lz4_partial.h5"
        write_specdata_uint16(h5_path, specdata, energy, compression='lz4')

        # Partial read (note: decompresses all, then slices)
        start_idx = 1000
        count = 500
        restored, _ = read_specdata_uint16(h5_path, start_idx=start_idx, count=count)

        assert restored.shape == (count, n_energy)
        np.testing.assert_array_equal(
            specdata[start_idx:start_idx+count].astype(np.uint16),
            restored
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
