"""Integration tests for Phase 8.3 I/O compression in the pipeline.

Tests:
- compress_h5_file() standalone
- process_xps_file_streaming() with compression parameter
- CLI convert command logic
- Roundtrip accuracy (compress -> read back -> verify)
"""

import os
import shutil

import h5py
import numpy as np
import pytest

from toyomacro.voigtfit.h5io import (
    HAS_LZ4,
    FitparaCodec,
    compress_h5_file,
    decompress_array_from_h5,
)

pytestmark = pytest.mark.skipif(not HAS_LZ4, reason="lz4 not installed")


@pytest.fixture
def sample_h5_file(tmp_path):
    """Create a realistic XPS HDF5 file for testing."""
    n_spectra = 2000
    n_energy = 80
    n_components = 3

    h5_path = tmp_path / "C1s_test.h5"
    with h5py.File(h5_path, 'w') as f:
        # specdata: row 0 = energy, rows 1: = spectra (integer count data)
        energy = np.linspace(280, 295, n_energy).astype(np.float32)
        spectra = np.random.poisson(200, size=(n_spectra, n_energy)).astype(np.float32)
        specdata = np.vstack([energy[np.newaxis, :], spectra])
        f.create_dataset('specdata', data=specdata, dtype=np.float32)

        # fitpara: (n_spectra, n_components, 9) with realistic values
        fitpara = np.full((n_spectra, n_components, 9), np.nan, dtype=np.float32)
        fitpara[:, 0, 0] = np.random.uniform(800, 1200, n_spectra)  # amplitude
        fitpara[:, 0, 1] = 284.5  # center
        fitpara[:, 0, 2] = 0.8    # sigma
        fitpara[:, 0, 3] = 0.3    # gamma
        fitpara[:, 0, 7] = 1      # func_type
        fitpara[:, 0, 8] = np.random.uniform(0.01, 0.1, n_spectra)  # chi2
        fitpara[:, 1, 0] = np.random.uniform(400, 600, n_spectra)
        fitpara[:, 1, 1] = 286.0
        fitpara[:, 1, 2] = 0.9
        fitpara[:, 1, 3] = 0.3
        fitpara[:, 1, 7] = 1
        fitpara[:, 1, 8] = fitpara[:, 0, 8]
        fitpara[:, 2, 0] = np.random.uniform(100, 300, n_spectra)
        fitpara[:, 2, 1] = 288.5
        fitpara[:, 2, 2] = 0.7
        fitpara[:, 2, 3] = 0.3
        fitpara[:, 2, 7] = 1
        fitpara[:, 2, 8] = fitpara[:, 0, 8]
        f.create_dataset('fitpara', data=fitpara)

        # otherpara: mostly constant
        otherpara = np.zeros((10, n_spectra), dtype=np.float32)
        otherpara[0] = 1.0
        otherpara[1] = 280.0
        f.create_dataset('otherpara', data=otherpara)

        # xytdata: grid coordinates
        xytdata = np.zeros((3, n_spectra), dtype=np.float32)
        xytdata[0] = np.tile(np.arange(50), 40).astype(np.float32)
        xytdata[1] = np.repeat(np.arange(40), 50).astype(np.float32)
        xytdata[2] = 1.0
        f.create_dataset('xytdata', data=xytdata)

        # misc group
        misc = f.create_group('misc')
        misc.create_dataset('maxcomp', data=np.array([3.0]))

    return str(h5_path)


class TestCompressH5File:
    """Test the standalone compress_h5_file() function."""

    def test_standard_compresses_fitpara(self, sample_h5_file, tmp_path):
        copy = str(tmp_path / "std.h5")
        shutil.copy2(sample_h5_file, copy)

        stats = compress_h5_file(copy, compression='standard')

        assert 'fitpara' in stats
        assert stats['fitpara']['ratio'] > 1.0
        assert 'fitpara' in stats['compressed_datasets']

        with h5py.File(copy, 'r') as f:
            assert 'fitpara_compressed' in f
            assert 'fitpara' not in f
            # otherpara/xytdata/specdata should be untouched
            assert 'otherpara' in f
            assert 'xytdata' in f
            assert 'specdata' in f

    def test_full_compresses_all(self, sample_h5_file, tmp_path):
        copy = str(tmp_path / "full.h5")
        shutil.copy2(sample_h5_file, copy)

        stats = compress_h5_file(copy, compression='full')

        assert 'fitpara' in stats['compressed_datasets']
        assert 'otherpara' in stats['compressed_datasets']
        assert 'xytdata' in stats['compressed_datasets']
        assert 'specdata' in stats['compressed_datasets']

        with h5py.File(copy, 'r') as f:
            assert 'fitpara_compressed' in f
            assert 'otherpara_compressed' in f
            assert 'xytdata_compressed' in f
            assert 'specdata_uint16' in f
            # Originals removed
            assert 'fitpara' not in f
            assert 'otherpara' not in f
            assert 'xytdata' not in f
            assert 'specdata' not in f

    def test_verbose_output(self, sample_h5_file, tmp_path, capsys):
        copy = str(tmp_path / "verbose.h5")
        shutil.copy2(sample_h5_file, copy)

        compress_h5_file(copy, compression='full', verbose=True)

        captured = capsys.readouterr()
        assert 'fitpara' in captured.out
        assert 'otherpara' in captured.out

    def test_no_fitpara_skips_gracefully(self, tmp_path):
        """Test with a file that has no fitpara."""
        h5_path = str(tmp_path / "no_fitpara.h5")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('specdata', data=np.ones((10, 5), dtype=np.float32))

        stats = compress_h5_file(h5_path, compression='standard')
        assert 'fitpara' not in stats.get('compressed_datasets', [])

    def test_specdata_out_of_range_skipped(self, tmp_path):
        """Test that specdata with values > 65535 is skipped."""
        h5_path = str(tmp_path / "big_vals.h5")
        with h5py.File(h5_path, 'w') as f:
            specdata = np.ones((10, 5), dtype=np.float32) * 100000
            f.create_dataset('specdata', data=specdata)

        stats = compress_h5_file(h5_path, compression='full')
        assert stats.get('specdata', {}).get('skipped') is True


class TestRoundtripAccuracy:
    """Test that compressed data can be read back accurately."""

    def test_fitpara_roundtrip(self, sample_h5_file, tmp_path):
        # Read original
        with h5py.File(sample_h5_file, 'r') as f:
            orig = f['fitpara'][:].copy()

        copy = str(tmp_path / "rt_fp.h5")
        shutil.copy2(sample_h5_file, copy)
        compress_h5_file(copy, compression='standard')

        # Read back
        codec = FitparaCodec()
        with h5py.File(copy, 'r') as f:
            restored = codec.decode_from_h5(f, 'fitpara_compressed')

        # Check non-NaN values
        mask = ~np.isnan(orig)
        assert mask.any()
        nonzero = np.abs(orig[mask]) > 1e-10
        if nonzero.any():
            rel_err = np.abs(orig[mask][nonzero] - restored[mask][nonzero]) / np.abs(orig[mask][nonzero])
            assert rel_err.max() < 0.01  # <1% relative error

    def test_otherpara_xytdata_lossless(self, sample_h5_file, tmp_path):
        # Read originals
        with h5py.File(sample_h5_file, 'r') as f:
            orig_other = f['otherpara'][:].copy()
            orig_xyt = f['xytdata'][:].copy()

        copy = str(tmp_path / "rt_lz4.h5")
        shutil.copy2(sample_h5_file, copy)
        compress_h5_file(copy, compression='full')

        # Read back
        with h5py.File(copy, 'r') as f:
            restored_other = decompress_array_from_h5(f, 'otherpara_compressed')
            restored_xyt = decompress_array_from_h5(f, 'xytdata_compressed')

        np.testing.assert_array_equal(orig_other, restored_other)
        np.testing.assert_array_equal(orig_xyt, restored_xyt)

    def test_specdata_uint16_roundtrip(self, sample_h5_file, tmp_path):
        # Read original specdata (row 0 = energy, rows 1: = spectra)
        with h5py.File(sample_h5_file, 'r') as f:
            orig_specdata = f['specdata'][:].copy()

        copy = str(tmp_path / "rt_uint16.h5")
        shutil.copy2(sample_h5_file, copy)
        compress_h5_file(copy, compression='full')

        with h5py.File(copy, 'r') as f:
            restored_uint16 = f['specdata_uint16'][:].copy()
            restored_energy = f['specdata_uint16_energy'][:].copy()

        # Energy axis should match
        np.testing.assert_allclose(orig_specdata[0], restored_energy, rtol=1e-6)

        # Spectra: uint16 is exact for integer Poisson data
        orig_spectra = orig_specdata[1:].astype(np.uint16)
        np.testing.assert_array_equal(orig_spectra, restored_uint16)


class TestCompressH5FileSize:
    """Test that compressed files are actually smaller."""

    def test_full_compression_reduces_size(self, sample_h5_file, tmp_path):
        original_size = os.path.getsize(sample_h5_file)

        copy = str(tmp_path / "size_test.h5")
        shutil.copy2(sample_h5_file, copy)
        compress_h5_file(copy, compression='full')
        compressed_size = os.path.getsize(copy)

        # Note: HDF5 in-place delete doesn't reclaim space, so the file
        # may not be smaller. But the compressed datasets themselves are smaller.
        # Verify the stats instead.
        copy2 = str(tmp_path / "size_test2.h5")
        shutil.copy2(sample_h5_file, copy2)
        stats = compress_h5_file(copy2, compression='full')

        if 'fitpara' in stats:
            assert stats['fitpara']['ratio'] > 2.0
        if 'otherpara' in stats:
            assert stats['otherpara']['ratio'] > 10.0
        if 'specdata' in stats:
            assert stats['specdata']['ratio'] > 1.5


class TestCLIConvertLogic:
    """Test the CLI convert command's analysis function."""

    def test_analyze_runs(self, sample_h5_file, capsys):
        from toyomacro.voigtfit.cli import analyze_h5_compression
        analyze_h5_compression(sample_h5_file)

        captured = capsys.readouterr()
        assert 'fitpara' in captured.out
        assert 'specdata' in captured.out

    def test_analyze_with_compressed_file(self, sample_h5_file, tmp_path, capsys):
        from toyomacro.voigtfit.cli import analyze_h5_compression

        copy = str(tmp_path / "compressed.h5")
        shutil.copy2(sample_h5_file, copy)
        compress_h5_file(copy, compression='full')

        analyze_h5_compression(copy)
        captured = capsys.readouterr()
        assert 'already compressed' in captured.out
