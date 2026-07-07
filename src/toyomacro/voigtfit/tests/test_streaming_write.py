"""
Tests for streaming fitpara write functionality.

Tests:
1. StreamingFitparaWriter basic functionality
2. FitResultSummary and lazy loading
3. Validation and recovery helpers
4. Integration with process_xps_file_streaming
"""

import os
import tempfile

import h5py
import numpy as np
import pytest

from toyomacro.voigtfit.h5io import (
    StreamingFitparaWriter,
    StreamingWriterConfig,
    clear_incomplete_marker,
    read_fitpara_as_fitresult,
    read_fitpara_by_indices,
    read_fitpara_downsampled,
    read_fitpara_slice,
    recover_partial_fit,
    validate_fitpara_file,
)


class MockFitResult:
    """Mock FitResult for testing."""
    def __init__(self, n_components: int, n_spectra: int):
        self.amplitudes = np.random.rand(n_components, n_spectra).astype(np.float32)
        self.chi2 = np.random.rand(n_spectra).astype(np.float32) * 0.1
        self.anomaly_mask = np.random.rand(n_spectra) > 0.95  # ~5% anomalies
        self.centers = None
        self.sigmas = None
        self.gammas = None


@pytest.fixture
def temp_h5_file():
    """Create a temporary HDF5 file for testing."""
    fd, path = tempfile.mkstemp(suffix='.h5')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_output_file():
    """Create a temporary output file path."""
    fd, path = tempfile.mkstemp(suffix='_output.h5')
    os.close(fd)
    os.unlink(path)  # Remove so writer can create it
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_input_file(temp_h5_file):
    """Create a sample input HDF5 file with specdata."""
    n_spectra = 1000
    n_energy = 100

    with h5py.File(temp_h5_file, 'w') as f:
        # Create specdata (row 0 = energy, rest = spectra)
        specdata = np.random.rand(n_spectra + 1, n_energy).astype(np.float32)
        specdata[0, :] = np.linspace(280, 290, n_energy)  # Energy axis
        f.create_dataset('specdata', data=specdata)

        # Create empty fitpara
        f.create_dataset('fitpara', shape=(n_spectra, 3, 9), dtype=np.float32)
        f['fitpara'][:] = np.nan

        # Create xytdata
        xytdata = np.random.rand(3, n_spectra).astype(np.float32)
        f.create_dataset('xytdata', data=xytdata)

    return temp_h5_file


class TestStreamingFitparaWriter:
    """Tests for StreamingFitparaWriter class."""

    def test_basic_write(self, temp_output_file):
        """Test basic batch writing."""
        n_spectra = 100
        n_components = 3
        batch_size = 25

        config = StreamingWriterConfig(
            output_path=temp_output_file,
            write_mode='new_file',
            n_spectra=n_spectra,
            n_components=n_components,
        )

        peak_config = {
            'centers': np.array([284.5, 286.0, 288.0]),
            'sigmas': np.array([0.8, 0.9, 0.7]),
            'gamma': 0.25,
        }

        with StreamingFitparaWriter(config) as writer:
            for batch_idx in range(n_spectra // batch_size):
                start = batch_idx * batch_size
                result = MockFitResult(n_components, batch_size)
                writer.write_batch(batch_idx, start, result, peak_config)

            summary = writer.get_summary()

        # Verify summary
        assert summary.n_spectra == n_spectra
        assert summary.n_components == n_components
        assert summary.chi2_mean > 0
        assert summary.output_path == temp_output_file

        # Verify file
        with h5py.File(temp_output_file, 'r') as f:
            assert 'fitpara' in f
            assert f['fitpara'].shape == (n_spectra, n_components, 9)
            # Check incomplete marker removed
            assert '_voigtfit_incomplete' not in f.attrs

    def test_write_with_metadata_copy(self, sample_input_file, temp_output_file):
        """Test writing with metadata copied from input file."""
        n_spectra = 1000
        n_components = 3

        config = StreamingWriterConfig(
            output_path=temp_output_file,
            write_mode='new_file',
            n_spectra=n_spectra,
            n_components=n_components,
            copy_metadata=True,
        )

        peak_config = {
            'centers': np.array([284.5, 286.0, 288.0]),
            'sigmas': np.array([0.8, 0.9, 0.7]),
            'gamma': 0.25,
        }

        with StreamingFitparaWriter(config, input_path=sample_input_file) as writer:
            result = MockFitResult(n_components, n_spectra)
            writer.write_batch(0, 0, result, peak_config)

        # Verify metadata was copied
        with h5py.File(temp_output_file, 'r') as f:
            assert 'specdata' in f
            assert 'xytdata' in f
            assert f['specdata'].shape[0] == n_spectra + 1

    def test_anomaly_tracking(self, temp_output_file):
        """Test anomaly tracking in summary."""
        n_spectra = 100
        n_components = 2

        config = StreamingWriterConfig(
            output_path=temp_output_file,
            write_mode='new_file',
            n_spectra=n_spectra,
            n_components=n_components,
        )

        peak_config = {'centers': np.array([285.0, 287.0]), 'sigmas': np.array([0.8, 0.8]), 'gamma': 0.25}

        with StreamingFitparaWriter(config) as writer:
            result = MockFitResult(n_components, n_spectra)
            # Force some anomalies
            result.anomaly_mask = np.zeros(n_spectra, dtype=bool)
            result.anomaly_mask[10] = True
            result.anomaly_mask[50] = True
            result.anomaly_mask[90] = True

            writer.write_batch(0, 0, result, peak_config, anomaly_mask=result.anomaly_mask)
            summary = writer.get_summary()

        assert summary.anomaly_count == 3
        assert summary.anomaly_rate == 0.03
        assert len(summary.anomaly_indices) == 3
        assert set(summary.anomaly_indices) == {10, 50, 90}


class TestFitResultSummary:
    """Tests for FitResultSummary lazy loading."""

    def test_load_slice(self, temp_output_file):
        """Test loading a slice of data."""
        n_spectra = 100
        n_components = 2

        # Create test file
        config = StreamingWriterConfig(
            output_path=temp_output_file,
            write_mode='new_file',
            n_spectra=n_spectra,
            n_components=n_components,
        )

        peak_config = {'centers': np.array([285.0, 287.0]), 'sigmas': np.array([0.8, 0.8]), 'gamma': 0.25}

        with StreamingFitparaWriter(config) as writer:
            result = MockFitResult(n_components, n_spectra)
            writer.write_batch(0, 0, result, peak_config)
            summary = writer.get_summary()

        # Test slice loading
        slice_result = summary.load_slice(10, 30)
        assert slice_result.amplitudes.shape == (n_components, 20)
        assert len(slice_result.chi2) == 20

    def test_load_downsampled(self, temp_output_file):
        """Test loading downsampled data."""
        n_spectra = 1000
        n_components = 2

        config = StreamingWriterConfig(
            output_path=temp_output_file,
            write_mode='new_file',
            n_spectra=n_spectra,
            n_components=n_components,
        )

        peak_config = {'centers': np.array([285.0, 287.0]), 'sigmas': np.array([0.8, 0.8]), 'gamma': 0.25}

        with StreamingFitparaWriter(config) as writer:
            result = MockFitResult(n_components, n_spectra)
            writer.write_batch(0, 0, result, peak_config)
            summary = writer.get_summary()

        # Test downsampled loading (every 10th)
        ds_result = summary.load_downsampled(step=10)
        assert ds_result.amplitudes.shape == (n_components, 100)


class TestValidationAndRecovery:
    """Tests for validation and recovery helpers."""

    def test_validate_complete_file(self, temp_output_file):
        """Test validation of a complete file."""
        n_spectra = 50
        n_components = 2

        config = StreamingWriterConfig(
            output_path=temp_output_file,
            write_mode='new_file',
            n_spectra=n_spectra,
            n_components=n_components,
        )

        peak_config = {'centers': np.array([285.0, 287.0]), 'sigmas': np.array([0.8, 0.8]), 'gamma': 0.25}

        with StreamingFitparaWriter(config) as writer:
            result = MockFitResult(n_components, n_spectra)
            writer.write_batch(0, 0, result, peak_config)

        validation = validate_fitpara_file(temp_output_file)
        assert validation['valid'] is True
        assert validation['incomplete'] is False
        assert validation['n_spectra'] == n_spectra

    def test_validate_incomplete_file(self, temp_output_file):
        """Test validation of an incomplete file."""
        n_spectra = 100
        n_components = 2

        config = StreamingWriterConfig(
            output_path=temp_output_file,
            write_mode='new_file',
            n_spectra=n_spectra,
            n_components=n_components,
        )

        peak_config = {'centers': np.array([285.0, 287.0]), 'sigmas': np.array([0.8, 0.8]), 'gamma': 0.25}

        # Simulate crash by not using context manager properly
        writer = StreamingFitparaWriter(config)
        writer._setup_output_file()
        result = MockFitResult(n_components, 50)
        writer.write_batch(0, 0, result, peak_config)
        # Don't call _finalize - simulate crash
        writer._h5_out.close()

        validation = validate_fitpara_file(temp_output_file)
        assert validation['valid'] is False
        assert validation['incomplete'] is True

    def test_recover_partial_fit(self, temp_output_file):
        """Test recovery information for partial file."""
        n_spectra = 100
        n_components = 2

        config = StreamingWriterConfig(
            output_path=temp_output_file,
            write_mode='new_file',
            n_spectra=n_spectra,
            n_components=n_components,
        )

        peak_config = {'centers': np.array([285.0, 287.0]), 'sigmas': np.array([0.8, 0.8]), 'gamma': 0.25}

        # Simulate partial write
        writer = StreamingFitparaWriter(config)
        writer._setup_output_file()
        result = MockFitResult(n_components, 50)
        writer.write_batch(0, 0, result, peak_config)
        writer._h5_out.close()

        recovery = recover_partial_fit(temp_output_file)
        assert recovery['can_resume'] is True
        assert recovery['resume_from_idx'] == 50
        assert recovery['n_spectra'] == n_spectra

    def test_clear_incomplete_marker(self, temp_output_file):
        """Test clearing incomplete marker."""
        n_spectra = 50
        n_components = 2

        config = StreamingWriterConfig(
            output_path=temp_output_file,
            write_mode='new_file',
            n_spectra=n_spectra,
            n_components=n_components,
        )

        # Simulate incomplete file
        writer = StreamingFitparaWriter(config)
        writer._setup_output_file()
        writer._h5_out.close()

        # Verify incomplete
        validation = validate_fitpara_file(temp_output_file)
        assert validation['incomplete'] is True

        # Clear marker
        result = clear_incomplete_marker(temp_output_file)
        assert result is True

        # Verify cleared
        validation = validate_fitpara_file(temp_output_file)
        assert validation['incomplete'] is False


class TestReadFunctions:
    """Tests for read helper functions."""

    @pytest.fixture
    def populated_file(self, temp_output_file):
        """Create a populated fitpara file."""
        n_spectra = 100
        n_components = 3

        config = StreamingWriterConfig(
            output_path=temp_output_file,
            write_mode='new_file',
            n_spectra=n_spectra,
            n_components=n_components,
        )

        peak_config = {
            'centers': np.array([284.5, 286.0, 288.0]),
            'sigmas': np.array([0.8, 0.9, 0.7]),
            'gamma': 0.25,
        }

        with StreamingFitparaWriter(config) as writer:
            result = MockFitResult(n_components, n_spectra)
            writer.write_batch(0, 0, result, peak_config)

        return temp_output_file

    def test_read_fitpara_as_fitresult(self, populated_file):
        """Test reading entire fitpara as FitResult."""
        result = read_fitpara_as_fitresult(populated_file)
        assert result.amplitudes.shape == (3, 100)
        assert len(result.chi2) == 100

    def test_read_fitpara_slice(self, populated_file):
        """Test reading slice of fitpara."""
        result = read_fitpara_slice(populated_file, 20, 40)
        assert result.amplitudes.shape == (3, 20)
        assert len(result.chi2) == 20

    def test_read_fitpara_downsampled(self, populated_file):
        """Test reading downsampled fitpara."""
        result = read_fitpara_downsampled(populated_file, step=5)
        assert result.amplitudes.shape == (3, 20)

    def test_read_fitpara_by_indices(self, populated_file):
        """Test reading specific indices."""
        indices = np.array([5, 10, 15, 90])
        result = read_fitpara_by_indices(populated_file, indices)
        assert result.amplitudes.shape == (3, 4)


class TestIntegration:
    """Integration tests with DepthProfilerBridge."""

    @pytest.fixture
    def large_input_file(self, temp_h5_file):
        """Create a larger input file for integration testing."""
        n_spectra = 5000
        n_energy = 100

        with h5py.File(temp_h5_file, 'w') as f:
            # Create specdata (row 0 = energy, rest = spectra)
            energy = np.linspace(280, 295, n_energy).astype(np.float32)
            spectra = np.random.rand(n_spectra, n_energy).astype(np.float32) * 100

            specdata = np.vstack([energy, spectra])
            f.create_dataset('specdata', data=specdata)

            # Create fitpara with some initial values
            fitpara = np.full((n_spectra, 3, 9), np.nan, dtype=np.float32)
            fitpara[:, 0, 1] = 284.5  # center 1
            fitpara[:, 1, 1] = 286.0  # center 2
            fitpara[:, 2, 1] = 288.5  # center 3
            f.create_dataset('fitpara', data=fitpara)

        return temp_h5_file

    def test_process_streaming_creates_summary(self, large_input_file, temp_output_file):
        """Test that streaming mode returns FitResultSummary."""
        # This test requires the full pipeline, which may not be available
        # in all environments. Skip if dependencies are missing.
        pytest.importorskip('voigtfit.pipeline')

        try:
            from toyomacro.voigtfit.h5io import FitResultSummary
            from toyomacro.voigtfit.integration import DepthProfilerBridge

            bridge = DepthProfilerBridge(use_mlx=False, enable_stage2=False)

            summary, metadata = bridge.process_xps_file_streaming(
                large_input_file,
                centers=np.array([284.5, 286.0, 288.5]),
                sigmas=np.array([0.8, 0.9, 0.7]),
                gamma=0.25,
                batch_size=1000,
                output_path=temp_output_file,
                write_mode='new_file',
            )

            # Verify return type
            assert isinstance(summary, FitResultSummary)
            assert summary.n_spectra == 5000
            assert summary.n_components == 3
            assert summary.output_path == temp_output_file

            # Verify file was created
            validation = validate_fitpara_file(temp_output_file)
            assert validation['valid'] is True
            assert validation['n_spectra'] == 5000

            # Test lazy loading
            slice_result = summary.load_slice(0, 100)
            assert slice_result.amplitudes.shape == (3, 100)

        except ImportError as e:
            pytest.skip(f"Pipeline dependencies not available: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
