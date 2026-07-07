"""Tests for FitDirtyMap — interactive fit result accumulator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from toyomacro.core.fitting_result import (
    BackgroundParams,
    FittingResult,
    LineshapeType,
    PeakComponent,
)
from toyomacro.io.fit_dirty_map import FitDirtyMap


def _make_fitting_result(
    center: float = 99.5,
    area: float = 100.0,
    n_energy: int = 200,
) -> FittingResult:
    """Create a minimal FittingResult for testing."""
    energy = np.linspace(95, 105, n_energy)
    comp = PeakComponent(
        area=area,
        center=center,
        fwhm_g=0.5,
        fwhm_l=0.3,
        lineshape=LineshapeType.VOIGT,
    )
    return FittingResult(
        components=[comp],
        background=BackgroundParams(type="linear"),
        energy=energy,
        fitted_intensity=np.zeros(n_energy),
        background_curve=np.zeros(n_energy),
        residuals=np.zeros(n_energy),
    )


def _make_h5(tmp_dir: Path, n_spectra: int = 100, maxcomp: int = 6) -> str:
    """Create a minimal H5 file with fitpara/otherpara datasets."""
    path = str(tmp_dir / "test_element.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "fitpara",
            shape=(n_spectra, maxcomp, 9),
            dtype=np.float32,
        )
        f.create_dataset(
            "otherpara",
            shape=(10, n_spectra),
            dtype=np.float32,
        )
        # Minimal specdata so HDF5Cache can open it
        f.create_dataset(
            "specdata",
            shape=(n_spectra + 1, 200),
            dtype=np.float32,
        )
    return path


class TestFitDirtyMap:
    """Unit tests for FitDirtyMap."""

    def test_empty(self):
        dm = FitDirtyMap()
        assert dm.dirty_count == 0
        assert not dm.is_dirty()
        assert dm.dirty_files == []

    def test_record_and_count(self):
        dm = FitDirtyMap()
        fr = _make_fitting_result()

        dm.record("/tmp/a.h5", 0, fr)
        assert dm.dirty_count == 1
        assert dm.is_dirty()

        dm.record("/tmp/a.h5", 1, fr)
        assert dm.dirty_count == 2

        dm.record("/tmp/b.h5", 0, fr)
        assert dm.dirty_count == 3
        assert set(dm.dirty_files) == {"/tmp/a.h5", "/tmp/b.h5"}

    def test_overwrite_same_index(self):
        dm = FitDirtyMap()
        fr1 = _make_fitting_result(area=100)
        fr2 = _make_fitting_result(area=200)

        dm.record("/tmp/a.h5", 5, fr1)
        dm.record("/tmp/a.h5", 5, fr2)
        assert dm.dirty_count == 1  # overwrite, not duplicate

    def test_clear_all(self):
        dm = FitDirtyMap()
        fr = _make_fitting_result()
        dm.record("/tmp/a.h5", 0, fr)
        dm.record("/tmp/b.h5", 0, fr)
        dm.clear()
        assert dm.dirty_count == 0

    def test_clear_single_path(self):
        dm = FitDirtyMap()
        fr = _make_fitting_result()
        dm.record("/tmp/a.h5", 0, fr)
        dm.record("/tmp/b.h5", 0, fr)
        dm.clear("/tmp/a.h5")
        assert dm.dirty_count == 1
        assert dm.dirty_files == ["/tmp/b.h5"]

    def test_flush_no_dirty(self):
        dm = FitDirtyMap()
        assert dm.flush() == 0

    def test_flush_writes_fitpara(self, tmp_path):
        path = _make_h5(tmp_path)
        dm = FitDirtyMap()

        fr = _make_fitting_result(center=99.5, area=123.4)
        dm.record(path, 10, fr)
        dm.record(path, 20, fr)

        n = dm.flush()
        assert n == 2
        assert dm.dirty_count == 0  # cleared after flush

        # Verify H5 contents
        with h5py.File(path, "r") as f:
            fp = f["fitpara"]
            # fitpara[10, 0, :] should have the component data
            row = fp[10, 0, :]
            assert row[1] == pytest.approx(99.5, abs=0.01)  # center
            # Verify index 20 was also written
            row20 = fp[20, 0, :]
            assert row20[1] == pytest.approx(99.5, abs=0.01)

            # Unused component slots should be NaN (MATLAB convention)
            assert np.all(np.isnan(fp[10, 1, :]))

    def test_flush_writes_otherpara(self, tmp_path):
        path = _make_h5(tmp_path)
        dm = FitDirtyMap()

        fr = _make_fitting_result()
        dm.record(path, 5, fr)

        dm.flush()

        with h5py.File(path, "r") as f:
            op = f["otherpara"]
            # otherpara[0, 5] = n_components = 1
            assert op[0, 5] == pytest.approx(1.0)

    def test_flush_missing_file(self, tmp_path, capsys):
        dm = FitDirtyMap()
        fr = _make_fitting_result()
        dm.record("/nonexistent/path.h5", 0, fr)

        n = dm.flush()
        assert n == 0  # skipped
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_flush_clears_entries(self, tmp_path):
        path = _make_h5(tmp_path)
        dm = FitDirtyMap()
        fr = _make_fitting_result()
        dm.record(path, 0, fr)

        dm.flush()
        assert not dm.is_dirty()
        # Second flush should be no-op
        assert dm.flush() == 0

    def test_flush_pads_components(self, tmp_path):
        """Verify fitpara is zero-padded to maxcomp."""
        path = _make_h5(tmp_path, maxcomp=4)
        dm = FitDirtyMap()
        fr = _make_fitting_result()  # 1 component

        dm.record(path, 0, fr)
        dm.flush()

        with h5py.File(path, "r") as f:
            fp = f["fitpara"]
            # Component 0 should have data, 1-3 should be NaN (MATLAB convention)
            assert fp[0, 0, 1] != 0  # center
            assert np.all(np.isnan(fp[0, 1, :]))
            assert np.all(np.isnan(fp[0, 2, :]))
            assert np.all(np.isnan(fp[0, 3, :]))

    def test_flush_auto_creates_fitpara(self, tmp_path):
        """Verify fitpara/otherpara are auto-created when missing."""
        # Create H5 with only specdata (no fitpara/otherpara)
        path = str(tmp_path / "no_fitpara.h5")
        n_spectra = 50
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "specdata",
                shape=(n_spectra + 1, 200),
                dtype=np.float32,
            )

        dm = FitDirtyMap()
        fr = _make_fitting_result(center=100.0, area=42.0)
        dm.record(path, 3, fr)

        n = dm.flush()
        assert n == 1

        with h5py.File(path, "r") as f:
            assert "fitpara" in f
            assert "otherpara" in f
            assert f["fitpara"].shape == (n_spectra, 6, 9)
            assert f["otherpara"].shape == (10, n_spectra)
            # Verify data was written
            assert f["fitpara"][3, 0, 1] == pytest.approx(100.0, abs=0.01)
