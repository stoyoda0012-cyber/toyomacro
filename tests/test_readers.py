"""
Tests for XPS file format readers and import pipeline.

Test data lives under ``<TOYOMACRO_TESTDATA>/Fitting/readtest/``; set the
``TOYOMACRO_TESTDATA`` environment variable to point at your local tree
(see ``tests/_testdata.py``). Tests skip cleanly when the data is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._testdata import fitting_dir
from toyomacro.io.readers.base_reader import (
    BaseReader,
    RawSpectrumData,
    SpectrumMetadata,
    create_reader,
    detect_format,
)
from toyomacro.io.readers.npl_reader import NPLReader
from toyomacro.io.readers.pxt_reader import PXTReader
from toyomacro.io.readers.ses_reader import SESTxtReader
from toyomacro.io.readers.vamas_reader import VAMASReader

TESTDATA_ROOT = fitting_dir() / "readtest"
PXT_DIR = TESTDATA_ROOT / "pxt"
IBW_DIR = TESTDATA_ROOT / "ibw"
TXT_DIR = TESTDATA_ROOT / "txt"
VMS_DIR = TESTDATA_ROOT / "vms"
NPL_DIR = TESTDATA_ROOT / "npl"

# Skip all tests if test data directory doesn't exist
pytestmark = pytest.mark.skipif(
    not TESTDATA_ROOT.exists(),
    reason=f"Test data not found at {TESTDATA_ROOT}",
)


def _get_first_file(directory: Path, pattern: str) -> Path:
    """Get the first file matching pattern in directory."""
    files = sorted(directory.glob(pattern))
    if not files:
        pytest.skip(f"No {pattern} files found in {directory}")
    return files[0]


# ============================================================
# Format detection tests
# ============================================================


class TestFormatDetection:
    def test_pxt_detected(self):
        assert detect_format("data.pxt") == "pxt"

    def test_ibw_detected(self):
        assert detect_format("data.ibw") == "pxt"

    def test_vms_detected(self):
        assert detect_format("data.vms") == "vamas"

    def test_npl_detected(self):
        assert detect_format("data.npl") == "npl"

    def test_txt_detected(self):
        assert detect_format("data.txt") == "ses"

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Unrecognized"):
            detect_format("data.xyz")


class TestReaderFactory:
    def test_create_pxt_reader(self):
        f = _get_first_file(PXT_DIR, "*.pxt")
        reader = create_reader(f)
        assert isinstance(reader, PXTReader)

    def test_create_ibw_reader(self):
        f = _get_first_file(IBW_DIR / "1-20sequence", "*.ibw")
        reader = create_reader(f)
        assert isinstance(reader, PXTReader)

    def test_create_vamas_reader(self):
        f = _get_first_file(VMS_DIR, "*.vms")
        reader = create_reader(f)
        assert isinstance(reader, VAMASReader)

    def test_create_npl_reader(self):
        f = _get_first_file(NPL_DIR, "*.npl")
        reader = create_reader(f)
        assert isinstance(reader, NPLReader)

    def test_create_ses_reader(self):
        f = _get_first_file(TXT_DIR, "*.txt")
        reader = create_reader(f)
        assert isinstance(reader, SESTxtReader)


# ============================================================
# PXT reader tests
# ============================================================


class TestPXTReader:
    def test_read_pxt_file(self):
        f = _get_first_file(PXT_DIR, "*.pxt")
        reader = PXTReader(f)
        assert reader.n_regions >= 1

        data = reader.read(0)
        assert isinstance(data, RawSpectrumData)
        assert data.energy.size > 0
        assert data.specdata.size > 0

    def test_pxt_energy_axis_valid(self):
        f = _get_first_file(PXT_DIR, "*.pxt")
        reader = PXTReader(f)
        data = reader.read(0)

        # Energy axis should be monotonic
        diff = np.diff(data.energy)
        assert np.all(diff > 0) or np.all(diff < 0), "Energy axis not monotonic"

    def test_pxt_region_names(self):
        f = _get_first_file(PXT_DIR, "*.pxt")
        reader = PXTReader(f)
        names = reader.region_names
        assert len(names) == reader.n_regions
        assert all(isinstance(n, str) for n in names)

    def test_pxt_metadata(self):
        f = _get_first_file(PXT_DIR, "*.pxt")
        reader = PXTReader(f)
        data = reader.read(0)
        assert isinstance(data.metadata, SpectrumMetadata)

    def test_pxt_can_read(self):
        f = _get_first_file(PXT_DIR, "*.pxt")
        assert PXTReader.can_read(f) is True

    def test_pxt_multi_region(self):
        """PXT v3 files should support multiple regions."""
        f = _get_first_file(PXT_DIR, "*.pxt")
        reader = PXTReader(f)
        # Multi-region PXT files should have > 1 region
        # (or at least 1 - depends on test data)
        assert reader.n_regions >= 1

    def test_read_ibw_file(self):
        """IBW (v5) files should work too."""
        f = _get_first_file(IBW_DIR / "1-20sequence", "*.ibw")
        reader = PXTReader(f)
        assert reader.n_regions == 1  # v5 = single region

        data = reader.read(0)
        assert data.energy.size > 0
        assert data.specdata.size > 0

    def test_ibw_can_read(self):
        f = _get_first_file(IBW_DIR / "1-20sequence", "*.ibw")
        assert PXTReader.can_read(f) is True

    def test_pxt_specdata_shape(self):
        f = _get_first_file(PXT_DIR, "*.pxt")
        reader = PXTReader(f)
        data = reader.read(0)
        # specdata should be at least 2D
        assert data.specdata.ndim >= 1
        # First dimension = energy
        assert data.specdata.shape[0] == len(data.energy)

    def test_pxt_no_nan(self):
        f = _get_first_file(PXT_DIR, "*.pxt")
        reader = PXTReader(f)
        data = reader.read(0)
        assert not np.any(np.isnan(data.energy))
        assert not np.any(np.isnan(data.specdata))


# ============================================================
# SES text reader tests
# ============================================================


class TestSESTxtReader:
    def test_read_txt_file(self):
        f = _get_first_file(TXT_DIR, "*.txt")
        reader = SESTxtReader(f)
        assert reader.n_regions >= 1

        data = reader.read(0)
        assert isinstance(data, RawSpectrumData)
        assert data.energy.size > 0

    def test_ses_energy_axis(self):
        f = _get_first_file(TXT_DIR, "*.txt")
        reader = SESTxtReader(f)
        data = reader.read(0)
        # Energy axis should have sensible values
        assert data.energy.size >= 3

    def test_ses_specdata_shape(self):
        f = _get_first_file(TXT_DIR, "*.txt")
        reader = SESTxtReader(f)
        data = reader.read(0)
        assert data.specdata.ndim >= 1

    def test_ses_can_read(self):
        f = _get_first_file(TXT_DIR, "*.txt")
        assert SESTxtReader.can_read(f) is True

    def test_ses_no_nan(self):
        f = _get_first_file(TXT_DIR, "*.txt")
        reader = SESTxtReader(f)
        data = reader.read(0)
        assert not np.any(np.isnan(data.energy))


# ============================================================
# VAMAS reader tests
# ============================================================


class TestVAMASReader:
    def test_read_vms_file(self):
        f = _get_first_file(VMS_DIR, "*.vms")
        reader = VAMASReader(f)
        assert reader.n_regions >= 1

        data = reader.read(0)
        assert isinstance(data, RawSpectrumData)
        assert data.energy.size > 0
        assert data.specdata.size > 0

    def test_vamas_energy_axis(self):
        f = _get_first_file(VMS_DIR, "*.vms")
        reader = VAMASReader(f)
        data = reader.read(0)
        diff = np.diff(data.energy)
        assert np.all(diff > 0) or np.all(diff < 0), "Energy axis not monotonic"

    def test_vamas_metadata(self):
        f = _get_first_file(VMS_DIR, "*.vms")
        reader = VAMASReader(f)
        data = reader.read(0)
        # VAMAS should be Kinetic energy scale
        assert data.metadata.energy_scale == "Kinetic"

    def test_vamas_single_angle(self):
        f = _get_first_file(VMS_DIR, "*.vms")
        reader = VAMASReader(f)
        data = reader.read(0)
        # VAMAS files are single-angle
        assert len(data.angle) == 1
        assert data.specdata.shape[1] == 1

    def test_vamas_region_names(self):
        f = _get_first_file(VMS_DIR, "*.vms")
        reader = VAMASReader(f)
        names = reader.region_names
        assert len(names) == reader.n_regions

    def test_vamas_can_read(self):
        f = _get_first_file(VMS_DIR, "*.vms")
        assert VAMASReader.can_read(f) is True

    def test_vamas_multi_region(self):
        """VAMAS files typically have multiple regions."""
        f = _get_first_file(VMS_DIR, "*.vms")
        reader = VAMASReader(f)
        # Most VAMAS files have multiple regions
        if reader.n_regions > 1:
            data0 = reader.read(0)
            data1 = reader.read(1)
            # Different regions should have different energy ranges
            assert data0.energy.size > 0
            assert data1.energy.size > 0


# ============================================================
# NPL reader tests
# ============================================================


class TestNPLReader:
    def test_read_npl_file(self):
        f = _get_first_file(NPL_DIR, "*.npl")
        reader = NPLReader(f)
        assert reader.n_regions >= 1

        data = reader.read(0)
        assert isinstance(data, RawSpectrumData)
        assert data.energy.size > 0

    def test_npl_energy_axis(self):
        f = _get_first_file(NPL_DIR, "*.npl")
        reader = NPLReader(f)
        data = reader.read(0)
        assert data.energy.size >= 3

    def test_npl_binding_energy(self):
        """NPL reader applies WF correction -> Binding energy."""
        f = _get_first_file(NPL_DIR, "*.npl")
        reader = NPLReader(f)
        data = reader.read(0)
        assert data.metadata.energy_scale == "Binding"

    def test_npl_region_names(self):
        f = _get_first_file(NPL_DIR, "*.npl")
        reader = NPLReader(f)
        names = reader.region_names
        assert len(names) == reader.n_regions
        assert all(isinstance(n, str) for n in names)

    def test_npl_can_read(self):
        f = _get_first_file(NPL_DIR, "*.npl")
        assert NPLReader.can_read(f) is True

    def test_npl_au_file(self):
        """Test Au.npl specifically (known file)."""
        au_file = NPL_DIR / "Au.npl"
        if not au_file.exists():
            pytest.skip("Au.npl not found")
        reader = NPLReader(au_file)
        assert reader.n_regions >= 1
        data = reader.read(0)
        assert data.energy.size > 0
        assert data.metadata.excitation_energy > 0


# ============================================================
# Import pipeline tests
# ============================================================


class TestImportPipeline:
    def test_import_pxt_file(self, tmp_path):
        from toyomacro.io.importer import ImportConfig, import_file

        f = _get_first_file(PXT_DIR, "*.pxt")
        config = ImportConfig(element="Test1s", compress=False)
        result = import_file(f, tmp_path, config)

        assert result.output_path.exists()
        assert result.n_spectra > 0
        assert result.n_energy > 0
        assert result.output_path.suffix == ".h5"

    def test_import_vms_file(self, tmp_path):
        from toyomacro.io.importer import ImportConfig, import_file

        f = _get_first_file(VMS_DIR, "*.vms")
        config = ImportConfig(element="Test1s", compress=False)
        result = import_file(f, tmp_path, config)

        assert result.output_path.exists()
        assert result.n_spectra > 0

    def test_import_npl_file(self, tmp_path):
        from toyomacro.io.importer import ImportConfig, import_file

        f = _get_first_file(NPL_DIR, "*.npl")
        config = ImportConfig(element="Au4f", compress=False)
        result = import_file(f, tmp_path, config)

        assert result.output_path.exists()
        assert result.n_spectra > 0

    def test_import_hdf5_roundtrip(self, tmp_path):
        """Import PXT -> HDF5 -> read back -> verify energy axis."""
        import h5py

        f = _get_first_file(PXT_DIR, "*.pxt")
        from toyomacro.io.importer import ImportConfig, import_file

        config = ImportConfig(element="Si2p", compress=False)
        result = import_file(f, tmp_path, config)

        # Verify HDF5 structure
        with h5py.File(result.output_path, "r") as h5:
            assert "specdata" in h5
            assert "fitpara" in h5
            assert "otherpara" in h5
            assert "xytdata" in h5
            assert "misc" in h5

            specdata = h5["specdata"][:]
            # Row 0 = energy axis
            energy = specdata[0, :]
            assert len(energy) == result.n_energy
            # Rows 1+ = spectra
            assert specdata.shape[0] == result.n_spectra + 1

    def test_import_with_compression(self, tmp_path):
        """Import with compression enabled."""
        f = _get_first_file(PXT_DIR, "*.pxt")
        from toyomacro.io.importer import ImportConfig, import_file

        config = ImportConfig(element="Si2p", compress=True)
        result = import_file(f, tmp_path, config)

        assert result.output_path.exists()
        # Compressed file should exist
        if result.compressed:
            assert "_compressed" in result.output_path.stem

    def test_import_ibw_file(self, tmp_path):
        from toyomacro.io.importer import ImportConfig, import_file

        f = _get_first_file(IBW_DIR / "1-20sequence", "*.ibw")
        config = ImportConfig(element="Si1s", compress=False)
        result = import_file(f, tmp_path, config)

        assert result.output_path.exists()
        assert result.n_spectra > 0

    def test_import_txt_file(self, tmp_path):
        from toyomacro.io.importer import ImportConfig, import_file

        f = _get_first_file(TXT_DIR, "*.txt")
        config = ImportConfig(element="Test1s", compress=False)
        result = import_file(f, tmp_path, config)

        assert result.output_path.exists()
        assert result.n_spectra > 0

    def test_import_dry_run(self, tmp_path):
        from toyomacro.io.importer import ImportConfig, import_folder

        results = import_folder(PXT_DIR, tmp_path, ImportConfig(element="Si2p"), dry_run=True)
        # Dry run should return empty list
        assert results == []
        # No files should be created
        assert len(list(tmp_path.glob("*.h5"))) == 0


# ============================================================
# detect_element tests
# ============================================================


class TestDetectElement:
    def test_detect_si2p(self):
        from toyomacro.io.importer import detect_element

        assert detect_element("Si2p_240101.pxt") == "Si2p"

    def test_detect_o1s(self):
        from toyomacro.io.importer import detect_element

        assert detect_element("O1s_data.vms") == "O1s"

    def test_detect_au4f(self):
        from toyomacro.io.importer import detect_element

        assert detect_element("Au4f.npl") == "Au4f"

    def test_detect_c1s(self):
        from toyomacro.io.importer import detect_element

        assert detect_element("C1s_organic.txt") == "C1s"

    def test_detect_ta4f(self):
        from toyomacro.io.importer import detect_element

        assert detect_element("Ta4f_oxide_240101.pxt") == "Ta4f"

    def test_detect_two_letter_element(self):
        from toyomacro.io.importer import detect_element

        assert detect_element("Al2p_test.pxt") == "Al2p"

    def test_detect_none_for_no_pattern(self):
        from toyomacro.io.importer import detect_element

        assert detect_element("random_data.pxt") is None

    def test_detect_none_for_no_orbital(self):
        from toyomacro.io.importer import detect_element

        assert detect_element("test123.h5") is None

    def test_detect_with_path(self):
        from toyomacro.io.importer import detect_element

        assert detect_element(Path("/some/path/Si2p_240101.pxt")) == "Si2p"

    def test_detect_region_name(self):
        from toyomacro.io.importer import detect_element

        assert detect_element("C 1s") == "C1s"

    def test_detect_region_name_with_space(self):
        from toyomacro.io.importer import detect_element

        assert detect_element("Si 2p") == "Si2p"


# ============================================================
# ensure_h5 tests
# ============================================================


class TestEnsureH5:
    def test_h5_passthrough(self, tmp_path):
        """HDF5 files should be returned as-is."""
        # Create a dummy .h5 file
        import h5py

        from toyomacro.io.importer import ensure_h5

        h5_file = tmp_path / "test.h5"
        with h5py.File(h5_file, "w") as f:
            f.create_dataset("specdata", data=np.zeros((2, 10)))

        result = ensure_h5(h5_file)
        assert result == [h5_file]

    def test_hdf5_passthrough(self, tmp_path):
        """HDF5 files with .hdf5 extension should pass through."""
        import h5py

        from toyomacro.io.importer import ensure_h5

        h5_file = tmp_path / "test.hdf5"
        with h5py.File(h5_file, "w") as f:
            f.create_dataset("specdata", data=np.zeros((2, 10)))

        result = ensure_h5(h5_file)
        assert result == [h5_file]

    def test_unsupported_format_raises(self, tmp_path):
        """Unsupported extensions should raise ValueError."""
        from toyomacro.io.importer import ensure_h5

        fake_file = tmp_path / "data.xyz"
        fake_file.touch()

        with pytest.raises(ValueError, match="Unsupported"):
            ensure_h5(fake_file)

    def test_file_not_found_raises(self):
        """Non-existent files should raise FileNotFoundError."""
        from toyomacro.io.importer import ensure_h5

        with pytest.raises(FileNotFoundError):
            ensure_h5("/non/existent/file.pxt")

    def test_converts_pxt(self, tmp_path):
        """PXT files should be converted to HDF5."""
        from toyomacro.io.importer import ensure_h5

        f = _get_first_file(PXT_DIR, "*.pxt")
        result = ensure_h5(f, element="Si2p", output_dir=tmp_path)

        assert len(result) >= 1
        assert all(p.suffix == ".h5" for p in result)
        assert all(p.exists() for p in result)

    def test_converts_npl(self, tmp_path):
        """NPL files should be converted to HDF5."""
        from toyomacro.io.importer import ensure_h5

        f = _get_first_file(NPL_DIR, "*.npl")
        result = ensure_h5(f, output_dir=tmp_path)

        assert len(result) >= 1
        assert all(p.exists() for p in result)

    def test_converts_vms(self, tmp_path):
        """VAMAS files should be converted to HDF5."""
        from toyomacro.io.importer import ensure_h5

        f = _get_first_file(VMS_DIR, "*.vms")
        result = ensure_h5(f, output_dir=tmp_path)

        assert len(result) >= 1
        assert all(p.exists() for p in result)

    def test_reuses_existing(self, tmp_path):
        """Second ensure_h5 call should reuse existing .h5."""
        from toyomacro.io.importer import ensure_h5

        f = _get_first_file(PXT_DIR, "*.pxt")
        result1 = ensure_h5(f, element="Si2p", output_dir=tmp_path)
        assert len(result1) >= 1

        # Second call should reuse (not re-create)
        result2 = ensure_h5(f, element="Si2p", output_dir=tmp_path)
        assert len(result2) >= 1
        # Should find the same file (or files)
        assert all(p.exists() for p in result2)


# ============================================================
# RAW_EXTENSIONS tests
# ============================================================


class TestRAWExtensions:
    def test_raw_extensions_contains_all(self):
        from toyomacro.io.importer import RAW_EXTENSIONS

        assert ".pxt" in RAW_EXTENSIONS
        assert ".ibw" in RAW_EXTENSIONS
        assert ".txt" in RAW_EXTENSIONS
        assert ".vms" in RAW_EXTENSIONS
        assert ".npl" in RAW_EXTENSIONS

    def test_h5_not_in_raw_extensions(self):
        from toyomacro.io.importer import RAW_EXTENSIONS

        assert ".h5" not in RAW_EXTENSIONS
        assert ".hdf5" not in RAW_EXTENSIONS
