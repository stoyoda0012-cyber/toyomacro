"""Reader/importer tests on synthetic, redistributable fixtures.

Unlike ``test_readers.py`` (which exercises private measured data and
skips when ``TOYOMACRO_TESTDATA`` is absent), these tests build minimal
but structurally valid PXT/IBW/VAMAS/NPL/SES files on the fly, so they
run on any machine and in public CI.

Covers the Phase A I/O enrichment contract:
- unknown excitation/pass energy is None (never 0)
- source format / version / region index provenance
- vendor metadata preservation with defensive copies
- intensity semantics stay "unknown" unless the file confirms them
- original shape + dimension roles (no angle guessing)
- transform history (BE/KE conversion, axis reversal, flatten, sweeps, dtype)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tests._synthetic_files import (
    make_notes,
    write_ibw,
    write_npl,
    write_pxt,
    write_ses_txt,
    write_vamas,
)
from toyomacro.io.readers.base_reader import (
    RawSpectrumData,
    ReaderTransform,
    ReaderWarning,
    SpectrumMetadata,
)
from toyomacro.io.readers.npl_reader import NPLReader
from toyomacro.io.readers.pxt_reader import PXTReader
from toyomacro.io.readers.ses_reader import SESTxtReader
from toyomacro.io.readers.vamas_reader import VAMASReader

FULL_NOTES = {
    "Region Name": "Si2p",
    "Excitation Energy": "1486.6",
    "Energy Scale": "Kinetic",
    "Lens Mode": "Angular56",
    "Pass Energy": 50,
    "Number of Sweeps": 4,
    "Number of Slices": 1,
    "Date": "2026-01-15",
    "Time": "10:30:00",
    # Not adopted into standard fields -> must survive as vendor metadata
    "Detector Mode": "ADC",
    "Sample": "SYNTH-01",
    "Instrument": "SES-2002",
}


def _spectrum_1d(n: int = 32) -> np.ndarray:
    return 100.0 + 10.0 * np.exp(-0.5 * ((np.arange(n) - n / 2) / 3.0) ** 2)


def _spectrum_2d(n_e: int = 32, n_a: int = 4) -> np.ndarray:
    return np.tile(_spectrum_1d(n_e)[:, None], (1, n_a)) + np.arange(n_a)


# ============================================================
# Backward compatibility of the data structures
# ============================================================


class TestBackwardCompat:
    def test_metadata_old_kwargs_still_work(self):
        md = SpectrumMetadata(
            region="Si2p",
            excitation_energy=1486.6,
            energy_scale="Kinetic",
            lens_mode="Angular",
            n_slices=2,
            n_sweeps=4,
            acquisition_mode="Swept",
            pass_energy=50.0,
            comments="",
        )
        assert md.excitation_energy == 1486.6
        assert md.pass_energy == 50.0
        # New fields get benign defaults
        assert md.source_format == "unknown"
        assert md.source_format_version is None
        assert md.source_region_index == 0
        assert dict(md.vendor_metadata) == {}
        assert md.intensity_semantics == "unknown"
        assert md.original_shape == ()
        assert md.dimension_roles == ()

    def test_metadata_defaults_are_none_not_zero(self):
        md = SpectrumMetadata()
        assert md.excitation_energy is None
        assert md.pass_energy is None

    def test_rawspectrumdata_without_transforms(self):
        data = RawSpectrumData(
            specdata=np.zeros((4, 1)),
            energy=np.arange(4.0),
            angle=np.array([0.0]),
            metadata=SpectrumMetadata(),
        )
        assert data.transforms == ()
        data.record_transform("dtype_conversion", {"from": "f4", "to": "f8"})
        assert data.transforms[0].name == "dtype_conversion"


# ============================================================
# PXT (packed experiment, wave v3)
# ============================================================


class TestSyntheticPXT:
    def test_roundtrip_arrays(self, tmp_path):
        spec = _spectrum_1d()
        f = write_pxt(tmp_path / "si2p.pxt", [(spec, make_notes(FULL_NOTES))])
        reader = PXTReader(f)
        assert reader.n_regions == 1
        assert reader.region_names == ["Si2p"]

        data = reader.read(0)
        np.testing.assert_allclose(data.specdata[:, 0], spec, rtol=1e-6)
        np.testing.assert_allclose(
            data.energy, 100.0 + 0.05 * np.arange(len(spec))
        )

    def test_source_format_version_region(self, tmp_path):
        f = write_pxt(
            tmp_path / "two.pxt",
            [
                (_spectrum_1d(), make_notes({**FULL_NOTES, "Region Name": "Si2p"})),
                (_spectrum_1d(), make_notes({**FULL_NOTES, "Region Name": "O1s"})),
            ],
        )
        reader = PXTReader(f)
        d0 = reader.read(0)
        d1 = reader.read(1)
        assert d0.metadata.source_format == "scienta_pxt"
        assert d0.metadata.source_format_version == "3"
        assert d0.metadata.source_region_index == 0
        assert d1.metadata.source_region_index == 1
        assert d1.metadata.region == "O1s"

    def test_standard_fields_typed(self, tmp_path):
        f = write_pxt(tmp_path / "a.pxt", [(_spectrum_1d(), make_notes(FULL_NOTES))])
        md = PXTReader(f).read(0).metadata
        assert md.excitation_energy == pytest.approx(1486.6)
        assert md.pass_energy == pytest.approx(50.0)
        assert md.n_sweeps == 4
        assert md.datetime is not None
        assert md.datetime.year == 2026 and md.datetime.hour == 10

    def test_unknown_energies_are_none(self, tmp_path):
        notes = {k: v for k, v in FULL_NOTES.items()
                 if k not in ("Excitation Energy", "Pass Energy")}
        f = write_pxt(tmp_path / "nohv.pxt", [(_spectrum_1d(), make_notes(notes))])
        md = PXTReader(f).read(0).metadata
        assert md.excitation_energy is None
        assert md.pass_energy is None

    def test_vendor_metadata_preserved(self, tmp_path):
        f = write_pxt(tmp_path / "a.pxt", [(_spectrum_1d(), make_notes(FULL_NOTES))])
        md = PXTReader(f).read(0).metadata
        assert md.vendor_metadata["Detector Mode"] == "ADC"
        assert md.vendor_metadata["Sample"] == "SYNTH-01"
        assert md.vendor_metadata["Instrument"] == "SES-2002"
        # Adopted keys are represented by standard fields, not duplicated
        assert "Excitation Energy" not in md.vendor_metadata
        assert "Region Name" not in md.vendor_metadata
        assert "Date" not in md.vendor_metadata

    def test_malformed_value_warns_and_stays_in_vendor(self, tmp_path):
        notes = {**FULL_NOTES, "Pass Energy": "not-a-number"}
        f = write_pxt(tmp_path / "bad.pxt", [(_spectrum_1d(), make_notes(notes))])
        with pytest.warns(ReaderWarning, match="Pass Energy"):
            data = PXTReader(f).read(0)
        assert data.metadata.pass_energy is None
        # Raw value not lost
        assert data.metadata.vendor_metadata["Pass Energy"] == "not-a-number"
        # Array reading unaffected
        assert data.specdata.shape[0] == 32

    def test_malformed_integer_warns_and_stays_in_vendor(self, tmp_path):
        notes = {**FULL_NOTES, "Number of Sweeps": "not-an-integer"}
        f = write_pxt(tmp_path / "bad-int.pxt", [(_spectrum_1d(), make_notes(notes))])
        with pytest.warns(ReaderWarning, match="Number of Sweeps"):
            data = PXTReader(f).read(0)
        assert data.metadata.n_sweeps == 1
        assert data.metadata.vendor_metadata["Number of Sweeps"] == "not-an-integer"

    def test_intensity_semantics_not_guessed(self, tmp_path):
        f = write_pxt(tmp_path / "a.pxt", [(_spectrum_1d(), make_notes(FULL_NOTES))])
        md = PXTReader(f).read(0).metadata
        assert md.intensity_semantics == "unknown"
        assert md.intensity_unit == "unknown"

    def test_original_shape_and_roles_2d_angular(self, tmp_path):
        f = write_pxt(
            tmp_path / "map.pxt",
            [(_spectrum_2d(), make_notes(FULL_NOTES))],
            x_ini=(100.0, -7.0),
            x_delta=(0.05, 1.0),
        )
        md = PXTReader(f).read(0).metadata
        assert md.original_shape == (32, 4)
        assert md.dimension_roles == ("energy", "emission_angle")

    def test_unconfirmed_dimension_is_unknown_not_angle(self, tmp_path):
        notes = {**FULL_NOTES, "Lens Mode": "T_HiPPHAXPES"}
        f = write_pxt(
            tmp_path / "hippie.pxt",
            [(_spectrum_2d(), make_notes(notes))],
            x_ini=(100.0, 0.0),
            x_delta=(0.05, 1.0),
        )
        md = PXTReader(f).read(0).metadata
        assert md.dimension_roles == ("energy", "unknown")

    def test_third_dimension_unknown(self, tmp_path):
        data3d = np.repeat(_spectrum_2d()[:, :, None], 3, axis=2)
        f = write_pxt(
            tmp_path / "sweep.pxt",
            [(data3d, make_notes(FULL_NOTES))],
            x_ini=(100.0, -7.0, 0.0),
            x_delta=(0.05, 1.0, 1.0),
        )
        md = PXTReader(f).read(0).metadata
        assert md.original_shape == (32, 4, 3)
        assert md.dimension_roles == ("energy", "emission_angle", "unknown")

    def test_slice_reversal_recorded(self, tmp_path):
        notes = {**FULL_NOTES, "Number of Slices": 4}
        f = write_pxt(
            tmp_path / "slices.pxt",
            [(_spectrum_2d(), make_notes(notes))],
            x_ini=(100.0, -7.0),
            x_delta=(0.05, 1.0),
        )
        data = PXTReader(f).read(0)
        targets = [
            (t.name, t.parameters.get("target")) for t in data.transforms
        ]
        assert ("angle_axis_reversal", "specdata") in targets
        assert ("angle_axis_reversal", "angle_values") in targets

    def test_dtype_conversion_recorded(self, tmp_path):
        f = write_pxt(tmp_path / "a.pxt", [(_spectrum_1d(), make_notes(FULL_NOTES))])
        data = PXTReader(f).read(0)
        dtypes = [t for t in data.transforms if t.name == "dtype_conversion"]
        assert len(dtypes) == 1
        assert dtypes[0].parameters == {"from": "float32", "to": "float64"}
        assert data.specdata.dtype == np.float64


# ============================================================
# IBW (standalone wave v5)
# ============================================================


class TestSyntheticIBW:
    def test_source_format_and_version(self, tmp_path):
        f = write_ibw(tmp_path / "one.ibw", _spectrum_1d(), make_notes(FULL_NOTES))
        reader = PXTReader(f)
        assert reader.n_regions == 1
        md = reader.read(0).metadata
        assert md.source_format == "igor_ibw"
        assert md.source_format_version == "5"
        assert md.source_region_index == 0
        assert md.excitation_energy == pytest.approx(1486.6)


# ============================================================
# VAMAS
# ============================================================


class TestSyntheticVAMAS:
    def test_unknowns_are_none(self, tmp_path):
        f = write_vamas(tmp_path / "a.vms")
        md = VAMASReader(f).read(0).metadata
        # The parsed block layout does not carry hv / pass energy
        assert md.excitation_energy is None
        assert md.pass_energy is None
        assert md.intensity_semantics == "unknown"

    def test_provenance_and_shape(self, tmp_path):
        f = write_vamas(tmp_path / "a.vms", n_energy=16, n_sweeps=4)
        md = VAMASReader(f).read(0).metadata
        assert md.source_format == "vamas"
        assert md.source_format_version is not None
        assert "VAMAS" in md.source_format_version
        assert md.source_region_index == 0
        assert md.original_shape == (16,)
        assert md.dimension_roles == ("energy",)
        assert md.n_sweeps == 4
        assert md.datetime is not None and md.datetime.year == 2026


# ============================================================
# NPL
# ============================================================


class TestSyntheticNPL:
    def test_ke_to_be_conversion_recorded(self, tmp_path):
        f = write_npl(tmp_path / "au.npl")
        data = NPLReader(f).read(0)
        assert data.metadata.energy_scale == "Binding"
        assert data.metadata.excitation_energy == pytest.approx(1486.6)
        assert data.metadata.source_format == "npl"

        conv = [t for t in data.transforms if t.name == "energy_scale_conversion"]
        assert len(conv) == 1
        assert conv[0].parameters["work_function_eV"] == pytest.approx(4.5)
        assert conv[0].parameters["excitation_energy_eV"] == pytest.approx(1486.6)
        # BE = hv - KE - WF reconstructs the original axis
        ke = 1486.6 - data.energy - 4.5
        np.testing.assert_allclose(ke, 1100.0 - 0.1 * np.arange(16), rtol=1e-9)

    def test_missing_hv_keeps_kinetic_and_warns(self, tmp_path):
        f = write_npl(tmp_path / "nohv.npl", excitation_energy="none")
        with pytest.warns(ReaderWarning, match="excitation energy"):
            data = NPLReader(f).read(0)
        assert data.metadata.excitation_energy is None
        # No fabricated conversion: axis stays kinetic, honestly labeled
        assert data.metadata.energy_scale == "Kinetic"
        assert all(t.name != "energy_scale_conversion" for t in data.transforms)
        np.testing.assert_allclose(data.energy, 1100.0 - 0.1 * np.arange(16))

    def test_binding_label_no_conversion(self, tmp_path):
        f = write_npl(
            tmp_path / "be.npl", energy_label="binding energy",
            e_ini=90.0, e_step=0.1,
        )
        data = NPLReader(f).read(0)
        assert data.metadata.energy_scale == "Binding"
        assert data.transforms == ()


# ============================================================
# SES text
# ============================================================


class TestSyntheticSES:
    def test_provenance_and_vendor(self, tmp_path):
        f = write_ses_txt(
            tmp_path / "arpes.txt",
            extra_info={"Detector Mode": "ADC", "User": "synth"},
        )
        md = SESTxtReader(f).read(0).metadata
        assert md.source_format == "ses_txt"
        assert md.source_format_version == "1.3.1"
        assert md.pass_energy == pytest.approx(50.0)
        assert md.excitation_energy == pytest.approx(1486.6)
        assert md.vendor_metadata["Detector Mode"] == "ADC"
        assert md.vendor_metadata["User"] == "synth"
        # Dimension scales are represented by the arrays, not vendor text
        assert "Dimension 1 scale" not in md.vendor_metadata

    def test_roles_angular(self, tmp_path):
        f = write_ses_txt(tmp_path / "arpes.txt", lens_mode="Angular56")
        md = SESTxtReader(f).read(0).metadata
        assert md.original_shape == (12, 4)
        assert md.dimension_roles == ("energy", "emission_angle")

    def test_simple_text_dimension_not_assumed_angle(self, tmp_path):
        f = tmp_path / "plain.txt"
        e = 100.0 + 0.1 * np.arange(10)
        table = np.column_stack([e, np.ones(10), 2 * np.ones(10)])
        np.savetxt(f, table)
        md = SESTxtReader(f).read(0).metadata
        assert md.source_format == "text_columns"
        assert md.dimension_roles == ("energy", "unknown")
        assert md.excitation_energy is None
        assert md.intensity_semantics == "unknown"


# ============================================================
# Defensive copies
# ============================================================


class TestDefensiveCopies:
    def test_vendor_metadata_isolated_from_caller(self):
        vendor = {"Sample": "A", "nested": {"k": 1}}
        md = SpectrumMetadata(vendor_metadata=vendor)
        vendor["Sample"] = "B"
        vendor["nested"]["k"] = 999
        assert md.vendor_metadata["Sample"] == "A"
        assert md.vendor_metadata["nested"]["k"] == 1

    def test_transform_parameters_isolated_from_caller(self):
        params = {"input_shape": [4, 2]}
        t = ReaderTransform(name="dimension_flattening", parameters=params)
        params["input_shape"].append(99)
        params["extra"] = True
        assert t.parameters == {"input_shape": [4, 2]}

    def test_transform_to_dict_is_a_copy(self):
        t = ReaderTransform(name="x", parameters={"a": [1]})
        d = t.to_dict()
        d["parameters"]["a"].append(2)
        assert t.parameters == {"a": [1]}


# ============================================================
# Importer: transform history + None handling end-to-end
# ============================================================


class TestImporterWithSyntheticFiles:
    def _pxt(self, tmp_path, notes=None, data=None, **kwargs):
        return write_pxt(
            tmp_path / "in.pxt",
            [(data if data is not None else _spectrum_1d(),
              make_notes(notes or FULL_NOTES))],
            **kwargs,
        )

    def test_import_pxt_end_to_end(self, tmp_path):
        from toyomacro.io.importer import ImportConfig, import_file

        f = self._pxt(tmp_path)
        result = import_file(f, tmp_path / "out", ImportConfig(element="Si2p", compress=False))
        assert result.output_path.exists()
        assert result.n_spectra == 1
        assert result.metadata["source_format"] == "scienta_pxt"
        assert result.metadata["source_format_version"] == "3"
        assert result.metadata["original_shape"] == [32]
        assert result.metadata["dimension_roles"] == ["energy"]

    def test_ke_to_be_conversion_in_history(self, tmp_path):
        from toyomacro.io.importer import ImportConfig, import_file

        f = self._pxt(tmp_path)
        result = import_file(
            f, tmp_path / "out",
            ImportConfig(element="Si2p", compress=False, energy_scale="BE"),
        )
        names = [t["name"] for t in result.metadata["transforms"]]
        assert "energy_scale_conversion" in names
        conv = next(t for t in result.metadata["transforms"]
                    if t["name"] == "energy_scale_conversion")
        assert conv["parameters"]["excitation_energy_eV"] == pytest.approx(1486.6)
        assert conv["parameters"]["from"] == "Kinetic"
        # BE = hv - KE: 100..101.55 -> 1385.05..1386.6
        assert result.energy_range[0] == pytest.approx(1486.6 - 101.55)
        assert result.energy_range[1] == pytest.approx(1486.6 - 100.0)

    def test_conversion_without_hv_raises(self, tmp_path):
        from toyomacro.io.importer import ImportConfig, import_file

        notes = {k: v for k, v in FULL_NOTES.items() if k != "Excitation Energy"}
        f = self._pxt(tmp_path, notes=notes)
        with pytest.raises(ValueError, match="excitation energy"):
            import_file(
                f, tmp_path / "out",
                ImportConfig(element="Si2p", compress=False, energy_scale="BE"),
            )

    def test_unknown_hv_writes_legacy_sentinel_to_h5(self, tmp_path):
        import h5py

        from toyomacro.io.importer import ImportConfig, import_file

        notes = {k: v for k, v in FULL_NOTES.items() if k != "Excitation Energy"}
        f = self._pxt(tmp_path, notes=notes)
        result = import_file(f, tmp_path / "out", ImportConfig(element="Si2p", compress=False))
        # In-memory truth stays None; the (unchanged) HDF5 schema keeps 0.0
        assert result.metadata["excitation_energy"] is None
        with h5py.File(result.output_path) as h5:
            assert float(h5["misc/fermienergy"][0]) == 0.0

    def test_flatten_recorded(self, tmp_path):
        from toyomacro.io.importer import ImportConfig, import_file

        data3d = np.repeat(_spectrum_2d()[:, :, None], 3, axis=2)
        f = self._pxt(
            tmp_path, data=data3d,
            x_ini=(100.0, -7.0, 0.0), x_delta=(0.05, 1.0, 1.0),
        )
        result = import_file(
            f, tmp_path / "out",
            ImportConfig(element="Si2p", compress=False, sweep_mode="individual"),
        )
        assert result.n_spectra == 12
        flat = next(t for t in result.metadata["transforms"]
                    if t["name"] == "dimension_flattening")
        assert flat["parameters"]["input_shape"] == [32, 4, 3]
        assert flat["parameters"]["output_shape"] == [32, 12]

    def test_sweep_integration_recorded(self, tmp_path):
        from toyomacro.io.importer import ImportConfig, import_file

        data3d = np.repeat(_spectrum_2d()[:, :, None], 3, axis=2)
        f = self._pxt(
            tmp_path, data=data3d,
            x_ini=(100.0, -7.0, 0.0), x_delta=(0.05, 1.0, 1.0),
        )
        result = import_file(
            f, tmp_path / "out",
            ImportConfig(element="Si2p", compress=False, sweep_mode="integrate"),
        )
        assert result.n_spectra == 4
        integ = next(t for t in result.metadata["transforms"]
                     if t["name"] == "sweep_integration")
        assert integ["parameters"]["input_shape"] == [32, 4, 3]
        assert integ["parameters"]["output_shape"] == [32, 4]

    def test_unknown_detector_dimension_not_written_as_angle(self, tmp_path):
        import h5py

        from toyomacro.io.importer import ImportConfig, import_file

        notes = {**FULL_NOTES, "Lens Mode": "T_HiPPHAXPES"}
        f = self._pxt(
            tmp_path,
            notes=notes,
            data=_spectrum_2d(),
            x_ini=(100.0, 0.0),
            x_delta=(0.05, 1.0),
        )
        result = import_file(
            f,
            tmp_path / "out",
            ImportConfig(element="Si2p", compress=False),
        )
        assert result.metadata["dimension_roles"] == ["energy", "unknown"]
        with h5py.File(result.output_path) as h5:
            assert int(h5["misc/dim0_is_angle"][0]) == 0

    def test_ensure_h5_on_synthetic_pxt(self, tmp_path):
        from toyomacro.io.importer import ensure_h5

        f = self._pxt(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ReaderWarning)
            paths = ensure_h5(f, output_dir=tmp_path / "out")
        assert len(paths) == 1
        assert paths[0].suffix == ".h5"
        assert paths[0].exists()
