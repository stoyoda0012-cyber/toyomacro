"""Tests for the HDF5 /provenance group (Phase B-1).

Toyomacro HDF5 is the project's local storage schema (not a vendor
format); /provenance records facts read from the upstream input files
plus the reader/importer transform history. These tests run entirely on
synthetic fixtures (tests/_synthetic_files.py).

Design contract: the provenance design note (non-public; see
CONTRIBUTING.md). The contract it fixes is asserted here.
"""

from __future__ import annotations

import json
import warnings

import h5py
import numpy as np
import pytest

from tests._synthetic_files import make_notes, write_pxt, write_vamas
from tests.test_readers_synthetic import FULL_NOTES, _spectrum_1d, _spectrum_2d
from toyomacro.io.importer import ImportConfig, import_file
from toyomacro.io.provenance import (
    HDF5Provenance,
    ProvenanceWarning,
    read_provenance,
    write_provenance,
)
from toyomacro.io.readers.base_reader import ReaderTransform, SpectrumMetadata
from toyomacro.io.readers.lazy_spectrum import LazySpectrum
from toyomacro.io.schema import ToyomacroSchema


def _import_pxt(tmp_path, notes=None, data=None, config=None, **pxt_kwargs):
    f = write_pxt(
        tmp_path / "in.pxt",
        [(data if data is not None else _spectrum_1d(),
          make_notes(notes or FULL_NOTES))],
        **pxt_kwargs,
    )
    cfg = config or ImportConfig(element="Si2p", compress=False)
    return import_file(f, tmp_path / "out", cfg)


# ============================================================
# Write path: attributes, omission rules, root version
# ============================================================


class TestProvenanceWrite:
    def test_group_and_standard_attrs(self, tmp_path):
        result = _import_pxt(tmp_path)
        with h5py.File(result.output_path) as f:
            assert "provenance" in f
            a = f["provenance"].attrs
            assert a["schema_version"] == ToyomacroSchema.PROVENANCE_SCHEMA_VERSION
            assert a["source_format"] == "scienta_pxt"
            assert a["source_format_version"] == "3"
            assert int(a["source_region_index"]) == 0
            assert a["region_name"] == "Si2p"
            assert float(a["excitation_energy_eV"]) == pytest.approx(1486.6)
            assert float(a["pass_energy_eV"]) == pytest.approx(50.0)
            assert a["intensity_semantics"] == "unknown"
            assert a["intensity_unit"] == "unknown"
            assert list(a["original_shape"]) == [32]
            assert list(a["dimension_roles"]) == ["energy"]
            # Not persisted in B-1: Phase A cannot guarantee these were
            # explicitly read from the file (audit correction).
            assert "n_sweeps" not in a
            assert "n_slices" not in a
            assert "lens_mode" not in a
            # acquisition_mode: FULL_NOTES has no explicit value -> absent
            assert "acquisition_mode" not in a
            prov = read_provenance(f)
        # Reserved fields restore as None, never as the structural default 1
        assert prov.n_sweeps is None
        assert prov.n_slices is None
        assert prov.lens_mode is None
        assert prov.acquisition_mode is None

    def test_root_schema_version_stamped(self, tmp_path):
        result = _import_pxt(tmp_path)
        with h5py.File(result.output_path) as f:
            assert (
                f.attrs[ToyomacroSchema.ATTR_SCHEMA_VERSION]
                == ToyomacroSchema.VERSION
                == "1.2.0"
            )

    def test_unknown_energies_have_no_attr(self, tmp_path):
        notes = {k: v for k, v in FULL_NOTES.items()
                 if k not in ("Excitation Energy", "Pass Energy")}
        result = _import_pxt(tmp_path, notes=notes)
        with h5py.File(result.output_path) as f:
            a = f["provenance"].attrs
            assert "excitation_energy_eV" not in a
            assert "pass_energy_eV" not in a
            prov = read_provenance(f)
        assert prov.excitation_energy is None
        assert prov.pass_energy is None

    def test_datetime_not_persisted_by_default(self, tmp_path):
        result = _import_pxt(tmp_path)  # FULL_NOTES includes Date/Time
        with h5py.File(result.output_path) as f:
            assert "datetime" not in f["provenance"].attrs

    def test_datetime_optin(self, tmp_path):
        result = _import_pxt(
            tmp_path,
            config=ImportConfig(element="Si2p", compress=False, persist_datetime=True),
        )
        with h5py.File(result.output_path) as f:
            assert f["provenance"].attrs["datetime"] == "2026-01-15T10:30:00"
            assert read_provenance(f).datetime == "2026-01-15T10:30:00"


# ============================================================
# Transform history
# ============================================================


class TestTransformHistory:
    def test_records_match_import_result_in_order(self, tmp_path):
        result = _import_pxt(
            tmp_path,
            config=ImportConfig(element="Si2p", compress=False, energy_scale="BE"),
        )
        with h5py.File(result.output_path) as f:
            prov = read_provenance(f)
        assert list(prov.transforms) == result.metadata["transforms"]
        names = [t["name"] for t in prov.transforms]
        # Reader transforms first, importer transforms last
        assert names == ["dtype_conversion", "energy_scale_conversion"]
        assert not prov.transform_history_incomplete
        assert prov.dropped_transform_records == 0

    def test_empty_history_has_no_dataset(self, tmp_path):
        f_in = write_vamas(tmp_path / "a.vms")
        result = import_file(
            f_in, tmp_path / "out", ImportConfig(element="Test1s", compress=False)
        )
        with h5py.File(result.output_path) as f:
            assert "transform_history" not in f["provenance"]
            prov = read_provenance(f)
        assert prov.transforms == ()
        assert prov.source_format == "vamas"

    def test_unknown_transform_name_tolerated(self, tmp_path):
        result = _import_pxt(tmp_path)
        with h5py.File(result.output_path, "r+") as f:
            grp = f["provenance"]
            records = [r for r in grp["transform_history"][:]]
            records.append(json.dumps(
                {"name": "future_transform_v9", "parameters": {"x": 1},
                 "source": "future", "reason": ""}
            ))
            del grp["transform_history"]
            grp.create_dataset(
                "transform_history", data=records,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
        with h5py.File(result.output_path) as f:
            prov = read_provenance(f)
        assert prov.transforms[-1]["name"] == "future_transform_v9"
        assert not prov.transform_history_incomplete

    def test_malformed_record_skipped_with_warning(self, tmp_path):
        result = _import_pxt(tmp_path)
        with h5py.File(result.output_path, "r+") as f:
            grp = f["provenance"]
            records = [r for r in grp["transform_history"][:]]
            n_valid = len(records)
            records.append("{not valid json")
            del grp["transform_history"]
            grp.create_dataset(
                "transform_history", data=records,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
        with h5py.File(result.output_path) as f:
            with pytest.warns(ProvenanceWarning, match="not valid JSON"):
                prov = read_provenance(f)
        assert len(prov.transforms) == n_valid
        assert prov.transform_history_incomplete
        assert prov.dropped_transform_records == 1

    def test_json_unsafe_transform_dropped_not_replaced(self, tmp_path):
        md = SpectrumMetadata(region="X", source_format="scienta_pxt")
        good = ReaderTransform(name="dtype_conversion",
                               parameters={"from": "f4", "to": "f8"})
        bad = ReaderTransform(name="weird", parameters={"bad": {1, 2, 3}})
        out = tmp_path / "direct.h5"
        with h5py.File(out, "w") as f:
            with pytest.warns(ProvenanceWarning, match="not JSON-safe"):
                write_provenance(f, md, [good, bad])
        with h5py.File(out) as f:
            grp = f["provenance"]
            assert bool(grp.attrs["transform_history_incomplete"]) is True
            assert int(grp.attrs["dropped_transform_records"]) == 1
            stored = [json.loads(r) for r in grp["transform_history"][:]]
            # No placeholder record: only the serializable one survives
            assert [r["name"] for r in stored] == ["dtype_conversion"]
            prov = read_provenance(f)
        assert prov.transform_history_incomplete
        assert prov.dropped_transform_records == 1

    def test_all_records_dropped_leaves_flags_only(self, tmp_path):
        md = SpectrumMetadata(region="X")
        bad = ReaderTransform(name="weird", parameters={"bad": {1, 2}})
        out = tmp_path / "direct.h5"
        with h5py.File(out, "w") as f:
            with pytest.warns(ProvenanceWarning):
                write_provenance(f, md, [bad])
        with h5py.File(out) as f:
            assert "transform_history" not in f["provenance"]
            prov = read_provenance(f)
        assert prov.transforms == ()
        assert prov.transform_history_incomplete
        assert prov.dropped_transform_records == 1


# ============================================================
# Vendor metadata policy
# ============================================================


class TestVendorMetadata:
    def test_not_persisted_by_default(self, tmp_path):
        result = _import_pxt(tmp_path)  # FULL_NOTES has vendor keys
        with h5py.File(result.output_path) as f:
            assert "vendor_metadata" not in f["provenance"]
            assert read_provenance(f).vendor_metadata is None

    def test_optin_requires_allowlist(self, tmp_path):
        result = _import_pxt(
            tmp_path,
            config=ImportConfig(element="Si2p", compress=False,
                                persist_vendor_metadata=True),
        )
        with h5py.File(result.output_path) as f:
            assert "vendor_metadata" not in f["provenance"]

    def test_allowlist_intersection_only(self, tmp_path):
        result = _import_pxt(
            tmp_path,
            config=ImportConfig(
                element="Si2p", compress=False,
                persist_vendor_metadata=True,
                vendor_metadata_allowlist=("Detector Mode", "Instrument", "NoSuchKey"),
            ),
        )
        with h5py.File(result.output_path) as f:
            prov = read_provenance(f)
            ds = f["provenance"]["vendor_metadata"]
            assert ds.attrs["policy_version"] == "1.0"
            assert list(ds.attrs["allowlist"]) == [
                "Detector Mode", "Instrument", "NoSuchKey"
            ]
        assert prov.vendor_metadata == {
            "Detector Mode": "ADC", "Instrument": "SES-2002"
        }
        assert "Sample" not in prov.vendor_metadata  # not allowlisted

    def test_unparsed_notes_never_persisted(self, tmp_path):
        md = SpectrumMetadata(
            region="X",
            vendor_metadata={"unparsed_notes": ["free text"], "Ok": "v"},
        )
        out = tmp_path / "direct.h5"
        with h5py.File(out, "w") as f:
            with pytest.warns(ProvenanceWarning, match="never persisted"):
                write_provenance(
                    f, md, (),
                    persist_vendor_metadata=True,
                    vendor_metadata_allowlist=("unparsed_notes", "Ok"),
                )
        with h5py.File(out) as f:
            prov = read_provenance(f)
        assert prov.vendor_metadata == {"Ok": "v"}


# ============================================================
# Read path: legacy files, versioning
# ============================================================


class TestReadCompat:
    def test_legacy_file_without_provenance(self, tmp_path):
        path = tmp_path / "legacy.h5"
        with h5py.File(path, "w") as f:
            data = np.zeros((3, 10), dtype=np.float32)
            data[0, :] = np.arange(10)
            f.create_dataset("specdata", data=data)
        with h5py.File(path) as f:
            assert ToyomacroSchema.ATTR_SCHEMA_VERSION not in f.attrs
            assert read_provenance(f) is None
        with LazySpectrum(path) as spec:
            assert spec.get_provenance() is None
            assert spec.n_spectra == 2  # legacy read works unchanged

    def test_lazyspectrum_reads_provenance(self, tmp_path):
        result = _import_pxt(tmp_path)
        with LazySpectrum(result.output_path) as spec:
            prov = spec.get_provenance()
        assert isinstance(prov, HDF5Provenance)
        assert prov.source_format == "scienta_pxt"
        assert prov.excitation_energy == pytest.approx(1486.6)

    def test_missing_schema_version_warns_best_effort(self, tmp_path):
        path = tmp_path / "odd.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("provenance")
            grp.attrs["source_format"] = "vamas"
        with h5py.File(path) as f:
            with pytest.warns(ProvenanceWarning, match="schema_version"):
                prov = read_provenance(f)
        assert prov.schema_version is None
        assert prov.source_format == "vamas"

    def test_major_version_mismatch_warns_best_effort(self, tmp_path):
        path = tmp_path / "future.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("provenance")
            grp.attrs["schema_version"] = "2.0"
            grp.attrs["source_format"] = "npl"
        with h5py.File(path) as f:
            with pytest.warns(ProvenanceWarning, match="unsupported major"):
                prov = read_provenance(f)
        assert prov.schema_version == "2.0"
        assert prov.source_format == "npl"


# ============================================================
# Malformed provenance: best-effort reads, never exceptions
# ============================================================


def _make_provenance_file(tmp_path, mutate):
    """Create a minimal valid provenance file, then apply *mutate*(grp)."""
    path = tmp_path / "malformed.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("provenance")
        grp.attrs["schema_version"] = "1.0"
        grp.attrs["source_format"] = "vamas"
        mutate(grp)
    return path


class TestMalformedProvenance:
    def test_broken_int_attrs_do_not_raise(self, tmp_path):
        def mutate(grp):
            grp.attrs["n_sweeps"] = "broken"
            grp.attrs["source_region_index"] = "x"

        path = _make_provenance_file(tmp_path, mutate)
        with h5py.File(path) as f:
            with pytest.warns(ProvenanceWarning, match="n_sweeps"):
                prov = read_provenance(f)
        assert prov.n_sweeps is None  # never the structural default 1
        assert prov.source_region_index == 0
        assert prov.source_format == "vamas"

    def test_broken_float_attrs_restore_none(self, tmp_path):
        def mutate(grp):
            grp.attrs["excitation_energy_eV"] = "abc"
            grp.attrs["pass_energy_eV"] = np.float64(np.nan)

        path = _make_provenance_file(tmp_path, mutate)
        with h5py.File(path) as f:
            with pytest.warns(ProvenanceWarning):
                prov = read_provenance(f)
        assert prov.excitation_energy is None
        assert prov.pass_energy is None

    def test_array_valued_scalar_attrs(self, tmp_path):
        def mutate(grp):
            grp.attrs["excitation_energy_eV"] = np.array([1.0, 2.0])
            grp.attrs["source_format"] = np.array([1, 2])

        path = _make_provenance_file(tmp_path, mutate)
        with h5py.File(path) as f:
            with pytest.warns(ProvenanceWarning):
                prov = read_provenance(f)
        assert prov.excitation_energy is None
        assert prov.source_format == "unknown"

    def test_original_shape_and_roles_malformed(self, tmp_path):
        def mutate(grp):
            grp.attrs["original_shape"] = "abc"
            grp.attrs["dimension_roles"] = "energy"  # scalar, not 1-D array

        path = _make_provenance_file(tmp_path, mutate)
        with h5py.File(path) as f:
            with pytest.warns(ProvenanceWarning):
                prov = read_provenance(f)
        assert prov.original_shape == ()
        assert prov.dimension_roles == ()

    def test_transform_history_scalar_dataset(self, tmp_path):
        def mutate(grp):
            grp.create_dataset("transform_history", data=3.14)  # scalar, wrong type

        path = _make_provenance_file(tmp_path, mutate)
        with h5py.File(path) as f:
            with pytest.warns(ProvenanceWarning, match="transform_history"):
                prov = read_provenance(f)
        assert prov.transforms == ()
        assert prov.transform_history_incomplete
        assert prov.source_format == "vamas"  # other fields still readable

    def test_transform_history_as_group(self, tmp_path):
        def mutate(grp):
            grp.create_group("transform_history")

        path = _make_provenance_file(tmp_path, mutate)
        with h5py.File(path) as f:
            with pytest.warns(ProvenanceWarning, match="transform_history"):
                prov = read_provenance(f)
        assert prov.transforms == ()
        assert prov.transform_history_incomplete

    def test_transform_history_wrong_dtype(self, tmp_path):
        def mutate(grp):
            grp.create_dataset("transform_history", data=np.arange(3.0))

        path = _make_provenance_file(tmp_path, mutate)
        with h5py.File(path) as f:
            with pytest.warns(ProvenanceWarning):
                prov = read_provenance(f)
        assert prov.transforms == ()
        assert prov.dropped_transform_records == 3

    def test_vendor_metadata_malformed_dataset(self, tmp_path):
        def mutate(grp):
            grp.create_dataset("vendor_metadata", data=1.0)  # scalar

        path = _make_provenance_file(tmp_path, mutate)
        with h5py.File(path) as f:
            with pytest.warns(ProvenanceWarning, match="vendor_metadata"):
                prov = read_provenance(f)
        assert prov.vendor_metadata is None
        assert prov.source_format == "vamas"

    def test_vendor_metadata_empty_dataset(self, tmp_path):
        def mutate(grp):
            grp.create_dataset(
                "vendor_metadata", shape=(0,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

        path = _make_provenance_file(tmp_path, mutate)
        with h5py.File(path) as f:
            with pytest.warns(ProvenanceWarning, match="vendor_metadata"):
                prov = read_provenance(f)
        assert prov.vendor_metadata is None

    def test_provenance_not_a_group(self, tmp_path):
        path = tmp_path / "notgroup.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("provenance", data=np.zeros(4))
        with h5py.File(path) as f:
            with pytest.warns(ProvenanceWarning, match="not an HDF5 group"):
                assert read_provenance(f) is None


# ============================================================
# Derived files: compression, repack, streaming fit output
# ============================================================


def _read_prov(path) -> HDF5Provenance:
    with h5py.File(path) as f:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProvenanceWarning)
            return read_provenance(f)


class TestDerivedFiles:
    def _source(self, tmp_path):
        return _import_pxt(
            tmp_path,
            data=_spectrum_2d(),
            config=ImportConfig(element="Si2p", compress=False, energy_scale="BE"),
            x_ini=(100.0, -7.0),
            x_delta=(0.05, 1.0),
        )

    @pytest.mark.parametrize("mode", ["full", "standard"])
    def test_compression_preserves_provenance(self, tmp_path, mode):
        from toyomacro.io.compression import HAS_LZ4, compress_h5_file_streaming

        if not HAS_LZ4:
            pytest.skip("lz4 not installed")
        result = self._source(tmp_path)
        out = tmp_path / "compressed.h5"
        compress_h5_file_streaming(result.output_path, out, compression=mode)
        # Semantic equality of provenance content (not file bit-identity)
        assert _read_prov(out) == _read_prov(result.output_path)
        with h5py.File(out) as f:
            assert f.attrs[ToyomacroSchema.ATTR_SCHEMA_VERSION] == "1.2.0"

    def test_repack_preserves_provenance(self, tmp_path):
        from toyomacro.voigtfit.tools.repack_h5 import repack_file

        result = self._source(tmp_path)
        out = tmp_path / "repacked.h5"
        repack_file(str(result.output_path), str(out), verbose=False)
        assert _read_prov(out) == _read_prov(result.output_path)
        with h5py.File(out) as f:
            assert f.attrs[ToyomacroSchema.ATTR_SCHEMA_VERSION] == "1.2.0"

    def test_streaming_writer_inherits_provenance_and_root_attrs(self, tmp_path):
        from toyomacro.io.writers.streaming_writer import (
            StreamingFitparaWriter,
            StreamingWriterConfig,
        )

        result = self._source(tmp_path)
        out = tmp_path / "fit_out.h5"
        cfg = StreamingWriterConfig(
            output_path=str(out),
            n_spectra=result.n_spectra,
            n_components=2,
            write_mode="new_file",
            compress_fitpara=False,
            compress_on_close=False,
        )
        with StreamingFitparaWriter(cfg, input_path=result.output_path):
            pass  # metadata/attribute inheritance happens on setup

        assert _read_prov(out) == _read_prov(result.output_path)
        with h5py.File(out) as f:
            assert f.attrs[ToyomacroSchema.ATTR_SCHEMA_VERSION] == "1.2.0"
            # Recovery markers must be this file's own state, not inherited
            assert "_toyomacro_incomplete" not in f.attrs
