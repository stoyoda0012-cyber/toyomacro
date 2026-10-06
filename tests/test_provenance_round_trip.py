"""What a file said survives HDF5 and comes back as what it was.

The five round-trip conditions of the v0.5.0 provenance design:

1. A value the file states (``n_sweeps=1``) comes back as a confirmed 1.
2. An unknown value comes back unknown — never as a structural default.
3. A user's override comes back with its origin and its history.
4. A file written before schema 1.1 is not given facts it never had.
5. Every ordinate series comes back exactly — label, unit, values,
   shape and the energy axis it belongs to — through import,
   compression, repack and the streaming writer.
"""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest

from tests._synthetic_files import make_notes, vamas_iso_block, write_pxt, write_vamas_iso
from tests.test_readers_synthetic import FULL_NOTES, _spectrum_1d
from toyomacro.io.importer import ImportConfig, import_file
from toyomacro.io.provenance import ProvenanceWarning, read_provenance
from toyomacro.io.readers.vamas_reader import VAMASReader
from toyomacro.io.schema import ToyomacroSchema


def _prov(path):
    with h5py.File(path) as f:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProvenanceWarning)
            return read_provenance(f)


def _import_vamas(tmp_path, reader=None, config=None, **block):
    f = write_vamas_iso(tmp_path / "in.vms", blocks=[vamas_iso_block(n_energy=12, **block)])
    cfg = config or ImportConfig(element="C1s", compress=False)
    return f, import_file(f, tmp_path / "out", cfg, reader=reader)


# --- 1. a stated value is a confirmed value ----------------------------------

def test_a_stated_one_comes_back_as_a_confirmed_one(tmp_path):
    _, result = _import_vamas(tmp_path, n_scans="1")
    prov = _prov(result.output_path)
    assert prov.schema_version == ToyomacroSchema.PROVENANCE_SCHEMA_VERSION == "1.1"
    assert prov.n_sweeps == 1
    assert prov.field_origins["n_sweeps"] == "file"
    assert prov.excitation_energy == pytest.approx(1486.6)
    assert prov.field_origins["excitation_energy"] == "file"
    assert prov.signal_mode == "pulse counting"
    assert prov.signal_collection_time == pytest.approx(0.05)
    assert prov.transmission_curve == "embedded"
    assert prov.transmission_curve_variable == "Transmission"


# --- 2. unknown stays unknown ------------------------------------------------

def test_an_unknown_value_comes_back_unknown(tmp_path):
    _, result = _import_vamas(tmp_path, n_scans="x", collection_time="1e+037")
    prov = _prov(result.output_path)
    # The reader keeps n_sweeps=1 in memory as a structural default; it
    # must not reach the file as a fact.
    assert prov.n_sweeps is None and "n_sweeps" not in prov.field_origins
    assert prov.signal_collection_time is None
    assert prov.n_slices is None and prov.lens_mode is None
    # A curve in the file says nothing about whether it was applied.
    assert prov.transmission_applied is None and prov.transmission_basis is None
    assert prov.transmission_curve_normalisation is None


# --- 3. a user's override, with its history -----------------------------------

def test_a_user_override_comes_back_with_origin_and_history(tmp_path):
    f = write_vamas_iso(tmp_path / "in.vms", blocks=[vamas_iso_block(n_energy=12)])
    reader = VAMASReader(f)
    data = reader.read(0)
    data.declare("n_sweeps", 8, "acquisition log: 8 sweeps, file records the export")
    data.declare("transmission_applied", "not_applied", "exported raw from the vendor software")
    result = import_file(f, tmp_path / "out", ImportConfig(element="C1s", compress=False),
                         data=data)
    prov = _prov(result.output_path)
    assert prov.n_sweeps == 8 and prov.field_origins["n_sweeps"] == "user"
    assert (prov.transmission_applied, prov.transmission_basis) == ("not_applied", "user")
    records = [t for t in prov.transforms if t["name"] == "user_declaration"]
    assert [r["parameters"]["field"] for r in records] == ["n_sweeps", "transmission_applied"]
    assert records[0]["parameters"]["previous"] == 4
    assert records[0]["parameters"]["previous_origin"] == "file"
    assert records[0]["reason"].startswith("acquisition log")
    assert records[1]["parameters"]["previous_origin"] == "unknown"


def test_a_declaration_needs_a_known_field_and_a_reason(tmp_path):
    data = VAMASReader(write_vamas_iso(tmp_path / "a.vms")).read(0)
    with pytest.raises(ValueError, match="cannot be declared"):
        data.declare("region", "x", "because")
    with pytest.raises(ValueError, match="reason"):
        data.declare("n_sweeps", 2, "")


# --- 4. old files are not given facts -----------------------------------------

def test_a_reader_that_records_no_origins_persists_no_reserved_fields(tmp_path):
    """PXT records no origins yet: lens mode, sweeps and slices stay unwritten."""
    f = write_pxt(tmp_path / "in.pxt", [(_spectrum_1d(), make_notes(FULL_NOTES))])
    result = import_file(f, tmp_path / "out", ImportConfig(element="Si2p", compress=False))
    prov = _prov(result.output_path)
    assert (prov.n_sweeps, prov.n_slices, prov.lens_mode) == (None, None, None)
    assert prov.series is None


def test_a_schema_1_0_group_reads_without_invented_facts(tmp_path):
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("provenance")
        g.attrs["schema_version"] = "1.0"
        g.attrs["source_format"] = "vamas"
        g.attrs["excitation_energy_eV"] = 1486.6
        f.create_group("misc").create_dataset("numberofslice", data=[1])
    prov = _prov(path)
    assert prov.excitation_energy == pytest.approx(1486.6)
    assert prov.field_origins == {}  # 1.0 recorded no origins
    assert prov.n_slices is None  # the legacy misc field is not a fact
    assert prov.n_sweeps is None and prov.series is None
    assert prov.transmission_applied is None and prov.transmission_curve is None


# --- 5. the series, exactly, through every derived file -----------------------

def test_series_come_back_exactly_on_the_files_own_axis(tmp_path):
    f, result = _import_vamas(
        tmp_path, config=ImportConfig(element="C1s", compress=False, energy_scale="BE"))
    original = VAMASReader(f).read(0)
    prov = _prov(result.output_path)
    # The importer converted the main axis to binding energy ...
    with h5py.File(result.output_path) as h5:
        assert any(t["name"] == "energy_scale_conversion" for t in prov.transforms)
    # ... the stored series stay on the kinetic-energy axis they belong to.
    np.testing.assert_array_equal(prov.series.energy, original.energy)
    assert prov.series.variables == original.corresponding_variables
    assert all(v.values.dtype == np.float64 for v in prov.series.variables)
    assert prov.series.variables[1].values.shape == (12,)


@pytest.fixture
def imported(tmp_path):
    values = [f"{1000.123456789 + 0.000000001 * i:.12f}" for i in range(24)]
    _, result = _import_vamas(tmp_path, ordinates=values)
    return result


def test_series_survive_repack_without_float32(imported, tmp_path):
    from toyomacro.voigtfit.tools.repack_h5 import repack_file

    out = tmp_path / "repacked.h5"
    repack_file(str(imported.output_path), str(out), verbose=False)
    assert _prov(out) == _prov(imported.output_path)
    with h5py.File(out) as h5:
        assert h5["provenance/corresponding/var0"].dtype == np.float64


@pytest.mark.parametrize("mode", ["full", "standard"])
def test_series_survive_compression(imported, tmp_path, mode):
    from toyomacro.io.compression import HAS_LZ4, compress_h5_file_streaming

    if not HAS_LZ4:
        pytest.skip("lz4 not installed")
    out = tmp_path / "compressed.h5"
    compress_h5_file_streaming(imported.output_path, out, compression=mode)
    assert _prov(out) == _prov(imported.output_path)


def test_series_survive_the_streaming_writer(imported, tmp_path):
    from toyomacro.io.writers.streaming_writer import (
        StreamingFitparaWriter,
        StreamingWriterConfig,
    )

    out = tmp_path / "fit_out.h5"
    cfg = StreamingWriterConfig(
        output_path=str(out), n_spectra=imported.n_spectra, n_components=2,
        write_mode="new_file", compress_fitpara=False, compress_on_close=False)
    with StreamingFitparaWriter(cfg, input_path=imported.output_path):
        pass
    assert _prov(out) == _prov(imported.output_path)


def test_a_malformed_series_is_ignored_with_a_warning(imported):
    with h5py.File(imported.output_path, "a") as h5:
        del h5["provenance/corresponding/energy"]
    with h5py.File(imported.output_path) as h5:
        with pytest.warns(ProvenanceWarning, match="corresponding"):
            prov = read_provenance(h5)
    assert prov.series is None


# --- what the first audit found -------------------------------------------------

def test_reusing_a_reader_keeps_the_files_axis_in_every_import(tmp_path):
    """The importer must not convert the reader's cached data in place."""
    f = write_vamas_iso(tmp_path / "in.vms", blocks=[vamas_iso_block(n_energy=12)])
    reader = VAMASReader(f)
    file_axis = reader.read(0).energy.copy()
    cfg = ImportConfig(element="C1s", compress=False, energy_scale="BE")
    first = import_file(f, tmp_path / "a", cfg, reader=reader)
    second = import_file(f, tmp_path / "b", cfg, reader=reader)
    np.testing.assert_array_equal(reader.read(0).energy, file_axis)
    for result in (first, second):
        np.testing.assert_array_equal(_prov(result.output_path).series.energy, file_axis)
    assert _prov(second.output_path) == _prov(first.output_path)


@pytest.mark.parametrize(("value", "stored"), [(np.int64(8), 8), (8.0, 8)])
def test_numpy_and_integral_declarations_are_stored(tmp_path, value, stored):
    f = write_vamas_iso(tmp_path / "in.vms", blocks=[vamas_iso_block(n_energy=12)])
    reader = VAMASReader(f)
    data = reader.read(0)
    data.declare("n_sweeps", value, "acquisition log")
    result = import_file(f, tmp_path / "out", ImportConfig(element="C1s", compress=False),
                         data=data)
    prov = _prov(result.output_path)
    assert (prov.n_sweeps, prov.field_origins["n_sweeps"]) == (stored, "user")
    assert not prov.transform_history_incomplete


@pytest.mark.parametrize(("field_name", "value"), [
    ("n_sweeps", True), ("n_sweeps", 8.5), ("n_sweeps", 0), ("excitation_energy", "abc"),
    ("excitation_energy", float("nan")), ("transmission_applied", "yes please"),
    ("transmission_curve", "somewhere"), ("lens_mode", ""), ("lens_mode", "unknown"),
    ("transmission_curve_normalisation", "Unknown"), ("excitation_energy", 0.0),
    ("pass_energy", -20.0),
])
def test_an_unstorable_declaration_changes_nothing(tmp_path, field_name, value):
    data = VAMASReader(write_vamas_iso(tmp_path / "a.vms")).read(0)
    before = (getattr(data.metadata, field_name), dict(data.metadata.field_origins),
              data.metadata.transmission_basis, data.transforms)
    with pytest.raises(ValueError):
        data.declare(field_name, value, "because")
    after = (getattr(data.metadata, field_name), dict(data.metadata.field_origins),
             data.metadata.transmission_basis, data.transforms)
    assert after == before


def test_no_curve_beside_other_series_is_inferred_not_stated(tmp_path):
    f = write_vamas_iso(tmp_path / "a.vms", blocks=[
        vamas_iso_block("A", variables=(("Counts", "d"), ("T(E)", "d"))),
        vamas_iso_block("B", variables=(("Counts", "d"),))])
    two, one = VAMASReader(f).read(0).metadata, VAMASReader(f).read(1).metadata
    assert (two.transmission_curve, two.field_origins["transmission_curve"]) == ("none", "inferred")
    assert (one.transmission_curve, one.field_origins["transmission_curve"]) == ("none", "file")


def test_an_unknown_origin_on_disk_is_dropped_with_a_warning(imported):
    with h5py.File(imported.output_path, "a") as h5:
        h5["provenance"].attrs["n_sweeps_origin"] = "guessed"
    with h5py.File(imported.output_path) as h5:
        with pytest.warns(ProvenanceWarning, match="n_sweeps_origin"):
            prov = read_provenance(h5)
    assert "n_sweeps" not in prov.field_origins


def test_series_keep_file_order_past_ten_variables(tmp_path):
    names = tuple((f"v{k}", "d") for k in range(12))
    ords = [str(float(100 * i + k)) for i in range(4) for k in range(12)]
    f = write_vamas_iso(tmp_path / "in.vms", blocks=[
        vamas_iso_block(n_energy=4, variables=names, ordinates=ords)])
    result = import_file(f, tmp_path / "out", ImportConfig(element="C1s", compress=False))
    series = _prov(result.output_path).series
    assert [v.label for v in series.variables] == [f"v{k}" for k in range(12)]
    assert series.variables[10].values[1] == pytest.approx(110.0)


# --- what the re-audit found ------------------------------------------------------

def test_a_declaration_on_a_pxt_region_reaches_the_file(tmp_path):
    """The PXT reader builds a new object on each read; pass the data."""
    from toyomacro.io.readers.pxt_reader import PXTReader

    f = write_pxt(tmp_path / "in.pxt", [(_spectrum_1d(), make_notes(FULL_NOTES))])
    reader = PXTReader(f)
    assert reader.read(0) is not reader.read(0)
    data = reader.read(0)
    data.declare("n_sweeps", 8, "acquisition log")
    data.declare("transmission_applied", "applied", "vendor export setting")
    result = import_file(f, tmp_path / "out", ImportConfig(element="Si2p", compress=False),
                         data=data)
    prov = _prov(result.output_path)
    assert (prov.n_sweeps, prov.field_origins["n_sweeps"]) == (8, "user")
    assert (prov.transmission_applied, prov.transmission_basis) == ("applied", "user")


def test_declaring_no_curve_clears_the_curve_variable(tmp_path):
    data = VAMASReader(write_vamas_iso(tmp_path / "a.vms")).read(0)
    data.declare("transmission_curve", "none", "the column is a different quantity")
    md = data.metadata
    assert md.transmission_curve_variable is None
    assert "transmission_curve_variable" not in md.field_origins
    assert data.transforms[-1].parameters["cleared"] == {
        "transmission_curve_variable": {"previous": "Transmission", "previous_origin": "file"}}


def test_declaring_an_embedded_curve_keeps_its_variable(tmp_path):
    data = VAMASReader(write_vamas_iso(tmp_path / "a.vms")).read(0)
    data.declare("transmission_curve", "embedded", "confirmed with the vendor")
    assert data.metadata.transmission_curve_variable == "Transmission"
    assert data.transforms[-1].parameters["cleared"] == {}


def test_inconsistent_or_empty_transmission_declarations_are_refused(tmp_path):
    data = VAMASReader(write_vamas_iso(tmp_path / "a.vms", blocks=[
        vamas_iso_block(variables=(("Counts", "d"),))])).read(0)
    with pytest.raises(ValueError, match="unknown"):
        data.declare("transmission_applied", "unknown", "not sure")
    with pytest.raises(ValueError, match="embedded"):
        data.declare("transmission_curve_variable", "Counts", "it is the curve")


def test_data_without_a_reader_does_not_parse_the_file_again(tmp_path):
    """Importing from data= neither re-parses (no second warning) nor
    needs the default layout (a legacy-layout read imports as is)."""
    from tests._synthetic_files import write_vamas
    from toyomacro.io.readers.base_reader import ReaderWarning

    f = write_vamas_iso(tmp_path / "in.vms", blocks=[
        vamas_iso_block("A", technique="AES dir"), vamas_iso_block("B", n_energy=12)])
    with pytest.warns(ReaderWarning):
        data = VAMASReader(f).read(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReaderWarning)
        result = import_file(f, tmp_path / "out", ImportConfig(element="C1s", compress=False),
                             data=data)
    assert result.n_regions is None

    legacy = write_vamas(tmp_path / "old.vms", n_energy=16)
    old = VAMASReader(legacy, layout="legacy_fixed").read(0)
    import_file(legacy, tmp_path / "out2", ImportConfig(element="C1s", compress=False), data=old)


def test_a_reader_for_another_file_is_refused(tmp_path):
    f = write_vamas_iso(tmp_path / "in.vms")
    other = write_vamas_iso(tmp_path / "other.vms")
    with pytest.raises(ValueError, match="wrong file"):
        import_file(other, tmp_path / "out", ImportConfig(element="C1s", compress=False),
                    reader=VAMASReader(f))


def test_a_numpy_value_set_by_a_reader_is_stored(tmp_path):
    f = write_vamas_iso(tmp_path / "in.vms", blocks=[vamas_iso_block(n_energy=12)])
    data = VAMASReader(f).read(0)
    data.metadata.n_sweeps = np.int64(3)  # as a reader might set it
    result = import_file(f, tmp_path / "out", ImportConfig(element="C1s", compress=False),
                         data=data)
    assert _prov(result.output_path).n_sweeps == 3


def test_a_value_that_cannot_be_stored_warns_and_writes_no_origin(tmp_path):
    f = write_vamas_iso(tmp_path / "in.vms", blocks=[vamas_iso_block(n_energy=12)])
    data = VAMASReader(f).read(0)
    data.metadata.n_sweeps = [3]  # not a scalar
    with pytest.warns(ProvenanceWarning, match="cannot be stored"):
        result = import_file(f, tmp_path / "out", ImportConfig(element="C1s", compress=False),
                             data=data)
    prov = _prov(result.output_path)
    assert prov.n_sweeps is None and "n_sweeps" not in prov.field_origins
