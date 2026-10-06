"""VAMAS (ISO 14976) reading: structure from the counts, scope, skipped blocks.

Offline tests build files with ``tests._synthetic_files.write_vamas_iso``
in the layout CasaXPS exports. One optional test reads the 15 public files
of Zenodo 10.5281/zenodo.7074887 (CC BY 4.0) when a local copy of the
archive is present; it is never downloaded by the test suite.
"""

from __future__ import annotations

import hashlib
import os
import warnings
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests._synthetic_files import vamas_iso_block, write_vamas_iso
from tests._testdata import testdata_root as _testdata_root
from toyomacro.io import ReaderWarning, RegionImportWarning
from toyomacro.io.importer import ImportConfig, ensure_h5, import_file
from toyomacro.io.provenance import ProvenanceWarning, read_provenance
from toyomacro.io.readers.vamas_reader import VAMASReader

# --- structure ---------------------------------------------------------------


def test_counts_not_line_positions_define_the_structure(tmp_path):
    """Variable comment lines, empty fields and two variables per block."""
    blocks = [
        vamas_iso_block("F 1s/2", n_energy=5, comments=(), transition=""),
        vamas_iso_block("C 1s/3", n_energy=7, comments=tuple(f"c{i}" for i in range(15)),
                        transition="1s"),
    ]
    f = write_vamas_iso(tmp_path / "a.vms", blocks=blocks)
    r = VAMASReader(f)
    assert r.n_regions == 2 and r.region_names == ["F 1s/2", "C 1s/3"]
    assert r.skipped_blocks == ()
    d = r.read(1)
    assert d.energy.shape == (7,) and d.energy[0] == pytest.approx(1186.6)
    assert [(v.label, v.unit) for v in d.corresponding_variables] == [
        ("Counts", "d"), ("Transmission", "d")]
    np.testing.assert_array_equal(d.specdata[:, 0], d.corresponding_variables[0].values)
    assert d.corresponding_variables[0].values[1] == pytest.approx(1010.5)
    assert d.corresponding_variables[1].values[1] == pytest.approx(0.901)
    assert d.metadata.vendor_metadata["transition_label"] == "1s"
    assert r.read(0).metadata.vendor_metadata["transition_label"] == ""


def test_header_facts_are_read_and_not_known_values_are_none(tmp_path):
    f = write_vamas_iso(tmp_path / "a.vms", blocks=[vamas_iso_block(
        source_energy="1486.68", pass_energy="20", work_function="1e+037",
        n_scans="2", collection_time="0.24")])
    md = VAMASReader(f).read(0).metadata
    assert md.excitation_energy == pytest.approx(1486.68)
    assert md.pass_energy == pytest.approx(20.0)
    assert md.n_sweeps == 2
    assert md.signal_collection_time == pytest.approx(0.24)
    assert md.signal_mode == "pulse counting"
    assert md.energy_scale == "Kinetic"
    assert md.vendor_metadata["work_function_eV"] is None
    # The stored numbers' meaning is not inferred from labels or integers.
    assert md.intensity_semantics == "unknown"


def test_transmission_curve_present_does_not_mean_applied(tmp_path):
    f = write_vamas_iso(tmp_path / "a.vms")
    md = VAMASReader(f).read(0).metadata
    assert md.transmission_curve == "embedded"
    assert md.transmission_curve_variable == "Transmission"
    assert md.transmission_applied == "unknown"
    assert md.transmission_basis == "unknown"
    assert md.transmission_curve_normalisation == "unknown"

    f1 = write_vamas_iso(tmp_path / "b.vms", blocks=[vamas_iso_block(
        variables=(("Counts", "d"),))])
    assert VAMASReader(f1).read(0).metadata.transmission_curve == "none"


def test_binding_energy_axis(tmp_path):
    f = write_vamas_iso(tmp_path / "a.vms", blocks=[vamas_iso_block(x_label="Binding Energy")])
    assert VAMASReader(f).read(0).metadata.energy_scale == "Binding"


def test_a_two_digit_year_is_not_given_a_century(tmp_path):
    f = write_vamas_iso(tmp_path / "a.vms", blocks=[vamas_iso_block(year="22")])
    md = VAMASReader(f).read(0).metadata
    assert md.datetime is None
    assert md.vendor_metadata["date_note"] == "two_digit_year_unresolved"
    assert md.vendor_metadata["date_raw"].startswith("22 1 15")


# --- scope and skipped blocks --------------------------------------------------


def test_out_of_scope_blocks_are_listed_not_dropped_silently(tmp_path):
    blocks = [
        vamas_iso_block("C 1s/1"),
        vamas_iso_block("Auger", technique="AES dir"),
        vamas_iso_block("Bad", ordinates=["1", "x"] * 16),
        vamas_iso_block("Wrong axis", x_label="Time"),
        vamas_iso_block("O 1s/5"),
    ]
    f = write_vamas_iso(tmp_path / "a.vms", blocks=blocks)
    r = VAMASReader(f)
    with pytest.warns(ReaderWarning, match="3 block"):
        assert r.region_names == ["C 1s/1", "O 1s/5"]
    assert [(b.index, b.block_id) for b in r.skipped_blocks] == [
        (1, "Auger"), (2, "Bad"), (3, "Wrong axis")]
    assert "not XPS" in r.skipped_blocks[0].reason
    assert "does not parse" in r.skipped_blocks[1].reason
    # The file position survives, so a block can be traced back.
    assert r.read(1).metadata.source_region_index == 4


@pytest.mark.parametrize("mode", [("MAP", "REGULAR"), ("SDP", "REGULAR"), ("NORM", "IRREGULAR")])
def test_unsupported_modes_stop_the_file(tmp_path, mode):
    f = write_vamas_iso(tmp_path / "a.vms", exp_mode=mode[0], scan_mode=mode[1])
    with pytest.raises(ValueError, match="not supported"):
        _ = VAMASReader(f).n_regions


def test_a_broken_count_stops_the_file(tmp_path):
    """A count that does not parse leaves the block boundaries unknown."""
    block = vamas_iso_block()
    i = block.index("Counts") - 1  # the number of corresponding variables
    block[i] = "two"
    f = write_vamas_iso(tmp_path / "a.vms", blocks=[block, vamas_iso_block()])
    with pytest.raises(ValueError, match="not an integer"):
        _ = VAMASReader(f).n_regions


def test_a_truncated_file_stops(tmp_path):
    f = write_vamas_iso(tmp_path / "a.vms")
    lines = f.read_bytes().decode().split("\r\n")
    f.write_bytes("\r\n".join(lines[:-10]).encode())
    with pytest.raises(ValueError, match="file ends"):
        _ = VAMASReader(f).n_regions


def test_a_missing_terminator_warns(tmp_path):
    f = write_vamas_iso(tmp_path / "a.vms", terminator="")
    with pytest.warns(ReaderWarning, match="end of experiment"):
        assert VAMASReader(f).n_regions == 1


def test_lines_left_after_the_last_block_stop_the_file(tmp_path):
    """Two extra lines: a count was wrong and the last block may be cut short."""
    block = vamas_iso_block(n_energy=8)
    i = block.index("16")  # number of ordinate values, 8 points x 2 variables
    block[i] = "14"
    f = write_vamas_iso(tmp_path / "a.vms", blocks=[block])
    with pytest.raises(ValueError, match="remain after 1 blocks"):
        _ = VAMASReader(f).n_regions


def test_an_ordinate_count_that_does_not_fit_the_variables_stops_the_file(tmp_path):
    block = vamas_iso_block(n_energy=8)
    block[block.index("16")] = "15"
    f = write_vamas_iso(tmp_path / "a.vms", blocks=[block, vamas_iso_block()])
    with pytest.raises(ValueError, match="block boundary cannot be determined"):
        _ = VAMASReader(f).n_regions


def test_manually_entered_items_are_consumed(tmp_path):
    f = write_vamas_iso(tmp_path / "a.vms", manual_items=("44", "0"))
    assert VAMASReader(f).n_regions == 1


def test_a_parameter_inclusion_list_stops_the_file(tmp_path):
    f = write_vamas_iso(tmp_path / "a.vms", inclusion=("3",))
    with pytest.raises(ValueError, match="inclusion"):
        _ = VAMASReader(f).n_regions


def test_a_sputter_ion_technique_stops_the_file_and_says_so(tmp_path):
    f = write_vamas_iso(tmp_path / "a.vms", blocks=[vamas_iso_block(technique="SIMS")])
    with pytest.raises(ValueError, match="technique SIMS"):
        _ = VAMASReader(f).n_regions


def test_an_aes_diff_block_is_skipped_and_the_next_block_still_reads(tmp_path):
    """AES diff carries one extra field; miscounting it would shift every
    later block."""
    f = write_vamas_iso(tmp_path / "a.vms", blocks=[
        vamas_iso_block("A", technique="AES diff"), vamas_iso_block("B", n_energy=9)])
    r = VAMASReader(f)
    with pytest.warns(ReaderWarning):
        assert r.region_names == ["B"]
    assert r.read(0).energy.size == 9


def test_a_retard_ratio_is_not_a_pass_energy(tmp_path):
    f = write_vamas_iso(tmp_path / "a.vms", blocks=[vamas_iso_block(
        analyser_mode="FRR", pass_energy="4")])
    md = VAMASReader(f).read(0).metadata
    assert md.pass_energy is None
    assert md.vendor_metadata["retard_ratio"] == 4.0


def test_a_non_finite_ordinate_rejects_the_block(tmp_path):
    ords = [str(1000.0 + i) for i in range(16)]
    ords[3] = "nan"
    f = write_vamas_iso(tmp_path / "a.vms", blocks=[
        vamas_iso_block("A", variables=(("Counts", "d"),), ordinates=ords),
        vamas_iso_block("B")])
    r = VAMASReader(f)
    with pytest.warns(ReaderWarning):
        assert r.region_names == ["B"]
    assert "non-finite" in r.skipped_blocks[0].reason


def test_an_unreadable_scan_count_is_kept_raw(tmp_path):
    f = write_vamas_iso(tmp_path / "a.vms", blocks=[vamas_iso_block(n_scans="x")])
    md = VAMASReader(f).read(0).metadata
    assert md.vendor_metadata["n_scans_raw"] == "x"


def test_errors_do_not_repeat_comment_text(tmp_path):
    """Comments hold paths and names; a misaligned read must not echo them."""
    f = write_vamas_iso(tmp_path / "a.vms")
    lines = f.read_bytes().decode().split("\r\n")
    lines[5] = "3"  # understate the four file comment lines
    f.write_bytes("\r\n".join(lines).encode())
    with pytest.raises(ValueError) as err:
        _ = VAMASReader(f).n_regions
    assert "someone" not in str(err.value) and "Users" not in str(err.value)


def test_unknown_layout_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="layout"):
        VAMASReader(write_vamas_iso(tmp_path / "a.vms"), layout="omicron")


# --- privacy and the importer ------------------------------------------------


def test_comments_with_paths_are_never_written_to_hdf5(tmp_path):
    f = write_vamas_iso(tmp_path / "a.vms")
    md = VAMASReader(f).read(0).metadata
    assert any("someone" in line for line in md.vendor_metadata["unparsed_notes"])
    with pytest.warns(ProvenanceWarning, match="never persisted"):
        result = import_file(f, tmp_path / "out", ImportConfig(
            element="C1s", compress=False, persist_vendor_metadata=True,
            vendor_metadata_allowlist=("unparsed_notes", "block_id")))
    with h5py.File(result.output_path) as h5:
        prov = read_provenance(h5)
    assert "unparsed_notes" not in (prov.vendor_metadata or {})
    assert b"someone" not in result.output_path.read_bytes()


def test_ensure_h5_writes_one_file_per_block_when_names_normalise_alike(tmp_path):
    """Kratos-style ids ("C 1s/3", "C 1s/6") and ids that differ only in
    spelling ("C1s", "C 1s", "C_1s") all normalise to C1s."""
    ids = ["F 1s/2", "C 1s/3", "F 1s/5", "C 1s/6", "C 1s/9", "C1s", "C 1s", "C_1s"]
    f = write_vamas_iso(tmp_path / "PVC.vms", blocks=[vamas_iso_block(b) for b in ids])
    out = tmp_path / "out"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        paths = ensure_h5(f, output_dir=out)
    assert len(set(paths)) == len(ids) == len(list(out.glob("*.h5")))
    with h5py.File(out / "C1s_6_PVC.h5") as h5:
        assert read_provenance(h5).source_region_index == 3


def test_a_block_id_cannot_leave_the_output_directory(tmp_path):
    f = write_vamas_iso(tmp_path / "a.vms", blocks=[
        vamas_iso_block("../../escaped"), vamas_iso_block("Survey/1")])
    out = tmp_path / "deep" / "out"
    paths = ensure_h5(f, output_dir=out)
    assert len(paths) == 2
    assert all(p.parent == out for p in paths)
    assert not list(tmp_path.glob("*.h5"))


def test_ensure_h5_reports_a_region_it_could_not_import(tmp_path, monkeypatch):
    import toyomacro.io.importer as importer

    f = write_vamas_iso(tmp_path / "a.vms", blocks=[
        vamas_iso_block("C 1s/1"), vamas_iso_block("O 1s/2")])
    real = importer.import_file

    def failing(path, out, config, **kwargs):
        if config.region_index == 1:
            raise RuntimeError("disk full")
        return real(path, out, config, **kwargs)

    monkeypatch.setattr(importer, "import_file", failing)
    with pytest.warns(RegionImportWarning, match=r"region 1 \('O_1s_2'\).*disk full"):
        paths = ensure_h5(f, output_dir=tmp_path / "out")
    assert len(paths) == 1


# --- public data (optional, local copy only) ----------------------------------

#: SHA-256 of "Degradation Polymers.zip", Zenodo record 7074887 v1.
POLYMER_ZIP_SHA256 = "99fea65bca138543dbac2f377ef4c4c0ce2317e5b78dcd9cfa71d108d8e81453"


def _polymer_zip() -> Path | None:
    env = os.environ.get("TOYOMACRO_VAMAS_POLYMER_ZIP")
    path = Path(env) if env else _testdata_root() / "zenodo-7074887" / "Degradation Polymers.zip"
    return path if path.is_file() else None


@pytest.mark.skipif(_polymer_zip() is None, reason=(
    "Zenodo 7074887 archive not present; set TOYOMACRO_VAMAS_POLYMER_ZIP"))
def test_the_15_public_files(tmp_path):
    archive = _polymer_zip()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == POLYMER_ZIP_SHA256
    blocks = {}
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".vms"))
        assert len(names) == 15
        for i, name in enumerate(names):
            path = tmp_path / f"f{i}.vms"
            path.write_bytes(z.read(name))
            with warnings.catch_warnings():
                warnings.simplefilter("error", ReaderWarning)
                r = VAMASReader(path)
                blocks[name] = r.n_regions
                assert r.skipped_blocks == ()
            if name == "Kratos Axis Ultra/PTFE.vms":
                f1s, c1s = r.read(0), r.read(1)
            if name == "Thermo KAlpha/PVC.vms":
                thermo = r.read(1)
    assert blocks == {
        "Kratos Axis Ultra/PTFE.vms": 132, "Kratos Axis Ultra/PVC.vms": 132,
        "Kratos Axis Ultra/PVF.vms": 132, "Kratos Axis Ultra/PVdCl.vms": 132,
        "Kratos Axis Ultra/PVdF.vms": 132, "Thermo KAlpha/PTFE1.vms": 502,
        "Thermo KAlpha/PVC.vms": 502, "Thermo KAlpha/PVF.vms": 502,
        "Thermo KAlpha/PVdCl.vms": 502, "Thermo KAlpha/PVdF.vms": 502,
        "Thermo KAlpha/pc4s iteration.vms": 502, "Thermo KAlpha/ppg iteration.vms": 502,
        "Thermo KAlpha/ptfe 30v iteration.vms": 502, "Thermo KAlpha/ptfe 40v t60.vms": 60,
        "Thermo KAlpha/pvp.vms": 753,
    }
    # The example's pair (ASTRA_ANSWERS Q1): source blocks 0 and 1.
    assert (f1s.metadata.region, c1s.metadata.region) == ("F 1s/2", "C 1s/3")
    assert (f1s.energy.size, c1s.energy.size) == (301, 251)
    assert f1s.metadata.pass_energy == 20.0 and f1s.metadata.n_sweeps == 4
    assert (f1s.metadata.signal_collection_time, c1s.metadata.signal_collection_time) == (0.2, 0.24)
    assert f1s.metadata.vendor_metadata["experimental_variables"] == {"Exp Variable": "0"}
    # A Thermo spot check: second block, Cl 2p, 201 points from 1281.6 eV KE.
    md = thermo.metadata
    assert (md.region, md.vendor_metadata["species_label"]) == ("Cl2p Scan", "Cl 2p")
    assert thermo.energy.size == 201
    assert (md.excitation_energy, md.pass_energy, md.n_sweeps) == (1486.68, 50.0, 4)
    assert [v.label for v in thermo.corresponding_variables] == ["Counts", "Transmission"]
