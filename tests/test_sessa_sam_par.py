"""Reading IMFP/TRMFP pairs out of a SESSA ``sam_par.txt``.

Every number in these fixtures is invented. No fragment of a real SESSA
output file is reproduced here: what is under test is the reader's
handling of the *format* — which column is which, what unit it declares,
what happens when a column is missing — not any SESSA-computed value.
The lengths were chosen to make the albedo come out at round numbers so
a wrong pairing is visible by eye.
"""

from __future__ import annotations

import pytest

from toyomacro.data.elastic_scattering import (
    overlayer_eal_report,
    single_scattering_albedo,
)
from toyomacro.data.sessa import (
    read_remarks,
    read_sam_par,
    select_row,
    sessa_source,
)

# Invented. Layer 1 is built so that omega = 20/(20+80) = 0.2 exactly and
# layer 2 so that omega = 5/(5+95) = 0.05 exactly.
TWO_LAYERS_TWO_PEAKS = """\
-sample interaction parameters----------------------------------------
-----------------------------------------------------------------------
 #interaction parameters for peak: 1 xx1s
 #lay |peak_no |peak_id  |Energy [eV]  |IMFP [A]   |EMFP [A]   |TRMFP [A]  |SEP [sqrt(1/eV)]|
  1   |  1     | xx1s    | 1000.00000  | 20.00000  [  1]| 12.00000  [  2]| 80.00000  [  2]|  0.30000  [  3]|
  2   |  1     | xx1s    | 1000.00000  |  5.00000  [  1]|  9.00000  [  2]| 95.00000  [  2]|  0.10000  [  3]|
-----------------------------------------------------------------------
 #interaction parameters for peak: 2 yy2p3
 #lay |peak_no |peak_id  |Energy [eV]  |IMFP [A]   |EMFP [A]   |TRMFP [A]  |SEP [sqrt(1/eV)]|
  1   |  2     | yy2p3   |  500.00000  | 10.00000  [  1]|  6.00000  [  2]| 40.00000  [  2]|  0.30000  [  3]|
  2   |  2     | yy2p3   |  500.00000  |  4.00000  [  1]|  7.00000  [  2]| 36.00000  [  2]|  0.10000  [  3]|
-----------------------------------------------------------------------
"""

# The same first block with the EMFP and SEP columns dropped: nothing
# here needs them.
NO_OPTIONAL_COLUMNS = """\
 #interaction parameters for peak: 1 xx1s
 #lay |peak_no |peak_id  |Energy [eV]  |IMFP [A]   |TRMFP [A]  |
  1   |  1     | xx1s    | 1000.00000  | 20.00000  [  1]| 80.00000  [  2]|
"""

# Two peaks of one layer whose kinetic energies differ by 0.2%, the way
# lines excited by one photon energy actually do. Invented values.
NEAR_DEGENERATE_PEAKS = """\
 #interaction parameters for peak: 1 aa4f7
 #lay |peak_no |peak_id  |Energy [eV]  |IMFP [A]   |TRMFP [A]  |
  1   |  1     | aa4f7   | 9000.00000  | 20.00000  [  1]| 80.00000  [  2]|
 #interaction parameters for peak: 2 bb1s
 #lay |peak_no |peak_id  |Energy [eV]  |IMFP [A]   |TRMFP [A]  |
  1   |  2     | bb1s    | 8982.00000  | 19.90000  [  1]| 40.00000  [  2]|
"""

# Invented remarks in SESSA's layout: a tag, then free text that may run
# over several lines. The wording is made up, not SESSA's own.
REMARKS = """\
[1]
An invented note about the inelastic mean free path, long enough to
wrap onto a second line.
[2]
 An invented note about elastic scattering.
[3]  An invented note with no continuation line.
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.fixture
def sam_par(tmp_path):
    return write(tmp_path, "runsam_par.txt", TWO_LAYERS_TWO_PEAKS)


# --- the format -----------------------------------------------------


def test_reads_every_layer_of_every_peak(sam_par):
    rows = read_sam_par(sam_par)
    assert [(r.layer, r.peak_no, r.peak_id) for r in rows] == [
        (1, 1, "xx1s"), (2, 1, "xx1s"), (1, 2, "yy2p3"), (2, 2, "yy2p3"),
    ]


def test_columns_are_mapped_by_name_not_position(sam_par):
    """IMFP, EMFP and TRMFP are adjacent and easy to swap."""
    row = select_row(read_sam_par(sam_par), layer=1, peak_id="xx1s")
    assert row.kinetic_energy_ev == 1000.0
    assert row.imfp == 20.0
    assert row.emfp == 12.0
    assert row.trmfp == 80.0
    assert row.surface_excitation_parameter == 0.3


def test_reference_tags_are_kept_per_quantity(sam_par):
    """The bracketed tags say where each number came from."""
    row = select_row(read_sam_par(sam_par), layer=1, peak_id="xx1s")
    assert (row.imfp_ref, row.emfp_ref, row.trmfp_ref, row.sep_ref) == (
        1, 2, 2, 3)


def test_optional_columns_may_be_absent(tmp_path):
    path = write(tmp_path, "sam_par.txt", NO_OPTIONAL_COLUMNS)
    (row,) = read_sam_par(path)
    assert (row.imfp, row.trmfp) == (20.0, 80.0)
    assert row.emfp is None
    assert row.surface_excitation_parameter is None


def test_a_file_without_a_trmfp_column_is_refused(tmp_path):
    """Half a pair is not a pair; there is no albedo to form."""
    path = write(tmp_path, "sam_par.txt",
                 NO_OPTIONAL_COLUMNS.replace("|TRMFP [A]  ", ""))
    with pytest.raises(ValueError, match="trmfp"):
        read_sam_par(path)


def test_a_short_data_row_is_refused_rather_than_shifted(tmp_path):
    """Dropping a value would silently slide TRMFP into the IMFP slot."""
    text = TWO_LAYERS_TWO_PEAKS.replace(
        "  1   |  1     | xx1s    | 1000.00000  | 20.00000  [  1]",
        "  1   |  1     | xx1s    | 1000.00000  ")
    path = write(tmp_path, "sam_par.txt", text)
    with pytest.raises(ValueError, match="fields but the header declares"):
        read_sam_par(path)


def test_a_file_with_no_rows_is_refused(tmp_path):
    path = write(tmp_path, "sam_par.txt", "nothing to see here\n")
    with pytest.raises(ValueError, match="no interaction-parameter rows"):
        read_sam_par(path)


def test_a_repeated_layer_and_peak_is_refused(tmp_path):
    """Two runs concatenated cannot be told apart row by row."""
    path = write(tmp_path, "sam_par.txt",
                 TWO_LAYERS_TWO_PEAKS + TWO_LAYERS_TWO_PEAKS)
    with pytest.raises(ValueError, match="already read at line"):
        read_sam_par(path)


# --- units ----------------------------------------------------------


def test_the_unit_is_read_from_the_header(sam_par):
    rows = read_sam_par(sam_par)
    assert all(row.unit == "angstrom" for row in rows)
    assert all(length.unit == "angstrom"
               for row in rows for length in row.lengths())


@pytest.mark.parametrize("declared", ["A", "a", "Å", "Å",
                                      "Angstrom", "angstrom"])
def test_every_spelling_of_angstrom_is_accepted(tmp_path, declared):
    """Both Unicode angstroms included -- they are different code points."""
    path = write(tmp_path, f"sam_par_{declared.encode('unicode_escape')}.txt",
                 NO_OPTIONAL_COLUMNS.replace("IMFP [A]", f"IMFP [{declared}]")
                 .replace("TRMFP [A]", f"TRMFP [{declared}]"))
    (row,) = read_sam_par(path)
    assert row.unit == "angstrom"
    assert row.imfp == 20.0


def test_an_unrecognised_length_unit_stops_the_read(tmp_path):
    """Not read as Angstrom on the grounds that SESSA usually writes it."""
    path = write(tmp_path, "sam_par.txt",
                 NO_OPTIONAL_COLUMNS.replace("IMFP [A]", "IMFP [bohr]"))
    with pytest.raises(ValueError, match="does not recognise"):
        read_sam_par(path)


def test_an_energy_column_not_in_ev_stops_the_read(tmp_path):
    path = write(tmp_path, "sam_par.txt",
                 NO_OPTIONAL_COLUMNS.replace("Energy [eV]", "Energy [keV]"))
    with pytest.raises(ValueError, match="not eV"):
        read_sam_par(path)


def test_conversion_to_nm_scales_lengths_and_leaves_the_albedo_alone(sam_par):
    """This package's own IMFPs are in nm, so the pair often must be."""
    row = select_row(read_sam_par(sam_par, unit="nm"), layer=1, peak_id="xx1s")
    assert row.unit == "nm"
    assert row.imfp == pytest.approx(2.0)
    assert row.trmfp == pytest.approx(8.0)
    assert row.emfp == pytest.approx(1.2)
    assert single_scattering_albedo(row.imfp, row.trmfp) == pytest.approx(
        0.2, rel=1e-12)
    # Not a length; not scaled.
    assert row.surface_excitation_parameter == 0.3


def test_an_unknown_output_unit_is_refused(sam_par):
    with pytest.raises(ValueError, match="must be 'nm' or 'angstrom'"):
        read_sam_par(sam_par, unit="bohr")


# --- declarations the file does not contain -------------------------


def test_materials_come_from_the_caller_and_default_to_undeclared(sam_par):
    plain = read_sam_par(sam_par)
    assert all(row.material is None for row in plain)

    declared = read_sam_par(sam_par, layer_materials={1: "Au", 2: "Si"})
    assert {row.layer: row.material for row in declared} == {1: "Au", 2: "Si"}


def test_a_partial_material_declaration_leaves_the_rest_undeclared(sam_par):
    rows = read_sam_par(sam_par, layer_materials={1: "Au"})
    assert {row.layer: row.material for row in rows} == {1: "Au", 2: None}


def test_a_material_for_a_layer_the_file_lacks_is_refused(sam_par):
    """SESSA numbers layers from 1; a 0-based mapping is a real mistake."""
    with pytest.raises(ValueError, match="does not contain"):
        read_sam_par(sam_par, layer_materials={0: "Au", 1: "Si"})


def test_the_source_string_states_only_what_was_declared():
    assert sessa_source() == "SESSA"
    assert sessa_source(version="2.2.2") == "SESSA 2.2.2"
    assert sessa_source(imfp_model="JTP") == "SESSA (JTP)"
    assert sessa_source(version="2.2.2", imfp_model="JTP") == (
        "SESSA 2.2.2 (JTP)")


def test_the_source_reaches_both_lengths_and_names_the_row(sam_par):
    row = select_row(
        read_sam_par(sam_par, version="2.2.2", imfp_model="JTP"),
        layer=1, peak_id="xx1s")
    assert row.source == "SESSA 2.2.2 (JTP)"
    imfp, trmfp = row.lengths()
    assert imfp.source == trmfp.source == row.row_source()
    assert imfp.source == "SESSA 2.2.2 (JTP); layer 1, peak 1 xx1s"


def test_an_explicit_source_cannot_be_combined_with_its_parts(sam_par):
    with pytest.raises(ValueError, match="not both"):
        read_sam_par(sam_par, source="SESSA 2.2.2", version="2.2.2")


# --- selecting a row ------------------------------------------------


def test_a_peak_id_alone_does_not_identify_a_row(sam_par):
    """It repeats in every layer, which is the whole point of the file."""
    rows = read_sam_par(sam_par)
    assert len([r for r in rows if r.peak_id == "xx1s"]) == 2
    assert select_row(rows, layer=2, peak_id="xx1s").imfp == 5.0
    assert select_row(rows, layer=2, peak_no=1).imfp == 5.0


def test_selecting_something_absent_raises_with_what_is_there(sam_par):
    with pytest.raises(ValueError, match=r"available \(layer, peak_id\)"):
        select_row(read_sam_par(sam_par), layer=3, peak_id="xx1s")


def test_selecting_needs_a_peak(sam_par):
    with pytest.raises(ValueError, match="peak_id or peak_no"):
        select_row(read_sam_par(sam_par), layer=1)


# --- the pair guarantee, and the way through to a report ------------


def test_both_lengths_come_from_one_row(sam_par):
    """Spreading one row cannot pair two different electrons."""
    row = select_row(read_sam_par(sam_par), layer=1, peak_id="xx1s")
    imfp, trmfp = row.lengths()
    assert imfp.value == 20.0
    assert trmfp.value == 80.0
    assert imfp.unit == trmfp.unit
    assert imfp.kinetic_energy_ev == trmfp.kinetic_energy_ev == 1000.0
    assert imfp.material == trmfp.material


def test_a_pair_split_across_layers_is_flagged(sam_par):
    """The one mispairing the other checks cannot see.

    SESSA writes the same kinetic energy for a peak in every layer, so
    the energy check is blind to it, and an overlayer on a substrate of
    the same material passes the material check too. Only the row in
    the source string distinguishes them.
    """
    rows = read_sam_par(sam_par, layer_materials={1: "Au", 2: "Au"},
                        version="2.2.2", imfp_model="JTP")
    top = select_row(rows, layer=1, peak_id="xx1s")
    bottom = select_row(rows, layer=2, peak_id="xx1s")
    assert top.kinetic_energy_ev == bottom.kinetic_energy_ev

    report = overlayer_eal_report(
        top.lengths()[0], bottom.lengths()[1], model="jp2020_unpolarized")

    record = report.to_dict()
    assert record["consistency"]["kinetic_energy"] == "consistent"
    assert record["consistency"]["material"] == "consistent"
    assert report.source_consistency == "inconsistent"
    assert any("source" in w for w in report.warnings)


def test_a_pair_split_across_widely_separated_peaks_raises(sam_par):
    """With energies 2x apart the kinetic-energy check catches it first."""
    rows = read_sam_par(sam_par, version="2.2.2", imfp_model="JTP")
    first = select_row(rows, layer=1, peak_id="xx1s")
    second = select_row(rows, layer=1, peak_id="yy2p3")
    with pytest.raises(ValueError, match="same electron"):
        overlayer_eal_report(
            first.lengths()[0], second.lengths()[1],
            model="jp2020_unpolarized")


def test_a_pair_split_across_near_degenerate_peaks_is_flagged(tmp_path):
    """The realistic case: the energy check does none of the work.

    Lines from one photon energy sit close together — at Ga K-alpha the
    Au and Si lines of the sibling HAXPES work are within a few tenths
    of a percent — so the 1% kinetic-energy tolerance is silent here by
    design, and the row in the source string is the only thing left.
    """
    path = write(tmp_path, "sam_par.txt", NEAR_DEGENERATE_PEAKS)
    rows = read_sam_par(path, layer_materials={1: "Au"}, version="2.2.2")
    a = select_row(rows, layer=1, peak_id="aa4f7")
    b = select_row(rows, layer=1, peak_id="bb1s")
    assert abs(a.kinetic_energy_ev - b.kinetic_energy_ev) < 0.01 * (
        a.kinetic_energy_ev)

    report = overlayer_eal_report(
        a.lengths()[0], b.lengths()[1], model="jp2020_polarized_haxpes")

    record = report.to_dict()
    assert record["consistency"]["kinetic_energy"] == "consistent"
    assert record["consistency"]["material"] == "consistent"
    assert report.source_consistency == "inconsistent"


def test_two_runs_under_one_label_are_not_told_apart(tmp_path):
    """A stated limit, pinned so it cannot quietly become a surprise.

    The row is (layer, peak_no, peak_id) and carries no file identity,
    so the same row of two different runs produces the same source
    string. Declaring the runs differently restores the distinction,
    which is what the module docstring tells the caller to do.
    """
    one = write(tmp_path, "one_sam_par.txt", TWO_LAYERS_TWO_PEAKS)
    two = write(tmp_path, "two_sam_par.txt",
                TWO_LAYERS_TWO_PEAKS.replace("| 80.00000", "| 30.00000"))
    a = select_row(read_sam_par(one, version="2.2.2"), layer=1,
                   peak_id="xx1s")
    b = select_row(read_sam_par(two, version="2.2.2"), layer=1,
                   peak_id="xx1s")
    assert a.trmfp != b.trmfp
    undetected = overlayer_eal_report(
        a.lengths()[0], b.lengths()[1], model="jp2020_unpolarized")
    assert undetected.source_consistency == "consistent"

    # The documented remedy.
    labelled = select_row(
        read_sam_par(two, source="SESSA 2.2.2, second run"),
        layer=1, peak_id="xx1s")
    detected = overlayer_eal_report(
        a.lengths()[0], labelled.lengths()[1], model="jp2020_unpolarized")
    assert detected.source_consistency == "inconsistent"


def test_a_row_feeds_overlayer_eal_report_with_every_check_satisfied(sam_par):
    rows = read_sam_par(sam_par, layer_materials={1: "Au", 2: "Si"},
                        version="2.2.2", imfp_model="JTP")
    row = select_row(rows, layer=1, peak_id="xx1s")

    report = overlayer_eal_report(
        *row.lengths(), model="jp2020_unpolarized", emission_angle_deg=45.0)

    assert report.albedo == pytest.approx(0.2, rel=1e-12)
    assert report.ratio == pytest.approx(1.0 - 0.738 * 0.2, rel=1e-12)
    assert report.value == pytest.approx(20.0 * (1.0 - 0.738 * 0.2), rel=1e-12)
    assert report.unit == "angstrom"
    assert report.source_consistency == "consistent"
    assert report.emission_angle_check == "within"
    assert report.warnings == ()

    record = report.to_dict()
    assert record["consistency"] == {
        "material": "consistent",
        "kinetic_energy": "consistent",
        "source": "consistent",
    }
    assert record["inputs"]["imfp"]["material"] == "Au"


def test_an_undeclared_material_is_reported_as_such_not_as_a_pass(sam_par):
    row = select_row(read_sam_par(sam_par), layer=1, peak_id="xx1s")
    record = overlayer_eal_report(
        *row.lengths(), model="jp2020_unpolarized").to_dict()
    assert record["consistency"]["material"] == "not_recorded"
    # The energy is in the file, so that one is a real check.
    assert record["consistency"]["kinetic_energy"] == "consistent"


def test_a_second_peak_carries_its_own_energy_and_albedo(sam_par):
    """One file, two electrons: the pairing must follow the peak."""
    rows = read_sam_par(sam_par)
    row = select_row(rows, layer=1, peak_id="yy2p3")
    imfp, trmfp = row.lengths()
    assert imfp.kinetic_energy_ev == 500.0
    assert single_scattering_albedo(imfp.value, trmfp.value) == pytest.approx(
        0.2, rel=1e-12)


# --- rems.txt -------------------------------------------------------


def test_remarks_resolve_the_tags_the_rows_carry(tmp_path, sam_par):
    remarks = read_remarks(write(tmp_path, "runrems.txt", REMARKS))
    row = select_row(read_sam_par(sam_par), layer=1, peak_id="xx1s")

    assert set(remarks) == {1, 2, 3}
    assert "inelastic mean free path" in remarks[row.imfp_ref]
    assert "elastic scattering" in remarks[row.trmfp_ref]
    # A remark running over two lines is joined, not truncated.
    assert remarks[1].endswith("wrap onto a second line.")
