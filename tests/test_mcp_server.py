"""Tests for the MCP server tools (called directly, no MCP transport)."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mcp")

from toyomacro import mcp_server  # noqa: E402


class TestReferenceDataTools:
    def test_lookup_binding_energy(self):
        out = json.loads(mcp_server.lookup_binding_energy("Si", "2p"))
        assert out["binding_energy_eV"] == pytest.approx(99.0, abs=1.0)

    def test_spin_orbit_split_is_the_reference_value(self):
        """Si 2p: 0.6 from the 0.1 eV table, where the integer table's 1.0
        was reported before."""
        out = json.loads(mcp_server.lookup_binding_energy("Si", "2p"))
        assert out["spin_orbit_split_eV"] == 0.6
        so = out["spin_orbit"]
        assert (so["precision_eV"], so["source_table"], so["state"]) == (
            0.1, "trzh2018", "provisional")

    def test_spin_orbit_state_is_reported_where_unsettled(self):
        ga = json.loads(mcp_server.lookup_binding_energy("Ga", "3d"))["spin_orbit"]
        assert (ga["split_eV"], ga["state"]) == (5.0, "to_verify")
        # No position for a value that is to be verified; raw value nested
        flat = json.loads(mcp_server.lookup_binding_energy("Ga", "3d"))
        assert flat["spin_orbit_split_eV"] is None
        assert "disagrees" in ga["state_reason"]
        s = json.loads(mcp_server.lookup_binding_energy("O", "1s"))
        assert s["spin_orbit_split_eV"] is None
        assert s["spin_orbit"]["state"] == "not_covered"

    def test_the_nested_entry_is_the_reference_entry(self):
        from toyomacro.data import BindingEnergy

        so = json.loads(mcp_server.lookup_binding_energy("Si", "2p"))["spin_orbit"]
        ref = BindingEnergy.get_spin_orbit("Si", "2p")
        assert so == {
            "j_plus": "2p3/2", "j_minus": "2p1/2", "split_eV": 0.6,
            "precision_eV": 0.1, "source_table": "trzh2018",
            "be_j_plus_eV": 98.9, "be_j_minus_eV": 99.5,
            "state": "provisional", "state_reason": ref.state_reason,
            "unresolved_at_precision": False, "statistical_ratio": 0.5,
        }

    def test_a_carried_pair_gives_no_headline(self):
        """Lu 2p: the binding_energy.json pair, not_covered."""
        out = json.loads(mcp_server.lookup_binding_energy("Lu", "2p"))
        assert out["spin_orbit_split_eV"] is None
        assert (out["spin_orbit"]["split_eV"], out["spin_orbit"]["state"]) == (
            1105.0, "not_covered")

    def test_the_headline_is_never_zero_negative_or_unsettled(self):
        from toyomacro.data import BindingEnergy

        given = 0
        for el in BindingEnergy.get_all_elements():
            for n in range(1, 8):
                for letter in "spdf":
                    out = json.loads(mcp_server.lookup_binding_energy(el, f"{n}{letter}"))
                    head = out.get("spin_orbit_split_eV")
                    if head is None:
                        continue
                    given += 1
                    so = out["spin_orbit"]
                    assert head > 0 and head == so["split_eV"]
                    assert so["state"] == "provisional" and not so["unresolved_at_precision"]
        assert given == 258

    @pytest.mark.parametrize("label", ["2p ", "2p\t", "2p\n"])
    def test_a_padded_label_is_still_answered(self, label):
        out = json.loads(mcp_server.lookup_binding_energy("Si", label))
        assert out["binding_energy_eV"] == 99.0 and out["spin_orbit_split_eV"] == 0.6

    @pytest.mark.parametrize("label", ["2p3", "2p1"])
    def test_a_label_the_reference_cannot_read_is_answered_without_it(self, label):
        """The binding-energy lookup accepts these; the answer must stay a
        JSON result, not an exception."""
        out = json.loads(mcp_server.lookup_binding_energy("Si", label))
        assert out["binding_energy_eV"] == 99.0
        assert out == {"element": "Si", "orbital": label, "binding_energy_eV": 99.0,
                       "spin_orbit_split_eV": None, "spin_orbit": None}

    def test_a_pair_printed_equal_is_not_reported_as_zero(self):
        out = json.loads(mcp_server.lookup_binding_energy("Am", "5f"))
        assert out["spin_orbit_split_eV"] is None
        so = out["spin_orbit"]
        assert (so["split_eV"], so["state"], so["unresolved_at_precision"]) == (
            0.0, "to_verify", True)

    def test_a_negative_pair_keeps_its_sign_in_the_nested_object(self):
        out = json.loads(mcp_server.lookup_binding_energy("Zn", "3d"))
        assert out["spin_orbit_split_eV"] is None
        assert (out["spin_orbit"]["split_eV"], out["spin_orbit"]["state"]) == (
            -0.1, "to_verify")

    def test_lookup_binding_energy_unknown(self):
        out = json.loads(mcp_server.lookup_binding_energy("Xx", "9z"))
        assert "error" in out

    def test_calculate_sensitivity(self):
        out = json.loads(mcp_server.calculate_sensitivity(
            "Si", "2p", photon_energy=1486.6, compound="SiO2",
        ))
        assert out["kinetic_energy_eV"] == pytest.approx(1486.6 - out["binding_energy_eV"])
        assert out["cross_section"] > 0
        assert out["imfp_nm"] > 0
        assert out["sensitivity"] == pytest.approx(
            out["cross_section"] * out["imfp_nm"])

    def test_calculate_sensitivity_flags_out_of_range_imfp(self):
        """TPP-2M is fitted over 50-2000 eV; the tool's default is HAXPES.

        A caller that only sees `imfp_nm` cannot tell an interpolated
        value from a 4x extrapolation, so the flag has to travel with
        the number.
        """
        al_ka = json.loads(mcp_server.calculate_sensitivity(
            "Si", "2p", photon_energy=1486.6, compound="SiO2",
        ))
        ga_ka = json.loads(mcp_server.calculate_sensitivity(
            "Si", "2p", photon_energy=9251.7, compound="SiO2",
        ))

        assert al_ka["imfp_fitted_range_eV"] == [50.0, 2000.0]
        assert al_ka["imfp_extrapolated"] is False
        assert ga_ka["imfp_extrapolated"] is True
        assert "inelastic only" in ga_ka["imfp_model"]

    def test_calculate_sensitivity_does_not_reconfirm_an_inferred_unit(self):
        """The unit's status has to survive the MCP layer.

        `unit_info()` deliberately keeps `unit` empty for a table whose
        unit was never read from a primary source. Collapsing that into a
        single `sensitivity_unit` string here would undo the separation
        one layer up, which is where a consuming agent would convert on
        it.
        """
        from toyomacro.data import CrossSection
        from toyomacro.data import cross_section as cs

        # No bundled table has an inferred unit now (Trzhaskovskaya's kb was
        # read from its tables' headings); exercise the path with one.
        saved = cs._TABLE_UNITS["trzhaskovskaya"]
        cs._TABLE_UNITS["trzhaskovskaya"] = (None, "kb", "inferred for this test")
        try:
            CrossSection.set_default_table("trzhaskovskaya")
            out = json.loads(mcp_server.calculate_sensitivity(
                "Si", "2p", photon_energy=1486.6, compound="SiO2",
            ))
        finally:
            CrossSection.set_default_table("yeh_lindau")
            cs._TABLE_UNITS["trzhaskovskaya"] = saved

        assert out["cross_section_unit"] is None
        assert out["cross_section_inferred_unit"] == "kb"
        assert out["cross_section_unit_status"] == "inferred"
        assert out["sensitivity_unit"] is None
        assert out["sensitivity_inferred_unit"] == "kb*nm"
        # The failure this guards against: a string that reads confirmed.
        assert "None" not in str(out["sensitivity_inferred_unit"])

    def test_calculate_sensitivity_reports_a_confirmed_unit_plainly(self):
        """And the confirmed case must not be hedged into uselessness."""
        out = json.loads(mcp_server.calculate_sensitivity(
            "Si", "2p", photon_energy=1486.6, compound="SiO2",
        ))
        assert out["cross_section_unit"] == "Mb"
        assert out["cross_section_inferred_unit"] is None
        assert out["cross_section_unit_status"] == "confirmed"
        assert out["sensitivity_unit"] == "Mb*nm"
        assert out["sensitivity_inferred_unit"] is None

    def test_calculate_sensitivity_names_what_it_is_not(self):
        """`sensitivity` is sigma x IMFP, not an AMRSF.

        The number alone reads like an instrument sensitivity factor, so
        the caveat has to travel with it — an agent consuming the JSON
        never sees the docstring.
        """
        out = json.loads(mcp_server.calculate_sensitivity(
            "Si", "2p", photon_energy=1486.6, compound="SiO2",
        ))

        model = out["sensitivity_model"]
        assert "not a complete AMRSF" in model
        for excluded in ("transmission", "elastic", "polarization", "geometry"):
            assert excluded in model.lower()

    def test_calculate_sensitivity_reports_the_table_it_actually_used(self):
        """`cross_section_table` must name the table sigma came from.

        The docstring claimed Scofield while the call resolves the
        process default; for Si 2p at Al K-alpha the two differ by 43%.
        So assert the reported name against a *literal*, and against the
        sigma an explicit lookup on that table returns — comparing it to
        `get_default_table()` would only restate how it was built.
        """
        from toyomacro.data import CrossSection

        out = json.loads(mcp_server.calculate_sensitivity(
            "Si", "2p", photon_energy=1486.6, compound="SiO2",
        ))

        assert out["cross_section_table"] == "yeh_lindau"
        assert out["cross_section"] == pytest.approx(
            CrossSection.lookup("Si", "2p", 1486.6, table="yeh_lindau"))
        # ...and that this is a distinguishable claim, not a tautology.
        assert out["cross_section"] != pytest.approx(
            CrossSection.lookup("Si", "2p", 1486.6, table="scofield"))

    def test_calculate_sensitivity_flags_out_of_range_cross_section(self):
        """Yeh & Lindau stops at 8047.8 eV; the tool's default is above it.

        The IMFP already carries an extrapolation flag. Flagging only
        the IMFP would imply the cross-section is on firmer ground at
        the same energy, which at the HAXPES default it is not.
        """
        al_ka = json.loads(mcp_server.calculate_sensitivity(
            "Si", "2p", photon_energy=1486.6, compound="SiO2",
        ))
        ga_ka = json.loads(mcp_server.calculate_sensitivity(
            "Si", "2p", photon_energy=9251.7, compound="SiO2",
        ))

        assert al_ka["cross_section_table_range_eV"] == [10.2, 8047.8]
        assert al_ka["cross_section_extrapolated"] is False
        assert ga_ka["cross_section_extrapolated"] is True
        # The default call extrapolates *both* factors, not just lambda.
        assert ga_ka["imfp_extrapolated"] is True

    def test_list_fitting_templates(self):
        out = json.loads(mcp_server.list_fitting_templates())
        names = [t["name"] for t in out]
        assert "Si2p_oxide" in names
        si = next(t for t in out if t["name"] == "Si2p_oxide")
        assert si["element"] == "Si2p"
        assert len(si["peaks"]) == 5


class TestFitSpectrumFile:
    def test_fit_two_column_txt(self, tmp_path):
        # Synthetic Si 2p doublet pair as a plain two-column text file
        from scipy.special import wofz

        energy = np.linspace(96.0, 108.0, 301)

        def voigt(c, s, g):
            z = ((energy - c) + 1j * g) / (s * np.sqrt(2))
            p = np.real(wofz(z))
            return p / p.max()

        intensity = 1000 * (voigt(99.3, 0.45, 0.1) + 0.5 * voigt(99.9, 0.45, 0.1))
        intensity += 800 * (voigt(103.4, 0.6, 0.1) + 0.5 * voigt(104.0, 0.6, 0.1))
        intensity += 50.0  # constant offset
        f = tmp_path / "si2p.txt"
        np.savetxt(f, np.column_stack([energy, intensity]))

        out = json.loads(mcp_server.fit_spectrum_file(str(f), element="Si2p"))
        assert out["success"]
        assert out["r_squared"] > 0.99
        centers = [c["center_eV"] for c in out["components"]]
        assert min(abs(c - 99.3) for c in centers) < 0.3
        assert min(abs(c - 103.4) for c in centers) < 0.3


class TestGVRTTools:
    def test_gvrt_run_demo(self, tmp_path):
        png = tmp_path / "rt.png"
        out = json.loads(mcp_server.gvrt_run(
            size=64, noise="None", output_png=str(png),
        ))
        assert out["n_spectra"] == 64 * 64
        assert out["psnr_dB"]["average"] > 15
        assert png.exists()

    def test_gvrt_sweep_monotonic(self):
        out = json.loads(mcp_server.gvrt_sweep(
            noise_levels="None,Strong", size=64,
        ))
        assert len(out) == 2
        # More noise -> lower average PSNR
        assert out[0]["psnr_dB"]["average"] > out[1]["psnr_dB"]["average"]


def test_sensitivity_table_argument_reaches_deep_levels():
    """Yeh-Lindau carries no Si 1s; the default call says so and names the
    way out, and table='scofield' answers it (v0.4.0)."""
    from toyomacro.mcp_server import calculate_sensitivity

    default = json.loads(calculate_sensitivity("Si", "1s"))
    assert "error" in default and "table='scofield'" in default["error"]
    sc = json.loads(calculate_sensitivity("Si", "1s", table="scofield"))
    assert sc["cross_section"] > 0 and sc["cross_section_table"] == "scofield"
    bad = json.loads(calculate_sensitivity("Si", "2p", table="nope"))
    assert "Unknown table" in bad["error"]


@pytest.mark.parametrize(("element", "orbital", "table"), [
    ("P", "1s", "scofield"),
    ("Cu", "2s", "scofield"),
    ("As", "3s", "yeh_lindau"),
    ("O", "2s", "yeh_lindau"),
])
def test_sensitivity_refuses_photon_energy_equal_to_binding_energy(element, orbital, table):
    """At hv == BE the table has a cross-section (the threshold gate is
    hv < BE), but ke = 0 and TPP-2M has no value. The tool returned no
    JSON there: IMFP.tpp2m's ValueError escaped it."""
    from toyomacro.data import BindingEnergy, CrossSection

    be = BindingEnergy.lookup(element, orbital)
    # The case is only a test of the guard if sigma alone would pass.
    assert CrossSection.lookup_with_status(element, orbital, be, table).value > 0

    out = json.loads(mcp_server.calculate_sensitivity(
        element, orbital, photon_energy=be, table=table))
    assert "kinetic energy" in out["error"]
    assert out["kinetic_energy_eV"] == 0.0
    assert out["cross_section_status"] in ("tabulated", "outside_range_below")
    assert "sensitivity" not in out


def test_trzhaskovskaya_at_its_threshold_is_a_json_error():
    """Read at hv = E + Eb, Trzhaskovskaya has no value at E = 0, so
    hv == BE never reaches the kinetic-energy check there; it must still
    come back as the tool's error object."""
    from toyomacro.data import BindingEnergy

    be = BindingEnergy.lookup("Si", "2p1/2")
    out = json.loads(mcp_server.calculate_sensitivity(
        "Si", "2p1/2", photon_energy=be, table="trzhaskovskaya"))
    assert "error" in out and "sensitivity" not in out


def test_sensitivity_refuses_photon_energy_below_binding_energy(monkeypatch):
    """hv < BE is refused as a JSON error, never a ValueError: by the
    cross-section's threshold gate where it applies, and by the tool's own
    ke check where it does not. Yeh-Lindau prints As 2s at Al K-alpha,
    below its 1527 eV binding energy; with the gate switched off (as on a
    line with no threshold) the table answers and only the ke check is left.
    """
    from toyomacro.data import CrossSection

    gated = json.loads(mcp_server.calculate_sensitivity(
        "As", "2s", photon_energy=1486.6, table="yeh_lindau"))
    assert gated["cross_section_status"] == "below_threshold"
    assert "sensitivity" not in gated

    monkeypatch.setattr(CrossSection, "_get_binding_energy",
                        classmethod(lambda cls, element, orbital: None))
    assert CrossSection.lookup_with_status(
        "As", "2s", 1486.6, "yeh_lindau").value > 0
    ungated = json.loads(mcp_server.calculate_sensitivity(
        "As", "2s", photon_energy=1486.6, table="yeh_lindau"))
    assert "kinetic energy" in ungated["error"]
    assert ungated["kinetic_energy_eV"] == pytest.approx(-40.4)
    assert ungated["cross_section_status"] == "tabulated"
    assert "sensitivity" not in ungated



def _lazy_tool_imports() -> set[str]:
    """Modules imported inside function bodies of mcp_server.py."""
    import ast

    tree = ast.parse(open(mcp_server.__file__, encoding="utf-8").read())
    names: set[str] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name == "_preload":
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
    return names


def test_preload_imports_every_module_the_tools_import():
    """The server imports the tools' lazy imports before reading stdin.

    A first tool call that hung on Windows until the client sent another
    message was reported; the first-time imports ran while the stdio
    reader held a pending read on stdin. Run in a fresh interpreter so
    the check sees what _preload loads, not what this test session has.
    """
    import subprocess
    import sys

    wanted = sorted(_lazy_tool_imports())
    assert "toyomacro.voigtfit.gvrt_service" in wanted and "matplotlib.pyplot" in wanted
    code = (
        "import sys, json\n"
        "from toyomacro import mcp_server\n"
        f"wanted = {wanted!r}\n"
        "before = [m for m in wanted if m in sys.modules]\n"
        "mcp_server._preload()\n"
        "import matplotlib\n"
        "print(json.dumps({'before': before,\n"
        "                  'missing': [m for m in wanted if m not in sys.modules],\n"
        "                  'backend': matplotlib.get_backend().lower()}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, check=True)
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["missing"] == []
    assert got["backend"] == "agg"
    # Importing the module alone stays cheap: the heavy modules wait for _preload.
    assert "toyomacro.voigtfit.gvrt_service" not in got["before"]
