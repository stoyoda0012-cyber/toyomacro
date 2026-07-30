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
