"""Smoke tests for the ``gvrt`` CLI subcommand."""

from __future__ import annotations

from toyomacro.voigtfit.cli import main


def test_gvrt_demo_roundtrip(tmp_path, capsys):
    out = tmp_path / "roundtrip.png"
    main(["gvrt", "--size", "64", "--noise", "None", "-o", str(out)])
    assert out.exists()
    captured = capsys.readouterr().out
    assert "4,096 spectra" in captured
    assert "PSNR" in captured


def test_gvrt_inspect_pixel(tmp_path):
    out = tmp_path / "roundtrip.png"
    main([
        "gvrt", "--size", "64", "--noise", "Weak",
        "--inspect", "32,32", "-o", str(out),
    ])
    assert out.exists()
    assert (tmp_path / "roundtrip_px32_32.png").exists()
