"""examples/10_composition_ptfe.py: estimate, condition dependence, assumptions, withheld.

Offline it runs on synthetic spectra; with the Zenodo 7074887 archive
present (``TOYOMACRO_VAMAS_POLYMER_ZIP``) it also runs on the real pair.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "10_composition_ptfe.py"


@pytest.fixture(scope="module")
def ex():
    spec = importlib.util.spec_from_file_location("_example_10", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod
    finally:
        del sys.modules[spec.name]


def _check(res):
    b = res["base"]
    assert b.status == "conditional" and set(b.fractions) == {"F", "C"}
    assert any("integrated_counts" in a for a in b.assumptions)
    assert any("share one normalisation" in a for a in b.assumptions)
    stat = next(n for n in b.not_evaluated if n.startswith("statistical uncertainty"))
    assert "not evaluated" in stat and "raw counts" in stat and "§7" in stat
    assert len(res["grid"].changes) == 4 and res["grid"].interaction is not None
    assert res["transmission_equal"].status == "conditional"
    assert res["count_rate"].status == "conditional"


def test_offline_run_on_synthetic_spectra(ex, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("TOYOMACRO_VAMAS_POLYMER_ZIP", raising=False)
    monkeypatch.setattr(ex, "OUT_DIR", tmp_path)

    def no_real_data(*_a, **_k):
        raise AssertionError("the offline run must not read or fetch the real data")

    monkeypatch.setattr(ex, "load_pair", no_real_data)
    monkeypatch.setattr(ex, "fetch", no_real_data)
    res = ex.main([])
    _check(res)
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "not an uncertainty" in out
    assert "assumed per scan and point" in out and "Eg 7.7 eV assumed" in out
    # The seeded stand-in pins the arithmetic offline (windows, exposure, curves);
    # these numbers say nothing about PTFE.
    f = res["base"].fractions["F"]
    assert f == pytest.approx(0.6373, abs=0.0005)
    assert res["count_rate"].fractions["F"] - f == pytest.approx(-0.0431, abs=0.0005)
    assert (tmp_path / "10_composition_ptfe.png").exists()


def test_a_wrong_archive_is_refused(ex, tmp_path):
    fake = tmp_path / "x.zip"
    fake.write_bytes(b"not it")
    with pytest.raises(ValueError, match="not the archive"):
        ex.main(["--data", str(fake)])


ARCHIVE = os.environ.get("TOYOMACRO_VAMAS_POLYMER_ZIP")


@pytest.mark.skipif(not ARCHIVE or not Path(ARCHIVE).is_file(),
                    reason="Zenodo 7074887 archive not present")
def test_the_real_pair(ex, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ex, "OUT_DIR", tmp_path)
    res = ex.main(["--data", ARCHIVE])
    _check(res)
    out = capsys.readouterr().out
    assert "Zenodo 7074887" in out and "SYNTHETIC" not in out
    # Pinned on the record of this version, not reference values for PTFE.
    f = res["base"].fractions["F"]
    assert f == pytest.approx(0.702, abs=0.001)
    assert res["count_rate"].fractions["F"] - f == pytest.approx(-0.0395, abs=0.001)
    assert res["transmission_equal"].fractions["F"] - f == pytest.approx(0.0292, abs=0.001)
