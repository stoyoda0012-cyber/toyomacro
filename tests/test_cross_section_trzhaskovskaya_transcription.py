"""Trzhaskovskaya cells corrected against the printed tables.

An audit compared the bundled table with ADNDT 77 (2001), 82 (2002) and
92 (2006), Table 2, cell by cell: all 11,160 cells at E = 100-5000 eV and
all 1,240 binding energies agreed, and 1,148 of 1,149 cells at 10 keV.
The exception, Eu 5p3/2 at 10 keV, was stored 10x low. Four lines the
2002 table prints were missing: Er, Tm and Yb 3p3/2 and Os 3d5/2, the
shallower member of a deep doublet whose deeper member ADNDT 82 omits
(it prints only subshells bound by less than 2000 eV). This file pins
those five corrections to the printed values (kb).
"""

import json

import pytest

from toyomacro.data.paths import get_cache_dir

E = [100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 10000.0]

# element, line, printed Eb, sigma at E (kb), pages (ADNDT 82; ADNDT 92 Table 2)
ADDED = [
    ("Er", "3p3/2", 1815.0,
     [154.6, 143.5, 115.1, 82.30, 60.91, 46.38, 28.74, 19.05, 13.28, 3.438], (275, 300)),
    ("Tm", "3p3/2", 1885.0,
     [150.1, 139.6, 112.8, 81.36, 60.67, 46.48, 29.08, 19.41, 13.61, 3.581], (276, 300)),
    ("Yb", "3p3/2", 1955.0,
     [145.8, 135.9, 110.5, 80.44, 60.41, 46.55, 29.39, 19.76, 13.93, 3.726], (277, 300)),
    ("Os", "3d5/2", 1964.0,
     [512.4, 461.1, 330.9, 200.8, 129.7, 87.73, 44.67, 25.25, 15.40, 2.500], (283, 301)),
]


@pytest.fixture(scope="module")
def raw():
    with open(get_cache_dir() / "trzhaskovskaya.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize(("el", "key", "eb", "sigma", "pages"), ADDED)
def test_a_printed_line_that_was_missing_is_present(raw, el, key, eb, sigma, pages):
    assert raw["photon_energies"] == E
    line = raw["data"][el][key]
    assert line["binding_energy"] == eb, pages
    assert line["cross_sections"] == pytest.approx(sigma, rel=1e-12), pages


def test_eu_5p3_2_at_10_kev_is_as_printed(raw):
    """ADNDT 92 Table 2 (p. 299) prints 1.133 x 10^-1 kb; it was stored as 0.01133."""
    assert raw["data"]["Eu"]["5p3/2"]["cross_sections"][-1] == pytest.approx(0.1133, rel=1e-12)
    # Its neighbours, which were right: 5p1/2 at 10 keV, 5p3/2 at 5 keV.
    assert raw["data"]["Eu"]["5p1/2"]["cross_sections"][-1] == pytest.approx(0.07077)
    assert raw["data"]["Eu"]["5p3/2"]["cross_sections"][-2] == pytest.approx(0.6036)
