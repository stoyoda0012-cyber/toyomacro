"""The fixtures in ``tests/fixtures/web/`` against the code that wrote them.

Three claims, each failing on its own kind of drift:

- the committed files are what the generator writes today, byte for byte
  (a change to a generator, its contract spectrum, or the numerical
  source -- through ``source_sha256`` -- shows up here first);
- toyomacro, called with its default settings, reproduces every expected
  value within the fixture's own tolerance (a solver or default change
  that moves results);
- where the model matches its generator, the optimum is consistent with
  the generating parameters (the fixture is not just self-consistent).

How the expected values and tolerances are defined is documented in
``tests/web_fixtures.py``. Test ids avoid the substring of the
paper-reproduction filter in ``tests/README.md``: these are contract tests.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests import web_fixtures as wf

CASES = wf.case_names()
IDS = [c.replace("si2p_", "silicon_").replace("_", "-") for c in CASES]


def _load(case: str) -> dict:
    return json.loads((wf.FIXTURE_DIR / f"{case}.json").read_text(encoding="utf-8"))


def _within(actual: float, entry: dict) -> bool:
    return abs(actual - entry["value"]) <= entry["tolerance"]


def test_committed_fixtures_equal_a_fresh_generation():
    commit = wf.resolve_source_commit()
    assert commit is not None, "no fixtures and no exact source commit; run --write"
    fresh = wf.generate(commit)
    on_disk = {p.name: p.read_bytes() for p in wf.FIXTURE_DIR.glob("*.json")}
    assert sorted(on_disk) == sorted(fresh)
    stale = [name for name in fresh if fresh[name] != on_disk[name]]
    assert not stale, f"stale fixtures {stale}: run `python -m tests.web_fixtures --write`"


def test_index_lists_every_case_with_its_digest():
    import hashlib

    index = json.loads((wf.FIXTURE_DIR / "index.json").read_text(encoding="utf-8"))
    assert [c["case"] for c in index["cases"]] == CASES
    for entry in index["cases"]:
        data = (wf.FIXTURE_DIR / entry["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert json.loads(data)["provenance"]["source_sha256"] == index["source_sha256"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_inputs_are_synthetic_and_carry_their_generator(case):
    data = _load(case)
    assert data["generation"]["kind"] == "synthetic"
    assert data["generation"]["function"].startswith("tests/")
    assert len(data["input"]["x"]) == len(data["input"]["y"]) == data["input"]["points"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_default_toyomacro_fit_reproduces_the_fixture(case):
    data = _load(case)
    x = np.array(data["input"]["x"])
    y = np.array(data["input"]["y"])
    expected = data["expected"]
    tol = data["tolerance"]

    if case in wf.SINGLE_CASES:
        truth = [data["truth"]["parameters"][n] for n in wf.SINGLE_NAMES]
        solution = wf.fit_single(x, y, wf.single_default_start(truth))  # default ftol
        values, cost = solution.x, solution.cost
        lower, upper = wf.single_bounds(x)
        active = wf._active(values, lower, upper)
    else:
        result, (solution, kwargs) = wf.fit_si2p(x, y, data["model"]["energy_type"])
        values = wf.si2p_parameters_from_result(result)
        cost = result.fitting_result.chi2 / 2.0
        lower, upper = kwargs["bounds"]
        active = wf._active(values, lower, upper)
        background = result.fitting_result.background_curve
        gap = np.max(np.abs(background - np.array(expected["background"])))
        assert gap <= tol["background_absolute"], f"background off by {gap}"

    entries = expected["parameters"]
    assert [e["name"] for e in entries] == (
        wf.SINGLE_NAMES if case in wf.SINGLE_CASES else wf.si2p_parameter_names()
    )
    assert [e["at_bound"] for e in entries] == active
    off = [(e["name"], v, e["value"], e["tolerance"])
           for e, v in zip(entries, values, strict=True) if not _within(v, e)]
    assert not off, f"outside tolerance (name, got, expected, tol): {off}"
    assert abs(cost / expected["cost"] - 1.0) <= tol["cost_relative"]


@pytest.mark.parametrize(
    "case", [c for c in CASES if c in wf.SINGLE_CASES],
    ids=[i for c, i in zip(CASES, IDS, strict=True) if c in wf.SINGLE_CASES],
)
def test_optimum_is_consistent_with_the_generator(case):
    data = _load(case)
    assert data["truth"]["model_matches_generator"] is True
    for entry in data["expected"]["parameters"]:
        truth = data["truth"]["parameters"][entry["name"]]
        z = (entry["value"] - truth) / entry["statistical_sd"]
        assert abs(z) <= wf.TRUTH_SD_LIMIT, f"{entry['name']}: {z:+.2f} sd from the truth"


@pytest.mark.parametrize("case", [c for c in CASES if c in wf.SI2P_CASES],
                         ids=[i for c, i in zip(CASES, IDS, strict=True) if c in wf.SI2P_CASES])
def test_misspecified_cases_say_so(case):
    truth = _load(case)["truth"]
    assert truth["model_matches_generator"] is False
    assert truth["misspecification"]
