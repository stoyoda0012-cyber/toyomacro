"""Pins the spin-orbit and cross-table magnitudes quoted in docs/API.md §5.

These are **known defects**, documented but not fixed: fixing them
changes returned numbers and belongs in its own audited change set. The
tests exist so the documentation cannot drift from the behavior, and so
that a fix has to break them deliberately rather than silently.

Four audit rounds found magnitudes in §5 that had been measured on a few
lines and written as if general — the fourth found it inside the test
written to prevent the third. So the rule has two halves:

1. A magnitude in §5 is pinned here, or it does not go in §5.
2. Pin over the **population** the claim describes, not over exemplars.
   A test that pins a magnitude on five hand-picked cases certifies an
   overgeneralization instead of catching it.

A corollary learned the same way: do not pin a population *percentage*
where a regime claim is what is meant. The fractions here depend on the
photon energy and on which elements a table happens to carry — 71% of
doublets fall within 3% of the degeneracy ratio at Al Kα but only 12% at
Ga Kα — so asserting a percentage breaks on a legitimate table
extension. Assert instead that the approximation is not exact, that it
is tight for the named lines people quantify, that it degrades for heavy
elements, and that it degrades further at HAXPES energies. Exact
population numbers live in comments, dated, and in §5's prose.
"""

import re
from fractions import Fraction

import pytest

from toyomacro.data.cross_section import CrossSection

AL_KA_EV = 1486.6
GA_KA_EV = 9251.7  # the MCP tool default

# The suffix search order in `_lookup_direct`. `1/2` is tried LAST, which
# is why p behaves unlike d and f.
SUFFIX_ORDER = ["3/2", "5/2", "7/2", "1/2"]

# (element, bare orbital, l, expected key hit, expected component degeneracy)
J_RESOLVED_CASES = [
    ("Si", "2p", 1, "2p3/2", 4),
    ("Cu", "2p", 1, "2p3/2", 4),
    ("Ag", "3d", 2, "3d3/2", 4),
    ("Au", "4d", 2, "4d3/2", 4),
    ("Au", "4f", 3, "4f5/2", 6),
]


def _keys(element, table):
    return CrossSection._get_data(table)["data"].get(element, {})


def _components(element, bare, table):
    return [k for k in _keys(element, table) if k.startswith(bare) and "/" in k]


@pytest.mark.parametrize("table", ["scofield", "trzhaskovskaya"])
@pytest.mark.parametrize("element,bare,l,expected_key,expected_deg", J_RESOLVED_CASES)
def test_bare_orbital_returns_one_component_by_suffix_order(
    table, element, bare, l, expected_key, expected_deg
):
    """A bare orbital returns the first suffix that exists, not the sum.

    Pins *which* component, because it is not always j = 3/2: `4f3/2`
    does not exist, so `'4f'` falls through to `4f5/2`.
    """
    keys = _keys(element, table)
    if not keys:
        pytest.skip(f"{element} absent from {table}")

    hit = next((f"{bare}{s}" for s in SUFFIX_ORDER if f"{bare}{s}" in keys), None)
    assert hit == expected_key

    bare_value = CrossSection.lookup(element, bare, AL_KA_EV, table=table)
    component = CrossSection.lookup(element, hit, AL_KA_EV, table=table)
    assert bare_value == pytest.approx(component), (
        f"{table} {element} {bare} should equal its {hit} component alone"
    )


def _all_doublets():
    """Every two-component non-s subshell in both j-resolved tables."""
    out = []
    for table in ("scofield", "trzhaskovskaya"):
        data = CrossSection._get_data(table)["data"]
        for element, keys in data.items():
            grouped: dict[str, list[str]] = {}
            for key in keys:
                m = re.match(r"^(\d[spdf])(\d/2)$", key)
                if m:
                    grouped.setdefault(m.group(1), []).append(key)
            for bare, components in grouped.items():
                if bare[1] != "s" and len(components) == 2:
                    out.append((table, element, bare, sorted(components)))
    return out


def _j(key, bare):
    """The j value encoded in a key suffix, e.g. '2p3/2' -> Fraction(3, 2)."""
    return Fraction(key[len(bare):])


def _shortfall(table, element, bare, components, photon_energy):
    """(measured sum/bare, degeneracy-predicted ratio) or None."""
    keys = _keys(element, table)
    hit = next((f"{bare}{s}" for s in SUFFIX_ORDER if f"{bare}{s}" in keys), None)
    values = [CrossSection.lookup(element, k, photon_energy, table=table)
              for k in components]
    bare_value = CrossSection.lookup(element, bare, photon_energy, table=table)
    if hit is None or bare_value in (None, 0) or not all(values):
        return None
    l = "spdf".index(bare[1])
    predicted = 2 * (2 * l + 1) / float(2 * _j(hit, bare) + 1)
    return sum(values) / bare_value, predicted


def _deviations(photon_energy):
    """{(table, element, bare): |measured/predicted - 1|} over all doublets."""
    out = {}
    for table, element, bare, components in _all_doublets():
        got = _shortfall(table, element, bare, components, photon_energy)
        if got is not None:
            measured, predicted = got
            out[(table, element, bare)] = abs(measured / predicted - 1)
    return out


def test_the_degeneracy_ratio_is_an_approximation_not_an_identity():
    """The invariant §5 depends on: the formula is NOT exact.

    Pinning this on hand-picked exemplars is what let an
    overgeneralization through — five cases were measured, the formula
    was written as an equality, and the test agreed with the five. So
    assert over the whole population, but assert only what is a *regime*
    claim rather than a population percentage: percentages depend on the
    photon energy and on which elements the table happens to carry (at
    Ga Kα only 12% fall within 3%, against 71% at Al Kα), so pinning
    them would break on a legitimate table extension.
    """
    devs = _deviations(AL_KA_EV)
    assert len(devs) > 400, f"expected the full population, got {len(devs)}"

    values = list(devs.values())
    within_3pct = sum(d <= 0.03 for d in values) / len(values)

    # This is the real invariant, and the guard against §5's "≈" quietly
    # being restated as "=". Everything else here is regime, not ratio.
    assert within_3pct < 1.0, "formula now exact to 3% — restate API.md §5"

    # Measured 2026-07-30 over 832 doublets at Al Kα, for the record:
    # within 1% 22.5%, 3% 70.9%, 10% 87.9%, 30% 100.0%. Recorded, not
    # asserted — see the docstring.
    assert 0.3 < within_3pct < 0.95, (
        f"within-3% fraction moved to {within_3pct:.1%}; if this is a real "
        "data change, update the numbers quoted in docs/API.md §5"
    )


def test_the_approximation_holds_tightly_for_the_lines_people_quantify():
    """§5 claims "good to a few percent for light and mid-Z levels".

    That is the claim worth pinning tightly, because it is the one a
    reader acts on. Named, characterized lines at Al Kα rather than a
    population fraction.
    """
    devs = _deviations(AL_KA_EV)
    for element, bare in [("Si", "2p"), ("Cu", "2p"), ("Ag", "3d"), ("Au", "4f")]:
        dev = devs.get(("scofield", element, bare))
        assert dev is not None, f"{element} {bare} missing from scofield"
        assert dev < 0.03, f"{element} {bare} deviates {dev:.1%}, was <3%"


def test_the_approximation_degrades_for_heavy_elements():
    """§5 says it "degrades for actinide p and d" — pin that as a fact.

    Asserted on the named worst case rather than as `max(devs) <= 0.30`.
    An exact bound sitting 0.4 pp from its own boundary (Pu 3d measures
    29.6% against a 30% ceiling) fails on a 0.3% data change with no
    regression having occurred.
    """
    devs = _deviations(AL_KA_EV)

    pu = devs.get(("scofield", "Pu", "3d"))
    assert pu is not None and 0.25 < pu < 0.35, (
        f"Pu 3d deviation moved to {pu!r}; §5 quotes +29.6%"
    )
    # And that heavy levels really are the tail, not one anomaly.
    assert sum(d > 0.10 for d in devs.values()) >= 20
    assert max(devs.values()) < 0.5, (
        f"worst deviation {max(devs.values()):.1%} — §5's characterization "
        "of the spread needs revisiting"
    )


def test_the_approximation_is_far_worse_at_haxpes_energies():
    """The regime this package defaults to is the regime it holds worst in.

    `calculate_sensitivity` defaults to Ga Kα and `examples/03` uses it,
    so §5 has to say the Al Kα spread is not the general case.
    """
    al = _deviations(AL_KA_EV)
    ga = _deviations(GA_KA_EV)

    frac_al = sum(d <= 0.03 for d in al.values()) / len(al)
    frac_ga = sum(d <= 0.03 for d in ga.values()) / len(ga)

    assert frac_ga < 0.30 < frac_al, (
        f"within-3% at Ga Kα {frac_ga:.1%} vs Al Kα {frac_al:.1%}"
    )
    assert max(ga.values()) > 1.0, (
        f"worst Ga Kα deviation {max(ga.values()):.1%}; §5 quotes >200%"
    )


def test_the_returned_key_is_the_first_existing_suffix_for_every_doublet():
    """Population-level version of the mechanism claim.

    This is the assertion that actually bites when the suffix order in
    `_lookup_direct` changes — verified by mutation. The magnitude is
    approximate; *which key is returned* is exact, for all of them.
    """
    doublets = _all_doublets()
    assert len(doublets) > 400

    for table, element, bare, components in doublets:
        keys = _keys(element, table)
        hit = next(f"{bare}{s}" for s in SUFFIX_ORDER if f"{bare}{s}" in keys)
        bare_value = CrossSection.lookup(element, bare, AL_KA_EV, table=table)
        component = CrossSection.lookup(element, hit, AL_KA_EV, table=table)
        if bare_value is None or component is None:
            continue
        assert bare_value == pytest.approx(component), (
            f"{table} {element} {bare} did not resolve to {hit}"
        )


def test_p_returns_the_higher_degeneracy_half_unlike_d_and_f():
    """Because `1/2` is tried LAST, p is the exception.

    Stating this as "returns the lower-degeneracy component" is wrong
    for p, and that error is what the suffix order exists to explain.
    Asserted against the tables rather than against restated constants:
    an earlier version of this test contained `assert 4 > 2`, which
    cannot fail and certified nothing.
    """
    for table in ("scofield", "trzhaskovskaya"):
        for element, bare, expected_hit, is_higher_degeneracy in [
            ("Si", "2p", "2p3/2", True),    # 4 of 6 -> the larger half
            ("Ag", "3d", "3d3/2", False),   # 4 of 10 -> the smaller
            ("Au", "4f", "4f5/2", False),   # 4f3/2 absent -> 6 of 14
        ]:
            keys = _keys(element, table)
            if not keys:
                continue
            hit = next(f"{bare}{s}" for s in SUFFIX_ORDER if f"{bare}{s}" in keys)
            assert hit == expected_hit

            j_hit = _j(hit, bare)
            others = [k for k in _components(element, bare, table) if k != hit]
            assert others, f"{element} {bare} has no second component in {table}"
            j_other = _j(others[0], bare)
            got_higher = (2 * j_hit + 1) > (2 * j_other + 1)
            assert got_higher is is_higher_degeneracy, (
                f"{table} {element} {bare}: returned {hit}, "
                f"higher-degeneracy={got_higher}, expected {is_higher_degeneracy}"
            )


def test_single_component_subshells_cannot_be_summed():
    """The precondition §5 has to state: 38 trzhaskovskaya subshells lack a partner.

    The "request the components and sum them" remedy assumes both halves
    are tabulated. Where only one is, the other resolves through the
    cross-element log(Z) fit and the sum is a real value plus a
    fabricated one — silently, and roughly doubling sigma.
    """
    singles = {"scofield": [], "trzhaskovskaya": []}
    for table in singles:
        data = CrossSection._get_data(table)["data"]
        for element, keys in data.items():
            grouped: dict[str, list[str]] = {}
            for key in keys:
                m = re.match(r"^(\d[spdf])(\d/2)$", key)
                if m:
                    grouped.setdefault(m.group(1), []).append(key)
            for bare, components in grouped.items():
                if bare[1] != "s" and len(components) == 1:
                    singles[table].append((element, bare))

    assert singles["scofield"] == []
    assert len(singles["trzhaskovskaya"]) == 38

    # And the failure is silent: nothing raises, a plausible number comes back.
    keys = _keys("Si", "trzhaskovskaya")
    assert "3p1/2" in keys and "3p3/2" not in keys
    fabricated = CrossSection.lookup("Si", "3p3/2", AL_KA_EV, table="trzhaskovskaya")
    assert fabricated is not None and fabricated > 0


def test_yeh_lindau_carries_no_j_resolved_keys_at_all():
    """The mirror case, and why the remedy must be per-table.

    API.md §5 states this generally over the whole table, so assert it
    over the whole table rather than for one element.
    """
    data = CrossSection._get_data("yeh_lindau")["data"]
    assert len(data) > 100
    offenders = {e: [k for k in ks if "/" in k] for e, ks in data.items()
                 if any("/" in k for k in ks)}
    assert offenders == {}


@pytest.mark.parametrize("element,bare,components,expected", [
    ("Si", "2p", ["2p1/2", "2p3/2"], 1.979),
    ("Cu", "2p", ["2p1/2", "2p3/2"], 1.979),
    ("Ag", "3d", ["3d3/2", "3d5/2"], 2.184),
    ("Au", "4f", ["4f5/2", "4f7/2"], 2.258),
])
def test_summing_components_on_yeh_lindau_double_counts(
    element, bare, components, expected
):
    """Each component request falls through to the log(Z) fit and returns ~the shell.

    Pins the factors quoted in §5. They are not all 1.98 — that is the
    p value, and d/f run higher.
    """
    bare_value = CrossSection.lookup(element, bare, AL_KA_EV, table="yeh_lindau")
    total = sum(
        CrossSection.lookup(element, k, AL_KA_EV, table="yeh_lindau")
        for k in components
    )
    assert total / bare_value == pytest.approx(expected, rel=0.01)


def test_the_three_tables_are_not_reconcilable_by_a_scale_factor():
    """Pins the 141x-553x span and 3.9x spread quoted in API.md §5.

    A pure unit error would be a single constant. The spread is the
    finding, so assert the spread and not just the endpoints.
    """
    lines = [("C", "1s"), ("N", "1s"), ("O", "1s"), ("F", "1s"),
             ("Al", "2p"), ("Si", "2p"), ("Ti", "2p"), ("Fe", "2p"),
             ("Ni", "2p"), ("Cu", "2p"), ("Zn", "2p"),
             ("Ag", "3d"), ("Sn", "3d"), ("Au", "4f")]

    ratios = []
    for el, orb in lines:
        t = CrossSection.lookup(el, orb, AL_KA_EV, table="trzhaskovskaya")
        y = CrossSection.lookup(el, orb, AL_KA_EV, table="yeh_lindau")
        if t and y:
            ratios.append(t / y)

    assert len(ratios) >= 12, "need the sampled set API.md quotes"
    lo, hi = min(ratios), max(ratios)
    assert 140 <= lo <= 142, f"low end moved: {lo:.1f}"
    assert 550 <= hi <= 555, f"high end moved: {hi:.1f}"
    assert hi / lo == pytest.approx(3.9, abs=0.1)
