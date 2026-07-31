"""Spin-orbit resolution and per-table units in ``CrossSection``.

This file previously pinned a *defect*: a bare subshell label returned
one j component instead of the doublet. The defect is fixed, so these
tests now pin the specification that replaced it.

Method, learned across five audit rounds on this section of
``docs/API.md``:

1. A magnitude quoted in section 5 is pinned here, or it does not go in
   section 5.
2. Pin over the **population** the claim describes, not over exemplars.
   A test that pins a magnitude on a few hand-picked cases certifies an
   overgeneralization instead of catching it.
3. Do not pin a population *percentage* where a regime claim is meant —
   percentages move with photon energy and with which elements a table
   happens to carry.

The evidence for the two tables' occupancy conventions is different and
does not transfer between them; the tests are separated accordingly.
"""

import re

import pytest

from toyomacro.data.cross_section import CrossSection

AL_KA_EV = 1486.6
J_RESOLVED_TABLES = ("scofield", "trzhaskovskaya")
SCAN_ENERGIES = (1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 8000.0)

# Scofield, UCRL-51326 (1973), Table A1 ("Hartree-Fock normalizations and
# binding energies"), column NE. Transcribed from the report; the report
# itself is not redistributed with this package.
SCOFIELD_TABLE_A1_NE = {
    ("B", "2p"): {"1/2": 0.33, "3/2": 0.67, "electrons": 1},
    ("C", "2p"): {"1/2": 0.67, "3/2": 1.33, "electrons": 2},
    ("N", "2p"): {"1/2": 1.00, "3/2": 2.00, "electrons": 3},
    ("O", "2p"): {"1/2": 1.33, "3/2": 2.67, "electrons": 4},
    ("Al", "3p"): {"1/2": 0.33, "3/2": 0.67, "electrons": 1},
    ("Si", "3p"): {"1/2": 0.67, "3/2": 1.33, "electrons": 2},
}

# Scofield, UCRL-51326 (1973), Table A2 ("PHOTOELECTRIC CROSS
# SECTIONS(BARNS)"). Alongside the individual subshells the table prints
# TOTAL and K/L/M SHELL columns. Those shell columns are *stated sums* —
# they are not carried in the bundled JSON, which stores j-resolved
# subshells only. So comparing our summed subshells against them tests
# the summation semantics against a number the source states, not
# against our own arithmetic. Transcribed from the report; the report
# itself is not redistributed with this package.
#
# Si, at the two tabulated energies that bracket Al K-alpha. Scofield
# prints K-SHELL = .0E+00 at both: BE(Si 1s) = 1.8285 keV, so the K
# shell is closed and TOTAL = L + M there.
SCOFIELD_TABLE_A2_SI_BARN = {
    1000.0: {
        "subshells": {
            "2s1/2": 3.0801e04,
            "2p1/2": 1.3149e04,
            "2p3/2": 2.5849e04,
            "3s1/2": 2.6478e03,
            "3p1/2": 2.2022e02,
            "3p3/2": 4.3296e02,
        },
        "L_shell": 6.9800e04,
        "M_shell": 3.3010e03,
        "total": 7.3101e04,
    },
    1500.0: {
        "subshells": {
            "2s1/2": 1.2766e04,
            "2p1/2": 3.6599e03,
            "2p3/2": 7.1743e03,
            "3s1/2": 1.0791e03,
            "3p1/2": 6.2936e01,
            "3p3/2": 1.2338e02,
        },
        "total": 2.4866e04,
    },
}
BARN_TO_MB = 1e-6
# The stated columns carry five significant figures, so agreement is
# bounded below by the source's own rounding, not by our interpolation:
# every comparison here reads the tabulated grid point directly.
A2_ROUNDING_REL = 1e-4

# Trzhaskovskaya carries exactly this many half-listed non-s subshells.
# The rule "an absent component is unoccupied" rests on that structure,
# so the count is pinned: if it moves, the rule must be re-established
# before `_ABSENT_COMPONENT_IS_UNOCCUPIED` may stay True.
TRZH_HALF_LISTED_COUNT = 38


def _keys(element, table):
    return CrossSection._get_data(table)["data"].get(element, {})


def _tabulated(table, element, orbital, hv):
    """The stored value at an exact grid energy, bypassing interpolation.

    The Table A2 comparisons must not route through the interpolator:
    that would fold fit error into a check whose whole purpose is to
    read the stored numbers against the source.
    """
    data = CrossSection._get_data(table)
    entry = data["data"].get(element, {}).get(orbital)
    if entry is None:
        return None
    return entry["cross_sections"][data["photon_energies"].index(hv)]


def _subshells(table):
    """{(element, bare): [component keys]} for every non-s subshell."""
    out = {}
    for element, keys in CrossSection._get_data(table)["data"].items():
        grouped: dict[str, list[str]] = {}
        for key in keys:
            m = re.match(r"^(\d[spdf])(\d/2)$", key)
            if m:
                grouped.setdefault(m.group(1), []).append(key)
        for bare, components in grouped.items():
            if bare[1] != "s":
                out[(element, bare)] = sorted(components)
    return out


# ---------------------------------------------------------------------------
# The core invariant: a bare label is the sum of its ionizable components
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", J_RESOLVED_TABLES)
def test_bare_orbital_equals_the_sum_of_ionizable_components(table):
    """Over every non-s subshell in the table, at six photon energies.

    Evaluating each component at the requested energy and adding is not
    the same as summing the stored arrays and interpolating once: the
    interpolator fits one polynomial across an element's whole range, so
    where components sit on different grids the two disagree by tens of
    percent. Pinning this over the population is what caught a missing
    per-component threshold check during implementation.
    """
    checked = 0
    for (element, bare), components in _subshells(table).items():
        for hv in SCAN_ENERGIES:
            total = CrossSection.lookup(element, bare, hv, table=table)
            parts = [
                CrossSection.lookup(element, key, hv, table=table)
                for key in components
            ]
            ionizable = [p for p in parts if p is not None]
            if total is None or not ionizable:
                continue
            checked += 1
            assert total == pytest.approx(sum(ionizable), rel=1e-12), (
                f"{table} {element} {bare} at {hv} eV: bare {total!r} != "
                f"sum of ionizable components {sum(ionizable)!r}"
            )

    assert checked > 2000, f"expected the full population, checked {checked}"


def test_between_the_two_thresholds_only_the_lower_component_contributes():
    """W 3d5/2 opens at 1809 eV, 3d3/2 at 1872 eV.

    Between them a measured envelope contains one line, so the bare label
    must return that component alone rather than interpolating the
    unopened one. Gating on the subshell instead of on each component
    silently adds it back.
    """
    below = CrossSection.lookup("W", "3d3/2", 1850.0, table="scofield")
    assert below is None, "3d3/2 should be closed at 1850 eV"

    bare = CrossSection.lookup("W", "3d", 1850.0, table="scofield")
    lower = CrossSection.lookup("W", "3d5/2", 1850.0, table="scofield")
    assert bare == pytest.approx(lower)

    # ...and above both thresholds it is genuinely the sum of two.
    bare_above = CrossSection.lookup("W", "3d", 2000.0, table="scofield")
    parts = [
        CrossSection.lookup("W", f"3d{j}", 2000.0, table="scofield")
        for j in ("3/2", "5/2")
    ]
    assert all(p is not None for p in parts)
    assert bare_above == pytest.approx(sum(parts))
    assert bare_above > lower


@pytest.mark.parametrize("table", J_RESOLVED_TABLES)
def test_a_bare_label_is_not_gated_on_the_subshell_binding_energy(table):
    """Co 3p: bare BE 60 eV, but 3p1/2 opens at 59 eV.

    A subshell-level threshold gate returns None at 59 eV even though
    3p1/2 is ionizable and the spectrum contains it. The gate has to be
    per component, which means a bare label must not be refused on the
    subshell's own binding energy on the way in.
    """
    from toyomacro.data import BindingEnergy

    assert BindingEnergy.lookup("Co", "3p") == 60.0
    assert BindingEnergy.lookup("Co", "3p1/2") == 59.0

    lower = CrossSection.lookup("Co", "3p1/2", 59.0, table=table)
    assert lower is not None and lower > 0
    assert CrossSection.lookup("Co", "3p3/2", 59.0, table=table) is None

    bare = CrossSection.lookup("Co", "3p", 59.0, table=table)
    assert bare == pytest.approx(lower), (
        "bare 3p was refused at 59 eV although 3p1/2 is open"
    )


def test_a_bare_label_is_still_refused_below_every_component_threshold():
    """The gate must not simply be removed.

    Below both thresholds nothing is ionizable and the answer is None,
    on the bare-keyed default table as well as the j-resolved ones.
    """
    assert CrossSection.lookup("Co", "3p", 40.0, table="scofield") is None
    assert CrossSection.lookup("Si", "2p", 50.0) is None  # yeh_lindau, BE 99


# ---------------------------------------------------------------------------
# Scofield: occupancy evidence is Table A1 of the primary source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("element", "bare"), sorted(SCOFIELD_TABLE_A1_NE), ids=lambda v: str(v)
)
def test_scofield_table_a1_occupations_sum_to_the_subshell_electron_count(
    element, bare
):
    """Scofield assumes fractional occupations, split by degeneracy.

    This is why the tabulated cross-sections may be summed as stored:
    the occupancy is already in them, and weighting again would
    double-correct. Source: UCRL-51326 (1973), Table A1, column NE.
    """
    ne = SCOFIELD_TABLE_A1_NE[(element, bare)]
    assert ne["1/2"] + ne["3/2"] == pytest.approx(ne["electrons"], abs=0.005)
    # Split in the degeneracy ratio 2 : 4 for a p subshell.
    assert ne["3/2"] == pytest.approx(2 * ne["1/2"], rel=0.02)


# ---------------------------------------------------------------------------
# Scofield: summation semantics against Table A2's stated shell columns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hv", sorted(SCOFIELD_TABLE_A2_SI_BARN))
def test_scofield_stated_shell_columns_equal_our_subshell_sums(hv):
    """Summing the stored subshells reproduces columns we do not store.

    This is the implementation-fidelity half of the two-layer rule in
    ``AGENTS.md``. The L/M/TOTAL columns are absent from the bundled
    JSON, so the agreement cannot be an artifact of comparing the data
    to itself: it says our summation convention is the source's own.
    """
    case = SCOFIELD_TABLE_A2_SI_BARN[hv]
    stored = {
        orb: _tabulated("scofield", "Si", orb, hv) for orb in case["subshells"]
    }
    assert None not in stored.values(), f"Si subshells missing at {hv} eV"

    sums = {
        "L_shell": sum(stored[o] for o in ("2s1/2", "2p1/2", "2p3/2")),
        "M_shell": sum(stored[o] for o in ("3s1/2", "3p1/2", "3p3/2")),
    }
    sums["total"] = sums["L_shell"] + sums["M_shell"]

    for column, stated_barn in case.items():
        if column == "subshells":
            continue
        assert sums[column] == pytest.approx(
            stated_barn * BARN_TO_MB, rel=A2_ROUNDING_REL
        ), f"Si {column} at {hv} eV should be Table A2's {stated_barn:.4E} barn"


@pytest.mark.parametrize("hv", sorted(SCOFIELD_TABLE_A2_SI_BARN))
def test_scofield_k_shell_is_closed_at_both_energies(hv):
    """Which is why TOTAL = L + M above, with no K term.

    Scofield prints K-SHELL = .0E+00 at 1.0 and 1.5 keV; BE(Si 1s) =
    1.8285 keV. If the bundled table ever gained a 1s value here, the
    sum above would silently stop matching the stated TOTAL.
    """
    assert _tabulated("scofield", "Si", "1s1/2", hv) is None


@pytest.mark.parametrize("hv", sorted(SCOFIELD_TABLE_A2_SI_BARN))
def test_scofield_stored_si_subshells_match_table_a2(hv):
    """Localizes a failure of the sum above to one subshell.

    Weaker evidence than that test on its own: these six numbers are
    what the JSON stores, so this pins them rather than confirming them
    independently. It earns its place by naming which entry drifted.
    """
    for orbital, barns in sorted(SCOFIELD_TABLE_A2_SI_BARN[hv]["subshells"].items()):
        got = _tabulated("scofield", "Si", orbital, hv)
        assert got == pytest.approx(barns * BARN_TO_MB, rel=A2_ROUNDING_REL), (
            f"Si {orbital} at {hv} eV should be Table A2's {barns:.4E} barn"
        )


def test_scofield_bare_2p_is_the_stated_l_shell_minus_2s():
    """The doublet fix, checked against the source rather than itself.

    ``lookup('Si', '2p')`` must be L-SHELL - 2s. Before the fix it
    returned the 2p3/2 component alone. Tolerance is looser than
    ``A2_ROUNDING_REL`` because ``lookup`` interpolates the summed
    curve onto the requested energy while the subtrahend is read off
    the grid.
    """
    case = SCOFIELD_TABLE_A2_SI_BARN[1000.0]
    expected = case["L_shell"] * BARN_TO_MB - _tabulated(
        "scofield", "Si", "2s1/2", 1000.0
    )
    got = CrossSection.lookup("Si", "2p", 1000.0, table="scofield")
    assert got == pytest.approx(expected, rel=1e-3)


def test_scofield_lists_both_components_of_every_subshell_it_carries():
    """So a missing Scofield component is a data gap, not an empty orbital.

    This is what licenses `_ABSENT_COMPONENT_IS_UNOCCUPIED["scofield"]`
    being False: the table gives no way to distinguish the two, so a
    half-listed subshell must refuse rather than silently under-count.
    """
    half_listed = {
        k: v for k, v in _subshells("scofield").items() if len(v) != 2
    }
    assert half_listed == {}


def test_scofield_open_valence_subshells_are_still_fully_listed():
    """Even where the upper-j component holds no electrons.

    B 2p3/2 is tabulated although boron's 2p3/2 occupation is 0.67 of an
    electron rather than empty (Table A1) — the point is that Scofield
    does not omit components, so absence carries no occupancy meaning.
    """
    for element, bare in [("B", "2p"), ("C", "2p"), ("Sc", "3d"), ("Ce", "4f")]:
        components = _subshells("scofield").get((element, bare))
        assert components is not None and len(components) == 2, (
            f"{element} {bare}: expected both components, got {components}"
        )


# ---------------------------------------------------------------------------
# Trzhaskovskaya: occupancy evidence is a structural audit of this table
# ---------------------------------------------------------------------------


def test_trzhaskovskaya_half_listed_population_is_unchanged():
    """Pin the structure the "absent means unoccupied" rule rests on.

    Deliberately *not* justified by Scofield's Table A1, which says
    nothing about this table. If a cache regeneration moves this count,
    the inclusion rule has to be re-established from the source before
    the flag may stay True.
    """
    half_listed = {
        k: v for k, v in _subshells("trzhaskovskaya").items() if len(v) == 1
    }
    assert len(half_listed) == TRZH_HALF_LISTED_COUNT

    # All of them are valence subshells missing the upper-j component.
    upper = {"p": "3/2", "d": "5/2", "f": "7/2"}
    for (element, bare), components in half_listed.items():
        assert not components[0].endswith(upper[bare[1]]), (
            f"{element} {bare} is listed as {components[0]}, but the rule "
            "assumes the *upper* j component is the one that can be absent"
        )


@pytest.mark.parametrize(
    ("element", "bare"),
    [("Cr", "3d"), ("Mo", "4d"), ("Eu", "4f"), ("Re", "5d")],
)
def test_configuration_exceptions_carry_both_components(element, bare):
    """The counter-examples that make the rule credible.

    Naive j-j filling would predict an empty upper-j component for these
    four; their real ground-state configurations put an electron there,
    and the table lists both. So the table tracks actual occupancy rather
    than a filling rule of its own.
    """
    components = _subshells("trzhaskovskaya").get((element, bare))
    assert components is not None and len(components) == 2


def test_half_listed_subshell_refuses_until_the_rule_is_confirmed():
    """A half-listed bare subshell returns None, conservatively.

    The structure above is consistent with "listed iff occupied", but
    that rule has not been read from ADNDT 77/82. Treating the absent
    component as a hard zero would put an unverified assumption into a
    returned number, so the subshell refuses instead. The j-resolved
    component that *is* listed remains available.

    Flip `_ABSENT_COMPONENT_IS_UNOCCUPIED["trzhaskovskaya"]` and update
    this test once the source states the rule.
    """
    assert CrossSection.lookup(
        "V", "3d", AL_KA_EV, table="trzhaskovskaya"
    ) is None
    listed = CrossSection.lookup("V", "3d3/2", AL_KA_EV, table="trzhaskovskaya")
    assert listed is not None and listed > 0


def test_an_unlisted_component_is_never_fabricated_by_extrapolation():
    """Whatever the inclusion rule turns out to be, do not invent one.

    The cross-element log(Z) fit would happily return a plausible number
    for a component the table does not carry.
    """
    assert CrossSection.lookup(
        "V", "3d5/2", AL_KA_EV, table="trzhaskovskaya"
    ) is None


def test_every_half_listed_subshell_behaves_the_same_way():
    """All 38, not the one or two that happened to get exemplified.

    For each: the bare subshell refuses, the component that *is* listed
    returns a positive value, and the absent one is neither returned nor
    fabricated by the extrapolation fallback.
    """
    upper = {"p": "3/2", "d": "5/2", "f": "7/2"}
    half_listed = {
        k: v for k, v in _subshells("trzhaskovskaya").items() if len(v) == 1
    }
    assert len(half_listed) == TRZH_HALF_LISTED_COUNT

    for (element, bare), components in half_listed.items():
        listed = components[0]
        absent = f"{bare}{upper[bare[1]]}"

        assert CrossSection.lookup(
            element, bare, AL_KA_EV, table="trzhaskovskaya"
        ) is None, f"{element} {bare} should refuse while the rule is unconfirmed"

        value = CrossSection.lookup(
            element, listed, AL_KA_EV, table="trzhaskovskaya"
        )
        assert value is not None and value > 0, (
            f"{element} {listed} is tabulated and must still resolve"
        )

        assert CrossSection.lookup(
            element, absent, AL_KA_EV, table="trzhaskovskaya"
        ) is None, f"{element} {absent} was fabricated"


def test_a_wholly_absent_subshell_still_reaches_the_extrapolation_path():
    """The counter-case showing the refusal is targeted, not blanket.

    Refusing a *half-listed* subshell is a statement about a table that
    covers the subshell and omits one component. Where the table carries
    no component of it at all there is nothing to contradict, so the
    cross-element log(Z) fallback still answers — otherwise the fix
    would have quietly disabled it everywhere.

    Ce 1s is the case to use: the orbital is real and occupied, and it is
    absent from Yeh-Lindau only because that table covers the soft-x-ray
    range where a 40 keV level is inaccessible. So the extrapolation is
    filling a coverage gap rather than inventing a cross-section for an
    orbital that holds no electrons — which is what an unoccupied-valence
    example would have pinned.
    """
    keys = _keys("Ce", "yeh_lindau")
    assert keys and "1s" not in keys, "Ce 1s should be outside this table"

    be = 40443.0  # Ce 1s, far above the table's 8047.8 eV ceiling
    assert CrossSection.lookup("Ce", "1s", AL_KA_EV) is None, (
        "below threshold it must still refuse"
    )

    value = CrossSection.lookup("Ce", "1s", 100_000.0, table="yeh_lindau")
    assert value is not None and value > 0
    assert 100_000.0 > be


# ---------------------------------------------------------------------------
# Yeh-Lindau: bare-keyed, so j-resolved requests are refused
# ---------------------------------------------------------------------------


def test_yeh_lindau_carries_no_j_resolved_keys_at_all():
    assert _subshells("yeh_lindau") == {}
    data = CrossSection._get_data("yeh_lindau")["data"]
    assert len(data) > 100
    assert not any("/" in k for keys in data.values() for k in keys)


@pytest.mark.parametrize(
    "orbital", ["2p3/2", "2p1/2", "3d5/2", "4f7/2"]
)
def test_j_resolved_request_on_a_bare_keyed_table_returns_none(orbital):
    """Refuse rather than split the total by an assumed branching ratio.

    The statistical ratio is not exact — Scofield's own Si 2p3/2 : 2p1/2
    is 1.966, not 2 — and a split value would be indistinguishable from a
    tabulated one.
    """
    element = {"2": "Si", "3": "Ag", "4": "Au"}[orbital[0]]
    assert CrossSection.lookup(
        element, orbital, AL_KA_EV, table="yeh_lindau"
    ) is None


def test_bare_labels_still_resolve_on_yeh_lindau():
    """The default table must keep answering ordinary questions."""
    for element, bare in [("Si", "2p"), ("Ag", "3d"), ("Au", "4f")]:
        value = CrossSection.lookup(element, bare, AL_KA_EV)
        assert value is not None and value > 0


# ---------------------------------------------------------------------------
# Literature cross-check
# ---------------------------------------------------------------------------


def test_scofield_o1s_over_si2p_is_the_doublet_ratio():
    """Regression on the ratio the bundled Scofield table yields.

    Not a literature comparison: no primary source for a published
    O 1s / Si 2p sensitivity ratio has been checked, and asserting one
    without it would be the overgeneralization this file exists to
    prevent. What is pinned is the arithmetic — the ratio follows from
    the summed doublet, and returning the 2p3/2 component alone gave
    5.40 instead. Promote this to a literature check if and when a
    source is confirmed.
    """
    o1s = CrossSection.lookup("O", "1s", AL_KA_EV, table="scofield")
    si2p = CrossSection.lookup("Si", "2p", AL_KA_EV, table="scofield")
    components = [
        CrossSection.lookup("Si", f"2p{j}", AL_KA_EV, table="scofield")
        for j in ("1/2", "3/2")
    ]

    assert si2p == pytest.approx(sum(components))
    assert o1s / si2p == pytest.approx(3.574, rel=0.005)
    # The pre-fix value, kept so the regression is unambiguous.
    assert o1s / components[1] == pytest.approx(5.398, rel=0.005)


# ---------------------------------------------------------------------------
# Units: confirmed and inferred must stay separable
# ---------------------------------------------------------------------------


def test_unit_info_separates_confirmed_from_inferred():
    for table in ("yeh_lindau", "scofield"):
        info = CrossSection.unit_info(table)
        assert info["unit"] == "Mb"
        assert info["inferred_unit"] is None
        assert info["status"] == "confirmed"
        assert info["values_rescaled"] is False

    info = CrossSection.unit_info("trzhaskovskaya")
    assert info["unit"] is None, "an unconfirmed unit must not fill `unit`"
    assert info["inferred_unit"] == "kb"
    assert info["status"] == "inferred"
    assert info["values_rescaled"] is False
    assert info["note"]


def test_unit_info_offers_no_conversion_factor():
    """Nothing a caller could multiply by without reading `status`.

    `bool` is a subclass of `int`, so `values_rescaled` has to be
    excluded explicitly rather than by an isinstance check.
    """
    info = CrossSection.unit_info("trzhaskovskaya")
    numeric = {
        k: v for k, v in info.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    assert numeric == {}
    assert not any("conversion" in k or "factor" in k for k in info)


def test_no_public_unit_accessor_that_hides_the_status():
    """`unit()` was deliberately not published.

    A caller reading a bare 'kb' string would convert on an inference.
    """
    assert not hasattr(CrossSection, "unit")


def test_unknown_table_is_rejected():
    with pytest.raises(ValueError):
        CrossSection.unit_info("not_a_table")


def test_stored_values_are_not_rescaled():
    """The inference must not have leaked into the data.

    Trzhaskovskaya values stay ~10^3 above the megabarn tables; if some
    future change divides them, `values_rescaled` has to change with it.
    """
    trzh = CrossSection.lookup("Si", "2p", AL_KA_EV, table="trzhaskovskaya")
    scof = CrossSection.lookup("Si", "2p", AL_KA_EV, table="scofield")
    assert trzh / scof > 100
    assert CrossSection.unit_info("trzhaskovskaya")["values_rescaled"] is False


@pytest.mark.parametrize("table", ["scofield", "trzhaskovskaya", "yeh_lindau"])
def test_get_rsf_is_the_ratio_of_two_lookups_on_the_same_table(table):
    """Which is why it is well defined even where the unit is inferred.

    Deliberately *not* an agreement check between tables. Two
    independent theoretical compilations need not agree, and for
    `trzhaskovskaya` the photoelectron-vs-photon energy axis error
    survives any ratio between lines of different binding energy — so a
    cross-table comparison would test neither the unit cancellation nor
    anything else this file is about.
    """
    ratio = CrossSection.get_rsf("Si", "2p", "C", "1s", AL_KA_EV, table=table)
    si = CrossSection.lookup("Si", "2p", AL_KA_EV, table=table)
    c = CrossSection.lookup("C", "1s", AL_KA_EV, table=table)

    assert ratio is not None and ratio > 0
    assert ratio == pytest.approx(si / c)
