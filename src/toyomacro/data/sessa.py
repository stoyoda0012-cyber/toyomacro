"""Read IMFP and TRMFP pairs out of a SESSA output file.

:mod:`toyomacro.data.elastic_scattering` needs a transport mean free
path, and none is bundled with this package (see ``docs/DATA_SOURCES.md``
under *Deliberately NOT bundled* for why). SESSA — NIST Standard
Reference Database 100 — writes one per layer and per peak, alongside the
inelastic mean free path it used for the same electron, which makes a
user's own SESSA run the practical way to get a consistent pair.

This module reads such a file. It does not bundle SESSA data, does not
call SESSA, and does not depend on it: the readers here are plain text
processing over a file the user generated under their own licence, in
the same arrangement as the Scienta transmission-function adapter.

Getting the file
----------------

In SESSA, after the sample and peaks are set up::

    PROJECT SAVE OUTPUT "/absolute/path/prefix"

writes ``<prefix>sam_par.txt`` (the interaction parameters read here)
together with ``<prefix>sam_peak.txt``, ``<prefix>rems.txt``,
``<prefix>refs.txt`` and the partial-intensity files. ``MODEL SAVE
SPECTRA`` does not write ``sam_par.txt``.

What the file does and does not say
-----------------------------------

``sam_par.txt`` holds, for each peak and each layer, the kinetic energy
and the IMFP, EMFP, TRMFP and surface-excitation parameter for that
electron in that layer, each tagged with a bracketed reference number
resolved by ``rems.txt`` (:func:`read_remarks`).

It does **not** record the layer compositions, and it does not record
the SESSA version. Those are declarations only the caller can make, so
:func:`read_sam_par` takes them as arguments and leaves them undeclared
otherwise — an undeclared material reaches
:func:`~toyomacro.data.elastic_scattering.overlayer_eal_report` as
``"not_recorded"``, which is not a passed check.

Note that the layer material is the matrix the electron travels
through, which is *not* in general the element named by ``peak_id``: a
row for peak ``au4f7`` in layer 2 is the path length of a 9.16 keV
electron in whatever layer 2 is made of.

The pairing
-----------

:meth:`SessaInteractionParameters.lengths` builds both
:class:`~toyomacro.data.elastic_scattering.DescribedLength` objects from
a single row of a single file, so ``overlayer_eal_report(*row.lengths(),
...)`` cannot pair an IMFP and a TRMFP that describe different
electrons. That is the requirement the ``elastic_scattering`` module
docstring states and cannot itself enforce.

It is a guarantee about that call, not about the module. Splitting the
pair — taking ``one_row.lengths()[0]`` and ``another_row.lengths()[1]``
— defeats it, and the checks in ``overlayer_eal_report`` are weaker
against this than they look: SESSA writes **the same kinetic energy for
a peak in every layer**, so the kinetic-energy check cannot see a
cross-layer mispair, and an overlayer on a substrate of the same
material — the geometry the overlayer EAL is defined for — passes the
material check too.

For that reason the lengths from :meth:`~SessaInteractionParameters.lengths`
carry the row in their ``source``, not just the run, so a pair split
within one file comes back with ``source_consistency == "inconsistent"``
and a warning instead of a plausible wrong number. It is a warning
rather than an error because that is what the report does with every
source disagreement; the way to not need it is to spread one row.

The row is ``(layer, peak_no, peak_id)``, which identifies a row within
a file but says nothing about *which* file. Reading two runs and
splitting a pair across the same ``(layer, peak_no, peak_id)`` in both
is therefore **not** detected, since the two source strings are equal.
Give the runs different labels — ``version``, ``imfp_model`` or an
explicit ``source`` — if you hold more than one at a time, and they
become distinguishable again.

Units are taken from the column headers rather than assumed. SESSA
writes Angstrom; a file whose header says otherwise is either converted
or rejected, never read as if it were Angstrom.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from toyomacro.data.elastic_scattering import DescribedLength, LengthUnit

__all__ = [
    "SessaInteractionParameters",
    "read_sam_par",
    "read_remarks",
    "select_row",
    "sessa_source",
]

# Column header units that name a length, and the DescribedLength unit
# they map to. SESSA writes "[A]"; the others are accepted so that a
# differently configured build is converted rather than misread. Keys
# are matched after casefolding and so are spelled casefolded here: a
# key written with a capital would never match. Both Unicode spellings
# of the angstrom symbol -- U+00C5 (latin capital A with ring above)
# and U+212B (angstrom sign) -- casefold to the single key below, so
# listing it once covers both.
_LENGTH_UNITS: dict[str, LengthUnit] = {
    "a": "angstrom",
    "\u00e5": "angstrom",
    "angstrom": "angstrom",
    "nm": "nm",
}

_ANGSTROM_PER_NM = 10.0

#: Columns this reader needs. EMFP and SEP are read when present and
#: left as ``None`` when absent, because nothing here uses them.
_REQUIRED_COLUMNS = ("lay", "peak_no", "peak_id", "energy", "imfp", "trmfp")

_HEADER_MARKER = "#lay"

# A numeric field, optionally followed by SESSA's bracketed reference
# tag, in the shape "12.34567  [  1]" (illustrative, not a real value).
_VALUE = re.compile(
    r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?)\s*(?:\[\s*(\d+)\s*\])?$"
)

# The tag that opens a remark block in rems.txt, in the shape
# "[3]  Some note about the numbers." (illustrative, not a real remark).
_REMARK_TAG = re.compile(r"^\s*\[\s*(\d+)\s*\]\s*(.*)$")


@dataclass(frozen=True)
class SessaInteractionParameters:
    """One row of ``sam_par.txt``: one peak in one layer.

    Lengths are kept in the unit the file declared (or the unit
    :func:`read_sam_par` was asked to convert to), which ``unit``
    names. ``*_ref`` are SESSA's bracketed reference tags, resolved to
    text by :func:`read_remarks`; they record where each quantity came
    from, which the single ``source`` string cannot.

    Attributes:
        layer: 1-based layer index as SESSA numbers them.
        peak_no: 1-based peak index within the run.
        peak_id: SESSA's peak label, e.g. ``"au4f7"``. Names the
            *emitting* level, not the layer material.
        kinetic_energy_ev: Electron kinetic energy, in eV.
        imfp: Inelastic mean free path, in ``unit``.
        trmfp: Transport mean free path, in ``unit``.
        emfp: Elastic mean free path, or ``None`` if the file had no
            such column. Not used by this package.
        surface_excitation_parameter: SESSA's SEP, or ``None``. Its
            unit is not a length and is not converted.
        unit: The unit ``imfp``, ``trmfp`` and ``emfp`` are in.
        material: The layer's composition if the caller declared it,
            else ``None``.
        source: Provenance string for the *run*; see
            :func:`sessa_source`. The lengths from
            :meth:`lengths` carry :meth:`row_source` instead, which
            is this plus the row.
    """

    layer: int
    peak_no: int
    peak_id: str
    kinetic_energy_ev: float
    imfp: float
    trmfp: float
    emfp: float | None
    surface_excitation_parameter: float | None
    unit: LengthUnit
    material: str | None
    source: str
    imfp_ref: int | None = None
    trmfp_ref: int | None = None
    emfp_ref: int | None = None
    sep_ref: int | None = None

    def lengths(self) -> tuple[DescribedLength, DescribedLength]:
        """Return ``(imfp, trmfp)`` as described lengths, in that order.

        Both come from this one row, so they describe the same electron
        in the same layer of the same run. The result is meant to be
        *spread* into a report function, which is what makes that a
        guarantee rather than a convention::

            report = overlayer_eal_report(
                *row.lengths(), model="jp2020_polarized_haxpes")

        The order is ``(imfp, trmfp)`` and it matters: nothing
        downstream can tell a reversed pair from a real one — both
        lengths then agree on unit, material, energy and source, and
        the albedo comes out as its own complement. Spread the tuple;
        do not reorder it.

        Both lengths carry the same ``source``, which names the run
        **and this row** — see :meth:`row_source`. Naming the row is
        what lets a split pair be detected; naming the run is what the
        report's source check is otherwise about. The per-quantity
        provenance — SESSA derives the TRMFP from elastic cross
        sections, not from the IMFP formula — stays in ``imfp_ref`` and
        ``trmfp_ref``.
        """
        common = {
            "unit": self.unit,
            "source": self.row_source(),
            "material": self.material,
            "kinetic_energy_ev": self.kinetic_energy_ev,
        }
        return (
            DescribedLength(value=self.imfp, **common),
            DescribedLength(value=self.trmfp, **common),
        )

    def row_source(self) -> str:
        """Return ``source`` narrowed to this row.

        An albedo's "one dataset" is one row, not one file: two rows of
        the same run describe different electrons, or the same electron
        in different matter. Carrying the row in the source string is
        what makes ``overlayer_eal_report`` report a split pair as
        ``"inconsistent"``, since the kinetic energy is identical across
        layers for a given peak and cannot distinguish them.

        The format is stable, and consumers that keep the record may
        rely on it::

            <run source>; layer <layer>, peak <peak_no> <peak_id>

        The run part is whatever :func:`sessa_source` built or the
        caller passed, so it is free text; the suffix is the fixed part.
        See the module docstring for what this does and does not catch.
        """
        return (f"{self.source}; layer {self.layer}, "
                f"peak {self.peak_no} {self.peak_id}")


def sessa_source(
    *, version: str | None = None, imfp_model: str | None = None
) -> str:
    """Build the provenance string for a SESSA run.

    Neither fact is in ``sam_par.txt``, so both are declarations. What
    is declared is included and what is not is left out, rather than
    guessed: ``sessa_source()`` is ``"SESSA"``, and
    ``sessa_source(version="2.2.2", imfp_model="JTP")`` is
    ``"SESSA 2.2.2 (JTP)"``.

    Args:
        version: SESSA version, e.g. ``"2.2.2"``.
        imfp_model: The IMFP formula the run was configured with, e.g.
            ``"JTP"`` (SESSA's default) or ``"TPP-2M"``. Check it
            against the run's own ``rems.txt`` rather than assuming;
            :func:`read_remarks` resolves the tag that
            ``SessaInteractionParameters.imfp_ref`` carries.

    Returns:
        The string to stamp on both lengths of the pair.
    """
    text = "SESSA"
    if version is not None and version.strip():
        text += f" {version.strip()}"
    if imfp_model is not None and imfp_model.strip():
        text += f" ({imfp_model.strip()})"
    return text


def _split_row(line: str) -> list[str]:
    """Split a pipe-delimited line, dropping the empty trailing field."""
    fields = [field.strip() for field in line.split("|")]
    if fields and not fields[-1]:
        fields.pop()
    return fields


def _parse_header(line: str, path: Path, lineno: int) -> tuple[
    list[str], dict[str, str]
]:
    """Return the header's column names and their bracketed units.

    Names are lowercased and stripped of the leading ``#`` and of any
    unit, so ``"#lay"`` is ``"lay"`` and ``"Energy [eV]"`` is
    ``"energy"``.
    """
    names: list[str] = []
    units: dict[str, str] = {}
    for field in _split_row(line):
        unit = ""
        if "[" in field and field.endswith("]"):
            head, _, bracket = field.partition("[")
            unit = bracket[:-1].strip()
            field = head
        name = field.strip().lstrip("#").strip().casefold()
        if not name:
            continue
        if name in units:
            raise ValueError(
                f"{path}:{lineno}: column {name!r} appears twice in the "
                "header; this is not a sam_par.txt this reader can map")
        names.append(name)
        units[name] = unit
    missing = [c for c in _REQUIRED_COLUMNS if c not in units]
    if missing:
        raise ValueError(
            f"{path}:{lineno}: sam_par.txt header is missing the "
            f"{', '.join(missing)} column(s); found {names}. The IMFP and "
            "the TRMFP must both be present — an albedo cannot be formed "
            "from one of them.")
    return names, units


def _length_unit(column: str, declared: str, path: Path) -> LengthUnit:
    unit = _LENGTH_UNITS.get(declared.strip().casefold())
    if unit is None:
        raise ValueError(
            f"{path}: the {column.upper()} column is headed with unit "
            f"{declared!r}, which this reader does not recognise as a "
            f"length. Known: {sorted(_LENGTH_UNITS)}. The unit is read "
            "from the header rather than assumed, so an unfamiliar one "
            "stops the read instead of being taken for Angstrom.")
    return unit


def _value(
    row: Mapping[str, str], column: str, path: Path, lineno: int
) -> tuple[float, int | None]:
    """Return one field as ``(number, reference tag or None)``."""
    text = row[column]
    match = _VALUE.match(text.strip())
    if match is None:
        raise ValueError(
            f"{path}:{lineno}: cannot read column {column!r} from "
            f"{text!r}; expected a number, optionally followed by a "
            "bracketed reference tag")
    number, tag = match.groups()
    return float(number.replace("D", "e").replace("d", "e")), (
        int(tag) if tag is not None else None)


def read_sam_par(
    path: str | Path,
    *,
    layer_materials: Mapping[int, str] | None = None,
    version: str | None = None,
    imfp_model: str | None = None,
    source: str | None = None,
    unit: LengthUnit | None = None,
) -> list[SessaInteractionParameters]:
    """Read a SESSA ``sam_par.txt`` into one row per layer and peak.

    The file's own column headers decide which column is which and what
    unit its numbers are in; nothing about the layout is assumed. A file
    without an IMFP or a TRMFP column, or with a length column in an
    unrecognised unit, raises rather than being read approximately.

    Args:
        path: The ``<prefix>sam_par.txt`` written by ``PROJECT SAVE
            OUTPUT``.
        layer_materials: Composition of each layer by SESSA's 1-based
            layer number, e.g. ``{1: "Au", 2: "Si"}``. The file does not
            contain this. Layers left out are reported as
            ``"not_recorded"`` downstream; a key that matches no layer
            in the file raises, since that is usually an off-by-one
            against SESSA's 1-based numbering.
        version: SESSA version, for the provenance string.
        imfp_model: The run's IMFP formula, for the provenance string.
        source: Provenance string to use verbatim, overriding
            ``version`` and ``imfp_model``.
        unit: Convert all lengths to this unit. The default keeps the
            unit the file declares (Angstrom, for SESSA). Pass ``"nm"``
            to pair these against this package's own IMFPs, which are
            in nm.

    Returns:
        One :class:`SessaInteractionParameters` per data row, in file
        order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file has no readable header, is missing a
            required column, declares an unrecognised unit, has a
            malformed data row, repeats a (layer, peak) pair, or if
            ``layer_materials`` names a layer the file does not have.
    """
    path = Path(path)
    if source is not None and (version is not None or imfp_model is not None):
        raise ValueError("pass either source, or version/imfp_model, not both")
    if source is None:
        source = sessa_source(version=version, imfp_model=imfp_model)
    if unit is not None and unit not in ("nm", "angstrom"):
        raise ValueError(f"unit must be 'nm' or 'angstrom', got {unit!r}")

    text = path.read_text(encoding="utf-8", errors="replace")

    names: list[str] | None = None
    out_unit: LengthUnit | None = None
    scale = 1.0
    rows: list[SessaInteractionParameters] = []
    seen: dict[tuple[int, int], int] = {}

    for lineno, line in enumerate(text.splitlines(), start=1):
        if _HEADER_MARKER in line.casefold():
            names, units = _parse_header(line, path, lineno)
            if units["energy"].strip().casefold() != "ev":
                raise ValueError(
                    f"{path}:{lineno}: the Energy column is headed with "
                    f"unit {units['energy']!r}, not eV")
            file_unit = _length_unit("imfp", units["imfp"], path)
            for column in ("trmfp", "emfp"):
                if column in units and _length_unit(
                        column, units[column], path) != file_unit:
                    raise ValueError(
                        f"{path}:{lineno}: the IMFP column is in "
                        f"{units['imfp']!r} but the {column.upper()} "
                        f"column is in {units[column]!r}; a pair read "
                        "from mixed units would give a wrong albedo")
            out_unit = unit if unit is not None else file_unit
            scale = (1.0 if out_unit == file_unit
                     else 1.0 / _ANGSTROM_PER_NM if out_unit == "nm"
                     else _ANGSTROM_PER_NM)
            continue

        if names is None or not line.lstrip()[:1].isdigit():
            continue

        fields = _split_row(line)
        if len(fields) != len(names):
            raise ValueError(
                f"{path}:{lineno}: row has {len(fields)} fields but the "
                f"header declares {len(names)}: {line!r}")
        row = dict(zip(names, fields))

        def value(column: str) -> tuple[float, int | None]:
            return _value(row, column, path, lineno)  # noqa: B023

        layer = int(value("lay")[0])
        peak_no = int(value("peak_no")[0])
        if (layer, peak_no) in seen:
            raise ValueError(
                f"{path}:{lineno}: layer {layer} peak {peak_no} was "
                f"already read at line {seen[(layer, peak_no)]}; two runs "
                "concatenated into one file cannot be told apart")
        seen[(layer, peak_no)] = lineno

        emfp, emfp_ref = value("emfp") if "emfp" in row else (None, None)
        sep, sep_ref = value("sep") if "sep" in row else (None, None)

        rows.append(SessaInteractionParameters(
            layer=layer,
            peak_no=peak_no,
            peak_id=row["peak_id"],
            kinetic_energy_ev=value("energy")[0],
            imfp=value("imfp")[0] * scale,
            trmfp=value("trmfp")[0] * scale,
            emfp=None if emfp is None else emfp * scale,
            surface_excitation_parameter=sep,
            unit=out_unit,
            material=None,
            source=source,
            imfp_ref=value("imfp")[1],
            trmfp_ref=value("trmfp")[1],
            emfp_ref=emfp_ref,
            sep_ref=sep_ref,
        ))

    if not rows:
        raise ValueError(
            f"{path}: no interaction-parameter rows found. A file written "
            "by MODEL SAVE SPECTRA rather than PROJECT SAVE OUTPUT looks "
            "like this.")

    if layer_materials:
        present = {row.layer for row in rows}
        unknown = sorted(set(layer_materials) - present)
        if unknown:
            raise ValueError(
                f"{path}: layer_materials names layer(s) {unknown} that "
                f"the file does not contain; it has {sorted(present)}. "
                "SESSA numbers layers from 1.")
        rows = [
            row if row.layer not in layer_materials
            else replace(row, material=layer_materials[row.layer])
            for row in rows
        ]

    return rows


def select_row(
    rows: Iterable[SessaInteractionParameters],
    *,
    layer: int,
    peak_id: str | None = None,
    peak_no: int | None = None,
) -> SessaInteractionParameters:
    """Return the single row matching a layer and a peak.

    A peak id repeats across layers and a layer holds every peak, so
    neither alone identifies a row. Matching nothing, or more than one
    row, raises rather than returning a first match.

    Args:
        rows: Rows from :func:`read_sam_par`.
        layer: SESSA's 1-based layer number.
        peak_id: Peak label, compared case-insensitively.
        peak_no: Peak index, as an alternative to ``peak_id``.

    Returns:
        The matching row.

    Raises:
        ValueError: If neither ``peak_id`` nor ``peak_no`` is given, or
            if the selection does not match exactly one row.
    """
    if peak_id is None and peak_no is None:
        raise ValueError("give peak_id or peak_no")
    rows = list(rows)
    found = [
        row for row in rows
        if row.layer == layer
        and (peak_no is None or row.peak_no == peak_no)
        and (peak_id is None
             or row.peak_id.casefold() == peak_id.casefold())
    ]
    if len(found) == 1:
        return found[0]
    wanted = f"layer {layer}" + (
        f", peak_id {peak_id!r}" if peak_id is not None else "") + (
        f", peak_no {peak_no}" if peak_no is not None else "")
    if not found:
        available = sorted({(row.layer, row.peak_id) for row in rows})
        raise ValueError(
            f"no row for {wanted}; available (layer, peak_id): {available}")
    raise ValueError(
        f"{len(found)} rows match {wanted}; narrow the selection")


def read_remarks(path: str | Path) -> dict[int, str]:
    """Read a SESSA ``rems.txt`` into ``{tag: remark text}``.

    The tags are the bracketed numbers ``sam_par.txt`` puts after each
    value, so this is how ``imfp_ref`` and ``trmfp_ref`` become
    statements — which IMFP formula the run used, whether a quantity was
    extrapolated, which potential the elastic cross sections came from.
    Consecutive lines belonging to one tag are joined with spaces.

    Args:
        path: The ``<prefix>rems.txt`` from the same run.

    Returns:
        Remark text by tag number. Tags with no text map to ``""``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    remarks: dict[int, list[str]] = {}
    current: int | None = None
    for line in path.read_text(
            encoding="utf-8", errors="replace").splitlines():
        match = _REMARK_TAG.match(line)
        if match is not None:
            current = int(match.group(1))
            remarks.setdefault(current, [])
            rest = match.group(2).strip()
            if rest:
                remarks[current].append(rest)
        elif current is not None and line.strip():
            remarks[current].append(line.strip())
    return {tag: " ".join(parts) for tag, parts in remarks.items()}
