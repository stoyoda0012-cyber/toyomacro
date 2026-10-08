# Using toyomacro with an AI agent — a starting point

> This document is the developers' recommended starting point for using
> toyomacro together with an AI agent. It is not a required procedure.
> Copy it and edit it for your research question, samples, instrument
> and laboratory practice. Where you depart from it, record what you
> changed and why in your analysis notes. Nothing in it widens what the
> software supports or makes a result scientifically valid.
>
> **Written for: toyomacro v0.5.0.** Read the copy that came with the
> version you have installed (`python -m toyomacro.guides`), not the
> repository's main branch, which may describe a newer version.

If you keep an edited copy, keep this header and the version line, and
keep what you added about *your usual* instrument or settings apart from
what you checked for *this* data set. An agent should not treat the
first as a fact about the second.

## 1. What toyomacro is for, and what it is not

Good at: reading XPS files (PXT/IBW, VAMAS, NPL, SES text), fitting
Voigt peaks to one spectrum or to millions at once, cross-sections,
binding energies and TPP-2M IMFPs with their state and source, and a
homogeneous-equivalent composition that lists what it rests on.

Not for: absolute composition traceable to a standard, layered or
segregated samples (the composition assumes homogeneity), depth
profiles, automatic peak identification or chemical-state assignment.
The authoritative statements are the README's "Scope" and
"Quantification boundary" sections.

## 2. Before an analysis, check

- **The energy axis**: kinetic or binding, the photon energy, and
  whether charging shifts it.
- **What the intensity is** (`SpectrumMetadata.intensity_semantics`):
  raw counts, a count rate, a corrected intensity, or unknown. Do not
  treat it as raw counts because the numbers are integers or because a
  column is called "Counts".
- **The acquisition**: dwell or collection time, number of scans or
  sweeps, pass energy, and where each came from
  (`SpectrumMetadata.field_origins`: read from the file, declared by a
  user, or inferred).
- **Transmission**: whether a curve came with the data
  (`transmission_curve`) and, separately, whether the intensity is
  already divided by it (`transmission_applied`). A curve in the file
  does not say which.

## 3. When something is not known

- Unknown intensity meaning: do not assume raw counts. If the analysis
  needs an assumption, name it (`assume_intensity`) so it travels with
  the result.
- Unknown transmission state: report it as not evaluated, or name the
  assumption (`assume_transmission`); never divide it out twice.
- Always state the elements in the denominator of a composition. Never
  renormalise over the elements that happened to work.
- A fact the file does not hold but you know — from the acquisition log,
  say — can be declared with a reason (`RawSpectrumData.declare`); it is
  then recorded as yours, not the file's.

## 4. How to work

Settle the conditions on one representative spectrum: the windows, the
background, the cross-section table, the matrix, and the assumptions.
Then apply them unchanged to the whole batch. Report anything that the
batch refused or flagged, instead of quietly dropping it.

## 5. Reading a result

Report these four things separately, and never fold them into one "±":

1. the **estimate**;
2. the **statistical uncertainty** — in v0.5.0 this is **withheld** for
   compositions: the bootstrap is calibrated on synthetic data, but no
   rule for choosing which data sets to report it for kept that
   calibration, so its validated scope is empty;
3. the **condition dependence** — how the estimate moves when the table,
   the background or an assumption is changed. It is a difference
   between conditions, not an uncertainty, and it tells you what
   dominates;
4. what was **not evaluated** — which does not mean zero.

A lesson from that validation that applies to any error bar: **choosing
which data sets get an error bar by how clean they came out changes what
the error bars mean** — on the selected data they no longer match the
real spread (in toyomacro's test they overstated it by up to 25 %).
Decide what you will report before you look at how well each fit went.

So, to "what is the error bar?", a v0.5.0 answer reads like: "the
statistical part is not available yet; the result moves by X when the
cross-section table is changed and by Y with the transmission
assumption, and the transmission assumption dominates."
`examples/10_composition_ptfe.py` shows exactly this on public data.

## 6. Keep the record with the result

Save the input files' provenance, the toyomacro version, the table, the
background and window, the matrix and its source, every assumption and
every declaration with its reason. Imported HDF5 files carry most of
this in their `/provenance` group (`toyomacro.io.read_provenance`).

## 7. The MCP server

`python -m toyomacro.mcp_server` (install with the `mcp` extra) exposes
binding-energy lookup, an intrinsic sensitivity (`calculate_sensitivity`)
and spectrum fitting to an agent. `calculate_sensitivity` is
cross-section × IMFP: **not** a complete relative sensitivity factor and
not an instrument RSF. Its result says whether the cross-section was
tabulated or extrapolated for that line, and refuses below a line's
first tabulated energy.

## 8. Where the details are

For the version this guide was written for:

- API and stability tiers: <https://github.com/stoyoda0012-cyber/toyomacro/blob/v0.5.0/docs/API.md>
- Data sources and their licences: <https://github.com/stoyoda0012-cyber/toyomacro/blob/v0.5.0/docs/DATA_SOURCES.md>
- The composition uncertainty validation: <https://github.com/stoyoda0012-cyber/toyomacro/blob/v0.5.0/docs/design/composition-uncertainty.md>
- Examples: <https://github.com/stoyoda0012-cyber/toyomacro/tree/v0.5.0/examples>

The repository's `AGENTS.md` and `CONTRIBUTING.md` (branches, audits,
commit practice) apply when you change the package itself or its public
documents — not to an analysis that only uses it.
