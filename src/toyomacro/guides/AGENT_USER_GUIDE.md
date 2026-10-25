# Using toyomacro with an AI agent — a starting point

> This document is the developers' recommended starting point for using
> toyomacro together with an AI agent. It is not a required procedure.
> Copy it and edit it for your research question, samples, instrument
> and laboratory practice. Where you depart from it, record what you
> changed and why in your analysis notes. Nothing in it widens what the
> software supports or makes a result scientifically valid.
>
> **Written for: toyomacro v0.7.0.** Read the copy that came with the
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
batch refused or flagged, instead of quietly dropping it. For a
spectrum whose statistical uncertainty you will report (§5), fix the
expected counts and the conditions from a plan or a separate
measurement independent of it: the spectrum you tuned the conditions
on does not satisfy that contract for itself.

## 5. Reading a result

Report these four things separately, and never fold them into one "±":

1. the **estimate**;
2. the **statistical uncertainty** — not computed by `composition()`
   itself. `composition_uncertainty(lines, conditions, declared=...)`
   computes it, and whether it does is decided from the **expected
   counts and the analysis conditions you fixed before looking at the
   data set** (a `DeclaredSpectrum`), never from the data set's own
   areas or relative noise; it is *not evaluated* when the composition
   cannot be estimated or the intensity is not stated to be raw counts,
   and *withheld* when any of the 1,000 bootstrap replicates is
   refused, when the declaration's predicted relative area noise is
   above the validated threshold (0.02 with a linear background, 0.01
   with Shirley), or when a required assumption is not declared;
3. the **condition dependence** — how the estimate moves when the table,
   the background or an assumption is changed. It is a difference
   between conditions, not an uncertainty, and it tells you what
   dominates;
4. what was **not evaluated** — which does not mean zero.

A lesson from the validation behind that threshold (three
pre-registered rounds on synthetic data) that applies to any error bar: **choosing
which data sets get an error bar by how clean they came out changes what
the error bars mean** — on the selected data they no longer match the
real spread (in toyomacro's test they overstated it by up to 25 %).
Decide what you will report before you look at how well each fit went.

What `composition_uncertainty` returns is the standard uncertainty and
covariance of the composition estimate under independent Poisson counts.
What was validated, for each registered synthetic condition, is the
agreement between the root-mean-square reported standard uncertainty
and the standard deviation of the estimate over the data sets that
return one. It does not guarantee the accuracy of any one data set's
uncertainty, the bias of the estimate, the coverage of any interval, or
the overall accuracy of the composition. The result's `applicability`
is always `assumed`: the registered calibration transfers to your case
only under the assumptions you name in the declaration
(`channel_independence`, `no_prior_correction`, `transfer`), and the
result lists what in your case differs from the registered values
(matrix, photon energy, window, channel width, lines, exposure). The
transmission state it accepts is `unknown` with
`assume_transmission="equal_across_lines"` (nothing divided out):
`applied` means the stored intensity was already divided by a curve and
is no longer counts, and `not_applied` would divide one out in the
calculation; neither is admitted, and the file's provenance is not to
be relabelled to fit.

In the registered sensitivity analysis, a declaration with twice the
peak and half the base gives a predicted relative noise of about
0.33–0.41 of the correct declaration's, so the true relative noise of a
case admitted at the threshold can reach about 3 times the threshold.
That is a ratio observed in the registered conditions, not a bound for
any declaration error: the levels at 2 and 4 times the threshold
passed, the range between them was not validated. The declaration
errors tried are nine discrete points on the peak and base amplitudes;
between the points, wrong shapes, different errors per line and real
data are separate assumptions. Use at more counts than the lowest
registered level is an extrapolation too — name it
(`more_counts_extrapolation`); a predicted noise below 0.005 alone does
not detect it. The rate of data sets that return no uncertainty is
known only as a finite-sample bound (0 of 2,000 gives an upper bound
of 0.15 %), never as "at most 0.1 %".

The noise model is independent Poisson counts per channel. On one
laboratory detector adjacent channels of the stored intensity
correlated at +0.16; where that holds the standard uncertainty is too
small, and `channel_independence` is your assumption, not a fact the
file states. The sensitivity σ × λ is **not** a complete relative
sensitivity factor: elastic scattering, the angular distribution,
polarization and geometry are not in it, and the statistical
uncertainty does not cover them — they are in `not_evaluated`.

So, to "what is the error bar?", a v0.7.0 answer reads like: "the
statistical standard uncertainty is X, computed under the declaration
I fixed beforehand and the named assumptions; the result moves by Y
when the cross-section table is changed, by Z with the background, and
by W with the assumptions about the stored data (what the intensity is,
whether the transmission was applied); these are the ones that matter
here." `examples/10_composition_ptfe.py` shows the condition dependence
on public data, where the two assumptions about the stored data move
the answer more than the table does.

Before reporting a width split into its Gaussian and Lorentzian parts,
or two overlapping peaks as two, check whether the spectrum can determine
it: `toyomacro.voigtfit.identifiability` (experimental).

- `assess_identifiability` labels each parameter.
- `identifiability_modes` names the combinations of peak parameters
  determined worst, and what each looks like in the spectrum.
- At a fit's result, use `identifiability_at_estimate(...,
  intensity="raw_counts")`. Any other intensity returns a withheld
  report with no numbers. For an `exact_k` candidate, use
  `identifiability_from_exact_k`, which evaluates the tied model that fit
  used.
- `mode_profile` follows the softest direction for a single peak, to
  see whether the Fisher ellipse holds there.

Report them for what they are. They are bounds computed from a model at
stated parameters, or a plug-in at the fit's estimate. They are not
confidence intervals, not the fit's error, and not weights for another
fit.

- The directions depend on the reference scales, which are a
  convention. Report a mode's relative bound and what its signature
  looks like, not its vector as a physical fact.
- `weakly_identified` says that this window, these counts and this
  background are not enough; more counts or a wider window change it.
  Only `rank_deficient` is a statement about the model.
- Where the profile meets a width bound within a fraction of a standard
  deviation, the Fisher numbers say nothing beyond that point.

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

- API and stability tiers: <https://github.com/stoyoda0012-cyber/toyomacro/blob/v0.7.0/docs/API.md>
- Data sources and their licences: <https://github.com/stoyoda0012-cyber/toyomacro/blob/v0.7.0/docs/DATA_SOURCES.md>
- The composition uncertainty validation: <https://github.com/stoyoda0012-cyber/toyomacro/blob/v0.7.0/docs/design/composition-uncertainty.md>
- The identifiability diagnostics (modes, the estimate, the profile): <https://github.com/stoyoda0012-cyber/toyomacro/blob/v0.7.0/docs/design/voigt-width-identifiability.md>
- Examples: <https://github.com/stoyoda0012-cyber/toyomacro/tree/v0.7.0/examples>

The repository's `AGENTS.md` and `CONTRIBUTING.md` (branches, audits,
commit practice) apply when you change the package itself or its public
documents — not to an analysis that only uses it.
