# Elastic scattering: SESSA comparison record

Status: **comparison record, not an acceptance.** This documents how the
`data.elastic_scattering` relations stand against a controlled SESSA
simulation series, why no accuracy figure is adopted from that
comparison yet, and what would have to change for one to be. Nothing in
this file is a measurement; every number below is the output of a
calculation, either SESSA's Monte Carlo or an analytic relation.

## The controlled series

SESSA v2.2.2 (NIST SRD 100) was driven by the maintainer with elastic
scattering as the only toggled variable, all other settings identical:

- photon energy 9251.7 eV (Ga Kα), SESSA's default source geometry and
  polarization settings (not explicitly set — a recorded limitation,
  since the analytic models are geometry-specific);
- Si substrate under an Au overlayer of 0, 25, 50, 75, 100, 150, 200,
  300 Å;
- analyzer at 0°, 30°, 45°, 60°, 75° from the surface normal;
- `MODEL SET SLA true` (straight-line approximation: elastic
  deflections neglected) against `MODEL SET SLA false` (full Mott
  elastic scattering);
- Si 1s and Au 4f intensities each integrated over ±8 eV around the
  peak (7405–7421 eV and 9158–9174 eV respectively), above a linear
  baseline interpolated from the window edges, inside simulation
  regions of 7388–7438 eV and 9136–9196 eV.

The intensity series is not stored in this repository; a bit-identical
copy is kept under version control in the maintainer's SESSA analysis
repository (SHA-256
`84645f338d4cda4de19efd9934e639e452d5c51a63dbf6b509f70826ebe43cf6`,
archived 2026-08-04, with provenance notes alongside it). The settings
above remain sufficient to regenerate it, and every number quoted below
was re-derived from the series rather than copied from earlier notes.

The 60° and 75° angles are excluded throughout: at high angle and
large thickness the substrate intensity is numerically near zero, so
peak extraction and Monte Carlo statistics dominate the fitted slope —
at 75° the extracted λ(Mott)/λ(SLA) even exceeds 1, a value this
control cannot interpret, so those angles are excluded rather than
explained.

## Input-level comparison (robust)

SESSA's own transport parameters for the Au layer at kinetic energy
7412.7 eV — IMFP 64.40 Å (its default JTP formula, *not* TPP-2M: this
package's `IMFP.tpp2m` gives 59.0 Å here, a 9.2% input difference),
TRMFP 244.90 Å — give a single-scattering albedo ω = 0.2082. The analytic relations evaluated
at that ω:

| relation | L/IMFP |
|---|---:|
| Jablonski & Powell 2020 Eq. (20), unpolarized | 0.846 |
| Jablonski & Powell 2020 Eq. (62), polarized HAXPES | 0.826 |
| Seah & Gilmore 2001 Eq. (35), Z = 79 | 0.854 |

Because ω is taken from SESSA's own tables, this is a comparison of two
calculation paths — analytic approximation against numerical transport
— **sharing the same underlying scattering data**, not two independent
determinations.

## Observable-level comparison (not yet usable for acceptance)

The Monte Carlo counterpart of L/IMFP is the ratio of the fitted
substrate-line decay lengths, λ(Mott)/λ(SLA), from
`I_Si(t) ∝ exp(−t/(λ cos θ))`. Re-deriving both from the archived
intensity series shows the observable is fragile against the fitted
thickness window:

| fit window | quantity | 0° | 30° | 45° |
|---|---|---:|---:|---:|
| 25–200 Å | λ(SLA), Å | 58.5 | 60.7 | 61.5 |
| | λ(Mott), Å | 52.5 | 48.1 | 48.3 |
| | λ(Mott)/λ(SLA) | 0.898 | 0.793 | 0.786 |
| 25–300 Å | λ(SLA), Å | 59.6 | 60.9 | 63.3 |
| | λ(Mott), Å | 47.3 | 48.3 | 51.3 |
| | λ(Mott)/λ(SLA) | 0.795 | 0.793 | 0.809 |

Two things disqualify this, for now, as an acceptance test:

1. **The control fails.** With elastic scattering off, the fitted
   λ(SLA) should recover SESSA's own IMFP of 64.40 Å nearly exactly —
   the decay law is then Beer–Lambert by construction. It comes back
   1.7–9.2% low depending on angle and window. Until the observable
   passes that control at the ~2% level, its Mott/SLA ratio cannot
   adjudicate a ~2% question such as which slope fits better.
2. **The window moves the answer.** At 0° the ratio shifts from 0.898
   to 0.795 — ±6% around the analytic predictions — when the window
   gains one thickness point. The analytic values (0.826–0.854) sit
   inside that spread, which is compatibility, not verification.

The remaining differences decompose as follows:

- **Equation.** Both 2020 relations are averages over an emission-angle
  range within which L is treated as constant; the Monte Carlo ratio is
  angle-resolved and genuinely varies with θ. No constant-in-θ relation
  can reproduce that variation, only its average.
- **Inputs.** ω comes from SESSA's tables in both columns, so input
  differences cancel here by construction. Anyone repeating this with a
  TPP-2M IMFP instead inherits the 9.2% IMFP difference directly — an
  input effect, not a model effect (see the sensitivity tests in
  `tests/test_elastic_scattering.py`).
- **Units.** All lengths in Å, ratios dimensionless; no conversion is
  involved anywhere.
- **Geometry and validity.** At 7413 eV the unpolarized Eq. (20) is an
  extrapolation (fitted 321–4426 eV); Eq. (62) covers the energy but
  assumes a specific polarized-HAXPES geometry, whereas a Ga Kα lab
  source is unpolarized; Eq. (35) was fitted at 300–1500 eV and its
  x-ray-to-analyzer angle condition was not verified against SESSA's
  default geometry. `overlayer_eal_report` flags the first of these
  automatically when the kinetic energy is declared.

## What would change this record

A revised SESSA-side observable — elastic-peak-only intensities, a fit
specification fixed before looking at the answers, and the SLA control
recovered to ~2% — would make the Mott/SLA ratio meaningful at the
level the slopes differ. Until then this record stands as: inputs
consistent, predictions within the (wide) spread of the numerical
observable, and no acceptance claim made.
