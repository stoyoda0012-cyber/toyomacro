---
title: 'toyomacro.voigtfit: high-throughput Voigt-profile fitting for X-ray photoelectron spectroscopy'
tags:
  - Python
  - XPS
  - photoelectron spectroscopy
  - Voigt profile
  - peak fitting
  - GPU
  - Apple Silicon
  - MLX
authors:
  - name: Satoshi Toyoda
    orcid: 0009-0007-1091-1579
    affiliation: 1
affiliations:
  - name: Advanced Equipment Division, Vacuum Products Corporation, Tokyo, Japan
    index: 1
date: 18 May 2026
bibliography: paper.bib
---

# Summary

`voigtfit` is a Python library for fitting overlapping Voigt-profile
peaks across very large X-ray photoelectron spectroscopy (XPS)
datasets, distributed as the `toyomacro.voigtfit` subpackage of the
`toyomacro` repository. It targets the hyperspectral and
angle-resolved (ARXPS) regimes, where every pixel of a
multi-megapixel map carries its own multi-peak fit problem. The
library exposes a ladder of solvers — amplitude-only projection,
Taylor-step residual projection, a precomputed dictionary refined by
parabolic interpolation, and a multipeak alternating-projection
routine — that trade precision for throughput along a single API.
The fastest path, an **amplitude-only projection kernel** that
reduces to a single `Y @ W` matmul against a precomputed weight
matrix, sustains a measured median of **478 million spectrum-vector
evaluations per second** (range 453–490 over nine repetitions) on an
Apple M3 Max with the MLX backend; the committed measurement record
under `paper/figures/results/` pins the environment, problem size,
and every individual timing. The Giga Voigt Round Trip (GVRT)
benchmark — which encodes an image as per-pixel Voigt parameters,
regenerates spectra, refits, and decodes back to an image —
previously completed an 8K UHD image at eight shot-noise severities
(about 265 million Voigt fits in total) in roughly ten minutes on
the same hardware. A pure-NumPy backend runs the same algorithms on
any CPython host, agreeing with the MLX path to within about
$10^{-3}$ relative (float32) on the shared solver paths.
Cramér–Rao Lower Bound (CRLB) utilities [@Kay1993] are bundled so
that solver output can be compared directly with the
information-theoretic precision limit. Throughout the paper a
"spectrum" is a single 1-D intensity array of ``n_energy`` channels
measured at one pixel of an XPS map; rates are reported per such
array.

# Statement of need

XPS quantification reduces, at its core, to fitting overlapping
Voigt profiles to each measured energy spectrum. The dominant
open-source route — `lmfit` [@lmfit] or `scipy.optimize.curve_fit`
[@scipy] wrapping per-spectrum least-squares optimization — assumes
a Python-loop "one spectrum at a time" workload. On the benchmark
problem shipped with this package (single Voigt peak, 151 channels,
free amplitude, position, and width at peak-count SNR 10), both
sustain a measured **~800 spectra per
second** in their normal single-thread loop on an Apple M3 Max
(committed record in
`paper/figures/results/solver_comparison.json`).

Modern instrumentation produces datasets that defeat this model. A
single experiment can now generate hundreds of millions of spectra:
4K or 8K hyperspectral imaging, ARXPS depth profiling at sixteen
emission angles × multiple elements × megapixel maps, or GVRT-class
validation batches in the $10^9$ range. At 800 spectra/s, a single
33-megapixel map is a twelve-hour job; the same fit recast
batch-first completes in seconds. `voigtfit` is built around that
recasting: amplitude recovery becomes a dense matrix product against
a weight matrix derived from the Faddeeva function [@Faddeeva],
energy-shift ($\delta E$) and width ($\delta\sigma$) recovery
proceed by Taylor residual projection or dictionary search with
parabolic refinement, and multipeak clusters use alternating
projection with joint block deflation, optionally polished by an
exact-Jacobian Gauss–Newton step that approaches the CRLB for
isolated peaks at realistic signal-to-noise ratios.

# State of the field

Open-source XPS fitting today serves the interactive, per-spectrum
workflow well. KherveFitting [@KherveFitting2025] provides a full
GUI application with background models and constraint systems;
`lmfitxps` [@lmfitxps] extends `lmfit` with XPS-specific lineshapes
and backgrounds (Tougaard, Shirley, slope); commercial suites ship
analogous per-spectrum routines. All inherit the per-spectrum optimizer
architecture, which is the right tool for tens to thousands of
spectra but not for hundreds of millions. A documented survey of
related software is maintained in the repository
(`docs/RELATED_SOFTWARE.md`).

Rather than contributing batch execution to an existing package,
`voigtfit` rebuilds the solver around a batch-first array contract:
one shared peak configuration is amortized across the whole batch as
precomputed weight/dictionary matrices, and every solver stage is
expressed as dense linear algebra that a GPU (or BLAS) executes
without per-spectrum Python overhead. That design cannot be
retrofitted onto a per-spectrum API, which re-evaluates the model
and re-allocates the optimizer for each spectrum.

The package ships a same-problem comparison benchmark: identical
seeded data (input hash committed), identical initialization, and
identical bounds, with accuracy scored on a common subset fitted by
every solver. It is a workflow-versus-workflow comparison on one
machine — the per-spectrum tools run their normal single-thread CPU
loops (SciPy via bounded trust-region reflective, lmfit via its
bounded `leastsq` workflow) while the batch solver uses the GPU. The recorded result is: `scipy.optimize.curve_fit` at 798 and
`lmfit` at 783 spectra/s versus the voigtfit `dict2d_parabola`
solver at **6.5 million spectra/s** — a factor of about 8,000 — at
matching accuracy on the common subset: mean absolute amplitude
error 0.022 vs. 0.022, position error 0.013 vs. 0.013 eV (full
per-parameter figures and environment in the committed
`solver_comparison.json`). The amplitude-only projection kernel is
reported separately because it solves a smaller problem (no
position/width recovery). Multipeak
throughputs of about 2 million spectra/s (two components) and 0.1
million spectra/s (ten components) were previously recorded on the
same hardware with the MLX [@mlx2023] backend.

Reference data for quantification — Scofield [@Scofield1973],
Yeh–Lindau [@YehLindau1985], and Trzhaskovskaya
[@Trzhaskovskaya2018; @Trzhaskovskaya2019] cross-sections, TPP-2M
inelastic mean free paths [@TPP2M], and Scienta-style analyzer
transmission corrections — ship as traceable tables
(`docs/DATA_SOURCES.md`), enabling quantitative analysis directly
from a fit result.

# Software design

The central trade-off is **shared configuration versus per-spectrum
flexibility**: all spectra in a batch share one peak model (centers,
widths, lineshapes), which is what makes precomputed weight and
dictionary matrices — and therefore matmul-shaped solvers — possible.
Workloads whose model varies per spectrum fall back to conventional
tools; hyperspectral maps, where one chemical system is imaged over
millions of pixels, are exactly the shared-configuration case.

Solver stages are ordered as an accuracy/throughput ladder with
explicit costs. Dictionary construction is a one-time setup expense
(recorded separately in every benchmark) amortized over the batch;
warm-cache throughput is the steady-state figure. Estimation
operates in float32 on the GPU — the NumPy path reproduces it to
about $10^{-3}$ relative, and the CRLB utilities quantify where shot
noise, not arithmetic precision, dominates the error budget.
Per-spectrum $\chi^2$ and quality flags mark pixels that need
escalation to a slower solver stage, so precision is spent only
where the data demand it. The MLX backend is opportunistic: an
installed-but-unusable MLX (e.g., a headless Mac without a Metal
device) is detected by a capability probe and the package falls back
to NumPy; `require_mlx()` asserts the GPU path for users who need a
guarantee.

# Performance model and validation

\autoref{fig:bottleneck} places the measured kernel throughput
against the hardware bounds of the test machine. The amplitude-only
projection kernel sustains a median 478 M spectra/s — 74% of the
645 M spectra/s memory-bandwidth bound and a factor of 33 below the
compute bound — confirming that the kernel is memory-bound. The
end-to-end bar is a *projection* from the HDF5/SSD efficiency bound
(about 9 M spectra/s); since that projected ceiling sits a factor of
50 below the measured kernel rate, disk input — not the solver — is
expected to dominate the full pipeline. This is a bound-based
argument, not a measurement; the figure script performs the live
end-to-end measurement when pointed at a real HDF5 file. \autoref{fig:gvrt} shows the bundled GVRT round
trip on a synthetic demo image across a shot-noise ladder spanning
peak-count SNR $10^4$ to $10^{-1}$. The three parameter channels
degrade in a reproducible order — width first, amplitude second,
peak position last — and collapse as the peak-count SNR approaches
unity, i.e., when the peak signal equals its own shot noise. Both
figures regenerate from self-contained scripts and committed
measurement records in `paper/figures/`.

The estimator class is also characterized against *systematic*
lineshape error. Because the solver is a variable-projection method
[@GolubPereyra1973], the bias that an unrepresentable model error
(asymmetry, satellites, background curvature) induces in a fitted
peak position follows a first-order projection law, validated on the
package's own Voigt basis in
`examples/04_projection_law_validation.py` — including the
production `VarProFitter` (slope 1.000, $R^2 > 0.9999$ against the
predicted bias; full three-level study in
`docs/projection_law_validation.md`). The law itself is derived in a
companion theoretical manuscript (in preparation). Together with the
CRLB utilities this covers both halves of the error budget:
statistical noise and systematic model error.

![Throughput bottleneck hierarchy on an Apple M3 Max. Hatched bars
are theoretical bounds derived from hardware specifications; the
solid bar is the measured median of nine repetitions (committed
record: `results/figure1_throughput.json`); the cross-hatched
end-to-end bar is projected from the HDF5 efficiency
bound.\label{fig:bottleneck}](figures/figure1_bottleneck.png)

![GVRT image round trip with the `dict2d_parabola` solver across the
shot-noise ladder. Top: original demo image and reconstructions from
the fitted per-pixel Voigt parameters (R = amplitude, G = energy
shift, B = FWHM). Bottom: per-channel PSNR versus peak-count SNR
($= 10^4/\mathrm{level}$); accuracy degrades in the order width
$\rightarrow$ amplitude $\rightarrow$ position and collapses as the
peak-count SNR approaches unity.\label{fig:gvrt}](figures/figure2_gvrt_noise.png)

# Research impact statement

The library's accuracy and throughput claims are reproducible from
committed, machine-readable benchmark records
(`paper/figures/results/`), and the GVRT suite validates parameter
recovery against ground truth at the $10^9$-spectrum scale —
including the shot-noise collapse ordering shown in
\autoref{fig:gvrt}. The engine was developed for, and is used in,
the author's ARXPS simulation-and-inversion research workflow, where
it removed the peak-fitting bottleneck between spectral simulation
and depth-profile reconstruction; it underpins a peer-reviewed
measurement-methodology paper now in press, and its
misspecification-bias behaviour is the experimental basis of the
companion theoretical manuscript in preparation. Results obtained
with the engine were presented at the 2026 spring meeting of the
Japan Society of Applied Physics.

# AI usage disclosure

Generative AI tools were used during the development of this
software and manuscript. **Tools:** Claude Code (Anthropic), using
Claude Opus 4-family and Claude 5-family models over the course of
development. **Where used:** implementation drafting, refactoring,
test scaffolding, benchmark scripts, documentation, and manuscript
editing. **Nature and scope:** the AI assistant drafted code against
algorithms, numerical targets, and acceptance criteria specified by
the author, in an iterative propose–review–revise loop in which the
author acted as reviewer and gatekeeper for every change. The author
made all core algorithmic and design decisions, and reviewed,
edited, validated, and tested all AI-assisted output; all benchmark
measurements reported here were executed and verified on the
author's hardware.

# Acknowledgments

`voigtfit` grew out of the MATLAB Toyomacro suite, whose
per-spectrum solver and chemical-state library shaped many of the
present-day API decisions.

This software was developed in the course of the author's work at
Vacuum Products Corporation, which approved its release under the
MIT License with the author as copyright holder. No external funding
was received. The open repository is self-contained: everything
described in this paper runs from the public code alone (see the
scope section of the README for what is deliberately out of scope).

# References
