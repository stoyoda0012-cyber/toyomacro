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
date: 26 July 2026
bibliography: paper.bib
---

# Summary

`voigtfit` is a Python library for fitting overlapping Voigt-profile
peaks across very large X-ray photoelectron spectroscopy (XPS)
datasets, shipped as the `toyomacro.voigtfit` subpackage. It targets
hyperspectral and angle-resolved (ARXPS) maps — every pixel its own
multi-peak fit — with a solver ladder trading precision for
throughput along one API. The fastest path, an **amplitude-only
projection kernel** reducing to a single `Y @ W` matmul, sustains a
measured median of **478 million spectrum-vector evaluations per
second** on an Apple M3 Max with the MLX backend [@mlx2023]
(committed record: `paper/figures/results/`). The Giga Voigt Round
Trip (GVRT) benchmark — an image encoded as per-pixel Voigt
parameters and recovered by refitting — previously turned an 8K UHD
image (265 million fits) around in about ten minutes. A pure-NumPy
backend runs the same algorithms on any CPython host; bundled
Cramér–Rao Lower Bound (CRLB) utilities [@Kay1993] compare solver
output with the information-theoretic limit. A "spectrum" throughout
is one pixel's 1-D intensity array; rates are per array.

# Statement of need

XPS quantification reduces to fitting overlapping Voigt profiles to
each measured energy spectrum, and the dominant open-source route —
`lmfit` [@lmfit] or `scipy.optimize.curve_fit` [@scipy] wrapping
per-spectrum least-squares optimization — assumes a "one spectrum at
a time" workload (baseline below). Modern instrumentation defeats
that model — hyperspectral imaging, multi-angle ARXPS depth
profiling, and validation batches reach $10^8$–$10^9$ spectra —
making a 33-megapixel map a twelve-hour job that, recast
batch-first, completes in seconds. `voigtfit` is built around that
recasting: amplitude recovery is a dense matrix product against a
Faddeeva-derived [@Faddeeva] weight matrix, shift and width recovery
proceed by Taylor residual projection or dictionary search with
parabolic refinement, and multipeak clusters by alternating
projection, optionally polished by an exact-Jacobian Gauss–Newton
step approaching the CRLB.

# State of the field

Open-source XPS fitting serves the interactive workflow well —
KherveFitting [@KherveFitting2025] is a full GUI application,
`lmfitxps` [@lmfitxps] extends `lmfit` with XPS-specific lineshapes
and backgrounds, commercial suites are analogous — but all are
per-spectrum optimizer architectures, right for thousands of
spectra, not hundreds of millions (survey:
`docs/RELATED_SOFTWARE.md`). `voigtfit` instead rebuilds the solver
around a batch-first array contract — one shared peak configuration,
precomputed weight/dictionary matrices, every stage dense linear
algebra on GPU or BLAS with no per-spectrum Python overhead — a
design that cannot be retrofitted onto a per-spectrum API.

The package ships a same-problem comparison benchmark (identical
seeded data — input hash committed — initialization, and bounds;
accuracy on a common subset), deliberately workflow-versus-workflow
on one machine: `scipy.optimize.curve_fit` at 845 and `lmfit` at 801
spectra/s single-threaded versus the GPU-batch `dict2d_parabola`
solver at **6.6 million spectra/s** — about 8,000× — at matching
accuracy (per-parameter figures and environment in the committed
`solver_comparison.json`; the amplitude-only kernel solves a smaller
problem and is reported separately). Traceable quantification tables
ship with the package (`docs/DATA_SOURCES.md`): Scofield
[@Scofield1973], Yeh–Lindau [@YehLindau1985], and Trzhaskovskaya
[@Trzhaskovskaya2018; @Trzhaskovskaya2019] cross-sections, TPP-2M
mean free paths [@TPP2M], and Scienta-style transmission
corrections.

# Software design

The shared peak model per batch is what makes matmul-shaped solvers
possible; per-spectrum model variation falls back to conventional
tools. Costs are explicit: dictionary construction is a one-time
expense recorded separately in every benchmark; warm-cache
throughput is the steady-state figure. Estimation runs in float32 on
the GPU; the NumPy path reproduces it to about $10^{-3}$ relative at 4×
lower measured throughput, so the batch formulation, not the GPU,
carries the speedup above. Per-spectrum $\chi^2$ and quality flags mark
pixels needing escalation to a slower stage. A capability probe detects an
installed-but-unusable MLX and falls back to NumPy; `require_mlx()`
asserts the GPU path.

# Performance model and validation

\autoref{fig:bottleneck} places the measured kernel throughput
against the test machine's hardware bounds: the kernel sustains 74%
of the memory-bandwidth bound (memory-bound), while the end-to-end
bar is a *projection* from the HDF5/SSD efficiency bound — disk
input, not the solver, is expected to dominate. This is a bound, not
a measurement; the figure script performs the live measurement when
pointed at a real HDF5 file. \autoref{fig:gvrt} shows the bundled
GVRT round trip across a shot-noise ladder: the parameter channels
degrade in a reproducible order — width, then amplitude, then
position — and collapse as the peak-count SNR approaches unity. Both
figures regenerate from self-contained scripts and committed records
in `paper/figures/`.

The estimator is also characterized against *systematic* lineshape
error. Because the solver is a variable-projection method
[@GolubPereyra1973], the bias an unrepresentable model error
(asymmetry, satellites, background curvature) induces in a fitted
peak position follows a first-order projection law, validated on the
package's own Voigt basis in
`examples/04_projection_law_validation.py` — including the
production `VarProFitter` (slope 1.000, $R^2 > 0.9999$; full study
in `docs/projection_law_validation.md`). The law is derived in a
companion theoretical manuscript (in preparation). With the CRLB
utilities this covers both halves of the error budget — statistical
and systematic.

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

All throughput and accuracy claims are reproducible from committed
benchmark records; the GVRT suite validates recovery against ground
truth at the $10^9$-spectrum scale. Developed for the author's ARXPS
simulation-and-inversion workflow, the engine removed the
peak-fitting bottleneck between spectral simulation and
depth-profile reconstruction; it underpins a peer-reviewed
measurement-methodology paper now in press, and its
misspecification-bias behaviour is the experimental basis of the
companion theoretical manuscript in preparation. Engine results were
presented at the Japan Society of Applied Physics 2026 spring
meeting.

# AI usage disclosure

Generative AI was used in developing this software and manuscript.
**Tools:** Claude Code (Anthropic), using Claude Opus 4-family and
Claude 5-family models. **Where used:** implementation, refactoring,
tests, benchmarks, documentation, manuscript editing. **Nature and
scope:** the assistant drafted code against algorithms, numerical
targets, and acceptance criteria specified by the author, in a
propose–review–revise loop with the author as gatekeeper; the author
made all core algorithmic and design decisions, reviewed and tested
all AI-assisted output, and ran and verified every reported
benchmark on the author's own hardware.

# Acknowledgments

`voigtfit` grew out of the MATLAB Toyomacro suite, whose solver and
chemical-state library shaped its API. It was developed during the
author's work at Vacuum Products Corporation, which approved its
release under the MIT License with the author as copyright holder.
No external funding was received. The repository is self-contained:
everything described here runs from the public code alone.

# References
