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

`voigtfit` is a Python library for fitting overlapping
Voigt-profile peaks across very large X-ray photoelectron
spectroscopy (XPS) datasets, distributed as the
`toyomacro.voigtfit` subpackage of the `toyomacro` repository. It
targets the hyperspectral and
angle-resolved (ARXPS) regimes, where every pixel of a
multi-megapixel map carries its own multi-peak fit problem. The
library exposes a ladder of solvers — amplitude-only matrix-matrix
product, Taylor-step residual projection, a precomputed dictionary
refined by parabolic interpolation, and a multipeak alternating
projection routine — that trade precision for throughput along a
single API. The fastest path, an amplitude-only kernel that reduces
to a single `Y @ W` matmul against a precomputed weight matrix,
reaches **~400 million spectrum-vector evaluations per second** on
an Apple M3 Max with the MLX backend. The Giga Voigt Round Trip
(GVRT) benchmark — which encodes an image as per-pixel Voigt
parameters, regenerates spectra, refits, and decodes back to an
image — fits an 8K UHD image at eight Poisson noise levels (about
265 million Voigt fits in total) in roughly ten minutes on the same
hardware; the 8K source image is user-supplied via the
`VOIGTFIT_DATA_ROOT` environment variable, and any 8K image
reproduces the throughput class. A pure-NumPy backend runs the same
algorithms on any CPython host at a few-fold lower throughput,
agreeing with the MLX path to within about $10^{-3}$ relative
(float32) on the shared solver paths.
Cramér-Rao Lower Bound utilities [@Kay1993] are bundled with the
package so that solver output can be compared directly
with the information-theoretic precision limit. Throughout the
paper a "spectrum" refers to a single 1-D intensity array of
``n_energy`` channels measured at one pixel of an XPS map; rates
are reported per such array.

# Statement of need

XPS quantification reduces, at its core, to fitting overlapping Voigt
profiles to each measured energy spectrum. The dominant open-source
tools — `lmfit` [@lmfit] and `scipy.optimize.curve_fit` [@scipy] —
wrap Levenberg-Marquardt and assume a Python-loop "one spectrum at a
time" workload. Commercial XPS analysis suites distribute analogous
per-spectrum routines. In both cases, realistic throughputs land in
the 10–1,000 spectra per second range on a single core.

Modern instrumentation produces datasets that defeat this model.
A single experiment can now generate hundreds of millions of
spectra: 4K or 8K hyperspectral imaging, ARXPS depth profiling at
sixteen emission angles × multiple elements × megapixel maps, or
GVRT-class benchmark batches in the 10⁹ range used to validate
algorithms at the noise-free limit. Per-pixel chemistry maps that
take overnight on a per-spectrum LM solver complete in minutes when
the same fit is recast batch-first.

`voigtfit` is built around that recasting. Stage 1 amplitude recovery
becomes a single dense matrix-matrix product against a precomputed
weight matrix derived from the Faddeeva function [@Faddeeva],
reaching memory-bandwidth-limited throughputs. Energy-shift (δE) and
width (δσ) recovery proceeds either as a 4-step Taylor residual
projection, which keeps the solver in the linear-algebra regime, or
as a parabolic refinement on top of a 2-D dictionary that crosses
the Taylor-validity radius without sacrificing precision. The
multipeak path uses alternating projection with joint block
deflation — a true orthogonal projection against all co-fitted
components — optionally polished by an exact-Jacobian Gauss–Newton
step with joint polynomial-background coupling and per-spectrum
quality diagnostics; the exact step re-evaluates the Voigt basis
each iteration and approaches the Cramér–Rao bound for isolated
peaks at realistic signal-to-noise ratios. The multipeak solver ships in
two interchangeable implementations: a pure-numpy fallback that runs
anywhere CPython runs, and an Apple MLX [@mlx2023] path that keeps
the iteration in a lazy GPU graph on Apple Silicon — including the
small per-spectrum linear solves, which use an unrolled batched
Cholesky factorization rather than a CPU round trip — sustaining
about two million spectra per second for two-component clusters and
about 0.1 million per second at ten components.

A reproducible GVRT benchmark suite is bundled with the package —
adding physical Poisson noise injection and per-channel PSNR scoring
to that loop — so that algorithmic claims can be independently
verified at the 10⁹-spectrum scale. Cross-section data from Scofield [@Scofield1973],
Yeh–Lindau [@YehLindau1985] and Trzhaskovskaya [@Trzhaskovskaya2018;
@Trzhaskovskaya2019] are shipped as reference tables, together with
TPP-2M inelastic mean free path [@TPP2M] and Scienta-style analyzer
transmission corrections, enabling Level 2 quantitative analysis
directly from a fit result. Open-source XPS fitting tools such as
KherveFitting [@KherveFitting2025] and the lmfit-based `lmfitxps`
[@lmfitxps] serve the interactive, per-spectrum workflow well; to
the authors' knowledge no other open-source Python package occupies
the batch-first niche: batched, GPU-friendly, with integrated CRLB
and traceable cross-section data.

# Functionality and example

The public API is intentionally narrow. `voigt_profile(energy,
center, sigma, gamma)` evaluates the lineshape. `HybridPipeline`
runs the full Stage 1 + Stage 2 fit given a batch of spectra and a
peak configuration dictionary; passing ``use_mlx=False`` forces the
NumPy backend without any other code change. `SpectraGenerator`
produces calibration-quality synthetic data, and `RoundtripBenchmark`
drives the GVRT end-to-end loop. A worked example in the project
README fits ten thousand synthetic Si 2p spectra with three
oxidation-state components (Si⁰, Si¹⁺, Si⁴⁺) and recovers per-pixel
amplitudes to a median absolute error of 0.0011 (true amplitudes
drawn from $[0.3, 1.2]$). The repository carries 1,111 automated
tests — 701 of them in the voigtfit engine suite — of which the
Si 2p sub-oxide suite is the most heavily exercised real-data case,
and runs on CPython 3.11 and 3.12 under both Linux and macOS through
GitHub Actions.

# Performance model and validation

\autoref{fig:bottleneck} places the measured throughput against the
hardware bounds of the test machine. The Stage 1 amplitude-only
kernel sustains 400 M spectra/s — 62% of the 645 M spectra/s
memory-bandwidth bound and a factor of 40 below the compute bound —
confirming that the fit path is memory-bound, and that end-to-end
rates (8.4 M spectra/s) are set by HDF5/SSD input rather than by the
solver. \autoref{fig:gvrt} shows the bundled GVRT round trip on the
synthetic demo image across a Poisson noise ladder
$\lambda = 1$–$10^5$. The three parameter channels degrade in a
reproducible order — width first, amplitude second, peak position
last — and collapse only once $\lambda$ exceeds the encoded peak
amplitude of $10^3$ counts. Both figures regenerate from
self-contained scripts in `paper/figures/`, so the claims can be
re-measured on any host.

![Throughput bottleneck hierarchy on an Apple M3 Max. Hatched bars
are theoretical bounds derived from hardware specifications; solid
bars are measured. The Stage 1 kernel operates near the
memory-bandwidth roofline, while end-to-end throughput is limited by
HDF5/SSD I/O.\label{fig:bottleneck}](figures/figure1_bottleneck.png)

![GVRT image round trip across the Poisson noise ladder with the
`dict2d_parabola` solver. Top: original demo image and
reconstructions from the fitted per-pixel Voigt parameters
(R = amplitude, G = energy shift, B = FWHM). Bottom: per-channel
PSNR; accuracy degrades in the order width $\rightarrow$ amplitude
$\rightarrow$ position, collapsing when the noise level exceeds the
encoded peak amplitude.\label{fig:gvrt}](figures/figure2_gvrt_noise.png)

# Acknowledgments

`voigtfit` grew out of the MATLAB Toyomacro suite, whose
per-spectrum solver and chemical-state library shaped many of the
present-day API decisions.

# References
