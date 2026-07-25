# Design Philosophy

Principles that shaped `toyomacro` and its `voigtfit` engine. None of
them are novel individually; the combination explains why the code
looks the way it does.

## 1. Throughput is a design constraint, not an optimization pass

The project's defining question changed from *"how do I fit this
spectrum well?"* to *"how do I fit 10⁹ spectra without lying about
any of them?"*. Every feature is evaluated against its cost per
spectrum. Anything that cannot be expressed as batched linear algebra
(a matmul against a precomputed basis, a residual projection, a
dictionary lookup) has to justify itself in a separate, explicitly
slower tier — which is why the solvers form a ladder
(amplitude-only → 4-step → dictionary → adaptive) instead of one
configurable monolith.

## 2. Speed discipline keeps the codebase small

Counterintuitively, optimizing for batch throughput suppresses code
growth. Features that survive must reduce to a small number of array
programs, so the temptation to accumulate special cases largely
disappears. The physics core is tiny — a Voigt profile, a spin-orbit
doublet, a background model, each a handful of lines. Most of the
remaining code exists to do that tiny thing a billion times, measure
how well it did, and stay honest about the answer.

## 3. Honest numbers or no numbers

Every performance claim states exactly what is measured: which
kernel, which backend, which hardware, end-to-end or solver-only.
Benchmark records are committed to the repository in machine-readable
form (`paper/figures/results/`), and headline numbers in the README
and paper are reproducible from scripts in
`src/toyomacro/voigtfit/benchmarks/`. A number that cannot be
regenerated does not go in the documentation.

## 4. Round-trip validation is the primary quality gate

The GVRT (Giga Voigt Round Trip) suite encodes an image into Voigt
parameters, synthesizes noisy spectra, fits them, and reconstructs
the image. If the reconstruction is wrong, the fit is wrong — no
amount of per-spectrum χ² can hide it. Per-parameter PSNR channels
(amplitude / position / width) expose *which* parameter degrades
first as noise grows, which is far more informative than a single
aggregate score.

## 5. Degrade gracefully across hardware

The MLX GPU backend is an accelerator, not a requirement. Every
solver has a NumPy fallback verified for parity (identical best-index
selection, ≤1e-6 numerical difference), so results are reproducible
on any machine that runs CPython. Tests that need the GPU skip
cleanly instead of failing.

## 6. Models are data, not code

Fitting models (peak count, spin-orbit splitting, branch ratios,
vary/fix flags) live in declarative templates, not in per-analysis
scripts. This is the oldest idea in the codebase: it descends from a
hand-me-down Classic Mac fitting program whose model files carried
the same information in the early 1990s, driven by macros at one or
two spectra per second. The definition format survived three decades
and a factor of ~10⁸ in throughput; the discipline it encodes —
separate the physics declaration from the execution engine — is the
part worth keeping.
