# Test inventory

This suite has **2,041 automated tests** across **80 files**, in two
locations:

| Location | Scope | Files | Tests |
|---|---|--:|--:|
| `tests/` | Library body — lineshapes, backgrounds, templates, I/O, quantification, meta | 43 | 1,065 |
| `src/toyomacro/voigtfit/tests/` | VoigtFit engine — solvers, encoders, information theory | 37 | 976 |

Every test here runs on a plain `pip install` (no GUI or instrument
data required). Counts below come from `pytest --collect-only` on an
install with the `mcp` extra, which is what CI uses; without it the 13
MCP-server tests collapse into one skipped module. The MLX extra does
not change what is collected. `test_readme_inventory.py` compares every
number on this page with a fresh collection and fails, listing the
values to copy in, when one of them drifts.

## Two kinds of test

Tests fall into two purposes. The distinction matters when deciding
what to run:

- **Contract / regression** (1,609 tests, 79%) — guarantee the library
  behaves correctly: lineshape math, background algorithms, solver
  routing, file readers, template conversion, and the reference-data
  tables. Fast, deterministic.
- **Paper reproduction** (432 tests, 21%) — reproduce the accuracy
  and throughput claims in the JOSS paper: the GVRT image round-trip,
  the Hilbert/Split parameter encoders, and the Si 2p sub-oxide fit.
  These sweep large parameter grids and are the reason the count looks
  large for a peak-fitting library.

> **What is GVRT?** *Giga Voigt Round Trip.* An image is encoded as
> per-pixel Voigt parameters, regenerated into synthetic spectra with
> physical Poisson noise, refitted by the solver, and decoded back
> into an image. Comparing the input and recovered images (per-channel
> PSNR) measures end-to-end solver fidelity — at up to 10⁹ spectra.
> It is a self-contained accuracy benchmark, not an XPS file format.

To run only the contract tests (skip the heavy reproductions):

```bash
pytest -k "not gvrt and not si2p and not encoder and not roundtrip and not multi_image and not ncomp"
```

## Library body — `tests/` (1,065)

### Claim guards — noise model, versions, backends, comparisons (58)
| Tests | File | Guards |
|--:|---|---|
| 19 | `test_noise_semantics.py` | `level` ↔ peak-SNR ↔ Poisson-mean conversions, empirically pinned to the sampler |
| 10 | `test_mlx_support.py` | MLX absent / installed-but-unusable / usable; NumPy fallback end-to-end |
| 10 | `test_solver_comparison.py` | Same-problem scipy/lmfit comparison benchmark stays runnable + seed reproducibility |
| 4 | `test_cli_surface.py` | CLI entry points stay importable and keep their documented flags |
| 4 | `test_readme_inventory.py` | Every count on this page — total, per location, per section, per file, contract / reproduction split — against a fresh `pytest --collect-only` |
| 3 | `test_version.py` | pyproject ↔ `toyomacro.__version__` ↔ voigtfit ↔ CLI consistency |
| 2 | `test_varpro_oracle.py` | `VarProFitter` against a SciPy least-squares oracle |
| 6 | `test_identifiability_mc.py` | Monte Carlo check of `voigtfit.identifiability`: 10⁴ simulated spectra fitted by exact constrained Poisson maximum likelihood (helper `_poisson_mle.py`, not shipped) against the inverse Fisher matrix in the interior; the estimator's distribution near the variance boundary is recorded, not judged |

### Lineshape & background — physics core (79)
| Tests | File | Guards |
|--:|---|---|
| 22 | `test_doniach_sunjic.py` | DS lineshape + MATLAB `LineshapeType` numbering |
| 27 | `test_fermi_edge.py` | Fermi-edge fit: agreement with `FermiDirac`, axis-shift and KE/BE invariance, 1σ pulls of E_F and resolution over 200 seeds, `success` gates (bounds, singular covariance, E_F off-window, collapsed width), fine-step starts, window-independent 10–90% width |
| 17 | `test_tougaard.py` | Tougaard background algorithm |
| 13 | `test_energy_axis.py` | BE / KE energy-axis handling through the pipeline |

### Templates & bridge (78)
| Tests | File | Guards |
|--:|---|---|
| 29 | `test_fitting_templates.py` | Fitting template system |
| 20 | `test_template_bridge.py` | `FittingTemplate` → `ComponentConfig` conversion |
| 17 | `test_fast_voigt_solvers.py` | `FastVoigtFitter` routing (incl. `use_mlx` on the multipeak path) + template binding-energy integration |
| 12 | `test_fit_dirty_map.py` | Interactive fit-result accumulator |

### GVRT / synthetic — paper reproduction (182)
| Tests | File | Guards |
|--:|---|---|
| 62 | `test_si2p_gvrt.py` | Si 2p doublet linear-encoder round trip (synthetic) |
| 47 | `test_si2p_suboxide.py` | Si 2p five sub-oxide-state fit — synthetic; a few tests additionally exercise a local measured map and **skip on a clean install** |
| 26 | `test_synthetic.py` | Synthetic data generation |
| 23 | `test_multi_image.py` | Multi-image GVRT generalization (smoke) |
| 9 | `test_ncomp_scaling.py` | `n_comp` scaling benchmark (smoke) |
| 6 | `test_example_05_map_from_file.py` | Example 05: reader columns → spectra, KE→BE, background sort, synthetic map recovered |
| 3 | `test_example_06_fermi_edge_calibration.py` | Example 06: a known axis offset and Au 4f7/2 = 84 eV recovered; default window finds E_F, not a steeper deeper band |
| 4 | `test_example_data.py` | `examples/data/` generator reproduces the shipped file |
| 2 | `test_gvrt_cli.py` | `gvrt` CLI subcommand (smoke) |

### I/O & readers (145)
| Tests | File | Guards |
|--:|---|---|
| 68 | `test_readers.py` | Format detection + reader factory (`.pxt` / `.vms` / `.npl`) |
| 37 | `test_readers_synthetic.py` | Reader behaviour on synthetic files — runs without local instrument data |
| 33 | `test_provenance_schema.py` | HDF5 provenance layout, versioning, and legacy-file fallback |
| 7 | `test_chunked_encoding.py` | Chunked vs monolithic `fitpara` encoder |

### Quantification data (510)
| Tests | File | Guards |
|--:|---|---|
| 123 | `test_imfp_tpp2m.py` | TPP-2M IMFP — implementation fidelity against the published table, and physical plausibility, kept separate |
| 45 | `test_cross_section_spin_orbit_limits.py` | Spin-orbit cross-section lookup and the limits of what it reports |
| 26 | `test_element_dedup.py` | Element-name dedup + `ElementInfo` utilities |
| 53 | `test_elastic_scattering.py` | Albedo-based EAL; required `model` keyword, published slopes, stated validity limits, and the `DescribedLength` / report provenance contract |
| 41 | `test_sampling_depth.py` | Mean escape depth and information depth; published slopes held apart from the EAL's, the required emission angle and percentage, the straight-line limits, and the source's own tabulated albedos |
| 37 | `test_sessa_sam_par.py` | SESSA `sam_par.txt` reader — header-driven columns and units, the one-row pairing and the detection of a pair split across rows, caller-declared material and version (invented fixtures only; no SESSA data bundled) |
| 25 | `test_compound_parameters.py` | `compounds.json` entries are arithmetically coherent with their own formulas — the one thing checkable without a recorded source |
| 22 | `test_compound_provenance.py` | `compounds_provenance.json` stays in step with `compounds.json` field by field, and never leaks into the values |
| 14 | `test_bundled_table_loading.py` | The `_cache/` tables are shipped data, not a rebuildable cache: a missing file raises rather than regenerating itself |
| 13 | `test_transmission_adapter.py` | Analyzer-transmission loader (synthetic fixtures only; no vendor data bundled) |
| 6 | `test_cross_section_tables.py` | Bundled cross-section tables load on a clean install |
| 102 | `test_angular_geometry.py` | θ (measured) vs ψ (derived) vs κ (the beam's own direction): the magic-angle identity, the factor-of-20 cost of confusing θ with ψ, the polarization average that puts the unpolarized angle on **k**, the two incidence-angle warnings and their boundaries, the tabulated fractions of subshells with a negative non-dipole term, and `L_full`'s present values pinned as a record while its convention is unresolved |
| 3 | `test_angular_correction_limits.py` | Angular-correction lookup rejects under-specified input |

### MCP server (13)
| Tests | File | Guards |
|--:|---|---|
| 13 | `test_mcp_server.py` | MCP server tools (direct call, no transport), including unit-status pass-through |

## VoigtFit engine — `src/toyomacro/voigtfit/tests/` (976)

### Solvers & fitting core (297)
| Tests | File | Guards |
|--:|---|---|
| 68 | `test_dictionary_solver.py` | δE / δE×δσ dictionary solver (single & two-peak) |
| 65 | `test_multipeak_solver.py` | Alternating-projection multipeak (2 / 3 / N components) |
| 35 | `test_gp_compact.py` | Compact Golub-Pereyra Hessian against the explicit form |
| 29 | `test_dictionary_solver_3d.py` | 3-D dictionary solver δE×δσ×δγ |
| 23 | `test_gp_jacobian.py` | Golub-Pereyra variable-projection Jacobian |
| 19 | `test_voigt_jacobian.py` | Voigt Jacobian / Hessian analytical derivatives |
| 17 | `test_param_roundtrip.py` | Parameter encode → solve round trip |
| 17 | `test_ds_basis.py` | Doniach-Sunjic basis generation (amplitude-only) |
| 15 | `test_gp_lm.py` | GP Levenberg-Marquardt acceptance rules and background designs |
| 5 | `test_integration.py` | Synthetic accuracy + MATLAB bridge + performance |
| 3 | `test_voigtfit.py` | Top-level smoke |
| 1 | `test_pipeline_mlx.py` | MLX pipeline smoke |

### GVRT & encoders — paper reproduction (225)
| Tests | File | Guards |
|--:|---|---|
| 37 | `test_split_encoder.py` | Hilbert 2-D scalar / vectorized / locality |
| 34 | `test_neighbor_preservation.py` | Fisher-whitened transform, bbox neighbor preservation |
| 30 | `test_hilbert_encoder.py` | 3-D Hilbert curve + `MultiPeakEncoder` |
| 28 | `test_gvrt_multipeak_1b.py` | 10⁹-spectrum multipeak benchmark |
| 25 | `test_gvrt_multipeak.py` | Multipeak GVRT presets & generation |
| 23 | `test_gvrt_service.py` | Interactive GVRT service (GUI / MCP engine) |
| 21 | `test_split_encoder_e2e.py` | Split encoder end-to-end pipeline |
| 14 | `test_gvrt_noise.py` | Noise-robustness visualization |
| 13 | `test_gvrt_tracking.py` | Parameter-tracking visualization |

### Statistics & information theory (366)
| Tests | File | Guards |
|--:|---|---|
| 108 | `test_identifiability.py` | Width identifiability: Voigt derivatives in the Gaussian variance down to σ = 0 (two routes and a quadrature reference, the asymptotic limit on the term count), Poisson Fisher matrix with a background, effective vs conditional information, the labels and their unit invariance, the condition scan |
| 37 | `test_fisher_transform.py` | Fisher coordinate transform / anisotropy / decorrelation |
| 36 | `test_fisher_information.py` | Fisher information basic / scaling / 3-D vs 4-D |
| 54 | `test_crlb.py` | Cramér-Rao lower bound consistency / symmetry / overlap |
| 27 | `test_grids.py` | Uniform / sinh / Chebyshev grids |
| 26 | `test_exact_k.py` | Component-count selection: candidate fits, residual gate, background handling |
| 24 | `test_extended_svd.py` | Extended SVD on Voigt spectra |
| 19 | `test_model_selection.py` | AIC / AICc / BIC scoring, profile likelihood, non-finite handling |
| 18 | `test_rank_diagnostics.py` | Numerical rank and conditioning of structured Voigt designs |
| 15 | `test_error_stats.py` | Error-statistics computation |
| 2 | `test_bench_rank_model_selection.py` | Rank/model-selection benchmark stays runnable |

### Infrastructure — compression, memory, I/O (88)
| Tests | File | Guards |
|--:|---|---|
| 34 | `test_memory.py` | Memory detection, optimal chunk size, dict3d cache size |
| 22 | `test_compression.py` | `fitpara` compression codec |
| 14 | `test_streaming_write.py` | Streaming HDF5 write |
| 11 | `test_compression_integration.py` | Pipeline I/O compression |
| 7 | `test_convergence_warnings.py` | Iterative refiners stay silent on their first pass — the infinity sentinel must not be divided by |

---

*Regenerate the numbers with `pytest --collect-only -q`.*
