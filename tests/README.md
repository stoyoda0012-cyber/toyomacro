# Test inventory

This suite has **1,585 automated tests** across **67 files**, in two
locations:

| Location | Scope | Files | Tests |
|---|---|--:|--:|
| `tests/` | Library body — lineshapes, backgrounds, templates, I/O, quantification, meta | 32 | 746 |
| `src/toyomacro/voigtfit/tests/` | VoigtFit engine — solvers, encoders, information theory | 35 | 839 |

Every test here runs on a plain `pip install` (no GUI or instrument
data required). Counts below come from `pytest --collect-only`.

## Two kinds of test

Tests fall into two purposes. The distinction matters when deciding
what to run:

- **Contract / regression** (1,153 tests, 73%) — guarantee the library
  behaves correctly: lineshape math, background algorithms, solver
  routing, file readers, template conversion, and the reference-data
  tables. Fast, deterministic.
- **Paper reproduction** (432 tests, 27%) — reproduce the accuracy
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

## Library body — `tests/` (746)

### Claim guards — noise model, versions, backends, comparisons (48)
| Tests | File | Guards |
|--:|---|---|
| 19 | `test_noise_semantics.py` | `level` ↔ peak-SNR ↔ Poisson-mean conversions, empirically pinned to the sampler |
| 10 | `test_mlx_support.py` | MLX absent / installed-but-unusable / usable; NumPy fallback end-to-end |
| 10 | `test_solver_comparison.py` | Same-problem scipy/lmfit comparison benchmark stays runnable + seed reproducibility |
| 4 | `test_cli_surface.py` | CLI entry points stay importable and keep their documented flags |
| 3 | `test_version.py` | pyproject ↔ `toyomacro.__version__` ↔ voigtfit ↔ CLI consistency |
| 2 | `test_varpro_oracle.py` | `VarProFitter` against a SciPy least-squares oracle |

### Lineshape & background — physics core (52)
| Tests | File | Guards |
|--:|---|---|
| 22 | `test_doniach_sunjic.py` | DS lineshape + MATLAB `LineshapeType` numbering |
| 17 | `test_tougaard.py` | Tougaard background algorithm |
| 13 | `test_energy_axis.py` | BE / KE energy-axis handling through the pipeline |

### Templates & bridge (76)
| Tests | File | Guards |
|--:|---|---|
| 29 | `test_fitting_templates.py` | Fitting template system |
| 20 | `test_template_bridge.py` | `FittingTemplate` → `ComponentConfig` conversion |
| 15 | `test_fast_voigt_solvers.py` | `FastVoigtFitter` routing + template binding-energy integration |
| 12 | `test_fit_dirty_map.py` | Interactive fit-result accumulator |

### GVRT / synthetic — paper reproduction (173)
| Tests | File | Guards |
|--:|---|---|
| 62 | `test_si2p_gvrt.py` | Si 2p doublet linear-encoder round trip (synthetic) |
| 47 | `test_si2p_suboxide.py` | Si 2p five sub-oxide-state fit — synthetic; a few tests additionally exercise a local measured map and **skip on a clean install** |
| 26 | `test_synthetic.py` | Synthetic data generation |
| 23 | `test_multi_image.py` | Multi-image GVRT generalization (smoke) |
| 9 | `test_ncomp_scaling.py` | `n_comp` scaling benchmark (smoke) |
| 4 | `test_example_data.py` | `examples/data/` generator reproduces the shipped file |
| 2 | `test_gvrt_cli.py` | `gvrt` CLI subcommand (smoke) |

### I/O & readers (145)
| Tests | File | Guards |
|--:|---|---|
| 68 | `test_readers.py` | Format detection + reader factory (`.pxt` / `.vms` / `.npl`) |
| 37 | `test_readers_synthetic.py` | Reader behaviour on synthetic files — runs without local instrument data |
| 33 | `test_provenance_schema.py` | HDF5 provenance layout, versioning, and legacy-file fallback |
| 7 | `test_chunked_encoding.py` | Chunked vs monolithic `fitpara` encoder |

### Quantification data (239)
| Tests | File | Guards |
|--:|---|---|
| 123 | `test_imfp_tpp2m.py` | TPP-2M IMFP — implementation fidelity against the published table, and physical plausibility, kept separate |
| 45 | `test_cross_section_spin_orbit_limits.py` | Spin-orbit cross-section lookup and the limits of what it reports |
| 26 | `test_element_dedup.py` | Element-name dedup + `ElementInfo` utilities |
| 23 | `test_elastic_scattering.py` | Albedo-based EAL; required `model` keyword, published slopes, stated validity limits |
| 13 | `test_transmission_adapter.py` | Analyzer-transmission loader (synthetic fixtures only; no vendor data bundled) |
| 6 | `test_cross_section_tables.py` | Bundled cross-section tables load on a clean install |
| 3 | `test_angular_correction_limits.py` | Angular-correction lookup rejects under-specified input |

### MCP server (13)
| Tests | File | Guards |
|--:|---|---|
| 13 | `test_mcp_server.py` | MCP server tools (direct call, no transport), including unit-status pass-through |

## VoigtFit engine — `src/toyomacro/voigtfit/tests/` (839)

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

### Statistics & information theory (236)
| Tests | File | Guards |
|--:|---|---|
| 37 | `test_fisher_transform.py` | Fisher coordinate transform / anisotropy / decorrelation |
| 36 | `test_fisher_information.py` | Fisher information basic / scaling / 3-D vs 4-D |
| 32 | `test_crlb.py` | Cramér-Rao lower bound consistency / symmetry / overlap |
| 27 | `test_grids.py` | Uniform / sinh / Chebyshev grids |
| 26 | `test_exact_k.py` | Component-count selection: candidate fits, residual gate, background handling |
| 24 | `test_extended_svd.py` | Extended SVD on Voigt spectra |
| 19 | `test_model_selection.py` | AIC / AICc / BIC scoring, profile likelihood, non-finite handling |
| 18 | `test_rank_diagnostics.py` | Numerical rank and conditioning of structured Voigt designs |
| 15 | `test_error_stats.py` | Error-statistics computation |
| 2 | `test_bench_rank_model_selection.py` | Rank/model-selection benchmark stays runnable |

### Infrastructure — compression, memory, I/O (81)
| Tests | File | Guards |
|--:|---|---|
| 34 | `test_memory.py` | Memory detection, optimal chunk size, dict3d cache size |
| 22 | `test_compression.py` | `fitpara` compression codec |
| 14 | `test_streaming_write.py` | Streaming HDF5 write |
| 11 | `test_compression_integration.py` | Pipeline I/O compression |

---

*Regenerate the numbers with `pytest --collect-only -q`.*
