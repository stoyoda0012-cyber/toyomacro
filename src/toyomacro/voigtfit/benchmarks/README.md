# voigtfit benchmarks

Maintainer tools for measuring solver throughput, accuracy ceilings,
and scaling behavior. **Not run in CI** — they are demonstration and
profiling scripts, not part of the test suite.

Run any module as:

```bash
python -m toyomacro.voigtfit.benchmarks.<module>
```

Docstrings occasionally cite `dev-log NN` — references to the
maintainer's development log; see CONTRIBUTING.md.

## Standalone (synthetic data only — run anywhere)

| Module | What it measures |
|--------|------------------|
| `theoretical_limits` | Roofline analysis: memory/compute/SSD bounds vs measured Stage 1 rate |
| `benchmark_rowmajor`, `benchmark_stage1_pipeline` | Stage 1 amplitude-kernel throughput |
| `bench_dict2d`, `bench_dict2d_sweep`, `bench_dict2d_margin` | Dict2D solver accuracy/throughput vs grid settings |
| `bench_dict2d_noise` | Dict2D vs Taylor crossover under Poisson noise |
| `bench_dict2d_adaptive`, `bench_dict2d_adaptive_fused` | Adaptive two-round solver; fused variant vs baseline |
| `bench_dict3d_psnr`, `bench_gamma_perturbation` | 3-D dictionary (γ recovery) accuracy |
| `bench_gvrt_1b`, `bench_gvrt_multipeak_1b`, `bench_gvrt_multipeak_roundtrip` | Giga-Voigt round trip at the 10⁹-spectra scale |
| `bench_multipeak_4d`, `bench_multipeak_sweep`, `bench_ncomp_scaling` | Multipeak alternating-projection scaling |
| `bench_split_encoder_e2e`, `bench_jitter_limits`, `bench_nonuniform_grid` | Encoder / grid robustness studies |
| `poisson_benchmark` | Poisson noise-injection kernel (spec: `poisson_benchmark_spec.md`) |
| `chunk_optimization_benchmark` | Chunk-size tuning for the MLX pipeline |
| `gp_newton_comparison` | Post-AP Newton Jacobian study: raw vs Kaufman vs Golub–Pereyra (report: `GP_JACOBIAN_VERIFICATION.md`) |
| `bench_gp_lm` | Compact GP + LM safeguard: solver comparison, assembly micro-bench, auto-routing (report: `GP_LM_HARDENING_REPORT.md`) |

## Data-driven (require external images/HDF5 via `VOIGTFIT_DATA_ROOT`)

| Module | What it measures |
|--------|------------------|
| `bottleneck_analysis` | Phase-level profiling of the fused gen+fit pipeline |
| `benchmark_mlx_pipeline`, `benchmark_h5` | End-to-end throughput incl. HDF5 I/O |
| `roundtrip_benchmark`, `param_roundtrip_benchmark` | Image → spectra → fit → image round trip (GVRT) |
| `psnr_noise_benchmark`, `error_map_benchmark` | Reconstruction quality vs noise level |
| `bench_gvrt_noise`, `bench_gvrt_tracking`, `bench_multi_image` | GVRT robustness/tracking on image sets |
| `giga_mosaic_benchmark`, `reconstruction_benchmark` | Large-mosaic scaling |

Paths are environment-configurable in `_data_paths.py`
(`VOIGTFIT_DATA_ROOT`, `VOIGTFIT_GAZOU_DIR`); no data ships with the
repository. A fully self-contained image round trip is available
without any external data via `toyomacro.voigtfit.gvrt_service`
(synthetic demo image) — see `paper/figures/make_figure2_gvrt_noise.py`
for a worked example.
