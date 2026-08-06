"""
MLX-Fused Pipeline Bottleneck Analysis

Detailed phase profiling at optimal chunk size (2M) to identify
where the gen+fit → fit-only throughput gap comes from.

Theoretical max: ~400M spec/s (VoigtFit fit-only, GigaVoigt dev-log 35)
Current gen+fit: ~28M spec/s (MLX-fused roundtrip E2E, dev-log 35)
Gap: ~93% — spectra generation + noise (~90% of E2E time) is the bottleneck, not fit.

Phases measured:
  1. A construction (NumPy): indexing + amplitude_scale multiply
  2. np→mx transfer: mx.array(A_np), mx.array(bg_np), mx.array(P_np)
  3. MLX matmul (lazy): A_mx @ P_mx + bg_mx (graph construction only)
  4. MLX noise (lazy): Cornish-Fisher Poisson on GPU
  5. mx.eval(): Force lazy graph evaluation (matmul + noise)
  6. VoigtFit process_rowmajor():
     6a. cache lookup (weight matrix)
     6b. Y already mx.array → skip conversion
     6c. stage1: Y_mx @ W (MLX matmul)
     6d. chi2 sampling
     6e. mx→np transfer: np.array(A_row).T
     6f. anomaly detection
     6g. tile/output construction
  7. Python loop overhead (for nl in noise_levels, for variant in variants)
  8. GC

Usage:
    python -m toyomacro.voigtfit.benchmarks.bottleneck_analysis
    python -m toyomacro.voigtfit.benchmarks.bottleneck_analysis --frames 20
"""

import gc
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx

    from toyomacro.voigtfit._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND the default device can execute work
except ImportError:
    HAS_MLX = False


@dataclass
class DetailedTimings:
    """Micro-phase timings for bottleneck analysis."""
    # Phase 1: A construction (NumPy)
    a_construction: float = 0
    # Phase 2: np→mx transfers
    np_to_mx: float = 0
    # Phase 3: MLX matmul (lazy graph build)
    matmul_lazy: float = 0
    # Phase 4: MLX noise (lazy graph build)
    noise_lazy: float = 0
    # Phase 5: mx.eval() — forces computation
    mx_eval: float = 0
    # Phase 5a/5b: separated matmul vs noise GPU time (split_eval mode)
    mx_eval_matmul: float = 0
    mx_eval_noise: float = 0
    # Phase 6: VoigtFit
    voigtfit_total: float = 0
    # Phase 6 sub-phases (from timing dict)
    vf_cache_lookup: float = 0
    vf_mlx_convert: float = 0
    vf_stage1: float = 0
    vf_anomaly: float = 0
    vf_other: float = 0
    # Phase 7: Python overhead (loop, dict ops, del)
    python_overhead: float = 0
    # Phase 8: GC
    gc_time: float = 0
    # Totals
    total: float = 0
    total_spectra: int = 0
    n_fit_calls: int = 0
    n_noise_calls: int = 0
    n_eval_calls: int = 0
    # Mode
    split_eval: bool = False

    @property
    def throughput(self) -> float:
        return self.total_spectra / self.total if self.total > 0 else 0

    @property
    def genfit_time(self) -> float:
        """Time excluding GC and Python overhead."""
        return (self.a_construction + self.np_to_mx + self.matmul_lazy +
                self.noise_lazy + self.mx_eval + self.voigtfit_total)


def _get_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / (1024 * 1024)


def run_bottleneck_analysis(
    n_test: int = 10,
    gif_path: str | None = None,
    input_path: str | None = None,
    split_eval: bool = False,
) -> DetailedTimings:
    """Run detailed bottleneck profiling of MLX-fused pipeline."""
    from toyomacro.voigtfit.image_utils import load_image
    from toyomacro.voigtfit.pipeline import HybridPipeline
    from toyomacro.voigtfit.spectra_generator import (
        GeneratorConfig,
        NoiseConfig,
        add_poisson_noise_mlx_fused,
        decompose_image_to_amplitudes,
        get_element_preset,
        voigt_profile,
    )
    from toyomacro.voigtfit.weight_cache import WeightMatrixCache

    from ._data_paths import roundtrip_image_dir

    # Resolve input path (--input takes precedence over --gif)
    media_path = input_path or gif_path
    if media_path is None:
        from ._data_paths import find_default_image

        found = find_default_image()
        if found is None:
            raise SystemExit(
                'No input media found — pass --input/--gif or set '
                'VOIGTFIT_DATA_ROOT to a tree containing images.'
            )
        media_path = str(found)

    frames = load_image(media_path, all_frames=True)
    n_frames, h, w, _ = frames.shape
    n_pixels = h * w
    print(f"Loaded: {w}x{h}, {n_frames} frames, {n_pixels:,} px/frame")
    print(f"Testing: {n_test} frames × 2 NL × 5 file_keys = "
          f"{n_test * 2 * 5:,} fit calls")
    print(f"Total spectra: {n_test * n_pixels * 2 * 5:,}")

    preset = get_element_preset('demo')
    config = GeneratorConfig()
    color_mapping = preset.color_mapping

    elem_by_file = {}
    for idx, elem in enumerate(preset.elements):
        fk = elem.file_key
        if fk not in elem_by_file:
            elem_by_file[fk] = []
        elem_by_file[fk].append((idx, elem))

    file_configs = {}
    for file_key, elem_list in elem_by_file.items():
        elem_specs = [e for _, e in elem_list]
        elem_indices = [idx for idx, _ in elem_list]
        binding_energies = [e.binding_energy for e in elem_specs]
        e_min = min(binding_energies) - config.energy_edge
        e_max = max(binding_energies) + config.energy_edge
        energy = np.arange(e_min, e_max + config.energy_step / 2,
                          config.energy_step, dtype=np.float64).astype(np.float32)
        profiles = []
        for elem in elem_specs:
            prof = voigt_profile(energy, elem.binding_energy, elem.sigma, elem.gamma)
            prof = (prof / (prof.max() + 1e-10)).astype(np.float32)
            profiles.append(prof)
        peak_config = {
            'centers': np.array([e.binding_energy for e in elem_specs], dtype=np.float32),
            'sigmas': np.array([e.sigma for e in elem_specs], dtype=np.float32),
            'gamma': elem_specs[0].gamma,
        }
        file_configs[file_key] = {
            'elem_specs': elem_specs, 'elem_indices': elem_indices,
            'energy': energy, 'profiles': profiles, 'peak_config': peak_config,
        }

    n_file_keys = len(file_configs)
    channel_sizes = [len(fc['energy']) for fc in file_configs.values()]
    print(f"File keys: {n_file_keys}, channels: {channel_sizes}")

    noise_levels = ['None', 'Moderate']
    noise_configs = {
        'None': NoiseConfig(noise_type='none'),
        'Moderate': NoiseConfig(noise_type='poisson', poisson_level=1e3),
    }

    # Decompose frames
    print(f"\nDecomposing {n_test} frames...", end='', flush=True)
    all_amps = []
    for fi in range(min(n_test, n_frames)):
        amp = decompose_image_to_amplitudes(frames[fi], color_mapping, method='pinv')
        all_amps.append(amp)
    print(" done")

    # Pre-convert profiles to MLX
    mlx_profiles = {}
    for file_key, fc in file_configs.items():
        P_np = np.stack(fc['profiles'], axis=0)
        mlx_profiles[file_key] = mx.array(P_np)

    # Warmup
    print("Warming up pipeline...", end='', flush=True)
    cache = WeightMatrixCache()
    pipeline = HybridPipeline(cache, enable_stage2=False, use_mlx=True)
    for file_key, fc in file_configs.items():
        dummy = mx.zeros((n_pixels, len(fc['energy'])), dtype=mx.float32)
        mx.eval(dummy)
        _ = pipeline.process_rowmajor(
            dummy, fc['elem_specs'][0].symbol, fc['elem_specs'][0].orbital,
            fc['energy'], fc['peak_config'],
        )
    gc.collect()
    print(" done\n")

    # === Main benchmark with micro-phase timing ===
    t = DetailedTimings()
    t.split_eval = split_eval
    t_total_start = time.perf_counter()

    if split_eval:
        print("  [split_eval mode: inserting mx.eval() between matmul and noise]")

    for ci in range(n_test):
        chunk_amp = all_amps[ci]

        t_loop_start = time.perf_counter()

        for file_key, fc in file_configs.items():
            elem_indices = fc['elem_indices']
            energy = fc['energy']
            peak_config = fc['peak_config']
            elem_specs = fc['elem_specs']
            n_profiles = len(fc['profiles'])

            # === Phase 1: A construction (NumPy indexing + scaling) ===
            t0 = time.perf_counter()
            A_np = np.zeros((n_pixels, n_profiles), dtype=np.float32)
            for j, elem_idx in enumerate(elem_indices):
                A_np[:, j] = np.maximum(chunk_amp[elem_idx, :], 0)
            bg_np = (A_np.sum(axis=1, keepdims=True)
                     * (config.amplitude_scale * config.bg_base)).astype(np.float32)
            A_scaled = A_np * config.amplitude_scale
            t.a_construction += time.perf_counter() - t0

            # === Phase 2: np→mx transfer ===
            t0 = time.perf_counter()
            A_mx = mx.array(A_scaled)
            bg_mx = mx.array(bg_np)
            P_mx = mlx_profiles[file_key]
            del A_np, A_scaled, bg_np
            t.np_to_mx += time.perf_counter() - t0

            # === Phase 3: MLX matmul (lazy) ===
            t0 = time.perf_counter()
            all_clean_mx = A_mx @ P_mx + bg_mx
            del A_mx, bg_mx
            t.matmul_lazy += time.perf_counter() - t0

            # === Split eval: force matmul computation before noise ===
            if split_eval:
                t0 = time.perf_counter()
                mx.eval(all_clean_mx)
                t.mx_eval_matmul += time.perf_counter() - t0

            for nl in noise_levels:
                nc = noise_configs[nl]

                # === Phase 4: MLX noise (lazy) ===
                t0 = time.perf_counter()
                if nc.noise_type == 'none':
                    spectra_mx = all_clean_mx
                elif nc.noise_type == 'poisson':
                    spectra_mx = add_poisson_noise_mlx_fused(
                        all_clean_mx, nc.poisson_level
                    )
                    t.n_noise_calls += 1
                else:
                    spectra_mx = all_clean_mx
                t.noise_lazy += time.perf_counter() - t0

                # === Phase 5: mx.eval() — forces GPU computation ===
                t0 = time.perf_counter()
                mx.eval(spectra_mx)
                t_eval = time.perf_counter() - t0
                t.mx_eval += t_eval
                t.n_eval_calls += 1

                # In split_eval mode, this eval is noise-only (matmul already done)
                if split_eval and nc.noise_type == 'poisson':
                    t.mx_eval_noise += t_eval

                # === Phase 6: VoigtFit process_rowmajor ===
                t0 = time.perf_counter()
                result = pipeline.process_rowmajor(
                    spectra_mx,
                    elem_specs[0].symbol, elem_specs[0].orbital,
                    energy, peak_config,
                    amplitudes_only=True,
                )
                t_vf = time.perf_counter() - t0
                t.voigtfit_total += t_vf
                t.total_spectra += spectra_mx.shape[0]
                t.n_fit_calls += 1

                # Extract VoigtFit sub-timings
                if result.timing:
                    t.vf_cache_lookup += result.timing.get('cache_lookup', 0)
                    t.vf_mlx_convert += result.timing.get('mlx_convert', 0)
                    t.vf_stage1 += result.timing.get('stage1', 0)
                    t.vf_anomaly += result.timing.get('anomaly_detection', 0)
                    t_sub = (result.timing.get('cache_lookup', 0) +
                             result.timing.get('mlx_convert', 0) +
                             result.timing.get('stage1', 0) +
                             result.timing.get('anomaly_detection', 0))
                    t.vf_other += max(t_vf - t_sub, 0)

                if spectra_mx is not all_clean_mx:
                    del spectra_mx

            del all_clean_mx

        # === Phase 8: GC ===
        t0 = time.perf_counter()
        gc.collect()
        t.gc_time += time.perf_counter() - t0

    t.total = time.perf_counter() - t_total_start

    # Phase 7: Python overhead = total - sum of all measured phases
    t_measured = (t.a_construction + t.np_to_mx + t.matmul_lazy +
                  t.noise_lazy + t.mx_eval + t.voigtfit_total + t.gc_time)
    t.python_overhead = max(t.total - t_measured, 0)

    return t


def print_report(t: DetailedTimings) -> None:
    """Print detailed bottleneck report."""
    print(f"\n{'='*70}")
    print("BOTTLENECK ANALYSIS: MLX-Fused Pipeline (Optimal 2M Chunk)")
    print(f"{'='*70}")
    print(f"Total spectra: {t.total_spectra:,}")
    print(f"Fit calls: {t.n_fit_calls}")
    print(f"Eval calls: {t.n_eval_calls}")
    print(f"Noise calls: {t.n_noise_calls}")

    theoretical_max = 400e6  # spec/s from GigaVoigt dev-log 35 fit-only
    current = t.throughput

    print(f"\nThroughput: {current/1e6:.2f} M spec/s")
    print(f"Theoretical max: {theoretical_max/1e6:.0f} M spec/s")
    print(f"Efficiency: {current/theoretical_max*100:.1f}%")
    print(f"Gap: {(theoretical_max - current)/1e6:.1f} M spec/s")

    # Phase breakdown
    phases = [
        ('1. A construction (NumPy)', t.a_construction),
        ('2. np→mx transfer',         t.np_to_mx),
        ('3. MLX matmul (lazy)',       t.matmul_lazy),
        ('4. MLX noise (lazy)',        t.noise_lazy),
        ('5. mx.eval() [GPU exec]',   t.mx_eval),
        ('6. VoigtFit total',         t.voigtfit_total),
        ('7. Python overhead',         t.python_overhead),
        ('8. GC',                      t.gc_time),
    ]

    print(f"\n{'Phase':<30s} {'Time(s)':>8s} {'%':>6s} {'ms/call':>8s}")
    print('-' * 56)

    for name, val in phases:
        pct = val / t.total * 100 if t.total > 0 else 0
        if 'eval' in name or 'VoigtFit' in name:
            n = t.n_eval_calls if 'eval' in name else t.n_fit_calls
        elif 'noise' in name.lower() and 'lazy' in name:
            n = t.n_noise_calls
        elif 'A constr' in name or 'np→mx' in name or 'matmul' in name:
            n = t.n_fit_calls // 2  # 1 per file_key per NL, but A/matmul is per file_key
        else:
            n = t.n_fit_calls
        ms_per = val / n * 1000 if n > 0 else 0
        print(f"  {name:<28s} {val:>7.3f}s {pct:>5.1f}% {ms_per:>7.2f}")

    print(f"  {'─'*28} {'─'*7} {'─'*5}")
    print(f"  {'TOTAL':<28s} {t.total:>7.3f}s 100.0%")

    # VoigtFit sub-phases
    print(f"\n  VoigtFit breakdown ({t.n_fit_calls} calls):")
    vf_sub = [
        ('cache_lookup',   t.vf_cache_lookup),
        ('mlx_convert',    t.vf_mlx_convert),
        ('stage1 (Y@W)',   t.vf_stage1),
        ('anomaly_detect', t.vf_anomaly),
        ('other (tile/etc)', t.vf_other),
    ]
    for name, val in vf_sub:
        pct_vf = val / t.voigtfit_total * 100 if t.voigtfit_total > 0 else 0
        ms_per = val / t.n_fit_calls * 1000 if t.n_fit_calls > 0 else 0
        print(f"    {name:<20s} {val:>7.3f}s ({pct_vf:>5.1f}%) {ms_per:>7.2f} ms/call")

    # Split eval breakdown
    if t.split_eval:
        print("\n  ★ mx.eval() Split Analysis (matmul vs noise):")
        total_gpu = t.mx_eval_matmul + t.mx_eval + t.mx_eval_noise
        # mx_eval_matmul: matmul-only eval (before noise loop)
        # mx_eval_noise: noise-only eval (poisson NLs in split mode)
        # mx_eval - mx_eval_noise: clean NL evals (matmul already done, near-zero)
        matmul_gpu = t.mx_eval_matmul
        noise_gpu = t.mx_eval_noise
        clean_eval = t.mx_eval - t.mx_eval_noise  # eval for 'None' NL (cached, ~0)
        total_gen_gpu = matmul_gpu + noise_gpu + clean_eval
        print(f"    Matmul GPU:    {matmul_gpu:.3f}s "
              f"({matmul_gpu/total_gen_gpu*100:.1f}%)" if total_gen_gpu > 0 else "")
        print(f"    Noise GPU:     {noise_gpu:.3f}s "
              f"({noise_gpu/total_gen_gpu*100:.1f}%)" if total_gen_gpu > 0 else "")
        print(f"    Clean eval:    {clean_eval:.3f}s "
              f"({clean_eval/total_gen_gpu*100:.1f}%)" if total_gen_gpu > 0 else "")
        print(f"    Total GPU gen: {total_gen_gpu:.3f}s")
        if matmul_gpu > 0 and noise_gpu > 0:
            print(f"    Ratio: matmul:noise = {matmul_gpu/noise_gpu:.2f}:1")

    # Bottleneck analysis
    print(f"\n{'='*70}")
    print("BOTTLENECK SUMMARY")
    print(f"{'='*70}")

    # Sort by time
    all_phases = sorted(phases, key=lambda x: x[1], reverse=True)
    for i, (name, val) in enumerate(all_phases[:3]):
        pct = val / t.total * 100
        lost = val / t.total * current  # spec/s "lost" to this phase
        print(f"  #{i+1}: {name} — {val:.3f}s ({pct:.1f}%) "
              f"≈ {lost/1e6:.1f}M spec/s consumed")

    # Key insight: what would happen if we eliminated each overhead
    print("\n  Hypothetical improvements:")
    gen_overhead = t.a_construction + t.np_to_mx + t.matmul_lazy + t.noise_lazy + t.mx_eval
    vf_only_time = t.voigtfit_total
    ideal_time = vf_only_time  # if all gen overhead was zero
    ideal_tp = t.total_spectra / ideal_time if ideal_time > 0 else 0
    print(f"    If gen overhead = 0: {ideal_tp/1e6:.1f} M spec/s "
          f"(= VoigtFit-only throughput)")

    no_gc_time = t.total - t.gc_time
    no_gc_tp = t.total_spectra / no_gc_time if no_gc_time > 0 else 0
    print(f"    If GC = 0: {no_gc_tp/1e6:.1f} M spec/s")

    no_python_time = t.total - t.python_overhead
    no_python_tp = t.total_spectra / no_python_time if no_python_time > 0 else 0
    print(f"    If Python overhead = 0: {no_python_tp/1e6:.1f} M spec/s")

    no_eval_time = t.total - t.mx_eval
    no_eval_tp = t.total_spectra / no_eval_time if no_eval_time > 0 else 0
    print(f"    If mx.eval() = 0: {no_eval_tp/1e6:.1f} M spec/s")

    # VoigtFit internal: mx→np in stage1
    vf_mx_np_time = t.vf_other  # Mostly np.array(A_row).T + output construction
    no_mx_np_time = t.total - vf_mx_np_time
    no_mx_np_tp = t.total_spectra / no_mx_np_time if no_mx_np_time > 0 else 0
    print(f"    If VoigtFit mx→np = 0: {no_mx_np_tp/1e6:.1f} M spec/s")

    # Combined: no gen + no gc + no python
    min_time = t.voigtfit_total
    min_tp = t.total_spectra / min_time if min_time > 0 else 0
    print(f"\n  Absolute minimum (VoigtFit only): {min_tp/1e6:.1f} M spec/s")
    print(f"  VoigtFit internal bottleneck: stage1={t.vf_stage1:.3f}s, "
          f"other={t.vf_other:.3f}s")


def plot_results(t: DetailedTimings, output_path: str | None = None) -> None:
    """Generate bottleneck visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Phase breakdown waterfall
    ax = axes[0, 0]
    phases = [
        ('A constr.', t.a_construction, '#e74c3c'),
        ('np→mx', t.np_to_mx, '#f39c12'),
        ('Matmul(lazy)', t.matmul_lazy, '#e67e22'),
        ('Noise(lazy)', t.noise_lazy, '#3498db'),
        ('mx.eval()', t.mx_eval, '#9b59b6'),
        ('VoigtFit', t.voigtfit_total, '#2ecc71'),
        ('Python OH', t.python_overhead, '#bdc3c7'),
        ('GC', t.gc_time, '#95a5a6'),
    ]
    names = [p[0] for p in phases]
    vals = [p[1] for p in phases]
    colors = [p[2] for p in phases]
    bars = ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Time (s)')
    ax.set_title('Phase Breakdown (Waterfall)')
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        pct = val / t.total * 100
        if val > 0.01:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}s ({pct:.1f}%)', va='center', fontsize=8)

    # Panel 2: Pie chart
    ax = axes[0, 1]
    gen_time = t.a_construction + t.np_to_mx + t.matmul_lazy + t.noise_lazy + t.mx_eval
    pie_phases = {
        'Gen overhead': gen_time,
        'VoigtFit': t.voigtfit_total,
        'Python OH': t.python_overhead,
        'GC': t.gc_time,
    }
    pie_colors = ['#e74c3c', '#2ecc71', '#bdc3c7', '#95a5a6']
    vals_pie = [v for v in pie_phases.values() if v > 0.01]
    lbls_pie = [k for k, v in pie_phases.items() if v > 0.01]
    ax.pie(vals_pie, labels=lbls_pie, colors=pie_colors[:len(vals_pie)],
           autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Time Distribution ({t.throughput/1e6:.1f} M spec/s)')

    # Panel 3: VoigtFit internal breakdown
    ax = axes[1, 0]
    vf_phases = [
        ('cache', t.vf_cache_lookup, '#27ae60'),
        ('mx convert', t.vf_mlx_convert, '#f1c40f'),
        ('stage1\n(Y@W+chi2)', t.vf_stage1, '#2ecc71'),
        ('anomaly', t.vf_anomaly, '#16a085'),
        ('other\n(mx→np+tile)', t.vf_other, '#1abc9c'),
    ]
    vf_names = [p[0] for p in vf_phases]
    vf_vals = [p[1] for p in vf_phases]
    vf_colors = [p[2] for p in vf_phases]
    bars = ax.barh(range(len(vf_names)), vf_vals, color=vf_colors)
    ax.set_yticks(range(len(vf_names)))
    ax.set_yticklabels(vf_names)
    ax.set_xlabel('Time (s)')
    ax.set_title('VoigtFit Internal Breakdown')
    ax.invert_yaxis()
    for bar, val in zip(bars, vf_vals):
        pct = val / t.voigtfit_total * 100 if t.voigtfit_total > 0 else 0
        if val > 0.005:
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}s ({pct:.1f}%)', va='center', fontsize=8)

    # Panel 4: Throughput waterfall — show what each phase costs
    ax = axes[1, 1]
    theoretical = 400.0  # M spec/s (fit-only, GigaVoigt dev-log 35)
    current_tp = t.throughput / 1e6

    # Calculate throughput loss per phase
    losses = []
    for name, val, _ in phases:
        tp_without = t.total_spectra / (t.total - val) / 1e6 if (t.total - val) > 0 else 0
        loss = tp_without - current_tp
        losses.append((name, loss))

    # Sort by loss
    losses.sort(key=lambda x: x[1], reverse=True)
    loss_names = [l[0] for l in losses[:6]]
    loss_vals = [l[1] for l in losses[:6]]
    loss_colors = ['#e74c3c' if v > 1 else '#f39c12' if v > 0.5 else '#95a5a6'
                   for v in loss_vals]

    bars = ax.barh(range(len(loss_names)), loss_vals, color=loss_colors)
    ax.set_yticks(range(len(loss_names)))
    ax.set_yticklabels(loss_names)
    ax.set_xlabel('Throughput gain if eliminated (M spec/s)')
    ax.set_title(f'Throughput Loss per Phase (current={current_tp:.1f}M)')
    ax.invert_yaxis()
    for bar, val in zip(bars, loss_vals):
        if val > 0.1:
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f'+{val:.1f}', va='center', fontsize=9)

    fig.suptitle(f'Bottleneck Analysis: MLX-Fused Pipeline @ 2M Chunk\n'
                 f'{current_tp:.1f} M spec/s / {theoretical:.0f} M theoretical '
                 f'= {current_tp/theoretical*100:.0f}% efficiency',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if output_path is None:
        output_path = (Path(__file__).parent.parent.parent
                       / 'outputs' / 'bottleneck_analysis.png')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='MLX-Fused Pipeline Bottleneck Analysis')
    parser.add_argument('--frames', type=int, default=10,
                       help='Number of frames to test (default: 10)')
    parser.add_argument('--input', type=str, default=None,
                       help='Input media path (GIF, image, etc.)')
    parser.add_argument('--gif', type=str, default=None,
                       help='(deprecated, use --input) Input GIF path')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output plot path')
    parser.add_argument('--split-eval', action='store_true',
                       help='Insert mx.eval() between matmul and noise to separate GPU times')
    args = parser.parse_args()

    t = run_bottleneck_analysis(
        n_test=args.frames,
        input_path=args.input,
        gif_path=args.gif,
        split_eval=args.split_eval,
    )
    print_report(t)
    plot_results(t, output_path=args.output)
