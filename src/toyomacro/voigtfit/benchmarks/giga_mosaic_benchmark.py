"""
Giga-Spectrum Mosaic Benchmark

Runs 8K roundtrip verification across 4 images × N noise levels,
collecting PSNR metrics + representative spectra + reconstructed crops
for a single summary figure.

Total: 33.2M px × 5 files × 4 images × 7 NL = 4.65 billion spectra (physical noise)

Usage:
    python -m voigtfit.benchmarks.giga_mosaic_benchmark
    python -m voigtfit.benchmarks.giga_mosaic_benchmark --output outputs/giga_mosaic_cache.npz
Date: 2026-02-18
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np

from toyomacro.voigtfit.benchmarks.roundtrip_benchmark import (
    NOISE_LEVELS,
    RoundtripBenchmark,
)
from toyomacro.voigtfit.spectra_generator import (
    NoiseConfig,
    add_poisson_noise,
    decompose_image_to_amplitudes,
    voigt_profile,
)

from ._data_paths import gazou_dir

# Default 8K images
_GAZOU_DIR = gazou_dir()
DEFAULT_IMAGES = {
    'fuji': _GAZOU_DIR / 'churei-tower-mount-fuji-in-japan-8k-68-7680x4320.jpg',
    'dobai': _GAZOU_DIR / '1912095.jpg',
    'planet': _GAZOU_DIR / 'wallpaperbetter.jpg',
    'aurora': _GAZOU_DIR / 'wallpapersden.com_aurora-borealis-over-winter-lake_7680x4320.jpg',
}

# Canonical noise levels (excluding aliases)
CANONICAL_NOISE = [
    'None', 'Minimal', 'Negligible', 'Subtle', 'Weak',
    'Small', 'Moderate', 'Strong', 'Intense', 'Extreme',
]

# Physical noise levels: λ ≥ 1 at peaks (Subtle through Intense)
# Minimal/Negligible produce λ >> 10^8 which breaks Poisson assumptions
# Extreme ≈ Intense (both saturated at PSNR floor)
PHYSICAL_NOISE = [
    'None', 'Subtle', 'Weak', 'Small', 'Moderate', 'Strong', 'Intense',
]

# √10-spaced noise levels for λ = 1-decade stepping
# level = 10000/√λ, so λ=10^N → level=10^(4-N/2)
# Existing NOISE_LEVELS covers integer decades of level (= even decades of λ)
# We add half-decades: level = 10^0.5, 10^1.5, 10^2.5, 10^3.5, 10^4.5
import math as _math

_SQRT10 = _math.sqrt(10)  # ≈ 3.162
NOISE_LEVELS_HALFDEC = {
    'lam7': _SQRT10 * 1e0,    # level≈3.16,   λ=10^7, SNR≈3162
    'lam5': _SQRT10 * 1e1,    # level≈31.6,   λ=10^5, SNR≈316
    'lam3': _SQRT10 * 1e2,    # level≈316,    λ=10^3, SNR≈31.6
    'lam1': _SQRT10 * 1e3,    # level≈3162,   λ=10^1, SNR≈3.16
    'lam-1': _SQRT10 * 1e4,   # level≈31623,  λ=10^-1, SNR≈0.316
}

# Merge into NOISE_LEVELS for use in spectra generation
for _k, _v in NOISE_LEVELS_HALFDEC.items():
    NOISE_LEVELS[_k] = _v

# λ = 1-decade stepping: 11 noise levels covering λ = 10^8 down to 10^-2
LAMBDA_DECADE_NOISE = [
    'None',     # NF
    'Subtle',   # λ=10^8
    'lam7',     # λ=10^7
    'Weak',     # λ=10^6
    'lam5',     # λ=10^5
    'Small',    # λ=10^4
    'lam3',     # λ=10^3
    'Moderate',  # λ=10^2
    'lam1',     # λ=10^1
    'Strong',   # λ=10^0
    'lam-1',    # λ=10^-1
    'Intense',  # λ=10^-2
]


def _generate_representative_spectra(
    bench: RoundtripBenchmark,
    noise_levels: list[str],
    pixel_index: int | None = None,
) -> dict:
    """Generate representative spectra for ALL file_keys at multiple noise levels.

    For each noise level, also runs VoigtFit to produce a fitted (reconstructed)
    spectrum, stored as 'spec_{fk}_fit_{nl_name}'.

    Returns:
        dict with keys per file_key:
          'file_keys': list of file_key names
          For each file_key fk:
            'spec_{fk}_energy': energy axis
            'spec_{fk}_noisefree': noisefree spectrum
            'spec_{fk}_label': display label
            'spec_{fk}_{noise_name}': noisy spectrum
            'spec_{fk}_fit_{noise_name}': VoigtFit reconstructed spectrum
    """
    from toyomacro.voigtfit.pipeline import HybridPipeline
    from toyomacro.voigtfit.weight_cache import WeightMatrixCache

    bench._init_generator(bench.base_output_dir, 'spectra_sample')

    config = bench._generator.config
    color_mapping = bench._color_mapping
    image = bench._generator.image

    # Get amplitudes for a single pixel
    image_amps = decompose_image_to_amplitudes(
        image, color_mapping, method='pinv'
    )

    # Default: pick a representative pixel from the center crop region
    # Choose a pixel near median brightness (not darkest/brightest extreme)
    H, W = image.shape[:2]
    if pixel_index is None:
        # Center crop region (same as main benchmark)
        cy, cx = H // 2, W // 2
        half = 256  # crop_size // 2
        y0, y1 = max(0, cy - half), min(H, cy + half)
        x0, x1 = max(0, cx - half), min(W, cx + half)
        crop_region = image[y0:y1, x0:x1]
        # Compute per-pixel brightness in crop
        brightness = crop_region.astype(np.float32).mean(axis=2).ravel()
        # Target: 75th percentile brightness (representative, not extreme)
        target = np.percentile(brightness, 75)
        crop_h_local, crop_w_local = crop_region.shape[:2]
        # Find pixel closest to target brightness (vectorized)
        best_flat = int(np.argmin(np.abs(brightness - target)))
        best_dy, best_dx = divmod(best_flat, crop_w_local)
        pixel_index = (y0 + best_dy) * W + (x0 + best_dx)
        py, px = divmod(pixel_index, W)
        pixel_rgb = image[py, px]
        print(f"  Representative pixel: ({px}, {py}), RGB={pixel_rgb}, "
              f"brightness={float(pixel_rgb.mean()):.0f} "
              f"(75th percentile of crop={target:.0f})")

    # Initialize VoigtFit pipeline (Stage 1 only) + weight cache for Phi
    weight_cache = WeightMatrixCache()
    pipeline = HybridPipeline(cache=weight_cache, use_mlx=False, enable_stage2=False)

    result = {
        'file_keys': list(bench._elem_by_file.keys()),
    }

    for file_key, elem_list in bench._elem_by_file.items():
        elem_specs = [e for _, e in elem_list]
        elem_indices = [idx for idx, _ in elem_list]

        # Energy axis for this file_key
        binding_energies = [e.binding_energy for e in elem_specs]
        e_min = min(binding_energies) - config.energy_edge
        e_max = max(binding_energies) + config.energy_edge
        energy = np.arange(e_min, e_max + config.energy_step / 2,
                           config.energy_step, dtype=np.float64).astype(np.float32)

        # Build max-normalized profiles for spectra generation (same as SpectraGenerator)
        profiles = []
        for elem in elem_specs:
            prof = voigt_profile(energy, elem.binding_energy, elem.sigma, elem.gamma)
            prof = (prof / (prof.max() + 1e-10)).astype(np.float32)
            profiles.append(prof)

        # Build noisefree spectrum for this pixel
        amps = np.array([
            max(image_amps[idx, pixel_index], 0) for idx in elem_indices
        ], dtype=np.float32)

        P = np.stack(profiles, axis=0)  # (n_comp, n_energy)
        spectrum_nf = (config.amplitude_scale * amps) @ P
        bg = float(amps.sum() * config.amplitude_scale * config.bg_base)
        spectrum_nf += bg

        # Peak config for VoigtFit
        peak_config = {
            'centers': np.array([e.binding_energy for e in elem_specs], dtype=np.float32),
            'sigmas': np.array([e.sigma for e in elem_specs], dtype=np.float32),
            'gamma': float(elem_specs[0].gamma),
        }

        # Get VoigtFit's internal Phi basis (un-normalized Voigt profiles)
        # This MUST match the basis used by the weight matrix for correct reconstruction
        Phi_fit = weight_cache.build_basis_voigt(
            energy=energy,
            centers=peak_config['centers'],
            sigmas=peak_config['sigmas'],
            gamma=peak_config['gamma'],
        )  # (n_energy, n_comp) — un-normalized Voigt profiles

        # Display label
        symbols = sorted(set(e.symbol for e in elem_specs))
        orbital = elem_specs[0].orbital
        label = f"{'/'.join(symbols)} {orbital}"

        result[f'spec_{file_key}_energy'] = energy
        result[f'spec_{file_key}_noisefree'] = spectrum_nf.copy()
        result[f'spec_{file_key}_label'] = label

        # Generate noisy spectra + fit each
        for nl_name in noise_levels:
            if nl_name == 'None':
                noisy_spec = spectrum_nf.copy()
            else:
                noise_value = NOISE_LEVELS.get(nl_name, 0)
                if noise_value == 0:
                    noisy_spec = spectrum_nf.copy()
                else:
                    spec_2d = spectrum_nf.copy()[np.newaxis, :]
                    noisy_spec = add_poisson_noise(spec_2d, noise_value)[0]

            result[f'spec_{file_key}_{nl_name}'] = noisy_spec.copy()

            # VoigtFit: fit this spectrum and reconstruct
            Y = noisy_spec[np.newaxis, :].copy()  # (1, n_energy), C-contiguous
            try:
                fit_result = pipeline.process_rowmajor(
                    Y=Y,
                    element=elem_specs[0].symbol,
                    orbital=orbital,
                    energy=energy,
                    peak_config=peak_config,
                    amplitudes_only=True,
                )
                fitted_amps = fit_result.amplitudes[:, 0]  # (n_comp,)
                # Reconstruct using VoigtFit's own Phi basis (consistent normalization)
                reconstructed = Phi_fit @ fitted_amps  # (n_energy,)
                result[f'spec_{file_key}_fit_{nl_name}'] = reconstructed
            except Exception:
                # Fallback: store NaN if fit fails
                result[f'spec_{file_key}_fit_{nl_name}'] = np.full_like(energy, np.nan)

    return result


def run_giga_benchmark(
    image_paths: dict[str, Path] | None = None,
    noise_levels: list[str] | None = None,
    output_npz: Path | None = None,
    crop_size: int = 512,
    elements: str = 'fuji',
    verbose: bool = True,
) -> dict:
    """Run 4-image × N-noise roundtrip benchmark, cache to NPZ.

    Args:
        image_paths: dict of {name: path}. Default: 4 standard 8K images.
        noise_levels: List of noise level names. Default: 10 canonical levels.
        output_npz: Path to save results. Default: outputs/giga_mosaic_cache.npz
        crop_size: Size of center crop for mosaic (pixels).
        elements: Element preset name.
        verbose: Print progress.

    Returns:
        dict with all collected data (same structure as NPZ).
    """
    if image_paths is None:
        image_paths = DEFAULT_IMAGES

    if noise_levels is None:
        noise_levels = LAMBDA_DECADE_NOISE

    if output_npz is None:
        output_npz = Path('outputs/giga_mosaic_cache.npz')

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    image_names = list(image_paths.keys())
    n_images = len(image_names)
    n_noise = len(noise_levels)

    if verbose:
        print("=" * 70)
        print("GIGA-SPECTRUM MOSAIC BENCHMARK")
        print("=" * 70)
        print(f"Images: {n_images} ({', '.join(image_names)})")
        print(f"Noise levels: {n_noise} ({', '.join(noise_levels)})")
        total_est = 0
        for name in image_names:
            p = image_paths[name]
            from PIL import Image as PILImage
            with PILImage.open(p) as img:
                w, h = img.size
            px = w * h
            total_est += px * 5 * n_noise  # 5 file_keys per pixel
            print(f"  {name}: {w}x{h} = {px/1e6:.1f}M px")
        print(f"Total spectra: {total_est/1e9:.2f} billion")
        print("=" * 70)

    # ---- Collect results ----
    all_gt_images = {}
    all_recon_crops = {}      # {image_name: {noise_name: crop}}
    all_psnr_vs_gt = {}       # {image_name: {noise_name: float}}
    all_psnr_vs_nf = {}       # {image_name: {noise_name: float}}
    all_throughput = {}        # {image_name: {noise_name: float}}
    all_time = {}             # {image_name: {noise_name: float}}
    all_gen_time = {}         # {image_name: {noise_name: float}}
    all_fit_time = {}         # {image_name: {noise_name: float}}
    representative_spectra = None

    total_spectra = 0
    total_time_all = 0.0
    t_start = time.perf_counter()

    for img_idx, img_name in enumerate(image_names):
        img_path = image_paths[img_name]

        if verbose:
            print(f"\n{'─'*70}")
            print(f"[{img_idx+1}/{n_images}] {img_name}: {img_path.name}")
            print(f"{'─'*70}")

        # Create benchmark instance (streaming mode, no disk I/O)
        tmp_dir = output_npz.parent / f"_tmp_{img_name}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        bench = RoundtripBenchmark(
            image_path=str(img_path),
            output_dir=str(tmp_dir),
            elements=elements,
            use_mlx=True,
            verbose=verbose,
        )

        # Store GT image
        all_gt_images[img_name] = bench.original_image.copy()

        H, W = bench.image_shape
        cy, cx = H // 2, W // 2
        half = crop_size // 2
        y0 = max(0, cy - half)
        y1 = min(H, cy + half)
        x0 = max(0, cx - half)
        x1 = min(W, cx + half)

        all_recon_crops[img_name] = {}
        all_psnr_vs_gt[img_name] = {}
        all_psnr_vs_nf[img_name] = {}
        all_throughput[img_name] = {}
        all_time[img_name] = {}
        all_gen_time[img_name] = {}
        all_fit_time[img_name] = {}

        # Generate representative spectra from first image only
        if img_idx == 0:
            representative_spectra = _generate_representative_spectra(
                bench, noise_levels
            )

        # Run noisefree first
        nf_result = bench.run(
            'None',
            output_dir=tmp_dir,
            save_spectra=False,
            keep_images=True,
            use_inmemory='streaming',
        )

        nf_recon = nf_result.reconstructed_image
        all_recon_crops[img_name]['None'] = nf_recon[y0:y1, x0:x1].copy()
        all_psnr_vs_gt[img_name]['None'] = float(nf_result.psnr_vs_original)
        all_psnr_vs_nf[img_name]['None'] = float('inf')
        all_throughput[img_name]['None'] = float(nf_result.throughput)
        all_time[img_name]['None'] = float(nf_result.total_time)
        all_gen_time[img_name]['None'] = float(nf_result.generation_time)
        all_fit_time[img_name]['None'] = float(nf_result.reconstruction_time)
        total_spectra += nf_result.n_spectra
        total_time_all += nf_result.total_time

        # Get noisefree fitted_amplitudes for vs NF comparison
        # Run streaming again to get the raw amplitudes
        bench._init_generator(tmp_dir, 'benchmark')
        nf_noise_config = NoiseConfig(noise_type='none')
        nf_fitted_amps, _, _ = bench._run_fused_streaming(nf_noise_config)

        # Run noisy levels
        for nl_name in noise_levels:
            if nl_name == 'None':
                continue

            result = bench.run(
                nl_name,
                output_dir=tmp_dir,
                save_spectra=False,
                keep_images=True,
                noisefree_amplitudes=nf_fitted_amps,
                noisefree_reconstructed=nf_recon,
                use_inmemory='streaming',
            )

            recon = result.reconstructed_image
            all_recon_crops[img_name][nl_name] = recon[y0:y1, x0:x1].copy()
            all_psnr_vs_gt[img_name][nl_name] = float(result.psnr_vs_original)
            psnr_nf = result.psnr_vs_noisefree
            all_psnr_vs_nf[img_name][nl_name] = float(psnr_nf) if psnr_nf is not None else float('nan')
            all_throughput[img_name][nl_name] = float(result.throughput)
            all_time[img_name][nl_name] = float(result.total_time)
            all_gen_time[img_name][nl_name] = float(result.generation_time)
            all_fit_time[img_name][nl_name] = float(result.reconstruction_time)
            total_spectra += result.n_spectra
            total_time_all += result.total_time

            del recon, result
            gc.collect()

        # Clean up temp dir
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

        del bench, nf_recon, nf_fitted_amps
        gc.collect()

    elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"\n{'='*70}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*70}")
        print(f"Total spectra: {total_spectra/1e9:.2f} billion")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Avg throughput: {total_spectra/total_time_all/1e6:.1f}M spec/s")

    # ---- Pack into saveable structure ----
    # GT images: store as uint8 crops too (full 8K is too big for NPZ)
    gt_crops = {}
    for name in image_names:
        gt = all_gt_images[name]
        H, W = gt.shape[:2]
        cy, cx = H // 2, W // 2
        half = crop_size // 2
        y0 = max(0, cy - half)
        y1 = min(H, cy + half)
        x0 = max(0, cx - half)
        x1 = min(W, cx + half)
        gt_crops[name] = gt[y0:y1, x0:x1].copy()

    # Build arrays
    psnr_gt_arr = np.zeros((n_images, n_noise), dtype=np.float64)
    psnr_nf_arr = np.zeros((n_images, n_noise), dtype=np.float64)
    throughput_arr = np.zeros((n_images, n_noise), dtype=np.float64)
    time_arr = np.zeros((n_images, n_noise), dtype=np.float64)
    gen_time_arr = np.zeros((n_images, n_noise), dtype=np.float64)
    fit_time_arr = np.zeros((n_images, n_noise), dtype=np.float64)

    for i, name in enumerate(image_names):
        for j, nl in enumerate(noise_levels):
            psnr_gt_arr[i, j] = all_psnr_vs_gt[name].get(nl, float('nan'))
            psnr_nf_arr[i, j] = all_psnr_vs_nf[name].get(nl, float('nan'))
            throughput_arr[i, j] = all_throughput[name].get(nl, 0.0)
            time_arr[i, j] = all_time[name].get(nl, 0.0)
            gen_time_arr[i, j] = all_gen_time[name].get(nl, 0.0)
            fit_time_arr[i, j] = all_fit_time[name].get(nl, 0.0)

    noise_values = np.array([NOISE_LEVELS.get(nl, 0) for nl in noise_levels],
                            dtype=np.float64)

    # Build crop stack: (n_images, n_noise, crop_h, crop_w, 3)
    sample_crop = all_recon_crops[image_names[0]][noise_levels[0]]
    crop_h, crop_w = sample_crop.shape[:2]
    recon_crop_arr = np.zeros((n_images, n_noise, crop_h, crop_w, 3), dtype=np.uint8)
    gt_crop_arr = np.zeros((n_images, crop_h, crop_w, 3), dtype=np.uint8)

    for i, name in enumerate(image_names):
        gt_crop_arr[i] = gt_crops[name]
        for j, nl in enumerate(noise_levels):
            if nl in all_recon_crops[name]:
                recon_crop_arr[i, j] = all_recon_crops[name][nl]

    # Also store downscaled GT for the top panel
    gt_thumbs = []
    thumb_h = 540  # reasonable for display
    for name in image_names:
        gt = all_gt_images[name]
        H, W = gt.shape[:2]
        scale = thumb_h / H
        thumb_w = int(W * scale)
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(gt)
        thumb = pil_img.resize((thumb_w, thumb_h), PILImage.LANCZOS)
        gt_thumbs.append(np.array(thumb))
    # Pad to same size
    max_w = max(t.shape[1] for t in gt_thumbs)
    gt_thumb_arr = np.zeros((n_images, thumb_h, max_w, 3), dtype=np.uint8)
    for i, t in enumerate(gt_thumbs):
        gt_thumb_arr[i, :, :t.shape[1]] = t

    # Representative spectra (multi-file_key)
    spec_dict = {}
    if representative_spectra is not None:
        file_keys = representative_spectra['file_keys']
        spec_dict['spec_file_keys'] = np.array(file_keys, dtype=object)
        for fk in file_keys:
            spec_dict[f'spec_{fk}_energy'] = representative_spectra[f'spec_{fk}_energy']
            spec_dict[f'spec_{fk}_noisefree'] = representative_spectra[f'spec_{fk}_noisefree']
            spec_dict[f'spec_{fk}_label'] = representative_spectra[f'spec_{fk}_label']
            for nl in noise_levels:
                key = f'spec_{fk}_{nl}'
                if key in representative_spectra:
                    spec_dict[key] = representative_spectra[key]
                # Also save fit (VoigtFit reconstruction) spectra
                fit_key = f'spec_{fk}_fit_{nl}'
                if fit_key in representative_spectra:
                    spec_dict[fit_key] = representative_spectra[fit_key]

    # Save
    save_dict = {
        'image_names': np.array(image_names, dtype=object),
        'noise_names': np.array(noise_levels, dtype=object),
        'noise_values': noise_values,
        'psnr_vs_gt': psnr_gt_arr,
        'psnr_vs_nf': psnr_nf_arr,
        'throughput': throughput_arr,
        'time': time_arr,
        'gen_time': gen_time_arr,
        'fit_time': fit_time_arr,
        'recon_crops': recon_crop_arr,
        'gt_crops': gt_crop_arr,
        'gt_thumbs': gt_thumb_arr,
        'crop_size': crop_size,
        'total_spectra': total_spectra,
        'total_time': elapsed,
        **spec_dict,
    }

    np.savez_compressed(output_npz, **save_dict)
    if verbose:
        size_mb = output_npz.stat().st_size / 1e6
        print(f"Saved: {output_npz} ({size_mb:.1f} MB)")

    return save_dict


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Giga-Spectrum Mosaic Benchmark: 4 images × N noise levels'
    )
    parser.add_argument('--images', nargs='*', type=str, default=None,
                       help='Image paths (default: 4 standard 8K images)')
    parser.add_argument('--image-names', nargs='*', type=str, default=None,
                       help='Image names (must match --images count)')
    parser.add_argument('--noise-levels', nargs='*', type=str, default=None,
                       help=f'Noise levels (default: {len(LAMBDA_DECADE_NOISE)} lambda-decade)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output NPZ path (default: outputs/giga_mosaic_cache.npz)')
    parser.add_argument('--crop-size', type=int, default=512,
                       help='Center crop size in pixels (default: 512)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: 4 noise levels only')

    args = parser.parse_args()

    # Resolve image paths
    image_paths = None
    if args.images:
        names = args.image_names or [Path(p).stem[:10] for p in args.images]
        if len(names) != len(args.images):
            print("Error: --image-names must match --images count")
            sys.exit(1)
        image_paths = {n: Path(p) for n, p in zip(names, args.images)}

    noise_levels = args.noise_levels
    if args.quick:
        noise_levels = ['None', 'Weak', 'Moderate', 'Strong', 'Intense']

    output = Path(args.output) if args.output else None

    run_giga_benchmark(
        image_paths=image_paths,
        noise_levels=noise_levels,
        output_npz=output,
        crop_size=args.crop_size,
    )
