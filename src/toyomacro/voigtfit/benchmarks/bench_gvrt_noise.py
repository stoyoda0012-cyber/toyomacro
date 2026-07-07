"""
GVRT Noise Robustness Visualization — CLI Runner
=================================================

Generates noise comparison figures:
  1. Main figure: NF / Moderate / Strong side-by-side
  2. Reconstructed images: Original vs each noise level
  3. Channel noise grid: RGB channels × 8 noise levels (--fine-grid)

Usage:
    ~/.local/bin/uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_noise
    ~/.local/bin/uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_noise \\
        --image /path/to/image.jpg --output /path/to/output.png
    ~/.local/bin/uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_noise \\
        --fine-grid
"""

import argparse
from pathlib import Path

from ._data_paths import roundtrip_image_dir

DEFAULT_IMAGE = str(
    roundtrip_image_dir() / 'fuji' / 'churei-tower-mount-fuji-in-japan-8k-68-7680x4320.jpg'
)


def main():
    parser = argparse.ArgumentParser(
        description='GVRT Noise Robustness Visualization',
    )
    parser.add_argument(
        '--image', type=str, default=None,
        help='Input RGB image path. Default: fuji 8K',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output figure path (PNG/PDF)',
    )
    parser.add_argument(
        '--max-height', type=int, default=540,
        help='Downsample image to this max height (default: 540)',
    )
    parser.add_argument(
        '--roi-preset', type=str, default='fuji',
        help='ROI preset name (default: fuji)',
    )
    parser.add_argument(
        '--roi-name', type=str, default='sky',
        help='Which ROI to use (default: sky)',
    )
    parser.add_argument(
        '--dpi', type=int, default=200,
        help='DPI for saved figure',
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Show figure interactively',
    )
    parser.add_argument(
        '--fine-grid', action='store_true',
        help='Generate 4×9 channel noise grid (8 noise levels)',
    )
    parser.add_argument(
        '--bg-fraction', type=float, default=None,
        help='Constant background fraction (default: BG_FRACTION in '
             'gvrt_tracking.py; set to 0 for solver-matched pure Voigt)',
    )
    parser.add_argument(
        '--width-px', type=int, default=None,
        help='Target output PNG width in pixels (overrides dpi for grid plot). '
             'Use 7680 for real 8K UHD.',
    )
    parser.add_argument(
        '--height-px', type=int, default=None,
        help='Target output PNG height in pixels (overrides dpi/figsize for grid plot). '
             'Use 4320 together with --width-px 7680 for exact 8K UHD.',
    )
    args = parser.parse_args()

    image_path = args.image or DEFAULT_IMAGE
    if not Path(image_path).exists():
        parser.error(f'Image not found: {image_path}')

    output_path = args.output
    if output_path is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / 'outputs'
        output_path = str(out_dir / 'gvrt_noise_comparison.png')

    from toyomacro.voigtfit.visualization.gvrt_noise_comparison import (
        FINE_NOISE_LEVELS,
        build_noise_comparison_data,
        plot_channel_noise_grid,
        plot_noise_comparison,
        plot_reconstructed_images,
    )
    from toyomacro.voigtfit.visualization.gvrt_tracking import BG_FRACTION

    bg_fraction = args.bg_fraction if args.bg_fraction is not None else BG_FRACTION
    print(f"BG fraction: {bg_fraction:g}"
          f"{' (pure Voigt)' if bg_fraction == 0.0 else ''}")

    print("=" * 60)
    print("GVRT Noise Robustness Visualization")
    print("=" * 60)

    if args.fine_grid:
        # Fine-grained 8-level noise sweep (skip sweeps/ROI for speed)
        print("\n--- Fine-Grained Channel Noise Grid (8 levels) ---")
        data_fine = build_noise_comparison_data(
            image_path=image_path,
            max_height=args.max_height,
            roi_preset=args.roi_preset,
            roi_name=args.roi_name,
            noise_levels=FINE_NOISE_LEVELS,
            skip_details=True,
            bg_fraction=bg_fraction,
        )

        for nr in data_fine.noise_results:
            print(f"\n{nr.label} (SNR\u2248{nr.snr_label}):")
            print(f"  PSNR: amp={nr.amp_psnr:.1f}dB  "
                  f"\u03b4E={nr.dE_psnr:.1f}dB  "
                  f"\u03b4\u03c3={nr.dsigma_psnr:.1f}dB")

        out = Path(output_path)
        if out.stem.endswith('_grid'):
            grid_path = str(out)
        else:
            grid_path = str(out.with_name('gvrt_channel_noise_grid.png'))
        fig3 = plot_channel_noise_grid(
            data_fine,
            output_path=grid_path,
            show=args.show,
            dpi=args.dpi,
            target_width_px=args.width_px,
            target_height_px=args.height_px,
        )
        print(f"\nChannel grid: {grid_path}")

        # Also generate reconstructed images row
        recon_path = str(Path(output_path).with_name('gvrt_noise_reconstructed_fine.png'))
        fig4 = plot_reconstructed_images(
            data_fine,
            output_path=recon_path,
            show=args.show,
            dpi=args.dpi,
        )
        print(f"Reconstructed (fine): {recon_path}")
    else:
        # Standard 3-level comparison
        data = build_noise_comparison_data(
            image_path=image_path,
            max_height=args.max_height,
            roi_preset=args.roi_preset,
            roi_name=args.roi_name,
            bg_fraction=bg_fraction,
        )

        for nr in data.noise_results:
            print(f"\n{nr.label} (SNR\u2248{nr.snr_label}):")
            print(f"  PSNR: amp={nr.amp_psnr:.1f}dB  "
                  f"\u03b4E={nr.dE_psnr:.1f}dB  "
                  f"\u03b4\u03c3={nr.dsigma_psnr:.1f}dB")

        fig = plot_noise_comparison(
            data,
            figsize=(20, 18),
            output_path=output_path,
            show=args.show,
            dpi=args.dpi,
        )
        print(f"\nFigure saved: {output_path}")

        recon_path = str(Path(output_path).with_name('gvrt_noise_reconstructed.png'))
        fig2 = plot_reconstructed_images(
            data,
            output_path=recon_path,
            show=args.show,
            dpi=args.dpi,
        )
        print(f"Reconstructed: {recon_path}")


if __name__ == '__main__':
    main()
