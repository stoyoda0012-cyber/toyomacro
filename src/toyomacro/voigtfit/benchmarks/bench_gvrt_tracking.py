"""
GVRT Parameter Tracking Visualization v4 — CLI Runner
======================================================

Generates a 5-panel figure:
  (A) Source image + ROI rectangles
  (B) Encoding scheme
  (C) Synthetic demo — isolated channel effects (R/G/B sweep)
  (D) Real image ROI — spectral bands
  (E) Accuracy summary

Usage:
    ~/.local/bin/uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_tracking
    ~/.local/bin/uv run python -m toyomacro.voigtfit.benchmarks.bench_gvrt_tracking \\
        --image /path/to/image.jpg --output /path/to/output.png
"""

import argparse
from pathlib import Path

from ._data_paths import roundtrip_image_dir

DEFAULT_IMAGE = str(
    roundtrip_image_dir() / 'fuji' / 'churei-tower-mount-fuji-in-japan-8k-68-7680x4320.jpg'
)


def main():
    parser = argparse.ArgumentParser(
        description='GVRT Parameter Tracking Visualization v4',
    )
    parser.add_argument(
        '--image', type=str, default=None,
        help='Input RGB image path. Default: fuji 8K (auto-downsampled)',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output figure path (PNG/PDF). Default: outputs/gvrt_tracking.png',
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
        '--n-roi-samples', type=int, default=500,
        help='Max pixels per ROI (default: 500)',
    )
    parser.add_argument(
        '--dpi', type=int, default=200,
        help='DPI for saved figure',
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Show figure interactively',
    )
    args = parser.parse_args()

    image_path = args.image or DEFAULT_IMAGE
    if not Path(image_path).exists():
        parser.error(f'Image not found: {image_path}')

    output_path = args.output
    if output_path is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / 'outputs'
        output_path = str(out_dir / 'gvrt_tracking.png')

    from toyomacro.voigtfit.visualization.gvrt_tracking import (
        build_tracking_data,
        plot_gvrt_tracking,
    )

    print("=" * 60)
    print("GVRT Parameter Tracking Visualization v4")
    print("=" * 60)

    data = build_tracking_data(
        image_path=image_path,
        max_height=args.max_height,
        roi_preset=args.roi_preset,
        n_roi_samples=args.n_roi_samples,
    )

    # Summary
    print(f"\nSweeps: {len(data.sweeps)} channels")
    for s in data.sweeps:
        print(f"  {s.channel} ({s.param_name}): "
              f"{s.sweep_values[0]:.2f} → {s.sweep_values[-1]:.2f}")

    print(f"\nROIs: {len(data.rois)} regions")
    for roi in data.rois:
        r0, c0, r1, c1 = roi.rect
        print(f"  {roi.name}: {roi.n_pixels} px "
              f"({r1 - r0}\u00d7{c1 - c0} region)")

    print(f"\nGlobal PSNR: amp={data.amp_psnr:.1f}dB  "
          f"\u03b4E={data.dE_psnr:.1f}dB  "
          f"\u03b4\u03c3={data.dsigma_psnr:.1f}dB")

    fig = plot_gvrt_tracking(
        data,
        figsize=(18, 14),
        output_path=output_path,
        show=args.show,
        dpi=args.dpi,
    )
    print(f"\nFigure saved: {output_path}")


if __name__ == '__main__':
    main()
