#!/usr/bin/env python3
"""
VoigtFit CLI - Command-line interface for VoigtFit processing
=============================================================

Usage:
    # Process synthetic data (benchmark mode)
    python -m voigtfit.cli benchmark --n-spectra 1000000

    # Process from HDF5 file
    python -m voigtfit.cli process input.h5 --output result.npz

    # GVRT roundtrip demo (image -> spectra -> fit -> image)
    python -m toyomacro.voigtfit.cli gvrt --noise Moderate --inspect 192,192

    # Interactive REPL mode
    python -m voigtfit.cli repl
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx

    from ._mlx_support import mlx_usable as _mlx_usable
    HAS_MLX = _mlx_usable()  # installed AND the default device can execute work
except ImportError:
    HAS_MLX = False


def cmd_benchmark(args):
    """Run benchmark with synthetic data."""
    from .pipeline import HybridPipeline
    from .weight_cache import WeightMatrixCache

    print("=" * 60)
    print("VoigtFit Benchmark")
    print("=" * 60)
    print(f"MLX available: {HAS_MLX}")
    print(f"Spectra count: {args.n_spectra:,}")
    print(f"Energy points: {args.n_energy}")
    print(f"Components: {args.n_components}")
    print(f"Stage 2: {'enabled' if args.stage2 else 'disabled'}")
    print()

    # Setup
    cache = WeightMatrixCache()
    pipeline = HybridPipeline(
        cache=cache,
        use_mlx=HAS_MLX,
        enable_stage2=args.stage2,
        chi2_threshold=3.0,
    )

    # Generate synthetic data
    np.random.seed(42)
    energy = np.linspace(280, 292, args.n_energy).astype(np.float32)
    centers = np.linspace(282, 290, args.n_components).astype(np.float32)
    sigmas = np.full(args.n_components, 0.8, dtype=np.float32)
    gamma = 0.3

    peak_config = {
        "centers": centers,
        "sigmas": sigmas,
        "gamma": gamma,
    }

    # Generate spectra with noise
    print("Generating synthetic spectra...")
    t0 = time.perf_counter()

    # Simple synthetic: random amplitudes + noise
    from .voigt_jacobian import voigt_profile
    amplitudes_true = np.abs(np.random.randn(args.n_components, args.n_spectra).astype(np.float32)) * 100

    Y = np.zeros((args.n_energy, args.n_spectra), dtype=np.float32)
    for i, (c, s) in enumerate(zip(centers, sigmas)):
        profile = voigt_profile(energy, c, s, gamma).astype(np.float32)
        Y += np.outer(profile, amplitudes_true[i])

    # Add noise
    noise_level = Y.max() / 100
    Y += noise_level * np.random.randn(*Y.shape).astype(np.float32)

    gen_time = time.perf_counter() - t0
    print(f"Generation time: {gen_time:.3f}s")

    # Warm-up (cache initialization)
    print("\nWarm-up run...")
    _ = pipeline.process(
        Y=Y[:, :min(1000, args.n_spectra)],
        element="C",
        orbital="1s",
        energy=energy,
        peak_config=peak_config,
    )

    # Benchmark run
    print("Benchmark run...")
    t0 = time.perf_counter()

    result = pipeline.process(
        Y=Y,
        element="C",
        orbital="1s",
        energy=energy,
        peak_config=peak_config,
    )

    total_time = time.perf_counter() - t0

    # Results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total time: {total_time:.3f}s")
    print(f"Rate: {args.n_spectra/total_time:,.0f} spectra/s")
    print()
    print("Timing breakdown:")
    for k, v in result.timing.items():
        if isinstance(v, float) and k not in ['stage1_rate', 'stage2_rate', 'anomaly_ratio']:
            print(f"  {k}: {v:.4f}s")
        elif k == 'stage1_rate':
            print(f"  {k}: {v:,.0f} spec/s")
        elif k == 'n_anomaly':
            print(f"  {k}: {v}")
        elif k == 'anomaly_ratio':
            print(f"  {k}: {v*100:.2f}%")
    print()

    # Accuracy check
    amp_error = np.abs(result.amplitudes - amplitudes_true).mean()
    amp_relative = amp_error / amplitudes_true.mean() * 100
    print(f"Mean amplitude error: {amp_error:.4f} ({amp_relative:.2f}%)")
    print(f"Mean χ²: {result.chi2.mean():.6f}")
    print(f"Anomalies: {result.anomaly_mask.sum()} ({100*result.anomaly_mask.mean():.2f}%)")

    # 8K estimate
    n_8k = 8192 * 4096  # 33.6M
    est_8k_time = n_8k / (args.n_spectra / total_time)
    print(f"\nEstimated 8K (33.6M spectra) time: {est_8k_time:.2f}s")


def cmd_process(args):
    """Process data from file."""
    from .integration import DepthProfilerBridge

    print(f"Processing: {args.input}")

    bridge = DepthProfilerBridge(
        use_mlx=HAS_MLX,
        enable_stage2=not args.no_stage2,
    )

    # Load data
    data = bridge.load_h5_data(args.input)
    print(f"Loaded data shape: {data['Ismp'].shape}")

    # For now, process as flat spectra
    # TODO: Add proper multi-element support
    print("Multi-element processing not yet implemented.")
    print("Use 'benchmark' command for performance testing.")


def cmd_repl(args):
    """Interactive REPL mode."""
    from .pipeline import HybridPipeline
    from .weight_cache import WeightMatrixCache

    print("=" * 60)
    print("VoigtFit Interactive Mode")
    print("=" * 60)
    print("Commands: benchmark, quit")
    print()

    cache = WeightMatrixCache()
    pipeline = HybridPipeline(cache=cache, use_mlx=HAS_MLX)

    while True:
        try:
            cmd = input("voigtfit> ").strip().lower()

            if cmd in ('quit', 'exit', 'q'):
                break
            elif cmd == 'benchmark':
                # Quick benchmark
                n_spectra = 100000
                energy = np.linspace(280, 292, 100).astype(np.float32)
                Y = np.random.randn(100, n_spectra).astype(np.float32)

                t0 = time.perf_counter()
                result = pipeline.process(
                    Y=Y,
                    element="C",
                    orbital="1s",
                    energy=energy,
                    peak_config={
                        "centers": np.array([284.5, 286.0, 288.5]),
                        "sigmas": np.array([0.8, 0.9, 0.7]),
                        "gamma": 0.3,
                    },
                )
                t = time.perf_counter() - t0
                print(f"Processed {n_spectra:,} spectra in {t:.3f}s ({n_spectra/t:,.0f}/s)")
            elif cmd == 'help':
                print("Commands: benchmark, quit, help")
            elif cmd:
                print(f"Unknown command: {cmd}")

        except (KeyboardInterrupt, EOFError):
            print()
            break


def analyze_h5_compression(h5_path: str):
    """Analyze HDF5 file and show compression potential."""
    import os

    import h5py

    file_size = os.path.getsize(h5_path)
    print(f"File: {h5_path}")
    print(f"Size: {file_size / 1e6:.1f} MB")
    print()
    print(f"{'Dataset':<20} {'Shape':<25} {'Dtype':<10} {'Size (MB)':<12} {'Potential'}")
    print("-" * 85)

    with h5py.File(h5_path, 'r') as f:
        for name in ['specdata', 'fitpara', 'otherpara', 'xytdata']:
            if name in f and isinstance(f[name], h5py.Dataset):
                ds = f[name]
                raw_mb = ds.nbytes / 1e6
                shape_str = str(ds.shape)
                dtype_str = str(ds.dtype)

                if name == 'fitpara':
                    potential = f"~{raw_mb/6:.1f} MB (6x FitparaCodec)"
                elif name in ('otherpara', 'xytdata'):
                    potential = f"~{raw_mb/200:.2f} MB (200x+ LZ4)"
                elif name == 'specdata':
                    potential = f"~{raw_mb/2:.1f} MB (uint16)"
                else:
                    potential = "N/A"

                print(f"{name:<20} {shape_str:<25} {dtype_str:<10} {raw_mb:<12.1f} {potential}")

        # Show already-compressed datasets
        for name in ['fitpara_compressed', 'otherpara_compressed',
                     'xytdata_compressed', 'specdata_uint16']:
            if name in f:
                if isinstance(f[name], h5py.Group):
                    size_mb = f[name]['data'].nbytes / 1e6
                    print(f"{name:<20} {'(compressed)':<25} {'uint8':<10} {size_mb:<12.2f} already compressed")
                elif isinstance(f[name], h5py.Dataset):
                    size_mb = f[name].nbytes / 1e6
                    print(f"{name:<20} {str(f[name].shape):<25} {str(f[name].dtype):<10} {size_mb:<12.1f} already compressed")


def cmd_convert(args):
    """Convert HDF5 file with compression codecs."""
    import os
    import shutil

    from .h5io import compress_h5_file

    input_path = args.input

    if not Path(input_path).exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if args.analyze:
        analyze_h5_compression(input_path)
        return

    output_path = args.output
    if output_path is None:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}_compressed.h5")

    compression = args.compression or 'standard'

    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Compression: {compression}")
    print()

    # Copy input to output, then compress + repack in-place
    shutil.copy2(input_path, output_path)
    stats = compress_h5_file(output_path, compression=compression, verbose=True, repack=True)

    # Print summary
    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    print()
    print(f"Input:  {input_size / 1e6:.1f} MB")
    print(f"Output: {output_size / 1e6:.1f} MB")
    if output_size > 0:
        print(f"Ratio:  {input_size / output_size:.1f}x")
    print(f"Compressed datasets: {', '.join(stats.get('compressed_datasets', []))}")


def cmd_gvrt(args):
    """Image -> Voigt spectra -> fit -> reconstructed image roundtrip."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .gvrt_service import (
        GVRTConfig,
        GVRTService,
        make_demo_image,
        prepare_image,
    )

    if args.image:
        if not Path(args.image).exists():
            print(f"Error: {args.image} not found")
            sys.exit(1)
        image = prepare_image(args.image, max_pixels=args.max_pixels)
    else:
        image = make_demo_image(args.size)

    config = GVRTConfig(
        solver=args.solver,
        noise_level=args.noise,
        exact_voigt=args.exact,
        max_pixels=args.max_pixels,
    )
    n_px = image.shape[0] * image.shape[1]
    print(f"GVRT roundtrip: {image.shape[1]}x{image.shape[0]} = {n_px:,} spectra  "
          f"solver={args.solver}  noise={args.noise}")

    service = GVRTService()
    result = service.run(image, config)
    print(result.summary)

    # 3-panel figure: original / reconstructed / |difference|
    diff = np.abs(
        result.original.astype(np.int16) - result.reconstructed.astype(np.int16)
    ).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    panels = [
        (result.original, "original (R=amp, G=dE, B=FWHM)"),
        (result.reconstructed, "reconstructed from fits"),
        (diff, "|difference|"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    p = result.psnr
    fig.suptitle(
        f"PSNR  R(amp)={p.amplitude:.1f}  G(dE)={p.shift:.1f}  "
        f"B(FWHM)={p.fwhm:.1f} dB   |   "
        f"{result.throughput / 1e6:.2f} M spectra/s",
        fontsize=10,
    )
    fig.tight_layout()
    out = Path(args.output)
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")

    if args.inspect:
        x, y = (int(v) for v in args.inspect.split(","))
        insp = service.inspect_pixel(x, y)
        fig2, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(insp.energy, insp.noisy, ".", ms=3, color="0.4",
                label="measured (noisy)")
        ax.plot(insp.energy, insp.clean, "-", color="tab:green", lw=1,
                label="ground truth")
        ax.plot(insp.energy, insp.fit, "-", color="tab:red", lw=1.2,
                label="fit")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Intensity (counts)")
        ax.set_title(
            f"pixel ({x}, {y}): amp GT {insp.gt_amp:.0f} / "
            f"fit {insp.fit_amp:.0f}   dE GT {insp.gt_dE:+.3f} / "
            f"fit {insp.fit_dE:+.3f} eV",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        fig2.tight_layout()
        out2 = out.with_name(f"{out.stem}_px{x}_{y}{out.suffix}")
        fig2.savefig(out2, dpi=150)
        print(f"saved: {out2}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="VoigtFit CLI - Fast Voigt profile fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmark')
    bench_parser.add_argument('--n-spectra', type=int, default=100000,
                             help='Number of spectra (default: 100000)')
    bench_parser.add_argument('--n-energy', type=int, default=100,
                             help='Number of energy points (default: 100)')
    bench_parser.add_argument('--n-components', type=int, default=3,
                             help='Number of components (default: 3)')
    bench_parser.add_argument('--stage2', action='store_true',
                             help='Enable Stage 2 refinement')

    # Process command
    proc_parser = subparsers.add_parser('process', help='Process data file')
    proc_parser.add_argument('input', help='Input HDF5 file')
    proc_parser.add_argument('--output', '-o', help='Output file (default: input_result.npz)')
    proc_parser.add_argument('--no-stage2', action='store_true',
                            help='Disable Stage 2 refinement')

    # Convert command
    conv_parser = subparsers.add_parser('convert', help='Convert HDF5 with compression')
    conv_parser.add_argument('input', help='Input HDF5 file')
    conv_parser.add_argument('output', nargs='?', help='Output file (default: input_compressed.h5)')
    conv_parser.add_argument('--compression', '-c', choices=['standard', 'full'],
                            default='standard',
                            help="'standard': fitpara only. 'full': all datasets (default: standard)")
    conv_parser.add_argument('--analyze', action='store_true',
                            help='Analyze file without converting')

    # GVRT command (image -> spectra -> fit -> image roundtrip)
    gvrt_parser = subparsers.add_parser(
        'gvrt',
        help='GVRT roundtrip demo: encode an image as Voigt spectra, '
             'add noise, fit, and reconstruct the image',
    )
    gvrt_parser.add_argument('image', nargs='?',
                             help='Input image file (default: synthetic demo image)')
    gvrt_parser.add_argument('--size', type=int, default=384,
                             help='Demo image size in pixels (default: 384)')
    gvrt_parser.add_argument('--solver', default='taylor',
                             choices=['taylor', 'taylor6', 'dict2d_parabola'],
                             help='Fitting solver (default: taylor)')
    gvrt_parser.add_argument('--noise', default='Moderate',
                             help='Noise level: None/Subtle/Weak/Small/Moderate/'
                                  'Strong/Intense or lamX (default: Moderate)')
    gvrt_parser.add_argument('--exact', action='store_true',
                             help='Exact Voigt generation (slower, no Taylor '
                                  'linearization artifacts)')
    gvrt_parser.add_argument('--max-pixels', type=int, default=1_048_576,
                             help='Downscale larger images (default: 1048576)')
    gvrt_parser.add_argument('--inspect', metavar='X,Y',
                             help='Also save a spectrum-space plot of one pixel')
    gvrt_parser.add_argument('--output', '-o', default='gvrt_roundtrip.png',
                             help='Output figure path (default: gvrt_roundtrip.png)')

    # REPL command
    subparsers.add_parser('repl', help='Interactive mode')

    args = parser.parse_args(argv)

    if args.command == 'benchmark':
        cmd_benchmark(args)
    elif args.command == 'process':
        cmd_process(args)
    elif args.command == 'convert':
        cmd_convert(args)
    elif args.command == 'gvrt':
        cmd_gvrt(args)
    elif args.command == 'repl':
        cmd_repl(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
