"""Command-line interface for Toyomacro."""

import argparse
import sys

from toyomacro import __version__


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="toyomacro",
        description="Toyomacro - XPS Peak Fitting Software",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Public command surface: only implemented, distributed functionality.
    # `gui` (private companion layer, not installed by the public package)
    # and `fit` (placeholder pending a data schema; fitting is available
    # through the Python APIs) are intentionally not registered.

    # Convert command (HDF5 compression)
    convert_parser = subparsers.add_parser(
        "convert", help="Compress HDF5 files for faster I/O"
    )
    convert_parser.add_argument("input_folder", help="Folder with HDF5 files")
    convert_parser.add_argument(
        "-o", "--output-folder",
        help="Output folder (default: {input}_compressed)",
    )
    convert_parser.add_argument(
        "--compression",
        choices=["standard", "full"],
        default="full",
        help="Compression mode: standard (fitpara only) or full (all datasets)",
    )
    convert_parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Specdata streaming chunk size (default: 500000)",
    )
    convert_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without executing",
    )

    # Import command (raw data -> HDF5 + compression)
    import_parser = subparsers.add_parser(
        "import", help="Import raw XPS data files to HDF5"
    )
    import_parser.add_argument("input", help="Input file or folder")
    import_parser.add_argument(
        "-e", "--element", required=True,
        help="Element name (e.g., Si2p, O1s)",
    )
    import_parser.add_argument(
        "-o", "--output",
        help="Output directory (default: same as input)",
    )
    import_parser.add_argument(
        "--format",
        choices=["auto", "pxt", "ses", "vamas", "npl"],
        default="auto",
        help="File format (default: auto-detect)",
    )
    import_parser.add_argument(
        "--max-components", type=int, default=6,
        help="Max fitting components (default: 6)",
    )
    import_parser.add_argument(
        "--energy-scale",
        choices=["auto", "BE", "KE"],
        default="auto",
        help="Energy scale override (default: auto from file)",
    )
    import_parser.add_argument(
        "--excitation-energy", type=float, default=None,
        help="Photon energy override (eV)",
    )
    import_parser.add_argument(
        "--region", type=int, default=0,
        help="Region index for multi-region files (default: 0)",
    )
    import_parser.add_argument(
        "--no-compress", action="store_true",
        help="Skip uint16+LZ4 compression",
    )
    import_parser.add_argument(
        "--sweep-mode",
        choices=["integrate", "individual"],
        default="integrate",
        help="Sweep handling: integrate (sum) or individual (default: integrate)",
    )
    import_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without executing",
    )

    args = parser.parse_args()

    if args.command == "convert":
        from toyomacro.io.compression import convert_folder
        convert_folder(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            compression=args.compression,
            chunk_size=args.chunk_size,
            dry_run=args.dry_run,
        )
    elif args.command == "import":
        from pathlib import Path

        from toyomacro.io.importer import ImportConfig, import_file, import_folder

        config = ImportConfig(
            element=args.element,
            format=args.format,
            max_components=args.max_components,
            energy_scale=args.energy_scale,
            excitation_energy=args.excitation_energy,
            region_index=args.region,
            compress=not args.no_compress,
            sweep_mode=args.sweep_mode,
        )

        input_path = Path(args.input)
        output_dir = Path(args.output) if args.output else input_path.parent
        if input_path.is_file():
            output_dir = Path(args.output) if args.output else input_path.parent

        if input_path.is_dir():
            results = import_folder(input_path, output_dir, config, dry_run=args.dry_run)
            if results:
                print(f"\nImported {len(results)} files successfully.")
        elif input_path.is_file():
            if args.dry_run:
                from toyomacro.io.readers.base_reader import create_reader
                reader = create_reader(input_path, config.format)
                print(f"File: {input_path.name}")
                print(f"Format: {reader.__class__.__name__}")
                print(f"Regions: {reader.n_regions} ({', '.join(reader.region_names)})")
                print(f"Element: {config.element}")
                print(f"Compress: {config.compress}")
                print(f"Output: {output_dir}")
            else:
                result = import_file(input_path, output_dir, config)
                print(f"Imported: {result.output_path}")
                print(f"  Spectra: {result.n_spectra}, Energy points: {result.n_energy}")
                print(f"  Energy range: {result.energy_range[0]:.2f} - {result.energy_range[1]:.2f} eV")
                print(f"  Compressed: {result.compressed}")
                print(f"  File size: {result.file_size_mb:.2f} MB")
                print(f"  Time: {result.elapsed_seconds:.2f}s")
        else:
            print(f"Error: {input_path} not found")
            sys.exit(1)
    else:
        # No subcommand: show help and exit successfully.
        parser.print_help()


if __name__ == "__main__":
    main()
