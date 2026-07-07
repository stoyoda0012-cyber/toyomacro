"""Demonstration and profiling benchmarks for voigtfit.

These modules are NOT run in CI — they are maintainer tools for
measuring solver throughput, accuracy ceilings, and scaling behavior.
Most run standalone on synthetic data; the image/data-driven variants
require ``VOIGTFIT_DATA_ROOT`` (see ``_data_paths.py``). See
``README.md`` in this directory for a categorized index.
"""
