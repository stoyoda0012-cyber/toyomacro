"""
Toyomacro - XPS Peak Fitting Software

A Python implementation of the Toyomacro XPS analysis toolkit.
"""

# Single version source: pyproject.toml [project] version, read from the
# installed package metadata.  The fallback covers running from a source
# checkout that has not been installed.
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version("toyomacro")
except PackageNotFoundError:  # source tree without installation
    __version__ = "0.0.0+unknown"

__author__ = "Satoshi Toyoda"

# Lazy imports for faster startup
def __getattr__(name):
    """Lazy import for heavy modules."""
    if name == "Spectrum":
        from toyomacro.core.spectrum import Spectrum
        return Spectrum
    elif name == "FittingResult":
        from toyomacro.core.fitting_result import FittingResult
        return FittingResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Spectrum",
    "FittingResult",
    "__version__",
]
