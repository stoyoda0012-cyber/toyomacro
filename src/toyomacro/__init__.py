"""
Toyomacro - XPS Peak Fitting Software

A Python implementation of the Toyomacro XPS analysis toolkit.
"""

__version__ = "0.1.0"
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
