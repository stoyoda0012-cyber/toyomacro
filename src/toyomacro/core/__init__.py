"""Core data structures for Toyomacro."""

from toyomacro.core.fitting_result import BackgroundParams, FittingResult, PeakComponent
from toyomacro.core.spectrum import Spectrum

__all__ = ["Spectrum", "FittingResult", "PeakComponent", "BackgroundParams"]
