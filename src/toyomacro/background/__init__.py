"""Background subtraction algorithms for XPS spectra."""

from toyomacro.background.base import BaseBackground
from toyomacro.background.linear import Linear
from toyomacro.background.shirley import Shirley
from toyomacro.background.tougaard import Tougaard

__all__ = ["BaseBackground", "Shirley", "Linear", "Tougaard"]
