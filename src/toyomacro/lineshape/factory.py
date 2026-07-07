"""Factory for creating lineshape instances."""


from toyomacro.lineshape.base import BaseLineshape
from toyomacro.lineshape.doniach_sunjic import DoniachSunjic
from toyomacro.lineshape.fermi_dirac import FermiDirac
from toyomacro.lineshape.gaussian import Gaussian
from toyomacro.lineshape.lorentzian import Lorentzian
from toyomacro.lineshape.pseudovoigt import PseudoVoigt
from toyomacro.lineshape.voigt import Voigt


class LineshapeFactory:
    """
    Factory for creating lineshape instances.

    Supports registration of custom lineshapes.
    """

    _registry: dict[str, type[BaseLineshape]] = {
        "gaussian": Gaussian,
        "lorentzian": Lorentzian,
        "voigt": Voigt,
        "pseudovoigt": PseudoVoigt,
        "doniach_sunjic": DoniachSunjic,
        "fermi_dirac": FermiDirac,
        # Aliases
        "g": Gaussian,
        "l": Lorentzian,
        "v": Voigt,
        "pv": PseudoVoigt,
        "ds": DoniachSunjic,
        "fd": FermiDirac,
    }

    # Numeric type mapping (MATLAB TypeID convention)
    _type_map: dict[int, str] = {
        1: "voigt",
        2: "pseudovoigt",
        3: "gaussian",
        4: "lorentzian",
        5: "doniach_sunjic",
        6: "fermi_dirac",
    }

    @classmethod
    def create(cls, name: str | int) -> BaseLineshape:
        """
        Create a lineshape instance.

        Args:
            name: Lineshape name (str) or type number (int)

        Returns:
            Lineshape instance

        Raises:
            ValueError: If lineshape is not registered
        """
        if isinstance(name, int):
            name = cls._type_map.get(name, "")

        name_lower = name.lower()
        if name_lower not in cls._registry:
            available = ", ".join(sorted(set(cls._registry.keys()) - {"g", "l", "v", "pv"}))
            raise ValueError(
                f"Unknown lineshape: {name}. Available: {available}"
            )

        return cls._registry[name_lower]()

    @classmethod
    def register(cls, name: str, lineshape_class: type[BaseLineshape]):
        """
        Register a custom lineshape.

        Args:
            name: Name to register under
            lineshape_class: Lineshape class to register
        """
        cls._registry[name.lower()] = lineshape_class

    @classmethod
    def available(cls) -> list[str]:
        """List available lineshape names (excluding short aliases)."""
        return sorted(k for k in cls._registry.keys() if len(k) > 2)


def create_lineshape(name: str | int) -> BaseLineshape:
    """
    Convenience function to create a lineshape.

    Args:
        name: Lineshape name or type number

    Returns:
        Lineshape instance
    """
    return LineshapeFactory.create(name)
