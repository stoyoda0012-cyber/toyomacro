"""
Element and compound database for XPS analysis.

Provides fast lookup of:
- Core-level binding energies
- Photoionization cross-sections
- Compound physical properties
- IMFP calculations (TPP-2M)

Usage:
    from toyomacro.data import BindingEnergy, CrossSection, CompoundDB, IMFP

    # Binding energy lookup
    be = BindingEnergy.lookup('Si', '2p')      # → 99.0 eV
    be = BindingEnergy.from_string('Si2p')     # → 99.0 eV
    so = BindingEnergy.get_spin_orbit('Si', '2p')  # → split_eV 0.6, provisional

    # Cross-section (Al K-α = 1486.6 eV)
    sigma = CrossSection.lookup('Si', '2p', 1486.6)

    # Compound properties
    props = CompoundDB.get_properties('SiO2')
    # → {'Nv': 16, 'density': 2.2, 'Mw': 60.08, 'Eg': 8.9}

    # IMFP calculation
    lambda_nm = IMFP.tpp2m(kinetic_energy=1386.6, compound='SiO2')
"""

from toyomacro.data.angular_correction import AngularCorrection
from toyomacro.data.binding_energy import BindingEnergy, SpinOrbitReference
from toyomacro.data.compound_db import CompoundDB
from toyomacro.data.cross_section import (
    ComponentLookup,
    CrossSection,
    CrossSectionLookup,
    SpinOrbitIntensityRatio,
)
from toyomacro.data.imfp import IMFP
from toyomacro.data.paths import (
    clear_cache,
    get_cache_dir,
    get_common_data_path,
    regenerate_cache,
)

__all__ = [
    # Main classes
    "AngularCorrection",
    "BindingEnergy",
    "CrossSection",
    "CrossSectionLookup",
    "ComponentLookup",
    "SpinOrbitReference",
    "SpinOrbitIntensityRatio",
    "CompoundDB",
    "IMFP",
    # Utility functions
    "get_common_data_path",
    "get_cache_dir",
    "clear_cache",
    "regenerate_cache",
]
