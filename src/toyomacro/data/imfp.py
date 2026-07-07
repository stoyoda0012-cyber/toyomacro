"""
Inelastic Mean Free Path (IMFP) calculation using TPP-2M formula.

The TPP-2M formula (Tanuma-Powell-Penn, 1994) provides accurate
IMFP values for XPS quantitative analysis.

Usage:
    from toyomacro.data import IMFP

    # Using compound name
    lambda_nm = IMFP.tpp2m(kinetic_energy=1386.6, compound='SiO2')

    # Using explicit parameters
    lambda_nm = IMFP.tpp2m(
        kinetic_energy=1386.6,
        Nv=16, density=2.2, Mw=60.08, Eg=8.9
    )
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class IMFP:
    """
    Inelastic Mean Free Path calculations.

    Implements TPP-2M formula for IMFP estimation.
    All methods are class methods - no instantiation needed.
    """

    @classmethod
    def tpp2m(
        cls,
        kinetic_energy: float,
        compound: str | None = None,
        Nv: float | None = None,
        density: float | None = None,
        Mw: float | None = None,
        Eg: float | None = None,
    ) -> float:
        """
        Calculate IMFP using TPP-2M formula.

        Either provide `compound` name (looks up from CompoundDB) or
        provide all four parameters directly.

        Args:
            kinetic_energy: Electron kinetic energy in eV
            compound: Compound name (e.g., 'SiO2') - looks up properties
            Nv: Number of valence electrons per molecule
            density: Density in g/cm³
            Mw: Molecular weight in g/mol
            Eg: Band gap in eV

        Returns:
            IMFP in nanometers (nm)

        Raises:
            ValueError: If compound not found or required parameters missing

        Examples:
            >>> IMFP.tpp2m(1386.6, compound='SiO2')
            2.34  # example value in nm

            >>> IMFP.tpp2m(1386.6, Nv=16, density=2.2, Mw=60.08, Eg=8.9)
            2.34
        """
        # Get parameters from compound database if needed
        if compound is not None:
            from toyomacro.data.compound_db import CompoundDB

            props = CompoundDB.get_properties(compound)
            if props is None:
                raise ValueError(f"Compound '{compound}' not found in database")

            Nv = props.get("Nv", Nv)
            density = props.get("density", density)
            Mw = props.get("Mw", Mw)
            Eg = props.get("Eg", Eg)

        # Validate parameters
        if Nv is None or density is None or Mw is None:
            raise ValueError(
                "Required parameters missing. Provide compound name or "
                "Nv, density, and Mw explicitly."
            )

        # Default band gap to 0 for metals
        if Eg is None:
            Eg = 0.0

        return cls._calculate_tpp2m(kinetic_energy, Nv, density, Mw, Eg)

    @staticmethod
    def _calculate_tpp2m(
        E: float,
        Nv: float,
        rho: float,
        M: float,
        Eg: float,
    ) -> float:
        """
        Core TPP-2M calculation.

        TPP-2M formula from:
        S. Tanuma, C.J. Powell, D.R. Penn,
        Surf. Interface Anal. 21 (1994) 165-176

        Args:
            E: Kinetic energy in eV
            Nv: Valence electrons per molecule
            rho: Density in g/cm³
            M: Molecular weight in g/mol
            Eg: Band gap in eV

        Returns:
            IMFP in nanometers
        """
        # Avogadro's number
        Na = 6.022e23

        # Calculate number density of atoms (atoms/cm³)
        # For compounds, this is molecules/cm³
        n = rho * Na / M

        # Plasmon energy (eV)
        # Ep = 28.8 * sqrt(Nv * rho / M)
        Ep = 28.8 * math.sqrt(Nv * rho / M)

        # TPP-2M parameters
        beta = -0.10 + 0.944 / math.sqrt(Ep**2 + Eg**2) + 0.069 * rho**0.1
        gamma = 0.191 * rho**(-0.50)
        C = 1.97 - 0.91 * (Ep / 1000)  # U parameter in paper
        D = 53.4 - 20.8 * (Ep / 1000)  # D parameter

        # IMFP in Angstroms
        # lambda = E / (Ep^2 * (beta * ln(gamma * E) - (C/E) + (D/E^2)))
        ln_term = beta * math.log(gamma * E)
        imfp_angstrom = E / (Ep**2 * (ln_term - C / E + D / E**2))

        # Convert to nanometers
        return imfp_angstrom / 10.0

    @classmethod
    def attenuation_length(
        cls,
        kinetic_energy: float,
        compound: str | None = None,
        Nv: float | None = None,
        density: float | None = None,
        Mw: float | None = None,
        Eg: float | None = None,
        correction_factor: float = 0.9,
    ) -> float:
        """
        Calculate Effective Attenuation Length (EAL).

        EAL ≈ IMFP × correction_factor

        The correction factor accounts for elastic scattering effects.
        Typical values are 0.8-1.0.

        Args:
            kinetic_energy: Electron kinetic energy in eV
            compound: Compound name
            Nv: Number of valence electrons
            density: Density in g/cm³
            Mw: Molecular weight in g/mol
            Eg: Band gap in eV
            correction_factor: EAL/IMFP ratio (default 0.9)

        Returns:
            EAL in nanometers
        """
        imfp = cls.tpp2m(
            kinetic_energy=kinetic_energy,
            compound=compound,
            Nv=Nv,
            density=density,
            Mw=Mw,
            Eg=Eg,
        )
        return imfp * correction_factor

    @classmethod
    def sampling_depth(
        cls,
        kinetic_energy: float,
        compound: str | None = None,
        Nv: float | None = None,
        density: float | None = None,
        Mw: float | None = None,
        Eg: float | None = None,
        fraction: float = 0.95,
    ) -> float:
        """
        Calculate sampling depth for a given signal fraction.

        For normal emission, the fraction F of signal from depth d is:
        F = 1 - exp(-d / lambda)

        So: d = -lambda * ln(1 - F)

        Args:
            kinetic_energy: Electron kinetic energy in eV
            compound: Compound name
            Nv: Number of valence electrons
            density: Density in g/cm³
            Mw: Molecular weight in g/mol
            Eg: Band gap in eV
            fraction: Signal fraction (default 0.95 = 95%)

        Returns:
            Sampling depth in nanometers (for normal emission)
        """
        imfp = cls.tpp2m(
            kinetic_energy=kinetic_energy,
            compound=compound,
            Nv=Nv,
            density=density,
            Mw=Mw,
            Eg=Eg,
        )
        return -imfp * math.log(1 - fraction)
