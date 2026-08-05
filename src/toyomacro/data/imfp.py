"""Inelastic mean free path (IMFP) from the TPP-2M predictive formula.

**Model.** TPP-2M, as published in S. Tanuma, C. J. Powell & D. R. Penn,
*Surf. Interface Anal.* **21**, 165 (1994), DOI
`10.1002/sia.740210302 <https://doi.org/10.1002/sia.740210302>`_ —
Eqns (3), (4b), (4c), (4d), (4e) and (8) of that paper. Nothing is
fitted, tabulated or interpolated here: the equations are evaluated from
material parameters the caller supplies.

**Quantity.** The *inelastic* mean free path only. Elastic scattering is
not included, so this is not an effective attenuation length (EAL), a
mean escape depth, or an information depth. Those are separately defined
quantities; see :mod:`toyomacro.data.elastic_scattering` for the EAL and
:meth:`IMFP.sampling_depth` for the straight-line information depth.

**Units.** Energies in eV, density in g/cm³, molecular weight in g/mol.
Eqn (3) returns Ångströms; this module's public functions return
**nanometres**.

**Energy.** ``kinetic_energy`` is the electron's kinetic energy in the
solid, in eV. Nothing is subtracted internally — no work function, no
analyzer offset, no photon energy. Callers working from binding energies
convert first (conventionally ``E = hv - E_B``). The paper's own
convention for the zero: "All energies are expressed with respect to the
Fermi level which, for insulators, is assumed to be located midway
between the valence band maximum and the conduction band minimum." The
insulator half of that matters — for a wide-gap oxide it is a ~E_g/2
ambiguity in where E_B is referenced, a few eV, which is negligible at
Al Ka but not for a low-kinetic-energy line.

**Validity.** The paper fits Eqn (3) over **50–2000 eV** and states that
Eqns (3) and (4) "should not be used for energies greater than 2000 eV";
it also notes that the largest deviations from optically-derived IMFPs
occur below 200 eV. Reported average RMS deviations against IMFPs from
optical data are 10.2% (elements), 8.5% (organic compounds) and 18.9%
(inorganic compounds) — Table 9. Note that the inorganic group was
*excluded* from the fit ("the optical data on which their IMFPs are
based are much less reliable"), which is worth knowing given that SiO2
is the matrix in this package's own examples.

This module does **not** refuse energies outside that window, because
callers routinely want HAXPES estimates, and the authors' later work is
reassuring about it rather than the reverse: Shinotsuka, Tanuma, Powell
& Penn, *Surf. Interface Anal.* **47**, 871 (2015), DOI
10.1002/sia.5789, report that "the TPP-2M equation was useful for
energies between 50 eV and 30 keV" — that sentence is about the
non-relativistic form, the one implemented here. It rests on IMFPs for
**41 elemental solids**, though, so it does not by itself carry to
compounds; SiO2, the matrix in this package's own examples, is not
covered by it any more than it was by the 1994 fit.

What that same paper adds is a **relativistic** form for reaching
higher energies, and *that* is not implemented here. It is the same
parameter set (their Eqn (29); beta, C and D are printed in nm units
and so are 10x this module's Angstrom-unit coefficients, while gamma —
in eV^-1, since it multiplies E inside a logarithm and carries no
length — is unchanged at 0.191 rho^-0.5. It also writes the plasmon
prefactor as 28.816 rather than the 1994 paper's 28.8.) with a factor

    alpha(T) = (1 + T/2 m_e c^2) / (1 + T/m_e c^2)^2

replacing T by alpha(T)T in the numerator and inside the logarithm,
while C/T and D/T^2 keep the plain T — their Eqn (26), which they
conclude is satisfactory up to 200 keV. Omitting alpha overestimates the
IMFP by roughly 0.3% at 1.4 keV, 1.8% at 7.4 keV, 2.3% at 9.3 keV and 7%
at 30 keV. Small next to the formula's ~10-19% RMS, but one-sided, and
it grows with energy.

**Material parameters are a separate input.** :class:`~toyomacro.data.compound_db.CompoundDB`
supplies N_v, density, M_w and E_g for convenience. Those constants are
curated data with their own provenance and their own errors; the
correctness of this formula says nothing about the correctness of any
particular entry. Pass the parameters explicitly when it matters.

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

#: Energy range over which Tanuma, Powell & Penn (1994) fit Eqn (3).
#: Outside it the formula still evaluates, but the paper does not
#: support the result — see the module docstring.
TPP2M_FITTED_RANGE_EV = (50.0, 2000.0)


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
        Calculate the inelastic mean free path from the TPP-2M formula.

        Either provide `compound` name (looks up from CompoundDB) or
        provide all four parameters directly.

        Elastic scattering is not included; see the module docstring for
        the model, its units, and the 50-2000 eV range over which the
        source paper fits it.

        Args:
            kinetic_energy: Electron kinetic energy in the solid, in eV.
                Nothing is subtracted internally — convert from binding
                energy before calling.
            compound: Compound name (e.g., 'SiO2') - looks up properties
                from `CompoundDB`, which is curated data with its own
                provenance, separate from this formula.
            Nv: Valence electrons per atom (elements) or per molecule
                (compounds)
            density: Density in g/cm³
            Mw: Atomic or molecular weight in g/mol
            Eg: Band gap in eV. The paper defines it "for
                non-conductors" and does not prescribe a value for
                conductors; 0 is this package's default and the usual
                convention. For a non-conductor, pass it.

        Returns:
            IMFP in nanometers (nm). Eqn (3) itself yields Ångströms.

        Raises:
            ValueError: If compound not found, required parameters are
                missing, or a parameter is outside the formula's domain

        Examples:
            >>> round(IMFP.tpp2m(1386.6, compound='SiO2'), 2)
            3.88

            >>> round(IMFP.tpp2m(1386.6, Nv=16, density=2.2, Mw=60.08, Eg=8.9), 2)
            3.88
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

        TPP-2M from S. Tanuma, C. J. Powell & D. R. Penn,
        *Surf. Interface Anal.* **21**, 165 (1994), DOI 10.1002/sia.740210302.
        (The PDF's running head reads 1993, but the CCC code
        ``0142-2421/94/030165-12`` and the copyright line give 1994, which
        is the publisher's bibliographic year — do not "correct" this back.)
        TPP-2M is Eqns (3), (4b), (4c), (4d), (4e) and (8) of that paper
        (p. 172); it differs from TPP-2 only in using Eqn (8) for beta.

            lambda = E / (Ep^2 [beta ln(gamma E) - C/E + D/E^2])     (3)
            gamma  = 0.191 rho^-0.5                                  (4b)
            C      = 1.97 - 0.91 U                                   (4c)
            D      = 53.4 - 20.8 U                                   (4d)
            U      = Nv rho / M = Ep^2 / 829.4                       (4e)
            beta   = -0.10 + 0.944 (Ep^2 + Eg^2)^-0.5 + 0.069 rho^0.1 (8)

        Fitted by the authors over 50-2000 eV; they state Eqns (3) and
        (4) should not be used above 2000 eV, and that deviations from
        optically-derived IMFPs are largest below 200 eV. Evaluating
        outside that window is allowed here and is an extrapolation.

        The bracket guard below is one-sided by nature: just *above* the
        root, Eqn (3) returns arbitrarily large positive IMFPs (thousands
        of nm) rather than negative ones, and nothing catches those. For
        every material in `compounds.json` the root sits below 38.4 eV,
        so this is confined to well under the 50 eV fit floor — but it is
        a reason to respect that floor rather than trust the guard.

        Args:
            E: Kinetic energy in the solid, in eV
            Nv: Valence electrons per atom (elements) or molecule
                (compounds)
            rho: Density in g/cm³
            M: Atomic or molecular weight in g/mol
            Eg: Band gap in eV; the paper defines it for non-conductors,
                and 0 is this package's convention for conductors

        Returns:
            IMFP in nanometers

        Raises:
            ValueError: If a parameter is non-physical, or if the
                modified-Bethe bracket is non-positive so that Eqn (3)
                has no meaningful solution
        """
        if not math.isfinite(E) or E <= 0.0:
            raise ValueError(f"kinetic energy must be finite and positive, got {E}")
        if not math.isfinite(Nv) or Nv <= 0.0:
            raise ValueError(f"Nv must be finite and positive, got {Nv}")
        if not math.isfinite(rho) or rho <= 0.0:
            raise ValueError(f"density must be finite and positive, got {rho}")
        if not math.isfinite(M) or M <= 0.0:
            raise ValueError(f"molecular weight must be finite and positive, got {M}")
        if not math.isfinite(Eg) or Eg < 0.0:
            raise ValueError(f"band gap must be finite and non-negative, got {Eg}")

        # Free-electron plasmon energy (eV)
        Ep = 28.8 * math.sqrt(Nv * rho / M)

        # Eqn (4e). U is in mol/cm³ and is NOT Ep/1000 — an earlier version
        # of this function substituted the latter, which inflated the IMFP by
        # up to ~900% at 50 eV (the C/E and D/E² terms dominate at low E).
        U = Nv * rho / M

        beta = -0.10 + 0.944 / math.sqrt(Ep**2 + Eg**2) + 0.069 * rho**0.1
        gamma = 0.191 * rho**(-0.50)
        C = 1.97 - 0.91 * U
        D = 53.4 - 20.8 * U

        # Eqn (3) — IMFP in Angstroms
        bracket = beta * math.log(gamma * E) - C / E + D / E**2
        if bracket <= 0.0:
            # Reachable only well below the 50 eV fit floor, and only for
            # a few high-density materials. Returning the negative length
            # Eqn (3) would give here is worse than refusing.
            raise ValueError(
                f"TPP-2M is out of domain at E={E} eV for these material "
                f"parameters: the modified-Bethe bracket is {bracket:.4g} "
                f"(must be positive). The formula is fitted over "
                f"{TPP2M_FITTED_RANGE_EV[0]:g}-{TPP2M_FITTED_RANGE_EV[1]:g} eV."
            )
        return (E / (Ep**2 * bracket)) / 10.0

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

        **Elastic scattering is not included.** ``lambda`` here is the
        TPP-2M inelastic mean free path, so this is the straight-line
        approximation at normal emission — the SLA information depth, not
        an elastic-scattering-corrected sampling or information depth.

        Do **not** correct it with the *EAL* slopes of
        ``data.elastic_scattering``: those are for the
        overlayer-thickness attenuation length L_TH, and the information
        depth follows a different one. For an elastic-scattering
        corrected information depth use
        ``data.elastic_scattering.information_depth`` (Jablonski & Powell
        2009 Eq. (29), 1 - 0.787 omega), which needs a TRMFP as well as
        this IMFP and reduces to exactly this function at normal
        emission when elastic scattering is switched off.

        **Note the different convention on the way across.** This method
        takes ``fraction`` in [0, 1); ``information_depth`` takes
        ``percentage`` in (0, 100). 95% is ``fraction=0.95`` here and
        ``percentage=95.0`` there. Both reject the other's units, so the
        confusion fails loudly rather than returning a wrong depth.

        Args:
            kinetic_energy: Electron kinetic energy in the solid, in eV.
                Nothing is subtracted internally — see `tpp2m`.
            compound: Compound name, looked up in `CompoundDB`
            Nv: Valence electrons per atom (elements) or per molecule
                (compounds)
            density: Density in g/cm³
            Mw: Atomic or molecular weight in g/mol
            Eg: Band gap in eV; 0 for conductors
            fraction: Signal fraction (default 0.95 = 95%)

        Returns:
            Sampling depth in nanometers (for normal emission)

        Raises:
            ValueError: If `fraction` is not in [0, 1), or if `tpp2m`
                rejects the energy or material parameters
        """
        if not math.isfinite(fraction) or not 0.0 <= fraction < 1.0:
            raise ValueError(
                f"fraction must be in [0, 1); F=1 is reached only at "
                f"infinite depth. Got {fraction}"
            )
        imfp = cls.tpp2m(
            kinetic_energy=kinetic_energy,
            compound=compound,
            Nv=Nv,
            density=density,
            Mw=Mw,
            Eg=Eg,
        )
        return -imfp * math.log(1 - fraction)
