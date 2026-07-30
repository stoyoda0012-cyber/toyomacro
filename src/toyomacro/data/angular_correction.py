"""Angular correction for photoionization cross-sections.

Non-dipole angular distribution parameters (β, γ, δ) from
Trzhaskovskaya & Yarzhemsky (2018) for HAXPES applications (1.5–10 keV).

Two geometry modes are supported:

1. **Linearly polarized** light (synchrotron):

    L_pol(ψ, φ) = 1 + β P₂(cos θ_ε) + (δ + γ cos²θ_ε) sin θ_ε cos φ

    where θ_ε = angle from polarization vector ε to electron emission,
          φ = azimuth around ε measured from the (ε, k) plane.

    For coplanar HAXPES:  ψ = α_xray − θ_emission,  φ = 0.

2. **Unpolarized** light (lab X-ray, e.g. Ga Kα):

    L_unpol(α) = 1 − (β/2) P₂(cos α) + (δ + γ/2 · sin²α) cos α

    where α = angle from X-ray beam direction k to electron emission.
    Derived by averaging L_pol over all ε ⊥ k.
    Axially symmetric around k (no φ dependence).

P₂(x) = (3x² − 1) / 2  (2nd Legendre polynomial).

For HAXPES geometry with grazing incidence (coplanar):
    ψ(θ) = α_xray − θ   where α_xray = X-ray angle from surface normal,
                                θ = emission angle from surface normal.

References
----------
- Trzhaskovskaya, M. B. & Yarzhemsky, V. G. (2018). At. Data Nucl. Data Tables.
- Willis et al. (2024). Digitisation of Trzhaskovskaya Dirac-Fock Photoionisation
  Parameters for HAXPES Applications (1.5-10 keV).
- Scofield, J. H. (1976). J. Electron Spectrosc. Relat. Phenom.

Usage
-----
    from toyomacro.data.angular_correction import AngularCorrection

    # Lookup parameters
    params = AngularCorrection.lookup('Si', '2p3/2', 9000)
    # → {'sigma': 0.01097, 'beta': 0.222, 'gamma': 0.925, 'delta': 0.289,
    #    'binding_energy': 98.9}

    # Angular correction factor
    L = AngularCorrection.L_full(beta=0.222, gamma=0.925, delta=0.289,
                                  psi_deg=60.0)

    # Full angular distribution for emission angles 0-60°
    result = AngularCorrection.angular_distribution(
        'Si', '2p3/2', 9000,
        emission_angles_deg=np.arange(0, 61),
        xray_from_normal_deg=88.0,
    )
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from toyomacro.data.paths import load_trzh2018_data, load_trzh2019_data


class AngularCorrection:
    """Non-dipole angular correction using Trzhaskovskaya parameters.

    Merges two datasets:
    - 2018: outer shells, Z=1-100, 1.5-10 keV (fixed grid)
    - 2019: inner shells (Eb ≥ ~1.5 keV), Z=13-100, 2-18 keV (variable grid)

    All methods are class methods — no instantiation needed.
    Parameters are lazily loaded from Excel (cached as JSON after first parse).
    """

    _data_2018: dict[str, Any] | None = None
    _data_2019: dict[str, Any] | None = None

    @classmethod
    def _get_data_2018(cls) -> dict[str, Any]:
        """Get cached Trzhaskovskaya 2018 data."""
        if cls._data_2018 is None:
            cls._data_2018 = load_trzh2018_data()
        return cls._data_2018

    @classmethod
    def _get_data_2019(cls) -> dict[str, Any]:
        """Get cached Trzhaskovskaya 2019 inner-shell data."""
        if cls._data_2019 is None:
            cls._data_2019 = load_trzh2019_data()
        return cls._data_2019

    @classmethod
    def available_elements(cls) -> list[str]:
        """List elements in the database (union of 2018 + 2019)."""
        elems = set(cls._get_data_2018().get("data", {}).keys())
        elems |= set(cls._get_data_2019().get("data", {}).keys())
        return sorted(elems)

    @classmethod
    def available_orbitals(cls, element: str) -> list[str]:
        """List orbitals for an element (union of 2018 + 2019)."""
        orbs = set()
        elem_2018 = cls._get_data_2018().get("data", {}).get(element, {})
        elem_2019 = cls._get_data_2019().get("data", {}).get(element, {})
        orbs.update(elem_2018.keys())
        orbs.update(elem_2019.keys())
        return sorted(orbs)

    @classmethod
    def photon_energies(cls) -> list[float]:
        """Tabulated photon energies from 2018 dataset (eV)."""
        return cls._get_data_2018().get("photon_energies", [])

    @classmethod
    def _find_orbital(
        cls, elem_data: dict[str, Any], orbital: str
    ) -> dict[str, Any] | None:
        """Find orbital in element data, with j-suffix fallback."""
        if orbital in elem_data:
            return elem_data[orbital]
        for suffix in ["3/2", "5/2", "7/2", "1/2"]:
            key = f"{orbital}{suffix}"
            if key in elem_data:
                return elem_data[key]
        return None

    @classmethod
    def lookup_raw(
        cls,
        element: str,
        orbital: str,
    ) -> dict[str, Any] | None:
        """Get raw data for element/orbital (no interpolation).

        Tries 2018 first, falls back to 2019 for inner shells.
        Returns dict with: binding_energy, sigma, beta, gamma, delta.
        For 2018 data, values are at common photon_energies().
        For 2019 data, includes per-orbital 'photon_energies' key.
        """
        # Try 2018 (outer shells) first
        data_2018 = cls._get_data_2018()
        elem_2018 = data_2018.get("data", {}).get(element, {})
        result = cls._find_orbital(elem_2018, orbital)
        if result is not None:
            return result

        # Fall back to 2019 (inner shells)
        data_2019 = cls._get_data_2019()
        elem_2019 = data_2019.get("data", {}).get(element, {})
        result = cls._find_orbital(elem_2019, orbital)
        if result is not None:
            return result

        return None

    @classmethod
    def lookup(
        cls,
        element: str,
        orbital: str,
        photon_energy: float,
    ) -> dict[str, float] | None:
        """Look up interpolated σ, β, γ, δ at a given photon energy.

        Uses log interpolation for σ and linear interpolation for β, γ, δ.
        Merges 2018 (outer shells) and 2019 (inner shells) datasets.

        Parameters
        ----------
        element : str
            Element symbol (e.g., 'Si', 'Au').
        orbital : str
            Orbital designation (e.g., '2p3/2', '1s1/2').
        photon_energy : float
            Photon energy in eV.

        Returns
        -------
        dict or None
            {'sigma': float, 'beta': float, 'gamma': float, 'delta': float,
             'binding_energy': float} or None if not found.
        """
        raw = cls.lookup_raw(element, orbital)
        if raw is None:
            return None

        # Determine photon energy grid:
        # 2019 data has per-orbital 'photon_energies', 2018 uses common grid
        pe = raw.get("photon_energies") or cls.photon_energies()
        if not pe:
            return None

        result = {"binding_energy": raw.get("binding_energy")}

        for key in ("sigma", "beta", "gamma", "delta"):
            values = raw.get(key, [])
            if not values:
                return None

            valid = [
                (e, v) for e, v in zip(pe, values)
                if v is not None and e is not None
            ]
            if not valid:
                return None

            if key == "sigma":
                result[key] = _interp_log_log(valid, photon_energy)
            else:
                result[key] = _interp_linear(valid, photon_energy)

        return result

    # ------------------------------------------------------------------
    # Angular correction formulae
    # ------------------------------------------------------------------

    @staticmethod
    def P2(cos_psi: float | np.ndarray) -> float | np.ndarray:
        """Second Legendre polynomial P₂(x) = (3x² - 1) / 2."""
        return (3.0 * cos_psi**2 - 1.0) / 2.0

    @staticmethod
    def L_dipole(
        beta: float,
        psi_deg: float | np.ndarray,
    ) -> float | np.ndarray:
        """Dipole-only angular correction factor.

        L = 1 - (β/2) P₂(cos ψ)

        Parameters
        ----------
        beta : float
            Angular asymmetry parameter.
        psi_deg : float or array
            Angle between photon direction and emission direction (degrees).
        """
        psi = np.radians(psi_deg)
        cos_psi = np.cos(psi)
        P2 = (3.0 * cos_psi**2 - 1.0) / 2.0
        return 1.0 - (beta / 2.0) * P2

    @staticmethod
    def L_full(
        beta: float,
        gamma: float,
        delta: float,
        psi_deg: float | np.ndarray,
        phi_deg: float = 0.0,
    ) -> float | np.ndarray:
        """Non-dipole angular correction for linearly polarized X-rays.

        L = 1 - (β/2) P₂(cos ψ) + (δ + γ cos²ψ) sin ψ cos φ

        For synchrotron sources. For unpolarized lab sources, use
        L_unpolarized() instead.

        Parameters
        ----------
        beta, gamma, delta : float
            Angular distribution parameters.
        psi_deg : float or array
            Angle from photon direction (degrees).
        phi_deg : float
            Azimuthal angle (degrees). 0 = scattering plane.
        """
        psi = np.radians(psi_deg)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        cos_phi = math.cos(math.radians(phi_deg))

        P2 = (3.0 * cos_psi**2 - 1.0) / 2.0
        dipole = 1.0 - (beta / 2.0) * P2
        nondipole = (delta + gamma * cos_psi**2) * sin_psi * cos_phi

        return dipole + nondipole

    @staticmethod
    def L_unpolarized(
        beta: float,
        gamma: float,
        delta: float,
        alpha_deg: float | np.ndarray,
    ) -> float | np.ndarray:
        """Non-dipole angular correction for unpolarized X-rays.

        Obtained by averaging L_full over all polarization directions ε ⊥ k.

        L = 1 - (β/2) P₂(cos α) + (δ + γ/2 · sin²α) cos α

        Parameters
        ----------
        beta, gamma, delta : float
            Angular distribution parameters.
        alpha_deg : float or array
            Angle from X-ray beam direction k (degrees).
            α = 90° when detector is perpendicular to beam.

        Returns
        -------
        L : float or array
            Angular correction factor (1 = isotropic).
        """
        alpha = np.radians(alpha_deg)
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)

        P2 = (3.0 * cos_a**2 - 1.0) / 2.0
        dipole = 1.0 - (beta / 2.0) * P2
        nondipole = (delta + gamma / 2.0 * sin_a**2) * cos_a

        return dipole + nondipole

    @classmethod
    def angular_distribution_unpolarized(
        cls,
        element: str,
        orbital: str,
        photon_energy: float,
        emission_angles_deg: np.ndarray | list[float],
        xray_from_normal_deg: float = 88.0,
    ) -> dict[str, Any] | None:
        """Compute angular correction for unpolarized X-rays.

        For lab X-ray sources (Ga Kα, etc.) where the radiation is
        unpolarized. The relevant angle α is between the X-ray beam
        direction and the photoelectron emission direction.

        Parameters
        ----------
        element : str
            Element symbol.
        orbital : str
            Orbital designation.
        photon_energy : float
            Photon energy in eV.
        emission_angles_deg : array-like
            Emission angles θ from surface normal (degrees).
        xray_from_normal_deg : float
            X-ray angle from surface normal (degrees).
            Default 88° = 2° grazing incidence.

        Returns
        -------
        dict or None
            {
                'emission_angles': array,
                'alpha': array (degrees, angle from beam),
                'L_dipole': array,
                'L_unpolarized': array,
                'beta': float, 'gamma': float, 'delta': float,
                'sigma': float, 'binding_energy': float,
            }
        """
        params = cls.lookup(element, orbital, photon_energy)
        if params is None:
            return None

        theta = np.asarray(emission_angles_deg, dtype=np.float64)
        # α = angle from beam = xray_from_normal - θ (coplanar geometry)
        alpha = xray_from_normal_deg - theta

        beta = params["beta"]
        gamma = params["gamma"]
        delta = params["delta"]

        L_dip = cls.L_dipole(beta, alpha)
        L_unp = cls.L_unpolarized(beta, gamma, delta, alpha)

        return {
            "emission_angles": theta,
            "alpha": alpha,
            "L_dipole": L_dip,
            "L_unpolarized": L_unp,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "sigma": params["sigma"],
            "binding_energy": params.get("binding_energy"),
        }

    @classmethod
    def angular_distribution(
        cls,
        element: str,
        orbital: str,
        photon_energy: float,
        emission_angles_deg: np.ndarray | list[float],
        xray_from_normal_deg: float = 88.0,
        phi_deg: float = 0.0,
    ) -> dict[str, Any] | None:
        """Compute angular correction over emission angles.

        For grazing incidence HAXPES:
            ψ = xray_from_normal - θ  (coplanar geometry)

        Parameters
        ----------
        element : str
            Element symbol.
        orbital : str
            Orbital designation.
        photon_energy : float
            Photon energy in eV.
        emission_angles_deg : array-like
            Emission angles θ from surface normal (degrees).
        xray_from_normal_deg : float
            X-ray angle from surface normal (degrees).
            Default 88° = 2° grazing incidence.
        phi_deg : float
            Azimuthal angle (degrees). 0 = scattering plane.

        Returns
        -------
        dict or None
            {
                'emission_angles': array,
                'psi': array (degrees),
                'L_dipole': array,
                'L_full': array,
                'beta': float, 'gamma': float, 'delta': float,
                'sigma': float, 'binding_energy': float,
            }
        """
        params = cls.lookup(element, orbital, photon_energy)
        if params is None:
            return None

        theta = np.asarray(emission_angles_deg, dtype=np.float64)
        psi = xray_from_normal_deg - theta  # coplanar geometry

        beta = params["beta"]
        gamma = params["gamma"]
        delta = params["delta"]

        L_dip = cls.L_dipole(beta, psi)
        L_ful = cls.L_full(beta, gamma, delta, psi, phi_deg)

        return {
            "emission_angles": theta,
            "psi": psi,
            "L_dipole": L_dip,
            "L_full": L_ful,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "sigma": params["sigma"],
            "binding_energy": params.get("binding_energy"),
        }


# ------------------------------------------------------------------
# Interpolation helpers
# ------------------------------------------------------------------

def _interp_log_log(
    points: list[tuple[float, float]],
    x: float,
) -> float:
    """Log-log linear interpolation / extrapolation."""
    points = sorted(points)
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    # Filter positive values for log
    mask = (xs > 0) & (ys > 0)
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 2:
        return ys[0] if len(ys) else 0.0

    log_xs = np.log(xs)
    log_ys = np.log(ys)
    log_x = math.log(max(x, xs[0]))

    result = float(np.interp(log_x, log_xs, log_ys))
    return math.exp(result)


def _interp_linear(
    points: list[tuple[float, float]],
    x: float,
) -> float:
    """Linear interpolation for β, γ, δ; clamped outside the grid.

    Outside the tabulated range this returns the nearest endpoint, which
    is a clamp and not an extrapolation — the caller gets parameters for
    a different energy with nothing to signal it. The grid starts at
    1500 eV, so Al Kα (1486.6 eV) is clamped. Pinned by
    ``tests/test_angular_correction_limits.py``.
    """
    points = sorted(points)
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    if len(xs) < 2:
        return float(ys[0]) if len(ys) else 0.0

    # np.interp clamps to the endpoints rather than extrapolating.
    return float(np.interp(x, xs, ys))
