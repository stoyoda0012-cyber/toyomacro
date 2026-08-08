"""Angular correction for photoionization cross-sections.

Non-dipole angular distribution parameters (β, γ, δ) from
Trzhaskovskaya & Yarzhemsky (2018) for HAXPES applications (1.5–10 keV).

Two geometry modes are supported:

1. **Linearly polarized** light (synchrotron):

    L_pol(ψ, φ) = 1 + β P₂(cos θ_ε) + (δ + γ cos²θ_ε) sin θ_ε cos φ

    where θ_ε = angle from polarization vector ε to electron emission,
          φ = azimuth around ε measured from the (ε, k) plane, taken
              φ = 0 in the half containing **+k**. That choice is what
              puts the unpolarized angle on k rather than on its
              supplement; the other half would land on ψ.

    For coplanar HAXPES:  ψ = σ_s − θ_emission,  φ = 0, with σ_s
    *signed* — as drawn below that is −56° − 27° = −83°, not the 29°
    the sketch's unsigned label would give.

2. **Unpolarized** light (lab X-ray, e.g. Ga Kα):

    L_unpol(α) = 1 − (β/2) P₂(cos α) + (δ + γ/2 · sin²α) cos α

    where α = angle from X-ray beam direction k to electron emission.
    Derived by averaging L_pol over all ε ⊥ k.
    Axially symmetric around k (no φ dependence).

P₂(x) = (3x² − 1) / 2  (2nd Legendre polynomial).

Which angle is which
--------------------

Six symbols appear below and only θ is measured. ``α_xray`` and ``σ_s``
name the same thing — where the source sits — but not the same number:
the sketch labels magnitudes, while every rule below consumes ``σ_s``
*signed* (negative for a source on the opposite side of the normal
from the analyzer, so −56° as drawn). ``α`` and ``α_xray`` differ by
more than a subscript::

     hν                        normal        e⁻
                                  ^
       \\                          |        /
          \\                       |       /
             \\.-------- ψ = 83° --|-----./
                \\                 |     /
                   \\              |    /
                     α_xray = 56° | θ = 27°
                         \\        |  /
                            \\     | /
                               \\  |/
        --------------------------+-----------------------
                           sample surface

Angles in the sketch are magnitudes, as drawn. The values are
representative, not any particular instrument. (Apparent angles depend
on the font's character-cell aspect ratio; the drawing assumes 2:1.)

- **θ** — emission angle from the surface normal. Measured; the
  analyzer's angle axis.
- **α_xray** — where the *source* sits, as an incidence angle from the
  normal. An instrument property; confirm it, there is no safe default.
- **ψ** — angle from the direction *toward the source* to the emission
  direction. As drawn, 56° + 27° = 83°.
- **α** — angle from the beam's *propagation direction* **k** to the
  emission direction. As drawn, 180° − 83° = **97°**.

ψ and α are supplements, not synonyms, and ``cos α = −cos ψ``. Which of
the two a formula wants is not a matter of taste:

- ``L_dipole`` cannot tell them apart. P₂ is even under ``cos → −cos``,
  so ψ and α give the same number. Nothing you do here is observable.
- ``L_unpolarized`` **wants α, from k**. Its non-dipole term carries a
  bare ``cos α``, so it is *odd about 90°*: feeding it ψ instead leaves
  the dipole part alone and reverses that term exactly.
- ``L_full`` is unresolved; see its docstring. Do not settle it by
  analogy with either of the above.

What fixes α to k, rather than to the direction toward the source, is
the polarization average this module already claims. Averaging the
polarized form above over ε ⊥ k, the factor ``sin θ_ε cos φ`` is the
coordinate-free ``k̂·p̂``; carrying the average through gives
``⟨P₂(cos θ_ε)⟩ = −P₂(cos α)/2`` and
``⟨(δ + γ cos²θ_ε)(k̂·p̂)⟩ = (δ + γ/2 sin²α) cos α`` — which is
``L_unpol`` with **α measured from k**. The derivation holds for any
signs of β, γ and δ, and ``test_angular_geometry.py`` carries it out
numerically.

Do **not** use "forward emission is enhanced" as the diagnostic. It is
the generic expectation, not a law. The non-dipole factor
``δ + γ/2 sin²α`` is linear in ``sin²α``, so it dips below zero
somewhere in 0° < α < 90° exactly when ``min(δ, δ + γ/2) < 0`` — and
that holds for **35.3% of the (element, orbital, energy) rows** in the
bundled 2018/2019 tables, or 71.9% of the distinct subshells at one
energy or more. Forward emission is *suppressed* there. Nor is
``γ > 0`` sufficient: δ can reverse it alone, which happens on 21.4% of
rows. A reader who checks their geometry against forward-peaking will
conclude their κ is wrong on those subshells when it is right.
``test_angular_geometry.py`` recomputes these fractions from the
bundled tables rather than trusting them.

Getting α from the API
----------------------

``angular_distribution`` and ``angular_distribution_unpolarized`` both
compute ``xray_from_normal_deg - theta``. For the unpolarized entry
point that difference is α, so ``xray_from_normal_deg`` must be **κ,
the propagation direction measured from the outward normal**, signed in
the same sense as θ. Writing σ_s for the signed angle at which the
source sits::

    κ = σ_s + 180°  (σ_s ≤ 0)         α = κ − θ
    κ = σ_s − 180°  (σ_s > 0)

That is a *reversal*, not a change of sign and not a supplement — both
of those give a different, and wrong, κ. Normal incidence is the case
that catches the tidier-looking ``σ_s − 180°·sign(σ_s)``: at σ_s = 0
it returns 0, which points out of the sample.

The beam has to travel *into* the sample, so a physical κ always has
**|κ| > 90°**. That is a check you can apply without knowing anything
about the instrument, and ``angular_distribution_unpolarized`` warns
when it fails.

Worked from the sketch: the source sits at σ_s = −56°, so κ = +124° and
α = 124° − 27° = 97°. Note that κ is *not* α_xray, and not −α_xray
either: passing −56° gives ψ = −83°, whose cosine has the wrong sign,
and passing +56° gives 29°, which is neither angle in the picture.
For a source mirrored onto the other side (σ_s = +56°) the answer is
κ = −124°, not +124°.

Two traps, in order of how much they cost:

1. **Passing θ where the derived angle belongs.** Does not raise;
   returns a plausible wrong number, off by a factor of 20 in the
   recorded case (see ``L_dipole``).
2. **Passing ψ where α belongs.** Does not raise; leaves ``L_dipole``
   unchanged to within rounding and inverts the non-dipole term in
   ``L_unpolarized``. Because the magnitudes 83° and 97° are both
   unremarkable and the dipole part is unchanged, nothing downstream
   looks wrong.

``L_full`` is not even in ψ: its non-dipole term flips with the sign.
Over the full range it is negative for 23% of ψ — on −178.8°…−137.8°
and −42.2°…−1.2° for the Si 1s 9.25 keV parameters — where it is not a
correction factor at all. Nothing in the API checks, so verify the sign
before using ``L_full``.

Sign convention
---------------

``L_unpol`` above is ``1 − (β/2) P₂``, the **unpolarized** form with the
angle measured from the beam **k**. The form quoted throughout the
synchrotron literature, ``1 + β P₂(cos θ_ε)``, differs in both sign and
reference axis: θ_ε is measured from the **polarization vector ε**.
Transcribing one for the other is a real error, not a cosmetic one.
``L_full`` currently mixes the two — see its docstring.

References
----------
- Trzhaskovskaya, M. B. & Yarzhemsky, V. G. (2018). At. Data Nucl. Data Tables.
- Willis et al. (2024). Digitisation of Trzhaskovskaya Dirac-Fock Photoionisation
  Parameters for HAXPES Applications (1.5-10 keV).
- Scofield, J. H. (1976). J. Electron Spectrosc. Relat. Phenom.

**Only j-resolved orbital labels are defined.** A bare label such as
``'2p'`` is not rejected at runtime yet, but it is undefined and
deprecated: it resolves to whichever j component the suffix search
happens to find first, rather than to the subshell. Unlike
`CrossSection`, this module cannot simply sum the components — the
2018/2019 tables give σ for *completely filled* subshells, and β, γ and δ
are per-component quantities that do not add. Making bare input well
defined is deferred rather than guessed at.

Usage
-----
    from toyomacro.data.angular_correction import AngularCorrection

    # Lookup parameters — j-resolved label, not a bare subshell
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
import warnings
from typing import Any

import numpy as np

from toyomacro.data.paths import load_trzh2018_data, load_trzh2019_data


class _UnstatedAngle(float):
    """A default that remembers the caller never stated the geometry.

    A plain ``88.0`` default cannot tell "I verified my instrument is at
    88°" from "I did not think about it", and the two deserve different
    treatment: the x-ray incidence angle sets the *shape* of the angular
    dependence, not just its scale. Subclassing ``float`` keeps the
    parameter a float with the same numeric default in the signature,
    while ``type(...) is _UnstatedAngle`` still identifies the unstated
    case. An explicit ``88.0`` from the caller is a plain float and does
    not warn.
    """

    __slots__ = ()


#: Placeholder incidence angle: 2° grazing, a synchrotron-like geometry.
#: Not a property of any particular instrument — see the warning below.
_XRAY_ANGLE_UNSTATED = _UnstatedAngle(88.0)


def _warn_if_geometry_unstated(xray_from_normal_deg: float) -> None:
    """Warn when the incidence angle was inherited rather than stated."""
    if type(xray_from_normal_deg) is not _UnstatedAngle:
        return
    warnings.warn(
        "xray_from_normal_deg was left at its placeholder value of 88.0° "
        "(2° grazing incidence). This is not a property of your "
        "instrument, and the assumed incidence angle sets the shape of "
        "the angular dependence, not merely its scale: for Si 1s at "
        "9.25 keV the spread of L_dipole across a 51°–9° emission fan is "
        "85% at 88° incidence and 216% at 55°. Pass the angle your "
        "instrument actually uses. On the unpolarized entry point that "
        "angle is κ, the propagation direction, for which 88° is not a "
        "valid value at all — a beam entering the sample has |κ| > 90°, "
        "so the placeholder inverts the non-dipole term as well.",
        UserWarning,
        stacklevel=3,
    )


def _warn_if_beam_leaves_the_sample(xray_from_normal_deg: float) -> None:
    """Warn when κ describes a beam travelling out of the surface.

    Only the unpolarized path calls this. There ``xray_from_normal_deg``
    is κ, the propagation direction from the outward normal, and a beam
    that enters the sample has ``|κ| > 90°``. Passing the source's own
    position instead — the natural mistake, since that is the number an
    instrument datasheet quotes — gives ``|κ| < 90°`` and silently
    inverts the non-dipole term. The dipole part is unchanged, so
    nothing downstream looks wrong.

    Not raised: the check cannot distinguish a genuine convention error
    from an exactly-grazing geometry, and the polarized entry point uses
    the same parameter under a convention that is still unresolved.
    """
    if type(xray_from_normal_deg) is _UnstatedAngle:
        return  # already warned about, and more specifically
    kappa = np.abs(np.asarray(xray_from_normal_deg, dtype=float)) % 360.0
    kappa = np.where(kappa > 180.0, 360.0 - kappa, kappa)
    if np.all(kappa > 90.0):
        return
    warnings.warn(
        f"xray_from_normal_deg={xray_from_normal_deg}° describes a beam "
        f"propagating out of the sample (|κ| = {np.min(kappa):g}° ≤ 90°). This "
        "parameter is the propagation direction κ from the outward "
        "normal, not the angle at which the source sits: for a source "
        "at signed incidence σ_s, κ = σ_s + 180° if σ_s <= 0, else "
        "σ_s − 180°. Passing the "
        "source angle leaves L_dipole unchanged and inverts the "
        "non-dipole term of L_unpolarized, because cos α = −cos ψ.",
        UserWarning,
        stacklevel=3,
    )


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
            Angle between the photon direction and the emission
            direction (degrees). Even under ``cos → −cos``, so it
            does not matter here whether that is measured from the
            beam k or from the direction toward the source — the
            distinction the module docstring draws is invisible to
            this function, and load-bearing for ``L_unpolarized``.

            **Not the emission angle from the surface normal.** In
            angle-resolved work θ (from the normal) is the measured
            variable and ψ is derived from it and the incidence
            geometry — for a coplanar setup
            ``psi = xray_from_normal - theta``. Passing θ here returns a
            plausible but wrong number instead of raising: for
            β = 1.9255 and θ = 8.78°, this gives 0.0709, where the
            correct ψ (79.22° at 88° incidence) gives 1.4309 — a factor
            of 20, and both are unremarkable numbers between 0 and 2.

            ``angular_distribution()`` takes θ directly and does this
            conversion for you; prefer it.
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

        .. warning::

           **Convention under review — do not rely on this value.**
           This expression combines two different angular conventions,
           and the module's own documentation contradicts it:

           - The module docstring gives the linearly polarized form as
             ``1 + β P₂(cos θ_ε) + (δ + γ cos²θ_ε) sin θ_ε cos φ`` with
             θ_ε measured **from the polarization vector ε**. The
             dipole term here is instead ``1 - (β/2) P₂``, which is the
             *polarization-averaged* coefficient, applied to an angle
             documented as being **from the photon direction k**.
           - ``L_unpolarized`` states it is "obtained by averaging
             L_full over all polarization directions ε ⊥ k". That
             relation does not hold for this implementation: averaging
             this expression over φ sends cos φ to zero, leaving exactly
             ``L_dipole``, not ``L_unpolarized``.

           Resolving it requires reading the printed formula in
           Trzhaskovskaya & Yarzhemsky, *At. Data Nucl. Data Tables*
           **119**, 99 (2018), which has not been done. Nothing is
           changed until then, so current values remain reproducible;
           ``test_angular_geometry.py`` pins them as a record of present
           behaviour, explicitly **not** as a claim of correctness.

           ``L_dipole`` and ``L_unpolarized`` are unaffected: each
           matches the corresponding form in the module docstring.

        For synchrotron sources. For unpolarized lab sources, use
        L_unpolarized() instead.

        Parameters
        ----------
        beta, gamma, delta : float
            Angular distribution parameters.
        psi_deg : float or array
            Angle from the photon direction (degrees) — but see the
            warning above: which axis this is measured from is the
            open question, and unlike ``L_dipole`` this function is
            not even in it, so the choice changes the value. Not the
            emission angle from the surface normal.
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
            Angle from the beam's **propagation direction k** to the
            electron emission direction (degrees). α = 90° when the
            detector is perpendicular to the beam.

            **Not the emission angle from the surface normal**, and
            **not the angle from the direction toward the source**. The
            two mistakes cost different things:

            - θ for α — the factor-of-20 trap described in
              ``L_dipole``.
            - ψ (from the source, the supplement of α) for α — the
              dipole part is unchanged and the non-dipole term inverts,
              because ``cos α = −cos ψ``. Silent, and the reason this
              parameter names the propagation direction explicitly.

            For a coplanar setup ``alpha = kappa - theta`` where κ is
            the propagation direction from the outward normal, with
            ``|κ| > 90°``. See the module docstring;
            ``angular_distribution_unpolarized()`` takes θ and does the
            conversion for you.

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
        xray_from_normal_deg: float = _XRAY_ANGLE_UNSTATED,
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
            **κ, the beam's propagation direction**, measured from the
            outward surface normal and signed in the same sense as θ.
            α is formed as ``kappa - theta``.

            This is not where the source sits: for a source at signed
            incidence σ_s, κ is σ_s *reversed* — ``σ_s + 180°`` if
            σ_s ≤ 0, else ``σ_s − 180°``. A beam that
            enters the sample always has ``|κ| > 90°``, and a value
            failing that emits a ``UserWarning`` — it describes a beam
            travelling back out of the surface, and inverts the
            non-dipole term. See the module docstring.

            The default 88° is a **placeholder, not an instrument
            value**, and leaving it in place emits a ``UserWarning``. It
            is also not a valid κ (it reads as 2° grazing expressed as a
            source position; the matching κ is −92°), which is why the
            placeholder is a particularly poor fit here. State your
            instrument's geometry.

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

        _warn_if_geometry_unstated(xray_from_normal_deg)
        _warn_if_beam_leaves_the_sample(xray_from_normal_deg)

        theta = np.asarray(emission_angles_deg, dtype=np.float64)
        # α = angle from the propagation direction k = κ − θ (coplanar)
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
        xray_from_normal_deg: float = _XRAY_ANGLE_UNSTATED,
        phi_deg: float = 0.0,
    ) -> dict[str, Any] | None:
        """Compute angular correction over emission angles.

        **This is the entry point that takes the angle you measured.**
        It accepts θ from the surface normal — the analyzer's angle axis
        — and derives ψ internally::

            ψ = xray_from_normal - θ   (coplanar geometry)

        Do not compute ψ yourself and call ``L_dipole`` with θ: that
        returns a plausible but wrong number rather than raising. See the
        warning in ``L_dipole``.

        The returned ``L_full`` carries an unresolved convention question
        — see ``L_full``'s docstring. ``L_dipole`` in the returned dict
        is unaffected.

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
            X-ray angle from surface normal (degrees), signed, forming
            ``ψ = xray_from_normal_deg − θ``.

            **This is not the same quantity the unpolarized entry point
            takes under the same parameter name.** There it is κ, the
            beam's propagation direction. Here ψ is the angle from the
            direction *toward the source*, so pass the source's own
            signed position — negative for a source on the opposite side
            of the normal from the analyzer, giving ψ = −(α_xray + θ).
            ``L_dipole`` is even in ψ and cannot see the difference;
            ``L_full`` is not, and its convention is unresolved, so this
            instruction is **provisional** and settles nothing about
            ``L_full``. See the module docstring.

            The default 88° (2° grazing) is a **placeholder, not an
            instrument value**, and leaving it in place emits a
            ``UserWarning``. The assumed incidence angle sets the shape
            of the angular dependence, not just its scale — verify it
            against your instrument.
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

        _warn_if_geometry_unstated(xray_from_normal_deg)

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
