"""Fitting templates for well-known XPS chemical systems.

Encodes domain knowledge: chemical shifts, width ordering, SO doublets,
and lineshape selections for multi-component fitting of known systems.

Pattern follows ``xps_database.py``: frozen dataclass + dict registry +
factory function.

Usage::

    from toyomacro.fitting.templates import get_template, list_templates

    template = get_template("Si2p_oxide")
    # Use with AutoFitter:
    result = fitter.fit_with_template(spectrum, "Si2p_oxide")
"""

from __future__ import annotations

from dataclasses import dataclass

from toyomacro.core.fitting_result import LineshapeType

# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------

@dataclass(frozen=True)
class TemplatePeak:
    """Definition of a single chemical-state peak within a template.

    Attributes:
        name: Human-readable label (e.g. ``"Si0"``, ``"C-C"``).
        delta_be: Chemical shift from reference peak in eV.
            Stored as delta in **binding energy**: positive means
            higher BE (more oxidised).  Reference peak has 0.0.
        fwhm_g: Estimated Gaussian FWHM (eV).
        fwhm_l: Estimated Lorentzian FWHM (eV).
        width_fraction: Interpolation fraction for width ordering
            (0.0 = narrowest, 1.0 = broadest).  ``None`` means the
            peak is not width-constrained.
        lineshape: Lineshape type for this peak.
        asymmetry: Doniach-Sunjic asymmetry index (only for DS).
        is_reference: Whether this is the reference peak (delta_be=0).
        relative_area: Rough expected relative area (for initial
            amplitude estimation only, not constrained in the fit).
    """

    name: str
    delta_be: float
    fwhm_g: float = 0.5
    fwhm_l: float = 0.3
    width_fraction: float | None = None
    lineshape: LineshapeType = LineshapeType.VOIGT
    asymmetry: float = 0.0
    is_reference: bool = False
    relative_area: float | None = None


@dataclass(frozen=True)
class FittingTemplate:
    """Complete fitting template for a chemical system.

    A template defines the element/orbital (for SO lookup), an ordered
    list of chemical-state peaks, and fitting strategy flags.

    Attributes:
        name: Template identifier (e.g. ``"Si2p_oxide"``).
        element: XPS label for SO lookup (e.g. ``"Si2p"``).
        description: Human-readable description.
        peaks: Ordered tuple of :class:`TemplatePeak` definitions.
        apply_so: Whether to add SO doublet partners.
        width_ordering: Whether to enforce monotonic width via
            interpolation constraints.
        reference_be: Optional default binding energy for the reference
            peak.  Used to auto-detect the reference position when the
            user does not supply one explicitly.
    """

    name: str
    element: str
    description: str
    peaks: tuple[TemplatePeak, ...]
    apply_so: bool = True
    width_ordering: bool = True
    reference_be: float | None = None

    def get_reference_be(self) -> float | None:
        """Get reference binding energy, with database fallback.

        Returns the explicitly-set ``reference_be`` if available,
        otherwise looks up from ``toyomacro.data.BindingEnergy``.
        """
        if self.reference_be is not None:
            return self.reference_be
        try:
            from toyomacro.data import BindingEnergy
            return BindingEnergy.from_string(self.element)
        except ImportError:
            return None

    @property
    def reference_peak(self) -> TemplatePeak:
        """Return the reference peak (delta_be == 0)."""
        for p in self.peaks:
            if p.is_reference:
                return p
        msg = f"Template '{self.name}' has no reference peak"
        raise ValueError(msg)

    @property
    def n_chemical_states(self) -> int:
        """Number of chemical states (before SO expansion)."""
        return len(self.peaks)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

_TEMPLATES: dict[str, FittingTemplate] = {}


def _register_builtin_templates() -> None:
    """Populate the registry with built-in templates."""

    # ----- Si2p_oxide: SiO2/Si interface (5 oxidation states) -----
    _TEMPLATES["Si2p_oxide"] = FittingTemplate(
        name="Si2p_oxide",
        element="Si2p",
        description="SiO2/Si interface: 5 oxidation states (Si0 to Si4+)",
        peaks=(
            TemplatePeak(
                name="Si0",
                delta_be=0.0,
                fwhm_g=0.35,
                fwhm_l=0.15,
                width_fraction=0.0,
                is_reference=True,
                relative_area=1.0,
            ),
            TemplatePeak(
                name="Si1+",
                delta_be=0.95,
                fwhm_g=0.45,
                fwhm_l=0.20,
                width_fraction=0.25,
                relative_area=0.3,
            ),
            TemplatePeak(
                name="Si2+",
                delta_be=1.80,
                fwhm_g=0.55,
                fwhm_l=0.25,
                width_fraction=0.50,
                relative_area=0.2,
            ),
            TemplatePeak(
                name="Si3+",
                delta_be=2.60,
                fwhm_g=0.65,
                fwhm_l=0.30,
                width_fraction=0.75,
                relative_area=0.15,
            ),
            TemplatePeak(
                name="Si4+",
                delta_be=3.60,
                fwhm_g=0.80,
                fwhm_l=0.35,
                width_fraction=1.0,
                relative_area=0.8,
            ),
        ),
        apply_so=True,
        width_ordering=True,
        reference_be=99.3,
    )

    # ----- Ta4f_oxide: Ta2O5 / Ta -----
    _TEMPLATES["Ta4f_oxide"] = FittingTemplate(
        name="Ta4f_oxide",
        element="Ta4f",
        description="Ta2O5/Ta: metallic + sub-oxide + oxide",
        peaks=(
            TemplatePeak(
                name="Ta0",
                delta_be=0.0,
                fwhm_g=0.30,
                fwhm_l=0.25,
                width_fraction=0.0,
                lineshape=LineshapeType.DONIACH_SUNJIC,
                asymmetry=0.04,
                is_reference=True,
                relative_area=1.0,
            ),
            TemplatePeak(
                name="Ta_sub",
                delta_be=2.0,
                fwhm_g=0.50,
                fwhm_l=0.30,
                width_fraction=0.5,
                relative_area=0.3,
            ),
            TemplatePeak(
                name="Ta5+",
                delta_be=4.5,
                fwhm_g=0.70,
                fwhm_l=0.35,
                width_fraction=1.0,
                relative_area=0.8,
            ),
        ),
        apply_so=True,
        width_ordering=True,
        reference_be=21.6,
    )

    # ----- C1s_organic: organic contamination -----
    _TEMPLATES["C1s_organic"] = FittingTemplate(
        name="C1s_organic",
        element="C1s",
        description="Organic contamination: C-C, C-O, C=O, O-C=O",
        peaks=(
            TemplatePeak(
                name="C-C",
                delta_be=0.0,
                fwhm_g=0.60,
                fwhm_l=0.20,
                is_reference=True,
                relative_area=1.0,
            ),
            TemplatePeak(
                name="C-O",
                delta_be=1.5,
                fwhm_g=0.65,
                fwhm_l=0.20,
                relative_area=0.3,
            ),
            TemplatePeak(
                name="C=O",
                delta_be=3.0,
                fwhm_g=0.65,
                fwhm_l=0.20,
                relative_area=0.15,
            ),
            TemplatePeak(
                name="O-C=O",
                delta_be=4.25,
                fwhm_g=0.70,
                fwhm_l=0.25,
                relative_area=0.1,
            ),
        ),
        apply_so=False,
        width_ordering=False,
        reference_be=285.0,
    )


_register_builtin_templates()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_template(name: str) -> FittingTemplate | None:
    """Look up a fitting template by name.

    Args:
        name: Template name (e.g. ``"Si2p_oxide"``).

    Returns:
        :class:`FittingTemplate` or ``None`` if not found.
    """
    return _TEMPLATES.get(name)


def list_templates() -> list[str]:
    """Return sorted list of all available template names."""
    return sorted(_TEMPLATES.keys())


def get_templates_for_element(element: str) -> list[FittingTemplate]:
    """Find all templates applicable to an element label.

    Args:
        element: XPS element label (e.g. ``"Si2p"``, ``"C1s"``).

    Returns:
        List of matching :class:`FittingTemplate` instances.
    """
    return [t for t in _TEMPLATES.values() if t.element == element]


def register_template(template: FittingTemplate) -> None:
    """Register a custom template.

    Args:
        template: :class:`FittingTemplate` to register.

    Raises:
        ValueError: If a template with the same name already exists.
    """
    if template.name in _TEMPLATES:
        msg = (
            f"Template '{template.name}' already registered. "
            f"Use a different name or remove the existing one first."
        )
        raise ValueError(msg)
    _TEMPLATES[template.name] = template
