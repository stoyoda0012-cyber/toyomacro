"""Bridge from FittingTemplate to ComponentConfig for hierarchical solving.

Converts domain-knowledge templates (chemical shifts, FWHM, SO doublets)
into the generic ComponentConfig format consumed by ``solve_grouped()``.

template → ComponentConfig → solve_grouped → GUI pipeline.

Usage::

    from toyomacro.fitting.template_bridge import template_to_configs
    from toyomacro.fitting.templates import get_template

    template = get_template("Si2p_oxide")
    configs = template_to_configs(template, reference_be=99.3)
    # → list[ComponentConfig] ready for solve_grouped()
"""

from __future__ import annotations

import warnings
from typing import Literal

from toyomacro.core.fitting_result import LineshapeType
from toyomacro.core.xps_database import get_so_info
from toyomacro.fitting.templates import FittingTemplate
from toyomacro.voigtfit.multipeak_config import ComponentConfig

# Gaussian FWHM → σ: FWHM_G = 2√(2 ln 2) σ ≈ 2.3548 σ
_FWHM_G_TO_SIGMA = 2.3548


def template_to_configs(
    template: FittingTemplate,
    reference_be: float,
    energy_type: Literal["BE", "KE"] = "BE",
) -> list[ComponentConfig]:
    """Convert a FittingTemplate to a list of ComponentConfig.

    Each TemplatePeak becomes one ComponentConfig.  Spin-orbit splitting
    is encoded in ``so_split`` / ``branch_ratio`` of the ComponentConfig
    (the solver handles doublet expansion internally).

    Doniach-Sunjic peaks emit a warning and are approximated as Voigt,
    since ``solve_grouped`` only supports Voigt lineshapes.

    Args:
        template: FittingTemplate with chemical shifts and widths.
        reference_be: Binding energy of the reference peak (eV).
            This anchors all delta_be offsets to absolute positions.
        energy_type: ``"BE"`` or ``"KE"``.  Determines sign convention
            for peak positions.

    Returns:
        List of ComponentConfig in the same order as ``template.peaks``.
        DS peaks are included with a Voigt approximation.

    Raises:
        ValueError: If template has no peaks.
    """
    if not template.peaks:
        msg = f"Template '{template.name}' has no peaks"
        raise ValueError(msg)

    is_be = energy_type == "BE"

    # Look up SO info if the template uses doublets
    so_info = get_so_info(template.element) if template.apply_so else None

    configs: list[ComponentConfig] = []

    for peak in template.peaks:
        # --- Position ---
        if is_be:
            center = reference_be + peak.delta_be
        else:
            center = reference_be - peak.delta_be

        # --- Widths ---
        sigma = peak.fwhm_g / _FWHM_G_TO_SIGMA
        gamma = peak.fwhm_l / 2.0

        # --- DS warning ---
        if peak.lineshape == LineshapeType.DONIACH_SUNJIC:
            warnings.warn(
                f"Template peak '{peak.name}' uses Doniach-Sunjic lineshape "
                f"(asymmetry={peak.asymmetry:.3f}). solve_grouped only "
                f"supports Voigt — using Voigt approximation.",
                stacklevel=2,
            )

        # --- SO doublet ---
        so_split = 0.0
        branch_ratio = 1.0
        if so_info is not None:
            so_split = so_info.so_split
            # SOInfo.branch_ratio = area(j-)/area(j+) e.g. 0.5 for p
            # ComponentConfig.branch_ratio = I_main/I_partner e.g. 2.0 for p
            branch_ratio = 1.0 / so_info.branch_ratio

        configs.append(
            ComponentConfig(
                center=center,
                sigma=sigma,
                gamma=gamma,
                so_split=so_split,
                branch_ratio=branch_ratio,
            )
        )

    return configs
