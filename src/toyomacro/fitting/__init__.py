"""Fitting engine for XPS peak analysis."""

from __future__ import annotations

import numpy as np

from toyomacro.fitting.autofitter import AutoFitConfig, AutoFitResult, AutoFitter
from toyomacro.fitting.batch_fitter import BatchFitConfig, BatchFitResult, BatchFitter
from toyomacro.fitting.fast_voigt import (
    NUMBA_AVAILABLE,
    VOIGTFIT_AVAILABLE,
    FastFitConfig,
    FastFitResult,
    FastVoigtFitter,
    shirley_batch_numba,
    shirley_calculate_numba,
)
from toyomacro.fitting.peak_model import MultiPeakModel, PeakModel
from toyomacro.fitting.template_bridge import template_to_configs
from toyomacro.fitting.templates import (
    FittingTemplate,
    TemplatePeak,
    get_template,
    get_templates_for_element,
    list_templates,
    register_template,
)


def extract_peak_config(result: AutoFitResult) -> dict | None:
    """Extract peak config (centers, sigmas, gamma) from AutoFitResult.

    Returns dict with numpy arrays suitable for voigtfit batch fitting,
    or None if result is unsuccessful or has no components.
    """
    if not result.success:
        return None
    fr = result.fitting_result
    if not fr.components:
        return None
    centers = np.array([c.center for c in fr.components])
    sigmas = np.array([c.fwhm_g / 2.355 for c in fr.components])
    gamma = fr.components[0].fwhm_l / 2
    return {'centers': centers, 'sigmas': sigmas, 'gamma': gamma}


__all__ = [
    "extract_peak_config",
    "AutoFitter",
    "AutoFitConfig",
    "AutoFitResult",
    "BatchFitter",
    "BatchFitConfig",
    "BatchFitResult",
    "PeakModel",
    "MultiPeakModel",
    # Fast engine
    "VOIGTFIT_AVAILABLE",
    "NUMBA_AVAILABLE",
    "FastVoigtFitter",
    "FastFitConfig",
    "FastFitResult",
    "shirley_calculate_numba",
    "shirley_batch_numba",
    # Templates
    "FittingTemplate",
    "TemplatePeak",
    "get_template",
    "list_templates",
    "get_templates_for_element",
    "register_template",
    # Template bridge
    "template_to_configs",
]
