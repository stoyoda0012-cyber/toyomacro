"""
Toyomacro HDF5 schema definition.

Compatible with voigtfit/DepthProfiler format (C-order, row-major).

HDF5 Layout (Python C-order):
    specdata:  (n_spectra+1, n_energy)  float32, contiguous
               Row 0 = energy axis, Row 1+ = spectra
    fitpara:   (n_spectra, maxcomp, 9)  float32, contiguous
    otherpara: (10, n_spectra)          float32, contiguous
    xytdata:   (3, n_spectra)           float32, contiguous

Note: MATLAB uses F-order (column-major), so the same data appears
      transposed when viewed in MATLAB vs Python.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import numpy as np


class ToyomacroSchema:
    """
    Toyomacro HDF5 schema definition (voigtfit compatible).

    Defines dataset paths, parameter indices, and data types.
    Uses C-order (row-major) layout for Python compatibility.
    """

    # 1.2.0: additive /provenance group + root schema-version attribute.
    # Files without the root attribute are legacy 1.1.x and read unchanged.
    VERSION: Final[str] = "1.2.0"

    # Root attribute carrying the file schema version (absent = legacy 1.1.x)
    ATTR_SCHEMA_VERSION: Final[str] = "toyomacro_schema_version"

    # Dataset paths
    PATH_SPECDATA: Final[str] = "/specdata"
    PATH_FITPARA: Final[str] = "/fitpara"
    PATH_OTHERPARA: Final[str] = "/otherpara"
    PATH_XYTDATA: Final[str] = "/xytdata"
    PATH_MISC: Final[str] = "/misc"
    # Provenance namespace: facts read from the upstream input file
    # (PXT/IBW/VAMAS/NPL/SES) plus the reader/importer transform history.
    # See docs/hdf5_provenance_phase_b1_design.md. The /uncertainty
    # namespace is reserved by that document and NOT created here.
    PATH_PROVENANCE: Final[str] = "/provenance"

    # Version of the /provenance group schema (independent of VERSION)
    PROVENANCE_SCHEMA_VERSION: Final[str] = "1.0"

    # Data types (all float32 for MATLAB compatibility)
    DTYPE_SPECDATA = np.float32
    DTYPE_FITPARA = np.float32
    DTYPE_OTHERPARA = np.float32
    DTYPE_XYTDATA = np.float32

    # Dimension definitions
    FITPARA_COLS: Final[int] = 9  # Last axis of fitpara
    OTHERPARA_ROWS: Final[int] = 10  # First axis of otherpara
    XYTDATA_ROWS: Final[int] = 3  # First axis of xytdata

    class FitparaIndex(IntEnum):
        """
        fitpara column indices (last axis, 0-based).

        Shape: (n_spectra, maxcomp, 9)
        Access: fitpara[spectrum_idx, component_idx, FitparaIndex.CENTER]
        """
        PEAK_HEIGHT = 0       # Peak height / amplitude (peakHeight)
        CENTER = 1            # Peak center energy (energyCentroid) [eV]
        FWHM_GAUSSIAN = 2     # Gaussian FWHM [eV]
        FWHM_LORENTZIAN = 3   # Lorentzian FWHM [eV]
        ASYMMETRY = 4         # Asymmetry / DS singularity (alpha)
        BRANCH_RATIO = 5      # Spin-orbit branch ratio (BR)
        SO_SPLIT = 6          # Spin-orbit splitting (SO) [eV]
        FUNCTION_TYPE = 7     # Lineshape type (1=Voigt, etc.)
        INTEGRATED_AREA = 8   # Integrated area [eV·counts] (trapz)

    class OtherparaIndex(IntEnum):
        """
        otherpara row indices (first axis, 0-based).

        Shape: (10, n_spectra)
        Access: otherpara[OtherparaIndex.CHI2, spectrum_idx]
        """
        NUM_COMPONENTS = 0
        X_RANGE_MIN = 1
        X_RANGE_MAX = 2
        BG_TYPE = 3
        BG_PARAM_A = 4
        BG_PARAM_B = 5
        BG_PARAM_C = 6
        MAX_HEIGHT = 7
        TOTAL_AREA = 8
        CHI2 = 9

    class FunctionType(IntEnum):
        """Lineshape function types (matching MATLAB convention)."""
        VOIGT = 1
        PSEUDO_VOIGT = 2
        GAUSSIAN = 3
        LORENTZIAN = 4
        DONIACH_SUNJIC = 5
        FERMI_DIRAC = 6

    class BackgroundType(IntEnum):
        """Background function types."""
        SHIRLEY = 1
        TOUGAARD = 2
        LINEAR = 3
        LEVEL = 4
        NONE = 5

    @classmethod
    def get_misc_defaults(cls, maxcomp: int = 6) -> dict:
        """
        Get default values for misc metadata.

        Args:
            maxcomp: Maximum number of peak components

        Returns:
            Dictionary with default misc values
        """
        return {
            "xdata_str": "",
            "ydata_str": "",
            "tdata_str": "",
            "numberofslice": 1,
            "fermienergy": 0.0,
            "bindingenergysign": 1,  # 1 = Binding Energy
            "sodeconv": 0,
            "maxcomp": maxcomp,
        }
