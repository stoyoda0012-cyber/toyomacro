"""I/O readers for Toyomacro."""

from toyomacro.io.readers.base_reader import (
    BaseReader,
    RawSpectrumData,
    ReaderTransform,
    ReaderWarning,
    SpectrumMetadata,
    create_reader,
    detect_format,
)
from toyomacro.io.readers.lazy_spectrum import DatasetInfo, LazySpectrum
from toyomacro.io.readers.mat_reader import MATReader, detect_mat_version
from toyomacro.io.readers.npl_reader import NPLReader
from toyomacro.io.readers.pxt_reader import PXTReader
from toyomacro.io.readers.ses_reader import SESTxtReader
from toyomacro.io.readers.vamas_reader import VAMASReader

__all__ = [
    # Base
    "BaseReader",
    "RawSpectrumData",
    "ReaderTransform",
    "ReaderWarning",
    "SpectrumMetadata",
    "create_reader",
    "detect_format",
    # Format readers
    "PXTReader",
    "SESTxtReader",
    "VAMASReader",
    "NPLReader",
    # Existing
    "LazySpectrum",
    "DatasetInfo",
    "MATReader",
    "detect_mat_version",
]
