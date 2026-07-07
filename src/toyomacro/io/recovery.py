"""
Recovery functions for incomplete HDF5 fitpara files.

Provides validation and recovery capabilities for files that were
interrupted during streaming write operations.

Usage:
    # Check if file is valid
    result = validate_fitpara_file('output.h5')
    if not result['valid']:
        print(f"Invalid: {result['error']}")

    # Get recovery info for incomplete file
    info = recover_partial_fit('output.h5')
    if info['can_resume']:
        print(f"Resume from index: {info['resume_from_idx']}")

    # Clear incomplete marker after manual recovery
    clear_incomplete_marker('output.h5')
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from toyomacro.io.schema import ToyomacroSchema


def validate_fitpara_file(h5_path: str | Path) -> dict[str, Any]:
    """
    Validate a fitpara HDF5 file for completeness and integrity.

    Checks for incomplete markers, presence of fitpara dataset,
    and verifies that data has been written (not all NaN).

    Args:
        h5_path: Path to HDF5 file

    Returns:
        Dict with keys:
            'valid': bool - True if file is complete and valid
            'incomplete': bool - True if file has incomplete marker
            'last_batch': int - Last completed batch index (-1 if not set)
            'n_spectra': int - Number of spectra in fitpara
            'n_components': int - Number of components
            'has_fitpara': bool - True if fitpara dataset exists
            'error': str or None - Error message if invalid
    """
    result: dict[str, Any] = {
        "valid": False,
        "incomplete": False,
        "last_batch": -1,
        "n_spectra": 0,
        "n_components": 0,
        "has_fitpara": False,
        "error": None,
    }

    try:
        with h5py.File(h5_path, "r") as f:
            # Check incomplete marker (support both toyomacro and voigtfit markers)
            if "_toyomacro_incomplete" in f.attrs:
                result["incomplete"] = True
                result["last_batch"] = f.attrs.get("_toyomacro_last_batch", -1)
            elif "_voigtfit_incomplete" in f.attrs:
                result["incomplete"] = True
                result["last_batch"] = f.attrs.get("_voigtfit_last_batch", -1)

            # Check fitpara dataset
            if ToyomacroSchema.PATH_FITPARA in f:
                result["has_fitpara"] = True
                fitpara = f[ToyomacroSchema.PATH_FITPARA]
                result["n_spectra"] = fitpara.shape[0]
                result["n_components"] = fitpara.shape[1]

                # Check for NaN values in first and last rows
                first_row = fitpara[0, :, :]
                last_row = fitpara[-1, :, :]

                if np.all(np.isnan(first_row)):
                    result["error"] = "First row is all NaN (no data written)"
                elif np.all(np.isnan(last_row)):
                    result["error"] = "Last row is all NaN (incomplete write)"
                elif result["incomplete"]:
                    result["error"] = "File marked as incomplete"
                else:
                    result["valid"] = True
            else:
                result["error"] = "fitpara dataset not found"

    except Exception as e:
        result["error"] = str(e)

    return result


def recover_partial_fit(h5_path: str | Path) -> dict[str, Any]:
    """
    Get information needed to resume a partial fit.

    Analyzes an incomplete file to determine where writing stopped,
    enabling resumption of interrupted fitting operations.

    Args:
        h5_path: Path to HDF5 file

    Returns:
        Dict with keys:
            'can_resume': bool - True if file can be resumed
            'last_batch': int - Last completed batch index
            'resume_from_idx': int - Index to resume from
            'n_spectra': int - Total number of spectra
            'n_components': int - Number of components
            'error': str or None - Error message if cannot resume
    """
    result: dict[str, Any] = {
        "can_resume": False,
        "last_batch": -1,
        "resume_from_idx": 0,
        "n_spectra": 0,
        "n_components": 0,
        "error": None,
    }

    try:
        with h5py.File(h5_path, "r") as f:
            # Check incomplete marker
            has_marker = (
                "_toyomacro_incomplete" in f.attrs
                or "_voigtfit_incomplete" in f.attrs
            )
            if not has_marker:
                result["error"] = "File is not marked as incomplete"
                return result

            result["last_batch"] = f.attrs.get(
                "_toyomacro_last_batch",
                f.attrs.get("_voigtfit_last_batch", -1),
            )

            if ToyomacroSchema.PATH_FITPARA not in f:
                result["error"] = "fitpara dataset not found"
                return result

            fitpara = f[ToyomacroSchema.PATH_FITPARA]
            result["n_spectra"] = fitpara.shape[0]
            result["n_components"] = fitpara.shape[1]

            # Find first row with all NaN (where writing stopped)
            # Check every 1000th row for efficiency
            step = max(1, result["n_spectra"] // 1000)
            resume_idx = result["n_spectra"]

            for idx in range(0, result["n_spectra"], step):
                row = fitpara[idx, :, :]
                if np.all(np.isnan(row)):
                    # Found NaN row, search backwards for exact position
                    for fine_idx in range(max(0, idx - step), idx):
                        row = fitpara[fine_idx, :, :]
                        if np.all(np.isnan(row)):
                            resume_idx = fine_idx
                            break
                    else:
                        resume_idx = idx
                    break

            result["resume_from_idx"] = resume_idx
            result["can_resume"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def clear_incomplete_marker(h5_path: str | Path) -> bool:
    """
    Remove incomplete marker from HDF5 file.

    Use after manual recovery or when you want to mark a file as complete.

    Args:
        h5_path: Path to HDF5 file

    Returns:
        True if marker was removed, False if not present
    """
    try:
        with h5py.File(h5_path, "r+") as f:
            removed = False
            # Handle both toyomacro and voigtfit markers
            for marker in [
                "_toyomacro_incomplete",
                "_toyomacro_last_batch",
                "_voigtfit_incomplete",
                "_voigtfit_last_batch",
            ]:
                if marker in f.attrs:
                    del f.attrs[marker]
                    removed = True
            return removed
    except Exception:
        return False


def get_file_status(h5_path: str | Path) -> dict[str, Any]:
    """
    Get comprehensive status information about an HDF5 file.

    Combines validation and recovery information into a single
    convenient function call.

    Args:
        h5_path: Path to HDF5 file

    Returns:
        Dict with comprehensive status information including:
            - validation results
            - recovery possibility
            - dataset sizes and shapes
    """
    validation = validate_fitpara_file(h5_path)
    recovery = recover_partial_fit(h5_path) if validation["incomplete"] else None

    status = {
        "exists": Path(h5_path).exists(),
        "validation": validation,
        "recovery": recovery,
        "file_size_mb": None,
        "datasets": {},
    }

    try:
        status["file_size_mb"] = Path(h5_path).stat().st_size / (1024 * 1024)

        with h5py.File(h5_path, "r") as f:
            for name in f.keys():
                obj = f[name]
                if isinstance(obj, h5py.Dataset):
                    status["datasets"][name] = {
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "chunks": obj.chunks,
                    }
    except Exception:
        pass

    return status
