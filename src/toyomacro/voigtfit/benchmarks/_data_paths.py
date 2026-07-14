"""Default data paths for voigtfit benchmark scripts.

Override the data root by setting the ``VOIGTFIT_DATA_ROOT``
environment variable. When unset, defaults to
``~/Documents/MATLAB/Simulation``.

The benchmark scripts under :mod:`toyomacro.voigtfit.benchmarks` look
up large image/roundtrip datasets that are not bundled with the
package. They were originally written against the maintainer's local
layout; these helpers let any user point them at their own data tree
without editing source.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    """Return the configured data root, expanding ``~``."""
    env = os.environ.get("VOIGTFIT_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Documents" / "MATLAB" / "Simulation"


def roundtrip_image_dir() -> Path:
    return data_root() / "Roundtrip" / "image"


def fuji_dir() -> Path:
    return data_root() / "Fuji" / "PSNRTest" / "Fuji"


def speedtest_dir() -> Path:
    return data_root() / "Fuji" / "SpeedTest"


def gazou_dir() -> Path:
    """Directory of sample images for GVRT demos.

    Override with the ``VOIGTFIT_GAZOU_DIR`` environment variable; when
    unset, defaults to ``~/xps-testdata/Gazou`` so the source carries no
    machine-specific path.
    """
    env = os.environ.get("VOIGTFIT_GAZOU_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / "xps-testdata" / "Gazou"
