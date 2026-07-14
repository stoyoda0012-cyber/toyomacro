"""Shared resolver for the local XPS test-data tree.

Several tests exercise the fitting/reader code against real measured
spectra that are **not** bundled with the package (they are large and,
in some cases, not ours to redistribute). Those tests skip cleanly when
the data is absent, so the suite stays green on any machine.

Point the tests at your own copy by setting the ``TOYOMACRO_TESTDATA``
environment variable to the root of the test-data tree, e.g.::

    export TOYOMACRO_TESTDATA="$HOME/xps-testdata"

The expected layout under that root is::

    <root>/Fitting/readtest/{pxt,ibw,txt,vms,npl}/   # reader round-trip
    <root>/Fitting/arpes/Si2p_arpes.txt              # Si 2p ARXPS
    <root>/Fitting/maptest/Si2p_maptest.h5           # Si 2p chemical map

When the variable is unset it defaults to ``~/xps-testdata`` so the
public source carries no machine-specific path.
"""

from __future__ import annotations

import os
from pathlib import Path


def testdata_root() -> Path:
    """Root of the local XPS test-data tree (``TOYOMACRO_TESTDATA``)."""
    env = os.environ.get("TOYOMACRO_TESTDATA")
    if env:
        return Path(env).expanduser()
    return Path.home() / "xps-testdata"


def fitting_dir() -> Path:
    """The ``Fitting`` subtree holding reader/ARXPS/map fixtures."""
    return testdata_root() / "Fitting"
