"""Version consistency: one authoritative source (pyproject.toml).

Every runtime-visible version must agree with the installed package
metadata, which in turn comes from ``pyproject.toml``.
"""

import re
import subprocess
import sys
from pathlib import Path

import toyomacro
import toyomacro.voigtfit


def _pyproject_version() -> str:
    import tomllib
    root = Path(__file__).resolve().parents[1]
    with open(root / 'pyproject.toml', 'rb') as f:
        return tomllib.load(f)['project']['version']


def test_package_version_matches_pyproject():
    assert toyomacro.__version__ == _pyproject_version()


def test_voigtfit_shares_package_version():
    assert toyomacro.voigtfit.__version__ == toyomacro.__version__


def test_cli_version_matches_package():
    out = subprocess.run(
        [sys.executable, '-c',
         'from toyomacro.cli import main; import sys; '
         'sys.argv=["toyomacro","--version"]; main()'],
        capture_output=True, text=True)
    printed = (out.stdout + out.stderr).strip()
    assert toyomacro.__version__ in printed


def test_readme_install_pin_matches_package():
    """README's no-clone install command pins the tag of this version."""
    readme = Path(__file__).resolve().parents[1] / 'README.md'
    if not readme.exists():
        import pytest
        pytest.skip('README.md is not part of this installation')
    m = re.search(r'toyomacro\.git@v(\d+\.\d+\.\d+)"', readme.read_text(encoding='utf-8'))
    assert m, 'the no-clone install command changed shape'
    assert m.group(1) == _pyproject_version()
