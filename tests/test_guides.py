"""The AI-agent user guide ships with the package and names its version.

The guide is read from the installed package so an agent gets the copy
that matches the code, not the repository's main branch. Its version line
and its links must agree, and it must never describe an older version
than the package it ships in.
"""

from __future__ import annotations

import re
import subprocess
import sys

import toyomacro
from toyomacro.guides import AGENT_USER_GUIDE, read_guide


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.match(r"(\d+)\.(\d+)\.(\d+)", text).groups())


def test_the_guide_is_readable_from_the_package():
    text = read_guide()
    assert text.startswith("# Using toyomacro with an AI agent")
    assert read_guide(AGENT_USER_GUIDE) == text


def test_the_version_line_and_the_links_agree():
    text = read_guide()
    stated = re.search(r"Written for: toyomacro v(\d+\.\d+\.\d+)", text).group(1)
    links = re.findall(r"github\.com/[^>\s]+", text)
    assert links and all(re.search(rf"/(?:blob|tree)/v{re.escape(stated)}/", u) for u in links)
    if toyomacro.__version__ != "0.0.0+unknown":
        assert _version(stated) >= _version(toyomacro.__version__)


def test_the_guide_keeps_what_v050_is_about():
    text = read_guide()
    for phrase in ("never fold them into one", "withheld", "not an uncertainty",
                   "changes what\nthe error bars mean", "not** a complete relative sensitivity"):
        assert phrase in text, phrase


def test_the_selection_lesson_keeps_its_direction():
    """The validation found admitted error bars too wide, not too narrow."""
    text = read_guide()
    assert "overstated it by up to 25 %" in text
    assert "optimistic" not in text and "understated" not in text


def test_only_shipped_guides_can_be_read():
    import pytest

    for name in ("../__init__.py", "/etc/hosts", "nope.md"):
        with pytest.raises(ValueError, match="no shipped guide"):
            read_guide(name)


def test_python_m_prints_it():
    out = subprocess.run([sys.executable, "-m", "toyomacro.guides"], capture_output=True,
                         text=True, check=True).stdout
    assert out == read_guide()
