"""Guides shipped with the package, readable from the installed version.

``python -m toyomacro.guides`` prints the AI-agent user guide for the
version you have installed, which is the one to give an agent — the
repository's main branch may describe a newer version.
"""

from __future__ import annotations

from importlib.resources import files

AGENT_USER_GUIDE = "AGENT_USER_GUIDE.md"

__all__ = ["AGENT_USER_GUIDE", "GUIDES", "read_guide"]


#: The guides this package ships; ``read_guide`` reads only these.
GUIDES = (AGENT_USER_GUIDE,)


def read_guide(name: str = AGENT_USER_GUIDE) -> str:
    """The text of a shipped guide (default: the AI-agent user guide).

    Raises:
        ValueError: ``name`` is not one of :data:`GUIDES`.
    """
    if name not in GUIDES:
        raise ValueError(f"no shipped guide {name!r}; one of {GUIDES}")
    return files(__package__).joinpath(name).read_text(encoding="utf-8")
