"""Test configuration — force pyqtgraph to use PySide6, and skip
Qt-dependent tests when PySide6 / pytest-qt is unavailable."""

import importlib.util
import os

import pytest

os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"


def _qt_stack_available() -> bool:
    return (
        importlib.util.find_spec("PySide6") is not None
        and importlib.util.find_spec("pytestqt") is not None
    )


def pytest_collection_modifyitems(config, items):
    """Skip GUI/Qt tests when the desktop stack is not installed.

    These tests use the ``qtbot`` fixture from pytest-qt and require
    a working PySide6. Skipping cleanly keeps the test suite green on
    hosts without the (separately developed) desktop stack installed.
    """
    if _qt_stack_available():
        return
    skip_qt = pytest.mark.skip(reason="PySide6 + pytest-qt not installed")
    qt_fixture_names = {"qtbot", "qapp"}
    for item in items:
        fixturenames = getattr(item, "fixturenames", ())
        if qt_fixture_names.intersection(fixturenames):
            item.add_marker(skip_qt)
