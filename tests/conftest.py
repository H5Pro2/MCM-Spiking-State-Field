# --------------------------------------------------
# tests/conftest.py
# --------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.debug_reader import dbr_debug


@pytest.fixture(autouse=True)
def debug_test_structure(request: pytest.FixtureRequest):
    """Schreibt eine einheitliche Print-/Debug-Struktur je Testlauf."""
    nodeid = request.node.nodeid
    start_line = f"[TEST START] {nodeid}"
    end_line = f"[TEST END] {nodeid}"

    print(start_line)
    dbr_debug(start_line, txt="tests_debug.txt")

    yield

    print(end_line)
    dbr_debug(end_line, txt="tests_debug.txt")
