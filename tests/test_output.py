# --------------------------------------------------
# tests/test_output.py
# --------------------------------------------------
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.output import (
    build_output_state,
    build_readout_labels,
    export_output_snapshot,
    serialize_output_state,
)


# --------------------------------------------------
# Labels
# --------------------------------------------------
def test_build_readout_labels_returns_positive_and_high_activity_labels() -> None:
    labels = build_readout_labels(
        mcm_state=0.8,
        neural_activity=0.7,
    )

    assert labels == ["positive_shift", "high_activity"]


def test_build_readout_labels_returns_negative_and_low_activity_labels() -> None:
    labels = build_readout_labels(
        mcm_state=-0.8,
        neural_activity=0.2,
    )

    assert labels == ["negative_shift", "low_activity"]


def test_build_readout_labels_returns_centered_label_in_middle_range() -> None:
    labels = build_readout_labels(
        mcm_state=0.1,
        neural_activity=0.49,
    )

    assert labels == ["centered", "low_activity"]


# --------------------------------------------------
# Output-State
# --------------------------------------------------
def test_build_output_state_builds_expected_output_mapping() -> None:
    output_state = build_output_state(
        neural_state={
            "activity": 0.75,
            "spike": 1.0,
        },
        mcm_state=0.9,
    )

    assert output_state["mcm_state"] == pytest.approx(0.9)
    assert output_state["neural_activity"] == pytest.approx(0.75)
    assert output_state["spike"] == pytest.approx(1.0)
    assert output_state["labels"] == ["positive_shift", "high_activity"]


def test_build_output_state_uses_default_values_when_neural_state_keys_are_missing() -> None:
    output_state = build_output_state(
        neural_state={},
        mcm_state=0.0,
    )

    assert output_state["mcm_state"] == pytest.approx(0.0)
    assert output_state["neural_activity"] == pytest.approx(0.0)
    assert output_state["spike"] == pytest.approx(0.0)
    assert output_state["labels"] == ["centered", "low_activity"]


# --------------------------------------------------
# Serialisierung
# --------------------------------------------------
def test_serialize_output_state_returns_json_string() -> None:
    serialized = serialize_output_state(
        {
            "mcm_state": 0.5,
            "neural_activity": 0.6,
            "spike": 1.0,
            "labels": ["centered", "high_activity"],
        }
    )

    parsed = json.loads(serialized)

    assert parsed["mcm_state"] == pytest.approx(0.5)
    assert parsed["neural_activity"] == pytest.approx(0.6)
    assert parsed["spike"] == pytest.approx(1.0)
    assert parsed["labels"] == ["centered", "high_activity"]


# --------------------------------------------------
# Export
# --------------------------------------------------
def test_export_output_snapshot_writes_serialized_file(tmp_path: Path) -> None:
    output_state = {
        "mcm_state": -0.5,
        "neural_activity": 0.1,
        "spike": 0.0,
        "labels": ["centered", "low_activity"],
    }
    output_path = tmp_path / "snapshots" / "output.json"

    written_path = export_output_snapshot(
        output_state=output_state,
        path=output_path,
    )

    assert written_path == output_path
    assert written_path.exists() is True

    parsed = json.loads(written_path.read_text(encoding="utf-8"))

    assert parsed["mcm_state"] == pytest.approx(-0.5)
    assert parsed["neural_activity"] == pytest.approx(0.1)
    assert parsed["spike"] == pytest.approx(0.0)
    assert parsed["labels"] == ["centered", "low_activity"]