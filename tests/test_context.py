# --------------------------------------------------
# tests/test_context.py
# --------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.context import (
    initialize_context_state,
    learn_context_transition,
    update_context_state,
)


def test_initialize_context_state_provides_expected_defaults() -> None:
    context_state = initialize_context_state()

    assert context_state["x_mean"] == pytest.approx(0.0)
    assert context_state["mean_density_variance"] == pytest.approx(0.0)
    assert context_state["dominant_cluster_id"] == pytest.approx(-1.0)


def test_update_context_state_updates_trend_and_replay_risk() -> None:
    updated = update_context_state(
        previous_context=initialize_context_state(),
        x_value=1.0,
        density_variance=0.4,
        cluster_id=2.0,
        input_energy=0.7,
        self_state={"cluster_stability": 0.5},
        decay=0.8,
    )

    assert updated["x_mean"] == pytest.approx(0.2)
    assert updated["mean_density_variance"] == pytest.approx(0.08)
    assert updated["drift_trend"] == pytest.approx(1.0)
    assert updated["replay_risk"] > 0.0


def test_learn_context_transition_counts_cluster_transitions() -> None:
    memory = learn_context_transition(
        previous_context={"dominant_cluster_id": 1.0},
        current_context={"dominant_cluster_id": 2.0},
        transition_memory={},
    )
    memory = learn_context_transition(
        previous_context={"dominant_cluster_id": 1.0},
        current_context={"dominant_cluster_id": 2.0},
        transition_memory=memory,
    )

    assert memory["1->2"] == pytest.approx(2.0)
