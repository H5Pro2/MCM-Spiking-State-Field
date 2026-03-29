# --------------------------------------------------
# tests/test_regulation.py
# --------------------------------------------------
from __future__ import annotations

import pytest

from src.core.meta_regulation import build_meta_regulation_state
from src.core.regulation import derive_regulation_parameters, regulate_replay_signal
from src.core.reflection import build_reflection_state
from src.core.self_state import build_self_state, build_self_state_labels


def test_build_self_state_and_labels() -> None:
    self_state = build_self_state(
        x_value=1.8,
        velocity=0.7,
        density_variance=0.9,
        cluster_stability=0.3,
    )
    labels = build_self_state_labels(self_state)

    assert self_state["center_distance"] == pytest.approx(1.8)
    assert "stressed" in labels
    assert "excited" in labels
    assert "diffuse" in labels


def test_meta_to_regulation_pipeline_reduces_replay_on_pressure() -> None:
    self_state = build_self_state(
        x_value=1.4,
        velocity=0.02,
        density_variance=0.8,
        cluster_stability=0.4,
    )
    reflection_state = build_reflection_state(
        x_value=1.4,
        velocity=0.02,
        context_state={"drift_trend": 0.01, "replay_risk": 0.9},
        previous_self_state={"center_distance": 1.1},
        matched_cluster_ids=(),
    )
    meta_state = build_meta_regulation_state(
        self_state=self_state,
        reflection_state=reflection_state,
        context_state={"replay_risk": 0.9, "stability_estimate": 0.4},
    )

    regulation_state = derive_regulation_parameters(meta_state)

    assert regulation_state["k_reg"] > 0.4
    assert regulation_state["replay_gain"] < 1.0


def test_regulate_replay_signal_scales_with_replay_gain() -> None:
    regulated = regulate_replay_signal(
        replay_signal=0.75,
        regulation_state={"replay_gain": 0.2},
    )

    assert regulated == pytest.approx(0.15)
