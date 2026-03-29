# --------------------------------------------------
# tests/test_reflection.py
# --------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.reflection import build_reflection_state


# --------------------------------------------------
# Reflexionszustand
# --------------------------------------------------
def test_build_reflection_state_detects_drift_loop_risk_and_known_pattern() -> None:
    reflection_state = build_reflection_state(
        x_value=1.2,
        velocity=0.02,
        context_state={
            "drift_trend": 0.1,
            "replay_risk": 0.8,
        },
        previous_self_state={
            "center_distance": 0.8,
        },
        matched_cluster_ids=(1.0, 2.0),
    )

    assert reflection_state["drift_detected"] == pytest.approx(1.0)
    assert reflection_state["return_toward_center"] == pytest.approx(0.0)
    assert reflection_state["loop_risk"] == pytest.approx(1.0)
    assert reflection_state["known_pattern"] == pytest.approx(1.0)
    assert reflection_state["center_distance"] == pytest.approx(1.2)
    assert reflection_state["context_alignment"] == pytest.approx(1.0 / (1.0 + abs(0.1 - 0.02)))


def test_build_reflection_state_detects_return_toward_center_without_loop_risk() -> None:
    reflection_state = build_reflection_state(
        x_value=0.4,
        velocity=0.2,
        context_state={
            "drift_trend": 0.3,
            "replay_risk": 0.4,
        },
        previous_self_state={
            "center_distance": 1.0,
        },
        matched_cluster_ids=(),
    )

    assert reflection_state["drift_detected"] == pytest.approx(0.0)
    assert reflection_state["return_toward_center"] == pytest.approx(1.0)
    assert reflection_state["loop_risk"] == pytest.approx(0.0)
    assert reflection_state["known_pattern"] == pytest.approx(0.0)
    assert reflection_state["center_distance"] == pytest.approx(0.4)
    assert reflection_state["context_alignment"] == pytest.approx(1.0 / (1.0 + abs(0.3 - 0.2)))


# --------------------------------------------------
# Fehlersignale
# --------------------------------------------------
def test_build_reflection_state_rejects_non_finite_x_value() -> None:
    with pytest.raises(ValueError, match="x_value must be finite"):
        build_reflection_state(
            x_value=float("nan"),
            velocity=0.0,
            context_state={
                "drift_trend": 0.0,
                "replay_risk": 0.0,
            },
            previous_self_state={
                "center_distance": 0.0,
            },
            matched_cluster_ids=(),
        )


def test_build_reflection_state_rejects_non_finite_velocity() -> None:
    with pytest.raises(ValueError, match="velocity must be finite"):
        build_reflection_state(
            x_value=0.0,
            velocity=float("inf"),
            context_state={
                "drift_trend": 0.0,
                "replay_risk": 0.0,
            },
            previous_self_state={
                "center_distance": 0.0,
            },
            matched_cluster_ids=(),
        )


def test_build_reflection_state_rejects_non_finite_context_values() -> None:
    with pytest.raises(ValueError, match="context drift_trend must be finite"):
        build_reflection_state(
            x_value=0.0,
            velocity=0.0,
            context_state={
                "drift_trend": float("nan"),
                "replay_risk": 0.0,
            },
            previous_self_state={
                "center_distance": 0.0,
            },
            matched_cluster_ids=(),
        )

    with pytest.raises(ValueError, match="context replay_risk must be finite"):
        build_reflection_state(
            x_value=0.0,
            velocity=0.0,
            context_state={
                "drift_trend": 0.0,
                "replay_risk": float("nan"),
            },
            previous_self_state={
                "center_distance": 0.0,
            },
            matched_cluster_ids=(),
        )


def test_build_reflection_state_rejects_non_finite_previous_center_distance() -> None:
    with pytest.raises(ValueError, match="previous_self_state center_distance must be finite"):
        build_reflection_state(
            x_value=0.0,
            velocity=0.0,
            context_state={
                "drift_trend": 0.0,
                "replay_risk": 0.0,
            },
            previous_self_state={
                "center_distance": float("nan"),
            },
            matched_cluster_ids=(),
        )