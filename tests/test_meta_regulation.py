# --------------------------------------------------
# tests/test_meta_regulation.py
# --------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.meta_regulation import build_meta_regulation_state


# --------------------------------------------------
# Meta-Regulation
# --------------------------------------------------
def test_build_meta_regulation_state_computes_expected_control_values() -> None:
    meta_state = build_meta_regulation_state(
        self_state={
            "center_distance": 1.2,
            "var_x": 0.8,
        },
        reflection_state={
            "loop_risk": 1.0,
            "drift_detected": 1.0,
        },
        context_state={
            "replay_risk": 0.6,
            "stability_estimate": 0.4,
        },
    )

    assert meta_state["overload_tendency"] == pytest.approx(2.3)
    assert meta_state["control_pressure"] == pytest.approx(2.6)
    assert meta_state["integration_capacity"] == pytest.approx(0.54)
    assert meta_state["protection_bias"] == pytest.approx(2.0)


def test_build_meta_regulation_state_uses_default_values_for_missing_keys() -> None:
    meta_state = build_meta_regulation_state(
        self_state={},
        reflection_state={},
        context_state={},
    )

    assert meta_state["overload_tendency"] == pytest.approx(0.0)
    assert meta_state["control_pressure"] == pytest.approx(0.0)
    assert meta_state["integration_capacity"] == pytest.approx(1.0)
    assert meta_state["protection_bias"] == pytest.approx(1.0)


# --------------------------------------------------
# Fehlersignale
# --------------------------------------------------
def test_build_meta_regulation_state_rejects_non_finite_self_state_values() -> None:
    with pytest.raises(ValueError, match="center_distance must be finite"):
        build_meta_regulation_state(
            self_state={
                "center_distance": float("nan"),
                "var_x": 0.0,
            },
            reflection_state={},
            context_state={},
        )

    with pytest.raises(ValueError, match="var_x must be finite"):
        build_meta_regulation_state(
            self_state={
                "center_distance": 0.0,
                "var_x": float("inf"),
            },
            reflection_state={},
            context_state={},
        )


def test_build_meta_regulation_state_rejects_non_finite_reflection_values() -> None:
    with pytest.raises(ValueError, match="loop_risk must be finite"):
        build_meta_regulation_state(
            self_state={},
            reflection_state={
                "loop_risk": float("nan"),
                "drift_detected": 0.0,
            },
            context_state={},
        )

    with pytest.raises(ValueError, match="drift_detected must be finite"):
        build_meta_regulation_state(
            self_state={},
            reflection_state={
                "loop_risk": 0.0,
                "drift_detected": float("inf"),
            },
            context_state={},
        )


def test_build_meta_regulation_state_rejects_non_finite_context_values() -> None:
    with pytest.raises(ValueError, match="replay_risk must be finite"):
        build_meta_regulation_state(
            self_state={},
            reflection_state={},
            context_state={
                "replay_risk": float("nan"),
                "stability_estimate": 1.0,
            },
        )

    with pytest.raises(ValueError, match="stability_estimate must be finite"):
        build_meta_regulation_state(
            self_state={},
            reflection_state={},
            context_state={
                "replay_risk": 0.0,
                "stability_estimate": float("inf"),
            },
        )