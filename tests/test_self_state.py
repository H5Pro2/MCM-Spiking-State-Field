# --------------------------------------------------
# tests/test_self_state.py
# --------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.self_state import (
    build_self_state,
    build_self_state_labels,
)


# --------------------------------------------------
# Self-State-Vektor
# --------------------------------------------------
def test_build_self_state_builds_expected_state_mapping() -> None:
    self_state = build_self_state(
        x_value=-1.2,
        velocity=0.7,
        density_variance=0.9,
        cluster_stability=0.4,
    )

    assert self_state["x"] == pytest.approx(-1.2)
    assert self_state["abs_x"] == pytest.approx(1.2)
    assert self_state["dx_dt"] == pytest.approx(0.7)
    assert self_state["var_x"] == pytest.approx(0.9)
    assert self_state["cluster_stability"] == pytest.approx(0.4)
    assert self_state["center_distance"] == pytest.approx(1.2)


def test_build_self_state_clamps_negative_density_variance_to_zero() -> None:
    self_state = build_self_state(
        x_value=0.3,
        velocity=0.1,
        density_variance=-0.5,
        cluster_stability=0.8,
    )

    assert self_state["var_x"] == pytest.approx(0.0)


# --------------------------------------------------
# Labels
# --------------------------------------------------
def test_build_self_state_labels_returns_stable_label_set() -> None:
    labels = build_self_state_labels(
        {
            "center_distance": 0.2,
            "dx_dt": 0.1,
            "var_x": 0.2,
        }
    )

    assert labels == ["stable"]


def test_build_self_state_labels_returns_active_excited_and_diffuse_labels() -> None:
    labels = build_self_state_labels(
        {
            "center_distance": 1.0,
            "dx_dt": 0.7,
            "var_x": 0.9,
        }
    )

    assert labels == ["active", "excited", "diffuse"]


def test_build_self_state_labels_returns_stressed_label_for_large_center_distance() -> None:
    labels = build_self_state_labels(
        {
            "center_distance": 1.8,
            "dx_dt": 0.1,
            "var_x": 0.2,
        }
    )

    assert labels == ["stressed"]


# --------------------------------------------------
# Fehlersignale
# --------------------------------------------------
def test_build_self_state_rejects_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="x_value must be finite"):
        build_self_state(
            x_value=float("nan"),
            velocity=0.0,
            density_variance=0.0,
            cluster_stability=1.0,
        )

    with pytest.raises(ValueError, match="velocity must be finite"):
        build_self_state(
            x_value=0.0,
            velocity=float("inf"),
            density_variance=0.0,
            cluster_stability=1.0,
        )

    with pytest.raises(ValueError, match="density_variance must be finite"):
        build_self_state(
            x_value=0.0,
            velocity=0.0,
            density_variance=float("nan"),
            cluster_stability=1.0,
        )

    with pytest.raises(ValueError, match="cluster_stability must be finite"):
        build_self_state(
            x_value=0.0,
            velocity=0.0,
            density_variance=0.0,
            cluster_stability=float("inf"),
        )


def test_build_self_state_labels_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="center_distance must be finite"):
        build_self_state_labels(
            {
                "center_distance": float("nan"),
                "dx_dt": 0.0,
                "var_x": 0.0,
            }
        )

    with pytest.raises(ValueError, match="dx_dt must be finite"):
        build_self_state_labels(
            {
                "center_distance": 0.0,
                "dx_dt": float("inf"),
                "var_x": 0.0,
            }
        )

    with pytest.raises(ValueError, match="var_x must be finite"):
        build_self_state_labels(
            {
                "center_distance": 0.0,
                "dx_dt": 0.0,
                "var_x": float("nan"),
            }
        )