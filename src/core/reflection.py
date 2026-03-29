# --------------------------------------------------
# src/core/reflection.py
# --------------------------------------------------
from __future__ import annotations

import math
from typing import Mapping, Sequence


# --------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------
def _ensure_finite(value: float, name: str) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


# --------------------------------------------------
# Reflexionszustand
# --------------------------------------------------
def build_reflection_state(
    x_value: float,
    velocity: float,
    context_state: Mapping[str, float],
    previous_self_state: Mapping[str, float],
    matched_cluster_ids: Sequence[float],
) -> dict[str, float]:
    x_numeric = _ensure_finite(x_value, "x_value")
    velocity_numeric = _ensure_finite(velocity, "velocity")

    context_drift = _ensure_finite(context_state.get("drift_trend", 0.0), "context drift_trend")
    replay_risk = _ensure_finite(context_state.get("replay_risk", 0.0), "context replay_risk")

    prev_center_distance = _ensure_finite(
        previous_self_state.get("center_distance", abs(x_numeric)),
        "previous_self_state center_distance",
    )
    center_distance = abs(x_numeric)

    drift_detected = 1.0 if center_distance > prev_center_distance + 1e-9 else 0.0
    return_toward_center = 1.0 if center_distance < prev_center_distance - 1e-9 else 0.0
    loop_risk = 1.0 if replay_risk > 0.5 and abs(velocity_numeric) < 0.05 else 0.0
    known_pattern = 1.0 if len(tuple(matched_cluster_ids)) > 0 else 0.0

    return {
        "drift_detected": drift_detected,
        "return_toward_center": return_toward_center,
        "loop_risk": loop_risk,
        "known_pattern": known_pattern,
        "context_alignment": 1.0 / (1.0 + abs(context_drift - velocity_numeric)),
        "center_distance": center_distance,
    }
