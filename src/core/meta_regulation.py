# --------------------------------------------------
# src/core/meta_regulation.py
# --------------------------------------------------
from __future__ import annotations

import math
from typing import Mapping


# --------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------
def _ensure_finite(value: float, name: str) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


# --------------------------------------------------
# Meta-Regulation
# --------------------------------------------------
def build_meta_regulation_state(
    self_state: Mapping[str, float],
    reflection_state: Mapping[str, float],
    context_state: Mapping[str, float],
) -> dict[str, float]:
    center_distance = _ensure_finite(self_state.get("center_distance", 0.0), "center_distance")
    variance = _ensure_finite(self_state.get("var_x", 0.0), "var_x")

    loop_risk = _ensure_finite(reflection_state.get("loop_risk", 0.0), "loop_risk")
    drift_detected = _ensure_finite(reflection_state.get("drift_detected", 0.0), "drift_detected")

    replay_risk = _ensure_finite(context_state.get("replay_risk", 0.0), "replay_risk")
    stability_estimate = _ensure_finite(context_state.get("stability_estimate", 1.0), "stability_estimate")

    overload_tendency = center_distance + variance + (0.5 * replay_risk)
    control_pressure = max(0.0, drift_detected + loop_risk + max(0.0, 1.0 - stability_estimate))

    return {
        "overload_tendency": overload_tendency,
        "control_pressure": control_pressure,
        "integration_capacity": max(0.1, 1.0 - min(1.0, overload_tendency * 0.2)),
        "protection_bias": min(2.0, 1.0 + control_pressure * 0.5),
    }