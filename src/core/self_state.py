# --------------------------------------------------
# src/core/self_state.py
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
# Self-State-Vektor
# --------------------------------------------------
def build_self_state(
    x_value: float,
    velocity: float,
    density_variance: float,
    cluster_stability: float,
) -> dict[str, float]:
    x_numeric = _ensure_finite(x_value, "x_value")
    velocity_numeric = _ensure_finite(velocity, "velocity")
    density_variance_numeric = _ensure_finite(density_variance, "density_variance")
    cluster_stability_numeric = _ensure_finite(cluster_stability, "cluster_stability")

    center_distance = abs(x_numeric)

    return {
        "x": x_numeric,
        "abs_x": center_distance,
        "dx_dt": velocity_numeric,
        "var_x": max(0.0, density_variance_numeric),
        "cluster_stability": cluster_stability_numeric,
        "center_distance": center_distance,
    }


# --------------------------------------------------
# Labels
# --------------------------------------------------
def build_self_state_labels(self_state: Mapping[str, float]) -> list[str]:
    center_distance = _ensure_finite(self_state.get("center_distance", 0.0), "center_distance")
    velocity = _ensure_finite(self_state.get("dx_dt", 0.0), "dx_dt")
    variance = _ensure_finite(self_state.get("var_x", 0.0), "var_x")

    labels: list[str] = []

    if center_distance < 0.5:
        labels.append("stable")
    elif center_distance < 1.5:
        labels.append("active")
    else:
        labels.append("stressed")

    if abs(velocity) > 0.5:
        labels.append("excited")

    if variance > 0.8:
        labels.append("diffuse")

    return labels
