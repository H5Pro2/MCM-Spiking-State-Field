# --------------------------------------------------
# src/core/context.py
# --------------------------------------------------
from __future__ import annotations

import math
from typing import Mapping


# --------------------------------------------------
# Defaults
# --------------------------------------------------
DEFAULT_CONTEXT_DECAY = 0.85
DEFAULT_REPLAY_RISK_GAIN = 0.5


# --------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------
def _ensure_finite(value: float, name: str) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


# --------------------------------------------------
# Kontextinitialisierung
# --------------------------------------------------
def initialize_context_state() -> dict[str, float]:
    return {
        "x_mean": 0.0,
        "mean_density_variance": 0.0,
        "drift_trend": 0.0,
        "dominant_cluster_id": -1.0,
        "replay_risk": 0.0,
        "stability_estimate": 1.0,
        "input_energy": 0.0,
    }


# --------------------------------------------------
# Kontextupdate
# --------------------------------------------------
def update_context_state(
    previous_context: Mapping[str, float],
    x_value: float,
    density_variance: float,
    cluster_id: float,
    input_energy: float,
    self_state: Mapping[str, float],
    decay: float = DEFAULT_CONTEXT_DECAY,
    replay_risk_gain: float = DEFAULT_REPLAY_RISK_GAIN,
) -> dict[str, float]:
    decay_value = _ensure_finite(decay, "decay")
    if not 0.0 <= decay_value <= 1.0:
        raise ValueError("decay must be in [0.0, 1.0]")

    replay_risk_gain_value = _ensure_finite(replay_risk_gain, "replay_risk_gain")
    if replay_risk_gain_value < 0.0:
        raise ValueError("replay_risk_gain must be >= 0.0")

    prev = dict(previous_context)
    prev_x = _ensure_finite(prev.get("x_mean", 0.0), "previous x_mean")
    prev_density_variance = _ensure_finite(
        prev.get("mean_density_variance", 0.0),
        "previous mean_density_variance",
    )

    x_numeric = _ensure_finite(x_value, "x_value")
    density_numeric = _ensure_finite(density_variance, "density_variance")
    cluster_numeric = _ensure_finite(cluster_id, "cluster_id")
    input_energy_numeric = _ensure_finite(input_energy, "input_energy")

    cluster_stability = _ensure_finite(
        self_state.get("cluster_stability", 0.0),
        "self_state cluster_stability",
    )

    updated_x_mean = (decay_value * prev_x) + ((1.0 - decay_value) * x_numeric)
    updated_density_variance = (decay_value * prev_density_variance) + (
        (1.0 - decay_value) * density_numeric
    )
    drift_trend = x_numeric - prev_x

    replay_risk = replay_risk_gain_value * (
        abs(updated_x_mean)
        + updated_density_variance
        + max(0.0, 1.0 - cluster_stability)
    )

    return {
        "x_mean": updated_x_mean,
        "mean_density_variance": updated_density_variance,
        "drift_trend": drift_trend,
        "dominant_cluster_id": cluster_numeric,
        "replay_risk": replay_risk,
        "stability_estimate": cluster_stability,
        "input_energy": max(0.0, input_energy_numeric),
    }


# --------------------------------------------------
# Kontextlernen
# --------------------------------------------------
def learn_context_transition(
    previous_context: Mapping[str, float],
    current_context: Mapping[str, float],
    transition_memory: Mapping[str, float],
) -> dict[str, float]:
    prev_cluster = int(_ensure_finite(previous_context.get("dominant_cluster_id", -1.0), "prev cluster"))
    curr_cluster = int(_ensure_finite(current_context.get("dominant_cluster_id", -1.0), "curr cluster"))

    transition_key = f"{prev_cluster}->{curr_cluster}"
    updated_memory = dict(transition_memory)
    current_count = _ensure_finite(updated_memory.get(transition_key, 0.0), "transition count")
    updated_memory[transition_key] = current_count + 1.0
    return updated_memory
