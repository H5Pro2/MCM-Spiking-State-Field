# --------------------------------------------------
# src/core/neural_core.py
# --------------------------------------------------
from __future__ import annotations

import math
from typing import Mapping, Sequence


# --------------------------------------------------
# Defaults
# --------------------------------------------------
DEFAULT_INPUT_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0)
DEFAULT_RECURRENT_GAIN = 0.25
DEFAULT_BIAS = 0.0
DEFAULT_THRESHOLD = 0.5
DEFAULT_RATE_SCALE = 1.0


# --------------------------------------------------
# Validierung
# --------------------------------------------------
def _validate_input_vector(input_vector: Sequence[float]) -> tuple[float, float, float, float, float]:
    if len(input_vector) != 5:
        raise ValueError("input_vector must contain 5 channels")

    values = tuple(float(value) for value in input_vector)
    for value in values:
        if not math.isfinite(value):
            raise ValueError("input_vector values must be finite")

    return values


# --------------------------------------------------
# Neural-Core-Aufbau
# --------------------------------------------------
def build_neural_core(config: Mapping[str, object] | None = None) -> dict[str, object]:
    config_map = dict(config or {})

    raw_weights = config_map.get("input_weights", DEFAULT_INPUT_WEIGHTS)
    input_weights = tuple(float(value) for value in raw_weights)
    if len(input_weights) != 5:
        raise ValueError("input_weights must contain 5 values")

    for value in input_weights:
        if not math.isfinite(value):
            raise ValueError("input_weights must be finite")

    recurrent_gain = float(config_map.get("recurrent_gain", DEFAULT_RECURRENT_GAIN))
    bias = float(config_map.get("bias", DEFAULT_BIAS))
    threshold = float(config_map.get("threshold", DEFAULT_THRESHOLD))
    rate_scale = float(config_map.get("rate_scale", DEFAULT_RATE_SCALE))

    for value in (recurrent_gain, bias, threshold, rate_scale):
        if not math.isfinite(value):
            raise ValueError("neural core config values must be finite")

    return {
        "input_weights": input_weights,
        "recurrent_gain": recurrent_gain,
        "bias": bias,
        "threshold": threshold,
        "rate_scale": rate_scale,
        "activity": 0.0,
        "spike": 0.0,
    }


# --------------------------------------------------
# Neural-Core-Schritt
# --------------------------------------------------
def step_neural_core(
    input_vector: Sequence[float],
    recurrent_feedback: float,
    neural_state: Mapping[str, object],
) -> dict[str, object]:
    input_values = _validate_input_vector(input_vector)

    recurrent_feedback_value = float(recurrent_feedback)
    if not math.isfinite(recurrent_feedback_value):
        raise ValueError("recurrent_feedback must be finite")

    state = dict(neural_state)

    input_weights = tuple(float(value) for value in state["input_weights"])
    recurrent_gain = float(state["recurrent_gain"])
    bias = float(state["bias"])
    threshold = float(state["threshold"])
    previous_activity = float(state.get("activity", 0.0))

    weighted_drive = sum(
        weight * value
        for weight, value in zip(input_weights, input_values)
    )
    total_drive = (
        weighted_drive
        + (recurrent_gain * previous_activity)
        + recurrent_feedback_value
        + bias
    )

    activity = math.tanh(total_drive)
    spike = 1.0 if activity >= threshold else 0.0

    state["activity"] = activity
    state["spike"] = spike
    return state


# --------------------------------------------------
# Readout
# --------------------------------------------------
def decode_neural_state(activity: float) -> float:
    activity_value = float(activity)
    if not math.isfinite(activity_value):
        raise ValueError("activity must be finite")
    return activity_value


# --------------------------------------------------
# Firing-Rate
# --------------------------------------------------
def compute_firing_rates(spikes: Sequence[float], dt: float) -> float:
    dt_value = float(dt)
    if dt_value <= 0.0:
        raise ValueError("dt must be > 0.0")

    spike_values = tuple(float(value) for value in spikes)
    for value in spike_values:
        if not math.isfinite(value):
            raise ValueError("spikes must be finite")

    if not spike_values:
        return 0.0

    return sum(spike_values) / (len(spike_values) * dt_value)