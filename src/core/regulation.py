# --------------------------------------------------
# src/core/regulation.py
# --------------------------------------------------
from __future__ import annotations

import math
from typing import Mapping


# --------------------------------------------------
# Defaults
# --------------------------------------------------
DEFAULT_BASE_REGULATION_GAIN = 0.4
DEFAULT_BASE_REPLAY_GAIN = 1.0
DEFAULT_BASE_INPUT_GAIN = 1.0
DEFAULT_BASE_NOISE_GAIN = 1.0


# --------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------
def _ensure_finite(value: float, name: str) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


def _clamp(value: float, min_value: float, max_value: float) -> float:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


# --------------------------------------------------
# Reglerableitung
# --------------------------------------------------
def derive_regulation_parameters(
    meta_state: Mapping[str, float],
    base_regulation_gain: float = DEFAULT_BASE_REGULATION_GAIN,
    base_replay_gain: float = DEFAULT_BASE_REPLAY_GAIN,
    base_input_gain: float = DEFAULT_BASE_INPUT_GAIN,
    base_noise_gain: float = DEFAULT_BASE_NOISE_GAIN,
) -> dict[str, float]:
    overload_tendency = _ensure_finite(meta_state.get("overload_tendency", 0.0), "overload_tendency")
    control_pressure = _ensure_finite(meta_state.get("control_pressure", 0.0), "control_pressure")
    integration_capacity = _ensure_finite(meta_state.get("integration_capacity", 1.0), "integration_capacity")
    protection_bias = _ensure_finite(meta_state.get("protection_bias", 1.0), "protection_bias")

    k_reg = _clamp(
        _ensure_finite(base_regulation_gain, "base_regulation_gain") + (0.08 * overload_tendency) + (0.05 * control_pressure),
        0.05,
        2.5,
    )
    replay_gain = _clamp(
        _ensure_finite(base_replay_gain, "base_replay_gain") * (1.0 / (1.0 + control_pressure * 0.5)) * integration_capacity,
        0.0,
        2.0,
    )
    input_gain = _clamp(
        _ensure_finite(base_input_gain, "base_input_gain") * (0.8 + (0.2 * integration_capacity)),
        0.2,
        2.0,
    )
    noise_gain = _clamp(
        _ensure_finite(base_noise_gain, "base_noise_gain") * (1.0 / protection_bias),
        0.0,
        2.0,
    )

    return {
        "k_reg": k_reg,
        "replay_gain": replay_gain,
        "input_gain": input_gain,
        "noise_gain": noise_gain,
    }


# --------------------------------------------------
# Replay-Regelung
# --------------------------------------------------
def regulate_replay_signal(replay_signal: float, regulation_state: Mapping[str, float]) -> float:
    replay_signal_value = _ensure_finite(replay_signal, "replay_signal")
    replay_gain = _ensure_finite(regulation_state.get("replay_gain", 1.0), "replay_gain")
    return replay_signal_value * max(0.0, replay_gain)