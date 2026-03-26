# --------------------------------------------------
# src/core/replay.py
# --------------------------------------------------
from __future__ import annotations

import math
from typing import Mapping, Sequence


# --------------------------------------------------
# Defaults
# --------------------------------------------------
DEFAULT_LOOP_THRESHOLD = 3
DEFAULT_MAX_REPLAY_SIGNAL = 1.5


# --------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------
def _ensure_finite(value: float, name: str) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


# --------------------------------------------------
# Replay-Gates
# --------------------------------------------------
def compute_replay_gates(
    memory_bank: Sequence[Mapping[str, float]],
    context_state: Mapping[str, float],
    regulation_state: Mapping[str, float],
) -> list[dict[str, float]]:
    context_x_mean = _ensure_finite(context_state.get("x_mean", 0.0), "context x_mean")
    context_density_variance = _ensure_finite(
        context_state.get("mean_density_variance", 0.0),
        "context mean_density_variance",
    )
    replay_gain = _ensure_finite(regulation_state.get("replay_gain", 1.0), "replay_gain")
    if replay_gain < 0.0:
        raise ValueError("replay_gain must be >= 0.0")

    replay_gates: list[dict[str, float]] = []

    for memory_entry in memory_bank:
        entry = dict(memory_entry)

        cluster_id = _ensure_finite(entry.get("cluster_id", -1.0), "cluster_id")
        x_mean = _ensure_finite(entry.get("x_mean", 0.0), "x_mean")
        density_variance = _ensure_finite(
            entry.get("mean_density_variance", 0.0),
            "mean_density_variance",
        )
        strength = _ensure_finite(entry.get("strength", 0.0), "strength")
        stability = _ensure_finite(entry.get("stability", 0.0), "stability")

        distance = abs(context_x_mean - x_mean) + abs(context_density_variance - density_variance)
        compatibility = 1.0 / (1.0 + distance)
        gate_value = replay_gain * compatibility * (strength + stability)

        replay_gates.append(
            {
                "cluster_id": cluster_id,
                "mu": x_mean,
                "gate": gate_value,
                "strength": strength,
                "stability": stability,
            }
        )

    replay_gates.sort(key=lambda item: item["gate"], reverse=True)
    return replay_gates


# --------------------------------------------------
# Replay-Signal
# --------------------------------------------------
def compute_replay_signal(
    memory_bank: Sequence[Mapping[str, float]],
    context_state: Mapping[str, float],
    mcm_state: float,
    regulation_state: Mapping[str, float] | None = None,
) -> float:
    regulation = dict(regulation_state or {})
    replay_gates = compute_replay_gates(
        memory_bank=memory_bank,
        context_state=context_state,
        regulation_state=regulation,
    )

    x_value = _ensure_finite(mcm_state, "mcm_state")
    replay_signal = 0.0

    for gate_entry in replay_gates:
        cluster_center = _ensure_finite(gate_entry["mu"], "mu")
        gate_value = _ensure_finite(gate_entry["gate"], "gate")
        replay_signal += gate_value * (cluster_center - x_value)

    return limit_replay_if_needed(
        replay_signal=replay_signal,
        regulation_state=regulation,
    )


# --------------------------------------------------
# Schleifenerkennung
# --------------------------------------------------
def detect_rumination_loop(
    replay_history: Sequence[float],
    loop_threshold: int = DEFAULT_LOOP_THRESHOLD,
) -> bool:
    if loop_threshold < 2:
        raise ValueError("loop_threshold must be >= 2")

    history_values = tuple(_ensure_finite(value, "replay_history value") for value in replay_history)
    if len(history_values) < loop_threshold:
        return False

    recent_values = history_values[-loop_threshold:]
    sign_reference = 0.0

    for value in recent_values:
        if abs(value) < 1e-9:
            return False
        current_sign = 1.0 if value > 0.0 else -1.0
        if sign_reference == 0.0:
            sign_reference = current_sign
            continue
        if current_sign != sign_reference:
            return False

    mean_abs_value = sum(abs(value) for value in recent_values) / len(recent_values)
    return mean_abs_value > 0.05


# --------------------------------------------------
# Replay-Begrenzung
# --------------------------------------------------
def limit_replay_if_needed(
    replay_signal: float,
    regulation_state: Mapping[str, float],
    max_replay_signal: float = DEFAULT_MAX_REPLAY_SIGNAL,
) -> float:
    replay_value = _ensure_finite(replay_signal, "replay_signal")
    replay_gain = _ensure_finite(regulation_state.get("replay_gain", 1.0), "replay_gain")
    if replay_gain < 0.0:
        raise ValueError("replay_gain must be >= 0.0")

    effective_limit = max_replay_signal * max(replay_gain, 1e-9)

    if replay_value > effective_limit:
        return effective_limit
    if replay_value < -effective_limit:
        return -effective_limit
    return replay_value