# --------------------------------------------------
# src/core/mcm_state.py
# --------------------------------------------------
from __future__ import annotations


# --------------------------------------------------
# Feldgrenzen
# --------------------------------------------------
FIELD_MIN = -3.0
FIELD_MAX = 3.0


# --------------------------------------------------
# Initialisierung
# --------------------------------------------------
def initialize_mcm_state(initial_x: float = 0.0) -> float:
    return clip_to_field_bounds(initial_x)


# --------------------------------------------------
# Bounds
# --------------------------------------------------
def clip_to_field_bounds(
    x_value: float,
    min_x: float = FIELD_MIN,
    max_x: float = FIELD_MAX,
) -> float:
    if x_value < min_x:
        return min_x
    if x_value > max_x:
        return max_x
    return x_value


# --------------------------------------------------
# Zustandsupdate
# --------------------------------------------------
def update_mcm_state(
    x_prev: float,
    neural_input: float,
    replay_input: float,
    regulation_gain: float,
    noise: float,
    dt: float,
) -> float:
    if dt <= 0.0:
        raise ValueError("dt must be > 0.0")

    dx_dt = (
        -regulation_gain * x_prev
        + neural_input
        + replay_input
        + noise
    )
    x_curr = x_prev + (dx_dt * dt)
    return clip_to_field_bounds(x_curr)


# --------------------------------------------------
# Geschwindigkeit
# --------------------------------------------------
def compute_state_velocity(x_prev: float, x_curr: float, dt: float) -> float:
    if dt <= 0.0:
        raise ValueError("dt must be > 0.0")
    return (x_curr - x_prev) / dt