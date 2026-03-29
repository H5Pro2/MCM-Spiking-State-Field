# --------------------------------------------------
# tests/test_mcm_state.py
# --------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.mcm_state import (
    FIELD_MAX,
    FIELD_MIN,
    clip_to_field_bounds,
    compute_state_velocity,
    initialize_mcm_state,
    update_mcm_state,
)


# --------------------------------------------------
# Initialisierung
# --------------------------------------------------
def test_initialize_mcm_state_clips_to_bounds() -> None:
    assert initialize_mcm_state(0.0) == 0.0
    assert initialize_mcm_state(10.0) == FIELD_MAX
    assert initialize_mcm_state(-10.0) == FIELD_MIN


# --------------------------------------------------
# Rueckkehr zum Zentrum
# --------------------------------------------------
def test_update_mcm_state_returns_toward_center_without_input() -> None:
    x_prev = 1.0

    x_curr = update_mcm_state(
        x_prev=x_prev,
        neural_input=0.0,
        replay_input=0.0,
        regulation_gain=0.4,
        noise=0.0,
        dt=0.1,
    )
    assert x_curr < x_prev
    assert x_curr > 0.0


# --------------------------------------------------
# Feldgrenzen
# --------------------------------------------------
def test_update_mcm_state_clips_to_field_bounds() -> None:
    high_value = update_mcm_state(
        x_prev=2.9,
        neural_input=10.0,
        replay_input=0.0,
        regulation_gain=0.0,
        noise=0.0,
        dt=0.1,
    )
    low_value = update_mcm_state(
        x_prev=-2.9,
        neural_input=-10.0,
        replay_input=0.0,
        regulation_gain=0.0,
        noise=0.0,
        dt=0.1,
    )

    assert high_value == FIELD_MAX
    assert low_value == FIELD_MIN
    assert clip_to_field_bounds(99.0) == FIELD_MAX
    assert clip_to_field_bounds(-99.0) == FIELD_MIN


# --------------------------------------------------
# Ableitung
# --------------------------------------------------
def test_compute_state_velocity_matches_delta_over_dt() -> None:
    velocity = compute_state_velocity(
        x_prev=0.2,
        x_curr=0.5,
        dt=0.1,
    )

    assert velocity == pytest.approx(3.0)


# --------------------------------------------------
# Fehlersignal
# --------------------------------------------------
def test_update_mcm_state_rejects_non_positive_dt() -> None:
    with pytest.raises(ValueError, match="dt must be > 0.0"):
        update_mcm_state(
            x_prev=0.0,
            neural_input=0.0,
            replay_input=0.0,
            regulation_gain=0.4,
            noise=0.0,
            dt=0.0,
        )

    with pytest.raises(ValueError, match="dt must be > 0.0"):
        compute_state_velocity(
            x_prev=0.0,
            x_curr=0.1,
            dt=-0.1,
        )