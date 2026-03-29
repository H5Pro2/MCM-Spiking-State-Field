# --------------------------------------------------
# tests/test_neural_core.py
# --------------------------------------------------
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.neural_core import (
    build_neural_core,
    compute_firing_rates,
    decode_neural_state,
    step_neural_core,
)


# --------------------------------------------------
# Neural-Core-Aufbau
# --------------------------------------------------
def test_build_neural_core_uses_defaults_and_custom_config() -> None:
    default_state = build_neural_core()
    custom_state = build_neural_core(
        {
            "input_weights": (0.5, 0.6, 0.7, 0.8, 0.9),
            "recurrent_gain": 0.4,
            "bias": 0.1,
            "threshold": 0.2,
            "rate_scale": 2.0,
        }
    )

    assert default_state["input_weights"] == pytest.approx((1.0, 1.0, 1.0, 1.0, 1.0))
    assert default_state["recurrent_gain"] == pytest.approx(0.25)
    assert default_state["bias"] == pytest.approx(0.0)
    assert default_state["threshold"] == pytest.approx(0.5)
    assert default_state["rate_scale"] == pytest.approx(1.0)
    assert default_state["activity"] == pytest.approx(0.0)
    assert default_state["spike"] == pytest.approx(0.0)

    assert custom_state["input_weights"] == pytest.approx((0.5, 0.6, 0.7, 0.8, 0.9))
    assert custom_state["recurrent_gain"] == pytest.approx(0.4)
    assert custom_state["bias"] == pytest.approx(0.1)
    assert custom_state["threshold"] == pytest.approx(0.2)
    assert custom_state["rate_scale"] == pytest.approx(2.0)

# --------------------------------------------------
# Neural-Core-Schritt
# --------------------------------------------------
def test_step_neural_core_updates_activity_and_spike_from_total_drive() -> None:
    neural_state = build_neural_core(
        {
            "input_weights": (1.0, 1.0, 1.0, 1.0, 1.0),
            "recurrent_gain": 0.5,
            "bias": 0.1,
            "threshold": 0.6,
        }
    )
    neural_state["activity"] = 0.2

    updated_state = step_neural_core(
        input_vector=(0.1, 0.2, 0.0, 0.0, 0.0),
        recurrent_feedback=0.1,
        neural_state=neural_state,
    )

    expected_total_drive = 0.3 + (0.5 * 0.2) + 0.1 + 0.1
    expected_activity = math.tanh(expected_total_drive)

    assert updated_state["activity"] == pytest.approx(expected_activity)
    assert updated_state["spike"] == pytest.approx(0.0)


def test_step_neural_core_keeps_spike_zero_below_threshold() -> None:
    neural_state = build_neural_core(
        {
            "input_weights": (1.0, 0.0, 0.0, 0.0, 0.0),
            "recurrent_gain": 0.0,
            "bias": 0.0,
            "threshold": 0.8,
        }
    )

    updated_state = step_neural_core(
        input_vector=(0.1, 0.0, 0.0, 0.0, 0.0),
        recurrent_feedback=0.0,
        neural_state=neural_state,
    )

    assert updated_state["activity"] == pytest.approx(math.tanh(0.1))
    assert updated_state["spike"] == pytest.approx(0.0)

# --------------------------------------------------
# Readout
# --------------------------------------------------
def test_decode_neural_state_and_compute_firing_rates_behave_as_expected() -> None:
    decoded = decode_neural_state(0.75)
    firing_rate = compute_firing_rates(
        spikes=(1.0, 0.0, 1.0, 1.0),
        dt=0.5,
    )

    assert decoded == pytest.approx(0.75)
    assert firing_rate == pytest.approx(1.5)


def test_compute_firing_rates_returns_zero_for_empty_spikes() -> None:
    firing_rate = compute_firing_rates(
        spikes=(),
        dt=0.1,
    )

    assert firing_rate == pytest.approx(0.0)


# --------------------------------------------------
# Fehlersignale
# --------------------------------------------------
def test_build_neural_core_rejects_invalid_input_weights_length() -> None:
    with pytest.raises(ValueError, match="input_weights must contain 5 values"):
        build_neural_core(
            {
                "input_weights": (1.0, 1.0, 1.0),
            }
        )


def test_step_neural_core_rejects_invalid_input_vector_length() -> None:
    neural_state = build_neural_core()

    with pytest.raises(ValueError, match="input_vector must contain 5 channels"):
        step_neural_core(
            input_vector=(1.0, 2.0, 3.0),
            recurrent_feedback=0.0,
            neural_state=neural_state,
        )


def test_step_neural_core_rejects_non_finite_recurrent_feedback() -> None:
    neural_state = build_neural_core()

    with pytest.raises(ValueError, match="recurrent_feedback must be finite"):
        step_neural_core(
            input_vector=(0.0, 0.0, 0.0, 0.0, 0.0),
            recurrent_feedback=float("inf"),
            neural_state=neural_state,
        )


def test_decode_neural_state_rejects_non_finite_activity() -> None:
    with pytest.raises(ValueError, match="activity must be finite"):
        decode_neural_state(float("nan"))


def test_compute_firing_rates_rejects_non_positive_dt() -> None:
    with pytest.raises(ValueError, match="dt must be > 0.0"):
        compute_firing_rates(
            spikes=(1.0, 0.0),
            dt=0.0,
        )


def test_compute_firing_rates_rejects_non_finite_spikes() -> None:
    with pytest.raises(ValueError, match="spikes must be finite"):
        compute_firing_rates(
            spikes=(1.0, float("nan")),
            dt=0.1,
        )

