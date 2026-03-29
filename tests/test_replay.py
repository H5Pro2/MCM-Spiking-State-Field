# --------------------------------------------------
# tests/test_replay.py
# --------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.replay import (
    compute_replay_gates,
    compute_replay_signal,
    detect_rumination_loop,
    limit_replay_if_needed,
)


# --------------------------------------------------
# Replay-Gates
# --------------------------------------------------
def test_compute_replay_gates_sorts_entries_by_gate_descending() -> None:
    replay_gates = compute_replay_gates(
        memory_bank=(
            {
                "cluster_id": 1.0,
                "x_mean": 0.1,
                "mean_density_variance": 0.2,
                "strength": 1.0,
                "stability": 0.5,
            },
            {
                "cluster_id": 2.0,
                "x_mean": 1.5,
                "mean_density_variance": 0.9,
                "strength": 0.8,
                "stability": 0.4,
            },
            {
                "cluster_id": 3.0,
                "x_mean": 0.2,
                "mean_density_variance": 0.25,
                "strength": 0.9,
                "stability": 0.6,
            },
        ),
        context_state={
            "x_mean": 0.1,
            "mean_density_variance": 0.2,
        },
        regulation_state={
            "replay_gain": 1.0,
        },
    )

    assert len(replay_gates) == 3
    assert replay_gates[0]["cluster_id"] == pytest.approx(1.0)
    assert replay_gates[1]["cluster_id"] == pytest.approx(3.0)
    assert replay_gates[2]["cluster_id"] == pytest.approx(2.0)
    assert replay_gates[0]["gate"] >= replay_gates[1]["gate"]
    assert replay_gates[1]["gate"] >= replay_gates[2]["gate"]


# --------------------------------------------------
# Replay-Signal
# --------------------------------------------------
def test_compute_replay_signal_accumulates_cluster_pull_and_respects_limit() -> None:
    replay_signal = compute_replay_signal(
        memory_bank=(
            {
                "cluster_id": 1.0,
                "x_mean": 1.0,
                "mean_density_variance": 0.0,
                "strength": 1.0,
                "stability": 1.0,
            },
        ),
        context_state={
            "x_mean": 1.0,
            "mean_density_variance": 0.0,
        },
        mcm_state=0.0,
        regulation_state={
            "replay_gain": 1.0,
        },
    )

    assert replay_signal == pytest.approx(1.5)


def test_compute_replay_signal_returns_zero_when_memory_is_empty() -> None:
    replay_signal = compute_replay_signal(
        memory_bank=(),
        context_state={
            "x_mean": 0.0,
            "mean_density_variance": 0.0,
        },
        mcm_state=0.3,
        regulation_state={
            "replay_gain": 1.0,
        },
    )

    assert replay_signal == pytest.approx(0.0)


# --------------------------------------------------
# Schleifenerkennung
# --------------------------------------------------
def test_detect_rumination_loop_detects_same_sign_replay_sequence() -> None:
    detected = detect_rumination_loop(
        replay_history=(0.2, 0.3, 0.4),
        loop_threshold=3,
    )

    assert detected is True


def test_detect_rumination_loop_returns_false_for_sign_change_or_zero() -> None:
    sign_change_detected = detect_rumination_loop(
        replay_history=(0.2, -0.3, 0.4),
        loop_threshold=3,
    )
    zero_detected = detect_rumination_loop(
        replay_history=(0.2, 0.0, 0.4),
        loop_threshold=3,
    )

    assert sign_change_detected is False
    assert zero_detected is False


# --------------------------------------------------
# Replay-Begrenzung
# --------------------------------------------------
def test_limit_replay_if_needed_clamps_to_effective_limit() -> None:
    limited_positive = limit_replay_if_needed(
        replay_signal=5.0,
        regulation_state={
            "replay_gain": 0.5,
        },
    )
    limited_negative = limit_replay_if_needed(
        replay_signal=-5.0,
        regulation_state={
            "replay_gain": 0.5,
        },
    )

    assert limited_positive == pytest.approx(0.75)
    assert limited_negative == pytest.approx(-0.75)


# --------------------------------------------------
# Fehlersignale
# --------------------------------------------------
def test_compute_replay_gates_rejects_negative_replay_gain() -> None:
    with pytest.raises(ValueError, match="replay_gain must be >= 0.0"):
        compute_replay_gates(
            memory_bank=(),
            context_state={
                "x_mean": 0.0,
                "mean_density_variance": 0.0,
            },
            regulation_state={
                "replay_gain": -0.1,
            },
        )


def test_detect_rumination_loop_rejects_too_small_loop_threshold() -> None:
    with pytest.raises(ValueError, match="loop_threshold must be >= 2"):
        detect_rumination_loop(
            replay_history=(0.1, 0.2),
            loop_threshold=1,
        )


def test_limit_replay_if_needed_rejects_negative_replay_gain() -> None:
    with pytest.raises(ValueError, match="replay_gain must be >= 0.0"):
        limit_replay_if_needed(
            replay_signal=1.0,
            regulation_state={
                "replay_gain": -0.1,
            },
        )