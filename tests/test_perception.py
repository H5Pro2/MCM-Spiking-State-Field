# --------------------------------------------------
# tests/test_perception.py
# --------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.perception import (
    apply_input_gain,
    build_input_vector,
    encode_perception,
    normalize_channels,
)


# --------------------------------------------------
# Normalisierung
# --------------------------------------------------
def test_normalize_channels_fills_missing_channels_with_zero() -> None:
    normalized = normalize_channels(
        {
            "valence": 0.5,
            "novelty": 0.2,
        }
    )

    assert normalized["valence"] == pytest.approx(0.5)
    assert normalized["novelty"] == pytest.approx(0.2)
    assert normalized["relevance"] == pytest.approx(0.0)
    assert normalized["uncertainty"] == pytest.approx(0.0)
    assert normalized["social_salience"] == pytest.approx(0.0)

# --------------------------------------------------
# Inputvektor
# --------------------------------------------------
def test_build_input_vector_and_apply_input_gain_return_expected_tuple() -> None:
    input_vector = build_input_vector(
        valence=0.1,
        novelty=0.2,
        relevance=0.3,
        uncertainty=0.4,
        social_salience=0.5,
    )
    gained_input_vector = apply_input_gain(
        input_vector=input_vector,
        gain_state=2.0,
    )

    assert input_vector == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5))
    assert gained_input_vector == pytest.approx((0.2, 0.4, 0.6, 0.8, 1.0))

# --------------------------------------------------
# Wahrnehmungscodierung
# --------------------------------------------------
def test_encode_perception_builds_complete_input_vector() -> None:
    encoded = encode_perception(
        {
            "valence": -0.5,
            "novelty": 0.4,
            "relevance": 0.7,
            "uncertainty": 0.1,
            "social_salience": 0.2,
        }
    )

    assert encoded == pytest.approx((-0.5, 0.4, 0.7, 0.1, 0.2))


# --------------------------------------------------
# Fehlersignale
# --------------------------------------------------
def test_normalize_channels_rejects_non_finite_channel_value() -> None:
    with pytest.raises(ValueError, match="channel 'valence' must be finite"):
        normalize_channels(
            {
                "valence": float("nan"),
            }
        )


def test_build_input_vector_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="input vector values must be finite"):
        build_input_vector(
            valence=0.1,
            novelty=0.2,
            relevance=float("inf"),
            uncertainty=0.4,
            social_salience=0.5,
        )


def test_apply_input_gain_rejects_non_finite_gain() -> None:
    with pytest.raises(ValueError, match="gain_state must be finite"):
        apply_input_gain(
            input_vector=(0.1, 0.2, 0.3, 0.4, 0.5),
            gain_state=float("nan"),
        )