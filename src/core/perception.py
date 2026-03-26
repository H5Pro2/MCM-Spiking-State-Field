# --------------------------------------------------
# src/core/perception.py
# --------------------------------------------------
from __future__ import annotations

import math
from typing import Mapping


# --------------------------------------------------
# Kanalnamen
# --------------------------------------------------
CHANNEL_NAMES = (
    "valence",
    "novelty",
    "relevance",
    "uncertainty",
    "social_salience",
)

# --------------------------------------------------
# Normalisierung
# --------------------------------------------------
def normalize_channels(raw_channels: Mapping[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}

    for channel_name in CHANNEL_NAMES:
        value = float(raw_channels.get(channel_name, 0.0))
        if not math.isfinite(value):
            raise ValueError(f"channel '{channel_name}' must be finite")
        normalized[channel_name] = value

    return normalized


# --------------------------------------------------
# Inputvektor
# --------------------------------------------------
def build_input_vector(
    valence: float,
    novelty: float,
    relevance: float,
    uncertainty: float,
    social_salience: float,
) -> tuple[float, float, float, float, float]:
    channel_values = (
        float(valence),
        float(novelty),
        float(relevance),
        float(uncertainty),
        float(social_salience),
    )

    for value in channel_values:
        if not math.isfinite(value):
            raise ValueError("input vector values must be finite")

    return channel_values


# --------------------------------------------------
# Gain
# --------------------------------------------------
def apply_input_gain(
    input_vector: tuple[float, float, float, float, float],
    gain_state: float,
) -> tuple[float, float, float, float, float]:
    gain_value = float(gain_state)
    if not math.isfinite(gain_value):
        raise ValueError("gain_state must be finite")

    return tuple(value * gain_value for value in input_vector)


# --------------------------------------------------
# Wahrnehmungscodierung
# --------------------------------------------------
def encode_perception(raw_input: Mapping[str, float]) -> tuple[float, float, float, float, float]:
    normalized_channels = normalize_channels(raw_input)
    return build_input_vector(
        valence=normalized_channels["valence"],
        novelty=normalized_channels["novelty"],
        relevance=normalized_channels["relevance"],
        uncertainty=normalized_channels["uncertainty"],
        social_salience=normalized_channels["social_salience"],
    )