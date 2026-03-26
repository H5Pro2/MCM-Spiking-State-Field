# --------------------------------------------------
# src/core/field_density.py
# --------------------------------------------------
from __future__ import annotations

import math
from typing import Sequence

from src.core.mcm_state import FIELD_MAX, FIELD_MIN


# --------------------------------------------------
# Feldachse
# --------------------------------------------------
def build_field_axis(
    num_points: int = 61,
    min_x: float = FIELD_MIN,
    max_x: float = FIELD_MAX,
) -> tuple[float, ...]:
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    if max_x <= min_x:
        raise ValueError("max_x must be > min_x")

    step = (max_x - min_x) / (num_points - 1)
    return tuple(min_x + (index * step) for index in range(num_points))


# --------------------------------------------------
# Gauß-Kern
# --------------------------------------------------
def _gaussian_kernel(x_value: float, center: float, sigma: float) -> float:
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.0")

    distance = x_value - center
    exponent = -((distance * distance) / (2.0 * sigma * sigma))
    return math.exp(exponent)


# --------------------------------------------------
# Normierung
# --------------------------------------------------
def normalize_density(density: Sequence[float]) -> tuple[float, ...]:
    density_values = tuple(float(value) for value in density)
    if not density_values:
        raise ValueError("density must not be empty")

    for value in density_values:
        if not math.isfinite(value):
            raise ValueError("density values must be finite")
        if value < 0.0:
            raise ValueError("density values must be >= 0.0")

    total = sum(density_values)
    if total <= 0.0:
        uniform_value = 1.0 / len(density_values)
        return tuple(uniform_value for _ in density_values)

    return tuple(value / total for value in density_values)


# --------------------------------------------------
# Rekonstruktion
# --------------------------------------------------
def reconstruct_density(
    neural_activity: Sequence[float],
    preferred_positions: Sequence[float],
    field_axis: Sequence[float] | None = None,
    sigma: float = 0.35,
) -> tuple[float, ...]:
    activity_values = tuple(float(value) for value in neural_activity)
    preferred_values = tuple(float(value) for value in preferred_positions)
    axis_values = tuple(field_axis) if field_axis is not None else build_field_axis()

    if len(activity_values) != len(preferred_values):
        raise ValueError("neural_activity and preferred_positions must have same length")
    if not axis_values:
        raise ValueError("field_axis must not be empty")

    for value in activity_values:
        if not math.isfinite(value):
            raise ValueError("neural_activity values must be finite")
        if value < 0.0:
            raise ValueError("neural_activity values must be >= 0.0")

    for value in preferred_values:
        if not math.isfinite(value):
            raise ValueError("preferred_positions values must be finite")

    for value in axis_values:
        if not math.isfinite(float(value)):
            raise ValueError("field_axis values must be finite")

    raw_density: list[float] = []
    for axis_value in axis_values:
        density_value = 0.0
        for activity, preferred_position in zip(activity_values, preferred_values):
            density_value += activity * _gaussian_kernel(
                x_value=float(axis_value),
                center=preferred_position,
                sigma=sigma,
            )
        raw_density.append(density_value)

    return normalize_density(raw_density)


# --------------------------------------------------
# Mittelwert
# --------------------------------------------------
def compute_density_mean(
    density: Sequence[float],
    field_axis: Sequence[float],
) -> float:
    density_values = normalize_density(density)
    axis_values = tuple(float(value) for value in field_axis)

    if len(density_values) != len(axis_values):
        raise ValueError("density and field_axis must have same length")

    return sum(
        axis_value * density_value
        for axis_value, density_value in zip(axis_values, density_values)
    )


# --------------------------------------------------
# Varianz
# --------------------------------------------------
def compute_density_variance(
    density: Sequence[float],
    field_axis: Sequence[float],
) -> float:
    density_values = normalize_density(density)
    axis_values = tuple(float(value) for value in field_axis)

    if len(density_values) != len(axis_values):
        raise ValueError("density and field_axis must have same length")

    mean_value = compute_density_mean(
        density=density_values,
        field_axis=axis_values,
    )
    return sum(
        ((axis_value - mean_value) ** 2) * density_value
        for axis_value, density_value in zip(axis_values, density_values)
    )


# --------------------------------------------------
# Peaks
# --------------------------------------------------
def find_density_peaks(
    density: Sequence[float],
    field_axis: Sequence[float],
    min_peak_height: float = 0.05,
) -> list[dict[str, float]]:
    density_values = normalize_density(density)
    axis_values = tuple(float(value) for value in field_axis)

    if len(density_values) != len(axis_values):
        raise ValueError("density and field_axis must have same length")
    if len(density_values) < 3:
        return []

    peaks: list[dict[str, float]] = []

    for index in range(1, len(density_values) - 1):
        left_value = density_values[index - 1]
        center_value = density_values[index]
        right_value = density_values[index + 1]

        if center_value < min_peak_height:
            continue
        if center_value < left_value:
            continue
        if center_value < right_value:
            continue

        peaks.append(
            {
                "index": float(index),
                "position": axis_values[index],
                "height": center_value,
            }
        )

    return peaks