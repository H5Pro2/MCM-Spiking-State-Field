# --------------------------------------------------
# tests/test_field_density.py
# --------------------------------------------------
from __future__ import annotations

import pytest

from src.core.field_density import (
    build_field_axis,
    compute_density_mean,
    compute_density_variance,
    find_density_peaks,
    normalize_density,
    reconstruct_density,
)
from src.core.mcm_state import FIELD_MAX, FIELD_MIN


# --------------------------------------------------
# Feldachse
# --------------------------------------------------
def test_build_field_axis_spans_field_bounds() -> None:
    field_axis = build_field_axis(num_points=7)

    assert len(field_axis) == 7
    assert field_axis[0] == pytest.approx(FIELD_MIN)
    assert field_axis[-1] == pytest.approx(FIELD_MAX)


# --------------------------------------------------
# Normierung
# --------------------------------------------------
def test_normalize_density_returns_probability_distribution() -> None:
    density = normalize_density((1.0, 2.0, 3.0))

    assert sum(density) == pytest.approx(1.0)
    assert density[0] == pytest.approx(1.0 / 6.0)
    assert density[1] == pytest.approx(2.0 / 6.0)
    assert density[2] == pytest.approx(3.0 / 6.0)


# --------------------------------------------------
# Mittelwert und Varianz
# --------------------------------------------------
def test_compute_density_mean_and_variance_match_density_shape() -> None:
    field_axis = (-1.0, 0.0, 1.0)
    density = (0.2, 0.6, 0.2)

    mean_value = compute_density_mean(
        density=density,
        field_axis=field_axis,
    )
    variance_value = compute_density_variance(
        density=density,
        field_axis=field_axis,
    )

    assert mean_value == pytest.approx(0.0)
    assert variance_value == pytest.approx(0.4)


# --------------------------------------------------
# Peaks
# --------------------------------------------------
def test_find_density_peaks_detects_local_maxima() -> None:
    field_axis = (-2.0, -1.0, 0.0, 1.0, 2.0)
    density = (0.1, 0.4, 0.1, 0.3, 0.1)

    peaks = find_density_peaks(
        density=density,
        field_axis=field_axis,
        min_peak_height=0.2,
    )

    assert len(peaks) == 2
    assert peaks[0]["position"] == pytest.approx(-1.0)
    assert peaks[0]["height"] == pytest.approx(0.4)
    assert peaks[1]["position"] == pytest.approx(1.0)
    assert peaks[1]["height"] == pytest.approx(0.3)


# --------------------------------------------------
# Rekonstruktion
# --------------------------------------------------
def test_reconstruct_density_requires_matching_lengths() -> None:
    with pytest.raises(ValueError, match="must have same length"):
        reconstruct_density(
            neural_activity=(0.5, 0.2),
            preferred_positions=(0.0,),
        )