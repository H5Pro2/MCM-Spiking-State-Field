# --------------------------------------------------
# src/viz/plot_field.py
# --------------------------------------------------
from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt


# --------------------------------------------------
# Dichteplot
# --------------------------------------------------
def plot_density(
    field_axis: Sequence[float],
    density: Sequence[float],
) -> tuple[plt.Figure, plt.Axes]:
    axis_values = [float(value) for value in field_axis]
    density_values = [float(value) for value in density]

    if len(axis_values) != len(density_values):
        raise ValueError("field_axis and density must have same length")

    figure, axes = plt.subplots()
    axes.plot(axis_values, density_values)
    axes.set_xlabel("x")
    axes.set_ylabel("rho(x,t)")
    axes.set_title("Felddichte")
    axes.grid(True)

    return figure, axes


# --------------------------------------------------
# Dichte-Heatmap
# --------------------------------------------------
def plot_density_heatmap(
    density_over_time: Sequence[Sequence[float]],
) -> tuple[plt.Figure, plt.Axes]:
    heatmap_values = [
        [float(value) for value in density]
        for density in density_over_time
    ]

    if not heatmap_values:
        raise ValueError("density_over_time must not be empty")

    figure, axes = plt.subplots()
    image = axes.imshow(
        heatmap_values,
        aspect="auto",
        origin="lower",
    )
    axes.set_xlabel("field index")
    axes.set_ylabel("time step")
    axes.set_title("Felddichte ueber Zeit")
    figure.colorbar(image, ax=axes)

    return figure, axes


# --------------------------------------------------
# Peak-Plot
# --------------------------------------------------
def plot_density_peaks(
    peaks: Sequence[Mapping[str, float]],
) -> tuple[plt.Figure, plt.Axes]:
    positions = [float(peak["position"]) for peak in peaks]
    heights = [float(peak["height"]) for peak in peaks]

    figure, axes = plt.subplots()
    axes.scatter(positions, heights)
    axes.set_xlabel("x")
    axes.set_ylabel("peak height")
    axes.set_title("Feldpeaks")
    axes.grid(True)

    return figure, axes