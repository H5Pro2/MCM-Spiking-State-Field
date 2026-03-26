# --------------------------------------------------
# src/viz/plot_spikes.py
# --------------------------------------------------
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt


# --------------------------------------------------
# Spike-Raster
# --------------------------------------------------
def plot_spike_raster(
    spike_data: Sequence[Sequence[float]],
) -> tuple[plt.Figure, plt.Axes]:
    raster_rows = [
        [float(value) for value in spike_row]
        for spike_row in spike_data
    ]

    if not raster_rows:
        raise ValueError("spike_data must not be empty")

    figure, axes = plt.subplots()

    for neuron_index, spike_row in enumerate(raster_rows):
        spike_times = [
            time_index
            for time_index, spike_value in enumerate(spike_row)
            if spike_value > 0.0
        ]

        if spike_times:
            axes.vlines(
                spike_times,
                neuron_index + 0.5,
                neuron_index + 1.5,
            )

    axes.set_xlabel("time step")
    axes.set_ylabel("neuron index")
    axes.set_title("Spike-Raster")
    axes.set_ylim(0.5, len(raster_rows) + 0.5)
    axes.grid(True)

    return figure, axes


# --------------------------------------------------
# Firing-Rates
# --------------------------------------------------
def plot_firing_rates(
    rate_data: Sequence[float],
) -> tuple[plt.Figure, plt.Axes]:
    rate_values = [float(value) for value in rate_data]

    if not rate_values:
        raise ValueError("rate_data must not be empty")

    figure, axes = plt.subplots()
    axes.plot(range(len(rate_values)), rate_values)
    axes.set_xlabel("time step")
    axes.set_ylabel("firing rate")
    axes.set_title("Firing-Rates")
    axes.grid(True)

    return figure, axes