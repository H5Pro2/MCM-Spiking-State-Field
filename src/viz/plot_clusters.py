# --------------------------------------------------
# src/viz/plot_clusters.py
# --------------------------------------------------
from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt


# --------------------------------------------------
# Cluster-Zentren
# --------------------------------------------------
def plot_cluster_centers(
    cluster_bank: Sequence[Mapping[str, float]],
) -> tuple[plt.Figure, plt.Axes]:
    cluster_ids = [float(cluster["cluster_id"]) for cluster in cluster_bank]
    x_means = [float(cluster["x_mean"]) for cluster in cluster_bank]

    figure, axes = plt.subplots()
    axes.scatter(cluster_ids, x_means)
    axes.set_xlabel("cluster_id")
    axes.set_ylabel("x_mean")
    axes.set_title("Cluster-Zentren")
    axes.grid(True)

    return figure, axes


# --------------------------------------------------
# Cluster-Staerken
# --------------------------------------------------
def plot_cluster_strengths(
    cluster_bank: Sequence[Mapping[str, float]],
) -> tuple[plt.Figure, plt.Axes]:
    cluster_ids = [float(cluster["cluster_id"]) for cluster in cluster_bank]
    strengths = [float(cluster["strength"]) for cluster in cluster_bank]

    figure, axes = plt.subplots()
    axes.bar(cluster_ids, strengths)
    axes.set_xlabel("cluster_id")
    axes.set_ylabel("strength")
    axes.set_title("Cluster-Staerken")
    axes.grid(True)

    return figure, axes