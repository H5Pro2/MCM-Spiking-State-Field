# --------------------------------------------------
# tests/test_clustering.py
# --------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.clustering import (
    build_cluster_features,
    fit_clusters,
    prune_weak_clusters,
)


# --------------------------------------------------
# Merkmalsbildung
# --------------------------------------------------
def test_build_cluster_features_computes_window_statistics() -> None:
    features = build_cluster_features(
        x_series=(0.0, 1.0, 2.0),
        density_series=(
            (0.2, 0.8),
            (0.5, 0.5),
            (0.1, 0.9),
        ),
        metrics_series=(
            {"peak_count": 1.0, "density_variance": 0.2},
            {"peak_count": 2.0, "density_variance": 0.4},
            {"peak_count": 1.0, "density_variance": 0.6},
        ),
    )

    assert features["x_mean"] == pytest.approx(1.0)
    assert features["x_variance"] == pytest.approx(2.0 / 3.0)
    assert features["x_span"] == pytest.approx(2.0)
    assert features["mean_peak_count"] == pytest.approx(4.0 / 3.0)
    assert features["mean_density_variance"] == pytest.approx(0.4)
    assert features["final_density_mass"] == pytest.approx(1.0)


# --------------------------------------------------
# Cluster-Fit
# --------------------------------------------------
def test_fit_clusters_merges_nearby_feature_vectors() -> None:

    cluster_bank = fit_clusters(
        feature_window=(
            {
                "x_mean": 0.10,
                "x_variance": 0.20,
                "x_span": 0.40,
                "mean_peak_count": 1.0,
                "mean_density_variance": 0.30,
                "final_density_mass": 1.0,
            },
            {
                "x_mean": 0.15,
                "x_variance": 0.25,
                "x_span": 0.45,
                "mean_peak_count": 1.0,
                "mean_density_variance": 0.35,
                "final_density_mass": 1.0,
            },
        ),
        assignment_threshold=0.75,
    )

    assert len(cluster_bank) == 1
    assert cluster_bank[0]["count"] == pytest.approx(2.0)
    assert cluster_bank[0]["strength"] == pytest.approx(2.0)


# --------------------------------------------------
# Pruning
# --------------------------------------------------
def test_prune_weak_clusters_removes_entries_below_thresholds() -> None:

    pruned_clusters = prune_weak_clusters(
        cluster_bank=(
            {
                "cluster_id": 0.0,
                "x_mean": 0.0,
                "x_variance": 0.1,
                "x_span": 0.2,
                "mean_peak_count": 1.0,
                "mean_density_variance": 0.1,
                "final_density_mass": 1.0,
                "count": 2.0,
                "age": 2.0,
                "strength": 1.5,
                "stability": 0.8,
            },
            {
                "cluster_id": 1.0,
                "x_mean": 1.0,
                "x_variance": 0.3,
                "x_span": 0.5,
                "mean_peak_count": 2.0,
                "mean_density_variance": 0.4,
                "final_density_mass": 1.0,
                "count": 1.0,
                "age": 4.0,
                "strength": 0.2,
                "stability": 0.1,
            },
        ),
        min_stability=0.15,
        min_strength=0.5,
    )

    assert len(pruned_clusters) == 1
    assert pruned_clusters[0]["cluster_id"] == pytest.approx(0.0)