# --------------------------------------------------
# src/core/clustering.py
# --------------------------------------------------
from __future__ import annotations

import math
from typing import Mapping, Sequence


# --------------------------------------------------
# Defaults
# --------------------------------------------------
DEFAULT_ASSIGNMENT_THRESHOLD = 0.75
DEFAULT_MIN_STABILITY = 0.15
DEFAULT_MIN_STRENGTH = 0.5


# --------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------
def _ensure_finite(value: float, name: str) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


# --------------------------------------------------
# Merkmalsbildung
# --------------------------------------------------
def build_cluster_features(
    x_series: Sequence[float],
    density_series: Sequence[Sequence[float]],
    metrics_series: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    x_values = tuple(_ensure_finite(value, "x_series value") for value in x_series)
    density_values = tuple(tuple(float(entry) for entry in density) for density in density_series)
    metric_values = tuple(dict(metrics) for metrics in metrics_series)

    if not x_values:
        raise ValueError("x_series must not be empty")
    if len(density_values) != len(metric_values):
        raise ValueError("density_series and metrics_series must have same length")
    if density_values and len(density_values) != len(x_values):
        raise ValueError("x_series and density_series must have same length")

    x_mean = sum(x_values) / len(x_values)
    x_variance = sum((value - x_mean) ** 2 for value in x_values) / len(x_values)
    x_span = max(x_values) - min(x_values)

    peak_count_values: list[float] = []
    density_variance_values: list[float] = []

    for metrics in metric_values:
        peak_count_values.append(_ensure_finite(metrics.get("peak_count", 0.0), "peak_count"))
        density_variance_values.append(
            _ensure_finite(metrics.get("density_variance", 0.0), "density_variance")
        )

    mean_peak_count = (
        sum(peak_count_values) / len(peak_count_values)
        if peak_count_values
        else 0.0
    )
    mean_density_variance = (
        sum(density_variance_values) / len(density_variance_values)
        if density_variance_values
        else 0.0
    )

    final_density_mass = 0.0
    if density_values:
        final_density_mass = _ensure_finite(sum(density_values[-1]), "final_density_mass")

    return {
        "x_mean": x_mean,
        "x_variance": x_variance,
        "x_span": x_span,
        "mean_peak_count": mean_peak_count,
        "mean_density_variance": mean_density_variance,
        "final_density_mass": final_density_mass,
    }


# --------------------------------------------------
# Cluster-Fit
# --------------------------------------------------
def fit_clusters(
    feature_window: Sequence[Mapping[str, float]],
    assignment_threshold: float = DEFAULT_ASSIGNMENT_THRESHOLD,
) -> list[dict[str, float]]:
    feature_values = [dict(features) for features in feature_window]
    threshold_value = _ensure_finite(assignment_threshold, "assignment_threshold")
    if threshold_value <= 0.0:
        raise ValueError("assignment_threshold must be > 0.0")

    cluster_bank: list[dict[str, float]] = []

    for features in feature_values:
        assignment = assign_cluster(
            current_features=features,
            cluster_bank=cluster_bank,
            assignment_threshold=threshold_value,
        )

        if assignment["matched"] >= 0.5:
            cluster_index = int(assignment["cluster_index"])
            cluster_entry = cluster_bank[cluster_index]
            count_before = cluster_entry["count"]
            count_after = count_before + 1.0

            cluster_entry["x_mean"] = (
                (cluster_entry["x_mean"] * count_before) + features["x_mean"]
            ) / count_after
            cluster_entry["x_variance"] = (
                (cluster_entry["x_variance"] * count_before) + features["x_variance"]
            ) / count_after
            cluster_entry["x_span"] = (
                (cluster_entry["x_span"] * count_before) + features["x_span"]
            ) / count_after
            cluster_entry["mean_peak_count"] = (
                (cluster_entry["mean_peak_count"] * count_before) + features["mean_peak_count"]
            ) / count_after
            cluster_entry["mean_density_variance"] = (
                (cluster_entry["mean_density_variance"] * count_before)
                + features["mean_density_variance"]
            ) / count_after
            cluster_entry["final_density_mass"] = (
                (cluster_entry["final_density_mass"] * count_before)
                + features["final_density_mass"]
            ) / count_after
            cluster_entry["count"] = count_after
            cluster_entry["age"] += 1.0
            cluster_entry["strength"] += 1.0
        else:
            cluster_bank.append(
                {
                    "cluster_id": float(len(cluster_bank)),
                    "x_mean": features["x_mean"],
                    "x_variance": features["x_variance"],
                    "x_span": features["x_span"],
                    "mean_peak_count": features["mean_peak_count"],
                    "mean_density_variance": features["mean_density_variance"],
                    "final_density_mass": features["final_density_mass"],
                    "count": 1.0,
                    "age": 1.0,
                    "strength": 1.0,
                    "stability": 1.0,
                }
            )

    return update_cluster_stability(cluster_bank)


# --------------------------------------------------
# Cluster-Zuordnung
# --------------------------------------------------
def assign_cluster(
    current_features: Mapping[str, float],
    cluster_bank: Sequence[Mapping[str, float]],
    assignment_threshold: float = DEFAULT_ASSIGNMENT_THRESHOLD,
) -> dict[str, float]:
    threshold_value = _ensure_finite(assignment_threshold, "assignment_threshold")
    if threshold_value <= 0.0:
        raise ValueError("assignment_threshold must be > 0.0")

    if not cluster_bank:
        return {
            "matched": 0.0,
            "cluster_index": -1.0,
            "distance": math.inf,
        }

    current_x_mean = _ensure_finite(current_features["x_mean"], "x_mean")
    current_x_variance = _ensure_finite(current_features["x_variance"], "x_variance")
    current_peak_count = _ensure_finite(current_features["mean_peak_count"], "mean_peak_count")
    current_density_variance = _ensure_finite(
        current_features["mean_density_variance"],
        "mean_density_variance",
    )

    best_index = -1
    best_distance = math.inf

    for index, cluster_entry in enumerate(cluster_bank):
        cluster_x_mean = _ensure_finite(cluster_entry["x_mean"], "cluster x_mean")
        cluster_x_variance = _ensure_finite(cluster_entry["x_variance"], "cluster x_variance")
        cluster_peak_count = _ensure_finite(
            cluster_entry["mean_peak_count"],
            "cluster mean_peak_count",
        )
        cluster_density_variance = _ensure_finite(
            cluster_entry["mean_density_variance"],
            "cluster mean_density_variance",
        )

        distance = math.sqrt(
            (current_x_mean - cluster_x_mean) ** 2
            + (current_x_variance - cluster_x_variance) ** 2
            + ((current_peak_count - cluster_peak_count) * 0.25) ** 2
            + ((current_density_variance - cluster_density_variance) * 0.5) ** 2
        )

        if distance < best_distance:
            best_distance = distance
            best_index = index

    if best_distance <= threshold_value:
        return {
            "matched": 1.0,
            "cluster_index": float(best_index),
            "distance": best_distance,
        }

    return {
        "matched": 0.0,
        "cluster_index": -1.0,
        "distance": best_distance,
    }


# --------------------------------------------------
# Stabilitaet
# --------------------------------------------------
def update_cluster_stability(
    cluster_state: Sequence[Mapping[str, float]],
) -> list[dict[str, float]]:
    updated_clusters: list[dict[str, float]] = []

    for cluster_entry in cluster_state:
        entry = dict(cluster_entry)
        count_value = _ensure_finite(entry.get("count", 0.0), "count")
        age_value = _ensure_finite(entry.get("age", 0.0), "age")
        x_variance_value = _ensure_finite(entry.get("x_variance", 0.0), "x_variance")
        density_variance_value = _ensure_finite(
            entry.get("mean_density_variance", 0.0),
            "mean_density_variance",
        )

        denominator = 1.0 + x_variance_value + density_variance_value
        stability = 0.0
        if age_value > 0.0:
            stability = (count_value / age_value) / denominator

        entry["stability"] = stability
        updated_clusters.append(entry)

    return updated_clusters


# --------------------------------------------------
# Pruning
# --------------------------------------------------
def prune_weak_clusters(
    cluster_bank: Sequence[Mapping[str, float]],
    min_stability: float = DEFAULT_MIN_STABILITY,
    min_strength: float = DEFAULT_MIN_STRENGTH,
) -> list[dict[str, float]]:
    stability_threshold = _ensure_finite(min_stability, "min_stability")
    strength_threshold = _ensure_finite(min_strength, "min_strength")

    pruned_clusters: list[dict[str, float]] = []

    for cluster_entry in cluster_bank:
        entry = dict(cluster_entry)
        stability_value = _ensure_finite(entry.get("stability", 0.0), "stability")
        strength_value = _ensure_finite(entry.get("strength", 0.0), "strength")

        if stability_value < stability_threshold:
            continue
        if strength_value < strength_threshold:
            continue

        pruned_clusters.append(entry)

    return pruned_clusters