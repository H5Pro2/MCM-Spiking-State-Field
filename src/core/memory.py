# --------------------------------------------------
# src/core/memory.py
# --------------------------------------------------
from __future__ import annotations

import math
from typing import Mapping, Sequence


# --------------------------------------------------
# Defaults
# --------------------------------------------------
DEFAULT_RETRIEVAL_LIMIT = 3
DEFAULT_DECAY_RATE = 0.05
DEFAULT_FORGET_THRESHOLD = 0.1


# --------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------
def _ensure_finite(value: float, name: str) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


# --------------------------------------------------
# Cluster speichern
# --------------------------------------------------
def store_cluster(
    cluster: Mapping[str, float],
    memory_bank: Sequence[Mapping[str, float]],
) -> list[dict[str, float]]:
    cluster_entry = dict(cluster)
    updated_memory_bank = [dict(entry) for entry in memory_bank]

    cluster_id = _ensure_finite(cluster_entry.get("cluster_id", -1.0), "cluster_id")
    strength = _ensure_finite(cluster_entry.get("strength", 1.0), "strength")
    age = _ensure_finite(cluster_entry.get("age", 1.0), "age")
    stability = _ensure_finite(cluster_entry.get("stability", 0.0), "stability")

    cluster_entry["cluster_id"] = cluster_id
    cluster_entry["strength"] = strength
    cluster_entry["age"] = age
    cluster_entry["stability"] = stability

    for index, memory_entry in enumerate(updated_memory_bank):
        memory_cluster_id = _ensure_finite(memory_entry.get("cluster_id", -1.0), "memory cluster_id")
        if memory_cluster_id == cluster_id:
            updated_memory_bank[index] = cluster_entry
            return updated_memory_bank

    updated_memory_bank.append(cluster_entry)
    return updated_memory_bank


# --------------------------------------------------
# Relevante Cluster abrufen
# --------------------------------------------------
def retrieve_relevant_clusters(
    context_state: Mapping[str, float],
    memory_bank: Sequence[Mapping[str, float]],
    retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT,
) -> list[dict[str, float]]:
    if retrieval_limit <= 0:
        raise ValueError("retrieval_limit must be > 0")

    context_x_mean = _ensure_finite(context_state.get("x_mean", 0.0), "context x_mean")
    context_density_variance = _ensure_finite(
        context_state.get("mean_density_variance", 0.0),
        "context mean_density_variance",
    )

    scored_clusters: list[tuple[float, dict[str, float]]] = []

    for memory_entry in memory_bank:
        entry = dict(memory_entry)

        x_mean = _ensure_finite(entry.get("x_mean", 0.0), "cluster x_mean")
        density_variance = _ensure_finite(
            entry.get("mean_density_variance", 0.0),
            "cluster mean_density_variance",
        )
        strength = _ensure_finite(entry.get("strength", 0.0), "cluster strength")
        stability = _ensure_finite(entry.get("stability", 0.0), "cluster stability")

        distance = abs(context_x_mean - x_mean) + abs(context_density_variance - density_variance)
        score = (strength + stability) - distance
        scored_clusters.append((score, entry))

    scored_clusters.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored_clusters[:retrieval_limit]]


# --------------------------------------------------
# Cluster staerken
# --------------------------------------------------
def strengthen_cluster(
    cluster_id: float,
    memory_bank: Sequence[Mapping[str, float]],
    amount: float = 1.0,
) -> list[dict[str, float]]:
    cluster_id_value = _ensure_finite(cluster_id, "cluster_id")
    amount_value = _ensure_finite(amount, "amount")
    if amount_value < 0.0:
        raise ValueError("amount must be >= 0.0")

    updated_memory_bank: list[dict[str, float]] = []

    for memory_entry in memory_bank:
        entry = dict(memory_entry)
        memory_cluster_id = _ensure_finite(entry.get("cluster_id", -1.0), "memory cluster_id")

        if memory_cluster_id == cluster_id_value:
            entry["strength"] = _ensure_finite(entry.get("strength", 0.0), "strength") + amount_value
            entry["age"] = _ensure_finite(entry.get("age", 0.0), "age") + 1.0

        updated_memory_bank.append(entry)

    return updated_memory_bank


# --------------------------------------------------
# Staerkeabbau
# --------------------------------------------------
def decay_cluster_strength(
    memory_bank: Sequence[Mapping[str, float]],
    decay_rate: float = DEFAULT_DECAY_RATE,
) -> list[dict[str, float]]:
    decay_rate_value = _ensure_finite(decay_rate, "decay_rate")
    if decay_rate_value < 0.0:
        raise ValueError("decay_rate must be >= 0.0")

    updated_memory_bank: list[dict[str, float]] = []

    for memory_entry in memory_bank:
        entry = dict(memory_entry)
        strength = _ensure_finite(entry.get("strength", 0.0), "strength")
        entry["strength"] = max(0.0, strength - decay_rate_value)
        updated_memory_bank.append(entry)

    return updated_memory_bank


# --------------------------------------------------
# Vergessen
# --------------------------------------------------
def forget_irrelevant_clusters(
    memory_bank: Sequence[Mapping[str, float]],
    forget_threshold: float = DEFAULT_FORGET_THRESHOLD,
) -> list[dict[str, float]]:
    threshold_value = _ensure_finite(forget_threshold, "forget_threshold")
    if threshold_value < 0.0:
        raise ValueError("forget_threshold must be >= 0.0")

    retained_memory_bank: list[dict[str, float]] = []

    for memory_entry in memory_bank:
        entry = dict(memory_entry)
        strength = _ensure_finite(entry.get("strength", 0.0), "strength")
        stability = _ensure_finite(entry.get("stability", 0.0), "stability")

        if strength < threshold_value and stability < threshold_value:
            continue

        retained_memory_bank.append(entry)

    return retained_memory_bank