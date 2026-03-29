# --------------------------------------------------
# tests/test_memory.py
# --------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.memory import (
    decay_cluster_strength,
    forget_irrelevant_clusters,
    retrieve_relevant_clusters,
    store_cluster,
    strengthen_cluster,
)


# --------------------------------------------------
# Cluster speichern
# --------------------------------------------------
def test_store_cluster_replaces_existing_cluster_by_id() -> None:
    memory_bank = [
        {
            "cluster_id": 1.0,
            "x_mean": 0.2,
            "mean_density_variance": 0.1,
            "strength": 0.5,
            "age": 2.0,
            "stability": 0.4,
        }
    ]

    updated_memory = store_cluster(
        cluster={
            "cluster_id": 1.0,
            "x_mean": 0.8,
            "mean_density_variance": 0.3,
            "strength": 1.5,
            "age": 4.0,
            "stability": 0.9,
        },
        memory_bank=memory_bank,
    )

    assert len(updated_memory) == 1
    assert updated_memory[0]["cluster_id"] == pytest.approx(1.0)
    assert updated_memory[0]["x_mean"] == pytest.approx(0.8)
    assert updated_memory[0]["strength"] == pytest.approx(1.5)
    assert updated_memory[0]["age"] == pytest.approx(4.0)
    assert updated_memory[0]["stability"] == pytest.approx(0.9)

# --------------------------------------------------
# Cluster anhaengen
# --------------------------------------------------
def test_store_cluster_appends_new_cluster_when_id_is_missing() -> None:
    updated_memory = store_cluster(
        cluster={
            "cluster_id": 2.0,
            "x_mean": -0.4,
            "mean_density_variance": 0.2,
            "strength": 1.0,
            "age": 1.0,
            "stability": 0.6,
        },
        memory_bank=(
            {
                "cluster_id": 1.0,
                "x_mean": 0.3,
                "mean_density_variance": 0.1,
                "strength": 0.8,
                "age": 2.0,
                "stability": 0.5,
            },
        ),
    )

    assert len(updated_memory) == 2
    assert updated_memory[-1]["cluster_id"] == pytest.approx(2.0)
    assert updated_memory[-1]["x_mean"] == pytest.approx(-0.4)

# --------------------------------------------------
# Retrieval
# --------------------------------------------------
def test_retrieve_relevant_clusters_returns_best_matches_first() -> None:
    retrieved = retrieve_relevant_clusters(
        context_state={
            "x_mean": 0.1,
            "mean_density_variance": 0.2,
        },
        memory_bank=(
            {
                "cluster_id": 1.0,
                "x_mean": 0.15,
                "mean_density_variance": 0.2,
                "strength": 1.0,
                "stability": 0.7,
            },
            {
                "cluster_id": 2.0,
                "x_mean": 1.5,
                "mean_density_variance": 0.9,
                "strength": 0.5,
                "stability": 0.2,
            },
            {
                "cluster_id": 3.0,
                "x_mean": 0.05,
                "mean_density_variance": 0.25,
                "strength": 0.9,
                "stability": 0.6,
            },
        ),
        retrieval_limit=2,
    )

    assert len(retrieved) == 2
    assert retrieved[0]["cluster_id"] == pytest.approx(1.0)
    assert retrieved[1]["cluster_id"] == pytest.approx(3.0)

# --------------------------------------------------
# Strengthening
# --------------------------------------------------
def test_strengthen_cluster_increases_strength_and_age_for_matching_cluster() -> None:
    updated_memory = strengthen_cluster(
        cluster_id=2.0,
        memory_bank=(
            {
                "cluster_id": 1.0,
                "strength": 0.5,
                "age": 1.0,
            },
            {
                "cluster_id": 2.0,
                "strength": 1.0,
                "age": 3.0,
            },
        ),
        amount=0.75,
    )

    assert updated_memory[0]["strength"] == pytest.approx(0.5)
    assert updated_memory[0]["age"] == pytest.approx(1.0)
    assert updated_memory[1]["strength"] == pytest.approx(1.75)
    assert updated_memory[1]["age"] == pytest.approx(4.0)

# --------------------------------------------------
# Decay
# --------------------------------------------------
def test_decay_cluster_strength_reduces_strength_without_going_below_zero() -> None:
    updated_memory = decay_cluster_strength(
        memory_bank=(
            {
                "cluster_id": 1.0,
                "strength": 0.2,
                "stability": 0.5,
            },
            {
                "cluster_id": 2.0,
                "strength": 1.0,
                "stability": 0.6,
            },
        ),
        decay_rate=0.3,
    )

    assert updated_memory[0]["strength"] == pytest.approx(0.0)
    assert updated_memory[1]["strength"] == pytest.approx(0.7)

# --------------------------------------------------
# Vergessen
# --------------------------------------------------
def test_forget_irrelevant_clusters_removes_weak_and_unstable_entries() -> None:
    retained_memory = forget_irrelevant_clusters(
        memory_bank=(
            {
                "cluster_id": 1.0,
                "strength": 0.05,
                "stability": 0.05,
            },
            {
                "cluster_id": 2.0,
                "strength": 0.2,
                "stability": 0.05,
            },
            {
                "cluster_id": 3.0,
                "strength": 0.05,
                "stability": 0.3,
            },
        ),
        forget_threshold=0.1,
    )

    retained_ids = [entry["cluster_id"] for entry in retained_memory]

    assert retained_ids == pytest.approx([2.0, 3.0])

# --------------------------------------------------
# Fehlersignale
# --------------------------------------------------
def test_retrieve_relevant_clusters_rejects_non_positive_retrieval_limit() -> None:
    with pytest.raises(ValueError, match="retrieval_limit must be > 0"):
        retrieve_relevant_clusters(
            context_state={
                "x_mean": 0.0,
                "mean_density_variance": 0.0,
            },
            memory_bank=(),
            retrieval_limit=0,
        )


def test_strengthen_cluster_rejects_negative_amount() -> None:
    with pytest.raises(ValueError, match="amount must be >= 0.0"):
        strengthen_cluster(
            cluster_id=1.0,
            memory_bank=(),
            amount=-0.1,
        )


def test_decay_cluster_strength_rejects_negative_decay_rate() -> None:
    with pytest.raises(ValueError, match="decay_rate must be >= 0.0"):
        decay_cluster_strength(
            memory_bank=(),
            decay_rate=-0.1,
        )


def test_forget_irrelevant_clusters_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="forget_threshold must be >= 0.0"):
        forget_irrelevant_clusters(
            memory_bank=(),
            forget_threshold=-0.1,
        )