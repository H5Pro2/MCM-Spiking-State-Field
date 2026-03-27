# --------------------------------------------------
# src/experiments/exp_phase_c.py
# --------------------------------------------------
from __future__ import annotations

from src.core.clustering import (
    build_cluster_features,
    fit_clusters,
    prune_weak_clusters,
)
from src.core.field_density import (
    build_field_axis,
    compute_density_mean,
    compute_density_variance,
    find_density_peaks,
    reconstruct_density,
)
from src.core.mcm_state import (
    compute_state_velocity,
    initialize_mcm_state,
    update_mcm_state,
)
from src.core.memory import (
    decay_cluster_strength,
    forget_irrelevant_clusters,
    retrieve_relevant_clusters,
    store_cluster,
    strengthen_cluster,
)
from src.core.neural_core import (
    build_neural_core,
    decode_neural_state,
    step_neural_core,
)
from src.core.output import build_output_state
from src.core.perception import apply_input_gain, encode_perception
from src.core.replay import (
    compute_replay_signal,
    detect_rumination_loop,
)


# --------------------------------------------------
# Phase-C-Inputs
# --------------------------------------------------
def build_phase_c_inputs() -> list[dict[str, float]]:
    return [
        {
            "valence": 0.8,
            "novelty": 0.4,
            "relevance": 0.7,
            "uncertainty": 0.1,
            "social_salience": 0.0,
        },
        {
            "valence": 0.8,
            "novelty": 0.4,
            "relevance": 0.7,
            "uncertainty": 0.1,
            "social_salience": 0.0,
        },
        {
            "valence": 0.0,
            "novelty": 0.0,
            "relevance": 0.0,
            "uncertainty": 0.0,
            "social_salience": 0.0,
        },
        {
            "valence": -0.8,
            "novelty": 0.5,
            "relevance": 0.7,
            "uncertainty": 0.3,
            "social_salience": 0.1,
        },
        {
            "valence": -0.8,
            "novelty": 0.5,
            "relevance": 0.7,
            "uncertainty": 0.3,
            "social_salience": 0.1,
        },
        {
            "valence": 0.0,
            "novelty": 0.0,
            "relevance": 0.0,
            "uncertainty": 0.0,
            "social_salience": 0.0,
        },
        {
            "valence": 0.8,
            "novelty": 0.4,
            "relevance": 0.7,
            "uncertainty": 0.1,
            "social_salience": 0.0,
        },
        {
            "valence": 0.8,
            "novelty": 0.4,
            "relevance": 0.7,
            "uncertainty": 0.1,
            "social_salience": 0.0,
        },
        {
            "valence": -0.8,
            "novelty": 0.5,
            "relevance": 0.7,
            "uncertainty": 0.3,
            "social_salience": 0.1,
        },
        {
            "valence": -0.8,
            "novelty": 0.5,
            "relevance": 0.7,
            "uncertainty": 0.3,
            "social_salience": 0.1,
        },
    ]


# --------------------------------------------------
# Phase-C-Experiment
# --------------------------------------------------
def run_phase_c_experiment() -> dict[str, object]:
    dt = 0.1
    regulation_gain = 0.4
    recurrent_feedback = 0.0
    noise = 0.0
    input_gain = 1.0
    window_size = 3

    field_axis = build_field_axis()
    preferred_positions = (-2.0, -1.0, 0.0, 1.0, 2.0)

    neural_state = build_neural_core()
    x_prev = initialize_mcm_state()

    replay_history: list[float] = []
    feature_history: list[dict[str, float]] = []
    memory_bank: list[dict[str, float]] = []
    cluster_bank: list[dict[str, float]] = []
    step_outputs: list[dict[str, object]] = []

    x_window: list[float] = []
    density_window: list[tuple[float, ...]] = []
    metrics_window: list[dict[str, float]] = []

    for step_index, raw_input in enumerate(build_phase_c_inputs()):
        context_state: dict[str, float]
        replay_input = 0.0

        if feature_history:
            context_state = {
                "x_mean": feature_history[-1]["x_mean"],
                "mean_density_variance": feature_history[-1]["mean_density_variance"],
            }
            replay_input = compute_replay_signal(
                memory_bank=memory_bank,
                context_state=context_state,
                mcm_state=x_prev,
                regulation_state={"replay_gain": 1.0},
            )
        else:
            context_state = {
                "x_mean": x_prev,
                "mean_density_variance": 0.0,
            }

        input_vector = encode_perception(raw_input)
        gained_input_vector = apply_input_gain(
            input_vector=input_vector,
            gain_state=input_gain,
        )

        neural_state = step_neural_core(
            input_vector=gained_input_vector,
            recurrent_feedback=recurrent_feedback,
            neural_state=neural_state,
        )
        neural_drive = decode_neural_state(float(neural_state["activity"]))

        x_curr = update_mcm_state(
            x_prev=x_prev,
            neural_input=neural_drive,
            replay_input=replay_input,
            regulation_gain=regulation_gain,
            noise=noise,
            dt=dt,
        )
        velocity = compute_state_velocity(
            x_prev=x_prev,
            x_curr=x_curr,
            dt=dt,
        )

        activity = float(neural_state["activity"])
        distributed_activity = (
            max(0.0, activity + abs(raw_input["valence"]) * 0.5),
            max(0.0, activity + raw_input["novelty"] * 0.4),
            max(0.0, activity + raw_input["relevance"] * 0.3),
            max(0.0, activity + raw_input["uncertainty"] * 0.2),
            max(0.0, activity + raw_input["social_salience"] * 0.1),
        )

        density = reconstruct_density(
            neural_activity=distributed_activity,
            preferred_positions=preferred_positions,
            field_axis=field_axis,
        )
        density_mean = compute_density_mean(
            density=density,
            field_axis=field_axis,
        )
        density_variance = compute_density_variance(
            density=density,
            field_axis=field_axis,
        )
        density_peaks = find_density_peaks(
            density=density,
            field_axis=field_axis,
        )

        x_window.append(x_curr)
        density_window.append(density)
        metrics_window.append(
            {
                "peak_count": float(len(density_peaks)),
                "density_variance": density_variance,
            }
        )

        if len(x_window) > window_size:
            x_window.pop(0)
            density_window.pop(0)
            metrics_window.pop(0)

        current_features: dict[str, float] | None = None
        relevant_clusters: list[dict[str, float]] = []
        rumination_loop = detect_rumination_loop(replay_history)

        if len(x_window) == window_size:
            current_features = build_cluster_features(
                x_series=x_window,
                density_series=density_window,
                metrics_series=metrics_window,
            )
            feature_history.append(current_features)

            cluster_bank = fit_clusters(feature_history)
            cluster_bank = prune_weak_clusters(cluster_bank)

            memory_bank = decay_cluster_strength(memory_bank)

            for cluster_entry in cluster_bank:
                memory_bank = store_cluster(
                    cluster=cluster_entry,
                    memory_bank=memory_bank,
                )

            relevant_clusters = retrieve_relevant_clusters(
                context_state=current_features,
                memory_bank=memory_bank,
            )

            if relevant_clusters:
                memory_bank = strengthen_cluster(
                    cluster_id=float(relevant_clusters[0]["cluster_id"]),
                    memory_bank=memory_bank,
                    amount=0.5,
                )

            memory_bank = forget_irrelevant_clusters(memory_bank)
            rumination_loop = detect_rumination_loop(replay_history)

        replay_history.append(replay_input)

        output_state = build_output_state(
            neural_state=neural_state,
            mcm_state=x_curr,
        )
        output_state["step"] = step_index
        output_state["velocity"] = velocity
        output_state["raw_input"] = dict(raw_input)
        output_state["density"] = density
        output_state["density_mean"] = density_mean
        output_state["density_variance"] = density_variance
        output_state["density_peaks"] = density_peaks
        output_state["replay_input"] = replay_input
        output_state["rumination_loop"] = rumination_loop
        output_state["feature_ready"] = current_features is not None
        output_state["cluster_count"] = len(cluster_bank)
        output_state["memory_count"] = len(memory_bank)
        output_state["relevant_cluster_ids"] = [
            float(cluster_entry["cluster_id"])
            for cluster_entry in relevant_clusters
        ]

        if current_features is None:
            output_state["cluster_features"] = {}
        else:
            output_state["cluster_features"] = dict(current_features)

        step_outputs.append(output_state)
        x_prev = x_curr

    return {
        "dt": dt,
        "regulation_gain": regulation_gain,
        "window_size": window_size,
        "field_axis": field_axis,
        "preferred_positions": preferred_positions,
        "steps": step_outputs,
        "cluster_bank": cluster_bank,
        "memory_bank": memory_bank,
        "replay_history": replay_history,
    }


# --------------------------------------------------
# Phase-C-Auswertung
# --------------------------------------------------
def evaluate_phase_c_results(run_data: dict[str, object]) -> dict[str, object]:
    step_outputs = list(run_data["steps"])
    cluster_bank = list(run_data["cluster_bank"])
    memory_bank = list(run_data["memory_bank"])
    replay_history = [float(value) for value in run_data["replay_history"]]

    cluster_counts = [
        int(step_output["cluster_count"])
        for step_output in step_outputs
    ]
    replay_non_zero_count = sum(
        1
        for replay_value in replay_history
        if abs(replay_value) > 1e-9
    )
    stable_cluster_count = sum(
        1
        for cluster_entry in cluster_bank
        if float(cluster_entry.get("stability", 0.0)) >= 0.15
    )
    rumination_detected = any(
        bool(step_output["rumination_loop"])
        for step_output in step_outputs
    )

    return {
        "clusters_detected": len(cluster_bank) > 0,
        "memory_populated": len(memory_bank) > 0,
        "replay_activated": replay_non_zero_count > 0,
        "stable_cluster_count": stable_cluster_count,
        "max_cluster_count": max(cluster_counts) if cluster_counts else 0,
        "memory_count": len(memory_bank),
        "replay_non_zero_count": replay_non_zero_count,
        "rumination_detected": rumination_detected,
        "num_steps": len(step_outputs),
    }