# --------------------------------------------------
# src/experiments/exp_phase_d.py
# --------------------------------------------------
from __future__ import annotations

from src.core.clustering import build_cluster_features, fit_clusters, prune_weak_clusters
from src.core.context import initialize_context_state, learn_context_transition, update_context_state
from src.core.field_density import build_field_axis, compute_density_variance, find_density_peaks, reconstruct_density
from src.core.mcm_state import compute_state_velocity, initialize_mcm_state, update_mcm_state
from src.core.neural_core import build_neural_core, decode_neural_state, step_neural_core
from src.core.perception import apply_input_gain, encode_perception
from src.core.self_state import build_self_state


def build_phase_d_inputs() -> list[dict[str, float]]:
    return [
        {"valence": 0.6, "novelty": 0.3, "relevance": 0.6, "uncertainty": 0.1, "social_salience": 0.0},
        {"valence": 0.6, "novelty": 0.3, "relevance": 0.6, "uncertainty": 0.1, "social_salience": 0.0},
        {"valence": -0.7, "novelty": 0.7, "relevance": 0.8, "uncertainty": 0.4, "social_salience": 0.1},
        {"valence": -0.7, "novelty": 0.7, "relevance": 0.8, "uncertainty": 0.4, "social_salience": 0.1},
        {"valence": 0.0, "novelty": 0.0, "relevance": 0.0, "uncertainty": 0.0, "social_salience": 0.0},
        {"valence": 0.4, "novelty": 0.2, "relevance": 0.2, "uncertainty": 0.2, "social_salience": 0.3},
    ]


def run_phase_d_experiment() -> dict[str, object]:
    dt = 0.1
    field_axis = build_field_axis()
    preferred_positions = (-2.0, -1.0, 0.0, 1.0, 2.0)

    neural_state = build_neural_core()
    x_prev = initialize_mcm_state()

    context_state = initialize_context_state()
    previous_context = dict(context_state)
    transition_memory: dict[str, float] = {}

    x_window: list[float] = []
    density_window: list[tuple[float, ...]] = []
    metrics_window: list[dict[str, float]] = []
    feature_history: list[dict[str, float]] = []
    cluster_bank: list[dict[str, float]] = []

    step_outputs: list[dict[str, object]] = []

    for step_index, raw_input in enumerate(build_phase_d_inputs()):
        input_vector = encode_perception(raw_input)
        gained_input_vector = apply_input_gain(input_vector=input_vector, gain_state=1.0)

        neural_state = step_neural_core(
            input_vector=gained_input_vector,
            recurrent_feedback=0.0,
            neural_state=neural_state,
        )
        neural_drive = decode_neural_state(float(neural_state["activity"]))

        x_curr = update_mcm_state(
            x_prev=x_prev,
            neural_input=neural_drive,
            replay_input=0.0,
            regulation_gain=0.4,
            noise=0.0,
            dt=dt,
        )
        velocity = compute_state_velocity(x_prev=x_prev, x_curr=x_curr, dt=dt)

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
        density_variance = compute_density_variance(density=density, field_axis=field_axis)
        density_peaks = find_density_peaks(density=density, field_axis=field_axis)

        x_window.append(x_curr)
        density_window.append(density)
        metrics_window.append({"peak_count": float(len(density_peaks)), "density_variance": density_variance})

        if len(x_window) > 3:
            x_window.pop(0)
            density_window.pop(0)
            metrics_window.pop(0)

        dominant_cluster_id = -1.0
        cluster_stability = 0.0

        if len(x_window) == 3:
            features = build_cluster_features(
                x_series=x_window,
                density_series=density_window,
                metrics_series=metrics_window,
            )
            feature_history.append(features)
            cluster_bank = prune_weak_clusters(fit_clusters(feature_history))

            if cluster_bank:
                dominant_cluster_id = float(cluster_bank[0]["cluster_id"])
                cluster_stability = float(cluster_bank[0].get("stability", 0.0))

        self_state = build_self_state(
            x_value=x_curr,
            velocity=velocity,
            density_variance=density_variance,
            cluster_stability=cluster_stability,
        )

        context_state = update_context_state(
            previous_context=context_state,
            x_value=x_curr,
            density_variance=density_variance,
            cluster_id=dominant_cluster_id,
            input_energy=sum(abs(value) for value in input_vector),
            self_state=self_state,
        )
        transition_memory = learn_context_transition(
            previous_context=previous_context,
            current_context=context_state,
            transition_memory=transition_memory,
        )
        previous_context = dict(context_state)

        step_outputs.append(
            {
                "step": step_index,
                "x": x_curr,
                "velocity": velocity,
                "density_variance": density_variance,
                "dominant_cluster_id": dominant_cluster_id,
                "context": dict(context_state),
                "transition_count": sum(float(v) for v in transition_memory.values()),
            }
        )
        x_prev = x_curr

    return {
        "dt": dt,
        "steps": step_outputs,
        "transition_memory": transition_memory,
        "cluster_bank": cluster_bank,
    }


def evaluate_phase_d_results(run_data: dict[str, object]) -> dict[str, object]:
    steps = list(run_data["steps"])
    transition_memory = dict(run_data["transition_memory"])

    replay_risk_values = [float(step["context"]["replay_risk"]) for step in steps]
    dominant_clusters = {int(float(step["dominant_cluster_id"])) for step in steps if float(step["dominant_cluster_id"]) >= 0.0}

    return {
        "context_built": len(steps) > 0,
        "context_learning_active": len(transition_memory) > 0,
        "transition_count": int(sum(float(value) for value in transition_memory.values())),
        "replay_risk_changes": (max(replay_risk_values) - min(replay_risk_values)) > 1e-9 if replay_risk_values else False,
        "dominant_cluster_varies": len(dominant_clusters) > 1,
        "num_steps": len(steps),
    }
